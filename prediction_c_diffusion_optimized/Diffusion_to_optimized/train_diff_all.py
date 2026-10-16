
import os
import time
import json
import math
import copy
import random
from functools import partial
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# =========================================================
# 0. Config
# =========================================================

@dataclass
class Config:
    TRAIN_CSV: str = str(PROJECT_ROOT / "dataset" / "wrc_class5_63468.txt")
    OUT_DIR: str = str(PROJECT_ROOT / "Diffusion_to_optimized" / "results_diff")

    SEP: str = "\t"
    NO_HEADER: bool = True

    SEQ_LEN: int = 80
    NUM_CLASSES: int = 5
    NUCLEOTIDES: Tuple[str, ...] = ("A", "C", "G", "T")

    EPOCHS: int = 2000
    BATCH_SIZE: int = 512
    LEARNING_RATE: float = 1e-4

    TIMESTEPS: int = 50
    BETA_END: float = 0.2

    LOSS_TYPE: str = "huber"
    P_UNCOND: float = 0.1

    EMA_BETA: float = 0.995
    EMA_START: int = 2000

    SAVE_EVERY: int = 100
    LOG_EVERY: int = 1

    SEED: int = 42
    NUM_WORKERS: int = 2
    GRAD_CLIP_NORM: float = 1.0

    BASE_DIM: int = 80
    DIM_MULTS: Tuple[int, ...] = (1, 2, 4)
    RESNET_BLOCK_GROUPS: int = 4
    LEARNED_SINUSOIDAL_DIM: int = 18

    RESUME_CHECKPOINT: str = ""
    RESUME_LOAD_EMA: bool = True
    RESUME_STRICT: bool = False

    # -------------------------
    # position importance
    # -------------------------
    USE_POSITION_IMPORTANCE_WEIGHT: bool = True
    CLASS_POSITION_WEIGHT_PATH: str = str(PROJECT_ROOT / "prior_condition" / "results_weights" / "class_specific_classifier_weights_full" / "class_position_weights_combined_5bins.npy")

    LOSS_WEIGHT_STRENGTH: float = 0.30
    LOSS_WEIGHT_MIN_CLAMP: float = 0.85
    LOSS_WEIGHT_MAX_CLAMP: float = 1.35
    UNCOND_USE_UNIFORM_WEIGHT: bool = True

    # -------------------------
    # base preference
    # -------------------------
    USE_BASE_PREF_CONDITION: bool = True
    CLASS_BASE_PREF_PATH: str = str(PROJECT_ROOT / "prior_condition" / "results_weights" / "class_specific_classifier_weights_full" / "class_base_preference_from_mutagenesis.npy")

    BASE_PREF_EPS: float = 1e-6
    BASE_PREF_CLIP: float = 3.0

    # -------------------------
    # importance gate
    # -------------------------
    USE_IMPORTANCE_ORDERED_GATE: bool = True
    GATE_CENTER: float = 0.55
    GATE_TEMP: float = 0.12
    PREF_GATE_STRENGTH: float = 0.35
    PREF_GATE_MIN: float = 0.65
    PREF_GATE_MAX: float = 1.35

    # -------------------------
    # adjacent differential prior
    # -------------------------
    USE_ADJ_DIFF_PRIOR: bool = False
    DELTA_PRIOR_USE_IMPORTANCE: bool = False
    DELTA_PRIOR_STRENGTH: float = 0.0
    DELTA_PREF_CLIP: float = 3.0

    # -------------------------
    # removed modules
    # -------------------------
    USE_SPATIAL_PRIOR_FILM: bool = False
    USE_FILM_IN_DOWNS: bool = False
    USE_FILM_IN_MID: bool = False
    USE_BOUNDARY_RESIDUAL_HEAD: bool = False
    USE_PROTO_INJECTION: bool = False

    # -------------------------
    # adjacent competitive loss
    # -------------------------
    USE_ADJ_COMPETITIVE_DENOISE: bool = True
    ADJ_COMPETITIVE_WEIGHT: float = 0.05
    ADJ_MARGIN: float = 0.03
    ADJ_T_MIN_FRAC: float = 0.05
    ADJ_T_MAX_FRAC: float = 0.75
    ADJ_SAMPLE_FRAC: float = 0.60
    ADJ_IMPORTANCE_STRENGTH: float = 0.60
    ADJ_WEIGHT_MIN_CLAMP: float = 0.80
    ADJ_WEIGHT_MAX_CLAMP: float = 1.60
    ADJ_CLASS_WEIGHTS: Tuple[float, ...] = (0.40, 1.20, 1.40, 1.20, 0.40)

    # -------------------------
    # prior Transformer
    # -------------------------
    USE_PRIOR_TOKEN_ENCODER: bool = True
    PRIOR_TOKEN_DIM: int = 128
    PRIOR_TOKEN_LAYERS: int = 4
    PRIOR_TOKEN_HEADS: int = 4
    PRIOR_TOKEN_DROPOUT: float = 0.05
    PRIOR_TOKEN_USE_TIME_LABEL_EMB: bool = True

    # -------------------------
    # cross-attention
    # -------------------------
    USE_CROSS_ATTN_IN_DOWNS: bool = False
    USE_CROSS_ATTN_IN_MID: bool = True
    USE_CROSS_ATTN_IN_UPS: bool = False

    CROSS_ATTN_HEADS: int = 4
    CROSS_ATTN_DIM_HEAD: int = 32
    CROSS_ATTN_SCALE_INIT: float = 0.005
    CROSS_ATTN_ZERO_INIT_OUT: bool = True

    DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"


CFG = Config()
NUCLEOTIDES = list(CFG.NUCLEOTIDES)

UNCOND_LABEL = 0
TOTAL_CLASS_NUMBER = CFG.NUM_CLASSES + 1


# =========================================================
# 1. Utils
# =========================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def safe_group_count(groups: int, channels: int) -> int:
    groups = min(groups, channels)

    while groups > 1 and channels % groups != 0:
        groups -= 1

    return groups


def match_spatial_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="nearest")


def clean_sequence(seq: str) -> str:
    return str(seq).strip().upper().replace(" ", "").replace("N", "")


def safe_torch_load(path: str, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


# =========================================================
# 2. EMA
# =========================================================

class EMA:
    def __init__(self, beta: float):
        self.beta = beta
        self.step = 0

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1.0 - self.beta) * new

    def update_model_average(self, ma_model: nn.Module, current_model: nn.Module):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            ma_params.data = self.update_average(ma_params.data, current_params.data)

    def reset_parameters(self, ema_model: nn.Module, model: nn.Module):
        ema_model.load_state_dict(model.state_dict())

    def step_ema(self, ema_model: nn.Module, model: nn.Module, step_start_ema: int = 2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
        else:
            self.update_model_average(ema_model, model)
        self.step += 1


# =========================================================
# 3. Diffusion schedule
# =========================================================

def linear_beta_schedule(timesteps: int, beta_end: float = 0.2) -> torch.Tensor:
    beta_start = 0.0001
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionSchedule:
    def __init__(self, timesteps: int = 50, beta_end: float = 0.2, device: str = "cpu"):
        self.timesteps = timesteps

        betas = linear_beta_schedule(timesteps, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def q_sample(
    schedule: DiffusionSchedule,
    x_start: torch.Tensor,
    t: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(schedule.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# =========================================================
# 4. Prior loading and building
# =========================================================

def load_class_position_weights(cfg: Config, device: str) -> torch.Tensor:
    path = cfg.CLASS_POSITION_WEIGHT_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"[ClassPositionWeight] File not found: {path}")

    weights = np.load(path).astype(np.float32)

    if weights.ndim != 2:
        raise ValueError(f"[ClassPositionWeight] Expected [num_classes, seq_len], got {weights.shape}")

    if weights.shape[0] != cfg.NUM_CLASSES:
        raise ValueError(f"[ClassPositionWeight] Expected {cfg.NUM_CLASSES} classes, got {weights.shape[0]}")

    if weights.shape[1] >= cfg.SEQ_LEN:
        weights = weights[:, :cfg.SEQ_LEN]
    else:
        pad_len = cfg.SEQ_LEN - weights.shape[1]
        weights = np.concatenate(
            [weights, np.ones((cfg.NUM_CLASSES, pad_len), dtype=np.float32)],
            axis=1,
        )

    weights = weights / (weights.mean(axis=1, keepdims=True) + 1e-8)

    print("=" * 80)
    print("[ClassPositionWeight]")
    print(f"Loaded: {path}")
    print(f"shape: {weights.shape}")

    for c in range(cfg.NUM_CLASSES):
        print(
            f"Class {c}: min={weights[c].min():.4f}, "
            f"max={weights[c].max():.4f}, "
            f"mean={weights[c].mean():.4f}"
        )

    print("=" * 80)

    return torch.from_numpy(weights).float().to(device)


def load_class_base_preference(cfg: Config, device: str) -> torch.Tensor:
    path = cfg.CLASS_BASE_PREF_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"[BasePreference] File not found: {path}")

    pref = np.load(path).astype(np.float32)

    if pref.ndim != 3:
        raise ValueError(f"[BasePreference] Expected 3D array, got {pref.shape}")

    if pref.shape[0] != cfg.NUM_CLASSES:
        raise ValueError(f"[BasePreference] Expected {cfg.NUM_CLASSES} classes, got {pref.shape[0]}")

    if pref.shape[1] == cfg.SEQ_LEN and pref.shape[2] == 4:
        pref = np.transpose(pref, (0, 2, 1))
    elif pref.shape[1] == 4 and pref.shape[2] >= cfg.SEQ_LEN:
        pref = pref[:, :, :cfg.SEQ_LEN]
    else:
        raise ValueError(f"[BasePreference] Expected [5,80,4] or [5,4,80], got {pref.shape}")

    if pref.shape[2] < cfg.SEQ_LEN:
        pad_len = cfg.SEQ_LEN - pref.shape[2]
        pad = np.full((cfg.NUM_CLASSES, 4, pad_len), 0.25, dtype=np.float32)
        pref = np.concatenate([pref, pad], axis=2)

    pref = np.clip(pref, cfg.BASE_PREF_EPS, None)
    pref = pref / (pref.sum(axis=1, keepdims=True) + cfg.BASE_PREF_EPS)

    logp = np.log(pref + cfg.BASE_PREF_EPS) - math.log(0.25)
    logp = logp - logp.mean(axis=1, keepdims=True)

    std = logp.std(axis=(1, 2), keepdims=True) + 1e-6
    feat = logp / std
    feat = np.clip(feat / cfg.BASE_PREF_CLIP, -1.0, 1.0)

    feat = feat[:, None, :, :]

    print("=" * 80)
    print("[BasePreference]")
    print(f"Loaded: {path}")
    print(f"shape after processing: {feat.shape}")

    for c in range(cfg.NUM_CLASSES):
        print(
            f"Class {c}: min={feat[c].min():.4f}, "
            f"max={feat[c].max():.4f}, "
            f"mean={feat[c].mean():.4f}, "
            f"std={feat[c].std():.4f}"
        )

    print("=" * 80)

    return torch.from_numpy(feat).float().to(device)


def build_adjacent_delta_pref_table(base_pref_table: torch.Tensor, cfg: Config) -> torch.Tensor:
    """
    """
    return torch.zeros_like(base_pref_table)


def make_base_pref_batch(
    classes_for_pref: torch.Tensor,
    class_base_pref: Optional[torch.Tensor],
    x_like: torch.Tensor,
) -> Optional[torch.Tensor]:
    if class_base_pref is None:
        return None

    device = x_like.device
    dtype = x_like.dtype
    b, _, _, l = x_like.shape

    out = torch.zeros((b, 1, 4, l), device=device, dtype=dtype)
    valid = classes_for_pref.long() > 0

    if valid.any():
        real_class = (classes_for_pref[valid].long() - 1).clamp(0, class_base_pref.shape[0] - 1)
        pref = class_base_pref[real_class].to(device=device, dtype=dtype)
        out[valid] = pref[:, :, :, :l]

    return out


def make_delta_pref_batch(
    classes_for_pref: torch.Tensor,
    class_delta_pref: Optional[torch.Tensor],
    class_position_weights: Optional[torch.Tensor],
    x_like: torch.Tensor,
    cfg: Config,
) -> Optional[torch.Tensor]:
    return None


def make_importance_map_batch(
    classes_for_pref: torch.Tensor,
    class_position_weights: Optional[torch.Tensor],
    x_like: torch.Tensor,
) -> Optional[torch.Tensor]:
    if class_position_weights is None:
        return None

    device = x_like.device
    dtype = x_like.dtype
    b, _, _, l = x_like.shape

    out = torch.zeros((b, 1, 4, l), device=device, dtype=dtype)
    valid = classes_for_pref.long() > 0

    if valid.any():
        real_class = (classes_for_pref[valid].long() - 1).clamp(0, class_position_weights.shape[0] - 1)

        w = class_position_weights[real_class].to(device=device, dtype=dtype)
        w = w[:, :l]
        w = w / (w.mean(dim=1, keepdim=True) + 1e-8)

        w_min = w.min(dim=1, keepdim=True)[0]
        w_max = w.max(dim=1, keepdim=True)[0]
        w_norm = (w - w_min) / (w_max - w_min + 1e-8)

        out[valid] = w_norm.view(-1, 1, 1, l).repeat(1, 1, 4, 1)

    return out


def make_importance_gate(
    classes_for_gate: torch.Tensor,
    class_position_weights: Optional[torch.Tensor],
    x_like: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    cfg: Config,
) -> torch.Tensor:
    device = x_like.device
    dtype = x_like.dtype
    b, _, _, l = x_like.shape

    gate = torch.ones((b, 1, 1, l), device=device, dtype=dtype)

    if (
        class_position_weights is None
        or not cfg.USE_IMPORTANCE_ORDERED_GATE
        or not cfg.USE_BASE_PREF_CONDITION
    ):
        return gate

    valid = classes_for_gate.long() > 0

    if valid.any():
        real_class = (classes_for_gate[valid].long() - 1).clamp(0, class_position_weights.shape[0] - 1)

        w = class_position_weights[real_class].to(device=device, dtype=dtype)
        w = w[:, :l]

        w_min = w.min(dim=1, keepdim=True)[0]
        w_max = w.max(dim=1, keepdim=True)[0]
        imp_norm = (w - w_min) / (w_max - w_min + 1e-8)

        t_frac = (t[valid].float() / max(schedule.timesteps - 1, 1)).view(-1, 1)

        gate_strength = torch.sigmoid(
            (cfg.GATE_CENTER - t_frac) / max(cfg.GATE_TEMP, 1e-6)
        )

        g = 1.0 + cfg.PREF_GATE_STRENGTH * gate_strength * (2.0 * imp_norm - 1.0)
        g = torch.clamp(g, min=cfg.PREF_GATE_MIN, max=cfg.PREF_GATE_MAX)

        gate[valid] = g.view(-1, 1, 1, l)

    return gate


def make_loss_weight_map(
    classes_for_weight: torch.Tensor,
    class_position_weights: Optional[torch.Tensor],
    x_like: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    device = x_like.device
    dtype = x_like.dtype
    b, _, _, l = x_like.shape

    loss_w = torch.ones((b, 1, 1, l), device=device, dtype=dtype)

    if class_position_weights is None or not cfg.USE_POSITION_IMPORTANCE_WEIGHT:
        return loss_w

    if cfg.UNCOND_USE_UNIFORM_WEIGHT:
        valid = classes_for_weight.long() > 0
    else:
        valid = torch.ones_like(classes_for_weight, dtype=torch.bool)

    if valid.any():
        real_class = (classes_for_weight[valid].long() - 1).clamp(0, class_position_weights.shape[0] - 1)

        w = class_position_weights[real_class].to(device=device, dtype=dtype)
        w = w[:, :l]

        w = 1.0 + cfg.LOSS_WEIGHT_STRENGTH * (w - 1.0)
        w = torch.clamp(w, min=cfg.LOSS_WEIGHT_MIN_CLAMP, max=cfg.LOSS_WEIGHT_MAX_CLAMP)

        loss_w[valid] = w.view(-1, 1, 1, l)

    return loss_w


def make_model_priors(
    classes_for_pref: torch.Tensor,
    x_like: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    position_weights: Optional[torch.Tensor],
    base_pref_table: Optional[torch.Tensor],
    delta_pref_table: Optional[torch.Tensor],
    cfg: Config,
):
    base_pref = make_base_pref_batch(
        classes_for_pref,
        base_pref_table,
        x_like,
    ) if cfg.USE_BASE_PREF_CONDITION else None

    delta_pref = None

    importance_map = make_importance_map_batch(
        classes_for_pref,
        position_weights,
        x_like,
    ) if cfg.USE_POSITION_IMPORTANCE_WEIGHT else None

    base_pref_gate = make_importance_gate(
        classes_for_pref,
        position_weights,
        x_like,
        t,
        schedule,
        cfg,
    )

    return base_pref, delta_pref, importance_map, base_pref_gate


# =========================================================
# 5. Model modules
# =========================================================

class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim: int, dim_out: Optional[int] = None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim: int, dim_out: Optional[int] = None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)

        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        return self.fn(x)


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0

        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: torch.Tensor):
        x = x.reshape(-1, 1).float()
        freqs = x * self.weights.reshape(1, -1) * 2.0 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)

        return fouriered


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()

        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(safe_group_count(groups, dim_out), dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1.0) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        time_emb_dim: Optional[int] = None,
        groups: int = 8,
    ):
        super().__init__()

        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out * 2),
            )
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        self.res_conv = (
            nn.Conv2d(dim, dim_out, 1)
            if dim != dim_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None):
        scale_shift = None

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.reshape(time_emb.shape[0], time_emb.shape[1], 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


def l2norm(t: torch.Tensor):
    return F.normalize(t, dim=-1)


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor):
        b, _, height, width = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = [
            t.reshape(b, self.heads, -1, height * width)
            for t in qkv
        ]

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (height * width)

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = out.reshape(b, -1, height, width)

        return self.to_out(out)


class Attention(nn.Module):
    """
    Full attention used only in the mid block.
    """
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        scale: float = 10.0,
    ):
        super().__init__()

        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor):
        b, _, height, width = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = [
            t.reshape(b, self.heads, -1, height * width).transpose(-2, -1)
            for t in qkv
        ]

        q, k = l2norm(q), l2norm(k)

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = out.transpose(-2, -1).reshape(b, -1, height, width)

        return self.to_out(out)


class SequencePriorTokenEncoder(nn.Module):


    def __init__(
        self,
        seq_len: int,
        token_dim: int,
        num_layers: int,
        nhead: int,
        dropout: float,
        num_classes: int,
        learned_sinusoidal_dim: int = 18,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.token_dim = token_dim

        self.input_proj = nn.Sequential(
            nn.Linear(16, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )

        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, token_dim) * 0.02)

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        self.label_emb = nn.Embedding(num_classes, token_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(token_dim)

    @staticmethod
    def _to_pos_feat(x: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        b, _, _, l = ref.shape
        device = ref.device
        dtype = ref.dtype

        if x is None:
            return torch.zeros((b, l, 4), device=device, dtype=dtype)

        x = x[:, :, :, :l]
        x = x.squeeze(1).permute(0, 2, 1).contiguous()
        return x

    def forward(
        self,
        x_noisy: torch.Tensor,
        time: Optional[torch.Tensor],
        classes: Optional[torch.Tensor],
        base_pref: Optional[torch.Tensor],
        delta_pref: Optional[torch.Tensor],
        importance_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, _, _, l = x_noisy.shape

        x_feat = self._to_pos_feat(x_noisy, x_noisy)
        base_feat = self._to_pos_feat(base_pref, x_noisy)
        delta_feat = self._to_pos_feat(delta_pref, x_noisy)
        imp_feat = self._to_pos_feat(importance_map, x_noisy)

        feat = torch.cat([x_feat, base_feat, delta_feat, imp_feat], dim=-1)

        tokens = self.input_proj(feat)
        tokens = tokens + self.pos_emb[:, :l, :].to(tokens.dtype)

        if time is not None:
            t_emb = self.time_mlp(time).to(tokens.dtype)
            tokens = tokens + t_emb.unsqueeze(1)

        if classes is not None:
            y_emb = self.label_emb(classes.long().clamp(0, TOTAL_CLASS_NUMBER - 1)).to(tokens.dtype)
            tokens = tokens + y_emb.unsqueeze(1)

        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        return tokens


class PriorCrossAttention2D(nn.Module):
    """
    U-Net feature as Query.
    Prior tokens as Key / Value.
    Residual injection:
        H' = H + tanh(scale) * CrossAttn(H, Z)
    """

    def __init__(
        self,
        feature_dim: int,
        token_dim: int,
        heads: int = 4,
        dim_head: int = 32,
        scale_init: float = 0.005,
        zero_init_out: bool = True,
    ):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        hidden = heads * dim_head

        self.norm_feat = nn.LayerNorm(feature_dim)
        self.norm_token = nn.LayerNorm(token_dim)

        self.to_q = nn.Linear(feature_dim, hidden, bias=False)
        self.to_k = nn.Linear(token_dim, hidden, bias=False)
        self.to_v = nn.Linear(token_dim, hidden, bias=False)

        self.to_out = nn.Linear(hidden, feature_dim)

        if zero_init_out:
            nn.init.zeros_(self.to_out.weight)
            if self.to_out.bias is not None:
                nn.init.zeros_(self.to_out.bias)

        self.scale = dim_head ** -0.5
        self.cross_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x: torch.Tensor, prior_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if prior_tokens is None:
            return x

        b, c, h, w = x.shape
        n = h * w

        feat = x.flatten(2).transpose(1, 2).contiguous()
        feat_norm = self.norm_feat(feat)
        tok_norm = self.norm_token(prior_tokens)

        q = self.to_q(feat_norm)
        k = self.to_k(tok_norm)
        v = self.to_v(tok_norm)

        q = q.reshape(b, n, self.heads, self.dim_head).transpose(1, 2)
        k = k.reshape(b, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.reshape(b, -1, self.heads, self.dim_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, self.heads * self.dim_head)
        out = self.to_out(out)

        out = out.transpose(1, 2).reshape(b, c, h, w)

        return x + torch.tanh(self.cross_scale) * out


# =========================================================
# 6. U-Net
# =========================================================

class Unet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg

        dim = cfg.BASE_DIM
        init_dim = dim
        channels = 1
        num_classes = cfg.NUM_CLASSES + 1

        self.channels = channels

        self.init_conv = nn.Conv2d(
            channels,
            init_dim,
            kernel_size=(7, 7),
            padding=3,
        )

        dims = [init_dim, *map(lambda m: dim * m, cfg.DIM_MULTS)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(cfg.LEARNED_SINUSOIDAL_DIM)
        fourier_dim = cfg.LEARNED_SINUSOIDAL_DIM + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.label_emb = nn.Embedding(num_classes, time_dim)

        if cfg.USE_PRIOR_TOKEN_ENCODER:
            self.prior_token_encoder = SequencePriorTokenEncoder(
                seq_len=cfg.SEQ_LEN,
                token_dim=cfg.PRIOR_TOKEN_DIM,
                num_layers=cfg.PRIOR_TOKEN_LAYERS,
                nhead=cfg.PRIOR_TOKEN_HEADS,
                dropout=cfg.PRIOR_TOKEN_DROPOUT,
                num_classes=num_classes,
                learned_sinusoidal_dim=cfg.LEARNED_SINUSOIDAL_DIM,
            )
        else:
            self.prior_token_encoder = None

        block_klass = partial(
            ResnetBlock,
            groups=cfg.RESNET_BLOCK_GROUPS,
        )

        self.downs = nn.ModuleList([])

        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]

        self.mid_block1 = block_klass(
            mid_dim,
            mid_dim,
            time_emb_dim=time_dim,
        )

        self.mid_attn = Residual(
            PreNorm(mid_dim, Attention(mid_dim))
        )

        self.mid_block2 = block_klass(
            mid_dim,
            mid_dim,
            time_emb_dim=time_dim,
        )

        if cfg.USE_CROSS_ATTN_IN_MID and cfg.USE_PRIOR_TOKEN_ENCODER:
            self.mid_cross_attn = PriorCrossAttention2D(
                feature_dim=mid_dim,
                token_dim=cfg.PRIOR_TOKEN_DIM,
                heads=cfg.CROSS_ATTN_HEADS,
                dim_head=cfg.CROSS_ATTN_DIM_HEAD,
                scale_init=cfg.CROSS_ATTN_SCALE_INIT,
                zero_init_out=cfg.CROSS_ATTN_ZERO_INIT_OUT,
            )
        else:
            self.mid_cross_attn = None

        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                        ),
                        block_klass(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                        ),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = block_klass(
            dim * 2,
            dim,
            time_emb_dim=time_dim,
        )

        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        classes: Optional[torch.Tensor],
        base_pref: Optional[torch.Tensor] = None,
        base_pref_gate: Optional[torch.Tensor] = None,
        delta_pref: Optional[torch.Tensor] = None,
        importance_map: Optional[torch.Tensor] = None,
    ):
        x_input = x

        if base_pref is not None and base_pref_gate is not None:
            base_pref_gated = base_pref * base_pref_gate
        else:
            base_pref_gated = base_pref

        if self.prior_token_encoder is not None:
            prior_tokens = self.prior_token_encoder(
                x_noisy=x_input,
                time=time,
                classes=classes,
                base_pref=base_pref_gated,
                delta_pref=None,
                importance_map=importance_map,
            )
        else:
            prior_tokens = None

        x = self.init_conv(x)
        r = x.clone()

        t_emb = self.time_mlp(time)

        if classes is not None:
            t_emb = t_emb + self.label_emb(classes.long().clamp(0, TOTAL_CLASS_NUMBER - 1))

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t_emb)
            h.append(x)

            x = block2(x, t_emb)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)

        if self.mid_cross_attn is not None:
            x = self.mid_cross_attn(x, prior_tokens)

        x = self.mid_block2(x, t_emb)

        for block1, block2, attn, upsample in self.ups:
            skip = h.pop()
            x = match_spatial_size(x, skip)
            x = torch.cat((x, skip), dim=1)
            x = block1(x, t_emb)

            skip = h.pop()
            x = match_spatial_size(x, skip)
            x = torch.cat((x, skip), dim=1)
            x = block2(x, t_emb)

            x = attn(x)
            x = upsample(x)

        x = match_spatial_size(x, r)
        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t_emb)

        return self.final_conv(x)


def build_model(cfg: Config) -> nn.Module:
    return Unet(cfg)


# =========================================================
# 7. Dataset
# =========================================================

def one_hot_minus_one(
    seq: str,
    seq_len: int,
    nucleotides: List[str] = NUCLEOTIDES,
) -> np.ndarray:
    arr = np.full((4, seq_len), -1.0, dtype=np.float32)
    idx_map = {b: i for i, b in enumerate(nucleotides)}

    seq = clean_sequence(seq)

    for pos, base in enumerate(seq[:seq_len]):
        if base in idx_map:
            arr[idx_map[base], pos] = 1.0

    return arr


def load_promoter_dataframe(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.TRAIN_CSV):
        raise FileNotFoundError(f"TRAIN_CSV not found: {cfg.TRAIN_CSV}")

    if cfg.NO_HEADER:
        df = pd.read_csv(
            cfg.TRAIN_CSV,
            sep=cfg.SEP,
            header=None,
            names=["sequence", "label"],
        )
    else:
        df = pd.read_csv(cfg.TRAIN_CSV, sep=cfg.SEP)

        if "sequence" not in df.columns or "label" not in df.columns:
            raise ValueError("如果 NO_HEADER=False，数据文件中必须包含 sequence 和 label 两列。")

        df = df[["sequence", "label"]].copy()

    n0 = len(df)

    df["sequence"] = df["sequence"].map(clean_sequence)
    df["label"] = df["label"].astype(int)

    df = df[df["sequence"].str.len() == cfg.SEQ_LEN].copy()
    df = df[df["sequence"].str.fullmatch(r"[ACGT]+").fillna(False)].copy()
    df = df[(df["label"] >= 0) & (df["label"] < cfg.NUM_CLASSES)].copy()

    df["cond_label"] = df["label"] + 1
    df = df.reset_index(drop=True)

    print("=" * 80)
    print("[Data]")
    print(f"Original rows: {n0}")
    print(f"Valid rows:    {len(df)}")
    print(f"Seq length:    {cfg.SEQ_LEN}")
    print(f"Labels:        {sorted(df['label'].unique().tolist())}")
    print("-" * 80)
    print(df["label"].value_counts().sort_index().to_string())
    print("=" * 80)

    if len(df) == 0:
        raise ValueError("过滤后没有有效序列，请检查 TRAIN_CSV、SEQ_LEN 或数据格式。")

    return df


class PromoterDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int):
        self.seqs = df["sequence"].tolist()
        self.labels = df["cond_label"].astype(int).to_numpy()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx: int):
        x = one_hot_minus_one(self.seqs[idx], self.seq_len)
        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# =========================================================
# 8. Losses
# =========================================================

def loss_map_fn(pred: torch.Tensor, target: torch.Tensor, loss_type: str):
    if loss_type == "l1":
        return F.l1_loss(pred, target, reduction="none")
    if loss_type == "l2":
        return F.mse_loss(pred, target, reduction="none")
    if loss_type == "huber":
        return F.smooth_l1_loss(pred, target, reduction="none")
    raise ValueError(f"Unknown loss_type: {loss_type}")


def sample_adjacent_internal_labels(classes: torch.Tensor, cfg: Config) -> torch.Tensor:
    device = classes.device
    real_y = (classes.long() - 1).clamp(0, cfg.NUM_CLASSES - 1)
    adj = real_y.clone()

    for i in range(real_y.shape[0]):
        y = int(real_y[i].item())

        if y == 0:
            a = 1
        elif y == cfg.NUM_CLASSES - 1:
            a = cfg.NUM_CLASSES - 2
        else:
            a = y - 1 if torch.rand((), device=device).item() < 0.5 else y + 1

        adj[i] = a

    return adj + 1


def get_adj_position_weights(
    classes: torch.Tensor,
    class_position_weights: torch.Tensor,
    x_like: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    device = x_like.device
    dtype = x_like.dtype
    _, _, _, l = x_like.shape

    real_class = (classes.long() - 1).clamp(0, cfg.NUM_CLASSES - 1)

    w = class_position_weights[real_class].to(device=device, dtype=dtype)
    w = w[:, :l]
    w = w / (w.mean(dim=1, keepdim=True) + 1e-8)

    w = 1.0 + cfg.ADJ_IMPORTANCE_STRENGTH * (w - 1.0)
    w = torch.clamp(
        w,
        min=cfg.ADJ_WEIGHT_MIN_CLAMP,
        max=cfg.ADJ_WEIGHT_MAX_CLAMP,
    )

    return w


def adjacent_competitive_denoise_loss_fast(
    denoise_model: nn.Module,
    schedule: DiffusionSchedule,
    x_noisy: torch.Tensor,
    t: torch.Tensor,
    true_noise: torch.Tensor,
    classes: torch.Tensor,
    predicted_correct: torch.Tensor,
    keep_cond: torch.Tensor,
    position_weights: torch.Tensor,
    base_pref_table: Optional[torch.Tensor],
    delta_pref_table: Optional[torch.Tensor],
    cfg: Config,
):
    device = x_noisy.device
    b = x_noisy.shape[0]

    if not cfg.USE_ADJ_COMPETITIVE_DENOISE or cfg.ADJ_COMPETITIVE_WEIGHT <= 0:
        zero = torch.tensor(0.0, device=device)
        return zero, 0.0, 0.0, 0.0

    t_frac = t.float() / max(schedule.timesteps - 1, 1)
    valid_t = (t_frac >= cfg.ADJ_T_MIN_FRAC) & (t_frac <= cfg.ADJ_T_MAX_FRAC)
    valid = valid_t & keep_cond & (classes.long() > 0)

    if valid.sum().item() == 0:
        zero = torch.tensor(0.0, device=device)
        return zero, float(valid.float().mean().detach().cpu().item()), 0.0, 0.0

    valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
    n_valid_before_sample = valid_idx.numel()

    if cfg.ADJ_SAMPLE_FRAC < 1.0:
        n_keep = max(1, int(n_valid_before_sample * cfg.ADJ_SAMPLE_FRAC))
        perm = torch.randperm(n_valid_before_sample, device=device)[:n_keep]
        valid_idx = valid_idx[perm]

    x_sel = x_noisy[valid_idx]
    t_sel = t[valid_idx]
    noise_sel = true_noise[valid_idx]
    classes_sel = classes[valid_idx]
    pred_correct_sel = predicted_correct[valid_idx]

    adj_classes = sample_adjacent_internal_labels(classes_sel, cfg)

    base_pref_adj, delta_pref_adj, imp_adj, gate_adj = make_model_priors(
        adj_classes,
        x_sel,
        t_sel,
        schedule,
        position_weights,
        base_pref_table,
        delta_pref_table,
        cfg,
    )

    pred_adjacent = denoise_model(
        x_sel,
        time=t_sel,
        classes=adj_classes,
        base_pref=base_pref_adj,
        base_pref_gate=gate_adj,
        delta_pref=None,
        importance_map=imp_adj,
    )

    err_correct_map = (pred_correct_sel - noise_sel).pow(2)
    err_adj_map = (pred_adjacent - noise_sel).pow(2)

    err_correct_pos = err_correct_map.mean(dim=2).squeeze(1)
    err_adj_pos = err_adj_map.mean(dim=2).squeeze(1)

    w = get_adj_position_weights(
        classes=classes_sel,
        class_position_weights=position_weights,
        x_like=x_sel,
        cfg=cfg,
    )

    err_correct = (err_correct_pos * w).sum(dim=1) / (w.sum(dim=1) + 1e-8)
    err_adj = (err_adj_pos * w).sum(dim=1) / (w.sum(dim=1) + 1e-8)

    raw_margin = F.relu(cfg.ADJ_MARGIN + err_correct - err_adj)

    class_w = torch.tensor(
        cfg.ADJ_CLASS_WEIGHTS,
        device=device,
        dtype=raw_margin.dtype,
    )

    real_class = (classes_sel.long() - 1).clamp(0, cfg.NUM_CLASSES - 1)
    sample_w = class_w[real_class]

    loss = (raw_margin * sample_w).sum() / sample_w.sum().clamp_min(1.0)

    with torch.no_grad():
        active_ratio = float(valid.float().mean().detach().cpu().item())
        selected_ratio = float(valid_idx.numel() / max(b, 1))
        satisfy = (err_correct + cfg.ADJ_MARGIN < err_adj).float()
        satisfy_ratio = float(satisfy.mean().detach().cpu().item())

    return loss, active_ratio, selected_ratio, satisfy_ratio


def p_losses(
    denoise_model: nn.Module,
    schedule: DiffusionSchedule,
    x_start: torch.Tensor,
    t: torch.Tensor,
    classes: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    loss_type: str = "huber",
    p_uncond: float = 0.1,
    position_weights: Optional[torch.Tensor] = None,
    base_pref_table: Optional[torch.Tensor] = None,
    delta_pref_table: Optional[torch.Tensor] = None,
    cfg: Config = CFG,
):
    device = x_start.device

    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(
        schedule=schedule,
        x_start=x_start,
        t=t,
        noise=noise,
    )

    keep_cond = torch.bernoulli(
        torch.full((classes.shape[0],), 1.0 - p_uncond, device=device)
    ).bool()

    classes_masked = torch.where(
        keep_cond,
        classes,
        torch.zeros_like(classes),
    ).long()

    base_pref, delta_pref, importance_map, base_pref_gate = make_model_priors(
        classes_masked,
        x_noisy,
        t,
        schedule,
        position_weights,
        base_pref_table,
        delta_pref_table,
        cfg,
    )

    predicted_noise = denoise_model(
        x_noisy,
        time=t,
        classes=classes_masked,
        base_pref=base_pref,
        base_pref_gate=base_pref_gate,
        delta_pref=None,
        importance_map=importance_map,
    )

    loss_map = loss_map_fn(
        predicted_noise,
        noise,
        loss_type=loss_type,
    )

    if cfg.USE_POSITION_IMPORTANCE_WEIGHT and position_weights is not None:
        loss_weight_map = make_loss_weight_map(
            classes_for_weight=classes_masked,
            class_position_weights=position_weights,
            x_like=x_start,
            cfg=cfg,
        )
        denoise_loss = (loss_map * loss_weight_map).mean()
    else:
        denoise_loss = loss_map.mean()

    adj_loss = torch.tensor(0.0, device=device)
    adj_active_ratio = 0.0
    adj_selected_ratio = 0.0
    adj_satisfy_ratio = 0.0

    if cfg.USE_ADJ_COMPETITIVE_DENOISE and position_weights is not None:
        adj_loss, adj_active_ratio, adj_selected_ratio, adj_satisfy_ratio = adjacent_competitive_denoise_loss_fast(
            denoise_model=denoise_model,
            schedule=schedule,
            x_noisy=x_noisy,
            t=t,
            true_noise=noise,
            classes=classes,
            predicted_correct=predicted_noise,
            keep_cond=keep_cond,
            position_weights=position_weights,
            base_pref_table=base_pref_table,
            delta_pref_table=delta_pref_table,
            cfg=cfg,
        )

    total_loss = denoise_loss + cfg.ADJ_COMPETITIVE_WEIGHT * adj_loss

    log_dict = {
        "total_loss": float(total_loss.detach().cpu().item()),
        "denoise_loss": float(denoise_loss.detach().cpu().item()),
        "adj_loss": float(adj_loss.detach().cpu().item()),
        "adj_active_ratio": adj_active_ratio,
        "adj_selected_ratio": adj_selected_ratio,
        "adj_satisfy_ratio": adj_satisfy_ratio,
    }

    return total_loss, log_dict


# =========================================================
# 9. Checkpoint helpers
# =========================================================

def load_resume_if_needed(model: nn.Module, cfg: Config, device: str):
    if not cfg.RESUME_CHECKPOINT:
        print("[Resume] No checkpoint provided. Train from scratch.")
        return

    if not os.path.exists(cfg.RESUME_CHECKPOINT):
        raise FileNotFoundError(f"RESUME_CHECKPOINT not found: {cfg.RESUME_CHECKPOINT}")

    ckpt = safe_torch_load(cfg.RESUME_CHECKPOINT, device)

    state_key = "ema_state_dict" if cfg.RESUME_LOAD_EMA and "ema_state_dict" in ckpt else "model_state_dict"

    missing, unexpected = model.load_state_dict(
        ckpt[state_key],
        strict=cfg.RESUME_STRICT,
    )

    print("=" * 80)
    print("[Resume]")
    print(f"Loaded: {cfg.RESUME_CHECKPOINT}")
    print(f"State key: {state_key}")
    print(f"Strict: {cfg.RESUME_STRICT}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if missing:
        print("First missing keys:")
        for k in missing[:20]:
            print(f"  {k}")

    if unexpected:
        print("First unexpected keys:")
        for k in unexpected[:20]:
            print(f"  {k}")

    print("=" * 80)


def save_checkpoint(
    path: str,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer,
    cfg: Config,
    epoch: int,
    global_step: int,
    loss_value: float,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss_value,
            "config": asdict(cfg),
            "nucleotides": NUCLEOTIDES,
            "label_mapping": {
                "real_label_0_to_4": "0,1,2,3,4",
                "internal_cond_label": "real_label + 1",
                "unconditional_label": 0,
            },
            "method": {
                "name": "Enhanced Structure Prior with Training Innovations",
                "standard_forward_noising": True,
                "onehot_range": "target=1, others=-1",
                "base_preference_condition": cfg.USE_BASE_PREF_CONDITION,
                "adjacent_differential_prior": False,
                "importance_map": cfg.USE_POSITION_IMPORTANCE_WEIGHT,
                "importance_gate": cfg.USE_IMPORTANCE_ORDERED_GATE,
                "importance_modulated_delta_prior": False,
                "prior_token_encoder": cfg.USE_PRIOR_TOKEN_ENCODER,
                "prior_token_time_label_embedding": True,
                "prior_token_position_embedding": True,
                "prior_token_layers": cfg.PRIOR_TOKEN_LAYERS,
                "prior_token_dim": cfg.PRIOR_TOKEN_DIM,
                "cross_attention_in_downs": False,
                "cross_attention_in_mid": True,
                "cross_attention_in_ups": False,
                "tanh_cross_attention_gate": True,
                "cross_attention_scale_init": cfg.CROSS_ATTN_SCALE_INIT,
                "double_skip_unet": True,
                "spatial_prior_film": False,
                "boundary_residual_head": False,
                "proto_injection": False,
                "position_weighted_denoising_loss": True,
                "adjacent_competitive_denoising": cfg.USE_ADJ_COMPETITIVE_DENOISE,
                "adjacent_competitive_weight": cfg.ADJ_COMPETITIVE_WEIGHT,
                "adjacent_margin": cfg.ADJ_MARGIN,
            },
        },
        path,
    )


# =========================================================
# 10. Main
# =========================================================

def main():
    cfg = CFG

    seed_everything(cfg.SEED)

    ensure_dir(cfg.OUT_DIR)
    ckpt_dir = os.path.join(cfg.OUT_DIR, "checkpoints")
    ensure_dir(ckpt_dir)

    config_path = os.path.join(cfg.OUT_DIR, "run_config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    df = load_promoter_dataframe(cfg)

    dataset = PromoterDataset(
        df=df,
        seq_len=cfg.SEQ_LEN,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    device = cfg.DEVICE
    print(f"[Device] {device}")

    model = build_model(cfg).to(device)

    load_resume_if_needed(model, cfg, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 80)
    print("[Model]")
    print(f"Total parameters:          {total_params:,}")
    print(f"Trainable parameters:      {trainable_params:,}")
    print(f"Input shape:               [B, 1, 4, {cfg.SEQ_LEN}]")
    print(f"One-hot range:             target=1, others=-1")
    print(f"Base dim:                  {cfg.BASE_DIM}")
    print(f"Base pref condition:       {cfg.USE_BASE_PREF_CONDITION}")
    print(f"Adj-diff prior:            {cfg.USE_ADJ_DIFF_PRIOR}")
    print(f"Importance map:            {cfg.USE_POSITION_IMPORTANCE_WEIGHT}")
    print(f"Importance gate:           {cfg.USE_IMPORTANCE_ORDERED_GATE}")
    print(f"Prior token encoder:       {cfg.USE_PRIOR_TOKEN_ENCODER}")
    print(f"Prior token layers:        {cfg.PRIOR_TOKEN_LAYERS}")
    print(f"Prior token position emb:  True")
    print(f"Prior token time/label:    True")
    print(f"Cross-attn downs:          {cfg.USE_CROSS_ATTN_IN_DOWNS}")
    print(f"Cross-attn mid:            {cfg.USE_CROSS_ATTN_IN_MID}")
    print(f"Cross-attn ups:            False")
    print(f"Cross-attn tanh gate:      True")
    print(f"Cross-attn scale init:     {cfg.CROSS_ATTN_SCALE_INIT}")
    print(f"Double skip U-Net:         True")
    print(f"Spatial prior FiLM:        False")
    print(f"Boundary residual head:    False")
    print(f"Proto injection:           False")
    print(f"Position weighted loss:    True")
    print(f"Adj competitive:           {cfg.USE_ADJ_COMPETITIVE_DENOISE}")
    print(f"Adj weight:                {cfg.ADJ_COMPETITIVE_WEIGHT}")
    print("=" * 80)

    optimizer = Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    schedule = DiffusionSchedule(
        timesteps=cfg.TIMESTEPS,
        beta_end=cfg.BETA_END,
        device=device,
    )

    position_weights = load_class_position_weights(cfg, device=device)
    base_pref_table = load_class_base_preference(cfg, device=device)
    delta_pref_table = build_adjacent_delta_pref_table(base_pref_table, cfg)

    ema = EMA(cfg.EMA_BETA)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    ema_model.requires_grad_(False)

    log_path = os.path.join(cfg.OUT_DIR, "train_log.csv")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,global_step,total_loss,denoise_loss,adj_loss,"
            "adj_active_ratio,adj_selected_ratio,adj_satisfy_ratio,lr,time_sec\n"
        )

    global_step = 0
    last_loss = float("nan")
    start_time = time.time()

    print("[Train] Start training.")
    print("=" * 80)

    for epoch in tqdm(range(1, cfg.EPOCHS + 1), desc="training"):
        model.train()

        total_losses = []
        denoise_losses = []
        adj_losses = []
        adj_active_ratios = []
        adj_selected_ratios = []
        adj_satisfy_ratios = []

        for x, y in loader:
            x = x.float().to(device, non_blocking=True)
            y = y.long().to(device, non_blocking=True)

            t = torch.randint(
                0,
                cfg.TIMESTEPS,
                (x.shape[0],),
                device=device,
            ).long()

            loss, log_dict = p_losses(
                denoise_model=model,
                schedule=schedule,
                x_start=x,
                t=t,
                classes=y,
                loss_type=cfg.LOSS_TYPE,
                p_uncond=cfg.P_UNCOND,
                position_weights=position_weights,
                base_pref_table=base_pref_table,
                delta_pref_table=delta_pref_table,
                cfg=cfg,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.GRAD_CLIP_NORM is not None and cfg.GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)

            optimizer.step()

            ema.step_ema(
                ema_model=ema_model,
                model=model,
                step_start_ema=cfg.EMA_START,
            )

            global_step += 1

            total_losses.append(log_dict["total_loss"])
            denoise_losses.append(log_dict["denoise_loss"])
            adj_losses.append(log_dict["adj_loss"])
            adj_active_ratios.append(log_dict["adj_active_ratio"])
            adj_selected_ratios.append(log_dict["adj_selected_ratio"])
            adj_satisfy_ratios.append(log_dict["adj_satisfy_ratio"])

        mean_total_loss = float(np.mean(total_losses))
        mean_denoise_loss = float(np.mean(denoise_losses))
        mean_adj_loss = float(np.mean(adj_losses))
        mean_adj_active = float(np.mean(adj_active_ratios))
        mean_adj_selected = float(np.mean(adj_selected_ratios))
        mean_adj_satisfy = float(np.mean(adj_satisfy_ratios))
        elapsed = time.time() - start_time

        last_loss = mean_total_loss

        if epoch % cfg.LOG_EVERY == 0:
            print(
                f"[Epoch {epoch:05d}/{cfg.EPOCHS}] "
                f"total={mean_total_loss:.6f} "
                f"denoise={mean_denoise_loss:.6f} "
                f"adj={mean_adj_loss:.6f} "
                f"adj_active={mean_adj_active:.4f} "
                f"adj_selected={mean_adj_selected:.4f} "
                f"adj_sat={mean_adj_satisfy:.4f} "
                f"step={global_step} "
                f"time={elapsed / 60:.2f} min"
            )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{global_step},"
                f"{mean_total_loss:.8f},{mean_denoise_loss:.8f},{mean_adj_loss:.8f},"
                f"{mean_adj_active:.8f},{mean_adj_selected:.8f},{mean_adj_satisfy:.8f},"
                f"{cfg.LEARNING_RATE},{elapsed:.2f}\n"
            )

        if epoch % cfg.SAVE_EVERY == 0:
            ckpt_path = os.path.join(
                ckpt_dir,
                f"epoch_{epoch:05d}.pt",
            )

            save_checkpoint(
                path=ckpt_path,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                global_step=global_step,
                loss_value=mean_total_loss,
            )

            print(f"[Save] {ckpt_path}")

    final_path = os.path.join(ckpt_dir, "gen_all.pt")

    save_checkpoint(
        path=final_path,
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        cfg=cfg,
        epoch=cfg.EPOCHS,
        global_step=global_step,
        loss_value=last_loss,
    )

    print("=" * 80)
    print("[Done] Training finished.")
    print(f"[Final checkpoint] {final_path}")
    print(f"[Train log] {log_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()