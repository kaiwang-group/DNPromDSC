

import os
import json
import math
import random
from functools import partial
from dataclasses import dataclass, fields
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


# =========================================================
# 0. Config
# =========================================================

@dataclass
class Config:
    TRAIN_CSV: str = "/data/stu1/wrc3_pycharm_project/WRC_mask_diffusion/wrc_masked_dataset/wrc_class5_63468.txt"
    OUT_DIR: str = "/data/stu1/wrc3_pycharm_project/WRC_mask_diffusion/wrc_gen_all/results_diff_all"

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

    USE_POSITION_IMPORTANCE_WEIGHT: bool = True
    CLASS_POSITION_WEIGHT_PATH: str = "/data/stu1/wrc3_pycharm_project/WRC_mask_diffusion/wrc_identification_key_regions_new/results_weights/class_specific_classifier_weights_full/class_position_weights_combined_5bins.npy"

    LOSS_WEIGHT_STRENGTH: float = 0.30
    LOSS_WEIGHT_MIN_CLAMP: float = 0.85
    LOSS_WEIGHT_MAX_CLAMP: float = 1.35
    UNCOND_USE_UNIFORM_WEIGHT: bool = True

    USE_BASE_PREF_CONDITION: bool = True
    CLASS_BASE_PREF_PATH: str = "/data/stu1/wrc3_pycharm_project/WRC_mask_diffusion/wrc_identification_key_regions_new/results_weights/class_specific_classifier_weights_full/class_base_preference_from_mutagenesis.npy"

    BASE_PREF_EPS: float = 1e-6
    BASE_PREF_CLIP: float = 3.0

    USE_IMPORTANCE_ORDERED_GATE: bool = True
    GATE_CENTER: float = 0.55
    GATE_TEMP: float = 0.12
    PREF_GATE_STRENGTH: float = 0.35
    PREF_GATE_MIN: float = 0.65
    PREF_GATE_MAX: float = 1.35

    USE_ADJ_DIFF_PRIOR: bool = False
    DELTA_PRIOR_USE_IMPORTANCE: bool = False
    DELTA_PRIOR_STRENGTH: float = 0.0
    DELTA_PREF_CLIP: float = 3.0

    USE_SPATIAL_PRIOR_FILM: bool = False
    USE_FILM_IN_DOWNS: bool = False
    USE_FILM_IN_MID: bool = False
    USE_BOUNDARY_RESIDUAL_HEAD: bool = False
    USE_PROTO_INJECTION: bool = False

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

    USE_PRIOR_TOKEN_ENCODER: bool = True
    PRIOR_TOKEN_DIM: int = 128
    PRIOR_TOKEN_LAYERS: int = 4
    PRIOR_TOKEN_HEADS: int = 4
    PRIOR_TOKEN_DROPOUT: float = 0.05
    PRIOR_TOKEN_USE_TIME_LABEL_EMB: bool = True

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
# 0. Generation Config
# =========================================================

CHECKPOINT = "/data/stu1/wrc3_pycharm_project/WRC_mask_diffusion/wrc_gen_all/checkpoints/gen_all.pt"

GEN_OUT_ROOT = "/data/stu1/wrc3_pycharm_project/WRC_mask_diffusion/wrc_gen_all/results_all_10sets/generated_all_10sets"

NUM_GENERATION_SETS = 10

GEN_OUT_DIR = GEN_OUT_ROOT

NUM_PER_LABEL = 2000
GEN_BATCH_SIZE = 128

LABELS = [0, 1, 2, 3, 4]

COND_WEIGHT = 1.0

USE_RAW_MODEL = False
SEED = 42

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SHOW_INNER_PROGRESS = False
SAVE_ALL_CSV = True


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


def safe_torch_load(path: str, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


# =========================================================
# 2. Diffusion schedule
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


# =========================================================
# 3. Prior loading and building
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


# =========================================================
# 4. Model modules
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
        _, _, _, l = x_noisy.shape

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
# 5. U-Net
# =========================================================

class Unet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        dim = cfg.BASE_DIM
        init_dim = dim
        channels = 1
        num_classes = cfg.NUM_CLASSES + 1

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
# 6. Config restore
# =========================================================

def cfg_from_checkpoint(ckpt) -> Config:
    cfg = Config()

    saved_cfg = ckpt.get("config", {}) or {}
    valid_keys = {f.name for f in fields(Config)}

    for k, v in saved_cfg.items():
        if k in valid_keys:
            setattr(cfg, k, v)

    cfg.USE_BASE_PREF_CONDITION = True
    cfg.USE_POSITION_IMPORTANCE_WEIGHT = True
    cfg.USE_IMPORTANCE_ORDERED_GATE = True

    cfg.USE_ADJ_DIFF_PRIOR = False
    cfg.DELTA_PRIOR_USE_IMPORTANCE = False
    cfg.DELTA_PRIOR_STRENGTH = 0.0

    cfg.USE_PRIOR_TOKEN_ENCODER = True
    cfg.PRIOR_TOKEN_USE_TIME_LABEL_EMB = True
    cfg.PRIOR_TOKEN_LAYERS = 4
    cfg.PRIOR_TOKEN_DIM = 128
    cfg.PRIOR_TOKEN_HEADS = 4

    cfg.USE_CROSS_ATTN_IN_DOWNS = False
    cfg.USE_CROSS_ATTN_IN_MID = True
    cfg.USE_CROSS_ATTN_IN_UPS = False

    cfg.CROSS_ATTN_SCALE_INIT = 0.005
    cfg.CROSS_ATTN_ZERO_INIT_OUT = True

    cfg.USE_SPATIAL_PRIOR_FILM = False
    cfg.USE_FILM_IN_DOWNS = False
    cfg.USE_FILM_IN_MID = False
    cfg.USE_BOUNDARY_RESIDUAL_HEAD = False
    cfg.USE_PROTO_INJECTION = False

    cfg.USE_ADJ_COMPETITIVE_DENOISE = True
    cfg.ADJ_COMPETITIVE_WEIGHT = 0.05

    return cfg


# =========================================================
# 7. DDPM + CFG
# =========================================================

@torch.no_grad()
def p_sample_guided(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    x: torch.Tensor,
    classes: torch.Tensor,
    t: torch.Tensor,
    t_index: int,
    cond_weight: float,
    position_weights: torch.Tensor,
    base_pref_table: torch.Tensor,
    delta_pref_table: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    batch_size = x.shape[0]

    x_double = torch.cat([x, x], dim=0)
    t_double = torch.cat([t, t], dim=0)

    classes_double = torch.cat(
        [classes, torch.zeros_like(classes)],
        dim=0,
    ).long()

    base_pref_double = make_base_pref_batch(
        classes_for_pref=classes_double,
        class_base_pref=base_pref_table,
        x_like=x_double,
    )

    delta_pref_double = None

    base_pref_gate_double = make_importance_gate(
        classes_for_gate=classes_double,
        class_position_weights=position_weights,
        x_like=x_double,
        t=t_double,
        schedule=schedule,
        cfg=cfg,
    )

    importance_map_double = make_importance_map_batch(
        classes_for_pref=classes_double,
        class_position_weights=position_weights,
        x_like=x_double,
    )

    eps_double = model(
        x_double,
        time=t_double,
        classes=classes_double,
        base_pref=base_pref_double,
        base_pref_gate=base_pref_gate_double,
        delta_pref=delta_pref_double,
        importance_map=importance_map_double,
    )

    eps_cond = eps_double[:batch_size]
    eps_uncond = eps_double[batch_size:]

    eps = (1.0 + cond_weight) * eps_cond - cond_weight * eps_uncond

    betas_t = extract(schedule.betas, t, x.shape)

    sqrt_one_minus_alphas_cumprod_t = extract(
        schedule.sqrt_one_minus_alphas_cumprod,
        t,
        x.shape,
    )

    sqrt_recip_alphas_t = extract(
        schedule.sqrt_recip_alphas,
        t,
        x.shape,
    )

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * eps / (sqrt_one_minus_alphas_cumprod_t + 1e-8)
    )

    if t_index == 0:
        return model_mean

    posterior_variance_t = extract(
        schedule.posterior_variance,
        t,
        x.shape,
    )

    noise = torch.randn_like(x)

    return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    image_size: int,
    classes: torch.Tensor,
    batch_size: int,
    channels: int,
    cond_weight: float,
    position_weights: torch.Tensor,
    base_pref_table: torch.Tensor,
    delta_pref_table: torch.Tensor,
    cfg: Config,
    show_progress: bool = False,
) -> torch.Tensor:
    device = next(model.parameters()).device

    img = torch.randn(
        (batch_size, channels, 4, image_size),
        device=device,
    )

    iterator = reversed(range(schedule.timesteps))

    if show_progress:
        iterator = tqdm(
            iterator,
            desc="sampling",
            total=schedule.timesteps,
        )

    for i in iterator:
        t = torch.full(
            (batch_size,),
            i,
            device=device,
            dtype=torch.long,
        )

        img = p_sample_guided(
            model=model,
            schedule=schedule,
            x=img,
            classes=classes,
            t=t,
            t_index=i,
            cond_weight=cond_weight,
            position_weights=position_weights,
            base_pref_table=base_pref_table,
            delta_pref_table=delta_pref_table,
            cfg=cfg,
        )

    return img


# =========================================================
# 8. Decode
# =========================================================

def decode_tensor_to_sequence(
    x: torch.Tensor,
    nucleotides: List[str] = NUCLEOTIDES,
) -> str:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.squeeze(x)

    if x.shape[0] != 4 and x.shape[-1] == 4:
        x = x.T

    idx = np.argmax(x, axis=0)
    seq = "".join(nucleotides[int(i)] for i in idx)

    return seq


def gc_ratio(seq: str) -> float:
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


# =========================================================
# 9. Generate one full set
# =========================================================

@torch.no_grad()
def generate_one_set(
    set_id: int,
    set_seed: int,
    set_out_dir: str,
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    position_weights: torch.Tensor,
    base_pref_table: torch.Tensor,
    delta_pref_table: torch.Tensor,
    cfg: Config,
    state_key: str,
):
    ensure_dir(set_out_dir)

    print("=" * 80)
    print(f"[Generate Set {set_id:02d}/{NUM_GENERATION_SETS}]")
    print(f"Output dir: {set_out_dir}")
    print(f"Recorded seed: {set_seed}")
    print("=" * 80)

    summary_rows = []
    all_rows = []

    model.eval()

    with torch.no_grad():
        for target_label in LABELS:
            if target_label < 0 or target_label >= cfg.NUM_CLASSES:
                raise ValueError(
                    f"target_label should be in [0, {cfg.NUM_CLASSES - 1}], got {target_label}"
                )

            internal_cond_label = target_label + 1

            label_rows = []
            n_done = 0

            pbar = tqdm(
                total=NUM_PER_LABEL,
                desc=f"set {set_id:02d} | label {target_label}",
            )

            while n_done < NUM_PER_LABEL:
                cur_bs = min(GEN_BATCH_SIZE, NUM_PER_LABEL - n_done)

                classes = torch.full(
                    (cur_bs,),
                    internal_cond_label,
                    dtype=torch.long,
                    device=DEVICE,
                )

                samples = sample(
                    model=model,
                    schedule=schedule,
                    image_size=cfg.SEQ_LEN,
                    classes=classes,
                    batch_size=cur_bs,
                    channels=1,
                    cond_weight=COND_WEIGHT,
                    position_weights=position_weights,
                    base_pref_table=base_pref_table,
                    delta_pref_table=delta_pref_table,
                    cfg=cfg,
                    show_progress=SHOW_INNER_PROGRESS,
                )

                seqs = [
                    decode_tensor_to_sequence(samples[j])
                    for j in range(samples.shape[0])
                ]

                for seq in seqs:
                    label_rows.append(
                        {
                            "sequence": seq,
                            "target_label": target_label,
                            "gc": gc_ratio(seq),
                            "length": len(seq),
                            "set_id": set_id,
                            "seed": set_seed,
                        }
                    )

                n_done += cur_bs
                pbar.update(cur_bs)

            pbar.close()

            df_label_full = pd.DataFrame(label_rows)

            df_label = df_label_full[["sequence"]].copy()

            out_path = os.path.join(
                set_out_dir,
                f"zhou_unet_label_{target_label}.csv",
            )

            df_label.to_csv(out_path, index=False)

            mean_gc = float(df_label_full["gc"].mean()) if len(df_label_full) > 0 else 0.0
            uniq = int(df_label_full["sequence"].nunique()) if len(df_label_full) > 0 else 0
            div = uniq / max(1, len(df_label_full))

            summary_rows.append(
                {
                    "set_id": set_id,
                    "seed": set_seed,
                    "label": target_label,
                    "n": len(df_label_full),
                    "unique": uniq,
                    "diversity": div,
                    "mean_gc": mean_gc,
                    "file": out_path,
                }
            )

            all_rows.extend(label_rows)

            print(
                f"[Set {set_id:02d} | Label {target_label}] saved={out_path} | "
                f"n={len(df_label_full)} unique={uniq} div={div:.4f} gc={mean_gc:.4f}"
            )

    if SAVE_ALL_CSV:
        all_df = pd.DataFrame(all_rows)
        all_path = os.path.join(set_out_dir, "zhou_unet_all_labels.csv")
        all_df.to_csv(all_path, index=False)
        print(f"[Set {set_id:02d} | All] saved={all_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(set_out_dir, "generation_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    meta = {
        "set_id": set_id,
        "set_seed": set_seed,
        "checkpoint": CHECKPOINT,
        "state_key": state_key,
        "seq_len": cfg.SEQ_LEN,
        "num_classes": cfg.NUM_CLASSES,
        "timesteps": cfg.TIMESTEPS,
        "beta_end": cfg.BETA_END,
        "labels": LABELS,
        "num_per_label": NUM_PER_LABEL,
        "batch_size": GEN_BATCH_SIZE,
        "cond_weight": COND_WEIGHT,
        "base_seed": SEED,
        "num_generation_sets": NUM_GENERATION_SETS,
        "seed_mode": "same initial seed only once; no per-set reseeding",
        "nucleotides": NUCLEOTIDES,
        "method": (
            "Enhanced structure prior with training innovations: "
            "base preference + importance map, time/label-aware 4-layer Prior Transformer, "
            "mid-only weak-gated cross-attention, position-weighted denoising loss, "
            "adjacent competitive denoising loss."
        ),
        "main_output_format": "CSV with only one column: sequence",
    }

    with open(
        os.path.join(set_out_dir, "generation_config.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[Set {set_id:02d}] Summary saved: {summary_path}")

    return summary_df


# =========================================================
# 10. Main
# =========================================================

def main():
    seed_everything(SEED)
    ensure_dir(GEN_OUT_ROOT)

    print("=" * 80)
    print("[Load checkpoint]")
    print(CHECKPOINT)
    print("=" * 80)

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    ckpt = safe_torch_load(CHECKPOINT, DEVICE)

    cfg = cfg_from_checkpoint(ckpt)
    cfg.DEVICE = DEVICE

    model = build_model(cfg).to(DEVICE)

    state_key = "model_state_dict" if USE_RAW_MODEL else "ema_state_dict"

    if state_key not in ckpt:
        print(f"[Warning] {state_key} not found. Use model_state_dict instead.")
        state_key = "model_state_dict"

    model.load_state_dict(ckpt[state_key], strict=True)
    model.eval()

    schedule = DiffusionSchedule(
        timesteps=cfg.TIMESTEPS,
        beta_end=cfg.BETA_END,
        device=DEVICE,
    )

    position_weights = load_class_position_weights(cfg, device=DEVICE)
    base_pref_table = load_class_base_preference(cfg, device=DEVICE)
    delta_pref_table = build_adjacent_delta_pref_table(base_pref_table, cfg)

    print("=" * 80)
    print("[Generation Config]")
    print(f"Checkpoint:                 {CHECKPOINT}")
    print(f"Model state:                {state_key}")
    print(f"Sequence length:            {cfg.SEQ_LEN}")
    print(f"Labels:                     {LABELS}")
    print(f"Num per label:              {NUM_PER_LABEL}")
    print(f"Num generation sets:        {NUM_GENERATION_SETS}")
    print(f"Batch size:                 {GEN_BATCH_SIZE}")
    print(f"Cond weight:                {COND_WEIGHT}")
    print(f"Base seed:                  {SEED}")
    print(f"Seed mode:                  same initial seed, no per-set reseeding")
    print(f"Device:                     {DEVICE}")
    print(f"Output root:                {GEN_OUT_ROOT}")
    print("-" * 80)
    print(f"Base dim:                   {getattr(cfg, 'BASE_DIM', None)}")
    print(f"Base pref condition:        {getattr(cfg, 'USE_BASE_PREF_CONDITION', None)}")
    print(f"Importance map:             {getattr(cfg, 'USE_POSITION_IMPORTANCE_WEIGHT', None)}")
    print(f"Importance gate:            {getattr(cfg, 'USE_IMPORTANCE_ORDERED_GATE', None)}")
    print(f"Adj-diff prior:             {getattr(cfg, 'USE_ADJ_DIFF_PRIOR', None)}")
    print(f"Prior token encoder:        {getattr(cfg, 'USE_PRIOR_TOKEN_ENCODER', None)}")
    print(f"Prior token layers:         {getattr(cfg, 'PRIOR_TOKEN_LAYERS', None)}")
    print(f"Prior token time/label emb: {getattr(cfg, 'PRIOR_TOKEN_USE_TIME_LABEL_EMB', None)}")
    print(f"Cross-attn downs:           {getattr(cfg, 'USE_CROSS_ATTN_IN_DOWNS', None)}")
    print(f"Cross-attn mid:             {getattr(cfg, 'USE_CROSS_ATTN_IN_MID', None)}")
    print(f"Cross-attn ups:             {getattr(cfg, 'USE_CROSS_ATTN_IN_UPS', None)}")
    print(f"Cross-attn tanh gate:       True")
    print(f"Cross-attn scale init:      {getattr(cfg, 'CROSS_ATTN_SCALE_INIT', None)}")
    print(f"Adj competitive:            {getattr(cfg, 'USE_ADJ_COMPETITIVE_DENOISE', None)}")
    print(f"Adj weight:                 {getattr(cfg, 'ADJ_COMPETITIVE_WEIGHT', None)}")
    print("=" * 80)

    all_set_summaries = []

    for set_id in range(1, NUM_GENERATION_SETS + 1):
        set_seed = SEED
        set_out_dir = os.path.join(GEN_OUT_ROOT, f"set_{set_id:02d}")

        summary_df = generate_one_set(
            set_id=set_id,
            set_seed=set_seed,
            set_out_dir=set_out_dir,
            model=model,
            schedule=schedule,
            position_weights=position_weights,
            base_pref_table=base_pref_table,
            delta_pref_table=delta_pref_table,
            cfg=cfg,
            state_key=state_key,
        )

        all_set_summaries.append(summary_df)

    if all_set_summaries:
        all_summary_df = pd.concat(all_set_summaries, ignore_index=True)
        all_summary_path = os.path.join(GEN_OUT_ROOT, "generation_summary_all_sets.csv")
        all_summary_df.to_csv(all_summary_path, index=False)

        numeric_cols = ["n", "unique", "diversity", "mean_gc"]
        agg_rows = []

        for col in numeric_cols:
            values = pd.to_numeric(all_summary_df[col], errors="coerce").dropna().values
            if len(values) > 0:
                agg_rows.append(
                    {
                        "metric": col,
                        "mean_across_all_labels_and_sets": float(np.mean(values)),
                        "std_across_all_labels_and_sets": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "var_across_all_labels_and_sets": float(np.var(values, ddof=1)) if len(values) > 1 else 0.0,
                        "n": int(len(values)),
                    }
                )

        pd.DataFrame(agg_rows).to_csv(
            os.path.join(GEN_OUT_ROOT, "generation_summary_mean_std_all_sets.csv"),
            index=False,
        )

        print("=" * 80)
        print("[Done]")
        print(f"Output root: {GEN_OUT_ROOT}")
        print(f"All-set generation summary: {all_summary_path}")
        print("=" * 80)
        print(all_summary_df)


if __name__ == "__main__":
    main()