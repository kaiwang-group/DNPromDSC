import os
from pathlib import Path

# To select a GPU, uncomment the next line before importing torch.
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import re
import sys
import random
from dataclasses import fields
from collections import Counter
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch


# =========================================================
# 0. Predictor import
# =========================================================

try:
    from model_optimized_predict import TransformerHybridMultiTask
except ImportError:
    print("Error: model_optimized_predict.py was not found. Make sure it is in the script directory or PYTHONPATH.")
    sys.exit(1)


# =========================================================
# 1. Import the current UNet training code
# =========================================================

from train_diff_all import (
    Config,
    build_model,
    DiffusionSchedule,
    seed_everything,
    ensure_dir,
    NUCLEOTIDES,
    extract,
    load_class_position_weights,
    load_class_base_preference,
    build_adjacent_delta_pref_table,
    make_base_pref_batch,
    make_delta_pref_batch,
    make_importance_gate,
    make_importance_map_batch,
)


# =========================================================
# 2. Paths and global configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = str(PROJECT_ROOT / "Diffusion_to_optimized" / "results_diff" / "checkpoints" / "gen_all.pt")

OUT_ROOT = str(PROJECT_ROOT / "Diffusion_to_optimized" / "results_optimized_10sets1")
OUT_DIR = OUT_ROOT

NUM_OPTIMIZATION_SETS = 10

DATASET_PATH = str(PROJECT_ROOT / "dataset" / "wrc_class5_63468.txt")

TARGET_LABELS = [0, 1, 2, 3, 4]
NUM_CLASSES = 5
NUM_LABELS = 5
SEQ_LEN = 80

POOL_SIZE = 2000
NUM_PER_LABEL = 2000

K = 6
KMERS_PER_SEQ = SEQ_LEN - K + 1
DIM_K = 4 ** K

SEED = 42
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

USE_RAW_MODEL = False
FORCE_SLIM_FLAGS = True

CFG_BASE = 1.0
CFG_SOFT = 1.5

MAX_ROUNDS = 80

BATCH_SIZE_UNET = 128
BATCH_SIZE_PRED = 256
BATCH_SIZE_REPAINT = 48

RESERVOIR_FILL = 512
RESERVOIR_EVOLVE = 384

TOP_K_REPLACE = 48

FILL_PCC_TOL = 0.0
FILL_GCERR_TOL = 0.0

BEST_PCC_MARGIN = 0.0
BEST_GCERR_MARGIN = 0.0
REQUIRE_BETTER_THAN_FILL = False

MIN_CLASS_GAIN = 1e-4
MIN_GC_GAIN = -5e-5
MIN_PCC_GAIN = 0.0

RELAXED_CLASS_GAIN = 0.010

REPAINT_START_RATIO = 0.45

GC_ABS_MAX_FILL = 0.10
GC_ABS_MAX_EVOLVE = 0.06
MAX_BASE_FRAC = 0.70
MAX_HOMOPOLY_RUN = 10

SAVE_ALL_CSV = True


# =========================================================
# 3. all_fusion predictor configuration: new 6-property PseDNC evaluator
# =========================================================

PREDICTOR_MODEL_TAG = "tf2_h8"
PREDICTOR_EXP_NAME = "all_fusion"

PREDICTOR_NUM_LAYERS = 2
PREDICTOR_NHEAD = 8
PREDICTOR_USE_CGR = True
PREDICTOR_USE_PSEDNC = True
PREDICTOR_PSEDNC_DIM = 22

PREDICTOR_BASE_ROOT = str(PROJECT_ROOT / "prediction" / "checkpoint_prediction")

PREDICTOR_CHECKPOINTS = [
    os.path.join(PREDICTOR_BASE_ROOT, f"prediction_f{i}.pth")
    for i in range(1, 6)
]

CHECK_FEATURE_COMPATIBILITY = True
K_FOLDS_ROOT = str(PROJECT_ROOT / "dataset" / "wrc_processed_5fold" / "kfold5_new")
FEATURE_CHECK_FOLD = 1
FEATURE_CHECK_N = 20
FEATURE_CHECK_WARN_MAE = 1e-5


# =========================================================
# 4. CGR / PseDNC: new 6-property PseDNC
# =========================================================

CGR_BINS = 16
LAMBDA = 6
WEIGHT = 0.05

DINUC = [
    "AA", "AC", "AG", "AT",
    "CA", "CC", "CG", "CT",
    "GA", "GC", "GG", "GT",
    "TA", "TC", "TG", "TT",
]

DN_MAP = {dn: i for i, dn in enumerate(DINUC)}

PHYCHEM = {
    "Twist": [
         0.06,  1.50,  0.78,  1.07,
        -1.38,  0.06, -1.66,  0.78,
        -0.08, -0.08,  0.06,  1.50,
        -1.23, -0.08, -1.38,  0.06,
    ],
    "Tilt": [
         0.50,  0.50,  0.36,  0.22,
        -1.36,  1.08, -1.22,  0.36,
         0.50,  0.22,  1.08,  0.50,
        -2.37,  0.50, -1.36,  0.50,
    ],
    "Roll": [
         0.27,  0.80,  0.09,  0.62,
        -0.27,  0.09, -0.44,  0.09,
         0.27,  1.33,  0.09,  0.80,
        -0.44,  0.27, -0.27,  0.27,
    ],
    "Shift": [
         1.59,  0.13,  0.68, -1.02,
        -0.86,  0.56, -0.82,  0.68,
         0.13, -0.35,  0.56,  0.13,
        -2.24,  0.13, -0.86,  1.59,
    ],
    "Slide": [
         0.11,  1.29, -0.24,  2.51,
        -0.62, -0.82, -0.29, -0.24,
        -0.39,  0.65, -0.82,  1.29,
        -1.51, -0.39, -0.62,  0.11,
    ],
    "Rise": [
        -0.11,  1.04, -0.62,  1.17,
        -1.25,  0.24, -1.39, -0.62,
         0.71,  1.59,  0.24,  1.04,
        -1.39,  0.71, -1.25, -0.11,
    ],
}

PHYCHEM_KEYS = [
    "Twist",
    "Tilt",
    "Roll",
    "Shift",
    "Slide",
    "Rise",
]


def clean_seq(seq: str, max_len: int = SEQ_LEN) -> str:
    seq = str(seq).upper().replace("U", "T").replace(" ", "").replace("N", "")
    seq = "".join(ch for ch in seq if ch in "ACGT")
    if len(seq) > max_len:
        seq = seq[:max_len]
    return seq


def get_psednc(seq, lam=LAMBDA, w=WEIGHT):
    """
    Standard six-property PseDNC-like descriptor.
    Output: 16 dinucleotide frequencies plus lam correlation factors.
    When lam=6, the output has 22 dimensions, not 16 + 6 * lam.
    At each lag k, average the six structural properties into one theta_k.
    """
    seq = clean_seq(seq, max_len=SEQ_LEN)
    L = len(seq)

    freqs = np.zeros(16, dtype=np.float32)

    if L < 2:
        return np.zeros(16 + lam, dtype=np.float32)

    for i in range(L - 1):
        dn = seq[i:i + 2]
        if dn in DN_MAP:
            freqs[DN_MAP[dn]] += 1.0

    freqs = freqs / max(L - 1, 1)

    vals = {key: [] for key in PHYCHEM_KEYS}

    for i in range(L - 1):
        dn = seq[i:i + 2]
        if dn in DN_MAP:
            idx = DN_MAP[dn]
            for key in PHYCHEM_KEYS:
                vals[key].append(PHYCHEM[key][idx])

    corrs = []

    if len(vals[PHYCHEM_KEYS[0]]) > lam:
        for k in range(1, lam + 1):
            c = 0.0
            for key in PHYCHEM_KEYS:
                v = np.asarray(vals[key], dtype=np.float32)
                c += np.mean((v[:-k] - v.mean()) * (v[k:] - v.mean()))
            corrs.append(c / float(len(PHYCHEM_KEYS)))
    else:
        corrs = [0.0] * lam

    denom = 1.0 + w * float(np.sum(corrs))

    if abs(denom) < 1e-12:
        denom = 1.0

    vec = [f / denom for f in freqs] + [w * t / denom for t in corrs]

    return np.asarray(vec, dtype=np.float32)


def compute_batch_psednc(seqs):
    return np.stack([get_psednc(s, LAMBDA, WEIGHT) for s in seqs]).astype(np.float32)


def seq_to_fcgr(seq, k):
    base_map = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
    }

    seq = clean_seq(seq, max_len=SEQ_LEN)

    N = 2 ** k
    matrix = np.zeros((N, N), dtype=np.float32)

    if len(seq) < k:
        return matrix

    for i in range(len(seq) - k + 1):
        sub = seq[i:i + k]
        row, col = 0, 0
        valid = True

        for j, char in enumerate(sub):
            if char not in base_map:
                valid = False
                break

            step = 2 ** (k - 1 - j)

            if char in ["G", "T"]:
                row += step
            if char in ["C", "G"]:
                col += step

        if valid:
            matrix[row, col] += 1.0

    if matrix.sum() > 0:
        matrix /= matrix.sum()

    return matrix.astype(np.float32)


def compute_batch_multires_cgr(seqs):
    batch_cgr = []

    for s in seqs:
        c4 = seq_to_fcgr(s, 4)
        c3 = seq_to_fcgr(s, 3)
        c3_up = c3.repeat(2, axis=0).repeat(2, axis=1)
        stacked = np.stack([c4, c3_up], axis=0)
        batch_cgr.append(stacked)

    return np.array(batch_cgr, dtype=np.float32)


def seq_to_one_hot_predictor(seq, max_len=SEQ_LEN):
    mapping = {
        "A": 1,
        "C": 2,
        "G": 3,
        "T": 4,
    }

    channels = 5
    one_hot = np.zeros((max_len, channels), dtype=np.float32)
    seq = clean_seq(seq, max_len=max_len)

    for i, char in enumerate(seq):
        if char in mapping:
            idx = mapping[char]
            if idx < channels:
                one_hot[i, idx] = 1.0

    return one_hot


# =========================================================
# 5. Basic utilities
# =========================================================

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def torch_load_safely(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k[len("module."):] if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }

    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {
            k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
            for k, v in state_dict.items()
        }

    return state_dict


def cfg_from_checkpoint(ckpt) -> Config:
    cfg = Config()
    saved_cfg = ckpt.get("config", {}) or {}
    valid_keys = {f.name for f in fields(Config)}

    for k, v in saved_cfg.items():
        if k in valid_keys:
            setattr(cfg, k, v)

    if FORCE_SLIM_FLAGS:
        if hasattr(cfg, "USE_SPATIAL_PRIOR_FILM"):
            cfg.USE_SPATIAL_PRIOR_FILM = False
        if hasattr(cfg, "USE_FILM_IN_DOWNS"):
            cfg.USE_FILM_IN_DOWNS = False
        if hasattr(cfg, "USE_FILM_IN_MID"):
            cfg.USE_FILM_IN_MID = False
        if hasattr(cfg, "USE_BOUNDARY_RESIDUAL_HEAD"):
            cfg.USE_BOUNDARY_RESIDUAL_HEAD = False

    return cfg


def gc_of_seq(seq: str) -> float:
    seq = clean_seq(seq)
    return (seq.count("G") + seq.count("C")) / max(1, len(seq))


def gc_score(gc: float, target_gc: float, sigma: float = 0.06) -> float:
    return float(np.exp(-((gc - target_gc) ** 2) / (2 * (sigma ** 2) + 1e-12)))


def low_complexity_bad(seq: str, max_base_frac=0.70, max_run=10) -> bool:
    seq = clean_seq(seq)

    if not seq:
        return True

    L = len(seq)
    c = Counter(seq)

    max_frac = max(
        c.get("A", 0),
        c.get("C", 0),
        c.get("G", 0),
        c.get("T", 0),
    ) / max(1, L)

    if max_frac > max_base_frac:
        return True

    run = 1
    best = 1

    for i in range(1, L):
        if seq[i] == seq[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1

    return best > max_run


def check_feature_compatibility():
    if not CHECK_FEATURE_COMPATIBILITY:
        return

    fold_dir = os.path.join(K_FOLDS_ROOT, f"fold{FEATURE_CHECK_FOLD}")
    val_csv = os.path.join(fold_dir, "val.csv")
    cgr_path = os.path.join(fold_dir, "fcgr_multires_k4k3_val.npy")
    pse_path = os.path.join(fold_dir, f"psednc_lam{LAMBDA}_val.npy")

    if not os.path.exists(val_csv) or not os.path.exists(cgr_path) or not os.path.exists(pse_path):
        print("[FeatureCheck] skipped: val.csv / CGR / PseDNC npy not found.")
        return

    try:
        df = pd.read_csv(val_csv)

        if "seq" not in df.columns:
            print("[FeatureCheck] skipped: val.csv has no seq column.")
            return

        seqs = df["seq"].astype(str).tolist()[:FEATURE_CHECK_N]

        cgr_ref = np.load(cgr_path)[:len(seqs)]
        pse_ref = np.load(pse_path)[:len(seqs)]

        cgr_new = compute_batch_multires_cgr(seqs)
        pse_new = compute_batch_psednc(seqs)

        cgr_mae = float(np.mean(np.abs(cgr_new - cgr_ref)))
        pse_mae = float(np.mean(np.abs(pse_new - pse_ref)))

        print("\n" + "=" * 100)
        print("[Feature Compatibility Check]")
        print("=" * 100)
        print(f"Fold: fold{FEATURE_CHECK_FOLD} | N={len(seqs)}")
        print(f"CGR shape: new={cgr_new.shape}, ref={cgr_ref.shape}, MAE={cgr_mae:.8e}")
        print(f"Pse shape: new={pse_new.shape}, ref={pse_ref.shape}, MAE={pse_mae:.8e}")

        if cgr_mae > FEATURE_CHECK_WARN_MAE or pse_mae > FEATURE_CHECK_WARN_MAE:
            print("Warning: Computed CGR/PseDNC features differ from the existing npy files.")
        else:
            print("Feature check passed.")

        print("=" * 100 + "\n")

    except Exception as e:
        print(f"[FeatureCheck] failed but continue: {e}")


# =========================================================
# 6. k-mer / natural reference
# =========================================================

_MAP_ARR = np.full(256, -1, dtype=np.int16)
_MAP_ARR[ord("A")] = 0
_MAP_ARR[ord("C")] = 1
_MAP_ARR[ord("G")] = 2
_MAP_ARR[ord("T")] = 3


def kmer_counts_batch_fast(seqs: List[str], k: int) -> np.ndarray:
    dim = 4 ** k
    out = np.zeros((len(seqs), dim), dtype=np.float32)
    base = 4 ** (k - 1)

    for b, s in enumerate(seqs):
        s = clean_seq(s)

        if len(s) < k:
            continue

        x = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
        v = _MAP_ARR[x]

        if np.any(v < 0):
            continue

        idx = 0

        for i in range(k):
            idx = idx * 4 + int(v[i])

        out[b, idx] += 1.0

        for i in range(k, len(v)):
            idx = (idx - int(v[i - k]) * base) * 4 + int(v[i])
            out[b, idx] += 1.0

    return out


def centered_pearson_to_ref(vec_freq, ref_c, ref_norm, eps=1e-8) -> float:
    v = vec_freq.astype(np.float32)
    v_c = v - v.mean()
    v_n = np.linalg.norm(v_c) + eps
    return float((v_c @ ref_c) / (v_n * ref_norm))


def centered_pearson_sim_batch(mat_freq, ref_freq, eps=1e-8) -> np.ndarray:
    if mat_freq.size == 0:
        return np.zeros((0,), dtype=np.float32)

    x = mat_freq.astype(np.float32)
    r = ref_freq.astype(np.float32)

    r_c = r - r.mean()
    r_n = np.linalg.norm(r_c) + eps

    x_c = x - x.mean(axis=1, keepdims=True)
    x_n = np.linalg.norm(x_c, axis=1) + eps

    dot = x_c @ r_c

    return (dot / (x_n * r_n)).astype(np.float32)


def parse_dataset_by_label(txt_path: str, labels: List[int]) -> Dict[int, List[str]]:
    grouped = {lb: [] for lb in labels}

    if not os.path.exists(txt_path):
        print(f"[Warning] DATASET_PATH not found: {txt_path}")
        return grouped

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#") or line.startswith(">"):
                continue

            toks = line.split()

            if len(toks) < 2:
                continue

            seq = clean_seq(toks[0])

            try:
                lb = int(re.findall(r"-?\d+", toks[1])[0])
            except Exception:
                continue

            if lb in grouped and len(seq) == SEQ_LEN:
                grouped[lb].append(seq)

    return grouped


def build_nat_ref_and_pool(dataset_path: str, labels: List[int], k: int = 6):
    grouped = parse_dataset_by_label(dataset_path, labels)
    ref_gc = {}
    ref_kfreq = {}

    dim = 4 ** k

    for lb in labels:
        seqs = grouped.get(lb, [])

        if len(seqs) == 0:
            ref_gc[lb] = 0.40
            ref_kfreq[lb] = np.zeros((dim,), dtype=np.float32)
            continue

        gcs = [gc_of_seq(s) for s in seqs]
        ref_gc[lb] = float(np.mean(gcs))

        sum_counts = np.zeros((dim,), dtype=np.float64)
        total = 0.0

        chunk = 512

        for i in range(0, len(seqs), chunk):
            batch = seqs[i:i + chunk]
            mat = kmer_counts_batch_fast(batch, k).astype(np.float64)
            sum_counts += mat.sum(axis=0)
            total += float(len(batch) * (SEQ_LEN - k + 1))

        if total > 0:
            ref_kfreq[lb] = (sum_counts / total).astype(np.float32)
        else:
            ref_kfreq[lb] = np.zeros((dim,), dtype=np.float32)

    return grouped, ref_gc, ref_kfreq


# =========================================================
# 7. all_fusion predictor
# =========================================================

def load_allfusion_predictors(device):
    print(f"[Load {PREDICTOR_MODEL_TAG}/{PREDICTOR_EXP_NAME} predictor]")

    models = []

    for path in PREDICTOR_CHECKPOINTS:
        if not os.path.exists(path):
            print(f"[Warning] Predictor checkpoint not found, skip: {path}")
            continue

        model = TransformerHybridMultiTask(
            input_size=5,
            hidden_size=256,
            num_classes=NUM_LABELS,
            conv_kernels=(5, 5, 3),
            num_layers=PREDICTOR_NUM_LAYERS,
            nhead=PREDICTOR_NHEAD,
            se_reduction=16,
            use_cgr=PREDICTOR_USE_CGR,
            use_psednc=PREDICTOR_USE_PSEDNC,
            psednc_dim=PREDICTOR_PSEDNC_DIM,
            mask_prob=0.0,
            pse_scale_init=0.05,
        ).to(device)

        try:
            ckpt = torch_load_safely(path, map_location=device)

            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt

            state_dict = strip_module_prefix(state_dict)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            models.append(model)

            print(f"[OK] Loaded predictor: {os.path.basename(path)}")

        except Exception as e:
            print(f"[Error] Failed to load predictor {os.path.basename(path)}: {e}")

    print(f"[Predictor] loaded folds={len(models)}")

    if len(models) == 0:
        raise RuntimeError(
            f"Failed to load any {PREDICTOR_MODEL_TAG}/{PREDICTOR_EXP_NAME} predictors. Check the checkpoint paths."
        )

    return models


@torch.no_grad()
def predict_probs_allfusion(models, sequences: List[str], device, batch_size=BATCH_SIZE_PRED) -> torch.Tensor:
    if len(sequences) == 0:
        return torch.zeros((0, NUM_CLASSES), device=device)

    all_probs = []

    for st in range(0, len(sequences), batch_size):
        ed = min(len(sequences), st + batch_size)
        batch_raw = sequences[st:ed]

        clean_batch = []
        batch_x = []

        for seq in batch_raw:
            seq = clean_seq(seq, max_len=SEQ_LEN)

            if len(seq) == 0:
                continue

            clean_batch.append(seq)
            batch_x.append(seq_to_one_hot_predictor(seq, max_len=SEQ_LEN))

        if len(batch_x) == 0:
            continue

        batch_cgr = compute_batch_multires_cgr(clean_batch)
        batch_pse = compute_batch_psednc(clean_batch)

        x = torch.tensor(np.array(batch_x), dtype=torch.float32, device=device)
        cgr_tensor = torch.tensor(batch_cgr, dtype=torch.float32, device=device)
        pse_tensor = torch.tensor(batch_pse, dtype=torch.float32, device=device)

        fold_probs = []

        for m in models:
            logits = m(
                x,
                extra_feat=cgr_tensor,
                psednc_feat=pse_tensor,
            )
            probs = torch.softmax(logits.float(), dim=1)
            fold_probs.append(probs)

        avg_probs = torch.stack(fold_probs, dim=0).mean(dim=0)
        all_probs.append(avg_probs.detach())

        del x, cgr_tensor, pse_tensor, fold_probs, avg_probs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_probs) == 0:
        return torch.zeros((0, NUM_CLASSES), device=device)

    return torch.cat(all_probs, dim=0)


# =========================================================
# 8. UNet DDPM sampling
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

    delta_pref_double = make_delta_pref_batch(
        classes_for_pref=classes_double,
        class_delta_pref=delta_pref_table,
        class_position_weights=position_weights,
        x_like=x_double,
        cfg=cfg,
    )

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

    sqrt_recip_alphas_t = extract(schedule.sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * eps / (sqrt_one_minus_alphas_cumprod_t + 1e-8)
    )

    if t_index == 0:
        return model_mean

    posterior_variance_t = extract(schedule.posterior_variance, t, x.shape)
    noise = torch.randn_like(x)

    return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_unet_tensor(
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
) -> torch.Tensor:
    device = next(model.parameters()).device

    img = torch.randn(
        (batch_size, channels, 4, image_size),
        device=device,
    )

    for i in reversed(range(schedule.timesteps)):
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


def decode_tensor_to_sequence(x: torch.Tensor, nucleotides: List[str] = NUCLEOTIDES) -> str:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.squeeze(x)

    if x.shape[0] != 4 and x.shape[-1] == 4:
        x = x.T

    idx = np.argmax(x, axis=0)

    return "".join(nucleotides[int(i)] for i in idx)


@torch.no_grad()
def generate_unet_candidates(
    model,
    schedule,
    target_label: int,
    n_generate: int,
    cond_weight: float,
    position_weights,
    base_pref_table,
    delta_pref_table,
    cfg,
    device,
) -> List[str]:
    seqs = []
    internal_cond_label = target_label + 1

    while len(seqs) < n_generate:
        cur_bs = min(BATCH_SIZE_UNET, n_generate - len(seqs))

        classes = torch.full(
            (cur_bs,),
            internal_cond_label,
            dtype=torch.long,
            device=device,
        )

        samples = sample_unet_tensor(
            model=model,
            schedule=schedule,
            image_size=cfg.SEQ_LEN,
            classes=classes,
            batch_size=cur_bs,
            channels=1,
            cond_weight=cond_weight,
            position_weights=position_weights,
            base_pref_table=base_pref_table,
            delta_pref_table=delta_pref_table,
            cfg=cfg,
        )

        seqs.extend([
            decode_tensor_to_sequence(samples[j])
            for j in range(samples.shape[0])
        ])

        del samples, classes

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return seqs


# =========================================================
# 9. Masked repaint
# =========================================================

def seqs_to_unet_onehot(seqs: List[str], seq_len: int, device) -> torch.Tensor:
    base_to_idx = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
    }

    arr = np.zeros((len(seqs), 1, 4, seq_len), dtype=np.float32)

    for i, seq in enumerate(seqs):
        s = clean_seq(seq)[:seq_len]

        for j, ch in enumerate(s):
            if ch in base_to_idx:
                arr[i, 0, base_to_idx[ch], j] = 1.0

    return torch.tensor(arr, dtype=torch.float32, device=device)


def get_schedule_sqrt_alphas_cumprod(schedule, device):
    if hasattr(schedule, "sqrt_alphas_cumprod"):
        return schedule.sqrt_alphas_cumprod

    if hasattr(schedule, "alphas_cumprod"):
        return torch.sqrt(schedule.alphas_cumprod)

    if hasattr(schedule, "betas"):
        betas = schedule.betas.to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return torch.sqrt(alphas_cumprod)

    raise AttributeError("schedule is missing sqrt_alphas_cumprod / alphas_cumprod / betas; cannot perform q_sample.")


def q_sample(schedule, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    device = x_start.device
    sqrt_alphas_cumprod = get_schedule_sqrt_alphas_cumprod(schedule, device)
    sqrt_alpha = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus = extract(schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alpha * x_start + sqrt_one_minus * noise


def get_position_importance(position_weights: torch.Tensor, target_label: int, seq_len: int) -> np.ndarray:
    pw = position_weights.detach().cpu().float().numpy()

    if pw.ndim == 1:
        arr = pw
    elif pw.ndim >= 2:
        idx = target_label + 1 if pw.shape[0] > NUM_CLASSES else target_label
        idx = min(idx, pw.shape[0] - 1)
        arr = pw[idx]

        while arr.ndim > 1:
            arr = arr.mean(axis=0)
    else:
        arr = np.ones((seq_len,), dtype=np.float32)

    arr = np.asarray(arr, dtype=np.float32).reshape(-1)

    if len(arr) < seq_len:
        arr = np.pad(arr, (0, seq_len - len(arr)), mode="edge")
    elif len(arr) > seq_len:
        arr = arr[:seq_len]

    arr = np.maximum(arr, 0.0)

    if float(arr.sum()) <= 1e-8:
        arr = np.ones((seq_len,), dtype=np.float32)

    arr = arr / (arr.sum() + 1e-8)

    return arr


def build_position_mask(
    batch_size: int,
    seq_len: int,
    target_label: int,
    mask_rate: float,
    position_weights: torch.Tensor,
    device,
    prior_ratio: float = 0.70,
) -> torch.Tensor:
    importance = get_position_importance(position_weights, target_label, seq_len)

    n_mask = max(1, int(round(seq_len * mask_rate)))
    n_prior = int(round(n_mask * prior_ratio))
    n_prior = min(n_prior, n_mask)

    masks = np.zeros((batch_size, seq_len), dtype=bool)
    all_pos = np.arange(seq_len)

    for b in range(batch_size):
        chosen = []

        if n_prior > 0:
            prior_pos = np.random.choice(
                all_pos,
                size=n_prior,
                replace=False,
                p=importance,
            )
            chosen.extend(prior_pos.tolist())

        chosen_set = set(chosen)
        remain = [p for p in all_pos.tolist() if p not in chosen_set]
        n_rand = n_mask - len(chosen)

        if n_rand > 0 and len(remain) > 0:
            rand_pos = np.random.choice(
                remain,
                size=min(n_rand, len(remain)),
                replace=False,
            )
            chosen.extend(rand_pos.tolist())

        masks[b, chosen] = True

    mask = torch.tensor(masks, dtype=torch.bool, device=device)
    mask = mask[:, None, None, :].expand(batch_size, 1, 4, seq_len)

    return mask


@torch.no_grad()
def masked_repaint_unet(
    model,
    schedule,
    parent_seqs: List[str],
    target_label: int,
    mask_rate: float,
    cond_weight: float,
    position_weights,
    base_pref_table,
    delta_pref_table,
    cfg,
    device,
) -> List[str]:
    if len(parent_seqs) == 0:
        return []

    out_seqs = []
    internal_cond_label = target_label + 1
    seq_len = cfg.SEQ_LEN

    for st in range(0, len(parent_seqs), BATCH_SIZE_REPAINT):
        ed = min(len(parent_seqs), st + BATCH_SIZE_REPAINT)
        batch = parent_seqs[st:ed]
        B = len(batch)

        x0 = seqs_to_unet_onehot(batch, seq_len=seq_len, device=device)

        mask = build_position_mask(
            batch_size=B,
            seq_len=seq_len,
            target_label=target_label,
            mask_rate=mask_rate,
            position_weights=position_weights,
            device=device,
            prior_ratio=0.70,
        )

        t_start = int(round((schedule.timesteps - 1) * REPAINT_START_RATIO))
        t_start = max(1, min(t_start, schedule.timesteps - 1))

        known_noise = torch.randn_like(x0)
        random_noise = torch.randn_like(x0)

        t0 = torch.full((B,), t_start, device=device, dtype=torch.long)
        x_known_t = q_sample(schedule, x0, t0, known_noise)

        x_t = torch.where(mask, random_noise, x_known_t)

        classes = torch.full(
            (B,),
            internal_cond_label,
            dtype=torch.long,
            device=device,
        )

        for i in reversed(range(t_start + 1)):
            t = torch.full((B,), i, device=device, dtype=torch.long)

            x_t = p_sample_guided(
                model=model,
                schedule=schedule,
                x=x_t,
                classes=classes,
                t=t,
                t_index=i,
                cond_weight=cond_weight,
                position_weights=position_weights,
                base_pref_table=base_pref_table,
                delta_pref_table=delta_pref_table,
                cfg=cfg,
            )

            if i > 0:
                t_prev = torch.full((B,), i - 1, device=device, dtype=torch.long)
                x_known_prev = q_sample(schedule, x0, t_prev, known_noise)
            else:
                x_known_prev = x0

            x_t = torch.where(mask, x_t, x_known_prev)

        out_seqs.extend([
            decode_tensor_to_sequence(x_t[j])
            for j in range(B)
        ])

        del x0, mask, known_noise, random_noise, x_known_t, x_t, classes

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out_seqs


# =========================================================
# 10. Action-level bandit
# =========================================================

ACTIONS = [
    {"name": "PoolLocalMutate", "n_mut": 1},
    {"name": "PoolLocalMutate", "n_mut": 2},
    {"name": "PoolLocalMutate", "n_mut": 3},

    {"name": "MaskedRepaint", "mask_rate": 0.10, "cond_weight": CFG_BASE},
    {"name": "MaskedRepaint", "mask_rate": 0.15, "cond_weight": CFG_BASE},
    {"name": "MaskedRepaint", "mask_rate": 0.20, "cond_weight": CFG_SOFT},

    {"name": "UNetRebirth", "cond_weight": CFG_BASE},
    {"name": "UNetRebirth", "cond_weight": CFG_SOFT},
]


class EpsGreedyActionBandit:
    def __init__(
        self,
        n_actions: int,
        eps_start=0.30,
        eps_end=0.10,
        decay=0.98,
        warmup_trials=2,
        stagnant_eps_boost=0.40,
    ):
        self.action_ids = list(range(n_actions))
        self.eps = float(eps_start)
        self.eps_end = float(eps_end)
        self.decay = float(decay)
        self.warmup_trials = int(warmup_trials)
        self.stagnant_eps_boost = float(stagnant_eps_boost)

        self.q = {a: 0.0 for a in self.action_ids}
        self.n = {a: 0 for a in self.action_ids}

    def select(self, global_stagnant_count: int = 0):
        under = [
            a for a in self.action_ids
            if self.n[a] < self.warmup_trials
        ]

        if len(under) > 0:
            min_n = min(self.n[a] for a in under)
            cand = [
                a for a in under
                if self.n[a] == min_n
            ]
            return random.choice(cand)

        eps_now = self.eps

        if global_stagnant_count >= 5:
            eps_now = max(eps_now, self.stagnant_eps_boost)

        if random.random() < eps_now:
            return random.choice(self.action_ids)

        max_q = max(self.q[a] for a in self.action_ids)

        best = [
            a for a in self.action_ids
            if abs(self.q[a] - max_q) < 1e-12
        ]

        return random.choice(best)

    def update(self, a, reward):
        self.n[a] += 1
        lr = 1.0 / float(self.n[a])
        self.q[a] = (1.0 - lr) * self.q[a] + lr * float(reward)
        self.eps = max(self.eps_end, self.eps * self.decay)

    def state_str(self):
        parts = []

        for a in self.action_ids:
            act = ACTIONS[a]

            if act["name"] == "PoolLocalMutate":
                name = f"mut{act['n_mut']}"
            elif act["name"] == "MaskedRepaint":
                name = f"mask{act['mask_rate']}_cfg{act['cond_weight']}"
            else:
                name = f"rebirth_cfg{act['cond_weight']}"

            parts.append(f"{a}:{name},q={self.q[a]:+.4f},n={self.n[a]}")

        return " | ".join(parts) + f" | eps={self.eps:.3f}"


# =========================================================
# 11. Local mutation and parent selection
# =========================================================

def mutate_seq_light(seq: str, target_gc: float, n_mut: int = 2) -> str:
    s = list(clean_seq(seq))

    if len(s) == 0:
        return "A" * SEQ_LEN

    L = len(s)
    gc_cnt = s.count("G") + s.count("C")
    changed = set()

    for _ in range(n_mut):
        pos = random.randrange(L)

        for _try in range(8):
            if pos not in changed:
                break
            pos = random.randrange(L)

        changed.add(pos)

        old = s[pos]
        cur_gc = gc_cnt / float(L)

        if cur_gc < target_gc:
            cand = ["G", "C", "A", "T"]
        else:
            cand = ["A", "T", "G", "C"]

        nb = old

        for _try in range(8):
            nb = random.choice(cand)
            if nb != old:
                break

        if old in "GC":
            gc_cnt -= 1

        if nb in "GC":
            gc_cnt += 1

        s[pos] = nb

    return "".join(s)


def select_parent_for_mutate(pool: List[Dict[str, Any]], target_label: int, n_parent: int) -> List[str]:
    if len(pool) == 0:
        return []

    close = [
        x for x in pool
        if (
            x["pred"] == target_label
            or abs(int(x["pred"]) - int(target_label)) == 1
            or x["prob"] > 0.25
        )
    ]

    if len(close) == 0:
        close = sorted(pool, key=lambda x: x["score"], reverse=True)[:min(500, len(pool))]
    else:
        close = sorted(close, key=lambda x: x["score"], reverse=True)[:min(800, len(close))]

    return [
        random.choice(close)["sequence"]
        for _ in range(n_parent)
    ]


def select_parent_for_mask(
    pool: List[Dict[str, Any]],
    target_label: int,
    n_parent: int,
    rescue_mode: bool,
) -> List[str]:
    if len(pool) == 0:
        return []

    if rescue_mode:
        cand = [
            x for x in pool
            if x["pred"] != target_label
            and abs(int(x["pred"]) - int(target_label)) == 1
        ]
    else:
        cand = [
            x for x in pool
            if x["pred"] == target_label
            and x["margin"] < 0.15
        ]

    if len(cand) == 0:
        cand = sorted(pool, key=lambda x: x["score"], reverse=True)[:min(500, len(pool))]
    else:
        cand = sorted(cand, key=lambda x: x["score"], reverse=True)[:min(800, len(cand))]

    return [
        random.choice(cand)["sequence"]
        for _ in range(n_parent)
    ]


# =========================================================
# 12. Scoring and pool stats
# =========================================================

def label_metrics_from_prob(prob_vec: np.ndarray, target_label: int):
    p_target = float(prob_vec[target_label])
    pred_label = int(np.argmax(prob_vec))

    other = [
        i for i in range(NUM_CLASSES)
        if i != target_label
    ]

    max_other = float(np.max(prob_vec[other]))
    margin = p_target - max_other

    adj = []

    if target_label - 1 >= 0:
        adj.append(target_label - 1)

    if target_label + 1 < NUM_CLASSES:
        adj.append(target_label + 1)

    if len(adj) > 0:
        max_adj = float(np.max(prob_vec[adj]))
        adj_margin = p_target - max_adj
    else:
        adj_margin = margin

    return p_target, margin, adj_margin, pred_label


def score_candidate_item(
    seq: str,
    prob_vec: np.ndarray,
    target_label: int,
    target_gc: float,
    self_pcc: float,
    gc_err: float,
) -> Dict[str, Any]:
    p_target, margin, adj_margin, pred_label = label_metrics_from_prob(
        prob_vec,
        target_label,
    )

    s_gc = gc_score(gc_of_seq(seq), target_gc)

    class_score = 0.7 * p_target + 0.3 * margin
    dist_score = 0.4 * s_gc + 0.6 * float(self_pcc)

    score = 0.50 * class_score + 0.35 * dist_score

    if pred_label == target_label:
        score += 0.08
    elif abs(pred_label - target_label) == 1:
        score -= 0.04
    else:
        score -= 0.12

    return {
        "sequence": seq,
        "prob": float(p_target),
        "margin": float(margin),
        "adj_margin": float(adj_margin),
        "pred": int(pred_label),
        "gc": float(gc_of_seq(seq)),
        "gc_err": float(gc_err),
        "self_pcc": float(self_pcc),
        "class_score": float(class_score),
        "dist_score": float(dist_score),
        "score": float(score),
    }


def build_reservoir_from_seqs(
    seqs: List[str],
    target_label: int,
    target_gc: float,
    ref_freq: np.ndarray,
    predictors,
    device,
    seen: set,
    nat_set: set,
    phase: str,
) -> List[Dict[str, Any]]:
    clean = []

    for s in seqs:
        s = clean_seq(s)

        if len(s) >= SEQ_LEN:
            s = s[:SEQ_LEN]

        if len(s) == SEQ_LEN:
            clean.append(s)

    if len(clean) == 0:
        return []

    probs = predict_probs_allfusion(
        models=predictors,
        sequences=clean,
        device=device,
        batch_size=BATCH_SIZE_PRED,
    ).detach().cpu().numpy().astype(np.float32)

    if len(probs) != len(clean):
        raise RuntimeError(
            f"The number of predictor probabilities differs from the number of sequences: probs={len(probs)} seqs={len(clean)}"
        )

    gcs = np.array(
        [gc_of_seq(s) for s in clean],
        dtype=np.float32,
    )

    gc_errs = np.abs(gcs - target_gc).astype(np.float32)

    counts_mat = kmer_counts_batch_fast(clean, K)
    freq_mat = counts_mat / float(KMERS_PER_SEQ)
    self_pcc = centered_pearson_sim_batch(freq_mat, ref_freq)

    reservoir = []
    gc_max = GC_ABS_MAX_FILL if phase == "FILL" else GC_ABS_MAX_EVOLVE

    for j, s in enumerate(clean):
        if s in seen or s in nat_set:
            continue

        if low_complexity_bad(s, MAX_BASE_FRAC, MAX_HOMOPOLY_RUN):
            continue

        if float(gc_errs[j]) > gc_max:
            continue

        item = score_candidate_item(
            seq=s,
            prob_vec=probs[j],
            target_label=target_label,
            target_gc=target_gc,
            self_pcc=float(self_pcc[j]),
            gc_err=float(gc_errs[j]),
        )

        item["counts"] = counts_mat[j].astype(np.float64)
        reservoir.append(item)

    reservoir.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return reservoir


def pool_pcc6(
    pool_sum_counts: np.ndarray,
    pool_total: float,
    ref_c: np.ndarray,
    ref_norm: float,
) -> float:
    if pool_total <= 0:
        return 0.0

    freq = (pool_sum_counts / pool_total).astype(np.float32)

    return centered_pearson_to_ref(
        freq,
        ref_c,
        ref_norm,
    )


def pool_stats(
    pool: List[Dict[str, Any]],
    target_label: int,
    pool_sum_counts: np.ndarray,
    pool_total: float,
    ref_c: np.ndarray,
    ref_norm: float,
) -> Dict[str, float]:
    if len(pool) == 0:
        return {
            "acc": 0.0,
            "avg_prob": 0.0,
            "avg_margin": 0.0,
            "adj_leak": 1.0,
            "avg_gc_err": 1.0,
            "avg_gc_score": 0.0,
            "pool_pcc6": 0.0,
            "class_score": 0.0,
            "dist_score": 0.0,
            "diversity": 0.0,
        }

    preds = np.array(
        [x["pred"] for x in pool],
        dtype=np.int64,
    )

    probs = np.array(
        [x["prob"] for x in pool],
        dtype=np.float32,
    )

    margins = np.array(
        [x["margin"] for x in pool],
        dtype=np.float32,
    )

    gc_errs = np.array(
        [x["gc_err"] for x in pool],
        dtype=np.float32,
    )

    seqs = [
        x["sequence"]
        for x in pool
    ]

    pcc = pool_pcc6(
        pool_sum_counts,
        pool_total,
        ref_c,
        ref_norm,
    )

    acc = float(np.mean(preds == target_label))
    avg_prob = float(np.mean(probs))
    avg_margin = float(np.mean(margins))
    adj_leak = float(np.mean(np.abs(preds - target_label) == 1))
    avg_gc_err = float(np.mean(gc_errs))
    diversity = float(len(set(seqs)) / max(1, len(seqs)))

    avg_gc_score = float(np.clip(1.0 - avg_gc_err / 0.08, 0.0, 1.0))

    class_score = 0.7 * avg_prob + 0.3 * avg_margin
    dist_score = 0.4 * avg_gc_score + 0.6 * pcc

    return {
        "acc": acc,
        "avg_prob": avg_prob,
        "avg_margin": avg_margin,
        "adj_leak": adj_leak,
        "avg_gc_err": avg_gc_err,
        "avg_gc_score": avg_gc_score,
        "pool_pcc6": pcc,
        "class_score": float(class_score),
        "dist_score": float(dist_score),
        "diversity": diversity,
    }


def compute_reward(
    old_stats: Dict[str, float],
    new_stats: Dict[str, float],
    safe_accept_rate: float,
) -> float:
    delta_class = new_stats["class_score"] - old_stats["class_score"]
    delta_acc = new_stats["acc"] - old_stats["acc"]
    delta_gc = old_stats["avg_gc_err"] - new_stats["avg_gc_err"]
    delta_pcc = new_stats["pool_pcc6"] - old_stats["pool_pcc6"]

    reward = (
        0.25 * delta_class
        + 0.15 * delta_acc
        + 0.30 * delta_gc
        + 0.30 * delta_pcc
        + 0.03 * float(safe_accept_rate)
    )

    if delta_gc < 0:
        reward -= 0.40 * abs(delta_gc)

    if delta_pcc < 0:
        reward -= 0.80 * abs(delta_pcc)

    if safe_accept_rate <= 1e-8:
        reward -= 0.02

    return float(reward)


def optimization_objective(stats: Dict[str, float]) -> float:
    gc_quality = float(np.clip(1.0 - stats["avg_gc_err"] / 0.08, 0.0, 1.0))

    obj = (
        0.40 * stats["acc"]
        + 0.40 * stats["pool_pcc6"]
        + 0.20 * gc_quality
    )

    return float(obj)


def clone_pool(pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    copied = []

    for x in pool:
        y = dict(x)
        if "counts" in y:
            y["counts"] = y["counts"].copy()
        copied.append(y)

    return copied


def pick_replace_indices(pool: List[Dict[str, Any]], topk: int) -> List[int]:
    if len(pool) == 0:
        return []

    scores = np.array(
        [x["score"] for x in pool],
        dtype=np.float32,
    )

    return np.argsort(scores)[:min(topk, len(pool))].tolist()


# =========================================================
# 13. Replacement: retain the original safe one-by-one replacement rule
# =========================================================

def candidate_passes_fill_guard(
    new_pool_pcc: float,
    new_avg_gcerr: float,
    fill_stats: Dict[str, float],
) -> bool:
    if fill_stats is None:
        return True

    if REQUIRE_BETTER_THAN_FILL:
        if new_pool_pcc < fill_stats["pool_pcc6"] + BEST_PCC_MARGIN:
            return False
        if new_avg_gcerr > fill_stats["avg_gc_err"] - BEST_GCERR_MARGIN:
            return False
    else:
        if new_pool_pcc < fill_stats["pool_pcc6"] - FILL_PCC_TOL:
            return False
        if new_avg_gcerr > fill_stats["avg_gc_err"] + FILL_GCERR_TOL:
            return False

    return True


def pareto_pccboost_replace(
    pool: List[Dict[str, Any]],
    reservoir: List[Dict[str, Any]],
    seen: set,
    pool_sum_counts: np.ndarray,
    pool_total: float,
    pool_gcerr_sum: float,
    top_k_replace: int,
    ref_c: np.ndarray,
    ref_norm: float,
    fill_stats: Dict[str, float],
) -> Tuple[int, np.ndarray, float]:
    if len(pool) == 0 or len(reservoir) == 0:
        return 0, pool_sum_counts, pool_gcerr_sum

    replace_indices = pick_replace_indices(pool, top_k_replace)
    used_cand = set()
    replaced = 0

    for wi in replace_indices:
        old = pool[wi]

        curr_pcc = pool_pcc6(
            pool_sum_counts,
            pool_total,
            ref_c,
            ref_norm,
        )

        best_ci = None
        best_gain = -1e18
        best_new_sum_counts = None
        best_new_gcerr_sum = None

        for ci, cand in enumerate(reservoir):
            if ci in used_cand:
                continue

            if cand["sequence"] in seen:
                continue

            class_gain = cand["class_score"] - old["class_score"]
            gc_gain = old["gc_err"] - cand["gc_err"]

            if class_gain <= MIN_CLASS_GAIN:
                continue

            new_sum_counts = pool_sum_counts - old["counts"] + cand["counts"]

            new_pool_pcc = pool_pcc6(
                new_sum_counts,
                pool_total,
                ref_c,
                ref_norm,
            )

            pcc_gain = new_pool_pcc - curr_pcc

            if pcc_gain < MIN_PCC_GAIN:
                continue

            new_gcerr_sum = pool_gcerr_sum - float(old["gc_err"]) + float(cand["gc_err"])
            new_avg_gcerr = new_gcerr_sum / float(len(pool))

            if not candidate_passes_fill_guard(
                new_pool_pcc=new_pool_pcc,
                new_avg_gcerr=new_avg_gcerr,
                fill_stats=fill_stats,
            ):
                continue

            strict_ok = (
                class_gain > MIN_CLASS_GAIN
                and gc_gain >= MIN_GC_GAIN
                and pcc_gain >= MIN_PCC_GAIN
            )

            relaxed_ok = False

            if not strict_ok:
                if class_gain > RELAXED_CLASS_GAIN and pcc_gain >= MIN_PCC_GAIN:
                    relaxed_ok = True

            if not strict_ok and not relaxed_ok:
                continue

            self_gain = cand["self_pcc"] - old["self_pcc"]

            replace_gain = (
                0.45 * class_gain
                + 0.20 * max(gc_gain, 0.0)
                + 0.25 * max(pcc_gain, 0.0)
                + 0.10 * max(self_gain, 0.0)
            )

            if relaxed_ok and not strict_ok:
                replace_gain *= 0.70

            if replace_gain > best_gain:
                best_gain = replace_gain
                best_ci = ci
                best_new_sum_counts = new_sum_counts
                best_new_gcerr_sum = new_gcerr_sum

        if best_ci is not None:
            cand = reservoir[best_ci]
            used_cand.add(best_ci)

            pool[wi] = cand
            seen.add(cand["sequence"])

            pool_sum_counts = best_new_sum_counts
            pool_gcerr_sum = best_new_gcerr_sum
            replaced += 1

    return replaced, pool_sum_counts, pool_gcerr_sum


# =========================================================
# 14. Action candidate generation
# =========================================================

def make_candidates_by_action(
    action: Dict[str, Any],
    pool: List[Dict[str, Any]],
    target_label: int,
    n_generate: int,
    target_gc: float,
    model,
    schedule,
    position_weights,
    base_pref_table,
    delta_pref_table,
    cfg,
    device,
):
    name = action["name"]

    if name == "PoolLocalMutate":
        parents = select_parent_for_mutate(
            pool=pool,
            target_label=target_label,
            n_parent=n_generate,
        )

        return [
            mutate_seq_light(
                seq=p,
                target_gc=target_gc,
                n_mut=action["n_mut"],
            )
            for p in parents
        ]

    if name == "MaskedRepaint":
        rescue_mode = False

        if len(pool) > 0:
            acc_now = float(np.mean([x["pred"] == target_label for x in pool]))
            rescue_mode = acc_now < 0.90

        parents = select_parent_for_mask(
            pool=pool,
            target_label=target_label,
            n_parent=n_generate,
            rescue_mode=rescue_mode,
        )

        return masked_repaint_unet(
            model=model,
            schedule=schedule,
            parent_seqs=parents,
            target_label=target_label,
            mask_rate=action["mask_rate"],
            cond_weight=action["cond_weight"],
            position_weights=position_weights,
            base_pref_table=base_pref_table,
            delta_pref_table=delta_pref_table,
            cfg=cfg,
            device=device,
        )

    if name == "UNetRebirth":
        return generate_unet_candidates(
            model=model,
            schedule=schedule,
            target_label=target_label,
            n_generate=n_generate,
            cond_weight=action["cond_weight"],
            position_weights=position_weights,
            base_pref_table=base_pref_table,
            delta_pref_table=delta_pref_table,
            cfg=cfg,
            device=device,
        )

    raise ValueError(f"Unknown action: {name}")


def print_pool_stats(prefix: str, stats: Dict[str, float]):
    print(
        f"{prefix} "
        f"Acc={stats['acc']:.4f} "
        f"Prob={stats['avg_prob']:.4f} "
        f"Margin={stats['avg_margin']:.4f} "
        f"AdjLeak={stats['adj_leak']:.4f} "
        f"GCerr={stats['avg_gc_err']:.4f} "
        f"PCC6={stats['pool_pcc6']:.4f} "
        f"Div={stats['diversity']:.4f}"
    )


# =========================================================
# 15. One label optimization
# =========================================================

def optimize_one_label(
    set_id: int,
    target_label: int,
    device,
    model,
    schedule,
    position_weights,
    base_pref_table,
    delta_pref_table,
    cfg,
    predictors,
    nat_grouped,
    ref_gc,
    ref_kfreq,
    nat_sets,
):
    target_gc = ref_gc[target_label]
    ref_freq = ref_kfreq[target_label].astype(np.float32)
    ref_c = ref_freq - ref_freq.mean()
    ref_norm = np.linalg.norm(ref_c) + 1e-8
    nat_set = nat_sets[target_label]

    print("\n" + "=" * 100)
    print(f"[Set {set_id:02d} | Label {target_label}] Start optimization")
    print("=" * 100)
    print(f"target_gc={target_gc:.6f}")
    print(f"natural_count={len(nat_grouped.get(target_label, []))}")

    pool = []
    seen = set()

    pool_sum_counts = np.zeros((DIM_K,), dtype=np.float64)
    pool_total = 0.0
    pool_gcerr_sum = 0.0

    fill_round = 0

    while len(pool) < POOL_SIZE:
        fill_round += 1

        raw = generate_unet_candidates(
            model=model,
            schedule=schedule,
            target_label=target_label,
            n_generate=RESERVOIR_FILL,
            cond_weight=CFG_BASE,
            position_weights=position_weights,
            base_pref_table=base_pref_table,
            delta_pref_table=delta_pref_table,
            cfg=cfg,
            device=device,
        )

        reservoir = build_reservoir_from_seqs(
            seqs=raw,
            target_label=target_label,
            target_gc=target_gc,
            ref_freq=ref_freq,
            predictors=predictors,
            device=device,
            seen=seen,
            nat_set=nat_set,
            phase="FILL",
        )

        add_count = 0

        for item in reservoir:
            if len(pool) >= POOL_SIZE:
                break

            s = item["sequence"]

            if s in seen:
                continue

            pool.append(item)
            seen.add(s)

            pool_sum_counts += item["counts"]
            pool_total += float(KMERS_PER_SEQ)
            pool_gcerr_sum += float(item["gc_err"])
            add_count += 1

        st = pool_stats(
            pool=pool,
            target_label=target_label,
            pool_sum_counts=pool_sum_counts,
            pool_total=pool_total,
            ref_c=ref_c,
            ref_norm=ref_norm,
        )

        print(
            f"[Set {set_id:02d} L{target_label} FILL {fill_round:03d}] "
            f"n={len(pool)} add={add_count} "
            f"acc={st['acc']:.4f} prob={st['avg_prob']:.4f} "
            f"margin={st['avg_margin']:.4f} adj_leak={st['adj_leak']:.4f} "
            f"gcerr={st['avg_gc_err']:.4f} pcc6={st['pool_pcc6']:.6f} "
            f"div={st['diversity']:.4f}"
        )

        if fill_round > 300:
            raise RuntimeError(f"FILL too many rounds for label {target_label}")

    fill_stats = pool_stats(
        pool=pool,
        target_label=target_label,
        pool_sum_counts=pool_sum_counts,
        pool_total=pool_total,
        ref_c=ref_c,
        ref_norm=ref_norm,
    )

    fill_obj = optimization_objective(fill_stats)

    print("-" * 100)
    print(
        f"[FillBaseline Set {set_id:02d} | Label {target_label}] "
        f"Fill Acc={fill_stats['acc']:.4f} "
        f"AvgProb={fill_stats['avg_prob']:.4f} "
        f"GCerr={fill_stats['avg_gc_err']:.4f} "
        f"PCC6={fill_stats['pool_pcc6']:.4f} "
        f"Div={fill_stats['diversity']:.4f}"
    )

    best_pool = clone_pool(pool)
    best_sum_counts = pool_sum_counts.copy()
    best_gcerr_sum = float(pool_gcerr_sum)
    best_stats = dict(fill_stats)
    best_obj = fill_obj

    print(f"[BestInit Set {set_id:02d} | Label {target_label}] best_obj={best_obj:.5f}")
    print("-" * 100)

    bandit = EpsGreedyActionBandit(n_actions=len(ACTIONS))
    stagnant_count = 0

    for rd in range(1, MAX_ROUNDS + 1):
        old_stats = pool_stats(
            pool=pool,
            target_label=target_label,
            pool_sum_counts=pool_sum_counts,
            pool_total=pool_total,
            ref_c=ref_c,
            ref_norm=ref_norm,
        )

        action_id = bandit.select(global_stagnant_count=stagnant_count)
        action = ACTIONS[action_id]

        print(f"[Set {set_id:02d} | Label {target_label} | Round {rd:03d} | EVOLVE] action={action}")

        raw = make_candidates_by_action(
            action=action,
            pool=pool,
            target_label=target_label,
            n_generate=RESERVOIR_EVOLVE,
            target_gc=target_gc,
            model=model,
            schedule=schedule,
            position_weights=position_weights,
            base_pref_table=base_pref_table,
            delta_pref_table=delta_pref_table,
            cfg=cfg,
            device=device,
        )

        reservoir = build_reservoir_from_seqs(
            seqs=raw,
            target_label=target_label,
            target_gc=target_gc,
            ref_freq=ref_freq,
            predictors=predictors,
            device=device,
            seen=seen,
            nat_set=nat_set,
            phase="EVOLVE",
        )

        before_len = len(pool)

        replaced, pool_sum_counts, pool_gcerr_sum = pareto_pccboost_replace(
            pool=pool,
            reservoir=reservoir,
            seen=seen,
            pool_sum_counts=pool_sum_counts,
            pool_total=pool_total,
            pool_gcerr_sum=pool_gcerr_sum,
            top_k_replace=TOP_K_REPLACE,
            ref_c=ref_c,
            ref_norm=ref_norm,
            fill_stats=fill_stats,
        )

        new_stats = pool_stats(
            pool=pool,
            target_label=target_label,
            pool_sum_counts=pool_sum_counts,
            pool_total=pool_total,
            ref_c=ref_c,
            ref_norm=ref_norm,
        )

        safe_accept_rate = float(replaced) / float(max(1, RESERVOIR_EVOLVE))
        reward = compute_reward(old_stats, new_stats, safe_accept_rate)

        bandit.update(action_id, reward)

        if replaced > 0:
            stagnant_count = 0
        else:
            stagnant_count += 1

        print(
            f"[RoundResult] "
            f"res={len(reservoir):4d} "
            f"add={max(0, len(pool) - before_len):3d} "
            f"safe_rep={replaced:3d} "
            f"pool={len(pool):4d} "
            f"safe_accept={safe_accept_rate:.4f} "
            f"reward={reward:+.5f}"
        )

        print(
            f"[PoolStats] "
            f"Acc={new_stats['acc']:.4f} "
            f"Prob={new_stats['avg_prob']:.4f} "
            f"Margin={new_stats['avg_margin']:.4f} "
            f"AdjLeak={new_stats['adj_leak']:.4f} "
            f"GCerr={new_stats['avg_gc_err']:.4f} "
            f"PCC6={new_stats['pool_pcc6']:.4f} "
            f"Div={new_stats['diversity']:.4f}"
        )

        print(f"[Bandit] {bandit.state_str()}")

        cur_obj = optimization_objective(new_stats)

        if cur_obj > best_obj:
            best_obj = cur_obj
            best_pool = clone_pool(pool)
            best_sum_counts = pool_sum_counts.copy()
            best_gcerr_sum = float(pool_gcerr_sum)
            best_stats = dict(new_stats)

            print(
                f"[BestPool Updated] "
                f"obj={best_obj:.5f} "
                f"Acc={best_stats['acc']:.4f} "
                f"GCerr={best_stats['avg_gc_err']:.4f} "
                f"PCC6={best_stats['pool_pcc6']:.4f}"
            )

    final_stats = pool_stats(
        pool=best_pool,
        target_label=target_label,
        pool_sum_counts=best_sum_counts,
        pool_total=pool_total,
        ref_c=ref_c,
        ref_norm=ref_norm,
    )

    final_obj = optimization_objective(final_stats)

    print("-" * 100)
    print(
        f"[FinalBest Set {set_id:02d} | Label {target_label}] "
        f"Obj={final_obj:.5f} "
        f"Acc={final_stats['acc']:.4f} "
        f"AvgProb={final_stats['avg_prob']:.4f} "
        f"GCerr={final_stats['avg_gc_err']:.4f} "
        f"PCC6={final_stats['pool_pcc6']:.4f} "
        f"Div={final_stats['diversity']:.4f}"
    )
    print("-" * 100)

    return best_pool, final_stats, fill_stats, best_stats


# =========================================================
# 16. Save
# =========================================================

def save_label_csv(seqs: List[str], path: str):
    df = pd.DataFrame({"sequence": seqs})
    df.to_csv(path, index=False)


def make_set_out_dir(set_id: int):
    return os.path.join(OUT_ROOT, f"set_{set_id:02d}")


def save_multi_set_summary(all_summary_dfs: List[pd.DataFrame]):
    if len(all_summary_dfs) == 0:
        return

    all_df = pd.concat(all_summary_dfs, axis=0, ignore_index=True)

    raw_path = os.path.join(
        OUT_ROOT,
        "optimization_summary_all_sets_raw.csv",
    )

    all_df.to_csv(raw_path, index=False)

    numeric_cols = [
        c for c in all_df.columns
        if c not in ["set_id", "target_label"]
        and pd.api.types.is_numeric_dtype(all_df[c])
    ]

    summary_rows = []

    for lb in TARGET_LABELS:
        sub = all_df[all_df["target_label"] == lb]

        row = {
            "target_label": lb,
            "n_sets": len(sub),
        }

        for c in numeric_cols:
            row[f"{c}_mean"] = float(sub[c].mean()) if len(sub) > 0 else np.nan
            row[f"{c}_std"] = float(sub[c].std(ddof=1)) if len(sub) > 1 else 0.0

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    label_path = os.path.join(
        OUT_ROOT,
        "optimization_summary_all_sets_by_label_mean_std.csv",
    )

    summary_df.to_csv(label_path, index=False)

    overall = {}

    for c in numeric_cols:
        overall[f"{c}_mean"] = float(all_df[c].mean())
        overall[f"{c}_std"] = float(all_df[c].std(ddof=1)) if len(all_df) > 1 else 0.0

    overall_path = os.path.join(
        OUT_ROOT,
        "optimization_summary_all_sets_overall_mean_std.csv",
    )

    pd.DataFrame([overall]).to_csv(overall_path, index=False)

    print("\n" + "=" * 100)
    print("[All Sets Summary Saved]")
    print(f"Raw:      {raw_path}")
    print(f"ByLabel:  {label_path}")
    print(f"Overall:  {overall_path}")
    print("=" * 100)
    print(summary_df)
    print(pd.DataFrame([overall]))


def optimize_one_set(
    set_id: int,
    out_dir: str,
    device,
    model,
    schedule,
    position_weights,
    base_pref_table,
    delta_pref_table,
    cfg,
    predictors,
    nat_grouped,
    ref_gc,
    ref_kfreq,
    nat_sets,
    state_key: str,
):
    ensure_dir(out_dir)

    summary_rows = []
    all_rows = []

    for target_label in TARGET_LABELS:
        best_pool, final_stats, fill_stats, best_stats = optimize_one_label(
            set_id=set_id,
            target_label=target_label,
            device=device,
            model=model,
            schedule=schedule,
            position_weights=position_weights,
            base_pref_table=base_pref_table,
            delta_pref_table=delta_pref_table,
            cfg=cfg,
            predictors=predictors,
            nat_grouped=nat_grouped,
            ref_gc=ref_gc,
            ref_kfreq=ref_kfreq,
            nat_sets=nat_sets,
        )

        best_pool = sorted(best_pool, key=lambda x: x["score"], reverse=True)[:NUM_PER_LABEL]
        label_seqs = [x["sequence"] for x in best_pool]

        label_csv = os.path.join(
            out_dir,
            f"zhou_unet_label_{target_label}.csv",
        )

        save_label_csv(label_seqs, label_csv)

        print(f"[Saved] {label_csv}")

        summary_rows.append({
            "set_id": set_id,
            "target_label": target_label,
            "n": len(label_seqs),
            "state_key": state_key,
            "cfg_base": CFG_BASE,
            "cfg_soft": CFG_SOFT,
            "predictor": f"{PREDICTOR_MODEL_TAG}/{PREDICTOR_EXP_NAME}",
            "fill_acc_by_allfusion": fill_stats["acc"],
            "fill_avg_prob": fill_stats["avg_prob"],
            "fill_avg_margin": fill_stats["avg_margin"],
            "fill_adj_leak": fill_stats["adj_leak"],
            "fill_avg_gc_err": fill_stats["avg_gc_err"],
            "fill_pool_pcc6": fill_stats["pool_pcc6"],
            "best_acc_by_allfusion": best_stats["acc"] if best_stats is not None else np.nan,
            "best_avg_prob": best_stats["avg_prob"] if best_stats is not None else np.nan,
            "best_avg_margin": best_stats["avg_margin"] if best_stats is not None else np.nan,
            "best_avg_gc_err": best_stats["avg_gc_err"] if best_stats is not None else np.nan,
            "best_pool_pcc6": best_stats["pool_pcc6"] if best_stats is not None else np.nan,
            "final_acc_by_allfusion": final_stats["acc"],
            "final_avg_prob": final_stats["avg_prob"],
            "final_avg_margin": final_stats["avg_margin"],
            "final_adj_leak": final_stats["adj_leak"],
            "final_avg_gc_err": final_stats["avg_gc_err"],
            "final_pool_pcc6": final_stats["pool_pcc6"],
            "final_diversity": final_stats["diversity"],
            "delta_acc_vs_fill": final_stats["acc"] - fill_stats["acc"],
            "delta_gcerr_vs_fill": final_stats["avg_gc_err"] - fill_stats["avg_gc_err"],
            "delta_pcc6_vs_fill": final_stats["pool_pcc6"] - fill_stats["pool_pcc6"],
            "target_gc": ref_gc[target_label],
        })

        for s in label_seqs:
            all_rows.append({
                "sequence": s,
                "target_label": target_label,
                "gc": gc_of_seq(s),
                "length": len(s),
                "set_id": set_id,
            })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(
        out_dir,
        "optimization_summary_allfusion_bandit_bestpool_pccboost.csv",
    )

    summary_df.to_csv(summary_path, index=False)

    if SAVE_ALL_CSV:
        all_df = pd.DataFrame(all_rows)

        all_path = os.path.join(
            out_dir,
            "zhou_unet_all_labels.csv",
        )

        all_df.to_csv(all_path, index=False)
        print(f"[All labels saved] {all_path}")

    print("\n" + "=" * 100)
    print(f"[Optimization Set {set_id:02d} Done]")
    print(f"OUT_DIR: {out_dir}")
    print(f"Summary: {summary_path}")
    print("=" * 100)
    print(summary_df)

    return summary_df


# =========================================================
# 17. Main
# =========================================================

def main():
    seed_everything(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = torch.device(DEVICE)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    ensure_dir(OUT_ROOT)

    log_path = os.path.join(
        OUT_ROOT,
        "optimization_log_allfusion_bandit_bestpool_pccboost_new_eval_psednc6_same_rule.txt",
    )

    sys.stdout = Logger(log_path)

    print("=" * 100)
    print("[LightPool Bandit Continuous Optimization: SAME RULE + New 6-prop PseDNC Evaluator]")
    print("=" * 100)
    print(f"CHECKPOINT: {CHECKPOINT}")
    print(f"OUT_ROOT:   {OUT_ROOT}")
    print(f"DATASET:    {DATASET_PATH}")
    print(f"DEVICE:     {device}")
    print(f"POOL_SIZE:  {POOL_SIZE}")
    print(f"MAX_ROUNDS: {MAX_ROUNDS}")
    print(f"CFG_BASE:   {CFG_BASE}")
    print(f"CFG_SOFT:   {CFG_SOFT}")
    print(f"Predictor:  {PREDICTOR_MODEL_TAG}/{PREDICTOR_EXP_NAME}")
    print(f"Predictor root: {PREDICTOR_BASE_ROOT}")
    print(f"NUM_OPTIMIZATION_SETS: {NUM_OPTIMIZATION_SETS}")
    print("Seed mode: seed_everything(SEED) only once at program start")
    print("FILL: label-agnostic UNetRebirth(cfg=1.0)")
    print("EVOLVE: original action-level epsilon-greedy bandit")
    print("Replacement: original pareto_pccboost_replace safe_rep / safe_accept rule")
    print("=" * 100)

    check_feature_compatibility()

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    print("[Load UNet checkpoint]")

    ckpt = torch.load(CHECKPOINT, map_location=device)

    cfg = cfg_from_checkpoint(ckpt)
    cfg.CLASS_POSITION_WEIGHT_PATH = str(PROJECT_ROOT / "prior_condition" / "results_weights" / "class_specific_classifier_weights_full" / "class_position_weights_combined_5bins.npy")
    cfg.CLASS_BASE_PREF_PATH = str(PROJECT_ROOT / "prior_condition" / "results_weights" / "class_specific_classifier_weights_full" / "class_base_preference_from_mutagenesis.npy")
    cfg.DEVICE = str(device)

    model = build_model(cfg).to(device)

    state_key = "model_state_dict" if USE_RAW_MODEL else "ema_state_dict"

    if state_key not in ckpt:
        print(f"[Warning] {state_key} not found. Use model_state_dict instead.")
        state_key = "model_state_dict"

    model.load_state_dict(ckpt[state_key], strict=True)
    model.eval()

    schedule = DiffusionSchedule(
        timesteps=cfg.TIMESTEPS,
        beta_end=cfg.BETA_END,
        device=device,
    )

    position_weights = load_class_position_weights(cfg, device=device)
    base_pref_table = load_class_base_preference(cfg, device=device)
    delta_pref_table = build_adjacent_delta_pref_table(base_pref_table, cfg)

    print("[UNet loaded]")
    print(f"state_key={state_key}")
    print(f"seq_len={cfg.SEQ_LEN}")
    print(f"timesteps={cfg.TIMESTEPS}")
    print("-" * 100)
    print(f"Base pref condition:    {getattr(cfg, 'USE_BASE_PREF_CONDITION', None)}")
    print(f"Adj-diff prior:         {getattr(cfg, 'USE_ADJ_DIFF_PRIOR', None)}")
    print(f"Prior token encoder:    {getattr(cfg, 'USE_PRIOR_TOKEN_ENCODER', None)}")
    print(f"Cross-attn downs:       {getattr(cfg, 'USE_CROSS_ATTN_IN_DOWNS', None)}")
    print(f"Cross-attn mid:         {getattr(cfg, 'USE_CROSS_ATTN_IN_MID', None)}")
    print("=" * 100)

    predictors = load_allfusion_predictors(device)

    print("[Load natural reference]")

    nat_grouped, ref_gc, ref_kfreq = build_nat_ref_and_pool(
        DATASET_PATH,
        TARGET_LABELS,
        k=K,
    )

    nat_sets = {
        lb: set(nat_grouped.get(lb, []))
        for lb in TARGET_LABELS
    }

    for lb in TARGET_LABELS:
        print(
            f"[Natural Ref] label={lb} "
            f"n={len(nat_grouped.get(lb, []))} "
            f"gc={ref_gc[lb]:.6f}"
        )

    all_set_summary_dfs = []

    for set_id in range(1, NUM_OPTIMIZATION_SETS + 1):
        out_dir = make_set_out_dir(set_id)

        summary_df = optimize_one_set(
            set_id=set_id,
            out_dir=out_dir,
            device=device,
            model=model,
            schedule=schedule,
            position_weights=position_weights,
            base_pref_table=base_pref_table,
            delta_pref_table=delta_pref_table,
            cfg=cfg,
            predictors=predictors,
            nat_grouped=nat_grouped,
            ref_gc=ref_gc,
            ref_kfreq=ref_kfreq,
            nat_sets=nat_sets,
            state_key=state_key,
        )

        all_set_summary_dfs.append(summary_df)

    save_multi_set_summary(all_set_summary_dfs)

    print("\n" + "=" * 100)
    print("[All Optimization Sets Done]")
    print(f"OUT_ROOT: {OUT_ROOT}")
    print("=" * 100)


if __name__ == "__main__":
    main()
