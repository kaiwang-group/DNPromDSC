import os
import re
import json
import random
import importlib
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from scipy.stats import rankdata
from scipy.spatial.distance import jensenshannon


# =========================================================
# =========================================================

@dataclass
class CFG:
    # -------------------------
    # -------------------------
    CLASS_DATA_PATH: str = "/data/stu1/wrc3_pycharm_project/WRC_mask_diffusion/wrc_masked_dataset/wrc_class5_63468.txt"

    OUT_DIR: str = "/data/stu1/wrc3_pycharm_project/WRC_mask_diffusion/wrc_identification_key_regions_new/results_weights/class_specific_classifier_weights_full"

    # -------------------------
    # -------------------------
    CLASSIFIER_MODULE: str = "model_optimized_predict"

    CLASSIFIER_CLASS: str = "TransformerHybridMultiTask"

    CLASSIFIER_KWARGS: Dict[str, Any] = field(default_factory=lambda: {
        "input_size": 5,
        "hidden_size": 256,
        "num_classes": 5,
        "dropout_rate": 0.2,
        "conv_kernels": (5, 5, 3),
        "num_layers": 3,
        "nhead": 8,
        "se_reduction": 16,

        "use_cgr": False,
        "use_psednc": False,

        "psednc_dim": 22,
        "mask_prob": 0.0,
        "pse_scale_init": 0.05,
    })

    CLASSIFIER_CHECKPOINTS: List[str] = field(default_factory=lambda: [
        "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Predictor/wrc_class5_cgr_gpt/new_simple/results_adaptive_film_v52_gpt_seed43/tf3_h8/seq_only/prediction_f1.pth",
        "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Predictor/wrc_class5_cgr_gpt/new_simple/results_adaptive_film_v52_gpt_seed43/tf3_h8/seq_only/prediction_f2.pth",
        "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Predictor/wrc_class5_cgr_gpt/new_simple/results_adaptive_film_v52_gpt_seed43/tf3_h8/seq_only/prediction_f3.pth",
        "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Predictor/wrc_class5_cgr_gpt/new_simple/results_adaptive_film_v52_gpt_seed43/tf3_h8/seq_only/prediction_f4.pth",
        "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Predictor/wrc_class5_cgr_gpt/new_simple/results_adaptive_film_v52_gpt_seed43/tf3_h8/seq_only/prediction_f5.pth",
    ])

    STRICT_LOAD: bool = True

    MODEL_INPUT_FORMAT: str = "B,L,5"

    SCORE_MODE: str = "margin"

    # -------------------------
    # -------------------------
    SEQ_LEN: int = 80
    NUM_CLASSES: int = 5
    BASES: str = "ACGT"

    # -------------------------
    # -------------------------
    RUN_A_BASE_DISTRIBUTION: bool = True
    RUN_B_IG: bool = True
    RUN_C_MUTAGENESIS: bool = True
    RUN_D_FUSION_BUCKET_WEIGHT: bool = True

    # -------------------------
    # -------------------------
    IG_STEPS: int = 30
    IG_MAX_SAMPLES_PER_CLASS: Optional[int] = None
    IG_BATCH_SIZE: int = 64

    # -------------------------
    # -------------------------
    MUT_MAX_SAMPLES_PER_CLASS: Optional[int] = None
    MUT_BATCH_SIZE: int = 512

    # -------------------------
    # -------------------------
    NUM_BUCKETS: int = 5

    FUSION_WEIGHT_BASE: float = 0.35
    FUSION_WEIGHT_IG: float = 0.30
    FUSION_WEIGHT_MUT: float = 0.35

    POSITION_WEIGHT_MIN: float = 0.8
    POSITION_WEIGHT_MAX: float = 1.8

    # -------------------------
    # -------------------------
    SEED: int = 42
    DEVICE: str = "cuda:1" if torch.cuda.is_available() else "cpu"


cfg = CFG()


# =========================================================
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_seq(seq: str) -> str:
    seq = str(seq).strip().upper()
    seq = re.sub(r"[^ACGT]", "", seq)
    return seq


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def strip_state_dict(state):
    """
      checkpoint
      {"model_state_dict": ...}
      {"state_dict": ...}
      DataParallel module.
      torch.compile _orig_mod.
    """
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

    if not isinstance(state, dict):
        return state

    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        new_state[nk] = v

    return new_state


def one_hot_encode_seq(seq: str, seq_len: int, bases: str = "ACGT") -> np.ndarray:
    """
    """
    base_to_idx = {b: i for i, b in enumerate(bases)}
    arr = np.zeros((seq_len, 4), dtype=np.float32)

    seq = clean_seq(seq)

    if len(seq) >= seq_len:
        seq = seq[:seq_len]
    else:
        seq = seq + "N" * (seq_len - len(seq))

    for i, b in enumerate(seq):
        if b in base_to_idx:
            arr[i, base_to_idx[b]] = 1.0

    return arr


def seqs_to_tensor(seqs: List[str], seq_len: int, bases: str = "ACGT") -> torch.Tensor:
    arr = np.stack([one_hot_encode_seq(s, seq_len, bases) for s in seqs], axis=0)
    return torch.tensor(arr, dtype=torch.float32)


def load_class_sequences(path: str, seq_len: int, num_classes: int) -> Dict[int, List[str]]:
    """
      sequence<TAB>label
      sequence label
      sequence,label
    """
    seqs_by_class = {i: [] for i in range(num_classes)}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            low = line.lower()
            if "sequence" in low and "label" in low:
                continue

            parts = re.split(r"[\t,\s,]+", line)

            if len(parts) < 2:
                continue

            seq = clean_seq(parts[0])

            try:
                label = int(float(parts[1]))
            except Exception:
                continue

            if len(seq) == seq_len and 0 <= label < num_classes:
                seqs_by_class[label].append(seq)

    total = sum(len(v) for v in seqs_by_class.values())

    if total == 0:
        raise ValueError(f"No valid sequences loaded from: {path}")

    print("=" * 80)
    print("[Data loaded]")
    print(f"Path: {path}")
    print(f"Total valid sequences: {total}")
    for c in range(num_classes):
        print(f"Class {c}: {len(seqs_by_class[c])}")
    print("=" * 80)

    return seqs_by_class


def sample_sequences(seqs: List[str], max_samples: Optional[int], seed: int) -> List[str]:
    """
    """
    if max_samples is None or len(seqs) <= max_samples:
        return list(seqs)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(seqs), size=max_samples, replace=False)
    return [seqs[i] for i in idx]


def percentile_rank_vector(x: np.ndarray) -> np.ndarray:
    """
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    if n <= 1:
        return np.zeros_like(x, dtype=np.float32)

    r = rankdata(x, method="average")
    r = (r - 1) / (n - 1)

    return r.astype(np.float32)


def make_buckets_by_quantile(importance: np.ndarray, num_buckets: int) -> np.ndarray:
    """
    importance: [L]
    return: [L], bucket id 0..num_buckets-1
    """
    imp = np.asarray(importance, dtype=np.float32)
    edges = np.quantile(imp, np.linspace(0, 1, num_buckets + 1))

    bucket = np.digitize(imp, edges[1:-1], right=True).astype(np.int64)
    bucket = np.clip(bucket, 0, num_buckets - 1)

    return bucket


def normalize_weight_mean_one(
    importance: np.ndarray,
    w_min: float,
    w_max: float,
) -> np.ndarray:
    """
    importance -> [w_min, w_max] -> mean=1
    """
    imp = np.asarray(importance, dtype=np.float32)

    min_v = float(np.min(imp))
    max_v = float(np.max(imp))

    if abs(max_v - min_v) < 1e-8:
        weights = np.ones_like(imp, dtype=np.float32)
    else:
        norm = (imp - min_v) / (max_v - min_v)
        weights = w_min + norm * (w_max - w_min)

    weights = weights / (weights.mean() + 1e-8)

    return weights.astype(np.float32)


# =========================================================
# =========================================================

def build_classifier_model(cfg: CFG):
    module = importlib.import_module(cfg.CLASSIFIER_MODULE)
    cls = getattr(module, cfg.CLASSIFIER_CLASS)
    model = cls(**cfg.CLASSIFIER_KWARGS)
    return model


def load_classifier_models(cfg: CFG, device: torch.device):
    models = []

    for ckpt_path in cfg.CLASSIFIER_CHECKPOINTS:
        if not os.path.exists(ckpt_path):
            print(f"⚠️ checkpoint not found, skip: {ckpt_path}")
            continue

        model = build_classifier_model(cfg).to(device)

        state = safe_torch_load(ckpt_path, device=device)
        state = strip_state_dict(state)

        try:
            model.load_state_dict(state, strict=cfg.STRICT_LOAD)
        except Exception as e:
            print("=" * 80)
            print(f"❌ Failed to load checkpoint: {ckpt_path}")
            print(e)
            print("-" * 80)
            print("常见原因：")
            print("1. 这个 checkpoint 不是 seq-only 模型，而是 CGR/PseDNC 融合模型；")
            print("2. CLASSIFIER_KWARGS 与训练时不一致；")
            print("3. input_size、num_layers、nhead、conv_kernels 等参数不一致。")
            print("=" * 80)
            raise

        model.eval()
        models.append(model)

        print(f"✅ Loaded classifier: {ckpt_path}")

    if len(models) == 0:
        raise RuntimeError("No classifier checkpoint loaded. Please check CLASSIFIER_CHECKPOINTS.")

    print("=" * 80)
    print(f"[Classifier loaded] n_models={len(models)}")
    print(f"Input format: {cfg.MODEL_INPUT_FORMAT}")
    print(f"Score mode: {cfg.SCORE_MODE}")
    print("=" * 80)

    return models


def prepare_model_input(x4: torch.Tensor, cfg: CFG) -> torch.Tensor:
    """
    x4: [B, L, 4]

    MODEL_INPUT_FORMAT="B,L,5":
      channel 1/2/3/4 = A/C/G/T
    """
    fmt = cfg.MODEL_INPUT_FORMAT

    if fmt == "B,L,4":
        return x4

    if fmt == "B,4,L":
        return x4.permute(0, 2, 1).contiguous()

    if fmt == "B,L,5":
        b, l, _ = x4.shape
        zero = torch.zeros((b, l, 1), device=x4.device, dtype=x4.dtype)
        return torch.cat([zero, x4], dim=2)

    if fmt == "B,5,L":
        b, l, _ = x4.shape
        zero = torch.zeros((b, 1, l), device=x4.device, dtype=x4.dtype)
        x_ch = x4.permute(0, 2, 1).contiguous()
        return torch.cat([zero, x_ch], dim=1)

    raise ValueError(f"Unknown MODEL_INPUT_FORMAT: {fmt}")


def extract_logits_from_output(output):
    if isinstance(output, dict):
        if "logits" in output:
            output = output["logits"]
        elif "pred" in output:
            output = output["pred"]
        else:
            first_key = list(output.keys())[0]
            output = output[first_key]

    if isinstance(output, (tuple, list)):
        output = output[0]

    return output


def forward_classifier_logits(model, x4: torch.Tensor, cfg: CFG) -> torch.Tensor:
    """
    x4: [B, L, 4]
    return logits: [B, NUM_CLASSES]
    """
    inp = prepare_model_input(x4, cfg)

    out = model(inp)

    logits = extract_logits_from_output(out)

    if logits.ndim != 2:
        logits = logits.view(logits.shape[0], -1)

    logits = logits[:, :cfg.NUM_CLASSES]

    return logits


def class_score_from_logits(logits: torch.Tensor, target_class: int, cfg: CFG) -> torch.Tensor:
    """
    return: [B]
    """
    if cfg.SCORE_MODE == "target_logit":
        return logits[:, target_class]

    if cfg.SCORE_MODE == "margin":
        target = logits[:, target_class]

        mask = torch.ones(
            logits.shape[1],
            dtype=torch.bool,
            device=logits.device,
        )
        mask[target_class] = False

        other_mean = logits[:, mask].mean(dim=1)
        return target - other_mean

    if cfg.SCORE_MODE == "prob":
        probs = torch.softmax(logits, dim=1)
        return probs[:, target_class]

    raise ValueError(f"Unknown SCORE_MODE: {cfg.SCORE_MODE}")


def ensemble_class_score(models, x4: torch.Tensor, target_class: int, cfg: CFG) -> torch.Tensor:
    scores = []

    for model in models:
        logits = forward_classifier_logits(model, x4, cfg)
        score = class_score_from_logits(logits, target_class, cfg)
        scores.append(score)

    return torch.stack(scores, dim=0).mean(dim=0)


@torch.no_grad()
def ensemble_class_score_no_grad(models, x4: torch.Tensor, target_class: int, cfg: CFG) -> torch.Tensor:
    return ensemble_class_score(models, x4, target_class, cfg)


# =========================================================
# A. class-specific base distribution importance
# =========================================================

def compute_base_distribution_importance(seqs_by_class: Dict[int, List[str]], cfg: CFG):
    """
    """
    base_to_idx = {b: i for i, b in enumerate(cfg.BASES)}

    def calc_freq(seqs: List[str]) -> np.ndarray:
        freq = np.zeros((cfg.SEQ_LEN, 4), dtype=np.float64)

        for seq in seqs:
            for i, b in enumerate(seq[:cfg.SEQ_LEN]):
                if b in base_to_idx:
                    freq[i, base_to_idx[b]] += 1.0

        freq += 1e-6
        freq = freq / freq.sum(axis=1, keepdims=True)

        return freq

    all_seqs = []
    for c in range(cfg.NUM_CLASSES):
        all_seqs.extend(seqs_by_class[c])

    global_freq = calc_freq(all_seqs)

    class_freq = np.zeros((cfg.NUM_CLASSES, cfg.SEQ_LEN, 4), dtype=np.float64)
    class_importance = np.zeros((cfg.NUM_CLASSES, cfg.SEQ_LEN), dtype=np.float32)

    for c in range(cfg.NUM_CLASSES):
        cfreq = calc_freq(seqs_by_class[c])
        class_freq[c] = cfreq

        for pos in range(cfg.SEQ_LEN):
            js = jensenshannon(cfreq[pos], global_freq[pos]) ** 2
            class_importance[c, pos] = float(js)

    np.save(os.path.join(cfg.OUT_DIR, "global_base_freq.npy"), global_freq)
    np.save(os.path.join(cfg.OUT_DIR, "class_base_freq.npy"), class_freq)
    np.save(os.path.join(cfg.OUT_DIR, "class_position_importance_base.npy"), class_importance)

    rows = []

    for pos in range(cfg.SEQ_LEN):
        row = {
            "position_index": pos,
            "position_relative": pos - cfg.SEQ_LEN,
        }

        for c in range(cfg.NUM_CLASSES):
            row[f"class{c}_base_importance"] = class_importance[c, pos]

            for bi, b in enumerate(cfg.BASES):
                row[f"class{c}_freq_{b}"] = class_freq[c, pos, bi]
                row[f"class{c}_diff_{b}"] = class_freq[c, pos, bi] - global_freq[pos, bi]

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        os.path.join(cfg.OUT_DIR, "class_position_importance_base.csv"),
        index=False,
    )

    print("=" * 80)
    print("[A Done] class-specific base distribution importance")
    print("Saved: class_position_importance_base.npy")
    print("Saved: class_position_importance_base.csv")
    print("=" * 80)

    return class_importance, class_freq, global_freq


# =========================================================
# B. seq-only classifier Integrated Gradients
# =========================================================

def integrated_gradients_batch_classifier(
    models,
    x: torch.Tensor,
    target_class: int,
    cfg: CFG,
    device: torch.device,
    steps: int = 30,
):
    """
    x: [B, L, 4]
    target_class: 0..4

    return:
      pos_abs_imp: [B, L]
      signed_attr: [B, L, 4]
    """
    for m in models:
        m.eval()

    x = x.to(device)
    baseline = torch.zeros_like(x, device=device)

    total_grads = torch.zeros_like(x, device=device)

    for k in range(1, steps + 1):
        alpha = float(k) / float(steps)

        xi = baseline + alpha * (x - baseline)
        xi = xi.detach().clone().requires_grad_(True)

        score = ensemble_class_score(
            models=models,
            x4=xi,
            target_class=target_class,
            cfg=cfg,
        )

        score_sum = score.sum()

        for m in models:
            m.zero_grad(set_to_none=True)

        grads = torch.autograd.grad(
            outputs=score_sum,
            inputs=xi,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
        )[0]

        total_grads += grads.detach()

    avg_grads = total_grads / float(steps)
    attr = (x - baseline) * avg_grads

    signed_attr = attr.detach().cpu().numpy()
    pos_abs_imp = np.abs(signed_attr).sum(axis=2)

    return pos_abs_imp.astype(np.float32), signed_attr.astype(np.float32)


def compute_ig_importance_classifier(seqs_by_class, models, cfg: CFG, device: torch.device):
    class_pos_imp = np.zeros((cfg.NUM_CLASSES, cfg.SEQ_LEN), dtype=np.float32)
    class_base_attr = np.zeros((cfg.NUM_CLASSES, cfg.SEQ_LEN, 4), dtype=np.float32)

    for target_class in range(cfg.NUM_CLASSES):
        seqs = sample_sequences(
            seqs_by_class[target_class],
            max_samples=cfg.IG_MAX_SAMPLES_PER_CLASS,
            seed=cfg.SEED + target_class,
        )

        print("=" * 80)
        print(f"[B IG classifier] target_class={target_class}, n={len(seqs)}")
        print("=" * 80)

        all_pos_imps = []
        all_signed_attrs = []

        for i in tqdm(range(0, len(seqs), cfg.IG_BATCH_SIZE), desc=f"IG class {target_class}"):
            batch_seqs = seqs[i:i + cfg.IG_BATCH_SIZE]
            x = seqs_to_tensor(batch_seqs, cfg.SEQ_LEN, cfg.BASES)

            pos_imp, signed_attr = integrated_gradients_batch_classifier(
                models=models,
                x=x,
                target_class=target_class,
                cfg=cfg,
                device=device,
                steps=cfg.IG_STEPS,
            )

            all_pos_imps.append(pos_imp)
            all_signed_attrs.append(signed_attr)

        all_pos_imps = np.concatenate(all_pos_imps, axis=0)
        all_signed_attrs = np.concatenate(all_signed_attrs, axis=0)

        class_pos_imp[target_class] = all_pos_imps.mean(axis=0)
        class_base_attr[target_class] = all_signed_attrs.mean(axis=0)

    np.save(
        os.path.join(cfg.OUT_DIR, "class_position_importance_ig_classifier.npy"),
        class_pos_imp,
    )
    np.save(
        os.path.join(cfg.OUT_DIR, "class_base_attribution_ig_classifier_signed.npy"),
        class_base_attr,
    )

    rows = []

    for pos in range(cfg.SEQ_LEN):
        row = {
            "position_index": pos,
            "position_relative": pos - cfg.SEQ_LEN,
        }

        for c in range(cfg.NUM_CLASSES):
            row[f"class{c}_ig_importance"] = class_pos_imp[c, pos]

            for bi, b in enumerate(cfg.BASES):
                row[f"class{c}_ig_signed_attr_{b}"] = class_base_attr[c, pos, bi]

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        os.path.join(cfg.OUT_DIR, "class_position_importance_ig_classifier.csv"),
        index=False,
    )

    print("=" * 80)
    print("[B Done] classifier-based class-specific IG")
    print("Saved: class_position_importance_ig_classifier.npy")
    print("Saved: class_position_importance_ig_classifier.csv")
    print("=" * 80)

    return class_pos_imp, class_base_attr


# =========================================================
# C. seq-only classifier mutagenesis
# =========================================================

@torch.no_grad()
def score_in_batches(
    models,
    x: torch.Tensor,
    target_class: int,
    cfg: CFG,
    device: torch.device,
    batch_size: int,
):
    scores = []

    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size].to(device)

        sb = ensemble_class_score_no_grad(
            models=models,
            x4=xb,
            target_class=target_class,
            cfg=cfg,
        )

        scores.append(sb.detach().cpu())

    return torch.cat(scores, dim=0)


def compute_mutagenesis_importance_classifier(seqs_by_class, models, cfg: CFG, device: torch.device):
    """

      class_position_importance_mutagenesis_classifier.npy [5,80]
      class_mutation_effect_signed.npy [5,80,4]
      class_mutation_effect_abs.npy [5,80,4]
      class_base_preference_from_mutagenesis.npy [5,80,4]
    """
    class_pos_imp = np.zeros((cfg.NUM_CLASSES, cfg.SEQ_LEN), dtype=np.float32)
    class_mut_effect_signed = np.zeros((cfg.NUM_CLASSES, cfg.SEQ_LEN, 4), dtype=np.float32)
    class_mut_effect_abs = np.zeros((cfg.NUM_CLASSES, cfg.SEQ_LEN, 4), dtype=np.float32)

    bases_eye = torch.eye(4, dtype=torch.float32)

    for target_class in range(cfg.NUM_CLASSES):
        seqs = sample_sequences(
            seqs_by_class[target_class],
            max_samples=cfg.MUT_MAX_SAMPLES_PER_CLASS,
            seed=cfg.SEED + 100 + target_class,
        )

        print("=" * 80)
        print(f"[C Mut classifier] target_class={target_class}, n={len(seqs)}")
        print("=" * 80)

        X = seqs_to_tensor(seqs, cfg.SEQ_LEN, cfg.BASES)
        _, L, _ = X.shape

        score_orig = score_in_batches(
            models=models,
            x=X,
            target_class=target_class,
            cfg=cfg,
            device=device,
            batch_size=cfg.MUT_BATCH_SIZE,
        )

        orig_base_idx = X.argmax(dim=2)

        for pos in tqdm(range(L), desc=f"Mut class {target_class}"):
            pos_sum_abs = 0.0
            pos_count = 0

            for b in range(4):
                valid_mask = orig_base_idx[:, pos] != b
                valid_idx = torch.where(valid_mask)[0]

                if valid_idx.numel() == 0:
                    continue

                sum_delta = 0.0
                sum_abs = 0.0
                count = 0

                for start in range(0, valid_idx.numel(), cfg.MUT_BATCH_SIZE):
                    idx = valid_idx[start:start + cfg.MUT_BATCH_SIZE]

                    X_mut = X[idx].clone()
                    X_mut[:, pos, :] = bases_eye[b]

                    score_mut = score_in_batches(
                        models=models,
                        x=X_mut,
                        target_class=target_class,
                        cfg=cfg,
                        device=device,
                        batch_size=cfg.MUT_BATCH_SIZE,
                    )

                    delta = score_mut - score_orig[idx]

                    sum_delta += float(delta.sum().item())
                    sum_abs += float(delta.abs().sum().item())
                    count += int(delta.numel())

                if count > 0:
                    mean_delta = sum_delta / count
                    mean_abs = sum_abs / count

                    class_mut_effect_signed[target_class, pos, b] = mean_delta
                    class_mut_effect_abs[target_class, pos, b] = mean_abs

                    pos_sum_abs += sum_abs
                    pos_count += count

            if pos_count > 0:
                class_pos_imp[target_class, pos] = pos_sum_abs / pos_count
            else:
                class_pos_imp[target_class, pos] = 0.0

    np.save(
        os.path.join(cfg.OUT_DIR, "class_position_importance_mutagenesis_classifier.npy"),
        class_pos_imp,
    )
    np.save(
        os.path.join(cfg.OUT_DIR, "class_mutation_effect_signed.npy"),
        class_mut_effect_signed,
    )
    np.save(
        os.path.join(cfg.OUT_DIR, "class_mutation_effect_abs.npy"),
        class_mut_effect_abs,
    )

    pref = class_mut_effect_signed.copy()
    pref = pref - pref.max(axis=2, keepdims=True)
    pref = np.exp(pref)
    pref = pref / (pref.sum(axis=2, keepdims=True) + 1e-8)
    pref = pref.astype(np.float32)

    np.save(
        os.path.join(cfg.OUT_DIR, "class_base_preference_from_mutagenesis.npy"),
        pref,
    )

    rows = []

    for pos in range(cfg.SEQ_LEN):
        row = {
            "position_index": pos,
            "position_relative": pos - cfg.SEQ_LEN,
        }

        for c in range(cfg.NUM_CLASSES):
            row[f"class{c}_mut_importance"] = class_pos_imp[c, pos]

            for bi, b in enumerate(cfg.BASES):
                row[f"class{c}_mut_signed_to_{b}"] = class_mut_effect_signed[c, pos, bi]
                row[f"class{c}_mut_abs_to_{b}"] = class_mut_effect_abs[c, pos, bi]
                row[f"class{c}_base_pref_{b}"] = pref[c, pos, bi]

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        os.path.join(cfg.OUT_DIR, "class_position_importance_mutagenesis_classifier.csv"),
        index=False,
    )

    print("=" * 80)
    print("[C Done] classifier-based class-specific mutagenesis")
    print("Saved: class_position_importance_mutagenesis_classifier.npy")
    print("Saved: class_base_preference_from_mutagenesis.npy")
    print("Saved: class_position_importance_mutagenesis_classifier.csv")
    print("=" * 80)

    return class_pos_imp, class_mut_effect_signed, class_mut_effect_abs, pref


# =========================================================
# D. rank fusion + buckets + continuous weights
# =========================================================

def load_optional(path: str):
    if os.path.exists(path):
        return np.load(path)
    return None


def fuse_and_make_buckets_weights(cfg: CFG):
    base_path = os.path.join(cfg.OUT_DIR, "class_position_importance_base.npy")
    ig_path = os.path.join(cfg.OUT_DIR, "class_position_importance_ig_classifier.npy")
    mut_path = os.path.join(cfg.OUT_DIR, "class_position_importance_mutagenesis_classifier.npy")

    base_imp = load_optional(base_path)
    ig_imp = load_optional(ig_path)
    mut_imp = load_optional(mut_path)

    available = []

    if base_imp is not None:
        available.append("base")

    if ig_imp is not None:
        available.append("ig_classifier")

    if mut_imp is not None:
        available.append("mutagenesis_classifier")

    if not available:
        raise ValueError("No importance files found for fusion.")

    print("=" * 80)
    print("[D Fusion]")
    print("Available:", available)
    print("=" * 80)

    combined = np.zeros((cfg.NUM_CLASSES, cfg.SEQ_LEN), dtype=np.float32)

    for c in range(cfg.NUM_CLASSES):
        parts = []
        weights = []

        if base_imp is not None:
            parts.append(percentile_rank_vector(base_imp[c]))
            weights.append(cfg.FUSION_WEIGHT_BASE)

        if ig_imp is not None:
            parts.append(percentile_rank_vector(ig_imp[c]))
            weights.append(cfg.FUSION_WEIGHT_IG)

        if mut_imp is not None:
            parts.append(percentile_rank_vector(mut_imp[c]))
            weights.append(cfg.FUSION_WEIGHT_MUT)

        weights = np.asarray(weights, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-8)

        fused = np.zeros(cfg.SEQ_LEN, dtype=np.float32)

        for p, w in zip(parts, weights):
            fused += w * p.astype(np.float32)

        combined[c] = fused

    bucket_ids = np.zeros((cfg.NUM_CLASSES, cfg.SEQ_LEN), dtype=np.int64)
    position_weights = np.zeros((cfg.NUM_CLASSES, cfg.SEQ_LEN), dtype=np.float32)

    for c in range(cfg.NUM_CLASSES):
        bucket_ids[c] = make_buckets_by_quantile(combined[c], cfg.NUM_BUCKETS)
        position_weights[c] = normalize_weight_mean_one(
            combined[c],
            w_min=cfg.POSITION_WEIGHT_MIN,
            w_max=cfg.POSITION_WEIGHT_MAX,
        )

    np.save(
        os.path.join(cfg.OUT_DIR, "class_position_importance_combined.npy"),
        combined,
    )
    np.save(
        os.path.join(cfg.OUT_DIR, f"class_specific_bucket_ids_combined_{cfg.NUM_BUCKETS}bins.npy"),
        bucket_ids,
    )
    np.save(
        os.path.join(cfg.OUT_DIR, f"class_position_weights_combined_{cfg.NUM_BUCKETS}bins.npy"),
        position_weights,
    )

    rows = []

    for pos in range(cfg.SEQ_LEN):
        row = {
            "position_index": pos,
            "position_relative": pos - cfg.SEQ_LEN,
        }

        for c in range(cfg.NUM_CLASSES):
            row[f"class{c}_combined_importance"] = combined[c, pos]
            row[f"class{c}_bucket"] = bucket_ids[c, pos]
            row[f"class{c}_weight"] = position_weights[c, pos]

            if base_imp is not None:
                row[f"class{c}_base_importance"] = base_imp[c, pos]
                row[f"class{c}_base_rank"] = percentile_rank_vector(base_imp[c])[pos]

            if ig_imp is not None:
                row[f"class{c}_ig_importance"] = ig_imp[c, pos]
                row[f"class{c}_ig_rank"] = percentile_rank_vector(ig_imp[c])[pos]

            if mut_imp is not None:
                row[f"class{c}_mut_importance"] = mut_imp[c, pos]
                row[f"class{c}_mut_rank"] = percentile_rank_vector(mut_imp[c])[pos]

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        os.path.join(cfg.OUT_DIR, "class_position_importance_combined.csv"),
        index=False,
    )

    summary_rows = []

    for c in range(cfg.NUM_CLASSES):
        summary_rows.append({
            "class": c,
            "combined_min": float(combined[c].min()),
            "combined_max": float(combined[c].max()),
            "combined_mean": float(combined[c].mean()),
            "weight_min": float(position_weights[c].min()),
            "weight_max": float(position_weights[c].max()),
            "weight_mean": float(position_weights[c].mean()),
            "bucket_counts": str(np.bincount(bucket_ids[c], minlength=cfg.NUM_BUCKETS).tolist()),
        })

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(cfg.OUT_DIR, "class_position_importance_summary.csv"),
        index=False,
    )

    print("=" * 80)
    print("[D Done] fusion + buckets + weights")
    print("Saved:")
    print("  class_position_importance_combined.npy")
    print(f"  class_specific_bucket_ids_combined_{cfg.NUM_BUCKETS}bins.npy")
    print(f"  class_position_weights_combined_{cfg.NUM_BUCKETS}bins.npy")
    print("  class_position_importance_combined.csv")
    print("  class_position_importance_summary.csv")
    print("-" * 80)

    for c in range(cfg.NUM_CLASSES):
        print(f"Class {c} bucket counts:", np.bincount(bucket_ids[c], minlength=cfg.NUM_BUCKETS))
        print(
            f"Class {c} weight min={position_weights[c].min():.4f}, "
            f"max={position_weights[c].max():.4f}, "
            f"mean={position_weights[c].mean():.4f}"
        )

    print("=" * 80)

    return combined, bucket_ids, position_weights


# =========================================================
# =========================================================

def main():
    set_seed(cfg.SEED)
    ensure_dir(cfg.OUT_DIR)

    with open(
        os.path.join(cfg.OUT_DIR, "run_config_seqonly_classifier_weights_full.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    if "cuda" in cfg.DEVICE and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.DEVICE)

    print("=" * 80)
    print("[Start seq-only classifier-based class-specific position weight generation - FULL DATA]")
    print(f"Output dir: {cfg.OUT_DIR}")
    print(f"Device: {device}")
    print("=" * 80)

    seqs_by_class = load_class_sequences(
        path=cfg.CLASS_DATA_PATH,
        seq_len=cfg.SEQ_LEN,
        num_classes=cfg.NUM_CLASSES,
    )

    models = None

    if cfg.RUN_A_BASE_DISTRIBUTION:
        compute_base_distribution_importance(seqs_by_class, cfg)

    if cfg.RUN_B_IG or cfg.RUN_C_MUTAGENESIS:
        models = load_classifier_models(cfg, device)

    if cfg.RUN_B_IG:
        compute_ig_importance_classifier(seqs_by_class, models, cfg, device)

    if cfg.RUN_C_MUTAGENESIS:
        compute_mutagenesis_importance_classifier(seqs_by_class, models, cfg, device)

    if cfg.RUN_D_FUSION_BUCKET_WEIGHT:
        fuse_and_make_buckets_weights(cfg)

    print("=" * 80)
    print("[All Done]")
    print(f"Results saved to: {cfg.OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()