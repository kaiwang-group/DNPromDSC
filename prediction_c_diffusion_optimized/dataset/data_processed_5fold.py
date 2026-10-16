import os
import re
import random
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

# ===== Configuration =====
DATASET_DIR = Path(__file__).resolve().parent
DATA_PATH = DATASET_DIR / "wrc_class5_63468.txt"  # Each line: sequence<TAB>label (0–4)
OUT_ROOT = DATASET_DIR / "wrc_processed_5fold"
SEQ_LEN = 80
NUM_CLASSES = 5
VAL_RATIO = 0.2
KFOLD = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ===== Read sequences and labels (0–4) =====
seq_str_list, label_list = [], []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for ln, line in enumerate(tqdm(f, desc="Reading"), start=1):
        s = line.strip()
        if not s:
            continue
        parts = re.split(r"[\t ]+", s)
        if len(parts) < 2:
            continue
        seq, lab_str = parts[0], parts[1]

        # Skip a header such as "Sequence<TAB>Class".
        if (not lab_str.replace(".", "", 1).isdigit()) and lab_str.lower() in {"class", "label"}:
            continue

        try:
            lab = int(float(lab_str))
        except ValueError:
            m = re.search(r"(\d+)", lab_str)
            if not m:
                raise
            lab = int(m.group(1))

        if not (0 <= lab < NUM_CLASSES):
            raise ValueError(f"Label out of range at line {ln}: {lab}")

        seq = seq.upper().replace("U", "T")
        seq_str_list.append(seq)
        label_list.append(lab)

y_all = np.asarray(label_list, dtype=np.int64)

print("Total samples:", len(seq_str_list))
uniq, cnt = np.unique(y_all, return_counts=True)
print("Label distribution:", dict(zip(uniq.tolist(), cnt.tolist())))

# ===== One-hot encoding: (N, SEQ_LEN, DEPTH=5) =====
PAD_TOKEN = "N"
vocab = ["A", "C", "G", "T"]
NUM_OOV = 1
tok2idx = {ch: i + NUM_OOV for i, ch in enumerate(vocab)}
DEPTH = len(vocab) + NUM_OOV  # 5

def pad_or_truncate(chars, L=SEQ_LEN, pad_value=PAD_TOKEN):
    """Pad or truncate a sequence to length L."""
    return chars[:L] if len(chars) >= L else chars + [pad_value] * (L - len(chars))

def one_hot_from_str(seq_str):
    """Convert a sequence string to an array of shape (SEQ_LEN, DEPTH)."""
    chars = list(seq_str)
    chars = pad_or_truncate(chars, L=SEQ_LEN)
    arr = np.zeros((SEQ_LEN, DEPTH), dtype=np.float32)
    for i, ch in enumerate(chars):
        idx = tok2idx.get(ch, 0)  # Map N and other characters to channel 0.
        arr[i, idx] = 1.0
    return arr

X_all = np.stack([one_hot_from_str(s) for s in seq_str_list], axis=0).astype(np.float32)
print("Encoded X:", X_all.shape, "y:", y_all.shape)

# ===== Compute CGR features =====
# Coordinates: A=(0,0), C=(0,1), G=(1,1), T=(1,0).
# Starting from (0.5,0.5), repeatedly move to the midpoint.
# Count points in a grid and normalize the counts to frequencies.
CGR_BINS = 16
CGR_DIM = CGR_BINS * CGR_BINS
base2coord = {
    "A": (0.0, 0.0),
    "C": (0.0, 1.0),
    "G": (1.0, 1.0),
    "T": (1.0, 0.0)
}

def cgr_from_str(seq_str, n_bins=CGR_BINS):
    # Exclude characters other than A, C, G, and T.
    seq = [ch for ch in seq_str if ch in base2coord]
    if len(seq) == 0:
        return np.zeros((n_bins * n_bins,), dtype=np.float32)

    x, y = 0.5, 0.5
    grid = np.zeros((n_bins, n_bins), dtype=np.float32)

    for ch in seq:
        vx, vy = base2coord[ch]
        x = (x + vx) / 2.0
        y = (y + vy) / 2.0
        ix = int(x * n_bins)
        iy = int(y * n_bins)
        if ix == n_bins:
            ix = n_bins - 1
        if iy == n_bins:
            iy = n_bins - 1
        grid[iy, ix] += 1.0

    total = grid.sum()
    if total > 0:
        grid /= total
    return grid.reshape(-1).astype(np.float32)

print("Computing CGR features...")
cgr_all = np.stack([cgr_from_str(s) for s in tqdm(seq_str_list, desc="CGR")], axis=0)
print("CGR_all shape:", cgr_all.shape)

os.makedirs(OUT_ROOT, exist_ok=True)

def save_pair(tr_idx, va_idx, out_dir, dump_lists=False):
    """Save training and validation datasets as .pt and .csv files."""
    os.makedirs(out_dir, exist_ok=True)
    import torch

    # Split encoded sequences, CGR features, and labels.
    Xtr = X_all[tr_idx].astype(np.float32)
    Xva = X_all[va_idx].astype(np.float32)
    Etr = cgr_all[tr_idx].astype(np.float32)
    Eva = cgr_all[va_idx].astype(np.float32)
    ytr = y_all[tr_idx].astype(np.int64)
    yva = y_all[va_idx].astype(np.int64)

    # Save datasets used for training.
    torch.save({"X": Xtr, "extra": Etr, "y": ytr}, os.path.join(out_dir, "train_dataset.pt"))
    torch.save({"X": Xva, "extra": Eva, "y": yva}, os.path.join(out_dir, "val_dataset.pt"))

    # Save original sequences and labels for inspection.
    train_seqs = [seq_str_list[i] for i in tr_idx]
    val_seqs = [seq_str_list[i] for i in va_idx]
    df_tr = pd.DataFrame({"seq": train_seqs, "label": y_all[tr_idx]})
    df_va = pd.DataFrame({"seq": val_seqs, "label": y_all[va_idx]})
    df_tr.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    df_va.to_csv(os.path.join(out_dir, "val.csv"), index=False)

    # Optionally save the complete sequence and label lists.
    if dump_lists:
        pickle.dump(seq_str_list, open(os.path.join(out_dir, "all_seq_list.pkl"), "wb"))
        pickle.dump(label_list, open(os.path.join(out_dir, "all_label_list.pkl"), "wb"))

    print(f"[SAVE] .pt + .csv -> {out_dir}")

# ===== 1) Stratified 80:20 split =====
sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
tr_idx, va_idx = next(sss.split(X_all, y_all))

holdout_dir = os.path.join(OUT_ROOT, f"preprocessed_data_class{NUM_CLASSES}_63468_8_2")
save_pair(tr_idx, va_idx, holdout_dir, dump_lists=True)

# ===== 2) Stratified five-fold split =====
kroot = os.path.join(OUT_ROOT, "kfold5_new")
os.makedirs(kroot, exist_ok=True)

skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
for i, (tr, va) in enumerate(skf.split(X_all, y_all), start=1):
    fold_dir = os.path.join(kroot, f"fold{i}")
    save_pair(tr, va, fold_dir, dump_lists=False)
    print(f"[KFOLD] fold{i} -> {fold_dir}  (Ntr={len(tr)}, Nva={len(va)})")

print("[DONE] preprocess OK.")