# -*- coding: utf-8 -*-
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ===== Config =====
DATASET_DIR = Path(__file__).resolve().parent
DATA_PATH = DATASET_DIR / "merged_dataset63468.csv"
OUT_ROOT = DATASET_DIR / "wrc_regression_63468_5fold"
SEQ_COL = "seq"
STRENGTH_COL = "strength"
NUM_BINS = 5
N_SPLITS = 5
SEED = 42

# ===== Load and clean data =====
df = pd.read_csv(DATA_PATH, header=None, names=["seq", "strength"])
print("=" * 100)
print("[Regression 5-Fold Preprocessing]")
print(f"DATA_PATH : {DATA_PATH}")
print(f"OUT_ROOT  : {OUT_ROOT}")
print(f"N_SPLITS  : {N_SPLITS}")
print(f"NUM_BINS  : {NUM_BINS}")
print(f"SEED      : {SEED}")
print("=" * 100)

before_n = len(df)
df = df.dropna(how="all").copy()
df["seq"] = df["seq"].astype(str).str.strip().str.upper()
df["strength"] = pd.to_numeric(df["strength"], errors="coerce")
df = df.dropna(subset=["seq", "strength"]).reset_index(drop=True)
after_n = len(df)

print(f"\nOriginal samples : {before_n}")
print(f"Valid samples    : {after_n}")
print(f"\nTotal samples: {len(df)}")
if len(df) != 63468:
    print(f"[Warning] Expected 63468 samples, but found {len(df)}.")

# ===== Create five equal-frequency strength bins =====
sorted_idx = np.argsort(df[STRENGTH_COL].values, kind="mergesort")
n_samples = len(df)
base_bin_size = n_samples // NUM_BINS
bin_sizes = [base_bin_size for _ in range(NUM_BINS - 1)]
last_bin_size = n_samples - base_bin_size * (NUM_BINS - 1)
bin_sizes.append(last_bin_size)

print("\nPlanned bin sizes:")
for i, size in enumerate(bin_sizes):
    print(f"Bin {i}: {size}")

bin_label = np.zeros(n_samples, dtype=np.int64)
start = 0
for bin_id, size in enumerate(bin_sizes):
    end = start + size
    indices_this_bin = sorted_idx[start:end]
    bin_label[indices_this_bin] = bin_id
    start = end
df["bin"] = bin_label

# ===== Check global bin distribution =====
print("\n" + "=" * 100)
print("Global Bin Distribution")
print("=" * 100)
global_counts = df["bin"].value_counts().sort_index()
for bin_id in range(NUM_BINS):
    count = int(global_counts.loc[bin_id])
    ratio = count / len(df)
    subset = df[df["bin"] == bin_id]
    print(
        f"Bin {bin_id} | N={count:6d} | Ratio={ratio:.6f} | "
        f"Strength range=[{subset[STRENGTH_COL].min():.6f}, "
        f"{subset[STRENGTH_COL].max():.6f}]"
    )

print("\nOverall strength statistics:")
print(f"Mean : {df[STRENGTH_COL].mean():.6f}")
print(f"Std  : {df[STRENGTH_COL].std():.6f}")
print(f"Min  : {df[STRENGTH_COL].min():.6f}")
print(f"Max  : {df[STRENGTH_COL].max():.6f}")

# ===== Save full dataset with bin labels =====
os.makedirs(OUT_ROOT, exist_ok=True)
full_data_path = os.path.join(OUT_ROOT, "full_dataset_with_bins.csv")
df.to_csv(full_data_path, index=False)
print(f"\nSaved full dataset with bins -> {full_data_path}")

# ===== Stratified five-fold split =====
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
all_indices = np.arange(len(df))
fold_summary_list = []

for fold_idx, (train_idx, val_idx) in enumerate(
    skf.split(all_indices, df["bin"].values), start=1
):
    print("\n" + "=" * 100)
    print(f"Fold {fold_idx}")
    print("=" * 100)

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)
    fold_dir = os.path.join(OUT_ROOT, f"fold{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    train_path = os.path.join(fold_dir, "train.csv")
    val_path = os.path.join(fold_dir, "val.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Train samples : {len(train_df)}")
    print(f"Val samples   : {len(val_df)}")
    print(f"Train ratio   : {len(train_df) / len(df):.6f}")
    print(f"Val ratio     : {len(val_df) / len(df):.6f}")

    print("\nTrain bin distribution:")
    train_bin_counts = train_df["bin"].value_counts().sort_index()
    for bin_id in range(NUM_BINS):
        count = int(train_bin_counts.loc[bin_id])
        ratio = count / len(train_df)
        print(f"  Bin {bin_id}: N={count:6d} | Ratio={ratio:.6f}")

    print("\nValidation bin distribution:")
    val_bin_counts = val_df["bin"].value_counts().sort_index()
    for bin_id in range(NUM_BINS):
        count = int(val_bin_counts.loc[bin_id])
        ratio = count / len(val_df)
        print(f"  Bin {bin_id}: N={count:6d} | Ratio={ratio:.6f}")

    train_mean = train_df[STRENGTH_COL].mean()
    train_std = train_df[STRENGTH_COL].std()
    train_min = train_df[STRENGTH_COL].min()
    train_max = train_df[STRENGTH_COL].max()
    val_mean = val_df[STRENGTH_COL].mean()
    val_std = val_df[STRENGTH_COL].std()
    val_min = val_df[STRENGTH_COL].min()
    val_max = val_df[STRENGTH_COL].max()

    print("\nContinuous strength statistics:")
    print(
        f"Train | Mean={train_mean:.6f} | Std={train_std:.6f} | "
        f"Min={train_min:.6f} | Max={train_max:.6f}"
    )
    print(
        f"Val   | Mean={val_mean:.6f} | Std={val_std:.6f} | "
        f"Min={val_min:.6f} | Max={val_max:.6f}"
    )
    print("\nTrain-Val difference:")
    print(f"Mean diff = {abs(train_mean - val_mean):.6f}")
    print(f"Std diff  = {abs(train_std - val_std):.6f}")

    overlap = np.intersect1d(train_idx, val_idx)
    if len(overlap) != 0:
        raise RuntimeError(f"Fold {fold_idx} has {len(overlap)} overlapping samples!")
    print("\nTrain / Val overlap: 0")

    report_rows = []
    for bin_id in range(NUM_BINS):
        tr_count = int(train_bin_counts.loc[bin_id])
        va_count = int(val_bin_counts.loc[bin_id])
        report_rows.append({
            "Fold": fold_idx,
            "Bin": bin_id,
            "Train_Count": tr_count,
            "Train_Ratio": tr_count / len(train_df),
            "Val_Count": va_count,
            "Val_Ratio": va_count / len(val_df)
        })

    report_path = os.path.join(fold_dir, "bin_distribution.csv")
    pd.DataFrame(report_rows).to_csv(report_path, index=False)

    fold_summary_list.append({
        "Fold": fold_idx,
        "Train_N": len(train_df),
        "Val_N": len(val_df),
        "Train_Mean": train_mean,
        "Train_Std": train_std,
        "Train_Min": train_min,
        "Train_Max": train_max,
        "Val_Mean": val_mean,
        "Val_Std": val_std,
        "Val_Min": val_min,
        "Val_Max": val_max,
        "Mean_Diff": abs(train_mean - val_mean),
        "Std_Diff": abs(train_std - val_std)
    })

    print("\nSaved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {report_path}")

# ===== Save fold summary =====
fold_summary_path = os.path.join(OUT_ROOT, "five_fold_summary.csv")
pd.DataFrame(fold_summary_list).to_csv(fold_summary_path, index=False)

# ===== Verify validation coverage =====
val_seen = np.zeros(len(df), dtype=np.int64)
for _, val_idx in skf.split(all_indices, df["bin"].values):
    val_seen[val_idx] += 1

unique_seen, seen_counts = np.unique(val_seen, return_counts=True)
print("\n" + "=" * 100)
print("Final Validation Coverage Check")
print("=" * 100)
print(dict(zip(unique_seen.tolist(), seen_counts.tolist())))

if not np.all(val_seen == 1):
    raise RuntimeError("Some samples were not used exactly once as validation samples.")
print("Every sample appears exactly once in the validation sets.")

print("\n" + "=" * 100)
print("[DONE] Regression stratified 5-fold preprocessing completed.")
print("=" * 100)
print("\nFull dataset:")
print(full_data_path)
print("\n5-fold summary:")
print(fold_summary_path)
print("\nFold folders:")
for fold_idx in range(1, N_SPLITS + 1):
    print(os.path.join(OUT_ROOT, f"fold{fold_idx}"))