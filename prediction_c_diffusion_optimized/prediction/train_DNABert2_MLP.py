import os
import random
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from model_dnabert2_mlp import DNABERT2_Hybrid_Regression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
SEED = 42
K_FOLDS_ROOT = str(ROOT.parent / "dataset" / "wrc_regression_63468_5fold")
SAVE_ROOT = str(ROOT / "checkpoint_dnabert2")
MODEL_PATH = str(ROOT / "DNABERT_2")
BATCH_SIZE, MAX_LEN = 256, 150
WEIGHT_DECAY, PATIENCE, MAX_EPOCHS, WARMUP_EPOCHS = 1e-2, 10, 150, 5
BERT_LR, HEAD_LR, GRAD_CLIP = 2e-5, 1e-3, 1.0
KMER, DO_KMER, NUM_WORKERS = 6, False, 4


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">> Random seed set to {seed}")


class DNASequenceRegressionDataset(Dataset):
    def __init__(self, sequences, strengths, tokenizer, max_len, kmer=6, do_kmer=False):
        self.sequences, self.strengths = list(sequences), list(strengths)
        self.tokenizer, self.max_len = tokenizer, max_len
        self.kmer, self.do_kmer = kmer, do_kmer

    def __len__(self):
        return len(self.sequences)

    def _seq2kmer(self, seq):
        k, seq = self.kmer, str(seq).upper()
        if len(seq) < k:
            return seq
        return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))

    def __getitem__(self, idx):
        seq = str(self.sequences[idx])
        text = self._seq2kmer(seq) if self.do_kmer else seq
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "strength": torch.tensor(float(self.strengths[idx]), dtype=torch.float32)
        }


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) != len(y_pred):
        raise RuntimeError(f"Length mismatch: true={len(y_true)}, pred={len(y_pred)}")

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        pearson = spearman = 0.0
    else:
        pearson = pearsonr(y_true, y_pred)[0]
        spearman = spearmanr(y_true, y_pred)[0]
        if np.isnan(pearson):
            pearson = 0.0
        if np.isnan(spearman):
            spearman = 0.0

    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Pearson": float(pearson),
        "Spearman": float(spearman)
    }


class DNABERT2_Regression_Trainer:
    def __init__(self, kfold_root=K_FOLDS_ROOT, save_root=SAVE_ROOT,
                 model_path=MODEL_PATH, batch_size=BATCH_SIZE, max_len=MAX_LEN,
                 weight_decay=WEIGHT_DECAY, patience=PATIENCE,
                 max_epochs=MAX_EPOCHS, warmup_epochs=WARMUP_EPOCHS,
                 kmer=KMER, do_kmer=DO_KMER):
        self.kfold_root, self.save_root, self.model_path = kfold_root, save_root, model_path
        self.batch_size, self.max_len, self.weight_decay = batch_size, max_len, weight_decay
        self.patience, self.max_epochs, self.warmup_epochs = patience, max_epochs, warmup_epochs
        self.kmer, self.do_kmer = kmer, do_kmer

        os.makedirs(self.save_root, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">> Loading Tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.use_amp = self.device.type == "cuda"
        print(f">> Trainer init | device={self.device}, batch_size={batch_size}, "
              f"max_len={max_len}, kmer={kmer}, do_kmer={do_kmer}")

    def load_fold_data(self, fold_idx):
        fold_dir = os.path.join(self.kfold_root, f"fold{fold_idx}")
        df_train = pd.read_csv(os.path.join(fold_dir, "train.csv"))
        df_val = pd.read_csv(os.path.join(fold_dir, "val.csv"))

        for name, df in (("train", df_train), ("val", df_val)):
            for col in ("seq", "strength"):
                if col not in df.columns:
                    raise ValueError(f"{name}.csv has no {col} column.")

        def loader(df, shuffle):
            ds = DNASequenceRegressionDataset(
                df["seq"].tolist(), df["strength"].tolist(),
                self.tokenizer, self.max_len, self.kmer, self.do_kmer
            )
            return DataLoader(
                ds, batch_size=self.batch_size, shuffle=shuffle,
                num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
            )

        print(f"[Fold {fold_idx}] Train={len(df_train)} | Val={len(df_val)} | "
              f"Train strength=[{df_train['strength'].min():.4f}, "
              f"{df_train['strength'].max():.4f}]")
        return loader(df_train, True), loader(df_val, False)

    def build_model(self):
        return DNABERT2_Hybrid_Regression(256, 0.2, True, "attn").to(self.device)

    @torch.no_grad()
    def evaluate(self, model, loader, criterion):
        model.eval()
        preds, targets, total_loss, total_samples = [], [], 0.0, 0
        for batch in loader:
            ids = batch["input_ids"].to(self.device, non_blocking=True)
            mask = batch["attention_mask"].to(self.device, non_blocking=True)
            target = batch["strength"].to(self.device, non_blocking=True).view(-1, 1)
            pred = model(ids, mask)
            if pred.shape != target.shape:
                raise RuntimeError(
                    f"DNABERT2 regression shape mismatch: "
                    f"pred={tuple(pred.shape)}, target={tuple(target.shape)}"
                )
            total_loss += criterion(pred, target).item() * target.size(0)
            total_samples += target.size(0)
            preds.append(pred.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())

        y_pred = np.concatenate(preds, axis=0).reshape(-1)
        y_true = np.concatenate(targets, axis=0).reshape(-1)
        return (
            total_loss / max(total_samples, 1),
            compute_metrics(y_true, y_pred),
            y_true,
            y_pred
        )

    def train_one_fold(self, fold_idx):
        print(f"\n{'=' * 100}\nDNABERT2 Regression | Fold {fold_idx}\n{'=' * 100}")
        fold_save_dir = os.path.join(self.save_root, f"fold{fold_idx}")
        os.makedirs(fold_save_dir, exist_ok=True)
        best_path = os.path.join(fold_save_dir, "best_model.pth")
        train_dl, val_dl = self.load_fold_data(fold_idx)
        model = self.build_model()
        criterion = nn.MSELoss()

        bert_params, head_params = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                (bert_params if name.startswith("bert.") else head_params).append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": bert_params, "lr": BERT_LR},
                {"params": head_params, "lr": HEAD_LR}
            ],
            weight_decay=self.weight_decay
        )
        scaler = GradScaler(enabled=self.use_amp)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=5 * len(train_dl), T_mult=2, eta_min=1e-6
        )
        best_r2, best_epoch, patience_counter, history = -float("inf"), 0, 0, []

        for ep in range(self.max_epochs):
            model.train()
            total_loss, total_samples = 0.0, 0

            if ep < self.warmup_epochs:
                factor = float(ep + 1) / float(self.warmup_epochs)
                optimizer.param_groups[0]["lr"] = BERT_LR * factor
                optimizer.param_groups[1]["lr"] = HEAD_LR * factor

            pbar = tqdm(train_dl, desc=f"[Fold {fold_idx}] Ep {ep + 1:03d}", leave=False)
            for batch in pbar:
                ids = batch["input_ids"].to(self.device, non_blocking=True)
                mask = batch["attention_mask"].to(self.device, non_blocking=True)
                target = batch["strength"].to(
                    self.device, non_blocking=True
                ).view(-1, 1)
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=self.use_amp):
                    pred = model(ids, mask)
                    if pred.shape != target.shape:
                        raise RuntimeError(
                            f"Regression shape mismatch: "
                            f"pred={tuple(pred.shape)}, target={tuple(target.shape)}"
                        )
                    loss = criterion(pred, target)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()

                if ep >= self.warmup_epochs:
                    scheduler.step()

                total_loss += loss.item() * target.size(0)
                total_samples += target.size(0)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    bert_lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                    head_lr=f"{optimizer.param_groups[1]['lr']:.1e}"
                )

            train_loss = total_loss / max(total_samples, 1)
            val_loss, metrics, _, _ = self.evaluate(model, val_dl, criterion)
            torch.cuda.empty_cache()
            history.append({
                "Epoch": ep + 1,
                "Train_Loss": train_loss,
                "Val_Loss": val_loss,
                **metrics
            })

            mark = ""
            if metrics["R2"] > best_r2:
                best_r2, best_epoch, patience_counter = metrics["R2"], ep + 1, 0
                torch.save(model.state_dict(), best_path)
                mark = "*"
            else:
                patience_counter += 1

            print(
                f"Fold {fold_idx} | Ep {ep + 1:03d}/{self.max_epochs} | "
                f"Train Loss {train_loss:.5f} | Val Loss {val_loss:.5f} | "
                f"R2 {metrics['R2']:.4f} | RMSE {metrics['RMSE']:.4f} | "
                f"MAE {metrics['MAE']:.4f} | Pearson {metrics['Pearson']:.4f} | "
                f"Spearman {metrics['Spearman']:.4f} {mark}"
            )
            if patience_counter >= self.patience:
                print(
                    f"Early stopping at Epoch {ep + 1}. "
                    f"Best Epoch={best_epoch}, Best R2={best_r2:.6f}"
                )
                break

        pd.DataFrame(history).to_csv(
            os.path.join(fold_save_dir, "training_log.csv"), index=False
        )

        final_model = self.build_model()
        final_model.load_state_dict(torch.load(best_path, map_location=self.device))
        _, final_metrics, y_true, y_pred = self.evaluate(
            final_model, val_dl, criterion
        )

        pd.DataFrame({
            "true_strength": y_true,
            "pred_strength": y_pred
        }).to_csv(
            os.path.join(fold_save_dir, "val_predictions.csv"), index=False
        )

        result = {"Fold": fold_idx, "Best_Epoch": best_epoch, **final_metrics}
        pd.DataFrame([result]).to_csv(
            os.path.join(fold_save_dir, "fold_metrics.csv"), index=False
        )

        print(f"\n{'-' * 80}\nFinal DNABERT2 Fold {fold_idx}\n{'-' * 80}")
        print(f"Best Epoch : {best_epoch}")
        for metric in ("R2", "RMSE", "MAE", "Pearson", "Spearman"):
            print(f"{metric:<11}: {result[metric]:.6f}")
        return result

    def run(self):
        results = []
        for fold in range(1, 6):
            seed_everything(SEED)
            results.append(self.train_one_fold(fold))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.save_summary(results)

    def save_summary(self, results):
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.save_root, "fold_metrics.csv"), index=False)
        metrics = ("R2", "RMSE", "MAE", "Pearson", "Spearman")

        summary = (
            "\n" + "=" * 100 + "\nDNABERT2 Regression 5-Fold Summary\n"
            + "=" * 100 + "\n\n"
        )
        summary += (
            f"{'Fold':<8}{'Epoch':<10}{'R2':<14}{'RMSE':<14}"
            f"{'MAE':<14}{'Pearson':<14}{'Spearman':<14}\n"
        )
        for _, row in df.iterrows():
            summary += (
                f"{int(row['Fold']):<8}{int(row['Best_Epoch']):<10}"
                f"{row['R2']:<14.6f}{row['RMSE']:<14.6f}{row['MAE']:<14.6f}"
                f"{row['Pearson']:<14.6f}{row['Spearman']:<14.6f}\n"
            )
        summary += "\n" + "-" * 100 + "\n"

        final_row = {"Model": "DNABERT2_last4_attn"}
        for metric in metrics:
            values = df[metric].values.astype(np.float64)
            mean, std = np.mean(values), np.std(values)
            summary += f"{metric:<10}: {mean:.6f} ± {std:.6f}\n"
            final_row[metric] = f"{mean:.4f} ± {std:.4f}"

        print(summary)
        with open(
            os.path.join(self.save_root, "final_summary.txt"),
            "w",
            encoding="utf-8"
        ) as f:
            f.write(summary)
        pd.DataFrame([final_row]).to_csv(
            os.path.join(self.save_root, "final_summary_table.csv"),
            index=False
        )


if __name__ == "__main__":
    seed_everything(42)
    trainer = DNABERT2_Regression_Trainer(
        kfold_root=K_FOLDS_ROOT,
        save_root=SAVE_ROOT,
        model_path=MODEL_PATH,
        batch_size=256,
        max_len=100,
        weight_decay=1e-2,
        patience=10,
        max_epochs=150,
        warmup_epochs=5,
        kmer=6,
        do_kmer=False
    )
    trainer.run()