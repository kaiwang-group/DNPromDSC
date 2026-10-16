from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model_dnabert2_mlp import DNABERT2_Hybrid_Regression
from train_DNABert2_MLP import DNASequenceRegressionDataset, compute_metrics

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "dataset" / "wrc_regression_63468_5fold"
CHECKPOINT_ROOT = ROOT / "checkpoint_dnabert2"
OUT_ROOT = ROOT / "results_evaluation_dnabert2_regression"
MODEL_PATH = ROOT / "DNABERT_2"
BATCH_SIZE, MAX_LEN = 256, 100


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    true, pred = [], []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        output = model(ids, mask)
        target = batch["strength"].view(-1, 1)

        if output.shape != target.shape:
            raise RuntimeError(
                f"Shape mismatch: {tuple(output.shape)} vs {tuple(target.shape)}"
            )

        true.append(target.numpy())
        pred.append(output.cpu().numpy())

    true = np.concatenate(true).reshape(-1)
    pred = np.concatenate(pred).reshape(-1)
    return compute_metrics(true, pred), true, pred


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), trust_remote_code=True
    )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []

    for fold in range(1, 6):
        data_path = DATA_ROOT / f"fold{fold}" / "val.csv"
        checkpoint = CHECKPOINT_ROOT / f"fold{fold}" / "best_model.pth"

        if not data_path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"Missing {data_path} or {checkpoint}")

        df = pd.read_csv(data_path)
        dataset = DNASequenceRegressionDataset(
            df["seq"], df["strength"], tokenizer, MAX_LEN,
            kmer=6, do_kmer=False
        )
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, drop_last=False
        )

        print(
            f"Fold {fold} | loading {checkpoint} "
            f"({checkpoint.stat().st_size} bytes)"
        )
        try:
            weights = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load {checkpoint}; check the checkpoint file itself."
            ) from exc

        model = DNABERT2_Hybrid_Regression(
            256, 0.2, True, "attn"
        ).to(device)
        model.load_state_dict(weights, strict=True)

        metrics, true, pred = evaluate(model, loader, device)
        results.append({"Fold": fold, **metrics})
        print(" | ".join(
            f"{name}={value:.6f}" for name, value in metrics.items()
        ))

        fold_out = OUT_ROOT / f"fold{fold}"
        fold_out.mkdir(exist_ok=True)
        pd.DataFrame({
            "true_strength": true,
            "pred_strength": pred
        }).to_csv(fold_out / "val_predictions.csv", index=False)
        pd.DataFrame([results[-1]]).to_csv(
            fold_out / "fold_metrics.csv", index=False
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    table = pd.DataFrame(results)
    table.to_csv(OUT_ROOT / "fold_metrics.csv", index=False)

    lines = [
        f"{name:<10}: {table[name].mean():.6f} ± "
        f"{table[name].std(ddof=0):.6f}"
        for name in ("R2", "RMSE", "MAE", "Pearson", "Spearman")
    ]
    summary = "DNABERT2 Regression 5-Fold Summary\n" + "\n".join(lines) + "\n"
    (OUT_ROOT / "final_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()