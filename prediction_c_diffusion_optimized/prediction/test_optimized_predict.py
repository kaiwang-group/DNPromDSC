from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
from model_optimized_predict import TransformerHybridMultiTask
from train_optimized_predict import (
    compute_batch_multires_cgr, compute_batch_psednc, compute_metrics
)

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "dataset" / "wrc_processed_5fold" / "kfold5_new"
CHECKPOINT_ROOT = ROOT / "checkpoint_prediction"
OUT_ROOT = ROOT / "evaluation_prediction"
BATCH_SIZE = 128


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, labels = [], []
    for x, cgr, pse, y in loader:
        logits = model(x.to(device), cgr.to(device), pse.to(device))
        predictions.append(logits.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    true = np.concatenate(labels)
    pred = np.concatenate(predictions)
    return compute_metrics(true, pred), true, pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []

    for fold in range(1, 6):
        folder = DATA_ROOT / f"fold{fold}"
        checkpoint = CHECKPOINT_ROOT / f"prediction_f{fold}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

        data = torch.load(
            folder / "val_dataset.pt", map_location="cpu", weights_only=False
        )
        seq = pd.read_csv(folder / "val.csv")["seq"].astype(str).tolist()
        x = torch.as_tensor(data["X"], dtype=torch.float32)
        y = torch.as_tensor(data["y"], dtype=torch.long)
        if len(seq) != len(x):
            raise RuntimeError(f"Fold {fold}: val.csv and val_dataset.pt have different lengths")

        cached_cgr = folder / "fcgr_multires_k4k3_val.npy"
        cgr = np.load(cached_cgr) if cached_cgr.exists() else compute_batch_multires_cgr(seq)
        pse = compute_batch_psednc(seq)
        if cgr.shape != (len(x), 2, 16, 16) or pse.shape != (len(x), 22):
            raise RuntimeError(f"Fold {fold}: unexpected feature shapes {cgr.shape}, {pse.shape}")

        loader = DataLoader(
            TensorDataset(
                x,
                torch.as_tensor(cgr, dtype=torch.float32),
                torch.as_tensor(pse, dtype=torch.float32),
                y,
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        print(f"Fold {fold}: loading {checkpoint}")
        model = TransformerHybridMultiTask(
            use_cgr=True, use_psednc=True, num_layers=2, nhead=8, psednc_dim=22
        ).to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True),
            strict=True,
        )

        metrics, true, pred = evaluate(model, loader, device)
        results.append({
            "fold": fold, "use_ema": True, "num_layers": 2,
            "nhead": 8, **metrics
        })
        print(" | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        pd.DataFrame({
            "seq": seq, "label": true, "prediction": pred
        }).to_csv(OUT_ROOT / f"fold{fold}_predictions.csv", index=False)
        (OUT_ROOT / f"fold{fold}_report.txt").write_text(
            classification_report(true, pred, digits=4), encoding="utf-8"
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    table = pd.DataFrame(results)
    table.to_csv(OUT_ROOT / "fold_metrics.csv", index=False)
    names = ("acc", "f1_m", "f1_w", "prec", "rec", "mcc")
    summary = "Classification 5-Fold Summary\n" + "\n".join(
        f"{k.upper():<5}: {table[k].mean():.4f} ± {table[k].std(ddof=0):.4f}"
        for k in names
    ) + "\n"
    (OUT_ROOT / "final_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()