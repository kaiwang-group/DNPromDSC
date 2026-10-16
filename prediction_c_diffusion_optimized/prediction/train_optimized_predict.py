import os
import random
import copy
import shutil
import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    classification_report
)

from model_optimized_predict import TransformerHybridMultiTask

warnings.filterwarnings("ignore")



K_FOLDS_ROOT = "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Data/SC/wrc_processed_5fold_CGR/kfold5"


BASE_SAVE_ROOT = "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Predictor/wrc_class5_cgr_gpt/new_simple/results_adaptive_film_v52_gpt_seed43_psednc6prop"

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
MAX_EPOCHS = 80

LR = 3e-4
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 5
SEQ_LEN = 80

LABEL_SMOOTH = 0.04
PATIENCE = 15
WARMUP_EPOCHS = 4

FORCE_RECOMPUTE_CGR = False
FORCE_RECOMPUTE_PSEDNC = True
BACKUP_OLD_PSEDNC_NPY = True

LAMBDA = 6
WEIGHT = 0.05
PSEDNC_DIM = 16 + LAMBDA

TRANSFORMER_CONFIGS = [
    {
        "tag": "tf2_h8",
        "num_layers": 2,
        "nhead": 8
    },
]

EXPERIMENTS = {
    "all_fusion": {
        "use_cgr": True,
        "use_psednc": True,
        "use_ema": True
    },
}

EXPERIMENTS_TO_RUN = [
    "all_fusion"
]



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
    seq = str(seq).upper().replace("U", "T").replace(" ", "")
    seq = "".join(ch for ch in seq if ch in "ACGT")
    if len(seq) > max_len:
        seq = seq[:max_len]
    return seq


def get_psednc(seq, lam=LAMBDA, w=WEIGHT):
    """


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
                c += np.mean(
                    (v[:-k] - v.mean()) * (v[k:] - v.mean())
                )

            c = c / float(len(PHYCHEM_KEYS))
            corrs.append(c)
    else:
        corrs = [0.0] * lam

    denom = 1.0 + w * float(np.sum(corrs))

    if abs(denom) < 1e-12:
        denom = 1.0

    vec = [f / denom for f in freqs] + [w * t / denom for t in corrs]

    return np.asarray(vec, dtype=np.float32)


def compute_batch_psednc(seqs):
    feats = [
        get_psednc(s, LAMBDA, WEIGHT)
        for s in seqs
    ]
    return np.stack(feats, axis=0).astype(np.float32)


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

    return np.asarray(batch_cgr, dtype=np.float32)


def backup_if_needed(path: str):
    if not os.path.exists(path):
        return

    backup_path = path + ".old_backup"

    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)
        print(f"[Backup] {path} -> {backup_path}")


def load_or_make_multires_cgr(fold_dir, seq_tr, seq_va, fold_idx):
    c_p_tr = os.path.join(fold_dir, "fcgr_multires_k4k3_train.npy")
    c_p_va = os.path.join(fold_dir, "fcgr_multires_k4k3_val.npy")

    need_make = (
        FORCE_RECOMPUTE_CGR
        or (not os.path.exists(c_p_tr))
        or (not os.path.exists(c_p_va))
    )

    if need_make:
        print(f"[Feature] Computing Multi-Res FCGR for Fold {fold_idx}...")
        C_tr = compute_batch_multires_cgr(seq_tr)
        C_va = compute_batch_multires_cgr(seq_va)

        np.save(c_p_tr, C_tr)
        np.save(c_p_va, C_va)

        print(f"[Feature] Saved: {c_p_tr} shape={C_tr.shape}")
        print(f"[Feature] Saved: {c_p_va} shape={C_va.shape}")
    else:
        C_tr = np.load(c_p_tr)
        C_va = np.load(c_p_va)
        print(f"[Feature] Loaded CGR npy Fold {fold_idx}: train={C_tr.shape}, val={C_va.shape}")

    return C_tr.astype(np.float32), C_va.astype(np.float32)


def load_or_make_psednc6(fold_dir, seq_tr, seq_va, fold_idx):
    p_p_tr = os.path.join(fold_dir, f"psednc_lam{LAMBDA}_train.npy")
    p_p_va = os.path.join(fold_dir, f"psednc_lam{LAMBDA}_val.npy")

    need_make = (
        FORCE_RECOMPUTE_PSEDNC
        or (not os.path.exists(p_p_tr))
        or (not os.path.exists(p_p_va))
    )

    if need_make:
        print(f"[Feature] Computing STANDARD 6-prop PseDNC for Fold {fold_idx}...")

        if BACKUP_OLD_PSEDNC_NPY:
            backup_if_needed(p_p_tr)
            backup_if_needed(p_p_va)

        P_tr = compute_batch_psednc(seq_tr)
        P_va = compute_batch_psednc(seq_va)

        if P_tr.shape[1] != PSEDNC_DIM or P_va.shape[1] != PSEDNC_DIM:
            raise RuntimeError(
                f"PseDNC dim error: train={P_tr.shape}, val={P_va.shape}, expected dim={PSEDNC_DIM}"
            )

        np.save(p_p_tr, P_tr)
        np.save(p_p_va, P_va)

        print(f"[Feature] Saved: {p_p_tr} shape={P_tr.shape}")
        print(f"[Feature] Saved: {p_p_va} shape={P_va.shape}")
    else:
        P_tr = np.load(p_p_tr)
        P_va = np.load(p_p_va)
        print(f"[Feature] Loaded PseDNC npy Fold {fold_idx}: train={P_tr.shape}, val={P_va.shape}")

    return P_tr.astype(np.float32), P_va.astype(np.float32)



def seed_everything(seed=43):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def create_ema(model):
    ema = copy.deepcopy(model)
    ema.eval()

    for p in ema.parameters():
        p.requires_grad_(False)

    return ema


@torch.no_grad()
def ema_update(model, ema, decay=0.999):
    msd = model.state_dict()
    esd = ema.state_dict()

    for k in esd:
        if esd[k].dtype.is_floating_point:
            esd[k].mul_(decay).add_(
                msd[k].detach().to(esd[k].device),
                alpha=1 - decay
            )
        else:
            esd[k].copy_(msd[k])


def get_ordinal_targets(y, num_classes):
    return (
        y.view(-1, 1)
        > torch.arange(num_classes - 1, device=y.device).view(1, -1)
    ).float()


def compute_metrics(y_true, y_pred):
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "f1_m": f1_score(y_true, y_pred, average="macro"),
        "f1_w": f1_score(y_true, y_pred, average="weighted"),
        "prec": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "rec": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }

    return metrics


def build_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )

        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda
    )



class Trainer:
    def __init__(self, save_root, exp_name, exp_cfg, model_cfg):
        self.save_root = save_root
        self.exp_name = exp_name
        self.model_cfg = model_cfg

        os.makedirs(save_root, exist_ok=True)

        self.use_cgr = exp_cfg["use_cgr"]
        self.use_pse = exp_cfg["use_psednc"]
        self.use_ema = exp_cfg.get("use_ema", False)

        self.num_layers = model_cfg["num_layers"]
        self.nhead = model_cfg["nhead"]
        self.config_tag = model_cfg["tag"]

        self.log_process_path = os.path.join(
            self.save_root,
            "training_process.txt"
        )
        self.log_summary_path = os.path.join(
            self.save_root,
            "summary_report.txt"
        )
        self.fold_metrics_path = os.path.join(
            self.save_root,
            "fold_metrics.csv"
        )
        self.final_summary_path = os.path.join(
            self.save_root,
            "final_summary.txt"
        )

    def log(self, msg, mode="process"):
        print(msg)

        path = self.log_process_path if mode == "process" else self.log_summary_path

        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def load_data(self, fold_idx):
        fold_dir = os.path.join(
            K_FOLDS_ROOT,
            f"fold{fold_idx}"
        )

        t_pt = torch.load(
            os.path.join(fold_dir, "train_dataset.pt"),
            weights_only=False
        )
        v_pt = torch.load(
            os.path.join(fold_dir, "val_dataset.pt"),
            weights_only=False
        )

        X_tr = torch.from_numpy(t_pt["X"]).float()
        y_tr = torch.from_numpy(t_pt["y"]).long()

        X_va = torch.from_numpy(v_pt["X"]).float()
        y_va = torch.from_numpy(v_pt["y"]).long()

        df_tr_path = os.path.join(fold_dir, "train.csv")
        df_va_path = os.path.join(fold_dir, "val.csv")

        if not os.path.exists(df_tr_path) or not os.path.exists(df_va_path):
            raise FileNotFoundError(
                f"Need train.csv / val.csv for feature recompute, but not found in {fold_dir}"
            )

        df_tr = pd.read_csv(df_tr_path)
        df_va = pd.read_csv(df_va_path)

        if "seq" not in df_tr.columns or "seq" not in df_va.columns:
            raise ValueError(
                f"train.csv / val.csv must contain column 'seq'. Current fold_dir={fold_dir}"
            )

        seq_tr = df_tr["seq"].astype(str).tolist()
        seq_va = df_va["seq"].astype(str).tolist()

        if len(seq_tr) != len(X_tr) or len(seq_va) != len(X_va):
            raise RuntimeError(
                f"Sequence count mismatch in fold{fold_idx}: "
                f"seq_tr={len(seq_tr)} X_tr={len(X_tr)}, "
                f"seq_va={len(seq_va)} X_va={len(X_va)}"
            )

        items_tr = [X_tr]
        items_va = [X_va]

        if self.use_cgr:
            C_tr, C_va = load_or_make_multires_cgr(
                fold_dir=fold_dir,
                seq_tr=seq_tr,
                seq_va=seq_va,
                fold_idx=fold_idx
            )

            items_tr.append(
                torch.from_numpy(C_tr).float()
            )
            items_va.append(
                torch.from_numpy(C_va).float()
            )

        if self.use_pse:
            P_tr, P_va = load_or_make_psednc6(
                fold_dir=fold_dir,
                seq_tr=seq_tr,
                seq_va=seq_va,
                fold_idx=fold_idx
            )

            items_tr.append(
                torch.from_numpy(P_tr).float()
            )
            items_va.append(
                torch.from_numpy(P_va).float()
            )

        items_tr.append(y_tr)
        items_va.append(y_va)

        train_loader = DataLoader(
            TensorDataset(*items_tr),
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=True
        )

        val_loader = DataLoader(
            TensorDataset(*items_va),
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=True
        )

        return train_loader, val_loader

    def unpack(self, batch):
        xb = batch[0]
        yb = batch[-1]

        cb, pb = None, None

        if self.use_cgr and self.use_pse:
            cb = batch[1]
            pb = batch[2]
        elif self.use_cgr:
            cb = batch[1]
        elif self.use_pse:
            pb = batch[1]

        return xb, cb, pb, yb

    def get_model(self):
        model = TransformerHybridMultiTask(
            use_cgr=self.use_cgr,
            use_psednc=self.use_pse,
            num_layers=self.num_layers,
            nhead=self.nhead,
            psednc_dim=PSEDNC_DIM
        ).to(DEVICE)

        return model

    def evaluate(self, model, val_loader):
        model.eval()

        preds, labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                xb, cb, pb, yb = self.unpack(batch)

                xb = xb.to(DEVICE)

                if cb is not None:
                    cb = cb.to(DEVICE)

                if pb is not None:
                    pb = pb.to(DEVICE)

                logits = model(
                    xb,
                    cb,
                    pb
                )

                preds.append(
                    logits.argmax(1).cpu()
                )
                labels.append(yb)

        y_pred = torch.cat(preds).numpy()
        y_true = torch.cat(labels).numpy()

        metrics = compute_metrics(
            y_true,
            y_pred
        )

        report = classification_report(
            y_true,
            y_pred,
            digits=4
        )

        return metrics, report

    def train_fold(self, fold):
        self.log(
            f"\nFold {fold}",
            "process"
        )

        self.log(
            f"Transformer: {self.num_layers} layers | {self.nhead} heads",
            "process"
        )

        self.log(
            f"Use CGR: {self.use_cgr} | "
            f"Use PseDNC: {self.use_pse} | "
            f"Use EMA: {self.use_ema}",
            "process"
        )

        train_loader, val_loader = self.load_data(fold)

        model = self.get_model()

        ema = create_ema(model) if self.use_ema else None

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

        total_steps = MAX_EPOCHS * len(train_loader)
        warmup_steps = WARMUP_EPOCHS * len(train_loader)

        scheduler = build_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )

        crit_ce = nn.CrossEntropyLoss(
            label_smoothing=LABEL_SMOOTH
        )
        crit_ord = nn.BCEWithLogitsLoss()

        best_acc = -1.0
        p_counter = 0

        best_path = os.path.join(
            self.save_root,
            f"best_f{fold}.pth"
        )

        for ep in range(1, MAX_EPOCHS + 1):
            model.train()

            l_sum, n = 0.0, 0

            for batch in train_loader:
                xb, cb, pb, yb = self.unpack(batch)

                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                if cb is not None:
                    cb = cb.to(DEVICE)

                if pb is not None:
                    pb = pb.to(DEVICE)

                optimizer.zero_grad()

                logits, aux = model(
                    xb,
                    cb,
                    pb,
                    return_aux=True
                )

                loss = crit_ce(
                    logits,
                    yb
                )

                ord_t = get_ordinal_targets(
                    yb,
                    NUM_CLASSES
                )

                loss = loss + 0.2 * crit_ord(
                    aux["ord"],
                    ord_t
                )

                loss.backward()
                optimizer.step()
                scheduler.step()

                if ema is not None:
                    ema_update(
                        model,
                        ema
                    )

                l_sum += loss.item()
                n += 1

            eval_model = ema if ema is not None else model

            metrics, report = self.evaluate(
                eval_model,
                val_loader
            )

            acc = metrics["acc"]
            f1_m = metrics["f1_m"]

            msg = (
                f"Ep {ep} | "
                f"L {l_sum / n:.4f} | "
                f"Acc {acc:.4f} "
                f"F1 {f1_m:.4f}"
            )

            self.log(
                msg,
                "process"
            )

            if acc > best_acc:
                best_acc = acc
                p_counter = 0

                torch.save(
                    eval_model.state_dict(),
                    best_path
                )

            else:
                p_counter += 1

            if p_counter >= PATIENCE:
                self.log(
                    f"--- Early Stopping triggered at Epoch {ep} ---",
                    "process"
                )
                break

        final_model = self.get_model()
        final_model.load_state_dict(
            torch.load(
                best_path,
                map_location=DEVICE
            )
        )
        final_model.to(DEVICE)

        final_metrics, final_report = self.evaluate(
            final_model,
            val_loader
        )

        self.log(
            f"\nFinal report Fold {fold} "
            f"(Best Acc: {final_metrics['acc']:.4f}, Use EMA: {self.use_ema}):\n"
            f"{final_report}\n",
            "summary"
        )

        fold_result = {
            "fold": fold,
            "use_ema": self.use_ema,
            "num_layers": self.num_layers,
            "nhead": self.nhead,
            **final_metrics
        }

        return fold_result

    def run(self):
        if os.path.exists(self.log_process_path):
            os.remove(self.log_process_path)

        if os.path.exists(self.log_summary_path):
            os.remove(self.log_summary_path)

        if os.path.exists(self.fold_metrics_path):
            os.remove(self.fold_metrics_path)

        if os.path.exists(self.final_summary_path):
            os.remove(self.final_summary_path)

        results = []

        for i in range(1, 6):
            fold_metrics = self.train_fold(i)
            results.append(fold_metrics)

        df = pd.DataFrame(results)
        df.to_csv(
            self.fold_metrics_path,
            index=False
        )

        keys = [
            "acc",
            "f1_m",
            "f1_w",
            "prec",
            "rec",
            "mcc"
        ]

        summary = "\n================ Final 5-Fold Summary ================\n"
        summary += f"Experiment: {self.exp_name}\n"
        summary += f"Transformer: {self.num_layers} layers | {self.nhead} heads\n"
        summary += (
            f"Use CGR: {self.use_cgr} | "
            f"Use PseDNC: {self.use_pse} | "
            f"Use EMA: {self.use_ema}\n\n"
        )

        for k in keys:
            vals = df[k].values
            summary += (
                f"{k.upper():<5}: "
                f"{np.mean(vals):.4f} ± {np.std(vals):.4f}\n"
            )

        self.log(
            summary,
            "summary"
        )

        with open(self.final_summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        print(summary)

        return df


def run_all_experiments():
    print("=" * 100)
    print("[Config]")
    print(f"K_FOLDS_ROOT: {K_FOLDS_ROOT}")
    print(f"BASE_SAVE_ROOT: {BASE_SAVE_ROOT}")
    print(f"DEVICE: {DEVICE}")
    print(f"LAMBDA: {LAMBDA}")
    print(f"WEIGHT: {WEIGHT}")
    print(f"PSEDNC_DIM: {PSEDNC_DIM}")
    print(f"FORCE_RECOMPUTE_CGR: {FORCE_RECOMPUTE_CGR}")
    print(f"FORCE_RECOMPUTE_PSEDNC: {FORCE_RECOMPUTE_PSEDNC}")
    print(f"BACKUP_OLD_PSEDNC_NPY: {BACKUP_OLD_PSEDNC_NPY}")
    print(f"PHYCHEM_KEYS: {PHYCHEM_KEYS}")
    print("=" * 100)

    for model_cfg in TRANSFORMER_CONFIGS:
        cfg_tag = model_cfg["tag"]

        print(
            f"\n================ Running Transformer Config: {cfg_tag} "
            f"({model_cfg['num_layers']} layers, {model_cfg['nhead']} heads) ================"
        )

        for name in EXPERIMENTS_TO_RUN:
            cfg = EXPERIMENTS[name]

            exp_root = os.path.join(
                BASE_SAVE_ROOT,
                cfg_tag,
                name
            )

            print(
                f"\nRunning EXP: {name} | "
                f"use_cgr={cfg['use_cgr']} "
                f"use_psednc={cfg['use_psednc']} "
                f"use_ema={cfg.get('use_ema', False)}"
            )

            Trainer(
                exp_root,
                name,
                cfg,
                model_cfg
            ).run()


if __name__ == "__main__":
    seed_everything(43)
    run_all_experiments()