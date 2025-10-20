#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMTNet vs. SVM / RandomForest / GradientBoosting on Iris, Wine, Breast Cancer.

MMTNet is a small, feasible Mental-Models-inspired classifier for tabular data:
- K "worldlets" (possibilities), each a tiny positive-weight MLP over [x, -x].
- Gates over -x act as "negation footnotes"; we penalize opened negation gates.
- Aggregation uses Possibility (noisy-OR) and Necessity (geometric) with a learned mix.
- A compute-aware cost penalizes the effective number of active worldlets.

Outputs:
- 5-fold stratified CV metrics (accuracy, F1-macro, ROC-AUC) for each dataset & model
- 80/20 holdout confusion matrices + classification reports
- CSVs under ./results/
- PNG confusion matrices under ./results/
"""

import os
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier  # (optional baseline, not used by default)
from sklearn.exceptions import ConvergenceWarning

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


import random

# --------------------
# Global config
# --------------------
SEED = random.randint(1, 10000)
print(f"Using random seed: {SEED}")
N_SPLITS = 5
TEST_SIZE = 0.2
RESULTS_DIR = "results"

np.random.seed(SEED)
torch.manual_seed(SEED)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Data loading
# --------------------
@dataclass
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray
    target_names: List[str]

def load_datasets() -> List[DatasetBundle]:
    iris = load_iris()
    wine = load_wine()
    bc = load_breast_cancer()
    return [
        DatasetBundle("Iris", iris.data, iris.target, list(iris.target_names)),
        DatasetBundle("Wine", wine.data, wine.target, list(wine.target_names)),
        DatasetBundle("BreastCancer", bc.data, bc.target, list(bc.target_names)),
    ]

# --------------------
# Baseline models (sklearn)
# --------------------
def build_baselines(random_state: int = SEED) -> Dict[str, object]:
    models = {}

    # SVM (RBF) with StandardScaler in a Pipeline
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=random_state
        ))
    ])
    models["SVM-RBF"] = svm

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state
    )
    models["RandomForest"] = rf

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=1.0,
        random_state=random_state
    )
    models["GradientBoosting"] = gb

    # Optional: classic MLP baseline (commented-out to keep focus on MMTNet)
    # mlp = Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("clf", MLPClassifier(
    #         hidden_layer_sizes=(64, 64),
    #         activation="relu",
    #         solver="adam",
    #         alpha=1e-4, learning_rate_init=1e-3,
    #         max_iter=500,
    #         early_stopping=True, n_iter_no_change=20,
    #         random_state=random_state
    #     ))
    # ])
    # models["MLP-Classic"] = mlp

    return models

# --------------------
# MMTNet: Mental-Model Tabular Network
# --------------------
class MMTNet(nn.Module):
    """
    Worldlet-based tabular classifier with truth-only bias and gated negations.

    Input: x in R^d (standardized)
    Augmented features: [x, -x] in R^(2d)
    Each worldlet k:
      - gating m_k in [0,1]^(2d): selects which features to use (neg half are "footnotes").
      - positive-weight MLP: Softplus-reparam weights ensure non-negative contributions.
      - outputs per-class probs p_k (softmax over classes).

    Aggregation over K worldlets:
      Possibility (noisy-OR):   q_poss[c] = 1 - Π_k (1 - π_k * p_k[c])
      Necessity (geometric):    q_nec[c]  = Π_k p_k[c]^{π_k}
      Final probs: q = σ(α) * q_poss + (1 - σ(α)) * q_nec; renormalize to sum=1.
      π_k(x): per-example worldlet weights from a tiny gating net; encourage few active via penalty.
    """
    def __init__(self, d_in: int, n_classes: int, K: int = 3, h: int = 64):
        super().__init__()
        self.d_in = d_in
        self.n_classes = n_classes
        self.K = K
        self.h = h
        d_aug = 2 * d_in

        # Worldlet parameters
        self.m_raw = nn.Parameter(torch.randn(K, d_aug) * 0.01)  # gates -> sigmoid(m_raw) in [0,1]
        # Positive weight MLPs per worldlet: W1, b1, W2, b2 with Softplus reparam
        self.W1_raw = nn.Parameter(torch.randn(K, d_aug, h) * 0.1)
        self.b1_raw = nn.Parameter(torch.zeros(K, h))
        self.W2_raw = nn.Parameter(torch.randn(K, h, n_classes) * 0.1)
        self.b2_raw = nn.Parameter(torch.zeros(K, n_classes))

        # Worldlet selector π(x): tiny gating net
        self.pi_net = nn.Sequential(
            nn.Linear(d_in, 16),
            nn.ReLU(),
            nn.Linear(16, K)
        )

        # Mix between Possibility and Necessity
        self.mix_param = nn.Parameter(torch.tensor(0.0))

        # Nonlinearities
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            q: (N, C) final probs
            extras: dict with per-batch stats for regularization/analysis
        """
        N = x.shape[0]
        eps = 1e-8

        # Augment with -x; do NOT detach so gradients flow
        x_aug = torch.cat([x, -x], dim=1)  # (N, 2d)

        # Per-worldlet processing
        m = torch.sigmoid(self.m_raw)  # (K, 2d), gates in [0,1]
        W1 = self.softplus(self.W1_raw)      # (K, 2d, h)  >= 0
        b1 = self.softplus(self.b1_raw)      # (K, h)     >= 0
        W2 = self.softplus(self.W2_raw)      # (K, h, C)  >= 0
        b2 = self.softplus(self.b2_raw)      # (K, C)     >= 0

        # Compute per-worldlet class probs p_k(x)
        # We'll loop over K to keep it readable & memory-friendly for tiny datasets
        pks = []
        for k in range(self.K):
            xk = x_aug * m[k].unsqueeze(0)                   # (N, 2d)
            h1 = F.relu(xk @ W1[k] + b1[k].unsqueeze(0))     # (N, h)
            logits_k = h1 @ W2[k] + b2[k].unsqueeze(0)       # (N, C)
            pk = F.softmax(logits_k, dim=1)                  # (N, C)
            pks.append(pk)
        P = torch.stack(pks, dim=1)  # (N, K, C)

        # π(x): per-example worldlet weights
        pi_logits = self.pi_net(x)            # (N, K)
        pi = F.softmax(pi_logits, dim=1)      # (N, K)

        # Possibility (noisy-OR): q_poss[c] = 1 - Π_k (1 - π_k * p_k[c])
        one_minus = 1.0 - (pi.unsqueeze(-1) * P)  # (N, K, C)
        q_poss = 1.0 - torch.exp(torch.clamp(torch.log(one_minus + eps).sum(dim=1), min=-50, max=50))  # (N, C)

        # Necessity (geometric): q_nec[c] = Π_k p_k[c]^{π_k}
        log_p = torch.log(P + eps)                          # (N, K, C)
        q_nec = torch.exp((pi.unsqueeze(-1) * log_p).sum(dim=1))  # (N, C)

        # Mix & normalize
        lam = torch.sigmoid(self.mix_param)
        q = lam * q_poss + (1 - lam) * q_nec                # (N, C)
        q = q / (q.sum(dim=1, keepdim=True) + eps)

        # Regularization helpers
        # Effective number of active worldlets: Peff = 1 / sum(pi^2) (per sample); average over batch
        Peff = 1.0 / (pi.pow(2).sum(dim=1) + eps)           # (N,)
        Peff_mean = Peff.mean()

        # Negation gates are second half of m (corresponding to -x)
        d_aug = x_aug.shape[1]
        d_in = d_aug // 2
        m_neg = m[:, d_in:]                                  # (K, d)
        neg_gate_sum = m_neg.sum() / (self.K * d_in)         # average negation openness

        # Overall gate sparsity (encourage fewer active features)
        gate_sparsity = m.sum() / (self.K * d_aug)

        extras = {
            "pi": pi,
            "Peff_mean": Peff_mean,
            "neg_gate_mean": neg_gate_sum,
            "gate_sparsity_mean": gate_sparsity
        }
        return q, extras

# --------------------
# Training utilities for MMTNet
# --------------------
@dataclass
class MMTNetConfig:
    K: int = 3
    h: int = 64
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 64
    patience: int = 20
    lambda_K: float = 0.01      # worldlet compute penalty
    lambda_eta: float = 0.05    # negation gate penalty
    lambda_gate: float = 0.001  # overall gate sparsity
    weight_decay: float = 1e-4

def train_mmtnet(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    cfg: MMTNetConfig
) -> Tuple[MMTNet, Dict[str, float]]:
    """
    Train MMTNet with early stopping on validation set.
    Returns best model (on val) and dict of final/aux metrics.
    """
    N_tr, d_in = X_tr.shape
    model = MMTNet(d_in=d_in, n_classes=n_classes, K=cfg.K, h=cfg.h).to(device)

    # Data
    Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
    ytr_t = torch.tensor(y_tr, dtype=torch.long)
    Xva_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    yva_t = torch.tensor(y_val, dtype=torch.long).to(device)

    ds = TensorDataset(Xtr_t, ytr_t)
    dl = DataLoader(ds, batch_size=min(cfg.batch_size, N_tr), shuffle=True)

    # Optimizer & scheduler
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            q, extras = model(xb)
            # Cross-entropy
            loss_ce = F.nll_loss(torch.log(q + 1e-8), yb)
            # Regularizers
            loss_K = cfg.lambda_K * extras["Peff_mean"]
            loss_eta = cfg.lambda_eta * extras["neg_gate_mean"]
            loss_gate = cfg.lambda_gate * extras["gate_sparsity_mean"]
            loss = loss_ce + loss_K + loss_eta + loss_gate
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu().item()) * xb.shape[0]
        epoch_loss /= max(1, len(ds))

        # Validation
        model.eval()
        with torch.no_grad():
            qv, extras_v = model(Xva_t)
            val_loss_ce = F.nll_loss(torch.log(qv + 1e-8), yva_t)
            val_loss = val_loss_ce \
                       + cfg.lambda_K * extras_v["Peff_mean"] \
                       + cfg.lambda_eta * extras_v["neg_gate_mean"] \
                       + cfg.lambda_gate * extras_v["gate_sparsity_mean"]
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val - 1e-6:
            best_val = float(val_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_loss": best_val}

def mmtnet_predict_proba(model: MMTNet, X: np.ndarray) -> np.ndarray:
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        q, _ = model(Xt)
    return q.cpu().numpy()

# --------------------
# Metrics helpers
# --------------------
def _get_scores_for_auc_sklearn(estimator, X_test: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_test)
    if hasattr(estimator, "decision_function"):
        dec = estimator.decision_function(X_test)
        if dec.ndim == 1:
            dec = dec.reshape(-1, 1)
        return dec
    raise ValueError("Estimator lacks predict_proba and decision_function.")

def compute_metrics_from_proba(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    y_pred = np.argmax(y_proba, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    n_classes = len(np.unique(y_true))
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan
    return acc, f1m, auc, y_pred

# --------------------
# Cross-validation & holdout
# --------------------
def crossval_eval_sklearn(model, X: np.ndarray, y: np.ndarray, n_splits: int = N_SPLITS) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    accs, f1s, aucs = [], [], []
    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        # Fit
        model_fit = model
        model_fit.fit(X_tr, y_tr)
        scores = _get_scores_for_auc_sklearn(model_fit, X_te)
        acc, f1m, auc, _ = compute_metrics_from_proba(y_te, scores)
        accs.append(acc); f1s.append(f1m); aucs.append(auc)
    return {
        "acc_mean": float(np.nanmean(accs)),
        "acc_std": float(np.nanstd(accs)),
        "f1_macro_mean": float(np.nanmean(f1s)),
        "f1_macro_std": float(np.nanstd(f1s)),
        "roc_auc_mean": float(np.nanmean(aucs)),
        "roc_auc_std": float(np.nanstd(aucs)),
    }

def crossval_eval_mmtnet(X: np.ndarray, y: np.ndarray, cfg: MMTNetConfig, n_splits: int = N_SPLITS) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    accs, f1s, aucs = [], [], []
    for tr_idx, te_idx in skf.split(X, y):
        X_tr_raw, X_te_raw = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Scale (fit on train only)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        # Train with small val split from X_tr
        X_tr2, X_val, y_tr2, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED
        )
        model, _ = train_mmtnet(
            X_tr2, y_tr2, X_val, y_val,
            n_classes=len(np.unique(y)), cfg=cfg
        )
        # Evaluate
        proba = mmtnet_predict_proba(model, X_te)
        acc, f1m, auc, _ = compute_metrics_from_proba(y_te, proba)
        accs.append(acc); f1s.append(f1m); aucs.append(auc)

    return {
        "acc_mean": float(np.nanmean(accs)),
        "acc_std": float(np.nanstd(accs)),
        "f1_macro_mean": float(np.nanmean(f1s)),
        "f1_macro_std": float(np.nanstd(f1s)),
        "roc_auc_mean": float(np.nanmean(aucs)),
        "roc_auc_std": float(np.nanstd(aucs)),
    }

def plot_confusion(cm: np.ndarray, class_names: List[str], title: str, save_path: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def holdout_eval_sklearn(models: Dict[str, object], ds: DatasetBundle) -> pd.DataFrame:
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=TEST_SIZE, stratify=ds.y, random_state=SEED
    )
    n_classes = len(np.unique(ds.y))
    rows = []
    for name, est in models.items():
        start = time.time()
        # Fit
        est.fit(X_tr_raw, y_tr)
        proba = _get_scores_for_auc_sklearn(est, X_te_raw)
        acc, f1m, auc, y_pred = compute_metrics_from_proba(y_te, proba)
        elapsed = time.time() - start

        # Confusion
        cm = confusion_matrix(y_te, y_pred, normalize="true")
        fn = os.path.join(RESULTS_DIR, f"confusion_{ds.name}_{name}.png")
        plot_confusion(cm, ds.target_names, f"{ds.name} – {name}", fn)

        report = classification_report(y_te, y_pred, target_names=ds.target_names, digits=3)
        print("=" * 80)
        print(f"[Holdout Report] Dataset: {ds.name} | Model: {name}")
        print(report)
        print(f"Confusion matrix saved to: {fn}")

        rows.append({
            "dataset": ds.name, "model": name,
            "holdout_accuracy": acc, "holdout_f1_macro": f1m,
            "holdout_roc_auc": auc, "train_time_sec": elapsed
        })
    return pd.DataFrame(rows)

def holdout_eval_mmtnet(ds: DatasetBundle, cfg: MMTNetConfig) -> pd.DataFrame:
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=TEST_SIZE, stratify=ds.y, random_state=SEED
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    start = time.time()
    # Split a small validation from train for early stopping
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED
    )
    model, _ = train_mmtnet(
        X_tr2, y_tr2, X_val, y_val,
        n_classes=len(np.unique(ds.y)), cfg=cfg
    )
    proba = mmtnet_predict_proba(model, X_te)
    acc, f1m, auc, y_pred = compute_metrics_from_proba(y_te, proba)
    elapsed = time.time() - start

    cm = confusion_matrix(y_te, y_pred, normalize="true")
    fn = os.path.join(RESULTS_DIR, f"confusion_{ds.name}_MMTNet.png")
    plot_confusion(cm, ds.target_names, f"{ds.name} – MMTNet", fn)

    report = classification_report(y_te, y_pred, target_names=ds.target_names, digits=3)
    print("=" * 80)
    print(f"[Holdout Report] Dataset: {ds.name} | Model: MMTNet")
    print(report)
    print(f"Confusion matrix saved to: {fn}")

    return pd.DataFrame([{
        "dataset": ds.name, "model": "MMTNet",
        "holdout_accuracy": acc, "holdout_f1_macro": f1m,
        "holdout_roc_auc": auc, "train_time_sec": elapsed
    }])

# --------------------
# Main benchmark
# --------------------
def benchmark():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    datasets = load_datasets()
    baselines = build_baselines(SEED)

    # MMTNet config (small & fast)
    cfg = MMTNetConfig(
        K=3, h=64, lr=1e-3, epochs=200, batch_size=64, patience=20,
        lambda_K=0.01, lambda_eta=0.05, lambda_gate=0.001, weight_decay=1e-4
    )

    cv_rows = []
    hold_rows = []

    for ds in datasets:
        print("\n" + "#" * 80)
        print(f"Dataset: {ds.name} | X shape: {ds.X.shape} | y classes: {len(np.unique(ds.y))}")
        print("#" * 80)

        # --- Cross-validation ---
        # Baselines
        for name, est in baselines.items():
            print(f"\n[Cross-Validation] {ds.name} – {name}")
            stats = crossval_eval_sklearn(est, ds.X, ds.y, n_splits=N_SPLITS)
            cv_rows.append({"dataset": ds.name, "model": name, **stats})
            print(
                f"  Acc {stats['acc_mean']:.4f} ± {stats['acc_std']:.4f} | "
                f"F1-macro {stats['f1_macro_mean']:.4f} ± {stats['f1_macro_std']:.4f} | "
                f"AUC {stats['roc_auc_mean']:.4f} ± {stats['roc_auc_std']:.4f}"
            )

        # MMTNet
        print(f"\n[Cross-Validation] {ds.name} – MMTNet (K={cfg.K})")
        mmtnet_stats = crossval_eval_mmtnet(ds.X, ds.y, cfg, n_splits=N_SPLITS)
        cv_rows.append({"dataset": ds.name, "model": "MMTNet", **mmtnet_stats})
        print(
            f"  Acc {mmtnet_stats['acc_mean']:.4f} ± {mmtnet_stats['acc_std']:.4f} | "
            f"F1-macro {mmtnet_stats['f1_macro_mean']:.4f} ± {mmtnet_stats['f1_macro_std']:.4f} | "
            f"AUC {mmtnet_stats['roc_auc_mean']:.4f} ± {mmtnet_stats['roc_auc_std']:.4f}"
        )

        # --- Holdout (80/20) ---
        print(f"\n[Holdout Evaluation] {ds.name} (test_size={TEST_SIZE})")
        df_b = holdout_eval_sklearn(baselines, ds)
        df_m = holdout_eval_mmtnet(ds, cfg)
        hold_rows.append(pd.concat([df_b, df_m], axis=0, ignore_index=True))

    # Summaries
    cv_df = pd.DataFrame(cv_rows).sort_values(["dataset", "model"])
    hold_df = pd.concat(hold_rows, axis=0, ignore_index=True).sort_values(["dataset", "model"])

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY (5-fold)")
    print("=" * 80)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 160):
        print(cv_df.to_string(index=False,
                              formatters={
                                  "acc_mean": "{:.4f}".format,
                                  "acc_std": "{:.4f}".format,
                                  "f1_macro_mean": "{:.4f}".format,
                                  "f1_macro_std": "{:.4f}".format,
                                  "roc_auc_mean": (lambda v: "nan" if pd.isna(v) else f"{v:.4f}"),
                                  "roc_auc_std": (lambda v: "nan" if pd.isna(v) else f"{v:.4f}")
                              }))

    print("\n" + "=" * 80)
    print("HOLDOUT (80/20) SUMMARY")
    print("=" * 80)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 160):
        print(hold_df.to_string(index=False,
                                formatters={
                                    "holdout_accuracy": "{:.4f}".format,
                                    "holdout_f1_macro": "{:.4f}".format,
                                    "holdout_roc_auc": (lambda v: "nan" if pd.isna(v) else f"{v:.4f}"),
                                    "train_time_sec": "{:.3f}".format
                                }))
    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cv_path = os.path.join(RESULTS_DIR, "cv_summary.csv")
    hold_path = os.path.join(RESULTS_DIR, "holdout_summary.csv")
    cv_df.to_csv(cv_path, index=False)
    hold_df.to_csv(hold_path, index=False)
    print(f"\nSaved CSVs:\n - {cv_path}\n - {hold_path}")
    print(f"Confusion matrices saved under: ./{RESULTS_DIR}/")

if __name__ == "__main__":
    benchmark()
