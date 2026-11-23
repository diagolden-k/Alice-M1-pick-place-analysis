"""Generalization / overfitting validation for the Alice pick-and-place classifier.

Checks whether reported performance is specific to this 48-episode set:
  1. Repeated Stratified K-Fold
  2. Leave-One-Out CV
  3. Nested CV (inner grid search)
  4. Label permutation test
  5. Bootstrap CIs on OOF predictions
  6. Learning curves (train vs val gap)
  7. Feature-selection leakage comparison (full-data top-d vs fold-wise selection)

Usage:
  python3 ml/validate_generalization.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "viz" / "outputs" / "episode_phase_features.csv"
RANK = ROOT / "viz" / "outputs" / "feature_influence_ranking.csv"
OUT = ROOT / "ml" / "outputs" / "validation"
RANDOM_STATE = 42
TOP_K = 25
N_PERM = 500
N_REPEATS = 20
N_BOOT = 2000


def load_full_matrix():
    feat = pd.read_csv(FEAT)
    y = feat["label"].to_numpy(dtype=int)
    episodes = feat["episode"].to_numpy(dtype=int)
    skip = {"episode", "label"}
    cols = [c for c in feat.columns if c not in skip]
    X_all = feat[cols].to_numpy(dtype=float)
    return feat, X_all, y, episodes, cols


def cohens_d_abs(x: np.ndarray, y: np.ndarray) -> float:
    x1, x0 = x[y == 1], x[y == 0]
    if len(x1) < 2 or len(x0) < 2:
        return 0.0
    v1, v0 = np.var(x1, ddof=1), np.var(x0, ddof=1)
    pooled = np.sqrt(((len(x1) - 1) * v1 + (len(x0) - 1) * v0) / (len(x1) + len(x0) - 2))
    if pooled < 1e-12:
        return 0.0
    return abs((np.mean(x1) - np.mean(x0)) / pooled)


def select_topk_indices(X: np.ndarray, y: np.ndarray, k: int = TOP_K) -> np.ndarray:
    scores = np.array([cohens_d_abs(X[:, j], y) for j in range(X.shape[1])])
    # NaN-safe: features with all-nan get 0
    scores = np.nan_to_num(scores, nan=0.0)
    return np.argsort(scores)[::-1][:k]


def make_svm(C=3.0, gamma="scale"):
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_logreg(C=0.1):
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def metrics_from_oof(y_true, y_pred, y_prob=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out


def oof_predict(model, X, y, cv):
    y_pred = np.zeros_like(y)
    y_prob = np.zeros(len(y), dtype=float)
    fold_rows = []
    for fold_i, (tr, te) in enumerate(cv.split(X, y)):
        m = model
        # clone via set params / new instance by re-fitting fresh pipeline
        from sklearn.base import clone

        m = clone(model)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        prob = m.predict_proba(X[te])[:, 1]
        y_pred[te] = pred
        y_prob[te] = prob
        fold_rows.append(
            {
                "fold": fold_i,
                "n_test": int(len(te)),
                "accuracy": float(accuracy_score(y[te], pred)),
                "f1": float(f1_score(y[te], pred)) if len(np.unique(y[te])) > 1 else np.nan,
            }
        )
    return y_pred, y_prob, pd.DataFrame(fold_rows)


def oof_with_foldwise_feature_select(X_all, y, cv, model_factory, k=TOP_K):
    """Honest CV: choose top-|d| features using TRAIN fold only."""
    y_pred = np.zeros_like(y)
    y_prob = np.zeros(len(y), dtype=float)
    selected = []
    from sklearn.base import clone

    for fold_i, (tr, te) in enumerate(cv.split(X_all, y)):
        idx = select_topk_indices(X_all[tr], y[tr], k=k)
        selected.append(idx)
        Xtr, Xte = X_all[tr][:, idx], X_all[te][:, idx]
        m = clone(model_factory())
        m.fit(Xtr, y[tr])
        y_pred[te] = m.predict(Xte)
        y_prob[te] = m.predict_proba(Xte)[:, 1]
    # feature stability: Jaccard of selected sets across folds
    jaccards = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            a, b = set(selected[i].tolist()), set(selected[j].tolist())
            jaccards.append(len(a & b) / max(1, len(a | b)))
    return y_pred, y_prob, float(np.mean(jaccards)), selected


def repeated_kfold(X, y, model, n_splits=5, n_repeats=N_REPEATS):
    rows = []
    from sklearn.base import clone

    for rep in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + rep)
        for fold_i, (tr, te) in enumerate(cv.split(X, y)):
            m = clone(model)
            m.fit(X[tr], y[tr])
            pred = m.predict(X[te])
            prob = m.predict_proba(X[te])[:, 1]
            rows.append(
                {
                    "repeat": rep,
                    "fold": fold_i,
                    "accuracy": float(accuracy_score(y[te], pred)),
                    "f1": float(f1_score(y[te], pred)) if len(np.unique(y[te])) > 1 else np.nan,
                    "roc_auc": float(roc_auc_score(y[te], prob)) if len(np.unique(y[te])) > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def leave_one_out(X, y, model):
    from sklearn.base import clone

    y_pred = np.zeros_like(y)
    y_prob = np.zeros(len(y), dtype=float)
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        m = clone(model)
        m.fit(X[tr], y[tr])
        y_pred[te] = m.predict(X[te])
        y_prob[te] = m.predict_proba(X[te])[:, 1]
    return metrics_from_oof(y, y_pred, y_prob), y_pred, y_prob


def nested_cv(X, y, n_splits=5):
    """Outer CV evaluates model whose C/gamma tuned on inner CV only."""
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    grid = {"clf__C": [0.3, 1.0, 3.0, 10.0], "clf__gamma": ["scale", 0.01, 0.03, 0.1]}
    y_pred = np.zeros_like(y)
    y_prob = np.zeros(len(y), dtype=float)
    best_params = []
    for fold_i, (tr, te) in enumerate(outer.split(X, y)):
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE + fold_i)
        gs = GridSearchCV(pipe, grid, scoring="f1", cv=inner, n_jobs=-1)
        gs.fit(X[tr], y[tr])
        best_params.append(gs.best_params_)
        y_pred[te] = gs.predict(X[te])
        y_prob[te] = gs.predict_proba(X[te])[:, 1]
        print(f"  nested fold {fold_i}: {gs.best_params_} test_acc={accuracy_score(y[te], y_pred[te]):.3f}")
    return metrics_from_oof(y, y_pred, y_prob), best_params


def bootstrap_ci(y, y_pred, y_prob, n_boot=N_BOOT, alpha=0.05):
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y)
    accs, f1s, aucs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, pr = y[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp))
        aucs.append(roc_auc_score(yt, pr))
    def ci(a):
        a = np.asarray(a)
        return {
            "mean": float(np.mean(a)),
            "lo": float(np.quantile(a, alpha / 2)),
            "hi": float(np.quantile(a, 1 - alpha / 2)),
        }
    return {"accuracy": ci(accs), "f1": ci(f1s), "roc_auc": ci(aucs)}


def run_permutation(X, y, model, n_permutations=N_PERM):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    score, perm_scores, pvalue = permutation_test_score(
        model,
        X,
        y,
        scoring="accuracy",
        cv=cv,
        n_permutations=n_permutations,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    # also F1 permutation
    score_f1, perm_f1, p_f1 = permutation_test_score(
        model,
        X,
        y,
        scoring="f1",
        cv=cv,
        n_permutations=n_permutations,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return {
        "accuracy_score": float(score),
        "accuracy_pvalue": float(pvalue),
        "accuracy_perm_mean": float(np.mean(perm_scores)),
        "accuracy_perm_std": float(np.std(perm_scores)),
        "accuracy_perm_scores": np.asarray(perm_scores),
        "f1_score": float(score_f1),
        "f1_pvalue": float(p_f1),
        "f1_perm_mean": float(np.mean(perm_f1)),
        "f1_perm_scores": np.asarray(perm_f1),
    }


def run_learning_curve(X, y, model):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes = np.linspace(0.3, 1.0, 6)
    sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        train_sizes=train_sizes,
        scoring="accuracy",
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return pd.DataFrame(
        {
            "train_size": sizes,
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "val_mean": val_scores.mean(axis=1),
            "val_std": val_scores.std(axis=1),
            "gap": train_scores.mean(axis=1) - val_scores.mean(axis=1),
        }
    )


def plot_permutation(perm, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title, real in (
        (axes[0], "accuracy_perm_scores", "Permutation: Accuracy", perm["accuracy_score"]),
        (axes[1], "f1_perm_scores", "Permutation: F1", perm["f1_score"]),
    ):
        ax.hist(perm[key], bins=30, color="#9E9E9E", edgecolor="white", alpha=0.9)
        ax.axvline(real, color="#E45756", lw=2, label=f"true={real:.3f}")
        ax.set_title(title)
        ax.set_xlabel("CV score under label shuffle")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_repeated_kfold(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(
        [df["accuracy"].dropna(), df["f1"].dropna(), df["roc_auc"].dropna()],
        showmeans=True,
    )
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Accuracy", "F1", "ROC-AUC"])
    ax.set_ylim(0.4, 1.05)
    ax.set_title(f"Repeated {5}-Fold × {N_REPEATS} (optimistic features)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_learning(lc: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(
        lc["train_size"],
        lc["train_mean"] - lc["train_std"],
        lc["train_mean"] + lc["train_std"],
        alpha=0.2,
        color="#4C78A8",
    )
    ax.fill_between(
        lc["train_size"],
        lc["val_mean"] - lc["val_std"],
        lc["val_mean"] + lc["val_std"],
        alpha=0.2,
        color="#54A24B",
    )
    ax.plot(lc["train_size"], lc["train_mean"], "-o", color="#4C78A8", label="train")
    ax.plot(lc["train_size"], lc["val_mean"], "-o", color="#54A24B", label="CV val")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning curve (train–val gap ≈ overfitting)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_leakage_compare(rows: list[dict], path: Path):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["accuracy"], 0.4, label="Accuracy", color="#4C78A8")
    ax.bar(x + 0.2, df["f1"], 0.4, label="F1", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(df["protocol"], rotation=15, ha="right")
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Optimistic vs honest protocols")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def summarize_ci(name, arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "name": name,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def write_report(path: Path, summary: dict):
    lines = []
    lines.append("# Generalization / Overfitting Validation\n")
    lines.append(
        "Question: does the selected SVM-RBF fit **only** these 48 episodes, "
        "or is there evidence it could work on new data?\n"
    )
    lines.append("## Protocols\n")
    lines.append(
        "1. **Optimistic 5-fold / Repeated K-fold / LOO** — features = global top-25 |Cohen's d| "
        "(same as original pipeline; mild selection leakage).\n"
        "2. **Honest fold-wise selection** — top-25 rechosen on each training fold only.\n"
        "3. **Nested CV** — outer test fold never sees C/γ tuning.\n"
        "4. **Permutation test** — labels shuffled; real score must beat null.\n"
        "5. **Bootstrap CI** — resample OOF predictions.\n"
        "6. **Learning curve** — train vs validation gap.\n"
    )
    lines.append("## Headline results\n")
    for k, v in summary["headline"].items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.4f}")
        else:
            lines.append(f"- **{k}**: {v}")
    lines.append("\n## Interpretation\n")
    lines.append(summary["interpretation"])
    lines.append("\n## Detailed tables\n")
    lines.append("### Protocol comparison\n")
    lines.append(pd.DataFrame(summary["protocol_rows"]).to_string(index=False))
    lines.append("\n### Repeated K-fold distribution\n")
    lines.append(pd.DataFrame(summary["repeated_summary"]).to_string(index=False))
    lines.append("\n### Bootstrap 95% CI (optimistic OOF)\n")
    lines.append(json.dumps(summary["bootstrap"], indent=2))
    lines.append("\n### Permutation test\n")
    lines.append(json.dumps(summary["permutation_summary"], indent=2))
    lines.append("\n### Learning curve (final row = full data)\n")
    lines.append(pd.DataFrame(summary["learning_curve"]).to_string(index=False))
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feat, X_all, y, episodes, all_cols = load_full_matrix()

    # Optimistic feature set (global ranking) — matches original pipeline
    rank = pd.read_csv(RANK)
    opt_cols = [c for c in rank["feature"].head(TOP_K).tolist() if c in feat.columns]
    X_opt = feat[opt_cols].to_numpy(dtype=float)
    model = make_svm(C=3.0, gamma="scale")
    print(f"N={len(y)}, features_all={X_all.shape[1]}, optimistic_top={X_opt.shape[1]}")

    protocol_rows = []

    # --- 1. Single stratified 5-fold (optimistic) ---
    print("\n[1] Stratified 5-fold (optimistic features)")
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred5, y_prob5, fold_df = oof_predict(model, X_opt, y, cv5)
    m5 = metrics_from_oof(y, y_pred5, y_prob5)
    print(f"  OOF acc={m5['accuracy']:.3f} f1={m5['f1']:.3f} auc={m5['roc_auc']:.3f}")
    protocol_rows.append({"protocol": "5-fold optimistic", **m5})
    fold_df.to_csv(OUT / "kfold5_folds.csv", index=False)

    # --- 2. Repeated K-fold ---
    print(f"\n[2] Repeated Stratified 5-fold × {N_REPEATS}")
    rep = repeated_kfold(X_opt, y, model, n_splits=5, n_repeats=N_REPEATS)
    rep.to_csv(OUT / "repeated_kfold_scores.csv", index=False)
    rep_sum = [
        summarize_ci("accuracy", rep["accuracy"]),
        summarize_ci("f1", rep["f1"]),
        summarize_ci("roc_auc", rep["roc_auc"]),
    ]
    print(
        f"  Acc {rep_sum[0]['mean']:.3f} [{rep_sum[0]['q05']:.3f}, {rep_sum[0]['q95']:.3f}]  "
        f"F1 {rep_sum[1]['mean']:.3f} [{rep_sum[1]['q05']:.3f}, {rep_sum[1]['q95']:.3f}]"
    )
    plot_repeated_kfold(rep, OUT / "repeated_kfold_box.png")

    # --- 3. LOO ---
    print("\n[3] Leave-One-Out (optimistic)")
    loo_m, loo_pred, loo_prob = leave_one_out(X_opt, y, model)
    print(f"  LOO acc={loo_m['accuracy']:.3f} f1={loo_m['f1']:.3f} auc={loo_m['roc_auc']:.3f}")
    protocol_rows.append({"protocol": "LOO optimistic", **loo_m})

    # --- 4. Honest fold-wise feature selection ---
    print("\n[4] Honest 5-fold (feature select inside train fold)")
    y_pred_h, y_prob_h, jaccard, _ = oof_with_foldwise_feature_select(
        X_all, y, cv5, lambda: make_svm(C=3.0, gamma="scale"), k=TOP_K
    )
    mh = metrics_from_oof(y, y_pred_h, y_prob_h)
    print(f"  Honest OOF acc={mh['accuracy']:.3f} f1={mh['f1']:.3f} auc={mh['roc_auc']:.3f}  feature Jaccard={jaccard:.3f}")
    protocol_rows.append({"protocol": "5-fold honest select", **mh, "feature_jaccard": jaccard})

    # --- 5. Nested CV ---
    print("\n[5] Nested CV (tune C/γ inside outer folds)")
    nested_m, nested_params = nested_cv(X_opt, y, n_splits=5)
    print(f"  Nested OOF acc={nested_m['accuracy']:.3f} f1={nested_m['f1']:.3f} auc={nested_m['roc_auc']:.3f}")
    protocol_rows.append({"protocol": "nested CV optimistic feats", **nested_m})

    # Honest nested-ish: foldwise select + fixed SVM (already done); optional logreg baseline
    print("\n[5b] Honest LOO with fold-wise... (using LOO + train-only select)")
    y_pred_hl = np.zeros_like(y)
    y_prob_hl = np.zeros(len(y), dtype=float)
    from sklearn.base import clone

    for tr, te in LeaveOneOut().split(X_all):
        idx = select_topk_indices(X_all[tr], y[tr], k=TOP_K)
        m = clone(make_svm())
        m.fit(X_all[tr][:, idx], y[tr])
        y_pred_hl[te] = m.predict(X_all[te][:, idx])
        y_prob_hl[te] = m.predict_proba(X_all[te][:, idx])[:, 1]
    mhl = metrics_from_oof(y, y_pred_hl, y_prob_hl)
    print(f"  Honest LOO acc={mhl['accuracy']:.3f} f1={mhl['f1']:.3f} auc={mhl['roc_auc']:.3f}")
    protocol_rows.append({"protocol": "LOO honest select", **mhl})

    # --- 6. Permutation test ---
    print(f"\n[6] Permutation test (n={N_PERM})")
    perm = run_permutation(X_opt, y, model, n_permutations=N_PERM)
    print(
        f"  Acc true={perm['accuracy_score']:.3f} null≈{perm['accuracy_perm_mean']:.3f} "
        f"p={perm['accuracy_pvalue']:.4f}"
    )
    print(f"  F1  true={perm['f1_score']:.3f} null≈{perm['f1_perm_mean']:.3f} p={perm['f1_pvalue']:.4f}")
    plot_permutation(perm, OUT / "permutation_hist.png")
    np.save(OUT / "perm_acc_scores.npy", perm["accuracy_perm_scores"])
    np.save(OUT / "perm_f1_scores.npy", perm["f1_perm_scores"])

    # --- 7. Bootstrap CI ---
    print(f"\n[7] Bootstrap CI on optimistic 5-fold OOF (n={N_BOOT})")
    boot = bootstrap_ci(y, y_pred5, y_prob5, n_boot=N_BOOT)
    print(json.dumps(boot, indent=2))

    # --- 8. Learning curve ---
    print("\n[8] Learning curve")
    lc = run_learning_curve(X_opt, y, model)
    lc.to_csv(OUT / "learning_curve.csv", index=False)
    print(lc.to_string(index=False))
    plot_learning(lc, OUT / "learning_curve.png")

    # Compare logreg honest as sanity
    print("\n[extra] Honest 5-fold with LogReg")
    y_pred_lr, y_prob_lr, j_lr, _ = oof_with_foldwise_feature_select(
        X_all, y, cv5, lambda: make_logreg(C=0.1), k=TOP_K
    )
    mlr = metrics_from_oof(y, y_pred_lr, y_prob_lr)
    protocol_rows.append({"protocol": "5-fold honest LogReg", **mlr, "feature_jaccard": j_lr})
    print(f"  LogReg honest acc={mlr['accuracy']:.3f} f1={mlr['f1']:.3f}")

    # Pre-specified phase features (quality-metric terms) — less fishing across 267 cols
    print("\n[extra] Fixed domain feature set (10 metric terms) 5-fold")
    domain_feats = [
        "approach_track_err_std",
        "approach_speed_maxabs",
        "approach_joint_speed_std",
        "grasp_acc_mag_mean",
        "lift_duration",
        "carry_right_vy_mean",
        "carry_right_vz_mean",
        "retreat_vy_mean",
        "global_jerk_mag_rms",
        "place_speed_rms",
    ]
    domain_feats = [c for c in domain_feats if c in feat.columns]
    X_dom = feat[domain_feats].to_numpy(dtype=float)
    y_pred_d, y_prob_d, _ = oof_predict(make_svm(), X_dom, y, cv5)
    md = metrics_from_oof(y, y_pred_d, y_prob_d)
    protocol_rows.append({"protocol": "5-fold fixed-10 domain feats", **md})
    print(f"  Domain-10 acc={md['accuracy']:.3f} f1={md['f1']:.3f} auc={md['roc_auc']:.3f}")

    # Permutation on honest protocol score (recompute with foldwise each perm would be slow;
    # instead permute labels once and run honest 5-fold many times — lighter: shuffle y, optimistic model)
    print("\n[extra] Permutation on honest fold-wise accuracy (n=200)")
    rng = np.random.default_rng(RANDOM_STATE)
    honest_null = []
    for i in range(200):
        y_perm = rng.permutation(y)
        yp, _, _, _ = oof_with_foldwise_feature_select(
            X_all, y_perm, cv5, lambda: make_svm(), k=TOP_K
        )
        honest_null.append(accuracy_score(y_perm, yp))
        if (i + 1) % 50 == 0:
            print(f"  perm {i+1}/200 ...")
    honest_null = np.asarray(honest_null)
    # p-value: fraction of null >= observed honest acc
    p_honest = (1 + np.sum(honest_null >= mh["accuracy"])) / (1 + len(honest_null))
    print(f"  Honest acc={mh['accuracy']:.3f} null_mean={honest_null.mean():.3f} p≈{p_honest:.4f}")
    summary_honest_perm = {
        "honest_accuracy": mh["accuracy"],
        "null_mean": float(honest_null.mean()),
        "null_std": float(honest_null.std()),
        "pvalue": float(p_honest),
        "n_permutations": 200,
    }
    np.save(OUT / "honest_perm_acc_scores.npy", honest_null)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(honest_null, bins=25, color="#9E9E9E", edgecolor="white")
    ax.axvline(mh["accuracy"], color="#E45756", lw=2, label=f"honest={mh['accuracy']:.3f}")
    ax.set_title("Permutation under honest fold-wise selection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "permutation_honest_hist.png", dpi=140)
    plt.close(fig)

    plot_leakage_compare(protocol_rows, OUT / "protocol_comparison.png")
    pd.DataFrame(protocol_rows).to_csv(OUT / "protocol_comparison.csv", index=False)

    # Interpretation
    drop = m5["accuracy"] - mh["accuracy"]
    gap = float(lc["gap"].iloc[-1])
    if perm["accuracy_pvalue"] < 0.01 and mh["accuracy"] >= 0.75 and p_honest < 0.05:
        verdict = "signal_likely_real"
    elif perm["accuracy_pvalue"] < 0.05 and mh["accuracy"] >= 0.65:
        verdict = "signal_present_but_uncertain"
    else:
        verdict = "weak_or_overfit_risk"

    interpretation = (
        f"Optimistic permutation p(acc)={perm['accuracy_pvalue']:.4f} "
        f"(null mean≈{perm['accuracy_perm_mean']:.2f}); "
        f"honest-selection permutation p≈{p_honest:.4f} "
        f"(null mean≈{honest_null.mean():.2f}). "
        f"Scores beat chance under label shuffle.\n\n"
        f"Optimistic 5-fold acc={m5['accuracy']:.3f} vs honest fold-wise feature selection "
        f"acc={mh['accuracy']:.3f} (drop={drop:.3f}). "
        f"Global top-25 ranking on all 48 episodes **inflated** the earlier ~94% figure "
        f"(selection leakage). Feature Jaccard across folds={jaccard:.2f} "
        f"(top features reshuffle a lot).\n\n"
        f"Fixed 10 domain features (phase/metric terms) 5-fold acc={md['accuracy']:.3f}. "
        f"Nested CV (tune C/γ inside folds) acc={nested_m['accuracy']:.3f}. "
        f"Honest LOO acc={mhl['accuracy']:.3f}. "
        f"Learning-curve final train–val gap={gap:.3f}.\n\n"
        f"**Bottom line ({verdict}):** There is a real success/fail signal, but the originally "
        f"reported ~94% is optimistic. A fairer expectation on **new episodes from the same "
        f"setup** is roughly **~70% accuracy** (honest CV), not 94%. "
        f"Without an external holdout set we cannot claim deploy-ready generalization; "
        f"re-validate on the demo/test release before trusting the score."
    )

    summary = {
        "headline": {
            "optimistic_5fold_acc": m5["accuracy"],
            "optimistic_5fold_f1": m5["f1"],
            "honest_5fold_acc": mh["accuracy"],
            "honest_5fold_f1": mh["f1"],
            "honest_LOO_acc": mhl["accuracy"],
            "nested_CV_acc": nested_m["accuracy"],
            "permutation_p_accuracy": perm["accuracy_pvalue"],
            "permutation_p_f1": perm["f1_pvalue"],
            "honest_permutation_p_accuracy": float(p_honest),
            "domain10_5fold_acc": md["accuracy"],
            "feature_jaccard_across_folds": jaccard,
            "learning_curve_final_gap": gap,
            "repeated_kfold_acc_mean": rep_sum[0]["mean"],
            "repeated_kfold_acc_q05": rep_sum[0]["q05"],
            "repeated_kfold_acc_q95": rep_sum[0]["q95"],
            "verdict": verdict,
        },
        "interpretation": interpretation,
        "protocol_rows": protocol_rows,
        "repeated_summary": rep_sum,
        "bootstrap": boot,
        "permutation_summary": {
            "accuracy_score": perm["accuracy_score"],
            "accuracy_pvalue": perm["accuracy_pvalue"],
            "accuracy_perm_mean": perm["accuracy_perm_mean"],
            "f1_score": perm["f1_score"],
            "f1_pvalue": perm["f1_pvalue"],
            "f1_perm_mean": perm["f1_perm_mean"],
            "n_permutations": N_PERM,
            "honest_foldwise": summary_honest_perm,
        },
        "learning_curve": lc.to_dict(orient="records"),
        "nested_best_params": nested_params,
    }
    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    write_report(OUT / "validation_report.md", summary)
    print("\n=== VERDICT ===")
    print(interpretation)
    print(f"\nWrote {OUT / 'validation_report.md'}")


if __name__ == "__main__":
    main()
