"""Sweep honest fold-wise selection size k.

Question: if we keep more features inside each train fold, does accuracy rise,
plateau, or fall (noise / multicollinearity)?

Sweeps k across pools (full / pool70 / pool55 / pool40 / pool17) and models.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path("/workspace")
OUT = ROOT / "ml" / "outputs" / "honest_k_sweep"
OUT.mkdir(parents=True, exist_ok=True)

EP = ROOT / "viz" / "outputs" / "episode_phase_features.csv"
CAND = ROOT / "ml" / "outputs" / "feature_candidates" / "candidate_features_per_episode.csv"
PROP = ROOT / "ml" / "outputs" / "proposed_features" / "proposed_features_per_episode.csv"
POOL_DEF = ROOT / "ml" / "outputs" / "expanded_physics_pool" / "pool_definitions.json"
RANDOM_STATE = 42

PHYSICS17 = [
    "approach_track_err_std",
    "approach_joint_speed_std",
    "approach_speed_maxabs",
    "grasp_acc_mag_mean",
    "lift_duration",
    "carry_right_vy_mean",
    "carry_right_vz_mean",
    "retreat_vy_mean",
    "cand_open_duration",
    "cand_open_duration_over_hold",
    "prop_retreat_burst_jspeed",
    "prop_retreat_burst_speed",
    "prop_tilt_pitch_release",
    "prop_z_release",
    "prop_sparc_palm",
    "prop_open_dur_ratio",
    "prop_grasp_tight_std",
]

K_GRID = [5, 8, 10, 12, 15, 17, 20, 25, 30, 40, 50, 60]
MODELS = ["logreg", "svm_rbf", "svm_linear", "ridge", "knn"]
POLICIES = ["topk", "corr070", "corr085", "family1"]


def family_key(name: str) -> str:
    return re.sub(r"_(mean|std|rms|maxabs|min|median)$", "", name)


def make_model(name: str, n_train: int):
    if name == "svm_rbf":
        clf = SVC(C=3.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)
    elif name == "svm_linear":
        clf = SVC(C=1.0, kernel="linear", class_weight="balanced", random_state=RANDOM_STATE)
    elif name == "logreg":
        clf = LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)
    elif name == "ridge":
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=RANDOM_STATE)
    elif name == "knn":
        clf = KNeighborsClassifier(n_neighbors=max(1, min(5, n_train)), weights="distance")
    else:
        raise ValueError(name)
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", clf),
        ]
    )


def cohens_d_abs(x: np.ndarray, y: np.ndarray) -> float:
    x1, x0 = x[y == 1], x[y == 0]
    x1, x0 = x1[np.isfinite(x1)], x0[np.isfinite(x0)]
    if len(x1) < 2 or len(x0) < 2:
        return 0.0
    pooled = np.sqrt(
        ((len(x1) - 1) * np.var(x1, ddof=1) + (len(x0) - 1) * np.var(x0, ddof=1))
        / (len(x1) + len(x0) - 2)
    )
    if pooled < 1e-12:
        return 0.0
    return abs(float((np.mean(x1) - np.mean(x0)) / pooled))


def impute_matrix(X: np.ndarray) -> np.ndarray:
    Xc = X.copy()
    for j in range(Xc.shape[1]):
        col = Xc[:, j]
        m = np.isfinite(col)
        Xc[:, j] = np.where(m, col, np.nanmedian(col[m]) if m.any() else 0.0)
    return Xc


def corr_matrix(X: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(impute_matrix(X).T)
    return np.nan_to_num(c, nan=0.0)


def select_idx(names: list[str], Xtr: np.ndarray, ytr: np.ndarray, policy: str, k: int) -> list[int]:
    d = np.array([cohens_d_abs(Xtr[:, j], ytr) for j in range(Xtr.shape[1])])
    k = min(k, Xtr.shape[1])
    if policy == "topk":
        return np.argsort(d)[::-1][:k].tolist()
    if policy in ("corr070", "corr085"):
        thr = 0.70 if policy == "corr070" else 0.85
        corr = corr_matrix(Xtr)
        kept: list[int] = []
        for j in np.argsort(d)[::-1]:
            if len(kept) >= k:
                break
            if any(abs(corr[j, i]) >= thr for i in kept):
                continue
            kept.append(int(j))
        return kept
    if policy == "family1":
        counts: dict[str, int] = defaultdict(int)
        kept = []
        for j in np.argsort(d)[::-1]:
            fam = family_key(names[j])
            if counts[fam] >= 1:
                continue
            kept.append(int(j))
            counts[fam] += 1
            if len(kept) >= k:
                break
        return kept
    raise ValueError(policy)


def jaccard_mean(sets: list[set[int]]) -> float:
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            vals.append(len(a & b) / max(1, len(a | b)))
    return float(np.mean(vals)) if vals else np.nan


def evaluate(pool_name: str, names: list[str], X: np.ndarray, y: np.ndarray, policy: str, k: int, model: str) -> dict:
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.full_like(y, -1)
    chosen: list[set[int]] = []
    n_feats = []

    for tr, te in cv.split(X, y):
        idx = select_idx(names, X[tr], y[tr], policy, k)
        chosen.append(set(idx))
        n_feats.append(len(idx))
        m = make_model(model, n_train=len(tr))
        m.fit(X[tr][:, idx], y[tr])
        y_pred[te] = m.predict(X[te][:, idx])

    mask = y_pred >= 0
    return {
        "pool": pool_name,
        "pool_size": X.shape[1],
        "policy": policy,
        "k_request": k,
        "k_actual_mean": float(np.mean(n_feats)),
        "model": model,
        "accuracy": float(accuracy_score(y[mask], y_pred[mask])),
        "f1": float(f1_score(y[mask], y_pred[mask])),
        "feature_jaccard": jaccard_mean(chosen),
    }


def unique(seq):
    return list(dict.fromkeys(seq))


def main():
    ep = pd.read_csv(EP)
    cand = pd.read_csv(CAND)
    prop = pd.read_csv(PROP)
    df = ep.merge(cand.drop(columns=["label"], errors="ignore"), on="episode")
    df = df.merge(prop.drop(columns=["label"], errors="ignore"), on="episode")
    y = df["label"].to_numpy(int)

    bank = [c for c in ep.columns if c not in ("episode", "label")]
    eng = [
        c
        for c in df.columns
        if c.startswith(("cand_", "prop_"))
        and c not in ("prop_dtw_to_success", "prop_release_pose_dev", "prop_tilt_dev_to_success")
    ]
    universe = unique(bank + eng)

    pools_json = json.loads(POOL_DEF.read_text()) if POOL_DEF.exists() else {}
    pool17 = [c for c in pools_json.get("pool17", PHYSICS17) if c in df.columns]
    pool40 = [c for c in pools_json.get("pool40", pool17) if c in df.columns]
    pool55 = [c for c in pools_json.get("pool55", pool40) if c in df.columns]
    pool70 = [c for c in pools_json.get("pool70", pool55) if c in df.columns]

    pools = {
        "pool17": pool17,
        "pool40": pool40,
        "pool55": pool55,
        "pool70": pool70,
        "full": universe,
    }

    # also fixed-all baselines (no selection) for each pool
    rows = []
    print("=== Honest k sweep ===")
    for pool_name, cols in pools.items():
        X = impute_matrix(df[cols].to_numpy(float))
        names = cols
        print(f"pool={pool_name} size={len(cols)}")

        # fixed whole pool baseline per model
        for model in MODELS:
            cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
            y_pred = np.zeros_like(y)
            for tr, te in cv.split(X, y):
                m = make_model(model, n_train=len(tr))
                m.fit(X[tr], y[tr])
                y_pred[te] = m.predict(X[te])
            rows.append(
                {
                    "pool": pool_name,
                    "pool_size": len(cols),
                    "policy": "fixed_all",
                    "k_request": len(cols),
                    "k_actual_mean": float(len(cols)),
                    "model": model,
                    "accuracy": float(accuracy_score(y, y_pred)),
                    "f1": float(f1_score(y, y_pred)),
                    "feature_jaccard": 1.0,
                }
            )

        for policy in POLICIES:
            for k in K_GRID:
                if k > len(cols) and policy == "topk":
                    # still run once at pool size via fixed; skip redundant
                    if k != len(cols):
                        continue
                for model in MODELS:
                    # skip absurd: requesting k >> pool for corr may still return <k
                    if k > len(cols) * 1.2:
                        continue
                    r = evaluate(pool_name, names, X, y, policy, k, model)
                    rows.append(r)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "honest_k_raw.csv", index=False)
    print(f"rows={len(res)}")

    # Aggregations
    # best k per pool×policy×model
    honest = res[res["policy"] != "fixed_all"].copy()
    idx = honest.groupby(["pool", "policy", "model"])["accuracy"].idxmax()
    best = honest.loc[idx].sort_values(["pool", "accuracy"], ascending=[True, False])
    best.to_csv(OUT / "best_k_per_setting.csv", index=False)

    # mean over models for curves
    curve = (
        honest.groupby(["pool", "policy", "k_request"], as_index=False)
        .agg(
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            acc_max=("accuracy", "max"),
            jac_mean=("feature_jaccard", "mean"),
            k_actual=("k_actual_mean", "mean"),
            n=("accuracy", "count"),
        )
        .sort_values(["pool", "policy", "k_request"])
    )
    curve.to_csv(OUT / "honest_k_curves.csv", index=False)

    # logreg + svm focus
    focus_models = ["logreg", "svm_rbf"]
    focus = honest[honest["model"].isin(focus_models)].copy()
    focus_curve = (
        focus.groupby(["pool", "policy", "model", "k_request"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), jaccard=("feature_jaccard", "mean"), k_actual=("k_actual_mean", "mean"))
        .sort_values(["pool", "model", "policy", "k_request"])
    )
    focus_curve.to_csv(OUT / "honest_k_focus_curves.csv", index=False)

    fixed = res[res["policy"] == "fixed_all"]
    fixed_sum = (
        fixed.groupby(["pool", "model"], as_index=False)
        .agg(accuracy=("accuracy", "mean"))
        .sort_values(["pool", "accuracy"], ascending=[True, False])
    )
    fixed_sum.to_csv(OUT / "fixed_pool_baselines.csv", index=False)

    # headline: for each pool, best honest vs fixed
    headlines = []
    for pool in pools:
        sub = honest[honest["pool"] == pool]
        if sub.empty:
            continue
        b = sub.loc[sub["accuracy"].idxmax()]
        fbest = fixed[fixed["pool"] == pool].sort_values("accuracy", ascending=False).iloc[0]
        # also best at small/medium/large k bands
        for band, lo, hi in [("small_k<=12", 1, 12), ("mid_k_13_25", 13, 25), ("large_k>=30", 30, 10_000)]:
            band_sub = sub[(sub["k_request"] >= lo) & (sub["k_request"] <= hi)]
            if band_sub.empty:
                continue
            bb = band_sub.loc[band_sub["accuracy"].idxmax()]
            headlines.append(
                {
                    "pool": pool,
                    "band": band,
                    "best_acc": float(bb["accuracy"]),
                    "best_model": bb["model"],
                    "best_policy": bb["policy"],
                    "best_k": int(bb["k_request"]),
                    "jaccard": float(bb["feature_jaccard"]),
                }
            )
        headlines.append(
            {
                "pool": pool,
                "band": "overall_best_honest",
                "best_acc": float(b["accuracy"]),
                "best_model": b["model"],
                "best_policy": b["policy"],
                "best_k": int(b["k_request"]),
                "jaccard": float(b["feature_jaccard"]),
            }
        )
        headlines.append(
            {
                "pool": pool,
                "band": "fixed_all_best_model",
                "best_acc": float(fbest["accuracy"]),
                "best_model": fbest["model"],
                "best_policy": "fixed_all",
                "best_k": int(fbest["k_request"]),
                "jaccard": 1.0,
            }
        )
    head = pd.DataFrame(headlines)
    head.to_csv(OUT / "headline_by_pool_band.csv", index=False)

    # correlation: k vs acc within pool×policy (mean models)
    sens = []
    for (pool, policy), g in curve.groupby(["pool", "policy"]):
        if g["k_request"].nunique() < 3:
            continue
        # compare small vs large
        small = g[g["k_request"] <= 12]["acc_mean"].mean()
        large = g[g["k_request"] >= 30]["acc_mean"].mean()
        peak_row = g.loc[g["acc_mean"].idxmax()]
        sens.append(
            {
                "pool": pool,
                "policy": policy,
                "peak_k": int(peak_row["k_request"]),
                "peak_acc": float(peak_row["acc_mean"]),
                "acc_k<=12": float(small) if np.isfinite(small) else np.nan,
                "acc_k>=30": float(large) if np.isfinite(large) else np.nan,
                "delta_large_minus_small": float(large - small) if np.isfinite(small) and np.isfinite(large) else np.nan,
            }
        )
    sens_df = pd.DataFrame(sens).sort_values(["pool", "delta_large_minus_small"])
    sens_df.to_csv(OUT / "k_sensitivity_summary.csv", index=False)

    summary = {
        "n_rows": int(len(res)),
        "k_grid": K_GRID,
        "models": MODELS,
        "policies": POLICIES,
        "pools": {k: len(v) for k, v in pools.items()},
        "sensitivity": sens_df.to_dict("records"),
        "top10": honest.sort_values("accuracy", ascending=False)
        .head(10)[["pool", "policy", "k_request", "model", "accuracy", "feature_jaccard"]]
        .to_dict("records"),
    }
    (OUT / "honest_k_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plots
    # 1) accuracy vs k for corr070, mean over models, by pool
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for pool, style in [("pool17", "-o"), ("pool40", "-s"), ("pool55", "-^"), ("pool70", "-d"), ("full", "--x")]:
        g = curve[(curve["pool"] == pool) & (curve["policy"] == "corr070")]
        if g.empty:
            continue
        ax.plot(g["k_request"], g["acc_mean"], style, label=pool)
    ax.set_xlabel("k (requested features)")
    ax.set_ylabel("mean accuracy over models")
    ax.set_title("Honest corr0.70: accuracy vs selection size k")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "plot_acc_vs_k_corr070.png", dpi=140)
    plt.close(fig)

    # 2) topk vs corr070 for pool40 and full
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, pool in zip(axes, ["pool40", "full"]):
        for policy, ls in [("topk", "-o"), ("corr070", "-s"), ("family1", "-^")]:
            g = curve[(curve["pool"] == pool) & (curve["policy"] == policy)]
            ax.plot(g["k_request"], g["acc_mean"], ls, label=policy)
        ax.set_title(pool)
        ax.set_xlabel("k")
        ax.legend(fontsize=8)
        ax.set_ylim(0.5, 1.0)
    axes[0].set_ylabel("mean accuracy")
    fig.suptitle("Policy comparison as k grows")
    fig.tight_layout()
    fig.savefig(OUT / "plot_policy_vs_k.png", dpi=140)
    plt.close(fig)

    # 3) logreg focus heatmap-ish lines
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, model in zip(axes, focus_models):
        for pool in ["pool17", "pool40", "full"]:
            g = focus_curve[(focus_curve["model"] == model) & (focus_curve["pool"] == pool) & (focus_curve["policy"] == "corr070")]
            ax.plot(g["k_request"], g["accuracy"], "-o", label=pool)
        ax.set_title(model)
        ax.set_xlabel("k")
        ax.legend(fontsize=8)
        ax.set_ylim(0.5, 1.0)
    axes[0].set_ylabel("accuracy")
    fig.suptitle("Honest corr0.70: LogReg / SVM-RBF vs k")
    fig.tight_layout()
    fig.savefig(OUT / "plot_logreg_svm_vs_k.png", dpi=140)
    plt.close(fig)

    # 4) jaccard vs k
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for pool in ["pool17", "pool40", "full"]:
        g = curve[(curve["pool"] == pool) & (curve["policy"] == "corr070")]
        ax.plot(g["k_request"], g["jac_mean"], "-o", label=pool)
    ax.set_xlabel("k")
    ax.set_ylabel("feature Jaccard across folds")
    ax.set_title("Selection stability vs k (corr0.70)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT / "plot_jaccard_vs_k.png", dpi=140)
    plt.close(fig)

    md = [
        "# Honest selection size (k) sweep",
        "",
        f"- Rows: {len(res)}",
        f"- Pools: { {k: len(v) for k, v in pools.items()} }",
        f"- k grid: {K_GRID}",
        "",
        "## Sensitivity (mean over models)",
        "",
        sens_df.to_string(index=False),
        "",
        "## Headlines by pool/band",
        "",
        head.to_string(index=False),
        "",
        "## Top honest configs",
        "",
        honest.sort_values("accuracy", ascending=False)
        .head(15)[["pool", "policy", "k_request", "model", "accuracy", "feature_jaccard"]]
        .to_string(index=False),
        "",
    ]
    (OUT / "honest_k_report.md").write_text("\n".join(md), encoding="utf-8")

    print("\nSensitivity:")
    print(sens_df.to_string(index=False))
    print("\nTop 10 honest:")
    print(
        honest.sort_values("accuracy", ascending=False)
        .head(10)[["pool", "policy", "k_request", "model", "accuracy", "feature_jaccard"]]
        .to_string(index=False)
    )
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
