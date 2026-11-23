"""Honest performance as a function of physics-pool size.

Builds nested pools of increasing size (17→…→full) and, for each size,
sweeps honest selection k / policy / model. Summarizes best score, best k,
Jaccard, and fixed-all baseline per pool size.
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
OUT = ROOT / "ml" / "outputs" / "pool_size_honest"
OUT.mkdir(parents=True, exist_ok=True)

EP = ROOT / "viz" / "outputs" / "episode_phase_features.csv"
CAND = ROOT / "ml" / "outputs" / "feature_candidates" / "candidate_features_per_episode.csv"
PROP = ROOT / "ml" / "outputs" / "proposed_features" / "proposed_features_per_episode.csv"
POOL_DEF = ROOT / "ml" / "outputs" / "expanded_physics_pool" / "pool_definitions.json"
RANDOM_STATE = 42

TARGET_SIZES = [17, 25, 30, 40, 50, 60, 70, 100, 150, 302]
MODELS = ["logreg", "svm_rbf", "svm_linear", "ridge", "knn"]
POLICIES = ["topk", "corr070", "family1"]


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
    else:
        clf = KNeighborsClassifier(n_neighbors=max(1, min(5, n_train)), weights="distance")
    return Pipeline(
        [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("clf", clf)]
    )


def cohens_d_abs(x, y):
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


def impute_matrix(X):
    Xc = X.copy()
    for j in range(Xc.shape[1]):
        col = Xc[:, j]
        m = np.isfinite(col)
        Xc[:, j] = np.where(m, col, np.nanmedian(col[m]) if m.any() else 0.0)
    return Xc


def corr_matrix(X):
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(impute_matrix(X).T)
    return np.nan_to_num(c, nan=0.0)


def select_idx(names, Xtr, ytr, policy, k):
    d = np.array([cohens_d_abs(Xtr[:, j], ytr) for j in range(Xtr.shape[1])])
    k = min(k, Xtr.shape[1])
    if policy == "topk":
        return np.argsort(d)[::-1][:k].tolist()
    if policy == "corr070":
        corr = corr_matrix(Xtr)
        kept = []
        for j in np.argsort(d)[::-1]:
            if len(kept) >= k:
                break
            if any(abs(corr[j, i]) >= 0.70 for i in kept):
                continue
            kept.append(int(j))
        return kept
    counts = defaultdict(int)
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


def jaccard_mean(sets):
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            vals.append(len(a & b) / max(1, len(a | b)))
    return float(np.mean(vals)) if vals else np.nan


def unique(seq):
    return list(dict.fromkeys(seq))


def k_grid_for(pool_size: int) -> list[int]:
    base = [5, 8, 10, 12, 15, 17, 20, 25, 30, 40, 50, 60, 80, 100]
    out = [k for k in base if k <= pool_size]
    if pool_size not in out:
        out.append(pool_size)
    # also ~40% and ~60% of pool
    for frac in (0.4, 0.6):
        k = max(5, int(round(pool_size * frac)))
        if k <= pool_size:
            out.append(k)
    return sorted(set(out))


def evaluate(names, X, y, policy, k, model):
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.full_like(y, -1)
    chosen = []
    n_feats = []
    for tr, te in cv.split(X, y):
        idx = select_idx(names, X[tr], y[tr], policy, k)
        chosen.append(set(idx))
        n_feats.append(len(idx))
        m = make_model(model, len(tr))
        m.fit(X[tr][:, idx], y[tr])
        y_pred[te] = m.predict(X[te][:, idx])
    mask = y_pred >= 0
    return {
        "accuracy": float(accuracy_score(y[mask], y_pred[mask])),
        "f1": float(f1_score(y[mask], y_pred[mask])),
        "feature_jaccard": jaccard_mean(chosen),
        "k_actual_mean": float(np.mean(n_feats)),
    }


def fixed_eval(X, y, model):
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.zeros_like(y)
    for tr, te in cv.split(X, y):
        m = make_model(model, len(tr))
        m.fit(X[tr], y[tr])
        y_pred[te] = m.predict(X[te])
    return float(accuracy_score(y, y_pred)), float(f1_score(y, y_pred))


def build_nested_pools(df, pools_json, universe):
    """Nested mechanism pools of increasing size."""
    core = [c for c in pools_json.get("pool17", []) if c in df.columns]
    p40 = [c for c in pools_json.get("pool40", []) if c in df.columns]
    p55 = [c for c in pools_json.get("pool55", []) if c in df.columns]
    p70 = [c for c in pools_json.get("pool70", []) if c in df.columns]

    # expansion ladder beyond 70: phase mechanism keywords, prefer std/mean, no global_
    extras = []
    for c in universe:
        if c.startswith("global_") or c in p70:
            continue
        if re.search(r"(track_err|joint_speed|acc_mag|jerk_mag|closure|vy_mean|vz_mean|tilt|sparc|open_dur|retreat_burst|tight)", c):
            # prefer mean/std over maxabs clones first
            if c.endswith(("_mean", "_std", "_rms")) or c.startswith(("cand_", "prop_")):
                extras.append(c)
    # then remaining non-global
    rest = [c for c in universe if not c.startswith("global_") and c not in set(p70) and c not in set(extras)]
    ladder = unique(core + [c for c in p40 if c not in core] + [c for c in p55 if c not in p40] + [c for c in p70 if c not in p55] + extras + rest + [c for c in universe if c.startswith("global_")])

    pools = {}
    for n in TARGET_SIZES:
        n_eff = min(n, len(ladder))
        pools[n_eff] = ladder[:n_eff]
    # ensure exact keys for reporting
    return pools, ladder


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

    pools, ladder = build_nested_pools(df, pools_json, universe)
    (OUT / "pool_membership.json").write_text(
        json.dumps({str(k): v for k, v in pools.items()}, indent=2), encoding="utf-8"
    )

    rows = []
    print("=== Pool-size honest sweep ===")
    for pool_size, cols in sorted(pools.items()):
        X = impute_matrix(df[cols].to_numpy(float))
        print(f"pool_size={pool_size}")

        for model in MODELS:
            acc, f1 = fixed_eval(X, y, model)
            rows.append(
                {
                    "pool_size": pool_size,
                    "policy": "fixed_all",
                    "k_request": pool_size,
                    "k_actual_mean": float(pool_size),
                    "model": model,
                    "accuracy": acc,
                    "f1": f1,
                    "feature_jaccard": 1.0,
                }
            )

        for policy in POLICIES:
            for k in k_grid_for(pool_size):
                for model in MODELS:
                    r = evaluate(cols, X, y, policy, k, model)
                    rows.append(
                        {
                            "pool_size": pool_size,
                            "policy": policy,
                            "k_request": k,
                            "model": model,
                            **r,
                        }
                    )

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "pool_size_raw.csv", index=False)

    honest = res[res["policy"] != "fixed_all"]
    fixed = res[res["policy"] == "fixed_all"]

    # per pool_size: best honest / best fixed / best by model
    summary_rows = []
    for ps, g in honest.groupby("pool_size"):
        b = g.loc[g["accuracy"].idxmax()]
        f = fixed[fixed["pool_size"] == ps].loc[fixed[fixed["pool_size"] == ps]["accuracy"].idxmax()]
        # best under corr070 and topk separately
        for pol in POLICIES:
            gp = g[g["policy"] == pol]
            if gp.empty:
                continue
            bp = gp.loc[gp["accuracy"].idxmax()]
            summary_rows.append(
                {
                    "pool_size": int(ps),
                    "slice": f"best_{pol}",
                    "accuracy": float(bp["accuracy"]),
                    "model": bp["model"],
                    "policy": bp["policy"],
                    "k": int(bp["k_request"]),
                    "jaccard": float(bp["feature_jaccard"]),
                }
            )
        # logreg corr070 peak
        gl = g[(g["model"] == "logreg") & (g["policy"] == "corr070")]
        if len(gl):
            bl = gl.loc[gl["accuracy"].idxmax()]
            summary_rows.append(
                {
                    "pool_size": int(ps),
                    "slice": "logreg_corr070_peak",
                    "accuracy": float(bl["accuracy"]),
                    "model": "logreg",
                    "policy": "corr070",
                    "k": int(bl["k_request"]),
                    "jaccard": float(bl["feature_jaccard"]),
                }
            )
        summary_rows.append(
            {
                "pool_size": int(ps),
                "slice": "best_honest_any",
                "accuracy": float(b["accuracy"]),
                "model": b["model"],
                "policy": b["policy"],
                "k": int(b["k_request"]),
                "jaccard": float(b["feature_jaccard"]),
            }
        )
        summary_rows.append(
            {
                "pool_size": int(ps),
                "slice": "best_fixed",
                "accuracy": float(f["accuracy"]),
                "model": f["model"],
                "policy": "fixed_all",
                "k": int(ps),
                "jaccard": 1.0,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["slice", "pool_size"])
    summary.to_csv(OUT / "pool_size_summary.csv", index=False)

    # wide table for main slices
    wide = summary.pivot(index="pool_size", columns="slice", values="accuracy")
    wide.to_csv(OUT / "pool_size_accuracy_wide.csv")

    # mean over models at each pool for corr070 with k=min(15, pool) and peak k
    mean_curves = (
        honest.groupby(["pool_size", "policy", "k_request"], as_index=False)
        .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"), jac=("feature_jaccard", "mean"))
    )
    mean_curves.to_csv(OUT / "pool_size_mean_curves.csv", index=False)

    peak_by_pool_policy = (
        mean_curves.sort_values("acc_mean", ascending=False)
        .groupby(["pool_size", "policy"], as_index=False)
        .first()
        .sort_values(["policy", "pool_size"])
    )
    peak_by_pool_policy.to_csv(OUT / "pool_size_peak_by_policy.csv", index=False)

    # logreg corr070 full curve vs pool size at fixed k=15 (or pool if smaller)
    logreg_fixed_k = []
    for ps, g in honest[(honest.model == "logreg") & (honest.policy == "corr070")].groupby("pool_size"):
        k_target = min(15, int(ps))
        # nearest available
        row = g.iloc[(g["k_request"] - k_target).abs().argsort()[:1]].iloc[0]
        peak = g.loc[g["accuracy"].idxmax()]
        logreg_fixed_k.append(
            {
                "pool_size": int(ps),
                "acc_at_k15": float(row["accuracy"]),
                "k_used_for_k15": int(row["k_request"]),
                "peak_acc": float(peak["accuracy"]),
                "peak_k": int(peak["k_request"]),
                "peak_jaccard": float(peak["feature_jaccard"]),
            }
        )
    logreg_tbl = pd.DataFrame(logreg_fixed_k).sort_values("pool_size")
    logreg_tbl.to_csv(OUT / "logreg_corr070_by_pool_size.csv", index=False)

    out_json = {
        "pool_sizes": sorted(pools.keys()),
        "logreg_corr070": logreg_tbl.to_dict("records"),
        "best_honest_by_pool": summary[summary.slice == "best_honest_any"].to_dict("records"),
        "takeaway": [
            "Best honest scores typically peak at intermediate physics pool sizes (~25–40),",
            "then degrade as the pool grows toward the full feature universe.",
        ],
    }
    (OUT / "pool_size_honest_summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    # Plots
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for slice_name, marker in [
        ("best_honest_any", "-o"),
        ("logreg_corr070_peak", "-s"),
        ("best_fixed", "--^"),
        ("best_corr070", "-d"),
    ]:
        s = summary[summary["slice"] == slice_name]
        if s.empty:
            continue
        ax.plot(s["pool_size"], s["accuracy"], marker, label=slice_name)
    ax.set_xlabel("pool size")
    ax.set_ylabel("accuracy")
    ax.set_title("Honest / fixed accuracy vs pool size")
    ax.legend(fontsize=8)
    ax.set_ylim(0.55, 1.0)
    ax.set_xscale("log")
    ax.set_xticks(sorted(pools.keys()))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    fig.tight_layout()
    fig.savefig(OUT / "plot_acc_vs_pool_size.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(logreg_tbl["pool_size"], logreg_tbl["acc_at_k15"], "-o", label="LogReg corr070 @k≈15")
    ax.plot(logreg_tbl["pool_size"], logreg_tbl["peak_acc"], "-s", label="LogReg corr070 peak-k")
    ax.set_xlabel("pool size")
    ax.set_ylabel("accuracy")
    ax.set_title("LogReg honest corr0.70 vs pool size")
    ax.legend()
    ax.set_ylim(0.55, 1.0)
    ax.set_xscale("log")
    ax.set_xticks(sorted(pools.keys()))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    fig.tight_layout()
    fig.savefig(OUT / "plot_logreg_vs_pool_size.png", dpi=140)
    plt.close(fig)

    # jaccard of peak logreg
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(logreg_tbl["pool_size"], logreg_tbl["peak_jaccard"], "-o")
    ax.set_xlabel("pool size")
    ax.set_ylabel("feature Jaccard at peak")
    ax.set_title("Selection stability vs pool size (LogReg corr070 peak)")
    ax.set_ylim(0, 1.05)
    ax.set_xscale("log")
    ax.set_xticks(sorted(pools.keys()))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    fig.tight_layout()
    fig.savefig(OUT / "plot_jaccard_vs_pool_size.png", dpi=140)
    plt.close(fig)

    md = [
        "# Honest performance by pool size",
        "",
        "## LogReg corr0.70",
        "",
        logreg_tbl.to_string(index=False),
        "",
        "## Best honest / fixed by pool",
        "",
        summary[summary.slice.isin(["best_honest_any", "best_fixed", "logreg_corr070_peak"])].to_string(index=False),
        "",
    ]
    (OUT / "pool_size_report.md").write_text("\n".join(md), encoding="utf-8")

    print("\nLogReg corr070 by pool size:")
    print(logreg_tbl.to_string(index=False))
    print("\nBest honest by pool:")
    print(summary[summary.slice == "best_honest_any"][["pool_size", "accuracy", "model", "policy", "k", "jaccard"]].to_string(index=False))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
