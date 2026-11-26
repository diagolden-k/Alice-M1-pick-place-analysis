"""Test strategies to exclude non-physical / redundant-similar features.

Strategies compared (all evaluated with fixed-set Stratified 5-fold SVM/LogReg
and a permutation sanity check on the final shortlist):

1. none_top25              — baseline: global top-25 |d| (optimistic reference)
2. corr_prune              — among |corr|>=thr pairs, keep higher |d| only
3. family_cap              — at most K features per phase×signal family
4. physics_allowlist       — only pre-specified physically motivated features
5. physics+corr            — allowlist then corr prune
6. stability_select        — bootstrap: keep features selected in >= freq folds
7. hybrid_must+corr        — must-keep physics set + corr-pruned fillers to n=15
8. vif_prune               — iterative VIF drop (multicollinearity)

Also reports: retained count, mean pairwise |corr|, physical coverage score.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path("/workspace")
OUT = ROOT / "ml" / "outputs" / "feature_exclusion"
OUT.mkdir(parents=True, exist_ok=True)

EP = ROOT / "viz" / "outputs" / "episode_phase_features.csv"
RANK = ROOT / "viz" / "outputs" / "feature_influence_ranking.csv"
CAND = ROOT / "ml" / "outputs" / "feature_candidates" / "candidate_features_per_episode.csv"
PROP = ROOT / "ml" / "outputs" / "proposed_features" / "proposed_features_per_episode.csv"
RANDOM_STATE = 42


# --- Physical taxonomy -------------------------------------------------------
# Features matching these families are considered physically motivated for tip-over.
PHYSICS_MUST = [
    # approach instability
    "approach_track_err_std",
    "approach_joint_speed_std",
    "approach_speed_maxabs",
    # grasp / transport
    "grasp_acc_mag_mean",
    "lift_duration",
    # carry / place geometry proxies
    "carry_right_vy_mean",
    "carry_right_vz_mean",
    # retreat / release (existing bank)
    "retreat_vy_mean",
    # engineered physics
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

# Patterns that are usually statistical clones / low physical specificity
LOW_VALUE_PATTERNS = [
    r".*_maxabs$",  # often redundant with rms/std for same signal
    r"^global_.*",  # diluted whole-episode averages (except we allow SPARC-like engineered)
    r".*_frac$",
    r"^n_frames$",
    r"^duration$",
]


def family_key(name: str) -> str:
    """Group near-duplicate stats of the same physical signal."""
    # strip trailing stat suffixes
    base = re.sub(r"_(mean|std|rms|maxabs|min|median)$", "", name)
    # also merge dy/dz variants lightly
    return base


def phase_of(name: str) -> str:
    for p in (
        "approach",
        "grasp",
        "lift",
        "carry_right",
        "place",
        "retreat",
        "global",
        "cand_",
        "prop_",
    ):
        if name.startswith(p):
            return p.rstrip("_")
    return "other"


def cohens_d_signed(x: np.ndarray, y: np.ndarray) -> float:
    x1, x0 = x[y == 1], x[y == 0]
    x1, x0 = x1[np.isfinite(x1)], x0[np.isfinite(x0)]
    if len(x1) < 2 or len(x0) < 2:
        return np.nan
    pooled = np.sqrt(
        ((len(x1) - 1) * np.var(x1, ddof=1) + (len(x0) - 1) * np.var(x0, ddof=1))
        / (len(x1) + len(x0) - 2)
    )
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(x1) - np.mean(x0)) / pooled)


def pairwise_abs_corr(X: np.ndarray) -> float:
    if X.shape[1] < 2:
        return 0.0
    # impute for corr
    Xc = X.copy()
    for j in range(Xc.shape[1]):
        col = Xc[:, j]
        m = np.isfinite(col)
        if m.any():
            col = np.where(m, col, np.nanmedian(col[m]))
        Xc[:, j] = col
    c = np.corrcoef(Xc.T)
    iu = np.triu_indices_from(c, k=1)
    return float(np.nanmean(np.abs(c[iu])))


def make_pipe(model: str):
    if model == "svm":
        clf = SVC(C=3.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)
    else:
        clf = LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", clf),
        ]
    )


def oof_scores(X: np.ndarray, y: np.ndarray, model: str = "svm") -> dict:
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    pred = np.zeros_like(y)
    for tr, te in cv.split(X, y):
        m = clone(make_pipe(model))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
    }


def corr_prune(cols: list[str], abs_d: dict[str, float], corr: pd.DataFrame, thr: float = 0.85) -> list[str]:
    """Greedy: sort by |d|, add if not highly correlated with already kept."""
    ordered = sorted(cols, key=lambda c: abs_d.get(c, 0.0), reverse=True)
    kept = []
    for c in ordered:
        ok = True
        for k in kept:
            r = corr.loc[c, k]
            if np.isfinite(r) and abs(r) >= thr:
                ok = False
                break
        if ok:
            kept.append(c)
    return kept


def family_cap(cols: list[str], abs_d: dict[str, float], k_per_family: int = 1) -> list[str]:
    ordered = sorted(cols, key=lambda c: abs_d.get(c, 0.0), reverse=True)
    counts = defaultdict(int)
    kept = []
    for c in ordered:
        fam = family_key(c)
        if counts[fam] < k_per_family:
            kept.append(c)
            counts[fam] += 1
    return kept


def drop_low_value_patterns(cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        if any(re.search(p, c) for p in LOW_VALUE_PATTERNS):
            # keep if in physics must
            if c not in PHYSICS_MUST:
                continue
        out.append(c)
    return out


def vif_prune(cols: list[str], Xdf: pd.DataFrame, max_vif: float = 10.0) -> list[str]:
    """Iteratively drop highest VIF until all <= max_vif or < 2 features."""
    from numpy.linalg import lstsq

    kept = cols[:]
    # simple VIF via R^2 of regressing each on others
    while len(kept) >= 3:
        X = Xdf[kept].to_numpy(float)
        # median impute
        for j in range(X.shape[1]):
            m = np.isfinite(X[:, j])
            X[~m, j] = np.nanmedian(X[m, j]) if m.any() else 0.0
        # standardize
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        vifs = []
        for j in range(X.shape[1]):
            yj = X[:, j]
            Xo = np.delete(X, j, axis=1)
            # least squares
            beta, *_ = lstsq(Xo, yj, rcond=None)
            pred = Xo @ beta
            ss_res = np.sum((yj - pred) ** 2)
            ss_tot = np.sum((yj - yj.mean()) ** 2) + 1e-12
            r2 = 1.0 - ss_res / ss_tot
            r2 = min(max(r2, 0.0), 0.9999)
            vifs.append(1.0 / (1.0 - r2))
        vmax = max(vifs)
        if vmax <= max_vif:
            break
        drop_i = int(np.argmax(vifs))
        kept.pop(drop_i)
    return kept


def stability_select(
    cols: list[str],
    Xdf: pd.DataFrame,
    y: np.ndarray,
    abs_d_full: dict[str, float],
    n_boot: int = 50,
    top_k: int = 15,
    freq_thr: float = 0.6,
) -> list[str]:
    """How often a feature appears in bootstrap top-k by |d|."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y)
    counts = defaultdict(int)
    X = Xdf[cols].to_numpy(float)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        scores = []
        for j, c in enumerate(cols):
            scores.append(abs(cohens_d_signed(X[idx, j], yy)))
        scores = np.nan_to_num(np.array(scores), nan=0.0)
        for j in np.argsort(scores)[::-1][:top_k]:
            counts[cols[j]] += 1
    freq = {c: counts[c] / n_boot for c in cols}
    selected = [c for c, f in freq.items() if f >= freq_thr]
    # if too few, take highest frequency
    if len(selected) < 8:
        selected = sorted(cols, key=lambda c: (freq.get(c, 0), abs_d_full.get(c, 0)), reverse=True)[:12]
    return selected, freq


def physical_coverage(cols: list[str]) -> dict:
    """Checklist: does the set cover key tip-over mechanisms?"""
    checks = {
        "approach_stability": any("approach" in c and ("track" in c or "speed" in c) for c in cols),
        "grasp_quality": any("grasp" in c or "tight" in c or "closure" in c for c in cols),
        "release_timing": any("open_dur" in c or "open_duration" in c for c in cols),
        "retreat_yank": any("retreat_burst" in c or (c.startswith("retreat_") and "speed" in c) or c == "retreat_vy_mean" for c in cols),
        "pose_orientation": any("tilt" in c or "z_release" in c for c in cols),
        "smoothness": any("sparc" in c or "jerk" in c for c in cols),
    }
    return {
        **checks,
        "n_covered": int(sum(checks.values())),
        "coverage_frac": float(np.mean(list(checks.values()))),
    }


def main():
    ep = pd.read_csv(EP)
    rank = pd.read_csv(RANK)
    cand = pd.read_csv(CAND)
    prop = pd.read_csv(PROP)
    df = ep.merge(cand.drop(columns=["label"], errors="ignore"), on="episode")
    df = df.merge(prop.drop(columns=["label"], errors="ignore"), on="episode")
    y = df["label"].to_numpy(int)

    # Candidate universe: original bank + engineered
    bank = [c for c in ep.columns if c not in ("episode", "label")]
    eng = [c for c in df.columns if c.startswith("cand_") or c.startswith("prop_")]
    # drop fold-dependent template feats from exclusion tests (already leakage-controlled elsewhere)
    eng = [c for c in eng if c not in ("prop_dtw_to_success", "prop_release_pose_dev", "prop_tilt_dev_to_success")]
    universe = list(dict.fromkeys(bank + eng))
    universe = [c for c in universe if c in df.columns]

    # effect sizes on full data (for ranking inside filters — acknowledged optimistic for selection,
    # but here we compare FILTERING policies on a common protocol)
    abs_d = {}
    signed_d = {}
    for c in universe:
        d = cohens_d_signed(df[c].to_numpy(float), y)
        signed_d[c] = d
        abs_d[c] = abs(d) if np.isfinite(d) else 0.0

    # correlation matrix (imputed)
    Xc = df[universe].copy()
    for c in universe:
        col = Xc[c].to_numpy(float)
        m = np.isfinite(col)
        Xc[c] = np.where(m, col, np.nanmedian(col[m]) if m.any() else 0.0)
    corr = Xc.corr().abs()

    # Seed pool: top-40 by |d| to keep computation sane for some filters
    top40 = sorted(universe, key=lambda c: abs_d[c], reverse=True)[:40]
    top25 = sorted(universe, key=lambda c: abs_d[c], reverse=True)[:25]

    # Build strategy shortlists
    strategies = {}

    strategies["none_top25"] = top25

    strategies["corr_prune_0.85"] = corr_prune(top40, abs_d, corr, thr=0.85)
    strategies["corr_prune_0.70"] = corr_prune(top40, abs_d, corr, thr=0.70)

    strategies["family_cap_1"] = family_cap(top40, abs_d, k_per_family=1)[:20]
    strategies["family_cap_2"] = family_cap(top40, abs_d, k_per_family=2)[:20]

    # pattern drop then top by d
    filtered = drop_low_value_patterns(top40)
    strategies["drop_global_maxabs_then_top15"] = sorted(filtered, key=lambda c: abs_d[c], reverse=True)[:15]

    # physics allowlist present in data
    phys = [c for c in PHYSICS_MUST if c in df.columns]
    strategies["physics_allowlist"] = phys
    strategies["physics+corr_0.85"] = corr_prune(phys, abs_d, corr, thr=0.85)

    # stability
    stab, freq = stability_select(top40, df, y, abs_d, n_boot=80, top_k=12, freq_thr=0.55)
    strategies["stability_freq>=0.55"] = stab
    # save frequencies
    pd.Series(freq).sort_values(ascending=False).head(30).to_csv(OUT / "stability_frequencies.csv", header=["freq"])

    # hybrid: must physics + corr-pruned fillers from top40 to reach ~15
    must = [c for c in phys if c in top40 or True]
    must = [c for c in PHYSICS_MUST if c in df.columns][:10]
    fillers = [c for c in corr_prune(top40, abs_d, corr, 0.85) if c not in must]
    hybrid = list(dict.fromkeys(must + fillers))[:16]
    strategies["hybrid_must+corr"] = hybrid

    # VIF on top25
    strategies["vif_prune_top25"] = vif_prune(top25, df, max_vif=10.0)

    # Evaluate
    rows = []
    print("=== Exclusion strategy comparison ===")
    for name, cols in strategies.items():
        cols = [c for c in cols if c in df.columns]
        if len(cols) < 2:
            continue
        X = df[cols].to_numpy(float)
        svm = oof_scores(X, y, "svm")
        lr = oof_scores(X, y, "logreg")
        cov = physical_coverage(cols)
        red = pairwise_abs_corr(X)
        n_phys = sum(1 for c in cols if c in PHYSICS_MUST)
        row = {
            "strategy": name,
            "n_features": len(cols),
            "svm_acc": svm["accuracy"],
            "svm_f1": svm["f1"],
            "logreg_acc": lr["accuracy"],
            "logreg_f1": lr["f1"],
            "mean_pairwise_abs_corr": red,
            "physics_coverage": cov["coverage_frac"],
            "n_physics_must_overlap": n_phys,
            "features": ",".join(cols),
        }
        rows.append(row)
        print(
            f"{name:32s} n={len(cols):2d}  SVM={svm['accuracy']:.3f}  LR={lr['accuracy']:.3f}  "
            f"corr={red:.2f}  phys_cov={cov['coverage_frac']:.2f}"
        )

    res = pd.DataFrame(rows).sort_values(["svm_acc", "physics_coverage"], ascending=False)
    res.to_csv(OUT / "exclusion_strategy_results.csv", index=False)

    # Permutation on best hybrid / physics sets
    print("\n=== Permutation on key shortlists ===")
    perm_rows = []
    for key in ["physics_allowlist", "physics+corr_0.85", "hybrid_must+corr", "corr_prune_0.85", "none_top25"]:
        cols = [c for c in strategies[key] if c in df.columns]
        X = df[cols].to_numpy(float)
        score, perm, p = permutation_test_score(
            make_pipe("svm"),
            X,
            y,
            scoring="accuracy",
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
            n_permutations=200,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        perm_rows.append(
            {"strategy": key, "n": len(cols), "acc": float(score), "null_mean": float(np.mean(perm)), "p": float(p)}
        )
        print(f"{key:32s} acc={score:.3f} null={np.mean(perm):.3f} p={p:.4f}")
    pd.DataFrame(perm_rows).to_csv(OUT / "exclusion_permutation.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot = res.sort_values("svm_acc")
    axes[0].barh(plot["strategy"], plot["svm_acc"], color="#4C78A8")
    axes[0].set_xlabel("SVM 5-fold Acc")
    axes[0].set_title("Accuracy after exclusion")
    axes[0].set_xlim(0.5, 1.0)
    axes[0].grid(True, axis="x", alpha=0.3)

    sc = axes[1].scatter(
        res["mean_pairwise_abs_corr"],
        res["svm_acc"],
        s=60 + 8 * res["n_features"],
        c=res["physics_coverage"],
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    for _, r in res.iterrows():
        axes[1].annotate(r["strategy"].replace("corr_prune", "corr").replace("stability_freq>=0.55", "stab"), (r["mean_pairwise_abs_corr"], r["svm_acc"]), fontsize=7, alpha=0.8)
    axes[1].set_xlabel("Mean pairwise |corr| (lower=less redundant)")
    axes[1].set_ylabel("SVM Acc")
    axes[1].set_title("Redundancy vs accuracy (color=physics coverage)")
    fig.colorbar(sc, ax=axes[1], label="physics coverage")
    fig.tight_layout()
    fig.savefig(OUT / "exclusion_comparison.png", dpi=140)
    plt.close(fig)

    # Recommend
    # Prefer: high phys coverage, lower redundancy, acc not much below top25
    score = (
        res["svm_acc"] * 0.45
        + res["physics_coverage"] * 0.35
        + (1 - res["mean_pairwise_abs_corr"]) * 0.20
    )
    res2 = res.copy()
    res2["composite"] = score
    best = res2.sort_values("composite", ascending=False).iloc[0]

    summary = {
        "best_strategy": best["strategy"],
        "best_features": best["features"].split(","),
        "best_metrics": {
            "n": int(best["n_features"]),
            "svm_acc": float(best["svm_acc"]),
            "logreg_acc": float(best["logreg_acc"]),
            "mean_corr": float(best["mean_pairwise_abs_corr"]),
            "physics_coverage": float(best["physics_coverage"]),
            "composite": float(best["composite"]),
        },
        "guidance": {
            "exclude_how": [
                "1) Start from physics allowlist (mechanism coverage), not from top-|d| of 267.",
                "2) Corr-prune |r|>=0.85 (or 0.70 if still redundant) keeping higher |d| or clearer physics name.",
                "3) Cap 1 feature per family (same base signal with mean/std/rms/maxabs).",
                "4) Drop global_* and pure duration/frac unless mechanism-linked.",
                "5) Optional: stability selection freq>=0.55 on a SMALL prefiltered pool.",
                "6) Do NOT use VIF/corr alone without physics — they keep strong but opaque clones.",
            ]
        },
    }
    (OUT / "exclusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # write human report
    lines = [
        "# Feature exclusion test report\n",
        "## Best by composite (acc + physics coverage + low redundancy)\n",
        f"- **{best['strategy']}** n={int(best['n_features'])} SVM={best['svm_acc']:.3f} "
        f"LR={best['logreg_acc']:.3f} mean|corr|={best['mean_pairwise_abs_corr']:.2f} "
        f"phys_cov={best['physics_coverage']:.2f}\n",
        "### Features\n",
    ]
    for f in best["features"].split(","):
        lines.append(f"- `{f}` (|d|={abs_d.get(f, float('nan')):.3f})")
    lines.append("\n## All strategies\n")
    lines.append(res[["strategy", "n_features", "svm_acc", "logreg_acc", "mean_pairwise_abs_corr", "physics_coverage"]].to_string(index=False))
    (OUT / "exclusion_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== BEST ===")
    print(best["strategy"], "n=", int(best["n_features"]), "SVM", round(best["svm_acc"], 3))
    print("features:", best["features"])
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
