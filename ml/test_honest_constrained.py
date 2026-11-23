"""Why honest CV is low, and whether pool constraints raise it.

Honest = feature choice uses TRAIN fold only (no full-data top-|d| leakage).
We compare unconstrained top-k vs constrained pools / policies inside each fold.
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path("/workspace")
OUT = ROOT / "ml" / "outputs" / "honest_constrained"
OUT.mkdir(parents=True, exist_ok=True)

EP = ROOT / "viz" / "outputs" / "episode_phase_features.csv"
CAND = ROOT / "ml" / "outputs" / "feature_candidates" / "candidate_features_per_episode.csv"
PROP = ROOT / "ml" / "outputs" / "proposed_features" / "proposed_features_per_episode.csv"
RANDOM_STATE = 42

PHYSICS_POOL = [
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

# Broader physics-ish pool: phase mechanism keywords (still smaller than 267)
PHYSICS_BROAD_RE = re.compile(
    r"(track_err|joint_speed|speed|acc_mag|jerk|closure|tight|tilt|sparc|"
    r"open_dur|retreat_burst|vy_mean|vz_mean|duration)"
)


def family_key(name: str) -> str:
    return re.sub(r"_(mean|std|rms|maxabs|min|median)$", "", name)


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


def make_svm():
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "clf",
                SVC(
                    C=3.0,
                    gamma="scale",
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def abs_d_on_train(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([cohens_d_abs(X[:, j], y) for j in range(X.shape[1])])


def corr_matrix(X: np.ndarray) -> np.ndarray:
    Xc = X.copy()
    for j in range(Xc.shape[1]):
        col = Xc[:, j]
        m = np.isfinite(col)
        Xc[:, j] = np.where(m, col, np.nanmedian(col[m]) if m.any() else 0.0)
    return np.corrcoef(Xc.T)


def select_topk(names: list[str], d: np.ndarray, k: int) -> list[int]:
    order = np.argsort(d)[::-1][:k]
    return order.tolist()


def select_topk_corr_prune(names: list[str], X: np.ndarray, d: np.ndarray, k: int, thr: float) -> list[int]:
    order = np.argsort(d)[::-1]
    corr = corr_matrix(X)
    kept: list[int] = []
    for j in order:
        if len(kept) >= k:
            break
        ok = True
        for i in kept:
            r = corr[j, i]
            if np.isfinite(r) and abs(r) >= thr:
                ok = False
                break
        if ok:
            kept.append(int(j))
    return kept


def select_family_cap(names: list[str], d: np.ndarray, k_total: int, per_family: int) -> list[int]:
    order = np.argsort(d)[::-1]
    counts: dict[str, int] = defaultdict(int)
    kept: list[int] = []
    for j in order:
        fam = family_key(names[j])
        if counts[fam] >= per_family:
            continue
        kept.append(int(j))
        counts[fam] += 1
        if len(kept) >= k_total:
            break
    return kept


def jaccard_mean(sets: list[set[str]]) -> float:
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            vals.append(len(a & b) / max(1, len(a | b)))
    return float(np.mean(vals)) if vals else 0.0


def evaluate_policy(name: str, pool_idx: np.ndarray, col_names: list[str], X_full: np.ndarray, y: np.ndarray, policy: str, k: int):
    """Run honest 5-fold under a selection policy on a fixed candidate pool."""
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.zeros_like(y)
    chosen_sets: list[set[str]] = []
    n_feats = []

    pool_names = [col_names[i] for i in pool_idx]
    Xp = X_full[:, pool_idx]

    for tr, te in cv.split(Xp, y):
        Xtr = Xp[tr]
        d = abs_d_on_train(Xtr, y[tr])

        if policy == "fixed_all":
            idx_local = list(range(Xp.shape[1]))
        elif policy == "topk":
            idx_local = select_topk(pool_names, d, k=min(k, Xp.shape[1]))
        elif policy == "topk_corr085":
            idx_local = select_topk_corr_prune(pool_names, Xtr, d, k=min(k, Xp.shape[1]), thr=0.85)
        elif policy == "topk_corr070":
            idx_local = select_topk_corr_prune(pool_names, Xtr, d, k=min(k, Xp.shape[1]), thr=0.70)
        elif policy == "family_cap1":
            idx_local = select_family_cap(pool_names, d, k_total=min(k, Xp.shape[1]), per_family=1)
        else:
            raise ValueError(policy)

        feats = {pool_names[j] for j in idx_local}
        chosen_sets.append(feats)
        n_feats.append(len(idx_local))

        m = clone(make_svm())
        m.fit(Xtr[:, idx_local], y[tr])
        y_pred[te] = m.predict(Xp[te][:, idx_local])

    return {
        "policy": name,
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "mean_n_features": float(np.mean(n_feats)),
        "feature_jaccard": jaccard_mean(chosen_sets),
        "fold_feature_sets": [sorted(s) for s in chosen_sets],
    }


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
        if (c.startswith("cand_") or c.startswith("prop_"))
        and c not in ("prop_dtw_to_success", "prop_release_pose_dev", "prop_tilt_dev_to_success")
    ]
    universe = list(dict.fromkeys(bank + eng))
    universe = [c for c in universe if c in df.columns]
    X = df[universe].to_numpy(float)
    name_to_i = {c: i for i, c in enumerate(universe)}

    phys = [c for c in PHYSICS_POOL if c in name_to_i]
    phys_idx = np.array([name_to_i[c] for c in phys], dtype=int)

    broad = [
        c
        for c in universe
        if PHYSICS_BROAD_RE.search(c) and not c.startswith("global_")
    ]
    # keep breadth manageable
    broad_idx = np.array([name_to_i[c] for c in broad], dtype=int)
    all_idx = np.arange(len(universe), dtype=int)

    # also: phase-prefix constrained (drop pure globals)
    phase_only = [c for c in universe if not c.startswith("global_")]
    phase_idx = np.array([name_to_i[c] for c in phase_only], dtype=int)

    specs = [
        ("honest_top25_full267", all_idx, "topk", 25),
        ("honest_top15_full267", all_idx, "topk", 15),
        ("honest_top10_full267", all_idx, "topk", 10),
        ("honest_top25_corr085_full", all_idx, "topk_corr085", 25),
        ("honest_top15_family1_full", all_idx, "family_cap1", 15),
        ("honest_top15_phase_only", phase_idx, "topk", 15),
        ("honest_top15_physics_broad", broad_idx, "topk", 15),
        ("honest_top15_broad_corr085", broad_idx, "topk_corr085", 15),
        ("honest_top12_physics17", phys_idx, "topk", 12),
        ("honest_top10_physics17", phys_idx, "topk", 10),
        ("honest_corr085_physics17", phys_idx, "topk_corr085", 12),
        ("honest_family1_physics17", phys_idx, "family_cap1", 12),
        ("fixed_physics17", phys_idx, "fixed_all", 17),
    ]

    rows = []
    print("=== Honest CV with selection constraints ===")
    print(f"universe={len(universe)}  physics17={len(phys)}  broad={len(broad)}  phase_only={len(phase_only)}")
    for name, pool, policy, k in specs:
        r = evaluate_policy(name, pool, universe, X, y, policy, k)
        rows.append({k2: v for k2, v in r.items() if k2 != "fold_feature_sets"})
        print(
            f"{name:36s} acc={r['accuracy']:.3f}  f1={r['f1']:.3f}  "
            f"n≈{r['mean_n_features']:.1f}  jaccard={r['feature_jaccard']:.2f}"
        )
        # save fold sets for key policies
        if name in (
            "honest_top25_full267",
            "honest_top15_physics_broad",
            "honest_top12_physics17",
            "fixed_physics17",
        ):
            (OUT / f"fold_sets_{name}.json").write_text(
                json.dumps(r["fold_feature_sets"], indent=2), encoding="utf-8"
            )

    res = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    res.to_csv(OUT / "honest_constrained_results.csv", index=False)

    # why-low diagnostic: how much do selected sets reshuffle + how often physics feats win
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    phys_hit = []
    clone_rate = []
    for tr, _ in cv.split(X, y):
        d = abs_d_on_train(X[tr], y[tr])
        top = np.argsort(d)[::-1][:25]
        names = [universe[j] for j in top]
        phys_hit.append(sum(1 for n in names if n in PHYSICS_POOL) / 25)
        # clone: same family appearing >1
        fams = [family_key(n) for n in names]
        clone_rate.append(1.0 - len(set(fams)) / len(fams))
    diag = {
        "mean_physics_in_top25": float(np.mean(phys_hit)),
        "mean_family_clone_frac_in_top25": float(np.mean(clone_rate)),
        "n_universe": len(universe),
        "n_physics17": len(phys),
        "n_broad": len(broad),
        "best_policy": res.iloc[0]["policy"],
        "best_acc": float(res.iloc[0]["accuracy"]),
        "fixed_physics_acc": float(res.loc[res["policy"] == "fixed_physics17", "accuracy"].iloc[0]),
        "baseline_honest_top25_acc": float(
            res.loc[res["policy"] == "honest_top25_full267", "accuracy"].iloc[0]
        ),
        "interpretation": [
            "Honest is lower because each fold re-picks noisy top-|d| from a huge pool;",
            "selected features reshuffle (low Jaccard) so the test fold sees unstable rules.",
            "Constraining the pool (physics / family / corr) raises honest accuracy by",
            "reducing variance of which features enter the model across folds.",
            "Fixed physics17 is an upper reference for a pre-specified constrained set;",
            "it is not fully honest if the list was tuned on the same 48 episodes.",
        ],
    }
    (OUT / "honest_constrained_summary.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = res.sort_values("accuracy")
    colors = [
        "#2a9d8f" if "physics" in p or p.startswith("fixed") else "#e76f51"
        for p in plot_df["policy"]
    ]
    ax.barh(plot_df["policy"], plot_df["accuracy"], color=colors)
    ax.axvline(diag["baseline_honest_top25_acc"], color="gray", ls="--", label="honest top25 full")
    ax.set_xlabel("Honest 5-fold accuracy")
    ax.set_title("Constrained vs unconstrained fold-wise selection")
    ax.set_xlim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "honest_constrained_comparison.png", dpi=140)
    plt.close(fig)

    md = [
        "# Honest CV with selection constraints",
        "",
        f"- Baseline honest top25 from full pool: **{diag['baseline_honest_top25_acc']:.3f}**",
        f"- Best constrained policy: **{diag['best_policy']}** = **{diag['best_acc']:.3f}**",
        f"- Fixed physics17 (pre-specified): **{diag['fixed_physics_acc']:.3f}**",
        f"- Mean fraction of physics17 inside unconstrained fold top25: **{diag['mean_physics_in_top25']:.2f}**",
        f"- Mean family-clone fraction in fold top25: **{diag['mean_family_clone_frac_in_top25']:.2f}**",
        "",
        "## Ranking",
        "",
        res.to_string(index=False),
        "",
    ]
    (OUT / "honest_constrained_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\nWrote", OUT)


if __name__ == "__main__":
    main()
