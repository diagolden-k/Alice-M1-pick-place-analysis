"""Honest selection inside expanded physics pools (dozens), not full universe.

Pools are mechanism-prefiltered tip-over candidates (~17 / ~40 / ~55), then
fold-wise top-k / corr-prune chooses inside that pool only.
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path("/workspace")
OUT = ROOT / "ml" / "outputs" / "expanded_physics_pool"
OUT.mkdir(parents=True, exist_ok=True)

EP = ROOT / "viz" / "outputs" / "episode_phase_features.csv"
CAND = ROOT / "ml" / "outputs" / "feature_candidates" / "candidate_features_per_episode.csv"
PROP = ROOT / "ml" / "outputs" / "proposed_features" / "proposed_features_per_episode.csv"
RANDOM_STATE = 42

# --- Pool definitions (mechanism-first, not |d|-first) -----------------------
# Strict 17 (previous allowlist)
POOL_17 = [
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

# Expanded ~40 total (=17 + ~23 extras): adjacent mechanism signals, 1–2 stats each
POOL_40_EXTRA = [
    "approach_track_err_rms",
    "approach_joint_speed_maxabs",
    "approach_acc_mag_mean",
    "approach_jerk_mag_rms",
    "grasp_acc_mag_rms",
    "grasp_closure_std",
    "grasp_joint_acc_mean",
    "prop_grasp_tight_mean",
    "lift_vz_mean",
    "lift_closure_std",
    "carry_right_track_err_std",
    "carry_right_speed_mean",
    "place_dz",
    "place_vz_mean",
    "prop_tilt_roll_release",
    "prop_vz_descent",
    "retreat_speed_maxabs",
    "prop_retreat_burst_vy",
    "prop_open_duration",
    "cand_post_open_joint_speed",
    "cand_post_open_vy_mean",
    "cand_release_slope",
    "cand_approach_last_track_std",
]

# Expanded ~55 total: a bit more phase coverage
POOL_55_EXTRA = [
    "approach_speed_std",
    "approach_vy_mean",
    "grasp_closure_mean",
    "grasp_speed_mean",
    "prop_finger_sync_std",
    "lift_closure_mean",
    "carry_right_acc_mag_mean",
    "place_closure_mean",
    "retreat_acc_mag_mean",
    "cand_hold_duration",
    "cand_pre_open_track_rms",
    "cand_approach_last_jerk_rms",
    "approach_joint_acc_mean",
    "grasp_jerk_mag_rms",
    "lift_track_err_std",
    "carry_right_track_err_rms",
    "place_track_err_std",
    "retreat_jerk_mag_rms",
    "cand_post_open_palm_speed",
    "cand_path_efficiency",
    "prop_hf_energy_sh_p",
]

# Broader ~70: previous wide pool extras kept for sensitivity
POOL_70_EXTRA = [
    "approach_track_err_mean",
    "grasp_track_err_std",
    "lift_speed_mean",
    "carry_right_jerk_mag_rms",
    "carry_right_joint_speed_mean",
    "place_speed_mean",
    "place_acc_mag_mean",
    "retreat_joint_speed_maxabs",
    "retreat_track_err_std",
    "cand_post_open_jerk_rms",
    "cand_z_peak_to_open",
    "cand_retreat_over_place_dur",
]


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


def make_pipe(kind: str = "svm"):
    if kind == "svm":
        clf = SVC(
            C=3.0,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    else:
        clf = LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", clf),
        ]
    )


def abs_d(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([cohens_d_abs(X[:, j], y) for j in range(X.shape[1])])


def corr_matrix(X: np.ndarray) -> np.ndarray:
    Xc = X.copy()
    for j in range(Xc.shape[1]):
        col = Xc[:, j]
        m = np.isfinite(col)
        Xc[:, j] = np.where(m, col, np.nanmedian(col[m]) if m.any() else 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(Xc.T)
    return np.nan_to_num(c, nan=0.0)


def select_topk(d: np.ndarray, k: int) -> list[int]:
    return np.argsort(d)[::-1][:k].tolist()


def select_corr_prune(X: np.ndarray, d: np.ndarray, k: int, thr: float) -> list[int]:
    order = np.argsort(d)[::-1]
    corr = corr_matrix(X)
    kept: list[int] = []
    for j in order:
        if len(kept) >= k:
            break
        if any(abs(corr[j, i]) >= thr for i in kept):
            continue
        kept.append(int(j))
    return kept


def select_family_cap(names: list[str], d: np.ndarray, k: int, per_family: int = 1) -> list[int]:
    order = np.argsort(d)[::-1]
    counts: dict[str, int] = defaultdict(int)
    kept: list[int] = []
    for j in order:
        fam = family_key(names[j])
        if counts[fam] >= per_family:
            continue
        kept.append(int(j))
        counts[fam] += 1
        if len(kept) >= k:
            break
    return kept


def jaccard_mean(sets: list[set[str]]) -> float:
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            vals.append(len(a & b) / max(1, len(a | b)))
    return float(np.mean(vals)) if vals else 0.0


def evaluate(
    label: str,
    pool_names: list[str],
    X_full: np.ndarray,
    all_names: list[str],
    y: np.ndarray,
    policy: str,
    k: int,
    model: str = "svm",
) -> dict:
    name_to_i = {c: i for i, c in enumerate(all_names)}
    present = [c for c in pool_names if c in name_to_i]
    missing = [c for c in pool_names if c not in name_to_i]
    pool_idx = np.array([name_to_i[c] for c in present], dtype=int)
    Xp = X_full[:, pool_idx]
    local_names = present

    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.zeros_like(y)
    sets: list[set[str]] = []
    n_feats = []

    for tr, te in cv.split(Xp, y):
        Xtr = Xp[tr]
        d = abs_d(Xtr, y[tr])
        if policy == "fixed_all":
            idx = list(range(Xp.shape[1]))
        elif policy == "topk":
            idx = select_topk(d, k=min(k, Xp.shape[1]))
        elif policy == "corr085":
            idx = select_corr_prune(Xtr, d, k=min(k, Xp.shape[1]), thr=0.85)
        elif policy == "corr070":
            idx = select_corr_prune(Xtr, d, k=min(k, Xp.shape[1]), thr=0.70)
        elif policy == "family1":
            idx = select_family_cap(local_names, d, k=min(k, Xp.shape[1]), per_family=1)
        else:
            raise ValueError(policy)

        sets.append({local_names[j] for j in idx})
        n_feats.append(len(idx))
        m = clone(make_pipe(model))
        m.fit(Xtr[:, idx], y[tr])
        y_pred[te] = m.predict(Xp[te][:, idx])

    return {
        "label": label,
        "pool_size": len(present),
        "pool_missing": len(missing),
        "policy": policy,
        "k": k,
        "model": model,
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "mean_n_features": float(np.mean(n_feats)),
        "feature_jaccard": jaccard_mean(sets),
        "fold_sets": [sorted(s) for s in sets],
        "pool_names": present,
    }


def unique_preserve(seq: list[str]) -> list[str]:
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
    universe = unique_preserve(bank + eng)
    X = df[universe].to_numpy(float)

    pool17 = [c for c in POOL_17 if c in universe]
    pool40 = unique_preserve(pool17 + [c for c in POOL_40_EXTRA if c in universe])
    pool55 = unique_preserve(pool40 + [c for c in POOL_55_EXTRA if c in universe])
    pool70 = unique_preserve(pool55 + [c for c in POOL_70_EXTRA if c in universe])

    # sanity: no globals in pools
    for name, pool in [("17", pool17), ("40", pool40), ("55", pool55), ("70", pool70)]:
        assert not any(c.startswith("global_") for c in pool), name

    specs = []
    # baselines
    specs.append(("full302_topk25", universe, "topk", 25))
    specs.append(("full302_corr085_15", universe, "corr085", 15))
    # fixed pools
    for pname, pool in [
        ("pool17", pool17),
        ("pool40", pool40),
        ("pool55", pool55),
        ("pool70", pool70),
    ]:
        specs.append((f"{pname}_fixed", pool, "fixed_all", len(pool)))
        for k in (10, 12, 15, 20):
            if k > len(pool):
                continue
            specs.append((f"{pname}_topk{k}", pool, "topk", k))
            specs.append((f"{pname}_corr085_{k}", pool, "corr085", k))
        specs.append((f"{pname}_family1_15", pool, "family1", min(15, len(pool))))
        specs.append((f"{pname}_corr070_15", pool, "corr070", min(15, len(pool))))

    rows = []
    fold_dump = {}
    print("=== Expanded physics-pool honest selection ===")
    print(f"pool17={len(pool17)} pool40={len(pool40)} pool55={len(pool55)} full={len(universe)}")
    for label, pool, policy, k in specs:
        r = evaluate(label, pool, X, universe, y, policy, k, model="svm")
        # also LR for key rows
        rows.append({kk: vv for kk, vv in r.items() if kk not in ("fold_sets", "pool_names")})
        print(
            f"{label:28s} pool={r['pool_size']:3d}  acc={r['accuracy']:.3f}  "
            f"f1={r['f1']:.3f}  n≈{r['mean_n_features']:.1f}  jac={r['feature_jaccard']:.2f}"
        )
        if label in (
            "pool40_topk15",
            "pool40_corr085_15",
            "pool40_corr070_15",
            "pool55_topk15",
            "pool55_corr070_15",
            "pool17_corr085_12",
            "pool70_corr070_15",
            "full302_topk25",
        ):
            fold_dump[label] = r["fold_sets"]
            # LR companion
            r_lr = evaluate(label + "_lr", pool, X, universe, y, policy, k, model="logreg")
            rows.append({kk: vv for kk, vv in r_lr.items() if kk not in ("fold_sets", "pool_names")})
            print(
                f"{label+'_lr':28s} pool={r_lr['pool_size']:3d}  acc={r_lr['accuracy']:.3f}  "
                f"f1={r_lr['f1']:.3f}  n≈{r_lr['mean_n_features']:.1f}  jac={r_lr['feature_jaccard']:.2f}"
            )

    res = pd.DataFrame(rows).sort_values(["accuracy", "feature_jaccard"], ascending=False)
    res.to_csv(OUT / "expanded_pool_results.csv", index=False)
    (OUT / "fold_sets.json").write_text(json.dumps(fold_dump, indent=2), encoding="utf-8")
    (OUT / "pool_definitions.json").write_text(
        json.dumps(
            {"pool17": pool17, "pool40": pool40, "pool55": pool55, "pool70": pool70},
            indent=2,
        ),
        encoding="utf-8",
    )

    # highlight table: best per pool size under honest (non-fixed) policies
    highlight = []
    for pool_tag, psize in [
        ("pool17", len(pool17)),
        ("pool40", len(pool40)),
        ("pool55", len(pool55)),
        ("pool70", len(pool70)),
        ("full302", len(universe)),
    ]:
        sub = res[res["label"].str.startswith(pool_tag) & ~res["label"].str.contains("fixed|_lr")]
        if sub.empty:
            continue
        best = sub.iloc[0]
        fixed = res[res["label"] == f"{pool_tag}_fixed"]
        # also report representative honest top15 / corr070_15 if present
        top15 = res[res["label"] == f"{pool_tag}_topk15"]
        corr15 = res[res["label"] == f"{pool_tag}_corr070_15"]
        highlight.append(
            {
                "pool": pool_tag,
                "pool_size": psize,
                "best_honest_policy": best["label"],
                "best_honest_acc": float(best["accuracy"]),
                "best_honest_jaccard": float(best["feature_jaccard"]),
                "topk15_acc": float(top15.iloc[0]["accuracy"]) if len(top15) else np.nan,
                "corr070_15_acc": float(corr15.iloc[0]["accuracy"]) if len(corr15) else np.nan,
                "fixed_acc": float(fixed.iloc[0]["accuracy"]) if len(fixed) else np.nan,
            }
        )
    hi = pd.DataFrame(highlight)
    hi.to_csv(OUT / "pool_size_highlight.csv", index=False)

    summary = {
        "pool_sizes": {
            "pool17": len(pool17),
            "pool40": len(pool40),
            "pool55": len(pool55),
            "pool70": len(pool70),
            "full": len(universe),
        },
        "highlight": highlight,
        "takeaway": [
            "Expanding the physics pool from 17 to ~40–55 still avoids full-universe fishing.",
            "Honest top-k / corr-prune inside the expanded pool should stay closer to pool17",
            "than to unconstrained top25, if mechanisms remain tip-over relevant.",
            "If accuracy collapses toward full302, the extras reintroduced fold-noise clones.",
        ],
    }
    (OUT / "expanded_pool_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # plot pool-size vs best honest / fixed
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(hi))
    w = 0.35
    ax.bar(x - w / 2, hi["best_honest_acc"], w, label="best honest in pool", color="#2a9d8f")
    ax.bar(x + w / 2, hi["fixed_acc"].fillna(0), w, label="fixed whole pool", color="#264653")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.pool}\n(n={int(r.pool_size)})" for r in hi.itertuples()])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("5-fold accuracy")
    ax.set_title("Honest selection inside expanded physics pools")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "expanded_pool_comparison.png", dpi=140)
    plt.close(fig)

    md = [
        "# Expanded physics-pool honest selection",
        "",
        f"- pool17={len(pool17)}, pool40={len(pool40)}, pool55={len(pool55)}, "
        f"pool70={len(pool70)}, full={len(universe)}",
        "",
        "## Highlight by pool size",
        "",
        hi.to_string(index=False),
        "",
        "## All SVM rows (sorted)",
        "",
        res[~res["label"].str.endswith("_lr")].to_string(index=False),
        "",
    ]
    (OUT / "expanded_pool_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\nHighlight:")
    print(hi.to_string(index=False))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
