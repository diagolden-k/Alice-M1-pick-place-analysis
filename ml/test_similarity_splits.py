"""Split episodes by similarity: similar-within vs dissimilar-across vs mixed.

Similarity is cosine distance in standardized physics-feature space (pool40),
clustered with agglomerative clustering. Protocols:

1. mixed_stratified     — usual StratifiedKFold (dissimilar ones mixed in folds)
2. similar_within       — train/test only inside each similarity cluster
3. dissimilar_groupk    — GroupKFold by cluster (test cluster unseen in train)
4. similar_paired_hold  — for each test ep, train on its k nearest same-label-balanced neighbors
5. dissimilar_paired    — for each test ep, train on its k farthest episodes (stratified subsample)

Reports SVM/LogReg accuracy and, for fold protocols, feature Jaccard under honest top-k.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, pairwise_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path("/workspace")
OUT = ROOT / "ml" / "outputs" / "similarity_splits"
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


def make_pipe(kind: str = "svm"):
    if kind == "svm":
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


def corr_prune_idx(X: np.ndarray, d: np.ndarray, k: int, thr: float = 0.70) -> list[int]:
    Xc = impute_matrix(X)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(Xc.T)
    corr = np.nan_to_num(corr, nan=0.0)
    kept: list[int] = []
    for j in np.argsort(d)[::-1]:
        if len(kept) >= k:
            break
        if any(abs(corr[j, i]) >= thr for i in kept):
            continue
        kept.append(int(j))
    return kept


def eval_splits(
    name: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
    X: np.ndarray,
    y: np.ndarray,
    feat_mode: str,
    pool_X: np.ndarray | None = None,
    k: int = 15,
) -> dict:
    """feat_mode: fixed | honest_corr070"""
    y_pred = np.full_like(y, fill_value=-1)
    fold_acc = []
    fold_meta = []
    chosen = []

    for fold_i, (tr, te) in enumerate(splits):
        if len(np.unique(y[tr])) < 2 or len(te) == 0:
            continue
        if feat_mode == "fixed":
            Xtr, Xte = X[tr], X[te]
            n_feat = X.shape[1]
        elif feat_mode == "honest_corr070":
            assert pool_X is not None
            d = np.array([cohens_d_abs(pool_X[tr, j], y[tr]) for j in range(pool_X.shape[1])])
            idx = corr_prune_idx(pool_X[tr], d, k=min(k, pool_X.shape[1]), thr=0.70)
            chosen.append(set(idx))
            Xtr, Xte = pool_X[tr][:, idx], pool_X[te][:, idx]
            n_feat = len(idx)
        else:
            raise ValueError(feat_mode)

        m = clone(make_pipe("svm"))
        m.fit(Xtr, y[tr])
        pred = m.predict(Xte)
        y_pred[te] = pred
        acc = float(accuracy_score(y[te], pred))
        fold_acc.append(acc)
        fold_meta.append(
            {
                "fold": fold_i,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "n_feat": int(n_feat),
                "acc": acc,
                "train_pos": int((y[tr] == 1).sum()),
                "test_pos": int((y[te] == 1).sum()),
            }
        )

    mask = y_pred >= 0
    if mask.sum() == 0:
        return {
            "protocol": name,
            "feat_mode": feat_mode,
            "accuracy": np.nan,
            "f1": np.nan,
            "n_scored": 0,
            "mean_fold_acc": np.nan,
            "std_fold_acc": np.nan,
            "feature_jaccard": np.nan,
            "folds": fold_meta,
        }

    jacc = np.nan
    if chosen:
        vals = []
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                a, b = chosen[i], chosen[j]
                vals.append(len(a & b) / max(1, len(a | b)))
        jacc = float(np.mean(vals)) if vals else np.nan

    return {
        "protocol": name,
        "feat_mode": feat_mode,
        "accuracy": float(accuracy_score(y[mask], y_pred[mask])),
        "f1": float(f1_score(y[mask], y_pred[mask])) if len(np.unique(y[mask])) > 1 else np.nan,
        "n_scored": int(mask.sum()),
        "mean_fold_acc": float(np.mean(fold_acc)) if fold_acc else np.nan,
        "std_fold_acc": float(np.std(fold_acc)) if fold_acc else np.nan,
        "feature_jaccard": jacc,
        "folds": fold_meta,
    }


def mean_pairwise_dist(D: np.ndarray, idx: np.ndarray) -> float:
    if len(idx) < 2:
        return 0.0
    sub = D[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), k=1)
    return float(sub[iu].mean())


def cross_mean_dist(D: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(D[np.ix_(a, b)].mean())


def build_similar_within_splits(labels_cluster: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inside each cluster: stratified 2-fold if possible, else LOO-like holdouts."""
    splits = []
    for c in sorted(np.unique(labels_cluster)):
        idx = np.where(labels_cluster == c)[0]
        if len(idx) < 4 or len(np.unique(y[idx])) < 2:
            continue
        # local stratified folds
        n_splits = 2 if len(idx) < 10 else 3
        # need enough per class
        yc = y[idx]
        if (yc == 0).sum() < n_splits or (yc == 1).sum() < n_splits:
            n_splits = 2
        if (yc == 0).sum() < n_splits or (yc == 1).sum() < n_splits:
            # fall back: leave-one-out but only when both classes remain in train
            for i in range(len(idx)):
                te = np.array([idx[i]])
                tr = np.array([j for j in idx if j != idx[i]])
                if len(np.unique(y[tr])) < 2:
                    continue
                splits.append((tr, te))
            continue
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
        for tr_l, te_l in cv.split(idx, yc):
            splits.append((idx[tr_l], idx[te_l]))
    return splits


def build_groupk_splits(labels_cluster: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = []
    for c in sorted(np.unique(labels_cluster)):
        te = np.where(labels_cluster == c)[0]
        tr = np.where(labels_cluster != c)[0]
        if len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        splits.append((tr, te))
    return splits


def build_neighbor_splits(
    D: np.ndarray,
    y: np.ndarray,
    mode: str,
    k_train: int = 24,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """LOO-style: train on nearest (similar) or farthest (dissimilar) k episodes."""
    n = len(y)
    splits = []
    for i in range(n):
        order = np.argsort(D[i])  # ascending distance
        order = order[order != i]
        if mode == "similar":
            # take nearest, but keep class balance roughly by quota
            cand = order
        else:
            cand = order[::-1]  # farthest first

        # greedily pick up to k_train with both classes present
        chosen = []
        n0 = n1 = 0
        target0 = k_train // 2
        target1 = k_train - target0
        for j in cand:
            if y[j] == 0 and n0 < target0:
                chosen.append(j)
                n0 += 1
            elif y[j] == 1 and n1 < target1:
                chosen.append(j)
                n1 += 1
            if len(chosen) >= k_train:
                break
        # fill remainder if needed
        if len(chosen) < k_train:
            for j in cand:
                if j not in chosen:
                    chosen.append(j)
                if len(chosen) >= k_train:
                    break
        tr = np.array(chosen, dtype=int)
        if len(np.unique(y[tr])) < 2:
            continue
        splits.append((tr, np.array([i], dtype=int)))
    return splits


def main():
    ep = pd.read_csv(EP)
    cand = pd.read_csv(CAND)
    prop = pd.read_csv(PROP)
    df = ep.merge(cand.drop(columns=["label"], errors="ignore"), on="episode")
    df = df.merge(prop.drop(columns=["label"], errors="ignore"), on="episode")
    y = df["label"].to_numpy(int)
    episodes = df["episode"].to_numpy(int)

    pools = json.loads(POOL_DEF.read_text()) if POOL_DEF.exists() else {}
    pool40 = [c for c in pools.get("pool40", PHYSICS17) if c in df.columns]
    pool17 = [c for c in PHYSICS17 if c in df.columns]

    # Similarity space = pool40 (mechanism features), not labels
    Xs = impute_matrix(df[pool40].to_numpy(float))
    Xs = StandardScaler().fit_transform(Xs)
    D = pairwise_distances(Xs, metric="cosine")

    # Cluster into similar groups (choose k with both classes when possible)
    n_clusters = 4
    clust = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    labels_c = clust.fit_predict(D)

    cluster_table = []
    for c in range(n_clusters):
        idx = np.where(labels_c == c)[0]
        cluster_table.append(
            {
                "cluster": int(c),
                "n": int(len(idx)),
                "n_success": int((y[idx] == 1).sum()),
                "n_fail": int((y[idx] == 0).sum()),
                "mean_within_cosine": mean_pairwise_dist(D, idx),
                "episodes": ",".join(map(str, episodes[idx].tolist())),
            }
        )
    ct = pd.DataFrame(cluster_table)
    ct.to_csv(OUT / "similarity_clusters.csv", index=False)

    # Diagnostics: within vs between cluster distance
    within = []
    between = []
    for c in range(n_clusters):
        idx = np.where(labels_c == c)[0]
        within.append(mean_pairwise_dist(D, idx))
        for c2 in range(c + 1, n_clusters):
            jdx = np.where(labels_c == c2)[0]
            between.append(cross_mean_dist(D, idx, jdx))
    sim_diag = {
        "n_clusters": n_clusters,
        "mean_within_cosine": float(np.mean(within)),
        "mean_between_cosine": float(np.mean(between)),
        "separation_ratio_between_over_within": float(np.mean(between) / max(1e-9, np.mean(within))),
        "cluster_sizes": ct[["cluster", "n", "n_success", "n_fail", "mean_within_cosine"]].to_dict("records"),
    }

    # Build protocols
    mixed_splits = list(StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(np.zeros(len(y)), y))
    similar_within = build_similar_within_splits(labels_c, y)
    dissimilar_group = build_groupk_splits(labels_c, y)
    similar_nn = build_neighbor_splits(D, y, mode="similar", k_train=24)
    dissimilar_nn = build_neighbor_splits(D, y, mode="dissimilar", k_train=24)

    # Split diversity diagnostics
    def split_diversity(splits):
        rows = []
        for i, (tr, te) in enumerate(splits):
            rows.append(
                {
                    "fold": i,
                    "train_within": mean_pairwise_dist(D, tr),
                    "test_within": mean_pairwise_dist(D, te) if len(te) > 1 else 0.0,
                    "train_test_cross": cross_mean_dist(D, tr, te),
                    "n_train": len(tr),
                    "n_test": len(te),
                }
            )
        return pd.DataFrame(rows)

    div_rows = []
    for pname, splits in [
        ("mixed_stratified", mixed_splits),
        ("similar_within", similar_within),
        ("dissimilar_groupk", dissimilar_group),
        ("similar_nn_train", similar_nn),
        ("dissimilar_nn_train", dissimilar_nn),
    ]:
        ddf = split_diversity(splits)
        ddf["protocol"] = pname
        div_rows.append(ddf)
        print(
            f"diversity {pname:22s} train_within={ddf['train_within'].mean():.3f}  "
            f"cross={ddf['train_test_cross'].mean():.3f}  n_folds={len(ddf)}"
        )
    div = pd.concat(div_rows, ignore_index=True)
    div.to_csv(OUT / "split_diversity.csv", index=False)

    X17 = impute_matrix(df[pool17].to_numpy(float))
    X40 = impute_matrix(df[pool40].to_numpy(float))

    protocols = [
        ("mixed_stratified", mixed_splits),
        ("similar_within", similar_within),
        ("dissimilar_groupk", dissimilar_group),
        ("similar_nn_train", similar_nn),
        ("dissimilar_nn_train", dissimilar_nn),
    ]

    results = []
    print("\n=== Accuracy by split protocol ===")
    for pname, splits in protocols:
        for feat_mode, Xuse, pool in [
            ("fixed_physics17", X17, None),
            ("honest_corr070_pool40", None, X40),
        ]:
            if feat_mode.startswith("fixed"):
                r = eval_splits(pname, splits, Xuse, y, "fixed")
            else:
                r = eval_splits(pname, splits, X17, y, "honest_corr070", pool_X=pool, k=15)
            r["feat_mode"] = feat_mode
            results.append({k: v for k, v in r.items() if k != "folds"})
            (OUT / f"folds_{pname}_{feat_mode}.json").write_text(json.dumps(r["folds"], indent=2), encoding="utf-8")
            print(
                f"{pname:22s} {feat_mode:24s}  acc={r['accuracy']:.3f}  "
                f"fold_mean={r['mean_fold_acc']:.3f}±{r['std_fold_acc']:.3f}  "
                f"n={r['n_scored']}  jac={r['feature_jaccard']}"
            )

    res = pd.DataFrame(results).sort_values(["feat_mode", "accuracy"], ascending=[True, False])
    res.to_csv(OUT / "similarity_split_results.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mode in zip(axes, ["fixed_physics17", "honest_corr070_pool40"]):
        sub = res[res["feat_mode"] == mode].set_index("protocol").reindex([p for p, _ in protocols])
        colors = {
            "mixed_stratified": "#6c757d",
            "similar_within": "#2a9d8f",
            "dissimilar_groupk": "#e76f51",
            "similar_nn_train": "#2a9d8f",
            "dissimilar_nn_train": "#e76f51",
        }
        ax.bar(
            range(len(sub)),
            sub["accuracy"],
            color=[colors.get(i, "#333") for i in sub.index],
        )
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub.index, rotation=25, ha="right")
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("accuracy")
        ax.set_title(mode)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
    fig.suptitle("Similar-within vs dissimilar-across splits")
    fig.tight_layout()
    fig.savefig(OUT / "similarity_split_comparison.png", dpi=140)
    plt.close(fig)

    # 2D embedding plot of clusters
    from sklearn.decomposition import PCA

    xy = PCA(2, random_state=RANDOM_STATE).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for c in range(n_clusters):
        m = labels_c == c
        ax.scatter(xy[m, 0], xy[m, 1], s=55, label=f"c{c} (n={m.sum()})", alpha=0.85)
    for i, lab in enumerate(y):
        ax.annotate("S" if lab == 1 else "F", (xy[i, 0], xy[i, 1]), fontsize=7, alpha=0.7)
    ax.legend(fontsize=8)
    ax.set_title("Episode clusters (PCA of physics pool40)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(OUT / "episode_similarity_clusters.png", dpi=140)
    plt.close(fig)

    summary = {
        "similarity_diagnostics": sim_diag,
        "results": results,
        "interpretation": [
            "similar_within / similar_nn_train: train & test come from nearby episodes → easier if local patterns dominate.",
            "dissimilar_groupk / dissimilar_nn_train: test is far from train → stress test for generalization.",
            "mixed_stratified: random mix baseline (typical CV).",
            "If similar >> dissimilar, the model partly relies on local episode neighborhoods, not only global tip-over mechanisms.",
            "If both stay high with physics features, signal is more mechanism-stable across styles.",
        ],
    }
    (OUT / "similarity_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Similarity-based train/test splits",
        "",
        f"- within-cluster cosine={sim_diag['mean_within_cosine']:.3f}, "
        f"between={sim_diag['mean_between_cosine']:.3f}, "
        f"ratio={sim_diag['separation_ratio_between_over_within']:.2f}",
        "",
        "## Clusters",
        "",
        ct.to_string(index=False),
        "",
        "## Results",
        "",
        res.to_string(index=False),
        "",
    ]
    (OUT / "similarity_split_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\nWrote", OUT)


if __name__ == "__main__":
    main()
