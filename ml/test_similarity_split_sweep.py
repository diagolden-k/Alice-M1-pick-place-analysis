"""Broad sweep: similarity split sizes × models × feature sets.

Addresses concern that prior similar/dissimilar gaps were artifacts of tiny
clusters. Sweeps n_clusters, neighbor train sizes, CV folds, models, and
feature modes; then aggregates sensitivity analysis.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, pairwise_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT = Path("/workspace")
OUT = ROOT / "ml" / "outputs" / "similarity_split_sweep"
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

N_CLUSTERS = [2, 3, 4, 6, 8]
K_TRAINS = [12, 16, 20, 24, 30, 36]
MIXED_FOLDS = [3, 5, 6, 8]
DIST_METRICS = ["cosine", "euclidean"]


def make_model(name: str, n_train: int | None = None):
    if name == "svm_rbf":
        clf = SVC(C=3.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)
    elif name == "svm_linear":
        clf = SVC(C=1.0, kernel="linear", class_weight="balanced", random_state=RANDOM_STATE)
    elif name == "logreg":
        clf = LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)
    elif name == "ridge":
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=RANDOM_STATE)
    elif name == "knn":
        k = 5 if n_train is None else max(1, min(5, n_train))
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    elif name == "rf":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=1 if (n_train is not None and n_train < 8) else 2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    elif name == "gb":
        clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
    else:
        raise ValueError(name)
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", clf),
        ]
    )


MODELS = ["svm_rbf", "svm_linear", "logreg", "ridge", "knn", "rf", "gb"]


def impute_matrix(X: np.ndarray) -> np.ndarray:
    Xc = X.copy()
    for j in range(Xc.shape[1]):
        col = Xc[:, j]
        m = np.isfinite(col)
        Xc[:, j] = np.where(m, col, np.nanmedian(col[m]) if m.any() else 0.0)
    return Xc


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


def mean_pairwise(D: np.ndarray, idx: np.ndarray) -> float:
    if len(idx) < 2:
        return 0.0
    sub = D[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), k=1)
    return float(sub[iu].mean())


def cross_mean(D: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(D[np.ix_(a, b)].mean())


def predict_with_pipe(model_name: str, Xtr, ytr, Xte):
    m = make_model(model_name, n_train=len(ytr))
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def eval_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    model_name: str,
    feat_mode: str,
    X_fixed: np.ndarray | None,
    X_pool: np.ndarray | None,
    k_feat: int = 15,
) -> dict:
    y_pred = np.full_like(y, -1)
    fold_acc = []
    n_train_list = []
    n_test_list = []
    cross_d = []  # filled by caller optionally via meta; keep placeholders

    for tr, te in splits:
        if len(te) == 0 or len(np.unique(y[tr])) < 2 or len(tr) < 3:
            continue
        if feat_mode == "fixed17" or feat_mode == "fixed40":
            Xtr, Xte = X_fixed[tr], X_fixed[te]
        elif feat_mode == "honest40":
            d = np.array([cohens_d_abs(X_pool[tr, j], y[tr]) for j in range(X_pool.shape[1])])
            idx = corr_prune_idx(X_pool[tr], d, k=min(k_feat, X_pool.shape[1]), thr=0.70)
            Xtr, Xte = X_pool[tr][:, idx], X_pool[te][:, idx]
        else:
            raise ValueError(feat_mode)

        try:
            pred = predict_with_pipe(model_name, Xtr, y[tr], Xte)
        except Exception:
            continue
        y_pred[te] = pred
        fold_acc.append(float(accuracy_score(y[te], pred)))
        n_train_list.append(len(tr))
        n_test_list.append(len(te))

    mask = y_pred >= 0
    if mask.sum() == 0:
        return {
            "accuracy": np.nan,
            "f1": np.nan,
            "n_scored": 0,
            "mean_fold_acc": np.nan,
            "std_fold_acc": np.nan,
            "mean_n_train": np.nan,
            "mean_n_test": np.nan,
            "n_folds_used": 0,
        }
    return {
        "accuracy": float(accuracy_score(y[mask], y_pred[mask])),
        "f1": float(f1_score(y[mask], y_pred[mask])) if len(np.unique(y[mask])) > 1 else np.nan,
        "n_scored": int(mask.sum()),
        "mean_fold_acc": float(np.mean(fold_acc)),
        "std_fold_acc": float(np.std(fold_acc)),
        "mean_n_train": float(np.mean(n_train_list)),
        "mean_n_test": float(np.mean(n_test_list)),
        "n_folds_used": int(len(fold_acc)),
    }


def build_similar_within(labels_c: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = []
    for c in sorted(np.unique(labels_c)):
        idx = np.where(labels_c == c)[0]
        if len(idx) < 4 or len(np.unique(y[idx])) < 2:
            continue
        yc = y[idx]
        n_splits = 3 if ((yc == 0).sum() >= 3 and (yc == 1).sum() >= 3 and len(idx) >= 9) else 2
        if (yc == 0).sum() < n_splits or (yc == 1).sum() < n_splits:
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


def build_groupk(labels_c: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = []
    for c in sorted(np.unique(labels_c)):
        te = np.where(labels_c == c)[0]
        tr = np.where(labels_c != c)[0]
        if len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        # require train reasonably large
        if len(tr) < 12:
            continue
        splits.append((tr, te))
    return splits


def build_nn(D: np.ndarray, y: np.ndarray, mode: str, k_train: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(y)
    splits = []
    k_train = min(k_train, n - 1)
    for i in range(n):
        order = np.argsort(D[i])
        order = order[order != i]
        cand = order if mode == "similar" else order[::-1]
        chosen = []
        n0 = n1 = 0
        t0 = k_train // 2
        t1 = k_train - t0
        for j in cand:
            if y[j] == 0 and n0 < t0:
                chosen.append(j)
                n0 += 1
            elif y[j] == 1 and n1 < t1:
                chosen.append(j)
                n1 += 1
            if len(chosen) >= k_train:
                break
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


def split_diversity(splits, D):
    rows = []
    for tr, te in splits:
        rows.append(
            {
                "train_within": mean_pairwise(D, tr),
                "test_within": mean_pairwise(D, te) if len(te) > 1 else 0.0,
                "cross": cross_mean(D, tr, te),
                "n_train": len(tr),
                "n_test": len(te),
            }
        )
    if not rows:
        return {"train_within": np.nan, "cross": np.nan, "mean_n_train": np.nan, "mean_n_test": np.nan, "n_folds": 0}
    df = pd.DataFrame(rows)
    return {
        "train_within": float(df["train_within"].mean()),
        "cross": float(df["cross"].mean()),
        "mean_n_train": float(df["n_train"].mean()),
        "mean_n_test": float(df["n_test"].mean()),
        "n_folds": int(len(df)),
    }


def main():
    ep = pd.read_csv(EP)
    cand = pd.read_csv(CAND)
    prop = pd.read_csv(PROP)
    df = ep.merge(cand.drop(columns=["label"], errors="ignore"), on="episode")
    df = df.merge(prop.drop(columns=["label"], errors="ignore"), on="episode")
    y = df["label"].to_numpy(int)

    pools = json.loads(POOL_DEF.read_text()) if POOL_DEF.exists() else {}
    pool40 = [c for c in pools.get("pool40", PHYSICS17) if c in df.columns]
    pool17 = [c for c in PHYSICS17 if c in df.columns]

    X17 = impute_matrix(df[pool17].to_numpy(float))
    X40 = impute_matrix(df[pool40].to_numpy(float))
    feat_map = {
        "fixed17": (X17, None),
        "fixed40": (X40, None),
        "honest40": (None, X40),
    }

    # Precompute distance / clusters per metric
    cluster_cache = {}
    dist_cache = {}
    for metric in DIST_METRICS:
        Xs = StandardScaler().fit_transform(X40)
        D = pairwise_distances(Xs, metric=metric)
        dist_cache[metric] = D
        for nc in N_CLUSTERS:
            clust = AgglomerativeClustering(n_clusters=nc, metric="precomputed", linkage="average")
            labels = clust.fit_predict(D)
            # cluster size stats
            sizes = [int((labels == c).sum()) for c in range(nc)]
            cluster_cache[(metric, nc)] = {
                "labels": labels,
                "sizes": sizes,
                "min_size": min(sizes),
                "median_size": float(np.median(sizes)),
                "max_size": max(sizes),
            }

    rows = []
    n_jobs_done = 0

    def run_one(protocol, param, metric, splits, D, model, feat_mode):
        nonlocal n_jobs_done
        Xf, Xp = feat_map[feat_mode]
        metrics = eval_splits(splits, y, model, feat_mode, Xf, Xp, k_feat=15)
        div = split_diversity(splits, D)
        row = {
            "protocol": protocol,
            "param": param,
            "dist_metric": metric,
            "model": model,
            "feat_mode": feat_mode,
            **metrics,
            "div_train_within": div["train_within"],
            "div_cross": div["cross"],
            "div_mean_n_train": div["mean_n_train"],
            "div_mean_n_test": div["mean_n_test"],
            "div_n_folds": div["n_folds"],
        }
        if protocol in ("similar_within", "dissimilar_groupk"):
            # param encodes nc
            nc = int(str(param).split("=")[1])
            info = cluster_cache[(metric, nc)]
            row["cluster_min"] = info["min_size"]
            row["cluster_median"] = info["median_size"]
            row["cluster_max"] = info["max_size"]
        rows.append(row)
        n_jobs_done += 1
        if n_jobs_done % 100 == 0:
            print(f"  ... {n_jobs_done} combos")

    print("=== Similarity split sweep ===")
    # 1) Mixed stratified
    for n_folds in MIXED_FOLDS:
        # skip if class counts insufficient
        if (y == 0).sum() < n_folds or (y == 1).sum() < n_folds:
            continue
        splits = list(StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE).split(np.zeros(len(y)), y))
        for metric in DIST_METRICS[:1]:  # diversity metric only once for mixed
            D = dist_cache[metric]
            for model in MODELS:
                for feat_mode in feat_map:
                    run_one("mixed_stratified", f"folds={n_folds}", metric, splits, D, model, feat_mode)

    # 2) Cluster protocols across n_clusters × metrics
    for metric in DIST_METRICS:
        D = dist_cache[metric]
        for nc in N_CLUSTERS:
            labels = cluster_cache[(metric, nc)]["labels"]
            within = build_similar_within(labels, y)
            groupk = build_groupk(labels, y)
            for protocol, splits in [("similar_within", within), ("dissimilar_groupk", groupk)]:
                if not splits:
                    continue
                for model in MODELS:
                    for feat_mode in feat_map:
                        run_one(protocol, f"n_clusters={nc}", metric, splits, D, model, feat_mode)

    # 3) Neighbor protocols across k_train × metrics
    for metric in DIST_METRICS:
        D = dist_cache[metric]
        for k_train in K_TRAINS:
            for mode, protocol in [("similar", "similar_nn"), ("dissimilar", "dissimilar_nn")]:
                splits = build_nn(D, y, mode=mode, k_train=k_train)
                for model in MODELS:
                    for feat_mode in feat_map:
                        run_one(protocol, f"k_train={k_train}", metric, splits, D, model, feat_mode)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "sweep_raw_results.csv", index=False)
    print(f"Total combos: {len(res)}")

    # ---------- Aggregations ----------
    # A) protocol × model summary (mean over params/metrics/feats)
    by_pm = (
        res.groupby(["protocol", "model"], as_index=False)
        .agg(
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            acc_q25=("accuracy", lambda s: s.quantile(0.25)),
            acc_q75=("accuracy", lambda s: s.quantile(0.75)),
            n=("accuracy", "count"),
            mean_n_train=("mean_n_train", "mean"),
            mean_n_test=("mean_n_test", "mean"),
        )
        .sort_values(["protocol", "acc_mean"], ascending=[True, False])
    )
    by_pm.to_csv(OUT / "agg_protocol_model.csv", index=False)

    # B) protocol × feat
    by_pf = (
        res.groupby(["protocol", "feat_mode"], as_index=False)
        .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"), n=("accuracy", "count"))
        .sort_values(["protocol", "acc_mean"], ascending=[True, False])
    )
    by_pf.to_csv(OUT / "agg_protocol_feat.csv", index=False)

    # C) sensitivity to n_clusters for groupk / within
    clus = res[res["protocol"].isin(["similar_within", "dissimilar_groupk"])].copy()
    clus["n_clusters"] = clus["param"].str.extract(r"(\d+)").astype(int)
    by_nc = (
        clus.groupby(["protocol", "n_clusters", "dist_metric"], as_index=False)
        .agg(
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            mean_n_test=("mean_n_test", "mean"),
            cluster_min=("cluster_min", "mean"),
            cluster_median=("cluster_median", "mean"),
            n=("accuracy", "count"),
        )
        .sort_values(["protocol", "dist_metric", "n_clusters"])
    )
    by_nc.to_csv(OUT / "agg_n_clusters.csv", index=False)

    # D) sensitivity to k_train
    nn = res[res["protocol"].isin(["similar_nn", "dissimilar_nn"])].copy()
    nn["k_train"] = nn["param"].str.extract(r"(\d+)").astype(int)
    by_k = (
        nn.groupby(["protocol", "k_train", "dist_metric"], as_index=False)
        .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"), n=("accuracy", "count"))
        .sort_values(["protocol", "dist_metric", "k_train"])
    )
    by_k.to_csv(OUT / "agg_k_train.csv", index=False)

    # E) mixed folds sensitivity
    mx = res[res["protocol"] == "mixed_stratified"].copy()
    mx["n_folds"] = mx["param"].str.extract(r"(\d+)").astype(int)
    by_f = (
        mx.groupby(["n_folds", "model", "feat_mode"], as_index=False)
        .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"))
        .sort_values(["n_folds", "acc_mean"], ascending=[True, False])
    )
    by_f.to_csv(OUT / "agg_mixed_folds.csv", index=False)

    # F) similar vs dissimilar gap (paired by model/feat/metric/comparable param)
    gap_rows = []
    # cluster-based gaps
    for (model, feat, metric, nc), g in clus.groupby(["model", "feat_mode", "dist_metric", "n_clusters"]):
        sim = g[g["protocol"] == "similar_within"]["accuracy"]
        dis = g[g["protocol"] == "dissimilar_groupk"]["accuracy"]
        if len(sim) and len(dis):
            gap_rows.append(
                {
                    "pair_type": "cluster",
                    "model": model,
                    "feat_mode": feat,
                    "dist_metric": metric,
                    "param": f"n_clusters={nc}",
                    "similar_acc": float(sim.iloc[0]),
                    "dissimilar_acc": float(dis.iloc[0]),
                    "gap_sim_minus_dis": float(sim.iloc[0] - dis.iloc[0]),
                    "mean_n_test_dis": float(g[g["protocol"] == "dissimilar_groupk"]["mean_n_test"].iloc[0]),
                    "cluster_min": float(g[g["protocol"] == "dissimilar_groupk"]["cluster_min"].iloc[0]),
                }
            )
    # nn gaps
    for (model, feat, metric, k), g in nn.groupby(["model", "feat_mode", "dist_metric", "k_train"]):
        sim = g[g["protocol"] == "similar_nn"]["accuracy"]
        dis = g[g["protocol"] == "dissimilar_nn"]["accuracy"]
        if len(sim) and len(dis):
            gap_rows.append(
                {
                    "pair_type": "neighbor",
                    "model": model,
                    "feat_mode": feat,
                    "dist_metric": metric,
                    "param": f"k_train={k}",
                    "similar_acc": float(sim.iloc[0]),
                    "dissimilar_acc": float(dis.iloc[0]),
                    "gap_sim_minus_dis": float(sim.iloc[0] - dis.iloc[0]),
                    "mean_n_test_dis": 1.0,
                    "cluster_min": np.nan,
                }
            )
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(OUT / "similar_vs_dissimilar_gaps.csv", index=False)

    gap_sum = (
        gaps.groupby(["pair_type", "model"], as_index=False)
        .agg(
            gap_mean=("gap_sim_minus_dis", "mean"),
            gap_std=("gap_sim_minus_dis", "std"),
            gap_q25=("gap_sim_minus_dis", lambda s: s.quantile(0.25)),
            gap_q75=("gap_sim_minus_dis", lambda s: s.quantile(0.75)),
            sim_mean=("similar_acc", "mean"),
            dis_mean=("dissimilar_acc", "mean"),
            n=("gap_sim_minus_dis", "count"),
        )
        .sort_values(["pair_type", "gap_mean"], ascending=[True, False])
    )
    gap_sum.to_csv(OUT / "gap_summary_by_model.csv", index=False)

    # Filter: only "large enough" dissimilar tests (cluster_min >= 6 or n_clusters<=4)
    large = gaps[(gaps["pair_type"] == "neighbor") | (gaps["cluster_min"] >= 6) | (gaps["param"].isin(["n_clusters=2", "n_clusters=3", "n_clusters=4"]))]
    # Better: for cluster, require mean_n_test_dis >= 8
    large_cluster = gaps[(gaps["pair_type"] == "cluster") & (gaps["mean_n_test_dis"] >= 8)]
    large_nn = gaps[(gaps["pair_type"] == "neighbor") & (gaps["param"].isin([f"k_train={k}" for k in [20, 24, 30, 36]]))]
    large_all = pd.concat([large_cluster, large_nn], ignore_index=True)
    large_sum = (
        large_all.groupby("pair_type", as_index=False)
        .agg(
            gap_mean=("gap_sim_minus_dis", "mean"),
            gap_std=("gap_sim_minus_dis", "std"),
            sim_mean=("similar_acc", "mean"),
            dis_mean=("dissimilar_acc", "mean"),
            n=("gap_sim_minus_dis", "count"),
            frac_gap_positive=("gap_sim_minus_dis", lambda s: float((s > 0).mean())),
        )
    )
    large_sum.to_csv(OUT / "gap_summary_large_splits.csv", index=False)

    # Headline stats
    protocol_overall = (
        res.groupby("protocol", as_index=False)
        .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"), acc_median=("accuracy", "median"), n=("accuracy", "count"))
        .sort_values("acc_mean", ascending=False)
    )
    protocol_overall.to_csv(OUT / "agg_protocol_overall.csv", index=False)

    summary = {
        "n_combos": int(len(res)),
        "models": MODELS,
        "n_clusters_grid": N_CLUSTERS,
        "k_train_grid": K_TRAINS,
        "mixed_folds_grid": MIXED_FOLDS,
        "feat_modes": list(feat_map.keys()),
        "dist_metrics": DIST_METRICS,
        "protocol_overall": protocol_overall.to_dict("records"),
        "gap_large_splits": large_sum.to_dict("records"),
        "gap_by_model_head": gap_sum.to_dict("records"),
        "takeaway": [
            "If similar>>dissimilar persists across large n_clusters=2–4, large k_train, and many models,",
            "the gap is not just an artifact of tiny leave-one-small-cluster splits.",
            "If the gap shrinks/vanishes for large splits only, earlier drop was size-driven.",
        ],
    }
    (OUT / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---------- Plots ----------
    # 1) protocol overall boxplot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = ["similar_within", "similar_nn", "mixed_stratified", "dissimilar_nn", "dissimilar_groupk"]
    data = [res.loc[res["protocol"] == p, "accuracy"].dropna().values for p in order]
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.set_ylabel("accuracy")
    ax.set_title(f"All combos (n={len(res)}): protocol distribution")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0.3, 1.05)
    fig.tight_layout()
    fig.savefig(OUT / "plot_protocol_boxplot.png", dpi=140)
    plt.close(fig)

    # 2) n_clusters sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, metric in zip(axes, DIST_METRICS):
        sub = by_nc[by_nc["dist_metric"] == metric]
        for prot, ls in [("similar_within", "-"), ("dissimilar_groupk", "--")]:
            s = sub[sub["protocol"] == prot]
            ax.plot(s["n_clusters"], s["acc_mean"], ls + "o", label=prot)
            ax.fill_between(
                s["n_clusters"],
                s["acc_mean"] - s["acc_std"].fillna(0),
                s["acc_mean"] + s["acc_std"].fillna(0),
                alpha=0.15,
            )
        ax.set_xlabel("n_clusters")
        ax.set_title(f"metric={metric}")
        ax.legend(fontsize=8)
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("mean accuracy ± std (over models/feats)")
    fig.suptitle("Sensitivity to number of clusters")
    fig.tight_layout()
    fig.savefig(OUT / "plot_n_clusters_sensitivity.png", dpi=140)
    plt.close(fig)

    # 3) k_train sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, metric in zip(axes, DIST_METRICS):
        sub = by_k[by_k["dist_metric"] == metric]
        for prot, ls in [("similar_nn", "-"), ("dissimilar_nn", "--")]:
            s = sub[sub["protocol"] == prot]
            ax.plot(s["k_train"], s["acc_mean"], ls + "o", label=prot)
            ax.fill_between(
                s["k_train"],
                s["acc_mean"] - s["acc_std"].fillna(0),
                s["acc_mean"] + s["acc_std"].fillna(0),
                alpha=0.15,
            )
        ax.set_xlabel("k_train")
        ax.set_title(f"metric={metric}")
        ax.legend(fontsize=8)
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("mean accuracy ± std")
    fig.suptitle("Sensitivity to neighbor train size")
    fig.tight_layout()
    fig.savefig(OUT / "plot_k_train_sensitivity.png", dpi=140)
    plt.close(fig)

    # 4) gap by model
    fig, ax = plt.subplots(figsize=(9, 4.5))
    piv = gap_sum.pivot(index="model", columns="pair_type", values="gap_mean")
    piv = piv.reindex(MODELS)
    x = np.arange(len(piv))
    w = 0.35
    ax.bar(x - w / 2, piv.get("cluster", pd.Series(index=piv.index)).fillna(0), w, label="cluster gap")
    ax.bar(x + w / 2, piv.get("neighbor", pd.Series(index=piv.index)).fillna(0), w, label="neighbor gap")
    ax.axhline(0, color="gray", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=20)
    ax.set_ylabel("similar − dissimilar accuracy")
    ax.set_title("Gap by model (mean over params/feats/metrics)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "plot_gap_by_model.png", dpi=140)
    plt.close(fig)

    # 5) heatmap protocol × model
    heat = by_pm.pivot(index="model", columns="protocol", values="acc_mean").reindex(MODELS)
    heat = heat.reindex(columns=order)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(heat.to_numpy(float), aspect="auto", vmin=0.5, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.to_numpy(float)[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.75 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, label="mean acc")
    ax.set_title("Mean accuracy: model × protocol")
    fig.tight_layout()
    fig.savefig(OUT / "plot_heatmap_model_protocol.png", dpi=140)
    plt.close(fig)

    # Markdown report
    md = [
        "# Similarity-split sweep analysis",
        "",
        f"- Combos run: **{len(res)}**",
        f"- Models: {', '.join(MODELS)}",
        f"- n_clusters: {N_CLUSTERS}; k_train: {K_TRAINS}; mixed folds: {MIXED_FOLDS}",
        f"- Feature modes: fixed17, fixed40, honest40; distances: {DIST_METRICS}",
        "",
        "## Protocol overall (all combos)",
        "",
        protocol_overall.to_string(index=False),
        "",
        "## Similar−dissimilar gap (large splits only)",
        "",
        large_sum.to_string(index=False),
        "",
        "## Gap by model",
        "",
        gap_sum.to_string(index=False),
        "",
        "## n_clusters sensitivity (mean over models/feats)",
        "",
        by_nc.to_string(index=False),
        "",
        "## k_train sensitivity",
        "",
        by_k.to_string(index=False),
        "",
    ]
    (OUT / "sweep_report.md").write_text("\n".join(md), encoding="utf-8")

    print("\nProtocol overall:")
    print(protocol_overall.to_string(index=False))
    print("\nLarge-split gaps:")
    print(large_sum.to_string(index=False))
    print("\nGap by model:")
    print(gap_sum.to_string(index=False))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
