"""Classification / Clustering / Quality Metric model selection pipeline.

Uses episode_phase_features.csv. With N=48, we restrict to top-|d| features,
run stratified CV model search, pick best classifier, then select an
interpretable clustering and a quality score formula.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ml" / "outputs"
FEAT = ROOT / "viz" / "outputs" / "episode_phase_features.csv"
RANK = ROOT / "viz" / "outputs" / "feature_influence_ranking.csv"

RANDOM_STATE = 42
TOP_K_FEATURES = 25
CV_SPLITS = 5


def load_xy(top_k: int = TOP_K_FEATURES):
    feat = pd.read_csv(FEAT)
    rank = pd.read_csv(RANK)
    cols = rank["feature"].head(top_k).tolist()
    # keep only columns that exist
    cols = [c for c in cols if c in feat.columns]
    X = feat[cols].to_numpy(dtype=float)
    y = feat["label"].to_numpy(dtype=int)
    episodes = feat["episode"].to_numpy(dtype=int)
    return feat, X, y, episodes, cols, rank


def classification_search(X, y, feature_names):
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    configs = [
        (
            "logreg",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            ),
            {
                "clf__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
        ),
        (
            "svm_rbf",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE)),
                ]
            ),
            {
                "clf__C": [0.1, 1.0, 3.0, 10.0],
                "clf__gamma": ["scale", 0.01, 0.03, 0.1],
            },
        ),
        (
            "svm_linear",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("clf", SVC(kernel="linear", class_weight="balanced", probability=True, random_state=RANDOM_STATE)),
                ]
            ),
            {"clf__C": [0.01, 0.1, 1.0, 3.0, 10.0]},
        ),
        (
            "rf",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            ),
            {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [3, 5, None],
                "clf__min_samples_leaf": [1, 2, 3],
            },
        ),
        (
            "extratrees",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("clf", ExtraTreesClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            ),
            {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [3, 5, None],
                "clf__min_samples_leaf": [1, 2, 3],
            },
        ),
        (
            "gb",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.03, 0.1],
                "clf__max_depth": [2, 3],
                "clf__min_samples_leaf": [2, 3],
            },
        ),
        (
            "knn",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("clf", KNeighborsClassifier()),
                ]
            ),
            {
                "clf__n_neighbors": [3, 5, 7],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        ),
    ]

    best_name, best_est, best_score = None, None, -1.0
    all_cv_rows = []

    for name, pipe, grid in configs:
        gs = GridSearchCV(
            pipe,
            grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )
        gs.fit(X, y)
        # also report accuracy of best params via CV
        y_pred = cross_val_predict(gs.best_estimator_, X, y, cv=cv)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred).tolist()
        row = {
            "model": name,
            "best_params": json.dumps(gs.best_params_),
            "cv_f1_gridbest": float(gs.best_score_),
            "oof_f1": float(f1),
            "oof_accuracy": float(acc),
            "confusion_matrix": json.dumps(cm),
            "n_candidates": int(len(gs.cv_results_["params"])),
        }
        results.append(row)
        all_cv_rows.append(row)
        print(f"[clf] {name:12s} oof_f1={f1:.3f} oof_acc={acc:.3f} params={gs.best_params_}")
        # Primary ranking key: OOF F1, then accuracy, then grid CV F1
        score = f1 + 0.01 * acc
        if score > best_score:
            best_score = score
            best_name = name
            best_est = gs.best_estimator_

    # Fit best on all data
    best_est.fit(X, y)
    y_prob = None
    if hasattr(best_est, "predict_proba"):
        y_prob = best_est.predict_proba(X)[:, 1]
    else:
        y_prob = best_est.decision_function(X)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-12)

    return pd.DataFrame(results).sort_values("oof_f1", ascending=False), best_name, best_est, y_prob


def prepare_matrix(X):
    imp = SimpleImputer(strategy="median")
    Xi = imp.fit_transform(X)
    sc = RobustScaler()
    Xs = sc.fit_transform(Xi)
    return Xs, imp, sc


def clustering_search(X, y, feature_names):
    Xs, imp, sc = prepare_matrix(X)
    pca = PCA(n_components=min(8, Xs.shape[1], Xs.shape[0] - 1), random_state=RANDOM_STATE)
    Z = pca.fit_transform(Xs)

    rows = []
    best = None
    best_key = -1e9

    # Candidate methods
    for k in range(3, 7):
        # KMeans
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        lab = km.fit_predict(Z)
        sil = silhouette_score(Z, lab) if len(set(lab)) > 1 else -1
        ari = adjusted_rand_score(y, lab)
        nmi = normalized_mutual_info_score(y, lab)
        rows.append({"method": "kmeans", "k": k, "silhouette": sil, "ARI": ari, "NMI": nmi, "labels": lab})
        print(f"[clu] kmeans k={k} sil={sil:.3f} ARI={ari:.3f} NMI={nmi:.3f}")

        # Agglomerative ward
        ag = AgglomerativeClustering(n_clusters=k, linkage="ward")
        lab = ag.fit_predict(Z)
        sil = silhouette_score(Z, lab)
        ari = adjusted_rand_score(y, lab)
        nmi = normalized_mutual_info_score(y, lab)
        rows.append({"method": "agglomerative_ward", "k": k, "silhouette": sil, "ARI": ari, "NMI": nmi, "labels": lab})
        print(f"[clu] ward   k={k} sil={sil:.3f} ARI={ari:.3f} NMI={nmi:.3f}")

        # GMM
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_STATE, n_init=5)
        lab = gmm.fit_predict(Z)
        sil = silhouette_score(Z, lab) if len(set(lab)) > 1 else -1
        ari = adjusted_rand_score(y, lab)
        nmi = normalized_mutual_info_score(y, lab)
        rows.append({"method": "gmm", "k": k, "silhouette": sil, "ARI": ari, "NMI": nmi, "labels": lab, "bic": float(gmm.bic(Z))})
        print(f"[clu] gmm    k={k} sil={sil:.3f} ARI={ari:.3f} NMI={nmi:.3f} bic={gmm.bic(Z):.1f}")

    # Selection: silhouette + partial label alignment + cluster-size balance + prefer k=4..5
    for r in rows:
        lab = np.asarray(r["labels"])
        counts = np.bincount(lab)
        p = counts / counts.sum()
        # entropy normalized by log(k): 1 = perfectly balanced
        balance = float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(counts)))
        min_frac = float(counts.min() / counts.sum())
        k_bonus = 0.03 if r["k"] in (4, 5) else 0.0
        # Prefer coherent clusters that are not a near-copy of binary labels,
        # and reject tiny clusters (<8% of data).
        key = (
            0.45 * r["silhouette"]
            + 0.35 * balance
            + 0.15 * (1.0 - abs(r["ARI"] - 0.4))
            + k_bonus
            - (0.25 if min_frac < 0.08 else 0.0)
        )
        r["balance"] = balance
        r["min_frac"] = min_frac
        r["select_score"] = key
        if key > best_key:
            best_key = key
            best = r

    summary = pd.DataFrame(
        [
            {
                "method": r["method"],
                "k": r["k"],
                "silhouette": r["silhouette"],
                "ARI": r["ARI"],
                "NMI": r["NMI"],
                "balance": r.get("balance", np.nan),
                "min_frac": r.get("min_frac", np.nan),
                "select_score": r["select_score"],
                "bic": r.get("bic", np.nan),
            }
            for r in rows
        ]
    ).sort_values("select_score", ascending=False)

    return summary, best, Z, pca, imp, sc


def name_clusters(feat: pd.DataFrame, labels: np.ndarray, feature_names: list[str]) -> dict[int, str]:
    """Assign human-readable names from cluster feature means vs global."""
    df = feat.copy()
    df["cluster"] = labels
    # Focus descriptors
    keys = [
        c
        for c in [
            "duration",
            "approach_track_err_std",
            "approach_speed_maxabs",
            "approach_joint_speed_std",
            "grasp_acc_mag_mean",
            "grasp_closure_mean",
            "lift_duration",
            "carry_right_vy_mean",
            "carry_right_vz_mean",
            "place_jerk_mag_rms",
            "retreat_vy_mean",
            "global_jerk_mag_rms",
            "global_track_err_std",
        ]
        if c in df.columns
    ]
    gmean = df[keys].mean()
    names = {}
    used = set()
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        success_rate = sub["label"].mean()
        m = sub[keys].mean()
        z = (m - gmean) / (df[keys].std().replace(0, 1))
        # pick top distinguishing traits
        top = z.abs().sort_values(ascending=False).head(3)
        traits = []
        for feat_name, zv in top.items():
            direction = "high" if z[feat_name] > 0 else "low"
            short = feat_name.replace("_mean", "").replace("_rms", "").replace("_std", "Var").replace("_maxabs", "Peak")
            traits.append(f"{direction}_{short}")
        zt = " ".join(traits)
        if success_rate >= 0.75:
            if "high_grasp_acc" in zt:
                name = "CleanSuccess_FirmGrasp"
            elif "low_approach" in zt:
                name = "CleanSuccess_SmoothApproach"
            else:
                name = "CleanSuccess"
        elif success_rate <= 0.25:
            if "high_lift_duration" in zt:
                name = "Fail_SlowLift_RoughApproach"
            elif "high_approach" in zt:
                name = "Fail_RoughApproach"
            elif "high_global_jerk" in zt or "high_place_jerk" in zt:
                name = "Fail_JerkyPlace"
            else:
                name = "Fail_Unstable"
        else:
            if "high_lift_duration" in zt:
                name = "Hesitant_SlowLift"
            else:
                name = "Mixed_Atypical"
        # uniqueness
        orig = name
        i = 2
        while name in used:
            name = f"{orig}_{i}"
            i += 1
        used.add(name)
        names[int(c)] = name
        print(f"  cluster {c}: n={len(sub)} success_rate={success_rate:.2f} -> {name} | traits={traits}")
    return names


def build_quality_metric(feat: pd.DataFrame, rank: pd.DataFrame, y_prob: np.ndarray | None):
    """Interpretable 0-100 score from signed effect-size weighted features + optional clf prob."""
    # Select complementary terms covering phases
    candidates = [
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
    # Map: for each feature, direction from Cohen's d (positive d => higher = better for success)
    dmap = dict(zip(rank["feature"], rank["cohens_d"]))
    terms = [c for c in candidates if c in feat.columns and c in dmap]

    # Robust scale to [0,1] within dataset
    scored = pd.DataFrame({"episode": feat["episode"], "label": feat["label"]})
    component_cols = []
    weights = []
    formulas = []

    for t in terms:
        x = feat[t].to_numpy(dtype=float)
        # winsorize-ish via quantiles
        lo, hi = np.nanpercentile(x, [5, 95])
        xw = np.clip(x, lo, hi)
        # normalize 0-1
        denom = hi - lo if hi > lo else 1.0
        u = (xw - lo) / denom
        # if higher is better for success (d>0), keep u; else invert
        if dmap[t] < 0:
            u = 1.0 - u
            direction = "invert"
        else:
            direction = "as_is"
        # weight by |d|
        w = abs(float(dmap[t]))
        col = f"term_{t}"
        scored[col] = u
        component_cols.append(col)
        weights.append(w)
        formulas.append({"feature": t, "direction": direction, "weight_|d|": w, "q05": float(lo), "q95": float(hi)})

    W = np.asarray(weights, dtype=float)
    W = W / W.sum()
    base = np.zeros(len(feat))
    for col, w in zip(component_cols, W):
        base += w * scored[col].to_numpy()

    # Optional blend with classifier probability (capped)
    if y_prob is not None:
        blend = 0.75 * base + 0.25 * y_prob
    else:
        blend = base

    score100 = 100.0 * blend
    scored["quality_0_1"] = blend
    scored["quality_0_100"] = score100
    if y_prob is not None:
        scored["clf_P_success"] = y_prob

    # Validate: should correlate with label
    r_pb = np.corrcoef(scored["quality_0_100"], scored["label"])[0, 1]
    # Separability
    mu_s = scored.loc[scored["label"] == 1, "quality_0_100"].mean()
    mu_f = scored.loc[scored["label"] == 0, "quality_0_100"].mean()

    meta = {
        "formula": "score = 100 * (0.75 * sum_i w_i * u_i + 0.25 * P_success)",
        "u_i": "robust [q05,q95] min-max; inverted if Cohen_d < 0",
        "w_i": "|Cohen_d| normalized to sum 1",
        "terms": formulas,
        "weights_normalized": {t: float(w) for t, w in zip(terms, W)},
        "corr_with_label": float(r_pb),
        "mean_success": float(mu_s),
        "mean_fail": float(mu_f),
        "gap": float(mu_s - mu_f),
    }
    return scored, meta


def plot_clf_comparison(clf_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(clf_df))
    ax.bar(x - 0.2, clf_df["oof_f1"], 0.4, label="OOF F1", color="#4C78A8")
    ax.bar(x + 0.2, clf_df["oof_accuracy"], 0.4, label="OOF Acc", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(clf_df["model"], rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_title("Classification model comparison (stratified OOF)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_clusters(Z, labels, y, names, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for c in np.unique(labels):
        m = labels == c
        axes[0].scatter(Z[m, 0], Z[m, 1], s=55, label=names.get(int(c), str(c)), alpha=0.85)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("Clusters (PCA)")
    axes[0].legend(fontsize=7, loc="best")
    axes[0].grid(True, alpha=0.3)

    for lab, name, color in ((1, "success", "#54A24B"), (0, "fail", "#E45756")):
        m = y == lab
        axes[1].scatter(Z[m, 0], Z[m, 1], s=55, c=color, label=name, alpha=0.85)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("True labels")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scores(score_df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for lab, name, color in ((1, "success", "#54A24B"), (0, "fail", "#E45756")):
        vals = score_df.loc[score_df["label"] == lab, "quality_0_100"]
        axes[0].hist(vals, bins=10, alpha=0.65, color=color, label=name)
    axes[0].set_xlabel("Quality score (0-100)")
    axes[0].set_title("Score distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    order = score_df.sort_values("quality_0_100")
    colors = ["#54A24B" if L == 1 else "#E45756" for L in order["label"]]
    axes[1].barh(order["episode"].astype(str), order["quality_0_100"], color=colors)
    axes[1].set_xlabel("Quality score")
    axes[1].set_ylabel("episode")
    axes[1].set_title("Per-episode scores")
    # readable: show fewer ticks
    axes[1].set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(
    path: Path,
    clf_df: pd.DataFrame,
    best_clf: str,
    best_params: str,
    clu_summary: pd.DataFrame,
    best_clu: dict,
    cluster_names: dict,
    metric_meta: dict,
    feature_names: list[str],
):
    lines = []
    lines.append("# ML Pipeline Results: Classification / Clustering / Quality Metric\n")
    lines.append(f"Feature set: top-{len(feature_names)} by |Cohen's d| from phase kinematics.\n")
    lines.append("## 1. Classification\n")
    lines.append("Stratified 5-fold OOF comparison (primary metric: F1).\n")
    lines.append(clf_df[["model", "oof_f1", "oof_accuracy", "cv_f1_gridbest", "best_params"]].to_string(index=False))
    lines.append(f"\n**Selected model:** `{best_clf}` with params `{best_params}`\n")
    lines.append("## 2. Clustering\n")
    lines.append("Embedding: median-impute → RobustScaler → PCA(8). Selection balances silhouette and partial ARI.\n")
    lines.append(clu_summary.head(12).to_string(index=False))
    lines.append(
        f"\n**Selected clustering:** `{best_clu['method']}` k={best_clu['k']} "
        f"(silhouette={best_clu['silhouette']:.3f}, ARI={best_clu['ARI']:.3f})\n"
    )
    lines.append("### Cluster names\n")
    for cid, name in sorted(cluster_names.items()):
        lines.append(f"- cluster {cid}: **{name}**")
    lines.append("\n## 3. Quality Metric\n")
    lines.append(f"Formula: `{metric_meta['formula']}`\n")
    lines.append(f"- corr(score, label) = {metric_meta['corr_with_label']:.3f}")
    lines.append(f"- mean success = {metric_meta['mean_success']:.1f}, mean fail = {metric_meta['mean_fail']:.1f}, gap = {metric_meta['gap']:.1f}\n")
    lines.append("### Terms\n")
    for t in metric_meta["terms"]:
        lines.append(
            f"- `{t['feature']}` ({t['direction']}), |d|-weight raw={t['weight_|d|']:.3f}, "
            f"q05={t['q05']:.4g}, q95={t['q95']:.4g}"
        )
    lines.append("\n### Normalized weights\n")
    for k, v in metric_meta["weights_normalized"].items():
        lines.append(f"- `{k}`: {v:.3f}")
    lines.append("\n## Artifacts\n")
    lines.append("- `classification_comparison.csv`")
    lines.append("- `best_classifier.joblib`")
    lines.append("- `clustering_comparison.csv`")
    lines.append("- `episode_results.csv` (pred, cluster, quality score)")
    lines.append("- `quality_metric.json`")
    lines.append("- figures: `clf_comparison.png`, `clusters_pca.png`, `quality_scores.png`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feat, X, y, episodes, feature_names, rank = load_xy(TOP_K_FEATURES)
    print(f"Loaded X={X.shape}, features={len(feature_names)}")

    # --- Classification ---
    clf_df, best_name, best_est, y_prob = classification_search(X, y, feature_names)
    clf_df.to_csv(OUT / "classification_comparison.csv", index=False)
    joblib.dump(
        {"model": best_est, "features": feature_names, "model_name": best_name},
        OUT / "best_classifier.joblib",
    )
    plot_clf_comparison(clf_df, OUT / "clf_comparison.png")
    best_params = clf_df.loc[clf_df["model"] == best_name, "best_params"].iloc[0]
    print(f"\n>>> BEST CLASSIFIER: {best_name} {best_params}")

    # --- Clustering ---
    clu_summary, best_clu, Z, pca, imp, sc = clustering_search(X, y, feature_names)
    clu_summary.to_csv(OUT / "clustering_comparison.csv", index=False)
    labels = best_clu["labels"]
    cluster_names = name_clusters(feat, labels, feature_names)
    plot_clusters(Z, labels, y, cluster_names, OUT / "clusters_pca.png")
    print(f"\n>>> BEST CLUSTERING: {best_clu['method']} k={best_clu['k']}")

    # Save clustering model pieces
    joblib.dump(
        {
            "method": best_clu["method"],
            "k": int(best_clu["k"]),
            "imputer": imp,
            "scaler": sc,
            "pca": pca,
            "labels_train": labels,
            "cluster_names": cluster_names,
            "features": feature_names,
        },
        OUT / "best_clustering.joblib",
    )

    # --- Quality metric ---
    score_df, metric_meta = build_quality_metric(feat, rank, y_prob)
    score_df.to_csv(OUT / "quality_scores.csv", index=False)
    (OUT / "quality_metric.json").write_text(json.dumps(metric_meta, indent=2), encoding="utf-8")
    plot_scores(score_df, OUT / "quality_scores.png")
    print(
        f"\n>>> QUALITY METRIC corr={metric_meta['corr_with_label']:.3f} "
        f"gap={metric_meta['gap']:.1f}"
    )

    # Combined episode table
    y_pred_full = best_est.predict(X)
    ep = pd.DataFrame(
        {
            "episode": episodes,
            "label": y,
            "pred": y_pred_full,
            "P_success": y_prob,
            "cluster": labels,
            "cluster_name": [cluster_names[int(c)] for c in labels],
            "quality_0_100": score_df["quality_0_100"].to_numpy(),
        }
    )
    ep.to_csv(OUT / "episode_results.csv", index=False)

    write_report(
        OUT / "ml_pipeline_report.md",
        clf_df,
        best_name,
        best_params,
        clu_summary,
        best_clu,
        cluster_names,
        metric_meta,
        feature_names,
    )
    print(f"\nWrote report: {OUT / 'ml_pipeline_report.md'}")


if __name__ == "__main__":
    main()
