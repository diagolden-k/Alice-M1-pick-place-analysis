"""Mathematical influence analysis of phase-kinematics features on success/fail."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mutual_info_score

ROOT = Path(__file__).resolve().parents[1]


def point_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r between continuous x and binary y (equivalent to point-biserial)."""
    if np.std(x) < 1e-12 or len(np.unique(y)) < 2:
        return np.nan
    r, _ = stats.pearsonr(x, y)
    return float(r)


def cohens_d(x1: np.ndarray, x0: np.ndarray) -> float:
    n1, n0 = len(x1), len(x0)
    if n1 < 2 or n0 < 2:
        return np.nan
    v1, v0 = np.var(x1, ddof=1), np.var(x0, ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2))
    if pooled < 1e-12:
        return np.nan
    return float((np.mean(x1) - np.mean(x0)) / pooled)


def rank_biserial_from_mwu(u: float, n1: int, n0: int) -> float:
    """Mann-Whitney rank-biserial correlation: r = 1 - 2U/(n1 n0)."""
    return float(1.0 - (2.0 * u) / (n1 * n0))


def analyze(feat_df: pd.DataFrame) -> pd.DataFrame:
    y = feat_df["label"].to_numpy(dtype=int)
    skip = {"episode", "label"}
    cols = [c for c in feat_df.columns if c not in skip]
    rows = []
    for col in cols:
        x = feat_df[col].to_numpy(dtype=float)
        mask = np.isfinite(x)
        if mask.sum() < 10 or np.unique(y[mask]).size < 2:
            continue
        xm, ym = x[mask], y[mask]
        xs, xf = xm[ym == 1], xm[ym == 0]
        # Welch t-test
        tstat, tp = stats.ttest_ind(xs, xf, equal_var=False, nan_policy="omit")
        # Mann-Whitney U
        try:
            ures = stats.mannwhitneyu(xs, xf, alternative="two-sided")
            u, up = float(ures.statistic), float(ures.pvalue)
            rrb = rank_biserial_from_mwu(u, len(xs), len(xf))
        except ValueError:
            u, up, rrb = np.nan, np.nan, np.nan

        # Mutual information via median split discretization of x (for MI with label)
        # Better: use sklearn mutual_info_classif style with kNN — but keep simple histogram MI
        try:
            # 5-bin quantile discretization
            qs = np.unique(np.quantile(xm, np.linspace(0, 1, 6)))
            if len(qs) < 3:
                mi = np.nan
            else:
                bins = np.digitize(xm, qs[1:-1], right=True)
                mi = float(mutual_info_score(ym, bins))
        except Exception:
            mi = np.nan

        rows.append(
            {
                "feature": col,
                "mean_success": float(np.mean(xs)) if len(xs) else np.nan,
                "mean_fail": float(np.mean(xf)) if len(xf) else np.nan,
                "diff_success_minus_fail": float(np.mean(xs) - np.mean(xf)) if len(xs) and len(xf) else np.nan,
                "cohens_d": cohens_d(xs, xf),
                "point_biserial_r": point_biserial(xm, ym),
                "welch_t": float(tstat),
                "welch_p": float(tp),
                "mannwhitney_U": u,
                "mannwhitney_p": up,
                "rank_biserial_r": rrb,
                "mutual_info_bits": mi,
            }
        )
    out = pd.DataFrame(rows)
    # FDR (Benjamini-Hochberg) on Welch p
    p = out["welch_p"].to_numpy()
    order = np.argsort(p)
    ranked = np.empty_like(p)
    m = len(p)
    prev = 1.0
    for i, idx in enumerate(order[::-1]):
        rank = m - i
        val = min(prev, p[idx] * m / rank)
        ranked[idx] = val
        prev = val
    out["welch_p_fdr"] = ranked
    out["abs_d"] = out["cohens_d"].abs()
    out["abs_r"] = out["point_biserial_r"].abs()
    return out.sort_values("abs_d", ascending=False).reset_index(drop=True)


def logistic_influence(feat_df: pd.DataFrame, top_n: int, ranking: pd.DataFrame) -> pd.DataFrame:
    """Standardized logistic regression on top features by |Cohen's d|."""
    y = feat_df["label"].to_numpy(dtype=int)
    candidates = ranking["feature"].head(top_n).tolist()
    X = feat_df[candidates].to_numpy(dtype=float)
    # Impute column means
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=min(5, int(y.sum()), int((1 - y).sum())), shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    pipe.fit(X, y)
    coef = pipe.named_steps["clf"].coef_.ravel()
    # Odd-ratio for +1 std change: exp(coef)
    return pd.DataFrame(
        {
            "feature": candidates,
            "logit_coef_std": coef,
            "odds_ratio_per_std": np.exp(coef),
            "cv_accuracy_mean": float(np.mean(scores)),
            "cv_accuracy_std": float(np.std(scores)),
        }
    ).sort_values("logit_coef_std", key=np.abs, ascending=False)


def phase_group_summary(ranking: pd.DataFrame) -> pd.DataFrame:
    def phase_of(feat: str) -> str:
        for ph in ("approach", "grasp", "lift", "carry_right", "place", "retreat", "global"):
            if feat.startswith(ph + "_") or feat == "duration" and ph == "global":
                return ph if feat.startswith(ph) else "global"
        if feat in ("duration", "n_frames") or feat.startswith("global_"):
            return "global"
        return "other"

    df = ranking.copy()
    df["phase_group"] = df["feature"].map(phase_of)

    def kin_family(feat: str) -> str:
        for k in ("jerk", "acc", "speed", "vy", "vz", "dy", "dz", "closure", "track", "duration", "frac", "path"):
            if k in feat:
                return k
        return "other"

    df["kin_family"] = df["feature"].map(kin_family)
    g = (
        df.groupby("phase_group")
        .agg(
            n_features=("feature", "count"),
            mean_abs_d=("abs_d", "mean"),
            max_abs_d=("abs_d", "max"),
            mean_abs_r=("abs_r", "mean"),
            n_sig_fdr05=("welch_p_fdr", lambda s: int(np.sum(s < 0.05))),
        )
        .sort_values("mean_abs_d", ascending=False)
    )
    return g.reset_index()


def plot_top(ranking: pd.DataFrame, out: Path, n: int = 20) -> None:
    top = ranking.head(n).iloc[::-1]
    colors = ["#54A24B" if d > 0 else "#E45756" for d in top["cohens_d"]]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["cohens_d"], color=colors)
    ax.axvline(0, color="#333", lw=1)
    ax.set_xlabel("Cohen's d (success − fail) / pooled std")
    ax.set_title(f"Top {n} features by |effect size|")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(format(float(v), floatfmt))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def write_report(
    ranking: pd.DataFrame,
    logit: pd.DataFrame,
    phase_sum: pd.DataFrame,
    out: Path,
) -> None:
    lines = []
    lines.append("# Feature influence analysis (success=1 vs fail=0)\n")
    lines.append("## Methods\n")
    lines.append(
        "- **Cohen's d**: standardized mean difference `(μ_success − μ_fail) / s_pooled`.\n"
        "- **Point-biserial r**: Pearson correlation between feature and binary label.\n"
        "- **Welch t-test** + Benjamini–Hochberg FDR on p-values.\n"
        "- **Mann–Whitney U** + rank-biserial correlation.\n"
        "- **Mutual information** (feature quantile-binned vs label).\n"
        "- **L2 logistic regression** on standardized top-|d| features; "
        "`odds_ratio_per_std = exp(β)` for +1σ change.\n"
    )
    lines.append("## Phase-group effect summary\n")
    lines.append(_md_table(phase_sum))
    lines.append("\n## Top 25 features by |Cohen's d|\n")
    cols = [
        "feature",
        "mean_success",
        "mean_fail",
        "cohens_d",
        "point_biserial_r",
        "welch_p_fdr",
        "rank_biserial_r",
        "mutual_info_bits",
    ]
    lines.append(_md_table(ranking[cols].head(25)))
    lines.append("\n## Logistic coefficients (standardized features)\n")
    lines.append(
        f"CV accuracy: {logit['cv_accuracy_mean'].iloc[0]:.3f} ± {logit['cv_accuracy_std'].iloc[0]:.3f}\n"
    )
    lines.append(_md_table(logit[["feature", "logit_coef_std", "odds_ratio_per_std"]]))
    lines.append(
        "\nInterpretation: `logit_coef_std > 0` increases log-odds of **success** when the "
        "feature rises by 1 standard deviation (other top features held constant).\n"
    )
    lines.append("\n## Figures\n")
    lines.append(
        "- `viz_phase_duration_by_label.png`\n"
        "- `viz_kinematics_boxplot_by_phase.png`\n"
        "- `viz_episode_*_timeline.png`\n"
        "- `viz_all_palm_paths.png`\n"
        "- `viz_feature_cohens_d_top20.png`\n"
        "- `viz_phase_group_influence.png`\n"
        "- `viz_logistic_coefs.png`\n"
        "- `viz_top2_feature_scatter.png`\n"
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features",
        type=Path,
        default=ROOT / "viz" / "outputs" / "episode_phase_features.csv",
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "viz" / "outputs")
    ap.add_argument("--top-n", type=int, default=25)
    args = ap.parse_args()

    feat_df = pd.read_csv(args.features)
    ranking = analyze(feat_df)
    logit = logistic_influence(feat_df, args.top_n, ranking)
    phase_sum = phase_group_summary(ranking)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.out_dir / "feature_influence_ranking.csv", index=False)
    logit.to_csv(args.out_dir / "logistic_feature_coefs.csv", index=False)
    phase_sum.to_csv(args.out_dir / "phase_group_influence.csv", index=False)
    plot_top(ranking, args.out_dir / "feature_cohens_d_top20.png", n=20)
    write_report(ranking, logit, phase_sum, args.out_dir / "feature_influence_report.md")

    print("=== Phase group summary ===")
    print(phase_sum.to_string(index=False))
    print("\n=== Top 15 by |Cohen's d| ===")
    print(
        ranking[
            ["feature", "cohens_d", "point_biserial_r", "welch_p_fdr", "mutual_info_bits"]
        ]
        .head(15)
        .to_string(index=False)
    )
    print("\n=== Logistic top coefs ===")
    print(logit.head(12).to_string(index=False))
    print(f"\nWrote report: {args.out_dir / 'feature_influence_report.md'}")


if __name__ == "__main__":
    main()
