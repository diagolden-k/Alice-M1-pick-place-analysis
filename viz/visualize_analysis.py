"""Summary visualizations for phase kinematics + feature influence."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phases import PHASE_COLORS, PHASE_NAMES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "viz" / "outputs"
ACTIVE = [p for p in PHASE_NAMES if p != "other"]


_SAVE_DIR: Path = OUT


def _save(fig: plt.Figure, name: str) -> Path:
    path = _SAVE_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def plot_phase_duration_by_label(phase_counts: pd.DataFrame) -> None:
    # frame counts -> seconds at 30fps
    rows = []
    for _, r in phase_counts.iterrows():
        for ph in ACTIVE:
            rows.append(
                {
                    "phase": ph,
                    "label": "success" if r["label"] == 1 else "fail",
                    "seconds": r[ph] / 30.0,
                }
            )
    df = pd.DataFrame(rows)
    means = df.groupby(["phase", "label"])["seconds"].mean().unstack("label")
    means = means.reindex(ACTIVE)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(ACTIVE))
    w = 0.35
    ax.bar(x - w / 2, means.get("success", 0), w, label="success", color="#54A24B")
    ax.bar(x + w / 2, means.get("fail", 0), w, label="fail", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(ACTIVE, rotation=20)
    ax.set_ylabel("Mean duration (s)")
    ax.set_title("Phase duration: success vs fail")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "viz_phase_duration_by_label.png")


def plot_kinematics_by_phase(frame_df: pd.DataFrame) -> None:
    metrics = [("speed", "Speed (m/s)"), ("acc_mag", "Acceleration (m/s²)"), ("jerk_mag", "Jerk (m/s³)")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
    for ax, (col, ylab) in zip(axes, metrics):
        data_s, data_f, labels = [], [], []
        for ph in ACTIVE:
            sub = frame_df[frame_df["phase"] == ph]
            data_s.append(sub.loc[sub["label"] == 1, col].dropna().to_numpy())
            data_f.append(sub.loc[sub["label"] == 0, col].dropna().to_numpy())
            labels.append(ph)
        pos = np.arange(len(ACTIVE))
        bp1 = ax.boxplot(
            data_s,
            positions=pos - 0.18,
            widths=0.3,
            patch_artist=True,
            showfliers=False,
        )
        bp2 = ax.boxplot(
            data_f,
            positions=pos + 0.18,
            widths=0.3,
            patch_artist=True,
            showfliers=False,
        )
        for patch in bp1["boxes"]:
            patch.set_facecolor("#54A24B")
            patch.set_alpha(0.7)
        for patch in bp2["boxes"]:
            patch.set_facecolor("#E45756")
            patch.set_alpha(0.7)
        ax.set_xticks(pos)
        ax.set_xticklabels(ACTIVE, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_title("Cartesian kinematics by phase")
    from matplotlib.patches import Patch

    axes[2].legend(
        handles=[
            Patch(color="#54A24B", label="success"),
            Patch(color="#E45756", label="fail"),
        ],
        loc="upper right",
    )
    _save(fig, "viz_kinematics_boxplot_by_phase.png")


def plot_example_episode_timeline(frame_df: pd.DataFrame, episode: int = 2) -> None:
    sub = frame_df[frame_df["episode"] == episode].copy()
    if sub.empty:
        # pick first success
        episode = int(frame_df.loc[frame_df["label"] == 1, "episode"].iloc[0])
        sub = frame_df[frame_df["episode"] == episode].copy()
    t = sub["timestamp"].to_numpy()
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    # phase bands
    for ax in axes:
        for ph in ACTIVE:
            m = sub["phase"] == ph
            if not m.any():
                continue
            idx = np.where(m.to_numpy())[0]
            # contiguous spans
            breaks = np.where(np.diff(idx) > 1)[0]
            starts = np.insert(idx[breaks + 1], 0, idx[0])
            ends = np.append(idx[breaks], idx[-1])
            for s, e in zip(starts, ends):
                ax.axvspan(t[s], t[e], color=PHASE_COLORS[ph], alpha=0.15, lw=0)

    axes[0].plot(t, sub["palm_y"], color="#4C78A8", label="Y (left+)")
    axes[0].plot(t, sub["palm_z"], color="#F58518", label="Z (up+)")
    axes[0].set_ylabel("position (m)")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"Episode {episode:06d} timeline (phase bands)")

    axes[1].plot(t, sub["speed"], color="#333")
    axes[1].set_ylabel("speed")

    axes[2].plot(t, sub["acc_mag"], color="#333", label="|a|")
    axes[2].plot(t, sub["jerk_mag"], color="#B279A2", alpha=0.8, label="|j|")
    axes[2].legend(loc="upper right")
    axes[2].set_ylabel("acc / jerk")

    axes[3].plot(t, sub["hand_closure"], color="#E45756")
    axes[3].set_ylabel("closure")
    axes[3].set_xlabel("time (s)")

    handles = [
        plt.matplotlib.patches.Patch(color=PHASE_COLORS[p], alpha=0.4, label=p) for p in ACTIVE
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, f"viz_episode_{episode:06d}_timeline.png")


def plot_mean_paths(frame_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, name, color in ((1, "success", "#54A24B"), (0, "fail", "#E45756")):
        sub = frame_df[frame_df["label"] == label]
        for ep, g in sub.groupby("episode"):
            axes[0].plot(g["palm_y"], g["palm_z"], color=color, alpha=0.25, lw=1)
            axes[1].plot(g["timestamp"] / g["timestamp"].max(), g["palm_z"], color=color, alpha=0.25, lw=1)
        axes[0].plot([], [], color=color, label=name)
        axes[1].plot([], [], color=color, label=name)
    axes[0].set_xlabel("Y left+ (m)")
    axes[0].set_ylabel("Z up (m)")
    axes[0].set_title("All palm paths (Y-Z)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("normalized time")
    axes[1].set_ylabel("Z up (m)")
    axes[1].set_title("Height vs normalized time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    _save(fig, "viz_all_palm_paths.png")


def plot_cohens_d(ranking: pd.DataFrame, n: int = 20) -> None:
    top = ranking.head(n).iloc[::-1]
    colors = ["#54A24B" if d > 0 else "#E45756" for d in top["cohens_d"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["cohens_d"], color=colors)
    ax.axvline(0, color="#333", lw=1)
    ax.set_xlabel("Cohen's d (success − fail)")
    ax.set_title(f"Top {n} features by |effect size|")
    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, "viz_feature_cohens_d_top20.png")


def plot_phase_group_influence(phase_sum: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = phase_sum.sort_values("mean_abs_d")
    colors = [PHASE_COLORS.get(p, "#9E9E9E") if p != "global" else "#555555" for p in order["phase_group"]]
    ax.barh(order["phase_group"], order["mean_abs_d"], color=colors)
    ax.set_xlabel("Mean |Cohen's d| across features")
    ax.set_title("Which phase group matters most?")
    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, "viz_phase_group_influence.png")


def plot_logit_coefs(logit: pd.DataFrame, n: int = 15) -> None:
    top = logit.head(n).iloc[::-1]
    colors = ["#54A24B" if c > 0 else "#E45756" for c in top["logit_coef_std"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["logit_coef_std"], color=colors)
    ax.axvline(0, color="#333", lw=1)
    ax.set_xlabel("Standardized logistic coefficient β")
    ax.set_title("Logistic influence (+ → success odds ↑)")
    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, "viz_logistic_coefs.png")


def plot_scatter_top2(feat_df: pd.DataFrame, ranking: pd.DataFrame) -> None:
    f1, f2 = ranking["feature"].iloc[0], ranking["feature"].iloc[1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for label, name, color in ((1, "success", "#54A24B"), (0, "fail", "#E45756")):
        m = feat_df["label"] == label
        ax.scatter(feat_df.loc[m, f1], feat_df.loc[m, f2], c=color, label=name, s=55, alpha=0.85)
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title("Top-2 effect-size features")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "viz_top2_feature_scatter.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or OUT

    feat = pd.read_csv(out_dir / "episode_phase_features.csv")
    frames = pd.read_csv(out_dir / "frame_phase_kinematics.csv")
    phases = pd.read_csv(out_dir / "episode_phase_counts.csv")
    ranking = pd.read_csv(out_dir / "feature_influence_ranking.csv")
    logit = pd.read_csv(out_dir / "logistic_feature_coefs.csv")
    phase_sum = pd.read_csv(out_dir / "phase_group_influence.csv")

    global _SAVE_DIR
    _SAVE_DIR = out_dir

    plot_phase_duration_by_label(phases)
    plot_kinematics_by_phase(frames)
    plot_example_episode_timeline(frames, episode=2)
    plot_mean_paths(frames)
    plot_cohens_d(ranking)
    plot_phase_group_influence(phase_sum)
    plot_logit_coefs(logit)
    plot_scatter_top2(feat, ranking)
    print("All analysis figures written.")


if __name__ == "__main__":
    main()
