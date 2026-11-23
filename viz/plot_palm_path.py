"""Static overview: left palm XYZ path colored by estimated phase."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))

from phases import PHASE_COLORS, PHASE_NAMES, estimate_phases  # noqa: E402
from visualize_arm_motion import (  # noqa: E402
    find_episode_csv,
    fk_trajectory,
    load_episode,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--source", choices=("state", "action"), default="state")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    path = find_episode_csv(args.episode)
    df = load_episode(path, args.source, args.stride)
    traj = fk_trajectory(df, args.source, "left")
    phases = estimate_phases(traj["palm"], traj["closure"], traj["timestamp"])
    palm = traj["palm"]
    colors = [PHASE_COLORS[ph] for ph in phases]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(f"Episode {args.episode:06d} ({path.parent.name}) left palm — torso frame")

    axes[0].scatter(palm[:, 1], palm[:, 0], c=colors, s=12)
    axes[0].set_xlabel("Y (left +)")
    axes[0].set_ylabel("X (forward +)")
    axes[0].set_title("Top view X-Y")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(palm[:, 1], palm[:, 2], c=colors, s=12)
    axes[1].set_xlabel("Y (left +)")
    axes[1].set_ylabel("Z (up +)")
    axes[1].set_title("Side view Y-Z (approach→place)")
    axes[1].grid(True, alpha=0.3)

    t = traj["timestamp"]
    axes[2].plot(t, palm[:, 2], label="Z", color="#333333")
    axes[2].plot(t, traj["closure"], label="hand closure", color="#F58518")
    axes[2].set_xlabel("time (s)")
    axes[2].set_title("Height & grasp")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PHASE_COLORS[n], markersize=8, label=n)
        for n in PHASE_NAMES[:-1]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out = args.out or (ROOT / "viz" / "outputs" / f"episode_{args.episode:06d}_palm.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
