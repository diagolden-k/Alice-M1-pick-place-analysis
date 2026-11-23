"""Animate Alice M1 arm stick figure from episode CSV using URDF FK.

Example:
  python viz/visualize_arm_motion.py --episode 2
  python viz/visualize_arm_motion.py --episode 2 --source action --save out.gif
  python viz/visualize_arm_motion.py --episode 1 --side both --stride 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "viz") not in sys.path:
    sys.path.insert(0, str(ROOT / "viz"))

from alice_arm_fk import (  # noqa: E402
    ARM_JOINT_NAMES,
    HAND_JOINT_NAMES,
    arm_chain,
    forward_kinematics,
    hand_closure,
    stick_points,
)
from phases import PHASE_COLORS, PHASE_NAMES, estimate_phases  # noqa: E402

DATA_ROOT = ROOT / "26_2_machinelearning_1st_data" / "26_2_machinelearning_1st_data"


def find_episode_csv(episode: int) -> Path:
    name = f"episode_{episode:06d}.csv"
    for split in ("success", "fail"):
        path = DATA_ROOT / split / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Episode CSV not found: {name} under {DATA_ROOT}")


def joint_cols(prefix: str, side: str) -> list[str]:
    # prefix: state / action, side: l / r
    return [f"{prefix}.{side}_{name}" for name in ARM_JOINT_NAMES]


def hand_cols(prefix: str, side: str) -> list[str]:
    return [f"{prefix}.{side}_{name}" for name in HAND_JOINT_NAMES]


def load_episode(path: Path, source: str, stride: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    needed = []
    for side in ("l", "r"):
        needed.extend(joint_cols(source, side))
        needed.extend(hand_cols(source, side))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {path.name}: {missing[:5]}...")
    return df


def fk_trajectory(df: pd.DataFrame, source: str, side: str) -> dict[str, np.ndarray]:
    chain = arm_chain(side)
    jcols = joint_cols(source, side[0])
    hcols = hand_cols(source, side[0])
    sticks = []
    palms = []
    closures = []
    for _, row in df.iterrows():
        q = row[jcols].to_numpy(dtype=float)
        pts = forward_kinematics(chain, q)
        stick = stick_points(pts)
        sticks.append(stick)
        palms.append(stick[-1])
        closures.append(hand_closure(row[hcols].to_numpy(dtype=float)))
    sticks_arr = np.stack(sticks, axis=0)  # (T, J, 3)
    palms_arr = np.stack(palms, axis=0)
    return {
        "sticks": sticks_arr,
        "palm": palms_arr,
        "closure": np.asarray(closures, dtype=float),
        "timestamp": df["timestamp"].to_numpy(dtype=float),
    }


def _axis_limits(points: np.ndarray, margin: float = 0.08) -> tuple[tuple[float, float], ...]:
    mins = points.reshape(-1, 3).min(axis=0) - margin
    maxs = points.reshape(-1, 3).max(axis=0) + margin
    # Equal aspect cube
    center = 0.5 * (mins + maxs)
    half = 0.5 * np.max(maxs - mins)
    half = max(half, 0.35)
    return tuple((float(center[i] - half), float(center[i] + half)) for i in range(3))


def animate(
    trajs: dict[str, dict[str, np.ndarray]],
    phases: list[str],
    title: str,
    interval_ms: float,
    save: Path | None,
    show: bool,
) -> None:
    sides = list(trajs.keys())
    all_pts = np.concatenate([trajs[s]["sticks"] for s in sides], axis=0)
    xlim, ylim, zlim = _axis_limits(all_pts)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)

    arm_lines = {}
    palm_dots = {}
    for side, color in zip(sides, ("#1f77b4", "#d62728")):
        (line,) = ax.plot([], [], [], "-o", color=color, lw=3, markersize=5, label=f"{side} arm")
        (dot,) = ax.plot([], [], [], "o", color=color, markersize=9)
        arm_lines[side] = line
        palm_dots[side] = dot

    trail_side = "left" if "left" in trajs else sides[0]
    trail = trajs[trail_side]["palm"]
    trail_colors = [PHASE_COLORS[p] for p in phases]
    ax.scatter(
        trail[:, 0],
        trail[:, 1],
        trail[:, 2],
        c=trail_colors,
        s=8,
        alpha=0.35,
        depthshade=False,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("X forward (m)")
    ax.set_ylabel("Y left (m)")
    ax.set_zlabel("Z up (m)")
    ax.legend(loc="upper left")
    ax.set_title(title)

    # Top-down path (X-Y) and height/closure timelines
    ts = trajs[trail_side]["timestamp"]
    palm = trajs[trail_side]["palm"]
    closure = trajs[trail_side]["closure"]
    ax2.plot(palm[:, 1], palm[:, 2], color="#bbbbbb", lw=1, label="Y-Z path")
    (phase_pt,) = ax2.plot([], [], "o", color="black", markersize=8)
    ax2.set_xlabel("Y left → right decreases (m)")
    ax2.set_ylabel("Z up (m)")
    ax2.set_title("Palm path + phase")
    ax2.grid(True, alpha=0.3)
    for name in PHASE_NAMES[:-1]:
        ax2.plot([], [], color=PHASE_COLORS[name], lw=4, label=name)
    ax2.legend(loc="best", fontsize=8)

    status = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    n = len(ts)

    def update(frame: int):
        for side in sides:
            stick = trajs[side]["sticks"][frame]
            arm_lines[side].set_data(stick[:, 0], stick[:, 1])
            arm_lines[side].set_3d_properties(stick[:, 2])
            palm_dots[side].set_data([stick[-1, 0]], [stick[-1, 1]])
            palm_dots[side].set_3d_properties([stick[-1, 2]])
        phase_pt.set_data([palm[frame, 1]], [palm[frame, 2]])
        status.set_text(
            f"t={ts[frame]:.2f}s  frame={frame}/{n-1}\n"
            f"phase={phases[frame]}  closure={closure[frame]:.3f}"
        )
        return [*arm_lines.values(), *palm_dots.values(), phase_pt, status]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        if save.suffix.lower() == ".gif":
            anim.save(save, writer="pillow", fps=max(1, int(1000 / interval_ms)))
        else:
            anim.save(save, fps=max(1, int(1000 / interval_ms)))
        print(f"Saved: {save}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def export_trajectory_csv(
    out: Path,
    timestamps: np.ndarray,
    palm: np.ndarray,
    closure: np.ndarray,
    phases: list[str],
) -> None:
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "palm_x": palm[:, 0],
            "palm_y": palm[:, 1],
            "palm_z": palm[:, 2],
            "hand_closure": closure,
            "phase": phases,
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote trajectory: {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize Alice M1 arm motion from CSV via FK")
    p.add_argument("--episode", type=int, required=True, help="Episode index, e.g. 2")
    p.add_argument("--source", choices=("state", "action"), default="state")
    p.add_argument("--side", choices=("left", "right", "both"), default="left")
    p.add_argument("--stride", type=int, default=2, help="Use every Nth frame (default 2)")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    p.add_argument("--save", type=Path, default=None, help="Save animation (.gif/.mp4)")
    p.add_argument("--export-csv", type=Path, default=None, help="Export palm XYZ + phase CSV")
    p.add_argument("--no-show", action="store_true", help="Do not open interactive window")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = find_episode_csv(args.episode)
    split = csv_path.parent.name
    df = load_episode(csv_path, args.source, args.stride)

    sides = ["left", "right"] if args.side == "both" else [args.side]
    trajs = {s: fk_trajectory(df, args.source, s) for s in sides}

    focus = "left" if "left" in trajs else sides[0]
    phases = estimate_phases(
        trajs[focus]["palm"],
        trajs[focus]["closure"],
        trajs[focus]["timestamp"],
    )

    if args.export_csv is not None:
        export_trajectory_csv(
            args.export_csv,
            trajs[focus]["timestamp"],
            trajs[focus]["palm"],
            trajs[focus]["closure"],
            phases,
        )

    # ~30fps source; stride reduces frames; interval for playback
    base_dt_ms = 1000.0 / 30.0 * args.stride
    interval_ms = max(1.0, base_dt_ms / max(args.speed, 1e-6))

    title = f"Episode {args.episode:06d} ({split}) | {args.source} | torso frame"
    animate(
        trajs,
        phases,
        title=title,
        interval_ms=interval_ms,
        save=args.save,
        show=not args.no_show,
    )

    # Phase summary for terminal
    from collections import Counter

    counts = Counter(phases)
    total = sum(counts.values())
    print(f"Loaded {csv_path}")
    print("Phase fractions:")
    for name in PHASE_NAMES:
        if name in counts:
            print(f"  {name:12s} {counts[name]:4d}  ({100 * counts[name] / total:.1f}%)")


if __name__ == "__main__":
    main()
