"""Build phase-segmented kinematics features (vel/acc/jerk) for all episodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))

from alice_arm_fk import ARM_JOINT_NAMES, HAND_JOINT_NAMES, arm_chain, forward_kinematics, hand_closure, stick_points
from phases import PHASE_NAMES, estimate_phases
from visualize_arm_motion import DATA_ROOT, find_episode_csv, hand_cols, joint_cols, load_episode

ACTIVE_PHASES = [p for p in PHASE_NAMES if p != "other"]


def derivatives(x: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return velocity, acceleration, jerk along time for (T,) or (T,D)."""
    v = np.gradient(x, t, axis=0)
    a = np.gradient(v, t, axis=0)
    j = np.gradient(a, t, axis=0)
    return v, a, j


def _stats(name: str, arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=float).ravel()
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return {f"{name}_mean": np.nan, f"{name}_std": np.nan, f"{name}_rms": np.nan, f"{name}_maxabs": np.nan}
    arr = arr[np.isfinite(arr)]
    return {
        f"{name}_mean": float(np.mean(arr)),
        f"{name}_std": float(np.std(arr)),
        f"{name}_rms": float(np.sqrt(np.mean(arr**2))),
        f"{name}_maxabs": float(np.max(np.abs(arr))),
    }


def episode_kinematics(df: pd.DataFrame, source: str = "state", side: str = "left") -> dict[str, np.ndarray]:
    chain = arm_chain(side)
    jcols = joint_cols(source, side[0])
    hcols = hand_cols(source, side[0])
    t = df["timestamp"].to_numpy(dtype=float)
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        t = np.arange(len(df), dtype=float) / 30.0

    q = df[jcols].to_numpy(dtype=float)  # (T,7)
    hand = df[hcols].to_numpy(dtype=float)
    closure = np.array([hand_closure(h) for h in hand], dtype=float)

    palms = []
    for qi in q:
        pts = forward_kinematics(chain, qi)
        palms.append(stick_points(pts)[-1])
    palm = np.stack(palms, axis=0)

    # Cartesian kinematics of palm
    v_p, a_p, j_p = derivatives(palm, t)
    speed = np.linalg.norm(v_p, axis=1)
    acc_mag = np.linalg.norm(a_p, axis=1)
    jerk_mag = np.linalg.norm(j_p, axis=1)

    # Joint-space kinematics
    v_q, a_q, j_q = derivatives(q, t)
    joint_speed = np.linalg.norm(v_q, axis=1)
    joint_acc = np.linalg.norm(a_q, axis=1)
    joint_jerk = np.linalg.norm(j_q, axis=1)

    # Tracking error if action present
    track = None
    try:
        qa = df[joint_cols("action", side[0])].to_numpy(dtype=float)
        track = np.linalg.norm(qa - q, axis=1)
    except Exception:
        track = np.full(len(df), np.nan)

    phases = estimate_phases(palm, closure, t)
    return {
        "t": t,
        "palm": palm,
        "closure": closure,
        "speed": speed,
        "acc_mag": acc_mag,
        "jerk_mag": jerk_mag,
        "v_p": v_p,
        "a_p": a_p,
        "j_p": j_p,
        "joint_speed": joint_speed,
        "joint_acc": joint_acc,
        "joint_jerk": joint_jerk,
        "track_err": track,
        "phases": np.asarray(phases),
        "q": q,
    }


def aggregate_episode_features(kin: dict[str, np.ndarray], label: int, episode: int) -> dict[str, float]:
    feats: dict[str, float] = {
        "episode": episode,
        "label": label,  # 1=success, 0=fail
        "duration": float(kin["t"][-1] - kin["t"][0]) if len(kin["t"]) else np.nan,
        "n_frames": float(len(kin["t"])),
    }

    # Global kinematics
    for key in ("speed", "acc_mag", "jerk_mag", "joint_speed", "joint_acc", "joint_jerk", "track_err", "closure"):
        feats.update(_stats(f"global_{key}", kin[key]))

    palm = kin["palm"]
    feats["global_path_length"] = float(np.sum(np.linalg.norm(np.diff(palm, axis=0), axis=1)))
    feats["global_dy"] = float(palm[-1, 1] - palm[0, 1])
    feats["global_dz"] = float(palm[-1, 2] - palm[0, 2])
    feats["global_z_range"] = float(palm[:, 2].max() - palm[:, 2].min())
    feats["global_y_range"] = float(palm[:, 1].max() - palm[:, 1].min())

    phases = kin["phases"]
    for ph in ACTIVE_PHASES:
        mask = phases == ph
        feats[f"{ph}_duration"] = float(np.sum(mask) / 30.0)  # approx seconds at 30fps
        feats[f"{ph}_frac"] = float(np.mean(mask)) if len(mask) else np.nan
        if not np.any(mask):
            for key in (
                "speed",
                "acc_mag",
                "jerk_mag",
                "joint_speed",
                "joint_acc",
                "joint_jerk",
                "track_err",
                "closure",
            ):
                feats.update(_stats(f"{ph}_{key}", np.array([])))
            feats[f"{ph}_dy"] = np.nan
            feats[f"{ph}_dz"] = np.nan
            feats[f"{ph}_vy_mean"] = np.nan
            feats[f"{ph}_vz_mean"] = np.nan
            continue

        for key in (
            "speed",
            "acc_mag",
            "jerk_mag",
            "joint_speed",
            "joint_acc",
            "joint_jerk",
            "track_err",
            "closure",
        ):
            feats.update(_stats(f"{ph}_{key}", kin[key][mask]))

        p = palm[mask]
        feats[f"{ph}_dy"] = float(p[-1, 1] - p[0, 1])
        feats[f"{ph}_dz"] = float(p[-1, 2] - p[0, 2])
        feats[f"{ph}_vy_mean"] = float(np.mean(kin["v_p"][mask, 1]))
        feats[f"{ph}_vz_mean"] = float(np.mean(kin["v_p"][mask, 2]))

    return feats


def frame_table(kin: dict[str, np.ndarray], label: int, episode: int) -> pd.DataFrame:
    """Per-frame table with phase + kinematics (left palm)."""
    palm = kin["palm"]
    v, a, j = kin["v_p"], kin["a_p"], kin["j_p"]
    return pd.DataFrame(
        {
            "episode": episode,
            "label": label,
            "timestamp": kin["t"],
            "phase": kin["phases"],
            "palm_x": palm[:, 0],
            "palm_y": palm[:, 1],
            "palm_z": palm[:, 2],
            "vx": v[:, 0],
            "vy": v[:, 1],
            "vz": v[:, 2],
            "ax": a[:, 0],
            "ay": a[:, 1],
            "az": a[:, 2],
            "jx": j[:, 0],
            "jy": j[:, 1],
            "jz": j[:, 2],
            "speed": kin["speed"],
            "acc_mag": kin["acc_mag"],
            "jerk_mag": kin["jerk_mag"],
            "joint_speed": kin["joint_speed"],
            "joint_acc": kin["joint_acc"],
            "joint_jerk": kin["joint_jerk"],
            "track_err": kin["track_err"],
            "hand_closure": kin["closure"],
        }
    )


def list_all_episodes() -> list[tuple[int, int, Path]]:
    rows = []
    for split, label in (("success", 1), ("fail", 0)):
        for path in sorted((DATA_ROOT / split).glob("episode_*.csv")):
            ep = int(path.stem.split("_")[1])
            rows.append((ep, label, path))
    return sorted(rows, key=lambda x: x[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("state", "action"), default="state")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "viz" / "outputs")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ep_rows = []
    frame_rows = []
    phase_counts = []

    for ep, label, path in list_all_episodes():
        df = load_episode(path, args.source, stride=1)
        kin = episode_kinematics(df, source=args.source, side="left")
        ep_rows.append(aggregate_episode_features(kin, label, ep))
        frame_rows.append(frame_table(kin, label, ep))
        vc = pd.Series(kin["phases"]).value_counts()
        phase_counts.append(
            {"episode": ep, "label": label, **{p: int(vc.get(p, 0)) for p in ACTIVE_PHASES}}
        )
        print(f"ep {ep:03d} label={label} phases={dict(vc)}")

    feat_df = pd.DataFrame(ep_rows)
    frame_df = pd.concat(frame_rows, ignore_index=True)
    phase_df = pd.DataFrame(phase_counts)

    feat_path = args.out_dir / "episode_phase_features.csv"
    frame_path = args.out_dir / "frame_phase_kinematics.csv"
    phase_path = args.out_dir / "episode_phase_counts.csv"
    feat_df.to_csv(feat_path, index=False)
    frame_df.to_csv(frame_path, index=False)
    phase_df.to_csv(phase_path, index=False)
    print(f"Wrote {feat_path} shape={feat_df.shape}")
    print(f"Wrote {frame_path} shape={frame_df.shape}")
    print(f"Wrote {phase_path}")


if __name__ == "__main__":
    main()
