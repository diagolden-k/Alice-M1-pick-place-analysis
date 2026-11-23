"""Pick-and-place phase segmentation for Alice M1 left-arm trajectories."""

from __future__ import annotations

import numpy as np

PHASE_NAMES = (
    "approach",
    "grasp",
    "lift",
    "carry_right",
    "place",
    "retreat",
    "other",
)

PHASE_COLORS = {
    "approach": "#4C78A8",
    "grasp": "#F58518",
    "lift": "#54A24B",
    "carry_right": "#B279A2",
    "place": "#E45756",
    "retreat": "#72B7B2",
    "other": "#9E9E9E",
}


def _smooth(x: np.ndarray, win: int = 7) -> np.ndarray:
    if len(x) < win:
        return np.asarray(x, dtype=float).copy()
    kern = np.ones(win) / win
    return np.convolve(np.asarray(x, dtype=float), kern, mode="same")


def _rising_edge(x: np.ndarray, thr: float, persist: int = 5) -> int:
    for i in range(0, len(x) - persist):
        if x[i] < thr and np.all(x[i + 1 : i + 1 + persist] >= thr):
            return i + 1
    # fallback: first above threshold
    idx = np.where(x >= thr)[0]
    return int(idx[0]) if len(idx) else len(x) // 3


def _falling_edge(x: np.ndarray, thr: float, start: int, persist: int = 5) -> int:
    for i in range(start, len(x) - persist):
        if x[i] >= thr and np.all(x[i + 1 : i + 1 + persist] < thr):
            return i + 1
    idx = np.where(x[start:] < thr)[0]
    return int(start + idx[0]) if len(idx) else len(x) - 1


def estimate_phases(palm: np.ndarray, closure: np.ndarray, timestamps: np.ndarray) -> list[str]:
    """Segment left-arm pick-and-place in torso frame.

    Axes: +X forward, +Y left, +Z up.
    Sequence: approach -> grasp -> lift -> carry_right -> place
    (descend while opening) -> retreat (open hand, clear left / +Y).
    """
    t = len(palm)
    if t < 12:
        return ["other"] * t

    py = _smooth(palm[:, 1])
    pz = _smooth(palm[:, 2])
    cl = _smooth(closure)

    ts = np.asarray(timestamps, dtype=float)
    if len(ts) != t or np.any(np.diff(ts) <= 0):
        ts = np.arange(t, dtype=float) / 30.0

    vy = np.gradient(py, ts)
    vz = np.gradient(pz, ts)

    cl_min, cl_max = float(np.min(cl)), float(np.max(cl))
    span = cl_max - cl_min
    if span < 1e-3:
        # Almost no grasp signal: split by kinematics only
        z_peak = int(np.argmax(pz))
        thirds = [t // 6, t // 3, z_peak, (z_peak + t) // 2, int(t * 0.85)]
        cuts = [int(np.clip(c, 1, t - 1)) for c in thirds]
        for i in range(1, len(cuts)):
            cuts[i] = max(cuts[i], cuts[i - 1] + 1)
        names = ["approach", "grasp", "lift", "carry_right", "place", "retreat"]
        phases = []
        bounds = [0, *cuts, t]
        for i in range(t):
            for k in range(len(names)):
                if bounds[k] <= i < bounds[k + 1]:
                    phases.append(names[k])
                    break
        return phases

    hi = cl_min + 0.55 * span
    lo = cl_min + 0.25 * span

    grasp_i = _rising_edge(cl, hi, persist=max(3, t // 80))
    release_i = _falling_edge(cl, lo, start=min(t - 2, grasp_i + max(8, t // 20)), persist=max(3, t // 80))
    if release_i <= grasp_i + 5:
        release_i = min(t - 1, int(grasp_i + 0.55 * (t - grasp_i)))

    # Lift / carry around height peak while grasped
    grasp_window = slice(grasp_i, max(grasp_i + 1, release_i))
    z_peak_i = grasp_i + int(np.argmax(pz[grasp_window]))
    z_peak_i = int(np.clip(z_peak_i, grasp_i + 1, release_i))

    lift_h = float(pz[z_peak_i] - pz[grasp_i])
    drop_thr = pz[z_peak_i] - max(0.02, 0.20 * abs(lift_h))

    # Place starts when descending after peak (or at release if earlier)
    descent_i = release_i
    for i in range(z_peak_i, release_i + 1):
        if pz[i] <= drop_thr and vz[i] < 0:
            descent_i = i
            break
    place_i = int(min(descent_i, release_i))
    place_i = int(np.clip(place_i, z_peak_i, release_i))

    # Short grasp hold right after closure onset
    grasp_end = int(np.clip(grasp_i + max(6, int(0.04 * t)), grasp_i + 1, z_peak_i))

    # Retreat: after release, open hand clears to the left (+Y)
    retreat_i = release_i
    for i in range(release_i, t - 1):
        if cl[i] <= hi and vy[i] > 0.01:
            retreat_i = i
            break
    else:
        retreat_i = min(t - 1, release_i + max(3, (t - release_i) // 3))

    # If place would be empty, keep a few frames before retreat
    if retreat_i <= place_i:
        retreat_i = min(t - 1, place_i + max(3, (t - place_i) // 4))
    if retreat_i <= release_i and release_i < t - 1:
        # place covers release..retreat, retreat the rest
        retreat_i = max(retreat_i, min(t - 1, release_i + max(2, (t - release_i) // 5)))

    # Enforce strict increasing cuts
    grasp_i = int(np.clip(grasp_i, 1, t - 6))
    grasp_end = int(np.clip(grasp_end, grasp_i + 1, t - 5))
    z_peak_i = int(np.clip(z_peak_i, grasp_end + 1, t - 4))
    place_i = int(np.clip(place_i, z_peak_i, t - 3))
    retreat_i = int(np.clip(retreat_i, place_i + 1, t - 1))

    # Guarantee carry has at least 1 frame when possible
    if place_i == z_peak_i and place_i < t - 3:
        place_i = min(place_i + 1, retreat_i - 1)

    phases: list[str] = []
    for i in range(t):
        if i < grasp_i:
            phases.append("approach")
        elif i < grasp_end:
            phases.append("grasp")
        elif i < z_peak_i:
            phases.append("lift")
        elif i < place_i:
            phases.append("carry_right")
        elif i < retreat_i:
            phases.append("place")
        else:
            phases.append("retreat")
    return phases
