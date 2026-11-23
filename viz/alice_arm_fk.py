"""Alice M1 dual-arm forward kinematics (torso_link frame).

Joint origins/axes match alice_m1.urdf. Waist/base joints are not in the
CSV, so all poses are relative to torso_link (shoulder attachment frame).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

ARM_JOINT_NAMES = ("sh_p", "sh_r", "sh_y", "el_p", "wr_y", "wr_p", "wr_r")
HAND_JOINT_NAMES = (
    "hand_index",
    "hand_middle",
    "hand_pinky",
    "hand_ring",
    "hand_thumb_y",
    "hand_thumb_p",
)


@dataclass(frozen=True)
class JointSpec:
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    fixed: bool = False


# URDF left/right arm chains (parent = torso_link for first joint).
LEFT_ARM: tuple[JointSpec, ...] = (
    JointSpec((0.0, 0.24, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # l_sh_p
    JointSpec((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),  # l_sh_r
    JointSpec((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # l_sh_y
    JointSpec((0.0, 0.0, -0.26), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # l_el_p
    JointSpec((0.0, 0.0, -0.255), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # l_wr_y
    JointSpec((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # l_wr_p
    JointSpec((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),  # l_wr_r
    JointSpec((0.0, 0.0, -0.077), (0.0, 3.1416, 1.5708), (1.0, 0.0, 0.0), fixed=True),
)

RIGHT_ARM: tuple[JointSpec, ...] = (
    JointSpec((0.0, -0.24, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # r_sh_p
    JointSpec((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),  # r_sh_r
    JointSpec((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # r_sh_y
    JointSpec((0.0, 0.0, -0.26), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # r_el_p
    JointSpec((0.0, 0.0, -0.255), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # r_wr_y
    JointSpec((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # r_wr_p
    JointSpec((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),  # r_wr_r
    JointSpec((0.0, 0.0, -0.077), (0.0, 3.1416, -1.5708), (1.0, 0.0, 0.0), fixed=True),
)

# Stick-figure indices into the pose chain (after each joint transform).
# 0: torso, 1: shoulder, 4: elbow, 5: wrist_yaw, 7: wrist_roll, 8: palm
STICK_INDICES = (0, 1, 4, 5, 7, 8)
STICK_LABELS = ("torso", "shoulder", "elbow", "wrist", "wrist_roll", "palm")


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _axis_angle_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    ax = np.asarray(axis, dtype=float)
    n = np.linalg.norm(ax)
    if n < 1e-12:
        return np.eye(3)
    x, y, z = ax / n
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def _homog(rot: np.ndarray, xyz: Sequence[float]) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = np.asarray(xyz, dtype=float)
    return t


def joint_transform(spec: JointSpec, q: float) -> np.ndarray:
    """URDF joint: T = Trans*RPY(origin) * Rot(axis, q)."""
    t_origin = _homog(_rpy_matrix(*spec.rpy), spec.xyz)
    angle = 0.0 if spec.fixed else float(q)
    t_joint = _homog(_axis_angle_matrix(spec.axis, angle), (0.0, 0.0, 0.0))
    return t_origin @ t_joint


def forward_kinematics(
    chain: Sequence[JointSpec],
    joint_angles: Sequence[float],
) -> np.ndarray:
    """Return (n_joints+1, 3) points: torso + after each joint.

    `joint_angles` length must equal number of non-fixed joints in chain.
    """
    q_iter = iter(joint_angles)
    t = np.eye(4)
    points = [t[:3, 3].copy()]
    for spec in chain:
        q = 0.0 if spec.fixed else next(q_iter)
        t = t @ joint_transform(spec, q)
        points.append(t[:3, 3].copy())
    # Ensure all movable joints were consumed.
    exhausted = object()
    if next(q_iter, exhausted) is not exhausted:
        raise ValueError("Too many joint angles for chain")
    return np.vstack(points)


def arm_chain(side: str) -> tuple[JointSpec, ...]:
    side = side.lower()
    if side in ("l", "left"):
        return LEFT_ARM
    if side in ("r", "right"):
        return RIGHT_ARM
    raise ValueError(f"Unknown side: {side}")


def stick_points(points: np.ndarray) -> np.ndarray:
    return points[list(STICK_INDICES)]


def hand_closure(hand_values: Iterable[float]) -> float:
    """Scalar grasp proxy: mean of finger/thumb command magnitudes."""
    vals = np.asarray(list(hand_values), dtype=float)
    return float(np.mean(np.abs(vals))) if vals.size else 0.0
