"""Score a new episode CSV with the selected classifier / metric.

Example:
  python ml/predict_episode.py --csv path/to/episode.csv
  python ml/predict_episode.py --episode 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))

from build_phase_features import aggregate_episode_features, episode_kinematics  # noqa: E402
from visualize_arm_motion import find_episode_csv, load_episode  # noqa: E402

OUT = ROOT / "ml" / "outputs"


def features_from_csv(path: Path, source: str = "state") -> pd.Series:
    df = load_episode(path, source, stride=1)
    kin = episode_kinematics(df, source=source, side="left")
    # label unknown
    feats = aggregate_episode_features(kin, label=-1, episode=-1)
    return pd.Series(feats)


def quality_from_row(row: pd.Series, metric: dict) -> float:
    acc = 0.0
    for feat, w in metric["weights_normalized"].items():
        term = next(t for t in metric["terms"] if t["feature"] == feat)
        x = float(row[feat])
        lo, hi = term["q05"], term["q95"]
        u = (np.clip(x, lo, hi) - lo) / (hi - lo if hi > lo else 1.0)
        if term["direction"] == "invert":
            u = 1.0 - u
        acc += w * u
    return float(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--source", default="state")
    args = ap.parse_args()

    if args.csv is None and args.episode is None:
        raise SystemExit("Provide --episode or --csv")
    path = args.csv if args.csv is not None else find_episode_csv(args.episode)

    bundle = joblib.load(OUT / "best_classifier.joblib")
    metric = json.loads((OUT / "quality_metric.json").read_text(encoding="utf-8"))
    model = bundle["model"]
    feats = bundle["features"]

    row = features_from_csv(path, source=args.source)
    X = row[feats].to_numpy(dtype=float).reshape(1, -1)
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0, 1])
    base = quality_from_row(row, metric)
    score = 100.0 * (0.75 * base + 0.25 * prob)

    print(f"file: {path}")
    print(f"pred_label: {pred} ({'success' if pred == 1 else 'fail'})")
    print(f"P_success: {prob:.3f}")
    print(f"quality_0_100: {score:.1f}")


if __name__ == "__main__":
    main()
