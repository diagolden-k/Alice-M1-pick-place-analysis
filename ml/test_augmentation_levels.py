"""Compare train-only trajectory augmentation levels: none / light / medium / heavy.

Design
------
Levels differ by copies-per-episode and transform strength (joint noise, time-warp,
temporal shift). Augmentations are tied to a parent episode id.

Leakage control
---------------
1) StratifiedKFold on the 48 *original* episodes.
2) Train = originals in fold ∪ synthetic rows whose parent is in train.
3) Test = originals only (never synthetic).

Features
--------
Re-extract phase bank + candidate + proposed physics features from each
(augmented) trajectory, then evaluate fixed physics17 and honest pool40.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT = Path("/workspace")
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "ml"))

from alice_arm_fk import ARM_JOINT_NAMES, HAND_JOINT_NAMES, arm_chain, forward_kinematics, hand_closure, stick_points  # noqa: E402
from build_phase_features import aggregate_episode_features, episode_kinematics, frame_table  # noqa: E402
from compare_new_features import engineer_candidates  # noqa: E402
from phases import estimate_phases  # noqa: E402
from visualize_arm_motion import DATA_ROOT, find_episode_csv, hand_cols, joint_cols, load_episode  # noqa: E402

OUT = ROOT / "ml" / "outputs" / "augmentation_levels"
OUT.mkdir(parents=True, exist_ok=True)
POOL_DEF = ROOT / "ml" / "outputs" / "expanded_physics_pool" / "pool_definitions.json"
CACHE = OUT / "augmented_feature_cache.csv"
RANDOM_STATE = 42

# Augmentation curricula -------------------------------------------------------
AUG_LEVELS = {
    "none": {"n_copies": 0, "joint_noise": 0.0, "time_warp": 0.0, "shift_frac": 0.0},
    "light": {"n_copies": 2, "joint_noise": 0.01, "time_warp": 0.05, "shift_frac": 0.02},
    "medium": {"n_copies": 4, "joint_noise": 0.02, "time_warp": 0.10, "shift_frac": 0.04},
    "heavy": {"n_copies": 8, "joint_noise": 0.04, "time_warp": 0.18, "shift_frac": 0.06},
}

MODELS = ["logreg", "svm_rbf", "svm_linear", "knn"]


def make_model(name: str):
    if name == "logreg":
        clf = LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)
    elif name == "svm_rbf":
        clf = SVC(C=3.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)
    elif name == "svm_linear":
        clf = SVC(C=1.0, kernel="linear", class_weight="balanced", random_state=RANDOM_STATE)
    else:
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("clf", clf)])


def state_action_cols() -> list[str]:
    cols = []
    for prefix in ("state", "action"):
        for side in ("l", "r"):
            cols.extend(joint_cols(prefix, side))
            cols.extend(hand_cols(prefix, side))
    return cols


def augment_trajectory(df: pd.DataFrame, rng: np.random.Generator, cfg: dict) -> pd.DataFrame:
    """Physics-light trajectory augmentation: noise + time-warp + shift."""
    out = df.copy().reset_index(drop=True)
    n = len(out)
    if n < 8:
        return out

    cols = [c for c in state_action_cols() if c in out.columns]

    # 1) temporal shift (crop then pad by edge)
    shift_frac = float(cfg["shift_frac"])
    if shift_frac > 0:
        max_shift = max(1, int(round(shift_frac * n)))
        s = int(rng.integers(-max_shift, max_shift + 1))
        if s > 0:
            chunk = out.iloc[s:].reset_index(drop=True)
            pad = pd.concat([out.iloc[[s]]] * s, ignore_index=True)
            out = pd.concat([chunk, pad], ignore_index=True).iloc[:n].reset_index(drop=True)
        elif s < 0:
            s = -s
            chunk = out.iloc[: n - s].reset_index(drop=True)
            pad = pd.concat([out.iloc[[0]]] * s, ignore_index=True)
            out = pd.concat([pad, chunk], ignore_index=True).iloc[:n].reset_index(drop=True)

    # 2) time warp via resampling onto warped grid, then back to n frames
    warp = float(cfg["time_warp"])
    if warp > 0:
        amt = float(rng.uniform(-warp, warp))
        x = np.linspace(0.0, 1.0, n)
        # monotonic warp: ease-in/out
        xw = np.clip(x + amt * np.sin(np.pi * x), 0.0, 1.0)
        xw = np.maximum.accumulate(xw)
        if xw[-1] <= xw[0]:
            xw = x
        xw = (xw - xw[0]) / max(1e-9, xw[-1] - xw[0])
        src = np.linspace(0.0, 1.0, n)
        warped = out.copy()
        for c in cols:
            warped[c] = np.interp(src, xw, out[c].to_numpy(float))
        if "timestamp" in warped.columns:
            t0, t1 = float(out["timestamp"].iloc[0]), float(out["timestamp"].iloc[-1])
            warped["timestamp"] = np.linspace(t0, t1, n)
        out = warped

    # 3) joint/hand Gaussian noise
    noise = float(cfg["joint_noise"])
    if noise > 0:
        for c in cols:
            out[c] = out[c].to_numpy(float) + rng.normal(0.0, noise, size=n)

    return out


def detect_release(cl: np.ndarray) -> tuple[int, int, int]:
    span = cl.max() - cl.min()
    hi = cl.min() + 0.55 * span if span > 1e-6 else np.quantile(cl, 0.7)
    lo = cl.min() + 0.25 * span if span > 1e-6 else np.quantile(cl, 0.3)
    closed = cl >= hi
    opened = cl <= lo
    close_i = 0
    for i in range(len(cl) - 3):
        if np.mean(closed[i : i + 3]) > 0.8:
            close_i = i
            break
    open_start = close_i
    for i in range(close_i + 5, len(cl) - 2):
        if cl[i] < hi and (cl[i] - cl[max(0, i - 1)]) < 0:
            open_start = i
            break
    open_end = open_start
    for i in range(open_start, len(cl)):
        if opened[i]:
            open_end = i
            break
    else:
        open_end = len(cl) - 1
    return close_i, open_start, open_end


def sparc_smoothness(speed: np.ndarray) -> float:
    s = speed[np.isfinite(speed)]
    if len(s) < 8:
        return np.nan
    s = s - np.mean(s)
    spec = np.fft.rfft(s)
    mag = np.abs(spec)
    mag = mag / (mag.sum() + 1e-12)
    mag = mag[mag > 0]
    return float(-np.sum(mag * np.log(mag + 1e-12)))


def proposed_from_kin(kin: dict, df: pd.DataFrame, label: int, episode: int) -> dict:
    t = kin["t"]
    palm = kin["palm"]
    cl = kin["closure"]
    q = kin["q"]
    speed = kin["speed"]
    jspeed = kin["joint_speed"]
    close_i, open_start, open_end = detect_release(cl)
    hand = df[hand_cols("state", "l")].to_numpy(float)
    vz = kin["v_p"][:, 2]
    pre = slice(max(0, open_start - 15), open_start)
    vz_pre = vz[pre]
    vz_descent = float(np.mean(vz_pre[vz_pre < 0])) if np.any(vz_pre < 0) else float(np.mean(vz_pre)) if len(vz_pre) else np.nan
    open_dur = float(t[open_end] - t[open_start])
    open_dur_ratio = open_dur / max(1e-3, float(t[-1] - t[0]))
    r0, r1 = open_end, min(len(t) - 1, open_end + 10)
    transport = slice(close_i, open_start)
    if transport.stop > transport.start:
        grasp_tight_mean = float(np.mean(np.abs(hand[transport])))
        grasp_tight_std = float(np.std(hand[transport]))
    else:
        grasp_tight_mean = grasp_tight_std = np.nan
    return {
        "episode": episode,
        "label": label,
        "prop_z_release": float(palm[open_end, 2]),
        "prop_tilt_pitch_release": float(q[open_end, 0] + q[open_end, 3] + q[open_end, 5]),
        "prop_tilt_roll_release": float(q[open_end, 6]),
        "prop_vz_descent": vz_descent,
        "prop_open_dur_ratio": open_dur_ratio,
        "prop_open_duration": open_dur,
        "prop_retreat_burst_speed": float(np.mean(speed[r0 : r1 + 1])),
        "prop_retreat_burst_vy": float(np.mean(kin["v_p"][r0 : r1 + 1, 1])),
        "prop_retreat_burst_jspeed": float(np.mean(jspeed[r0 : r1 + 1])),
        "prop_grasp_tight_mean": grasp_tight_mean,
        "prop_grasp_tight_std": grasp_tight_std,
        "prop_sparc_palm": sparc_smoothness(speed),
    }


def extract_feature_row(df: pd.DataFrame, episode: int, label: int) -> dict:
    kin = episode_kinematics(df, source="state", side="left")
    phase = aggregate_episode_features(kin, label, episode)
    frames = frame_table(kin, label, episode)
    cand = engineer_candidates(frames).iloc[0].to_dict()
    prop = proposed_from_kin(kin, df, label, episode)
    row = {}
    row.update(phase)
    row.update({k: v for k, v in cand.items() if k not in ("episode", "label")})
    row.update({k: v for k, v in prop.items() if k not in ("episode", "label")})
    row["episode"] = episode
    row["label"] = label
    return row


def list_episodes() -> list[tuple[int, int, Path]]:
    rows = []
    for split, label in (("success", 1), ("fail", 0)):
        for path in sorted((DATA_ROOT / split).glob("episode_*.csv")):
            ep = int(path.stem.split("_")[1])
            rows.append((ep, label, path))
    return sorted(rows, key=lambda x: x[0])


def build_or_load_cache(force: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not force:
        print(f"Loading cache {CACHE}")
        return pd.read_csv(CACHE)

    rows = []
    episodes = list_episodes()
    print(f"Building augmented feature cache for {len(episodes)} episodes × levels {list(AUG_LEVELS)}")
    for ep, label, path in episodes:
        base_df = load_episode(path, "state", stride=1)
        # original
        feat = extract_feature_row(base_df, ep, label)
        feat.update({"aug_level": "none", "parent_episode": ep, "is_synthetic": 0, "aug_id": 0})
        rows.append(feat)
        print(f"  ep {ep:03d} original")

        for level, cfg in AUG_LEVELS.items():
            if level == "none":
                continue
            rng = np.random.default_rng(RANDOM_STATE + ep * 100 + hash(level) % 97)
            for i in range(int(cfg["n_copies"])):
                aug_df = augment_trajectory(base_df, rng, cfg)
                feat = extract_feature_row(aug_df, ep, label)
                feat.update(
                    {
                        "aug_level": level,
                        "parent_episode": ep,
                        "is_synthetic": 1,
                        "aug_id": i + 1,
                    }
                )
                rows.append(feat)
            print(f"  ep {ep:03d} level={level} copies={cfg['n_copies']}")

    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    print(f"Wrote {CACHE} shape={df.shape}")
    return df


def cohens_d_abs(x, y):
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


def corr_prune(X, d, k, thr=0.70):
    Xc = X.copy()
    for j in range(Xc.shape[1]):
        col = Xc[:, j]
        m = np.isfinite(col)
        Xc[:, j] = np.where(m, col, np.nanmedian(col[m]) if m.any() else 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(Xc.T)
    corr = np.nan_to_num(corr, nan=0.0)
    kept = []
    for j in np.argsort(d)[::-1]:
        if len(kept) >= k:
            break
        if any(abs(corr[j, i]) >= thr for i in kept):
            continue
        kept.append(int(j))
    return kept


def eval_level(
    cache: pd.DataFrame,
    level: str,
    feat_cols: list[str],
    model_name: str,
    feat_mode: str,
    n_splits: int = 5,
) -> dict:
    """feat_mode: fixed | honest_corr070"""
    originals = cache[cache["is_synthetic"] == 0].sort_values("episode").reset_index(drop=True)
    y_all = originals["label"].to_numpy(int)
    parent_ids = originals["episode"].to_numpy(int)

    # rows usable for this level: originals always + synthetics of this level
    if level == "none":
        level_rows = cache[cache["is_synthetic"] == 0].copy()
    else:
        level_rows = cache[(cache["is_synthetic"] == 0) | (cache["aug_level"] == level)].copy()

    cv = StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.full(len(originals), -1)
    n_train_rows = []

    for tr_idx, te_idx in cv.split(originals, y_all):
        train_parents = set(parent_ids[tr_idx].tolist())
        test_parents = set(parent_ids[te_idx].tolist())

        train_df = level_rows[level_rows["parent_episode"].isin(train_parents)]
        # test: originals only
        test_df = originals.iloc[te_idx]

        Xtr_full = train_df[feat_cols].to_numpy(float)
        ytr = train_df["label"].to_numpy(int)
        Xte_full = test_df[feat_cols].to_numpy(float)
        yte = test_df["label"].to_numpy(int)

        if feat_mode == "fixed":
            idx = list(range(len(feat_cols)))
        else:
            # honest select using TRAIN rows only (includes augs)
            d = np.array([cohens_d_abs(Xtr_full[:, j], ytr) for j in range(Xtr_full.shape[1])])
            idx = corr_prune(Xtr_full, d, k=min(15, Xtr_full.shape[1]), thr=0.70)

        m = make_model(model_name)
        m.fit(Xtr_full[:, idx], ytr)
        pred = m.predict(Xte_full[:, idx])
        y_pred[te_idx] = pred
        n_train_rows.append(len(train_df))

    return {
        "aug_level": level,
        "model": model_name,
        "feat_mode": feat_mode,
        "accuracy": float(accuracy_score(y_all, y_pred)),
        "f1": float(f1_score(y_all, y_pred)),
        "mean_train_rows": float(np.mean(n_train_rows)),
        "n_test_originals": int(len(originals)),
        "n_synth_available": int(((cache["aug_level"] == level) & (cache["is_synthetic"] == 1)).sum()) if level != "none" else 0,
    }


def main():
    pools = json.loads(POOL_DEF.read_text())
    pool17 = pools["pool17"]
    pool40 = pools["pool40"]

    cache = build_or_load_cache(force=False)
    # ensure feature columns exist
    for name, cols in [("pool17", pool17), ("pool40", pool40)]:
        missing = [c for c in cols if c not in cache.columns]
        if missing:
            raise RuntimeError(f"Missing {name} cols: {missing[:10]}")

    # sanity: class balance of synthetics
    bal_rows = []
    for level in AUG_LEVELS:
        if level == "none":
            sub = cache[cache["is_synthetic"] == 0]
        else:
            sub = cache[cache["aug_level"] == level]
        bal_rows.append(
            {
                "aug_level": level,
                "n_rows": len(sub),
                "n_success": int((sub["label"] == 1).sum()),
                "n_fail": int((sub["label"] == 0).sum()),
                "success_frac": float((sub["label"] == 1).mean()),
            }
        )
    bal = pd.DataFrame(bal_rows)
    bal.to_csv(OUT / "augmentation_balance.csv", index=False)

    results = []
    print("\n=== Train/eval by augmentation level ===")
    for level in AUG_LEVELS:
        for model in MODELS:
            for feat_mode, cols, tag in [
                ("fixed", pool17, "fixed17"),
                ("honest_corr070", pool40, "honest40"),
            ]:
                r = eval_level(cache, level, cols, model, feat_mode if feat_mode != "fixed" else "fixed")
                r["feat_tag"] = tag
                results.append(r)
                print(
                    f"{level:7s} {model:11s} {tag:9s}  acc={r['accuracy']:.3f}  "
                    f"f1={r['f1']:.3f}  train_rows≈{r['mean_train_rows']:.0f}"
                )

    res = pd.DataFrame(results)
    res.to_csv(OUT / "augmentation_level_results.csv", index=False)

    # aggregates
    by_level = (
        res.groupby(["aug_level", "feat_tag"], as_index=False)
        .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"), f1_mean=("f1", "mean"))
    )
    # keep level order
    order = list(AUG_LEVELS.keys())
    by_level["aug_level"] = pd.Categorical(by_level["aug_level"], categories=order, ordered=True)
    by_level = by_level.sort_values(["feat_tag", "aug_level"])
    by_level.to_csv(OUT / "augmentation_by_level.csv", index=False)

    by_lm = (
        res.groupby(["aug_level", "model", "feat_tag"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), f1=("f1", "mean"), mean_train_rows=("mean_train_rows", "mean"))
    )
    by_lm["aug_level"] = pd.Categorical(by_lm["aug_level"], categories=order, ordered=True)
    by_lm = by_lm.sort_values(["feat_tag", "model", "aug_level"])
    by_lm.to_csv(OUT / "augmentation_by_level_model.csv", index=False)

    # delta vs none
    deltas = []
    base = res[res["aug_level"] == "none"].set_index(["model", "feat_tag"])["accuracy"]
    for _, row in res[res["aug_level"] != "none"].iterrows():
        key = (row["model"], row["feat_tag"])
        deltas.append(
            {
                "aug_level": row["aug_level"],
                "model": row["model"],
                "feat_tag": row["feat_tag"],
                "accuracy": row["accuracy"],
                "delta_vs_none": float(row["accuracy"] - base.loc[key]),
            }
        )
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(OUT / "augmentation_delta_vs_none.csv", index=False)

    summary = {
        "levels": AUG_LEVELS,
        "balance": bal.to_dict("records"),
        "by_level": by_level.to_dict("records"),
        "best_overall": res.sort_values("accuracy", ascending=False).head(8).to_dict("records"),
        "mean_delta_vs_none": delta_df.groupby("aug_level")["delta_vs_none"].mean().to_dict(),
        "protocol": [
            "Split originals first (Stratified 5-fold).",
            "Add synthetics only if parent episode is in the train fold.",
            "Evaluate on original test episodes only.",
        ],
    }
    (OUT / "augmentation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, tag in zip(axes, ["fixed17", "honest40"]):
        sub = res[res["feat_tag"] == tag]
        for model in MODELS:
            s = sub[sub["model"] == model].set_index("aug_level").reindex(order)
            ax.plot(range(len(order)), s["accuracy"], "-o", label=model)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.set_title(tag)
        ax.set_xlabel("augmentation level")
        ax.legend(fontsize=8)
        ax.set_ylim(0.55, 1.0)
        ax.axhline(sub[sub.aug_level == "none"]["accuracy"].mean(), color="gray", ls=":", lw=1)
    axes[0].set_ylabel("accuracy (original test only)")
    fig.suptitle("Train-only trajectory augmentation vs none/light/medium/heavy")
    fig.tight_layout()
    fig.savefig(OUT / "plot_aug_levels_accuracy.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for tag, ls in [("fixed17", "-o"), ("honest40", "-s")]:
        s = by_level[by_level["feat_tag"] == tag].set_index("aug_level").reindex(order)
        ax.plot(range(len(order)), s["acc_mean"], ls, label=tag)
        ax.fill_between(
            range(len(order)),
            s["acc_mean"] - s["acc_std"].fillna(0),
            s["acc_mean"] + s["acc_std"].fillna(0),
            alpha=0.15,
        )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("mean accuracy ± std over models")
    ax.set_title("Augmentation level effect (mean over models)")
    ax.legend()
    ax.set_ylim(0.55, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "plot_aug_levels_mean.png", dpi=140)
    plt.close(fig)

    md = [
        "# Trajectory augmentation level comparison",
        "",
        "## Levels",
        "",
        json.dumps(AUG_LEVELS, indent=2),
        "",
        "## Balance",
        "",
        bal.to_string(index=False),
        "",
        "## Mean by level",
        "",
        by_level.to_string(index=False),
        "",
        "## Per model",
        "",
        by_lm.to_string(index=False),
        "",
        "## Delta vs none",
        "",
        delta_df.to_string(index=False),
        "",
    ]
    (OUT / "augmentation_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\nMean by level:")
    print(by_level.to_string(index=False))
    print("\nMean delta vs none:")
    print(delta_df.groupby("aug_level")["delta_vs_none"].agg(["mean", "std"]))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
