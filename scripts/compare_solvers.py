"""Side-by-side comparison of the tracker variants (pose vs pixel measurement, EKF vs iterated EKF)
and of the old reactive aimer vs the attack-window aimer.

Inputs: out/runs_<variant>/*.jsonl from out/batch_variants.ps1, out/runs/*.jsonl (default config), and
out/runs_old_aimer/*.jsonl (the previous reactive aimer).
Outputs: docs/report/12_solver_comparison.png, 13_solver_timeseries.png, 14_aimer_window.png,
docs/report/solver_comparison.md.

The fair accuracy metric is pixel prediction error: each variant's estimate from an earlier frame is
propagated forward, its plate is projected into the later frame and compared with the detected
corners. It needs no ground truth and favors neither measurement space.
"""
import json
import pathlib

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).parent.parent
REPORT = BASE / "docs" / "report"
VARIANTS = ["pose_ekf", "pose_iekf", "pixel_ekf", "pixel_iekf"]
LABELS = {"pose_ekf": "pose, EKF", "pose_iekf": "pose, iterated", "pixel_ekf": "pixel, EKF", "pixel_iekf": "pixel, iterated"}
COLORS = {"pose_ekf": "#9ecae1", "pose_iekf": "#3182bd", "pixel_ekf": "#fdae6b", "pixel_iekf": "#e6550d", "old": "#999999"}
SPINNING = ["spinning_2m", "spinning_3m", "spinning_4m", "spinning_5m"]
MOVING = SPINNING + ["translate", "translate_spinning"]
STATIC = ["static_2m", "static_2m_angled", "static_3m", "static_4m", "static_5m", "static_6m"]
HORIZONS = [("next frame", None), ("100 ms", 0.1), ("300 ms", 0.3)]
TRUTH_R, TRUTH_H = 0.25, 0.08
DOF = {"pose": 4, "pixel": 8}

robot = yaml.safe_load((BASE / "configs" / "robots" / "realsense_test.yaml").read_text(encoding="utf-8-sig"))
FX, FY, CX, CY = robot["camera"]["k"]
TILT = np.radians(15.0)
WIDTH = {"hero": 0.230, "basebottom": 0.230}


def load(variant, video):
    path = BASE / "out" / variant / f"{video}.jsonl"
    return [json.loads(line) for line in open(path)] if path.exists() else None


def plate(x, i, n, dt=0.0):
    cx, vx, cy, vy, cz, vz, yaw, w, r, l, h = x
    odd = i % 2 == 1 and n == 4
    a = yaw + w * dt + i * 2 * np.pi / n
    ri = r + (l if odd else 0.0)
    return np.array([cx + vx * dt - ri * np.cos(a), cy + vy * dt - ri * np.sin(a), cz + vz * dt + (h if odd else 0.0)]), a


def project(p):
    """World (x forward, y left, z up), level camera at the origin, no distortion."""
    return np.array([FX * -p[1] / p[0] + CX, FY * -p[2] / p[0] + CY])


def corners_px(center, yaw, icon):
    w, hh = WIDTH.get(icon, 0.135) / 2, 0.056 / 2
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    ry = np.array([[np.cos(TILT), 0, np.sin(TILT)], [0, 1, 0], [-np.sin(TILT), 0, np.cos(TILT)]])
    return np.array([project(center + rz @ ry @ np.array(c)) for c in [(0, w, hh), (0, w, -hh), (0, -w, -hh), (0, -w, hh)]])


def detected_corners(row, obs_position):
    target = project(np.array(obs_position))
    dets = [np.array(d["corners"]) for d in row["detections"]]
    return min(dets, key=lambda c: np.linalg.norm(c.mean(0) - target)) if dets else None


def prediction_errors(rows, horizon):
    """(t, pixel error, 3D error) for every plate update, predicted from an earlier frame of the same track."""
    times = np.array([r["t"] for r in rows])
    out = []
    for j, row in enumerate(rows):
        est = row["record"]["estimate"]
        if not est or not est["updates"]:
            continue
        k = j - 1 if horizon is None else int(np.argmin(np.abs(times - (row["t"] - horizon))))
        prev = rows[k]["record"]["estimate"] if k >= 0 else None
        if k < 0 or k == j or not prev or est["frames_seen"] - prev["frames_seen"] != j - k:
            continue
        if horizon is not None and abs(row["t"] - rows[k]["t"] - horizon) > 0.01:
            continue
        for u in est["updates"]:
            det = detected_corners(row, u["observation"]["position"])
            center, yaw = plate(prev["x"], u["id"], prev["plate_count"], row["t"] - prev["t"])
            if det is None or center[0] <= 0.1:
                continue
            px = np.linalg.norm(corners_px(center, yaw, est["icon"]) - det, axis=1).mean()
            out.append((row["t"], px, np.linalg.norm(center - np.array(u["observation"]["position"]))))
    return np.array(out).reshape(-1, 3)


def nis_stats(rows, space):
    ups = [u for r in rows if r["record"]["estimate"] for u in r["record"]["estimate"]["updates"]]
    nis = np.array([u["nis"] for u in ups])
    return np.median(nis) / DOF[space], np.mean([u["accepted"] for u in ups])


def geometry(rows):
    xs = np.array([r["record"]["estimate"]["x"] for r in rows if r["record"]["estimate"] and r["record"]["estimate"]["mode"] == "Robot"])
    if len(xs) < 30:
        return None
    xs = xs[len(xs) // 3:]
    return {"r": np.median(xs[:, 8]), "l": np.median(xs[:, 9]), "h": np.median(np.abs(xs[:, 10])), "w": np.median(np.abs(xs[:, 7])), "w_std": np.std(xs[:, 7])}


def static_jitter(rows):
    """Frame-to-frame spread of the estimated plate 0 position, mm (static robot)."""
    ps = np.array([plate(r["record"]["estimate"]["x"], 0, 4)[0] for r in rows if r["record"]["estimate"] and r["record"]["estimate"]["state"] != "Tentative"])
    ps = ps[len(ps) // 3:]
    return 1000 * np.linalg.norm(np.std(ps, axis=0)) if len(ps) > 10 else np.nan


def aimer_stats(rows):
    ticks = [(c, r["record"]["target"]) for r in rows for c in r["record"]["commands"] if r["record"]["estimate"]]
    if not ticks:
        return None
    fire = np.mean([c["fire"] for c, _ in ticks])
    acc = max(abs(c["yaw_acc"]) for c, _ in ticks)
    kinds = [t.get("kind", "Plate") for _, t in ticks if t]
    return {"fire": fire, "peak_acc": acc, **{k: kinds.count(k) / max(len(kinds), 1) for k in ["Plate", "PreAim", "Center"]}}


def main():
    data = {v: {video: load(f"runs_{v}", video) for video in MOVING + STATIC} for v in VARIANTS}
    data["old"] = {video: load("runs_old_aimer", video) for video in MOVING + STATIC}
    data["new"] = {video: load("runs", video) for video in MOVING + STATIC}

    pred = {v: {name: np.concatenate([prediction_errors(data[v][vid], h) for vid in MOVING if data[v][vid]]) for name, h in HORIZONS} for v in VARIANTS}
    geo = {v: {vid: geometry(data[v][vid]) for vid in SPINNING if data[v][vid]} for v in VARIANTS}
    nis = {v: {vid: nis_stats(data[v][vid], v.split("_")[0]) for vid in MOVING + STATIC if data[v][vid]} for v in VARIANTS}
    jitter = {v: {vid: static_jitter(data[v][vid]) for vid in STATIC if data[v][vid]} for v in VARIANTS}
    aim = {v: {vid: aimer_stats(data[v][vid]) for vid in MOVING + STATIC if data[v][vid]} for v in ["old", "new"]}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    width = 0.2
    ax = axes[0, 0]
    for i, v in enumerate(VARIANTS):
        ax.bar(np.arange(3) + (i - 1.5) * width, [np.median(pred[v][n][:, 1]) for n, _ in HORIZONS], width, color=COLORS[v], label=LABELS[v])
    ax.set_xticks(range(3), [f"{n} (median)" for n, _ in HORIZONS])
    ax.set_ylabel("predicted vs detected corners (px)")
    ax.set_title("Prediction error in the image, spinning + translate videos")
    ax.legend()
    ax = axes[0, 1]
    for i, v in enumerate(VARIANTS):
        ax.bar(np.arange(3) + (i - 1.5) * width, [np.percentile(pred[v][n][:, 1], 90) for n, _ in HORIZONS], width, color=COLORS[v])
    ax.set_xticks(range(3), [f"{n} (p90)" for n, _ in HORIZONS])
    ax.set_ylabel("px")
    ax.set_title("Prediction error, 90th percentile")
    ax = axes[1, 0]
    for i, v in enumerate(VARIANTS):
        vids = [vid for vid in SPINNING if geo[v].get(vid)]
        ax.plot([geo[v][vid]["r"] for vid in vids], "o-", color=COLORS[v], label=f"{LABELS[v]} r")
        ax.plot([geo[v][vid]["h"] for vid in vids], "s--", color=COLORS[v], label=f"{LABELS[v]} |h|")
    ax.axhline(TRUTH_R, color="black", lw=0.8)
    ax.axhline(TRUTH_H, color="black", lw=0.8)
    ax.text(0, TRUTH_R + 0.004, "true r 0.25", fontsize=8)
    ax.text(0, TRUTH_H + 0.004, "true |h| 0.08", fontsize=8)
    ax.set_xticks(range(len(SPINNING)), SPINNING)
    ax.set_ylabel("m")
    ax.set_title("Robot geometry (robot mode, last 2/3 of each video)")
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1, 1]
    for i, v in enumerate(VARIANTS):
        ax.bar(np.arange(len(STATIC)) + (i - 1.5) * width, [jitter[v].get(vid, np.nan) for vid in STATIC], width, color=COLORS[v])
    ax.set_xticks(range(len(STATIC)), STATIC, rotation=20)
    ax.set_ylabel("mm")
    ax.set_title("Static plate: spread of the estimated position")
    fig.tight_layout()
    fig.savefig(REPORT / "12_solver_comparison.png", dpi=130, facecolor="white")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    rows = data["pose_iekf"]["translate_spinning"]
    for v in ["pose_iekf", "pixel_iekf"]:
        e = prediction_errors(data[v]["translate_spinning"], 0.1)
        axes[0].scatter(e[:, 0], e[:, 1], s=5, color=COLORS[v], label=LABELS[v], alpha=0.7)
        xs = [(r["t"], r["record"]["estimate"]["x"][8]) for r in data[v]["translate_spinning"] if r["record"]["estimate"]]
        axes[1].plot(*zip(*xs), color=COLORS[v], label=f"{LABELS[v]} r")
    dist = [(r["t"], np.hypot(*np.array(r["record"]["estimate"]["x"])[[0, 2]])) for r in rows if r["record"]["estimate"]]
    twin = axes[0].twinx()
    twin.plot(*zip(*dist), color="black", lw=0.8, label="center distance")
    twin.set_ylabel("center distance (m)")
    axes[0].set_ylim(0, 15)
    axes[0].set_ylabel("100 ms prediction error (px)")
    axes[0].legend(loc="upper left", markerscale=3)
    axes[0].set_title("translate_spinning: pose vs pixel measurement, both iterated")
    axes[1].axhline(TRUTH_R, color="black", lw=0.8)
    axes[1].set_ylim(0.1, 0.4)
    axes[1].set_ylabel("radius r (m)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(REPORT / "13_solver_timeseries.png", dpi=130, facecolor="white")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    for ax, vid, span in [(axes[0], "translate_spinning", (6.0, 8.0)), (axes[1], "spinning_3m", (4.0, 6.0))]:
        ticks = [(c["t"], c["yaw"], r["record"]["target"]) for r in data["new"][vid] for c in r["record"]["commands"] if r["record"]["target"]]
        ticks = [t for t in ticks if span[0] <= t[0] <= span[1]]
        t = np.array([x[0] for x in ticks])
        ax.plot(t, [np.degrees(x[1]) for x in ticks], color="black", lw=1.2, label="gimbal command")
        ax.plot(t, [np.degrees(x[2]["shot"]["yaw"]) for x in ticks], ".", ms=2, color="#3182bd", label="aim point")
        for kind, color in [("PreAim", "#fdae6b"), ("Center", "#bcbddc")]:
            for x in ticks:
                if x[2]["kind"] == kind:
                    ax.axvspan(x[0] - 0.0025, x[0] + 0.0025, color=color, alpha=0.4, lw=0)
        fires = [c["t"] for r in data["new"][vid] for c in r["record"]["commands"] if c["fire"] and span[0] <= c["t"] <= span[1]]
        lo = ax.get_ylim()[0]
        ax.plot(fires, np.full(len(fires), lo), "|", color="#d62728", ms=8, label="fire")
        ax.set_ylabel("yaw (deg)")
        ax.set_title(f"{vid}: orange = pre-aim on the window edge, purple = center aim, white = following a plate")
        ax.legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(REPORT / "14_aimer_window.png", dpi=130, facecolor="white")

    lines = ["# Solver and aimer comparison", "", "Generated by `scripts/compare_solvers.py`.", "",
             "## Tracker variants", "",
             "Prediction error: each variant's estimate is propagated from an earlier frame and its plate is projected onto the later frame's detected corners. Spinning and translate videos.", "",
             "| Variant | Next frame, median px | 100 ms, median px | 300 ms, median px | 300 ms, p90 px | 300 ms, median 3D cm | NIS median / dof | Accepted |",
             "|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        n = np.median([x[0] for x in nis[v].values()])
        a = np.mean([x[1] for x in nis[v].values()])
        p = pred[v]
        lines.append(f"| {LABELS[v]} | {np.median(p['next frame'][:, 1]):.2f} | {np.median(p['100 ms'][:, 1]):.2f} | {np.median(p['300 ms'][:, 1]):.2f} | {np.percentile(p['300 ms'][:, 1], 90):.2f} | {100 * np.median(p['300 ms'][:, 2]):.1f} | {n:.2f} | {100 * a:.1f}% |")
    lines += ["", "Geometry on the spinning videos (truth r = 0.25 m, |h| = 0.08 m, l = 0):", "",
              "| Variant | " + " | ".join(SPINNING) + " |", "|---|" + "---|" * len(SPINNING)]
    for v in VARIANTS:
        lines.append(f"| {LABELS[v]} | " + " | ".join(f"r {g['r']:.3f}, \\|h\\| {g['h']:.3f}, l {g['l']:.3f}, w {g['w']:.1f}±{g['w_std']:.2f}" if (g := geo[v].get(vid)) else "-" for vid in SPINNING) + " |")
    lines += ["", "Static plate position spread, mm:", "", "| Variant | " + " | ".join(STATIC) + " |", "|---|" + "---|" * len(STATIC)]
    for v in VARIANTS:
        lines.append(f"| {LABELS[v]} | " + " | ".join(f"{jitter[v].get(vid, np.nan):.1f}" for vid in STATIC) + " |")
    lines += ["", "## Aimer", "",
              "Share of control ticks that fire, and how the target was engaged. `old` is the previous reactive aimer (center aim above 6 rad/s, smoother bandwidth 30, plain EKF); `new` is the default config (attack window, smoother bandwidth 60, iterated EKF).", "",
              "| Video | old fire | new fire | new: plate / pre-aim / center | peak yaw acc old / new (rad/s²) |",
              "|---|---|---|---|---|"]
    for vid in MOVING + STATIC:
        if not aim["old"].get(vid) or not aim["new"].get(vid):
            continue
        b = aim["new"][vid]
        lines.append(f"| {vid} | {100 * aim['old'][vid]['fire']:.0f}% | {100 * b['fire']:.0f}% | {100 * b['Plate']:.0f} / {100 * b['PreAim']:.0f} / {100 * b['Center']:.0f}% | {aim['old'][vid]['peak_acc']:.1f} / {b['peak_acc']:.1f} |")
    timing = {v: sum(float(l.split(",")[1]) for l in open(BASE / "out" / f"runs_{v}" / "timing.csv").readlines()[1:]) for v in VARIANTS}
    lines += ["", "Wall time for all 12 videos (detector dominates): " + ", ".join(f"{LABELS[v]} {timing[v]:.0f} s" for v in VARIANTS), ""]
    (REPORT / "solver_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
