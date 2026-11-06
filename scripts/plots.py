#!/usr/bin/env python3
import json
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
OUT_RUNS = BASE / "out" / "runs"
OUT_NOREFINE = BASE / "out" / "runs_norefine"
OUT_REPLAY = BASE / "out" / "replay"
OUT_REPORT = BASE / "out" / "report"
OUT_REPORT.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    "static_2m", "static_2m_angled", "static_3m", "static_4m", "static_5m", "static_6m",
    "spinning_2m", "spinning_3m", "spinning_4m", "spinning_5m",
    "translate", "translate_spinning"
]
STATIC_VIDEOS = ["static_2m", "static_2m_angled", "static_3m", "static_4m", "static_5m", "static_6m"]
STATIC_DIST = ["static_2m", "static_3m", "static_4m", "static_5m", "static_6m"]
SPINNING_VIDEOS = ["spinning_2m", "spinning_3m", "spinning_4m", "spinning_5m"]

DPI = 150
PALETTE = {"red": "#e74c3c", "blue": "#3498db", "green": "#2ecc71"}

PRED = {}
JITTER = {}


def load_jsonl(path):
    """Load JSONL file, return list of dicts, empty list if not found."""
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def load_timing():
    """Load timing.csv as dict {video: seconds}."""
    timing = {}
    path = OUT_RUNS / "timing.csv"
    if path.exists():
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                timing[row["video"]] = float(row["seconds"])
    return timing

def fig_01_detection():
    """Detection rate and median confidence per video."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("01: Detection Rate and Confidence", fontsize=14, fontweight='bold')

    detection_rates, confidences = [], []
    videos_found = []

    for video in VIDEOS:
        data = load_jsonl(OUT_RUNS / f"{video}.jsonl")
        if not data:
            continue
        videos_found.append(video)
        frames_with_det = sum(1 for f in data if f.get("detections"))
        detection_rates.append(frames_with_det / len(data) if data else 0)

        all_conf = [d["confidence"] for f in data for d in f.get("detections", [])]
        confidences.append(np.median(all_conf) if all_conf else 0)

    x = np.arange(len(videos_found))
    axes[0].bar(x, detection_rates, color="#3498db", alpha=0.7)
    axes[0].set_ylabel("Detection Rate", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(videos_found, rotation=45, ha='right')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(x, confidences, color="#e74c3c", alpha=0.7)
    axes[1].set_ylabel("Median Confidence", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(videos_found, rotation=45, ha='right')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_REPORT / "01_detection.png", dpi=DPI, facecolor='white', bbox_inches='tight')
    plt.close()

def fig_02_distance():
    """Distance distribution boxplot for static videos."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("02: Observed vs Nominal Distance (Static Videos)", fontsize=14, fontweight='bold')

    distances_data = []
    nominal_dists = {"static_2m": 2.0, "static_3m": 3.0, "static_4m": 4.0, "static_5m": 5.0, "static_6m": 6.0}

    for video in STATIC_DIST:
        data = load_jsonl(OUT_RUNS / f"{video}.jsonl")
        dists = []
        for f in data:
            record = f.get("record", {})
            est = record.get("estimate")
            if est and est.get("updates"):
                for upd in est["updates"]:
                    if upd.get("accepted"):
                        pos = upd.get("observation", {}).get("position", [0, 0, 0])
                        d = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
                        dists.append(d)
        if dists:
            distances_data.append(dists)

    bp = ax.boxplot(distances_data, labels=STATIC_DIST[:len(distances_data)], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.6)

    for i, nom in enumerate(nominal_dists.values()):
        if i < len(distances_data):
            ax.axhline(nom, color='black', linestyle='--', alpha=0.5, linewidth=0.8)

    ax.set_ylabel("Distance (m)", fontsize=11)
    ax.set_xlabel("Video", fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_REPORT / "02_distance.png", dpi=DPI, facecolor='white', bbox_inches='tight')
    plt.close()

def corner_jitter(data):
    """Mean over the 4 corners of the per-axis std-dev (px), frames with exactly one detection."""
    pts = np.array([f["detections"][0]["corners"] for f in data if len(f["detections"]) == 1])
    return float(pts.std(axis=0).mean()) if len(pts) > 1 else float("nan")


def fig_03_corner_jitter():
    """Static robot: keypoints should not move. Lower spread = less measurement noise."""
    videos = [v for v in STATIC_VIDEOS if (OUT_NOREFINE / f"{v}.jsonl").exists() and v != "static_2m_angled"]
    off = [corner_jitter(load_jsonl(OUT_NOREFINE / f"{v}.jsonl")) for v in videos]
    on = [corner_jitter(load_jsonl(OUT_RUNS / f"{v}.jsonl")) for v in videos]
    xs = np.arange(len(videos))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(xs - 0.2, off, 0.4, color="#d62728", label="network keypoints")
    ax.bar(xs + 0.2, on, 0.4, color="#1f77b4", label="after lightbar refinement")
    ax.set_xticks(xs, videos)
    ax.set_ylabel("corner std-dev over time (px)")
    ax.set_title("03: keypoint jitter on static robots")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_REPORT / "03_corner_jitter.png", dpi=DPI, facecolor="white")
    plt.close(fig)
    JITTER.update(dict(zip(videos, zip(off, on))))


def fig_04_spin_rate():
    """Spin rate w(t) for spinning videos."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    fig.suptitle("04: Spin Rate Over Time (Shaded = Robot Mode)", fontsize=14, fontweight='bold')

    for idx, video in enumerate(SPINNING_VIDEOS):
        data = load_jsonl(OUT_RUNS / f"{video}.jsonl")
        if not data:
            axes[idx].text(0.5, 0.5, f"{video}\n(no data)", ha='center', va='center')
            continue

        times, ws, modes = [], [], []
        for f in data:
            record = f.get("record", {})
            est = record.get("estimate")
            if est:
                t = est.get("t", 0)
                x = est.get("x", [0]*11)
                mode = est.get("mode", "")
                times.append(t)
                ws.append(x[7] if len(x) > 7 else 0)
                modes.append(mode == "Robot")

        if times:
            axes[idx].plot(times, ws, color="#3498db", linewidth=1.5)
            for i in range(1, len(times)):
                if modes[i]:
                    axes[idx].axvspan(times[i-1], times[i], alpha=0.2, color="#2ecc71")
            axes[idx].set_xlabel("Time (s)", fontsize=10)
            axes[idx].set_ylabel("ω (rad/s)", fontsize=10)
            axes[idx].set_title(f"{video}", fontsize=11)
            axes[idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_REPORT / "04_spin_rate.png", dpi=DPI, facecolor='white', bbox_inches='tight')
    plt.close()

def fig_05_geometry():
    """Geometry parameters r, l, |h| over time for spinning videos."""
    fig, axes = plt.subplots(len(SPINNING_VIDEOS), 1, figsize=(12, 3*len(SPINNING_VIDEOS)))
    if len(SPINNING_VIDEOS) == 1:
        axes = [axes]
    fig.suptitle("05: Geometry Parameters (Spinning Videos)", fontsize=14, fontweight='bold')

    GT = {"r": 0.25, "l": 0.0, "h": 0.08}

    for idx, video in enumerate(SPINNING_VIDEOS):
        data = load_jsonl(OUT_RUNS / f"{video}.jsonl")
        if not data:
            continue

        ax = axes[idx] if len(SPINNING_VIDEOS) > 1 else axes
        times, rs, ls, hs = [], [], [], []

        for f in data:
            record = f.get("record", {})
            est = record.get("estimate")
            if est and est.get("mode") == "Robot":
                t = est.get("t", 0)
                x = est.get("x", [0]*11)
                times.append(t)
                rs.append(x[8] if len(x) > 8 else 0)
                ls.append(x[9] if len(x) > 9 else 0)
                hs.append(abs(x[10] if len(x) > 10 else 0))

        ax.axhline(GT["r"], color='black', linestyle='--', label=f'GT r={GT["r"]}', linewidth=1)
        ax.axhline(GT["l"], color='black', linestyle='--', linewidth=1)
        ax.axhline(GT["h"], color='black', linestyle='--', label=f'GT |h|={GT["h"]}', linewidth=1)

        if times:
            ax.plot(times, rs, label='r', color="#e74c3c", alpha=0.7)
            ax.plot(times, ls, label='l', color="#2ecc71", alpha=0.7)
            ax.plot(times, hs, label='|h|', color="#3498db", alpha=0.7)

            if len(times) > 0:
                last_half_idx = len(times) // 2
                med_r = np.median(rs[last_half_idx:]) if rs[last_half_idx:] else 0
                med_l = np.median(ls[last_half_idx:]) if ls[last_half_idx:] else 0
                med_h = np.median(hs[last_half_idx:]) if hs[last_half_idx:] else 0
                ax.text(0.02, 0.95, f"Last 50%: r={med_r:.3f}, l={med_l:.3f}, |h|={med_h:.3f}",
                       transform=ax.transAxes, verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_ylabel("Value (m)", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_title(f"{video}", fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_REPORT / "05_geometry.png", dpi=DPI, facecolor='white', bbox_inches='tight')
    plt.close()

def fig_06_nis():
    """NIS histogram with chi-square pdf overlay."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("06: Update NIS Distribution vs Chi-Square(4 dof)", fontsize=14, fontweight='bold')

    nis_vals, accepted_count, total_count = [], 0, 0
    for video in SPINNING_VIDEOS:
        data = load_jsonl(OUT_RUNS / f"{video}.jsonl")
        for f in data:
            record = f.get("record", {})
            est = record.get("estimate")
            if est and est.get("updates"):
                for upd in est["updates"]:
                    total_count += 1
                    nis = upd.get("nis", 0)
                    nis_vals.append(nis)
                    if upd.get("accepted"):
                        accepted_count += 1

    if nis_vals:
        ax.hist(nis_vals, bins=50, density=True, alpha=0.6, color="#3498db", label="Observed")

        x = np.linspace(0, 30, 300)
        chi2_pdf = x * np.exp(-x/2) / 4
        ax.plot(x, chi2_pdf, 'k-', linewidth=2, label="χ²(4 dof)")

        ax.axvline(13.28, color='red', linestyle='--', linewidth=2, label="Gate=13.28")

        acceptance_rate = accepted_count / total_count if total_count > 0 else 0
        ax.text(0.98, 0.97, f"Accepted: {acceptance_rate:.1%}", transform=ax.transAxes,
               ha='right', va='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        ax.set_xlabel("NIS", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_REPORT / "06_nis.png", dpi=DPI, facecolor='white', bbox_inches='tight')
    plt.close()

def predict_plate(x, plate_id, dt):
    """Plate center after dt seconds; x = [cx, vx, cy, vy, cz, vz, yaw, w, r, l, h]."""
    cx, vx, cy, vy, cz, vz, yaw, w, r, l, h = x
    odd = plate_id % 2 == 1
    yaw_i = yaw + w * dt + plate_id * np.pi / 2
    r_i = r + (l if odd else 0.0)
    return np.array([cx + vx * dt - r_i * np.cos(yaw_i), cy + vy * dt - r_i * np.sin(yaw_i), cz + vz * dt + (h if odd else 0.0)])


def fig_07_prediction():
    """Where the filter says each plate will be dt seconds ahead, vs where it was then observed."""
    horizons = [0.05, 0.1, 0.15, 0.2, 0.3]
    model, hold = {d: [] for d in horizons}, {d: [] for d in horizons}
    for video in SPINNING_VIDEOS:
        data = load_jsonl(OUT_RUNS / f"{video}.jsonl")
        if not data:
            continue
        times = np.array([f["t"] for f in data])
        for fi in data:
            est = fi["record"]["estimate"]
            if not est or est["state"] != "Confirmed" or est["mode"] != "Robot":
                continue
            seen = [u for u in est["updates"] if u["accepted"]]
            for d in horizons:
                j = int(np.argmin(np.abs(times - (fi["t"] + d))))
                if abs(times[j] - fi["t"] - d) > 0.01 or not data[j]["record"]["estimate"]:
                    continue
                for u in data[j]["record"]["estimate"]["updates"]:
                    if not u["accepted"]:
                        continue
                    obs = np.array(u["observation"]["position"])
                    model[d].append(np.linalg.norm(predict_plate(est["x"], u["id"], data[j]["t"] - est["t"]) - obs))
                    if seen:
                        hold[d].append(min(np.linalg.norm(np.array(s["observation"]["position"]) - obs) for s in seen))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for errs, name, color in [(model, "EKF robot model", "#1f77b4"), (hold, "hold last observed plate", "#d62728")]:
        ds = [d for d in horizons if errs[d]]
        ax.plot(ds, [np.median(errs[d]) * 100 for d in ds], "o-", color=color, lw=2, label=f"{name} (median)")
        ax.plot(ds, [np.percentile(errs[d], 75) * 100 for d in ds], "s--", color=color, alpha=0.6, label=f"{name} (75th pct)")
    ax.set_xlabel("prediction horizon (s)")
    ax.set_ylabel("plate position error (cm)")
    ax.set_title("07: predicted vs observed plate position, spinning videos (robot mode)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_REPORT / "07_prediction.png", dpi=DPI, facecolor="white")
    plt.close(fig)
    PRED.update({d: (float(np.median(model[d])), float(np.median(hold[d])) if hold[d] else float("nan"), len(model[d])) for d in horizons if model[d]})


def fig_08_aimer():
    """Aimer control during spin: commanded vs target yaw, fire ticks."""
    data = load_jsonl(OUT_RUNS / "spinning_3m.jsonl")
    if not data:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("08: Aimer Control (spinning_3m, t=6..7.5s)", fontsize=14, fontweight='bold')

    t_start, t_end = 6.0, 7.5
    times, yaws_cmd, yaws_target, fires = [], [], [], []

    for f in data:
        record = f.get("record", {})
        t_frame = record.get("estimate", {}).get("t", 0)
        if t_start <= t_frame <= t_end:
            target = record.get("target")
            if target:
                shot = target.get("shot", {})
                yaws_target.append(shot.get("yaw", 0))
                times.append(t_frame)

            for cmd in record.get("commands", []):
                cmd_time = cmd.get("t", t_frame)
                if t_start <= cmd_time <= t_end:
                    yaws_cmd.append(cmd.get("yaw", 0))
                    if cmd.get("fire"):
                        fires.append(cmd_time)

    if times:
        axes[0].plot(times, yaws_target, 'o-', label='Target yaw', color="#e74c3c", linewidth=1.5, markersize=4)
        if yaws_cmd:
            times_cmd = np.linspace(t_start, t_end, len(yaws_cmd))
            axes[0].plot(times_cmd, yaws_cmd, '-', label='Commanded yaw', color="#3498db", alpha=0.7, linewidth=1)
        if fires:
            axes[0].scatter(fires, [yaws_target[min(int((f-t_start)/(t_end-t_start)*len(yaws_target)), len(yaws_target)-1)] for f in fires],
                           marker='*', s=200, color='red', label='Fire', zorder=5)
        axes[0].set_ylabel("Yaw (rad)", fontsize=11)
        axes[0].set_xlabel("Time (s)", fontsize=11)
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)

    yaw_accs = []
    for cmd in [c for f in data for c in f.get("record", {}).get("commands", [])]:
        if t_start <= cmd.get("t", 0) <= t_end:
            yaw_accs.append(abs(cmd.get("yaw_acc", 0)))

    if yaw_accs:
        axes[1].hist(yaw_accs, bins=30, color="#3498db", alpha=0.7, edgecolor='black')
        axes[1].axvline(60, color='red', linestyle='--', linewidth=2, label='Limit: 60 rad/s²')
        max_acc = max(yaw_accs)
        fire_frac = len(fires) / max(len(yaw_accs), 1) if fires else 0
        axes[1].text(0.98, 0.97, f"Max |a|: {max_acc:.1f} rad/s²\nFire frac: {fire_frac:.1%}",
                    transform=axes[1].transAxes, ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        axes[1].set_xlabel("|Yaw Acceleration| (rad/s²)", fontsize=11)
        axes[1].set_ylabel("Count", fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_REPORT / "08_aimer.png", dpi=DPI, facecolor='white', bbox_inches='tight')
    plt.close()

def fig_09_ballistics():
    """Ballistics: flight time vs horizontal distance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("09: Ballistics Analysis", fontsize=14, fontweight='bold')

    flight_times, dists = [], []
    for video in VIDEOS:
        data = load_jsonl(OUT_RUNS / f"{video}.jsonl")
        for f in data:
            target = f.get("record", {}).get("target")
            if target and target.get("shot"):
                shot = target["shot"]
                flight_time = shot.get("flight_time", 0)
                pos = target.get("position", [0, 0, 0])
                dist = np.sqrt(pos[0]**2 + pos[1]**2)
                flight_times.append(flight_time)
                dists.append(dist)

    if flight_times:
        ax.scatter(dists, flight_times, alpha=0.6, s=20, color="#3498db", label="Observed")

        d_range = np.linspace(0, max(dists) * 1.1, 100)
        v = 23.0
        flight_time_vacuum = d_range / (v * np.cos(np.deg2rad(15)))
        ax.plot(d_range, flight_time_vacuum, 'k--', linewidth=2, label="Vacuum (v=23 m/s, θ=15°)")

        ax.set_xlabel("Horizontal Distance (m)", fontsize=11)
        ax.set_ylabel("Flight Time (s)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_REPORT / "09_ballistics.png", dpi=DPI, facecolor='white', bbox_inches='tight')
    plt.close()

def fig_10_translate():
    """Distance vs time for translate videos."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("10: Translation Maneuver (Center Distance vs Time)", fontsize=14, fontweight='bold')

    for idx, video in enumerate(["translate", "translate_spinning"]):
        data = load_jsonl(OUT_RUNS / f"{video}.jsonl")
        if not data:
            continue

        times, dists = [], []
        for f in data:
            est = f.get("record", {}).get("estimate")
            if est:
                t = est.get("t", 0)
                x = est.get("x", [0]*11)
                cx, cy = x[0], x[2]
                dist = np.sqrt(cx**2 + cy**2)
                times.append(t)
                dists.append(dist)

        if times:
            axes[idx].plot(times, dists, color="#3498db", linewidth=1.5)
            axes[idx].set_xlabel("Time (s)", fontsize=11)
            axes[idx].set_ylabel("Distance (m)", fontsize=11)
            axes[idx].set_title(f"{video}", fontsize=11)
            axes[idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_REPORT / "10_translate.png", dpi=DPI, facecolor='white', bbox_inches='tight')
    plt.close()

def fig_11_latency():
    """FPS from timing.csv."""
    timing = load_timing()
    if not timing:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("11: Processing Latency (FPS, CPU Only)", fontsize=14, fontweight='bold')

    videos_sorted = sorted([v for v in VIDEOS if v in timing], key=lambda x: timing[x])
    fps_list = []
    for video in videos_sorted:
        frame_counts = {}
        data = load_jsonl(OUT_RUNS / f"{video}.jsonl")
        frame_counts[video] = len(data)
        fps = frame_counts[video] / timing[video] if timing[video] > 0 else 0
        fps_list.append(fps)

    colors = ["#3498db" if "static" in v else "#e74c3c" if "spinning" in v else "#2ecc71" for v in videos_sorted]
    bars = ax.bar(range(len(videos_sorted)), fps_list, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(videos_sorted)))
    ax.set_xticklabels(videos_sorted, rotation=45, ha='right')
    ax.set_ylabel("FPS", fontsize=11)
    ax.set_xlabel("Video", fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_REPORT / "11_latency.png", dpi=DPI, facecolor='white', bbox_inches='tight')
    plt.close()

def generate_summary():
    """Generate summary.md table."""
    timing = load_timing()

    rows = []
    for video in VIDEOS:
        data = load_jsonl(OUT_RUNS / f"{video}.jsonl")
        if not data:
            rows.append(f"| {video} | - | - | - | - | - | - | - | - | - | - | - |")
            continue

        frames = len(data)
        det_rate = sum(1 for f in data if f.get("detections")) / frames if frames else 0

        total_upd, accepted_upd, confirmed_frames, robot_frames, fires = 0, 0, 0, 0, 0
        nises = []
        w_vals, r_vals, l_vals, h_vals = [], [], [], []

        for f in data:
            record = f.get("record", {})
            est = record.get("estimate")
            if est:
                if est.get("state") == "Confirmed":
                    confirmed_frames += 1
                if est.get("mode") == "Robot":
                    robot_frames += 1
                    x = est.get("x", [0]*11)
                    if len(x) > 7: w_vals.append(x[7])
                    if len(x) > 8: r_vals.append(x[8])
                    if len(x) > 9: l_vals.append(x[9])
                    if len(x) > 10: h_vals.append(abs(x[10]))

                for upd in est.get("updates", []):
                    total_upd += 1
                    if upd.get("accepted"):
                        accepted_upd += 1
                    nis = upd.get("nis", 0)
                    nises.append(nis)

            for cmd in record.get("commands", []):
                if cmd.get("fire"):
                    fires += 1

        confirmed_frac = confirmed_frames / frames if frames else 0
        robot_frac = robot_frames / frames if frames else 0
        median_nis = np.median(nises) if nises else 0

        if robot_frames > 0:
            last_half_idx = robot_frames // 2
            med_w = np.median(w_vals[last_half_idx:]) if w_vals[last_half_idx:] else 0
            med_r = np.median(r_vals[last_half_idx:]) if r_vals[last_half_idx:] else 0
            med_l = np.median(l_vals[last_half_idx:]) if l_vals[last_half_idx:] else 0
            med_h = np.median(h_vals[last_half_idx:]) if h_vals[last_half_idx:] else 0
        else:
            med_w = med_r = med_l = med_h = 0

        total_cmds = sum(len(f.get("record", {}).get("commands", [])) for f in data)
        fire_frac = fires / total_cmds if total_cmds else 0

        row = f"| {video} | {frames} | {det_rate:.1%} | {confirmed_frac:.1%} | {robot_frac:.1%} | {accepted_upd}/{total_upd} | {median_nis:.2f} | {med_w:.3f} | {med_r:.3f} | {med_l:.3f} | {med_h:.3f} | {fire_frac:.1%} |"
        rows.append(row)

    summary_lines = [
        "# Analysis Summary\n",
        "## Per-Video Statistics\n",
        "| Video | Frames | Det % | Conf % | Robot % | Updates | Median NIS | w | r | l | |h| | Fire % |",
        "|-------|--------|-------|--------|---------|---------|------------|---|---|---|-----|---------|"
    ]
    summary_lines.extend(rows)

    data_main = load_jsonl(OUT_RUNS / "spinning_3m.jsonl")
    data_replay_a = load_jsonl(OUT_REPLAY / "spinning_3m_a.jsonl")
    data_replay_b = load_jsonl(OUT_REPLAY / "spinning_3m_b.jsonl")

    summary_lines.append("\n## Replay Determinism\n")

    path_a = OUT_REPLAY / "spinning_3m_a.jsonl"
    path_b = OUT_REPLAY / "spinning_3m_b.jsonl"
    if path_a.exists() and path_b.exists():
        with open(path_a, 'rb') as f:
            bytes_a = f.read()
        with open(path_b, 'rb') as f:
            bytes_b = f.read()
        identical = "Yes (byte-identical)" if bytes_a == bytes_b else "No (byte differences)"
        summary_lines.append(f"- **Replay files identical**: {identical}\n")

    if data_main and data_replay_a:
        names = ["cx", "vx", "cy", "vy", "cz", "vz", "yaw", "w", "r", "l", "h"]
        diffs = np.array([np.abs(np.array(m["record"]["estimate"]["x"]) - np.array(a["record"]["estimate"]["x"]))
                          for m, a in zip(data_main, data_replay_a) if m["record"]["estimate"] and a["record"]["estimate"]])
        summary_lines.append("\nThe log stores JPEG frames, so replay differs slightly from the run on the raw video.")
        summary_lines.append("\n| state | " + " | ".join(names) + " |")
        summary_lines.append("|---" * (len(names) + 1) + "|")
        summary_lines.append("| median abs diff (video vs replay) | " + " | ".join(f"{v:.4f}" for v in np.median(diffs, axis=0)) + " |")

    summary_lines.append("\n## Prediction (figure 07)\n")
    summary_lines.append("| horizon (s) | EKF median error (cm) | hold-last-plate median error (cm) | samples |")
    summary_lines.append("|---|---|---|---|")
    for d, (m, h, n) in sorted(PRED.items()):
        summary_lines.append(f"| {d:.2f} | {m * 100:.1f} | {h * 100:.1f} | {n} |")

    summary_lines.append("\n## Corner jitter (figure 03)\n")
    summary_lines.append("| video | network keypoints (px) | after refinement (px) |")
    summary_lines.append("|---|---|---|")
    for v, (off, on) in JITTER.items():
        summary_lines.append(f"| {v} | {off:.3f} | {on:.3f} |")

    summary_lines.append("\n## Figures\n")
    figures = [
        ("01_detection.png", "Detection rate and median confidence per video"),
        ("02_distance.png", "Observed plate distance vs nominal for static videos"),
        ("03_corner_jitter.png", "Detected corner position stability with refinement ON vs OFF"),
        ("04_spin_rate.png", "Spin rate over time, shaded for robot mode"),
        ("05_geometry.png", "Geometric parameters (r, l, |h|) during spinning motion"),
        ("06_nis.png", "NIS distribution of updates against chi-square(4) gate"),
        ("07_prediction.png", "Prediction error vs horizon with baseline comparison"),
        ("08_aimer.png", "Aimer commanded vs target yaw and acceleration limits"),
        ("09_ballistics.png", "Flight time vs distance with vacuum ballistics curve"),
        ("10_translate.png", "Robot center distance during translation maneuvers"),
        ("11_latency.png", "Processing frames per second per video"),
    ]
    for fname, desc in figures:
        summary_lines.append(f"- **{fname}**: {desc}")

    with open(OUT_REPORT / "summary.md", "w") as f:
        f.write("\n".join(summary_lines))

if __name__ == "__main__":
    print("Generating figures...")
    fig_01_detection()
    print("  01_detection.png")
    fig_02_distance()
    print("  02_distance.png")
    fig_03_corner_jitter()
    print("  03_corner_jitter.png")
    fig_04_spin_rate()
    print("  04_spin_rate.png")
    fig_05_geometry()
    print("  05_geometry.png")
    fig_06_nis()
    print("  06_nis.png")
    fig_07_prediction()
    print("  07_prediction.png")
    fig_08_aimer()
    print("  08_aimer.png")
    fig_09_ballistics()
    print("  09_ballistics.png")
    fig_10_translate()
    print("  10_translate.png")
    fig_11_latency()
    print("  11_latency.png")

    print("Generating summary.md...")
    generate_summary()
    print("Done!")
