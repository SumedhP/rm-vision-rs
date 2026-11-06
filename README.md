# auto_aim_rs

A RoboMaster auto-aim pipeline in Rust, built from [`docs/design.md`](docs/design.md). It is tested end to end on the RealSense recordings using the SZURP armor model (`Infantry-v8n-fp16-20260726-D1.8w-B16.onnx`). Calls made while building it are in [`docs/decisions.md`](docs/decisions.md).

```
CameraSource â”€â”
              â”œâ”€â–º Sync â”€â–º Detector pool â”€â–º Tracker â”€â–º Aimer â”€â–º AimCommand
RobotSource â”€â”€â”˜   (pose @ frame time)       (EKF)     (ballistics + fire)
                                    â”‚
                                    â””â”€â–º JSONL / MCAP log / live Foxglove / overlay video
```

## Layout

| Path | Design doc section |
|---|---|
| `src/io/` | Inputs. `CameraSource::{Video, Log}`, `RobotSource::{Defaults, Log}` |
| `src/sync/pose_buffer.rs` | Time sync. Gimbal pose slerped to the frame time; link latencies are fixed config values |
| `src/detector/` | Detector. `onnx.rs` (model I/O, backends + fallback), `pool.rs` (N workers, strict frame order), `refine.rs` (lightbar corner refinement) |
| `src/tracker/` | Tracker. `measurement.rs` (plate pose), `ekf.rs` (11-state EKF, chi-square gate), `model.rs` (robot model), `mod.rs` (selection, association, lifecycle, plate/robot mode, priors) |
| `src/aimer/` | Aimer. `ballistics.rs` (RK4 + drag), `trajectory.rs` (acceleration-limited smoother), `aimer.rs` (plate choice, center aim, fire) |
| `src/pipeline.rs` | The loop. Runs in data time, so replays are deterministic |
| `src/viz/` | Logging + live visualization. `foxglove.rs` (MCAP + websocket), `overlay.rs` (annotated video) |
| `configs/` | `general.yaml` (tunings), `robots/realsense_test.yaml` (camera, latencies, gimbal), `logging.yaml` |
| `scripts/plots.py` | Report figures from the per-frame JSONL |

## Run

Needs Rust (edition 2024), and `ffmpeg`/`ffprobe` on PATH. The SZURP model is committed in `assets/models/`. The videos are not: put `assets/videos/*.mp4` from `rm-auto-aim` branch `basic-infra` there.

```sh
cargo build --release                      # add --features gpu on the Jetson (CUDA/TensorRT)
cargo test --release                       # unit tests + detector tests on the real videos

# One video: per-frame JSON, overlay video, MCAP log
target/release/auto-aim --video assets/videos/spinning_3m.mp4 \
    --jsonl out/runs/spinning_3m.jsonl --overlay out/overlay/spinning_3m.mp4 --mcap out/logs/spinning_3m.mcap

# Live view: serve a websocket and open Foxglove at ws://<host>:8765
target/release/auto-aim --video assets/videos/spinning_3m.mp4 --ws 8765

# Replay a recorded log (same input, same output)
target/release/auto-aim --replay out/logs/spinning_3m.mcap --jsonl out/replay/a.jsonl

python scripts/plots.py                    # figures + out/report/summary.md
```

`cargo clippy --release --all-targets` is clean under the lint set in `Cargo.toml`.

## Not tested here

- **GPU backends**: compiled behind `--features gpu`. On this laptop the CPU fallback path is exercised and tested instead.
- **Serial and live camera**: no hardware. The `Live` source variants and the MCU packet format are still to be written. Robot state comes from config defaults or a recorded log.
- **Latency values**: these are fixed config numbers. `docs/design.md` describes how to measure them.

## Results on the RealSense recordings (CPU)

Figures and tables are in [`docs/report/`](docs/report/) (`summary.md` plus 11 figures). The overlay videos are `docs/report/spinning_3m.mp4` and `translate_spinning.mp4`.

| What | Result | Evidence |
|---|---|---|
| Detection | 100% of frames on all static and spinning videos at 2 to 6 m; 68 to 71% on the translate videos, where the robot drives out of view | `01_detection.png` |
| Plate distance | Medians of 2.13 / 3.10 / 4.18 / 4.99 / 6.04 m against nominal "about" 2 / 3 / 4 / 5 / 6 m, with frame-to-frame spread from 4 mm at 2 m to 7 cm at 6 m | `02_distance.png` |
| Spin rate | Locks at 14.7 to 14.9 rad/s on all spinning videos, and at -11.7 rad/s while also translating; follows the spin-up and stop | `04_spin_rate.png` |
| Robot geometry | r = 0.243 to 0.247 m (truth 0.25), l = 0.02 m (truth 0), \|h\| = 0.093 to 0.095 m (truth 0.08) | `05_geometry.png` |
| Filter consistency | 99.5% of spinning-video updates pass the 99% chi-square gate | `06_nis.png` |
| Prediction | Median plate position error 6.4 cm at 50 ms and 8.3 cm at 300 ms, against 15 to 22 cm for holding the last observed plate | `07_prediction.png` |
| Aimer | Commanded yaw acceleration at most 5.9 rad/sÂ² (limit 60). Fires on 96 to 98% of static frames and 42 to 52% of spinning ticks | `08_aimer.png`, `summary.md` |
| Corner refinement | Keypoints are already sub-pixel stable (0.07 to 0.14 px). Refinement lowers jitter on 4 of 5 static videos and slightly raises it at 2 m, so its gain on this footage is small | `03_corner_jitter.png` |
| Replay | Two replays of the same log are byte-identical. Replay vs the raw-video run differs by about 5 mm in center position and 0.02 rad/s in spin rate, because the log stores JPEG frames | `summary.md` |
| Throughput | About 16 fps end to end on a Ryzen AI 7 350 CPU (3 detector workers, including JSON and overlay output) | `11_latency.png` |

Known gap: |h| reads about 1.4 cm high on every video. That's consistent with a systematic plate-height bias in the fixed-tilt pose fit, and it's worth checking against the real plate tilt and lightbar length.
