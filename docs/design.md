# Auto-Aim System Design

## Language: Rust

Why: decent ROS compatibility, and modern tooling for building and adding libraries.

| Crate | Use |
| --- | --- |
| nalgebra | Linear algebra (like Eigen) |
| kornia-rs | Image ops (like OpenCV): resize, warp, undistort. Its PnP (`kornia-3d`/`kornia-pnp`) is EPnP + LM refinement only, and EPnP is weak on 4 coplanar armor corners. Write IPPE ourselves (closed form, small) or seed LM from the tracker's prediction |
| ort | Run the detector model on TensorRT, CUDA, or CPU (see Detector → Backends). `tensorrt` feature supports aarch64 Linux (needs CUDA 11.6+, TensorRT 8.4+, which JetPack has) |
| foxglove | One API for both logging (writes MCAP) and live streaming (websocket server Foxglove connects to). Comes with standard message schemas (images, annotations, 3D scenes) |
| mcap | Reading logs back for replay |
| crossbeam-channel | Bounded channels between pipeline threads |
| serialport | Blocking serial on its own thread. Lowest, most predictable latency; everything else is plain threads, so no need to pull in tokio just for serial |
| tracing | Logs |
| color-eyre | Pretty errors |
| serde + serde-saphyr | YAML config |
| *(camera SDK)* | Hikrobot/MindVision ship a C SDK; needs a small `bindgen` wrapper |

## High level overview

```
CameraSource â”€â”
              â”œâ”€â–º Sync â”€â–º Detector â”€â–º Tracker â”€â–º Aimer â”€â–º MCU
RobotSource â”€â”€â”˜   (pose @ frame time)   (EKF)    (ballistics + fire)
      â”‚                                                     â”‚
      â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º Logger â—„â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â”œâ”€â”€â–º MCAP file â”€â”€â–º Replay
                              â””â”€â”€â–º Live websocket â”€â”€â–º Foxglove (laptop)
```

## Inputs to the system

```rust
struct CameraFrame {
    image: Image,
    frame_id: u64,        // detect drops, restore order after the detector pool
    t_jetson: Instant,    // stamped on arrival
}
// Intrinsics + distortion come from the per-robot config, not per frame.
```

```rust
struct RobotState {
    t: f64,               // Jetson arrival time minus imu_delay (fixed, measured offline)
    q: UnitQuaternion,    // gimbal orientation (or roll/pitch/yaw; quaternion avoids wrap/gimbal-lock bugs)
    bullet_speed: f32,    // last *measured* speed; fall back to config default if out of range
    robot_id: u8,         // own id; enemy color is derived from this
}
```

## System output

```rust
struct AimCommand {
    yaw: f32, pitch: f32,
    yaw_vel: f32, pitch_vel: f32,
    yaw_acc: f32, pitch_acc: f32,
    fire: bool,
}
```

The command is computed for `now + downlink_delay`, where `downlink_delay` is a single tunable number in config.

## System overview

**`CameraSource`** and **`RobotSource`** are traits with two implementations each: `Live` and `Log`.

The `Log` version of `RobotSource` might not exist, e.g. you only have a video you want to analyze. In that case it falls back to defaults from config (fixed pose, bullet speed, robot id).

The camera queue keeps only the **latest** frame. If processing falls behind, drop old frames, never new ones.

### Time sync

Everything is on the Jetson clock, and each link has **one fixed, stored latency** in the per-robot config:

| Value | Meaning |
|---|---|
| `camera_delay` | Exposure to frame arrival on the Jetson. `t_frame = t_arrival − camera_delay`. |
| `imu_delay` | IMU sample on the MCU to packet arrival on the Jetson. `t_pose = t_arrival − imu_delay`. |
| `downlink_delay` | Command sent by the Jetson to applied on the MCU. Commands are computed for `now + downlink_delay`. |

For each frame, the pose is a slerp of the buffered IMU samples at `t_frame` (`src/sync/pose_buffer.rs`). There is no online clock estimation in this codebase.

**Suggested way to measure the latencies (out of scope here):**
- `camera_delay − imu_delay`: point the camera at a fixed ChArUco board and sweep the gimbal back and forth at several speeds. Log frames and IMU packets, each stamped on arrival. Get the camera's rotation from the board pose, then find the time shift that best lines up its angular rate with the IMU's (cross-correlation or least squares over the shift). The residual should be flat across sweep speeds. Only this difference matters for pose alignment.
- `imu_delay` alone: echo a packet MCU → Jetson → MCU and take half the round trip (or toggle a GPIO the camera can see).
- `downlink_delay`: command a step in yaw and time it to the first encoder motion on the MCU, minus the gimbal's own response time; or tune it by hand until the aim stops lagging a constant-velocity target.
- Re-measure when exposure, resolution, USB/serial setup or firmware changes.
## Detector

`Frame → Frame + Detections`

- A detection contains color, icon, confidence, and 4 keypoints.
- Corner refinement happens here.
- Multi-threaded to improve throughput. Outputs must preserve frame order (reorder by `frame_id`), otherwise we bork the filters.
- Drop detections of our own color here, so the tracker never sees them.

### Backends

Config holds an ordered list, e.g. `backends: [tensorrt, cuda, cpu]`. ort already handles the fallback: execution providers are tried in order, and one that fails to register is skipped (CPU is always last).

What we add on top:
- **Log which backend actually loaded**, loudly, at startup and in the live viz. Silent fallback is the danger: TensorRT fails to load and you're running on CPU at a fraction of the frame rate without knowing.
- **`require_backend` option** (on for matches): use ort's `error_on_failure()` on the first entry, so a broken GPU setup fails at startup instead of silently degrading.
- **TensorRT engine cache** on disk. The first TensorRT build of the model takes minutes; caching makes later startups fast. Rebuild if the model or TensorRT version changes.
- **FP16** as a config flag for TensorRT/CUDA.
- Build with ort's `load-dynamic` feature so the same binary runs on a laptop with no CUDA (falls through to CPU) and on the Jetson.

Even with TensorRT, a node TensorRT doesn't support runs on CUDA/CPU instead, so preprocessing and odd ops can still land on the CPU. Check the per-stage latency plot once after any model change.

## Tracker

`Frame + Detections + Pose → RobotEstimate`

**Target selection.** We track by icon. If multiple robots are present, grab the icon closest to the center of the screen. Once we have a robot estimate, keep feeding it detections that match its icon.
- Also check that each detection is consistent with the prediction (a chi-square gate on the innovation), since icon matching alone won't stop one misclassified frame or a reflection from yanking the state.
- Drop the target and re-pick after it's been unseen for N frames.

**Robot filter.** The EKF estimates the whole robot: center position, velocity, yaw, spin rate, radius, 2nd radius, and plate heights.
- When the spin rate is low, radius and center aren't observable. In that case, aim straight at the observed plate's predicted position (plate mode). Switch to aiming from the full model (robot mode) once it's clearly spinning. Use separate on/off thresholds so it doesn't flicker.
- Reset the covariance too when re-initializing, not just the state.

**Priors.** Track the icon over several frames. Once a track passes a minimum number of frames and percentage of agreeing icons, store its radius and heights keyed by icon when we lose it. When that robot is picked up again, start the filter from the stored radius and heights instead of defaults. Fall back to per-class defaults (in config) for robots we haven't seen yet.

## Aimer

`RobotEstimate + RobotState → AimCommand`

1. **Pick the plate**: in robot mode, choose the plate that will face us at impact time, not the one nearest right now.
2. **Solve the impact time**: iterate `t_hit = latency + flight_time(predicted position)` until it converges (2–3 iterations).
3. **Ballistics**: gravity + drag model, using the *measured* bullet speed. Drag coefficient lives in config and gets tuned from logged shots.
4. **Trajectory**: smooth the yaw/pitch target over time within the gimbal's max acceleration (quintic segments or MPC), so switching plates doesn't jerk the gimbal.
5. **Fire**: fire when the gimbal error is within the plate's angular half-width (shrinks as the plate turns away). Don't fire when the target is only being predicted (no recent detection) or when the smoothed trajectory hasn't caught up to the target yet.

## Logging

`Log` (MCAP, so it opens in Foxglove):
- Every robot state input
- Every aim command output
- A fraction of camera frames, as frames per second, set in config

`Replay`:
- Feeds logged inputs in timestamp order as fast as possible, without sleeping between them.
- Same input gives the same output.
  - That means using the logged timestamps, not the wall clock.
  - It also means the detector pool's reordering has to make results independent of thread timing.
- Optional real-time mode (sleep to match the timestamps) for debugging threading issues.

## Live visualization

Same channels as the log, also sent to a live Foxglove websocket. One `log()` call per message goes to both outputs, so what you watch live is exactly what you can replay later. Open Foxglove on a laptop, connect to `ws://<jetson>:8765`.

What to publish:
- **Camera view**: frame + detection overlays (keypoints, icon, confidence), the predicted plates reprojected onto the image, and the aim point.
- **3D scene**: own gimbal, estimated robot (center, radius, all plates), the selected plate, and the bullet trajectory.
- **Plots**: spin rate, radius, innovation (how surprised the filter is per update), plate/robot mode, fire, gimbal error vs fire threshold, latency per stage.

Keep it from hurting the pipeline:
- Visualization runs on its own thread with a bounded channel. If it falls behind, **drop** messages, never block.
- Images are downscaled + JPEG'd before sending, at a capped rate (e.g. 15 Hz). Wi-Fi to the laptop is the bottleneck, not the Jetson.
- Off by default in matches, or capped low enough to stay within the radio bandwidth limit.
- Replay publishes to the same websocket, so you can watch a log play back exactly like a live session.

## Config

- **Detector**: model path, `backends` list, `require_backend`, FP16, engine cache path, thresholds.
- **General** (robot agnostic): noise values, gates, mode thresholds, drag coefficient, per-class geometry defaults, chi-square gate.
- **Per robot**:
  - camera intrinsics
  - camera → turret → barrel transforms
  - `camera_delay` and `downlink_delay`
  - gimbal max velocity/acceleration
  - default robot state (for logs without one)
- **Logging**: frame rate to log, output path.
- **Live viz**: enabled, port, image rate, image scale, JPEG quality.

## Threads

Pipeline threads are connected by bounded crossbeam channels.

- **Camera**: grab → latest-frame slot.
- **Serial**: RX parse, stamp on arrival → pose buffer. TX of aim commands also goes through this thread, so only one thread touches the port.
- **Detector pool**: N workers → reorder buffer.
- **Tracker**: per frame.
- **Aimer**: fixed rate (100–200 Hz), predicts from the latest estimate. It doesn't wait for new frames.
- **Logger / viz**: writes MCAP and serves the live websocket. Never blocks the pipeline; drops messages for viz when behind (still writes everything that's configured to be logged).
