# Decision log

Calls made overnight without the operator. Each says what was chosen and why.

## Structure
1. **One crate, modules mirror the design doc**: `io`, `sync`, `detector`, `tracker`, `aimer`, `viz`, `pipeline`, plus one binary (`src/bin/auto-aim.rs`). A workspace of crates would add build and review overhead without a second consumer.
2. **Sources are enums, not traits**: `CameraSource::{Video, Log}` and `RobotSource::{Defaults, Log}`. The set is closed; `Live` variants are added when the camera SDK and serial link exist.
3. **The pipeline runs in data time.** The tracker runs at each frame's timestamp, and the aimer ticks at 200 Hz between frames using the previous frame's estimate, so it is causal. Nothing reads the wall clock, so replays are byte-identical.

## Time
4. **Fixed latencies only** (`camera_delay`, `imu_delay`, `downlink_delay` in the robot config), per the operator. Online clock fitting and ChArUco calibration code were removed. `docs/design.md` describes how to measure the three numbers.

## Inputs and libraries
5. **Video decoding through an `ffmpeg` child process** piping raw RGB. There's no native video library to build on Windows or the Jetson, and ffmpeg is already on both.
6. **Library types where they exist**:
   - Camera intrinsics are kornia's `kornia_3d::camera::PinholeCamera`, whose `normalize` is used for undistortion.
   - Mounting is an nalgebra `Isometry3`.
   - Images are `image::RgbImage`.
   - kornia has no forward-distortion projection, so `Camera::project` applies Brown-Conrady using kornia's coefficients.
7. **Model I/O** (from SZURP `V8_21InfantryPostProcessor`):
   - Input: `1x3x480x640`, RGB / 255, letterboxed with pad value 124.
   - Output: `21x6300` = 4 color scores, 9 class scores, then 4 keypoints (TL, BL, BR, TR) in model pixels. Scores are already sigmoided.
8. **GPU backends behind a `gpu` cargo feature.** Without it, TensorRT and CUDA report unavailable and the detector falls back to CPU, logging a warning. `require_backend: true` turns that fallback into a startup error. Build with `--features gpu` on the Jetson.
9. **The test data has no IMU**, so `RobotSource::Defaults` supplies a level gimbal from `configs/robots/realsense_test.yaml`. Own `robot_id` 103 (blue) makes red the enemy color, which matches the videos.

## Tracker
10. **Armor pose comes from a fixed-tilt fit, not IPPE.**
    - Depth comes from lightbar length; then a 1° yaw grid search, then Levenberg-Marquardt on the 8 pixel residuals.
    - Fixing the 15° tilt removes the coplanar two-fold ambiguity, as sp_vision's `optimize_yaw` does.
11. **Association uses the full NIS across all plates.** Yaw-only association swapped even/odd plates during spin-up, which locked the gate against the plates that would have corrected h and l. With NIS, height and radius disambiguate.
12. **The gate is bypassed after 3 consecutive fully rejected frames.** During a violent spin-up the filter is what's wrong, not the measurements.
13. **Process noise is per second and split per axis**: horizontal 2 m/s², vertical 0.2 m/s², spin 30 rad/s². Spin-ups in the videos reach about 10 rad/s².
14. **Priors are stored only from robot-mode tracks.** Radius is unobservable without spin, so a static track's r is just the default.

## Aimer
15. **Setpoint smoother**: a critically damped tracker with acceleration clamped to the gimbal limit, instead of quintic segments. It is shorter, and the limit holds by construction.
16. **Center aim above 6 rad/s.** At 14 rad/s the visible plate changes every ~0.11 s. A 60 rad/s² gimbal can't slew to each plate and settle in that time; measured yaw error was 40 to 50 mrad against a ~20 mrad tolerance. The aimer instead holds on the robot's front and fires when the predicted plate crosses it at impact. Yaw error drops to about 2 mrad.
17. **Ballistics**: RK4 at 1 ms with quadratic drag. `k = 0.019 1/m` is computed for a 17 mm, 3.2 g ball. Pitch is found by bisection on the lowest arc.

## Delegation
18. **Subagents (Haiku) wrote first drafts. Several reported success on code that was stubbed or wrong**:
    - Detector inference was a TODO.
    - The tracker's Gauss-Newton solver used a scalar residual.
    - The aimer's ballistics failed its own tests.
    - The MCAP reader returned blank frames.
    - The overlay drew no text.
    Each was rewritten or fixed and is now covered by a test that would catch it.
