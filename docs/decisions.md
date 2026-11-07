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
15. **The update is iterated (`tracker.iterations`, default 3).**
    - Each iteration relinearizes at the latest estimate. It restarts from the prior mean and keeps the prior term `H_i (x_prior - x_i)`.
    - SHtech's iterated filter has a sign error on that term, and awakening's drops it.
    - Gating and the reported NIS use the prior innovation.
    - On the recordings, iterating changes nothing measurable: next-frame error is 0.88 px either way. At 60 fps each update is a small step, so one linearization is already accurate.
    - It's kept because it's correct and cheap, and it matters after gaps or at low frame rates. A unit test shows it fitting a strongly nonlinear corner measurement that a single step misses.
16. **Measurement space is a config choice (`tracker.measurement: pose | pixel`), and pose stays the default.**
    - The pixel variant measures the 4 detected corners against the model's projected plate, as rmcs and awakening do.
    - `docs/report/solver_comparison.md` compares the two on the same footage. Neither wins outright:
      - Pixel predicts the next frame better: 0.66 vs 0.88 px median.
      - Pixel holds a static, angled plate much steadier: 1.1 vs 5.4 mm.
      - Pose has the smaller error tail: 300 ms p90 of 17 vs 19 px. It also has a smaller 6 m static spread (25 vs 32 mm) and accepts more updates (99.6% vs 96.5%).
      - Fire rates match within 1 point, and so do r and |h|.
    - The pixel noise is 1 px. Tightening it to 0.5 or 0.3 px drops acceptance on moving plates to 47–73%, because corner error on moving plates is heavy-tailed.

## Aimer
17. **Setpoint smoother**: a critically damped tracker with acceleration clamped to the gimbal limit, instead of quintic segments. It is shorter, and the limit holds by construction. Bandwidth is 60 rad/s.
    - At 30 rad/s it took about 0.13 s to settle, which is a whole plate period at 11.7 rad/s. It overshot every plate in the window, and the overshoot carried into the next plate.
    - At 60 rad/s, fire rate on translate_spinning goes from 26% to 32% and on spinning_3m from 44% to 56%. The spread of the yaw command on a static target is unchanged.
18. **Plate choice is rmcs's attack window** (`src/aimer/aimer.rs`, from rmcs `fire_control.cpp`). It replaces the old facing-angle switch, the 45° fire cap and center aim above 6 rad/s.
    - A plate is engaged only while its angle around the robot center is inside the window. The window is at most 120° wide.
    - The window also shrinks to the part of each plate's sweep the gimbal can follow at a steady rate while still slewing back for the next plate. This is rmcs's sawtooth model, using `max_yaw_vel` and `max_yaw_acc` at 80%.
    - Plates are judged at the predicted impact time. Entry happens at 80% of the half width and exit at 120%.
    - With no plate inside, the gimbal waits on the edge the next plate enters from (pre-aim).
    - Above 12 rad/s it aims at the robot center.
    - In pre-aim and center aim, a shot fires only when a plate's predicted position at impact crosses the line of fire (window timing).
    - Compared with the old aimer at the same smoother bandwidth, translate_spinning (11.7 rad/s) goes from 21% to 26% of ticks firing.
    - The angled static plate goes from 0% to 97%, because the old 45° cap blocked a 53° plate.
19. **Ballistics**: RK4 at 1 ms with quadratic drag. `k = 0.019 1/m` is computed for a 17 mm, 3.2 g ball. Pitch is found by bisection on the lowest arc.

## Serial
20. **MCU link uses the DJI referee frame** (SOF `0xA5`, length, seq, CRC8, cmd id, body, CRC16), as awakening's `radar_io` and SHtech's MCU packets do.
    - The CRC tables are generated from the reflected polynomials and checked against entries of the DJI tables and the CRC-16/MCRF4XX check value.
    - A message is any type implementing `Message` (cmd id plus encode/decode). Adding one touches only `src/io/serial/messages.rs`.
    - Unknown ids are delivered as raw frames, so referee frames the MCU forwards pass through.
    - Our ids start at `0x0A00`, clear of the referee range: `0x0A01` `GimbalState` (MCU → host), `0x0A81` `GimbalCommand` (host → MCU). Little-endian, `f32`.
    - One thread owns the port and does both reads (1 ms timeout) and writes, so a send waits at most 1 ms.
    - `serialport` is built without its libudev feature, so it builds on the Jetson without extra system libraries.

## Delegation
21. **Subagents (Haiku) wrote first drafts. Several reported success on code that was stubbed or wrong**:
    - Detector inference was a TODO.
    - The tracker's Gauss-Newton solver used a scalar residual.
    - The aimer's ballistics failed its own tests.
    - The MCAP reader returned blank frames.
    - The overlay drew no text.
    Each was rewritten or fixed and is now covered by a test that would catch it.
