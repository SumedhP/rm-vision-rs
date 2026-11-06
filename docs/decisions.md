# Decision log

Calls made while building this overnight without the operator. Each says what was chosen, why, and what would change it.

1. **One crate, modules mirror the design doc** (`io`, `sync`, `detector`, `tracker`, `aimer`, `viz`, `calib`, `pipeline`), binaries in `src/bin`. A workspace of crates would add build and review overhead without a second consumer.
2. **Sources are enums, not traits** (`CameraSource::{Video, Log}`, `RobotSource::{Defaults, Log}`). The set is closed; `Live` variants are added when the camera SDK and serial exist. No dynamic dispatch needed.
3. **Video decoding through an `ffmpeg` child process** piping raw RGB. No native video library to build on Windows/Jetson; ffmpeg is already on both.
4. **`image` crate instead of kornia-rs** for resize/JPEG. The pipeline only needs letterbox resize, JPEG encode and pixel access; `image` is pure Rust and already covers it.
5. **Armor pose: center ray + lightbar-length depth, then yaw search with the 15 degree tilt fixed** (the sp_vision `optimize_yaw` idea) refined by Gauss-Newton on reprojection error, instead of IPPE. With the tilt fixed the coplanar two-fold ambiguity disappears.
6. **Model I/O** (read from SZURP `V8_21InfantryPostProcessor`): input `1x3x480x640` RGB / 255, letterboxed with pad value 124; output `21x6300` = 4 color scores, 9 class scores, 4 keypoints (TL, BL, BR, TR) in model pixels. Scores are already sigmoided.
7. **Test data has no IMU**, so `RobotSource::Defaults` supplies a fixed level gimbal from `configs/robots/realsense_test.yaml`. Ground truth from `docs/realsense_dataset.md`: radius 0.25 m, plate heights 0.08 / 0.16 m (h = 0.08), distances 2 to 6 m.
