//! Pose interpolation at frame time. Both link latencies are fixed config values
//! (`imu_delay`, `downlink_delay`); see `docs/design.md` for how to measure them.

pub mod pose_buffer;

pub use pose_buffer::PoseBuffer;
