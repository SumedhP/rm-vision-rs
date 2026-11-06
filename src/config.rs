//! YAML configuration, split the way the design doc describes: general tunings, per-robot
//! geometry/timing, and logging.

use std::path::{Path, PathBuf};

use color_eyre::eyre::WrapErr;
use serde::Deserialize;
use serde::de::DeserializeOwned;

use crate::aimer::GimbalLimits;
use crate::geometry::{CameraExtrinsics, CameraIntrinsics};
use crate::io::RobotState;

/// Robot-agnostic tunings (`configs/general.yaml`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralConfig {
    /// Detector settings.
    pub detector: crate::detector::DetectorConfig,
    /// Tracker settings.
    pub tracker: crate::tracker::TrackerConfig,
    /// Aimer settings.
    pub aimer: crate::aimer::AimerConfig,
}

/// Per-robot geometry and timing (`configs/robots/<name>.yaml`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotConfig {
    /// Camera intrinsics.
    pub camera: CameraIntrinsics,
    /// Camera mounting on the gimbal.
    pub extrinsics: CameraExtrinsics,
    /// Exposure-to-arrival delay from the ChArUco calibration, seconds.
    pub camera_delay: f64,
    /// Host-to-MCU command delay, seconds (tuned).
    pub downlink_delay: f64,
    /// Gimbal acceleration limits.
    pub gimbal: GimbalLimits,
    /// Robot state used when the source has none (plain videos).
    pub default_state: RobotState,
}

/// Logging and live visualization (`configs/logging.yaml`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoggingConfig {
    /// Directory MCAP logs are written to.
    pub dir: PathBuf,
    /// Camera frames per second to store in the log.
    pub frame_rate: f64,
}

/// Reads and parses one YAML file.
///
/// # Errors
/// File missing or not matching the schema.
pub fn load<T: DeserializeOwned>(path: &Path) -> color_eyre::Result<T> {
    let text = std::fs::read_to_string(path).wrap_err_with(|| format!("reading {}", path.display()))?;
    serde_saphyr::from_str(&text).wrap_err_with(|| format!("parsing {}", path.display()))
}
