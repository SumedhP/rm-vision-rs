//! MCU link over serial: DJI `RoboMaster` frames ([`frame`]) carrying typed messages
//! ([`messages`]), moved by one thread that owns the port ([`link`]).

pub mod frame;
pub mod link;
pub mod messages;

pub use frame::Frame;
pub use link::{Received, SerialLink};
pub use messages::{GimbalCommand, GimbalState, Message};
