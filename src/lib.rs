#![allow(clippy::let_and_return)]
#![allow(clippy::zero_prefixed_literal)]
#![warn(clippy::redundant_pub_crate)]
#![warn(clippy::print_stdout)]
#![warn(clippy::print_stderr)]
//#![warn(clippy::pedantic)]
#![warn(clippy::unwrap_used)]

//! A crate for reading video frames and processing them as images, using gstreamer as a backend.
//!
//! To start reading frames, you create a [`VideoFrameIterBuilder`] with a URI to the location of the video. Then call `spawn_gray` of `spawn_rgb` to receive
//! an iterator over video frames.
//!
//! Has integration with the [`image`] crate for easy image processing (but also allows direct access to raw pixels if that's what you want)
//!
//! The interface is small and minimal.
//!
//!
//! # Supported operating systems
//! Currently only tested on Ubuntu Linux 22.04. This crate should work in MacOS and windows but this has not been tested.
//!
//! # Installing
//! You should follow the detailed instructions written for gstreamer-rs [here.](https://github.com/sdroege/gstreamer-rs#installation)
//!

/// Utilities for getting duration and dimensions of a video without decoding frames.
pub mod mediainfo_utils;

// Provides [`VideoFrameIter`] and [`VideoFrameIterBuilder`]
/// Functions for decoding on your Nvidia GPU.
pub mod extras;
pub mod frame_iter;

pub use frame_iter::GrayFrame;
#[cfg(feature="dep:image")]
pub use frame_iter::ImageFns;
pub use frame_iter::RgbFrame;
pub use frame_iter::VideoFrameIter;
pub use frame_iter::VideoFrameIterBuilder;


/// Initialize gstreamer. You must call this function before calling any other function in this crate.
pub fn init_gstreamer() {
    gstreamer::init().expect("Failed to initialize gstreamer")
}
