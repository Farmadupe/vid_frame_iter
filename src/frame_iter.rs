// This example demonstrates how to extract all frames from a video file for
//further processing
//

use std::iter::FusedIterator;

use gstreamer::{prelude::*, ClockTime, CoreError, MessageView, StateChangeSuccess};

use gstreamer_video::VideoFrameExt;

#[cfg(feature = "dep:image")]
use image::GenericImageView;

#[derive(Debug, Clone)]
pub struct VideoFrameIterBuilder {
    uri: String,
    fps: Option<(u64, u64)>,
    start_offset: Option<f64>,
}

impl VideoFrameIterBuilder {
    #[must_use]
    /// Create a [`VideoFrameIterBuilder`] from the given URI.
    pub fn from_uri(uri: impl AsRef<str>) -> VideoFrameIterBuilder {
        Self {
            uri: uri.as_ref().to_string(),
            fps: None,
            start_offset: None,
        }
    }

    pub fn uri(&self) -> &str {
        &self.uri
    }

    /// Change the frame rate of the iterator. The argument is a fraction, for example:
    /// * For a framerate of one per 3 seconds, use (1, 3).
    /// * For a framerate of 12.34 frames per second use (1234 / 100).
    pub fn frame_rate(&mut self, fps: (u64, u64)) {
        self.fps = Some(fps);
    }

    /// Jump to the given time in seconds before beginning to return frames.
    pub fn start_offset(&mut self, seconds: f64) {
        self.start_offset = Some(seconds);
    }

    /// Consumes the builder and creates an iterator returning video frames.
    /// Frames are grayscale, with 8 bits per pixel.
    pub fn spawn_gray(&self) -> Result<VideoFrameIter<GrayFrame>, glib::Error> {
        self.create_pipeline::<GrayFrame>()
    }

    /// Consumes the builder and creates an iterator returning video frames.
    /// Frames are Rgb, with 8 bits per colour.
    pub fn spawn_rgb(&self) -> Result<VideoFrameIter<RgbFrame>, glib::Error> {
        self.create_pipeline::<RgbFrame>()
    }

    fn create_pipeline<RF: VideoFrame>(&self) -> Result<VideoFrameIter<RF>, glib::Error> {
        let fps_arg = match self.fps {
            None => String::from(""),
            Some((numer, denom)) => {
                format!("videorate name=rate ! capsfilter name=ratefilter ! video/x-raw,framerate={numer}/{denom} ! ")
            }
        };

        // Create our pipeline from a pipeline description string.
        let src_path = &self.uri;
        let pipeline_desc =
            format!("uridecodebin uri=\"{src_path}\" buffer-size=1 ! {fps_arg} videoconvert ! appsink name=sink");

        let pipeline = gstreamer::parse::launch(&pipeline_desc)?
            .downcast::<gstreamer::Pipeline>()
            .expect("Expected a gstreamer::Pipeline");

        // Get access to the appsink element.
        let appsink = pipeline
            .by_name("sink")
            .expect("Sink element not found")
            .downcast::<gstreamer_app::AppSink>()
            .expect("Sink element is expected to be an appsink!");

        // Don't synchronize on the clock.
        appsink.set_property("sync", false);

        // To save memory and CPU time prevent the sink element decoding any more than the minimum.
        appsink.set_max_buffers(1);
        appsink.set_drop(false);

        // Tell the appsink what format we want.
        // This can be set after linking the two objects, because format negotiation between
        // both elements will happen during pre-rolling of the pipeline.
        appsink.set_caps(Some(
            &gstreamer::Caps::builder("video/x-raw")
                .field("format", RF::gst_video_format().to_str())
                .build(),
        ));

        let pipeline = VideoFrameIter::<RF> {
            pipeline,
            fused: false,
            _phantom: std::marker::PhantomData,
        };
        pipeline.pause()?;

        if let Some(skip_amount) = self.start_offset {
            pipeline.seek_accurate(skip_amount)?;
        }

        pipeline.play()?;
        Ok(pipeline)
    }
}

fn change_state_blocking(
    pipeline: &gstreamer::Pipeline,
    new_state: gstreamer::State,
) -> Result<(), glib::Error> {
    use StateChangeSuccess::*;
    let timeout = 10 * gstreamer::ClockTime::SECOND;

    let state_change_error = match pipeline.set_state(new_state) {
        Ok(Success | NoPreroll) => return Ok(()),
        Ok(Async) => {
            let (result, _curr, _pending) = pipeline.state(timeout);
            match result {
                Ok(Success | NoPreroll) => return Ok(()),

                //state change failed within timeout. Treat as error
                Ok(Async) => None,
                Err(e) => Some(e),
            }
        }

        Err(e) => Some(e),
    };

    //If there was any error then return that.
    //If no error but timed out then say so.
    //If no error and no timeout then any report will do.
    let error: glib::Error =
        if let Some(e) = get_bus_errors(&pipeline.bus().expect("failed to get gst bus")).next() {
            e
        } else if let Some(_e) = state_change_error {
            glib::Error::new(
                gstreamer::CoreError::TooLazy,
                "Gstreamer State Change Error",
            )
        } else {
            glib::Error::new(gstreamer::CoreError::TooLazy, "Internal Gstreamer error")
        };

    //before returning, close down the pipeline to prevent memory leaks.
    //But if the pipeline can't close, cause a panic (preferable to memory leak)
    match change_state_blocking(pipeline, gstreamer::State::Null) {
        Ok(()) => Err(error),
        Err(e) => panic!("{e:?}"),
    }
}

fn into_glib_error(msg: gstreamer::Message) -> glib::Error {
    match msg.view() {
        MessageView::Error(e) => e.error(),
        MessageView::Warning(w) => w.error(),
        _ => {
            panic!("Only Warning and Error messages can be converted into GstreamerError")
        }
    }
}

// Drain all messages from the bus, keeping track of eos and error.
//(This prevents messages piling up and causing memory leaks)
fn get_bus_errors(bus: &gstreamer::Bus) -> impl Iterator<Item = glib::Error> + '_ {
    let errs_warns = [
        gstreamer::MessageType::Error,
        gstreamer::MessageType::Warning,
    ];

    std::iter::from_fn(move || bus.pop_filtered(&errs_warns).map(into_glib_error))
}

pub(crate) mod private {
    pub trait VideoFrameInternal {
        fn new(sample: gstreamer::Sample) -> Self;
        fn gst_video_format() -> gstreamer_video::VideoFormat;
    }
}
use private::VideoFrameInternal;

pub trait VideoFrame: VideoFrameInternal {
    /// Get a reference to the raw framebuffer data from gstreamer
    fn raw_frame(&self) -> &gstreamer_video::VideoFrame<gstreamer_video::video_frame::Readable>;

    /// The time of this frame
    fn pts(&self) -> std::time::Duration;
}

/// Conversion functions to types provided by the popular[`image`] crate.
#[cfg(feature = "dep:image")]
pub trait ImageFns {
    type IB;

    /// Get a [`image::FlatSamples`] for this frame with a borrowed reference to the underlying frame data.
    fn as_flat(&self) -> image::FlatSamples<&[u8]>;

    /// Copy the underlying frame data into an owned [`image::ImageBuffer`].
    fn to_imagebuffer(&self) -> Self::IB;
}

// Iterates over all the frames in a video.
// The iterator will prduce Ok(frame) until
// no more frames can be read. If any error occurred
// (regardless of whether it stopped the underlying pipeline or not)
// the iterator will produce Err(error).
// Once all frames and the first error has been produced the iterator
// will produce None.
#[derive(Debug)]
pub struct VideoFrameIter<RF: VideoFrame> {
    //Source of video frames
    pipeline: gstreamer::Pipeline,

    //Whether the last frame has been returned
    fused: bool,

    _phantom: std::marker::PhantomData<RF>,
}

impl<RF: VideoFrame> FusedIterator for VideoFrameIter<RF> {}
impl<RF: VideoFrame> Iterator for VideoFrameIter<RF> {
    type Item = Result<RF, glib::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_buf().map(|res| res.map(RF::new))
    }
}

impl<RF: VideoFrame> VideoFrameIter<RF> {
    fn next_buf(&mut self) -> Option<Result<gstreamer::Sample, glib::Error>> {
        //the amount of time to wait for a frame before assuming there are
        //none left.
        let try_pull_sample_timeout = 30 * gstreamer::ClockTime::SECOND;

        //required for FusedIterator
        if self.fused {
            return None;
        }

        let bus = self
            .pipeline
            .bus()
            .expect("Failed to get pipeline from bus. Shouldn't happen!");

        // Get access to the appsink element.
        let appsink = self
            .pipeline
            .by_name("sink")
            .expect("Sink element not found")
            .downcast::<gstreamer_app::AppSink>()
            .expect("Sink element is expected to be an appsink!");

        //If any error/warning occurred, then return it now.
        if let Some(error) = Self::try_find_error(&bus) {
            return Some(Err(error));
        }

        let sample = appsink.try_pull_sample(try_pull_sample_timeout);
        match sample {
            //If a frame was extracted then return it.
            Some(sample) => Some(Ok(sample)),

            None => {
                // Make sure no more frames can be drawn if next is called again
                self.fused = true;

                //if no sample was returned then we might have hit the timeout.
                //If so check for any possible error being written into the log
                //at that time
                let ret = if let Some(error) = Self::try_find_error(&bus) {
                    Some(Err(error))
                }
                // if gstreamer produced no frame but also didn't return any error
                // (maybe CPU was overloaded?), so inform the end user
                else if !appsink.is_eos() {
                    Some(Err(glib::Error::new(
                        CoreError::TooLazy,
                        "Gstreamer timed out",
                    )))

                // Otherwise we hit EOS and nothing else suspicious happened
                } else {
                    None
                };

                match change_state_blocking(&self.pipeline, gstreamer::State::Null) {
                    Ok(()) => ret,
                    Err(e) => panic!("{e:?}"),
                }
            }
        }
    }

    fn pause(&self) -> Result<(), glib::Error> {
        change_state_blocking(&self.pipeline, gstreamer::State::Paused)
    }

    fn play(&self) -> Result<(), glib::Error> {
        change_state_blocking(&self.pipeline, gstreamer::State::Playing)
    }

    /// Seek to the given position in the file, passing the 'accurate' flag to gstreamer.
    /// If you want to make large jumps in a video file this may be faster than setting a
    /// very low framerate (because with a low framerate, gstreamer still decodes every frame).
    pub fn seek_accurate(&self, time: f64) -> Result<(), glib::Error> {
        use gstreamer::SeekFlags;
        let time_ns_f64 = time * ClockTime::SECOND.nseconds() as f64;
        let time_ns_u64 = time_ns_f64 as u64;
        let flags = SeekFlags::ACCURATE.union(SeekFlags::FLUSH);

        self.pipeline
            .seek_simple(flags, gstreamer::ClockTime::from_nseconds(time_ns_u64))
            .map_err(|e| glib::Error::new(CoreError::TooLazy, &e.message))
    }

    fn try_find_error(bus: &gstreamer::Bus) -> Option<glib::Error> {
        bus.pop_filtered(&[
            gstreamer::MessageType::Error,
            gstreamer::MessageType::Warning,
        ])
        .filter(|msg| matches!(msg.view(), MessageView::Error(_) | MessageView::Warning(_)))
        .map(into_glib_error)
    }
}

//Must ensure all refcounted gobjects are cleaned up by the glib runtime.
//This won't happen unless we set the pipeline state to null
//
//Unfortunately there's no way to capture if something goes wrong, which
//could cause silent memory leaks, so prefer to panic instead here.
impl<T: VideoFrame> Drop for VideoFrameIter<T> {
    fn drop(&mut self) {
        match change_state_blocking(&self.pipeline, gstreamer::State::Null) {
            Ok(()) => (),
            Err(e) => panic!("{e:?}"),
        };
    }
}

/// A single video frame, with 8 bits per pixel, grayscale encoding.
///
/// You can access the raw data by:
/// * calling [`ImageFns::to_imagebuffer`] to copy the raw pixels into an owned [`image::ImageBuffer`].
/// * calling [`ImageFns::as_flat`] to get a [`image::FlatSamples`] struct representing the layout of the frame's raw data.
/// * directly indexing individual pixels using the functions from the [`image::GenericImageView`] trait.
/// * calling [`VideoFrame::raw_frame`] to get a reference to the raw data in gstreamer's internal format.
///
/// # Lifetimes and ownership
/// The underlying raw frame data is owned and reference counted by gstreamer, so it is generally cheap to clone frames.
/// If you want to pass frames around in your code, it is better to clone them instead of handing outreferences. In other
/// words, you can treat frames as if they were wrapped by an [`std::rc::Rc`]
///
/// Most functions have been written to avoid copying raw frames. Currently the only function that does copy is [`ImageFns::to_imagebuffer`].
///
#[derive(Debug)]
pub struct GrayFrame {
    frame: gstreamer_video::VideoFrame<gstreamer_video::video_frame::Readable>,
    pts: std::time::Duration,
}

// Safety: Not safe by default because of raw pointer to pixel data (see other safety notes for the raw pointer)
// However the backing memory is internally managed by gstreamer with a refcounter (which will never get zeroed
// during the lifetime of any GrayFrame, as the existence GrayFrame's .frame object guarantees that the refcount
// can never be zero)
impl Clone for GrayFrame {
    /// Clone this video frame. This operation is cheap because it does not clone the underlying
    /// data (it actually relies on gstreamer's refcounting mechanism)
    fn clone(&self) -> Self {
        let buffer = self.frame.buffer_owned();
        let frame = gstreamer_video::VideoFrame::from_buffer_readable(buffer, self.frame.info())
            .expect("Failed to map buffer readable");
        Self {
            frame,
            pts: self.pts,
        }
    }
}

#[cfg(feature = "dep:image")]
impl GenericImageView for GrayFrame {
    type Pixel = image::Luma<u8>;

    fn dimensions(&self) -> (u32, u32) {
        self.as_flat()
            .as_view::<image::Luma<u8>>()
            .expect("unreachable")
            .dimensions()
    }

    fn get_pixel(&self, x: u32, y: u32) -> Self::Pixel {
        self.as_flat()
            .as_view::<image::Luma<u8>>()
            .expect("unreachable")
            .get_pixel(x, y)
    }
}

impl VideoFrameInternal for GrayFrame {
    fn new(sample: gstreamer::Sample) -> Self {
        let caps = sample.caps().expect("Sample without caps");
        let info = gstreamer_video::VideoInfo::from_caps(caps).expect("Failed to parse caps");

        let pts_ns = sample
            .buffer()
            .expect("Failed to get buffer from appsink")
            .pts()
            .expect("failed to get pts from buffer")
            .nseconds();
        let pts = std::time::Duration::from_nanos(pts_ns);

        let buffer = sample
            .buffer_owned()
            .expect("Failed to get buffer from appsink");

        let frame = gstreamer_video::VideoFrame::from_buffer_readable(buffer, &info)
            .expect("Failed to map buffer readable");

        Self { frame, pts }
    }

    fn gst_video_format() -> gstreamer_video::VideoFormat {
        gstreamer_video::VideoFormat::Gray8
    }
}

impl VideoFrame for GrayFrame {
    fn raw_frame(&self) -> &gstreamer_video::VideoFrame<gstreamer_video::video_frame::Readable> {
        &self.frame
    }

    fn pts(&self) -> std::time::Duration {
        self.pts
    }
}

#[cfg(feature = "dep:image")]
impl ImageFns for GrayFrame {
    type IB = image::GrayImage;

    fn as_flat(&self) -> image::FlatSamples<&[u8]> {
        //safety: See safety note for send/sync impl (gstreamer guarantees that this pointer exists and does not move
        //for the life of self)
        let data_ref: &[u8] = self
            .frame
            .plane_data(0)
            .expect("gray frames only have one plane");

        let layout = image::flat::SampleLayout {
            channels: 1,
            channel_stride: 1,
            width: self.frame.width(),
            width_stride: 1,
            height: self.frame.height(),
            height_stride: self.frame.plane_stride()[0] as usize,
        };

        let flat = image::FlatSamples {
            samples: data_ref,
            layout,

            color_hint: Some(image::ColorType::L8),
        };

        flat
    }

    #[must_use]
    fn to_imagebuffer(&self) -> Self::IB {
        let width = self.frame.width();
        let height = self.frame.height();

        let flat = self.as_flat();
        let view = flat.as_view().expect("unreachable");

        image::ImageBuffer::from_fn(width, height, |x, y| view.get_pixel(x, y))
    }
}

/// A single video frame, with 24 bits per pixel, Rgb encoding.
///
/// You can access the raw data by:
/// * calling [`ImageFns::to_imagebuffer`] to copy the raw pixels into an owned [`image::ImageBuffer`].
/// * calling [`ImageFns::as_flat`] to get a [`image::FlatSamples`] struct representing the layout of the frame's raw data.
/// * directly indexing individual pixels using the functions from the [`image::GenericImageView`] trait.
/// * calling [`VideoFrame::raw_frame`] to get a reference to the raw data in gstreamer's internal format.
///
/// # Lifetimes and ownership
/// The underlying raw frame data is owned and reference counted by gstreamer, so it is generally cheap to clone frames.
/// If you want to pass frames around in your code, it is better to clone them instead of handing outreferences. In other
/// words, you can treat frames as if they were wrapped by an [`std::rc::Rc`]
///
/// Most functions have been written to avoid copying raw frames. Currently the only function that does copy is [`ImageFns::to_imagebuffer`].
///
/// # Examples
#[derive(Debug)]
pub struct RgbFrame {
    frame: gstreamer_video::VideoFrame<gstreamer_video::video_frame::Readable>,
    pts: std::time::Duration,
}

// Safety: See safety note for GrayFrame.
impl Clone for RgbFrame {
    /// Clone this video frame. This operation is cheap because it does not clone the underlying
    /// data (it actually relies on gstreamer's refcounting mechanism)
    fn clone(&self) -> Self {
        let buffer = self.frame.buffer_owned();
        let frame = gstreamer_video::VideoFrame::from_buffer_readable(buffer, self.frame.info())
            .expect("Failed to map buffer readable");

        Self {
            frame,
            pts: self.pts,
        }
    }
}

#[cfg(feature = "dep:image")]
impl GenericImageView for RgbFrame {
    type Pixel = image::Rgb<u8>;

    fn dimensions(&self) -> (u32, u32) {
        self.as_flat()
            .as_ref()
            .as_view::<image::Rgb<u8>>()
            .expect("unreachable")
            .dimensions()
    }

    fn get_pixel(&self, x: u32, y: u32) -> Self::Pixel {
        self.as_flat()
            .as_view::<image::Rgb<u8>>()
            .expect("unreachable")
            .get_pixel(x, y)
    }
}

impl VideoFrameInternal for RgbFrame {
    fn new(sample: gstreamer::Sample) -> Self {
        let caps = sample.caps().expect("Sample without caps");
        let info = gstreamer_video::VideoInfo::from_caps(caps).expect("Failed to parse caps");

        let pts_ns = sample
            .buffer()
            .expect("Failed to get buffer from appsink")
            .pts()
            .expect("failed to get pts from buffer")
            .nseconds();
        let pts = std::time::Duration::from_nanos(pts_ns);

        let buffer = sample
            .buffer_owned()
            .expect("Failed to get buffer from appsink");

        let frame = gstreamer_video::VideoFrame::from_buffer_readable(buffer, &info)
            .expect("Failed to map buffer readable");

        Self { frame, pts }
    }

    fn gst_video_format() -> gstreamer_video::VideoFormat {
        gstreamer_video::VideoFormat::Rgb
    }
}

impl VideoFrame for RgbFrame {
    fn raw_frame(&self) -> &gstreamer_video::VideoFrame<gstreamer_video::video_frame::Readable> {
        &self.frame
    }

    fn pts(&self) -> std::time::Duration {
        self.pts
    }
}

#[cfg(feature = "dep:image")]
impl ImageFns for RgbFrame {
    type IB = image::RgbImage;

    fn as_flat(&self) -> image::FlatSamples<&[u8]> {
        //safety: See safety note for send/sync impl (gstreamer guarantees that this pointer exists and does not move
        //for the life of self)
        let data_ref: &[u8] = self.frame.plane_data(0).expect("rgb frames have 1 plane");
        let layout = image::flat::SampleLayout {
            channels: 3,
            channel_stride: 1,
            width: self.frame.width(),
            width_stride: 3,
            height: self.frame.height(),
            height_stride: self.frame.plane_stride()[0] as usize,
        };

        image::FlatSamples {
            samples: data_ref,
            layout,
            color_hint: Some(image::ColorType::Rgb8),
        }
    }

    #[must_use]
    fn to_imagebuffer(&self) -> Self::IB {
        let width = self.frame.width();
        let height = self.frame.height();

        let flat = self.as_flat();
        let view = flat.as_view().expect("unreachable");

        image::ImageBuffer::from_fn(width, height, |x, y| view.get_pixel(x, y))
    }
}
