//! The image a detector reads: 8-bit grayscale pixels, row-major.

use crate::DetectError;

/// A borrowed 8-bit grayscale image: `width × height` pixels, row-major,
/// one byte per pixel, no row padding.
///
/// Decode and convert the image with any library, then view its luminance
/// buffer; a colour image goes through its luma conversion first.
#[derive(Debug, Clone, Copy)]
pub struct GrayImageView<'a> {
    width: u32,
    height: u32,
    pixels: &'a [u8],
}

impl<'a> GrayImageView<'a> {
    /// View `pixels` as a `width × height` image.
    ///
    /// # Errors
    ///
    /// [`DetectError::InvalidConfig`] when `pixels` does not hold exactly
    /// `width × height` bytes.
    pub fn new(width: u32, height: u32, pixels: &'a [u8]) -> Result<Self, DetectError> {
        let expected = width as usize * height as usize;
        if pixels.len() != expected {
            return Err(DetectError::InvalidConfig(format!(
                "a {width}x{height} grayscale image needs {expected} bytes, got {}",
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The pixels, row-major.
    pub fn pixels(&self) -> &'a [u8] {
        self.pixels
    }

    /// An owned copy as the image type the wrapped detectors take.
    pub(crate) fn to_gray_image(self) -> image::GrayImage {
        image::GrayImage::from_raw(self.width, self.height, self.pixels.to_vec())
            .expect("GrayImageView::new checked the buffer length")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_the_buffer_length() {
        let pixels = [0u8; 12];
        let view = GrayImageView::new(4, 3, &pixels).unwrap();
        assert_eq!((view.width(), view.height()), (4, 3));
        assert!(GrayImageView::new(5, 3, &pixels).is_err());
    }
}
