//! [`ImageGrid`]: the sample nodes a dense map is evaluated at.

use crate::Error;

/// Sample nodes over an image: the centres of `stride × stride` pixel
/// blocks, row-major from the top-left.
///
/// Pixel centres are at integer coordinates, so pixel `(0, 0)` covers
/// `[-0.5, 0.5]²`. Block `(c, r)` holds the pixels with
/// `c·stride ≤ x < (c + 1)·stride` (and likewise `y`), clipped to the image;
/// its node is the centre of the pixels it holds,
/// `x = (c·s + min((c + 1)·s, w) − 1) / 2`. A trailing partial block is
/// therefore centred on its in-image pixels.
///
/// The grid has `columns() × rows()` nodes. A plane over it (a distortion
/// or residual map) is a row-major `Vec<f64>` of [`len`](Self::len) values.
///
/// ```
/// use vision_calibration_core::ImageGrid;
///
/// let grid = ImageGrid::new([10, 4], 4)?;
/// assert_eq!((grid.columns(), grid.rows()), (3, 1));
/// assert_eq!(grid.node(0, 0), [1.5, 1.5]);
/// // The last block holds pixels 8 and 9 only.
/// assert_eq!(grid.node(2, 0), [8.5, 1.5]);
/// # Ok::<(), vision_calibration_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ImageGrid {
    image_size: [u32; 2],
    stride: u32,
}

impl ImageGrid {
    /// A grid of `stride × stride` pixel blocks over an image of
    /// `image_size = [width, height]` pixels.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a zero width, height or stride.
    pub fn new(image_size: [u32; 2], stride: u32) -> Result<Self, Error> {
        let [width, height] = image_size;
        if width == 0 || height == 0 {
            return Err(Error::invalid_input(format!(
                "image size {width}×{height} has no pixels"
            )));
        }
        if stride == 0 {
            return Err(Error::invalid_input("grid stride must be positive"));
        }
        Ok(Self { image_size, stride })
    }

    /// A grid of at most `columns` nodes per row: the stride is
    /// `ceil(width / columns)`, the same in both directions.
    ///
    /// The grid can have fewer columns than asked when the width is not a
    /// multiple of the stride (1000 px at 64 columns: stride 16, 63
    /// columns).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for zero `columns`, width or height.
    pub fn with_columns(image_size: [u32; 2], columns: u32) -> Result<Self, Error> {
        if columns == 0 {
            return Err(Error::invalid_input("grid columns must be positive"));
        }
        Self::new(image_size, image_size[0].div_ceil(columns).max(1))
    }

    /// `[width, height]` of the image, pixels.
    pub fn image_size(&self) -> [u32; 2] {
        self.image_size
    }

    /// The side of a block, pixels.
    pub fn stride(&self) -> u32 {
        self.stride
    }

    /// Nodes per row, `ceil(width / stride)`.
    pub fn columns(&self) -> usize {
        self.image_size[0].div_ceil(self.stride) as usize
    }

    /// Rows of nodes, `ceil(height / stride)`.
    pub fn rows(&self) -> usize {
        self.image_size[1].div_ceil(self.stride) as usize
    }

    /// The number of nodes, `columns() · rows()`; never zero.
    #[allow(clippy::len_without_is_empty)] // A grid always has at least one node.
    pub fn len(&self) -> usize {
        self.columns() * self.rows()
    }

    /// The pixel position of node `(column, row)`, the centre of its block's
    /// in-image pixels. Indices are `column < columns()`, `row < rows()`;
    /// others give positions off the image.
    pub fn node(&self, column: usize, row: usize) -> [f64; 2] {
        [
            Self::centre(column, self.stride, self.image_size[0]),
            Self::centre(row, self.stride, self.image_size[1]),
        ]
    }

    /// Every node's position, row-major from the top-left: the order of a
    /// plane's values.
    pub fn nodes(&self) -> impl Iterator<Item = [f64; 2]> + '_ {
        let columns = self.columns();
        (0..self.rows()).flat_map(move |r| (0..columns).map(move |c| self.node(c, r)))
    }

    /// The centre of block `i`'s pixels along an axis of `size` pixels.
    fn centre(i: usize, stride: u32, size: u32) -> f64 {
        let start = i as f64 * f64::from(stride);
        let end = ((i + 1) as f64 * f64::from(stride)).min(f64::from(size));
        (start + end - 1.0) / 2.0
    }
}
