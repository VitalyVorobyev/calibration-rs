//! Dense diagnostic maps over an image.
//!
//! A map is a set of planes over an [`ImageGrid`]: row-major `Vec<f64>`
//! values, one per node, with `NaN` where the map has no value.

mod grid;

pub use grid::ImageGrid;

#[cfg(test)]
mod tests;
