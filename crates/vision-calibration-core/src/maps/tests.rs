//! Tests of the diagnostic maps.

use super::*;

// ─── A7: the grid ───────────────────────────────────────────────────────────

#[test]
fn grid_nodes_are_block_centres_row_major() {
    let grid = ImageGrid::new([1280, 960], 16).unwrap();
    assert_eq!((grid.columns(), grid.rows(), grid.len()), (80, 60, 4800));
    assert_eq!(grid.node(0, 0), [7.5, 7.5]);
    assert_eq!(grid.node(79, 59), [1271.5, 951.5]);
    let nodes: Vec<[f64; 2]> = grid.nodes().collect();
    assert_eq!(nodes.len(), grid.len());
    assert_eq!(nodes[1], [23.5, 7.5]);
    assert_eq!(nodes[80], [7.5, 23.5]);
    assert_eq!(nodes[4799], grid.node(79, 59));
}

#[test]
fn grid_partial_last_block_is_centred_on_its_pixels() {
    // 1000 = 62·16 + 8: the last column holds pixels 992..=999.
    let grid = ImageGrid::new([1000, 3], 16).unwrap();
    assert_eq!((grid.columns(), grid.rows()), (63, 1));
    assert_eq!(grid.node(62, 0), [995.5, 1.0]);
    // Stride 1: one node per pixel, at the pixel centre.
    let pixels = ImageGrid::new([3, 2], 1).unwrap();
    let nodes: Vec<[f64; 2]> = pixels.nodes().collect();
    assert_eq!(
        nodes,
        vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0]
        ]
    );
    // One block larger than the image: its node is the image centre.
    let one = ImageGrid::new([5, 3], 8).unwrap();
    assert_eq!(one.len(), 1);
    assert_eq!(one.node(0, 0), [2.0, 1.0]);
}

#[test]
fn grid_with_columns_rounds_the_stride_up() {
    let grid = ImageGrid::with_columns([1280, 960], 64).unwrap();
    assert_eq!((grid.stride(), grid.columns(), grid.rows()), (20, 64, 48));
    let grid = ImageGrid::with_columns([1000, 750], 64).unwrap();
    assert_eq!((grid.stride(), grid.columns(), grid.rows()), (16, 63, 47));
    // More columns than pixels: stride 1, one node per pixel.
    let grid = ImageGrid::with_columns([10, 5], 64).unwrap();
    assert_eq!((grid.stride(), grid.columns(), grid.rows()), (1, 10, 5));
    assert_eq!(grid.image_size(), [10, 5]);
}

/// The layout of the viewer's dense-plane format: `columns × rows` cells of
/// `stride` pixels, the trailing cell partial.
#[test]
fn grid_matches_the_viewer_plane_layout() {
    for (size, stride) in [([1920, 1080], 7), ([640, 480], 32), ([4000, 3000], 15)] {
        let grid = ImageGrid::new(size, stride).unwrap();
        let columns = size[0].div_ceil(stride) as usize;
        let rows = size[1].div_ceil(stride) as usize;
        assert_eq!((grid.columns(), grid.rows()), (columns, rows));
        for (c, r) in [(0, 0), (columns - 1, rows - 1), (columns / 2, rows / 3)] {
            let [x, y] = grid.node(c, r);
            // Inside the image and within half a block of the cell's centre.
            assert!((0.0..=f64::from(size[0] - 1)).contains(&x));
            assert!((0.0..=f64::from(size[1] - 1)).contains(&y));
            let half = f64::from(stride) / 2.0;
            assert!((x - (c as f64 * f64::from(stride) + half - 0.5)).abs() <= half);
            assert!((y - (r as f64 * f64::from(stride) + half - 0.5)).abs() <= half);
        }
    }
}

#[test]
fn grid_rejects_empty_input() {
    assert!(ImageGrid::new([0, 10], 4).is_err());
    assert!(ImageGrid::new([10, 0], 4).is_err());
    assert!(ImageGrid::new([10, 10], 0).is_err());
    assert!(ImageGrid::with_columns([10, 10], 0).is_err());
    assert!(ImageGrid::with_columns([0, 10], 4).is_err());
}
