//! The diagnostic maps of a run's camera as value planes: the library's
//! distortion and residual maps (`vision_calibration::analysis`) and the
//! projection σ, all on one [`ImageGrid`], each plane a `VAM1` blob with
//! what its channels hold. Also the per-feature residual records and their
//! statistics, from the export's `per_feature_residuals`.
//!
//! Every displacement is `actual − reference` in pixels: a distortion
//! layer is the camera's pixel minus the pixel without that stage, a
//! residual is `observed − projected`.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use vision_calibration::analysis::{
    DisplacementPlane, ImageGrid, ReprojectionStats, TargetFeatureResidual,
};

use crate::blobs::{BlobRef, BlobStore};
use crate::planes;

/// The grid columns a map has unless asked otherwise: a residual field is
/// resolvable only at bandwidths of a few percent of the diagonal, and the
/// distortion and σ fields are smooth, so a finer grid adds cost, not
/// detail (the viewer interpolates).
pub const DEFAULT_COLUMNS: u32 = 64;
/// The most grid columns a map may have.
pub const MAX_COLUMNS: u32 = 1024;

/// The nodes every plane of a map is sampled at: the centres of
/// `stride × stride` pixel blocks over the camera's image (its region of
/// interest for a tiled camera), row-major from the top-left. Pixel
/// centres are at integer coordinates; block `(c, r)`'s node is at
/// `x = (c·stride + min((c + 1)·stride, width) − 1) / 2` (likewise `y`), so
/// a trailing partial block is centred on its in-image pixels. The
/// `VAM1` header of every plane carries `columns`, `rows` and `stride`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MapGrid {
    /// `[width, height]` of the camera's image, pixels.
    pub image_size: [u32; 2],
    /// The block size, pixels.
    pub stride: u32,
    pub columns: usize,
    pub rows: usize,
}

impl MapGrid {
    pub fn of(grid: &ImageGrid) -> Self {
        Self {
            image_size: grid.image_size(),
            stride: grid.stride(),
            columns: grid.columns(),
            rows: grid.rows(),
        }
    }
}

/// The grid of at most `columns` columns over `image_size`.
pub fn grid(image_size: [u32; 2], columns: u32) -> Result<ImageGrid, String> {
    if columns == 0 || columns > MAX_COLUMNS {
        return Err(format!(
            "a map has 1 to {MAX_COLUMNS} columns, not {columns}"
        ));
    }
    ImageGrid::with_columns(image_size, columns).map_err(|e| e.to_string())
}

/// One channel of a [`MapPlane`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PlaneChannel {
    /// What the channel holds (`dx`, `dy`, `magnitude`, `rms`, …).
    pub name: String,
    /// Its unit: `px`, `chi` (a significance), `samples`, `views`, `flag`
    /// (1 or 0).
    pub unit: String,
    /// `[min, max]` of its finite values; `null` when no node has one.
    pub domain: Option<[f64; 2]>,
}

/// A value plane of a map: a `VAM1` blob (the `blob` resource) whose
/// channels, in order, are `channels`. `NaN` marks a node without a value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MapPlane {
    pub blob: BlobRef,
    pub channels: Vec<PlaneChannel>,
}

/// The finite range of `values`.
fn domain(values: &[f64]) -> Option<[f64; 2]> {
    values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(None, |range, v| match range {
            None => Some([v, v]),
            Some([lo, hi]) => Some([lo.min(v), hi.max(v)]),
        })
}

/// `channels` (`(name, unit, values)`, each one value per node of `grid`)
/// encoded as one plane and kept in `blobs` as `id`.
pub fn put_plane(
    blobs: &BlobStore,
    id: String,
    grid: &ImageGrid,
    channels: &[(&str, &str, &[f64])],
) -> Result<MapPlane, String> {
    let as_f32: Vec<Vec<f32>> = channels
        .iter()
        .map(|(_, _, values)| values.iter().map(|&v| v as f32).collect())
        .collect();
    let refs: Vec<&[f32]> = as_f32.iter().map(Vec::as_slice).collect();
    let bytes = planes::encode(grid.columns(), grid.rows(), grid.stride(), &refs)
        .map_err(|e| format!("encode the plane: {e}"))?;
    Ok(MapPlane {
        blob: blobs.put(id, planes::MEDIA_TYPE, bytes),
        channels: channels
            .iter()
            .map(|(name, unit, values)| PlaneChannel {
                name: (*name).to_owned(),
                unit: (*unit).to_owned(),
                domain: domain(values),
            })
            .collect(),
    })
}

/// A displacement layer as its three channels: `dx`, `dy` and their
/// length, pixels.
pub fn displacement_channels(plane: &DisplacementPlane) -> [(&'static str, Vec<f64>); 3] {
    [
        ("dx", plane.dx.clone()),
        ("dy", plane.dy.clone()),
        ("magnitude", plane.magnitude()),
    ]
}

/// The major-axis σ of a symmetric 2×2 covariance `[[a, b], [b, c]]`.
pub fn major_axis_sigma(a: f64, b: f64, c: f64) -> f64 {
    let mean = 0.5 * (a + c);
    let half = 0.5 * (a - c);
    (mean + half.hypot(b)).max(0.0).sqrt()
}

/// Reprojection statistics of a group of records, from the library's
/// [`ReprojectionStats`], with the records whose projection failed.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GroupStats {
    /// Mean, RMS and largest error length `|observed − projected|`, pixels,
    /// over the records with a projection (all zero when there is none).
    pub stats: ReprojectionStats,
    /// Records without a projection (it diverged), not in `stats`.
    pub unprojected: usize,
}

#[derive(Default)]
struct Errors {
    errors: Vec<f64>,
    unprojected: usize,
}

impl Errors {
    fn push(&mut self, record: &TargetFeatureResidual) {
        match record.residual_px() {
            Some([dx, dy]) if dx.is_finite() && dy.is_finite() => self.errors.push(dx.hypot(dy)),
            _ => self.unprojected += 1,
        }
    }

    fn stats(&self) -> GroupStats {
        GroupStats {
            stats: ReprojectionStats::from_errors(&self.errors),
            unprojected: self.unprojected,
        }
    }
}

/// One camera's statistics.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CameraStats {
    pub camera: usize,
    #[serde(flatten)]
    pub group: GroupStats,
    /// The views it has records in.
    pub views: usize,
}

/// One view's statistics, over every camera that saw it.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ViewStats {
    pub view: usize,
    #[serde(flatten)]
    pub group: GroupStats,
}

/// One camera's statistics in one view.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CellStats {
    pub camera: usize,
    pub view: usize,
    #[serde(flatten)]
    pub group: GroupStats,
}

/// The reprojection statistics of a run's per-feature records: per camera,
/// per view (cameras pooled) and per (camera, view), each ascending.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ResidualStats {
    pub cameras: Vec<CameraStats>,
    pub views: Vec<ViewStats>,
    pub cells: Vec<CellStats>,
}

pub fn residual_stats(records: &[TargetFeatureResidual]) -> ResidualStats {
    let mut cameras: BTreeMap<usize, Errors> = BTreeMap::new();
    let mut views: BTreeMap<usize, Errors> = BTreeMap::new();
    let mut cells: BTreeMap<(usize, usize), Errors> = BTreeMap::new();
    for record in records {
        cameras.entry(record.camera).or_default().push(record);
        views.entry(record.view).or_default().push(record);
        cells
            .entry((record.camera, record.view))
            .or_default()
            .push(record);
    }
    let views_of = |camera: usize| cells.keys().filter(|(c, _)| *c == camera).count();
    ResidualStats {
        cameras: cameras
            .iter()
            .map(|(&camera, errors)| CameraStats {
                camera,
                group: errors.stats(),
                views: views_of(camera),
            })
            .collect(),
        views: views
            .iter()
            .map(|(&view, errors)| ViewStats {
                view,
                group: errors.stats(),
            })
            .collect(),
        cells: cells
            .iter()
            .map(|(&(camera, view), errors)| CellStats {
                camera,
                view,
                group: errors.stats(),
            })
            .collect(),
    }
}

/// A camera's per-feature residual records, one column per field; entry
/// `i` of every column is one record.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ResidualRecords {
    /// The record's index in the export's `per_feature_residuals.target`
    /// (what `residual_map`'s `outliers` name).
    pub index: Vec<usize>,
    /// Its view.
    pub view: Vec<usize>,
    /// Its feature's index in the view's detection.
    pub feature: Vec<usize>,
    /// The target point, metres, as the calibration used it.
    pub target_xyz_m: Vec<[f64; 3]>,
    /// The detected pixel.
    pub observed_px: Vec<[f64; 2]>,
    /// The model's projection of the target point; `null` where it
    /// diverged.
    pub projected_px: Vec<Option<[f64; 2]>>,
    /// `observed − projected`, pixels; `null` where the projection
    /// diverged.
    pub residual_px: Vec<Option<[f64; 2]>>,
}

/// The records of `camera` (of one `view`, when given), in export order.
pub fn residual_records(
    records: &[TargetFeatureResidual],
    camera: usize,
    view: Option<usize>,
) -> ResidualRecords {
    let mut out = ResidualRecords::default();
    for (index, record) in records.iter().enumerate() {
        if record.camera != camera || view.is_some_and(|v| v != record.view) {
            continue;
        }
        out.index.push(index);
        out.view.push(record.view);
        out.feature.push(record.feature);
        out.target_xyz_m.push(record.target_xyz_m);
        out.observed_px.push(record.observed_px);
        out.projected_px.push(record.projected_px);
        out.residual_px.push(record.residual_px());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(
        camera: usize,
        view: usize,
        observed: [f64; 2],
        projected: Option<[f64; 2]>,
    ) -> TargetFeatureResidual {
        let mut r = TargetFeatureResidual::default();
        r.camera = camera;
        r.view = view;
        r.observed_px = observed;
        r.projected_px = projected;
        r
    }

    #[test]
    fn statistics_group_by_camera_view_and_cell() {
        let records = [
            record(0, 0, [3.0, 4.0], Some([0.0, 0.0])),
            record(0, 1, [1.0, 0.0], Some([0.0, 0.0])),
            record(1, 1, [0.0, 0.0], None),
        ];
        let stats = residual_stats(&records);
        assert_eq!(stats.cameras.len(), 2);
        let cam0 = &stats.cameras[0].group;
        assert_eq!(cam0.stats.count, 2);
        assert!((cam0.stats.mean - 3.0).abs() < 1e-12);
        assert!((cam0.stats.max - 5.0).abs() < 1e-12);
        assert!((cam0.stats.rms - 13f64.sqrt()).abs() < 1e-12);
        assert_eq!(stats.cameras[0].views, 2);
        assert_eq!(stats.cameras[1].group.unprojected, 1);
        assert_eq!(stats.cameras[1].group.stats.count, 0);
        // View 1 pools both cameras.
        assert_eq!(stats.views[1].group.stats.count, 1);
        assert_eq!(stats.views[1].group.unprojected, 1);
        assert_eq!(
            stats
                .cells
                .iter()
                .map(|c| (c.camera, c.view))
                .collect::<Vec<_>>(),
            [(0, 0), (0, 1), (1, 1)]
        );
    }

    #[test]
    fn records_are_columns_of_observed_minus_projected() {
        let records = [
            record(0, 0, [10.5, 20.0], Some([10.0, 20.25])),
            record(1, 0, [0.0, 0.0], Some([0.0, 0.0])),
            record(0, 1, [1.0, 1.0], None),
        ];
        let all = residual_records(&records, 0, None);
        assert_eq!(all.index, [0, 2]);
        assert_eq!(all.residual_px, [Some([0.5, -0.25]), None]);
        let one = residual_records(&records, 0, Some(1));
        assert_eq!(one.index, [2]);
    }

    #[test]
    fn a_plane_states_each_channel_and_its_finite_range() {
        let grid = ImageGrid::new([4, 2], 2).unwrap();
        let blobs = BlobStore::default();
        let plane = put_plane(
            &blobs,
            "p".into(),
            &grid,
            &[
                ("a", "px", &[1.0, f64::NAN]),
                ("b", "px", &[f64::NAN, f64::NAN]),
            ],
        )
        .unwrap();
        assert_eq!(plane.channels[0].domain, Some([1.0, 1.0]));
        assert_eq!(plane.channels[1].domain, None);
        let (_, bytes) = blobs.get(&plane.blob.id).unwrap();
        assert_eq!(&bytes[..4], b"VAM1");
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 2);
        assert!(super::grid([10, 10], 0).is_err());
        assert!(super::grid([10, 10], MAX_COLUMNS + 1).is_err());
    }

    #[test]
    fn the_major_axis_of_a_diagonal_covariance_is_its_larger_sigma() {
        assert!((major_axis_sigma(4.0, 0.0, 1.0) - 2.0).abs() < 1e-12);
        assert!((major_axis_sigma(1.0, 1.0, 1.0) - 2f64.sqrt()).abs() < 1e-12);
    }
}
