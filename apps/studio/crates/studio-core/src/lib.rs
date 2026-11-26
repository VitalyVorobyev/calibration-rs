//! Calibration Studio's back end: what the desktop app and the browser
//! viewer compute, with no Tauri and no HTTP in it. The shells
//! (`src-tauri`'s Tauri commands and its `calib-viewer` server) are thin
//! wrappers over these functions.
//!
//! - [`project`] — project folders: create, open, save the manifest, save
//!   a run (`runs/<name>/`). See ADR 0014.
//! - [`run`] — the calibration runner, with stage progress and
//!   cancellation.
//! - [`ops`] — the viewer commands' bodies: export loading, undistortion,
//!   epipolar overlays, uncertainty, disparity; over the [`export_cache`].
//! - [`epipolar`], [`uncertainty`], [`disparity`] — the math behind them.
//! - [`laser`] — the laser-line extractor the laser topologies use.
//! - [`scope`] — the file scope: paths judged against root folders after
//!   canonicalising.

pub mod disparity;
pub mod epipolar;
pub mod export_cache;
pub mod laser;
pub mod ops;
pub mod project;
pub mod run;
pub mod scope;
pub mod uncertainty;

/// Command payload/response types re-exported for the `emit_schemas`
/// binary. Public only under `schema-export` so it never widens the app's
/// runtime surface. These are exactly the roots `emit_schemas` calls
/// `subschema_for::<…>()` on directly; `PointCloud` is *not* listed here
/// even though it's public in `disparity` — it only reaches the emitted
/// schema transitively, as a `$defs` entry nested inside `DisparityResult`.
#[cfg(feature = "schema-export")]
pub mod schema_types {
    pub use crate::disparity::DisparityResult;
    pub use crate::epipolar::EpipolarOverlay;
    pub use crate::project::{ProjectError, ProjectInfo};
    pub use crate::run::RunProgress;
    pub use vision_calibration::optim::ProjectionUncertainty;
}
