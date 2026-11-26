//! Calibration Studio's back end: what the desktop app and the browser
//! viewer compute, with no Tauri and no HTTP in it. The shells
//! (`src-tauri`'s Tauri commands and its `calib-viewer` server) dispatch
//! through one command table, [`api`].
//!
//! - [`api`] — the command table, its context ([`api::Ctx`]), the error
//!   envelope, jobs and the event stream, and the schema the TypeScript
//!   client is generated from.
//! - [`project`] — project folders: create, open, save the manifest, save
//!   a run (`runs/<name>/`). See ADR 0014.
//! - [`run`] — the calibration runner, with stage progress and
//!   cancellation.
//! - [`ops`] — the viewer commands' bodies: export loading, undistortion,
//!   epipolar overlays, uncertainty, disparity; over the [`export_cache`].
//! - [`epipolar`], [`uncertainty`], [`disparity`] — the math behind them.
//! - [`laser`] — the laser-line extractor the laser topologies use.
//! - [`images`], [`blobs`], [`planes`] — what resources serve: source
//!   images (passed through, transcoded, thumbnails, in an on-disk cache),
//!   derived blobs (in memory, content-addressed) and the `VAM1` value
//!   plane format.
//! - [`scope`] — the file scope: paths judged against root folders after
//!   canonicalising.
//! - [`run_source`] — the run the browser viewer serves.

pub mod api;
pub mod blobs;
pub mod disparity;
pub mod epipolar;
pub mod export_cache;
pub mod images;
pub mod laser;
pub mod ops;
pub mod planes;
pub mod project;
pub mod run;
pub mod run_source;
pub mod scope;
pub mod uncertainty;
