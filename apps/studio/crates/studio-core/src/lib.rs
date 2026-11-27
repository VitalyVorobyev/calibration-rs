//! Calibration Studio's back end: what the desktop app and the server
//! compute, with no Tauri and no HTTP in it. The shells (`src-tauri`'s
//! Tauri commands and `studio-server`'s HTTP routes) dispatch through one
//! command table, [`api`].
//!
//! - [`api`] — the command table, its context ([`api::Ctx`]), the error
//!   envelope, jobs and the event stream, and the schema the TypeScript
//!   client is generated from. Projects and their runs are the library's
//!   (`vision_calibration::project`); the commands only judge their paths.
//! - [`runs`] — runs as commands name them (project folder and run id) and
//!   the cache of their parsed exports.
//! - [`epipolar`], [`uncertainty`], [`disparity`] — the math behind the
//!   viewers' commands; [`maps`] — a run's diagnostic maps as value planes
//!   and its per-feature residuals.
//! - [`laser`] — the laser-line extractor the laser topologies use.
//! - [`images`], [`blobs`], [`planes`] — what resources serve: source
//!   images (passed through, transcoded, thumbnails, in an on-disk cache),
//!   derived blobs (in memory, content-addressed) and the `VAM1` value
//!   plane format.
//! - [`scope`] — the file scope: paths judged against root folders after
//!   canonicalising, and every path a project and its manifest name.
//! - [`settings`] — the desktop's roots and the recent projects, kept
//!   between runs.

pub mod api;
pub mod blobs;
pub mod disparity;
pub mod epipolar;
pub mod images;
pub mod laser;
pub mod maps;
pub mod planes;
pub mod runs;
pub mod scope;
pub mod settings;
pub mod uncertainty;
