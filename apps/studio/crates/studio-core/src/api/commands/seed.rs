//! Seeds through the library's `device_seed`: the seed of a run or the
//! aggregate of several (`seed_preview`, writes nothing), that seed written
//! to a file (`export_seed`), and a project's `[seed]` set or cleared
//! through the project's save path (`set_project_seed`) after the
//! library's pre-flight check.

use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use vision_calibration::device_seed::{DeviceSeed, SeedSpread};
use vision_calibration::project::{DeviceHistory, Project, ProjectError, SeedSource};

use super::device::{DeviceKey, all_histories};
use super::project::{ProjectView, save_spec};
use crate::api::{Ctx, ErrorKind, StudioError};
use crate::runs::{RunRef, open_project, resolve, slash};
use crate::scope::UserPath;
use crate::seeds::seed_of_run;

/// What a seed is made from.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SeedFrom {
    /// One run's seed: its cameras whole, the mounts of a rig of several,
    /// its laser planes and hand-eye (the library's `DeviceSeed::from_run`).
    Run { run: RunRef },
    /// The aggregate of several runs' seeds, one run per unit (the
    /// library's `DeviceSeed::aggregate`).
    Runs { runs: Vec<RunRef> },
    /// The aggregate of each device's representative run (its latest
    /// starred run, else its latest pass run calibrated in a project).
    Units { devices: Vec<DeviceKey> },
}

/// The seed `from` makes, without writing anything: its values, each
/// value's spread across the runs of an aggregate, and what the aggregate
/// left out. A run's seed carries its cameras, the mounts of a rig of
/// several cameras, its laser planes and its hand-eye; never anything of
/// the capture (target poses, robot-pose corrections, residuals) nor the
/// uncertainty. `seed` when a run makes no seed (a run without a record, a
/// record without its image sizes) or the runs do not aggregate (other
/// cameras, image sizes, models or hand-eye modes); `invalid` (on
/// `devices[i]`) for a device with no representative run.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SeedPreviewArgs {
    pub from: SeedFrom,
}

/// A seed and where it came from.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SeedPreview {
    /// The seed: what a seeded run starts from.
    pub seed: DeviceSeed,
    /// The runs it was made from, in the order given (for `units`, each
    /// device's representative run).
    pub runs: Vec<SeedRun>,
    /// Each aggregated value's values across the runs, median, 1.4826·MAD,
    /// min and max (the library's labels: `cam0.fx`, `cam1.cam_se3_rig.tx`,
    /// `cam0.laser_plane.nx`, `handeye.rz`); empty for one run's seed.
    pub spread: Vec<SeedSpread>,
    /// What only some of the aggregated runs have (`<camera>.cam_se3_rig`,
    /// `<camera>.laser_plane`, `handeye`), left out of the aggregate rather
    /// than filled in; without mounts, a seed of several cameras leaves out
    /// the hand-eye too. Empty for one run's seed.
    pub omitted: Vec<String>,
}

/// A run a seed was made from.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SeedRun {
    /// The project folder.
    pub project: String,
    pub id: String,
}

pub fn seed_preview(ctx: &Ctx, args: SeedPreviewArgs) -> Result<SeedPreview, StudioError> {
    make_seed(ctx, &args.from)
}

/// The seed `from` makes.
fn make_seed(ctx: &Ctx, from: &SeedFrom) -> Result<SeedPreview, StudioError> {
    let runs: Vec<(SeedRun, PathBuf)> = match from {
        SeedFrom::Run { run } => {
            let resolved = resolve(ctx, run)?;
            let seed = seed_of_run(&ctx.scope, &resolved.info.dir)?;
            return Ok(SeedPreview {
                seed,
                runs: vec![SeedRun {
                    project: slash(resolved.project.dir()),
                    id: resolved.info.id,
                }],
                spread: Vec::new(),
                omitted: Vec::new(),
            });
        }
        SeedFrom::Runs { runs } => runs
            .iter()
            .map(|run| {
                let resolved = resolve(ctx, run)?;
                Ok((
                    SeedRun {
                        project: slash(resolved.project.dir()),
                        id: resolved.info.id,
                    },
                    resolved.info.dir,
                ))
            })
            .collect::<Result<_, StudioError>>()?,
        SeedFrom::Units { devices } => {
            let (histories, _) = all_histories(ctx);
            units(&histories, devices)?
        }
    };
    let seeds = runs
        .iter()
        .map(|(_, dir)| seed_of_run(&ctx.scope, dir))
        .collect::<Result<Vec<_>, _>>()?;
    let aggregate = DeviceSeed::aggregate(&seeds)
        .map_err(|e| StudioError::new(ErrorKind::Seed, e.to_string()))?;
    Ok(SeedPreview {
        seed: aggregate.seed,
        runs: runs.into_iter().map(|(run, _)| run).collect(),
        spread: aggregate.spread,
        omitted: aggregate.omitted,
    })
}

/// Each device's representative run, with its folder.
fn units(
    histories: &[DeviceHistory],
    devices: &[DeviceKey],
) -> Result<Vec<(SeedRun, PathBuf)>, StudioError> {
    devices
        .iter()
        .enumerate()
        .map(|(i, key)| {
            let entry = histories
                .iter()
                .find(|h| key.names(&h.device))
                .and_then(|h| h.representative_run())
                .ok_or_else(|| {
                    StudioError::invalid(format!(
                        "the device {} has no representative run: no starred run and no \
                             pass run calibrated in a project",
                        key.id
                    ))
                    .on_field(format!("devices[{i}]"))
                })?;
            Ok((
                SeedRun {
                    project: slash(&entry.project),
                    id: entry.run.id.clone(),
                },
                entry.run.dir.clone(),
            ))
        })
        .collect()
}

/// Write the seed `from` makes (as `seed_preview` answers it) to the file
/// `dest` inside a root: TOML by its `.toml` extension, JSON by `.json`,
/// the format a project's `[seed] file` reads. An existing file is
/// `file_exists` unless `overwrite`; the file's folder must exist.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExportSeedArgs {
    pub from: SeedFrom,
    pub dest: UserPath,
    #[serde(default)]
    pub overwrite: bool,
}

/// A seed written to a file.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ExportedSeed {
    /// The file written, absolute.
    pub path: String,
    /// The seed and where it came from, as `seed_preview` answers it.
    pub preview: SeedPreview,
}

pub fn export_seed(ctx: &Ctx, args: ExportSeedArgs) -> Result<ExportedSeed, StudioError> {
    let dest = ctx
        .scope
        .check_lexical(Path::new(args.dest.as_str()))
        .map_err(|e| e.on_field("dest"))?;
    let extension = dest
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase());
    let toml = match extension.as_deref() {
        Some("toml") => true,
        Some("json") => false,
        _ => {
            return Err(StudioError::invalid("a seed file is .json or .toml").on_field("dest"));
        }
    };
    if dest.exists() && !args.overwrite {
        return Err(
            StudioError::new(ErrorKind::FileExists, format!("{} exists", dest.display()))
                .at(slash(&dest))
                .on_field("dest"),
        );
    }
    let preview = make_seed(ctx, &args.from)?;
    let text = if toml {
        toml::to_string_pretty(&preview.seed)
            .map_err(|e| StudioError::internal(format!("the seed as TOML: {e}")))?
    } else {
        serde_json::to_string_pretty(&preview.seed)
            .map_err(|e| StudioError::internal(format!("the seed as JSON: {e}")))?
            + "\n"
    };
    std::fs::write(&dest, text).map_err(|e| {
        StudioError::new(ErrorKind::Io, format!("{}: {e}", dest.display())).at(slash(&dest))
    })?;
    Ok(ExportedSeed {
        path: slash(&dest),
        preview,
    })
}

/// Set the project's `[seed]` (a run's folder, a device specification or
/// a seed file, relative to the project folder unless absolute), or clear
/// it with `null`, and save the project as `save_project` does (`revision`
/// as `open_project` answered it, else `changed_on_disk`). The seed's path
/// must lie inside a root. The library's pre-flight check runs on the
/// edited project before anything is written: a seed that cannot be made
/// or does not fit the dataset's cameras and the configuration's camera
/// model is refused with the library's error (`seed`, on the field
/// `seed`). When the project fails that check for another reason (an open
/// manifest choice, a stale exclusion) but its dataset and configuration
/// resolve, the seed is still made, so one that cannot be is refused; its
/// fit is checked by the next run.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SetProjectSeedArgs {
    pub dir: UserPath,
    pub revision: String,
    #[serde(default)]
    pub seed: Option<SeedSource>,
}

pub fn set_project_seed(ctx: &Ctx, args: SetProjectSeedArgs) -> Result<ProjectView, StudioError> {
    let project = open_project(ctx, &args.dir)?;
    let mut spec = project.spec().clone();
    spec.seed = args.seed;
    save_spec(ctx, project, &args.revision, spec)
}

/// The library's pre-flight check of `project`'s seed, on the project as
/// edited (not yet saved): `Project::check`, refused when it fails on the
/// seed; when it fails earlier, `Project::input_digest`, which makes the
/// seed without fitting it.
pub(super) fn check_seed(project: &Project) -> Result<(), StudioError> {
    if project.spec().seed.is_none() {
        return Ok(());
    }
    let about_seed = |e: &ProjectError| {
        matches!(
            e,
            ProjectError::SeedUnresolved { .. }
                | ProjectError::Calibrate(
                    vision_calibration::dataset_runner::CalibrateDatasetError::Seed(_)
                )
        )
    };
    match project.check() {
        Ok(()) => Ok(()),
        Err(e) if about_seed(&e) => Err(e.into()),
        Err(_) => match project.input_digest() {
            Err(e) if about_seed(&e) => Err(e.into()),
            _ => Ok(()),
        },
    }
}
