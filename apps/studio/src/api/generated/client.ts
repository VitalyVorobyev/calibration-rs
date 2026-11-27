/**
 * DO NOT EDIT — the typed client, generated from the back end's command table by
 * `bun run generate:api`. The source of truth is studio-core's
 * `api::table` and the `#[derive(schemars::JsonSchema)]` types; edit those
 * and regenerate.
 */
import type { Job, JobControl, JobHub } from "../jobs";
import { attachJob, startJob } from "../jobs";
import type { Transport } from "../transport";
import type * as T from "./types";

/** Every plain command: its argument object and its reply. */
export interface Commands {
  server_info: { args: T.ServerInfoArgs; reply: T.ServerInfo };
  add_root: { args: T.AddRootArgs; reply: T.ServerInfo };
  remove_root: { args: T.RemoveRootArgs; reply: T.ServerInfo };
  recent_projects: { args: T.RecentProjectsArgs; reply: T.RecentProjectsReply };
  forget_recent: { args: T.ForgetRecentArgs; reply: T.ForgetRecentReply };
  list_dir: { args: T.ListDirArgs; reply: T.DirListing };
  read_manifest: { args: T.ReadManifestArgs; reply: T.DatasetSpec };
  sniff_folder: { args: T.SniffFolderArgs; reply: T.SniffedFolder };
  config_form: { args: T.ConfigFormArgs; reply: T.FormSchema };
  dataset_form: { args: T.DatasetFormArgs; reply: T.FormSchema };
  resolve_config: { args: T.ResolveConfigArgs; reply: T.ResolveConfigReply };
  config_patch: { args: T.ConfigPatchArgs; reply: T.ConfigPatch };
  scan_library: { args: T.ScanLibraryArgs; reply: T.LibraryScan };
  open_project: { args: T.OpenProjectArgs; reply: T.ProjectView };
  create_project: { args: T.CreateProjectArgs; reply: T.ProjectView };
  save_project: { args: T.SaveProjectArgs; reply: T.ProjectView };
  project_views: { args: T.ProjectViewsArgs; reply: T.ProjectViews };
  list_runs: { args: T.ListRunsArgs; reply: T.ListRunsReply };
  set_run_notes: { args: T.SetRunNotesArgs; reply: T.RunInfo };
  delete_run: { args: T.DeleteRunArgs; reply: T.DeleteRunReply };
  import_export: { args: T.ImportExportArgs; reply: T.RunInfo };
  set_project_seed: { args: T.SetProjectSeedArgs; reply: T.ProjectView };
  device_history: { args: T.DeviceHistoryArgs; reply: T.DeviceHistories };
  device_series: { args: T.DeviceSeriesArgs; reply: T.DeviceSeriesSet };
  seed_preview: { args: T.SeedPreviewArgs; reply: T.SeedPreview };
  export_seed: { args: T.ExportSeedArgs; reply: T.ExportedSeed };
  run_info: { args: T.RunInfoArgs; reply: T.RunInfo };
  run_export: { args: T.RunExportArgs; reply: T.RunExport };
  run_quality: { args: T.RunQualityArgs; reply: T.QualityReport };
  undistort_points: { args: T.UndistortPointsArgs; reply: T.UndistortPointsReply };
  epipolar_overlay: { args: T.EpipolarOverlayArgs; reply: T.EpipolarOverlay };
  epipolar_overlay_undistorted: {
    args: T.EpipolarOverlayUndistortedArgs;
    reply: T.EpipolarOverlay;
  };
  camera_parameter_values: {
    args: T.CameraParameterValuesArgs;
    reply: T.CameraParameters;
  };
  projection_uncertainty: {
    args: T.ProjectionUncertaintyArgs;
    reply: T.ProjectionUncertainty;
  };
  distortion_map: { args: T.DistortionMapArgs; reply: T.DistortionMap };
  residual_map: { args: T.ResidualMapArgs; reply: T.ResidualMap };
  projection_sigma_map: { args: T.ProjectionSigmaMapArgs; reply: T.ProjectionSigmaMap };
  residual_records: { args: T.ResidualRecordsArgs; reply: T.ResidualRecords };
  residual_stats: { args: T.ResidualStatsArgs; reply: T.ResidualStats };
  detection_overview: { args: T.DetectionOverviewArgs; reply: T.DetectionOverview };
  detection_detail: { args: T.DetectionDetailArgs; reply: T.DetectionDetail };
  detect_preview: { args: T.DetectPreviewArgs; reply: T.DetectionDetail };
  list_jobs: { args: T.ListJobsArgs; reply: T.ListJobsReply };
  job_status: { args: T.JobStatusArgs; reply: T.JobInfo };
  cancel_job: { args: T.CancelJobArgs; reply: T.JobInfo };
}

/** Every job: its argument object, its progress event and its output. */
export interface Jobs {
  fetch_sample: {
    args: T.FetchSampleArgs;
    event: T.FetchSampleEvent;
    output: T.SampleProject;
  };
  calibrate: { args: T.CalibrateArgs; event: T.Progress; output: T.RunInfo };
  disparity: {
    args: T.DisparityArgs;
    event: T.DisparityEvent;
    output: T.DisparityResult;
  };
  detect_all: {
    args: T.DetectAllArgs;
    event: T.DetectProgress;
    output: T.DetectionOverview;
  };
}

/** Every resource: its query object. */
export interface Resources {
  image: T.ImageQuery;
  undistorted_image: T.UndistortedImageQuery;
  blob: T.BlobQuery;
  report: T.ReportQuery;
}

/** The back end's API over `t`; `hub` follows its jobs. */
export function createClient(t: Transport, hub: JobHub) {
  const control: JobControl = {
    status: (id) => t.invoke("job_status", { id }) as Promise<T.JobInfo>,
    cancel: (id) => t.invoke("cancel_job", { id }) as Promise<T.JobInfo>,
  };
  return {
    /**
     * What this back end is and what it serves. The frontend branches on
     * these values, never on which transport carries them.
     */
    serverInfo: (args: T.ServerInfoArgs = {}): Promise<T.ServerInfo> =>
      t.invoke("server_info", args) as Promise<T.ServerInfo>,
    /**
     * Add the folder `path` to the root folders (the desktop: the folder the
     * user picked in the native dialog, the one act that grants a folder).
     * `roots_fixed` on a server.
     */
    addRoot: (args: T.AddRootArgs): Promise<T.ServerInfo> =>
      t.invoke("add_root", args) as Promise<T.ServerInfo>,
    /**
     * Remove `path` from the root folders (the desktop). The samples' folder
     * is always a root.
     */
    removeRoot: (args: T.RemoveRootArgs): Promise<T.ServerInfo> =>
      t.invoke("remove_root", args) as Promise<T.ServerInfo>,
    /**
     * The projects opened most recently on this back end, newest first.
     */
    recentProjects: (args: T.RecentProjectsArgs = {}): Promise<T.RecentProjectsReply> =>
      t.invoke("recent_projects", args) as Promise<T.RecentProjectsReply>,
    /**
     * Drop `dir` from the recent projects (one that no longer opens).
     */
    forgetRecent: (args: T.ForgetRecentArgs): Promise<T.ForgetRecentReply> =>
      t.invoke("forget_recent", args) as Promise<T.ForgetRecentReply>,
    /**
     * The folder `path`, or the root folders without one. Entries whose name
     * starts with a dot are left out, and so is a symbolic link that leads out
     * of the roots; past 5 000 entries the listing is `truncated`.
     */
    listDir: (args: T.ListDirArgs): Promise<T.DirListing> =>
      t.invoke("list_dir", args) as Promise<T.DirListing>,
    /**
     * Read the dataset manifest at `path`: TOML for a `.toml` extension, JSON
     * otherwise.
     */
    readManifest: (args: T.ReadManifestArgs): Promise<T.DatasetSpec> =>
      t.invoke("read_manifest", args) as Promise<T.DatasetSpec>,
    /**
     * Infer a draft dataset manifest from the folder `folder` (the same
     * inference as `vcal project init`): images grouped into cameras, a
     * robot-pose file, a guessed topology. What cannot be determined is left
     * at a placeholder and listed in `unresolved`; a run refuses the manifest
     * until each is set and cleared from its `_unresolved`.
     */
    sniffFolder: (args: T.SniffFolderArgs): Promise<T.SniffedFolder> =>
      t.invoke("sniff_folder", args) as Promise<T.SniffedFolder>,
    /**
     * The config form of the workflow `topology`: the schema of its config
     * (`PlanarIntrinsicsConfig`, `RigHandeyeConfig`, …) and the library's
     * default configuration.
     */
    configForm: (args: T.ConfigFormArgs): Promise<T.FormSchema> =>
      t.invoke("config_form", args) as Promise<T.FormSchema>,
    /**
     * The dataset manifest form: the schema of `DatasetSpec` and the manifest
     * a new setup starts from (one chessboard camera, planar intrinsics).
     */
    datasetForm: (args: T.DatasetFormArgs = {}): Promise<T.FormSchema> =>
      t.invoke("dataset_form", args) as Promise<T.FormSchema>,
    /**
     * The configuration of the workflow `topology` that `patch` makes from its
     * defaults: tables merge key by key, anything else replaces (the library's
     * rule, as `vcal run --config` applies it), then checked. A field the
     * workflow does not have, or an invalid value, is a `config` error.
     */
    resolveConfig: (args: T.ResolveConfigArgs): Promise<T.ResolveConfigReply> =>
      t.invoke("resolve_config", args) as Promise<T.ResolveConfigReply>,
    /**
     * The configuration changes that make `config` (the workflow `topology`'s
     * config object, as a form edits it) from the library's defaults: the
     * fewest that do, as a project file stores them. The inverse of
     * `resolve_config`; an invalid config is a `config` error.
     */
    configPatch: (args: T.ConfigPatchArgs): Promise<T.ConfigPatch> =>
      t.invoke("config_patch", args) as Promise<T.ConfigPatch>,
    /**
     * Every project under the root folders, with its runs; a folder or
     * project that cannot be read is listed as a failure, never an error.
     */
    scanLibrary: (args: T.ScanLibraryArgs = {}): Promise<T.LibraryScan> =>
      t.invoke("scan_library", args) as Promise<T.LibraryScan>,
    /**
     * Open the project in `dir`: its definition, the resolved dataset and
     * configuration, the digests of its inputs and its runs. A version 1
     * project of the app's earlier format opens as it is (it is migrated when
     * saved). `not_a_project` for a folder without a project file,
     * `forbidden_path` when the project names a data folder or manifest
     * outside the roots. It becomes the most recent project. Opening again is
     * how a client reloads a project: the back end keeps none open.
     */
    openProject: (args: T.OpenProjectArgs): Promise<T.ProjectView> =>
      t.invoke("open_project", args) as Promise<T.ProjectView>,
    /**
     * Make the folder `dir` (created when missing, inside a root) a project of
     * `spec`, written as `calibration-project.toml`. Every path the spec names
     * must lie inside a root; `project_exists` when the folder already holds a
     * project. It becomes the most recent project.
     */
    createProject: (args: T.CreateProjectArgs): Promise<T.ProjectView> =>
      t.invoke("create_project", args) as Promise<T.ProjectView>,
    /**
     * Replace the project's definition with `spec` and write it, in its
     * format (a version 1 project becomes version 2). `revision` is the one
     * `open_project` answered: when the file changed on disk since, nothing is
     * written and the answer is `changed_on_disk` (open it again and apply the
     * edit). Every path the spec names must lie inside a root. A changed
     * `seed` is checked as `set_project_seed` checks it.
     */
    saveProject: (args: T.SaveProjectArgs): Promise<T.ProjectView> =>
      t.invoke("save_project", args) as Promise<T.ProjectView>,
    /**
     * The views the project's images pair into, without reading an image, and
     * what is wrong with the project as a run would find it before detection:
     * of the saved project, or of the draft `spec` when given (nothing is
     * written). Each problem is a message with the field it is about. With
     * robot poses or laser images in the manifest, each view also names the
     * pose and the laser frames it pairs with, and what pairs with no view is
     * listed.
     *
     * With a `spec`, `dir` may also be a folder that is not a project yet (a
     * new project's data folder): the draft is then judged as a project in
     * that folder, its inline manifest's `root` read against it.
     */
    projectViews: (args: T.ProjectViewsArgs): Promise<T.ProjectViews> =>
      t.invoke("project_views", args) as Promise<T.ProjectViews>,
    /**
     * The project's runs (as `open_project` lists them).
     */
    listRuns: (args: T.ListRunsArgs): Promise<T.ListRunsReply> =>
      t.invoke("list_runs", args) as Promise<T.ListRunsReply>,
    /**
     * Replace the notes (label, notes, star) of run `id`; empty notes remove
     * them. Answers the run as listed after.
     */
    setRunNotes: (args: T.SetRunNotesArgs): Promise<T.RunInfo> =>
      t.invoke("set_run_notes", args) as Promise<T.RunInfo>,
    /**
     * Delete run `id` with every file in it. Answers the runs left.
     */
    deleteRun: (args: T.DeleteRunArgs): Promise<T.DeleteRunReply> =>
      t.invoke("delete_run", args) as Promise<T.DeleteRunReply>,
    /**
     * Add an export calibrated elsewhere to the project as its next run: an
     * export file, a `vcal run -o` folder (its `calibration.json`) or a run
     * folder (its `export.json`). `import_mismatch` for another workflow or
     * camera count than the project's; the export, and every image its image
     * manifest names, must lie inside a root.
     */
    importExport: (args: T.ImportExportArgs): Promise<T.RunInfo> =>
      t.invoke("import_export", args) as Promise<T.RunInfo>,
    /**
     * Set the project's `[seed]` (a run's folder, a device specification or
     * a seed file, relative to the project folder unless absolute), or clear
     * it with `null`, and save the project as `save_project` does (`revision`
     * as `open_project` answered it, else `changed_on_disk`). The seed's path
     * must lie inside a root. The library's pre-flight check runs on the
     * edited project before anything is written: a seed that cannot be made
     * or does not fit the dataset's cameras and the configuration's camera
     * model is refused with the library's error (`seed`, on the field
     * `seed`). When the project fails that check for another reason (an open
     * manifest choice, a stale exclusion) but its dataset and configuration
     * resolve, the seed is still made, so one that cannot be is refused; its
     * fit is checked by the next run.
     */
    setProjectSeed: (args: T.SetProjectSeedArgs): Promise<T.ProjectView> =>
      t.invoke("set_project_seed", args) as Promise<T.ProjectView>,
    /**
     * The calibration history of a device (or of every device of a product):
     * every run of every project under the roots that names it, in the
     * library's order (runs without a timestamp first, then by timestamp, then
     * by project folder and listing order), with the run that stands for each
     * device (its latest starred run, else its latest pass run calibrated in a
     * project). A device no project names has no history (`devices` is
     * empty); a folder or project the scan cannot read is a failure, not an
     * error.
     */
    deviceHistory: (args: T.DeviceHistoryArgs): Promise<T.DeviceHistories> =>
      t.invoke("device_history", args) as Promise<T.DeviceHistories>,
    /**
     * The values of a device's parameters (or of each device of a product)
     * over its history, for a trend chart: every camera parameter, each
     * camera's mount in a rig, the hand-eye and each laser plane, as each run
     * estimated them, with their σ. A parameter only some runs have is a gap
     * in the others (`null`), never dropped. `starred_only` keeps the starred
     * runs. Each series of a camera parameter names its spread across the
     * runs, the library's robust summary (median, 1.4826·MAD, min, max) of
     * the runs' seeds.
     */
    deviceSeries: (args: T.DeviceSeriesArgs): Promise<T.DeviceSeriesSet> =>
      t.invoke("device_series", args) as Promise<T.DeviceSeriesSet>,
    /**
     * The seed `from` makes, without writing anything: its values, each
     * value's spread across the runs of an aggregate, and what the aggregate
     * left out. A run's seed carries its cameras, the mounts of a rig of
     * several cameras, its laser planes and its hand-eye; never anything of
     * the capture (target poses, robot-pose corrections, residuals) nor the
     * uncertainty. `seed` when a run makes no seed (a run without a record, a
     * record without its image sizes) or the runs do not aggregate (other
     * cameras, image sizes, models or hand-eye modes); `invalid` (on
     * `devices[i]`) for a device with no representative run.
     */
    seedPreview: (args: T.SeedPreviewArgs): Promise<T.SeedPreview> =>
      t.invoke("seed_preview", args) as Promise<T.SeedPreview>,
    /**
     * Write the seed `from` makes (as `seed_preview` answers it) to the file
     * `dest` inside a root: TOML by its `.toml` extension, JSON by `.json`,
     * the format a project's `[seed] file` reads. An existing file is
     * `file_exists` unless `overwrite`; the file's folder must exist.
     */
    exportSeed: (args: T.ExportSeedArgs): Promise<T.ExportedSeed> =>
      t.invoke("export_seed", args) as Promise<T.ExportedSeed>,
    /**
     * A run's listing entry: its state (with the record of a run this library
     * wrote) and its notes.
     */
    runInfo: (args: T.RunInfoArgs): Promise<T.RunInfo> =>
      t.invoke("run_info", args) as Promise<T.RunInfo>,
    /**
     * A run's export, with the folder its image manifest's frames resolve
     * against.
     */
    runExport: (args: T.RunExportArgs): Promise<T.RunExport> =>
      t.invoke("run_export", args) as Promise<T.RunExport>,
    /**
     * A run's quality report; `not found` (`io`) for an imported export whose
     * quality could not be assessed.
     */
    runQuality: (args: T.RunQualityArgs): Promise<T.QualityReport> =>
      t.invoke("run_quality", args) as Promise<T.QualityReport>,
    /**
     * Map raw (distorted) pixels of `camera` of the run into its undistorted
     * pixel frame; `null` where back-projection has no finite ray.
     */
    undistortPoints: (args: T.UndistortPointsArgs): Promise<T.UndistortPointsReply> =>
      t.invoke("undistort_points", args) as Promise<T.UndistortPointsReply>,
    /**
     * The epipolar polyline and epipole in camera `cam_b` (distorted pixels)
     * of the pixel `point_px` of camera `cam_a`, through each camera's full
     * model of the run.
     */
    epipolarOverlay: (args: T.EpipolarOverlayArgs): Promise<T.EpipolarOverlay> =>
      t.invoke("epipolar_overlay", args) as Promise<T.EpipolarOverlay>,
    /**
     * A straight epipolar segment in camera `cam_b`'s undistorted frame,
     * clipped to its `image_width_b` × `image_height_b` image, of the pixel
     * `point_px` of camera `cam_a`'s undistorted frame.
     */
    epipolarOverlayUndistorted: (
      args: T.EpipolarOverlayUndistortedArgs,
    ): Promise<T.EpipolarOverlay> =>
      t.invoke("epipolar_overlay_undistorted", args) as Promise<T.EpipolarOverlay>,
    /**
     * Every parameter of every camera of the run, whatever its model: its
     * label in the export's uncertainty, name, group, value and unit, its σ
     * or why it has none (no covariance and the library's reason, held fixed,
     * not listed), and whether the solve ended with it on a bound. Also the
     * poses the run estimates (each camera's pose in a rig of several, the
     * hand-eye: translation in metres, axis-angle rotation in degrees, the σ
     * of the perturbation the covariance is over) and its laser planes (normal
     * and distance with their σ).
     */
    cameraParameterValues: (
      args: T.CameraParameterValuesArgs,
    ): Promise<T.CameraParameters> =>
      t.invoke("camera_parameter_values", args) as Promise<T.CameraParameters>,
    /**
     * The projection uncertainty of `camera` of the run over an image of
     * `image_width` × `image_height` pixels.
     */
    projectionUncertainty: (
      args: T.ProjectionUncertaintyArgs,
    ): Promise<T.ProjectionUncertainty> =>
      t.invoke("projection_uncertainty", args) as Promise<T.ProjectionUncertainty>,
    /**
     * The lens and sensor displacement of camera `camera` of the run over its
     * image: three layers, each a plane of `dx`, `dy` and `magnitude` in
     * pixels, `actual − reference` at the same ray (where the ray lands minus
     * where it would land without the stage), indexed by the camera's own
     * pixel. `distortion` is the lens alone (the camera without its sensor
     * tilt minus the camera without distortion), `sensor` what the sensor
     * model adds (zero for an untilted sensor), `total` their sum. `NaN` where
     * no ray reaches a node. `columns` (default 64, at most 1024) sets the
     * grid every map of the camera shares.
     */
    distortionMap: (args: T.DistortionMapArgs): Promise<T.DistortionMap> =>
      t.invoke("distortion_map", args) as Promise<T.DistortionMap>,
    /**
     * The systematic part of camera `camera`'s reprojection residuals over its
     * image, the library's local-linear residual map: one plane of eight
     * channels, `mean_dx`, `mean_dy`, `mean_magnitude` (the smoothed mean of
     * `observed − projected`, px), `rms` (px), `std_error` (px, view-clustered),
     * `significance` (χ: above 3.03 is p < 0.01), `effective_samples` and
     * `effective_views`. Records sit at their projection. `NaN` where the
     * support is too thin (`effective_samples` below
     * `options.min_effective_samples`) and, for `std_error` and
     * `significance`, with too few effective views. `options.views` maps a
     * subset of the views (their indices in the export); one view, or too few
     * effective views, gives a mean that includes the views' pose error and no
     * significance. `columns` as `distortion_map`'s. `invalid` for a run
     * without per-feature residuals of the camera.
     */
    residualMap: (args: T.ResidualMapArgs): Promise<T.ResidualMap> =>
      t.invoke("residual_map", args) as Promise<T.ResidualMap>,
    /**
     * The projection uncertainty of camera `camera` on the maps' grid: one
     * plane, `sigma`, the major-axis σ in pixels of where the camera-frame ray
     * through each node lands, from the covariance of the camera's
     * parameters; `NaN` where no ray reaches a node. `unavailable`, with the
     * reason, for a run without a covariance or whose camera parameters were
     * all held fixed. `columns` as `distortion_map`'s.
     */
    projectionSigmaMap: (args: T.ProjectionSigmaMapArgs): Promise<T.ProjectionSigmaMap> =>
      t.invoke("projection_sigma_map", args) as Promise<T.ProjectionSigmaMap>,
    /**
     * Camera `camera`'s per-feature residual records (of one `view`, when
     * given), one column per field, in the export's order: each record's
     * index in the export, view, feature, target point, observed and
     * projected pixel and `observed − projected`. The target point is the one
     * the calibration used: an exclusion names a feature by its detection's
     * `world_xyz` (`detection_detail`), which a rig of plain chessboards may
     * have relabelled.
     */
    residualRecords: (args: T.ResidualRecordsArgs): Promise<T.ResidualRecords> =>
      t.invoke("residual_records", args) as Promise<T.ResidualRecords>,
    /**
     * The run's reprojection statistics from its per-feature residuals (the
     * library's mean, RMS, largest error and count of `|observed −
     * projected|`, px), per camera, per view over every camera, and per
     * camera and view, with the records whose projection diverged.
     */
    residualStats: (args: T.ResidualStatsArgs): Promise<T.ResidualStats> =>
      t.invoke("residual_stats", args) as Promise<T.ResidualStats>,
    /**
     * What the project's detection cache holds of every image, view by view
     * and camera by camera (the views of `project_views`): the features found
     * and the detector's outcome, or that the image has no cached detection
     * with the project's detector settings. No image is decoded or detected;
     * each is read to find its cache entry. `left_out` gives why run `run`
     * (default: the latest calibrated run) left an image out.
     */
    detectionOverview: (args: T.DetectionOverviewArgs): Promise<T.DetectionOverview> =>
      t.invoke("detection_overview", args) as Promise<T.DetectionOverview>,
    /**
     * Camera `camera`'s detection of `image` (relative to the data folder, as
     * `project_views` lists it) with what an overlay needs: the features (in
     * the camera's region of interest when it has one), the target's per-kind
     * detail aligned with them, the candidates the detector left out and the
     * image's outcome. Read from the project's detection cache, or detected
     * and stored there (features and detail) when it holds none: the
     * detection a calibration of the project uses. A feature is excluded by
     * its `world_xyz`. `invalid` for an image that is not one of the
     * camera's.
     */
    detectionDetail: (args: T.DetectionDetailArgs): Promise<T.DetectionDetail> =>
      t.invoke("detection_detail", args) as Promise<T.DetectionDetail>,
    /**
     * `detection_detail` of one image with `detector` in place of the
     * project's detector settings (the manifest's `detector` section; `{}` is
     * the detectors' defaults), to preview an edit: detected now, and nothing
     * is written (neither the project nor its detection cache).
     */
    detectPreview: (args: T.DetectPreviewArgs): Promise<T.DetectionDetail> =>
      t.invoke("detect_preview", args) as Promise<T.DetectionDetail>,
    /**
     * Every job the back end still knows of, live or recently finished, in
     * start order.
     */
    listJobs: (args: T.ListJobsArgs = {}): Promise<T.ListJobsReply> =>
      t.invoke("list_jobs", args) as Promise<T.ListJobsReply>,
    /**
     * The job `id`: its state, its last progress report, its output once
     * done.
     */
    jobStatus: (args: T.JobStatusArgs): Promise<T.JobInfo> =>
      t.invoke("job_status", args) as Promise<T.JobInfo>,
    /**
     * Cancel the job `id`: a queued job at once, a running one at its next
     * image or solver iteration. A finished job is answered as it is.
     */
    cancelJob: (args: T.CancelJobArgs): Promise<T.JobInfo> =>
      t.invoke("cancel_job", args) as Promise<T.JobInfo>,
    jobs: {
      /**
       * Make the sample dataset `id` (one of `server_info.samples`) available
       * as a project and answer its folder: the files are downloaded into the
       * back end's samples folder (always one of its roots), checked against
       * their SHA-256 and reused later, and the folder becomes a project
       * referencing the sample's manifest (once; an existing project is kept
       * with its runs). A back end built without samples answers `invalid`.
       */
      fetchSample: (
        args: T.FetchSampleArgs,
      ): Promise<Job<T.FetchSampleEvent, T.SampleProject>> =>
        startJob<T.FetchSampleEvent, T.SampleProject>(
          t,
          hub,
          control,
          "fetch_sample",
          args,
        ),
      /**
       * Calibrate the project as it is on disk and write the run as its next
       * `runs/run-NNN/` (export, quality report, record, the HTML report, and
       * `notes` when given). `config_patch` changes the configuration for this
       * run only, over the project's, and is recorded with it; `redetect`
       * detects every image again instead of reading the detection cache. Every
       * file the project names must lie inside a root. Its events are the
       * library's progress reports; a cancelled run writes nothing and ends the
       * job `cancelled`.
       */
      calibrate: (args: T.CalibrateArgs): Promise<Job<T.Progress, T.RunInfo>> =>
        startJob<T.Progress, T.RunInfo>(t, hub, control, "calibrate", args),
      /**
       * A disparity map of the stereo pair `path_a` (camera `cam_a`) and
       * `path_b` (camera `cam_b`) at `pose`: rectified from the run's export and
       * dense-matched. The output names its fields as blobs: a value plane of
       * disparity, depth and the rectified left image, the point cloud, and the
       * rectified pair. The same request again answers from the blob store while
       * it still holds them.
       */
      disparity: (
        args: T.DisparityArgs,
      ): Promise<Job<T.DisparityEvent, T.DisparityResult>> =>
        startJob<T.DisparityEvent, T.DisparityResult>(t, hub, control, "disparity", args),
      /**
       * Detect every image of the project into its detection cache (images it
       * already holds are read from it, unless `redetect`), several at a time;
       * its events count the images done, its output is the
       * `detection_overview` of the result. An image that cannot be read or
       * detected is reported in the overview, not as the job's failure.
       * Cancelling stops between images; what was detected stays cached.
       */
      detectAll: (
        args: T.DetectAllArgs,
      ): Promise<Job<T.DetectProgress, T.DetectionOverview>> =>
        startJob<T.DetectProgress, T.DetectionOverview>(
          t,
          hub,
          control,
          "detect_all",
          args,
        ),
      /** Follow the job `id` (started earlier, say before a reload). */
      attach: <E, R>(id: string): Promise<Job<E, R>> => attachJob<E, R>(hub, control, id),
    },
    /** The URL of each resource: for an `<img>`, a `fetch` or `fetchPlane`. */
    urls: {
      /**
       * The image file at `path`, as a browser can decode it: JPEG, PNG, WebP,
       * GIF and BMP are the file itself; TIFF, PNM and the other formats the
       * back end reads become an 8-bit PNG (a 16-bit image reduced as the
       * detectors reduce it). With `max`, a JPEG thumbnail whose longest side is
       * at most `max`, snapped up to 160, 320, 640 or 1280.
       */
      image: (query: T.ImageQuery): string => t.resourceUrl("image", query),
      /**
       * The image at `path` rendered into `camera`'s undistorted pixel frame of
       * the run's export (a PNG), cropped to `roi` when given.
       */
      undistortedImage: (query: T.UndistortedImageQuery): string =>
        t.resourceUrl("undistorted_image", query),
      /**
       * The blob `id` a command made (a value plane, raw floats, a PNG).
       * `blob_expired` once the store has evicted it: run the command again.
       */
      blob: (query: T.BlobQuery): string => t.resourceUrl("blob", query),
      /**
       * The run's page, `report.html` (written with every calibration this app
       * runs), served sandboxed: its scripts run without access to the app.
       * `io` when the run has none.
       */
      report: (query: T.ReportQuery): string => t.resourceUrl("report", query),
    },
  };
}

export type Client = ReturnType<typeof createClient>;
