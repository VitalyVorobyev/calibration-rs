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
  create_project: { args: T.CreateProjectArgs; reply: T.ProjectInfo };
  open_project: { args: T.OpenProjectArgs; reply: T.ProjectInfo };
  save_project_manifest: {
    args: T.SaveProjectManifestArgs;
    reply: T.SaveProjectManifestReply;
  };
  save_project_run: { args: T.SaveProjectRunArgs; reply: T.RunSummary };
  read_manifest: { args: T.ReadManifestArgs; reply: T.DatasetSpec };
  sniff_folder: { args: T.SniffFolderArgs; reply: T.DatasetSpec };
  default_config: { args: T.DefaultConfigArgs; reply: T.DefaultConfigReply };
  load_export: { args: T.LoadExportArgs; reply: T.LoadExportResult };
  set_active_export: { args: T.SetActiveExportArgs; reply: T.SetActiveExportReply };
  undistort_points: { args: T.UndistortPointsArgs; reply: T.UndistortPointsReply };
  epipolar_overlay: { args: T.EpipolarOverlayArgs; reply: T.EpipolarOverlay };
  epipolar_overlay_undistorted: {
    args: T.EpipolarOverlayUndistortedArgs;
    reply: T.EpipolarOverlay;
  };
  camera_parameter_values: {
    args: T.CameraParameterValuesArgs;
    reply: T.CameraParameterValuesReply;
  };
  projection_uncertainty: {
    args: T.ProjectionUncertaintyArgs;
    reply: T.ProjectionUncertainty;
  };
  disparity: { args: T.DisparityArgs; reply: T.DisparityResult };
  list_jobs: { args: T.ListJobsArgs; reply: T.ListJobsReply };
  job_status: { args: T.JobStatusArgs; reply: T.JobInfo };
  cancel_job: { args: T.CancelJobArgs; reply: T.JobInfo };
}

/** Every job: its argument object, its progress event and its output. */
export interface Jobs {
  fetch_sample: {
    args: T.FetchSampleArgs;
    event: T.FetchSampleEvent;
    output: T.FetchSampleOutput;
  };
  run_calibration: {
    args: T.RunCalibrationArgs;
    event: T.RunProgress;
    output: T.RunResponse;
  };
}

/** Every resource: its query object. */
export interface Resources {
  undistorted_image: T.UndistortedImageQuery;
}

/** The back end's API over `t`; `hub` follows its jobs. */
export function createClient(t: Transport, hub: JobHub) {
  const control: JobControl = {
    status: (id) => t.invoke("job_status", { id }) as Promise<T.JobInfo>,
    cancel: (id) => t.invoke("cancel_job", { id }) as Promise<T.JobInfo>,
  };
  return {
    /**
     * Make the folder `dir` a project for the dataset in `dataset_dir`,
     * described by `manifest`. Fails with `project_exists` when `dir` already
     * holds one.
     */
    createProject: (args: T.CreateProjectArgs): Promise<T.ProjectInfo> =>
      t.invoke("create_project", args) as Promise<T.ProjectInfo>,
    /**
     * Open the project in `dir`; `not_a_project` for any other folder.
     */
    openProject: (args: T.OpenProjectArgs): Promise<T.ProjectInfo> =>
      t.invoke("open_project", args) as Promise<T.ProjectInfo>,
    /**
     * Replace the project's dataset manifest (`dataset.json`).
     */
    saveProjectManifest: (
      args: T.SaveProjectManifestArgs,
    ): Promise<T.SaveProjectManifestReply> =>
      t.invoke("save_project_manifest", args) as Promise<T.SaveProjectManifestReply>,
    /**
     * Save a run as `runs/<name>/`; `run_exists` rather than overwrite one.
     */
    saveProjectRun: (args: T.SaveProjectRunArgs): Promise<T.RunSummary> =>
      t.invoke("save_project_run", args) as Promise<T.RunSummary>,
    /**
     * Read the dataset manifest at `path`: TOML for a `.toml` extension, JSON
     * otherwise. Its folder becomes one the app may read.
     */
    readManifest: (args: T.ReadManifestArgs): Promise<T.DatasetSpec> =>
      t.invoke("read_manifest", args) as Promise<T.DatasetSpec>,
    /**
     * Infer a dataset manifest from the folder `folder`: images grouped into
     * cameras, a robot-pose file, a guessed topology. What cannot be
     * determined is left at a placeholder and listed in the manifest's
     * `_unresolved` (the same inference as `vcal init`).
     */
    sniffFolder: (args: T.SniffFolderArgs): Promise<T.DatasetSpec> =>
      t.invoke("sniff_folder", args) as Promise<T.DatasetSpec>,
    /**
     * The default `*Config` of a manifest topology (`planar_intrinsics`,
     * `rig_handeye`, …): the library's `Config::default()`.
     */
    defaultConfig: (args: T.DefaultConfigArgs): Promise<T.DefaultConfigReply> =>
      t.invoke("default_config", args) as Promise<T.DefaultConfigReply>,
    /**
     * Read and parse the calibration export at `path` and check its schema
     * version. It does not become active: the frontend validates it first and
     * then calls `set_active_export`. Its folder and image root become
     * folders the app may read.
     */
    loadExport: (args: T.LoadExportArgs): Promise<T.LoadExportResult> =>
      t.invoke("load_export", args) as Promise<T.LoadExportResult>,
    /**
     * Make `export` the active export, the one every viewer command reads.
     */
    setActiveExport: (args: T.SetActiveExportArgs): Promise<T.SetActiveExportReply> =>
      t.invoke("set_active_export", args) as Promise<T.SetActiveExportReply>,
    /**
     * Map raw (distorted) pixels of `camera` into its undistorted pixel frame;
     * `null` where back-projection has no finite ray.
     */
    undistortPoints: (args: T.UndistortPointsArgs): Promise<T.UndistortPointsReply> =>
      t.invoke("undistort_points", args) as Promise<T.UndistortPointsReply>,
    /**
     * The epipolar polyline and epipole in camera `cam_b` (distorted pixels)
     * of the pixel `point_px` of camera `cam_a`, through each camera's full
     * model.
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
     * The value of each labelled parameter of the active export that is a
     * camera parameter, aligned with `labels`; `null` for poses and planes.
     */
    cameraParameterValues: (
      args: T.CameraParameterValuesArgs,
    ): Promise<T.CameraParameterValuesReply> =>
      t.invoke("camera_parameter_values", args) as Promise<T.CameraParameterValuesReply>,
    /**
     * The projection uncertainty of `camera` of the active export over an
     * image of `image_width` × `image_height` pixels.
     */
    projectionUncertainty: (
      args: T.ProjectionUncertaintyArgs,
    ): Promise<T.ProjectionUncertainty> =>
      t.invoke("projection_uncertainty", args) as Promise<T.ProjectionUncertainty>,
    /**
     * A disparity map of the stereo pair `path_a` (camera `cam_a`) and
     * `path_b` (camera `cam_b`) at `pose`: rectified from the active export,
     * dense-matched, with colour-mapped images and quality metrics.
     */
    disparity: (args: T.DisparityArgs): Promise<T.DisparityResult> =>
      t.invoke("disparity", args) as Promise<T.DisparityResult>,
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
       * Make the sample dataset `id` (one of the library's sample catalogue)
       * available locally and answer the absolute path of its manifest. The
       * files are downloaded into the platform cache (`VISION_CALIBRATION_DATA`
       * overrides it), checked against their SHA-256 and reused later, so the
       * app carries no dataset.
       */
      fetchSample: (
        args: T.FetchSampleArgs,
      ): Promise<Job<T.FetchSampleEvent, T.FetchSampleOutput>> =>
        startJob<T.FetchSampleEvent, T.FetchSampleOutput>(
          t,
          hub,
          control,
          "fetch_sample",
          args,
        ),
      /**
       * Run a calibration end to end: detection (cached), the solve and the
       * export, reporting each stage and the library's reports within it. The
       * folder `manifest_dir` is what the manifest's relative paths resolve
       * against; it becomes one the app may read. A successful run's export
       * becomes the active export.
       */
      runCalibration: (
        args: T.RunCalibrationArgs,
      ): Promise<Job<T.RunProgress, T.RunResponse>> =>
        startJob<T.RunProgress, T.RunResponse>(t, hub, control, "run_calibration", args),
      /** Follow the job `id` (started earlier, say before a reload). */
      attach: <E, R>(id: string): Promise<Job<E, R>> => attachJob<E, R>(hub, control, id),
    },
    resources: {
      /**
       * The image at `path` rendered into `camera`'s undistorted pixel frame (a
       * PNG), cropped to `roi` when given.
       */
      undistortedImage: (query: T.UndistortedImageQuery): Promise<ArrayBuffer> =>
        t.fetchResource("undistorted_image", query),
    },
  };
}

export type Client = ReturnType<typeof createClient>;
