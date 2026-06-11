export type Vec2 = [number, number];
export type Vec3 = [number, number, number];
export type Vec4 = [number, number, number, number];
export type Matrix4Rows = [Vec4, Vec4, Vec4, Vec4];

export interface TransformRecord {
  translation_m: Vec3;
  quaternion_xyzw: Vec4;
  matrix4_row_major: Matrix4Rows;
}

export interface PlaneRecord {
  normal: Vec3;
  distance_m: number;
}

export interface DatasetInfo {
  name: string;
  source_dir: string;
  num_cameras: number;
  board_rows: number;
  board_cols: number;
  cell_size_mm: number;
  full_image_size: [number, number];
  tile_size: [number, number];
}

export interface PoseRecord {
  index: number;
  snap_type: string;
  target_image: string;
  laser_image?: string | null;
  base_se3_gripper: TransformRecord;
}

export interface CameraRecord {
  index: number;
  intrinsics: [number, number, number, number, number];
  distortion: [number, number, number, number, number];
  scheimpflug: [number, number];
  cam_se3_rig: TransformRecord;
  cam_to_rig: TransformRecord;
  laser_plane_camera: PlaneRecord;
  laser_plane_rig: PlaneRecord;
}

export interface PerCameraStats {
  mean_reproj_error_px: number;
  reproj_count: number;
  max_reproj_error_px: number;
  reproj_histogram_px: [number, number, number, number, number];
  mean_laser_err_m: number;
  max_laser_err_m: number;
  laser_histogram_m: [number, number, number, number, number];
  mean_laser_err_px: number;
  max_laser_err_px: number;
  laser_count: number;
}

export interface TargetFeatureRecord {
  pose: number;
  camera: number;
  feature: number;
  target_xyz_m: Vec3;
  observed_px: Vec2;
  projected_px?: Vec2 | null;
  error_px?: number | null;
}

export interface LaserFeatureRecord {
  pose: number;
  camera: number;
  feature: number;
  observed_px: Vec2;
  residual_m?: number | null;
  residual_px?: number | null;
  projected_line_px?: [Vec2, Vec2] | null;
}

export interface StageGeometry {
  handeye: TransformRecord;
  target_ref: TransformRecord;
  cameras: CameraRecord[];
}

export interface ViewerStage {
  id: string;
  label: string;
  mean_reproj_error_px: number;
  final_cost?: number | null;
  robot_deltas: [number, number, number, number, number, number][];
  per_camera_stats: PerCameraStats[];
  geometry: StageGeometry;
  target_features: TargetFeatureRecord[];
  laser_features: LaserFeatureRecord[];
}

export interface ViewerManifest {
  schema_version: number;
  generator: string;
  dataset: DatasetInfo;
  frame_conventions: {
    cam_se3_rig: string;
    cam_to_rig: string;
    eye_to_hand_chain: string;
    robot_delta: string;
  };
  poses: PoseRecord[];
  cameras: CameraRecord[];
  handeye_mode: string;
  handeye: TransformRecord;
  target_ref: TransformRecord;
  stages: ViewerStage[];
}

export interface ReprojectionStats {
  mean: number;
  rms: number;
  max: number;
  count: number;
}

export interface LevelStats {
  mean: number;
  median: number;
  rms: number;
  p95: number;
  max: number;
  count: number;
}

export type ReprojLevel = 'intrinsic' | 'rig_extrinsic' | 'hand_eye' | 'laser';

export interface BenchIdent {
  dataset_id: string;
  problem: string;
  tier: string;
  git_sha: string;
  timestamp_rfc3339: string;
  config_hash: number;
  bench_schema_version: number;
  features: string[];
}

export interface BenchConvergence {
  init_ok: boolean;
  converged: boolean;
  report: {
    final_cost: number;
    num_iters: number;
  };
}

export interface BenchFit {
  overall: ReprojectionStats;
  per_camera: ReprojectionStats[];
  per_camera_hist: number[][];
  reported_mean_reproj_px: number;
  reported_per_cam_px: number[];
}

export interface DetectionStat {
  camera_id: string;
  images_total: number;
  images_used: number;
  features_detected: number;
  features_expected: number;
  coverage_pct: number;
  detect_ms: number;
}

export interface DetectionSummary {
  per_camera: DetectionStat[];
  total_detected: number;
  total_expected: number;
}

export interface LaserCamStat {
  camera_id: string;
  images_total: number;
  images_used: number;
  points_extracted: number;
  extract_ms: number;
  plane_residual_m?: ReprojectionStats | null;
  line_residual_px?: ReprojectionStats | null;
  inlier_ratio?: number | null;
}

export interface LaserMetrics {
  per_camera: LaserCamStat[];
  total_points: number;
  total_images_used: number;
  extract_ms: number;
}

export interface RobotCorrectionSummary {
  count: number;
  mean_rot_deg: number;
  max_rot_deg: number;
  mean_trans_mm: number;
  max_trans_mm: number;
  prior_rot_deg?: number | null;
  prior_trans_mm?: number | null;
  max_rot_prior_ratio?: number | null;
  max_trans_prior_ratio?: number | null;
  exceeds_prior?: boolean;
}

export interface IntrinsicsArtifact {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  skew: number;
}

export interface DistortionArtifact {
  k1: number;
  k2: number;
  k3: number;
  p1: number;
  p2: number;
}

export interface ScheimpflugArtifact {
  tilt_x_rad: number;
  tilt_y_rad: number;
}

export interface CameraArtifact {
  camera_id: string;
  camera_matrix_px: [[number, number, number], [number, number, number], [number, number, number]];
  intrinsics_px: IntrinsicsArtifact;
  distortion_model: string;
  distortion: DistortionArtifact;
  scheimpflug?: ScheimpflugArtifact | null;
}

export interface TransformArtifact {
  name: string;
  to_frame: string;
  from_frame: string;
  translation_mm: Vec3;
  rotation_quat_xyzw: Vec4;
  rotation_rotvec_deg: Vec3;
}

export interface CalibrationArtifacts {
  spatial_unit: string;
  angle_unit: string;
  cameras: CameraArtifact[];
  transforms: TransformArtifact[];
}

export interface BenchTiming {
  init_ms: number;
  optimize_ms: number;
  total_ms: number;
  detection_ms: number;
}

export interface ReprojLevelGap {
  from: ReprojLevel;
  to: ReprojLevel;
  mean_delta_px: number;
  ratio_to_previous?: number | null;
  ratio_to_intrinsic?: number | null;
}

export interface CompactLevelReport {
  level: ReprojLevel;
  overall: LevelStats;
  per_camera: LevelStats[];
  per_view: LevelStats[];
  residual_count: number;
  top_outliers: TargetFeatureRecord[];
}

export interface CompactReprojReport {
  headline_px: number;
  levels: CompactLevelReport[];
  gaps: ReprojLevelGap[];
}

export interface BenchRecord {
  ident: BenchIdent;
  convergence: BenchConvergence;
  fit: BenchFit;
  generalization?: unknown | null;
  stability?: unknown | null;
  detection?: DetectionSummary | null;
  laser?: LaserMetrics | null;
  robot_corrections?: RobotCorrectionSummary | null;
  artifacts?: CalibrationArtifacts | null;
  delta_to_prior?: unknown | null;
  timing: BenchTiming;
  reproj_report?: CompactReprojReport | null;
}

function isNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isVec(value: unknown, length: number): value is number[] {
  return Array.isArray(value) && value.length === length && value.every(isNumber);
}

function requireArray<T>(value: unknown, name: string): T[] {
  if (!Array.isArray(value)) {
    throw new Error(`${name} must be an array`);
  }
  return value as T[];
}

export function parseManifest(value: unknown): ViewerManifest {
  if (!value || typeof value !== 'object') {
    throw new Error('manifest must be an object');
  }
  const manifest = value as ViewerManifest;
  if (manifest.schema_version !== 1) {
    throw new Error(`unsupported manifest schema_version ${manifest.schema_version}`);
  }
  if (!manifest.dataset || !Number.isInteger(manifest.dataset.num_cameras)) {
    throw new Error('dataset.num_cameras is required');
  }
  if (!isVec(manifest.dataset.tile_size, 2)) {
    throw new Error('dataset.tile_size must be [width, height]');
  }
  const stages = requireArray<ViewerStage>(manifest.stages, 'stages');
  if (stages.length === 0) {
    throw new Error('manifest must contain at least one stage');
  }
  for (const stage of stages) {
    if (!stage.id || !stage.label) {
      throw new Error('each stage requires id and label');
    }
    requireArray<CameraRecord>(stage.geometry?.cameras, `stage ${stage.id} geometry.cameras`);
    requireArray<TargetFeatureRecord>(stage.target_features, `stage ${stage.id} target_features`);
    requireArray<LaserFeatureRecord>(stage.laser_features, `stage ${stage.id} laser_features`);
  }
  return manifest;
}

export function looksLikeBenchRecord(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false;
  const maybe = value as Partial<BenchRecord>;
  return typeof maybe.ident?.dataset_id === 'string'
    && Number.isInteger(maybe.ident?.bench_schema_version)
    && typeof maybe.fit === 'object';
}

export function parseBenchRecord(value: unknown): BenchRecord {
  if (!value || typeof value !== 'object') {
    throw new Error('benchmark record must be an object');
  }
  const record = value as BenchRecord;
  if (!record.ident || record.ident.bench_schema_version !== 3) {
    throw new Error(`unsupported bench_schema_version ${record.ident?.bench_schema_version}`);
  }
  if (!record.ident.dataset_id || !record.ident.problem || !record.ident.tier) {
    throw new Error('benchmark ident requires dataset_id, problem, and tier');
  }
  if (!record.fit?.overall || !isNumber(record.fit.overall.mean)) {
    throw new Error('benchmark fit.overall.mean is required');
  }
  if (!record.convergence?.report || !isNumber(record.convergence.report.final_cost)) {
    throw new Error('benchmark convergence.report.final_cost is required');
  }
  if (!record.timing || !isNumber(record.timing.total_ms)) {
    throw new Error('benchmark timing.total_ms is required');
  }
  if (record.reproj_report) {
    requireArray<CompactLevelReport>(record.reproj_report.levels, 'reproj_report.levels');
    requireArray<ReprojLevelGap>(record.reproj_report.gaps, 'reproj_report.gaps');
    for (const level of record.reproj_report.levels) {
      if (!level.level || !level.overall || !isNumber(level.overall.mean)) {
        throw new Error('each reproj_report level requires level and overall.mean');
      }
      requireArray<LevelStats>(level.per_camera, `reproj_report ${level.level} per_camera`);
      requireArray<LevelStats>(level.per_view, `reproj_report ${level.level} per_view`);
      requireArray<TargetFeatureRecord>(level.top_outliers, `reproj_report ${level.level} top_outliers`);
    }
  }
  if (record.artifacts) {
    requireArray<CameraArtifact>(record.artifacts.cameras, 'artifacts.cameras');
    requireArray<TransformArtifact>(record.artifacts.transforms, 'artifacts.transforms');
    for (const camera of record.artifacts.cameras) {
      if (!camera.camera_id || !isVec(camera.camera_matrix_px?.[0], 3)) {
        throw new Error('each artifact camera requires camera_id and camera_matrix_px');
      }
    }
  }
  return record;
}

export function resolveAssetUrl(manifestUrl: string, relativePath: string): string {
  return new URL(relativePath, manifestUrl).toString();
}
