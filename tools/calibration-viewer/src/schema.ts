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

export function resolveAssetUrl(manifestUrl: string, relativePath: string): string {
  return new URL(relativePath, manifestUrl).toString();
}
