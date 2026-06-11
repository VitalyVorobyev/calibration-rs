import { describe, expect, it } from 'vitest';
import { composeTransforms, matrixFromRows } from '../src/geometry';
import {
  looksLikeBenchRecord,
  parseBenchRecord,
  parseManifest,
  resolveAssetUrl,
  type BenchRecord,
  type TransformRecord,
  type ViewerManifest,
} from '../src/schema';

const manifest: ViewerManifest = {
  schema_version: 1,
  generator: 'test',
  dataset: {
    name: 'synthetic',
    source_dir: '.',
    num_cameras: 1,
    board_rows: 2,
    board_cols: 2,
    cell_size_mm: 1,
    full_image_size: [100, 50],
    tile_size: [100, 50],
  },
  frame_conventions: {
    cam_se3_rig: 'T_C_R',
    cam_to_rig: 'T_R_C',
    eye_to_hand_chain: 'T_C_T = T_C_R * T_R_B * T_B_G * T_G_T',
    robot_delta: 'T_B_G_corr = exp(delta) * T_B_G',
  },
  poses: [],
  cameras: [],
  handeye_mode: 'EyeToHand',
  handeye: {
    translation_m: [0, 0, 0],
    quaternion_xyzw: [0, 0, 0, 1],
    matrix4_row_major: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
  },
  target_ref: {
    translation_m: [0, 0, 0],
    quaternion_xyzw: [0, 0, 0, 1],
    matrix4_row_major: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
  },
  stages: [{
    id: 'stage4',
    label: 'Joint BA',
    mean_reproj_error_px: 0.5,
    final_cost: 1,
    robot_deltas: [],
    per_camera_stats: [],
    geometry: {
      handeye: {
        translation_m: [0, 0, 0],
        quaternion_xyzw: [0, 0, 0, 1],
        matrix4_row_major: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      },
      target_ref: {
        translation_m: [0, 0, 0],
        quaternion_xyzw: [0, 0, 0, 1],
        matrix4_row_major: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      },
      cameras: [],
    },
    target_features: [],
    laser_features: [],
  }],
};

const benchRecord: BenchRecord = {
  ident: {
    dataset_id: 'stereo_rig',
    problem: 'rig_extrinsics',
    tier: 'b',
    git_sha: 'deadbeef',
    timestamp_rfc3339: '2026-05-31T12:00:00Z',
    config_hash: 42,
    bench_schema_version: 3,
    features: ['tier-b'],
  },
  convergence: {
    init_ok: true,
    converged: true,
    report: {
      final_cost: 0.25,
      num_iters: 8,
    },
  },
  fit: {
    overall: { mean: 0.25, rms: 0.3, max: 1.2, count: 120 },
    per_camera: [{ mean: 0.2, rms: 0.25, max: 0.9, count: 60 }],
    per_camera_hist: [],
    reported_mean_reproj_px: 0.25,
    reported_per_cam_px: [0.2],
  },
  generalization: null,
  stability: null,
  detection: {
    total_detected: 118,
    total_expected: 120,
    per_camera: [{
      camera_id: 'cam0',
      images_total: 10,
      images_used: 10,
      features_detected: 118,
      features_expected: 120,
      coverage_pct: 98.333,
      detect_ms: 50,
    }],
  },
  laser: null,
  robot_corrections: {
    count: 10,
    mean_rot_deg: 0.01,
    max_rot_deg: 0.02,
    mean_trans_mm: 0.2,
    max_trans_mm: 0.4,
  },
  artifacts: {
    spatial_unit: 'mm',
    angle_unit: 'deg',
    cameras: [{
      camera_id: 'cam0',
      camera_matrix_px: [[800, 0, 320], [0, 801, 240], [0, 0, 1]],
      intrinsics_px: { fx: 800, fy: 801, cx: 320, cy: 240, skew: 0 },
      distortion_model: 'brown_conrady5',
      distortion: { k1: -0.1, k2: 0.02, k3: 0, p1: 0.001, p2: -0.001 },
      scheimpflug: null,
    }],
    transforms: [{
      name: 'cam0_se3_rig',
      to_frame: 'cam0',
      from_frame: 'rig',
      translation_mm: [1, 2, 3],
      rotation_quat_xyzw: [0, 0, 0, 1],
      rotation_rotvec_deg: [0, 0, 0],
    }],
  },
  delta_to_prior: null,
  timing: {
    init_ms: 2,
    optimize_ms: 10,
    total_ms: 70,
    detection_ms: 50,
  },
  reproj_report: {
    headline_px: 0.25,
    levels: [{
      level: 'intrinsic',
      overall: { mean: 0.18, median: 0.16, rms: 0.2, p95: 0.4, max: 0.8, count: 120 },
      per_camera: [{ mean: 0.18, median: 0.16, rms: 0.2, p95: 0.4, max: 0.8, count: 120 }],
      per_view: [{ mean: 0.18, median: 0.16, rms: 0.2, p95: 0.4, max: 0.8, count: 12 }],
      residual_count: 120,
      top_outliers: [{
        pose: 0,
        camera: 0,
        feature: 1,
        target_xyz_m: [0, 0, 0],
        observed_px: [10, 10],
        projected_px: [10.2, 10.1],
        error_px: 0.22,
      }],
    }, {
      level: 'rig_extrinsic',
      overall: { mean: 0.25, median: 0.22, rms: 0.3, p95: 0.6, max: 1.2, count: 120 },
      per_camera: [{ mean: 0.25, median: 0.22, rms: 0.3, p95: 0.6, max: 1.2, count: 120 }],
      per_view: [{ mean: 0.25, median: 0.22, rms: 0.3, p95: 0.6, max: 1.2, count: 12 }],
      residual_count: 120,
      top_outliers: [],
    }],
    gaps: [{
      from: 'intrinsic',
      to: 'rig_extrinsic',
      mean_delta_px: 0.07,
      ratio_to_previous: 1.389,
      ratio_to_intrinsic: 1.389,
    }],
  },
};

describe('viewer manifest schema', () => {
  it('accepts the v1 manifest shape', () => {
    expect(parseManifest(manifest).dataset.num_cameras).toBe(1);
  });

  it('rejects unsupported schema versions', () => {
    expect(() => parseManifest({ ...manifest, schema_version: 99 })).toThrow(/unsupported/);
  });

  it('resolves asset paths relative to the manifest URL', () => {
    expect(resolveAssetUrl('http://localhost/viewer-data/puzzle/viewer_manifest.json', 'images/target_0.png'))
      .toBe('http://localhost/viewer-data/puzzle/images/target_0.png');
  });

  it('converts row-major transforms into Three matrices', () => {
    const matrix = matrixFromRows([[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]]);
    expect(matrix.elements[12]).toBe(2);
    expect(matrix.elements[13]).toBe(3);
    expect(matrix.elements[14]).toBe(4);
  });

  it('composes exported transforms in chain order', () => {
    const a: TransformRecord = {
      translation_m: [1, 0, 0],
      quaternion_xyzw: [0, 0, 0, 1],
      matrix4_row_major: [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    };
    const b: TransformRecord = {
      translation_m: [0, 2, 0],
      quaternion_xyzw: [0, 0, 0, 1],
      matrix4_row_major: [[1, 0, 0, 0], [0, 1, 0, 2], [0, 0, 1, 0], [0, 0, 0, 1]],
    };
    const matrix = composeTransforms(a, b);
    expect(matrix.elements[12]).toBe(1);
    expect(matrix.elements[13]).toBe(2);
  });
});

describe('benchmark record schema', () => {
  it('accepts compact v3 benchmark records', () => {
    expect(looksLikeBenchRecord(benchRecord)).toBe(true);
    expect(parseBenchRecord(benchRecord).reproj_report?.headline_px).toBe(0.25);
    expect(parseBenchRecord(benchRecord).artifacts?.cameras[0].camera_matrix_px[0][0]).toBe(800);
  });

  it('rejects unsupported benchmark schemas', () => {
    expect(() => parseBenchRecord({
      ...benchRecord,
      ident: { ...benchRecord.ident, bench_schema_version: 99 },
    })).toThrow(/unsupported/);
  });
});
