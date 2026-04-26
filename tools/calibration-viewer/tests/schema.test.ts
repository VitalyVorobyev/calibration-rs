import { describe, expect, it } from 'vitest';
import { composeTransforms, matrixFromRows } from '../src/geometry';
import { parseManifest, resolveAssetUrl, type TransformRecord, type ViewerManifest } from '../src/schema';

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
