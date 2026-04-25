import { Matrix4, Quaternion, Vector3 } from 'three';
import type { Matrix4Rows, TransformRecord, Vec3 } from './schema';

export function matrixFromRows(rows: Matrix4Rows): Matrix4 {
  const m = new Matrix4();
  m.set(
    rows[0][0],
    rows[0][1],
    rows[0][2],
    rows[0][3],
    rows[1][0],
    rows[1][1],
    rows[1][2],
    rows[1][3],
    rows[2][0],
    rows[2][1],
    rows[2][2],
    rows[2][3],
    rows[3][0],
    rows[3][1],
    rows[3][2],
    rows[3][3],
  );
  return m;
}

export function transformToPose(transform: TransformRecord): {
  position: Vector3;
  quaternion: Quaternion;
  matrix: Matrix4;
} {
  return {
    position: new Vector3(...transform.translation_m),
    quaternion: new Quaternion(...transform.quaternion_xyzw),
    matrix: matrixFromRows(transform.matrix4_row_major),
  };
}

export function composeTransforms(...transforms: TransformRecord[]): Matrix4 {
  const result = new Matrix4().identity();
  for (const transform of transforms) {
    result.multiply(matrixFromRows(transform.matrix4_row_major));
  }
  return result;
}

export function vec3(v: Vec3): Vector3 {
  return new Vector3(v[0], v[1], v[2]);
}

export function formatMm(valueM: number): string {
  return `${(valueM * 1000).toFixed(2)} mm`;
}

export function formatPx(value: number): string {
  return `${value.toFixed(3)} px`;
}
