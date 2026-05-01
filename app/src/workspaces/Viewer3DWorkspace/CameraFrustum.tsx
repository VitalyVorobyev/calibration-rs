import { useMemo } from "react";
import { BufferGeometry, Float32BufferAttribute } from "three";
import { iso3InverseFromWire } from "../../lib/se3";
import type { Iso3Wire, PinholeCameraWire } from "../../store/types";

interface CameraFrustumProps {
  camera: PinholeCameraWire;
  /** `cam_se3_rig` (T_C_R, rig→camera). Inverted internally to place
   * the frustum object in the rig frame. */
  camSe3Rig: Iso3Wire;
  /** Image dimensions in pixels. Falls back to 1024×768 when the
   * manifest doesn't supply them. Determines the aspect ratio of the
   * frustum's far plane. */
  imageWidth: number;
  imageHeight: number;
  /** Distance from apex to far plane (meters). */
  farDepth: number;
  /** Wireframe color. */
  color: string;
  /** True when this is the active camera (cameraA). Drawn slightly
   * thicker / more saturated. */
  active: boolean;
  /** Click handler — clicking the frustum sets cameraA in the store. */
  onSelect?: () => void;
}

/** Wireframe pyramid representing a calibrated pinhole camera. The
 * apex sits at the camera origin; the four edges go to the corners of
 * the image plane projected back to depth `farDepth`. */
export function CameraFrustum({
  camera,
  camSe3Rig,
  imageWidth,
  imageHeight,
  farDepth,
  color,
  active,
  onSelect,
}: CameraFrustumProps) {
  const matrix = useMemo(() => iso3InverseFromWire(camSe3Rig), [camSe3Rig]);

  const corners = useMemo(() => frustumCorners(camera, imageWidth, imageHeight, farDepth), [
    camera,
    imageWidth,
    imageHeight,
    farDepth,
  ]);

  const geom = useMemo(() => buildEdgeGeometry(corners), [corners]);

  return (
    <group matrix={matrix} matrixAutoUpdate={false} onClick={onSelect}>
      <lineSegments geometry={geom}>
        <lineBasicMaterial color={color} linewidth={active ? 2 : 1} />
      </lineSegments>
      {/* Solid apex marker so clicks register on a non-edge target. */}
      <mesh>
        <sphereGeometry args={[farDepth * 0.06, 12, 8]} />
        <meshBasicMaterial color={color} transparent opacity={active ? 0.9 : 0.45} />
      </mesh>
    </group>
  );
}

function frustumCorners(
  camera: PinholeCameraWire,
  width: number,
  height: number,
  depth: number,
): [number, number, number][] {
  const { fx, fy, cx, cy } = camera.k;
  // Four image corners projected through the camera at distance `depth`.
  const cornersPx: [number, number][] = [
    [0, 0],
    [width, 0],
    [width, height],
    [0, height],
  ];
  return cornersPx.map(([px, py]) => {
    const xn = (px - cx) / fx;
    const yn = (py - cy) / fy;
    return [xn * depth, yn * depth, depth] as [number, number, number];
  });
}

function buildEdgeGeometry(
  corners: [number, number, number][],
): BufferGeometry {
  // Edges: apex→each corner (×4), plus the far-plane rectangle (×4).
  const apex: [number, number, number] = [0, 0, 0];
  const verts: number[] = [];
  for (const c of corners) {
    verts.push(...apex, ...c);
  }
  for (let i = 0; i < 4; i++) {
    const a = corners[i];
    const b = corners[(i + 1) % 4];
    verts.push(...a, ...b);
  }
  const g = new BufferGeometry();
  g.setAttribute("position", new Float32BufferAttribute(verts, 3));
  return g;
}
