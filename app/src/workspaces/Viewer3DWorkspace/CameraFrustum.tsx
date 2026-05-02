import { useMemo } from "react";
import { BufferGeometry, Float32BufferAttribute, type Object3D } from "three";
import { iso3InverseFromWire } from "../../lib/se3";
import type { Iso3Wire, PinholeCameraWire } from "../../store/types";

/** R3F's `raycast={fn}` prop expects a `Object3D['raycast']`. We use
 * a no-op to opt the visual-only meshes out of the raycaster pipeline,
 * letting clicks pass through to the dedicated hitbox quad below.
 * `Object3D['raycast']` returns void, so an empty function is the
 * canonical "no hit" signal. */
const NO_RAYCAST: Object3D["raycast"] = () => {};

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

  // Far-plane hitbox: a transparent quad covering the full far plane
  // of the frustum. Three.js's raycaster doesn't intersect line
  // geometry by default, so without this hitbox the only clickable
  // surface is the small apex sphere — and most users naturally try
  // to click on the visible far quad. The mesh has `visible={false}`
  // so it draws nothing but still participates in raycasting.
  const hitboxGeom = useMemo(() => buildFarPlaneHitbox(corners), [corners]);

  return (
    <group
      matrix={matrix}
      matrixAutoUpdate={false}
      onClick={(e) => {
        if (!onSelect) return;
        // Stop propagation so clicks on overlapping frustums don't
        // double-fire onto whichever group is rendered next.
        e.stopPropagation();
        onSelect();
      }}
      onPointerOver={(e) => {
        if (!onSelect) return;
        e.stopPropagation();
        document.body.style.cursor = "pointer";
      }}
      onPointerOut={() => {
        document.body.style.cursor = "";
      }}
    >
      {/* Wireframe edges — visual only. Opt out of raycast so clicks
        pass through to the hitbox below. */}
      <lineSegments geometry={geom} raycast={NO_RAYCAST}>
        <lineBasicMaterial color={color} linewidth={active ? 2 : 1} />
      </lineSegments>
      {/* Apex marker — visual only. */}
      <mesh raycast={NO_RAYCAST}>
        <sphereGeometry args={[farDepth * 0.06, 12, 8]} />
        <meshBasicMaterial color={color} transparent opacity={active ? 0.9 : 0.45} />
      </mesh>
      {/* Invisible far-plane hitbox. The `visible={false}` flag stops
        Three from rasterising this mesh, but the raycaster still
        considers it. */}
      <mesh geometry={hitboxGeom} visible={false}>
        <meshBasicMaterial transparent opacity={0} />
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

/** Build a two-triangle BufferGeometry covering the four far-plane
 * corners. Used as an invisible click target — Three.js's raycaster
 * picks up triangulated meshes by default but skips line geometry. */
function buildFarPlaneHitbox(
  corners: [number, number, number][],
): BufferGeometry {
  // Triangulate the quad as (0, 1, 2) + (0, 2, 3).
  const [a, b, c, d] = corners;
  const verts: number[] = [
    ...a, ...b, ...c,
    ...a, ...c, ...d,
  ];
  const g = new BufferGeometry();
  g.setAttribute("position", new Float32BufferAttribute(verts, 3));
  return g;
}
