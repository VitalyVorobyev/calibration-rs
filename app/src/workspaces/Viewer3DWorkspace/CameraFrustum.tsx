import { Html } from "@react-three/drei";
import { useMemo } from "react";
import { BufferGeometry, DoubleSide, Float32BufferAttribute, type Object3D } from "three";
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
  /** Compact label shown for the active camera. */
  label?: string;
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
  label,
}: CameraFrustumProps) {
  const matrix = useMemo(() => iso3InverseFromWire(camSe3Rig), [camSe3Rig]);

  const corners = useMemo(() => frustumCorners(camera, imageWidth, imageHeight, farDepth), [
    camera,
    imageWidth,
    imageHeight,
    farDepth,
  ]);

  const geom = useMemo(() => buildEdgeGeometry(corners), [corners]);
  const farPlaneGeom = useMemo(() => buildFarPlaneHitbox(corners), [corners]);

  // Whole-frustum hitbox: transparent side faces + far plane. Three.js's
  // raycaster doesn't intersect line geometry, and a single front-sided
  // far plane made selection feel arbitrary from most orbit angles.
  const hitboxGeom = useMemo(() => buildFrustumHitbox(corners), [corners]);

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
        <lineBasicMaterial
          color={color}
          linewidth={active ? 3 : 1}
          transparent
          opacity={active ? 1 : 0.55}
        />
      </lineSegments>
      {active && (
        <lineSegments geometry={geom} raycast={NO_RAYCAST}>
          <lineBasicMaterial color="#ffffff" linewidth={1} transparent opacity={0.55} />
        </lineSegments>
      )}
      <mesh geometry={farPlaneGeom} raycast={NO_RAYCAST}>
        <meshBasicMaterial
          color={color}
          transparent
          opacity={active ? 0.16 : 0.035}
          side={DoubleSide}
          depthWrite={false}
        />
      </mesh>
      {/* Apex marker — visual only. */}
      <mesh raycast={NO_RAYCAST}>
        <sphereGeometry args={[farDepth * (active ? 0.095 : 0.06), 16, 10]} />
        <meshBasicMaterial color={color} transparent opacity={active ? 1 : 0.45} />
      </mesh>
      {active &&
        corners.map((corner, i) => (
          <mesh key={`corner-${i}`} position={corner} raycast={NO_RAYCAST}>
            <sphereGeometry args={[farDepth * 0.05, 12, 8]} />
            <meshBasicMaterial color={color} transparent opacity={0.95} />
          </mesh>
        ))}
      {active && label && (
        <Html position={[0, -farDepth * 0.18, 0]} center style={{ pointerEvents: "none" }}>
          <span className="rounded border border-brand/70 bg-bg-soft/90 px-1.5 py-0.5 font-mono text-[10px] text-brand shadow-sm">
            {label}
          </span>
        </Html>
      )}
      {/* Invisible but raycastable camera body. */}
      <mesh geometry={hitboxGeom}>
        <meshBasicMaterial transparent opacity={0} side={DoubleSide} depthWrite={false} />
      </mesh>
      <mesh>
        <sphereGeometry args={[farDepth * 0.14, 12, 8]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} />
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

/** Build a two-sided transparent frustum hull for picking: four side
 * triangles plus the far-plane quad. Corners are padded around the
 * far-plane center so near-edge clicks still feel intentional. */
function buildFrustumHitbox(
  corners: [number, number, number][],
): BufferGeometry {
  const padded = padCorners(corners, 1.18);
  const apex: [number, number, number] = [0, 0, 0];
  const [a, b, c, d] = padded;
  const verts: number[] = [
    ...apex, ...a, ...b,
    ...apex, ...b, ...c,
    ...apex, ...c, ...d,
    ...apex, ...d, ...a,
    ...a, ...b, ...c,
    ...a, ...c, ...d,
  ];
  const g = new BufferGeometry();
  g.setAttribute("position", new Float32BufferAttribute(verts, 3));
  return g;
}

function padCorners(
  corners: [number, number, number][],
  scale: number,
): [number, number, number][] {
  const center = corners.reduce<[number, number, number]>(
    (acc, c) => [acc[0] + c[0] / 4, acc[1] + c[1] / 4, acc[2] + c[2] / 4],
    [0, 0, 0],
  );
  return corners.map((c) => [
    center[0] + (c[0] - center[0]) * scale,
    center[1] + (c[1] - center[1]) * scale,
    center[2] + (c[2] - center[2]) * scale,
  ]);
}
