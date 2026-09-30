import { Html } from "@react-three/drei";
import type { ThreeEvent } from "@react-three/fiber";
import { invertIso3, matrixFromIso3 } from "@vitavision/three";
import { CameraFrustum as FrustumGizmo } from "@vitavision/three-react";
import { useMemo } from "react";
import type { Object3D } from "three";
import type { Iso3Wire, PinholeCameraWire } from "../../store/types";
import { useBorderRays } from "./useBorderRays";

/** Opts the visual-only markers out of raycasting, so picks land on the
 * frustum's hitbox. An empty function is three's "no hit". */
const NO_RAYCAST: Object3D["raycast"] = () => {};

/** How much larger than the drawn frustum its pick hull is. The package
 * also makes a sphere of `(PICK_PADDING - 1) × farDepth` around the
 * optical centre pickable, where a small camera is most often clicked.
 * 1.18 is the padding this viewer used before the package took over. */
const PICK_PADDING = 1.18;

interface CameraFrustumProps {
  /** Index into the export's `cameras`; names the camera to the backend's
   * back-projection (see `useBorderRays`). */
  cameraIndex: number;
  camera: PinholeCameraWire;
  /** `cam_se3_rig` (T_C_R, rig→camera). Inverted to place the frustum in
   * the rig frame. */
  camSe3Rig: Iso3Wire;
  /** Image dimensions in pixels: the border whose rays bound the frustum. */
  imageWidth: number;
  imageHeight: number;
  /** Distance from apex to far outline (meters). */
  farDepth: number;
  /** Colour of the apex and corner markers; the frustum itself takes the
   * scene's `signal` / `fg-muted` tokens from `@vitavision/three-react`. */
  color: string;
  /** True when this is the active camera (cameraA). */
  active: boolean;
  /** Click handler — clicking the frustum sets cameraA in the store. */
  onSelect?: () => void;
  /** Compact label shown for the active camera. */
  label?: string;
}

/** A calibrated camera: `@vitavision/three-react`'s `CameraFrustum` (edges
 * from the optical centre to the image border at `farDepth`, a faint far
 * face, an invisible padded picking hull with a pickable apex), plus what
 * this viewer adds around it — an apex marker, corner markers and a DOM
 * label for the active camera. */
export function CameraFrustum({
  cameraIndex,
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
  const matrix = useMemo(() => matrixFromIso3(invertIso3(camSe3Rig)), [camSe3Rig]);
  const borderRays = useBorderRays(cameraIndex, camera.k, imageWidth, imageHeight);
  const corners = useMemo(() => farCorners(borderRays, farDepth), [borderRays, farDepth]);

  return (
    // Picks bubble here from the frustum's hull and its apex sphere, so both
    // select through one handler.
    <group
      matrix={matrix}
      matrixAutoUpdate={false}
      onClick={(e: ThreeEvent<MouseEvent>) => {
        if (!onSelect) return;
        // Stop propagation so clicks on overlapping frustums don't
        // double-fire onto whichever group is rendered next.
        e.stopPropagation();
        onSelect();
      }}
      onPointerOver={(e: ThreeEvent<PointerEvent>) => {
        if (!onSelect) return;
        e.stopPropagation();
        document.body.style.cursor = "pointer";
      }}
      onPointerOut={() => {
        document.body.style.cursor = "";
      }}
    >
      <FrustumGizmo
        borderRays={borderRays}
        depth={farDepth}
        active={active}
        pickPadding={PICK_PADDING}
      />
      {/* Apex marker — visual only. */}
      <mesh raycast={NO_RAYCAST}>
        <sphereGeometry args={[farDepth * (active ? 0.095 : 0.06), 16, 10]} />
        <meshBasicMaterial color={color} transparent opacity={active ? 1 : 0.45} />
      </mesh>
      {active &&
        corners.map((corner) => (
          <mesh key={`corner-${corner.join(",")}`} position={corner} raycast={NO_RAYCAST}>
            <sphereGeometry args={[farDepth * 0.05, 12, 8]} />
            <meshBasicMaterial color={color} transparent opacity={0.95} />
          </mesh>
        ))}
      {active && label && (
        <Html
          position={[0, -farDepth * 0.18, 0]}
          center
          style={{ pointerEvents: "none" }}
        >
          <span className="rounded border border-brand/70 bg-bg-soft/90 px-1.5 py-0.5 font-mono text-[10px] text-brand shadow-sm">
            {label}
          </span>
        </Html>
      )}
    </group>
  );
}

/** The four image corners at `depth`: points 0, n, 2n, 3n of a border
 * sampled `n` per edge (`imageBorderPixels`). */
function farCorners(rays: Float64Array, depth: number): [number, number, number][] {
  const count = rays.length / 3;
  return [0, 1, 2, 3].map((q) => {
    const i = 3 * Math.round((q * count) / 4);
    const s = depth / rays[i + 2]!;
    return [rays[i]! * s, rays[i + 1]! * s, depth];
  });
}
