import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { GIZMO_LAYER } from "@vitavision/three";
import { FrameAxes, useSceneColors } from "@vitavision/three-react";
import { useMemo } from "react";
import { Vector3 } from "three";
import { cameraPositionInRig, iso3FromWire } from "../../lib/se3";
import { useStore } from "../../store";
import type {
  AnyExport,
  Iso3Wire,
  LaserPlaneWire,
  PinholeCameraWire,
} from "../../store/types";
import type { TargetFeatureResidual } from "../../types";
import { CameraFrustum } from "./CameraFrustum";
import { laserFanPose } from "./laserFanPose";
import { LaserPlane } from "./LaserPlane";
import { LaserTargetCuts } from "./LaserTargetCuts";
import { TargetBoard } from "./TargetBoard";

interface SceneProps {
  data: AnyExport;
  showAllPoses: boolean;
  /** Render `laser_planes_rig` as bounded translucent quads. */
  showLaserPlanes: boolean;
  /** Per-camera image dimensions (pixels). Built from the manifest's
   * ROI metadata in the workspace; falls back to a sensible default
   * when a camera has no manifest entry. The frustum aspect ratio
   * comes from these — using the same value for every camera skews the
   * field of view on tiled rigs (e.g. the puzzle 130×130 6×720×540). */
  cameraDimensions: Map<number, { width: number; height: number }>;
  /** Last-resort fallback when a camera index has no manifest entry. */
  fallbackImage: { width: number; height: number };
}

const FAR_DEPTH_M = 0.05; // 5 cm — long enough to read on the puzzle
// 130×130 rig, short enough not to occlude
// the target board on small workspaces.

// Stable fallbacks for the optional export arrays. `data.cam_se3_rig ?? []`
// would otherwise mint a fresh `[]` every render, changing the identity of
// the `useMemo` dependencies below on every render even when the export
// itself hasn't changed.
const EMPTY_ISO3: Iso3Wire[] = [];
const EMPTY_LASER_PLANES: LaserPlaneWire[] = [];
const EMPTY_CAMERAS: PinholeCameraWire[] = [];

/** R3F scene root. Renders rig origin + per-camera frustums + the
 * active pose's target board (or all poses as ghosts). Click events
 * on a frustum or board drive the workspace selection state. */
export function Scene({
  data,
  showAllPoses,
  showLaserPlanes,
  cameraDimensions,
  fallbackImage,
}: SceneProps) {
  // Tokens via `@vitavision/three-react`; index.css aliases its token
  // names onto this app's palette.
  const colors = useSceneColors();
  const cameras = data.cameras ?? EMPTY_CAMERAS;
  const camSe3Rig = data.cam_se3_rig ?? EMPTY_ISO3;
  const rigSe3Target = data.rig_se3_target ?? EMPTY_ISO3;
  const laserPlanesRig = data.laser_planes_rig ?? EMPTY_LASER_PLANES;
  const cameraA = useStore((s) => s.cameraA);
  const selectedPose = useStore((s) => s.selectedPose);
  const setCamera = useStore((s) => s.setCamera);
  const setSelectedPose = useStore((s) => s.setSelectedPose);

  // Auto-fit: place the orbit camera so all frustums + the active
  // target board fit comfortably. Computed once per export — not on
  // every selection — so the user keeps their orbit pose while
  // navigating.
  const fit = useMemo(
    () => computeFit(camSe3Rig, rigSe3Target),
    [camSe3Rig, rigSe3Target],
  );

  // Pre-bucket residuals by pose for the target board. Avoids a linear
  // scan per board, especially when "show all poses" is on.
  const residualsByPose = useMemo(() => {
    const m = new Map<number, TargetFeatureResidual[]>();
    for (const r of data.per_feature_residuals.target) {
      const arr = m.get(r.pose);
      if (arr) arr.push(r);
      else m.set(r.pose, [r]);
    }
    return m;
  }, [data]);

  const visiblePoses: number[] = showAllPoses
    ? rigSe3Target.map((_, i) => i)
    : [selectedPose].filter((i) => i >= 0 && i < rigSe3Target.length);
  const activePose =
    selectedPose >= 0 && selectedPose < rigSe3Target.length
      ? rigSe3Target[selectedPose]
      : null;
  const activePoseResiduals =
    selectedPose >= 0 ? (residualsByPose.get(selectedPose) ?? []) : [];

  // Root each laser plane's fan at its owning camera (plane i belongs to
  // camera i); see `laserFanPose`.
  const laserFans = useMemo(
    () =>
      laserPlanesRig.map((plane, camera) => ({
        camera,
        ...laserFanPose(plane, camSe3Rig[camera]),
      })),
    [laserPlanesRig, camSe3Rig],
  );

  return (
    <Canvas
      orthographic={false}
      camera={{
        position: fit.cameraPos,
        fov: 35,
        near: 0.001,
        far: 50,
      }}
      gl={{ antialias: true }}
      style={{ background: colors.background }}
      // `@vitavision/three` puts its gizmos on GIZMO_LAYER. A canvas other
      // than the package's `SceneCanvas` must enable it on its camera and
      // raycaster, or they are neither drawn nor picked.
      onCreated={({ camera, raycaster }) => {
        camera.layers.enable(GIZMO_LAYER);
        raycaster.layers.enable(GIZMO_LAYER);
      }}
    >
      <ambientLight intensity={0.5} />
      <directionalLight position={[1, 2, 1]} intensity={0.4} />

      {/* Rig origin: X defect, Y normal, Z signal. */}
      <FrameAxes size={0.05} />

      {cameras
        .map((camera, id) => ({ camera, id }))
        .map(({ camera, id }) => {
          const pose = camSe3Rig[id];
          if (!pose) return null;
          const dims = cameraDimensions.get(id) ?? fallbackImage;
          return (
            <CameraFrustum
              key={`cam-${id}`}
              cameraIndex={id}
              camera={camera}
              camSe3Rig={pose}
              imageWidth={dims.width}
              imageHeight={dims.height}
              farDepth={FAR_DEPTH_M}
              color={id === cameraA ? colors.signal : colors.muted}
              active={id === cameraA}
              onSelect={() => setCamera(id, "A")}
              label={`cam ${id}`}
            />
          );
        })}

      {showLaserPlanes &&
        laserFans.map((fan) => (
          <LaserPlane
            key={`laser-plane-${fan.camera}`}
            matrix={fan.matrix}
            reach={fan.reach}
            active={fan.camera === cameraA}
            onSelect={() => setCamera(fan.camera, "A")}
          />
        ))}

      {visiblePoses.map((poseIdx) => {
        const pose = rigSe3Target[poseIdx];
        if (!pose) return null;
        const residuals = residualsByPose.get(poseIdx) ?? [];
        return (
          <TargetBoard
            key={`board-${poseIdx}`}
            rigSe3Target={pose}
            residuals={residuals}
            color={poseIdx === selectedPose ? colors.signal : colors.muted}
            fillColor={colors.signal}
            ghost={showAllPoses && poseIdx !== selectedPose}
            onSelect={() => setSelectedPose(poseIdx, "A")}
          />
        );
      })}

      {activePose && laserPlanesRig.length > 0 && (
        <LaserTargetCuts
          rigSe3Target={activePose}
          residuals={activePoseResiduals}
          planesRig={laserPlanesRig}
        />
      )}

      <OrbitControls
        target={fit.target}
        makeDefault
        enableDamping
        dampingFactor={0.08}
        minDistance={0.05}
        maxDistance={20}
      />
    </Canvas>
  );
}

interface FitResult {
  cameraPos: [number, number, number];
  target: [number, number, number];
}

/** Compute an OrbitControls target + camera position that frames the
 * rig + visible target boards. Falls back to a sensible default when
 * no rig data is available. */
function computeFit(
  camSe3Rig: {
    rotation: [number, number, number, number];
    translation: [number, number, number];
  }[],
  rigSe3Target: {
    rotation: [number, number, number, number];
    translation: [number, number, number];
  }[],
): FitResult {
  const points: Vector3[] = [new Vector3(0, 0, 0)]; // rig origin
  for (const c of camSe3Rig) {
    const [x, y, z] = cameraPositionInRig(c);
    points.push(new Vector3(x, y, z));
  }
  for (const t of rigSe3Target) {
    const m = iso3FromWire(t);
    points.push(new Vector3().setFromMatrixPosition(m));
  }
  if (points.length <= 1) {
    return { cameraPos: [0.5, 0.4, 0.5], target: [0, 0, 0] };
  }
  const center = new Vector3();
  for (const p of points) center.add(p);
  center.divideScalar(points.length);
  let radius = 0;
  for (const p of points) {
    radius = Math.max(radius, p.distanceTo(center));
  }
  // Place the camera one radius back along an isometric-ish vector.
  const offset = new Vector3(1, 0.7, 1).normalize().multiplyScalar(radius * 3);
  const camPos = center.clone().add(offset);
  return {
    cameraPos: [camPos.x, camPos.y, camPos.z],
    target: [center.x, center.y, center.z],
  };
}
