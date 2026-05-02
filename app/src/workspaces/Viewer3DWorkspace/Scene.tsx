import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";
import { Vector3 } from "three";
import { cameraPositionInRig, iso3FromWire } from "../../lib/se3";
import { useStore } from "../../store";
import type { AnyExport } from "../../store/types";
import type { TargetFeatureResidual } from "../../types";
import { CameraFrustum } from "./CameraFrustum";
import { TargetBoard } from "./TargetBoard";
import { useThemeColors } from "./useThemeColors";

interface SceneProps {
  data: AnyExport;
  showAllPoses: boolean;
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

/** R3F scene root. Renders rig origin + per-camera frustums + the
 * active pose's target board (or all poses as ghosts). Click events
 * on a frustum or board drive the workspace selection state. */
export function Scene({
  data,
  showAllPoses,
  cameraDimensions,
  fallbackImage,
}: SceneProps) {
  const colors = useThemeColors();
  const cameras = data.cameras ?? [];
  const camSe3Rig = data.cam_se3_rig ?? [];
  const rigSe3Target = data.rig_se3_target ?? [];
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
    >
      <ambientLight intensity={0.5} />
      <directionalLight position={[1, 2, 1]} intensity={0.4} />

      {/* Rig origin gizmo. axesHelper isn't theme-aware; use thin
          line segments so it reads as a quiet reference. */}
      <axesHelper args={[0.05]} />

      {cameras.map((camera, i) => {
        const pose = camSe3Rig[i];
        if (!pose) return null;
        const dims = cameraDimensions.get(i) ?? fallbackImage;
        return (
          <CameraFrustum
            key={`cam-${i}`}
            camera={camera}
            camSe3Rig={pose}
            imageWidth={dims.width}
            imageHeight={dims.height}
            farDepth={FAR_DEPTH_M}
            color={i === cameraA ? colors.active : colors.inactive}
            active={i === cameraA}
            onSelect={() => setCamera(i, "A")}
            label={`cam ${i}`}
          />
        );
      })}

      {visiblePoses.map((poseIdx) => {
        const pose = rigSe3Target[poseIdx];
        if (!pose) return null;
        const residuals = residualsByPose.get(poseIdx) ?? [];
        return (
          <TargetBoard
            key={`board-${poseIdx}`}
            rigSe3Target={pose}
            residuals={residuals}
            color={poseIdx === selectedPose ? colors.active : colors.inactive}
            fillColor={colors.boardFill}
            ghost={showAllPoses && poseIdx !== selectedPose}
            onSelect={() => setSelectedPose(poseIdx, "A")}
          />
        );
      })}

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
  camSe3Rig: { rotation: [number, number, number, number]; translation: [number, number, number] }[],
  rigSe3Target: { rotation: [number, number, number, number]; translation: [number, number, number] }[],
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
