import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";
import { useThemeColors } from "../Viewer3DWorkspace/useThemeColors";

interface PointCloudViewProps {
  /** Flat `[x, y, z, …]` positions (reference-camera frame, metres). */
  positions: Float32Array;
  /** Flat `[r, g, b, …]` per-point colours in `[0, 1]`. */
  colors: Float32Array;
}

/** Orbitable WebGL point cloud reprojected from the disparity map. Lazy-loaded
 * (Three.js is heavy) and framed automatically from the cloud bounds. */
export function PointCloudView({ positions, colors }: PointCloudViewProps) {
  const theme = useThemeColors();
  const frame = useMemo(() => computeFrame(positions), [positions]);

  return (
    // Remount on a new cloud so the auto-framed camera re-applies.
    <Canvas
      key={positions.length}
      camera={{ position: frame.cameraPos, fov: 40, near: 0.001, far: 100 }}
      gl={{ antialias: true }}
      style={{ background: theme.background }}
    >
      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
          <bufferAttribute attach="attributes-color" args={[colors, 3]} />
        </bufferGeometry>
        <pointsMaterial size={2} sizeAttenuation={false} vertexColors />
      </points>
      {/* Reference-camera frame gizmo at the origin. */}
      <axesHelper args={[frame.radius * 0.4]} />
      <OrbitControls
        target={frame.target}
        makeDefault
        enableDamping
        dampingFactor={0.08}
        minDistance={frame.radius * 0.1}
        maxDistance={frame.radius * 30}
      />
    </Canvas>
  );
}

interface Frame {
  target: [number, number, number];
  cameraPos: [number, number, number];
  radius: number;
}

/** Centroid + bounding radius → an OrbitControls target and a camera placed on
 * the reference-camera side (−z) so the surface faces the viewer. */
function computeFrame(positions: Float32Array): Frame {
  const n = Math.max(positions.length / 3, 1);
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (let i = 0; i < positions.length; i += 3) {
    cx += positions[i];
    cy += positions[i + 1];
    cz += positions[i + 2];
  }
  cx /= n;
  cy /= n;
  cz /= n;

  let r = 1e-3;
  for (let i = 0; i < positions.length; i += 3) {
    const dx = positions[i] - cx;
    const dy = positions[i + 1] - cy;
    const dz = positions[i + 2] - cz;
    r = Math.max(r, Math.hypot(dx, dy, dz));
  }

  const d = r * 2.6;
  const len = Math.hypot(0.4, -0.5, -1);
  const dir: [number, number, number] = [0.4 / len, -0.5 / len, -1 / len];
  return {
    target: [cx, cy, cz],
    cameraPos: [cx + dir[0] * d, cy + dir[1] * d, cz + dir[2] * d],
    radius: r,
  };
}
