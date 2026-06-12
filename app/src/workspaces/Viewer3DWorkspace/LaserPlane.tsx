import { useMemo } from "react";
import { DoubleSide, Quaternion, Vector3 } from "three";
import type { LaserPlaneWire } from "../../store/types";

interface LaserPlaneProps {
  /** Plane in rig frame: unit `normal`, signed `distance`
   * (`n · p + d = 0`). */
  plane: LaserPlaneWire;
  /** Point on (or near) the plane the quad is centred at — typically
   * the owning camera's position projected onto the plane, so the quad
   * shows up where the device actually measures, not at the rig
   * origin's closest point. Projected onto the plane here in case the
   * caller passes an off-plane anchor. */
  anchor: [number, number, number];
  /** Quad half-extent in meters. */
  halfExtent: number;
  /** Outline color (matches the owning camera's frustum selection). */
  color: string;
  active?: boolean;
  onSelect?: () => void;
}

/** Laser line color, independent of the theme: the fill stays
 * recognizably "laser red" against the brand-tinted target boards. */
const LASER_FILL = "#e74c3c";

/** Translucent bounded quad for one calibrated laser plane, with a
 * sharp outline like `TargetBoard` so it reads at any opacity. */
export function LaserPlane({
  plane,
  anchor,
  halfExtent,
  color,
  active = false,
  onSelect,
}: LaserPlaneProps) {
  const { position, quaternion } = useMemo(() => {
    const n = new Vector3(...plane.normal).normalize();
    const p = new Vector3(...anchor);
    // Snap the anchor onto the plane: p ← p − (n·p + d) n.
    p.addScaledVector(n, -(n.dot(p) + plane.distance));
    // planeGeometry lies in the local XY plane with +Z normal; rotate
    // +Z onto the plane normal. setFromUnitVectors handles the
    // antiparallel case internally.
    const q = new Quaternion().setFromUnitVectors(new Vector3(0, 0, 1), n);
    return { position: p, quaternion: q };
  }, [plane, anchor]);

  const outline = useMemo(() => {
    const e = halfExtent;
    return new Float32Array(
      [
        [-e, -e, 0],
        [e, -e, 0],
        [e, e, 0],
        [-e, e, 0],
        [-e, -e, 0],
      ].flat(),
    );
  }, [halfExtent]);

  return (
    <group position={position} quaternion={quaternion} onClick={onSelect}>
      <mesh>
        <planeGeometry args={[halfExtent * 2, halfExtent * 2]} />
        <meshBasicMaterial
          color={LASER_FILL}
          transparent
          opacity={active ? 0.22 : 0.08}
          depthWrite={false}
          side={DoubleSide}
        />
      </mesh>
      <line>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[outline, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color={color} transparent opacity={active ? 1 : 0.4} />
      </line>
    </group>
  );
}
