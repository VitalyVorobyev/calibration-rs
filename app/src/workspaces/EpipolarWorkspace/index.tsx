import { Empty } from "@vitavision/ui";
import type { ReactNode } from "react";
import { useStore } from "../../store";
import { exportKindLabel } from "../../store/exportKind";
import { EpipolarBody } from "./EpipolarBody";

/** Epipolar workspace: the pair view of a rig export, or why there is
 * none. */
export function EpipolarWorkspace() {
  const data = useStore((s) => s.data);
  const kind = useStore((s) => s.kind);

  if (!data || !kind) {
    return (
      <EmptyEpipolar>
        Load a rig export to inspect epipolar geometry between cameras.
      </EmptyEpipolar>
    );
  }

  const isRig =
    Array.isArray(data.cameras) &&
    Array.isArray(data.cam_se3_rig) &&
    data.cameras.length >= 2;
  if (!isRig) {
    return (
      <EmptyEpipolar>
        {`Epipolar geometry only applies to rig exports (rig_extrinsics, rig_handeye, rig_handeye_laserline, rig_laserline_device). The current export is ${exportKindLabel(
          kind,
        )}.`}
      </EmptyEpipolar>
    );
  }

  return <EpipolarBody data={data} kind={kind} />;
}

function EmptyEpipolar({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <h2 className="text-sm font-semibold tracking-tight text-fg">Epipolar geometry</h2>
      <Empty className="flex-1 justify-center rounded-panel border border-dashed border-line bg-surface px-6 [&>p]:max-w-md">
        {children}
      </Empty>
    </div>
  );
}
