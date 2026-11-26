import { Button } from "@vitavision/ui";
import { PoseStepper } from "./PoseStepper";

interface PoseCameraStepperProps {
  poseValues: number[];
  selectedPose: number;
  onSelectPose: (next: number) => void;
  cameraValues: number[];
  selectedCamera: number;
  onSelectCamera: (next: number) => void;
}

/** Pose + camera selectors side by side. Pose has a typeable value
 * (handy for sparse manifests where stepping is slow); camera is a
 * compact stepper because rigs rarely have more than a handful. */
export function PoseCameraStepper({
  poseValues,
  selectedPose,
  onSelectPose,
  cameraValues,
  selectedCamera,
  onSelectCamera,
}: PoseCameraStepperProps) {
  return (
    <div className="flex items-center gap-4">
      <PoseStepper
        poseValues={poseValues}
        selectedPose={selectedPose}
        onSelectPose={onSelectPose}
      />
      <CameraStepper
        cameraValues={cameraValues}
        selectedCamera={selectedCamera}
        onSelectCamera={onSelectCamera}
      />
    </div>
  );
}

interface CameraStepperProps {
  cameraValues: number[];
  selectedCamera: number;
  onSelectCamera: (next: number) => void;
}

function CameraStepper({
  cameraValues,
  selectedCamera,
  onSelectCamera,
}: CameraStepperProps) {
  const stepBy = (delta: number) => {
    if (cameraValues.length === 0) return;
    const idx = cameraValues.indexOf(selectedCamera);
    if (idx < 0) {
      onSelectCamera(cameraValues[delta >= 0 ? 0 : cameraValues.length - 1]!);
      return;
    }
    const n = cameraValues.length;
    onSelectCamera(cameraValues[(((idx + delta) % n) + n) % n]!);
  };

  const idx = cameraValues.indexOf(selectedCamera);
  const ordinalLabel =
    idx >= 0 ? `${idx + 1}/${cameraValues.length}` : `?/${cameraValues.length}`;

  return (
    <div className="flex items-center gap-1">
      <span className="font-mono text-[11px] text-fg-muted">cam</span>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => stepBy(-1)}
        aria-label="previous camera"
        className="w-7 px-0 font-mono"
        title="previous camera"
      >
        ◀
      </Button>
      <span className="min-w-[1.5rem] text-center font-mono text-xs tabular-nums">
        {selectedCamera}
      </span>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => stepBy(+1)}
        aria-label="next camera"
        className="w-7 px-0 font-mono"
        title="next camera"
      >
        ▶
      </Button>
      <span
        className="font-mono text-[10px] tabular-nums text-fg-muted"
        title={`${idx + 1} of ${cameraValues.length} cameras`}
      >
        {ordinalLabel}
      </span>
    </div>
  );
}
