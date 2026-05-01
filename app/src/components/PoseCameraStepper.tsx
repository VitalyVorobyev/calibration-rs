interface PoseCameraStepperProps {
  /** 1-indexed ordinal of the current pose within the meaningful set
   * (the manifest's available poses for the current camera, or all
   * distinct poses if the parent treats them globally). */
  poseOrdinal: number;
  /** Count of meaningful poses — the denominator in `i / N`. */
  poseTotal: number;
  cameraOrdinal: number;
  cameraTotal: number;
  /** Step the pose by `delta` (typically ±1). The parent handles
   * wraparound and ROI-aware skipping over manifest gaps. */
  onPoseStep: (delta: number) => void;
  onCameraStep: (delta: number) => void;
}

/** Two stepper widgets side by side: pose (◀ / ▶) and camera (◀ / ▶).
 * The arrow keys are also bound at the window level via
 * `useKeyboardNav`; these buttons are the mouse-only path. The
 * stepper itself is dumb — the parent decides what "next" means
 * (especially for sparse manifests where some `(pose, camera)`
 * combinations don't exist). */
export function PoseCameraStepper({
  poseOrdinal,
  poseTotal,
  cameraOrdinal,
  cameraTotal,
  onPoseStep,
  onCameraStep,
}: PoseCameraStepperProps) {
  return (
    <div className="flex items-center gap-4">
      <Stepper
        label="pose"
        ordinal={poseOrdinal}
        total={poseTotal}
        onPrev={() => onPoseStep(-1)}
        onNext={() => onPoseStep(+1)}
      />
      <Stepper
        label="cam"
        ordinal={cameraOrdinal}
        total={cameraTotal}
        onPrev={() => onCameraStep(-1)}
        onNext={() => onCameraStep(+1)}
      />
    </div>
  );
}

interface StepperProps {
  label: string;
  ordinal: number;
  total: number;
  onPrev: () => void;
  onNext: () => void;
}

function Stepper({ label, ordinal, total, onPrev, onNext }: StepperProps) {
  return (
    <div className="flex items-center gap-1">
      <span className="font-mono text-[11px] text-muted-foreground">{label}</span>
      <button
        type="button"
        onClick={onPrev}
        aria-label={`previous ${label}`}
        className="grid h-7 w-7 place-items-center !p-0 font-mono text-xs"
        title={`previous ${label}`}
      >
        ◀
      </button>
      <span className="min-w-[3.5rem] text-center font-mono text-xs tabular-nums">
        {ordinal} / {total}
      </span>
      <button
        type="button"
        onClick={onNext}
        aria-label={`next ${label}`}
        className="grid h-7 w-7 place-items-center !p-0 font-mono text-xs"
        title={`next ${label}`}
      >
        ▶
      </button>
    </div>
  );
}
