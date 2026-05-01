interface PoseCameraStepperProps {
  pose: number;
  camera: number;
  numPoses: number;
  numCameras: number;
  onPose: (next: number) => void;
  onCamera: (next: number) => void;
}

/** Two stepper widgets side by side: pose (←/→) and camera (←/→).
 * Wraps around at the boundaries. The arrow keys are also bound at
 * the window level via `useKeyboardNav`; these buttons are the
 * mouse-only path. */
export function PoseCameraStepper({
  pose,
  camera,
  numPoses,
  numCameras,
  onPose,
  onCamera,
}: PoseCameraStepperProps) {
  const wrap = (i: number, n: number) => ((i % n) + n) % n;
  return (
    <div className="flex items-center gap-4">
      <Stepper
        label="pose"
        index={pose}
        count={numPoses}
        onChange={(next) => onPose(wrap(next, numPoses))}
      />
      <Stepper
        label="cam"
        index={camera}
        count={numCameras}
        onChange={(next) => onCamera(wrap(next, numCameras))}
      />
    </div>
  );
}

interface StepperProps {
  label: string;
  index: number;
  count: number;
  onChange: (next: number) => void;
}

function Stepper({ label, index, count, onChange }: StepperProps) {
  return (
    <div className="flex items-center gap-1">
      <span className="font-mono text-[11px] text-muted-foreground">{label}</span>
      <button
        type="button"
        onClick={() => onChange(index - 1)}
        aria-label={`previous ${label}`}
        className="grid h-7 w-7 place-items-center !p-0 font-mono text-xs"
        title={`previous ${label}`}
      >
        ◀
      </button>
      <span className="min-w-[3.5rem] text-center font-mono text-xs tabular-nums">
        {index + 1} / {count}
      </span>
      <button
        type="button"
        onClick={() => onChange(index + 1)}
        aria-label={`next ${label}`}
        className="grid h-7 w-7 place-items-center !p-0 font-mono text-xs"
        title={`next ${label}`}
      >
        ▶
      </button>
    </div>
  );
}
