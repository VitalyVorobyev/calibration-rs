import { Button, Input } from "@vitavision/ui";
import { useRef, useState } from "react";

interface PoseStepperProps {
  /** All valid pose values (from manifest / export). Sorted ascending. */
  poseValues: number[];
  /** Currently selected pose. */
  selectedPose: number;
  /** Called with the next valid pose value. The component snaps a
   * typed value to the nearest entry in `poseValues`; this callback
   * always receives a member of that list (or the unchanged
   * `selectedPose` if the typed text was unparseable). */
  onSelectPose: (next: number) => void;
  /** Optional label override; default "pose". */
  label?: string;
}

/** Pose selector with a typeable value. Combines step arrows (one
 * pose at a time, wrapping) with a small text input so engineers can
 * jump straight to "pose 47" instead of clicking the arrow 47 times.
 *
 * Typed values snap to the nearest member of `poseValues` on commit
 * (Enter / blur) — convenient when the manifest is sparse and a
 * raw typed integer might not exist. Invalid text reverts. */
export function PoseStepper({
  poseValues,
  selectedPose,
  onSelectPose,
  label = "pose",
}: PoseStepperProps) {
  const [draft, setDraft] = useState<string>(() => String(selectedPose));
  const inputRef = useRef<HTMLInputElement | null>(null);

  // Keep the visible text in sync with the canonical selection
  // whenever the parent moves it (arrow keys, click on a frustum,
  // wrap-around step). Avoid clobbering the user's typing while the
  // input is focused.
  const [focused, setFocused] = useState(false);
  const [syncedPose, setSyncedPose] = useState(selectedPose);
  if (syncedPose !== selectedPose) {
    setSyncedPose(selectedPose);
    if (!focused) setDraft(String(selectedPose));
  }

  const stepBy = (delta: number) => {
    if (poseValues.length === 0) return;
    const idx = poseValues.indexOf(selectedPose);
    if (idx < 0) {
      onSelectPose(poseValues[delta >= 0 ? 0 : poseValues.length - 1]!);
      return;
    }
    const n = poseValues.length;
    const next = poseValues[(((idx + delta) % n) + n) % n]!;
    onSelectPose(next);
  };

  const commit = () => {
    const parsed = Number.parseInt(draft, 10);
    if (Number.isNaN(parsed) || poseValues.length === 0) {
      setDraft(String(selectedPose));
      return;
    }
    // Snap to nearest valid pose. Sparse manifests routinely have gaps
    // (poses 0, 2, 5, …) so an exact match isn't guaranteed.
    let best = poseValues[0]!;
    let bestDist = Math.abs(best - parsed);
    for (const v of poseValues) {
      const d = Math.abs(v - parsed);
      if (d < bestDist) {
        best = v;
        bestDist = d;
      }
    }
    setDraft(String(best));
    if (best !== selectedPose) onSelectPose(best);
  };

  const idx = poseValues.indexOf(selectedPose);
  const ordinalLabel =
    idx >= 0 ? `${idx + 1}/${poseValues.length}` : `?/${poseValues.length}`;

  return (
    <div className="flex items-center gap-1">
      <span className="font-mono text-[11px] text-fg-muted">{label}</span>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => stepBy(-1)}
        aria-label={`previous ${label}`}
        className="w-7 px-0 font-mono"
        title={`previous ${label}`}
      >
        ◀
      </Button>
      <Input
        ref={inputRef}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onFocus={() => setFocused(true)}
        onBlur={() => {
          setFocused(false);
          commit();
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            commit();
            inputRef.current?.blur();
          } else if (e.key === "Escape") {
            setDraft(String(selectedPose));
            inputRef.current?.blur();
          }
        }}
        inputMode="numeric"
        className="h-7 w-14 px-1 text-center font-mono text-xs tabular-nums"
        aria-label={`${label} value`}
      />
      <Button
        variant="ghost"
        size="sm"
        onClick={() => stepBy(+1)}
        aria-label={`next ${label}`}
        className="w-7 px-0 font-mono"
        title={`next ${label}`}
      >
        ▶
      </Button>
      <span
        className="font-mono text-[10px] tabular-nums text-fg-muted"
        title={`${idx + 1} of ${poseValues.length} poses`}
      >
        {ordinalLabel}
      </span>
    </div>
  );
}
