import { Button, Callout } from "@vitavision/ui";
import { tidy } from "../../project/configCheck";
import type { RunFailure } from "../../project/runEditor";

/** A calibration that wrote no run: what failed (the library's own words
 * and kind), where it stopped, and what to try. The editor still holds the
 * settings that failed; nothing is restored behind the user's back. */
export function FailureView({
  failure,
  onDismiss,
  onReset,
}: {
  failure: RunFailure;
  onDismiss: () => void;
  onReset: () => void;
}) {
  const { error } = failure;
  return (
    <Callout
      tone="error"
      title={
        failure.stage
          ? `The calibration failed ${failure.stage}`
          : "The calibration failed"
      }
      actions={
        <>
          <Button size="sm" variant="secondary" onClick={onReset}>
            Back to the project's settings
          </Button>
          <Button size="sm" variant="ghost" onClick={onDismiss}>
            Dismiss
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-2 text-xs">
        <p className="break-words text-fg" data-testid="failure-message">
          {tidy(error.message)}{" "}
          <span className="font-mono text-[11px] text-fg-muted">({error.kind})</span>
        </p>
        <p className="text-fg-muted">
          <span className="font-medium text-fg">What to try. </span>
          {failure.advice}
        </p>
        <p className="text-fg-subtle">
          No run was written. The editor still holds the settings that failed
          {failure.fromRun ? `, started from ${failure.fromRun}` : ""}.
        </p>
      </div>
    </Callout>
  );
}
