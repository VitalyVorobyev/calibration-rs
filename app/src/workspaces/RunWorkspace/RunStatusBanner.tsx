import { Button, Callout } from "@vitavision/ui";
import { Check, LoaderCircle, X } from "lucide-react";
import { useEffect, useState } from "react";
import type { RunStage } from "../../lib/runCalibration";
import { computeStageRows, formatElapsed, type StageRow } from "./runStages";
import type { RunStatus } from "./runStatus";

interface RunStatusBannerProps {
  status: RunStatus;
  onCancel: () => void;
}

/** The banner over the Run workspace for every non-idle state. `ask_user`
 * is a dialog (`AskUserDialog`), not a banner. */
export function RunStatusBanner({ status, onCancel }: RunStatusBannerProps) {
  switch (status.kind) {
    case "idle":
    case "ask_user":
      return null;
    case "running":
      return (
        <RunProgressPanel
          stage={status.stage}
          startedAt={status.startedAt}
          cancelRequested={status.cancelRequested}
          onCancel={onCancel}
        />
      );
    case "cancelled":
      return (
        <Callout tone="info" title="Run cancelled">
          <span className="font-mono text-xs">
            stopped at the stage boundary — no export was produced
          </span>
        </Callout>
      );
    case "ok":
      return (
        <Callout tone="success" title="Solve completed">
          <span className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-xs">
              {status.durationMs} ms · {status.usable}/{status.total} usable views
              {status.cacheUsed ? " · cache" : ""}
              {status.savedAs ? ` · saved as runs/${status.savedAs}` : ""}
            </span>
            <span className="ml-auto text-xs">Routing to /diagnose…</span>
          </span>
        </Callout>
      );
    case "validation":
      return (
        <Callout tone="info" title="Validation failed">
          <code className="font-mono text-xs">{status.message}</code>
        </Callout>
      );
    case "error":
      return (
        <Callout tone="error" title={`Run failed (${status.category})`}>
          <code className="font-mono text-xs text-fg">{status.message}</code>
        </Callout>
      );
  }
}

/** Live elapsed-time readout for the running banner. Owns its own 200ms
 * interval so only this leaf re-renders 5x/s while a run is in flight,
 * not the whole form-heavy workspace tree above it. */
function ElapsedClock({ startedAt }: { startedAt: number }) {
  const [elapsedMs, setElapsedMs] = useState(() => Date.now() - startedAt);
  useEffect(() => {
    const id = setInterval(() => setElapsedMs(Date.now() - startedAt), 200);
    return () => clearInterval(id);
  }, [startedAt]);
  return (
    <span className="ml-auto font-mono text-xs font-normal text-fg-muted tabular-nums">
      {formatElapsed(elapsedMs)}
    </span>
  );
}

interface RunProgressPanelProps {
  stage: RunStage | null;
  /** Epoch ms the run started — display-only, never enters any export. */
  startedAt: number;
  cancelRequested: boolean;
  onCancel: () => void;
}

/** Stage checklist + live elapsed clock + Cancel button for an in-flight
 * run. The three stages (detect → solve → export) come from the runner's
 * progress channel; cancellation takes effect at the next boundary. */
function RunProgressPanel({
  stage,
  startedAt,
  cancelRequested,
  onCancel,
}: RunProgressPanelProps) {
  const rows = computeStageRows(stage, { cancelled: cancelRequested });
  return (
    <Callout
      tone="info"
      title={
        <span className="flex items-center gap-2">
          {cancelRequested ? (
            "Cancelling…"
          ) : (
            <>
              <Spinner />
              Detection + calibration in progress…
            </>
          )}
          <ElapsedClock startedAt={startedAt} />
        </span>
      }
      actions={
        <Button
          size="sm"
          onClick={onCancel}
          disabled={cancelRequested}
          title="Stop the run at the next stage boundary"
        >
          {cancelRequested ? "Cancelling…" : "Cancel"}
        </Button>
      }
    >
      <ul className="flex flex-col gap-1">
        {rows.map((row) => (
          <StageRowItem key={row.id} row={row} />
        ))}
      </ul>
      {!cancelRequested && (
        <span className="mt-2 block font-mono text-[10px] text-fg-muted">
          first run is slowest; second hits the detection cache
        </span>
      )}
    </Callout>
  );
}

const STAGE_TEXT: Record<StageRow["state"], string> = {
  done: "text-normal",
  active: "text-fg",
  cancelled: "text-fg-muted line-through",
  pending: "text-fg-muted",
};

function StageRowItem({ row }: { row: StageRow }) {
  return (
    <li className="flex items-center gap-2 font-mono text-[11px]">
      <StageStateIcon state={row.state} />
      <span className={STAGE_TEXT[row.state]}>{row.label}</span>
    </li>
  );
}

function StageStateIcon({ state }: { state: StageRow["state"] }) {
  const box = "flex size-3.5 shrink-0 items-center justify-center";
  switch (state) {
    case "active":
      return <Spinner />;
    case "done":
      return (
        <span className={`${box} text-normal`}>
          <Check aria-label="done" className="size-2.5" />
        </span>
      );
    case "cancelled":
      return (
        <span className={`${box} text-fg-muted`}>
          <X aria-label="cancelled" className="size-2.5" />
        </span>
      );
    case "pending":
      return (
        <span className={box} aria-label="pending">
          <span className="size-1.5 rounded-full border border-fg-muted/60" />
        </span>
      );
  }
}

function Spinner() {
  return <LoaderCircle aria-label="Running" className="size-3.5 shrink-0 animate-spin" />;
}
