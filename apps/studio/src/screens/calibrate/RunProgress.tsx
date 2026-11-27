import { LineChart } from "@vitavision/charts";
import { Button, Progress, cn } from "@vitavision/ui";
import { Check, Circle, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { formatDuration } from "../../lib/format";
import { describeJob } from "../../session/jobModel";
import { useJobs } from "../../session/jobs";
import { stageChecklist } from "./stages";

/** Seconds a run has been going, for the panel's "elapsed" line only. */
function useElapsed(): number {
  const [started] = useState(() => Date.now());
  const [now, setNow] = useState(started);
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, []);
  return now - started;
}

/** The job view of "This run" while a calibration is queued or running:
 * the stage checklist, the solver's cost, the time it has taken and Cancel.
 * `jobId` is `null` between the press of Run and the job's id. */
export function RunProgress({ jobId }: { jobId: string | null }) {
  const row = useJobs((s) => s.rows.find((r) => r.id === jobId));
  const cancel = useJobs((s) => s.cancel);
  const elapsed = useElapsed();
  const line = row ? describeJob(row) : null;
  const stages = stageChecklist(row?.event ?? null, false);
  const queued = row?.state.kind === "queued";

  return (
    <section
      aria-label="Calibration in progress"
      className="flex flex-col gap-3 rounded-panel border border-line bg-surface p-3"
    >
      <div className="flex items-center gap-2">
        <Loader2 aria-hidden className="size-3.5 animate-spin text-signal" />
        <h3 className="text-xs font-semibold text-fg">
          {queued ? "Waiting for its turn" : "Calibrating"}
        </h3>
        <span className="ml-auto font-mono text-[11px] text-fg-muted">
          {formatDuration(elapsed)}
        </span>
      </div>
      {queued && line && <p className="text-xs text-fg-muted">{line.stage}</p>}
      <ol className="flex flex-col gap-1.5" aria-label="Stages">
        {stages.map((stage) => (
          <li
            key={stage.id}
            aria-current={stage.state === "active" ? "step" : undefined}
            className="flex items-center gap-2 text-xs"
          >
            {stage.state === "done" ? (
              <Check aria-hidden className="size-3.5 text-normal" />
            ) : stage.state === "active" && !queued ? (
              <Loader2 aria-hidden className="size-3.5 animate-spin text-signal" />
            ) : (
              <Circle aria-hidden className="size-3.5 text-fg-subtle" />
            )}
            <span
              className={cn(stage.state === "pending" ? "text-fg-subtle" : "text-fg")}
            >
              {stage.label}
              <span className="sr-only"> ({stage.state})</span>
            </span>
            {stage.detail && (
              <span className="ml-auto font-mono text-[11px] text-fg-muted">
                {stage.detail}
              </span>
            )}
          </li>
        ))}
      </ol>
      {row?.state.kind === "running" && (
        <Progress
          aria-label="Calibration progress"
          value={line?.progress ? line.progress.done : null}
          max={line?.progress ? line.progress.total : 100}
        />
      )}
      {row && row.costs.length > 1 && (
        <LineChart
          label="Solver cost per iteration"
          xLabel="iteration"
          yLabel="cost"
          logY
          variant="fluid"
          height={110}
          showLegend={false}
          series={[
            {
              name: "cost",
              points: row.costs.filter((c) => c > 0).map((y, x) => ({ x: x + 1, y })),
            },
          ]}
        />
      )}
      <div className="flex items-center justify-between gap-2">
        <p className="text-[11px] text-fg-muted">
          Edits stay possible; they apply to the next run.
        </p>
        <Button
          size="sm"
          variant="secondary"
          disabled={!row || row.cancelling}
          onClick={() => row && void cancel(row.id).catch(() => {})}
        >
          {row?.cancelling ? "Cancelling…" : "Cancel"}
        </Button>
      </div>
    </section>
  );
}
