import { Badge, Button, DensityProvider, ErrorBox, Progress } from "@vitavision/ui";
import { LineChart } from "@vitavision/charts";
import { X } from "lucide-react";
import { Link } from "react-router";
import {
  describeJob,
  isLive,
  jobTitle,
  resultChip,
  trayOrder,
  type JobRow,
} from "../session/jobModel";
import { useJobs } from "../session/jobs";
import { placePath, type Place } from "./screens";

/** The bottom tray: a row per job (queued, running, the last few that
 * ended, every failure until dismissed) with its stage, its bar, Cancel, a
 * small plot of the solver's cost while a calibration solves, and on
 * success a chip that opens the run it wrote. */
export function JobTray({ place }: { place: Place }) {
  const rows = useJobs((s) => s.rows);
  const open = useJobs((s) => s.trayOpen);
  const setOpen = useJobs((s) => s.setTrayOpen);
  if (!open) return null;

  return (
    <section
      id="job-tray"
      aria-label="Jobs"
      className="max-h-[40vh] overflow-y-auto border-t border-line bg-surface"
    >
      <DensityProvider value="compact">
        <div className="flex items-center justify-between px-3 pt-2">
          <h2 className="text-[10px] font-semibold tracking-wider text-fg-muted uppercase">
            Jobs
          </h2>
          <Button
            size="sm"
            variant="ghost"
            aria-label="Close the job tray"
            onClick={() => setOpen(false)}
          >
            <X aria-hidden className="size-3.5" />
          </Button>
        </div>
        {rows.length === 0 ? (
          <p className="px-3 pb-3 text-xs text-fg-muted">No jobs yet.</p>
        ) : (
          <ul className="flex flex-col gap-px p-2 pt-1">
            {trayOrder(rows).map((row) => (
              <li key={row.id}>
                <JobItem row={row} place={place} />
              </li>
            ))}
          </ul>
        )}
      </DensityProvider>
    </section>
  );
}

function JobItem({ row, place }: { row: JobRow; place: Place }) {
  const cancel = useJobs((s) => s.cancel);
  const dismiss = useJobs((s) => s.dismiss);
  const line = describeJob(row);
  const live = isLive(row.state);
  const chip = resultChip(row);

  return (
    <div className="flex flex-col gap-1.5 rounded-control px-2 py-1.5 hover:bg-raised/60">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-fg">{jobTitle(row.command)}</span>
        <span className="font-mono text-[11px] text-fg-muted">{row.id}</span>
        {row.state.kind === "queued" && <Badge>Queued</Badge>}
        {row.state.kind === "cancelled" && <Badge>Cancelled</Badge>}
        {row.state.kind === "done" && <Badge tone="normal">Done</Badge>}
        {row.state.kind === "failed" && <Badge tone="defect">Failed</Badge>}
        <span className="ml-auto font-mono text-[11px] text-fg-muted">
          {live ? line.stage : ""}
        </span>
        {live && (
          <Button
            size="sm"
            variant="secondary"
            disabled={row.cancelling}
            onClick={() => void cancel(row.id).catch(() => {})}
          >
            {row.cancelling ? "Cancelling…" : "Cancel"}
          </Button>
        )}
        {!live && (
          <Button
            size="sm"
            variant="ghost"
            aria-label={`Dismiss ${jobTitle(row.command)} ${row.id}`}
            onClick={() => dismiss(row.id)}
          >
            <X aria-hidden className="size-3.5" />
          </Button>
        )}
      </div>
      {row.state.kind === "running" && (
        <Progress
          aria-label={`${jobTitle(row.command)} progress`}
          value={line.progress ? line.progress.done : null}
          max={line.progress ? line.progress.total : 100}
        />
      )}
      {row.state.kind === "running" && row.costs.length > 1 && (
        <LineChart
          label="Solver cost per iteration"
          xLabel="iteration"
          yLabel="cost"
          logY
          variant="fluid"
          height={96}
          showLegend={false}
          series={[
            {
              name: "cost",
              points: row.costs.filter((c) => c > 0).map((y, x) => ({ x: x + 1, y })),
            },
          ]}
        />
      )}
      {chip && (
        <div>
          <Button asChild size="sm" variant="secondary">
            <Link
              to={placePath({
                ...place,
                screen: "inspect",
                tab: place.tab ?? "frames",
                project: chip.projectDir,
                run: chip.runId,
              })}
            >
              {chip.text}
            </Link>
          </Button>
        </div>
      )}
      {row.state.kind === "failed" && (
        <ErrorBox>
          {row.state.error.message}{" "}
          <span className="font-mono text-[11px]">({row.state.error.kind})</span>
        </ErrorBox>
      )}
    </div>
  );
}
