import { Progress, Tooltip } from "@vitavision/ui";
import { Activity } from "lucide-react";
import { indicatorLine } from "../session/jobModel";
import { useJobs } from "../session/jobs";

/** The header's job button. Nothing at all while no job has been heard of;
 * a quiet icon once there is a finished one to look at; while one runs, its
 * bar and stage (`Detecting 41/120`, `Solving · iter 12 · cost 3.2e−1`,
 * `Queued · position 2`). It opens and closes the tray. */
export function JobIndicator() {
  const rows = useJobs((s) => s.rows);
  const open = useJobs((s) => s.trayOpen);
  const setOpen = useJobs((s) => s.setTrayOpen);
  const line = indicatorLine(rows);

  if (rows.length === 0) return null;
  const label = line ? `Jobs: ${line.text}` : "Jobs";

  return (
    <Tooltip content={line ? line.text : "Recent jobs"}>
      <button
        type="button"
        aria-label={label}
        aria-expanded={open}
        aria-controls="job-tray"
        onClick={() => setOpen(!open)}
        data-running={line !== null}
        className="inline-flex h-7 items-center gap-2 rounded-control px-2 text-xs text-fg-muted outline-none hover:bg-raised hover:text-fg focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-signal aria-expanded:bg-raised aria-expanded:text-fg"
      >
        <Activity aria-hidden className="size-3.5" />
        {line && (
          <>
            <Progress
              aria-label="Job progress"
              className="w-20"
              value={line.progress ? line.progress.done : null}
              max={line.progress ? line.progress.total : 100}
            />
            <span className="hidden max-w-72 truncate font-mono text-[11px] text-fg lg:inline">
              {line.text}
            </span>
          </>
        )}
      </button>
    </Tooltip>
  );
}
