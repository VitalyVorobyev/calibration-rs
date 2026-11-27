/** The jobs the shell shows, as pure rules over the event stream: a row
 * per job from its events, what a row says about its progress, and which
 * rows stay in the tray. A job is the back end's (`calibrate`,
 * `fetch_sample`, `disparity`); this module knows their progress reports
 * only as far as a line of text and a bar need. */
import type {
  JobEvent,
  JobInfo,
  JobState,
  Progress,
  RunInfo,
} from "../api/generated/types";
import { formatCost, formatCount, formatPixels } from "../lib/format";
import { dirnamePath } from "../lib/paths";
import { runRms } from "../project/projectModel";

/** One job as the tray lists it. */
export interface JobRow {
  id: string;
  /** The job command (`calibrate`). */
  command: string;
  state: JobState;
  /** Its latest progress report. */
  event: unknown;
  /** The solver's cost per iteration, for the plot; empty for other jobs. */
  costs: number[];
  /** When this client first heard of it, ordering rows (not a clock). */
  order: number;
  /** The user asked to cancel and the job has not ended yet. */
  cancelling: boolean;
}

/** Points kept of a solver's cost (a long solve is thinned, not unbounded). */
const MAX_COSTS = 500;
/** Finished jobs kept in the tray besides the live ones. */
export const KEEP_FINISHED = 3;
/** Failed jobs kept until dismissed. */
export const MAX_FAILED = 5;

export function isLive(state: JobState): boolean {
  return state.kind === "queued" || state.kind === "running";
}

function isProgress(event: unknown): event is Progress {
  return (
    typeof event === "object" &&
    event !== null &&
    typeof (event as { kind?: unknown }).kind === "string"
  );
}

function withEvent(row: JobRow, event: unknown): JobRow {
  if (event === undefined || event === null) return row;
  const costs =
    row.command === "calibrate" && isProgress(event) && event.kind === "iteration"
      ? [...row.costs, event.cost].slice(-MAX_COSTS)
      : row.costs;
  return { ...row, event, costs };
}

/** `rows` with the job of `body` added or advanced. */
export function applyJobEvent(rows: JobRow[], body: JobEvent): JobRow[] {
  const existing = rows.find((r) => r.id === body.id);
  const base: JobRow = existing ?? {
    id: body.id,
    command: body.command,
    state: body.state,
    event: null,
    costs: [],
    order: rows.reduce((m, r) => Math.max(m, r.order), 0) + 1,
    cancelling: false,
  };
  const next = withEvent(
    { ...base, state: body.state, cancelling: base.cancelling && isLive(body.state) },
    body.event,
  );
  return existing ? rows.map((r) => (r.id === body.id ? next : r)) : [...rows, next];
}

/** `rows` brought up to what `list_jobs` says: jobs this client missed
 * (another tab's, one that ran while the stream was down) are added, and a
 * known job takes the newer state. */
export function mergeListed(rows: JobRow[], listed: readonly JobInfo[]): JobRow[] {
  let next = rows;
  for (const info of listed) {
    const known = next.find((r) => r.id === info.id);
    // A terminal state the stream already delivered stays.
    if (known && !isLive(known.state)) continue;
    next = applyJobEvent(next, {
      id: info.id,
      command: info.command,
      state: info.state,
      event: info.last_event ?? null,
    });
  }
  return next;
}

/** The rows the tray keeps, oldest first: every live job, the last few
 * that finished, and the failures until they are dismissed (a failure is
 * not a thing to lose by looking away; the last `MAX_FAILED` stay). */
export function pruneFinished(rows: JobRow[], keep = KEEP_FINISHED): JobRow[] {
  const newestFirst = (a: JobRow, b: JobRow) => b.order - a.order;
  const ended = rows.filter(
    (r) => r.state.kind === "done" || r.state.kind === "cancelled",
  );
  const failed = rows.filter((r) => r.state.kind === "failed");
  const drop = new Set(
    [
      ...ended.sort(newestFirst).slice(keep),
      ...failed.sort(newestFirst).slice(MAX_FAILED),
    ].map((r) => r.id),
  );
  return rows.filter((r) => !drop.has(r.id)).sort((a, b) => a.order - b.order);
}

/** The tray's order: what runs first, then what waits (in the order it
 * will run), then what ended, newest first. */
export function trayOrder(rows: readonly JobRow[]): JobRow[] {
  const rank = (r: JobRow) => (r.state.kind === "running" ? 0 : isLive(r.state) ? 1 : 2);
  return [...rows].sort((a, b) => {
    const byRank = rank(a) - rank(b);
    if (byRank !== 0) return byRank;
    return rank(a) === 2 ? b.order - a.order : a.order - b.order;
  });
}

const TITLES: Record<string, string> = {
  calibrate: "Calibration",
  fetch_sample: "Sample download",
  disparity: "Dense stereo",
  detect_all: "Detection",
};

export function jobTitle(command: string): string {
  return TITLES[command] ?? command.replace(/_/g, " ");
}

/** What a row says: a one-line stage and, when the report gives done and
 * total, a determinate bar. */
export interface JobLine {
  /** `Detecting 41/120`, `Solving · iter 12 · cost 3.2e−1`, `Queued · position 2`. */
  stage: string;
  /** Done and total of a counted stage; `null` for an unknown amount. */
  progress: { done: number; total: number } | null;
}

export function describeJob(row: JobRow): JobLine {
  const { state } = row;
  switch (state.kind) {
    case "queued":
      return { stage: `Queued · position ${state.position}`, progress: null };
    case "done":
      return { stage: "Done", progress: null };
    case "cancelled":
      return { stage: "Cancelled", progress: null };
    case "failed":
      return { stage: "Failed", progress: null };
    case "running":
      break;
  }
  if (row.cancelling) return { stage: "Cancelling…", progress: null };
  const event = row.event;
  if (!isProgress(event)) return { stage: "Running", progress: null };
  switch (event.kind) {
    case "detection":
      return {
        stage: `Detecting ${formatCount(event.done)}/${formatCount(event.total)}`,
        progress: { done: event.done, total: event.total },
      };
    case "laser_extraction":
      return {
        stage: `Extracting laser ${formatCount(event.done)}/${formatCount(event.total)}`,
        progress: { done: event.done, total: event.total },
      };
    case "step":
      return { stage: event.name, progress: null };
    case "iteration":
      return {
        stage: `Solving · iter ${event.iter} · cost ${formatCost(event.cost)}`,
        progress: { done: event.iter, total: event.max_iters },
      };
    default:
      return { stage: "Running", progress: null };
  }
}

/** The run a finished `calibrate` job wrote, `run-008 · Pass · 0.231 px`;
 * `null` for any other job or a job that did not finish. */
export function resultChip(
  row: JobRow,
): { runId: string; projectDir: string; text: string } | null {
  if (row.command !== "calibrate" || row.state.kind !== "done") return null;
  const run = row.state.output as RunInfo | null;
  if (!run || typeof run.id !== "string") return null;
  const rms = runRms(run);
  const verdict =
    run.state.kind === "recorded" ? (run.state.summary.verdict ?? null) : null;
  const parts = [run.id];
  if (verdict) parts.push(VERDICT_WORD[verdict]);
  if (rms !== null) parts.push(formatPixels(rms));
  // The run's folder is `<project>/runs/<id>`.
  return {
    runId: run.id,
    projectDir: dirnamePath(dirnamePath(run.dir)),
    text: parts.join(" · "),
  };
}

const VERDICT_WORD = { pass: "Pass", warn: "Check", fail: "Fail" } as const;

/** The header indicator's text: the running job's stage (with how many are
 * behind it), the queue when none runs, or `null` when idle. */
export function indicatorLine(rows: readonly JobRow[]): {
  text: string;
  progress: { done: number; total: number } | null;
} | null {
  const live = rows.filter((r) => isLive(r.state));
  if (live.length === 0) return null;
  const running = live.find((r) => r.state.kind === "running") ?? live[0]!;
  const line = describeJob(running);
  const behind = live.length - 1;
  return {
    text: behind > 0 ? `${line.stage} (+${behind})` : line.stage,
    progress: line.progress,
  };
}
