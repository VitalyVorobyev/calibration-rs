/** The jobs the back end runs and whether its event stream is up, from the
 * one standing subscription. Every tab hears every job, whoever started
 * it; `list_jobs` fills in what ran before this page loaded or while the
 * stream was down. The rules for a row are `jobModel.ts`. */
import { toast } from "@vitavision/ui";
import { create } from "zustand";
import { api, asStudioError, events, isNetworkError, type Connection } from "../api";
import type { ServerEvent } from "../api/generated/types";
import {
  applyJobEvent,
  jobTitle,
  mergeListed,
  pruneFinished,
  resultChip,
  type JobRow,
} from "./jobModel";

/** How long to wait between asks while the back end does not answer. */
const PROBE_MS = 3000;

export interface JobsState {
  rows: JobRow[];
  connection: Connection;
  /** When the stream last delivered anything (epoch ms; for the "last
   * heard" line only, never exported). */
  lastEventAt: number | null;
  /** Whether the job tray is open. */
  trayOpen: boolean;
  setTrayOpen: (open: boolean) => void;
  /** Ask the back end to cancel job `id`; the end arrives as an event. */
  cancel: (id: string) => Promise<void>;
  /** Drop a finished or failed row from the tray. */
  dismiss: (id: string) => void;
}

export const useJobs = create<JobsState>()((set, get) => ({
  rows: [],
  connection: "connecting",
  lastEventAt: null,
  trayOpen: false,
  setTrayOpen: (trayOpen) => set({ trayOpen }),

  cancel: async (id) => {
    set({ rows: get().rows.map((r) => (r.id === id ? { ...r, cancelling: true } : r)) });
    try {
      await api.cancelJob({ id });
    } catch (e) {
      // It ended on its own first, or the back end is gone: say no more.
      set({
        rows: get().rows.map((r) => (r.id === id ? { ...r, cancelling: false } : r)),
      });
      if (asStudioError(e).kind !== "job_not_found") throw e;
    }
  },

  dismiss: (id) => set({ rows: get().rows.filter((r) => r.id !== id) }),
}));

/** Whether what writes should be disabled: the connection is lost, so a
 * command would only fail. */
export function useReadOnly(): boolean {
  return useJobs((s) => s.connection === "lost");
}

/** Take one event of the stream into the store; a job that just finished
 * or failed says so in a toast. */
function onJobEvent(body: Extract<ServerEvent["body"], { kind: "job" }>): void {
  const before = useJobs.getState().rows.find((r) => r.id === body.id);
  const rows = pruneFinished(applyJobEvent(useJobs.getState().rows, body));
  useJobs.setState({ rows });
  const row = rows.find((r) => r.id === body.id);
  if (!row || before?.state.kind === row.state.kind) return;
  if (row.state.kind === "done") {
    const chip = resultChip(row);
    toast({
      title: `${jobTitle(row.command)} finished`,
      ...(chip ? { description: chip.text } : {}),
      tone: "success",
    });
  }
}

/** Fill the rows in from `list_jobs`: what ran before this page, and what
 * ended while the stream was down. */
export async function syncJobs(): Promise<void> {
  try {
    const listed = await api.listJobs();
    useJobs.setState({
      rows: pruneFinished(mergeListed(useJobs.getState().rows, listed)),
    });
  } catch {
    // The back end is not answering: the connection state says so.
  }
}

/** Start watching: the standing stream, its state, and the jobs. `onBack`
 * is called each time the stream comes back after a drop (the caller
 * refetches what it shows). Returns the stop function. Idempotent for the
 * caller that keeps the stop. */
export function startJobWatch(onBack: () => void): () => void {
  let wasLost = false;
  const stopConnection = events.onConnection((connection) => {
    useJobs.setState({ connection });
    if (connection === "lost") wasLost = true;
    if (connection === "open") {
      void syncJobs();
      if (wasLost) onBack();
      wasLost = false;
    }
  });
  const stopListen = events.listen((event) => {
    useJobs.setState({ lastEventAt: Date.now() });
    if (event.body.kind === "job") onJobEvent(event.body);
  });

  // A command that never reached the back end says it is gone even while
  // the stream has not noticed: ask it, until it answers.
  let probe: ReturnType<typeof setTimeout> | null = null;
  let stopped = false;
  const schedule = () => {
    if (probe !== null || stopped) return;
    probe = setTimeout(() => {
      probe = null;
      api.serverInfo().then(back, (e: unknown) => {
        if (isNetworkError(e)) schedule();
        else back();
      });
    }, PROBE_MS);
  };
  const back = () => {
    if (useJobs.getState().connection === "lost")
      useJobs.setState({ connection: "open" });
    void syncJobs();
    wasLost = false;
    onBack();
  };
  const stopUnreachable = events.onUnreachable(() => {
    wasLost = true;
    if (useJobs.getState().connection !== "lost")
      useJobs.setState({ connection: "lost" });
    schedule();
  });

  void events.connect();
  return () => {
    stopped = true;
    if (probe !== null) clearTimeout(probe);
    stopConnection();
    stopListen();
    stopUnreachable();
    events.disconnect();
  };
}
