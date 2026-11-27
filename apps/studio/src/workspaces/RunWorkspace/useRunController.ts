/** Runs the open project as the back end's `calibrate` job and follows it
 * to a terminal state. Unsaved edits are saved first (the project on disk
 * is what runs); the run the job writes is listed with the project's runs,
 * shown in the viewers, and the app routes to Diagnose. */
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { api, type Job } from "../../api";
import type { Progress, RunInfo } from "../../api/generated/types";
import { isDirty, useDraft } from "../../project/draftStore";
import { useProject } from "../../project/projectStore";
import { newRunId, statusFromError, withProgress, type RunStatus } from "./runStatus";

/** How long the success banner shows before the hand-off to Diagnose. */
const HANDOFF_MS = 600;

/** The run in flight, if any. Module-scoped so it survives the workspace
 * unmounting and remounting: a second run cannot start while the first is
 * still running or loading its result. */
let activeRunId: string | null = null;
/** Its job, once the back end has queued it. */
let activeJob: Job<Progress, RunInfo> | null = null;
/** Cancel was asked for before the job was queued. */
let cancelWanted = false;

export interface RunController {
  status: RunStatus;
  setStatus: (status: RunStatus) => void;
  run: () => Promise<void>;
  cancel: () => void;
}

export function useRunController(): RunController {
  const navigate = useNavigate();
  const [status, setStatus] = useState<RunStatus>({ kind: "idle" });
  const handoffTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (handoffTimerRef.current !== null) {
        clearTimeout(handoffTimerRef.current);
      }
    };
  }, []);

  const run = async () => {
    if (activeRunId !== null) return;
    if (!useProject.getState().project) {
      setStatus({
        kind: "error",
        category: "no_project",
        message: "Open or create a project first: a calibration runs on a project.",
      });
      return;
    }
    if (handoffTimerRef.current !== null) {
      clearTimeout(handoffTimerRef.current);
      handoffTimerRef.current = null;
    }
    const runId = newRunId();
    activeRunId = runId;
    cancelWanted = false;
    setStatus({
      kind: "running",
      runId,
      stage: null,
      report: null,
      step: null,
      startedAt: Date.now(),
      cancelRequested: false,
    });
    try {
      // The project on disk runs: unsaved edits are saved first.
      if (isDirty(useDraft.getState()) && !(await useDraft.getState().save())) {
        setStatus({
          kind: "error",
          category: "save",
          message:
            useProject.getState().error?.message ?? "The project could not be saved.",
        });
        return;
      }
      const project = useProject.getState().project;
      if (!project) return;
      const job = await api.jobs.calibrate({ dir: project.dir });
      activeJob = job;
      if (cancelWanted) void job.cancel().catch(() => {});
      job.onEvent((report) =>
        // Only advance if this run is still the active one (a stale event
        // must not resurrect a finished run).
        setStatus((prev) =>
          prev.kind === "running" && prev.runId === runId
            ? withProgress(prev, report)
            : prev,
        ),
      );
      let written: RunInfo;
      try {
        written = await job.result;
      } catch (e) {
        setStatus(statusFromError(e));
        return;
      }
      await useProject.getState().refreshRuns();
      await useProject.getState().openRun(written.id);
      setStatus({ kind: "ok", run: written });
      // The user left the Run workspace meanwhile: no hand-off.
      if (!mountedRef.current) return;
      handoffTimerRef.current = setTimeout(() => {
        handoffTimerRef.current = null;
        void navigate("/diagnose");
      }, HANDOFF_MS);
    } catch (e) {
      setStatus(statusFromError(e));
    } finally {
      if (activeRunId === runId) {
        activeRunId = null;
        activeJob = null;
      }
    }
  };

  const cancel = () => {
    if (status.kind !== "running") return;
    setStatus({ ...status, cancelRequested: true });
    // Fire-and-forget: the terminal `cancelled` state arrives as the
    // job's end. Before the job is queued, it is cancelled once it is.
    if (activeJob === null) {
      cancelWanted = true;
      return;
    }
    void activeJob.cancel().catch(() => {
      /* benign — the run may have finished between click and the request */
    });
  };

  return { status, setStatus, run, cancel };
}
