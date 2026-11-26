/** Runs the current setup and follows it to a terminal state. A successful
 * run with a project open is saved into the project and reopened from its
 * `export.json`; without a project the export goes straight to the viewers.
 * Either way the app then routes to Diagnose. */
import { toast } from "@vitavision/ui";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { cancelRun, runCalibration, type RunSuccess } from "../../lib/runCalibration";
import { runExportPath } from "../../project/projectApi";
import { useProject } from "../../project/projectStore";
import { useStore } from "../../store";
import { topologyOf, useRunSetup } from "../../store/runSetup";
import { newRunId, statusFromResponse, withProgress, type RunStatus } from "./runStatus";

/** How long the success banner shows before the hand-off to Diagnose. */
const HANDOFF_MS = 600;

/** The run in flight, if any. Module-scoped so it survives the workspace
 * unmounting and remounting: a second run cannot start while the first is
 * still saving or loading its result. */
let activeRunId: string | null = null;

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
    const { manifest, config, manifestDir } = useRunSetup.getState();
    if (!manifestDir) {
      setStatus({
        kind: "error",
        category: "no_dir",
        message: "Set a dataset folder first (pick a preset or use the folder button).",
      });
      return;
    }
    if (handoffTimerRef.current !== null) {
      clearTimeout(handoffTimerRef.current);
      handoffTimerRef.current = null;
    }
    const runId = newRunId();
    activeRunId = runId;
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
      const response = await runCalibration({
        runId,
        manifest,
        config,
        manifestDir,
        onProgress: (progress) =>
          // Only advance the stage if this run is still the active one
          // (a stale channel message must not resurrect a finished run).
          setStatus((prev) =>
            prev.kind === "running" && prev.runId === runId
              ? withProgress(prev, progress)
              : prev,
          ),
      });
      if (response.kind !== "ok") {
        setStatus(statusFromResponse(response));
        return;
      }
      const savedAs = await deliver(response, { manifest, config, manifestDir });
      setStatus(statusFromResponse(response, savedAs));
      // The user left the Run workspace meanwhile: no hand-off.
      if (!mountedRef.current) return;
      handoffTimerRef.current = setTimeout(() => {
        handoffTimerRef.current = null;
        void navigate("/diagnose");
      }, HANDOFF_MS);
    } catch (e) {
      setStatus({ kind: "error", category: "ipc", message: String(e) });
    } finally {
      if (activeRunId === runId) {
        activeRunId = null;
      }
    }
  };

  const cancel = () => {
    if (status.kind !== "running") return;
    setStatus({ ...status, cancelRequested: true });
    // Fire-and-forget: the terminal `cancelled` state arrives via the
    // run promise resolving to `RunResponse::Cancelled`.
    void cancelRun(status.runId).catch(() => {
      /* benign — the run may have finished between click and IPC */
    });
  };

  return { status, setStatus, run, cancel };
}

/** Hand a successful run's export to the viewers: saved into the open
 * project and reopened from disk, or held in memory without a project (or
 * when saving fails, which the project store reports). Returns the saved
 * run's name, if it was saved. */
async function deliver(
  response: RunSuccess,
  setup: { manifest: unknown; config: unknown; manifestDir: string },
): Promise<string | null> {
  const project = useProject.getState().project;
  if (project) {
    const saved = await useProject.getState().saveRun({
      manifest: setup.manifest,
      config: setup.config,
      export: response.export,
      dataset_dir: setup.manifestDir,
      diagnostics: {
        topology: topologyOf(setup.manifest),
        usable_views: response.usable_views,
        total_views: response.total_views,
        duration_ms: response.duration_ms,
        cache_used: response.cache_used,
      },
    });
    if (saved) {
      await useStore.getState().loadExport(runExportPath(project.dir, saved.name));
      return saved.name;
    }
    toast({
      title: "The run was not saved",
      description: "It is shown, but not in the project.",
      tone: "warn",
    });
  }
  useStore.getState().acceptLiveRunExport(response.export, setup.manifestDir);
  return null;
}
