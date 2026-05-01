import { useCallback, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { isTauriContext, joinPath } from "../lib/tauri";
import { useKeyboardNav } from "../hooks/useKeyboardNav";
import type {
  FrameKey,
  LoadExportResult,
  PlanarExport,
  TargetFeatureResidual,
} from "../types";
import { FrameCanvas, colorForError } from "./FrameCanvas";
import { PoseCameraStepper } from "./PoseCameraStepper";

interface ExportState {
  exportPath: string;
  exportDir: string;
  data: PlanarExport;
  frames: FrameKey[];
  /** `pose` values run [0, numPoses); `camera` values run [0, numCameras). */
  numPoses: number;
  numCameras: number;
}

export function ResidualViewer() {
  const [state, setState] = useState<ExportState | null>(null);
  const [pose, setPose] = useState<number>(0);
  const [camera, setCamera] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const tauriOk = isTauriContext();

  const handleOpen = async () => {
    setError(null);
    if (!isTauriContext()) {
      setError(
        "Tauri runtime not detected. Launch the app with `bun run tauri dev` " +
          "(or the bundled binary). Plain `bun run dev` only starts Vite, so " +
          "the file-dialog and IPC commands aren't wired up.",
      );
      return;
    }
    let chosen: string | null = null;
    try {
      chosen = await openDialog({
        multiple: false,
        directory: false,
        filters: [{ name: "Calibration export", extensions: ["json"] }],
      });
    } catch (e) {
      setError(`File dialog error: ${e}`);
      return;
    }
    if (!chosen) return;

    try {
      const result = await invoke<LoadExportResult>("load_export", {
        path: chosen,
      });
      const manifest = result.export.image_manifest;
      if (!manifest) {
        setError(
          "This export has no image_manifest field. v0 requires the manifest " +
            "to render the source images alongside the residuals.",
        );
        return;
      }
      const root = joinPath(result.export_dir, manifest.root);
      const frames: FrameKey[] = manifest.frames.map((f) => ({
        pose: f.pose,
        camera: f.camera,
        label: `pose ${f.pose} · cam ${f.camera}`,
        abs_path: joinPath(root, f.path),
        roi: f.roi,
      }));
      const numPoses =
        frames.reduce((m, f) => Math.max(m, f.pose), -1) + 1;
      const numCameras =
        frames.reduce((m, f) => Math.max(m, f.camera), -1) + 1;
      setState({
        exportPath: chosen,
        exportDir: result.export_dir,
        data: result.export,
        frames,
        numPoses: Math.max(numPoses, 1),
        numCameras: Math.max(numCameras, 1),
      });
      setPose(0);
      setCamera(0);
    } catch (e) {
      setError(`Could not load export: ${e}`);
    }
  };

  const frame = useMemo<FrameKey | null>(() => {
    if (!state) return null;
    return (
      state.frames.find((f) => f.pose === pose && f.camera === camera) ?? null
    );
  }, [state, pose, camera]);

  const stepPose = useCallback(
    (delta: number) => {
      if (!state) return;
      setPose((p) => ((p + delta) % state.numPoses + state.numPoses) % state.numPoses);
    },
    [state],
  );
  const stepCamera = useCallback(
    (delta: number) => {
      if (!state) return;
      setCamera(
        (c) => ((c + delta) % state.numCameras + state.numCameras) % state.numCameras,
      );
    },
    [state],
  );

  useKeyboardNav({
    onPoseStep: stepPose,
    onCameraStep: stepCamera,
    enabled: state != null,
  });

  return (
    <section className="flex min-h-0 flex-1 flex-col gap-2.5">
      {!tauriOk && (
        <div className="rounded-md border-l-2 border-brand bg-brand/[0.06] p-2.5 text-[13px] text-foreground">
          Tauri runtime not detected — you appear to be running plain Vite
          (<code>bun run dev</code>) in a browser. Launch the desktop app
          with <code>bun run tauri dev</code> to use the file dialog.
        </div>
      )}
      <div className="flex flex-wrap items-center gap-3">
        <button onClick={handleOpen} disabled={!tauriOk}>
          Open Export…
        </button>
        {state && (
          <PoseCameraStepper
            pose={pose}
            camera={camera}
            numPoses={state.numPoses}
            numCameras={state.numCameras}
            onPose={setPose}
            onCamera={setCamera}
          />
        )}
        {state && (
          <span className="ml-auto font-mono text-xs text-muted-foreground">
            mean reproj: {state.data.mean_reproj_error.toFixed(3)} px
          </span>
        )}
      </div>

      {error && (
        <div className="rounded-md border-l-2 border-destructive bg-destructive/[0.08] p-2.5 text-[13px] text-foreground">
          {error}
        </div>
      )}

      <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto rounded-md bg-bg-soft">
        {state && frame ? (
          <FrameCanvas
            frame={frame}
            residuals={state.data.per_feature_residuals.target}
            onError={setError}
          />
        ) : (
          <div className="text-[13px] text-muted-foreground">
            Open an <code>export.json</code> from a calibration run with an
            <code> image_manifest</code>. Use ← / → for pose, ↑ / ↓ for
            camera once an export is loaded.
          </div>
        )}
      </div>

      {state && frame && (
        <ResidualLegend
          residuals={state.data.per_feature_residuals.target.filter(
            (r) => r.pose === frame.pose && r.camera === frame.camera,
          )}
        />
      )}
    </section>
  );
}

function ResidualLegend({ residuals }: { residuals: TargetFeatureResidual[] }) {
  const errs = residuals
    .map((r) => r.error_px)
    .filter((e): e is number => typeof e === "number");
  const mean = errs.length > 0 ? errs.reduce((a, b) => a + b, 0) / errs.length : 0;
  const max = errs.length > 0 ? Math.max(...errs) : 0;
  const diverged = residuals.length - errs.length;
  return (
    <div className="flex flex-wrap items-center gap-x-3.5 gap-y-1 font-mono text-[11px] text-muted-foreground">
      <span>features {residuals.length}</span>
      <span>diverged {diverged}</span>
      <span>mean {mean.toFixed(3)} px</span>
      <span>max {max.toFixed(3)} px</span>
      <span className="flex items-center gap-2">
        <Swatch err={0.5} label="<1" />
        <Swatch err={1.5} label="<2" />
        <Swatch err={3} label="<5" />
        <Swatch err={7} label="<10" />
        <Swatch err={20} label="≥10" />
      </span>
    </div>
  );
}

function Swatch({ err, label }: { err: number; label: string }) {
  // Color is data-driven (matches the canvas error ramp), so it stays
  // inline; the dot is a Tailwind utility.
  return (
    <span className="inline-flex items-center gap-1">
      <span
        className="inline-block h-2 w-2 rounded-[2px]"
        style={{ background: colorForError(err) }}
      />
      {label}
    </span>
  );
}
