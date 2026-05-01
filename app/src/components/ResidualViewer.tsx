import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { isTauriContext, joinPath } from "../lib/tauri";
import { useKeyboardNav } from "../hooks/useKeyboardNav";
import {
  getPixelLum,
  rectHistogram,
  useImageData,
} from "../hooks/useImageData";
import type {
  CursorReadout,
  FrameKey,
  LoadExportResult,
  PlanarExport,
  TargetFeatureResidual,
} from "../types";
import {
  CompareViewer,
  type CompareViewerHandle,
} from "./CompareViewer";
import {
  FrameCanvas,
  type FrameCanvasHandle,
  colorForError,
} from "./FrameCanvas";
import { Histogram } from "./Histogram";
import { PoseCameraStepper } from "./PoseCameraStepper";

const HISTOGRAM_BINS = 64;

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
  const [poseRight, setPoseRight] = useState<number>(0);
  const [cameraRight, setCameraRight] = useState<number>(1);
  const [compare, setCompare] = useState<boolean>(false);
  const [linked, setLinked] = useState<boolean>(true);
  const [activePane, setActivePane] = useState<"left" | "right">("left");
  const [error, setError] = useState<string | null>(null);
  const [cursor, setCursor] = useState<CursorReadout | null>(null);
  const tauriOk = isTauriContext();
  const canvasHandleRef = useRef<FrameCanvasHandle | null>(null);
  const compareHandleRef = useRef<CompareViewerHandle | null>(null);

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
      setPoseRight(0);
      setCameraRight(Math.min(1, Math.max(numCameras - 1, 0)));
      setCompare(false);
      setActivePane("left");
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

  const rightFrame = useMemo<FrameKey | null>(() => {
    if (!state) return null;
    return (
      state.frames.find(
        (f) => f.pose === poseRight && f.camera === cameraRight,
      ) ?? null
    );
  }, [state, poseRight, cameraRight]);

  const imageData = useImageData(frame, setError);

  // Static ROI histogram — recomputed only when the frame's image data
  // changes, which is the cheap path. (A viewport-tracking variant
  // can come later; for diagnostic use the static distribution is
  // more stable to read.)
  const roiHistogram = useMemo<number[] | null>(() => {
    if (!imageData || !frame) return null;
    const r = frame.roi ?? {
      x: 0,
      y: 0,
      w: imageData.naturalWidth,
      h: imageData.naturalHeight,
    };
    return rectHistogram(imageData, r, HISTOGRAM_BINS);
  }, [imageData, frame]);

  const cursorBin = useMemo<number | null>(() => {
    if (!cursor || cursor.intensity == null) return null;
    const bin = Math.floor((cursor.intensity * HISTOGRAM_BINS) / 256);
    return Math.min(bin, HISTOGRAM_BINS - 1);
  }, [cursor]);

  // Reset cursor whenever the frame changes — the previous cursor's
  // (x, y) no longer maps to the new image.
  useEffect(() => {
    setCursor(null);
  }, [frame?.abs_path]);

  const handleCursor = useCallback(
    (c: { x: number; y: number } | null) => {
      if (!c || !imageData) {
        setCursor(null);
        return;
      }
      const roi = frame?.roi;
      // Translate ROI-local coords back to source-image coords for the
      // luminance lookup; residual frame and pixel buffer differ when
      // the manifest crops out a tile.
      const srcX = (roi?.x ?? 0) + c.x;
      const srcY = (roi?.y ?? 0) + c.y;
      setCursor({
        x: c.x,
        y: c.y,
        intensity: getPixelLum(imageData, srcX, srcY),
      });
    },
    [imageData, frame],
  );

  const stepPose = useCallback(
    (delta: number) => {
      if (!state) return;
      const apply = (p: number) =>
        ((p + delta) % state.numPoses + state.numPoses) % state.numPoses;
      if (compare && activePane === "right") {
        setPoseRight(apply);
      } else {
        setPose(apply);
      }
    },
    [state, compare, activePane],
  );
  const stepCamera = useCallback(
    (delta: number) => {
      if (!state) return;
      const apply = (c: number) =>
        ((c + delta) % state.numCameras + state.numCameras) % state.numCameras;
      if (compare && activePane === "right") {
        setCameraRight(apply);
      } else {
        setCamera(apply);
      }
    },
    [state, compare, activePane],
  );

  useKeyboardNav({
    onPoseStep: stepPose,
    onCameraStep: stepCamera,
    enabled: state != null,
  });

  // Single-key shortcuts for zoom controls. Bound at the window level
  // so they work without needing the canvas to be focused; ignored
  // when an editable element has focus.
  useEffect(() => {
    if (!state) return;
    const handler = (e: KeyboardEvent) => {
      const tgt = e.target as HTMLElement | null;
      if (
        tgt &&
        (tgt.tagName === "INPUT" ||
          tgt.tagName === "SELECT" ||
          tgt.tagName === "TEXTAREA" ||
          tgt.isContentEditable)
      ) {
        return;
      }
      const fit = () =>
        compare ? compareHandleRef.current?.fitActive() : canvasHandleRef.current?.fit();
      const oneToOne = () =>
        compare
          ? compareHandleRef.current?.reset1to1Active()
          : canvasHandleRef.current?.reset1to1();
      const zoom = (factor: number) =>
        compare
          ? compareHandleRef.current?.zoomActiveBy(factor)
          : canvasHandleRef.current?.zoomBy(factor);
      switch (e.key) {
        case "f":
        case "F":
          fit();
          e.preventDefault();
          break;
        case "1":
          oneToOne();
          e.preventDefault();
          break;
        case "+":
        case "=":
          zoom(1.25);
          e.preventDefault();
          break;
        case "-":
        case "_":
          zoom(1 / 1.25);
          e.preventDefault();
          break;
        case "l":
        case "L":
          if (compare) {
            setLinked((v) => !v);
            e.preventDefault();
          }
          break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [state, compare]);

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
            pose={compare && activePane === "right" ? poseRight : pose}
            camera={compare && activePane === "right" ? cameraRight : camera}
            numPoses={state.numPoses}
            numCameras={state.numCameras}
            onPose={(next) =>
              compare && activePane === "right" ? setPoseRight(next) : setPose(next)
            }
            onCamera={(next) =>
              compare && activePane === "right"
                ? setCameraRight(next)
                : setCamera(next)
            }
          />
        )}
        {state && (
          <ZoomControls
            onFit={() =>
              compare
                ? compareHandleRef.current?.fitActive()
                : canvasHandleRef.current?.fit()
            }
            onOneToOne={() =>
              compare
                ? compareHandleRef.current?.reset1to1Active()
                : canvasHandleRef.current?.reset1to1()
            }
            onZoomIn={() =>
              compare
                ? compareHandleRef.current?.zoomActiveBy(1.25)
                : canvasHandleRef.current?.zoomBy(1.25)
            }
            onZoomOut={() =>
              compare
                ? compareHandleRef.current?.zoomActiveBy(1 / 1.25)
                : canvasHandleRef.current?.zoomBy(1 / 1.25)
            }
          />
        )}
        {state && (
          <button
            type="button"
            onClick={() => setCompare((v) => !v)}
            className={`h-7 px-2 font-mono text-[11px] ${
              compare ? "border-brand text-brand" : ""
            }`}
            title="Toggle compare mode"
          >
            {compare ? "Compare ✓" : "Compare"}
          </button>
        )}
        {state && compare && (
          <button
            type="button"
            onClick={() => setLinked((v) => !v)}
            className={`h-7 px-2 font-mono text-[11px] ${
              linked ? "border-brand text-brand" : ""
            }`}
            title="Toggle linked viewport (L)"
          >
            {linked ? "Linked" : "Unlinked"}
          </button>
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

      <div className="relative flex min-h-0 flex-1 overflow-hidden rounded-md bg-bg-soft p-2">
        {state && frame && compare && rightFrame ? (
          <CompareViewer
            leftFrame={frame}
            rightFrame={rightFrame}
            residuals={state.data.per_feature_residuals.target}
            activePane={activePane}
            onActivePane={setActivePane}
            linked={linked}
            onCursorChange={setCursor}
            onError={setError}
            innerRef={compareHandleRef}
          />
        ) : state && frame ? (
          <FrameCanvas
            ref={canvasHandleRef}
            frame={frame}
            residuals={state.data.per_feature_residuals.target}
            image={imageData?.image ?? null}
            onCursor={handleCursor}
            onError={setError}
          />
        ) : (
          <div className="m-auto text-[13px] text-muted-foreground">
            Open an <code>export.json</code> from a calibration run with an
            <code> image_manifest</code>. Use ← / → for pose, ↑ / ↓ for
            camera; toggle Compare to view two frames side by side.
          </div>
        )}
      </div>

      {state && frame && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <ResidualLegend
            residuals={state.data.per_feature_residuals.target.filter(
              (r) => r.pose === frame.pose && r.camera === frame.camera,
            )}
          />
          {roiHistogram && (
            <div className="flex items-center gap-2">
              <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                histogram
              </span>
              <Histogram bins={roiHistogram} cursorBin={cursorBin} />
            </div>
          )}
          <CursorChip cursor={cursor} />
        </div>
      )}
    </section>
  );
}

function CursorChip({ cursor }: { cursor: CursorReadout | null }) {
  return (
    <span className="inline-flex min-w-[12rem] items-center gap-2 rounded-md border border-border bg-surface px-2 py-1 font-mono text-[11px] text-muted-foreground">
      <span className="uppercase tracking-wider">cursor</span>
      {cursor ? (
        <span className="text-foreground tabular-nums">
          ({cursor.x.toFixed(0)}, {cursor.y.toFixed(0)})
          {cursor.intensity != null
            ? ` · I=${cursor.intensity}`
            : ""}
        </span>
      ) : (
        <span>—</span>
      )}
    </span>
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

interface ZoomControlsProps {
  onFit: () => void;
  onOneToOne: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
}

function ZoomControls({
  onFit,
  onOneToOne,
  onZoomIn,
  onZoomOut,
}: ZoomControlsProps) {
  return (
    <div className="flex items-center gap-1">
      <button
        type="button"
        onClick={onZoomOut}
        title="Zoom out (−)"
        aria-label="Zoom out"
        className="grid h-7 w-7 place-items-center !p-0 font-mono text-xs"
      >
        −
      </button>
      <button
        type="button"
        onClick={onZoomIn}
        title="Zoom in (+)"
        aria-label="Zoom in"
        className="grid h-7 w-7 place-items-center !p-0 font-mono text-xs"
      >
        +
      </button>
      <button
        type="button"
        onClick={onFit}
        title="Fit (f)"
        className="h-7 px-2 font-mono text-[11px]"
      >
        Fit
      </button>
      <button
        type="button"
        onClick={onOneToOne}
        title="1:1 (1)"
        className="h-7 px-2 font-mono text-[11px]"
      >
        1:1
      </button>
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
