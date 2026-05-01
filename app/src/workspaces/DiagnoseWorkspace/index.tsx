import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CompareViewer,
  type CompareViewerHandle,
} from "../../components/CompareViewer";
import {
  FrameCanvas,
  type FrameCanvasHandle,
  colorForError,
} from "../../components/FrameCanvas";
import { Histogram } from "../../components/Histogram";
import { PoseCameraStepper } from "../../components/PoseCameraStepper";
import {
  getPixelLum,
  rectHistogram,
  useImageData,
} from "../../hooks/useImageData";
import { useKeyboardNav } from "../../hooks/useKeyboardNav";
import { useStore } from "../../store";
import type { CursorReadout, FrameKey, TargetFeatureResidual } from "../../types";

const HISTOGRAM_BINS = 64;

export function DiagnoseWorkspace() {
  // Cross-workspace state from the store.
  const data = useStore((s) => s.data);
  const frames = useStore((s) => s.frames);
  const poseValues = useStore((s) => s.poseValues);
  const cameraValues = useStore((s) => s.cameraValues);
  const selectedPose = useStore((s) => s.selectedPose);
  const selectedPoseB = useStore((s) => s.selectedPoseB);
  const cameraA = useStore((s) => s.cameraA);
  const cameraB = useStore((s) => s.cameraB);
  const stepPose = useStore((s) => s.stepPose);
  const stepCamera = useStore((s) => s.stepCamera);

  // Diagnose-specific UI state stays local — these affordances don't
  // exist in the other workspaces.
  const [compare, setCompare] = useState<boolean>(false);
  const [linked, setLinked] = useState<boolean>(true);
  const [activePane, setActivePane] = useState<"left" | "right">("left");
  const [cursor, setCursor] = useState<CursorReadout | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canvasHandleRef = useRef<FrameCanvasHandle | null>(null);
  const compareHandleRef = useRef<CompareViewerHandle | null>(null);

  const which: "A" | "B" = compare && activePane === "right" ? "B" : "A";

  const frame = useMemo<FrameKey | null>(() => {
    return (
      frames.find((f) => f.pose === selectedPose && f.camera === cameraA) ??
      null
    );
  }, [frames, selectedPose, cameraA]);

  const rightFrame = useMemo<FrameKey | null>(() => {
    return (
      frames.find((f) => f.pose === selectedPoseB && f.camera === cameraB) ??
      null
    );
  }, [frames, selectedPoseB, cameraB]);

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

  const onPoseStep = useCallback(
    (delta: number) => stepPose(delta, which),
    [stepPose, which],
  );
  const onCameraStep = useCallback(
    (delta: number) => stepCamera(delta, which),
    [stepCamera, which],
  );

  useKeyboardNav({
    onPoseStep,
    onCameraStep,
    enabled: data != null,
  });

  // Single-key shortcuts for zoom controls — bound at the window level
  // so they work without canvas focus; ignored over editable elements.
  useEffect(() => {
    if (!data) return;
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
        compare
          ? compareHandleRef.current?.fitActive()
          : canvasHandleRef.current?.fit();
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
  }, [data, compare]);

  return (
    <section className="flex min-h-0 flex-1 flex-col gap-2.5">
      <div className="flex flex-wrap items-center gap-3">
        {data && (
          <PoseCameraStepper
            poseOrdinal={
              poseValues.indexOf(which === "B" ? selectedPoseB : selectedPose) +
              1
            }
            poseTotal={poseValues.length}
            cameraOrdinal={
              cameraValues.indexOf(which === "B" ? cameraB : cameraA) + 1
            }
            cameraTotal={cameraValues.length}
            onPoseStep={onPoseStep}
            onCameraStep={onCameraStep}
          />
        )}
        {data && (
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
        {data && (
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
        {data && compare && (
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
        {data && (
          <span className="ml-auto font-mono text-xs text-muted-foreground">
            mean reproj: {data.mean_reproj_error.toFixed(3)} px
          </span>
        )}
      </div>

      {error && (
        <div className="rounded-md border-l-2 border-destructive bg-destructive/[0.08] p-2.5 text-[13px] text-foreground">
          {error}
        </div>
      )}

      <div className="relative flex min-h-0 flex-1 overflow-hidden rounded-md bg-bg-soft p-2">
        {data && frame && compare && rightFrame ? (
          <CompareViewer
            leftFrame={frame}
            rightFrame={rightFrame}
            residuals={data.per_feature_residuals.target}
            activePane={activePane}
            onActivePane={setActivePane}
            linked={linked}
            onCursorChange={setCursor}
            onError={setError}
            innerRef={compareHandleRef}
          />
        ) : data && frame ? (
          <FrameCanvas
            ref={canvasHandleRef}
            frame={frame}
            residuals={data.per_feature_residuals.target}
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

      {data && frame && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <ResidualLegend
            residuals={data.per_feature_residuals.target.filter(
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
          {cursor.intensity != null ? ` · I=${cursor.intensity}` : ""}
        </span>
      ) : (
        <span>—</span>
      )}
    </span>
  );
}

function ResidualLegend({
  residuals,
}: {
  residuals: TargetFeatureResidual[];
}) {
  const errs = residuals
    .map((r) => r.error_px)
    .filter((e): e is number => typeof e === "number");
  const mean =
    errs.length > 0 ? errs.reduce((a, b) => a + b, 0) / errs.length : 0;
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
