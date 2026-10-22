import { DensityProvider, ErrorBox, ToggleChip } from "@vitavision/ui";
import { useCallback, useMemo, useRef, useState } from "react";
import { CompareViewer, type CompareViewerHandle } from "../../components/CompareViewer";
import {
  FrameCanvas,
  type FrameCanvasHandle,
  type LaserOverlayData,
} from "../../components/FrameCanvas";
import { PoseCameraStepper } from "../../components/PoseCameraStepper";
import { ZoomControls } from "../../components/ZoomControls";
import { useImageData } from "../../hooks/useImageData";
import { useKeyboardNav } from "../../hooks/useKeyboardNav";
import {
  useZoomShortcuts,
  ZOOM_STEP,
  type ZoomCommands,
} from "../../hooks/useZoomShortcuts";
import { getPixelLum, rectHistogram } from "../../lib/imagePixels";
import { exportUncertainty } from "../../lib/uncertainty";
import { useStore } from "../../store";
import type {
  CursorReadout,
  FrameKey,
  LaserFeatureResidual,
  TargetFeatureResidual,
} from "../../types";
import { ResidualStatsPanels } from "./ResidualStatsPanels";
import { CursorChip, LaserResidualLegend, ResidualLegend } from "./ResidualLegends";
import { RoiHistogram } from "./RoiHistogram";
import { UncertaintyPanels } from "./UncertaintyPanels";

const HISTOGRAM_BINS = 64;

/** Stable empty array so laser view doesn't retrigger the canvas draw
 * effect with a fresh `[]` identity on every render. */
const NO_TARGET_RESIDUALS: TargetFeatureResidual[] = [];

export function DiagnoseWorkspace() {
  // Cross-workspace state from the store.
  const data = useStore((s) => s.data);
  const frames = useStore((s) => s.frames);
  const laserFrames = useStore((s) => s.laserFrames);
  const poseValues = useStore((s) => s.poseValues);
  const cameraValues = useStore((s) => s.cameraValues);
  const selectedPose = useStore((s) => s.selectedPose);
  const selectedPoseB = useStore((s) => s.selectedPoseB);
  const cameraA = useStore((s) => s.cameraA);
  const cameraB = useStore((s) => s.cameraB);
  const stepPose = useStore((s) => s.stepPose);
  const stepCamera = useStore((s) => s.stepCamera);
  const setSelectedPose = useStore((s) => s.setSelectedPose);
  const setCamera = useStore((s) => s.setCamera);

  // Diagnose-specific UI state stays local — these affordances don't
  // exist in the other workspaces.
  const [compare, setCompare] = useState<boolean>(false);
  const [linked, setLinked] = useState<boolean>(true);
  // Laser view plots per_feature_residuals.laser onto the laser-kind
  // frame for the active (pose, camera). Mutually exclusive with
  // compare mode — both repurpose the single canvas area.
  const [laserView, setLaserView] = useState<boolean>(false);
  // Residual stats side panel: per-pose table + cross-camera matrix.
  const [showStats, setShowStats] = useState<boolean>(false);
  // Parameter uncertainty side panel: value ± σ, correlations, projection σ.
  const [showUncertainty, setShowUncertainty] = useState<boolean>(false);
  const [activePane, setActivePane] = useState<"left" | "right">("left");
  const [cursor, setCursor] = useState<CursorReadout | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canvasHandleRef = useRef<FrameCanvasHandle | null>(null);
  const compareHandleRef = useRef<CompareViewerHandle | null>(null);

  const which: "A" | "B" = compare && activePane === "right" ? "B" : "A";

  const frame = useMemo<FrameKey | null>(() => {
    return frames.find((f) => f.pose === selectedPose && f.camera === cameraA) ?? null;
  }, [frames, selectedPose, cameraA]);

  const rightFrame = useMemo<FrameKey | null>(() => {
    return frames.find((f) => f.pose === selectedPoseB && f.camera === cameraB) ?? null;
  }, [frames, selectedPoseB, cameraB]);

  const laserResiduals = useMemo<LaserFeatureResidual[]>(
    () => data?.per_feature_residuals.laser ?? [],
    [data],
  );
  const laserOverlay = useMemo<LaserOverlayData>(
    () => ({
      residuals: laserResiduals,
      lines: data?.per_feature_residuals.laser_lines ?? [],
    }),
    [data, laserResiduals],
  );
  const hasLaser = laserFrames.length > 0 || laserResiduals.length > 0;

  const targetResiduals = useMemo<TargetFeatureResidual[]>(
    () => data?.per_feature_residuals.target ?? NO_TARGET_RESIDUALS,
    [data],
  );
  const hasStats = targetResiduals.length > 0;

  // A non-target export (or one loaded while stats are open) must not
  // strand the workspace on an empty panel. Adjusting state during render
  // (rather than in an effect) avoids painting the stale panel first.
  if (!hasStats && showStats) setShowStats(false);

  // Loading a non-laser export while laser view is active would strand
  // the workspace on a mode with no data — drop back to target view.
  if (!hasLaser && laserView) setLaserView(false);

  const laserFrame = useMemo<FrameKey | null>(() => {
    return (
      laserFrames.find((f) => f.pose === selectedPose && f.camera === cameraA) ?? null
    );
  }, [laserFrames, selectedPose, cameraA]);

  // The frame the canvas actually shows: the laser frame in laser view
  // (when one exists for the slot), the target frame otherwise.
  const showLaser = laserView && !compare;
  const activeFrame = showLaser ? laserFrame : frame;

  const imageData = useImageData(activeFrame, setError);

  const uncertainty = useMemo(() => (data ? exportUncertainty(data) : null), [data]);
  // The camera's own image: its ROI in a tiled frame, else the whole image.
  const imageSize = useMemo<[number, number] | null>(() => {
    if (!imageData || !activeFrame) return null;
    const roi = activeFrame.roi;
    return roi ? [roi.w, roi.h] : [imageData.naturalWidth, imageData.naturalHeight];
  }, [imageData, activeFrame]);

  // Static ROI histogram — recomputed only when the frame's image data
  // changes, which is the cheap path. (A viewport-tracking variant
  // can come later; for diagnostic use the static distribution is
  // more stable to read.)
  const roiHistogram = useMemo<number[] | null>(() => {
    if (!imageData || !activeFrame) return null;
    const r = activeFrame.roi ?? {
      x: 0,
      y: 0,
      w: imageData.naturalWidth,
      h: imageData.naturalHeight,
    };
    return rectHistogram(imageData, r, HISTOGRAM_BINS);
  }, [imageData, activeFrame]);

  // Reset cursor whenever the frame changes — the previous cursor's
  // (x, y) no longer maps to the new image.
  const activePath = activeFrame?.abs_path;
  const [cursorPath, setCursorPath] = useState(activePath);
  if (cursorPath !== activePath) {
    setCursorPath(activePath);
    setCursor(null);
  }

  const handleCursor = useCallback(
    (c: { x: number; y: number } | null) => {
      if (!c || !imageData) {
        setCursor(null);
        return;
      }
      const roi = activeFrame?.roi;
      const srcX = (roi?.x ?? 0) + c.x;
      const srcY = (roi?.y ?? 0) + c.y;
      setCursor({
        x: c.x,
        y: c.y,
        intensity: getPixelLum(imageData, Math.round(srcX), Math.round(srcY)),
      });
    },
    [imageData, activeFrame],
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

  // Zoom acts on the active compare pane, or on the single frame view.
  const zoom: ZoomCommands = {
    fit: () =>
      compare ? compareHandleRef.current?.fitActive() : canvasHandleRef.current?.fit(),
    oneToOne: () =>
      compare
        ? compareHandleRef.current?.reset1to1Active()
        : canvasHandleRef.current?.reset1to1(),
    zoomBy: (factor) =>
      compare
        ? compareHandleRef.current?.zoomActiveBy(factor)
        : canvasHandleRef.current?.zoomBy(factor),
  };
  useZoomShortcuts(zoom, {
    enabled: data != null,
    onLink: compare ? () => setLinked((v) => !v) : undefined,
  });

  return (
    <section className="flex min-h-0 flex-1 flex-col gap-2.5">
      <div className="flex flex-wrap items-center gap-3">
        {data && (
          <PoseCameraStepper
            poseValues={poseValues}
            selectedPose={which === "B" ? selectedPoseB : selectedPose}
            onSelectPose={(next) => setSelectedPose(next, which)}
            cameraValues={cameraValues}
            selectedCamera={which === "B" ? cameraB : cameraA}
            onSelectCamera={(next) => setCamera(next, which)}
          />
        )}
        {data && (
          <ZoomControls
            hints
            onFit={zoom.fit}
            onOneToOne={zoom.oneToOne}
            onZoomIn={() => zoom.zoomBy(ZOOM_STEP)}
            onZoomOut={() => zoom.zoomBy(1 / ZOOM_STEP)}
          />
        )}
        {data && (
          <ToggleChip
            checked={compare}
            onCheckedChange={(next) => {
              if (next) setLaserView(false);
              setCompare(next);
            }}
            title="Toggle compare mode"
          >
            Compare
          </ToggleChip>
        )}
        {data && hasLaser && (
          <ToggleChip
            checked={laserView}
            onCheckedChange={(next) => {
              if (next) setCompare(false);
              setLaserView(next);
            }}
            title="Show the laser frame with point-to-plane residuals"
          >
            Laser
          </ToggleChip>
        )}
        {data && compare && (
          <ToggleChip
            checked={linked}
            onCheckedChange={setLinked}
            title="Toggle linked viewport (L)"
          >
            Linked
          </ToggleChip>
        )}
        {data && hasStats && (
          <ToggleChip
            checked={showStats}
            onCheckedChange={setShowStats}
            title="Per-pose residual stats + cross-camera matrix"
          >
            Stats
          </ToggleChip>
        )}
        {data && (
          <ToggleChip
            checked={showUncertainty}
            onCheckedChange={setShowUncertainty}
            title="Parameter values ± σ, correlations and projection σ"
          >
            Uncertainty
          </ToggleChip>
        )}
        {data && typeof data.mean_reproj_error === "number" && (
          <span className="ml-auto font-mono text-xs text-fg-muted">
            mean reproj: {data.mean_reproj_error.toFixed(3)} px
          </span>
        )}
      </div>

      {error && <ErrorBox>{error}</ErrorBox>}

      <div className="flex min-h-0 flex-1 gap-2.5 overflow-hidden">
        <div className="relative flex min-h-0 flex-1 overflow-hidden rounded-control bg-raised p-2">
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
          ) : data && activeFrame ? (
            <FrameCanvas
              ref={canvasHandleRef}
              frame={activeFrame}
              residuals={
                showLaser ? NO_TARGET_RESIDUALS : data.per_feature_residuals.target
              }
              laser={showLaser ? laserOverlay : undefined}
              image={imageData?.image ?? null}
              onCursor={handleCursor}
              onError={setError}
            />
          ) : data && showLaser ? (
            <div className="m-auto text-[13px] text-fg-muted">
              No laser frame in the manifest for pose {selectedPose} · cam {cameraA}.
            </div>
          ) : (
            <div className="m-auto text-[13px] text-fg-muted">
              Open an <code>export.json</code> from a calibration run with an
              <code> image_manifest</code>. Use ← / → for pose, ↑ / ↓ for camera; toggle
              Compare to view two frames side by side.
            </div>
          )}
        </div>

        {data && ((showStats && hasStats) || showUncertainty) && (
          <DensityProvider value="compact">
            <aside className="flex min-h-0 w-[19rem] shrink-0 flex-col gap-2.5 overflow-y-auto">
              {showUncertainty && (
                <UncertaintyPanels
                  uncertainty={uncertainty}
                  camera={cameraA}
                  imageSize={imageSize}
                />
              )}
              {showStats && hasStats && (
                <ResidualStatsPanels
                  residuals={targetResiduals}
                  selectedPose={selectedPose}
                  selectedCamera={cameraA}
                  onSelectPose={(pose) => setSelectedPose(pose, "A")}
                  onSelectFrame={(pose, camera) => {
                    setSelectedPose(pose, "A");
                    setCamera(camera, "A");
                  }}
                />
              )}
            </aside>
          </DensityProvider>
        )}
      </div>

      {data && activeFrame && (
        // Reserve a stable height so the panel doesn't shimmy when the
        // histogram briefly drops out during a frame switch (imageData
        // unloads → roiHistogram becomes null until the next decode).
        // The histogram block always mounts; its bins go empty during
        // the gap and re-fill once decode lands.
        <div className="flex min-h-[3.25rem] flex-wrap items-center gap-x-4 gap-y-2">
          {showLaser ? (
            <LaserResidualLegend
              residuals={laserResiduals.filter(
                (r) => r.pose === activeFrame.pose && r.camera === activeFrame.camera,
              )}
              hist={data.per_feature_residuals.laser_hist_per_camera?.[cameraA]}
            />
          ) : (
            <ResidualLegend
              residuals={data.per_feature_residuals.target.filter(
                (r) => r.pose === activeFrame.pose && r.camera === activeFrame.camera,
              )}
            />
          )}
          <div className="flex items-center gap-2">
            <span className="font-mono text-[10px] uppercase tracking-wider text-fg-muted">
              histogram
            </span>
            <RoiHistogram
              counts={roiHistogram ?? []}
              intensity={cursor?.intensity ?? null}
            />
          </div>
          <CursorChip cursor={cursor} />
        </div>
      )}
    </section>
  );
}
