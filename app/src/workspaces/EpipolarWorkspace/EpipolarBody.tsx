import { ErrorBox, Select, ToggleChip } from "@vitavision/ui";
import { invoke } from "@tauri-apps/api/core";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FrameCanvasHandle } from "../../components/FrameCanvas";
import { PoseStepper } from "../../components/PoseStepper";
import { ZoomControls } from "../../components/ZoomControls";
import { useUndistortedImageData } from "../../hooks/useImageData";
import { ZOOM_STEP } from "../../hooks/useZoomShortcuts";
import { relativeCameraPose } from "../../lib/se3";
import { useStore } from "../../store";
import { exportKindLabel } from "../../store/exportKind";
import type { AnyExport, ExportKind } from "../../store/types";
import type { FrameKey, TargetFeatureResidual, ViewportTransform } from "../../types";
// Generated wire type; aliased to avoid colliding with the overlay component.
import type { EpipolarOverlay as EpipolarOverlayResult } from "../../types/generated/diagnose-wire";
import type { OverlayPoint } from "./EpipolarOverlay";
import { EpipolarPane } from "./EpipolarPane";
import {
  distanceToPolyline,
  featureDots,
  ghostDistanceTone,
  snapToFeature,
} from "./epipolarGeometry";
import { RelativePoseStrip } from "./RelativePoseStrip";
import { useUndistortedResiduals } from "./useUndistortedResiduals";

/** The epipolar pair of a rig export: pick a point in pane A, see its
 * epipolar line (and the matching feature) in pane B, both undistorted. */
export function EpipolarBody({ data, kind }: { data: AnyExport; kind: ExportKind }) {
  const frames = useStore((s) => s.frames);
  const cameraValues = useStore((s) => s.cameraValues);
  const camerasByPose = useStore((s) => s.camerasByPose);
  const selectedPose = useStore((s) => s.selectedPose);
  const cameraA = useStore((s) => s.cameraA);
  const cameraB = useStore((s) => s.cameraB);
  const setSelectedPose = useStore((s) => s.setSelectedPose);
  const setCamera = useStore((s) => s.setCamera);

  const [linked, setLinked] = useState(false);
  const [showFeatures, setShowFeatures] = useState(true);
  const [showTieLines, setShowTieLines] = useState(false);
  const [transformA, setTransformA] = useState<ViewportTransform | null>(null);
  const [transformB, setTransformB] = useState<ViewportTransform | null>(null);
  const [linkedTransform, setLinkedTransform] = useState<ViewportTransform | null>(null);
  const [picked, setPicked] = useState<{
    px: [number, number];
    feature: number | null;
  } | null>(null);
  const [overlay, setOverlay] = useState<EpipolarOverlayResult | null>(null);
  const [overlayError, setOverlayError] = useState<string | null>(null);
  const handleARef = useRef<FrameCanvasHandle | null>(null);
  const handleBRef = useRef<FrameCanvasHandle | null>(null);
  const latestRequestIdRef = useRef(0);

  const numPoses = data.rig_se3_target?.length ?? 0;
  const poseIndices = useMemo(
    () => Array.from({ length: numPoses }, (_, i) => i),
    [numPoses],
  );

  const frameA = useMemo<FrameKey | null>(
    () => frames.find((f) => f.pose === selectedPose && f.camera === cameraA) ?? null,
    [frames, selectedPose, cameraA],
  );
  const frameB = useMemo<FrameKey | null>(
    () => frames.find((f) => f.pose === selectedPose && f.camera === cameraB) ?? null,
    [frames, selectedPose, cameraB],
  );

  // Residuals of the active pose, per camera, so the overlay / tie-points /
  // snap logic doesn't re-scan the full residual array.
  const residualsForPose = useMemo<TargetFeatureResidual[]>(
    () =>
      data.diagnostics.per_feature_residuals.target.filter(
        (r) => r.pose === selectedPose,
      ),
    [data, selectedPose],
  );
  const residualsA = useMemo(
    () => residualsForPose.filter((r) => r.camera === cameraA),
    [residualsForPose, cameraA],
  );
  const residualsB = useMemo(
    () => residualsForPose.filter((r) => r.camera === cameraB),
    [residualsForPose, cameraB],
  );
  const undistortedA = useUndistortedResiduals(cameraA, residualsA, setOverlayError);
  const undistortedB = useUndistortedResiduals(cameraB, residualsB, setOverlayError);

  const imageA = useUndistortedImageData(frameA, cameraA, setOverlayError);
  const imageB = useUndistortedImageData(frameB, cameraB, setOverlayError);

  // Reset the overlay whenever the working tuple changes (during render, so
  // the stale overlay never paints), and bump the request id so an
  // in-flight response for the previous tuple is dropped.
  const tuple = `${selectedPose}|${cameraA}|${cameraB}`;
  const [shownTuple, setShownTuple] = useState(tuple);
  if (shownTuple !== tuple) {
    setShownTuple(tuple);
    setPicked(null);
    setOverlay(null);
    setOverlayError(null);
  }
  useEffect(() => {
    latestRequestIdRef.current += 1;
  }, [tuple]);

  const camerasInPose = camerasByPose.get(selectedPose) ?? cameraValues;

  const handlePickA = useCallback(
    async (pixel: { x: number; y: number }) => {
      const snapped = snapToFeature([pixel.x, pixel.y], undistortedA);
      setPicked(snapped);
      setOverlayError(null);
      const request = ++latestRequestIdRef.current;
      if (!imageB) {
        setOverlay(null);
        setOverlayError(
          "epipolar overlay failed: pane-B undistorted image is not ready yet",
        );
        return;
      }
      try {
        const result = await invoke<EpipolarOverlayResult>(
          "compute_epipolar_overlay_undistorted",
          {
            camA: cameraA,
            camB: cameraB,
            pointPx: snapped.px,
            imageWidthB: imageB.naturalWidth,
            imageHeightB: imageB.naturalHeight,
          },
        );
        if (request === latestRequestIdRef.current) setOverlay(result);
      } catch (e) {
        if (request !== latestRequestIdRef.current) return;
        setOverlay(null);
        setOverlayError(`epipolar overlay failed: ${String(e)}`);
      }
    },
    [cameraA, cameraB, imageB, undistortedA],
  );

  const ghostInB = useMemo<TargetFeatureResidual | null>(() => {
    if (!picked || picked.feature == null) return null;
    return undistortedB.find((r) => r.feature === picked.feature) ?? null;
  }, [picked, undistortedB]);

  const clippedPolyline =
    overlay && overlay.line_b.length >= 2 ? overlay.line_b : undefined;

  // The picked feature's epipolar residual: the pane-B feature's distance
  // to the line, shown next to its crosshair.
  const ghostDistancePx = useMemo<number | null>(() => {
    if (!ghostInB || !overlay || overlay.line_b.length < 2) return null;
    return distanceToPolyline(ghostInB.observed_px, overlay.line_b);
  }, [ghostInB, overlay]);

  const relativePose = useMemo(() => {
    const cs = data.cam_se3_rig;
    if (!cs || cs.length <= cameraA || cs.length <= cameraB) return null;
    if (cameraA === cameraB) return null;
    return relativeCameraPose(cs[cameraA]!, cs[cameraB]!);
  }, [data, cameraA, cameraB]);

  const markersA: OverlayPoint[] = [
    ...(showFeatures ? featureDots(undistortedA, 6) : []),
    ...(showTieLines ? featureDots(undistortedA, 4) : []),
    ...(picked ? [{ px: picked.px, color: "var(--signal)", size: 8 }] : []),
  ];
  const markersB: OverlayPoint[] = [
    ...(showFeatures ? featureDots(undistortedB, 6) : []),
    ...(showTieLines ? featureDots(undistortedB, 4) : []),
    ...(ghostInB ? [{ px: ghostInB.observed_px, color: "var(--signal)", size: 8 }] : []),
    ...(overlay?.epipole_b
      ? [{ px: overlay.epipole_b, color: "var(--defect)", size: 7 }]
      : []),
  ];

  // Linked mode shares one transform between both panes; flipping into it
  // forces a fresh fit so the two start aligned.
  useEffect(() => {
    if (linked) handleARef.current?.fit();
  }, [linked]);

  const drivePane = (action: (h: FrameCanvasHandle) => void) => {
    // Linked: either pane drives the shared transform; else pane A.
    const h = linked ? (handleARef.current ?? handleBRef.current) : handleARef.current;
    if (h) action(h);
  };

  return (
    <section className="flex min-h-0 flex-1 flex-col gap-2.5">
      <div className="flex flex-wrap items-center gap-3">
        {numPoses > 0 && (
          <PoseStepper
            poseValues={poseIndices}
            selectedPose={selectedPose}
            onSelectPose={(next) => setSelectedPose(next, "A")}
          />
        )}
        {(["A", "B"] as const).map((which) => (
          <div key={which} className="flex items-center gap-1.5">
            <span className="font-mono text-[11px] tracking-wider text-fg-muted uppercase">
              cam {which}
            </span>
            <Select
              aria-label={`cam ${which}`}
              className="h-7 w-16 font-mono text-xs"
              value={String(which === "A" ? cameraA : cameraB)}
              options={camerasInPose.map((i) => ({ value: String(i), label: String(i) }))}
              onValueChange={(v) => setCamera(Number(v), which)}
            />
          </div>
        ))}
        <ZoomControls
          onFit={() => drivePane((h) => h.fit())}
          onOneToOne={() => drivePane((h) => h.reset1to1())}
          onZoomIn={() => drivePane((h) => h.zoomBy(ZOOM_STEP))}
          onZoomOut={() => drivePane((h) => h.zoomBy(1 / ZOOM_STEP))}
        />
        <ToggleChip
          checked={linked}
          onCheckedChange={setLinked}
          title="Share zoom + pan between both panes (common AOI)"
        >
          Linked
        </ToggleChip>
        <ToggleChip
          checked={showFeatures}
          onCheckedChange={setShowFeatures}
          title="Show every detected feature as a clickable dot"
        >
          Features
        </ToggleChip>
        <ToggleChip
          checked={showTieLines}
          onCheckedChange={setShowTieLines}
          title="Render every observed feature as a faint dot (both panes)"
        >
          Tie-points
        </ToggleChip>
        <span className="ml-auto font-mono text-[11px] text-fg-muted">
          {exportKindLabel(kind)}
        </span>
      </div>

      <RelativePoseStrip
        relativePose={relativePose}
        cameraA={cameraA}
        cameraB={cameraB}
        ghostDistancePx={ghostDistancePx}
        pickedFeature={picked?.feature ?? null}
        samplesClipped={overlay?.samples_clipped ?? null}
      />

      {overlayError && <ErrorBox>{overlayError}</ErrorBox>}

      <div className="grid min-h-0 flex-1 grid-cols-2 gap-2 overflow-hidden rounded-control bg-raised p-2">
        <EpipolarPane
          frame={frameA}
          transform={linked ? linkedTransform : transformA}
          onTransformChange={linked ? setLinkedTransform : setTransformA}
          onPick={(pixel) => void handlePickA(pixel)}
          markers={markersA}
          image={imageA?.image ?? null}
          caption={
            frameA
              ? `pane A · cam ${cameraA} · undistorted${showFeatures ? " · click feature or anywhere" : " · click to pick"}`
              : undefined
          }
          handleRef={handleARef}
        />
        <EpipolarPane
          frame={frameB}
          transform={linked ? linkedTransform : transformB}
          onTransformChange={linked ? setLinkedTransform : setTransformB}
          polyline={clippedPolyline}
          polylineColor="var(--signal)"
          markers={markersB}
          image={imageB?.image ?? null}
          caption={frameB ? `pane B · cam ${cameraB} · undistorted` : undefined}
          handleRef={handleBRef}
          annotation={
            ghostInB && ghostDistancePx != null
              ? {
                  px: ghostInB.observed_px,
                  text: `Δ ${ghostDistancePx.toFixed(2)} px`,
                  color: `var(--${ghostDistanceTone(ghostDistancePx)})`,
                }
              : undefined
          }
        />
      </div>
    </section>
  );
}
