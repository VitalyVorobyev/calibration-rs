import { invoke } from "@tauri-apps/api/core";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  FrameCanvas,
  type FrameCanvasHandle,
} from "../../components/FrameCanvas";
import { PoseStepper } from "../../components/PoseStepper";
import { useImageData } from "../../hooks/useImageData";
import {
  iso3DistanceM,
  iso3EulerXYZDeg,
  iso3RotationAngleDeg,
  relativeCameraPose,
} from "../../lib/se3";
import { useStore } from "../../store";
import { exportKindLabel } from "../../store/exportShape";
import type {
  FrameKey,
  TargetFeatureResidual,
  ViewportTransform,
} from "../../types";
import { IDENTITY_TRANSFORM } from "../../types";
import {
  EpipolarOverlay,
  type OverlayPoint,
} from "./EpipolarOverlay";

interface EpipolarOverlayResult {
  line_b: [number, number][];
  epipole_b: [number, number] | null;
  samples_clipped: number;
}

/** Pixel-radius around the click within which we snap to the nearest
 * residual feature. Outside this radius the click is a free pixel
 * pick (no corresponding pane-B ghost is drawn). */
const FEATURE_SNAP_PX = 8;

export function EpipolarWorkspace() {
  const data = useStore((s) => s.data);
  const kind = useStore((s) => s.kind);
  const frames = useStore((s) => s.frames);
  const cameraValues = useStore((s) => s.cameraValues);
  const camerasByPose = useStore((s) => s.camerasByPose);
  const selectedPose = useStore((s) => s.selectedPose);
  const cameraA = useStore((s) => s.cameraA);
  const cameraB = useStore((s) => s.cameraB);
  const setSelectedPose = useStore((s) => s.setSelectedPose);
  const setCamera = useStore((s) => s.setCamera);

  const [linked, setLinked] = useState<boolean>(false);
  const [showFeatures, setShowFeatures] = useState<boolean>(true);
  const [showTieLines, setShowTieLines] = useState<boolean>(false);

  const [transformA, setTransformA] = useState<ViewportTransform>(IDENTITY_TRANSFORM);
  const [transformB, setTransformB] = useState<ViewportTransform>(IDENTITY_TRANSFORM);
  const [linkedTransform, setLinkedTransform] = useState<ViewportTransform>(
    IDENTITY_TRANSFORM,
  );

  const [picked, setPicked] = useState<{
    px: [number, number];
    feature: number | null;
  } | null>(null);
  const [overlay, setOverlay] = useState<EpipolarOverlayResult | null>(null);
  const [overlayError, setOverlayError] = useState<string | null>(null);

  const handleARef = useRef<FrameCanvasHandle | null>(null);
  const handleBRef = useRef<FrameCanvasHandle | null>(null);

  if (!data || !kind) {
    return (
      <Empty body="Load a rig export to inspect epipolar geometry between cameras." />
    );
  }

  const isRig =
    Array.isArray(data.cameras) &&
    Array.isArray(data.cam_se3_rig) &&
    data.cameras.length >= 2;
  if (!isRig) {
    return (
      <Empty
        body={`Epipolar geometry only applies to rig exports (rig_extrinsics, rig_handeye, rig_laserline_device). The current export is ${exportKindLabel(
          kind,
        )}.`}
      />
    );
  }

  return (
    <EpipolarBody
      data={data}
      kind={kind}
      frames={frames}
      cameraValues={cameraValues}
      camerasByPose={camerasByPose}
      selectedPose={selectedPose}
      cameraA={cameraA}
      cameraB={cameraB}
      setSelectedPose={setSelectedPose}
      setCamera={setCamera}
      linked={linked}
      setLinked={setLinked}
      showFeatures={showFeatures}
      setShowFeatures={setShowFeatures}
      showTieLines={showTieLines}
      setShowTieLines={setShowTieLines}
      transformA={transformA}
      transformB={transformB}
      linkedTransform={linkedTransform}
      setTransformA={setTransformA}
      setTransformB={setTransformB}
      setLinkedTransform={setLinkedTransform}
      picked={picked}
      setPicked={setPicked}
      overlay={overlay}
      setOverlay={setOverlay}
      overlayError={overlayError}
      setOverlayError={setOverlayError}
      handleARef={handleARef}
      handleBRef={handleBRef}
    />
  );
}

interface BodyProps {
  data: NonNullable<ReturnType<typeof useStore.getState>["data"]>;
  kind: NonNullable<ReturnType<typeof useStore.getState>["kind"]>;
  frames: FrameKey[];
  cameraValues: number[];
  camerasByPose: Map<number, number[]>;
  selectedPose: number;
  cameraA: number;
  cameraB: number;
  setSelectedPose: (pose: number, which?: "A" | "B") => void;
  setCamera: (camera: number, which?: "A" | "B") => void;
  linked: boolean;
  setLinked: (v: boolean | ((prev: boolean) => boolean)) => void;
  showFeatures: boolean;
  setShowFeatures: (v: boolean | ((prev: boolean) => boolean)) => void;
  showTieLines: boolean;
  setShowTieLines: (v: boolean | ((prev: boolean) => boolean)) => void;
  transformA: ViewportTransform;
  transformB: ViewportTransform;
  linkedTransform: ViewportTransform;
  setTransformA: (t: ViewportTransform) => void;
  setTransformB: (t: ViewportTransform) => void;
  setLinkedTransform: (t: ViewportTransform) => void;
  picked: { px: [number, number]; feature: number | null } | null;
  setPicked: (p: { px: [number, number]; feature: number | null } | null) => void;
  overlay: EpipolarOverlayResult | null;
  setOverlay: (o: EpipolarOverlayResult | null) => void;
  overlayError: string | null;
  setOverlayError: (msg: string | null) => void;
  handleARef: React.MutableRefObject<FrameCanvasHandle | null>;
  handleBRef: React.MutableRefObject<FrameCanvasHandle | null>;
}

function EpipolarBody(props: BodyProps) {
  const {
    data,
    kind,
    frames,
    cameraValues,
    camerasByPose,
    selectedPose,
    cameraA,
    cameraB,
    setSelectedPose,
    setCamera,
    linked,
    setLinked,
    showFeatures,
    setShowFeatures,
    showTieLines,
    setShowTieLines,
    transformA,
    transformB,
    linkedTransform,
    setTransformA,
    setTransformB,
    setLinkedTransform,
    picked,
    setPicked,
    overlay,
    setOverlay,
    overlayError,
    setOverlayError,
    handleARef,
    handleBRef,
  } = props;

  const numPoses = data.rig_se3_target?.length ?? 0;
  const poseIndices = useMemo(
    () => Array.from({ length: numPoses }, (_, i) => i),
    [numPoses],
  );

  const frameA = useMemo<FrameKey | null>(
    () =>
      frames.find((f) => f.pose === selectedPose && f.camera === cameraA) ??
      null,
    [frames, selectedPose, cameraA],
  );
  const frameB = useMemo<FrameKey | null>(
    () =>
      frames.find((f) => f.pose === selectedPose && f.camera === cameraB) ??
      null,
    [frames, selectedPose, cameraB],
  );

  // Bucket residuals by (camera) for the active pose so the overlay /
  // tie-points / snap logic doesn't re-scan the full residual array.
  const residualsForPose = useMemo<TargetFeatureResidual[]>(
    () =>
      data.per_feature_residuals.target.filter((r) => r.pose === selectedPose),
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

  const latestRequestIdRef = useRef(0);

  // Reset the overlay whenever the working tuple changes; bump the
  // request id so any in-flight response from the previous tuple is
  // dropped on arrival.
  useEffect(() => {
    latestRequestIdRef.current += 1;
    setPicked(null);
    setOverlay(null);
    setOverlayError(null);
  }, [selectedPose, cameraA, cameraB, setPicked, setOverlay, setOverlayError]);

  const camerasInPose = camerasByPose.get(selectedPose) ?? cameraValues;

  const handlePickA = useCallback(
    async (pixel: { x: number; y: number }) => {
      const px: [number, number] = [pixel.x, pixel.y];
      let feature: number | null = null;
      let bestDist = FEATURE_SNAP_PX;
      let snappedPx: [number, number] = px;
      for (const r of residualsA) {
        const dx = r.observed_px[0] - px[0];
        const dy = r.observed_px[1] - px[1];
        const d = Math.hypot(dx, dy);
        if (d < bestDist) {
          bestDist = d;
          feature = r.feature;
          snappedPx = [r.observed_px[0], r.observed_px[1]];
        }
      }
      setPicked({ px: snappedPx, feature });
      setOverlayError(null);
      latestRequestIdRef.current += 1;
      const myRequestId = latestRequestIdRef.current;
      try {
        const result = await invoke<EpipolarOverlayResult>(
          "compute_epipolar_overlay",
          { camA: cameraA, camB: cameraB, pointPx: snappedPx },
        );
        if (myRequestId !== latestRequestIdRef.current) return;
        setOverlay(result);
      } catch (e) {
        if (myRequestId !== latestRequestIdRef.current) return;
        setOverlay(null);
        setOverlayError(`epipolar overlay failed: ${e}`);
      }
    },
    [cameraA, cameraB, residualsA, setOverlay, setOverlayError, setPicked],
  );

  const ghostInB = useMemo<TargetFeatureResidual | null>(() => {
    if (!picked || picked.feature == null) return null;
    return residualsB.find((r) => r.feature === picked.feature) ?? null;
  }, [picked, residualsB]);

  // Polyline clipped strictly to pane-B's image bounds + split on
  // unphysically large jumps. The Rust side sweeps depths from 0.05 m
  // to 5 m, which is wider than any real dataset's working volume —
  // most samples land outside the image and pull the polyline into a
  // long curve (distortion gets visually amplified far from the
  // principal point). On top of that, when cam A's ray approaches
  // cam B's optical axis the projection passes through a near-
  // singular fold zone and consecutive samples can land hundreds of
  // pixels apart on opposite sides of the epipolar line — visually a
  // "parabola" / fork. We keep only the longest contiguous in-image
  // run AND require that consecutive surviving samples are close
  // enough that the line is physically plausible.
  const clippedPolyline = useMemo<[number, number][] | undefined>(() => {
    if (!overlay || overlay.line_b.length < 2) return undefined;
    if (!frameB) return overlay.line_b;
    const w = frameB.roi?.w ?? 0;
    const h = frameB.roi?.h ?? 0;
    if (w === 0 || h === 0) return overlay.line_b;
    // Quarter-frame jumps between consecutive depth samples are
    // physically impossible for a smooth epipolar line and reliably
    // indicate a fold-through-singularity in the projection.
    const jumpPx = Math.min(w, h) * 0.25;
    return longestContinuousRun(overlay.line_b, w, h, jumpPx);
  }, [overlay, frameB]);

  // Distance from the corresponding pane-B feature to the polyline in
  // pixels — the calibration's epipolar residual for the picked
  // feature pair. Surfaced as an annotation next to the cam-B
  // crosshair so the engineer can read it without leaving the workspace.
  const ghostDistancePx = useMemo<number | null>(() => {
    if (!ghostInB || !overlay || overlay.line_b.length < 2) return null;
    return distanceToPolyline(ghostInB.observed_px, overlay.line_b);
  }, [ghostInB, overlay]);

  // Camera relative pose (cam B as seen from cam A). Read here so the
  // info strip shows it alongside the overlay status.
  const relativePose = useMemo(() => {
    const cs = data.cam_se3_rig;
    if (!cs || cs.length <= cameraA || cs.length <= cameraB) return null;
    if (cameraA === cameraB) return null;
    return relativeCameraPose(cs[cameraA], cs[cameraB]);
  }, [data, cameraA, cameraB]);

  const tieMarkersA = useMemo<OverlayPoint[]>(() => {
    if (!showTieLines) return [];
    return residualsA.map((r) => ({
      px: r.observed_px,
      color: "var(--color-muted-foreground, #888)",
      dot: true,
      size: 4,
    }));
  }, [showTieLines, residualsA]);

  const tieMarkersB = useMemo<OverlayPoint[]>(() => {
    if (!showTieLines) return [];
    return residualsB.map((r) => ({
      px: r.observed_px,
      color: "var(--color-muted-foreground, #888)",
      dot: true,
      size: 4,
    }));
  }, [showTieLines, residualsB]);

  const featureMarkersA = useMemo<OverlayPoint[]>(() => {
    if (!showFeatures) return [];
    return residualsA.map((r) => ({
      px: r.observed_px,
      color: "var(--color-accent, #888)",
      dot: true,
      size: 6,
    }));
  }, [showFeatures, residualsA]);
  const featureMarkersB = useMemo<OverlayPoint[]>(() => {
    if (!showFeatures) return [];
    return residualsB.map((r) => ({
      px: r.observed_px,
      color: "var(--color-accent, #888)",
      dot: true,
      size: 6,
    }));
  }, [showFeatures, residualsB]);

  const markersA: OverlayPoint[] = [
    ...featureMarkersA,
    ...tieMarkersA,
    ...(picked
      ? [
          {
            px: picked.px,
            color: "var(--color-brand, #1abc9c)",
            size: 8,
          } satisfies OverlayPoint,
        ]
      : []),
  ];
  const markersB: OverlayPoint[] = [
    ...featureMarkersB,
    ...tieMarkersB,
    ...(ghostInB
      ? [
          {
            px: ghostInB.observed_px,
            color: "var(--color-brand, #1abc9c)",
            size: 8,
          } satisfies OverlayPoint,
        ]
      : []),
    ...(overlay?.epipole_b
      ? [
          {
            px: overlay.epipole_b,
            color: "var(--color-destructive, #e74c3c)",
            size: 7,
          } satisfies OverlayPoint,
        ]
      : []),
  ];

  // Linked-mode shares one transform between both panes; flipping into
  // linked mode forces a fresh fit so the two start aligned.
  useEffect(() => {
    if (linked) handleARef.current?.fit();
  }, [linked, handleARef]);

  const drivePane = (which: "A" | "B", action: (h: FrameCanvasHandle) => void) => {
    if (linked) {
      // Either ref drives the linked transform.
      const h = handleARef.current ?? handleBRef.current;
      if (h) action(h);
      return;
    }
    const h = which === "A" ? handleARef.current : handleBRef.current;
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
        <Selector
          label="cam A"
          value={cameraA}
          options={camerasInPose}
          onChange={(v) => setCamera(v, "A")}
        />
        <Selector
          label="cam B"
          value={cameraB}
          options={camerasInPose}
          onChange={(v) => setCamera(v, "B")}
        />
        <ZoomBar
          onFit={() =>
            drivePane("A", (h) => h.fit())
          }
          onOneToOne={() =>
            drivePane("A", (h) => h.reset1to1())
          }
          onZoomIn={() =>
            drivePane("A", (h) => h.zoomBy(1.25))
          }
          onZoomOut={() =>
            drivePane("A", (h) => h.zoomBy(1 / 1.25))
          }
        />
        <Toggle
          active={linked}
          onClick={() => setLinked((v) => !v)}
          label="Linked"
          activeLabel="Linked ✓"
          title="Share zoom + pan between both panes (common AOI)"
        />
        <Toggle
          active={showFeatures}
          onClick={() => setShowFeatures((v) => !v)}
          label="Features"
          activeLabel="Features ✓"
          title="Show every detected feature as a clickable dot"
        />
        <Toggle
          active={showTieLines}
          onClick={() => setShowTieLines((v) => !v)}
          label="Tie-points"
          activeLabel="Tie-points ✓"
          title="Render every observed feature as a faint dot (both panes)"
        />
        <span className="ml-auto font-mono text-[11px] text-muted-foreground">
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

      {overlayError && (
        <div className="rounded-md border-l-2 border-destructive bg-destructive/[0.08] p-2.5 text-[13px] text-foreground">
          {overlayError}
        </div>
      )}

      <div className="grid min-h-0 flex-1 grid-cols-2 gap-2 overflow-hidden rounded-md bg-bg-soft p-2">
        <Pane
          frame={frameA}
          residuals={residualsForPose}
          transform={linked ? linkedTransform : transformA}
          onTransformChange={linked ? setLinkedTransform : setTransformA}
          onPick={handlePickA}
          markers={markersA}
          caption={frameA ? `pane A · cam ${cameraA}${showFeatures ? " · click feature or anywhere" : " · click to pick"}` : undefined}
          handleRef={handleARef}
        />
        <Pane
          frame={frameB}
          residuals={residualsForPose}
          transform={linked ? linkedTransform : transformB}
          onTransformChange={linked ? setLinkedTransform : setTransformB}
          polyline={clippedPolyline}
          polylineColor="var(--color-brand, #1abc9c)"
          markers={markersB}
          caption={frameB ? `pane B · cam ${cameraB}` : undefined}
          handleRef={handleBRef}
          annotation={
            ghostInB && ghostDistancePx != null
              ? {
                  px: ghostInB.observed_px,
                  text: `Δ ${ghostDistancePx.toFixed(2)} px`,
                  color:
                    ghostDistancePx < 1
                      ? "var(--color-brand, #1abc9c)"
                      : ghostDistancePx < 5
                        ? "var(--color-foreground, #888)"
                        : "var(--color-destructive, #e74c3c)",
                }
              : undefined
          }
        />
      </div>
    </section>
  );
}

/** Return the longest contiguous run of polyline points that
 * (a) fall inside `[0, w] × [0, h]` AND
 * (b) sit within `jumpPx` of the previous in-run point.
 *
 * The Rust side densely samples depths along the back-projected ray;
 * only some samples project inside pane B's image. Filtering
 * in-bounds points and re-stitching them into one polyline draws a
 * "straight bridge" across out-of-image gaps, which looks like a
 * curve. Worse, when the ray passes near cam B's principal axis the
 * projection wraps around a near-singularity and consecutive in-image
 * samples can land hundreds of pixels apart on opposite sides of the
 * actual epipolar line — visually a fork / "parabola". Picking the
 * single longest run that survives both filters sidesteps both
 * artefacts. */
function longestContinuousRun(
  pts: [number, number][],
  w: number,
  h: number,
  jumpPx: number,
): [number, number][] {
  let bestStart = 0;
  let bestLen = 0;
  let curStart = -1;
  let curLen = 0;
  const flushIfBetter = () => {
    if (curLen > bestLen) {
      bestLen = curLen;
      bestStart = curStart;
    }
  };
  const reset = (i: number) => {
    flushIfBetter();
    curStart = i;
    curLen = 0;
  };
  const inBounds = (x: number, y: number) => x >= 0 && x <= w && y >= 0 && y <= h;
  for (let i = 0; i < pts.length; i++) {
    const [x, y] = pts[i];
    if (!inBounds(x, y)) {
      if (curLen > 0) reset(-1);
      continue;
    }
    if (curStart < 0) {
      curStart = i;
      curLen = 1;
      continue;
    }
    const [px, py] = pts[curStart + curLen - 1];
    const dx = x - px;
    const dy = y - py;
    if (dx * dx + dy * dy > jumpPx * jumpPx) {
      // Singularity-induced fold: end the current run at `i-1` and
      // start a new one at `i`.
      reset(i);
      curLen = 1;
      continue;
    }
    curLen += 1;
  }
  flushIfBetter();
  return bestLen > 0 ? pts.slice(bestStart, bestStart + bestLen) : [];
}

function distanceToPolyline(
  point: [number, number],
  polyline: [number, number][],
): number {
  let best = Infinity;
  for (let i = 0; i < polyline.length - 1; i++) {
    const d = distanceToSegment(point, polyline[i], polyline[i + 1]);
    if (d < best) best = d;
  }
  return best;
}

function distanceToSegment(
  p: [number, number],
  a: [number, number],
  b: [number, number],
): number {
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const lenSq = abx * abx + aby * aby;
  if (lenSq === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  const t = Math.max(
    0,
    Math.min(1, ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / lenSq),
  );
  const cx = a[0] + t * abx;
  const cy = a[1] + t * aby;
  return Math.hypot(p[0] - cx, p[1] - cy);
}

interface PaneProps {
  frame: FrameKey | null;
  residuals: TargetFeatureResidual[];
  transform: ViewportTransform;
  onTransformChange: (t: ViewportTransform) => void;
  onPick?: (pixel: { x: number; y: number }) => void;
  polyline?: [number, number][];
  polylineColor?: string;
  markers?: OverlayPoint[];
  caption?: string;
  handleRef: React.MutableRefObject<FrameCanvasHandle | null>;
  annotation?: { px: [number, number]; text: string; color: string };
}

function Pane({
  frame,
  residuals,
  transform,
  onTransformChange,
  onPick,
  polyline,
  polylineColor,
  markers,
  caption,
  handleRef,
  annotation,
}: PaneProps) {
  if (!frame) {
    return (
      <div className="flex h-full items-center justify-center rounded-md border border-dashed border-border text-[12px] text-muted-foreground">
        no frame for this (pose, camera)
      </div>
    );
  }
  return (
    <PaneInner
      key={frame.abs_path}
      frame={frame}
      residuals={residuals}
      transform={transform}
      onTransformChange={onTransformChange}
      onPick={onPick}
      polyline={polyline}
      polylineColor={polylineColor}
      markers={markers}
      caption={caption}
      handleRef={handleRef}
      annotation={annotation}
    />
  );
}

function PaneInner({
  frame,
  residuals,
  transform,
  onTransformChange,
  onPick,
  polyline,
  polylineColor,
  markers,
  caption,
  handleRef,
  annotation,
}: PaneProps & { frame: FrameKey }) {
  const image = useImageData(frame);
  return (
    <div className="relative flex h-full flex-col">
      <div className="relative flex-1 overflow-hidden">
        <FrameCanvas
          ref={handleRef}
          frame={frame}
          residuals={residuals}
          image={image?.image ?? null}
          transform={transform}
          onTransformChange={onTransformChange}
          onPick={onPick}
        />
        <EpipolarOverlay
          transform={transform}
          polyline={polyline}
          polylineColor={polylineColor}
          markers={markers}
          caption={caption}
          annotation={annotation}
        />
      </div>
    </div>
  );
}

interface ZoomBarProps {
  onFit: () => void;
  onOneToOne: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
}

function ZoomBar({ onFit, onOneToOne, onZoomIn, onZoomOut }: ZoomBarProps) {
  return (
    <div className="flex items-center gap-1">
      <button
        type="button"
        onClick={onZoomOut}
        title="Zoom out"
        aria-label="Zoom out"
        className="grid h-7 w-7 place-items-center !p-0 font-mono text-xs"
      >
        −
      </button>
      <button
        type="button"
        onClick={onZoomIn}
        title="Zoom in"
        aria-label="Zoom in"
        className="grid h-7 w-7 place-items-center !p-0 font-mono text-xs"
      >
        +
      </button>
      <button
        type="button"
        onClick={onFit}
        title="Fit"
        className="h-7 px-2 font-mono text-[11px]"
      >
        Fit
      </button>
      <button
        type="button"
        onClick={onOneToOne}
        title="1:1"
        className="h-7 px-2 font-mono text-[11px]"
      >
        1:1
      </button>
    </div>
  );
}

interface ToggleProps {
  active: boolean;
  onClick: () => void;
  label: string;
  activeLabel?: string;
  title?: string;
}

function Toggle({ active, onClick, label, activeLabel, title }: ToggleProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`h-7 px-2 font-mono text-[11px] ${
        active ? "border-brand text-brand" : ""
      }`}
      title={title}
    >
      {active ? (activeLabel ?? `${label} ✓`) : label}
    </button>
  );
}

interface RelativePoseStripProps {
  relativePose: ReturnType<typeof relativeCameraPose> | null;
  cameraA: number;
  cameraB: number;
  ghostDistancePx: number | null;
  pickedFeature: number | null;
  samplesClipped: number | null;
}

function RelativePoseStrip({
  relativePose,
  cameraA,
  cameraB,
  ghostDistancePx,
  pickedFeature,
  samplesClipped,
}: RelativePoseStripProps) {
  if (!relativePose) {
    return (
      <div className="font-mono text-[11px] text-muted-foreground">
        cam {cameraA} ↔ cam {cameraB}: pick two distinct cameras for relative pose
      </div>
    );
  }
  const dist = iso3DistanceM(relativePose);
  const euler = iso3EulerXYZDeg(relativePose);
  const angle = iso3RotationAngleDeg(relativePose);
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[11px] text-muted-foreground">
      <span>
        cam {cameraA} ⇒ cam {cameraB}:
      </span>
      <span>
        baseline <span className="text-foreground tabular-nums">{(dist * 1000).toFixed(1)} mm</span>
      </span>
      <span>
        rot <span className="text-foreground tabular-nums">{angle.toFixed(2)}°</span>
        <span className="ml-1">
          ({euler.x.toFixed(1)}, {euler.y.toFixed(1)}, {euler.z.toFixed(1)})°
        </span>
      </span>
      {pickedFeature != null && (
        <span>
          feature <span className="text-foreground tabular-nums">#{pickedFeature}</span>
        </span>
      )}
      {ghostDistancePx != null && (
        <span>
          ghost Δ
          <span
            className={`ml-1 tabular-nums ${
              ghostDistancePx < 1
                ? "text-brand"
                : ghostDistancePx < 5
                  ? "text-foreground"
                  : "text-destructive"
            }`}
          >
            {ghostDistancePx.toFixed(2)} px
          </span>
        </span>
      )}
      {samplesClipped != null && samplesClipped > 0 && (
        <span title="number of depth samples whose projection diverged">
          clipped <span className="text-foreground tabular-nums">{samplesClipped}</span>
        </span>
      )}
    </div>
  );
}

function Selector({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: number;
  options: number[];
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex items-center gap-1.5 font-mono text-[11px] text-muted-foreground">
      <span className="uppercase tracking-wider">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-7 rounded-md border border-border bg-surface px-1 text-foreground"
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  );
}

function Empty({ body }: { body: string }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <header className="flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-tight">
          Epipolar geometry
        </h2>
      </header>
      <div className="flex min-h-0 flex-1 items-center justify-center rounded-md border border-dashed border-border bg-bg-soft">
        <p className="max-w-[28rem] p-6 text-center text-[13px] text-muted-foreground">
          {body}
        </p>
      </div>
    </div>
  );
}
