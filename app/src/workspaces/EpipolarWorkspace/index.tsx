import { invoke } from "@tauri-apps/api/core";
import { useCallback, useEffect, useMemo, useState } from "react";
import { FrameCanvas } from "../../components/FrameCanvas";
import { useImageData } from "../../hooks/useImageData";
import { useStore } from "../../store";
import { exportKindLabel } from "../../store/exportShape";
import type { FrameKey, TargetFeatureResidual, ViewportTransform } from "../../types";
import { IDENTITY_TRANSFORM } from "../../types";
import { EpipolarOverlay, type OverlayPoint } from "./EpipolarOverlay";

interface EpipolarOverlayResult {
  line_b: [number, number][];
  epipole_b: [number, number] | null;
  samples_clipped: number;
}

/** Pixel-radius around the click within which we snap the selected
 * feature index; otherwise the click is a free pixel pick (no
 * corresponding pane-B ghost). */
const FEATURE_SNAP_PX = 6;

export function EpipolarWorkspace() {
  const data = useStore((s) => s.data);
  const kind = useStore((s) => s.kind);
  const frames = useStore((s) => s.frames);
  const cameraValues = useStore((s) => s.cameraValues);
  const poseValues = useStore((s) => s.poseValues);
  const camerasByPose = useStore((s) => s.camerasByPose);
  const selectedPose = useStore((s) => s.selectedPose);
  const cameraA = useStore((s) => s.cameraA);
  const cameraB = useStore((s) => s.cameraB);
  const setSelectedPose = useStore((s) => s.setSelectedPose);
  const setCamera = useStore((s) => s.setCamera);

  const [transformA, setTransformA] = useState<ViewportTransform>(
    IDENTITY_TRANSFORM,
  );
  const [transformB, setTransformB] = useState<ViewportTransform>(
    IDENTITY_TRANSFORM,
  );
  const [picked, setPicked] = useState<{ px: [number, number]; feature: number | null } | null>(
    null,
  );
  const [overlay, setOverlay] = useState<EpipolarOverlayResult | null>(null);
  const [overlayError, setOverlayError] = useState<string | null>(null);
  const [showTieLines, setShowTieLines] = useState(false);

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
      poseValues={poseValues}
      cameraValues={cameraValues}
      camerasByPose={camerasByPose}
      selectedPose={selectedPose}
      cameraA={cameraA}
      cameraB={cameraB}
      setSelectedPose={setSelectedPose}
      setCamera={setCamera}
      transformA={transformA}
      transformB={transformB}
      setTransformA={setTransformA}
      setTransformB={setTransformB}
      picked={picked}
      setPicked={setPicked}
      overlay={overlay}
      setOverlay={setOverlay}
      overlayError={overlayError}
      setOverlayError={setOverlayError}
      showTieLines={showTieLines}
      setShowTieLines={setShowTieLines}
    />
  );
}

interface BodyProps {
  data: NonNullable<ReturnType<typeof useStore.getState>["data"]>;
  kind: NonNullable<ReturnType<typeof useStore.getState>["kind"]>;
  frames: FrameKey[];
  poseValues: number[];
  cameraValues: number[];
  camerasByPose: Map<number, number[]>;
  selectedPose: number;
  cameraA: number;
  cameraB: number;
  setSelectedPose: (pose: number, which?: "A" | "B") => void;
  setCamera: (camera: number, which?: "A" | "B") => void;
  transformA: ViewportTransform;
  transformB: ViewportTransform;
  setTransformA: (t: ViewportTransform) => void;
  setTransformB: (t: ViewportTransform) => void;
  picked: { px: [number, number]; feature: number | null } | null;
  setPicked: (p: { px: [number, number]; feature: number | null } | null) => void;
  overlay: EpipolarOverlayResult | null;
  setOverlay: (o: EpipolarOverlayResult | null) => void;
  overlayError: string | null;
  setOverlayError: (msg: string | null) => void;
  showTieLines: boolean;
  setShowTieLines: (v: boolean | ((prev: boolean) => boolean)) => void;
}

function EpipolarBody(props: BodyProps) {
  const {
    data,
    kind,
    frames,
    poseValues,
    cameraValues,
    camerasByPose,
    selectedPose,
    cameraA,
    cameraB,
    setSelectedPose,
    setCamera,
    transformA,
    transformB,
    setTransformA,
    setTransformB,
    picked,
    setPicked,
    overlay,
    setOverlay,
    overlayError,
    setOverlayError,
    showTieLines,
    setShowTieLines,
  } = props;

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

  // Bucket the residuals once per (pose) so the overlays / tie-lines
  // don't re-scan the full residual array on every render.
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

  // Reset the picked overlay whenever the user changes the working
  // tuple (pose / cam-A / cam-B). Stale polylines from a previous
  // configuration are misleading.
  useEffect(() => {
    setPicked(null);
    setOverlay(null);
    setOverlayError(null);
  }, [selectedPose, cameraA, cameraB, setPicked, setOverlay, setOverlayError]);

  const camerasInPose = camerasByPose.get(selectedPose) ?? cameraValues;

  const handlePickA = useCallback(
    async (pixel: { x: number; y: number }) => {
      const px: [number, number] = [pixel.x, pixel.y];
      // Snap to the nearest residual feature in cam A within
      // FEATURE_SNAP_PX so the workspace can ghost the matching feature
      // in pane B; outside that radius the click is a free pixel pick
      // and no ghost is shown.
      let feature: number | null = null;
      let bestDist = FEATURE_SNAP_PX;
      for (const r of residualsA) {
        const dx = r.observed_px[0] - px[0];
        const dy = r.observed_px[1] - px[1];
        const d = Math.hypot(dx, dy);
        if (d < bestDist) {
          bestDist = d;
          feature = r.feature;
        }
      }
      setPicked({ px, feature });
      setOverlayError(null);
      try {
        const result = await invoke<EpipolarOverlayResult>(
          "compute_epipolar_overlay",
          { camA: cameraA, camB: cameraB, pointPx: px },
        );
        setOverlay(result);
      } catch (e) {
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

  const markersA: OverlayPoint[] = [
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

  return (
    <section className="flex min-h-0 flex-1 flex-col gap-2.5">
      <div className="flex flex-wrap items-center gap-3">
        <Selector
          label="pose"
          value={selectedPose}
          options={poseValues}
          onChange={(v) => setSelectedPose(v, "A")}
        />
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
        <button
          type="button"
          onClick={() => setShowTieLines((v) => !v)}
          className={`h-7 px-2 font-mono text-[11px] ${
            showTieLines ? "border-brand text-brand" : ""
          }`}
          title="Render every feature observation as a faint dot in both panes"
        >
          {showTieLines ? "Tie-points ✓" : "Tie-points"}
        </button>
        <span className="ml-auto font-mono text-[11px] text-muted-foreground">
          {exportKindLabel(kind)}
        </span>
      </div>

      {overlayError && (
        <div className="rounded-md border-l-2 border-destructive bg-destructive/[0.08] p-2.5 text-[13px] text-foreground">
          {overlayError}
        </div>
      )}

      <div className="grid min-h-0 flex-1 grid-cols-2 gap-2 overflow-hidden rounded-md bg-bg-soft p-2">
        <Pane
          frame={frameA}
          residuals={residualsForPose}
          transform={transformA}
          onTransformChange={setTransformA}
          onPick={handlePickA}
          markers={markersA}
          caption={frameA ? `pane A · cam ${cameraA} · click to pick` : undefined}
        />
        <Pane
          frame={frameB}
          residuals={residualsForPose}
          transform={transformB}
          onTransformChange={setTransformB}
          polyline={overlay?.line_b}
          polylineColor="var(--color-brand, #1abc9c)"
          markers={markersB}
          caption={frameB ? `pane B · cam ${cameraB}` : undefined}
        />
      </div>

      {!cameraValues.includes(cameraB) && (
        <div className="rounded-md border-l-2 border-brand bg-brand/[0.06] p-2 text-[12px] text-foreground">
          Pane-B camera {cameraB} is not in this export.
        </div>
      )}
    </section>
  );
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
}: PaneProps & { frame: FrameKey }) {
  const image = useImageData(frame);
  return (
    <div className="relative flex h-full flex-col">
      <div className="relative flex-1 overflow-hidden">
        <FrameCanvas
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
        />
      </div>
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
        <h2 className="text-sm font-semibold tracking-tight">Epipolar geometry</h2>
      </header>
      <div className="flex min-h-0 flex-1 items-center justify-center rounded-md border border-dashed border-border bg-bg-soft">
        <p className="max-w-[28rem] p-6 text-center text-[13px] text-muted-foreground">
          {body}
        </p>
      </div>
    </div>
  );
}
