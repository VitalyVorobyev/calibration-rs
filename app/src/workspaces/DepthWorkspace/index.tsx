import { invoke } from "@tauri-apps/api/core";
import { lazy, Suspense, useMemo, useState } from "react";
import { PoseStepper } from "../../components/PoseStepper";
import { useStore } from "../../store";

// Three.js is heavy (~900 KB); only pull it in when the 3D view is opened.
const PointCloudView = lazy(() =>
  import("./PointCloudView").then((m) => ({ default: m.PointCloudView })),
);

/** A reprojected 3D point cloud (flat position/colour arrays). */
interface PointCloud {
  positions: number[];
  colors: number[];
  count: number;
}

/** Result of the `compute_disparity` Tauri command (camelCase, matching the
 * Rust `#[serde(rename_all = "camelCase")]` struct). All images are
 * `data:image/png` base64 URLs. */
interface DisparityResult {
  rectifiedPairPng: string;
  disparityPng: string;
  overlayPng: string;
  depthPng: string;
  pointCloud: PointCloud;
  width: number;
  height: number;
  density: number;
  dispMin: number;
  dispMax: number;
  planeRms: number;
  planeInliers: number;
  baselineM: number;
  semiGlobal: boolean;
}

type ViewMode = "rectified" | "disparity" | "overlay" | "depth" | "3d";

// Matching is done at reduced resolution to keep the disparity search (and the
// dev-build solve time) tractable; rebuild in release for full resolution.
const BLOCK_SIZE = 11;
const DOWNSCALE = 4;

const VIEW_LABEL: Record<ViewMode, string> = {
  rectified: "rectified pair",
  disparity: "disparity",
  overlay: "overlay",
  depth: "depth",
  "3d": "3D cloud",
};

/** Dense stereo / depth workspace: rectify a synchronized pair and dense-match
 * it server-side, then show the colormapped disparity. */
export function DepthWorkspace() {
  const data = useStore((s) => s.data);
  const kind = useStore((s) => s.kind);
  const frames = useStore((s) => s.frames);
  const poseValues = useStore((s) => s.poseValues);
  const cameraValues = useStore((s) => s.cameraValues);
  const selectedPose = useStore((s) => s.selectedPose);
  const cameraA = useStore((s) => s.cameraA);
  const cameraB = useStore((s) => s.cameraB);
  const setSelectedPose = useStore((s) => s.setSelectedPose);
  const setCamera = useStore((s) => s.setCamera);

  const [semiGlobal, setSemiGlobal] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>("overlay");
  const [result, setResult] = useState<DisparityResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const frameA = useMemo(
    () => frames.find((f) => f.pose === selectedPose && f.camera === cameraA) ?? null,
    [frames, selectedPose, cameraA],
  );
  const frameB = useMemo(
    () => frames.find((f) => f.pose === selectedPose && f.camera === cameraB) ?? null,
    [frames, selectedPose, cameraB],
  );
  const cloud = useMemo(
    () =>
      result
        ? {
            positions: new Float32Array(result.pointCloud.positions),
            colors: new Float32Array(result.pointCloud.colors),
          }
        : null,
    [result],
  );

  if (!data || !kind) {
    return <Empty body="Load a rig export (two cameras + extrinsics) to compute dense disparity." />;
  }
  const isRig =
    Array.isArray(data.cameras) && Array.isArray(data.cam_se3_rig) && data.cameras.length >= 2;
  if (!isRig) {
    return <Empty body="Dense matching needs a stereo rig export (two cameras with extrinsics)." />;
  }

  const canCompute = frameA != null && frameB != null && cameraA !== cameraB && !loading;

  const compute = async () => {
    if (!frameA || !frameB) return;
    setLoading(true);
    setError(null);
    try {
      const res = await invoke<DisparityResult>("compute_disparity", {
        pathA: frameA.abs_path,
        pathB: frameB.abs_path,
        camA: cameraA,
        camB: cameraB,
        pose: selectedPose,
        params: { blockSize: BLOCK_SIZE, semiGlobal, downscale: DOWNSCALE },
      });
      setResult(res);
    } catch (e) {
      setError(String(e));
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const imgSrc =
    result == null
      ? null
      : viewMode === "disparity"
        ? result.disparityPng
        : viewMode === "overlay"
          ? result.overlayPng
          : viewMode === "depth"
            ? result.depthPng
            : viewMode === "rectified"
              ? result.rectifiedPairPng
              : null; // "3d" is rendered as WebGL below

  return (
    <section className="flex min-h-0 flex-1 flex-col gap-2.5">
      <div className="flex flex-wrap items-center gap-3">
        <PoseStepper
          poseValues={poseValues}
          selectedPose={selectedPose}
          onSelectPose={(p) => setSelectedPose(p)}
        />
        <Selector label="left" value={cameraA} options={cameraValues} onChange={(v) => setCamera(v, "A")} />
        <Selector label="right" value={cameraB} options={cameraValues} onChange={(v) => setCamera(v, "B")} />
        <Toggle
          active={semiGlobal}
          onClick={() => setSemiGlobal((v) => !v)}
          label="SGM"
          title="Semi-global aggregation: fills low-texture regions"
        />
        <button
          type="button"
          onClick={compute}
          disabled={!canCompute}
          className="h-7 px-2 font-mono text-[11px] border-brand text-brand disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? "Matching…" : "Compute disparity"}
        </button>
        {result && (
          <div className="ml-auto flex items-center gap-1">
            {(Object.keys(VIEW_LABEL) as ViewMode[]).map((m) => (
              <Toggle key={m} active={viewMode === m} onClick={() => setViewMode(m)} label={VIEW_LABEL[m]} />
            ))}
          </div>
        )}
      </div>

      {error && (
        <div className="rounded-md border-l-2 border-destructive bg-destructive/[0.08] p-2.5 text-[13px] text-foreground">
          {error}
        </div>
      )}

      <div className="relative grid min-h-0 flex-1 place-items-center overflow-hidden rounded-md bg-bg-soft">
        {viewMode === "3d" && cloud ? (
          <Suspense fallback={<Hint>Loading 3D…</Hint>}>
            <div className="absolute inset-0">
              <PointCloudView positions={cloud.positions} colors={cloud.colors} />
            </div>
          </Suspense>
        ) : imgSrc ? (
          <img
            src={imgSrc}
            alt={VIEW_LABEL[viewMode]}
            className="max-h-full max-w-full object-contain p-2 [image-rendering:pixelated]"
          />
        ) : (
          <Hint>
            {cameraA === cameraB
              ? "Pick two different cameras (left ≠ right)."
              : frameA && frameB
                ? "Press “Compute disparity” to rectify and dense-match this pair."
                : "No image pair for this pose / camera selection."}
          </Hint>
        )}
      </div>

      {result && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[11px] text-muted-foreground">
          <Stat label="mode" value={result.semiGlobal ? "semi-global" : "block"} />
          <Stat label="density" value={`${(result.density * 100).toFixed(1)}%`} />
          <Stat label="disparity" value={`${result.dispMin.toFixed(1)}–${result.dispMax.toFixed(1)} px`} />
          <Stat label="planarity RMS" value={`${result.planeRms.toFixed(2)} px`} />
          <Stat label="plane inliers" value={`${result.planeInliers}`} />
          <Stat label="cloud points" value={`${result.pointCloud.count}`} />
          <Stat label="baseline" value={`${(result.baselineM * 1000).toFixed(0)} mm`} />
          <Stat label="match size" value={`${result.width}×${result.height}`} />
        </div>
      )}
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span>
      {label} <span className="text-foreground tabular-nums">{value}</span>
    </span>
  );
}

function Hint({ children }: { children: React.ReactNode }) {
  return (
    <p className="max-w-[26rem] text-center text-[12px] text-muted-foreground">{children}</p>
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

function Toggle({
  active,
  onClick,
  label,
  title,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className={`h-7 px-2 font-mono text-[11px] ${active ? "border-brand text-brand" : ""}`}
    >
      {label}
    </button>
  );
}

function Empty({ body }: { body: string }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <header className="flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-tight">Depth (dense stereo)</h2>
      </header>
      <div className="flex min-h-0 flex-1 items-center justify-center rounded-md border border-dashed border-border bg-bg-soft">
        <p className="max-w-[28rem] p-6 text-center text-[13px] text-muted-foreground">{body}</p>
      </div>
    </div>
  );
}
