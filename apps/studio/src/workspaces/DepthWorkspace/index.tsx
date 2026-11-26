import { api } from "../../api";
import {
  Button,
  Empty,
  ErrorBox,
  SegmentedControl,
  Select,
  ToggleChip,
} from "@vitavision/ui";
import { lazy, Suspense, useMemo, useRef, useState } from "react";
import { NoImageList } from "../../components/NoImageList";
import { PoseStepper } from "../../components/PoseStepper";
import { useStore } from "../../store";
import type { DisparityResult } from "../../api/generated/types";

// Three.js is heavy (~900 KB); only pull it in when the 3D view is opened.
const PointCloudView = lazy(() =>
  import("./PointCloudView").then((m) => ({ default: m.PointCloudView })),
);

/** A reprojected 3D point cloud (flat position/colour arrays). */
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

const VIEW_MODES = Object.keys(VIEW_LABEL) as ViewMode[];

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
  const [resultState, setResultState] = useState<{
    key: string;
    result: DisparityResult | null;
    error: string | null;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const requestSeqRef = useRef(0);

  const frameA = useMemo(
    () => frames.find((f) => f.pose === selectedPose && f.camera === cameraA) ?? null,
    [frames, selectedPose, cameraA],
  );
  const frameB = useMemo(
    () => frames.find((f) => f.pose === selectedPose && f.camera === cameraB) ?? null,
    [frames, selectedPose, cameraB],
  );
  const selectionKey = `${poseValues.length}|${selectedPose}|${cameraA}|${cameraB}|${frameA?.abs_path ?? ""}|${frameB?.abs_path ?? ""}`;
  const result = resultState?.key === selectionKey ? resultState.result : null;
  const error = resultState?.key === selectionKey ? resultState.error : null;
  const cloud = useMemo(
    () =>
      result
        ? {
            positions: new Float32Array(result.point_cloud.positions),
            colors: new Float32Array(result.point_cloud.colors),
          }
        : null,
    [result],
  );

  if (!data || !kind) {
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-3">
        <h2 className="text-sm font-semibold tracking-tight text-fg">
          Depth (dense stereo)
        </h2>
        <Empty className="flex-1 justify-center rounded-panel border border-dashed border-line bg-surface px-6 [&>p]:max-w-md">
          Load a rig export (two cameras + extrinsics) to compute dense disparity.
        </Empty>
      </div>
    );
  }
  const isRig =
    Array.isArray(data.cameras) &&
    Array.isArray(data.cam_se3_rig) &&
    data.cameras.length >= 2;
  if (!isRig) {
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-3">
        <h2 className="text-sm font-semibold tracking-tight text-fg">
          Depth (dense stereo)
        </h2>
        <Empty className="flex-1 justify-center rounded-panel border border-dashed border-line bg-surface px-6 [&>p]:max-w-md">
          Dense matching needs a stereo rig export (two cameras with extrinsics).
        </Empty>
      </div>
    );
  }

  if (!frames.some((f) => f.abs_path !== null)) {
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-3">
        <h2 className="text-sm font-semibold tracking-tight text-fg">
          Depth (dense stereo)
        </h2>
        <div className="flex flex-1 rounded-panel border border-dashed border-line bg-surface">
          <NoImageList />
        </div>
      </div>
    );
  }

  const canCompute = frameA != null && frameB != null && cameraA !== cameraB && !loading;

  const compute = async () => {
    if (!frameA?.abs_path || !frameB?.abs_path) return;
    const seq = ++requestSeqRef.current;
    const key = selectionKey;
    setLoading(true);
    setResultState((prev) =>
      prev?.key === key
        ? { key, result: prev.result, error: null }
        : { key, result: null, error: null },
    );
    try {
      const res = await api.disparity({
        path_a: frameA.abs_path,
        path_b: frameB.abs_path,
        cam_a: cameraA,
        cam_b: cameraB,
        pose: selectedPose,
        params: { block_size: BLOCK_SIZE, semi_global: semiGlobal, downscale: DOWNSCALE },
      });
      if (requestSeqRef.current === seq) {
        setResultState({ key, result: res, error: null });
      }
    } catch (e) {
      if (requestSeqRef.current === seq) {
        setResultState({ key, result: null, error: String(e) });
      }
    } finally {
      if (requestSeqRef.current === seq) {
        setLoading(false);
      }
    }
  };

  const imgSrc =
    result == null
      ? null
      : viewMode === "disparity"
        ? result.disparity_png
        : viewMode === "overlay"
          ? result.overlay_png
          : viewMode === "depth"
            ? result.depth_png
            : viewMode === "rectified"
              ? result.rectified_pair_png
              : null; // "3d" is rendered as WebGL below

  return (
    <section className="flex min-h-0 flex-1 flex-col gap-2.5">
      <div className="flex flex-wrap items-center gap-3">
        <PoseStepper
          poseValues={poseValues}
          selectedPose={selectedPose}
          onSelectPose={(p) => setSelectedPose(p)}
        />
        <div className="flex items-center gap-1.5">
          <span className="font-mono text-[11px] tracking-wider text-fg-muted uppercase">
            left
          </span>
          <Select
            aria-label="left"
            className="h-7 w-16 font-mono text-xs"
            value={String(cameraA)}
            options={cameraValues.map((i) => ({ value: String(i), label: String(i) }))}
            onValueChange={(v) => setCamera(Number(v), "A")}
          />
        </div>
        <div className="flex items-center gap-1.5">
          <span className="font-mono text-[11px] tracking-wider text-fg-muted uppercase">
            right
          </span>
          <Select
            aria-label="right"
            className="h-7 w-16 font-mono text-xs"
            value={String(cameraB)}
            options={cameraValues.map((i) => ({ value: String(i), label: String(i) }))}
            onValueChange={(v) => setCamera(Number(v), "B")}
          />
        </div>
        <ToggleChip
          checked={semiGlobal}
          onCheckedChange={setSemiGlobal}
          title="Semi-global aggregation: fills low-texture regions"
        >
          SGM
        </ToggleChip>
        <Button
          variant="primary"
          size="sm"
          loading={loading}
          onClick={() => void compute()}
          disabled={!canCompute}
        >
          {loading ? "Matching…" : "Compute disparity"}
        </Button>
        {result && (
          <SegmentedControl
            aria-label="View"
            className="ml-auto"
            value={viewMode}
            options={VIEW_MODES.map((m) => ({ value: m, label: VIEW_LABEL[m] }))}
            onValueChange={(v) => setViewMode(v as ViewMode)}
          />
        )}
      </div>

      {error && <ErrorBox>{error}</ErrorBox>}

      <div className="relative grid min-h-0 flex-1 place-items-center overflow-hidden rounded-control bg-raised">
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
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[11px] text-fg-muted">
          <Stat label="mode" value={result.semi_global ? "semi-global" : "block"} />
          <Stat label="density" value={`${(result.density * 100).toFixed(1)}%`} />
          <Stat
            label="disparity"
            value={`${result.disp_min.toFixed(1)}–${result.disp_max.toFixed(1)} px`}
          />
          <Stat
            label="planarity RMS"
            value={result.plane_rms == null ? "—" : `${result.plane_rms.toFixed(2)} px`}
          />
          <Stat label="plane inliers" value={`${result.plane_inliers}`} />
          <Stat label="cloud points" value={`${result.point_cloud.count}`} />
          <Stat label="baseline" value={`${(result.baseline_m * 1000).toFixed(0)} mm`} />
          <Stat label="match size" value={`${result.width}×${result.height}`} />
        </div>
      )}
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span>
      {label} <span className="text-fg tabular-nums">{value}</span>
    </span>
  );
}

function Hint({ children }: { children: React.ReactNode }) {
  return (
    <p className="max-w-[26rem] text-center text-[12px] text-fg-muted">{children}</p>
  );
}
