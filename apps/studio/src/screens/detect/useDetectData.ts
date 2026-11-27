/** What the Detect screen reads and starts: the overview of the project's
 * detection cache, one image's detail, a preview with edited settings, and
 * the `detect_all` job. The cache's answers come through the app's query
 * cache (a saved project marks them stale, so a changed detector or an
 * exclusion refetches them); a preview is the screen's own, never cached. */
import { useCallback, useEffect, useRef, useState } from "react";
import { api, asStudioError, isCancelled, type StudioError } from "../../api";
import type {
  DetectorSpec,
  DetectionDetail,
  DetectionOverview,
} from "../../api/generated/types";
import { isLive } from "../../session/jobModel";
import { useJobs } from "../../session/jobs";
import { queryCache, useQuery, type Query } from "../../session/query";

export function useOverview(dir: string): Query<DetectionOverview> {
  return useQuery("detection_overview", { dir }, () => api.detectionOverview({ dir }));
}

/** An image's detail. `enabled: false` waits (the image has no cached
 * detection and nobody asked to detect it). */
export function useDetail(
  dir: string,
  camera: number | null,
  image: string | null,
  enabled: boolean,
): Query<DetectionDetail> {
  const ready = enabled && camera !== null && image !== null;
  return useQuery(
    "detection_detail",
    { dir, camera, image },
    () => api.detectionDetail({ dir, camera: camera ?? 0, image: image ?? "" }),
    ready,
  );
}

export interface PreviewState {
  /** `camera|image|detector` the answer is for. */
  key: string;
  detail: DetectionDetail | null;
  loading: boolean;
  error: StudioError | null;
}

/** `detect_preview` of one image with edited settings: asked on demand
 * (`run`), forgotten when the image changes (`clear`). Nothing is written. */
export function usePreview(dir: string) {
  const [state, setState] = useState<PreviewState | null>(null);
  const seq = useRef(0);

  const run = useCallback(
    async (camera: number, image: string, detector: DetectorSpec) => {
      const key = `${camera}|${image}|${JSON.stringify(detector)}`;
      const mine = ++seq.current;
      setState({ key, detail: null, loading: true, error: null });
      try {
        const detail = await api.detectPreview({ dir, camera, image, detector });
        if (mine === seq.current) setState({ key, detail, loading: false, error: null });
      } catch (e) {
        if (mine === seq.current)
          setState({ key, detail: null, loading: false, error: asStudioError(e) });
      }
    },
    [dir],
  );
  const clear = useCallback(() => {
    seq.current++;
    setState(null);
  }, []);
  return { preview: state, run, clear };
}

/** The `detect_all` job: whether one is live (this tab's or another's), how
 * far it is, Start and Cancel. A finished job marks the project's queries
 * stale, so the matrix and the open image read the cache again. */
export function useDetectAll(dir: string) {
  const row = useJobs((s) =>
    s.rows.find((r) => r.command === "detect_all" && isLive(r.state)),
  );
  const cancel = useJobs((s) => s.cancel);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<StudioError | null>(null);

  const start = useCallback(
    async (redetect: boolean) => {
      setStarting(true);
      setError(null);
      try {
        const job = await api.jobs.detectAll({ dir, ...(redetect ? { redetect } : {}) });
        setStarting(false);
        await job.result;
      } catch (e) {
        if (!isCancelled(e)) setError(asStudioError(e));
      } finally {
        setStarting(false);
        queryCache.invalidateProject(dir);
      }
    },
    [dir],
  );

  // A job that ended in another tab changes the cache too.
  const live = row !== undefined;
  const wasLive = useRef(false);
  useEffect(() => {
    if (wasLive.current && !live) queryCache.invalidateProject(dir);
    wasLive.current = live;
  }, [live, dir]);

  return {
    start,
    cancel: () => (row ? void cancel(row.id).catch(() => {}) : undefined),
    running: starting || live,
    row,
    error,
    clearError: () => setError(null),
  };
}
