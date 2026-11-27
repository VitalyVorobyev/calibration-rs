/** The maps of a run's camera as queries: each command's reply (the
 * planes' blob ids, their channels, the grid) and, per plane, the decoded
 * `VAM1` values, both held by the query cache so a tab asks once however
 * often it is shown. A blob the store dropped answers `blob_expired`; the
 * command runs again (same request, same ids) and the plane is read once
 * more. */
import type { ValuePlane } from "@vitavision/stage2d";
import { api, asStudioError, fetchPlane, type StudioError } from "../../../api";
import type {
  BlobRef,
  DistortionMap,
  MapGrid,
  PlaneChannel,
  ProjectionSigmaMap,
  ResidualMap,
  RunRef,
} from "../../../api/generated/types";
import { useQuery, type Query } from "../../../session/query";
import type { MapSource } from "./mapSource";

/** The values at `blob`; if the store dropped it, `rerun` makes the command
 * answer again and the fresh blob is read. */
export async function readPlane(
  blob: BlobRef,
  rerun: () => Promise<BlobRef>,
): Promise<ValuePlane> {
  try {
    return await fetchPlane(api.urls.blob({ id: blob.id }));
  } catch (e) {
    if (asStudioError(e).kind !== "blob_expired") throw e;
    const fresh = await rerun();
    return fetchPlane(api.urls.blob({ id: fresh.id }));
  }
}

/** The decoded plane of `blob`, once: the blob's id is its content's
 * address, so it is the key. */
export function usePlane(
  blob: BlobRef | null,
  rerun: () => Promise<BlobRef>,
): Query<ValuePlane> {
  return useQuery(
    "plane",
    { id: blob?.id ?? null },
    () => (blob ? readPlane(blob, rerun) : Promise.reject(new Error("no blob"))),
    blob !== null,
  );
}

/** What a tab needs of one plane of a map: the decoded values with their
 * grid and channels once there, or why not. */
export interface MapState {
  /** The command's error, else the plane's. */
  error: StudioError | null;
  loading: boolean;
  source: MapSource | null;
  refetch: () => void;
}

function join(
  command: Query<unknown>,
  plane: Query<ValuePlane>,
  grid: MapGrid | null,
  channels: PlaneChannel[] | null,
  refetch: () => void,
): MapState {
  const source =
    grid && channels && plane.data ? { plane: plane.data, grid, channels } : null;
  const error = command.error ?? plane.error;
  return { error, loading: source === null && error === null, source, refetch };
}

/** One layer of camera `camera`'s distortion map. */
export function useDistortionMap(
  run: RunRef | null,
  camera: number,
  layer: "distortion" | "sensor" | "total",
): { map: DistortionMap | undefined; state: MapState } {
  const q = useQuery(
    "distortion_map",
    { run, camera },
    () => api.distortionMap({ run: run!, camera }),
    run !== null,
  );
  const plane = q.data?.[layer];
  const p = usePlane(plane?.blob ?? null, () =>
    api.distortionMap({ run: run!, camera }).then((m) => m[layer].blob),
  );
  return {
    map: q.data,
    state: join(q, p, q.data?.grid ?? null, plane?.channels ?? null, q.refetch),
  };
}

/** Camera `camera`'s residual map over `views` (`null`: every view). */
export function useResidualMap(
  run: RunRef | null,
  camera: number,
  views: number[] | null,
): { map: ResidualMap | undefined; state: MapState } {
  const options = views === null ? {} : { views };
  const q = useQuery(
    "residual_map",
    { run, camera, options },
    () => api.residualMap({ run: run!, camera, options }),
    run !== null,
  );
  const p = usePlane(q.data?.plane.blob ?? null, () =>
    api.residualMap({ run: run!, camera, options }).then((m) => m.plane.blob),
  );
  return {
    map: q.data,
    state: join(q, p, q.data?.grid ?? null, q.data?.plane.channels ?? null, q.refetch),
  };
}

/** Camera `camera`'s projection σ map; `unavailable` says why there is none. */
export function useSigmaMap(
  run: RunRef | null,
  camera: number,
): { map: ProjectionSigmaMap | undefined; state: MapState } {
  const q = useQuery(
    "projection_sigma_map",
    { run, camera },
    () => api.projectionSigmaMap({ run: run!, camera }),
    run !== null,
  );
  const available = q.data?.status === "available" ? q.data : null;
  const p = usePlane(available?.plane.blob ?? null, () =>
    api.projectionSigmaMap({ run: run!, camera }).then((m) => {
      if (m.status !== "available") throw new Error("the σ map is no longer available");
      return m.plane.blob;
    }),
  );
  return {
    map: q.data,
    state: join(
      q,
      p,
      available?.grid ?? null,
      available?.plane.channels ?? null,
      q.refetch,
    ),
  };
}
