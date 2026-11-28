/** The Compare screen's two queries: the comparison of the chosen runs
 * (the first is the reference), and one camera's projection difference
 * between the reference and another run, whose plane is read by URL once
 * (`readPlane`: a dropped blob runs the command again). */
import { api } from "../../api";
import type {
  CompareRunsOptions,
  Compensation,
  ProjectionDifferenceMap,
  RunComparison,
} from "../../api/generated/types";
import { useQuery } from "../../session/query";
import { usePlane, type MapState } from "../inspect/maps/useMaps";

/** The comparison of `runs` of the project in `dir`; none until two runs
 * are chosen. */
export function useComparison(
  dir: string | null,
  runs: string[],
  options: CompareRunsOptions = {},
) {
  return useQuery<RunComparison>(
    "compare_runs",
    { dir, runs, options },
    () => api.compareRuns({ dir: dir!, runs, options }),
    dir !== null && runs.length >= 2,
  );
}

/** What the map tab needs: the reply (the numbers, or why there is no map)
 * and the plane's state for the viewer. */
export function useProjectionDifference(
  dir: string | null,
  reference: string | null,
  other: string | null,
  camera: string | null,
  compensation: Compensation,
): { map: ProjectionDifferenceMap | undefined; state: MapState } {
  const enabled = dir !== null && reference !== null && other !== null && camera !== null;
  const args = { dir, reference, other, camera, compensation };
  const ask = () =>
    api.projectionDifferenceMap({
      dir: dir!,
      reference: reference!,
      other: other!,
      camera: camera!,
      compensation,
    });
  const q = useQuery("projection_difference_map", args, ask, enabled);
  const compared = q.data?.status === "compared" ? q.data : null;
  const p = usePlane(compared?.plane.blob ?? null, () =>
    ask().then((m) => {
      if (m.status !== "compared") throw new Error("the difference map is gone");
      return m.plane.blob;
    }),
  );
  const source =
    compared && p.data
      ? { plane: p.data, grid: compared.grid, channels: compared.plane.channels }
      : null;
  const error = q.error ?? p.error;
  return {
    map: q.data,
    state: {
      error,
      // An unavailable map is an answer, not a wait.
      loading:
        enabled && q.data?.status !== "unavailable" && source === null && error === null,
      source,
      refetch: q.refetch,
    },
  };
}
