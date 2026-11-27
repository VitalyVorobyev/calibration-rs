/** Wire-typed project fixtures for component tests: an open project as
 * `open_project` answers it, and a run the `calibrate` job wrote. */
import type { DatasetSpec, ProjectView, RunInfo } from "../api/generated/types";

export const MANIFEST = {
  version: 1,
  topology: "planar_intrinsics",
  target: { kind: "chessboard", rows: 6, cols: 8, square_size_m: 0.03 },
  cameras: [{ id: "cam0", images: { kind: "glob", pattern: "img/*.png" } }],
} as unknown as DatasetSpec;

export const DIGEST = {
  dataset: "sha256:d",
  exclusions: "sha256:e",
  config: "sha256:c",
  quality: "sha256:q",
};

/** Run `id` of the project in `dir`, calibrated from `DIGEST`'s inputs. */
export function calibratedRun(dir: string, id: string, views = [9, 10]): RunInfo {
  return {
    id,
    dir: `${dir}/runs/${id}`,
    notes: {},
    state: {
      kind: "recorded",
      version: 1,
      timestamp: 1,
      tool: "test",
      library_version: "1.0.0",
      summary: { kind: "planar_intrinsics", verdict: "pass", cameras: [] },
      source: {
        kind: "calibrated",
        digest: { ...DIGEST, inputs: "sha256:i", detections: "sha256:x" },
        usable_views: views[0]!,
        total_views: views[1]!,
        left_out: [],
        redetected: false,
        duration_ms: 42,
        inputs: {} as never,
      },
    },
  };
}

/** The project in `dir`, its manifest in its file, with `runs`. */
export function projectView(dir: string, runs: RunInfo[] = []): ProjectView {
  return {
    dir,
    name: dir.slice(dir.lastIndexOf("/") + 1),
    file_version: 2,
    revision: "sha256:r1",
    spec: { version: 2, dataset: { root: ".", manifest: MANIFEST } },
    data_root: dir,
    dataset: { kind: "ok", spec: MANIFEST },
    dataset_editable: true,
    config: { kind: "ok", config: { solver: { max_iters: 50 } } },
    input_digest: DIGEST,
    runs,
  };
}
