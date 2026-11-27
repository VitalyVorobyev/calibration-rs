/** The Device screen's back end over HTTP, on the fixture library: the
 * history of `Stereo-X1 / SN-0042` across its two projects, its parameters
 * over it with the library's spread, the seeds its runs make (previewed and
 * written to a file), and a project's `[seed]` set through the library's
 * pre-flight check and cleared. Nothing here drives the page. */
import { expect, test } from "@playwright/test";
import type {
  DeviceHistories,
  DeviceSeriesSet,
  ExportedSeed,
  ProjectView,
  SeedPreview,
  StudioError,
} from "../../src/api/generated/types";
import { command, libraryRoot } from "./support";

const SN42 = { kind: "device", product: "Stereo-X1", id: "SN-0042" };

test("a device's history spans its projects, with its representative run", async ({
  request,
}) => {
  const reply = await command(request, "device_history", { device: SN42 });
  expect(reply.status).toBe(200);
  const { devices } = reply.body as unknown as DeviceHistories;
  expect(devices).toHaveLength(1);
  const runs = devices[0].runs;
  expect(runs.map((r) => [r.project_name, r.id])).toEqual([
    ["Bench calibration", "run-001"],
    ["Bench calibration", "run-002"],
    ["Field recheck", "run-001"],
  ]);
  expect(runs.map((r) => r.representative)).toEqual([false, true, false]);
  expect(runs[1].cameras.map((c) => c.rms_px)).toEqual([0.231, 0.247]);

  const type = await command(request, "device_history", {
    device: { kind: "type", product: "Stereo-X1" },
  });
  const ids = (type.body as unknown as DeviceHistories).devices.map((d) => d.device.id);
  expect(ids).toEqual(["SN-0042", "SN-0057"]);
});

test("a device's parameters over its runs, with σ and the library's spread", async ({
  request,
}) => {
  const reply = await command(request, "device_series", { device: SN42 });
  expect(reply.status).toBe(200);
  const series = (reply.body as unknown as DeviceSeriesSet).devices[0];
  expect(series.runs.map((r) => r.order)).toEqual([0, 1, 2]);
  const fx = series.parameters.find((p) => p.key === "cam0.fx")!;
  expect(fx.points.map((p) => p?.value)).toEqual([200, 200.6, 199.8]);
  expect(fx.points[0]?.sigma).toEqual({ kind: "estimated", sigma: 0.5 });
  expect(fx.spread).toBe("cam0.fx");
  const statistics = series.statistics;
  if (statistics.status !== "available") throw new Error("no statistics");
  const spread = statistics.spread.find((s) => s.label === "cam0.fx")!;
  expect(spread.median).toBe(200);
  expect(spread.min).toBe(199.8);
  expect(spread.max).toBe(200.6);
});

test("seeds: previewed from a run and an aggregate, written as JSON and TOML, set on a project and cleared", async ({
  request,
}) => {
  const root = await libraryRoot(request);
  const run = (project: string, id: string) => ({ dir: `${root}/${project}`, id });

  const one = await command(request, "seed_preview", {
    from: { kind: "run", run: run("stereo-sn0042", "run-002") },
  });
  expect(one.status).toBe(200);
  const seed = (one.body as unknown as SeedPreview).seed;
  expect(seed.cameras.map((c) => [c.id, c.image_size])).toEqual([
    ["cam0", [96, 72]],
    ["cam1", [96, 72]],
  ]);

  const aggregate = await command(request, "seed_preview", {
    from: {
      kind: "runs",
      runs: [run("stereo-sn0042", "run-002"), run("stereo-sn0042-field", "run-001")],
    },
  });
  expect(aggregate.status).toBe(200);
  const preview = aggregate.body as unknown as SeedPreview;
  expect(preview.spread.find((s) => s.label === "cam0.fx")?.values).toEqual([
    200.6, 199.8,
  ]);
  expect(preview.omitted).toEqual([]);

  // Written to a file inside the root; not over an existing one unless asked.
  const from = { kind: "units", devices: [{ product: "Stereo-X1", id: "SN-0042" }] };
  const dest = `${root}/stereo-sn0057/type-seed.json`;
  const written = await command(request, "export_seed", { from, dest });
  expect(written.status).toBe(200);
  expect((written.body as unknown as ExportedSeed).path).toBe(dest);
  const again = await command(request, "export_seed", { from, dest });
  expect(again.status).toBe(409);
  expect((again.body as unknown as StudioError).kind).toBe("file_exists");
  const toml = await command(request, "export_seed", {
    from,
    dest: `${root}/stereo-sn0057/type-seed.toml`,
  });
  expect(toml.status).toBe(200);

  // The project's [seed]: the library refuses a seed of another camera
  // model before anything is written.
  const dir = `${root}/stereo-sn0057`;
  const opened = (await command(request, "open_project", { dir }))
    .body as unknown as ProjectView;
  const refused = await command(request, "set_project_seed", {
    dir,
    revision: opened.revision,
    seed: { file: "type-seed.json" },
  });
  expect(refused.status).toBe(422);
  expect(refused.body).toMatchObject({ kind: "seed", field: "seed" });

  // With the project's camera model undistorted, the seed fits.
  const original = opened.spec;
  const undistorted = await command(request, "save_project", {
    dir,
    revision: opened.revision,
    spec: { ...original, config: { camera: { model: { distortion: "none" } } } },
  });
  expect(undistorted.status).toBe(200);
  const set = await command(request, "set_project_seed", {
    dir,
    revision: (undistorted.body as unknown as ProjectView).revision,
    seed: { file: "type-seed.json" },
  });
  expect(set.status).toBe(200);
  const seeded = set.body as unknown as ProjectView;
  expect(seeded.spec.seed).toEqual({ file: "type-seed.json" });
  expect(seeded.input_digest?.seed).toMatch(/^sha256:/);

  // Cleared, and the project's configuration put back.
  const cleared = await command(request, "set_project_seed", {
    dir,
    revision: seeded.revision,
    seed: null,
  });
  expect(cleared.status).toBe(200);
  const view = cleared.body as unknown as ProjectView;
  expect(view.spec.seed).toBeUndefined();
  const restored = await command(request, "save_project", {
    dir,
    revision: view.revision,
    spec: original,
  });
  expect(restored.status).toBe(200);
});
