/** The Compare screen's back end over HTTP, on the fixture library: the two
 * runs of `stereo-sn0042` side by side (imported runs, so the relation is
 * unknown and every test says why it is not made), and one camera's
 * projection difference as a value plane behind the blob resource. Nothing
 * here drives the page. */
import { expect, test } from "@playwright/test";
import type {
  ProjectionDifferenceMap,
  RunComparison,
  StudioError,
} from "../../src/api/generated/types";
import { E2E_PORT, E2E_TOKEN, command, libraryRoot } from "./support";

test("two runs side by side, each test the relation does not support with its reason", async ({
  request,
}) => {
  const dir = `${await libraryRoot(request)}/stereo-sn0042`;
  const reply = await command(request, "compare_runs", {
    dir,
    runs: ["run-001", "run-002"],
  });
  expect(reply.status).toBe(200);
  const comparison = reply.body as unknown as RunComparison;
  expect(comparison.runs.map((r) => r.name)).toEqual(["run-001", "run-002"]);
  const [difference] = comparison.differences;
  expect(difference.relation).toEqual({ kind: "unknown" });
  expect(difference.cameras.map((c) => c.id)).toEqual(["cam0", "cam1"]);
  const parameters = difference.cameras[0].parameters;
  if (parameters.kind !== "same_model") throw new Error(parameters.kind);
  const fx = parameters.parameters.find((p) => p.label === "fx")!;
  expect([fx.reference, fx.other]).toEqual([200, 200.6]);
  expect(fx.test).toEqual({ kind: "not_available", reason: "unknown_relation" });

  const one = await command(request, "compare_runs", { dir, runs: ["run-001"] });
  expect(one.status).toBe(400);
  expect(one.body as unknown as StudioError).toMatchObject({
    kind: "invalid_args",
    field: "runs",
  });
});

test("a camera's projection difference is a plane loaded by URL", async ({ request }) => {
  const dir = `${await libraryRoot(request)}/stereo-sn0042`;
  const reply = await command(request, "projection_difference_map", {
    dir,
    reference: "run-001",
    other: "run-002",
    camera: "cam1",
  });
  expect(reply.status).toBe(200);
  const map = reply.body as unknown as ProjectionDifferenceMap;
  if (map.status !== "compared") throw new Error(map.status);
  expect(map.grid.image_size).toEqual([96, 72]);
  expect(map.plane.channels.map((c) => c.name)).toEqual([
    "dx",
    "dy",
    "magnitude",
    "fitted",
  ]);

  const q = encodeURIComponent(JSON.stringify({ id: map.plane.blob.id }));
  const blob = await request.get(`/api/r/blob?q=${q}`, {
    headers: {
      Authorization: `Bearer ${E2E_TOKEN}`,
      Origin: `http://127.0.0.1:${E2E_PORT}`,
    },
  });
  expect(blob.status()).toBe(200);
  expect(blob.headers()["content-type"]).toBe("application/x-vam1");
  expect(blob.headers()["content-length"]).toBe(String(map.plane.blob.len));
  expect((await blob.text()).startsWith("VAM1")).toBe(true);

  const missing = await command(request, "projection_difference_map", {
    dir,
    reference: "run-001",
    other: "run-002",
    camera: "cam9",
  });
  expect(missing.status).toBe(422);
  expect(missing.body as unknown as StudioError).toMatchObject({
    kind: "invalid",
    field: "camera",
  });
});
