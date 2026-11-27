// @vitest-environment jsdom
/** The maps as queries: a plane is decoded once however often it is asked,
 * and a blob the store dropped is read again after the command runs again. */
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { setTransport } from "../../../api";
import { createMockTransport, type MockTransport } from "../../../api/transport/mock";
import { queryCache } from "../../../session/query";
import { distortionPlane, distortionReply, vam1 } from "../../../test/planeFixtures";
import { readPlane, useDistortionMap } from "./useMaps";

const RUN = { dir: "/work/cell", id: "run-001" };
let mock: MockTransport;
let blobs: Record<string, number>;
let expired: Set<string>;

beforeEach(() => {
  queryCache.clear();
  blobs = {};
  expired = new Set();
  mock = createMockTransport({
    commands: { distortion_map: () => distortionReply() },
    resources: { blob: ({ id }) => `mock://blob/${id}` },
  });
  setTransport(mock);
  vi.stubGlobal("fetch", (url: string) => {
    const id = url.split("/").pop() ?? "";
    blobs[id] = (blobs[id] ?? 0) + 1;
    if (expired.has(id)) {
      expired.delete(id);
      return Promise.resolve(
        new Response(JSON.stringify({ kind: "blob_expired", message: "gone" }), {
          status: 404,
        }),
      );
    }
    return Promise.resolve(new Response(vam1(distortionPlane())));
  });
});

afterEach(() => {
  setTransport(null);
  vi.unstubAllGlobals();
});

describe("useDistortionMap", () => {
  it("joins the command's grid and channels to the decoded plane, once", async () => {
    const first = renderHook(() => useDistortionMap(RUN, 0, "distortion"));
    await waitFor(() => expect(first.result.current.state.source).not.toBeNull());
    const source = first.result.current.state.source!;
    expect(source.grid.columns).toBe(4);
    expect(source.channels.map((c) => c.name)).toEqual(["dx", "dy", "magnitude"]);
    expect(source.plane.width).toBe(4);

    // Another tab asking for the same layer reads the cache, not the wire.
    const second = renderHook(() => useDistortionMap(RUN, 0, "distortion"));
    await waitFor(() => expect(second.result.current.state.source).not.toBeNull());
    expect(blobs["lens"]).toBe(1);
    expect(mock.calls.filter((c) => c.command === "distortion_map")).toHaveLength(1);
  });

  it("reads each layer's plane of its own", async () => {
    const lens = renderHook(() => useDistortionMap(RUN, 0, "distortion"));
    await waitFor(() => expect(lens.result.current.state.source).not.toBeNull());
    const total = renderHook(() => useDistortionMap(RUN, 0, "total"));
    await waitFor(() => expect(total.result.current.state.source).not.toBeNull());
    expect(Object.keys(blobs).sort()).toEqual(["lens", "total"]);
  });

  it("says loading, then why, when the command fails", async () => {
    setTransport(
      createMockTransport({
        commands: {
          distortion_map: () => {
            throw new Error("no such run");
          },
        },
      }),
    );
    const { result } = renderHook(() => useDistortionMap(RUN, 0, "distortion"));
    expect(result.current.state.loading).toBe(true);
    await waitFor(() => expect(result.current.state.error).not.toBeNull());
    expect(result.current.state.loading).toBe(false);
    expect(result.current.state.source).toBeNull();
  });
});

describe("readPlane", () => {
  const blob = { id: "lens", len: 1, media_type: "application/x-vam1" };

  it("runs the command again when the store dropped the blob", async () => {
    expired.add("lens");
    const rerun = vi.fn(() => Promise.resolve(blob));
    const plane = await readPlane(blob, rerun);
    expect(rerun).toHaveBeenCalledTimes(1);
    expect(plane.width).toBe(4);
    expect(blobs["lens"]).toBe(2);
  });

  it("does not, for any other failure", async () => {
    vi.stubGlobal("fetch", () =>
      Promise.resolve(
        new Response(JSON.stringify({ kind: "internal", message: "boom" }), {
          status: 500,
        }),
      ),
    );
    const rerun = vi.fn(() => Promise.resolve(blob));
    await expect(readPlane(blob, rerun)).rejects.toMatchObject({ message: "boom" });
    expect(rerun).not.toHaveBeenCalled();
  });
});
