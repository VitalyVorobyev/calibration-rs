/** The resource readers against the back end's own bytes: the golden value
 * plane studio-core writes (`planes.rs`, `__fixtures__/golden.vam1`) and
 * the error envelope. */
import { afterEach, describe, expect, it, vi } from "vitest";
import goldenUrl from "./__fixtures__/golden.vam1?inline";
import { fetchBytes, fetchFloats, fetchPlane } from "./resources";

// The file inlined as a `data:` URL, decoded here.
const golden = Uint8Array.from(atob(goldenUrl.slice(goldenUrl.indexOf(",") + 1)), (c) =>
  c.charCodeAt(0),
);

function serve(body: BodyInit | null, init?: ResponseInit) {
  vi.stubGlobal("fetch", () => Promise.resolve(new Response(body, init)));
}

afterEach(() => vi.unstubAllGlobals());

describe("resources", () => {
  it("decodes the plane studio-core encodes", async () => {
    serve(golden);
    const plane = await fetchPlane("/api/r/blob?q={}");
    expect([plane.width, plane.height, plane.stride, plane.channels]).toEqual([
      3, 2, 1, 2,
    ]);
    const values = Array.from(plane.values);
    expect(values.slice(0, 4)).toEqual([0, 1.5, -2, 3.25]);
    expect(Number.isNaN(values[4])).toBe(true);
    expect(values.slice(5)).toEqual([1e6, 10, 20, 30, 40, 50, 60]);
  });

  it("reads raw floats and refuses a ragged buffer", async () => {
    serve(new Float32Array([1, 2.5]).buffer);
    expect(Array.from(await fetchFloats("/f"))).toEqual([1, 2.5]);
    serve(new Uint8Array([1, 2, 3]));
    await expect(fetchFloats("/f")).rejects.toMatchObject({ kind: "internal" });
  });

  it("rejects with the back end's envelope", async () => {
    serve(JSON.stringify({ kind: "blob_expired", message: "gone" }), { status: 404 });
    await expect(fetchBytes("/b")).rejects.toMatchObject({
      kind: "blob_expired",
      message: "gone",
    });
    serve("nope", { status: 502, statusText: "Bad Gateway" });
    await expect(fetchBytes("/b")).rejects.toMatchObject({ message: "502 Bad Gateway" });
  });
});
