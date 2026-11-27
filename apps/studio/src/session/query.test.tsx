// @vitest-environment jsdom
/** The query cache: one fetch per key however many panels ask, the last
 * good answer kept under an error or an invalidation, and a project's
 * entries marked stale when the back end says it changed. */
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioError } from "../api";
import { QueryCache, queryCache, queryKey, useQuery } from "./query";

afterEach(() => {
  cleanup();
  queryCache.clear();
});

describe("queryKey", () => {
  it("is the same for the same arguments in any key order", () => {
    expect(queryKey("c", { a: 1, b: { y: 2, x: 3 } })).toBe(
      queryKey("c", { b: { x: 3, y: 2 }, a: 1 }),
    );
    expect(queryKey("c", { a: 1 })).not.toBe(queryKey("d", { a: 1 }));
    expect(queryKey("c", undefined)).toBe(queryKey("c", null));
  });
});

describe("QueryCache", () => {
  it("fetches once for concurrent askers and holds the answer", async () => {
    const cache = new QueryCache();
    const fetcher = vi.fn(() => Promise.resolve(["run-001"]));
    await Promise.all([
      cache.fetch("list_runs", { dir: "/p" }, fetcher),
      cache.fetch("list_runs", { dir: "/p" }, fetcher),
    ]);
    await cache.fetch("list_runs", { dir: "/p" }, fetcher);
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(cache.get("list_runs", { dir: "/p" })).toMatchObject({
      data: ["run-001"],
      loading: false,
      stale: false,
      error: null,
    });
  });

  it("keeps the last good answer, marked stale, when the project changes", async () => {
    const cache = new QueryCache();
    let n = 0;
    const fetcher = () => Promise.resolve(++n);
    await cache.fetch("open_project", { dir: "/p" }, fetcher);
    await cache.fetch("open_project", { dir: "/other" }, fetcher);
    cache.invalidateProject("/p");
    expect(cache.get("open_project", { dir: "/p" })).toMatchObject({
      data: 1,
      stale: true,
    });
    expect(cache.get("open_project", { dir: "/other" }).stale).toBe(false);
    await cache.fetch("open_project", { dir: "/p" }, fetcher);
    expect(cache.get("open_project", { dir: "/p" })).toMatchObject({
      data: 3,
      stale: false,
    });
  });

  it("matches a project named by a run reference", async () => {
    const cache = new QueryCache();
    await cache.fetch("run_export", { run: { dir: "/p", id: "run-001" } }, () =>
      Promise.resolve("export"),
    );
    cache.invalidateProject("/p");
    expect(cache.get("run_export", { run: { dir: "/p", id: "run-001" } }).stale).toBe(
      true,
    );
  });

  it("keeps the data and reports the error when a refetch fails", async () => {
    const cache = new QueryCache();
    await cache.fetch("c", {}, () => Promise.resolve(7));
    cache.invalidate(() => true);
    await cache.fetch("c", {}, () =>
      Promise.reject(new StudioError({ kind: "io", message: "disk" })),
    );
    expect(cache.get("c", {})).toMatchObject({ data: 7, loading: false });
    expect(cache.get("c", {}).error?.message).toBe("disk");
  });

  it("fetches again after an invalidation that arrived mid-fetch", async () => {
    const cache = new QueryCache();
    let release!: (v: number) => void;
    const first = cache.fetch(
      "c",
      {},
      () => new Promise<number>((resolve) => (release = resolve)),
    );
    cache.invalidate(() => true);
    release(1);
    await first;
    expect(cache.get("c", {})).toMatchObject({ data: 1, stale: true });
  });

  it("tells subscribers when an entry changes", async () => {
    const cache = new QueryCache();
    const listener = vi.fn();
    cache.subscribe(listener);
    await cache.fetch("c", {}, () => Promise.resolve(1));
    expect(listener).toHaveBeenCalled();
  });
});

function Probe({ enabled = true }: { enabled?: boolean }) {
  const q = useQuery(
    "list_runs",
    { dir: "/p" },
    () => Promise.resolve(["a", "b"]),
    enabled,
  );
  return (
    <p>
      {q.loading ? "loading" : q.data ? q.data.join(",") : "none"}
      {q.stale ? " stale" : ""}
    </p>
  );
}

describe("useQuery", () => {
  it("fetches, shows the answer, and refetches after an invalidation", async () => {
    render(<Probe />);
    await waitFor(() => expect(screen.getByText("a,b")).toBeTruthy());
    act(() => queryCache.invalidateProject("/p"));
    await waitFor(() => expect(screen.getByText("a,b")).toBeTruthy());
    expect(queryCache.get("list_runs", { dir: "/p" }).stale).toBe(false);
  });

  it("waits while disabled", async () => {
    render(<Probe enabled={false} />);
    await act(async () => {
      await Promise.resolve();
    });
    expect(screen.getByText(/none/)).toBeTruthy();
  });
});
