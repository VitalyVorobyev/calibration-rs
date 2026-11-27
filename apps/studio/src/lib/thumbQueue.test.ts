import { describe, expect, it } from "vitest";
import { ThumbQueue, type ThumbLoader } from "./thumbQueue";

/** A loader whose loads finish when the test says so. */
function manualLoader() {
  const started: string[] = [];
  const released: string[] = [];
  const settle = new Map<string, { ok: (s: string) => void; fail: (e: Error) => void }>();
  const loader: ThumbLoader = {
    load: (url) =>
      new Promise((ok, fail) => {
        started.push(url);
        settle.set(url, { ok, fail });
      }),
    release: (src) => released.push(src),
  };
  return { loader, started, released, settle };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

describe("ThumbQueue", () => {
  it("runs at most `limit` loads at a time, newest request first", async () => {
    const { loader, started, settle } = manualLoader();
    const q = new ThumbQueue(loader, 2);
    for (const u of ["a", "b", "c", "d"]) q.subscribe(u, () => {});
    // a and b took the free slots as they were asked for; c and d wait.
    expect(started).toEqual(["a", "b"]);
    settle.get("a")!.ok("blob:a");
    await flush();
    // The next to start is the most recently asked for.
    expect(started).toEqual(["a", "b", "d"]);
    expect(q.get("a")).toEqual({ kind: "ready", src: "blob:a" });
    expect(q.get("c")).toEqual({ kind: "idle" });
  });

  it("never starts a load nobody waits for any more", () => {
    const { loader, started } = manualLoader();
    const q = new ThumbQueue(loader, 1);
    q.subscribe("a", () => {});
    const stop = q.subscribe("b", () => {});
    stop();
    expect(started).toEqual(["a"]);
    expect(q.get("b")).toEqual({ kind: "idle" });
  });

  it("tells its listeners and keeps a failure's reason", async () => {
    const { loader, settle } = manualLoader();
    const q = new ThumbQueue(loader, 1);
    let heard = 0;
    q.subscribe("a", () => heard++);
    settle.get("a")!.fail(new Error("404 Not Found"));
    await flush();
    expect(q.get("a")).toEqual({ kind: "error", message: "404 Not Found" });
    expect(heard).toBe(2); // loading, error
  });

  it("drops the least recently used answers nobody shows, releasing them", async () => {
    const { loader, released, settle } = manualLoader();
    const q = new ThumbQueue(loader, 1, 2);
    for (const u of ["a", "b", "c"]) {
      const stop = q.subscribe(u, () => {});
      settle.get(u)!.ok(`blob:${u}`);
      await flush();
      stop();
    }
    expect(released).toEqual(["blob:a"]);
    expect(q.get("a")).toEqual({ kind: "idle" });
    expect(q.get("c")).toEqual({ kind: "ready", src: "blob:c" });
  });
});
