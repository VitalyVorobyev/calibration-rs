// @vitest-environment jsdom
/** The HTTP transport's contract with `calibration-studio-server`: commands are posted
 * to `/api/cmd/<name>` as JSON, a failure rejects with the error envelope,
 * a resource's query travels as encoded JSON, and the event stream resumes
 * after the last event seen. */
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ServerEvent } from "../generated/types";
import { isNetworkError, StudioError } from "../transport";
import { httpTransport, resourceUrl } from "./http";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("the HTTP transport", () => {
  it("posts a command's argument object and answers its JSON", async () => {
    const fetchMock = vi.fn((_url: string, _init?: RequestInit) =>
      Promise.resolve(jsonResponse([[1, 2]])),
    );
    vi.stubGlobal("fetch", fetchMock);
    const args = { camera: 1, points_px: [[3, 4]] };
    expect(await httpTransport.invoke("undistort_points", args)).toEqual([[1, 2]]);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/cmd/undistort_points");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual(args);
  });

  it("rejects with the error envelope, or with the status when there is none", async () => {
    vi.stubGlobal("fetch", () =>
      Promise.resolve(
        jsonResponse({ kind: "forbidden_path", message: "out", path: "/etc" }, 403),
      ),
    );
    const error = await httpTransport
      .invoke("read_manifest", {})
      .catch((e: unknown) => e);
    expect(error).toBeInstanceOf(StudioError);
    expect(error).toMatchObject({ kind: "forbidden_path", message: "out", path: "/etc" });
    expect(String(error)).toBe("out");

    vi.stubGlobal("fetch", () =>
      Promise.resolve(new Response("oops", { status: 502, statusText: "Bad Gateway" })),
    );
    await expect(httpTransport.invoke("x", {})).rejects.toMatchObject({
      kind: "internal",
      message: "502 Bad Gateway",
    });
  });

  it("puts a resource's query in the URL as encoded JSON", () => {
    const query = { path: "/r/a b&c.png", camera: 0, roi: null };
    const url = resourceUrl("undistorted_image", query);
    expect(url.startsWith("/api/r/undistorted_image?q=")).toBe(true);
    const q = new URL(url, "http://h").searchParams.get("q");
    expect(JSON.parse(q ?? "")).toEqual(query);

    expect(httpTransport.resourceUrl("undistorted_image", query)).toBe(url);
  });

  it("subscribes to the event stream after the last event seen", () => {
    const sources: FakeEventSource[] = [];
    class FakeEventSource {
      onmessage: ((m: MessageEvent<string>) => void) | null = null;
      closed = false;
      constructor(readonly url: string) {
        sources.push(this);
      }
      addEventListener() {}
      close() {
        this.closed = true;
      }
      send(event: ServerEvent) {
        this.onmessage?.({ data: JSON.stringify(event) } as MessageEvent<string>);
      }
    }
    vi.stubGlobal("EventSource", FakeEventSource);
    const seen: number[] = [];
    const fresh = httpTransport.subscribe((e) => seen.push(e.seq));
    expect(sources[0]!.url).toBe("/api/events");
    const body: ServerEvent["body"] = {
      kind: "job",
      id: "1-ab",
      command: "calibrate",
      state: { kind: "running" },
    };
    sources[0]!.send({ seq: 4, body });
    sources[0]!.send({ seq: 3, body }); // older than what was seen
    sources[0]!.send({ seq: 5, body });
    expect(seen).toEqual([4, 5]);
    fresh.close();
    expect(sources[0]!.closed).toBe(true);

    httpTransport.subscribe(() => {}, 5);
    expect(sources[1]!.url).toBe("/api/events?after=5");
  });

  it("reports the stream's drops and returns, and reopens one the browser gave up", () => {
    vi.useFakeTimers();
    const sources: FakeEventSource[] = [];
    class FakeEventSource {
      onmessage: ((m: MessageEvent<string>) => void) | null = null;
      readyState = 0;
      private readonly handlers = new Map<string, (() => void)[]>();
      constructor(readonly url: string) {
        sources.push(this);
      }
      addEventListener(type: string, handler: () => void) {
        this.handlers.set(type, [...(this.handlers.get(type) ?? []), handler]);
      }
      fire(type: string) {
        for (const handler of this.handlers.get(type) ?? []) handler();
      }
      close() {
        this.readyState = 2;
      }
      send(event: ServerEvent) {
        this.onmessage?.({ data: JSON.stringify(event) } as MessageEvent<string>);
      }
    }
    vi.stubGlobal("EventSource", FakeEventSource);
    const status: boolean[] = [];
    const body: ServerEvent["body"] = { kind: "project_changed", dir: "/p" };
    const subscription = httpTransport.subscribe(
      () => {},
      undefined,
      (open) => status.push(open),
    );
    sources[0]!.fire("open");
    sources[0]!.send({ seq: 7, body });
    // A dropped connection: the browser retries it (still connecting).
    sources[0]!.fire("error");
    sources[0]!.fire("open");
    expect(status).toEqual([true, false, true]);
    expect(sources).toHaveLength(1);
    // An answer that is not a stream closes the source: a new one, after
    // the last event, once the delay has passed.
    sources[0]!.readyState = 2;
    sources[0]!.fire("error");
    expect(sources).toHaveLength(1);
    vi.advanceTimersByTime(2000);
    expect(sources).toHaveLength(2);
    expect(sources[1]!.url).toBe("/api/events?after=7");
    // Closing stops the retries.
    sources[1]!.readyState = 2;
    sources[1]!.fire("error");
    subscription.close();
    vi.advanceTimersByTime(10_000);
    expect(sources).toHaveLength(2);
    vi.useRealTimers();
  });

  it("marks a failure to reach the server as a network error", async () => {
    vi.stubGlobal("fetch", () => Promise.reject(new TypeError("Failed to fetch")));
    const error = await httpTransport.invoke("server_info", {}).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(StudioError);
    expect(isNetworkError(error)).toBe(true);
    vi.stubGlobal("fetch", () =>
      Promise.resolve(jsonResponse({ kind: "io", message: "disk" }, 500)),
    );
    const answered = await httpTransport.invoke("x", {}).catch((e: unknown) => e);
    expect(isNetworkError(answered)).toBe(false);
  });
});
