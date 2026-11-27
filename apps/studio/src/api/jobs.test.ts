/** Following jobs through the event stream: progress reaches the job (the
 * reports sent before the first listener too), the output settles its
 * result, a failure and a cancellation reject it, and the one subscription
 * closes when no followed job is live. */
import { describe, expect, it } from "vitest";
import { createClient } from "./generated/client";
import { JobHub } from "./jobs";
import { createMockTransport } from "./transport/mock";
import { isCancelled, StudioError, type Transport } from "./transport";

function client(transport: Transport) {
  const hub = new JobHub(transport);
  return createClient(transport, hub);
}

/** A transport counting its open subscriptions. */
function counted(transport: ReturnType<typeof createMockTransport>) {
  let open = 0;
  const subscribe: Transport["subscribe"] = (onEvent, after) => {
    open += 1;
    const inner = transport.subscribe(onEvent, after);
    return {
      ready: inner.ready,
      close: () => {
        open -= 1;
        inner.close();
      },
    };
  };
  return { transport: { ...transport, subscribe }, open: () => open };
}

describe("jobs", () => {
  it("reports progress, replayed to a late listener, and settles with the output", async () => {
    const mock = createMockTransport({
      jobs: {
        calibrate: (args, control) => {
          control.emit({ kind: "detection", done: 1, total: 2 });
          control.emit({ kind: "step", name: "optimize" });
          return {
            id: "run-001",
            dir: `${args.dir}/runs/run-001`,
            notes: {},
            state: { kind: "unreadable", message: "test" },
          };
        },
      },
    });
    const { transport, open } = counted(mock);
    const api = client(transport);
    const job = await api.jobs.calibrate({ dir: "/p" });
    expect(open()).toBe(1);
    const output = await job.result;
    const stages: string[] = [];
    job.onEvent((progress) => stages.push(progress.kind));
    expect(stages).toEqual(["detection", "step"]);
    expect(output.id).toBe("run-001");
    expect(open()).toBe(0);
    expect(mock.calls.map((c) => c.command)).toEqual(["calibrate"]);
  });

  it("rejects with the job's error, and with `cancelled` once it is cancelled", async () => {
    const api = client(
      createMockTransport({
        jobs: {
          fetch_sample: async (args, control) => {
            if (args.id === "bad")
              throw new StudioError({ kind: "io", message: "offline" });
            await control.cancelled;
            throw new StudioError({ kind: "cancelled", message: "cancelled" });
          },
        },
      }),
    );
    const failing = await api.jobs.fetchSample({ id: "bad" });
    await expect(failing.result).rejects.toMatchObject({
      kind: "io",
      message: "offline",
    });

    const waiting = await api.jobs.fetchSample({ id: "kuka_1" });
    await waiting.cancel();
    const error = await waiting.result.catch((e: unknown) => e);
    expect(isCancelled(error)).toBe(true);
  });

  it("attaches to a job started earlier and finds it ended", async () => {
    const mock = createMockTransport({
      jobs: { fetch_sample: () => ({ dir: "/cache/kuka_1" }) },
    });
    const api = client(mock);
    const started = await api.jobs.fetchSample({ id: "kuka_1" });
    await started.result;
    // A fresh client (a reloaded page) follows it by its id.
    const again = await client(mock).jobs.attach<null, { dir: string }>(started.id);
    await expect(again.result).resolves.toEqual({ dir: "/cache/kuka_1" });
    const gone = await client(mock).jobs.attach<null, { dir: string }>("9-none");
    await expect(gone.result).rejects.toMatchObject({ kind: "job_not_found" });
  });
});

describe("the standing watch", () => {
  it("keeps the stream open with no job, and gives every event to its listeners", async () => {
    const mock = createMockTransport({
      jobs: { fetch_sample: () => ({ dir: "/cache/x" }) },
    });
    const { transport, open } = counted(mock);
    const hub = new JobHub(transport);
    const seen: string[] = [];
    hub.listen((e) => seen.push(e.body.kind));
    await hub.connect();
    expect(open()).toBe(1);
    mock.emit({ kind: "project_changed", dir: "/p" });
    expect(seen).toEqual(["project_changed"]);
    // A job that comes and goes does not close it.
    const job = await createClient(transport, hub).jobs.fetchSample({ id: "x" });
    await job.result;
    expect(open()).toBe(1);
    expect(seen).toContain("job");
    hub.disconnect();
    expect(open()).toBe(0);
  });

  it("tells the connection's state, the current one at once", async () => {
    const mock = createMockTransport();
    const hub = new JobHub(mock);
    const states: string[] = [];
    hub.onConnection((c) => states.push(c));
    await hub.connect();
    mock.setConnected(false);
    mock.setConnected(true);
    expect(states).toEqual(["connecting", "open", "lost", "open"]);
    hub.disconnect();
    expect(states[states.length - 1]).toBe("connecting");
  });

  it("stops hearing after its unsubscribe", async () => {
    const mock = createMockTransport();
    const hub = new JobHub(mock);
    const seen: number[] = [];
    const stop = hub.listen((e) => seen.push(e.seq));
    await hub.connect();
    mock.emit({ kind: "project_changed", dir: "/p" });
    stop();
    mock.emit({ kind: "project_changed", dir: "/p" });
    expect(seen).toEqual([1]);
  });
});
