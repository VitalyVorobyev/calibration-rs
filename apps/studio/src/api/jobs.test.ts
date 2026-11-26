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
        run_calibration: (_args, control) => {
          control.emit({ stage: "detect" });
          control.emit({ stage: "solve" });
          return { kind: "cancelled" };
        },
      },
    });
    const { transport, open } = counted(mock);
    const api = client(transport);
    const job = await api.jobs.runCalibration({
      manifest: {},
      config: {},
      manifest_dir: "/d",
    });
    expect(open()).toBe(1);
    const output = await job.result;
    const stages: string[] = [];
    job.onEvent((progress) => stages.push(progress.stage));
    expect(stages).toEqual(["detect", "solve"]);
    expect(output).toEqual({ kind: "cancelled" });
    expect(open()).toBe(0);
    expect(mock.calls.map((c) => c.command)).toEqual(["run_calibration"]);
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
    const mock = createMockTransport({ jobs: { fetch_sample: () => "/cache/m.toml" } });
    const api = client(mock);
    const started = await api.jobs.fetchSample({ id: "kuka_1" });
    await started.result;
    // A fresh client (a reloaded page) follows it by its id.
    const again = await client(mock).jobs.attach<null, string>(started.id);
    await expect(again.result).resolves.toBe("/cache/m.toml");
    const gone = await client(mock).jobs.attach<null, string>("9-none");
    await expect(gone.result).rejects.toMatchObject({ kind: "job_not_found" });
  });
});
