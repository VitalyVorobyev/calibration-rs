import { describe, expect, it } from "vitest";
import type { JobEvent } from "../api/generated/types";
import { calibratedRun } from "../test/projectFixtures";
import {
  applyJobEvent,
  describeJob,
  indicatorLine,
  isLive,
  jobTitle,
  mergeListed,
  pruneFinished,
  resultChip,
  trayOrder,
  type JobRow,
} from "./jobModel";

const event = (
  id: string,
  state: JobEvent["state"],
  progress?: unknown,
  command = "calibrate",
): JobEvent => ({ id, command, state, event: progress ?? null });

function rowsOf(...events: JobEvent[]): JobRow[] {
  return events.reduce<JobRow[]>((rows, e) => applyJobEvent(rows, e), []);
}

describe("applyJobEvent", () => {
  it("adds a job on its first event and advances it on the next", () => {
    const rows = rowsOf(
      event("1-a", { kind: "queued", position: 1 }),
      event("1-a", { kind: "running" }, { kind: "detection", done: 3, total: 12 }),
    );
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ id: "1-a", state: { kind: "running" } });
    expect(rows[0]!.event).toEqual({ kind: "detection", done: 3, total: 12 });
  });

  it("keeps the last report when an event carries none", () => {
    const rows = rowsOf(
      event("1-a", { kind: "running" }, { kind: "step", name: "optimize" }),
      event("1-a", { kind: "running" }),
    );
    expect(rows[0]!.event).toEqual({ kind: "step", name: "optimize" });
  });

  it("collects a calibration's solver cost, in order", () => {
    const iter = (i: number, cost: number) => ({
      kind: "iteration",
      iter: i,
      max_iters: 50,
      cost,
    });
    const rows = rowsOf(
      event("1-a", { kind: "running" }, iter(1, 9)),
      event("1-a", { kind: "running" }, iter(2, 3)),
      event("2-b", { kind: "running" }, iter(1, 1), "disparity"),
    );
    expect(rows.find((r) => r.id === "1-a")!.costs).toEqual([9, 3]);
    expect(rows.find((r) => r.id === "2-b")!.costs).toEqual([]);
  });

  it("orders jobs by when they were first heard", () => {
    const rows = rowsOf(
      event("1-a", { kind: "running" }),
      event("2-b", { kind: "queued", position: 1 }),
    );
    expect(rows.map((r) => r.id)).toEqual(["1-a", "2-b"]);
    expect(rows[1]!.order).toBeGreaterThan(rows[0]!.order);
  });

  it("clears a cancel request once the job has ended", () => {
    const rows = rowsOf(event("1-a", { kind: "running" }));
    const asked = rows.map((r) => ({ ...r, cancelling: true }));
    expect(applyJobEvent(asked, event("1-a", { kind: "running" }))[0]!.cancelling).toBe(
      true,
    );
    expect(applyJobEvent(asked, event("1-a", { kind: "cancelled" }))[0]!.cancelling).toBe(
      false,
    );
  });
});

describe("mergeListed", () => {
  it("adds jobs the stream never showed and keeps a delivered end", () => {
    const rows = rowsOf(event("1-a", { kind: "cancelled" }));
    const merged = mergeListed(rows, [
      { id: "1-a", command: "calibrate", state: { kind: "running" } },
      {
        id: "2-b",
        command: "calibrate",
        state: { kind: "running" },
        last_event: { kind: "step", name: "init" },
      },
    ]);
    expect(merged.find((r) => r.id === "1-a")!.state.kind).toBe("cancelled");
    expect(merged.find((r) => r.id === "2-b")!.event).toEqual({
      kind: "step",
      name: "init",
    });
  });

  it("takes a live job's newer state", () => {
    const rows = rowsOf(event("1-a", { kind: "queued", position: 2 }));
    const merged = mergeListed(rows, [
      { id: "1-a", command: "calibrate", state: { kind: "running" } },
    ]);
    expect(merged[0]!.state.kind).toBe("running");
  });
});

describe("pruneFinished", () => {
  it("keeps every live job and the last three finished", () => {
    const rows = rowsOf(
      ...["a", "b", "c", "d", "e"].map((id) => event(id, { kind: "cancelled" })),
      event("f", { kind: "running" }),
    );
    const kept = pruneFinished(rows);
    expect(kept.map((r) => r.id)).toEqual(["c", "d", "e", "f"]);
    expect(kept.filter((r) => isLive(r.state))).toHaveLength(1);
  });

  it("keeps a failure however many jobs end after it", () => {
    const rows = rowsOf(
      event("x", { kind: "failed", error: { kind: "calibration", message: "no" } }),
      ...["a", "b", "c", "d", "e"].map((id) => event(id, { kind: "cancelled" })),
    );
    expect(pruneFinished(rows).map((r) => r.id)).toEqual(["x", "c", "d", "e"]);
  });
});

describe("describeJob", () => {
  const running = (progress: unknown) =>
    rowsOf(event("1-a", { kind: "running" }, progress))[0]!;

  it("counts detections", () => {
    expect(describeJob(running({ kind: "detection", done: 41, total: 120 }))).toEqual({
      stage: "Detecting 41/120",
      progress: { done: 41, total: 120 },
    });
  });

  it("counts the images of detect_all, whose report has no kind", () => {
    const row = rowsOf(
      event(
        "1-a",
        { kind: "running" },
        { camera: 0, image: "a.png", done: 3, total: 12 },
        "detect_all",
      ),
    )[0]!;
    expect(describeJob(row)).toEqual({
      stage: "Detecting 3/12",
      progress: { done: 3, total: 12 },
    });
  });

  it("says the solver's iteration and cost", () => {
    expect(
      describeJob(running({ kind: "iteration", iter: 12, max_iters: 50, cost: 0.32 })),
    ).toEqual({
      stage: "Solving · iter 12 · cost 3.2e−1",
      progress: { done: 12, total: 50 },
    });
  });

  it("names a step, or just runs when it has no report", () => {
    expect(describeJob(running({ kind: "step", name: "Exporting" })).stage).toBe(
      "Exporting",
    );
    expect(describeJob(rowsOf(event("1-a", { kind: "running" }))[0]!).stage).toBe(
      "Running",
    );
  });

  it("says the queue position and a pending cancel", () => {
    expect(
      describeJob(rowsOf(event("1-a", { kind: "queued", position: 2 }))[0]!).stage,
    ).toBe("Queued · position 2");
    const cancelling = { ...running({ kind: "step", name: "x" }), cancelling: true };
    expect(describeJob(cancelling).stage).toBe("Cancelling…");
  });

  it("titles the known jobs and spells out the others", () => {
    expect(jobTitle("calibrate")).toBe("Calibration");
    expect(jobTitle("some_new_job")).toBe("some new job");
  });
});

describe("indicatorLine", () => {
  it("is nothing while idle", () => {
    expect(indicatorLine([])).toBeNull();
    expect(indicatorLine(rowsOf(event("1-a", { kind: "cancelled" })))).toBeNull();
  });

  it("shows the running job with how many wait behind it", () => {
    const rows = rowsOf(
      event("2-b", { kind: "queued", position: 1 }),
      event("1-a", { kind: "running" }, { kind: "detection", done: 1, total: 4 }),
    );
    expect(indicatorLine(rows)).toEqual({
      text: "Detecting 1/4 (+1)",
      progress: { done: 1, total: 4 },
    });
  });

  it("shows the queue when nothing runs yet", () => {
    expect(
      indicatorLine(rowsOf(event("1-a", { kind: "queued", position: 1 })))?.text,
    ).toBe("Queued · position 1");
  });
});

describe("trayOrder", () => {
  it("lists the running job, then the queue in its order, then what ended, newest first", () => {
    const rows = rowsOf(
      event("a", { kind: "cancelled" }),
      event("b", { kind: "queued", position: 2 }),
      event("c", { kind: "running" }),
      event("d", { kind: "queued", position: 1 }),
      event("e", { kind: "cancelled" }),
    );
    expect(trayOrder(rows).map((r) => r.id)).toEqual(["c", "b", "d", "e", "a"]);
  });
});

describe("resultChip", () => {
  it("names the run a calibration wrote, with its verdict", () => {
    const run = calibratedRun("/p", "run-008");
    const row = rowsOf(event("1-a", { kind: "done", output: run }))[0]!;
    expect(resultChip(row)).toEqual({
      runId: "run-008",
      projectDir: "/p",
      text: "run-008 · Pass",
    });
  });

  it("is nothing for another job or an unfinished one", () => {
    expect(resultChip(rowsOf(event("1-a", { kind: "running" }))[0]!)).toBeNull();
    expect(
      resultChip(
        rowsOf(event("1-a", { kind: "done", output: {} }, undefined, "disparity"))[0]!,
      ),
    ).toBeNull();
  });
});
