import { describe, expect, it } from "vitest";
import type { RunInfo } from "../api/generated/types";
import { calibratedRun, DIGEST } from "../test/projectFixtures";
import { runPills } from "./runPills";

function withVerdict(run: RunInfo, verdict: "pass" | "warn" | "fail" | null): RunInfo {
  if (run.state.kind !== "recorded") throw new Error("fixture");
  return {
    ...run,
    state: { ...run.state, summary: { ...run.state.summary, verdict } },
  };
}

describe("runPills", () => {
  it("says a current, passing run is Recorded and Pass, with nothing stale", () => {
    const pills = runPills(calibratedRun("/p", "run-001"), DIGEST);
    expect(pills.state).toMatchObject({ label: "Recorded", tone: "neutral" });
    expect(pills.verdict).toEqual({ label: "Pass", tone: "normal" });
    expect(pills.stale).toBeNull();
  });

  it("says each verdict in words and in its tone", () => {
    const run = calibratedRun("/p", "run-001");
    expect(runPills(withVerdict(run, "warn"), DIGEST).verdict).toEqual({
      label: "Check",
      tone: "warning",
    });
    expect(runPills(withVerdict(run, "fail"), DIGEST).verdict).toEqual({
      label: "Fail",
      tone: "defect",
    });
    expect(runPills(withVerdict(run, null), DIGEST).verdict).toBeNull();
  });

  it("is stale in the words of the inputs that changed", () => {
    const pills = runPills(calibratedRun("/p", "run-001"), {
      ...DIGEST,
      dataset: "sha256:other",
      config: "sha256:other",
    });
    expect(pills.stale?.label).toBe("Stale");
    expect(pills.stale?.title).toBe(
      "The project's dataset, configuration changed since this run was made.",
    );
  });

  it("cannot be stale when the project's inputs do not resolve", () => {
    expect(runPills(calibratedRun("/p", "run-001"), null).stale).toBeNull();
  });

  it("warns that an imported or earlier run is limited, and marks an unreadable one", () => {
    const run = calibratedRun("/p", "run-001");
    if (run.state.kind !== "recorded") throw new Error("fixture");
    const imported: RunInfo = {
      ...run,
      state: { ...run.state, source: { kind: "imported", file_name: "e.json" } },
    };
    expect(runPills(imported, DIGEST).state.label).toBe("Imported");
    expect(runPills(imported, DIGEST).state.title).toMatch(/Limited/);
    expect(runPills(imported, DIGEST).stale).toBeNull();
    const unreadable: RunInfo = {
      ...run,
      state: { kind: "unreadable", message: "v9" },
    };
    expect(runPills(unreadable, DIGEST).state).toMatchObject({
      label: "Unreadable",
      tone: "defect",
    });
    expect(runPills(unreadable, DIGEST).verdict).toBeNull();
  });
});
