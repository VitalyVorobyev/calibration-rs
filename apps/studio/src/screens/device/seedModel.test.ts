import { describe, expect, it } from "vitest";
import { PREVIEW_RUN, SN1, SN2 } from "./__fixtures__/deviceFixtures";
import { runKey, unitKey } from "./deviceModel";
import {
  describeSource,
  exportDest,
  runSeedSource,
  seedable,
  seedRequest,
  seedValues,
  type SeedChoice,
} from "./seedModel";

const choice = (c: Partial<SeedChoice>): SeedChoice => ({
  kind: "run",
  run: null,
  runs: [],
  units: [],
  ...c,
});
const runs = SN1.runs;
const rep = runKey("/data/bench", "run-002");

describe("what a source asks for", () => {
  it("a run", () => {
    expect(seedRequest(choice({ run: rep }), runs, [SN1])).toEqual({
      from: { kind: "run", run: { dir: "/data/bench", id: "run-002" } },
    });
    expect(seedRequest(choice({}), runs, [SN1])).toEqual({
      problem: "Choose the run to seed from.",
    });
  });

  it("chosen runs, two at least", () => {
    const one = seedRequest(choice({ kind: "runs", runs: [rep] }), runs, [SN1]);
    expect("problem" in one && one.problem).toMatch(/at least two runs/);
    const two = seedRequest(
      choice({ kind: "runs", runs: [rep, runKey("/data/field", "run-001")] }),
      runs,
      [SN1],
    );
    expect(two).toEqual({
      from: {
        kind: "runs",
        runs: [
          { dir: "/data/bench", id: "run-002" },
          { dir: "/data/field", id: "run-001" },
        ],
      },
    });
  });

  it("the type's units, each with a representative run", () => {
    const stray = { ...SN2, device: { id: "SN-0099", product: "Stereo-X1" }, runs: [] };
    const request = seedRequest(
      choice({
        kind: "units",
        units: [SN1, SN2, stray].map((u) => unitKey(u.device)),
      }),
      runs,
      [SN1, SN2, stray],
    );
    expect(request).toEqual({
      from: {
        kind: "units",
        devices: [
          { id: "SN-0042", product: "Stereo-X1" },
          { id: "SN-0057", product: "Stereo-X1" },
        ],
      },
    });
    const few = seedRequest(
      choice({ kind: "units", units: [unitKey(SN1.device)] }),
      runs,
      [SN1],
    );
    expect("problem" in few && few.problem).toMatch(/at least two units/);
  });

  it("offers the runs whose files read", () => {
    expect(runs.map(seedable)).toEqual([true, true, true, false]);
  });
});

describe("the seed's values", () => {
  it("are the intrinsics, distortion without its iteration count, and the sensor", () => {
    const rows = seedValues(PREVIEW_RUN.seed).map(
      (r) => `${r.camera}.${r.label}=${r.value}`,
    );
    expect(rows).toEqual([
      "cam0.fx=200.6",
      "cam0.fy=200.4",
      "cam0.cx=48",
      "cam0.cy=36",
      "cam0.k1=-0.1",
      "cam0.k2=0",
      "cam0.k3=0",
      "cam0.p1=0",
      "cam0.p2=0",
    ]);
  });

  it("carry a mount, a laser plane and the hand-eye", () => {
    const seed = {
      ...PREVIEW_RUN.seed,
      handeye: {
        mode: "eye_in_hand" as const,
        gripper_se3_rig: {
          rotation: [0, 0, 0, 1] as [number, number, number, number],
          translation: [0.1, 0, 0] as [number, number, number],
        },
      },
      cameras: [
        {
          ...PREVIEW_RUN.seed.cameras[0]!,
          laser_plane: { normal: [0, 1, 0] as [number, number, number], distance: 0.2 },
        },
      ],
    };
    const labels = seedValues(seed).map((r) => r.label);
    expect(labels).toContain("laser_plane.distance");
    expect(labels).toContain("gripper_se3_rig.tx");
  });
});

describe("the file", () => {
  it("is the folder and the name, its format by the extension", () => {
    expect(exportDest("/data/p", "seed.json")).toEqual({
      dest: "/data/p/seed.json",
      format: "json",
    });
    expect(exportDest("/data/p/", " seed.TOML ")).toEqual({
      dest: "/data/p/seed.TOML",
      format: "toml",
    });
  });

  it("says what is wrong with the name", () => {
    const problem = (folder: string | null, name: string) => {
      const r = exportDest(folder, name);
      return "problem" in r ? r.problem : null;
    };
    expect(problem(null, "a.json")).toMatch(/folder/);
    expect(problem("/d", " ")).toMatch(/Name the file/);
    expect(problem("/d", "a/b.json")).toMatch(/not a path/);
    expect(problem("/d", "seed.txt")).toMatch(/\.json or \.toml/);
  });
});

describe("the project's seed", () => {
  it("is a run's folder for one run, a file for anything else", () => {
    expect(
      runSeedSource({ kind: "run", run: { dir: "/data/bench", id: "run-002" } }),
    ).toEqual({ run: "/data/bench/runs/run-002" });
    expect(runSeedSource({ kind: "runs", runs: [] })).toBeNull();
  });

  it("is described in words", () => {
    expect(describeSource(null)).toBe("none");
    expect(describeSource({ file: "seeds/x.json" })).toBe("file seeds/x.json");
    expect(describeSource({ run: "/r" })).toBe("run /r");
    expect(describeSource({ device_spec: "s.json" })).toBe("device specification s.json");
  });
});
