import { describe, expect, it } from "vitest";
import type { DataRelation } from "../../api/generated/types";
import { comparison } from "./__fixtures__/compareFixtures";
import {
  cameraBlocks,
  maxProjectionSigma,
  pooledRms,
  positionName,
  residualsComparable,
  signed,
  testState,
  testTally,
  verdictFacts,
  visibleRows,
} from "./compareModel";
import { covarianceWords, refusalWords, relationWords } from "./compareWords";

describe("test states", () => {
  it("reads a p-value against 5 %, and a missing test as not available", () => {
    expect(testState({ kind: "independent", z: 2, p_value: 0.049, sigma: 1 })).toBe(
      "significant",
    );
    expect(testState({ kind: "nested", z: 1, p_value: 0.05, sigma: 1 })).toBe(
      "consistent",
    );
    expect(testState({ kind: "not_available", reason: "unknown_relation" })).toBe(
      "not_available",
    );
    expect(testState({ kind: "independent", chi2: 3, dof: 6, p_value: 0.8 })).toBe(
      "consistent",
    );
  });
});

describe("camera blocks", () => {
  const blocks = cameraBlocks(comparison());

  it("lays each parameter across the runs after the reference", () => {
    expect(blocks.map((b) => b.id)).toEqual(["cam0", "cam1"]);
    const cam0 = blocks[0]!;
    expect(cam0.rows.map((r) => r.label)).toEqual(["fx", "fy", "k3"]);
    const fy = cam0.rows[1]!;
    expect(fy.reference).toBe(200);
    expect(fy.others).toHaveLength(2);
    // run-003 (same data) has no fy: the cell is empty, not a zero.
    expect(fy.others[1]).toBeNull();
    expect(cam0.status).toEqual(["same_model", "same_model"]);
    expect(blocks[1]!.status).toEqual(["same_model", "different_models"]);
    expect(blocks[1]!.joints[1]).toBeNull();
  });

  it("marks what matters: a significant test, or untested a change of 1 σ", () => {
    const cam0 = blocks[0]!;
    const by = Object.fromEntries(cam0.rows.map((r) => [r.label, r.matters]));
    // fx: 0.35 / 0.5 = 0.7 σ, test consistent. fy: significant. k3: no change.
    expect(by).toEqual({ fx: false, fy: true, k3: false });
    // fx of run-003 is untested and 0.1 / 0.5 = 0.2 σ: it does not matter.
    expect(visibleRows(cam0.rows, true).map((r) => r.label)).toEqual(["fy"]);
    expect(visibleRows(cam0.rows, false)).toHaveLength(3);
  });

  it("calls an untested change material only from 1 reference σ", () => {
    const c = comparison();
    const run3 = c.differences[1]!.cameras[0]!.parameters;
    if (run3.kind !== "same_model") throw new Error("model");
    run3.parameters[0]!.delta_over_reference_sigma = -1.5;
    expect(cameraBlocks(c)[0]!.rows[0]!.matters).toBe(true);
  });

  it("tallies the tests made, not the cells without one", () => {
    expect(testTally(blocks[0]!.rows)).toEqual({ tested: 2, significant: 1 });
    expect(testTally(blocks[1]!.rows)).toEqual({ tested: 1, significant: 0 });
  });

  it("keeps a camera's joint test per run", () => {
    expect(blocks[0]!.joints[0]).toMatchObject({ kind: "independent", dof: 8 });
    expect(blocks[0]!.joints[1]).toMatchObject({ kind: "not_available" });
  });
});

describe("numbers", () => {
  it("writes a signed change with three significant digits", () => {
    expect(signed(0.012345)).toBe("+0.0123");
    expect(signed(-0.4)).toBe("−0.4");
    expect(signed(0)).toBe("0");
    expect(signed(Number.NaN)).toBe("–");
  });

  it("pools a run's RMS by residual count and finds its largest σ", () => {
    const run = comparison().runs[0]!;
    expect(pooledRms(run)).toBeCloseTo(Math.sqrt((0.2 ** 2 + (0.2 * 1.05) ** 2) / 2), 9);
    expect(maxProjectionSigma(run)).toBe(0.2);
    expect(pooledRms({ ...run, cameras: [] })).toBeNull();
  });

  it("names the positions: the first is the reference", () => {
    expect([0, 1, 2].map(positionName)).toEqual(["Reference", "B", "C"]);
  });

  it("compares residuals only for the same observations", () => {
    expect(["same_data", "identical"].map(residualsComparable)).toEqual([true, true]);
    expect(
      ["independent", "nested", "overlapping", "unknown"].some(residualsComparable),
    ).toBe(false);
  });
});

describe("what the data supports", () => {
  const names = ["run-001", "run-002", "run-003"];

  it("names the lowest RMS but not as better when the runs differ in model or data", () => {
    const facts = verdictFacts(comparison(), names);
    const rms = facts.find((f) => f.includes("lowest RMS"))!;
    expect(rms).toContain("run-003");
    expect(rms).toContain("camera models differ");
    expect(rms).toContain("not evidence of a better calibration");
  });

  it("states the lowest RMS plainly only on the same observations and model", () => {
    const c = comparison();
    c.differences = [{ ...c.differences[1]!, model_changes: [], run: 1 }];
    c.runs = [c.runs[0]!, c.runs[2]!];
    const facts = verdictFacts(c, ["run-001", "run-003"]);
    expect(facts[0]).toBe(
      "run-003 has the lowest RMS (0.154 px) of the runs that do not fail, on the same observations.",
    );
  });

  it("says the RMS does not compare across different observations", () => {
    const c = comparison();
    c.differences = [c.differences[0]!];
    c.runs = [c.runs[0]!, c.runs[1]!];
    expect(verdictFacts(c, names)[0]).toContain("used different observations");
  });

  it("leaves out a failing run's RMS and says it fails; recorded verdicts win", () => {
    const facts = verdictFacts(comparison(), names, ["fail", null, null]);
    expect(facts.some((f) => f.startsWith("run-001 fails its quality check"))).toBe(true);
    expect(facts.some((f) => f.includes("run-001 has the lowest"))).toBe(false);
  });

  it("counts the rejecting tests, and says when none applies", () => {
    const facts = verdictFacts(comparison(), names);
    expect(facts[facts.length - 1]).toBe(
      "1 of 3 parameter tests reject agreement at the 5 % level; read the joint test of each camera first.",
    );
    const c = comparison();
    c.differences = [c.differences[1]!];
    c.runs = [c.runs[0]!, c.runs[2]!];
    expect(verdictFacts(c, ["run-001", "run-003"]).pop()).toBe(
      "No parameter test applies to these runs; differences are read as materiality only.",
    );
  });

  it("makes no statement for fewer than two runs", () => {
    const c = comparison();
    c.runs = [c.runs[0]!];
    expect(verdictFacts(c, ["run-001"])).toEqual([]);
  });

  it("never says best", () => {
    const text = verdictFacts(comparison(), names).join(" ");
    expect(text).not.toMatch(/\bbest\b/i);
  });
});

describe("relation words", () => {
  const kinds: DataRelation[] = [
    { kind: "unknown" },
    { kind: "independent" },
    { kind: "identical" },
    { kind: "same_data" },
    { kind: "nested", smaller: "other" },
    { kind: "overlapping" },
  ];

  it("says what each relation allows, naming the runs", () => {
    const words = kinds.map((k) => relationWords(k, "B", "A"));
    expect(words.map((w) => w.label)).toEqual([
      "Relation unknown",
      "Independent captures",
      "Identical",
      "Same data",
      "Nested",
      "Overlapping",
    ]);
    expect(words[0]!.text).toContain("no parameter test is made");
    expect(words[3]!.text).toContain("no test is valid");
    // The smaller run is named, and the exclusion caveat is stated.
    expect(words[4]!.text).toContain("B used a subset of A");
    expect(words[4]!.text).toContain("without looking at their residuals");
  });

  it("names the smaller run of a nested pair on either side", () => {
    expect(
      relationWords({ kind: "nested", smaller: "reference" }, "B", "A").text,
    ).toContain("A used a subset of B");
  });

  it("says a refused relation and what was done instead", () => {
    expect(
      refusalWords({ stated: { kind: "independent" }, reason: "shared_features" }),
    ).toBe(
      "The stated relation (independent) was refused: the runs share features. The pair is compared as overlapping.",
    );
  });

  it("says why a covariance is missing, or nothing when it is there", () => {
    expect(covarianceWords({ status: "available", shared_parameters: 3 })).toBeNull();
    expect(
      covarianceWords({
        status: "unavailable",
        reason: { kind: "rank_deficient", unobservable: ["k3"] },
      }),
    ).toBe("rank deficient: k3 not observable");
  });
});
