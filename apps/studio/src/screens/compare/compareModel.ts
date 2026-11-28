/** The Compare screen's rules over the comparison's reply: the parameter
 * rows of each camera across the runs, which differences matter, the runs'
 * own evidence in one line each, and the facts a verdict may state. Pure;
 * nothing here ranks runs, and a statement says only what the reply
 * supports. */
import type {
  CameraEvidence,
  DifferenceTest,
  JointTest,
  ParameterComparison,
  ParameterDifference,
  RunComparison,
  RunEvidence,
  Verdict,
} from "../../api/generated/types";

/** The level of a test read as "significant" (the book's 5 %). */
export const ALPHA = 0.05;

/** A difference with no valid test matters when it is this many of the
 * reference's σ: materiality, which is not a test. */
export const MATERIAL_SIGMA = 1;

export type TestState = "significant" | "consistent" | "not_available";

export function testState(test: DifferenceTest | JointTest): TestState {
  if (test.kind === "not_available") return "not_available";
  return test.p_value < ALPHA ? "significant" : "consistent";
}

/** One parameter of one camera across the runs after the reference. */
export interface ParameterRow {
  label: string;
  reference: number;
  referenceSigma: number | null;
  /** One per run after the reference; `null` where that run lacks it. */
  others: (ParameterDifference | null)[];
  /** The largest |Δ| in σ of the reference over the runs, `null` when the
   * reference has no σ. */
  size: number | null;
  /** Whether it is a difference that matters: a significant test, or, with
   * no test, a change of at least `MATERIAL_SIGMA` reference σ. */
  matters: boolean;
}

/** What a run's camera is compared as. */
export type CameraStatus = "same_model" | "different_models" | "unparameterized";

export interface CameraBlock {
  id: string;
  rows: ParameterRow[];
  /** Per run after the reference: how the camera was compared. */
  status: CameraStatus[];
  /** Per run after the reference: the joint test of the camera's
   * parameters; `null` where they were not compared. */
  joints: (JointTest | null)[];
}

function mattersOf(row: Pick<ParameterRow, "others">): boolean {
  return row.others.some((p) => {
    if (!p) return false;
    if (p.test.kind !== "not_available") return testState(p.test) === "significant";
    return (
      p.delta_over_reference_sigma != null &&
      Math.abs(p.delta_over_reference_sigma) >= MATERIAL_SIGMA
    );
  });
}

/** The cameras of the comparison, in the reference's order then the
 * others', each with its parameters laid across the runs. */
export function cameraBlocks(comparison: RunComparison): CameraBlock[] {
  const ids: string[] = [];
  for (const d of comparison.differences) {
    for (const c of d.cameras) if (!ids.includes(c.id)) ids.push(c.id);
  }
  return ids.map((id) => {
    const per = comparison.differences.map(
      (d) => d.cameras.find((c) => c.id === id)?.parameters ?? null,
    );
    const same = (p: ParameterComparison | null) => (p?.kind === "same_model" ? p : null);
    const labels: string[] = [];
    for (const p of per) {
      for (const q of same(p)?.parameters ?? [])
        if (!labels.includes(q.label)) labels.push(q.label);
    }
    const rows = labels.map((label): ParameterRow => {
      const others = per.map(
        (p) => same(p)?.parameters.find((q) => q.label === label) ?? null,
      );
      const first = others.find((p) => p !== null)!;
      const sizes = others
        .map((p) => p?.delta_over_reference_sigma)
        .filter((v): v is number => v != null && Number.isFinite(v))
        .map(Math.abs);
      const row = {
        label,
        reference: first.reference,
        referenceSigma: first.reference_sigma ?? null,
        others,
        size: sizes.length > 0 ? Math.max(...sizes) : null,
      };
      return { ...row, matters: mattersOf(row) };
    });
    return {
      id,
      rows,
      status: per.map((p) => p?.kind ?? "unparameterized"),
      joints: per.map((p) => same(p)?.joint ?? null),
    };
  });
}

/** The rows to show: all, or those that matter. */
export function visibleRows(rows: ParameterRow[], onlyMatters: boolean): ParameterRow[] {
  return onlyMatters ? rows.filter((r) => r.matters) : rows;
}

/** How many tests a block held, and how many of them rejected. */
export function testTally(rows: ParameterRow[]): { tested: number; significant: number } {
  let tested = 0;
  let significant = 0;
  for (const row of rows)
    for (const p of row.others) {
      if (!p || p.test.kind === "not_available") continue;
      tested += 1;
      if (testState(p.test) === "significant") significant += 1;
    }
  return { tested, significant };
}

/** `+0.0123` / `−0.4`: a signed change, three significant digits. */
export function signed(value: number): string {
  if (!Number.isFinite(value)) return "–";
  if (value === 0) return "0";
  const text = Number(value.toPrecision(3)).toString();
  return value > 0 ? `+${text}` : text.replace("-", "−");
}

/** A run's reprojection RMS over its cameras, pooled by residual count;
 * `null` when no camera has one. */
export function pooledRms(run: RunEvidence): number | null {
  let squares = 0;
  let count = 0;
  for (const c of run.cameras) {
    const r = c.reprojection;
    if (!r || r.count <= 0) continue;
    squares += r.count * r.rms * r.rms;
    count += r.count;
  }
  return count > 0 ? Math.sqrt(squares / count) : null;
}

/** The largest projection σ of a run over its cameras, px. */
export function maxProjectionSigma(run: RunEvidence): number | null {
  const values = run.cameras
    .map((c) => c.max_projection_sigma_px)
    .filter((v): v is number => v != null && Number.isFinite(v));
  return values.length > 0 ? Math.max(...values) : null;
}

/** Whether `run`'s in-sample residuals compare with the reference's:
 * only runs of the same observations do. */
export function residualsComparable(kind: string): boolean {
  return kind === "same_data" || kind === "identical";
}

/** The cameras the reference and run `index` (0-based among the others)
 * both have, which a projection map can be made for. */
export function commonCameras(comparison: RunComparison, index: number): string[] {
  const d = comparison.differences[index];
  return d ? d.cameras.map((c) => c.id) : [];
}

export function evidenceCamera(run: RunEvidence, id: string): CameraEvidence | null {
  return run.cameras.find((c) => c.id === id) ?? null;
}

/** The facts the data supports about which run to keep, as sentences. It
 * names the lowest RMS only among passing runs and says when that is no
 * evidence of a better calibration; it never says "best". */
export function verdictFacts(
  comparison: RunComparison,
  names: string[],
  /** Each run's verdict as recorded with it, where there is one; the
   * comparison's own (default thresholds) otherwise. */
  verdicts: (Verdict | null)[] = [],
): string[] {
  const facts: string[] = [];
  const runs = comparison.runs;
  if (runs.length < 2) return facts;

  const verdictOf = (i: number): Verdict => verdicts[i] ?? runs[i]!.verdict;
  const passing = runs
    .map((r, i) => ({ r, i, rms: pooledRms(r) }))
    .filter((e) => verdictOf(e.i) !== "fail" && e.rms !== null);
  const allSameData = comparison.differences.every((d) =>
    residualsComparable(d.relation.kind),
  );
  const modelsDiffer = comparison.differences.some((d) => d.model_changes.length > 0);
  if (passing.length >= 2) {
    const lowest = passing.reduce((a, b) => (b.rms! < a.rms! ? b : a));
    const tie = passing.filter(
      (e) => Math.round(e.rms! * 1000) === Math.round(lowest.rms! * 1000),
    );
    if (tie.length > 1) {
      facts.push(
        "The runs that do not fail read the same RMS to the thousandth of a pixel.",
      );
    } else if (allSameData && !modelsDiffer) {
      facts.push(
        `${names[lowest.i]} has the lowest RMS (${lowest.rms!.toFixed(3)} px) of the runs that do not fail, on the same observations.`,
      );
    } else if (modelsDiffer) {
      facts.push(
        `${names[lowest.i]} has the lowest RMS (${lowest.rms!.toFixed(3)} px), but the camera models differ, and in-sample RMS favours the larger model: it is not evidence of a better calibration.`,
      );
    } else {
      facts.push(
        `${names[lowest.i]} has the lowest RMS (${lowest.rms!.toFixed(3)} px), but the runs used different observations, so their RMS does not compare.`,
      );
    }
  }
  const sigmas = runs
    .map((r, i) => ({ i, sigma: maxProjectionSigma(r), ok: verdictOf(i) !== "fail" }))
    .filter((e) => e.ok && e.sigma !== null);
  if (sigmas.length >= 2) {
    const tightest = sigmas.reduce((a, b) => (b.sigma! < a.sigma! ? b : a));
    facts.push(
      `${names[tightest.i]} is the most precise: its largest projection σ is ${tightest.sigma!.toFixed(3)} px.`,
    );
  }
  const failing = runs
    .map((_, i) => (verdictOf(i) === "fail" ? names[i] : null))
    .filter(Boolean);
  if (failing.length > 0)
    facts.push(
      `${failing.join(", ")} ${failing.length === 1 ? "fails its" : "fail their"} quality check.`,
    );

  const blocks = cameraBlocks(comparison);
  const tally = blocks.reduce(
    (t, b) => {
      const x = testTally(b.rows);
      return { tested: t.tested + x.tested, significant: t.significant + x.significant };
    },
    { tested: 0, significant: 0 },
  );
  if (tally.tested > 0) {
    facts.push(
      tally.significant === 0
        ? `None of the ${tally.tested} parameter tests rejects agreement at the 5 % level.`
        : `${tally.significant} of ${tally.tested} parameter tests reject agreement at the 5 % level; read the joint test of each camera first.`,
    );
  } else {
    facts.push(
      "No parameter test applies to these runs; differences are read as materiality only.",
    );
  }
  return facts;
}

const VERDICT_WORD: Record<Verdict, string> = {
  pass: "pass",
  warn: "check",
  fail: "fail",
};
export const verdictWord = (v: Verdict): string => VERDICT_WORD[v];

/** The position's name in the set: the first is the reference. */
export const positionName = (index: number): string =>
  index === 0 ? "Reference" : String.fromCharCode(65 + index);
