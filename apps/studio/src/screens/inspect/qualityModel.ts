/** The checks of a run's quality report in plain words: what each examines,
 * its verdict, the measured value and the limit it was held to. Pure. */
import type {
  CheckKind,
  CheckStatus,
  Measurement,
  QualityCheck,
  Unit,
  Verdict,
} from "../../api/generated/types";

/** What each check examines, as a title and a sentence. */
export const CHECK_WORDS: Record<CheckKind, { title: string; about: string }> = {
  observability: {
    title: "Every parameter is determined",
    about: "Whether the data determines every estimated parameter.",
  },
  convergence: {
    title: "The solve converged",
    about: "Whether the final solve converged.",
  },
  variance_factor: {
    title: "Residual noise matches the stated noise",
    about: "Whether the reprojection noise matches the stated feature noise.",
  },
  projection_sigma: {
    title: "Projection is precise",
    about: "How precisely the camera places a ray, at worst, over its image.",
  },
  feature_coverage: {
    title: "The features cover the image",
    about:
      "How much of the image the observed features reached, in cells of an 8 by 6 grid.",
  },
  view_outliers: {
    title: "No view stands out",
    about: "Views whose residuals are far above the noise level.",
  },
  parameter_correlation: {
    title: "Parameters can be told apart",
    about:
      "Camera parameters the data cannot tell apart: the largest absolute correlation of a pair.",
  },
  covariance_linearity: {
    title: "The covariance describes the error",
    about:
      "Whether the covariance describes the joint error of the shared parameters, or a combination of them errs well beyond its σ at second order.",
  },
  parameter_conditioning: {
    title: "No parameter's σ is inflated",
    about:
      "Camera parameters whose σ the data inflates by trading them off against a combination of the camera's other parameters.",
  },
};

export const VERDICT_WORDS: Record<Verdict, string> = {
  pass: "Pass",
  warn: "Warning",
  fail: "Fail",
};

export const STATUS_WORDS: Record<CheckStatus, string> = {
  pass: "Pass",
  warn: "Warning",
  fail: "Fail",
  not_applicable: "Not applicable",
};

/** The badge tone of a status; verdict colour lives only here and in the
 * context bar. */
export function statusTone(
  status: CheckStatus,
): "normal" | "warning" | "defect" | "neutral" {
  switch (status) {
    case "pass":
      return "normal";
    case "warn":
      return "warning";
    case "fail":
      return "defect";
    case "not_applicable":
      return "neutral";
  }
}

const UNIT_TEXT: Record<Unit, string> = {
  pixel: "px",
  fraction: "",
  ratio: "",
  sigma: "σ̂",
  correlation: "",
};

/** A measured value or a limit with its unit (`0.31 px`, `62 %`). */
export function measureText(value: number, unit: Unit): string {
  if (unit === "fraction") return `${Number((value * 100).toPrecision(3))} %`;
  const text = String(Number(value.toPrecision(3))).replace("-", "−");
  const suffix = UNIT_TEXT[unit];
  return suffix ? `${text} ${suffix}` : text;
}

/** `measured 0.31 px, limit 0.5 px`; `null` for a check with no measurement. */
export function measurementText(m: Measurement | null | undefined): string | null {
  if (!m) return null;
  return `measured ${measureText(m.value, m.unit)}, limit ${measureText(m.threshold, m.unit)}`;
}

/** The checks to list: the result-wide ones first, then each camera's (the
 * report's own order), with the ones that could not be assessed last of
 * their group. `camera` keeps the whole-result checks and that camera's. */
export function checksFor(
  checks: readonly QualityCheck[],
  camera: number | null,
): QualityCheck[] {
  return checks.filter((c) => camera === null || c.camera == null || c.camera === camera);
}

/** How many checks hold each status. */
export function tally(checks: readonly QualityCheck[]): Record<CheckStatus, number> {
  const t: Record<CheckStatus, number> = { pass: 0, warn: 0, fail: 0, not_applicable: 0 };
  for (const c of checks) t[c.status] += 1;
  return t;
}
