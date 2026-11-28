/** What the comparison's typed answers say, in plain words: how two runs'
 * observations relate, why a test is not made, why a map or a covariance
 * is missing. The words follow the book's chapter on comparing
 * calibrations and claim no more than it does. Pure. */
import type { Tone } from "@vitavision/ui";
import type {
  CovarianceStatus,
  DataRelation,
  NoTest,
  ProjectionUnavailable,
  RelationRefusal,
} from "../../api/generated/types";

export interface RelationWords {
  label: string;
  tone: Tone;
  /** What the relation allows, in a sentence. */
  text: string;
}

/** How a run's observations relate to the reference's, and what that lets
 * the comparison say. `other` is the run's name, `reference` the first's. */
export function relationWords(
  relation: DataRelation,
  other: string,
  reference: string,
): RelationWords {
  switch (relation.kind) {
    case "independent":
      return {
        label: "Independent captures",
        tone: "info",
        text: `${other} and ${reference} share no observation. A two-sample test applies: a difference is the device's change plus each capture's error.`,
      };
    case "nested": {
      const smaller = relation.smaller === "reference" ? reference : other;
      const larger = relation.smaller === "reference" ? other : reference;
      return {
        label: "Nested",
        tone: "info",
        text: `${smaller} used a subset of ${larger}'s observations. The test asks whether the left-out observations pulled the estimate; it is valid only for observations left out without looking at their residuals.`,
      };
    }
    case "same_data":
      return {
        label: "Same data",
        tone: "neutral",
        text: `${other} and ${reference} used the same observations under another configuration or model. Their errors are correlated: no test is valid, and a difference reads as how much it matters, not as significant.`,
      };
    case "identical":
      return {
        label: "Identical",
        tone: "neutral",
        text: `${other} and ${reference} hold the same observations and settings: there is nothing to test.`,
      };
    case "overlapping":
      return {
        label: "Overlapping",
        tone: "neutral",
        text: `${other} and ${reference} share some observations and not others: no test is valid.`,
      };
    default:
      return {
        label: "Relation unknown",
        tone: "neutral",
        text: `It is not known how ${other}'s observations relate to ${reference}'s (an imported run says nothing of its data), so no parameter test is made. Differences are shown with what they are worth against the reference's σ.`,
      };
  }
}

const RELATION_NAME: Record<DataRelation["kind"], string> = {
  independent: "independent",
  nested: "nested",
  same_data: "the same data",
  identical: "identical",
  overlapping: "overlapping",
  unknown: "unknown",
};

const REFUSAL: Record<RelationRefusal["reason"], string> = {
  shared_features: "the runs share features",
  not_a_subset: "the smaller run's observations are not a subset of the larger's",
  configurations_differ: "the runs' configurations differ",
};

/** The stated relation the runs contradict, as a sentence. */
export function refusalWords(refusal: RelationRefusal): string {
  return `The stated relation (${RELATION_NAME[refusal.stated.kind]}) was refused: ${REFUSAL[refusal.reason]}. The pair is compared as overlapping.`;
}

export interface NoTestWords {
  /** In a table cell. */
  short: string;
  /** In a sentence. */
  long: string;
}

export const NO_TEST: Record<NoTest, NoTestWords> = {
  unknown_relation: {
    short: "relation unknown",
    long: "how the runs' data relate is not known",
  },
  identical: {
    short: "identical runs",
    long: "the runs hold the same observations and settings",
  },
  shared_data: {
    short: "shared data",
    long: "the runs share observations, so their errors are correlated",
  },
  no_covariance: { short: "no covariance", long: "a run has no covariance" },
  not_estimated: {
    short: "not estimated",
    long: "a run did not estimate this (it holds it fixed)",
  },
  mixed_residuals: {
    short: "mixed residuals",
    long: "a run's covariance combines several residual families or priors",
  },
  no_variance_factor: {
    short: "no variance factor",
    long: "an export states no variance factor",
  },
  nested_variance_not_positive: {
    short: "variance not positive",
    long: "leaving the data out did not grow the variance (a small exclusion)",
  },
  nested_joint: {
    short: "no joint test",
    long: "a nested pair has no joint test; each parameter has its own",
  },
  gauge_differs: {
    short: "rig frame differs",
    long: "the runs hold different cameras as the rig frame",
  },
  singular_covariance: {
    short: "singular covariance",
    long: "the covariance of the difference is singular",
  },
};

const PROJECTION: Record<ProjectionUnavailable, string> = {
  invalid_image: "an image size of one of the runs is not valid",
  too_few_nodes:
    "the two calibrations have too few image positions in common to be aligned and compared",
  singular:
    "the alignment of the two camera frames could not be solved (degenerate geometry)",
  not_central:
    "a camera is not a central projection (a telecentric lens, say), so rays cannot be compared",
};

export function projectionUnavailableWords(reason: ProjectionUnavailable): string {
  return PROJECTION[reason];
}

/** Why a run has no covariance, or `null` when it has one. */
export function covarianceWords(c: CovarianceStatus): string | null {
  if (c.status === "available") return null;
  const r = c.reason;
  switch (r.kind) {
    case "not_converged":
      return "the solve did not converge";
    case "rank_deficient":
      return `rank deficient: ${r.unobservable.join(", ") || "some parameters"} not observable`;
    case "at_bound":
      return `${r.labels.join(", ") || "a parameter"} at a bound`;
    case "no_redundancy":
      return "no redundancy: as many measurements as unknowns";
    case "not_computed":
      return "not computed";
    default:
      return "the calibration was imported without a covariance";
  }
}
