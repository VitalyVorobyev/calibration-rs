/** What the context bar says of the run it shows: the state of its files,
 * its verdict in words, and whether the project's inputs have changed
 * since it was made. Pure rules over the wire types; the bar only draws
 * them. */
import type { Tone } from "@vitavision/ui";
import type { ProjectDigest, RunInfo, Verdict } from "../api/generated/types";
import {
  runStateName,
  runVerdict,
  staleParts,
  type RunStateName,
} from "../project/projectModel";

export interface Pill {
  label: string;
  tone: Tone;
  /** What hovering says, when there is more to say. */
  title?: string;
}

export interface RunPills {
  state: Pill;
  /** `null` for a run with no verdict (the quality could not be judged). */
  verdict: Pill | null;
  /** `null` when the run is current or cannot be told apart. */
  stale: Pill | null;
}

const STATE_TITLE: Record<RunStateName, string | undefined> = {
  Recorded: undefined,
  Imported: "Limited: an export made elsewhere, with no record of its configuration.",
  Legacy: "Limited: a run of the app's earlier format, with no record of its inputs.",
  Unreadable: "The run's files do not read.",
};

const VERDICT: Record<Verdict, { label: string; tone: Tone }> = {
  pass: { label: "Pass", tone: "normal" },
  warn: { label: "Check", tone: "warning" },
  fail: { label: "Fail", tone: "defect" },
};

/** The four inputs of a run, as the words the user knows them by. */
const PART_WORD: Record<keyof ProjectDigest, string> = {
  dataset: "dataset",
  exclusions: "exclusions",
  config: "configuration",
  quality: "quality thresholds",
};

export function runPills(
  run: RunInfo,
  current: ProjectDigest | null | undefined,
): RunPills {
  const name = runStateName(run);
  const verdict = runVerdict(run);
  const parts = staleParts(run, current);
  const title = STATE_TITLE[name];
  return {
    state: {
      label: name,
      tone: name === "Unreadable" ? "defect" : "neutral",
      ...(title ? { title } : {}),
    },
    verdict: verdict ? { ...VERDICT[verdict] } : null,
    stale:
      parts && parts.length > 0
        ? {
            label: "Stale",
            tone: "neutral",
            title: `The project's ${parts.map((p) => PART_WORD[p]).join(", ")} changed since this run was made.`,
          }
        : null,
  };
}
