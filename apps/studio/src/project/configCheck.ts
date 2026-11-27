/** What the library says of an edited configuration, as the form shows it,
 * and what a failed calibration says to try.
 *
 * `resolve_config` answers a typed `config` error whose message starts with
 * the field it is about (`solver.max_iters must be positive`,
 * `optimize.reprojection_loss (huber): scale …`). The form shows a message
 * on the field it names, when the config has that field; any other message
 * is the form's. */
import { getAtPath } from "@vitavision/forms";
import type { ErrorKind, StudioError } from "../api/generated/types";

/** The path a config error is about: the leading dotted name of its
 * message (after `invalid input:`), cut back to the longest prefix the
 * config has; `""` when it names no field of it. */
export function errorFieldPath(message: string, config: unknown): string {
  const text = message.replace(/^(invalid input|serialization error):\s*/i, "");
  const lead = /^([a-z_][a-z0-9_]*(?:\.[a-z0-9_]+)*)/.exec(text)?.[1];
  if (!lead) return "";
  const segments = lead.split(".");
  while (segments.length > 0) {
    const path = segments.join(".");
    if (getAtPath(config, path) !== undefined) return path;
    segments.pop();
  }
  return "";
}

/** The messages of a failed check, keyed for `SchemaValueForm`. */
export function errorMessages(
  error: Pick<StudioError, "message"> | null,
  config: unknown,
): Record<string, string> {
  if (!error) return {};
  return { [errorFieldPath(error.message, config)]: tidy(error.message) };
}

/** A library message without its wrapper (`invalid input: …`). */
export function tidy(message: string): string {
  return message.replace(/^(invalid input|serialization error):\s*/i, "");
}

/** Where a failed calibration stopped, in words, from its last progress
 * report (`detection`, `step`, `iteration`); `null` when it sent none. */
export function stageOf(event: unknown): string | null {
  if (typeof event !== "object" || event === null) return null;
  const e = event as {
    kind?: unknown;
    name?: unknown;
    iter?: unknown;
    done?: unknown;
    total?: unknown;
  };
  // The last count reaching the total means the stage had finished: what
  // failed came after it (screening the detections, the first solve step).
  const finished = e.done !== undefined && e.done === e.total;
  switch (e.kind) {
    case "detection":
      return finished
        ? "after detecting the images"
        : `while detecting the images (${String(e.done)} of ${String(e.total)})`;
    case "laser_extraction":
      return finished
        ? "after extracting the laser line"
        : "while extracting the laser line";
    case "step":
      return typeof e.name === "string" ? `in the step "${e.name}"` : null;
    case "iteration":
      return typeof e.iter === "number"
        ? `while solving (iteration ${e.iter})`
        : "while solving";
    default:
      return null;
  }
}

/** What to try after a failure of this kind. */
export function advice(kind: ErrorKind, message: string): string {
  const lower = message.toLowerCase();
  if (/need \d+, got \d+|too few|not enough/.test(lower)) {
    return "The dataset gives too few usable views. Check the dataset's images and the exclusions, then run again.";
  }
  switch (kind) {
    case "config":
      return "A setting is not valid. Look for the field named in the message, or reset the editor to the project's configuration.";
    case "calibration":
      return "The solver could not calibrate with these settings or this data. Try the project's configuration, a robust loss, more iterations, or fewer free parameters.";
    case "detection":
      return "A target could not be detected. Open the images in Detect, or run again with Re-detect images on.";
    case "manifest":
    case "exclusion":
    case "needs_input":
      return "The dataset is not complete. Fix it on the Dataset screen first.";
    case "busy":
      return "Another job is using the back end. It runs when that one ends.";
    case "io":
    case "forbidden_path":
      return "A file could not be read. Check that the dataset's folder is where the project says.";
    case "changed_on_disk":
      return "The project changed on disk. Reload it, then run again.";
    default:
      return "Run again. If it fails the same way, copy the message from the job tray.";
  }
}
