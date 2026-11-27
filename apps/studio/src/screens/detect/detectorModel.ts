/** The detector settings the Detector tab edits: the manifest's `detector`
 * section, its form (the library's own schema, cut down to that one
 * section), and what applying it writes. Pure. */
import type { JsonSchema, UiSchema } from "@vitavision/forms";
import type {
  DatasetSpec,
  DetectorSpec,
  ProjectSpec,
  ProjectView,
} from "../../api/generated/types";
import { withManifest } from "../../project/projectModel";

/** The detector settings the project saves (`{}`: every detector's own defaults). */
export function savedDetector(project: ProjectView): DetectorSpec {
  const manifest = project.dataset.kind === "ok" ? project.dataset.spec : null;
  return manifest?.detector ?? {};
}

export function sameDetector(a: DetectorSpec, b: DetectorSpec): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** The manifest's schema reduced to its `detector` section as a required,
 * non-null block, so the form shows the settings themselves and not a
 * switch that sets them. `null` while the schema has none. */
export function detectorSchema(schema: JsonSchema | undefined): JsonSchema | null {
  const props = (schema as { properties?: Record<string, unknown> } | undefined)
    ?.properties;
  if (!schema || !props || !("detector" in props)) return null;
  return {
    ...schema,
    properties: { detector: { $ref: "#/$defs/DetectorSpec" } },
    required: ["detector"],
  };
}

/** The form's layout: the section's own block, named for the tab. */
export const DETECTOR_UI: UiSchema = {
  fields: { detector: { label: "Detector settings" } },
};

/** Whether the manifest is written in the project file, so the detector
 * settings can be saved with it. */
export function canApply(project: ProjectView): boolean {
  return project.dataset_editable && project.dataset.kind === "ok";
}

/** The project's spec with `detector` as the manifest's detector section
 * (an empty one is dropped: the detectors' defaults). `null` when the
 * manifest is not the project file's. */
export function specWithDetector(
  project: ProjectView,
  detector: DetectorSpec,
): ProjectSpec | null {
  if (!canApply(project) || project.dataset.kind !== "ok") return null;
  const manifest: DatasetSpec = {
    ...project.dataset.spec,
    detector: Object.keys(detector).length === 0 ? null : detector,
  };
  return withManifest(project.spec, manifest);
}
