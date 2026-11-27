/** The Dataset screen's pure rules: the library's messages about a draft
 * placed on the form's fields (the library names the field for some and not
 * for others, so the rest are placed by what the message says, until it
 * does), what a camera's pattern matched, which groups of the form a
 * workflow uses, and the draft's problems counted by severity. */
import type { FieldMessage, FieldMessages } from "@vitavision/forms";
import type { UiSchema } from "@vitavision/forms";
import type {
  CameraSource,
  DatasetSpec,
  ProblemKind,
  ProjectViews,
  StudioError,
} from "../../api/generated/types";
import { exportKindLabel } from "../../store/exportKind";

/** A message about the draft, placed (`path`) on a field of the manifest
 * form when it can be, else listed (`path` is `null`). */
export interface DraftMessage {
  /** The form's dot path (`cameras.1.images`), or `null`. */
  path: string | null;
  severity: "error" | "warning";
  text: string;
  /** What the library offers for the field, when it does (`needs_input`). */
  suggestions: readonly string[];
}

/** The library's `cameras[1].images` as the form's `cameras.1.images`. */
export function toFormPath(field: string): string {
  return field.replace(/\[(\d+)\]/g, ".$1");
}

/** Whether `path` is a field of the manifest (and so of the form); the
 * project's own fields (`device.id`, `exclude…`) are not. */
export function isManifestPath(path: string): boolean {
  const root = path.split(".")[0] ?? "";
  return [
    "cameras",
    "target",
    "topology",
    "pose_pairing",
    "robot_poses",
    "pose_convention",
    "detector",
    "laser",
    "upstream_calibration",
    "description",
  ].includes(root);
}

/** The field a library message is about when it names none: read from the
 * message's own words (the library's phrasings), against the draft's
 * cameras. `null` when it says nothing of a field. */
export function inferField(
  message: string,
  cameras: readonly CameraSource[],
): string | null {
  if (/^manifest validation failed: invalid target/.test(message)) return "target";
  if (/duplicate camera id/.test(message)) return "cameras";
  if (/expects .* camera\(s\)/.test(message)) return "cameras";
  if (/requires robot_poses/.test(message)) return "robot_poses";
  const roi = /camera "([^"]*)" has degenerate ROI/.exec(message);
  if (roi) {
    const i = cameras.findIndex((c) => c.id === roi[1]);
    return i >= 0 ? `cameras.${i}.roi_xywh` : "cameras";
  }
  if (/no named capture group/.test(message)) return "pose_pairing.group";
  if (/pose pairing regex .* failed to compile/.test(message))
    return "pose_pairing.regex";
  return null;
}

/** The library's messages about a draft, each placed on its field. */
export function placeMessages(
  messages: readonly StudioError[],
  manifest: DatasetSpec | null,
): DraftMessage[] {
  return messages.map((m) => {
    const named = m.field ? toFormPath(m.field) : null;
    const path = named ?? (manifest ? inferField(m.message, manifest.cameras) : null);
    return {
      path: path !== null && isManifestPath(path) ? path : null,
      severity: "error" as const,
      text: m.message,
      suggestions: m.suggestions ?? [],
    };
  });
}

/** What a camera's pattern matched, from the views the library paired. */
export interface PatternMatch {
  /** Images the camera has among the views. */
  files: number;
  first: string | null;
  last: string | null;
}

export function patternMatch(
  views: Pick<ProjectViews, "views">,
  camera: number,
): PatternMatch {
  let files = 0;
  let first: string | null = null;
  let last: string | null = null;
  for (const v of views.views) {
    const image = v.images[camera];
    if (image == null) continue;
    files++;
    first ??= image;
    last = image;
  }
  return { files, first, last };
}

/** What the app itself sees wrong that the library does not say: a camera
 * whose pattern matched nothing (a warning on its pattern: a run would find
 * no images), and a camera missing from some views (listed in the matrix). */
export function derivedWarnings(
  reply: Pick<ProjectViews, "cameras" | "views"> | null,
  manifest: DatasetSpec | null,
): DraftMessage[] {
  if (!reply || !manifest) return [];
  return reply.cameras.flatMap((_id, i) => {
    if (patternMatch(reply, i).files > 0) return [];
    const images = manifest.cameras[i]?.images;
    const what = images?.kind === "glob" ? `“${images.pattern}”` : "its list of images";
    return [
      {
        path: `cameras.${i}.images`,
        severity: "warning" as const,
        text: `No image matches ${what}.`,
        suggestions: [],
      },
    ];
  });
}

/** The messages as the form takes them, by path; a path with several
 * messages shows them one after another. */
export function formMessages(messages: readonly DraftMessage[]): FieldMessages {
  const out: Record<string, FieldMessage> = {};
  for (const m of messages) {
    if (m.path === null) continue;
    const prior = out[m.path];
    // An error is never hidden by a warning on the same field.
    if (prior && (prior.severity === "error" || m.severity === "warning")) {
      out[m.path] = { ...prior, text: `${prior.text} ${m.text}` };
    } else {
      out[m.path] = { severity: m.severity, text: m.text };
    }
  }
  return out;
}

export interface ProblemCounts {
  errors: number;
  warnings: number;
}

export function countProblems(messages: readonly DraftMessage[]): ProblemCounts {
  return {
    errors: messages.filter((m) => m.severity === "error").length,
    warnings: messages.filter((m) => m.severity === "warning").length,
  };
}

/** The workflows whose robot poses the manifest names. */
const HANDEYE: readonly ProblemKind[] = [
  "single_cam_handeye",
  "rig_handeye",
  "rig_handeye_laserline",
];
const LASER: readonly ProblemKind[] = [
  "laserline_device",
  "rig_handeye_laserline",
  "rig_laserline_device",
];

export const usesPoses = (t: ProblemKind) => HANDEYE.includes(t);
export const usesLaser = (t: ProblemKind) => LASER.includes(t);
/** A single camera with no robot poses needs no pairing. */
export const usesPairing = (t: ProblemKind, cameras: number) =>
  cameras > 1 || usesPoses(t) || usesLaser(t);

/** The form's layout for `topology`: five groups, the ones a workflow does
 * not use hidden (their fields stay in the manifest as they are), the
 * bookkeeping fields never shown. */
export function datasetUi(topology: ProblemKind, cameras: number): UiSchema {
  const hide = (paths: string[]) =>
    Object.fromEntries(paths.map((p) => [p, { hidden: true as const }]));
  const hidden = [
    "_unresolved",
    "version",
    ...(usesPoses(topology) ? [] : ["robot_poses", "pose_convention"]),
    ...(usesLaser(topology) ? [] : ["laser"]),
    ...(topology === "rig_laserline_device" ? [] : ["upstream_calibration"]),
    ...(usesPairing(topology, cameras) ? [] : ["pose_pairing"]),
  ];
  return {
    groups: [
      { id: "workflow", title: "Workflow", fields: ["topology"] },
      { id: "cameras", title: "Cameras", fields: ["cameras"] },
      { id: "pairing", title: "Pairing", fields: ["pose_pairing"] },
      { id: "target", title: "Target", fields: ["target"] },
      {
        id: "poses",
        title: "Poses & motion",
        fields: ["robot_poses", "pose_convention"],
      },
      { id: "laser", title: "Laser", fields: ["laser", "upstream_calibration"] },
      {
        id: "detector",
        title: "Detector",
        fields: ["detector"],
        collapsible: true,
        defaultOpen: false,
      },
      {
        id: "notes",
        title: "Description",
        fields: ["description"],
        collapsible: true,
        defaultOpen: false,
      },
    ],
    fields: {
      ...hide(hidden),
      topology: {
        label: "What is calibrated",
        enumLabels: Object.fromEntries(
          [
            "planar_intrinsics",
            "single_cam_handeye",
            "laserline_device",
            "rig_extrinsics",
            "rig_handeye",
            "rig_handeye_laserline",
            "rig_laserline_device",
          ].map((k) => [k, exportKindLabel(k as ProblemKind)]),
        ),
        descriptionAs: "inline",
      },
    },
  };
}
