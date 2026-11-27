/** A camera's parameters as the Parameters tab writes them: grouped, each
 * `value ± σ` with its unit, or the stated reason there is no σ. Pure. */
import type {
  CameraParameter,
  CameraParameterSet,
  DistortionKind,
  ParameterCovariance,
  ParameterGroup,
  ProjectionKind,
  SensorKind,
  SigmaUnavailable,
} from "../../api/generated/types";
import { formatWithSigma, type Uncertainty } from "../../lib/format";
import { parameterRows, unavailableText, type ParameterRow } from "../../lib/uncertainty";

/** The groups in the order the tab lists them. A group the back end may add
 * later (a pose) is listed after these by its own name. */
export const GROUP_ORDER: readonly ParameterGroup[] = [
  "intrinsics",
  "projection",
  "distortion",
  "sensor",
];

export const GROUP_TITLES: Record<string, string> = {
  intrinsics: "Intrinsics",
  projection: "Projection",
  distortion: "Distortion",
  sensor: "Sensor",
};

const DISTORTION_NAMES: Record<DistortionKind, string> = {
  none: "no distortion",
  brown_conrady5: "Brown–Conrady (5)",
  rational8: "rational (8)",
  thin_prism9: "thin prism (9)",
  division1: "division (1)",
  kannala_brandt4: "Kannala–Brandt (4)",
};

const PROJECTION_NAMES: Record<ProjectionKind, string> = {
  pinhole: "pinhole",
  equidistant: "equidistant",
};

/** `pinhole · Brown–Conrady (5) · tilted sensor`. */
export function modelSummary(set: CameraParameterSet): string {
  const sensor: SensorKind | null | undefined = set.sensor;
  const parts = [
    PROJECTION_NAMES[set.projection] ?? set.projection,
    DISTORTION_NAMES[set.distortion] ?? set.distortion,
  ];
  if (sensor === "scheimpflug") parts.push("tilted sensor");
  else if (sensor === null) parts.push("sensor homography");
  return parts.join(" · ");
}

/** The parameters of a set by group, in the tab's order. */
export function groupParameters(
  parameters: readonly CameraParameter[],
): { group: string; title: string; parameters: CameraParameter[] }[] {
  const byGroup = new Map<string, CameraParameter[]>();
  for (const p of parameters) {
    const list = byGroup.get(p.group) ?? [];
    list.push(p);
    byGroup.set(p.group, list);
  }
  const known = GROUP_ORDER.filter((g) => byGroup.has(g)) as string[];
  const rest = [...byGroup.keys()].filter((g) => !known.includes(g)).sort();
  return [...known, ...rest].map((group) => ({
    group,
    title: GROUP_TITLES[group] ?? group,
    parameters: byGroup.get(group)!,
  }));
}

/** The sentence for why a parameter (or a projection) has no σ. */
export function describeSigmaUnavailable(reason: SigmaUnavailable): string {
  switch (reason.kind) {
    case "fixed":
      return "held fixed, not estimated";
    case "not_listed":
      return "not among the estimated parameters of this run";
    case "unsupported_model":
      return reason.message;
    case "no_covariance":
      return unavailableText(reason.reason);
  }
}

/** Why a parameter has no σ; `null` when it has one. */
export function sigmaReason(p: CameraParameter): string | null {
  return p.sigma.kind === "estimated" ? null : describeSigmaUnavailable(p.sigma.reason);
}

/** The σ of a parameter as `formatWithSigma` takes it. */
export function uncertaintyOf(p: CameraParameter): Uncertainty {
  if (p.sigma.kind === "estimated") return { kind: "sigma", sigma: p.sigma.sigma };
  if (p.sigma.reason.kind === "fixed") return { kind: "fixed" };
  return { kind: "unavailable", reason: sigmaReason(p) ?? "unavailable" };
}

/** A parameter as shown: radians are written in degrees (the sensor tilt),
 * everything else as the back end gives it. */
export interface ParameterView {
  parameter: CameraParameter;
  /** The value's digits with `± σ` when there is one. */
  text: string;
  unit: string;
  /** σ as a share of the value, for a scale parameter with a σ. */
  relative: string | null;
  /** Why there is no σ, when there is none. */
  reason: string | null;
}

export function viewOf(p: CameraParameter): ParameterView {
  const radians = p.unit === "rad";
  const scale = radians ? 180 / Math.PI : 1;
  const unc = uncertaintyOf(p);
  const shown: Uncertainty =
    unc.kind === "sigma" ? { kind: "sigma", sigma: unc.sigma * scale } : unc;
  const f = formatWithSigma(p.value * scale, shown);
  const scaleParameter = p.unit === "px" && p.value !== 0;
  const relative =
    scaleParameter && unc.kind === "sigma"
      ? `${((unc.sigma / Math.abs(p.value)) * 100).toPrecision(2)} %`
      : null;
  return {
    parameter: p,
    text: f.text,
    unit: radians ? "°" : p.unit,
    relative,
    reason: sigmaReason(p),
  };
}

/** The set as JSON for the clipboard: the wire's own values (radians stay
 * radians), σ or the reason it is missing. */
export function parametersJson(set: CameraParameterSet): string {
  return JSON.stringify(
    {
      camera: set.camera,
      model: {
        projection: set.projection,
        distortion: set.distortion,
        sensor: set.sensor,
      },
      parameters: set.parameters.map((p) => ({
        name: p.name,
        group: p.group,
        value: p.value,
        unit: p.unit,
        sigma: p.sigma.kind === "estimated" ? p.sigma.sigma : null,
        sigma_unavailable: sigmaReason(p),
        at_bound: p.at_bound,
      })),
    },
    null,
    2,
  );
}

/** The shared parameters of the covariance that are no camera's own (a
 * rig's camera-to-rig pose, a hand-eye pose, a laser plane), as σ-only rows:
 * their covariance is over a perturbation of the pose, so there is no value
 * to write. Of `camera`: its own `cam<n>.` poses and those of no camera. */
export function otherRows(
  cov: ParameterCovariance,
  owned: ReadonlySet<string>,
  camera: number,
): ParameterRow[] {
  return parameterRows(
    cov,
    cov.labels.map(() => null),
  ).filter(
    (r) =>
      !owned.has(r.label) &&
      (r.label.startsWith(`cam${camera}.`) || !/^cam\d+\./.test(r.label)),
  );
}
