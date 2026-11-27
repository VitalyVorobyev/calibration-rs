import {
  Badge,
  Button,
  Callout,
  Disclosure,
  ErrorBox,
  Skeleton,
  Table,
  type Column,
} from "@vitavision/ui";
import { useState } from "react";
import { api } from "../../api";
import type { CameraParameterSet, RunRef } from "../../api/generated/types";
import { significant } from "../../lib/format";
import {
  exportUncertainty,
  residualKindName,
  residualKindUnit,
  unavailableText,
} from "../../lib/uncertainty";
import type { ParameterRow } from "../../lib/uncertainty";
import { useQuery } from "../../session/query";
import { useStore } from "../../store";
import { CameraFrame } from "./CameraFrame";
import { CorrelationMatrix } from "./CorrelationMatrix";
import {
  coveredLabels,
  groupParameters,
  modelSummary,
  otherRows,
  parametersJson,
  sharedOf,
  viewOf,
  type ParameterView,
} from "./parameterModel";
import { PlaneValues, PoseValues } from "./PoseValues";

const COLUMNS: Column<ParameterView>[] = [
  {
    key: "name",
    header: "Parameter",
    width: "9rem",
    cell: (r) => <span className="font-mono text-fg">{r.parameter.name}</span>,
  },
  {
    key: "value",
    header: "Value ± σ",
    numeric: true,
    width: "14rem",
    cell: (r) => (
      <span title={r.reason ?? undefined} data-testid={`value-${r.parameter.label}`}>
        {r.text}
      </span>
    ),
  },
  {
    key: "unit",
    header: "Unit",
    width: "4rem",
    cell: (r) => <span className="text-fg-muted">{r.unit || "–"}</span>,
  },
  {
    key: "relative",
    header: "σ / value",
    numeric: true,
    width: "6rem",
    cell: (r) => r.relative ?? <span className="text-fg-muted">–</span>,
  },
  {
    key: "note",
    header: "Note",
    cell: (r) => (
      <span className="flex flex-wrap items-center gap-1.5 text-xs text-fg-muted">
        {r.parameter.at_bound && <Badge tone="warning">on a bound</Badge>}
        {r.reason ?? ""}
      </span>
    ),
  },
];

const OTHER_COLUMNS: Column<ParameterRow>[] = [
  {
    key: "label",
    header: "Parameter",
    cell: (r) => <span className="font-mono text-fg">{r.label}</span>,
  },
  {
    key: "sigma",
    header: "σ",
    numeric: true,
    cell: (r) => `± ${significant(r.sigma, 3)}`,
  },
  {
    key: "unit",
    header: "Unit",
    width: "4rem",
    cell: (r) => <span className="text-fg-muted">{r.unit || "–"}</span>,
  },
];

/** The pose parameters the covariance is over besides the camera's own and
 * the poses and planes listed with their values (a target's pose): their
 * uncertainty, with no value. */
function OtherParameters({ rows }: { rows: ParameterRow[] }) {
  if (rows.length === 0) return null;
  return (
    <section aria-label="Other poses" className="flex flex-col gap-1">
      <h3 className="text-[11px] uppercase tracking-wider text-fg-muted">Other poses</h3>
      <Table
        caption="Pose and plane parameters"
        columns={OTHER_COLUMNS}
        rows={rows}
        rowKey={(r) => r.label}
      />
      <p className="text-xs text-fg-muted">
        Rotations in degrees, translations in millimetres. A pose is listed by its
        uncertainty alone: its covariance is over a small change of it, not over a value.
      </p>
    </section>
  );
}

function ParameterGroups({ set }: { set: CameraParameterSet }) {
  if (set.parameters.length === 0) {
    return (
      <Callout tone="info">
        {set.unsupported ?? "This camera model has no parameters the solver estimates."}
      </Callout>
    );
  }
  return (
    <>
      {groupParameters(set.parameters).map((g) => (
        <section key={g.group} aria-label={g.title} className="flex flex-col gap-1">
          <h3 className="text-[11px] uppercase tracking-wider text-fg-muted">
            {g.title}
          </h3>
          <Table
            caption={`${g.title} parameters`}
            columns={COLUMNS}
            rows={g.parameters.map(viewOf)}
            rowKey={(r) => r.parameter.label}
          />
        </section>
      ))}
    </>
  );
}

/** The fit's own numbers and what the covariance could not give. */
function FitStrip() {
  const data = useStore((s) => s.data);
  const uncertainty = data ? exportUncertainty(data) : null;
  if (uncertainty === null) {
    return <Callout tone="info">This run was written without uncertainty.</Callout>;
  }
  if (uncertainty.status === "unavailable") {
    return (
      <Callout tone="warning" title="No parameter covariance">
        {unavailableText(uncertainty.reason)}
      </Callout>
    );
  }
  return (
    <div className="flex flex-col gap-2">
      <p className="font-mono text-xs text-fg-muted">
        {uncertainty.dof} degrees of freedom · {uncertainty.num_residuals} residuals
        {(uncertainty.fixed ?? []).length > 0
          ? ` · held fixed: ${(uncertainty.fixed ?? []).join(", ")}`
          : ""}
      </p>
      {uncertainty.noise.map((n) => (
        <p
          key={`${residualKindName(n.kind)}-${n.camera ?? "all"}`}
          className="font-mono text-xs text-fg-muted"
        >
          {residualKindName(n.kind)}
          {n.camera != null ? ` · camera ${n.camera}` : ""}
          {"sigma0" in n
            ? ` noise σ̂₀ ≈ ${significant(n.sigma0, 3)} × stated`
            : ` noise σ ≈ ${significant(n.sigma, 3)}${residualKindUnit(n.kind)}`}{" "}
          ({n.count} residuals)
        </p>
      ))}
      <Disclosure
        summary="Correlation between parameters"
        count={uncertainty.labels.length}
      >
        <CorrelationMatrix cov={uncertainty} />
      </Disclosure>
    </div>
  );
}

function CopyJson({ set }: { set: CameraParameterSet }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  return (
    <Button
      size="sm"
      onClick={() => {
        navigator.clipboard.writeText(parametersJson(set)).then(
          () => setState("copied"),
          () => setState("failed"),
        );
      }}
    >
      {state === "copied"
        ? "Copied"
        : state === "failed"
          ? "Copy failed"
          : "Copy as JSON"}
    </Button>
  );
}

/** Every parameter of the selected camera, whatever its model: value ± σ at
 * the decimals of σ, its unit, and for a parameter with no σ the reason. */
export function ParametersTab() {
  return (
    <CameraFrame title="Parameters">
      {(target) => <Body run={target.run} camera={target.camera} />}
    </CameraFrame>
  );
}

function Body({ run, camera }: { run: RunRef; camera: number }) {
  const q = useQuery("camera_parameter_values", { run }, () =>
    api.cameraParameterValues({ run }),
  );
  const data = useStore((s) => s.data);
  const uncertainty = data ? exportUncertainty(data) : null;
  if (q.error && !q.data) {
    return (
      <ErrorBox>
        Could not read the parameters: {q.error.message}{" "}
        <Button size="sm" onClick={q.refetch}>
          Retry
        </Button>
      </ErrorBox>
    );
  }
  if (!q.data) return <Skeleton className="h-64 w-full" />;
  const set = q.data.cameras.find((c) => c.camera === camera);
  const other =
    uncertainty?.status === "available"
      ? otherRows(uncertainty, coveredLabels(q.data), camera)
      : [];
  const shared = sharedOf(q.data, camera);
  if (!set)
    return <Callout tone="info">Camera {camera} has no parameters in this run.</Callout>;
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1">
      <div className="flex flex-wrap items-center gap-3">
        <p className="text-sm text-fg" data-testid="model-summary">
          {modelSummary(set)}
        </p>
        <CopyJson set={set} />
      </div>
      <ParameterGroups set={set} />
      <PoseValues poses={shared.poses} />
      <PlaneValues planes={shared.planes} />
      <OtherParameters rows={other} />
      <FitStrip />
    </div>
  );
}
