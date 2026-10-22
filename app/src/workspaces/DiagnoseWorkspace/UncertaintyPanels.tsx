import { ColormapLegend, colormapValue } from "@vitavision/charts";
import { Callout, ErrorBox, Panel, Table, type Column } from "@vitavision/ui";
import { invoke } from "@tauri-apps/api/core";
import { Fragment, useEffect, useMemo, useState } from "react";
import {
  correlationMatrix,
  parameterRows,
  unavailableText,
  type ParameterRow,
} from "../../lib/uncertainty";
import type {
  ParameterCovariance,
  ProjectionUncertainty,
  Uncertainty,
} from "../../types/generated/diagnose-wire";

interface UncertaintyPanelsProps {
  uncertainty: Uncertainty | null;
  /** The camera the viewer shows; the projection map is this camera's. */
  camera: number;
  /** `[width, height]` of that camera's images, once a frame is decoded. */
  imageSize: [number, number] | null;
}

/** The export's parameter uncertainty: each parameter's value ± σ, their
 * correlations and the active camera's projection σ map. */
export function UncertaintyPanels({
  uncertainty,
  camera,
  imageSize,
}: UncertaintyPanelsProps) {
  if (!uncertainty) {
    return (
      <Panel title="Parameters">
        <p className="text-[11px] text-fg-muted">
          This export was written without uncertainty.
        </p>
      </Panel>
    );
  }
  if (uncertainty.status === "unavailable") {
    return (
      <Callout tone="warning" title="No covariance">
        {unavailableText(uncertainty.reason)}
      </Callout>
    );
  }
  return (
    <>
      <ParametersPanel cov={uncertainty} />
      <CorrelationPanel cov={uncertainty} />
      <ProjectionPanel cov={uncertainty} camera={camera} imageSize={imageSize} />
    </>
  );
}

/** Six significant figures for a value, three for its σ. */
function formatValue(v: number): string {
  return Math.abs(v) >= 1e-3 || v === 0
    ? Number(v.toPrecision(6)).toString()
    : v.toExponential(3);
}

function formatSigma(v: number): string {
  return Math.abs(v) >= 1e-3 || v === 0
    ? Number(v.toPrecision(3)).toString()
    : v.toExponential(2);
}

const NO_VALUES: (number | null)[] = [];

const PARAMETER_COLUMNS: Column<ParameterRow>[] = [
  {
    key: "label",
    header: "parameter",
    cell: (r) => <span className="font-mono text-fg">{r.label}</span>,
  },
  {
    key: "value",
    header: "value",
    numeric: true,
    cell: (r) =>
      r.value == null ? <span className="text-fg-muted">—</span> : formatValue(r.value),
  },
  {
    key: "sigma",
    header: "± σ",
    numeric: true,
    cell: (r) => (
      <span>
        {formatSigma(r.sigma)}
        {r.unit && <span className="ml-1 text-fg-muted">{r.unit}</span>}
      </span>
    ),
  },
];

function ParametersPanel({ cov }: { cov: ParameterCovariance }) {
  // Values tagged with the labels they were fetched for, so a newly loaded
  // export never pairs its σ with the previous export's values.
  const [fetched, setFetched] = useState<{
    labels: string[];
    values: (number | null)[];
  } | null>(null);
  const labels = cov.labels;
  useEffect(() => {
    let cancelled = false;
    invoke<(number | null)[]>("camera_parameter_values", { labels })
      .then((values) => {
        if (!cancelled) setFetched({ labels, values });
      })
      // Values are a convenience next to σ; without them the table still
      // shows every σ.
      .catch(() => {
        if (!cancelled) setFetched({ labels, values: [] });
      });
    return () => {
      cancelled = true;
    };
  }, [labels]);
  const values = fetched?.labels === labels ? fetched.values : NO_VALUES;
  const rows = useMemo(() => parameterRows(cov, values), [cov, values]);
  const fixed = cov.fixed ?? [];

  return (
    <Panel
      title="Parameters"
      actions={
        <span className="font-mono text-[10px] text-fg-muted">
          {cov.dof} dof · {cov.num_residuals} residuals
        </span>
      }
    >
      <section aria-label="Parameter uncertainty" className="flex flex-col gap-2">
        <Table columns={PARAMETER_COLUMNS} rows={rows} rowKey={(r) => r.label} />
        {fixed.length > 0 && (
          <p className="text-[11px] text-fg-muted">
            <span className="uppercase tracking-wider">fixed</span>{" "}
            <span className="font-mono">{fixed.join(", ")}</span>
          </p>
        )}
        {cov.noise.map((n) => (
          <p
            key={`${n.kind}-${n.camera ?? "all"}`}
            className="font-mono text-[11px] text-fg-muted"
          >
            {n.kind.replace("_", " ")}
            {n.camera != null ? ` · cam ${n.camera}` : ""}
            {"sigma0" in n
              ? ` noise σ̂₀ ≈ ${formatSigma(n.sigma0)} × stated`
              : ` noise σ ≈ ${formatSigma(n.sigma)}${n.kind === "laser_plane" ? " m" : " px"}`}{" "}
            ({n.count})
          </p>
        ))}
      </section>
    </Panel>
  );
}

/** |ρ| on a perceptually uniform map; the sign is in each cell's tooltip. */
function CorrelationPanel({ cov }: { cov: ParameterCovariance }) {
  const rho = useMemo(() => correlationMatrix(cov), [cov]);
  const n = cov.labels.length;
  if (n < 2) return null;
  return (
    <Panel title="Correlation">
      <section aria-label="Parameter correlation" className="flex flex-col gap-1.5">
        {/* Rows carry their label; columns follow the same order, so the
            diagonal reads which column is which. */}
        <div
          className="grid items-center gap-x-1.5 font-mono text-[9px] text-fg-muted"
          style={{ gridTemplateColumns: "max-content 1fr" }}
        >
          {rho.map((row, i) => {
            const rowLabel = cov.labels[i]!;
            return (
              <Fragment key={rowLabel}>
                <span className="truncate text-right">{rowLabel}</span>
                <svg
                  viewBox={`0 0 ${n} 1`}
                  className="block w-full"
                  role="img"
                  shapeRendering="crispEdges"
                >
                  <title>{`${rowLabel} correlations`}</title>
                  {row.map((v, j) => (
                    <rect
                      key={cov.labels[j]}
                      x={j}
                      y={0}
                      width={1}
                      height={1}
                      fill={
                        Number.isFinite(v)
                          ? colormapValue("viridis", Math.abs(v), [0, 1])
                          : "none"
                      }
                    >
                      <title>{`${rowLabel} · ${cov.labels[j]}: ρ = ${
                        Number.isFinite(v) ? v.toFixed(3) : "—"
                      }`}</title>
                    </rect>
                  ))}
                </svg>
              </Fragment>
            );
          })}
        </div>
        <ColormapLegend map="viridis" domain={[0, 1]} label="|ρ|" />
      </section>
    </Panel>
  );
}

interface ProjectionPanelProps {
  cov: ParameterCovariance;
  camera: number;
  imageSize: [number, number] | null;
}

type ProjectionResult =
  { status: "ready"; map: ProjectionUncertainty } | { status: "error"; message: string };

/** What a projection map was computed for: a result is current only while
 * all of it still holds. */
interface ProjectionRequest {
  cov: ParameterCovariance;
  camera: number;
  width: number;
  height: number;
}

function ProjectionPanel({ cov, camera, imageSize }: ProjectionPanelProps) {
  const [fetched, setFetched] = useState<{
    request: ProjectionRequest;
    result: ProjectionResult;
  } | null>(null);
  const [width, height] = imageSize ?? [0, 0];
  useEffect(() => {
    if (width === 0 || height === 0) return;
    let cancelled = false;
    const request = { cov, camera, width, height };
    invoke<ProjectionUncertainty>("projection_uncertainty", {
      camera,
      imageWidth: width,
      imageHeight: height,
    })
      .then((map) => {
        if (!cancelled) setFetched({ request, result: { status: "ready", map } });
      })
      .catch((e: unknown) => {
        if (!cancelled)
          setFetched({ request, result: { status: "error", message: String(e) } });
      });
    return () => {
      cancelled = true;
    };
  }, [cov, camera, width, height]);
  const current =
    fetched &&
    fetched.request.cov === cov &&
    fetched.request.camera === camera &&
    fetched.request.width === width &&
    fetched.request.height === height
      ? fetched.result
      : null;

  return (
    <Panel
      title="Projection uncertainty"
      actions={<span className="font-mono text-[10px] text-fg-muted">cam {camera}</span>}
    >
      {imageSize === null ? (
        <p className="text-[11px] text-fg-muted">Waiting for the frame to decode.</p>
      ) : current === null ? (
        <p className="text-[11px] text-fg-muted">Computing…</p>
      ) : current.status === "error" ? (
        <ErrorBox>{current.message}</ErrorBox>
      ) : (
        <ProjectionMap map={current.map} />
      )}
    </Panel>
  );
}

/** The map's nodes with their grid position, row by row from the top-left. */
function gridNodes(map: ProjectionUncertainty) {
  return map.sigma.map((sigma, k) => ({
    column: k % map.columns,
    row: Math.floor(k / map.columns),
    sigma,
  }));
}

/** The grid of major-axis σ, cells spanning the image in its aspect. */
function ProjectionMap({ map }: { map: ProjectionUncertainty }) {
  const [w, h] = map.image_size;
  const domain: [number, number] = [0, map.max];
  // Nodes run from pixel 0 to w − 1 (h − 1); each cell is centred on its
  // node, and the view box crops the half cells beyond the image edge.
  const cellW = (w - 1) / (map.columns - 1);
  const cellH = (h - 1) / (map.rows - 1);
  return (
    <section aria-label="Projection uncertainty map" className="flex flex-col gap-1.5">
      <svg
        viewBox={`-0.5 -0.5 ${w} ${h}`}
        className="w-full"
        role="img"
        shapeRendering="crispEdges"
      >
        {gridNodes(map).map(({ column, row, sigma }) => (
          <rect
            key={`${column},${row}`}
            x={(column - 0.5) * cellW}
            y={(row - 0.5) * cellH}
            width={cellW}
            height={cellH}
            fill={sigma == null ? "none" : colormapValue("viridis", sigma, domain)}
          >
            <title>{sigma == null ? "no ray" : `${sigma.toFixed(3)} px`}</title>
          </rect>
        ))}
      </svg>
      <ColormapLegend map="viridis" domain={domain} unit="px" label="σ" />
      <p className="font-mono text-[11px] text-fg-muted">
        centre {formatSigma(map.center)} · corner{" "}
        {map.corner == null ? "—" : formatSigma(map.corner)} · max {formatSigma(map.max)}{" "}
        px
      </p>
      <p className="text-[11px] text-fg-muted">
        Where a camera-frame ray lands, from the camera&apos;s own parameters; a
        re-estimated pose absorbs part of a principal-point error.
      </p>
    </section>
  );
}
