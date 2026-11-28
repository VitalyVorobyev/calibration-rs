import { Callout, SegmentedControl, Table, type Column } from "@vitavision/ui";
import { useMemo, useState } from "react";
import type {
  CompareRunsOptions,
  Compensation,
  ProjectionComparison,
  ProjectionDifferenceMap,
  RunComparison,
} from "../../api/generated/types";
import { ToolbarGroup } from "../../components/ViewerToolbar";
import { fixed } from "../../lib/format";
import { gridText } from "../inspect/maps/mapText";
import type { MapSource } from "../inspect/maps/mapSource";
import { MapPanel, type MapDescription } from "../inspect/maps/MapPanel";
import { positionName } from "./compareModel";
import { projectionUnavailableWords } from "./compareWords";
import { useComparison, useProjectionDifference } from "./useCompareData";

type Component = "magnitude" | "dx" | "dy";

const COMPONENTS: Record<Component, { label: string; signed: boolean }> = {
  magnitude: { label: "|d|", signed: false },
  dx: { label: "dx", signed: true },
  dy: { label: "dy", signed: true },
};

const px = (v: number) => fixed(v, 3);

/** Below this the largest difference rounds to 0.000 px: the plane holds
 * rounding error, and its colour scale and iso-lines would draw that. */
const IDENTICAL_PX = 5e-4;

interface Pair {
  /** Index among the runs after the reference (0-based). */
  other: number;
  name: string;
  camera: string;
  /** The summary under the chosen alignment; `null` while it is read. */
  projection: ProjectionComparison | null;
}

/** The pairs of the comparison, each with its projection summary from
 * `summary` (the same comparison under the alignment chosen). */
function pairsOf(comparison: RunComparison, summary: RunComparison | undefined): Pair[] {
  return comparison.differences.flatMap((d, other) =>
    d.cameras.map((c) => ({
      other,
      name: comparison.runs[d.run]?.name ?? "",
      camera: c.id,
      projection:
        summary?.differences[other]?.cameras.find((x) => x.id === c.id)?.projection ??
        null,
    })),
  );
}

/** What the map says, from the reply's numbers only. */
function mapStatement(
  map: Extract<ProjectionDifferenceMap, { status: "compared" }>,
  reference: string,
  other: string,
): string {
  const outside = map.valid_nodes - map.fit_nodes;
  const extrapolated =
    outside > 0
      ? ` ${outside} of ${map.valid_nodes} nodes lie outside what the features of either run cover; there the difference is an extrapolation of both models.`
      : "";
  return `${other} projects within ${px(map.max_fit_px)} px of ${reference} over the covered region (rms ${px(map.rms_fit_px)} px).${extrapolated}`;
}

interface Context {
  reference: string;
  other: string;
  camera: string;
  component: Component;
  map: Extract<ProjectionDifferenceMap, { status: "compared" }> | null;
}

function build(source: MapSource, c: Context): MapDescription {
  const quantity = COMPONENTS[c.component];
  return {
    quantity: {
      channel: c.component,
      label: quantity.label,
      kind: quantity.signed ? "signed" : "magnitude",
    },
    readout: [
      { channel: "dx", label: "dx" },
      { channel: "dy", label: "dy" },
      { channel: "magnitude", label: "|d|" },
      { channel: "fitted", label: "covered" },
    ],
    arrows: { dx: "dx", dy: "dy", on: true },
    isoLines: true,
    mask: {
      channel: "fitted",
      minimum: 0.5,
      words: {
        switch: "Fade extrapolated",
        title: "Fade the nodes outside the region the runs' features cover",
        key: "outside the covered region: both models extrapolate",
      },
    },
    name: `Projection difference of ${c.camera}`,
    provenance: `${c.other} − ${c.reference} · ${c.camera} · ${gridText(source.grid)} · ${source.grid.image_size[0]} × ${source.grid.image_size[1]} px · frames ${!c.map ? "" : c.map.compensation === "rotation" ? `aligned by a rotation of ${fixed(c.map.rotation_deg, 3)}°` : "as calibrated"}`,
    statement: c.map ? mapStatement(c.map, c.reference, c.other) : null,
    children: (
      <p>
        The displacement is where a ray lands under {c.other} minus where it lands under{" "}
        {c.reference}, in pixels; arrows point the way the pixel moves. It is a difference
        of projections, not of parameters, so it compares any two models. It has no σ of
        its own.
      </p>
    ),
  };
}

const COMPENSATION_WORDS: Record<Compensation, string> = {
  rotation:
    "The orientation of a camera frame is not separately observable from what uses the camera, so the rotation of the other run's frame that brings it closest to the reference is removed first. What remains is the part no re-orientation removes.",
  none: "The rays are compared as the two calibrations place them, with no re-orientation: for camera frames used as calibrated.",
};

/** Where the projections of the runs differ over the image: a table of
 * every run and camera (a row picks the pair), and for the picked pair the
 * difference map: magnitude and direction over the sensor, faded outside
 * the region the features cover, with the numbers the comparison gives. */
export function MapTab({
  dir,
  comparison,
  options,
}: {
  dir: string;
  comparison: RunComparison;
  /** What the comparison was asked with (the project's thresholds). */
  options: CompareRunsOptions;
}) {
  const [pick, setPick] = useState<{ other: number; camera: string } | null>(null);
  const [component, setComponent] = useState<Component>("magnitude");
  const [compensation, setCompensation] = useState<Compensation>("rotation");
  // The default alignment is the comparison's own; another is read once more.
  const summary = useComparison(
    dir,
    comparison.runs.map((r) => r.name),
    compensation === "rotation" ? options : { ...options, compensation },
  );
  const pairs = useMemo(
    () => pairsOf(comparison, summary.data),
    [comparison, summary.data],
  );
  const chosen =
    pairs.find((p) => p.other === pick?.other && p.camera === pick.camera) ??
    pairs[0] ??
    null;
  const reference = comparison.runs[0]?.name ?? "";
  const { map, state } = useProjectionDifference(
    dir,
    reference,
    chosen?.name ?? null,
    chosen?.camera ?? null,
    compensation,
  );

  const columns = useMemo(() => pairColumns(), []);
  const compared = map?.status === "compared" ? map : null;

  const controls = (
    <>
      <ToolbarGroup label="Component">
        <SegmentedControl
          aria-label="Component"
          value={component}
          onValueChange={(v) => setComponent((v || "magnitude") as Component)}
          options={(Object.keys(COMPONENTS) as Component[]).map((c) => ({
            value: c,
            label: COMPONENTS[c].label,
          }))}
        />
      </ToolbarGroup>
      <ToolbarGroup label="Align frames">
        <SegmentedControl
          aria-label="Align frames"
          value={compensation}
          onValueChange={(v) => setCompensation(v === "none" ? "none" : "rotation")}
          options={[
            { value: "rotation", label: "Rotation" },
            { value: "none", label: "None" },
          ]}
        />
      </ToolbarGroup>
    </>
  );

  if (pairs.length === 0) {
    return (
      <Callout tone="info" title="No camera to compare">
        The runs have no camera in common.
      </Callout>
    );
  }
  return (
    <div className="flex flex-col gap-3">
      <Table
        caption="Projection difference of each run and camera against the reference"
        columns={columns}
        rows={pairs}
        rowKey={(p) => `${p.other}:${p.camera}`}
        isRowActive={(p) => p === chosen}
        onRowClick={(p) => setPick({ other: p.other, camera: p.camera })}
      />
      <p className="max-w-3xl text-xs text-fg-muted">
        {COMPENSATION_WORDS[compensation]}
      </p>
      {map?.status === "unavailable" ? (
        <>
          <MapControlsOnly>{controls}</MapControlsOnly>
          <Callout
            tone="info"
            title={`No projection difference for ${chosen?.camera} between ${reference} and ${chosen?.name}`}
          >
            The map is not made because {projectionUnavailableWords(map.reason)}.
          </Callout>
        </>
      ) : compared && compared.max_valid_px < IDENTICAL_PX ? (
        <>
          <MapControlsOnly>{controls}</MapControlsOnly>
          <Callout tone="info" title="Nothing to draw: the projections are the same">
            {chosen?.name} sends every ray of {chosen?.camera} to the same pixel as{" "}
            {reference}: the largest difference is {px(compared.max_valid_px)} px, below
            what a map can show. A map of it would only draw rounding error.
          </Callout>
        </>
      ) : (
        <MapPanel
          state={state}
          identity={`${chosen?.name}:${chosen?.camera}:${compensation}`}
          controls={controls}
          build={build}
          context={
            {
              reference,
              other: chosen?.name ?? "",
              camera: chosen?.camera ?? "",
              component,
              map: compared,
            } satisfies Context
          }
          failure={(message) => `Could not make the difference map: ${message}`}
        />
      )}
    </div>
  );
}

function MapControlsOnly({ children }: { children: React.ReactNode }) {
  return (
    <div
      role="toolbar"
      aria-label="Map display"
      className="flex flex-wrap gap-x-4 gap-y-1.5"
    >
      {children}
    </div>
  );
}

function pairColumns(): Column<Pair>[] {
  const num =
    (f: (p: Extract<ProjectionComparison, { status: "compared" }>) => string) =>
    (p: Pair) =>
      p.projection === null
        ? "…"
        : p.projection.status === "compared"
          ? f(p.projection)
          : "–";
  return [
    {
      key: "run",
      header: "Run",
      cell: (p) => (
        <span className="font-mono text-xs">
          {positionName(p.other + 1)} · {p.name}
        </span>
      ),
    },
    {
      key: "camera",
      header: "Camera",
      cell: (p) => <span className="font-mono text-xs">{p.camera}</span>,
    },
    {
      key: "rms",
      header: "RMS px",
      group: "Covered region",
      numeric: true,
      cell: (p) =>
        p.projection === null ? (
          "…"
        ) : p.projection.status === "compared" ? (
          px(p.projection.rms_fit_px)
        ) : (
          <span className="font-sans text-xs text-fg-muted">
            unavailable: {projectionUnavailableWords(p.projection.reason)}
          </span>
        ),
    },
    {
      key: "max",
      header: "Max px",
      group: "Covered region",
      numeric: true,
      cell: num((p) => px(p.max_fit_px)),
    },
    {
      key: "rmsv",
      header: "RMS px",
      group: "Every node",
      numeric: true,
      cell: num((p) => px(p.rms_valid_px)),
    },
    {
      key: "maxv",
      header: "Max px",
      group: "Every node",
      numeric: true,
      cell: num((p) => px(p.max_valid_px)),
    },
    {
      key: "rot",
      header: "Rotation removed",
      numeric: true,
      cell: num((p) =>
        p.compensation === "rotation" ? `${fixed(p.rotation_deg, 3)}°` : "none",
      ),
    },
    {
      key: "nodes",
      header: "Nodes in the fit",
      numeric: true,
      cell: num((p) => `${p.fit_nodes} of ${p.valid_nodes}`),
    },
  ];
}
