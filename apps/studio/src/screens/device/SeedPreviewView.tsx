import { Callout, Table, type Column } from "@vitavision/ui";
import { useMemo } from "react";
import type { SeedPreview, SeedSpread } from "../../api/generated/types";
import { significant } from "../../lib/format";
import { folderName } from "../../project/projectModel";
import { lensLine, seedValues, type SeedValueRow } from "./seedModel";

/** What a seed will contain: where it comes from, each value (with its
 * spread across the runs of an aggregate) and what was left out. */
export function SeedPreviewView({ preview }: { preview: SeedPreview }) {
  const { seed, runs, spread, omitted } = preview;
  const aggregate = spread.length > 0;
  return (
    <div className="flex flex-col gap-3">
      <p className="text-xs text-fg-muted">
        Made from {runs.length === 1 ? "run" : `${runs.length} runs`}:{" "}
        {runs.map((r, i) => (
          <span key={`${r.project}/${r.id}`}>
            {i > 0 && ", "}
            <span className="font-mono text-fg" title={r.project}>
              {r.id}
            </span>{" "}
            of {folderName(r.project)}
          </span>
        ))}
        .
      </p>
      {omitted.length > 0 && (
        <Callout tone="warning" title="Left out of the seed">
          Only some of the runs have {omitted.join(", ")}, so the seed does not carry{" "}
          {omitted.length === 1 ? "it" : "them"} rather than filling{" "}
          {omitted.length === 1 ? "it" : "them"} in.
        </Callout>
      )}
      <ul className="flex flex-col gap-1 text-xs">
        {seed.cameras.map((camera) => (
          <li key={camera.id} className="flex gap-2">
            <span className="font-mono text-fg">{camera.id}</span>
            <span className="text-fg-muted">
              {lensLine(camera)}
              {camera.cam_se3_rig ? " · mount" : ""}
              {camera.laser_plane ? " · laser plane" : ""}
            </span>
          </li>
        ))}
        {seed.handeye && (
          <li className="flex gap-2">
            <span className="font-mono text-fg">hand-eye</span>
            <span className="text-fg-muted">{seed.handeye.mode.replace(/_/g, " ")}</span>
          </li>
        )}
      </ul>
      {aggregate ? <SpreadTable spread={spread} /> : <ValueTable seed={preview} />}
    </div>
  );
}

function ValueTable({ seed }: { seed: SeedPreview }) {
  const rows = useMemo(() => seedValues(seed.seed), [seed.seed]);
  return (
    <Table
      caption="Values of the seed"
      columns={VALUE_COLUMNS}
      rows={rows}
      rowKey={(r) => `${r.camera}/${r.label}`}
      empty="The seed holds no values."
    />
  );
}

/** The aggregate's values: the median is the seeded value; the robust σ,
 * min and max say how much the runs disagree. */
function SpreadTable({ spread }: { spread: SeedSpread[] }) {
  return (
    <Table
      caption="Values of the seed and their spread across the runs"
      columns={SPREAD_COLUMNS}
      rows={spread}
      rowKey={(r) => r.label}
    />
  );
}

const VALUE_COLUMNS: Column<SeedValueRow>[] = [
  {
    key: "label",
    header: "Value",
    cell: (r) => (
      <span className="font-mono text-xs">
        {r.camera ? `${r.camera}.${r.label}` : r.label}
      </span>
    ),
  },
  {
    key: "value",
    header: "Seeded value",
    numeric: true,
    cell: (r) => significant(r.value),
  },
  {
    key: "unit",
    header: "Unit",
    cell: (r) => <span className="text-fg-muted">{r.unit}</span>,
  },
];

const SPREAD_COLUMNS: Column<SeedSpread>[] = [
  {
    key: "label",
    header: "Value",
    cell: (r) => <span className="font-mono text-xs">{r.label}</span>,
  },
  {
    key: "median",
    header: "Seeded (median)",
    numeric: true,
    cell: (r) => significant(r.median),
  },
  {
    key: "sigma",
    header: "Robust σ",
    numeric: true,
    cell: (r) => significant(r.mad_sigma, 3),
  },
  { key: "min", header: "Min", numeric: true, cell: (r) => significant(r.min) },
  { key: "max", header: "Max", numeric: true, cell: (r) => significant(r.max) },
  { key: "n", header: "Runs", numeric: true, cell: (r) => r.values.length },
];
