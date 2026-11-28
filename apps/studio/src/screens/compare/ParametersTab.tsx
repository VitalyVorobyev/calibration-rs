import { Callout, DeltaCell, Switch, Table, type Column } from "@vitavision/ui";
import { useMemo, useState } from "react";
import type { JointTest, RunComparison } from "../../api/generated/types";
import { fixed, formatWithSigma } from "../../lib/format";
import {
  ALPHA,
  MATERIAL_SIGMA,
  cameraBlocks,
  signed,
  testState,
  testTally,
  visibleRows,
  type CameraBlock,
  type ParameterRow,
  positionName,
} from "./compareModel";
import { NO_TEST } from "./compareWords";
import { PoseBlocks } from "./PoseBlocks";
import { TestCell } from "./TestCell";

const valueText = (value: number, sigma: number | null | undefined) =>
  formatWithSigma(
    value,
    sigma != null && sigma > 0 ? { kind: "sigma", sigma } : { kind: "fixed" },
  ).text.replace(" ·", "");

/** The parameters of each camera across the runs: value ± σ of the
 * reference, then for each other run its value with the change from the
 * reference, the change in the reference's σ (how much it matters, which
 * is not a test) and the test where one is valid. The joint test of a
 * camera's parameters is read first. */
export function ParametersTab({ comparison }: { comparison: RunComparison }) {
  const [onlyMatters, setOnlyMatters] = useState(false);
  const blocks = useMemo(() => cameraBlocks(comparison), [comparison]);
  const names = comparison.runs.map((r) => r.name);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Switch
          checked={onlyMatters}
          label="Only differences that matter"
          onCheckedChange={setOnlyMatters}
        />
        <p className="max-w-2xl text-xs text-fg-muted">
          Matters: a test rejects agreement at the {ALPHA * 100} % level or, where no test
          applies, the change is at least {MATERIAL_SIGMA} σ of the reference. Δ in σ says
          how much a change is worth at the precision the reference claims; two fits of
          the same images can differ by many σ without disagreeing.
        </p>
      </div>
      {blocks.length === 0 && (
        <Callout tone="info" title="No camera to compare">
          The runs have no camera in common.
        </Callout>
      )}
      {blocks.map((block) => (
        <CameraTable
          key={block.id}
          block={block}
          names={names}
          onlyMatters={onlyMatters}
        />
      ))}
      <PoseBlocks comparison={comparison} />
    </div>
  );
}

function jointLine(joint: JointTest | null, name: string): string {
  if (joint === null) return `${name}: no joint test`;
  if (joint.kind === "not_available")
    return `${name}: no joint test (${NO_TEST[joint.reason].long})`;
  const verdict = testState(joint) === "significant" ? "disagree" : "agree";
  return `${name}: χ² ${fixed(joint.chi2, 1)} on ${joint.dof} dof, p ${joint.p_value < 0.001 ? "< 0.001" : fixed(joint.p_value, 3)}: the runs ${verdict} at the ${ALPHA * 100} % level`;
}

function CameraTable({
  block,
  names,
  onlyMatters,
}: {
  block: CameraBlock;
  names: string[];
  onlyMatters: boolean;
}) {
  const rows = visibleRows(block.rows, onlyMatters);
  const tally = testTally(block.rows);
  const columns = useMemo(() => buildColumns(names), [names]);
  const notCompared = block.status
    .map((s, i) => (s === "same_model" ? null : names[i + 1]))
    .filter((n): n is string => n !== null);

  return (
    <section aria-label={`Parameters of ${block.id}`} className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
        <h3 className="font-mono text-sm text-fg">{block.id}</h3>
        <span className="text-xs text-fg-muted">
          {rows.length === block.rows.length
            ? `${block.rows.length} parameters`
            : `${rows.length} of ${block.rows.length} parameters`}
          {tally.tested > 0 &&
            ` · ${tally.significant} of ${tally.tested} tests significant`}
        </span>
      </div>
      {block.joints.some((j) => j !== null) && (
        <ul className="flex flex-col gap-0.5 text-xs text-fg-muted">
          {block.joints.map((j, i) =>
            j === null ? null : (
              <li key={names[i + 1]} className="font-mono text-[11px]">
                {jointLine(j, names[i + 1] ?? "")}
              </li>
            ),
          )}
        </ul>
      )}
      {notCompared.length > 0 && (
        <p className="text-xs text-fg-muted">
          {notCompared.join(", ")}: the camera model differs from the reference&apos;s, so
          its parameters are not compared here.
        </p>
      )}
      {block.rows.length > 0 && (
        <Table
          caption={`Parameters of ${block.id} across the runs`}
          columns={columns}
          rows={rows}
          rowKey={(r) => r.label}
          pinFirstColumn
          empty="No parameter differs enough to matter."
        />
      )}
    </section>
  );
}

function buildColumns(names: string[]): Column<ParameterRow>[] {
  const cols: Column<ParameterRow>[] = [
    {
      key: "label",
      header: "Parameter",
      sortable: true,
      sortValue: (r) => r.label,
      cell: (r) => <span className="font-mono text-xs">{r.label}</span>,
    },
    {
      key: "ref",
      header: "Value ± σ",
      group: `${names[0]} (reference)`,
      numeric: true,
      cell: (r) => valueText(r.reference, r.referenceSigma),
    },
  ];
  names.slice(1).forEach((name, k) => {
    const group = `${positionName(k + 1)} · ${name}`;
    cols.push(
      {
        key: `v${k}`,
        header: "Value ± σ (Δ)",
        group,
        numeric: true,
        cell: (r) => {
          const p = r.others[k];
          if (!p) return <span className="text-fg-subtle">not in this run</span>;
          const size = p.delta_over_reference_sigma;
          return (
            <DeltaCell
              value={p.other}
              delta={p.delta}
              magnitude={size == null ? null : Math.abs(size) / 3}
              format={() => valueText(p.other, p.other_sigma)}
              deltaFormat={(d) => signed(d).replace(/^[+−]/, "")}
              deltaLabel="change from the reference"
            />
          );
        },
      },
      {
        key: `s${k}`,
        header: "Δ in σ",
        group,
        numeric: true,
        sortable: true,
        firstSort: "descending",
        sortValue: (r) => {
          const v = r.others[k]?.delta_over_reference_sigma;
          return v == null ? null : Math.abs(v);
        },
        cell: (r) => {
          const v = r.others[k]?.delta_over_reference_sigma;
          return v == null ? (
            <span
              className="text-fg-subtle"
              title="The reference has no σ for this parameter"
            >
              –
            </span>
          ) : (
            signed(v)
          );
        },
      },
      {
        key: `t${k}`,
        header: "Test",
        group,
        cell: (r) => {
          const p = r.others[k];
          return p ? <TestCell test={p.test} /> : null;
        },
      },
    );
  });
  return cols;
}
