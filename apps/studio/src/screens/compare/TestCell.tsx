import { Badge } from "@vitavision/ui";
import type { DifferenceTest, JointTest } from "../../api/generated/types";
import { fixed } from "../../lib/format";
import { testState } from "./compareModel";
import { NO_TEST } from "./compareWords";

const pText = (p: number) => (p < 0.001 ? "p < 0.001" : `p ${fixed(p, 3)}`);

/** A test's outcome in a cell: significant or consistent at the 5 % level
 * with its statistic and p-value, or, for a test that is not made, the
 * reason in words (never a blank, never a statistic). */
export function TestCell({ test }: { test: DifferenceTest | JointTest }) {
  if (test.kind === "not_available") {
    const words = NO_TEST[test.reason];
    return (
      <span
        className="flex flex-col text-xs text-fg-muted"
        title={`Not tested: ${words.long}`}
      >
        <span>not tested</span>
        <span className="text-[11px] text-fg-subtle">{words.short}</span>
      </span>
    );
  }
  const significant = testState(test) === "significant";
  const statistic =
    "z" in test
      ? `z ${test.z > 0 ? "+" : test.z < 0 ? "−" : ""}${fixed(Math.abs(test.z), 2)}`
      : `χ² ${fixed(test.chi2, 1)} · ${test.dof} dof`;
  return (
    <span className="flex flex-col items-start gap-0.5">
      <Badge tone={significant ? "info" : "neutral"}>
        {significant ? "significant" : "consistent"}
      </Badge>
      <span className="font-mono text-[11px] text-fg-muted">
        {statistic} · {pText(test.p_value)}
      </span>
    </span>
  );
}
