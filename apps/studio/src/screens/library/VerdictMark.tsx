import { Badge, StatusDot } from "@vitavision/ui";
import type { Verdict } from "../../api/generated/types";
import { VERDICT } from "../../shell/runPills";

/** A verdict in words, in its colour: the badge of a table cell. A project
 * with no verdict says so quietly. */
export function VerdictBadge({ verdict }: { verdict: Verdict | null }) {
  if (verdict === null) return <span className="text-xs text-fg-subtle">no run</span>;
  const { label, tone } = VERDICT[verdict];
  return <Badge tone={tone}>{label}</Badge>;
}

/** The worst verdict of a group as a dot with its word for assistive
 * technology and hover (never colour alone). */
export function VerdictDot({ verdict }: { verdict: Verdict | null }) {
  if (verdict === null) return null;
  const { label, tone } = VERDICT[verdict];
  return (
    <span title={`Worst verdict: ${label}`}>
      <StatusDot tone={tone}>
        <span className="sr-only">Worst verdict: {label}</span>
      </StatusDot>
    </span>
  );
}
