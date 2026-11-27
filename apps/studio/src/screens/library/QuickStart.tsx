import { Badge, Button, Disclosure } from "@vitavision/ui";
import type { SampleInfo } from "../../api/generated/types";
import { openSample } from "../../project/projectActions";
import { useReadOnly } from "../../session/jobs";
import { exportKindLabel } from "../../store/exportKind";

/** The Quick Start samples a back end can fetch: each opens as a project of
 * its own (downloaded on first use). */
export function QuickStart({
  samples,
  collapsed = false,
}: {
  samples: readonly SampleInfo[];
  /** Folded away (a library that has projects does not need them in view). */
  collapsed?: boolean;
}) {
  const readOnly = useReadOnly();
  if (samples.length === 0) return null;
  const list = (
    <ul className="flex flex-col divide-y divide-line rounded-panel border border-line bg-surface">
      {samples.map((sample) => (
        <li key={sample.id} className="flex items-center gap-3 px-3 py-2">
          <div className="flex min-w-0 flex-1 flex-col">
            <span className="text-sm font-medium text-fg">{sample.id}</span>
            <span className="truncate text-xs text-fg-muted">{sample.description}</span>
          </div>
          <Badge>{exportKindLabel(sample.topology)}</Badge>
          <Button
            size="sm"
            disabled={readOnly}
            onClick={() => void openSample(sample.id)}
          >
            Open sample
          </Button>
        </li>
      ))}
    </ul>
  );
  if (collapsed) {
    return (
      <Disclosure summary="Quick start samples" count={samples.length}>
        <div className="pt-2">{list}</div>
      </Disclosure>
    );
  }
  return (
    <section aria-labelledby="samples-title" className="flex flex-col gap-2">
      <h2
        id="samples-title"
        className="text-[10px] font-semibold tracking-wider text-fg-muted uppercase"
      >
        Quick start
      </h2>
      {list}
    </section>
  );
}
