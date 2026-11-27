import { Badge, Button, Panel } from "@vitavision/ui";
import type { SampleInfo } from "../../api/generated/types";
import { topologyInfo } from "./topologies";

interface QuickStartGridProps {
  /** The samples the back end offers (`server_info.samples`). */
  samples: SampleInfo[];
  /** The sample being fetched, if any. */
  busy: string | null;
  onUse: (id: string) => void;
}

/** The Quick Start samples: each is fetched on first use (into the back
 * end's samples folder) and opened as a project of its own. */
export function QuickStartGrid({ samples, busy, onUse }: QuickStartGridProps) {
  if (samples.length === 0) return null;
  return (
    <section className="flex flex-col gap-3">
      <h3 className="text-[12px] font-semibold tracking-tight text-fg">Quick start</h3>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {samples.map((sample) => (
          <Panel
            key={sample.id}
            title={sample.id}
            actions={<Badge tone="info">{topologyInfo(sample.topology).label}</Badge>}
          >
            <p className="text-[11px] leading-relaxed text-fg-muted">
              {sample.description}
            </p>
            <div className="mt-2 flex items-center justify-between gap-2">
              <span className="font-mono text-[10px] text-fg-subtle">
                {sample.licence}
              </span>
              <Button
                size="sm"
                variant="primary"
                loading={busy === sample.id}
                disabled={busy !== null}
                onClick={() => onUse(sample.id)}
                aria-label={`Use the ${sample.id} sample`}
              >
                Use
              </Button>
            </div>
          </Panel>
        ))}
      </div>
    </section>
  );
}
