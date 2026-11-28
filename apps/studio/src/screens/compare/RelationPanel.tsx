import { Badge, Disclosure } from "@vitavision/ui";
import type { Exclusions, RunComparison } from "../../api/generated/types";
import { formatCount } from "../../lib/format";
import { positionName } from "./compareModel";
import { refusalWords, relationWords } from "./compareWords";

const count = (e: Exclusions | undefined) =>
  (e?.features?.length ?? 0) + (e?.images?.length ?? 0) + (e?.views?.length ?? 0);

/** How each run relates to the reference, in words, before any number:
 * whether their observations are the same, nested or independent, which
 * decides which tests exist, and what changed between them (camera
 * models, settings, exclusions). A stated relation the exports contradict
 * is said as refused. */
export function RelationPanel({ comparison }: { comparison: RunComparison }) {
  const reference = comparison.runs[0]?.name ?? "";
  return (
    <section aria-label="How the runs relate" className="flex flex-col gap-2">
      <h2 className="text-[10px] font-semibold tracking-wider text-fg-muted uppercase">
        How the runs relate to {reference}
      </h2>
      <ul className="flex flex-col gap-2">
        {comparison.differences.map((d) => {
          const name = comparison.runs[d.run]?.name ?? `run ${d.run}`;
          const words = relationWords(d.relation, name, reference);
          const settings = d.config_changes ?? [];
          const only = d.exclusion_changes;
          return (
            <li
              key={d.run}
              className="flex flex-col gap-1 rounded-control border border-line bg-surface px-3 py-2"
            >
              <div className="flex flex-wrap items-center gap-2">
                <Badge>{positionName(d.run)}</Badge>
                <span className="font-mono text-xs">{name}</span>
                <Badge tone={words.tone}>{words.label}</Badge>
                {d.relation_refused && (
                  <Badge tone="warning">Stated relation refused</Badge>
                )}
              </div>
              <p className="text-xs text-fg">{words.text}</p>
              {d.relation_refused && (
                <p className="text-xs text-warn">{refusalWords(d.relation_refused)}</p>
              )}
              {d.data && (
                <p className="font-mono text-[11px] text-fg-muted">
                  {formatCount(d.data.shared_images)} images and{" "}
                  {formatCount(d.data.shared_features)} features in common · only{" "}
                  {reference}: {formatCount(d.data.reference_only_images)} images,{" "}
                  {formatCount(d.data.reference_only_features)} features · only {name}:{" "}
                  {formatCount(d.data.other_only_images)} images,{" "}
                  {formatCount(d.data.other_only_features)} features
                </p>
              )}
              {d.model_changes.length > 0 && (
                <p className="text-xs text-fg-muted">
                  The camera model differs for {d.model_changes.join(", ")}: its
                  parameters name different quantities and are not compared; the
                  projection map is the comparison.
                </p>
              )}
              {d.unmatched_cameras.length > 0 && (
                <p className="text-xs text-fg-muted">
                  Not in both runs, so not compared: {d.unmatched_cameras.join(", ")}.
                </p>
              )}
              {only && count(only.reference_only) + count(only.other_only) > 0 && (
                <p className="text-xs text-fg-muted">
                  Exclusions differ: {reference} made {count(only.reference_only)} that{" "}
                  {name} did not, {name} made {count(only.other_only)} that {reference}{" "}
                  did not.
                </p>
              )}
              {settings.length > 0 && (
                <Disclosure summary="Settings that differ" count={settings.length}>
                  <ul className="flex flex-col gap-0.5 font-mono text-[11px]">
                    {settings.map((c) => (
                      <li key={c.path} className="flex flex-wrap gap-x-2">
                        <span className="text-fg">{c.path}</span>
                        <span className="text-fg-muted">
                          {c.reference ?? "not set"} → {c.other ?? "not set"}
                        </span>
                      </li>
                    ))}
                  </ul>
                </Disclosure>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
