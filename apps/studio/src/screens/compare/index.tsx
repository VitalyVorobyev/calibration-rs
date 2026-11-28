import {
  Button,
  Callout,
  DensityProvider,
  Empty,
  PageHeader,
  Select,
  SkeletonRows,
  Tabs,
} from "@vitavision/ui";
import { RefreshCw } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";
import { useSearchParams } from "react-router";
import type { CompareRunsOptions } from "../../api/generated/types";
import { useProject } from "../../project/projectStore";
import { runStateName, runVerdict } from "../../project/projectModel";
import { useCompareSet } from "../../session/compareSet";
import { useReadOnly } from "../../session/jobs";
import { verdictFacts } from "./compareModel";
import {
  COMPARE_TABS,
  runsOf,
  tabOf,
  withCompare,
  type CompareTab,
} from "./compareParams";
import { FitTab } from "./FitTab";
import { MapTab } from "./MapTab";
import { ParametersTab } from "./ParametersTab";
import { RelationPanel } from "./RelationPanel";
import { RunSet } from "./RunSet";
import { useComparison } from "./useCompareData";

const TAB_LABEL: Record<CompareTab, string> = {
  parameters: "Parameters",
  fit: "Fit quality",
  maps: "Projection difference",
};

/** The Compare screen: the runs ticked in the run list set side by side
 * against the first, the reference. It says how the runs relate before any
 * number, then shows what changed (parameters with their σ, differences
 * and the tests that are valid), how well each fits, and where the
 * projections differ over the image. It ranks nothing: it states what the
 * data supports and leaves the star, the run to keep, to the person. */
export function CompareScreen() {
  const project = useProject((s) => s.project);
  const readOnly = useReadOnly();
  const [params, setParams] = useSearchParams();
  const setDir = useCompareSet((s) => s.dir);
  const stored = useCompareSet((s) => s.ids);
  const dir = project?.dir ?? null;
  const ids = useMemo(() => (setDir === dir ? stored : []), [setDir, dir, stored]);
  const tab = tabOf(params);

  // Address → set: an address that names runs (a link, a pasted address)
  // sets them; the address with none leaves the set alone.
  const urlRuns = params.get("runs") ?? "";
  const applied = useRef(false);
  const tabsRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!project) return;
    const named = runsOf(params).filter((id) => project.runs.some((r) => r.id === id));
    if (named.length === 0 || named.join() === ids.join()) return;
    applied.current = true;
    useCompareSet.getState().setIds(project.dir, named);
    // Only a change of the address's runs or of the project acts.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [urlRuns, project?.dir]);
  // Set → address: the runs and the tab are written, replacing the entry.
  useEffect(() => {
    if (applied.current) {
      applied.current = false;
      return;
    }
    const next = withCompare(params, ids, tab);
    if (next.toString() !== params.toString()) setParams(next, { replace: true });
    // Only a change of the set or the tab acts.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [ids, tab]);

  // The project's own quality thresholds judge each run, as they did when it ran.
  const quality = project?.spec.quality ?? null;
  const options = useMemo<CompareRunsOptions>(
    () => (quality ? { thresholds: quality } : {}),
    [quality],
  );
  const comparison = useComparison(dir, ids, options);
  const names = comparison.data?.runs.map((r) => r.name) ?? ids;
  const runs = project?.runs;
  // Each run's verdict as recorded with it, in the set's order.
  const verdicts = useMemo(
    () =>
      names.map((n) => {
        const run = runs?.find((r) => r.id === n);
        return run ? runVerdict(run) : null;
      }),
    [names, runs],
  );
  const facts = useMemo(
    () => (comparison.data ? verdictFacts(comparison.data, names, verdicts) : []),
    [comparison.data, names, verdicts],
  );

  if (!project) return <Empty>Open a project to compare its runs.</Empty>;
  const addable = project.runs.filter(
    (r) => !ids.includes(r.id) && runStateName(r) !== "Unreadable",
  );
  const pending =
    comparison.data === undefined && comparison.error === null && ids.length >= 2;

  return (
    <DensityProvider value="comfortable">
      <div className="flex h-full min-h-0 flex-col gap-3 overflow-hidden">
        <PageHeader
          title="Compare"
          meta={
            ids.length > 0 ? (
              <span className="font-mono">
                {ids.length} {ids.length === 1 ? "run" : "runs"} · reference {ids[0]}
              </span>
            ) : undefined
          }
          actions={
            <div className="flex items-center gap-2">
              {addable.length > 0 && ids.length < 6 && (
                <Select
                  aria-label="Add a run to the comparison"
                  value=""
                  placeholder="Add a run…"
                  onValueChange={(id) =>
                    id && useCompareSet.getState().toggle(project.dir, id)
                  }
                  options={addable.map((r) => ({
                    value: r.id,
                    label: r.notes.label ? `${r.id} · ${r.notes.label}` : r.id,
                  }))}
                  className="w-44"
                />
              )}
              <Button
                size="sm"
                variant="ghost"
                aria-label="Compare again"
                title="Compare again"
                disabled={ids.length < 2}
                onClick={() => comparison.refetch()}
              >
                <RefreshCw aria-hidden className="size-3.5" />
              </Button>
            </div>
          }
        />

        {ids.length < 2 ? (
          <div className="mx-auto flex max-w-xl flex-col gap-3 pt-6">
            <Empty>
              {ids.length === 0
                ? "Tick two or more runs in the run list to compare them, or add runs above. The first is the reference."
                : "One run is chosen. Tick another in the run list, or add one above: a comparison needs at least two."}
            </Empty>
            {ids.length === 1 && (
              <RunSet
                ids={ids}
                runs={project.runs}
                evidence={undefined}
                readOnly={readOnly}
              />
            )}
          </div>
        ) : (
          <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1 pb-2 [&>*]:shrink-0">
            <RunSet
              ids={ids}
              runs={project.runs}
              evidence={comparison.data?.runs}
              readOnly={readOnly}
            />
            {comparison.error && (
              <Callout
                tone="error"
                title="The runs could not be compared"
                actions={
                  <Button size="sm" onClick={() => comparison.refetch()}>
                    Try again
                  </Button>
                }
              >
                {comparison.error.message}
              </Callout>
            )}
            {pending && (
              <div
                role="status"
                aria-busy
                aria-label="Comparing the runs"
                className="max-w-md"
              >
                <SkeletonRows rows={5} />
              </div>
            )}
            {comparison.data && (
              <>
                <div className="grid gap-3 xl:grid-cols-2">
                  <RelationPanel comparison={comparison.data} />
                  <section aria-label="What the data supports">
                    <Callout tone="info" title="What the data supports">
                      <ul className="flex list-disc flex-col gap-0.5 pl-4">
                        {facts.map((f) => (
                          <li key={f}>{f}</li>
                        ))}
                      </ul>
                      <p className="mt-1.5 text-fg-muted">
                        Nothing here ranks the runs: star the one you accept.
                      </p>
                    </Callout>
                  </section>
                </div>
                <div ref={tabsRef}>
                  <Tabs
                    label="Comparison"
                    idPrefix="compare"
                    active={tab}
                    items={COMPARE_TABS.map((id) => ({ id, label: TAB_LABEL[id] }))}
                    onSelect={(next) => {
                      setParams(withCompare(params, ids, next), { replace: true });
                      // The tab's content, not the summary above it.
                      requestAnimationFrame(() =>
                        tabsRef.current?.scrollIntoView?.({
                          block: "start",
                          behavior: "smooth",
                        }),
                      );
                    }}
                  />
                </div>
                <div
                  role="tabpanel"
                  id={`compare-panel-${tab}`}
                  aria-labelledby={`compare-tab-${tab}`}
                  className="flex flex-col gap-3"
                >
                  {tab === "parameters" && <ParametersTab comparison={comparison.data} />}
                  {tab === "fit" && (
                    <FitTab comparison={comparison.data} verdicts={verdicts} />
                  )}
                  {tab === "maps" && (
                    <MapTab
                      dir={project.dir}
                      comparison={comparison.data}
                      options={options}
                    />
                  )}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </DensityProvider>
  );
}
