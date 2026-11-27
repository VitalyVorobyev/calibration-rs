import {
  Button,
  ConfirmDialog,
  DensityProvider,
  Empty,
  SegmentedControl,
  toast,
} from "@vitavision/ui";
import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { api } from "../api";
import type { ProjectView, RunInfo } from "../api/generated/types";
import { runsNewestFirst } from "../project/projectModel";
import { useProject } from "../project/projectStore";
import { useRunEditor } from "../project/runEditor";
import { lowestRmsRun, runRow, type RunRow } from "../project/runRows";
import { useCompareSet } from "../session/compareSet";
import { useReadOnly } from "../session/jobs";
import { useStore } from "../store";
import { RunCard } from "./RunCard";
import { RunDetails } from "./RunDetails";
import { parsePlace, placePath, screenById } from "./screens";

type Order = "newest" | "rms";

/** The persistent history of the open project's runs, beside every screen
 * that reads runs. Each run is a card: its verdict, its RMS (and per
 * camera), the views and features it used, what its settings changed from
 * the project's, whether it is stale, with a tick for Compare and a star
 * to keep it. Choosing one makes it the current run (the same choice as
 * the context bar's) and opens its details: label, notes, and how its
 * settings differ from the editor's. */
export function RunList({ project }: { project: ProjectView | null }) {
  const shown = useStore((s) => s.run);
  const ticked = useCompareSet((s) => s.ids);
  const readOnly = useReadOnly();
  const navigate = useNavigate();
  const { pathname, search } = useLocation();
  const [order, setOrder] = useState<Order>("newest");
  const [pendingDelete, setPendingDelete] = useState<RunInfo | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    useCompareSet
      .getState()
      .follow(project?.dir ?? null, project?.runs.map((r) => r.id) ?? []);
  }, [project]);

  const rows = useMemo<{ run: RunInfo; row: RunRow }[]>(() => {
    if (!project) return [];
    const mapped = runsNewestFirst(project.runs).map((run) => ({
      run,
      row: runRow(run, project),
    }));
    return order === "rms"
      ? [...mapped].sort((a, b) => (a.row.rms ?? Infinity) - (b.row.rms ?? Infinity))
      : mapped;
  }, [project, order]);
  const lowest = useMemo(() => lowestRmsRun(rows.map((r) => r.row)), [rows]);

  if (!project) return null;
  const place = parsePlace(pathname, search);
  const selectedId = shown !== null && shown.dir === project.dir ? shown.id : null;

  const open = (screen: "inspect" | "calibrate" | "compare", id: string) =>
    void navigate(
      placePath({
        screen,
        tab: screen === "inspect" ? (place.tab ?? "frames") : null,
        project: project.dir,
        run: id,
      }),
    );

  const rerun = async (run: RunInfo) => {
    if (await useRunEditor.getState().loadRun(run)) {
      void useProject.getState().openRun(run.id);
      open("calibrate", run.id);
      // Focus the editor, as the action promises.
      requestAnimationFrame(() =>
        document
          .querySelector<HTMLElement>(
            'section[aria-label="Parameters"] input, section[aria-label="Parameters"] button',
          )
          ?.focus(),
      );
    }
  };

  const remove = async () => {
    if (!pendingDelete) return;
    setDeleting(true);
    await useProject.getState().deleteRun(pendingDelete.id);
    setDeleting(false);
    setPendingDelete(null);
  };

  return (
    <DensityProvider value="compact">
      <div className="flex flex-col gap-2.5 p-2.5">
        <div className="flex items-end justify-between gap-2">
          <div className="flex min-w-0 flex-col gap-0.5">
            <h2 className="text-[10px] font-semibold tracking-wider text-fg-muted uppercase">
              Runs · {rows.length}
            </h2>
            <span
              className="truncate font-mono text-[10px] text-fg-subtle"
              title={project.data_root}
            >
              data {project.data_root === project.dir ? "." : project.data_root}
            </span>
          </div>
          {rows.length > 1 && (
            <SegmentedControl
              aria-label="Order runs by"
              value={order}
              onValueChange={(v) => setOrder(v === "rms" ? "rms" : "newest")}
              options={[
                { value: "newest", label: "Newest" },
                { value: "rms", label: "RMS" },
              ]}
            />
          )}
        </div>

        {ticked.length >= 2 && (
          <div className="flex items-center justify-between gap-2 rounded-control border border-line bg-surface px-2 py-1.5">
            <span className="text-xs text-fg-muted">{ticked.length} runs ticked</span>
            <div className="flex gap-1">
              <Button
                size="sm"
                variant="ghost"
                onClick={() => useCompareSet.getState().clear()}
              >
                Clear
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => open("compare", ticked[0] ?? "")}
                disabled={!screenById("compare").built}
                title={
                  screenById("compare").built
                    ? "Compare the ticked runs"
                    : "Compare is coming in a later release"
                }
              >
                Compare {ticked.length} ▸
              </Button>
            </div>
          </div>
        )}

        {rows.length === 0 ? (
          <Empty className="py-6">
            No runs yet. Set up the parameters and press Run.
          </Empty>
        ) : (
          <ul aria-label="Saved runs" className="flex flex-col gap-0.5">
            {rows.map(({ run, row }) => {
              const selected = run.id === selectedId;
              const calibrated =
                run.state.kind === "recorded" && run.state.source.kind === "calibrated";
              return (
                <li key={run.id}>
                  <RunCard
                    run={run}
                    row={row}
                    digest={project.input_digest}
                    projectPatch={project.spec.config}
                    selected={selected}
                    ticked={ticked.includes(run.id)}
                    lowest={lowest === run.id}
                    readOnly={readOnly}
                    actions={{
                      onSelect: () => void useProject.getState().openRun(run.id),
                      onTick: () => useCompareSet.getState().toggle(project.dir, run.id),
                      onStar: (starred) =>
                        void useProject.getState().patchNotes(run.id, { starred }),
                      onOpenInspect: () => {
                        void useProject.getState().openRun(run.id);
                        open("inspect", run.id);
                      },
                      onRerun: () => void rerun(run),
                      onEdit: () => {
                        void useProject.getState().openRun(run.id);
                        // The details open under the selected run; the label is first.
                        setTimeout(
                          () =>
                            document
                              .querySelector<HTMLElement>('input[aria-label="Run label"]')
                              ?.focus(),
                          60,
                        );
                      },
                      onDelete: () => setPendingDelete(run),
                      onCopyId: () => {
                        void navigator.clipboard?.writeText(run.id).then(
                          () => toast({ title: `${run.id} copied`, tone: "success" }),
                          () => undefined,
                        );
                      },
                      reportUrl: calibrated
                        ? api.urls.report({ run: { dir: project.dir, id: run.id } })
                        : null,
                    }}
                  />
                  {selected && (
                    <div className="px-1 pt-1 pb-1.5">
                      <RunDetails key={run.id} run={run} row={row} />
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>
      <ConfirmDialog
        open={pendingDelete !== null}
        onOpenChange={(o) => {
          if (!o) setPendingDelete(null);
        }}
        title={`Delete ${pendingDelete?.id ?? "the run"}?`}
        description={`The run folder ${pendingDelete?.dir ?? ""} is deleted with its export, quality report and notes. This cannot be undone.`}
        confirmLabel="Delete"
        destructive
        loading={deleting}
        onConfirm={() => void remove()}
      />
    </DensityProvider>
  );
}
