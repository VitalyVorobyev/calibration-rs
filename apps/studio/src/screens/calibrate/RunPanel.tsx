import { getAtPath } from "@vitavision/forms";
import { Badge, Button, Callout, Input, Switch } from "@vitavision/ui";
import { Play, Save } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router";
import { api } from "../../api";
import type { ProjectView } from "../../api/generated/types";
import { changedPaths, pathLabel, shortValue } from "../../project/configEdit";
import { projectConfig, useRunEditor, workingConfig } from "../../project/runEditor";
import { useReadOnly } from "../../session/jobs";
import { useQuery } from "../../session/query";
import { placePath } from "../../shell/screens";
import { FailureView } from "./FailureView";
import { RunProgress } from "./RunProgress";
import { SeedRow } from "./SeedRow";
import type { EditorCheck } from "./useEditorCheck";

/** "This run": what pressing Run will do, from the project's own state,
 * the Run and Save-as-default buttons, and, while a calibration goes, its
 * progress; after a failure, what failed and what to try. */
export function RunPanel({
  project,
  check,
  onRun,
}: {
  project: ProjectView;
  check: EditorCheck;
  onRun: (opts: { label: string; redetect: boolean }) => void;
}) {
  const editor = useRunEditor();
  const readOnly = useReadOnly();
  const [label, setLabel] = useState("");
  const [redetect, setRedetect] = useState(false);
  const [saving, setSaving] = useState(false);
  const working = workingConfig(editor, project);
  const fromProject = projectConfig(project, editor.form);

  const views = useQuery(
    "project_views",
    { dir: project.dir },
    () => api.projectViews({ dir: project.dir }),
    project.dataset.kind === "ok",
  );
  const blockers = useMemo(() => views.data?.messages ?? [], [views.data]);

  const changes = useMemo(
    () => changedPaths(working, fromProject),
    [working, fromProject],
  );
  const exclusions = countExclusions(project);
  const busy = editor.starting || editor.jobId !== null;

  const reason = readOnly
    ? "The connection to the back end is lost."
    : project.dataset.kind !== "ok"
      ? "The dataset cannot be read."
      : blockers.length > 0
        ? `Fix the dataset first: ${blockers[0]!.message}`
        : editor.form === null
          ? "The configuration form is loading."
          : check.state.status === "error"
            ? "A parameter does not check; see the editor."
            : check.state.status === "checking"
              ? "Checking the parameters…"
              : busy
                ? "A calibration is running."
                : null;

  const run = () => {
    if (reason === null) onRun({ label, redetect });
  };
  // ⌘↵ / Ctrl+↵ runs, also from inside a field.
  useRunKey(run);

  const saveDefault = async () => {
    setSaving(true);
    await editor.saveAsDefault(project, working);
    setSaving(false);
  };

  const tone = project.dataset.kind === "ok" ? undefined : "defect";
  return (
    <aside aria-label="This run" className="flex flex-col gap-4">
      <h2 className="text-sm font-semibold tracking-tight text-fg">This run</h2>

      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-xs">
        <dt className="text-fg-muted">Dataset</dt>
        <dd className="text-fg">
          {views.data
            ? `${views.data.views.length} views · ${views.data.cameras.length} camera${views.data.cameras.length === 1 ? "" : "s"}`
            : project.dataset.kind === "ok"
              ? "Reading…"
              : "Cannot be read"}
        </dd>
        <dt className="text-fg-muted">Exclusions</dt>
        <dd className="text-fg">{exclusions === 0 ? "None" : exclusions}</dd>
        <dt className="text-fg-muted">Seed</dt>
        <dd className="min-w-0 text-fg">
          <SeedRow project={project} />
        </dd>
      </dl>

      {blockers.length > 0 && (
        <Callout
          tone="error"
          title="Fix the dataset first"
          actions={
            <Button asChild size="sm" variant="secondary">
              <Link
                to={placePath({
                  screen: "dataset",
                  tab: null,
                  project: project.dir,
                  run: null,
                })}
              >
                Open Dataset
              </Link>
            </Button>
          }
        >
          <ul className="list-disc pl-4 text-xs">
            {blockers.slice(0, 3).map((m) => (
              <li key={`${m.field ?? ""}${m.message}`}>{m.message}</li>
            ))}
          </ul>
        </Callout>
      )}

      <section aria-label="Changes from the project" className="flex flex-col gap-1.5">
        <div className="flex items-center gap-2">
          <h3 className="text-xs font-semibold text-fg">Configuration</h3>
          <Badge tone={changes.length > 0 ? "info" : "neutral"}>
            {changes.length === 0
              ? "the project's"
              : `${changes.length} change${changes.length === 1 ? "" : "s"}`}
          </Badge>
        </div>
        {changes.length > 0 && (
          <ul className="flex flex-col gap-1 text-xs" aria-label="Changed fields">
            {changes.slice(0, 8).map((path) => (
              <li key={path} className="flex min-w-0 items-baseline gap-2">
                <span className="min-w-0 break-words text-fg-muted" title={path}>
                  {pathLabel(path, 3)}
                </span>
                <span className="ml-auto flex shrink-0 items-baseline gap-1 font-mono text-[11px]">
                  <span className="text-fg-subtle line-through">
                    {shortValue(getAtPath(fromProject, path))}
                  </span>
                  <span aria-hidden>→</span>
                  <span className="text-fg">{shortValue(getAtPath(working, path))}</span>
                </span>
              </li>
            ))}
            {changes.length > 8 && (
              <li className="text-fg-subtle">and {changes.length - 8} more</li>
            )}
          </ul>
        )}
        {changes.length > 0 && (
          <p className="text-[11px] text-fg-subtle">
            A run records these as its own; the project's file is not touched.
          </p>
        )}
      </section>

      <div className="flex flex-col gap-2">
        <Input
          aria-label="Label for this run"
          placeholder="Label for this run (optional)"
          value={label}
          disabled={readOnly}
          onChange={(e) => setLabel(e.target.value)}
        />
        <Switch
          label="Re-detect images"
          description="Detect every image again instead of reading the detection cache."
          checked={redetect}
          onCheckedChange={setRedetect}
        />
      </div>

      <div className="flex flex-col gap-2">
        <div className="flex flex-col gap-2">
          <Button
            variant="primary"
            disabled={reason !== null}
            loading={busy}
            title={reason ?? "Calibrate with these parameters (⌘↵)"}
            onClick={run}
          >
            <Play aria-hidden className="size-3.5" />
            {busy ? "Running…" : "Run"}
          </Button>
          <Button
            variant="secondary"
            disabled={
              readOnly || saving || changes.length === 0 || check.state.status !== "ok"
            }
            loading={saving}
            title="Write these parameters into the project file as its configuration"
            onClick={() => void saveDefault()}
          >
            <Save aria-hidden className="size-3.5" />
            Save as project default
          </Button>
        </div>
        {reason !== null && !busy && (
          <p
            className={
              tone === "defect" ? "text-xs text-defect" : "text-xs text-fg-muted"
            }
          >
            {reason}
          </p>
        )}
        {editor.cancelled && (
          <p role="status" className="text-xs text-fg-muted">
            Cancelled. No run was written.
          </p>
        )}
      </div>

      {busy && <RunProgress jobId={editor.jobId} />}
      {editor.failure && (
        <FailureView
          failure={editor.failure}
          onDismiss={editor.clearFailure}
          onReset={() => editor.reset()}
        />
      )}
    </aside>
  );
}

function countExclusions(project: ProjectView): number {
  const e = project.spec.exclude;
  return (e?.features?.length ?? 0) + (e?.images?.length ?? 0) + (e?.views?.length ?? 0);
}

function useRunKey(run: () => void): void {
  const ref = useRef(run);
  useEffect(() => {
    ref.current = run;
  });
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && !e.altKey && !e.shiftKey && e.key === "Enter") {
        e.preventDefault();
        ref.current();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}
