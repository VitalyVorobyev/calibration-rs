import { SchemaValueForm } from "@vitavision/forms";
import { Badge, Button, Callout, Disclosure, Select, Skeleton } from "@vitavision/ui";
import { RotateCcw } from "lucide-react";
import { useMemo } from "react";
import { api } from "../../api";
import type { ProjectView } from "../../api/generated/types";
import { changedPaths } from "../../project/configEdit";
import {
  baselineValue,
  projectTopology,
  useRunEditor,
  workingConfig,
  type Baseline,
} from "../../project/runEditor";
import { useReadOnly } from "../../session/jobs";
import { useQuery } from "../../session/query";
import { exportKindLabel } from "../../store/exportKind";
import { curateConfig, seedLossScales } from "./curate";
import { cameraRenderer } from "./cameraRenderer";
import { PatchView } from "./PatchView";
import { RerunNotice } from "./RerunNotice";
import type { EditorCheck } from "./useEditorCheck";

/** The parameter editor: the configuration the next run takes, curated
 * groups first and every other field behind them, each field marked when
 * it differs from the baseline, with the library's own words when a value
 * does not check. */
export function ParameterEditor({
  project,
  check,
}: {
  project: ProjectView;
  check: EditorCheck;
}) {
  const editor = useRunEditor();
  const readOnly = useReadOnly();
  const { form } = editor;
  const working = workingConfig(editor, project);
  const base = baselineValue(editor.baseline, project, form);
  const topology = projectTopology(project);

  // The layout reads the schema and the defaults, never the edits, so it
  // is built once per workflow.
  const ui = useMemo(
    () => (form ? curateConfig(form.schema, form.default) : null),
    [form],
  );
  const lossPaths = useMemo(
    () => ui?.groups?.find((g) => g.id === "loss")?.fields ?? [],
    [ui],
  );
  const views = useQuery("project_views", { dir: project.dir }, () =>
    api.projectViews({ dir: project.dir }),
  );
  const cameras = views.data?.cameras;
  const renderField = useMemo(
    () => cameraRenderer(working, base, cameras ?? []),
    [working, base, cameras],
  );
  const changes = useMemo(() => changedPaths(working, base), [working, base]);
  const runs = project.runs.filter(
    (r) => r.state.kind === "recorded" && r.state.source.kind === "calibrated",
  );

  if (editor.formError) {
    return (
      <Callout tone="error" title="The configuration form did not load">
        {editor.formError}
      </Callout>
    );
  }
  if (!form || !ui || topology === null) {
    return (
      <div role="status" aria-busy className="flex flex-col gap-3">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-40 w-full" />
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }

  return (
    <section aria-label="Parameters" className="flex flex-col gap-4">
      <header className="flex flex-wrap items-end justify-between gap-x-4 gap-y-2">
        <div className="flex min-w-0 flex-col gap-0.5">
          <h2 className="text-sm font-semibold tracking-tight text-fg">Parameters</h2>
          <p className="text-xs text-fg-muted">
            {exportKindLabel(topology)} · edits apply to the next run only
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-2 text-xs text-fg-muted">
            Compare with
            <Select
              aria-label="Baseline"
              className="h-7 w-44 text-xs"
              value={baselineKey(editor.baseline)}
              disabled={readOnly}
              onValueChange={(value) => {
                if (value === "default" || value === "project") {
                  void editor.setBaseline(value);
                } else if (value.startsWith("run:")) {
                  void editor.setBaseline("run", value.slice(4));
                }
              }}
              options={[
                { value: "default", label: "Library default" },
                { value: "project", label: "Project configuration" },
                ...runs.map((r) => ({
                  value: `run:${r.id}`,
                  label: r.id,
                  note: r.notes.label ?? "",
                })),
              ]}
            />
          </label>
          <Badge tone={changes.length > 0 ? "info" : "neutral"}>
            {changes.length === 0
              ? "No changes"
              : `${changes.length} change${changes.length === 1 ? "" : "s"}`}
          </Badge>
          <Button
            size="sm"
            variant="ghost"
            disabled={readOnly || (editor.edits === null && editor.fromRun === null)}
            title="Put the editor back to the project's configuration"
            onClick={() => editor.reset()}
          >
            <RotateCcw aria-hidden className="size-3.5" />
            Reset all
          </Button>
        </div>
      </header>

      <RerunNotice project={project} />
      {project.config.kind === "invalid" && (
        <Callout tone="warning" title="The project's configuration does not resolve">
          <span className="text-xs">
            {project.config.error.message} The editor starts from the library defaults.
          </span>
        </Callout>
      )}

      <SchemaValueForm
        key={form.topology}
        schema={form.schema}
        value={working}
        baseline={base}
        ui={ui}
        messages={check.messages}
        renderField={renderField}
        columns={1}
        disabled={readOnly}
        onValueChange={(next) => editor.edit(seedLossScales(working, next, lossPaths))}
      />

      <Disclosure summary="Resulting patch">
        <PatchView patch={editor.editorPatch} />
      </Disclosure>
    </section>
  );
}

function baselineKey(baseline: Baseline): string {
  return baseline.kind === "run" ? `run:${baseline.id}` : baseline.kind;
}
