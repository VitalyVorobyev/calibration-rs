/** Run workspace: calibrate the open project.
 *
 * A calibration runs on a project (`calibrate`, the library's
 * `Project::calibrate`): without one, the workspace offers to open one, to
 * make one of a dataset folder or a manifest, or a Quick Start sample.
 * With one:
 *   1. Header — the project, its workflow and target; Save and Run.
 *   2. Status banner — progress and cancellation while the job runs, then
 *      its outcome (`RunStatusBanner`); a choice the manifest leaves open is
 *      a dialog (`AskUserDialog`).
 *   3. Unresolved-fields notice — the manifest's `_unresolved`; blocks Run.
 *   4. Manifest, calibration config and the advanced JSON editor, as
 *      collapsible sections. They edit the project's unsaved draft
 *      (`useDraft`); a manifest file of its own is shown, not edited.
 */
import { Badge, Button, Callout, Disclosure } from "@vitavision/ui";
import { SchemaValueForm } from "@vitavision/forms";
import { useMemo, useState } from "react";

import type { DatasetSpec, SampleInfo } from "../../api/generated/types";
import { isDirty, topologyOf, useDraft } from "../../project/draftStore";
import {
  newProjectFromFolder,
  newProjectFromManifest,
  openProject,
  openSample,
} from "../../project/projectActions";
import { useProject } from "../../project/projectStore";
import { useReadOnly } from "../../session/jobs";
import { useSession } from "../../session/session";
import { AskUserDialog } from "./AskUserDialog";
import { applyAskUserChoice, clearUnresolved, unresolvedPaths } from "./manifestFields";
import { QuickStartGrid } from "./QuickStart";
import { RunStatusBanner } from "./RunStatusBanner";
import { SetupJsonEditor } from "./SetupJsonEditor";
import {
  configSummary,
  manifestSummary,
  runBlockReason,
  targetKindOf,
} from "./setupSummary";
import { SECTION_CLASS, SectionSummary, UnresolvedNotice } from "./SetupStrips";
import { topologyInfo } from "./topologies";
import { useForms } from "./useForms";
import { useRunController } from "./useRunController";

export function RunWorkspace() {
  const project = useProject((s) => s.project);
  const { status, setStatus, run, cancel } = useRunController();
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1">
      {project ? (
        <ProjectSetup status={status} onRun={run} onCancel={cancel} />
      ) : (
        <NoProject />
      )}
      {project === null && <RunStatusBanner status={status} onCancel={cancel} />}
      {status.kind === "ask_user" && (
        <AskUserDialog
          field={status.field}
          prompt={status.prompt}
          suggestions={status.suggestions}
          onApply={(choice) => {
            const { manifest, setManifest } = useDraft.getState();
            if (manifest !== null) {
              setManifest(
                applyAskUserChoice(manifest, status.field, choice) as DatasetSpec,
              );
            }
            setStatus({ kind: "idle" });
          }}
          onDismiss={() => setStatus({ kind: "idle" })}
        />
      )}
    </div>
  );
}

/** A stable empty list for the selector (a fresh `[]` would re-render forever). */
const NO_SAMPLES: SampleInfo[] = [];

/** No project open: the ways to one. */
function NoProject() {
  const samples = useSession((s) =>
    s.status.kind === "ready" ? s.status.info.samples : NO_SAMPLES,
  );
  const [busy, setBusy] = useState<string | null>(null);
  return (
    <>
      <header className="flex flex-col gap-0.5">
        <h2 className="text-sm font-semibold tracking-tight text-fg">Run calibration</h2>
        <p className="text-[12px] text-fg-muted">
          A calibration runs on a project: a dataset, its settings and its runs, in one
          folder. Open one, or make one of a dataset folder or a manifest.
        </p>
      </header>
      <div className="flex flex-wrap gap-2">
        <Button variant="primary" onClick={() => void openProject()}>
          Open project…
        </Button>
        <Button onClick={() => void newProjectFromFolder()}>
          New project from a dataset folder…
        </Button>
        <Button onClick={() => void newProjectFromManifest()}>
          New project from a manifest…
        </Button>
      </div>
      <QuickStartGrid
        samples={samples}
        busy={busy}
        onUse={(id) => {
          setBusy(id);
          void openSample(id).finally(() => setBusy(null));
        }}
      />
    </>
  );
}

interface ProjectSetupProps {
  status: ReturnType<typeof useRunController>["status"];
  onRun: () => Promise<void>;
  onCancel: () => void;
}

/** The open project's setup: its manifest and configuration, Save and Run. */
function ProjectSetup({ status, onRun, onCancel }: ProjectSetupProps) {
  const project = useProject((s) => s.project);
  const manifest = useDraft((s) => s.manifest);
  const config = useDraft((s) => s.config);
  const editable = useDraft((s) => s.editable);
  const dirty = useDraft(isDirty);
  const setManifest = useDraft((s) => s.setManifest);
  const setConfig = useDraft((s) => s.setConfig);
  const [saving, setSaving] = useState(false);
  const readOnly = useReadOnly();

  const isRunning = status.kind === "running";
  const topology = topologyOf(manifest) ?? "";
  const info = topologyInfo(topology);
  const forms = useForms(topology);
  const unresolved = useMemo<string[]>(() => unresolvedPaths(manifest), [manifest]);
  const blockReason = runBlockReason({
    supported: info.supported,
    unsupportedReason: info.unsupportedReason,
    unresolved: unresolved.length,
    hasProject: project !== null,
    configLoaded: config !== null,
  });
  if (!project) return null;
  const datasetError =
    project.dataset.kind === "unreadable" ? project.dataset.error : null;

  const save = async () => {
    setSaving(true);
    await useDraft.getState().save();
    setSaving(false);
  };

  return (
    <>
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-col gap-0.5">
          <h2 className="text-sm font-semibold tracking-tight text-fg">
            Run calibration · {project.name}
          </h2>
          <p className="font-mono text-[11px] text-fg-muted">
            {info.label} + {targetKindOf(manifest)}, end-to-end
            {!info.supported && info.unsupportedReason
              ? ` — ${info.unsupportedReason}`
              : ""}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            onClick={() => void save()}
            disabled={!dirty || isRunning || readOnly}
            loading={saving}
            title="Write the edits to the project file"
          >
            Save
          </Button>
          <Button
            variant="primary"
            className="px-5"
            loading={isRunning}
            onClick={() => void onRun()}
            disabled={isRunning || readOnly || blockReason !== undefined}
            title={
              blockReason ?? (dirty ? "Save the edits and run the project" : undefined)
            }
          >
            {isRunning ? "Running…" : "Run"}
          </Button>
        </div>
      </header>

      <RunStatusBanner status={status} onCancel={onCancel} />

      {datasetError && (
        <Callout tone="error" title="The dataset cannot be read">
          <code className="font-mono text-xs">{datasetError.message}</code>
        </Callout>
      )}
      {!editable && manifest !== null && (
        <Callout tone="info" title="The manifest is a file of its own">
          <span className="text-xs">
            The project references{" "}
            <code className="font-mono">
              {typeof project.spec.dataset.manifest === "string"
                ? project.spec.dataset.manifest
                : "a manifest file"}
            </code>
            , which the app shows but never writes: edit it there, then reload the
            project.
          </span>
        </Callout>
      )}
      {unresolved.length > 0 && (
        <UnresolvedNotice
          paths={unresolved}
          onResolve={(path) => {
            if (manifest !== null)
              setManifest(clearUnresolved(manifest, path) as DatasetSpec);
          }}
        />
      )}

      <Disclosure
        className={SECTION_CLASS}
        defaultOpen={unresolved.length > 0}
        summary={
          <SectionSummary
            title="Manifest"
            summary={manifestSummary(manifest)}
            badge={
              unresolved.length > 0 ? (
                <Badge tone="defect">{unresolved.length} unresolved</Badge>
              ) : editable ? undefined : (
                <Badge tone="info">read-only</Badge>
              )
            }
          />
        }
      >
        {forms.dataset && manifest !== null ? (
          <SchemaValueForm
            schema={forms.dataset}
            value={manifest}
            onValueChange={(next) => setManifest(next as DatasetSpec)}
            disabled={!editable}
            columns={1}
          />
        ) : (
          <p className="text-[12px] text-fg-muted">Loading the manifest form…</p>
        )}
      </Disclosure>

      <Disclosure
        className={SECTION_CLASS}
        summary={
          <SectionSummary title="Calibration config" summary={configSummary(config)} />
        }
      >
        {project.config.kind === "invalid" && (
          <p className="mb-2 text-[12px] text-fg-muted">
            The project's configuration does not resolve ({project.config.error.message});
            the form starts from the workflow's defaults.
          </p>
        )}
        {forms.config && config !== null ? (
          <SchemaValueForm
            key={topology}
            schema={forms.config}
            value={config}
            onValueChange={setConfig}
            columns={1}
          />
        ) : (
          <p className="text-[12px] text-fg-muted">
            {info.unsupportedReason ??
              forms.configError ??
              `Loading the config form for "${topology}"…`}
          </p>
        )}
      </Disclosure>

      <Disclosure
        className={SECTION_CLASS}
        summary={
          <SectionSummary
            title="Advanced JSON editor"
            summary="merged manifest + config"
          />
        }
      >
        <SetupJsonEditor
          manifest={manifest}
          config={config}
          onApply={(next) => {
            if (next.manifest !== undefined && editable) {
              setManifest(next.manifest as DatasetSpec);
            }
            if (next.config !== undefined) setConfig(next.config);
          }}
        />
      </Disclosure>
    </>
  );
}
