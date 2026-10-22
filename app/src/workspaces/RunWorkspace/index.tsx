/** Run workspace.
 *
 * Information architecture:
 *   1. Header strip — title, topology/detector subtitle, "Sniff folder" +
 *      Run buttons.
 *   2. Status banner — progress and cancellation while a run is in flight,
 *      then its outcome (`RunStatusBanner`). Runner `ask_user` ambiguities
 *      surface as a dialog (`AskUserDialog`).
 *   3. Quick-start panel — preset card grid. Collapses to a compact
 *      "preset: name ✓  change" row once a preset is applied.
 *   4. Compact paths strip — folder + manifest paths, once a dataset folder
 *      is known.
 *   5. Unresolved-fields notice — shown when the manifest carries
 *      `_unresolved` (e.g. from a sniffed folder); blocks Run.
 *   6. Manifest, calibration config and the advanced JSON editor, as
 *      collapsible sections.
 *
 * The setup itself (manifest, config, dataset folder) lives in the
 * app-wide `useRunSetup` store, so opening a project fills it from
 * anywhere; this component only lays it out.
 */
import { Badge, Button, Disclosure } from "@vitavision/ui";
import { SchemaValueForm, type JsonSchema } from "@vitavision/forms";
import { useMemo, useState } from "react";

import datasetSchemaJson from "../../schemas/dataset_spec.json";
import { topologyOf, useRunSetup } from "../../store/runSetup";
import { AskUserDialog } from "./AskUserDialog";
import { applyAskUserChoice, clearUnresolved, unresolvedPaths } from "./manifestFields";
import { PRESETS, type EnabledPreset } from "./presets";
import { ActivePresetBar, QuickStartGrid } from "./QuickStart";
import { RunStatusBanner } from "./RunStatusBanner";
import { SetupJsonEditor } from "./SetupJsonEditor";
import { setupActions } from "./setupActions";
import {
  configSummary,
  manifestSummary,
  runBlockReason,
  targetKindOf,
} from "./setupSummary";
import {
  PathsStrip,
  SECTION_CLASS,
  SectionSummary,
  UnresolvedNotice,
} from "./SetupStrips";
import { topologyInfo } from "./topologies";
import { useRunController } from "./useRunController";

// schemars-emitted JSON Schema; cast through unknown since both shapes
// are JSON-compatible (forms' JsonSchema interface is intentionally loose).
const datasetSchema = datasetSchemaJson as unknown as JsonSchema;

export function RunWorkspace() {
  const manifest = useRunSetup((s) => s.manifest);
  const config = useRunSetup((s) => s.config);
  const manifestDir = useRunSetup((s) => s.manifestDir);
  const manifestPath = useRunSetup((s) => s.manifestPath);
  const presetId = useRunSetup((s) => s.presetId);
  const setManifest = useRunSetup((s) => s.setManifest);
  const setConfig = useRunSetup((s) => s.setConfig);
  const updateManifest = useRunSetup((s) => s.updateManifest);

  const { status, setStatus, run, cancel } = useRunController();
  const actions = setupActions(setStatus);

  // Whether the quick-start grid is visible (collapses once a setup loads).
  const [gridExpanded, setGridExpanded] = useState(manifestDir === null);

  const isRunning = status.kind === "running";
  const topology = topologyOf(manifest);
  const info = topologyInfo(topology);

  // Fields the sniffer/runner could not determine. A non-empty list blocks
  // Run until the user fills them in and clears them.
  const unresolved = useMemo<string[]>(() => unresolvedPaths(manifest), [manifest]);
  const hasUnresolved = unresolved.length > 0;
  const blockReason = runBlockReason({
    supported: info.supported,
    unsupportedReason: info.unsupportedReason,
    unresolved: unresolved.length,
    manifestDir,
  });
  const runBlocked = isRunning || !manifestDir || !info.supported || hasUnresolved;

  const collapseOnLoad = (loaded: boolean) => {
    if (loaded) setGridExpanded(false);
  };

  // Apply an AskUser modal choice into the manifest, then dismiss so the
  // user can review and re-run.
  const handleAskUserApply = (choice: string) => {
    if (status.kind !== "ask_user") return;
    const field = status.field;
    updateManifest((prev) => applyAskUserChoice(prev, field, choice));
    setStatus({ kind: "idle" });
  };

  const activePreset = presetId
    ? (PRESETS.find((p) => p.id === presetId) as EnabledPreset | undefined)
    : undefined;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-col gap-0.5">
          <h2 className="text-sm font-semibold tracking-tight text-fg">
            Run calibration
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
            onClick={() => void actions.sniffFolder().then(collapseOnLoad)}
            disabled={isRunning}
            title="Pick a dataset folder and auto-generate a manifest"
          >
            Sniff folder
          </Button>
          <Button
            variant="primary"
            className="px-5"
            loading={isRunning}
            onClick={() => void run()}
            disabled={runBlocked}
            title={blockReason}
          >
            {isRunning ? "Running…" : "Run"}
          </Button>
        </div>
      </header>

      <RunStatusBanner status={status} onCancel={cancel} />

      {gridExpanded ? (
        <QuickStartGrid
          activePresetId={presetId}
          onUse={(preset) => void actions.applyPreset(preset).then(collapseOnLoad)}
          onCollapse={() => setGridExpanded(false)}
        />
      ) : (
        <ActivePresetBar
          preset={activePreset}
          onChangePreset={() => setGridExpanded(true)}
        />
      )}

      {manifestDir && (
        <PathsStrip
          manifestDir={manifestDir}
          manifestPath={manifestPath}
          onPickFolder={() => void actions.pickDatasetFolder()}
          onPickManifest={() => void actions.pickManifest()}
        />
      )}

      {hasUnresolved && (
        <UnresolvedNotice
          paths={unresolved}
          onResolve={(path) => updateManifest((prev) => clearUnresolved(prev, path))}
        />
      )}

      <Disclosure
        className={SECTION_CLASS}
        defaultOpen={hasUnresolved}
        summary={
          <SectionSummary
            title="Manifest"
            summary={manifestSummary(manifest)}
            badge={
              hasUnresolved ? (
                <Badge tone="defect">{unresolved.length} unresolved</Badge>
              ) : manifestDir ? undefined : (
                <Badge tone="info">unset</Badge>
              )
            }
          />
        }
      >
        <SchemaValueForm
          schema={datasetSchema}
          value={manifest}
          onValueChange={setManifest}
          columns={1}
        />
      </Disclosure>

      <Disclosure
        className={SECTION_CLASS}
        summary={
          <SectionSummary title="Calibration config" summary={configSummary(config)} />
        }
      >
        {info.schema ? (
          <SchemaValueForm
            key={topology}
            schema={info.schema}
            value={config}
            onValueChange={setConfig}
            columns={1}
          />
        ) : (
          <p className="text-[12px] text-fg-muted">
            {info.unsupportedReason ?? `No config form for topology "${topology}" yet.`}
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
            if (next.manifest !== undefined) setManifest(next.manifest);
            if (next.config !== undefined) setConfig(next.config);
          }}
        />
      </Disclosure>

      {status.kind === "ask_user" && (
        <AskUserDialog
          field={status.field}
          prompt={status.prompt}
          suggestions={status.suggestions}
          onApply={handleAskUserApply}
          onDismiss={() => setStatus({ kind: "idle" })}
        />
      )}
    </div>
  );
}
