/** Run workspace.
 *
 * Information architecture (approved plan):
 *   1. Header strip — title, topology/detector subtitle, "Sniff folder" +
 *      Run buttons.
 *   2. Quick-start panel — preset card grid. Collapses to a compact
 *      "preset: name ✓  change" row once a preset is applied.
 *   3. Compact paths strip — folder + manifest paths (font-mono, muted).
 *      Visible once a manifest dir is known.
 *   3b. Unresolved-fields notice — shown when the manifest carries
 *      `_unresolved` (e.g. from a sniffed folder); blocks Run (ADR 0019).
 *   4. Manifest section — collapsible, schema-driven SchemaValueForm.
 *   5. Calibration config section — collapsible, schema-driven SchemaValueForm.
 *   6. Advanced JSON editor — third collapsible, lowest priority.
 *   7. Status banner — sticky top-of-workspace during / after a run.
 *      Runner `ask_user` ambiguities surface as a dialog (AskUserDialog).
 *
 * All 8 topologies + 4 detectors run end-to-end, plus "Sniff
 * folder" → heuristic manifest (the `sniff_folder` Tauri command), with the
 * fields the sniffer can't determine left in `_unresolved` and surfaced as
 * red badges + a blocked Run until the user fills and clears them.
 */
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { useEffect, useEffectEvent, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router";
import * as TOML from "toml";

import {
  Badge,
  Button,
  Callout,
  Dialog,
  DialogClose,
  Disclosure,
  Input,
  Textarea,
} from "@vitavision/ui";
import { SchemaValueForm, type JsonSchema } from "@vitavision/forms";
import {
  cancelRun,
  runCalibration,
  type RunResponse,
  type RunStage,
} from "../../lib/runCalibration";
import { dirnamePath, isTauriContext, joinPath, repoRoot } from "../../lib/tauri";
import datasetSchemaJson from "../../schemas/dataset_spec.json";
import planarDefaultConfig from "../../schemas/planar_intrinsics_config.default.json";
import { useStore } from "../../store";
import {
  applyAskUserChoice,
  clearUnresolved,
  hintFor,
  unresolvedPaths,
} from "./manifestFields";
import { PresetCard } from "./PresetCard";
import { BUILTIN_PRESETS, mergeConfig, type EnabledPreset } from "./presets";
import { computeStageRows, formatElapsed, type StageRow } from "./runStages";
import { topologyInfo } from "./topologies";

// schemars-emitted JSON Schema; cast through unknown since both shapes
// are JSON-compatible (forms' JsonSchema interface is intentionally loose).
const datasetSchema = datasetSchemaJson as unknown as JsonSchema;

// ── Default form values ──────────────────────────────────────────────────────

const DEFAULT_DATASET: unknown = {
  version: 1,
  cameras: [
    {
      id: "cam0",
      images: { kind: "glob", pattern: "*.png" },
    },
  ],
  target: {
    kind: "chessboard",
    rows: 9,
    cols: 6,
    square_size_m: 0.025,
  },
  topology: "planar_intrinsics",
};

// Browser-context fallback only — inside Tauri the defaults come from
// `default_config_cmd` (Rust `Config::default()`), the single source of
// truth. This is the same value, emitted by `cargo xtask emit-schemas` (and
// drift-checked in CI), so it cannot fall behind the Rust default.
const DEFAULT_PLANAR_CONFIG: unknown = planarDefaultConfig;

/** Topology of a manifest value, defaulting to planar. */
function topologyOf(manifest: unknown): string {
  const m = manifest as Record<string, unknown> | null;
  return typeof m?.topology === "string" ? m.topology : "planar_intrinsics";
}

/** Fresh correlation id for one run — used to route the progress channel
 * and cancellation. `crypto.randomUUID` is available in every Tauri
 * webview and modern browser; fall back to a timestamp+random string in
 * the rare environment that lacks it (older jsdom). */
/** Wall-clock epoch ms, for display-only timestamps. */
function nowMs(): number {
  return Date.now();
}

function newRunId(): string {
  const c = globalThis.crypto;
  if (c && typeof c.randomUUID === "function") return c.randomUUID();
  return `run-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

/** Default config for a topology: Rust-side defaults inside Tauri,
 * a static fallback in plain-browser dev. */
async function fetchDefaultConfig(topology: string, inTauri: boolean): Promise<unknown> {
  if (inTauri) {
    try {
      return await invoke<unknown>("default_config_cmd", { topology });
    } catch {
      // Fall through to the static fallback (e.g. unknown topology).
    }
  }
  return topology === "planar_intrinsics" ? DEFAULT_PLANAR_CONFIG : {};
}

// ── Run status ───────────────────────────────────────────────────────────────

type RunStatus =
  | { kind: "idle" }
  | {
      kind: "running";
      /** Correlation id for `cancelRun`. */
      runId: string;
      /** Last stage the runner announced; `null` until the first message. */
      stage: RunStage | null;
      /** Epoch ms the run started — UI elapsed clock only, never exported. */
      startedAt: number;
      /** True once the user clicked Cancel (awaiting the boundary stop). */
      cancelRequested: boolean;
    }
  | { kind: "cancelled" }
  | { kind: "ok"; durationMs: number; usable: number; total: number; cacheUsed: boolean }
  | { kind: "error"; category: string; message: string }
  | { kind: "validation"; message: string }
  | { kind: "ask_user"; field: string; prompt: string; suggestions: string[] };

// ── Component ────────────────────────────────────────────────────────────────

export function RunWorkspace() {
  const navigate = useNavigate();
  const inTauri = isTauriContext();
  const acceptLiveRunExport = useStore((s) => s.acceptLiveRunExport);

  // Preset state — null means "no preset selected / custom".
  const [activePresetId, setActivePresetId] = useState<string | null>(null);
  // Whether the quick-start grid is visible (collapses after preset pick).
  const [gridExpanded, setGridExpanded] = useState(true);

  // Paths
  const [manifestDir, setManifestDir] = useState<string | null>(null);
  const [manifestPath, setManifestPath] = useState<string | null>(null);

  // Form state
  const [manifest, setManifest] = useState<unknown>(DEFAULT_DATASET);
  const [config, setConfig] = useState<unknown>(DEFAULT_PLANAR_CONFIG);

  // Run status
  const [status, setStatus] = useState<RunStatus>({ kind: "idle" });

  // Ref to the JSON editor textarea so we can sync it when form state changes
  // from a preset load (the textarea uses defaultValue for perf reasons).
  const jsonEditorRef = useRef<HTMLTextAreaElement>(null);

  const isRunning = status.kind === "running";

  // ── Topology-driven config schema + defaults ─────────────────────────────

  const topology = topologyOf(manifest);
  const info = topologyInfo(topology);

  // Fields the sniffer/runner could not determine. A non-empty list blocks
  // Run until the user fills them in and clears them (ADR 0019).
  const unresolved = useMemo<string[]>(() => unresolvedPaths(manifest), [manifest]);
  const hasUnresolved = unresolved.length > 0;
  const runBlocked = isRunning || !manifestDir || !info.supported || hasUnresolved;
  const runBlockReason = !info.supported
    ? info.unsupportedReason
    : hasUnresolved
      ? `Resolve ${unresolved.length} unresolved manifest field${unresolved.length !== 1 ? "s" : ""} first`
      : !manifestDir
        ? "Set a dataset folder first"
        : undefined;

  // When the manifest's topology changes (form edit, JSON blur, preset
  // load), swap the config schema and reset the config to that
  // topology's defaults.
  const prevTopologyRef = useRef(topology);
  // Reads the latest manifest without making the effect re-fire on every
  // manifest edit: it must only fire on topology *changes*.
  const syncJsonEditor = useEffectEvent((defaults: unknown) => {
    if (jsonEditorRef.current) {
      jsonEditorRef.current.value = JSON.stringify(
        { manifest, config: defaults },
        null,
        2,
      );
    }
  });
  useEffect(() => {
    if (prevTopologyRef.current === topology) return;
    prevTopologyRef.current = topology;
    let cancelled = false;
    void fetchDefaultConfig(topology, inTauri).then((defaults) => {
      if (cancelled) return;
      setConfig(defaults);
      syncJsonEditor(defaults);
    });
    return () => {
      cancelled = true;
    };
  }, [topology, inTauri]);

  // ── Derived summary strings for collapsible headers ──────────────────────

  const manifestSummary = useMemo<string>(() => {
    const m = manifest as Record<string, unknown>;
    const topology = typeof m?.topology === "string" ? m.topology : "?";
    const target = m?.target as Record<string, unknown> | undefined;
    const kind = typeof target?.kind === "string" ? target.kind : "?";
    const cameras = Array.isArray(m?.cameras) ? m.cameras.length : 0;
    return `${topology} · ${kind} · ${cameras} camera${cameras !== 1 ? "s" : ""}`;
  }, [manifest]);

  const configSummary = useMemo<string>(() => {
    const c = config as Record<string, unknown>;
    const iters = typeof c?.max_iters === "number" ? c.max_iters : "?";
    const loss = robustLossLabel(c?.robust_loss);
    return `max_iters=${iters} · loss=${loss}`;
  }, [config]);

  const mergedJson = useMemo<string>(() => {
    return JSON.stringify({ manifest, config }, null, 2);
  }, [manifest, config]);

  // The editor sits in a `Disclosure` (a native <details>), which keeps it
  // mounted while collapsed, so `defaultValue` alone would go stale after a
  // form edit. Re-sync whenever the merged state changes — except while the
  // editor has focus, so an in-progress edit (applied on blur) is not clobbered.
  useEffect(() => {
    const editor = jsonEditorRef.current;
    if (editor && editor.ownerDocument.activeElement !== editor)
      editor.value = mergedJson;
  }, [mergedJson]);

  // ── Preset loader ────────────────────────────────────────────────────────

  const handleUsePreset = async (preset: EnabledPreset) => {
    if (!inTauri) {
      setStatus({
        kind: "error",
        category: "no_tauri",
        message:
          "Preset loading reads from disk and requires the Tauri runtime (bun run tauri dev).",
      });
      return;
    }

    try {
      // Preset manifest paths are repo-root-relative (presets.ts); resolve
      // to an absolute path before touching disk.
      const root = await repoRoot();
      if (!root) {
        setStatus({
          kind: "error",
          category: "no_tauri",
          message: "Could not resolve the workspace repo root (repo_root_cmd failed).",
        });
        return;
      }
      const absManifestPath = joinPath(root, preset.manifestPath);

      // Load the TOML from disk using the load_text_file command.
      const raw = await invoke<string>("load_text_file", { path: absManifestPath });
      // TOML.parse's own types return `any`; narrow to `unknown` immediately
      // since `manifest`/`config` state is `unknown` everywhere else in this
      // component.
      const parsed: unknown = TOML.parse(raw);
      const manifestWithOverrides = preset.manifestOverrides
        ? mergeConfig(parsed, preset.manifestOverrides)
        : parsed;

      // Derive manifestDir from the manifest file path.
      const dir = dirnamePath(absManifestPath);

      setManifestDir(dir);
      setManifestPath(absManifestPath);
      setManifest(manifestWithOverrides);
      // Reset config to the preset topology's defaults (the topology
      // effect above only fires on topology *changes*, and switching
      // between same-topology presets must still reset), then apply
      // the preset's overrides — datasets like rtv3d need non-default
      // config (Scheimpflug sensors, EyeToHand).
      let defaults = await fetchDefaultConfig(topologyOf(manifestWithOverrides), inTauri);
      if (preset.configOverrides) {
        defaults = mergeConfig(defaults, preset.configOverrides);
      }
      prevTopologyRef.current = topologyOf(manifestWithOverrides);
      setConfig(defaults);
      setActivePresetId(preset.id);
      setGridExpanded(false);
      setStatus({ kind: "idle" });

      // Sync the JSON textarea if it happens to be mounted.
      if (jsonEditorRef.current) {
        jsonEditorRef.current.value = JSON.stringify(
          { manifest: manifestWithOverrides, config: defaults },
          null,
          2,
        );
      }
    } catch (e) {
      setStatus({
        kind: "error",
        category: "preset_load",
        message: `Failed to load preset "${preset.name}": ${String(e)}`,
      });
    }
  };

  // ── Manual path pickers ──────────────────────────────────────────────────

  const handlePickFolder = async () => {
    if (!inTauri) {
      setStatus({
        kind: "error",
        category: "no_tauri",
        message: "Folder picker requires Tauri (bun run tauri dev).",
      });
      return;
    }
    try {
      const picked = await open({
        directory: true,
        multiple: false,
        title: "Pick the dataset folder",
      });
      if (typeof picked === "string") {
        setManifestDir(picked);
        // Clear preset selection — user is taking manual control.
        setActivePresetId(null);
      }
    } catch (e) {
      setStatus({ kind: "error", category: "dialog", message: String(e) });
    }
  };

  // Pick a foreign dataset folder and heuristically infer a manifest
  //. The sniffer leaves fields it can't determine in `_unresolved`,
  // which drives the red badge + blocked Run below.
  const handleSniffFolder = async () => {
    if (!inTauri) {
      setStatus({
        kind: "error",
        category: "no_tauri",
        message: "Folder sniffing requires Tauri (bun run tauri dev).",
      });
      return;
    }
    try {
      const picked = await open({
        directory: true,
        multiple: false,
        title: "Pick a dataset folder to sniff",
      });
      if (typeof picked !== "string") return;
      const spec = await invoke<Record<string, unknown>>("sniff_folder", {
        folder: picked,
      });

      setManifestDir(picked);
      setManifestPath(null);
      setManifest(spec);
      const defaults = await fetchDefaultConfig(topologyOf(spec), inTauri);
      prevTopologyRef.current = topologyOf(spec);
      setConfig(defaults);
      setActivePresetId(null);
      setGridExpanded(false);
      setStatus({ kind: "idle" });

      if (jsonEditorRef.current) {
        jsonEditorRef.current.value = JSON.stringify(
          { manifest: spec, config: defaults },
          null,
          2,
        );
      }
    } catch (e) {
      setStatus({
        kind: "error",
        category: "sniff",
        message: `Sniff failed: ${String(e)}`,
      });
    }
  };

  // Mark one `_unresolved` field as handled (the user filled it in the form).
  const handleResolveField = (path: string) => {
    setManifest((prev: unknown) => clearUnresolved(prev, path));
  };

  // Apply an AskUser modal choice into the manifest, then dismiss so the
  // user can review and re-run.
  const handleAskUserApply = (choice: string) => {
    if (status.kind !== "ask_user") return;
    const field = status.field;
    setManifest((prev: unknown) => applyAskUserChoice(prev, field, choice));
    setStatus({ kind: "idle" });
  };

  const handlePickManifest = async () => {
    if (!inTauri) {
      setStatus({
        kind: "error",
        category: "no_tauri",
        message: "File picker requires Tauri (bun run tauri dev).",
      });
      return;
    }
    try {
      const picked = await open({
        multiple: false,
        title: "Pick a dataset.toml or dataset.json manifest",
        filters: [
          { name: "Manifest", extensions: ["toml", "json"] },
          { name: "All files", extensions: ["*"] },
        ],
      });
      if (typeof picked === "string") {
        setManifestPath(picked);
        const dir = dirnamePath(picked);
        setManifestDir(dir);
        setActivePresetId(null);

        // Attempt to load + parse the manifest from disk. On failure we
        // keep the default form state; the user can edit manually.
        try {
          const raw = await invoke<string>("load_text_file", { path: picked });
          const ext = picked.split(".").pop()?.toLowerCase();
          // TOML.parse/JSON.parse return `any`; narrow to `unknown` immediately.
          const parsed: unknown = ext === "toml" ? TOML.parse(raw) : JSON.parse(raw);
          setManifest(parsed);
        } catch {
          // Non-fatal: file might be unreadable or malformed; let the user fix it in the form.
        }
      }
    } catch (e) {
      setStatus({ kind: "error", category: "dialog", message: String(e) });
    }
  };

  // ── Run ──────────────────────────────────────────────────────────────────

  const handleRun = async () => {
    if (!manifestDir) {
      setStatus({
        kind: "error",
        category: "no_dir",
        message: "Set a dataset folder first (pick a preset or use the folder button).",
      });
      return;
    }
    const runId = newRunId();
    setStatus({
      kind: "running",
      runId,
      stage: null,
      startedAt: nowMs(),
      cancelRequested: false,
    });
    try {
      const response = await runCalibration({
        runId,
        manifest,
        config,
        manifestDir,
        onProgress: (stage) =>
          // Only advance the stage if this run is still the active one
          // (a stale channel message must not resurrect a finished run).
          setStatus((prev) =>
            prev.kind === "running" && prev.runId === runId ? { ...prev, stage } : prev,
          ),
      });
      handleResponse(response);
    } catch (e) {
      setStatus({ kind: "error", category: "ipc", message: String(e) });
    }
  };

  const handleCancel = () => {
    if (status.kind !== "running") return;
    const runId = status.runId;
    setStatus({ ...status, cancelRequested: true });
    // Fire-and-forget: the terminal `cancelled` state arrives via the
    // run promise resolving to `RunResponse::Cancelled`.
    void cancelRun(runId).catch(() => {
      /* benign — the run may have finished between click and IPC */
    });
  };

  const handleResponse = (response: RunResponse) => {
    if (response.kind === "ok") {
      if (manifestDir) {
        acceptLiveRunExport(response.export, manifestDir);
      }
      setStatus({
        kind: "ok",
        durationMs: response.duration_ms,
        usable: response.usable_views,
        total: response.total_views,
        cacheUsed: response.cache_used,
      });
      // Hand off to /diagnose after a brief success flash.
      setTimeout(() => navigate("/diagnose"), 600);
    } else if (response.kind === "cancelled") {
      setStatus({ kind: "cancelled" });
    } else if (response.kind === "ask_user") {
      setStatus({
        kind: "ask_user",
        field: response.field,
        prompt: response.prompt,
        suggestions: response.suggestions,
      });
    } else if (response.kind === "validation_failed") {
      setStatus({ kind: "validation", message: response.message });
    } else {
      setStatus({
        kind: "error",
        category: response.category,
        message: response.message,
      });
    }
  };

  // ── JSON editor round-trip ───────────────────────────────────────────────

  const handleJsonBlur = (text: string) => {
    try {
      const parsed = JSON.parse(text) as Record<string, unknown>;
      if (parsed.manifest != null) setManifest(parsed.manifest);
      if (parsed.config != null) setConfig(parsed.config);
    } catch {
      // Keep previous values; v0 doesn't surface parse errors yet.
    }
  };

  // ── Active preset metadata ───────────────────────────────────────────────

  const activePreset = activePresetId
    ? (BUILTIN_PRESETS.find((p) => p.id === activePresetId) as EnabledPreset | undefined)
    : undefined;

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1">
      {/* 1. Header strip */}
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
            onClick={() => void handleSniffFolder()}
            disabled={isRunning}
            title="Pick a dataset folder and auto-generate a manifest"
          >
            Sniff folder
          </Button>

          <Button
            variant="primary"
            className="px-5"
            loading={isRunning}
            onClick={() => void handleRun()}
            disabled={runBlocked}
            title={runBlockReason}
          >
            {isRunning ? "Running…" : "Run"}
          </Button>
        </div>
      </header>

      {/* 7. Status banner — sticky at top of content */}
      {status.kind !== "idle" && <StatusBanner status={status} onCancel={handleCancel} />}

      {/* 2. Quick-start grid / active-preset bar */}
      {gridExpanded ? (
        <QuickStartGrid
          activePresetId={activePresetId}
          onUse={(preset) => void handleUsePreset(preset)}
          onCollapse={() => setGridExpanded(false)}
        />
      ) : (
        <ActivePresetBar
          preset={activePreset}
          onChangePreset={() => setGridExpanded(true)}
        />
      )}

      {/* 3. Compact paths strip — visible once a dir is known */}
      {manifestDir && (
        <PathsStrip
          manifestDir={manifestDir}
          manifestPath={manifestPath}
          onPickFolder={() => void handlePickFolder()}
          onPickManifest={() => void handlePickManifest()}
        />
      )}

      {/* 3b. Unresolved-fields notice — sniffed manifests block Run until
          every domain-knowledge field is filled and cleared (ADR 0019). */}
      {hasUnresolved && (
        <UnresolvedNotice paths={unresolved} onResolve={handleResolveField} />
      )}

      {/* 4. Manifest section */}
      <Disclosure
        className={SECTION_CLASS}
        defaultOpen={hasUnresolved}
        summary={
          <SectionSummary
            title="Manifest"
            summary={manifestSummary}
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

      {/* 5. Calibration config section */}
      <Disclosure
        className={SECTION_CLASS}
        summary={<SectionSummary title="Calibration config" summary={configSummary} />}
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

      {/* 6. Advanced JSON editor */}
      <Disclosure
        className={SECTION_CLASS}
        summary={
          <SectionSummary
            title="Advanced JSON editor"
            summary="merged manifest + config"
          />
        }
      >
        <Textarea
          ref={jsonEditorRef}
          aria-label="Merged manifest and config JSON"
          defaultValue={mergedJson}
          onBlur={(e) => handleJsonBlur(e.target.value)}
          rows={18}
          spellCheck={false}
          className="resize-y px-3 py-2 font-mono text-[11px] leading-relaxed"
        />
        <p className="mt-1.5 text-[11px] text-fg-muted">
          Edits applied on blur. Both <code className="font-mono">manifest</code> and{" "}
          <code className="font-mono">config</code> keys are required at the top level.
        </p>
      </Disclosure>

      {/* AskUser dialog — runner ambiguity that needs a choice (ADR 0019). */}
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

// ── Unresolved-fields notice ───────────────────────────────────────────────────

interface UnresolvedNoticeProps {
  paths: string[];
  onResolve: (path: string) => void;
}

function UnresolvedNotice({ paths, onResolve }: UnresolvedNoticeProps) {
  return (
    <Callout
      tone="error"
      title={`${paths.length} field${paths.length !== 1 ? "s" : ""} need your input before this dataset can run`}
    >
      <p className="text-xs">
        The sniffer left these blank rather than guess. Fill each one in the Manifest form
        below, then mark it resolved.
      </p>
      <ul className="mt-2 flex flex-col gap-1.5">
        {paths.map((path) => (
          <li key={path} className="flex items-start gap-2">
            <code className="mt-0.5 shrink-0 font-mono text-[11px] text-fg">{path}</code>
            <span className="min-w-0 flex-1 text-[11px] text-fg-muted">
              {hintFor(path) ?? "Provide a value in the Manifest form."}
            </span>
            <Button
              size="sm"
              onClick={() => onResolve(path)}
              className="h-6 shrink-0 px-2 text-[11px]"
              title="Remove this field from _unresolved once you've filled it in"
            >
              mark resolved
            </Button>
          </li>
        ))}
      </ul>
    </Callout>
  );
}

function targetKindOf(manifest: unknown): string {
  const m = manifest as Record<string, unknown> | null;
  const target = m?.target as Record<string, unknown> | undefined;
  return typeof target?.kind === "string" ? target.kind : "?";
}

function robustLossLabel(value: unknown): string {
  if (typeof value === "string") return value;
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>;
    if (typeof obj.type === "string") return obj.type;
    const key = Object.keys(obj)[0];
    if (key) return key;
  }
  return "None";
}

// ── Quick-start grid ─────────────────────────────────────────────────────────

interface QuickStartGridProps {
  activePresetId: string | null;
  onUse: (preset: EnabledPreset) => void;
  onCollapse: () => void;
}

function QuickStartGrid({ activePresetId, onUse, onCollapse }: QuickStartGridProps) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-[12px] font-semibold tracking-tight text-fg">Quick start</h3>
        {activePresetId && (
          <Button variant="ghost" size="sm" onClick={onCollapse}>
            collapse
          </Button>
        )}
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {BUILTIN_PRESETS.map((preset) => (
          <PresetCard
            key={preset.id}
            preset={preset}
            isActive={preset.id === activePresetId}
            onUse={(p) => onUse(p)}
          />
        ))}
      </div>
    </section>
  );
}

// ── Active preset bar ────────────────────────────────────────────────────────

interface ActivePresetBarProps {
  preset: EnabledPreset | undefined;
  onChangePreset: () => void;
}

function ActivePresetBar({ preset, onChangePreset }: ActivePresetBarProps) {
  return (
    <div className="flex items-center justify-between rounded-control border border-signal/40 bg-signal/[0.05] px-3 py-2">
      <div className="flex items-center gap-2">
        {/* Brand check mark */}
        <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-signal/20 text-signal">
          <svg width="8" height="8" viewBox="0 0 8 8" fill="none" aria-hidden="true">
            <path
              d="M1 4L3 6L7 2"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
        <span className="text-[12px] font-medium text-fg">
          {preset ? preset.name : "Custom"}
        </span>
        {preset && (
          <span className="font-mono text-[11px] text-fg-muted">
            {preset.targetSummary}
          </span>
        )}
      </div>
      <Button variant="ghost" size="sm" onClick={onChangePreset}>
        change
      </Button>
    </div>
  );
}

// ── Compact paths strip ──────────────────────────────────────────────────────

interface PathsStripProps {
  manifestDir: string | null;
  manifestPath: string | null;
  onPickFolder: () => void;
  onPickManifest: () => void;
}

function PathsStrip({
  manifestDir,
  manifestPath,
  onPickFolder,
  onPickManifest,
}: PathsStripProps) {
  return (
    <div className="flex flex-col gap-1.5 rounded-control border border-line bg-raised px-3 py-2">
      <PathRow
        label="Folder"
        value={manifestDir}
        onEdit={onPickFolder}
        editTitle="Change dataset folder"
      />
      <PathRow
        label="Manifest"
        value={manifestPath}
        onEdit={onPickManifest}
        editTitle="Change manifest file"
      />
    </div>
  );
}

interface PathRowProps {
  label: string;
  value: string | null;
  onEdit: () => void;
  editTitle: string;
}

function PathRow({ label, value, onEdit, editTitle }: PathRowProps) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-14 shrink-0 text-[11px] font-medium text-fg-muted">{label}</span>
      <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-fg">
        {value ?? <span className="text-fg-muted">—</span>}
      </span>
      <Button
        variant="ghost"
        size="sm"
        onClick={onEdit}
        title={editTitle}
        className="h-6 shrink-0 px-2 text-[11px]"
      >
        edit
      </Button>
    </div>
  );
}

// ── Status banner ─────────────────────────────────────────────────────────────

interface StatusBannerProps {
  status: RunStatus;
  onCancel: () => void;
}

function StatusBanner({ status, onCancel }: StatusBannerProps) {
  if (status.kind === "idle") return null;

  if (status.kind === "running") {
    return (
      <RunProgressPanel
        stage={status.stage}
        startedAt={status.startedAt}
        cancelRequested={status.cancelRequested}
        onCancel={onCancel}
      />
    );
  }

  if (status.kind === "cancelled") {
    return (
      <Callout tone="info" title="Run cancelled">
        <span className="font-mono text-xs">
          stopped at the stage boundary — no export was produced
        </span>
      </Callout>
    );
  }

  if (status.kind === "ok") {
    return (
      <Callout tone="success" title="Solve completed">
        <span className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-xs">
            {status.durationMs} ms · {status.usable}/{status.total} usable views
            {status.cacheUsed ? " · cache" : ""}
          </span>
          <span className="ml-auto text-xs">Routing to /diagnose…</span>
        </span>
      </Callout>
    );
  }

  if (status.kind === "validation") {
    return (
      <Callout tone="info" title="Validation failed">
        <code className="font-mono text-xs">{status.message}</code>
      </Callout>
    );
  }

  // `ask_user` is presented as a dialog (see AskUserDialog), not an inline banner.
  if (status.kind === "ask_user") return null;

  // Error
  return (
    <Callout tone="error" title={`Run failed (${status.category})`}>
      <code className="font-mono text-xs text-fg">{status.message}</code>
    </Callout>
  );
}

// ── Elapsed clock ────────────────────────────────────────────────────────────

interface ElapsedClockProps {
  /** Epoch ms the run started — display-only, never enters any export. */
  startedAt: number;
}

/** Live elapsed-time readout for the running banner. Owns its own 200ms
 * interval so only this leaf re-renders 5x/s while a run is in flight,
 * not the whole form-heavy workspace tree above it. */
function ElapsedClock({ startedAt }: ElapsedClockProps) {
  const [elapsedMs, setElapsedMs] = useState(() => Date.now() - startedAt);
  useEffect(() => {
    const id = setInterval(() => setElapsedMs(Date.now() - startedAt), 200);
    return () => clearInterval(id);
  }, [startedAt]);
  return (
    <span className="ml-auto font-mono text-xs font-normal text-fg-muted tabular-nums">
      {formatElapsed(elapsedMs)}
    </span>
  );
}

// ── Run progress panel ─────────────────────────────────────────────────────────

interface RunProgressPanelProps {
  stage: RunStage | null;
  startedAt: number;
  cancelRequested: boolean;
  onCancel: () => void;
}

/** Stage checklist + live elapsed clock + Cancel button for an in-flight
 * run. The three stages (detect → solve → export) come from the runner's
 * progress channel; cancellation takes effect at the next boundary. */
function RunProgressPanel({
  stage,
  startedAt,
  cancelRequested,
  onCancel,
}: RunProgressPanelProps) {
  const rows = computeStageRows(stage, { cancelled: cancelRequested });
  return (
    <Callout
      tone="info"
      title={
        <span className="flex items-center gap-2">
          {cancelRequested ? (
            "Cancelling…"
          ) : (
            <>
              <SpinnerIcon />
              Detection + calibration in progress…
            </>
          )}
          <ElapsedClock startedAt={startedAt} />
        </span>
      }
      actions={
        <Button
          size="sm"
          onClick={onCancel}
          disabled={cancelRequested}
          title="Stop the run at the next stage boundary"
        >
          {cancelRequested ? "Cancelling…" : "Cancel"}
        </Button>
      }
    >
      <ul className="flex flex-col gap-1">
        {rows.map((row) => (
          <StageRowItem key={row.id} row={row} />
        ))}
      </ul>
      {!cancelRequested && (
        <span className="mt-2 block font-mono text-[10px] text-fg-muted">
          first run is slowest; second hits the detection cache
        </span>
      )}
    </Callout>
  );
}

function StageRowItem({ row }: { row: StageRow }) {
  return (
    <li className="flex items-center gap-2 font-mono text-[11px]">
      <StageStateIcon state={row.state} />
      <span
        className={
          row.state === "done"
            ? "text-normal"
            : row.state === "active"
              ? "text-fg"
              : row.state === "cancelled"
                ? "text-fg-muted line-through"
                : "text-fg-muted"
        }
      >
        {row.label}
      </span>
    </li>
  );
}

function StageStateIcon({ state }: { state: StageRow["state"] }) {
  if (state === "active") return <SpinnerIcon />;
  if (state === "done") {
    return (
      <span className="flex h-3.5 w-3.5 shrink-0 items-center justify-center text-normal">
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-label="done">
          <path
            d="M1.5 5L4 7.5L8.5 2.5"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </span>
    );
  }
  if (state === "cancelled") {
    return (
      <span className="flex h-3.5 w-3.5 shrink-0 items-center justify-center text-fg-muted">
        <svg width="9" height="9" viewBox="0 0 9 9" fill="none" aria-label="cancelled">
          <path
            d="M1.5 1.5L7.5 7.5M7.5 1.5L1.5 7.5"
            stroke="currentColor"
            strokeWidth="1.4"
            strokeLinecap="round"
          />
        </svg>
      </span>
    );
  }
  // pending
  return (
    <span
      className="flex h-3.5 w-3.5 shrink-0 items-center justify-center"
      aria-label="pending"
    >
      <span className="h-1.5 w-1.5 rounded-full border border-fg-muted/60" />
    </span>
  );
}

// ── Spinner ──────────────────────────────────────────────────────────────────

function SpinnerIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 14 14"
      fill="none"
      aria-label="Running"
      className="shrink-0 animate-spin"
    >
      <circle
        cx="7"
        cy="7"
        r="5.5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeOpacity="0.25"
      />
      <path
        d="M7 1.5A5.5 5.5 0 0 1 12.5 7"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

// ── Collapsible section summary ─────────────────────────────────────────────

/** The bordered box every collapsible form section (`Disclosure`) sits in. */
const SECTION_CLASS = "rounded-panel border border-line bg-surface px-3 py-2";

interface SectionSummaryProps {
  title: string;
  /** Short descriptor, shown only while the section is collapsed. */
  summary: string;
  badge?: React.ReactNode;
}

/** The `summary` line of a form section: the title, an optional badge, and
 * the collapsed-state descriptor (hidden once the `<details>` is open). */
function SectionSummary({ title, summary, badge }: SectionSummaryProps) {
  return (
    <>
      <span className="text-xs font-semibold tracking-tight text-fg">{title}</span>
      {badge}
      <span className="ml-auto min-w-0 truncate pl-3 font-mono text-[11px] font-normal text-fg-muted group-open:hidden">
        {summary}
      </span>
    </>
  );
}

// ── AskUser dialog ───────────────────────────────────────────────────────────

interface AskUserDialogProps {
  field: string;
  prompt: string;
  suggestions: string[];
  /** Apply a chosen value for `field` to the manifest. */
  onApply: (choice: string) => void;
  /** Close without applying. */
  onDismiss: () => void;
}

/** Dialog for a runner `ask_user` event (ADR 0019 fail-fast).
 *
 * The dataset runner raises an `ask_user` event when it hits an ambiguity it
 * refuses to guess (e.g. how images pair into views). This surfaces the
 * field + prompt, renders each suggestion as a click-to-apply button, and
 * offers a free-text input for open-ended fields. Applying writes the choice
 * into the manifest (via `applyAskUserChoice`) and dismisses — the user
 * reviews the form and re-runs, staying in control. */
function AskUserDialog({
  field,
  prompt,
  suggestions,
  onApply,
  onDismiss,
}: AskUserDialogProps) {
  const [freeText, setFreeText] = useState("");
  const hint = hintFor(field);
  const trimmed = freeText.trim();

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onDismiss();
      }}
      title="Input needed"
      description={prompt}
      footer={
        <DialogClose asChild>
          <Button variant="ghost">Dismiss</Button>
        </DialogClose>
      }
    >
      <div className="mt-3 flex flex-col gap-3">
        <code className="font-mono text-[11px] text-fg-muted">{field}</code>
        {hint && hint !== prompt && (
          <p className="text-xs leading-relaxed text-fg-muted">{hint}</p>
        )}

        {suggestions.length > 0 && (
          <div className="flex flex-col gap-1.5">
            <span className="text-xs font-medium text-fg-muted">Choose one:</span>
            <div className="flex flex-wrap gap-1.5">
              {suggestions.map((s) => (
                <Button
                  key={s}
                  size="sm"
                  className="font-mono"
                  onClick={() => onApply(s)}
                >
                  {s}
                </Button>
              ))}
            </div>
          </div>
        )}

        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-fg-muted">Or enter a value:</span>
          <div className="flex gap-1.5">
            <Input
              type="text"
              aria-label={`Value for ${field}`}
              value={freeText}
              onChange={(e) => setFreeText(e.target.value)}
              placeholder={field}
              className="min-w-0 flex-1 font-mono text-xs"
            />
            <Button
              variant="primary"
              disabled={trimmed === ""}
              onClick={() => onApply(trimmed)}
            >
              Apply
            </Button>
          </div>
        </div>
      </div>
    </Dialog>
  );
}
