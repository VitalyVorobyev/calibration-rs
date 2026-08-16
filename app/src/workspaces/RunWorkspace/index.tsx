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
 *   4. Manifest section — collapsible, schema-driven ConfigForm.
 *   5. Calibration config section — collapsible, schema-driven ConfigForm.
 *   6. Advanced JSON editor — third collapsible, lowest priority.
 *   7. Status banner — sticky top-of-workspace during / after a run.
 *      Runner `ask_user` ambiguities surface as a modal (AskUserModal).
 *
 * All 8 topologies + 4 detectors run end-to-end, plus "Sniff
 * folder" → heuristic manifest (the `sniff_folder` Tauri command), with the
 * fields the sniffer can't determine left in `_unresolved` and surfaced as
 * red badges + a blocked Run until the user fills and clears them.
 */
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import * as TOML from "toml";

import { Banner, Button } from "../../components/ui";
import { ConfigForm, type JsonSchema } from "../../lib/configForm";
import {
  cancelRun,
  runCalibration,
  type RunResponse,
  type RunStage,
} from "../../lib/runCalibration";
import { dirnamePath, isTauriContext, joinPath, repoRoot } from "../../lib/tauri";
import datasetSchemaJson from "../../schemas/dataset_spec.json";
import { useStore } from "../../store";
import { AskUserModal } from "./AskUserModal";
import { CollapsibleSection } from "./CollapsibleSection";
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
// are JSON-compatible (our JsonSchema interface is intentionally loose).
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
// `default_config_cmd` (Rust `Config::default()`), the single source
// of truth. Grouped shape per ADR 0024 — mirrors `PlanarIntrinsicsConfig::default()`.
const DEFAULT_PLANAR_CONFIG: unknown = {
  init: {
    init_iterations: 2,
    fix_k3: true,
    fix_tangential: false,
    zero_skew: true,
  },
  solver: {
    max_iters: 50,
    verbosity: 0,
    robust_loss: "None",
  },
  distortion_model: "brown_conrady5",
  fix_camera: {
    intrinsics: { fx: false, fy: false, cx: false, cy: false },
    distortion: { k1: false, k2: false, k3: true, p1: false, p2: false },
  },
  fix_poses: [],
};

/** Topology of a manifest value, defaulting to planar. */
function topologyOf(manifest: unknown): string {
  const m = manifest as Record<string, unknown> | null;
  return typeof m?.topology === "string" ? m.topology : "planar_intrinsics";
}

/** Fresh correlation id for one run — used to route the progress channel
 * and cancellation. `crypto.randomUUID` is available in every Tauri
 * webview and modern browser; fall back to a timestamp+random string in
 * the rare environment that lacks it (older jsdom). */
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
  useEffect(() => {
    if (prevTopologyRef.current === topology) return;
    prevTopologyRef.current = topology;
    let cancelled = false;
    void fetchDefaultConfig(topology, inTauri).then((defaults) => {
      if (cancelled) return;
      setConfig(defaults);
      if (jsonEditorRef.current) {
        jsonEditorRef.current.value = JSON.stringify(
          { manifest, config: defaults },
          null,
          2,
        );
      }
    });
    return () => {
      cancelled = true;
    };
    // `manifest` is intentionally read, not depended on: this effect
    // only fires on topology *changes*.
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
      startedAt: Date.now(),
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
          <h2 className="text-sm font-semibold tracking-tight">Run calibration</h2>
          <p className="font-mono text-[11px] text-muted-foreground">
            {info.label} + {targetKindOf(manifest)}, end-to-end
            {!info.supported && info.unsupportedReason
              ? ` — ${info.unsupportedReason}`
              : ""}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            size="md"
            onClick={() => void handleSniffFolder()}
            disabled={isRunning}
            title="Pick a dataset folder and auto-generate a manifest"
          >
            Sniff folder
          </Button>

          <Button
            variant="primary"
            size="md"
            className="!px-5"
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
      <CollapsibleSection
        title="Manifest"
        summary={manifestSummary}
        defaultOpen={hasUnresolved}
        badge={
          hasUnresolved
            ? `${unresolved.length} unresolved`
            : manifestDir
              ? undefined
              : "unset"
        }
        badgeVariant={hasUnresolved ? "destructive" : "default"}
      >
        <ConfigForm
          schema={datasetSchema}
          value={manifest}
          onChange={setManifest}
          rootLabel="dataset"
        />
      </CollapsibleSection>

      {/* 5. Calibration config section */}
      <CollapsibleSection title="Calibration config" summary={configSummary}>
        {info.schema ? (
          <ConfigForm
            schema={info.schema}
            value={config}
            onChange={setConfig}
            rootLabel="config"
          />
        ) : (
          <p className="text-[12px] text-muted-foreground">
            {info.unsupportedReason ?? `No config form for topology "${topology}" yet.`}
          </p>
        )}
      </CollapsibleSection>

      {/* 6. Advanced JSON editor */}
      <CollapsibleSection title="Advanced JSON editor" summary="merged manifest + config">
        <textarea
          ref={jsonEditorRef}
          defaultValue={mergedJson}
          onBlur={(e) => handleJsonBlur(e.target.value)}
          rows={18}
          spellCheck={false}
          className="w-full rounded-md border border-border bg-bg-soft px-3 py-2 font-mono text-[11px] leading-relaxed text-foreground outline-none focus:border-brand/60 resize-y"
        />
        <p className="mt-1.5 text-[11px] text-muted-foreground">
          Edits applied on blur. Both <code className="font-mono">manifest</code> and{" "}
          <code className="font-mono">config</code> keys are required at the top level.
        </p>
      </CollapsibleSection>

      {/* AskUser modal — runner ambiguity that needs a choice (ADR 0019). */}
      {status.kind === "ask_user" && (
        <AskUserModal
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
    <Banner variant="error" className="flex flex-col gap-2 !p-3 text-[12px]">
      <p className="font-semibold text-destructive">
        {paths.length} field{paths.length !== 1 ? "s" : ""} need your input before this
        dataset can run
      </p>
      <p className="text-[11px] text-muted-foreground">
        The sniffer left these blank rather than guess. Fill each one in the Manifest form
        below, then mark it resolved.
      </p>
      <ul className="flex flex-col gap-1.5">
        {paths.map((path) => (
          <li key={path} className="flex items-start gap-2">
            <code className="mt-0.5 shrink-0 font-mono text-[11px] text-foreground">
              {path}
            </code>
            <span className="min-w-0 flex-1 text-[11px] text-muted-foreground">
              {hintFor(path) ?? "Provide a value in the Manifest form."}
            </span>
            <button
              type="button"
              onClick={() => onResolve(path)}
              className="shrink-0 rounded border border-border px-2 py-0.5 text-[10px] text-muted-foreground transition-colors hover:text-foreground"
              title="Remove this field from _unresolved once you've filled it in"
            >
              mark resolved
            </button>
          </li>
        ))}
      </ul>
    </Banner>
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
        <h3 className="text-[12px] font-semibold tracking-tight text-foreground">
          Quick start
        </h3>
        {activePresetId && (
          <button
            type="button"
            onClick={onCollapse}
            className="text-[11px] text-muted-foreground hover:text-foreground transition-colors"
          >
            collapse
          </button>
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
    <div className="flex items-center justify-between rounded-md border border-brand/40 bg-brand/[0.05] px-3 py-2">
      <div className="flex items-center gap-2">
        {/* Brand check mark */}
        <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-brand/20 text-brand">
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
        <span className="text-[12px] font-medium text-foreground">
          {preset ? preset.name : "Custom"}
        </span>
        {preset && (
          <span className="font-mono text-[11px] text-muted-foreground">
            {preset.targetSummary}
          </span>
        )}
      </div>
      <button
        type="button"
        onClick={onChangePreset}
        className="text-[11px] text-muted-foreground hover:text-foreground transition-colors"
      >
        change
      </button>
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
    <div className="flex flex-col gap-1.5 rounded-md border border-border bg-bg-soft px-3 py-2">
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
      <span className="w-14 shrink-0 text-[11px] font-medium text-muted-foreground">
        {label}
      </span>
      <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-foreground">
        {value ?? <span className="text-muted-foreground">—</span>}
      </span>
      <button
        type="button"
        onClick={onEdit}
        title={editTitle}
        className="shrink-0 rounded border border-border px-2 py-0.5 text-[10px] text-muted-foreground hover:text-foreground transition-colors"
      >
        edit
      </button>
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
      <Banner variant="neutral" className="flex items-center gap-2 !p-3 text-[12px]">
        <span className="font-medium text-foreground">Run cancelled</span>
        <span className="font-mono text-muted-foreground">
          stopped at the stage boundary — no export was produced
        </span>
      </Banner>
    );
  }

  if (status.kind === "ok") {
    return (
      <Banner variant="success" className="flex items-center gap-2 !p-3 text-[12px]">
        <span className="text-success">Solve completed</span>
        <span className="font-mono text-muted-foreground">
          {status.durationMs} ms · {status.usable}/{status.total} usable views
          {status.cacheUsed ? " · cache" : ""}
        </span>
        <span className="ml-auto text-muted-foreground">Routing to /diagnose…</span>
      </Banner>
    );
  }

  if (status.kind === "validation") {
    return (
      <Banner variant="neutral" className="!p-3 text-[12px]">
        <span className="font-medium text-foreground">Validation failed: </span>
        <code className="font-mono text-muted-foreground">{status.message}</code>
      </Banner>
    );
  }

  // `ask_user` is presented as a modal (see AskUserModal), not an inline banner.
  if (status.kind === "ask_user") return null;

  // Error
  return (
    <Banner variant="error" className="!p-3 text-[12px]">
      <span className="font-semibold text-destructive">
        Run failed ({status.category}):{" "}
      </span>
      <code className="font-mono text-foreground">{status.message}</code>
    </Banner>
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
    const tick = () => setElapsedMs(Date.now() - startedAt);
    tick();
    const id = setInterval(tick, 200);
    return () => clearInterval(id);
  }, [startedAt]);
  return (
    <span className="ml-auto font-mono text-muted-foreground tabular-nums">
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
    <Banner variant="neutral" className="flex flex-col gap-2 !p-3 text-[12px]">
      <div className="flex items-center gap-2">
        {cancelRequested ? (
          <span className="text-foreground">Cancelling…</span>
        ) : (
          <>
            <SpinnerIcon />
            <span className="text-foreground">Detection + calibration in progress…</span>
          </>
        )}
        <ElapsedClock startedAt={startedAt} />
        <Button
          size="sm"
          className="!px-2.5 font-medium"
          onClick={onCancel}
          disabled={cancelRequested}
          title="Stop the run at the next stage boundary"
        >
          {cancelRequested ? "Cancelling…" : "Cancel"}
        </Button>
      </div>
      <ul className="flex flex-col gap-1">
        {rows.map((row) => (
          <StageRowItem key={row.id} row={row} />
        ))}
      </ul>
      {!cancelRequested && (
        <span className="font-mono text-[10px] text-muted-foreground">
          first run is slowest; second hits the detection cache
        </span>
      )}
    </Banner>
  );
}

function StageRowItem({ row }: { row: StageRow }) {
  return (
    <li className="flex items-center gap-2 font-mono text-[11px]">
      <StageStateIcon state={row.state} />
      <span
        className={
          row.state === "done"
            ? "text-success"
            : row.state === "active"
              ? "text-foreground"
              : row.state === "cancelled"
                ? "text-muted-foreground line-through"
                : "text-muted-foreground"
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
      <span className="flex h-3.5 w-3.5 shrink-0 items-center justify-center text-success">
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
      <span className="flex h-3.5 w-3.5 shrink-0 items-center justify-center text-muted-foreground">
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
      <span className="h-1.5 w-1.5 rounded-full border border-muted-foreground/60" />
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
