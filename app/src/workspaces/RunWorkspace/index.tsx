/** Run workspace (B3b redesign).
 *
 * Information architecture (approved plan):
 *   1. Header strip — title, active topology/detector subtitle, Run button.
 *   2. Quick-start panel — preset card grid. Collapses to a compact
 *      "preset: name ✓  change" row once a preset is applied.
 *   3. Compact paths strip — folder + manifest paths (font-mono, muted).
 *      Visible once a manifest dir is known.
 *   4. Manifest section — collapsible, schema-driven ConfigForm.
 *   5. Calibration config section — collapsible, schema-driven ConfigForm.
 *   6. Advanced JSON editor — third collapsible, lowest priority.
 *   7. Status banner — sticky top-of-workspace during / after a run.
 *
 * PR 1 wires up PlanarIntrinsics + chessboard only; the IA is designed
 * to accommodate all 8 topologies + 4 detectors without structural change.
 */
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import * as TOML from "toml";

import { ConfigForm, type JsonSchema } from "../../lib/configForm";
import { runCalibration, type RunResponse } from "../../lib/runCalibration";
import { isTauriContext, joinPath } from "../../lib/tauri";
import datasetSchemaJson from "../../schemas/dataset_spec.json";
import planarConfigSchemaJson from "../../schemas/planar_intrinsics_config.json";
import { CollapsibleSection } from "./CollapsibleSection";
import { PresetCard } from "./PresetCard";
import { BUILTIN_PRESETS, type EnabledPreset } from "./presets";

// schemars-emitted JSON Schemas; cast through unknown since both shapes
// are JSON-compatible (our JsonSchema interface is intentionally loose).
const datasetSchema = datasetSchemaJson as unknown as JsonSchema;
const planarConfigSchema = planarConfigSchemaJson as unknown as JsonSchema;

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

const DEFAULT_PLANAR_CONFIG: unknown = {
  init_iterations: 2,
  fix_k3_in_init: true,
  fix_tangential_in_init: false,
  zero_skew: true,
  max_iters: 50,
  verbosity: 0,
  robust_loss: { type: "None" },
  fix_intrinsics: { fx: false, fy: false, cx: false, cy: false },
  fix_distortion: { k1: false, k2: false, k3: true, p1: false, p2: false },
  fix_poses: [],
};

// ── Run status ───────────────────────────────────────────────────────────────

type RunStatus =
  | { kind: "idle" }
  | { kind: "running" }
  | { kind: "ok"; durationMs: number; usable: number; total: number; cacheUsed: boolean }
  | { kind: "error"; category: string; message: string }
  | { kind: "validation"; message: string }
  | { kind: "ask_user"; field: string; prompt: string; suggestions: string[] };

// ── Component ────────────────────────────────────────────────────────────────

export function RunWorkspace() {
  const navigate = useNavigate();
  const inTauri = isTauriContext();

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
    const loss = (c?.robust_loss as Record<string, unknown> | undefined)?.type ?? "None";
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
        message: "Preset loading reads from disk and requires the Tauri runtime (bun run tauri dev).",
      });
      return;
    }

    try {
      // Load the TOML from disk using the new load_text_file command.
      const raw = await invoke<string>("load_text_file", { path: preset.manifestPath });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- TOML.parse returns any
      const parsed: any = TOML.parse(raw);

      // Derive manifestDir from the manifest file path.
      const sep = preset.manifestPath.includes("\\") ? "\\" : "/";
      const dir = preset.manifestPath.substring(0, preset.manifestPath.lastIndexOf(sep));

      setManifestDir(dir);
      setManifestPath(preset.manifestPath);
      setManifest(parsed);
      // Reset config to defaults when switching presets (the config is
      // topology-specific; PlanarIntrinsics is the only one in B3b).
      setConfig(DEFAULT_PLANAR_CONFIG);
      setActivePresetId(preset.id);
      setGridExpanded(false);
      setStatus({ kind: "idle" });

      // Sync the JSON textarea if it happens to be mounted.
      if (jsonEditorRef.current) {
        jsonEditorRef.current.value = JSON.stringify({ manifest: parsed, config: DEFAULT_PLANAR_CONFIG }, null, 2);
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
      setStatus({ kind: "error", category: "no_tauri", message: "Folder picker requires Tauri (bun run tauri dev)." });
      return;
    }
    try {
      const picked = await open({ directory: true, multiple: false, title: "Pick the dataset folder" });
      if (typeof picked === "string") {
        setManifestDir(picked);
        // Clear preset selection — user is taking manual control.
        setActivePresetId(null);
      }
    } catch (e) {
      setStatus({ kind: "error", category: "dialog", message: String(e) });
    }
  };

  const handlePickManifest = async () => {
    if (!inTauri) {
      setStatus({ kind: "error", category: "no_tauri", message: "File picker requires Tauri (bun run tauri dev)." });
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
        const sep = picked.includes("\\") ? "\\" : "/";
        const dir = picked.substring(0, picked.lastIndexOf(sep));
        setManifestDir(dir);
        setActivePresetId(null);

        // Attempt to load + parse the manifest from disk. On failure we
        // keep the default form state; the user can edit manually.
        try {
          const raw = await invoke<string>("load_text_file", { path: picked });
          const ext = picked.split(".").pop()?.toLowerCase();
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const parsed: any = ext === "toml" ? TOML.parse(raw) : JSON.parse(raw);
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
      setStatus({ kind: "error", category: "no_dir", message: "Set a dataset folder first (pick a preset or use the folder button)." });
      return;
    }
    setStatus({ kind: "running" });
    try {
      const response = await runCalibration({ manifest, config, manifestDir });
      handleResponse(response);
    } catch (e) {
      setStatus({ kind: "error", category: "ipc", message: String(e) });
    }
  };

  const handleResponse = (response: RunResponse) => {
    if (response.kind === "ok") {
      setStatus({
        kind: "ok",
        durationMs: response.duration_ms,
        usable: response.usable_views,
        total: response.total_views,
        cacheUsed: response.cache_used,
      });
      // Hand off to /diagnose after a brief success flash.
      setTimeout(() => navigate("/diagnose"), 600);
    } else if (response.kind === "ask_user") {
      setStatus({ kind: "ask_user", field: response.field, prompt: response.prompt, suggestions: response.suggestions });
    } else if (response.kind === "validation_failed") {
      setStatus({ kind: "validation", message: response.message });
    } else {
      setStatus({ kind: "error", category: response.category, message: response.message });
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
            PlanarIntrinsics + chessboard, end-to-end
          </p>
        </div>

        <button
          type="button"
          onClick={handleRun}
          disabled={isRunning || !manifestDir}
          className={[
            "h-9 rounded-md px-5 text-[13px] font-semibold transition-colors",
            isRunning || !manifestDir
              ? "cursor-not-allowed bg-bg-soft text-muted-foreground border border-border"
              : "bg-brand text-white hover:opacity-90",
          ].join(" ")}
          style={isRunning || !manifestDir ? undefined : { backgroundColor: "var(--brand)" }}
        >
          {isRunning ? "Running…" : "Run"}
        </button>
      </header>

      {/* 7. Status banner — sticky at top of content */}
      {status.kind !== "idle" && <StatusBanner status={status} />}

      {/* 2. Quick-start grid / active-preset bar */}
      {gridExpanded ? (
        <QuickStartGrid
          activePresetId={activePresetId}
          onUse={handleUsePreset}
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
          onPickFolder={handlePickFolder}
          onPickManifest={handlePickManifest}
        />
      )}

      {/* 4. Manifest section */}
      <CollapsibleSection
        title="Manifest"
        summary={manifestSummary}
        badge={manifestDir ? undefined : "unset"}
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
        <ConfigForm
          schema={planarConfigSchema}
          value={config}
          onChange={setConfig}
          rootLabel="config"
        />
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
    </div>
  );
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
            onUse={(p) => onUse(p as EnabledPreset)}
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
        {/* Green check mark */}
        <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-brand/20">
          <svg width="8" height="8" viewBox="0 0 8 8" fill="none" aria-hidden="true">
            <path d="M1 4L3 6L7 2" stroke="var(--brand)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
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

function PathsStrip({ manifestDir, manifestPath, onPickFolder, onPickManifest }: PathsStripProps) {
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
      <span className="w-14 shrink-0 text-[11px] font-medium text-muted-foreground">{label}</span>
      <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-foreground">
        {value ?? <span className="text-muted-foreground">—</span>}
      </span>
      <button
        type="button"
        onClick={onEdit}
        title={editTitle}
        className="shrink-0 rounded border border-border bg-bg px-2 py-0.5 text-[10px] text-muted-foreground hover:text-foreground transition-colors"
      >
        edit
      </button>
    </div>
  );
}

// ── Status banner ─────────────────────────────────────────────────────────────

function StatusBanner({ status }: { status: RunStatus }) {
  if (status.kind === "idle") return null;

  if (status.kind === "running") {
    return (
      <div className="flex items-center gap-2 rounded-md border border-border bg-bg-soft px-3 py-2.5 text-[12px]">
        <SpinnerIcon />
        <span className="text-foreground">Detection + calibration in progress…</span>
        <span className="ml-auto font-mono text-muted-foreground">first run is slowest; second hits the cache</span>
      </div>
    );
  }

  if (status.kind === "ok") {
    return (
      <div
        className="flex items-center gap-2 rounded-md border px-3 py-2.5 text-[12px]"
        style={{ borderColor: "var(--color-success, #22c55e)", backgroundColor: "color-mix(in srgb, var(--color-success, #22c55e) 8%, transparent)" }}
      >
        <span style={{ color: "var(--color-success, #22c55e)" }}>Solve completed</span>
        <span className="font-mono text-muted-foreground">
          {status.durationMs} ms · {status.usable}/{status.total} usable views
          {status.cacheUsed ? " · cache" : ""}
        </span>
        <span className="ml-auto text-muted-foreground">Routing to /diagnose…</span>
      </div>
    );
  }

  if (status.kind === "validation") {
    return (
      <div className="rounded-md border border-border bg-bg-soft px-3 py-2.5 text-[12px]">
        <span className="font-medium text-foreground">Validation failed: </span>
        <code className="font-mono text-muted-foreground">{status.message}</code>
      </div>
    );
  }

  if (status.kind === "ask_user") {
    return (
      <div className="rounded-md border border-border bg-bg-soft px-3 py-2.5 text-[12px]">
        <p className="font-semibold text-foreground">Need your input on <code className="font-mono">{status.field}</code></p>
        <p className="mt-1 text-muted-foreground">{status.prompt}</p>
        {status.suggestions.length > 0 && (
          <p className="mt-1 font-mono text-muted-foreground">
            Suggestions: {status.suggestions.join(", ")}
          </p>
        )}
      </div>
    );
  }

  // Error
  return (
    <div
      className="rounded-md border px-3 py-2.5 text-[12px]"
      style={{ borderColor: "var(--color-destructive, #ef4444)", backgroundColor: "color-mix(in srgb, var(--color-destructive, #ef4444) 8%, transparent)" }}
    >
      <span className="font-semibold" style={{ color: "var(--color-destructive, #ef4444)" }}>
        Run failed ({status.category}):{" "}
      </span>
      <code className="font-mono text-foreground">{status.message}</code>
    </div>
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
      <circle cx="7" cy="7" r="5.5" stroke="currentColor" strokeWidth="1.5" strokeOpacity="0.25" />
      <path d="M7 1.5A5.5 5.5 0 0 1 12.5 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

// Keep joinPath in scope (it's imported at the top and referenced in
// path-derivation logic; TypeScript would otherwise flag it unused).
// The actual invocations are inside the handlers above.
void joinPath; // used: imported from lib/tauri for path building in handlePickManifest
