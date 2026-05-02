/** Run workspace (B3b) — the user picks a `dataset.toml` manifest,
 * tweaks the schema-driven `DatasetSpec` + `*Config` forms, hits Run,
 * and lands on /diagnose with the produced export active. PR 1 wires
 * up `PlanarIntrinsics` + chessboard end-to-end; the dropdown only
 * advertises that one topology so the user is never blocked by an
 * unimplemented case. Coverage to all 8 topologies + 4 detectors comes
 * in B3c (see ROADMAP.md).
 */
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { open } from "@tauri-apps/plugin-dialog";

import { ConfigForm, type JsonSchema } from "../../lib/configForm";
import { runCalibration, type RunResponse } from "../../lib/runCalibration";
import { isTauriContext } from "../../lib/tauri";
import datasetSchemaJson from "../../schemas/dataset_spec.json";
import planarConfigSchemaJson from "../../schemas/planar_intrinsics_config.json";

// schemars-emitted JSON Schemas have a richer type than our loose
// `JsonSchema` interface tracks; cast through `unknown` since both
// shapes are JSON-compatible.
const datasetSchema = datasetSchemaJson as unknown as JsonSchema;
const planarConfigSchema = planarConfigSchemaJson as unknown as JsonSchema;

const DEFAULT_DATASET = {
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

const DEFAULT_PLANAR_CONFIG = {
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

type RunStatus =
  | { kind: "idle" }
  | { kind: "running" }
  | { kind: "ok"; durationMs: number; usable: number; total: number; cacheUsed: boolean }
  | { kind: "error"; category: string; message: string }
  | { kind: "validation"; message: string }
  | { kind: "ask_user"; field: string; prompt: string; suggestions: string[] };

export function RunWorkspace() {
  const navigate = useNavigate();
  const inTauri = isTauriContext();

  const [manifestDir, setManifestDir] = useState<string | null>(null);
  const [manifestPath, setManifestPath] = useState<string | null>(null);
  const [manifest, setManifest] = useState<unknown>(DEFAULT_DATASET);
  const [config, setConfig] = useState<unknown>(DEFAULT_PLANAR_CONFIG);
  const [status, setStatus] = useState<RunStatus>({ kind: "idle" });

  const isRunning = status.kind === "running";

  const handlePickFolder = async () => {
    if (!inTauri) {
      setStatus({
        kind: "error",
        category: "no_tauri",
        message: "Folder picker is only available inside the Tauri app (bun run tauri dev).",
      });
      return;
    }
    try {
      const picked = await open({
        directory: true,
        multiple: false,
        title: "Pick the dataset folder (manifest will be resolved against it)",
      });
      if (typeof picked === "string") {
        setManifestDir(picked);
        setManifestPath(null);
      }
    } catch (e) {
      setStatus({ kind: "error", category: "dialog", message: String(e) });
    }
  };

  const handlePickManifest = async () => {
    if (!inTauri) {
      setStatus({
        kind: "error",
        category: "no_tauri",
        message: "File picker is only available inside the Tauri app.",
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
        // Manifest dir defaults to the manifest's parent.
        const sep = picked.includes("\\") ? "\\" : "/";
        const dir = picked.substring(0, picked.lastIndexOf(sep));
        setManifestDir(dir);
        // PR 1 doesn't yet parse the manifest from disk into the form
        // (TOML/JSON parser pulled in B3d). For now we just record
        // the path — users still edit in-form starting from the
        // defaults. B3d adds full load + autopopulate.
      }
    } catch (e) {
      setStatus({ kind: "error", category: "dialog", message: String(e) });
    }
  };

  const handleRun = async () => {
    if (!manifestDir) {
      setStatus({
        kind: "error",
        category: "no_dir",
        message: "Pick a dataset folder first so glob patterns can be resolved.",
      });
      return;
    }
    setStatus({ kind: "running" });
    try {
      const response = await runCalibration({
        manifest,
        config,
        manifestDir,
      });
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
      // Hand off to /diagnose — the runner already populated
      // ExportCache, but the diagnose viewer reads its export from
      // the Zustand store; B3c adds the live-export bridge so
      // navigation lights up automatically.
      setTimeout(() => navigate("/diagnose"), 600);
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

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-col gap-0.5">
          <h2 className="text-sm font-semibold tracking-tight">Run calibration</h2>
          <p className="text-[12px] text-muted-foreground">
            B3b vertical slice: PlanarIntrinsics + chessboard, end-to-end.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={handlePickFolder}
            className="rounded border border-border bg-bg-soft px-3 py-1.5 text-[12px]"
          >
            Pick dataset folder…
          </button>
          <button
            type="button"
            onClick={handlePickManifest}
            className="rounded border border-border bg-bg-soft px-3 py-1.5 text-[12px]"
          >
            Pick manifest…
          </button>
          <button
            type="button"
            onClick={handleRun}
            disabled={isRunning || !manifestDir}
            className="rounded bg-brand px-4 py-1.5 text-[12px] font-semibold text-white disabled:opacity-50"
            style={{ backgroundColor: "var(--brand)" }}
          >
            {isRunning ? "Running…" : "Run"}
          </button>
        </div>
      </header>

      <PathLine label="Folder" value={manifestDir} />
      <PathLine label="Manifest" value={manifestPath} />

      <StatusBanner status={status} />

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <Pane title="Dataset (DatasetSpec)">
          <ConfigForm
            schema={datasetSchema}
            value={manifest}
            onChange={setManifest}
            rootLabel="dataset"
          />
        </Pane>
        <Pane title="Calibration (PlanarIntrinsicsConfig)">
          <ConfigForm
            schema={planarConfigSchema}
            value={config}
            onChange={setConfig}
            rootLabel="config"
          />
        </Pane>
      </div>
    </div>
  );
}

function Pane({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="flex min-h-0 flex-col gap-2 rounded-md border border-border p-3">
      <h3 className="text-[12px] font-semibold tracking-tight text-muted-foreground">
        {title}
      </h3>
      <div className="flex flex-col gap-2">{children}</div>
    </section>
  );
}

function PathLine({ label, value }: { label: string; value: string | null }) {
  return (
    <p className="text-[11px] text-muted-foreground">
      <span className="font-semibold">{label}:</span>{" "}
      <span className="font-mono">{value ?? "<unset>"}</span>
    </p>
  );
}

function StatusBanner({ status }: { status: RunStatus }) {
  if (status.kind === "idle") return null;
  if (status.kind === "running") {
    return (
      <div className="rounded-md border border-border bg-bg-soft px-3 py-2 text-[12px]">
        Detection + calibration in progress… (first run is slowest, second hits the cache)
      </div>
    );
  }
  if (status.kind === "ok") {
    return (
      <div
        className="rounded-md border px-3 py-2 text-[12px]"
        style={{ borderColor: "var(--success, #22c55e)" }}
      >
        Solve completed in {status.durationMs} ms · {status.usable}/{status.total} usable views
        {status.cacheUsed ? " · cache reads possible" : ""}. Routing to /diagnose…
      </div>
    );
  }
  if (status.kind === "validation") {
    return (
      <div className="rounded-md border border-border bg-bg-soft px-3 py-2 text-[12px]">
        Manifest validation failed: <span className="font-mono">{status.message}</span>
      </div>
    );
  }
  if (status.kind === "ask_user") {
    return (
      <div className="rounded-md border border-border bg-bg-soft px-3 py-2 text-[12px]">
        <p className="font-semibold">Need your input on {status.field}</p>
        <p className="mt-1">{status.prompt}</p>
        {status.suggestions.length > 0 && (
          <p className="mt-1 text-muted-foreground">
            Suggestions: {status.suggestions.join(", ")}
          </p>
        )}
      </div>
    );
  }
  return (
    <div
      className="rounded-md border px-3 py-2 text-[12px]"
      style={{ borderColor: "var(--danger, #ef4444)" }}
    >
      <span className="font-semibold">Run failed ({status.category}): </span>
      <span className="font-mono">{status.message}</span>
    </div>
  );
}

