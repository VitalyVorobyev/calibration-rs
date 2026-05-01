import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import type {
  FrameKey,
  LoadExportResult,
  PlanarExport,
  TargetFeatureResidual,
} from "../types";

// True iff the page is running inside a Tauri webview (i.e. the IPC
// internals have been injected). When the user runs `bun run dev` and
// loads localhost:1420 in a regular browser, this is false and any
// `invoke` / dialog call would throw `Cannot read properties of
// undefined (reading 'invoke')`. We guard up-front so the failure mode
// is a readable banner instead of an opaque TypeError.
function isTauriContext(): boolean {
  return (
    typeof window !== "undefined" &&
    "__TAURI_INTERNALS__" in window &&
    (window as unknown as { __TAURI_INTERNALS__?: unknown })
      .__TAURI_INTERNALS__ != null
  );
}

// Path joining for absolute filesystem paths. Tauri exposes a path
// utility but for v0 the manifest only ever uses POSIX-style relatives,
// so a hand-rolled join keeps the dependency surface tiny.
function joinPath(dir: string, ...rest: string[]): string {
  let out = dir.replace(/[\\/]+$/, "");
  for (const segment of rest) {
    if (!segment) continue;
    const sep = out.includes("\\") && !out.includes("/") ? "\\" : "/";
    out = `${out}${sep}${segment.replace(/^[\\/]+/, "")}`;
  }
  return out;
}

interface ExportState {
  exportPath: string;
  exportDir: string;
  data: PlanarExport;
  frames: FrameKey[];
}

export function ResidualViewer() {
  const [state, setState] = useState<ExportState | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const tauriOk = isTauriContext();

  const handleOpen = async () => {
    setError(null);
    if (!isTauriContext()) {
      setError(
        "Tauri runtime not detected. Launch the app with `bun run tauri dev` " +
          "(or the bundled binary). Plain `bun run dev` only starts Vite, so " +
          "the file-dialog and IPC commands aren't wired up.",
      );
      return;
    }
    let chosen: string | null = null;
    try {
      chosen = await openDialog({
        multiple: false,
        directory: false,
        filters: [{ name: "Calibration export", extensions: ["json"] }],
      });
    } catch (e) {
      setError(`File dialog error: ${e}`);
      return;
    }
    if (!chosen) return;

    try {
      const result = await invoke<LoadExportResult>("load_export", {
        path: chosen,
      });
      const manifest = result.export.image_manifest;
      if (!manifest) {
        setError(
          "This export has no image_manifest field. v0 requires the manifest " +
            "to render the source images alongside the residuals.",
        );
        return;
      }
      const root = joinPath(result.export_dir, manifest.root);
      const frames: FrameKey[] = manifest.frames.map((f) => ({
        pose: f.pose,
        camera: f.camera,
        label: `pose ${f.pose} · cam ${f.camera}`,
        abs_path: joinPath(root, f.path),
        roi: f.roi,
      }));
      setState({
        exportPath: chosen,
        exportDir: result.export_dir,
        data: result.export,
        frames,
      });
      setSelected(frames[0] ? frameKey(frames[0]) : null);
    } catch (e) {
      setError(`Could not load export: ${e}`);
    }
  };

  const frame = useMemo<FrameKey | null>(() => {
    if (!state || !selected) return null;
    return state.frames.find((f) => frameKey(f) === selected) ?? null;
  }, [state, selected]);

  // Load the image bytes when the selection changes.
  useEffect(() => {
    if (!frame) {
      setImgUrl(null);
      return;
    }
    let cancelled = false;
    setImgUrl(null);
    invoke<string>("load_image", { path: frame.abs_path })
      .then((dataUrl) => {
        if (!cancelled) setImgUrl(dataUrl);
      })
      .catch((e) => {
        if (!cancelled) setError(`Could not load image: ${e}`);
      });
    return () => {
      cancelled = true;
    };
  }, [frame]);

  // Draw image + arrows whenever the selection or image changes.
  useEffect(() => {
    if (!state || !frame || !imgUrl) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const img = new Image();
    img.onload = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const roi = frame.roi;
      const sx = roi?.x ?? 0;
      const sy = roi?.y ?? 0;
      const sw = roi?.w ?? img.naturalWidth;
      const sh = roi?.h ?? img.naturalHeight;
      canvas.width = sw;
      canvas.height = sh;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
      drawResidualArrows(ctx, state.data.per_feature_residuals.target, frame, sx, sy);
    };
    img.onerror = () => setError("Image failed to decode.");
    img.src = imgUrl;
  }, [state, frame, imgUrl]);

  return (
    <section
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        minHeight: 0,
      }}
    >
      {!tauriOk && (
        <div
          style={{
            border: "1px solid #d4a72c",
            background: "#fff7d6",
            color: "#5a4400",
            padding: "10px",
            borderRadius: 6,
            fontSize: "13px",
          }}
        >
          Tauri runtime not detected — you appear to be running plain Vite
          (<code>bun run dev</code>) in a browser. Launch the desktop app
          with <code>bun run tauri dev</code> to use the file dialog.
        </div>
      )}
      <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
        <button onClick={handleOpen} disabled={!tauriOk}>
          Open Export…
        </button>
        {state && (
          <select
            value={selected ?? ""}
            onChange={(e) => setSelected(e.target.value)}
          >
            {state.frames.map((f) => (
              <option key={frameKey(f)} value={frameKey(f)}>
                {f.label}
              </option>
            ))}
          </select>
        )}
        {state && (
          <span style={{ opacity: 0.7, fontSize: "12px" }}>
            mean reprojection error: {state.data.mean_reproj_error.toFixed(3)} px
          </span>
        )}
      </div>

      {error && (
        <div
          style={{
            border: "1px solid #c0392b",
            background: "#fdecea",
            color: "#7d2118",
            padding: "10px",
            borderRadius: 6,
            fontSize: "13px",
          }}
        >
          {error}
        </div>
      )}

      <div
        style={{
          flex: 1,
          minHeight: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "rgba(127,127,127,0.08)",
          borderRadius: 6,
          overflow: "auto",
        }}
      >
        {state ? (
          <canvas
            ref={canvasRef}
            style={{
              maxWidth: "100%",
              maxHeight: "100%",
              imageRendering: "pixelated",
              boxShadow: "0 1px 4px rgba(0,0,0,0.2)",
            }}
          />
        ) : (
          <div style={{ opacity: 0.6, fontSize: "13px" }}>
            Open an <code>export.json</code> from a calibration run with an
            <code> image_manifest</code>.
          </div>
        )}
      </div>

      {state && frame && (
        <ResidualLegend
          residuals={state.data.per_feature_residuals.target.filter(
            (r) => r.pose === frame.pose && r.camera === frame.camera,
          )}
        />
      )}
    </section>
  );
}

function frameKey(f: { pose: number; camera: number }): string {
  return `${f.pose}/${f.camera}`;
}

/** Cap on rendered arrow length so a freak diverged feature does not
 * dominate the canvas. The numeric value still reads correctly off the
 * legend if needed. */
const ARROW_LENGTH_CAP_PX = 40;
/** Multiplier applied to the residual vector so sub-pixel errors are
 * actually visible on a 640×480 canvas. */
const ARROW_GAIN = 30;

function drawResidualArrows(
  ctx: CanvasRenderingContext2D,
  all: TargetFeatureResidual[],
  frame: FrameKey,
  roiX: number,
  roiY: number,
) {
  const arrows = all.filter((r) => r.pose === frame.pose && r.camera === frame.camera);
  for (const r of arrows) {
    if (!r.projected_px) continue;
    const ox = r.observed_px[0] - roiX;
    const oy = r.observed_px[1] - roiY;
    const dx0 = r.projected_px[0] - r.observed_px[0];
    const dy0 = r.projected_px[1] - r.observed_px[1];
    const mag = Math.hypot(dx0, dy0);
    if (mag < 1e-6) continue;
    const gain = Math.min(ARROW_GAIN, ARROW_LENGTH_CAP_PX / Math.max(mag, 1e-6) / 1);
    const dx = dx0 * gain;
    const dy = dy0 * gain;
    ctx.strokeStyle = colorForError(r.error_px ?? mag);
    ctx.lineWidth = 1.5;
    drawArrow(ctx, ox, oy, ox + dx, oy + dy);
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.beginPath();
    ctx.arc(ox, oy, 1.6, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = 4;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.lineTo(
    x2 - head * Math.cos(angle - Math.PI / 6),
    y2 - head * Math.sin(angle - Math.PI / 6),
  );
  ctx.moveTo(x2, y2);
  ctx.lineTo(
    x2 - head * Math.cos(angle + Math.PI / 6),
    y2 - head * Math.sin(angle + Math.PI / 6),
  );
  ctx.stroke();
}

function colorForError(err: number): string {
  // Cool→warm ramp matching the histogram bucket edges (1, 2, 5, 10 px)
  // already used for `target_hist_per_camera`. Keeps the visual scale
  // consistent across UI surfaces if a histogram view is added later.
  if (err < 1) return "#1abc9c"; // teal
  if (err < 2) return "#2ecc71"; // green
  if (err < 5) return "#f1c40f"; // yellow
  if (err < 10) return "#e67e22"; // orange
  return "#e74c3c"; // red
}

function ResidualLegend({ residuals }: { residuals: TargetFeatureResidual[] }) {
  const errs = residuals
    .map((r) => r.error_px)
    .filter((e): e is number => typeof e === "number");
  const mean = errs.length > 0 ? errs.reduce((a, b) => a + b, 0) / errs.length : 0;
  const max = errs.length > 0 ? Math.max(...errs) : 0;
  const diverged = residuals.length - errs.length;
  return (
    <div style={{ display: "flex", gap: 14, flexWrap: "wrap", fontSize: 12, opacity: 0.85 }}>
      <span>features: {residuals.length}</span>
      <span>diverged: {diverged}</span>
      <span>mean error: {mean.toFixed(3)} px</span>
      <span>max error: {max.toFixed(3)} px</span>
      <span>
        legend (px):
        <SwatchLabel color="#1abc9c" label="<1" />
        <SwatchLabel color="#2ecc71" label="<2" />
        <SwatchLabel color="#f1c40f" label="<5" />
        <SwatchLabel color="#e67e22" label="<10" />
        <SwatchLabel color="#e74c3c" label="≥10" />
      </span>
      <span>
        arrows scaled ×{ARROW_GAIN} (cap {ARROW_LENGTH_CAP_PX}px) for
        sub-pixel visibility.
      </span>
    </div>
  );
}

function SwatchLabel({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 3, marginLeft: 6 }}>
      <span
        style={{
          display: "inline-block",
          width: 10,
          height: 10,
          background: color,
          borderRadius: 2,
        }}
      />
      {label}
    </span>
  );
}
