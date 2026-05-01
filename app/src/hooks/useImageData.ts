import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { FrameKey } from "../types";

export interface ImageData {
  /** The decoded HTMLImageElement, ready for `ctx.drawImage`. */
  image: HTMLImageElement;
  /** Width / height of the original PNG in pixels. */
  naturalWidth: number;
  naturalHeight: number;
  /** Per-pixel luminance for the full image, row-major
   * (`y * naturalWidth + x`). 8-bit. We store luminance only — that's
   * what the histogram + cursor readout need, and one byte per pixel
   * keeps memory at reasonable bounds (8 MB for a 4320×540 strip). */
  luminance: Uint8Array;
}

/** Loads a frame's PNG via the Tauri `load_image` command and decodes
 * its luminance buffer once, so both `FrameCanvas` (drawing) and
 * `ResidualViewer` (cursor readout + histogram) can share the same
 * decode without the IPC roundtrip happening twice. */
export function useImageData(
  frame: FrameKey | null,
  onError?: (msg: string) => void,
): ImageData | null {
  const [data, setData] = useState<ImageData | null>(null);

  useEffect(() => {
    if (!frame) {
      setData(null);
      return;
    }
    let cancelled = false;
    setData(null);
    invoke<string>("load_image", { path: frame.abs_path })
      .then((dataUrl) => {
        if (cancelled) return;
        const img = new Image();
        img.onload = () => {
          if (cancelled) return;
          try {
            const luminance = decodeLuminance(img);
            setData({
              image: img,
              naturalWidth: img.naturalWidth,
              naturalHeight: img.naturalHeight,
              luminance,
            });
          } catch (e) {
            onError?.(`Pixel decode failed: ${e}`);
          }
        };
        img.onerror = () => {
          if (!cancelled) onError?.("Image failed to decode.");
        };
        img.src = dataUrl;
      })
      .catch((e) => {
        if (!cancelled) onError?.(`Could not load image: ${e}`);
      });
    return () => {
      cancelled = true;
    };
  }, [frame?.abs_path, onError]);

  return data;
}

/** Pull luminance out of an HTMLImageElement by drawing it into an
 * OffscreenCanvas (or a hidden 2D canvas for older runtimes) and
 * extracting RGBA. We compute Rec. 601 luma and discard the rest. */
type Canvas2D = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;

function decodeLuminance(img: HTMLImageElement): Uint8Array {
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  const ctx = acquireContext(w, h);
  if (!ctx) throw new Error("could not acquire 2D context");
  ctx.drawImage(img, 0, 0);
  const rgba = ctx.getImageData(0, 0, w, h).data;
  const lum = new Uint8Array(w * h);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j++) {
    // Rec. 601 luma — close enough for diagnostic readout.
    lum[j] = (0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]) | 0;
  }
  return lum;
}

function acquireContext(w: number, h: number): Canvas2D | null {
  if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(w, h).getContext(
      "2d",
    ) as OffscreenCanvasRenderingContext2D | null;
  }
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c.getContext("2d");
}

/** Read the luminance at integer image coords, or null if out of bounds. */
export function getPixelLum(data: ImageData, x: number, y: number): number | null {
  const xi = x | 0;
  const yi = y | 0;
  if (xi < 0 || yi < 0 || xi >= data.naturalWidth || yi >= data.naturalHeight) {
    return null;
  }
  return data.luminance[yi * data.naturalWidth + xi];
}

/** Compute a fixed-bin histogram over a sub-rectangle of the image. */
export function rectHistogram(
  data: ImageData,
  rect: { x: number; y: number; w: number; h: number },
  binCount: number,
): number[] {
  const bins = new Array<number>(binCount).fill(0);
  const w = data.naturalWidth;
  const h = data.naturalHeight;
  const x0 = Math.max(0, Math.floor(rect.x));
  const y0 = Math.max(0, Math.floor(rect.y));
  const x1 = Math.min(w, Math.ceil(rect.x + rect.w));
  const y1 = Math.min(h, Math.ceil(rect.y + rect.h));
  if (x1 <= x0 || y1 <= y0) return bins;
  const scale = binCount / 256;
  const lum = data.luminance;
  for (let y = y0; y < y1; y++) {
    const row = y * w;
    for (let x = x0; x < x1; x++) {
      const v = lum[row + x];
      let bin = (v * scale) | 0;
      if (bin >= binCount) bin = binCount - 1;
      bins[bin]++;
    }
  }
  return bins;
}
