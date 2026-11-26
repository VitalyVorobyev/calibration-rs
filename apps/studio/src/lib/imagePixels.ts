/** Decoded frame pixels: the image for drawing, plus its luminance for the
 * cursor readout and the ROI histogram. */
export interface ImageData {
  /** The decoded HTMLImageElement, ready for `ctx.drawImage`. */
  image: HTMLImageElement;
  /** Width / height of the original image in pixels. */
  naturalWidth: number;
  naturalHeight: number;
  /** Per-pixel luminance for the full image, row-major
   * (`y * naturalWidth + x`). 8-bit. Luminance only — that's what the
   * histogram + cursor readout need, and one byte per pixel keeps memory
   * at reasonable bounds (8 MB for a 4320×540 strip). */
  luminance: Uint8Array;
}

/** Load `src` into an image and decode its luminance. `crossOrigin` is
 * set so an asset-protocol image (another origin than the page) does not
 * taint the canvas the luminance is read from; the protocol answers with
 * the page's origin in `Access-Control-Allow-Origin`. */
export function decodeImage(src: string): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      try {
        resolve({
          image: img,
          naturalWidth: img.naturalWidth,
          naturalHeight: img.naturalHeight,
          luminance: decodeLuminance(img),
        });
      } catch (e) {
        reject(new Error(`Pixel decode failed: ${String(e)}`));
      }
    };
    img.onerror = () => reject(new Error("Image failed to decode."));
    img.src = src;
  });
}

type Canvas2D = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;

/** Pull luminance out of an image by drawing it into an OffscreenCanvas
 * (or a hidden 2D canvas for older runtimes) and reading back RGBA:
 * Rec. 601 luma, the rest discarded. */
function decodeLuminance(img: HTMLImageElement): Uint8Array {
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  const ctx = acquireContext(w, h);
  if (!ctx) throw new Error("could not acquire 2D context");
  ctx.drawImage(img, 0, 0);
  return rgbaToLuminance(ctx.getImageData(0, 0, w, h).data);
}

/** Rec. 601 luma of an RGBA buffer, one byte per pixel. */
export function rgbaToLuminance(rgba: ArrayLike<number>): Uint8Array {
  const lum = new Uint8Array(rgba.length >> 2);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j++) {
    lum[j] = (0.299 * rgba[i]! + 0.587 * rgba[i + 1]! + 0.114 * rgba[i + 2]!) | 0;
  }
  return lum;
}

function acquireContext(w: number, h: number): Canvas2D | null {
  if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(w, h).getContext("2d");
  }
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c.getContext("2d");
}

type Pixels = Pick<ImageData, "naturalWidth" | "naturalHeight" | "luminance">;

/** Read the luminance at integer image coords, or null if out of bounds. */
export function getPixelLum(data: Pixels, x: number, y: number): number | null {
  const xi = x | 0;
  const yi = y | 0;
  if (xi < 0 || yi < 0 || xi >= data.naturalWidth || yi >= data.naturalHeight) {
    return null;
  }
  return data.luminance[yi * data.naturalWidth + xi]!;
}

/** Compute a fixed-bin histogram over a sub-rectangle of the image. */
export function rectHistogram(
  data: Pixels,
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
      const v = lum[row + x]!;
      let bin = (v * scale) | 0;
      if (bin >= binCount) bin = binCount - 1;
      bins[bin]!++;
    }
  }
  return bins;
}
