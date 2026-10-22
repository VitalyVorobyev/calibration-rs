import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { decodeImage, type ImageData } from "../lib/imagePixels";
import { assetUrl } from "../lib/tauri";
import type { FrameKey } from "../types";

export type { ImageData } from "../lib/imagePixels";

/** Decoded pixels tagged with the request key they were loaded for, so a
 * hook can drop them as soon as its inputs move on to a different key. */
interface Loaded {
  key: string;
  data: ImageData;
}

/** Loads a frame's source image from disk through the asset protocol and
 * decodes its luminance once, so both `FrameCanvas` (drawing) and
 * `DiagnoseWorkspace` (cursor readout + histogram) share one decode. */
export function useImageData(
  frame: FrameKey | null,
  onError?: (msg: string) => void,
): ImageData | null {
  // Depend on the path only, not `frame` itself: the caller may hand us a
  // fresh `FrameKey` object each render even when the path is unchanged,
  // and re-loading on every render would defeat the point of this hook.
  const path = frame?.abs_path ?? null;
  const [loaded, setLoaded] = useState<Loaded | null>(null);

  useEffect(() => {
    if (path === null) return;
    let cancelled = false;
    decodeImage(assetUrl(path))
      .then((data) => {
        if (!cancelled) setLoaded({ key: path, data });
      })
      .catch((e: unknown) => {
        if (!cancelled) onError?.(`Could not load image ${path}: ${errorText(e)}`);
      });
    return () => {
      cancelled = true;
    };
  }, [path, onError]);

  return path !== null && loaded?.key === path ? loaded.data : null;
}

/** Loads a frame through the backend's canonical camera model into the
 * undistorted pixel frame. The backend answers with PNG bytes, shown
 * through a blob URL that lives as long as the image is current. The
 * returned image is already
 * ROI-local when `frame.roi` is set, so callers should render it without
 * applying the ROI crop a second time. */
export function useUndistortedImageData(
  frame: FrameKey | null,
  camera: number,
  onError?: (msg: string) => void,
): ImageData | null {
  // Depend on the individual `frame` / `frame.roi` fields actually read,
  // not the object itself (see `useImageData`).
  const path = frame?.abs_path ?? null;
  const roiX = frame?.roi?.x;
  const roiY = frame?.roi?.y;
  const roiW = frame?.roi?.w;
  const roiH = frame?.roi?.h;
  const key = `${path}|${camera}|${roiX},${roiY},${roiW},${roiH}`;
  const [loaded, setLoaded] = useState<Loaded | null>(null);

  useEffect(() => {
    if (path === null) return;
    const roi =
      roiX === undefined || roiY === undefined || roiW === undefined || roiH === undefined
        ? null
        : { x: roiX, y: roiY, w: roiW, h: roiH };
    let cancelled = false;
    let url: string | null = null;
    invoke<ArrayBuffer>("load_undistorted_image", { path, camera, roi })
      .then((bytes) => {
        if (cancelled) return null;
        url = URL.createObjectURL(new Blob([bytes], { type: "image/png" }));
        return decodeImage(url);
      })
      .then((data) => {
        if (!cancelled && data) setLoaded({ key, data });
      })
      .catch((e: unknown) => {
        if (!cancelled) onError?.(`Could not load undistorted image: ${errorText(e)}`);
      });
    return () => {
      cancelled = true;
      // The frame viewer displays the image by its URL, so the URL lives
      // until this key's image is no longer returned.
      if (url) URL.revokeObjectURL(url);
    };
  }, [path, roiX, roiY, roiW, roiH, camera, key, onError]);

  return path !== null && loaded?.key === key ? loaded.data : null;
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
