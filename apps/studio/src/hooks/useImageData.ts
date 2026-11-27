import { useEffect, useState } from "react";
import { api } from "../api";
import { useStore } from "../store";
import { decodeImage, type ImageData } from "../lib/imagePixels";
import type { FrameKey } from "../types";

export type { ImageData } from "../lib/imagePixels";

/** Decoded pixels tagged with the request key they were loaded for, so a
 * hook can drop them as soon as its inputs move on to a different key. */
interface Loaded {
  key: string;
  data: ImageData;
}

/** Loads a frame's source image through the back end's `image` resource
 * (the file itself, or a PNG of a format the browser cannot decode) and
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
    decodeImage(api.urls.image({ path }))
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
 * undistorted pixel frame (the `undistorted_image` resource, a PNG). The
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
  const run = useStore((s) => s.run);
  const key = `${run?.dir}|${run?.id}|${path}|${camera}|${roiX},${roiY},${roiW},${roiH}`;
  const [loaded, setLoaded] = useState<Loaded | null>(null);

  useEffect(() => {
    if (path === null || run === null) return;
    const roi =
      roiX === undefined || roiY === undefined || roiW === undefined || roiH === undefined
        ? null
        : { x: roiX, y: roiY, w: roiW, h: roiH };
    let cancelled = false;
    decodeImage(api.urls.undistortedImage({ run, path, camera, roi }))
      .then((data) => {
        if (!cancelled) setLoaded({ key, data });
      })
      .catch((e: unknown) => {
        if (!cancelled) onError?.(`Could not load undistorted image: ${errorText(e)}`);
      });
    return () => {
      cancelled = true;
    };
  }, [run, path, roiX, roiY, roiW, roiH, camera, key, onError]);

  return path !== null && loaded?.key === key ? loaded.data : null;
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
