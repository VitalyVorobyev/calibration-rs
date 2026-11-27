/** The decoded size of an image, read by loading it once. */
import { useEffect, useState } from "react";

export type ImageSize = { width: number; height: number };

/** The decoded size of the image at `url`; `"error"` when it does not
 * open, `null` while it loads (or without a url). */
export function useImageSize(url: string | null): ImageSize | "error" | null {
  const [result, setResult] = useState<{ url: string; size: ImageSize | "error" } | null>(
    null,
  );
  useEffect(() => {
    if (url === null) return;
    let live = true;
    const img = new Image();
    img.onload = () => {
      if (live)
        setResult({ url, size: { width: img.naturalWidth, height: img.naturalHeight } });
    };
    img.onerror = () => {
      if (live) setResult({ url, size: "error" });
    };
    img.src = url;
    return () => {
      live = false;
    };
  }, [url]);
  return url !== null && result?.url === url ? result.size : null;
}
