import { useCallback, useState } from "react";

/** The width of an element as it is laid out, kept current; `null` before
 * the first measure (and where there is no layout, as in a test). Pass the
 * returned `ref` to the element. */
export function useElementWidth<T extends HTMLElement>(): [
  (node: T | null) => void,
  number | null,
] {
  const [width, setWidth] = useState<number | null>(null);
  const [observer] = useState(() =>
    typeof ResizeObserver === "undefined"
      ? null
      : new ResizeObserver((entries) => {
          const w = entries[0]?.contentRect.width;
          if (w !== undefined && w > 0) setWidth(Math.round(w));
        }),
  );
  const ref = useCallback(
    (node: T | null) => {
      observer?.disconnect();
      if (node) observer?.observe(node);
    },
    [observer],
  );
  return [ref, width];
}
