import { useEffect, useState } from "react";

export interface SceneColors {
  /** Active camera frustum / target board border. Brand cyan. */
  active: string;
  /** Inactive camera frustums. Muted foreground HSL. */
  inactive: string;
  /** Rig origin axes — picks the foreground token so it adapts to theme. */
  axes: string;
  /** Target board fill (translucent). */
  boardFill: string;
  /** Canvas background. Reads --color-background as a hex so R3F's
   * `gl.setClearColor` matches the surrounding panel. */
  background: string;
}

function read(name: string): string {
  if (typeof document === "undefined") return "#999";
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  if (!value) return "#999";
  // Stored as `H S% L%` triplets (per index.css). Wrap in hsl().
  if (/^\d/.test(value)) return `hsl(${value})`;
  return value;
}

/** Reads vitavision tokens from the document and re-reads when the
 * `.dark` class flips. Keeps the 3D scene visually in sync with the
 * rest of the app. */
export function useThemeColors(): SceneColors {
  const compute = (): SceneColors => ({
    active: read("--brand"),
    inactive: read("--muted-foreground"),
    axes: read("--foreground"),
    boardFill: `hsl(${getComputedStyle(document.documentElement)
      .getPropertyValue("--brand")
      .trim()} / 0.18)`,
    background: read("--bg-soft"),
  });

  const [colors, setColors] = useState<SceneColors>(() =>
    typeof window === "undefined"
      ? {
          active: "#5cd0e0",
          inactive: "#7a8aa0",
          axes: "#c8d4e0",
          boardFill: "rgba(92, 208, 224, 0.18)",
          background: "#0e1421",
        }
      : compute(),
  );

  useEffect(() => {
    const obs = new MutationObserver(() => setColors(compute()));
    obs.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });
    return () => obs.disconnect();
  }, []);

  return colors;
}
