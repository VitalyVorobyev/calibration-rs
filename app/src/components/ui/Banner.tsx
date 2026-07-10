import type { HTMLAttributes } from "react";
import { cx } from "./cx";

export type BannerVariant = "error" | "success" | "warning" | "neutral";

const VARIANT_CLASSES: Record<BannerVariant, string> = {
  error: "border-destructive bg-destructive/[0.08]",
  success: "border-success bg-success/[0.08]",
  warning: "border-warning bg-warning/[0.08]",
  neutral: "border-border bg-bg-soft",
};

export interface BannerProps extends HTMLAttributes<HTMLDivElement> {
  variant?: BannerVariant;
}

/** Left-accent status banner (load errors, run status, unresolved-fields
 * notices). One shape, four semantic colors — this consolidates two
 * banner idioms that had drifted apart: the left-accent-bar style four
 * files already used verbatim, and RunWorkspace's separate full-border
 * rounded style (which also leaned on inline `style={{ color: "var(--brand)" }}`
 * — invalid CSS, since `--brand` is a bare `H S% L%` triplet, not a
 * `hsl(...)` color; those declarations were silently dropped). Content
 * composition (bold prefixes, secondary text, inline actions) stays with
 * the caller via `children`. */
export function Banner({
  variant = "neutral",
  className,
  children,
  ...rest
}: BannerProps) {
  return (
    <div
      className={cx(
        "rounded-md border-l-2 p-2.5 text-[13px] text-foreground",
        VARIANT_CLASSES[variant],
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}
