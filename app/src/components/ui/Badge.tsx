import type { HTMLAttributes } from "react";
import { cx } from "./cx";

export type BadgeVariant =
  "brand" | "accent" | "success" | "warning" | "destructive" | "neutral";

const VARIANT_CLASSES: Record<BadgeVariant, string> = {
  brand: "bg-brand/[0.12] text-brand",
  accent: "bg-accent/[0.12] text-accent",
  success: "bg-success/[0.12] text-success",
  warning: "bg-warning/[0.12] text-warning",
  destructive: "bg-destructive/[0.12] text-destructive",
  neutral: "border border-border text-muted-foreground",
};

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

/** Small pill label (CollapsibleSection's unresolved-count badge,
 * PresetCard's topology/milestone/active badges). Replaces per-file
 * `color-mix(in srgb, var(--color-destructive, #ef4444) …)` hacks with
 * the real `--color-destructive` token now that it (and `--success` /
 * `--warning`) are declared in index.css. */
export function Badge({ variant = "neutral", className, ...rest }: BadgeProps) {
  return (
    <span
      className={cx(
        "rounded-full px-1.5 py-px text-[10px] font-medium",
        VARIANT_CLASSES[variant],
        className,
      )}
      {...rest}
    />
  );
}
