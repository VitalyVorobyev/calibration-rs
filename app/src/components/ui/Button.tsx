import { forwardRef } from "react";
import type { ButtonHTMLAttributes } from "react";
import { cx } from "./cx";

export type ButtonVariant = "default" | "primary";
export type ButtonSize = "sm" | "md" | "icon";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** `default` — bordered, neutral (the global `button` base in index.css).
   * `primary` — brand-filled CTA (Run, Apply). Only one exists because
   * that's the only two the app actually uses; see the ui/README note in
   * app/README.md before adding a third. */
  variant?: ButtonVariant;
  /** `sm` (h-7, toolbar default) · `md` (h-9, page-level CTA) · `icon`
   * (h-7 square, e.g. zoom −/+, theme toggle, steppers). */
  size?: ButtonSize;
  /** Toggle/pressed visual state (brand border + text) and `aria-pressed`
   * wiring for on/off toolbar toggles (Compare, Linked, SGM, …). Omit for
   * plain action buttons — passing `false` still emits `aria-pressed`,
   * which is correct for a toggle sitting at rest but wrong for a button
   * that was never a toggle. */
  pressed?: boolean;
}

const SIZE_CLASSES: Record<ButtonSize, string> = {
  sm: "h-7 px-2 font-mono text-[11px]",
  md: "h-9 px-3 text-[13px] font-medium",
  icon: "grid h-7 w-7 place-items-center !p-0 font-mono text-xs",
};

/** Shared button primitive. Sizing/variant deltas are layered on top of
 * the global `button { … }` base rule in index.css (border, background,
 * hover, disabled, focus-visible) rather than re-declaring it — see the
 * design note in app/README.md. */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = "default",
    size = "sm",
    pressed,
    disabled,
    className,
    type = "button",
    ...rest
  },
  ref,
) {
  const variantClass =
    variant === "primary"
      ? disabled
        ? "cursor-not-allowed border-border bg-bg-soft text-muted-foreground"
        : "border-brand bg-brand text-white hover:opacity-90"
      : cx("disabled:cursor-not-allowed", pressed && "border-brand text-brand");

  return (
    <button
      ref={ref}
      type={type}
      disabled={disabled}
      aria-pressed={pressed}
      className={cx(SIZE_CLASSES[size], variantClass, className)}
      {...rest}
    />
  );
});
