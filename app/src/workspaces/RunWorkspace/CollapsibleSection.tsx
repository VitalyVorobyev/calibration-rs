/** Reusable collapsible section for the Run workspace.
 *
 * The header bar is always visible and acts as the toggle trigger. The
 * body is conditionally rendered (not just hidden) to avoid paying the
 * render cost of ConfigForm when collapsed — the schema-driven form tree
 * is moderately expensive on first mount.
 *
 * Props
 * -----
 * title       — section heading (left-aligned in the header bar)
 * summary     — optional short descriptor shown when collapsed (replaces
 *               the expand prompt so the user knows what's inside)
 * defaultOpen — initial open state; the component is uncontrolled by
 *               default so parent doesn't need to track per-section state
 * children    — section body (any React node)
 * badge       — optional text rendered as a small badge after the title
 */
import { useState } from "react";

interface CollapsibleSectionProps {
  title: string;
  summary?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
  badge?: string;
}

export function CollapsibleSection({
  title,
  summary,
  defaultOpen = false,
  children,
  badge,
}: CollapsibleSectionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <section className="rounded-md border border-border overflow-hidden">
      {/* Header bar — always visible, click to toggle */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={[
          "flex w-full items-center justify-between gap-3 px-3 py-2.5",
          "text-left transition-colors",
          open ? "bg-bg-soft" : "bg-bg hover:bg-bg-soft",
        ].join(" ")}
        aria-expanded={open}
      >
        <div className="flex min-w-0 items-center gap-2">
          <ChevronIcon open={open} />
          <span className="text-[12px] font-semibold tracking-tight text-foreground">
            {title}
          </span>
          {badge && (
            <span className="rounded-full bg-brand/[0.12] px-1.5 py-px text-[10px] font-medium text-brand">
              {badge}
            </span>
          )}
        </div>

        {!open && summary && (
          <span className="truncate font-mono text-[11px] text-muted-foreground">
            {summary}
          </span>
        )}
      </button>

      {/* Body — unmounted when collapsed */}
      {open && (
        <div className="border-t border-border bg-bg p-3">
          {children}
        </div>
      )}
    </section>
  );
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 12 12"
      fill="none"
      aria-hidden="true"
      className={[
        "shrink-0 text-muted-foreground transition-transform duration-150",
        open ? "rotate-90" : "rotate-0",
      ].join(" ")}
    >
      <path
        d="M4 2.5L7.5 6L4 9.5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
