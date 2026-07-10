import { cx } from "./cx";

export interface SectionHeaderProps {
  title: string;
  /** Right-aligned mono caption (e.g. "cam 2 → pose 5"). */
  subtitle?: string;
  className?: string;
}

/** Small uppercase subsection heading with an optional right-aligned
 * subtitle, underlined by a hairline border. Used inside `Panel`s
 * (Diagnose's stats rail, the 3D viewer's info rail) — this was two
 * byte-for-byte-duplicated `PanelHeading` functions before this pass. */
export function SectionHeader({ title, subtitle, className }: SectionHeaderProps) {
  return (
    <header
      className={cx(
        "flex items-baseline justify-between border-b border-border pb-1",
        className,
      )}
    >
      <h3 className="text-[11px] font-semibold uppercase tracking-wider text-foreground">
        {title}
      </h3>
      {subtitle && (
        <span className="font-mono text-[10px] text-muted-foreground">{subtitle}</span>
      )}
    </header>
  );
}
