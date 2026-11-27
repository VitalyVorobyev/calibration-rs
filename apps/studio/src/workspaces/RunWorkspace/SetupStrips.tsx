import { Button, Callout } from "@vitavision/ui";
import type { ReactNode } from "react";

interface UnresolvedNoticeProps {
  paths: string[];
  onResolve: (path: string) => void;
}

/** The manifest fields the sniffer left blank rather than guess. Run stays
 * blocked until each is filled in and marked resolved. */
export function UnresolvedNotice({ paths, onResolve }: UnresolvedNoticeProps) {
  return (
    <Callout
      tone="error"
      title={`${paths.length} field${paths.length !== 1 ? "s" : ""} need your input before this dataset can run`}
    >
      <p className="text-xs">
        The sniffer left these blank rather than guess. Fill each one in the Manifest form
        below, then mark it resolved.
      </p>
      <ul className="mt-2 flex flex-col gap-1.5">
        {paths.map((path) => (
          <li key={path} className="flex items-start gap-2">
            <code className="mt-0.5 shrink-0 font-mono text-[11px] text-fg">{path}</code>
            <span className="min-w-0 flex-1 text-[11px] text-fg-muted">
              Set it in the Manifest form below.
            </span>
            <Button
              size="sm"
              onClick={() => onResolve(path)}
              className="h-6 shrink-0 px-2 text-[11px]"
              title="Remove this field from _unresolved once you've filled it in"
            >
              mark resolved
            </Button>
          </li>
        ))}
      </ul>
    </Callout>
  );
}

/** The bordered box every collapsible form section (`Disclosure`) sits in. */
export const SECTION_CLASS = "rounded-panel border border-line bg-surface px-3 py-2";

interface SectionSummaryProps {
  title: string;
  /** Short descriptor, shown only while the section is collapsed. */
  summary: string;
  badge?: ReactNode;
}

/** The `summary` line of a form section: the title, an optional badge, and
 * the collapsed-state descriptor (hidden once the `<details>` is open). */
export function SectionSummary({ title, summary, badge }: SectionSummaryProps) {
  return (
    <>
      <span className="text-xs font-semibold tracking-tight text-fg">{title}</span>
      {badge}
      <span className="ml-auto min-w-0 truncate pl-3 font-mono text-[11px] font-normal text-fg-muted group-open:hidden">
        {summary}
      </span>
    </>
  );
}
