import { Button, Popover, PopoverClose, cn } from "@vitavision/ui";
import { Check, ChevronDown } from "lucide-react";

export interface Sibling {
  id: string;
  label: string;
  /** A quiet second line (the device's id, a project's folder). */
  note?: string | undefined;
}

/** One segment of the breadcrumb: its name, and on click the others of its
 * kind (the device type's other types, the type's other devices) to move
 * to. The current one is marked; choosing one closes the list. */
export function SiblingPopover({
  label,
  title,
  options,
  current,
  onChoose,
  muted = false,
}: {
  label: string;
  /** What the segment is, for its accessible name and tooltip. */
  title: string;
  options: readonly Sibling[];
  current: string | null;
  onChoose: (id: string) => void;
  muted?: boolean;
}) {
  return (
    <Popover
      aria-label={title}
      className="min-w-48 max-w-80"
      trigger={
        <Button size="sm" variant="ghost" aria-label={title} title={title}>
          <span className={cn("max-w-40 truncate", muted ? "text-fg-muted" : "text-fg")}>
            {label}
          </span>
          <ChevronDown aria-hidden className="size-3 text-fg-subtle" />
        </Button>
      }
    >
      <ul className="flex flex-col">
        {options.length === 0 && (
          <li className="px-2 py-1.5 text-xs text-fg-muted">Nothing else here.</li>
        )}
        {options.map((o) => (
          <li key={o.id}>
            <PopoverClose>
              <button
                type="button"
                aria-current={o.id === current ? "true" : undefined}
                onClick={() => onChoose(o.id)}
                className="flex w-full items-center gap-2 rounded-control px-2 py-1.5 text-left outline-none hover:bg-raised focus-visible:bg-raised focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-signal"
              >
                <Check
                  aria-hidden
                  className={cn(
                    "size-3.5 shrink-0",
                    o.id === current ? "text-signal" : "invisible",
                  )}
                />
                <span className="flex min-w-0 flex-col">
                  <span className="truncate text-sm text-fg">{o.label}</span>
                  {o.note && (
                    <span className="truncate font-mono text-[10px] text-fg-muted">
                      {o.note}
                    </span>
                  )}
                </span>
              </button>
            </PopoverClose>
          </li>
        ))}
      </ul>
    </Popover>
  );
}
