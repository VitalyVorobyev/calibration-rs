import { Input, Select, ToggleChip } from "@vitavision/ui";
import { Search } from "lucide-react";
import type { ProblemKind, Verdict } from "../../api/generated/types";
import { exportKindLabel } from "../../store/exportKind";
import { isFiltering, NO_FILTER, type LibraryFilter } from "./libraryModel";

const CHIPS: { value: Verdict | "none"; label: string }[] = [
  { value: "pass", label: "Pass" },
  { value: "warn", label: "Check" },
  { value: "fail", label: "Fail" },
  { value: "none", label: "No run" },
];

const ALL = "__all__";

/** Search and filters of the Library: one text box (name, device type and
 * id, label, workflow, run notes), the verdicts as chips and the workflow
 * as a select of the ones the library has. */
export function LibraryToolbar({
  filter,
  onChange,
  topologies,
}: {
  filter: LibraryFilter;
  onChange: (next: LibraryFilter) => void;
  /** The workflows the scan has projects of. */
  topologies: readonly ProblemKind[];
}) {
  const toggle = (value: Verdict | "none", on: boolean) => {
    const verdicts = new Set(filter.verdicts);
    if (on) verdicts.add(value);
    else verdicts.delete(value);
    onChange({ ...filter, verdicts });
  };
  return (
    <div
      role="search"
      aria-label="Find projects"
      className="flex flex-wrap items-center gap-2"
    >
      <div className="relative min-w-56 flex-1 sm:max-w-sm">
        <Search
          aria-hidden
          className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-fg-subtle"
        />
        <Input
          type="search"
          aria-label="Search projects"
          placeholder="Search projects, devices, labels…"
          value={filter.query}
          className="pl-8"
          onChange={(e) => onChange({ ...filter, query: e.target.value })}
        />
      </div>
      <div role="group" aria-label="Verdict" className="flex items-center gap-1">
        {CHIPS.map((chip) => (
          <ToggleChip
            key={chip.value}
            size="sm"
            checked={filter.verdicts.has(chip.value)}
            onCheckedChange={(on) => toggle(chip.value, on)}
          >
            {chip.label}
          </ToggleChip>
        ))}
      </div>
      <Select
        aria-label="Workflow"
        className="w-52"
        value={filter.topology ?? ALL}
        onValueChange={(v) =>
          onChange({ ...filter, topology: v === ALL ? null : (v as ProblemKind) })
        }
        options={[
          { value: ALL, label: "All workflows" },
          ...topologies.map((t) => ({ value: t, label: exportKindLabel(t) })),
        ]}
      />
      {isFiltering(filter) && (
        <button
          type="button"
          onClick={() => onChange(NO_FILTER)}
          className="rounded-control px-1.5 py-1 text-xs text-fg-muted underline-offset-2 hover:text-fg hover:underline focus-visible:outline-2 focus-visible:outline-signal"
        >
          Clear
        </button>
      )}
    </div>
  );
}
