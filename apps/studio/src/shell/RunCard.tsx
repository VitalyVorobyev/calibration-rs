import {
  Badge,
  Checkbox,
  DropdownMenu,
  MenuItem,
  MenuSeparator,
  cn,
} from "@vitavision/ui";
import {
  Copy,
  ExternalLink,
  Microscope,
  MoreHorizontal,
  Pencil,
  RotateCcw,
  Star,
  Trash2,
} from "lucide-react";
import type { ProjectDigest, RunInfo } from "../api/generated/types";
import { formatCount, formatPixels } from "../lib/format";
import { formatWhen, type RunRow } from "../project/runRows";
import { runPills } from "./runPills";

/** The cameras shown in a card's line of per-camera RMS; the rest are
 * counted (a rig of ten does not make a card ten lines). */
const MAX_CAMERAS = 4;

export interface RunCardActions {
  onSelect: () => void;
  onTick: (ticked: boolean) => void;
  onStar: (starred: boolean) => void;
  onOpenInspect: () => void;
  onRerun: () => void;
  onEdit: () => void;
  onDelete: () => void;
  onCopyId: () => void;
  /** The `report` resource of the run, when it has one. */
  reportUrl: string | null;
}

/** One run in the history: its key numbers at a glance and its actions.
 * The card's own button selects the run (the same choice as the context
 * bar's); the tick feeds Compare; the star marks the run kept. The tick,
 * the star, the verdict and the menu sit over the button's corners, so the
 * button holds the run's text alone. */
export function RunCard({
  run,
  row,
  digest,
  projectPatch,
  selected,
  ticked,
  lowest,
  readOnly,
  actions,
}: {
  run: RunInfo;
  row: RunRow;
  digest: ProjectDigest | null | undefined;
  projectPatch: unknown;
  selected: boolean;
  ticked: boolean;
  lowest: boolean;
  readOnly: boolean;
  actions: RunCardActions;
}) {
  const pills = runPills(run, digest, projectPatch);
  const shown = row.cameras.slice(0, MAX_CAMERAS);
  const hidden = row.cameras.length - shown.length;
  const canRerun = row.changes !== null;
  const staleParts = row.stale ?? [];

  return (
    <div
      data-selected={selected ? "" : undefined}
      className={cn(
        "relative rounded-control border transition-colors",
        selected ? "border-signal/50 bg-raised" : "border-transparent hover:bg-raised/60",
      )}
    >
      <div className="absolute top-1.5 left-1.5 flex items-center gap-1">
        <Checkbox
          aria-label={`Compare ${row.id}`}
          checked={ticked}
          onCheckedChange={actions.onTick}
        />
        <button
          type="button"
          aria-pressed={row.starred}
          aria-label={row.starred ? `Unstar ${row.id}` : `Star ${row.id}`}
          title={
            row.starred
              ? "Starred: a run to keep. Click to unstar"
              : "Star as a run to keep"
          }
          disabled={readOnly || row.state === "Unreadable"}
          onClick={() => actions.onStar(!row.starred)}
          className="grid size-5 place-items-center rounded-control text-fg-subtle hover:text-fg focus-visible:outline-2 focus-visible:outline-signal disabled:opacity-40"
        >
          <Star
            aria-hidden
            className={cn("size-3.5", row.starred && "fill-current text-warn")}
          />
        </button>
      </div>

      <div className="absolute top-1 right-1 flex items-center gap-0.5">
        {pills.verdict && <Badge tone={pills.verdict.tone}>{pills.verdict.label}</Badge>}
        <DropdownMenu
          align="end"
          trigger={
            <button
              type="button"
              aria-label={`Actions for ${row.id}`}
              className="grid size-6 place-items-center rounded-control text-fg-muted hover:bg-raised hover:text-fg focus-visible:outline-2 focus-visible:outline-signal"
            >
              <MoreHorizontal aria-hidden className="size-4" />
            </button>
          }
        >
          <MenuItem icon={<Microscope />} onSelect={actions.onOpenInspect}>
            Open in Inspect
          </MenuItem>
          <MenuItem
            icon={<RotateCcw />}
            disabled={!canRerun || readOnly}
            onSelect={actions.onRerun}
          >
            {canRerun ? "Rerun from this run" : "Rerun (no record of its settings)"}
          </MenuItem>
          <MenuItem icon={<Pencil />} onSelect={actions.onEdit}>
            Edit label and notes
          </MenuItem>
          {actions.reportUrl && (
            <MenuItem
              icon={<ExternalLink />}
              onSelect={() => window.open(actions.reportUrl!, "_blank", "noreferrer")}
            >
              Open report
            </MenuItem>
          )}
          <MenuItem icon={<Copy />} onSelect={actions.onCopyId}>
            Copy run id
          </MenuItem>
          <MenuSeparator />
          <MenuItem
            icon={<Trash2 />}
            tone="defect"
            disabled={readOnly}
            onSelect={actions.onDelete}
          >
            Delete…
          </MenuItem>
        </DropdownMenu>
      </div>

      <button
        type="button"
        onClick={actions.onSelect}
        aria-current={selected ? "true" : undefined}
        aria-label={`Run ${row.id}${row.label ? `, ${row.label}` : ""}`}
        className="flex w-full min-w-0 flex-col gap-0.5 rounded-control py-1.5 pr-2 pl-[3.9rem] text-left focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal"
      >
        <span className="flex min-w-0 items-baseline gap-1.5 pr-[4.6rem]">
          <span className="shrink-0 font-mono text-xs font-medium text-fg">{row.id}</span>
          {row.label && (
            <span className="truncate text-xs text-fg-muted" title={row.label}>
              {row.label}
            </span>
          )}
        </span>
        {row.state === "Unreadable" ? (
          <span className="text-[11px] text-defect">{row.problem ?? "Unreadable"}</span>
        ) : (
          <>
            <span className="flex flex-wrap items-baseline gap-x-2 text-[11px] text-fg-muted">
              <span className="font-mono text-xs font-medium text-fg tabular-nums">
                {row.rms === null ? "no RMS" : formatPixels(row.rms)}
              </span>
              {row.views && (
                <span
                  className="whitespace-nowrap tabular-nums"
                  title="views used of the views the dataset formed"
                >
                  {row.views.used}/{row.views.total} views
                </span>
              )}
              {row.features !== null && (
                <span className="whitespace-nowrap tabular-nums">
                  {formatCount(row.features)} features
                </span>
              )}
            </span>
            {shown.length > 1 && (
              <span
                className="flex flex-wrap gap-x-2 font-mono text-[10px] text-fg-subtle tabular-nums"
                aria-label="RMS per camera"
              >
                {shown.map((c) => (
                  <span key={c.id}>
                    {c.id} {c.rms === null ? "–" : c.rms.toFixed(3)}
                  </span>
                ))}
                {hidden > 0 && <span>+{hidden}</span>}
              </span>
            )}
            <span className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-[11px] text-fg-subtle">
              {row.changes !== null && (
                <span
                  title={
                    row.changes.length > 0
                      ? `Differs from the project's configuration: ${row.changes.join(", ")}`
                      : "Made with the project's configuration"
                  }
                >
                  {row.changes.length === 0
                    ? "project settings"
                    : `${row.changes.length} change${row.changes.length === 1 ? "" : "s"}`}
                </span>
              )}
              {row.timestamp !== null && (
                <span className="whitespace-nowrap">· {formatWhen(row.timestamp)}</span>
              )}
            </span>
          </>
        )}
        {(lowest || row.state !== "Recorded" || staleParts.length > 0) && (
          <span className="mt-0.5 flex flex-wrap items-center gap-1">
            {lowest && (
              <span title="The lowest reprojection RMS among the runs whose verdict is not Fail. A lower error is not a better model by itself: star the run you accept.">
                <Badge tone="info">Lowest RMS</Badge>
              </span>
            )}
            {row.state !== "Recorded" && (
              <span title={pills.state.title}>
                <Badge tone={pills.state.tone}>{pills.state.label}</Badge>
              </span>
            )}
            {staleParts.length > 0 && (
              <span
                title={`The project's ${staleParts.join(", ")} changed since this run was made.`}
              >
                <Badge>Stale</Badge>
              </span>
            )}
          </span>
        )}
      </button>
    </div>
  );
}
