import { Button, ConfirmDialog, Input, ToggleChip } from "@vitavision/ui";
import { ExternalLink, Star, Trash2 } from "lucide-react";
import { useState } from "react";
import { api } from "../api";
import type { ProjectView, RunInfo } from "../api/generated/types";
import { staleParts } from "../project/projectModel";
import { useProject } from "../project/projectStore";

/** The chosen run's notes (label, star), its report page, whether it was
 * made from the project's current inputs, and its deletion. */
export function RunActions({ project, run }: { project: ProjectView; run: RunInfo }) {
  const [label, setLabel] = useState(run.notes.label ?? "");
  const [confirm, setConfirm] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const stale = staleParts(run, project.input_digest);

  const saveNotes = (next: { label?: string; starred?: boolean }) => {
    const notes = {
      ...run.notes,
      label: (next.label ?? label).trim() || null,
      starred: next.starred ?? run.notes.starred ?? false,
    };
    void useProject.getState().setNotes(run.id, notes);
  };

  const remove = async () => {
    setDeleting(true);
    await useProject.getState().deleteRun(run.id);
    setDeleting(false);
    setConfirm(false);
  };

  const hasReport =
    run.state.kind === "recorded" && run.state.source.kind === "calibrated";

  return (
    <div className="flex flex-col gap-2 rounded-control border border-line p-2">
      <div className="flex items-center gap-1.5">
        <Input
          aria-label="Run label"
          placeholder="Label"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          onBlur={() => {
            if (label.trim() !== (run.notes.label ?? "")) saveNotes({ label });
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") saveNotes({ label });
          }}
          className="h-7 text-[11px]"
        />
        <ToggleChip
          title="Starred: a run to keep"
          checked={run.notes.starred ?? false}
          onCheckedChange={(starred) => saveNotes({ starred })}
        >
          <Star aria-hidden className="size-3.5" />
        </ToggleChip>
      </div>
      {stale && stale.length > 0 && (
        <p className="text-[11px] text-fg-muted">
          Made from other inputs than the project's now: {stale.join(", ")} changed.
        </p>
      )}
      <div className="flex items-center gap-1.5">
        {hasReport && (
          <Button asChild size="sm" variant="ghost">
            <a
              href={api.urls.report({ run: { dir: project.dir, id: run.id } })}
              target="_blank"
              rel="noreferrer"
            >
              <ExternalLink aria-hidden className="size-3.5" />
              Report
            </a>
          </Button>
        )}
        <Button
          size="sm"
          variant="ghost"
          className="ml-auto"
          onClick={() => setConfirm(true)}
        >
          <Trash2 aria-hidden className="size-3.5" />
          Delete run
        </Button>
      </div>
      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title={`Delete ${run.id}?`}
        description={`The run folder ${run.dir} is deleted with its export, quality report and notes.`}
        confirmLabel="Delete"
        destructive
        loading={deleting}
        onConfirm={() => void remove()}
      />
    </div>
  );
}
