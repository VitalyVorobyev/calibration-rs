import { Input, Textarea } from "@vitavision/ui";
import { useState } from "react";
import type { RunInfo } from "../api/generated/types";
import { patchLeaves, pathLabel } from "../project/configEdit";
import { formatDuration } from "../lib/format";
import { useProject } from "../project/projectStore";
import { useRunEditor } from "../project/runEditor";
import {
  differencesFromEditor,
  leafText,
  recordedPatch,
  type RunRow,
} from "../project/runRows";
import { useReadOnly } from "../session/jobs";

/** The chosen run in full: its label and notes (saved when the field is
 * left), the settings it changed from the project's, how they differ from
 * the editor's now, and what its record says of the run. */
export function RunDetails({ run, row }: { run: RunInfo; row: RunRow }) {
  const readOnly = useReadOnly();
  const editorPatch = useRunEditor((s) => s.editorPatch);
  const [label, setLabel] = useState(run.notes.label ?? "");
  const [notes, setNotes] = useState(run.notes.notes ?? "");

  const save = (next: { label?: string; notes?: string }) => {
    const nextLabel = (next.label ?? label).trim();
    const nextNotes = (next.notes ?? notes).trim();
    if (nextLabel === (run.notes.label ?? "") && nextNotes === (run.notes.notes ?? ""))
      return;
    void useProject.getState().patchNotes(run.id, {
      label: nextLabel || null,
      notes: nextNotes || null,
    });
  };

  const patch = recordedPatch(run);
  const leaves = patch ? patchLeaves(patch) : null;
  const differs = patch && editorPatch ? differencesFromEditor(patch, editorPatch) : null;
  const calibrated =
    run.state.kind === "recorded" && run.state.source.kind === "calibrated"
      ? run.state.source
      : null;

  return (
    <div className="flex flex-col gap-3 rounded-control border border-line bg-surface p-2.5">
      <div className="flex flex-col gap-1.5">
        <Input
          aria-label="Run label"
          placeholder="Label"
          value={label}
          disabled={readOnly}
          onChange={(e) => setLabel(e.target.value)}
          onBlur={() => save({ label })}
          onKeyDown={(e) => {
            if (e.key === "Enter") save({ label });
          }}
          className="h-7 text-[11px]"
        />
        <Textarea
          aria-label="Run notes"
          placeholder="Notes"
          minRows={2}
          maxRows={6}
          value={notes}
          disabled={readOnly}
          onChange={(e) => setNotes(e.target.value)}
          onBlur={() => save({ notes })}
          className="text-[11px]"
        />
      </div>

      {row.changes !== null && (
        <section aria-label="Settings of this run" className="flex flex-col gap-1">
          <h3 className="text-[11px] font-semibold text-fg">
            {row.changes.length === 0
              ? "Made with the project's settings"
              : `${row.changes.length} change${row.changes.length === 1 ? "" : "s"} from the project's settings`}
          </h3>
          {row.changes.length > 0 && (
            <ul className="flex flex-col gap-0.5 text-[11px]">
              {row.changes.map((path) => (
                <li key={path} className="flex min-w-0 items-baseline gap-2">
                  <span className="min-w-0 break-words text-fg-muted" title={path}>
                    {pathLabel(path, 3)}
                  </span>
                  <span className="ml-auto shrink-0 font-mono text-fg">
                    {leafText(leaves?.get(path))}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {differs && (
        <section aria-label="This run against the editor" className="flex flex-col gap-1">
          <h3 className="text-[11px] font-semibold text-fg">
            {differs.length === 0
              ? "Same settings as the editor"
              : `${differs.length} differ${differs.length === 1 ? "s" : ""} from the editor`}
          </h3>
          {differs.length > 0 && (
            <ul className="flex flex-col gap-0.5 text-[11px]">
              {differs.slice(0, 8).map((d) => (
                <li key={d.path} className="flex min-w-0 items-baseline gap-2">
                  <span className="truncate text-fg-muted" title={d.path}>
                    {pathLabel(d.path)}
                  </span>
                  <span className="ml-auto flex shrink-0 gap-1 font-mono">
                    <span className="text-fg">{leafText(d.run)}</span>
                    <span aria-hidden className="text-fg-subtle">
                      vs
                    </span>
                    <span className="text-fg-muted">{leafText(d.editor)}</span>
                  </span>
                </li>
              ))}
              {differs.length > 8 && (
                <li className="text-fg-subtle">and {differs.length - 8} more</li>
              )}
            </ul>
          )}
        </section>
      )}

      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
        {calibrated && (
          <>
            <dt className="text-fg-muted">Took</dt>
            <dd className="text-right font-mono text-fg">
              {formatDuration(calibrated.duration_ms)}
            </dd>
            <dt className="text-fg-muted">Left out</dt>
            <dd className="text-right font-mono text-fg">
              {calibrated.left_out.length} image
              {calibrated.left_out.length === 1 ? "" : "s"}
            </dd>
            {calibrated.redetected && (
              <>
                <dt className="text-fg-muted">Detection</dt>
                <dd className="text-right text-fg">detected again</dd>
              </>
            )}
          </>
        )}
        {run.state.kind === "recorded" && (
          <>
            <dt className="text-fg-muted">Made by</dt>
            <dd className="truncate text-right font-mono text-fg" title={run.state.tool}>
              {run.state.tool}
            </dd>
          </>
        )}
      </dl>
    </div>
  );
}
