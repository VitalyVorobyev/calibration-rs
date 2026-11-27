import { Button, Callout, ConfirmDialog } from "@vitavision/ui";
import { useState } from "react";
import type { ProjectView } from "../../api/generated/types";
import { useProject } from "../../project/projectStore";
import {
  describeDifferences,
  exclusionDifferences,
  runExclusions,
} from "../../project/rerun";
import { useRunEditor } from "../../project/runEditor";
import { useReadOnly } from "../../session/jobs";

/** What the editor says when it was loaded from a run: whose settings it
 * holds, and, when the project's exclusions are no longer that run's, the
 * choice between restoring them (a project edit, confirmed) and keeping the
 * current ones (the rerun is then not like for like, and the run will be
 * marked stale). */
export function RerunNotice({ project }: { project: ProjectView }) {
  const fromRun = useRunEditor((s) => s.fromRun);
  const readOnly = useReadOnly();
  const [keptFor, setKeptFor] = useState<string | null>(null);
  const [confirm, setConfirm] = useState(false);
  const [saving, setSaving] = useState(false);
  const run = project.runs.find((r) => r.id === fromRun);
  if (!fromRun || !run) return null;

  const theirs = runExclusions(run);
  const diffs = exclusionDifferences(theirs, project.spec.exclude);
  const differs = theirs !== null && diffs.length > 0 && keptFor !== fromRun;

  const restore = async () => {
    if (!theirs) return;
    setSaving(true);
    await useProject.getState().save({ ...project.spec, exclude: theirs });
    setSaving(false);
    setConfirm(false);
  };

  return (
    <>
      <Callout tone="info" title={`Started from the settings of ${fromRun}`}>
        <span className="text-xs">
          The editor holds that run's recorded configuration, and compares with it. The
          project's own configuration is untouched.
        </span>
      </Callout>
      {differs && (
        <Callout
          tone="warning"
          title="The project's exclusions are not those of that run"
          actions={
            <>
              <Button
                size="sm"
                variant="secondary"
                disabled={readOnly}
                onClick={() => setConfirm(true)}
              >
                Restore its exclusions
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setKeptFor(fromRun)}>
                Keep current exclusions
              </Button>
            </>
          }
        >
          <span className="text-xs">
            {describeDifferences(diffs)}. Rerunning with the current ones is not a
            like-for-like rerun, and the run will be marked stale against the old.
          </span>
        </Callout>
      )}
      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title="Restore the exclusions of that run?"
        description={`The project's exclusions become those of ${fromRun}, written to the project file (${describeDifferences(diffs)}).`}
        confirmLabel="Restore"
        loading={saving}
        onConfirm={() => void restore()}
      />
    </>
  );
}
