import {
  Button,
  Callout,
  Dialog,
  ErrorBox,
  Field,
  Input,
  Section,
  Select,
  Textarea,
} from "@vitavision/ui";
import { FileInput, FolderOpen } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router";
import { api, asStudioError } from "../../api";
import type { RunInfo } from "../../api/generated/types";
import { useProjectDialog } from "../../project/dialogs";
import { useProject } from "../../project/projectStore";
import { useLibraryScan } from "../../project/useLibraryScan";
import { useReadOnly } from "../../session/jobs";
import { userRoots, useSession } from "../../session/session";
import { pickExportFile, pickFolder } from "../../shell/dialogs";
import { placePath } from "../../shell/screens";
import { exportKindLabel } from "../../store/exportKind";
import { relativeToRoots } from "./libraryModel";

/** Import an export calibrated elsewhere (an export file, or the folder a
 * `vcal run -o` wrote) as the next run of a project: which file, which
 * project, an optional label. The library refuses an export of another
 * workflow or camera count, and the reason is said here with what was
 * entered kept, so another project can be chosen. The result is a run in
 * the `Imported` state, opened in Inspect. */
export function ImportDialog({ dir }: { dir?: string }) {
  const close = useProjectDialog((s) => s.close);
  const navigate = useNavigate();
  const readOnly = useReadOnly();
  const scan = useLibraryScan();
  const status = useSession((s) => s.status);
  const roots = useMemo(
    () => (status.kind === "ready" ? userRoots(status.info) : []),
    [status],
  );
  const open = useProject((s) => s.project?.dir ?? null);

  const [file, setFile] = useState("");
  const [target, setTarget] = useState(dir ?? open ?? "");
  const [label, setLabel] = useState("");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<{ run: RunInfo; dir: string } | null>(null);

  const projects = useMemo(
    () =>
      (scan.data?.projects ?? []).map((p) => ({
        value: p.dir,
        label: p.name,
        note: `${p.topology ? exportKindLabel(p.topology) : "manifest unreadable"} · ${relativeToRoots(p.dir, roots)}`,
      })),
    [scan.data, roots],
  );

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const notesBody = {
        ...(label.trim() ? { label: label.trim() } : {}),
        ...(notes.trim() ? { notes: notes.trim() } : {}),
      };
      const imported = await api.importExport({
        dir: target,
        export: file,
        notes: notesBody,
      });
      setDone({ run: imported, dir: target });
      await useProject.getState().open(target);
    } catch (e) {
      const err = asStudioError(e);
      // The library's sentence says why (workflow, camera count); another
      // project can be chosen without losing the rest.
      setError(
        err.kind === "import_mismatch"
          ? `This export does not belong to that project. ${err.message}`
          : err.message,
      );
    } finally {
      setBusy(false);
    }
  };

  const toInspect = async () => {
    if (!done) return;
    await useProject.getState().openRun(done.run.id);
    close();
    void navigate(
      placePath({
        screen: "inspect",
        tab: "frames",
        project: done.dir,
        run: done.run.id,
      }),
    );
  };

  return (
    <Dialog
      open
      onOpenChange={(o) => {
        if (!o) close();
      }}
      title="Import an export"
      description="Add a calibration made elsewhere as a run of a project. It is read, not recalculated."
      className="w-[min(36rem,calc(100vw-2rem))]"
      footer={
        done ? (
          <>
            <Button variant="ghost" onClick={close}>
              Close
            </Button>
            <Button variant="primary" onClick={() => void toInspect()}>
              Open in Inspect
            </Button>
          </>
        ) : (
          <>
            <Button variant="ghost" onClick={close}>
              Cancel
            </Button>
            <Button
              variant="primary"
              disabled={readOnly || busy || file.trim() === "" || target === ""}
              onClick={() => void run()}
            >
              {busy ? "Importing…" : "Import"}
            </Button>
          </>
        )
      }
    >
      {done ? (
        <div className="mt-4">
          <Callout tone="success" title={`Imported as ${done.run.id}`}>
            The run is in the project, marked Imported: it has no record of the inputs it
            was calibrated from, so it is never called stale.
          </Callout>
        </div>
      ) : (
        <div className="mt-4 flex flex-col gap-5">
          <Section step={1} title="The export">
            <div className="flex flex-col gap-2">
              <Input
                aria-label="Export file or run folder"
                value={file}
                spellCheck={false}
                placeholder="calibration.json, export.json, or a folder"
                className="font-mono text-xs"
                onChange={(e) => setFile(e.target.value)}
              />
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => void pickExportFile().then((p) => p && setFile(p))}
                >
                  <FileInput aria-hidden className="size-3.5" />
                  Choose a file…
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() =>
                    void pickFolder("Pick a vcal run folder").then((p) => p && setFile(p))
                  }
                >
                  <FolderOpen aria-hidden className="size-3.5" />
                  Choose a run folder…
                </Button>
              </div>
            </div>
          </Section>
          <Section
            step={2}
            title="The project"
            hint="Its workflow and camera count must match the export's."
          >
            <Select
              aria-label="Project"
              value={target}
              onValueChange={setTarget}
              placeholder={
                projects.length === 0 ? "No projects in the library" : "Choose a project…"
              }
              options={projects}
            />
          </Section>
          <Section step={3} title="Label" hint="Optional.">
            <div className="flex flex-col gap-2">
              <Input
                aria-label="Run label"
                value={label}
                placeholder="vendor tool, 2026-11"
                onChange={(e) => setLabel(e.target.value)}
              />
              <Field label="Notes">
                <Textarea
                  value={notes}
                  rows={2}
                  onChange={(e) => setNotes(e.target.value)}
                />
              </Field>
            </div>
          </Section>
          {error && <ErrorBox>{error}</ErrorBox>}
        </div>
      )}
    </Dialog>
  );
}
