import { Button, Callout, ConfirmDialog, Input, toast } from "@vitavision/ui";
import { FolderOpen } from "lucide-react";
import { useState } from "react";
import { api, asStudioError } from "../../api";
import type { SeedFrom, SeedSource } from "../../api/generated/types";
import { useProject } from "../../project/projectStore";
import { useReadOnly } from "../../session/jobs";
import { pickFolder } from "../../shell/dialogs";
import { describeSource, exportDest, runSeedSource } from "./seedModel";

/** Writing the seed: to a file (JSON or TOML, asking before it replaces
 * one), and as the open project's `[seed]`. All of it waits while the
 * connection is lost. */
export function SeedActions({ from }: { from: SeedFrom | null }) {
  const readOnly = useReadOnly();
  const project = useProject((s) => s.project);
  const [folder, setFolder] = useState<string | null>(project?.dir ?? null);
  const [name, setName] = useState("seed.json");
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [exported, setExported] = useState<{ path: string; of: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [seedError, setSeedError] = useState<string | null>(null);

  const fromKey = JSON.stringify(from);
  const dest = exportDest(folder, name);
  const file = exported !== null && exported.of === fromKey ? exported.path : null;
  const direct = from ? runSeedSource(from) : null;

  const choose = async () => {
    const picked = await pickFolder("Choose the folder to write the seed to");
    if (picked !== null) setFolder(picked);
  };

  const write = async (overwrite: boolean) => {
    if (from === null || "problem" in dest) return;
    setBusy(true);
    setError(null);
    try {
      const done = await api.exportSeed({ from, dest: dest.dest, overwrite });
      setExported({ path: done.path, of: fromKey });
      setConfirm(false);
    } catch (e) {
      const err = asStudioError(e);
      if (err.kind === "file_exists") setConfirm(true);
      else setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const setSeed = async (seed: SeedSource | null) => {
    if (!project) return;
    setBusy(true);
    setSeedError(null);
    try {
      const view = await api.setProjectSeed({
        dir: project.dir,
        revision: project.revision,
        seed,
      });
      useProject.setState({ project: view, error: null });
      toast({
        title: seed ? "Seed set" : "Seed cleared",
        description: project.name,
        tone: "success",
      });
    } catch (e) {
      setSeedError(asStudioError(e).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <section aria-label="Write to a file" className="flex flex-col gap-2">
        <h3 className="text-sm font-semibold text-fg">Write to a file</h3>
        <div className="flex flex-wrap items-end gap-2">
          <div className="flex min-w-0 flex-col gap-1 text-xs text-fg-muted">
            Folder
            <div className="flex items-center gap-2">
              <span
                className="max-w-[22rem] truncate rounded-control border border-line bg-surface px-2 py-1.5 font-mono text-xs text-fg"
                title={folder ?? undefined}
              >
                {folder ?? "none chosen"}
              </span>
              <Button size="sm" variant="secondary" onClick={() => void choose()}>
                <FolderOpen aria-hidden className="size-3.5" />
                Choose…
              </Button>
            </div>
          </div>
          <label className="flex flex-col gap-1 text-xs text-fg-muted">
            File name
            <Input
              aria-label="File name"
              className="w-48 font-mono"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <Button
            variant="primary"
            disabled={readOnly || busy || from === null || "problem" in dest}
            onClick={() => void write(false)}
          >
            Export
          </Button>
        </div>
        <p className="text-[11px] text-fg-muted">
          {"problem" in dest
            ? dest.problem
            : `Writes ${dest.format.toUpperCase()}; the extension picks the format.`}
        </p>
        {error && <Callout tone="error">{error}</Callout>}
        {file && (
          <Callout tone="success" title="Seed written">
            <span className="font-mono text-xs break-all">{file}</span>
          </Callout>
        )}
      </section>

      <section aria-label="This project's seed" className="flex flex-col gap-2">
        <h3 className="text-sm font-semibold text-fg">This project&apos;s seed</h3>
        {project ? (
          <>
            <p className="text-xs text-fg-muted">
              <span className="text-fg">{project.name}</span> starts its calibrations
              from: <span className="font-mono">{describeSource(project.spec.seed)}</span>
            </p>
            <div className="flex flex-wrap gap-2">
              {direct && (
                <Button
                  variant="secondary"
                  disabled={readOnly || busy}
                  onClick={() => void setSeed(direct)}
                >
                  Use this run as the project&apos;s seed
                </Button>
              )}
              {file && (
                <Button
                  variant="secondary"
                  disabled={readOnly || busy}
                  onClick={() => void setSeed({ file })}
                >
                  Use the exported file as the project&apos;s seed
                </Button>
              )}
              <Button
                variant="ghost"
                disabled={readOnly || busy || !project.spec.seed}
                onClick={() => void setSeed(null)}
              >
                Clear seed
              </Button>
            </div>
            {from !== null && direct === null && file === null && (
              <p className="text-[11px] text-fg-muted">
                A seed made from several runs is a file: write it first, then use the
                file.
              </p>
            )}
            {seedError && (
              <Callout tone="error" title="The project cannot take this seed">
                {seedError}
              </Callout>
            )}
          </>
        ) : (
          <p className="text-xs text-fg-muted">Open a project to use a seed for it.</p>
        )}
      </section>

      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title="Replace the file?"
        description={
          <>
            <span className="font-mono break-all">
              {"problem" in dest ? "" : dest.dest}
            </span>{" "}
            already exists. Replacing it cannot be undone.
          </>
        }
        confirmLabel="Replace"
        destructive
        loading={busy}
        onConfirm={() => void write(true)}
      />
    </div>
  );
}
