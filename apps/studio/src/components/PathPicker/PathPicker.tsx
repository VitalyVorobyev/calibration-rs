/** The shell's path picker: a dialog over the back end's folder listing
 * (`list_dir`), on the desktop and in the browser alike. It starts at the
 * root folders and never leaves them; what it offers depends on the pick
 * (`pickRules.ts`). Mounted once by the shell; `requestPath` opens it. */
import { Button, Dialog, Empty, ErrorBox, Listbox } from "@vitavision/ui";
import { useEffect, useState } from "react";
import { api } from "../../api";
import type { DirEntry, DirListing, Show } from "../../api/generated/types";
import { AddRootButton } from "../../shell/AddRootButton";
import { usePathPicker } from "./pickerStore";
import {
  breadcrumb,
  MODE_NOUN,
  selectable,
  showsFiles,
  type PickMode,
} from "./pickRules";

/** The folder the picker showed last, reopened next time. */
let lastDir: string | null = null;

export function PathPickerHost() {
  const request = usePathPicker((s) => s.request);
  const answer = usePathPicker((s) => s.answer);
  if (request === null) return null;
  return (
    <PathPicker
      key={`${request.mode}:${request.title}`}
      title={request.title}
      mode={request.mode}
      onAnswer={answer}
    />
  );
}

function PathPicker({
  title,
  mode,
  onAnswer,
}: {
  title: string;
  mode: PickMode;
  onAnswer: (path: string | null) => void;
}) {
  const [dir, setDir] = useState<string | null>(lastDir);
  const [listing, setListing] = useState<DirListing | null>(null);
  const [roots, setRoots] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    let live = true;
    const show: Show = showsFiles(mode) ? "all" : "dirs";
    const args = dir === null ? { show } : { path: dir, show };
    api.listDir(args).then(
      (next) => {
        if (!live) return;
        setListing(next);
        setSelected(null);
        setError(null);
        if (next.path === null) setRoots(next.entries.map((e) => e.path));
        lastDir = next.path ?? null;
      },
      (e: unknown) => {
        if (!live) return;
        // A folder that went away (or out of reach): back to the roots.
        setError(String(e));
        if (dir !== null) setDir(null);
      },
    );
    return () => {
      live = false;
    };
  }, [dir, mode, reload]);

  // The roots, for the breadcrumb, even when the picker opens deeper.
  useEffect(() => {
    let live = true;
    api.listDir({ show: "dirs" }).then(
      (r) => {
        if (live) setRoots(r.entries.map((e) => e.path));
      },
      () => {},
    );
    return () => {
      live = false;
    };
  }, [reload]);

  const entries = listing?.entries ?? [];
  const chosen = entries.find((e) => e.path === selected) ?? null;
  const open = (entry: DirEntry) => {
    if (entry.kind === "dir") setDir(entry.path);
  };
  const choice =
    chosen !== null && selectable(chosen, mode)
      ? chosen.path
      : mode === "folder" && chosen === null && dir !== null
        ? dir
        : null;

  return (
    <Dialog
      open
      onOpenChange={(isOpen) => {
        if (!isOpen) onAnswer(null);
      }}
      title={title}
      description={`Choose ${MODE_NOUN[mode]} inside the folders this back end serves.`}
      className="w-[min(40rem,92vw)]"
      footer={
        <>
          <Button variant="ghost" onClick={() => onAnswer(null)}>
            Cancel
          </Button>
          <Button
            disabled={chosen === null || chosen.kind !== "dir"}
            onClick={() => chosen && open(chosen)}
          >
            Open folder
          </Button>
          <Button
            variant="primary"
            disabled={choice === null}
            onClick={() => onAnswer(choice)}
          >
            {mode === "folder" && chosen === null ? "Choose this folder" : "Choose"}
          </Button>
        </>
      }
    >
      <nav aria-label="Folder" className="mb-2 flex flex-wrap items-center gap-1">
        {breadcrumb(listing?.path ?? null, roots).map((crumb, i, all) => (
          <Button
            key={crumb.path ?? "roots"}
            size="sm"
            variant="ghost"
            disabled={i === all.length - 1}
            onClick={() => setDir(crumb.path)}
            className="max-w-[16rem] truncate font-mono text-[11px]"
          >
            {crumb.label}
          </Button>
        ))}
      </nav>
      {error && <ErrorBox className="mb-2">{error}</ErrorBox>}
      {listing?.path === null && entries.length === 0 ? (
        <Empty action={<AddRootButton onAdded={() => setReload((n) => n + 1)} />}>
          No folders yet: Calibration Studio reads only the folders you add.
        </Empty>
      ) : (
        <div
          className="max-h-[50vh] overflow-y-auto rounded-control border border-line"
          // On the whole row, not its text: the first click of a double click
          // has selected the entry under the pointer.
          onDoubleClick={(event) => {
            if (!(event.target as HTMLElement).closest('[role="option"]')) return;
            if (chosen === null) return;
            if (selectable(chosen, mode) && chosen.kind === "file") onAnswer(chosen.path);
            else open(chosen);
          }}
        >
          <Listbox
            aria-label="Entries"
            value={selected}
            onValueChange={setSelected}
            options={entries.map((e) => ({
              value: e.path,
              label: e.kind === "dir" ? `${e.name}/` : e.name,
              description: e.is_project ? "project" : undefined,
              disabled: e.kind === "file" && !selectable(e, mode),
            }))}
            renderOption={(option) => (
              <span className="flex items-center justify-between gap-3 font-mono text-[12px]">
                <span className="truncate">{option.label}</span>
                {option.description && (
                  <span className="text-[10px] text-fg-muted uppercase">
                    {option.description}
                  </span>
                )}
              </span>
            )}
          />
          {listing?.truncated && (
            <p className="px-3 py-2 text-[11px] text-fg-muted">
              Only the first 5 000 entries are listed.
            </p>
          )}
        </div>
      )}
    </Dialog>
  );
}
