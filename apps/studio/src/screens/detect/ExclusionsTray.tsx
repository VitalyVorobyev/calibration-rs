import { Button, Dialog, Input, Table, type Column } from "@vitavision/ui";
import { Ban, ListChecks, Undo2 } from "lucide-react";
import { useMemo, useState } from "react";
import type { Exclusions } from "../../api/generated/types";
import { useReadOnly } from "../../session/jobs";
import {
  countExclusions,
  describeCounts,
  totalExcluded,
  withNote,
  without,
} from "./exclusions";
import { changeExclusions, undoExclusion, useExclusionHistory } from "./useExclusions";

type Scope = "views" | "images" | "features";

interface Row {
  scope: Scope;
  /** Index in its list of the project's `[exclude]`. */
  index: number;
  what: string;
  camera: string;
  image: string;
  points: number | null;
  note: string;
}

function rowsOf(exclude: Exclusions): Row[] {
  return [
    ...(exclude.views ?? []).map((e, index): Row => ({
      scope: "views",
      index,
      what: "View",
      camera: "all",
      image: "token" in e.view ? e.view.token : e.view.image,
      points: null,
      note: e.note ?? "",
    })),
    ...(exclude.images ?? []).map((e, index): Row => ({
      scope: "images",
      index,
      what: "Image",
      camera: e.camera,
      image: e.image,
      points: null,
      note: e.note ?? "",
    })),
    ...(exclude.features ?? []).map((e, index): Row => ({
      scope: "features",
      index,
      what: "Features",
      camera: e.camera,
      image: e.image,
      points: e.points.length,
      note: e.note ?? "",
    })),
  ];
}

/** The foot of the stage: what the project leaves out, Undo, and the
 * panel that lists every exclusion with its note and Restore. */
export function ExclusionsTray({ exclude }: { exclude: Exclusions }) {
  const readOnly = useReadOnly();
  const [open, setOpen] = useState(false);
  const counts = countExclusions(exclude);
  const undoable = useExclusionHistory((s) => s.before.length > 0);

  return (
    <div
      role="region"
      aria-label="Exclusions"
      className="flex items-center gap-2 rounded-control border border-line bg-surface px-2.5 py-1.5"
    >
      <Button size="sm" variant="ghost" onClick={() => setOpen(true)}>
        <ListChecks aria-hidden className="size-3.5" />
        Exclusions
        <span className="font-mono text-[11px] text-fg-muted">
          {describeCounts(counts)}
        </span>
      </Button>
      <Button
        size="sm"
        variant="ghost"
        className="ml-auto"
        disabled={readOnly || !undoable}
        title="Undo the last exclusion (⌘Z)"
        onClick={() => void undoExclusion()}
      >
        <Undo2 aria-hidden className="size-3.5" />
        Undo
      </Button>
      <ExclusionsPanel open={open} onOpenChange={setOpen} exclude={exclude} />
    </div>
  );
}

function ExclusionsPanel({
  open,
  onOpenChange,
  exclude,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  exclude: Exclusions;
}) {
  const readOnly = useReadOnly();
  const rows = useMemo(() => rowsOf(exclude), [exclude]);
  const columns = useMemo(() => exclusionColumns(readOnly), [readOnly]);

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Exclusions"
      description="What the project leaves out of every run. Features are named by their board point, so they stay excluded when the image is detected again."
      className="w-[min(56rem,94vw)]"
      footer={
        <Button
          size="sm"
          variant="secondary"
          disabled={readOnly || totalExcluded(countExclusions(exclude)) === 0}
          onClick={() => void changeExclusions(() => ({}))}
        >
          Restore all
        </Button>
      }
    >
      <Table
        caption="Exclusions of the project"
        columns={columns}
        rows={rows}
        rowKey={(r) => `${r.scope}${r.index}`}
        empty={
          <span className="flex items-center gap-2 py-6 text-xs text-fg-muted">
            <Ban aria-hidden className="size-4" />
            Nothing is excluded. Press X on an image, ⇧X on a view, or pick features on
            the image.
          </span>
        }
      />
    </Dialog>
  );
}

/** The panel's columns: scope, camera, image or view, points, an editable
 * note, and Restore. */
function exclusionColumns(readOnly: boolean): Column<Row>[] {
  return [
    { key: "what", header: "Scope", width: "5.5rem", cell: (r) => r.what },
    {
      key: "camera",
      header: "Camera",
      width: "4.5rem",
      cell: (r) => <span className="font-mono">{r.camera}</span>,
    },
    {
      key: "image",
      header: "Image or view",
      cell: (r) => (
        <span className="font-mono break-all" title={r.image}>
          {r.image}
        </span>
      ),
    },
    {
      key: "points",
      header: "Points",
      numeric: true,
      width: "4rem",
      cell: (r) => (r.points === null ? "–" : r.points),
    },
    {
      key: "note",
      header: "Why",
      width: "11rem",
      cell: (r) => (
        <NoteField key={`${r.scope}${r.index}${r.note}`} row={r} disabled={readOnly} />
      ),
    },
    {
      key: "restore",
      header: <span className="sr-only">Restore</span>,
      width: "5rem",
      cell: (r) => (
        <Button
          size="sm"
          variant="ghost"
          disabled={readOnly}
          aria-label={`Restore ${r.what.toLowerCase()} ${r.image}`}
          onClick={() => void changeExclusions((e) => without(e, r.scope, r.index))}
        >
          Restore
        </Button>
      ),
    },
  ];
}

/** A note, saved when the field loses focus. */
function NoteField({ row, disabled }: { row: Row; disabled: boolean }) {
  const [text, setText] = useState(row.note);
  return (
    <Input
      aria-label={`Why ${row.what.toLowerCase()} ${row.image} is excluded`}
      placeholder="Why…"
      value={text}
      disabled={disabled}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => {
        if (text.trim() !== row.note)
          void changeExclusions((e) => withNote(e, row.scope, row.index, text), {
            quiet: true,
          });
      }}
    />
  );
}
