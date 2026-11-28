import { Badge, Button, Input, toast } from "@vitavision/ui";
import { ArrowLeft, ArrowRight, Crown, Star, X } from "lucide-react";
import { useState } from "react";
import type { RunEvidence, RunInfo } from "../../api/generated/types";
import { formatCount, formatPixels } from "../../lib/format";
import { runVerdict } from "../../project/projectModel";
import { useProject } from "../../project/projectStore";
import { useCompareSet } from "../../session/compareSet";
import { VERDICT } from "../../shell/runPills";
import { pooledRms, positionName } from "./compareModel";

/** The chosen runs, in order: which is the reference (every difference is
 * against it), each run's own verdict and error, its star and label, and
 * where it stands in the set. A run is starred here as in the run list;
 * "Keep" stars it and unstars the others of the set. */
export function RunSet({
  ids,
  runs,
  evidence,
  readOnly,
}: {
  ids: string[];
  runs: RunInfo[];
  /** The comparison's evidence for each id, in the same order, once there. */
  evidence: RunEvidence[] | undefined;
  readOnly: boolean;
}) {
  return (
    <ol
      aria-label="Runs compared"
      className="grid grid-cols-[repeat(auto-fill,minmax(17rem,1fr))] gap-2"
    >
      {ids.map((id, i) => (
        <li key={id} className="min-w-0">
          <RunTile
            id={id}
            index={i}
            count={ids.length}
            run={runs.find((r) => r.id === id) ?? null}
            evidence={evidence?.[i] ?? null}
            others={ids.filter((o) => o !== id)}
            readOnly={readOnly}
          />
        </li>
      ))}
    </ol>
  );
}

function RunTile({
  id,
  index,
  count,
  run,
  evidence,
  others,
  readOnly,
}: {
  id: string;
  index: number;
  count: number;
  run: RunInfo | null;
  evidence: RunEvidence | null;
  others: string[];
  readOnly: boolean;
}) {
  const starred = run?.notes.starred ?? false;
  const saved = run?.notes.label ?? "";
  const [label, setLabel] = useState(saved);
  const [seen, setSeen] = useState(saved);
  // A label saved elsewhere (the run list) replaces the one being typed.
  if (seen !== saved) {
    setSeen(saved);
    setLabel(saved);
  }
  const reference = index === 0;
  const verdict = (run ? runVerdict(run) : null) ?? evidence?.verdict ?? null;
  const rms = evidence ? pooledRms(evidence) : null;
  const views = evidence
    ? evidence.cameras.reduce((n, c) => Math.max(n, c.views), 0)
    : null;
  const features = evidence ? evidence.cameras.reduce((n, c) => n + c.features, 0) : null;
  const set = useCompareSet.getState();
  const store = useProject.getState();

  const saveLabel = () => {
    const next = label.trim();
    if (next === saved) return;
    void store.patchNotes(id, { label: next || null });
  };
  const keep = async () => {
    await store.patchNotes(id, { starred: true });
    const unstar = others.filter(
      (o) => useProject.getState().project?.runs.find((r) => r.id === o)?.notes.starred,
    );
    await Promise.all(unstar.map((o) => store.patchNotes(o, { starred: false })));
    toast({
      title:
        unstar.length > 0
          ? `${id} starred; ${unstar.join(", ")} unstarred.`
          : `${id} starred.`,
    });
  };

  return (
    <div
      className={
        "flex h-full flex-col gap-2 rounded-control border bg-surface p-2.5 " +
        (reference ? "border-signal" : "border-line")
      }
    >
      <div className="flex items-center gap-1.5">
        {reference ? (
          <Badge tone="info">
            <Crown aria-hidden className="mr-1 size-3" />
            Reference
          </Badge>
        ) : (
          <Badge>{positionName(index)}</Badge>
        )}
        <span className="min-w-0 flex-1 truncate font-mono text-sm text-fg">{id}</span>
        <button
          type="button"
          aria-pressed={starred}
          aria-label={starred ? `Unstar ${id}` : `Star ${id}`}
          title={
            starred ? "Starred: a run to keep. Click to unstar" : "Star as a run to keep"
          }
          disabled={readOnly || run === null}
          onClick={() => void store.patchNotes(id, { starred: !starred })}
          className="grid size-6 place-items-center rounded-control text-fg-subtle hover:text-fg focus-visible:outline-2 focus-visible:outline-signal disabled:opacity-40"
        >
          <Star
            aria-hidden
            className={starred ? "size-4 fill-current text-warn" : "size-4"}
          />
        </button>
      </div>

      <Input
        aria-label={`Label of ${id}`}
        placeholder="Label"
        value={label}
        disabled={readOnly || run === null}
        onChange={(e) => setLabel(e.target.value)}
        onBlur={saveLabel}
        onKeyDown={(e) => e.key === "Enter" && saveLabel()}
      />

      <p className="flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-xs text-fg-muted">
        {verdict ? (
          <span title="The quality verdict recorded with the run">
            <Badge tone={VERDICT[verdict].tone}>{VERDICT[verdict].label}</Badge>
          </span>
        ) : null}
        <span className="whitespace-nowrap text-fg">
          {rms === null ? "–" : formatPixels(rms)}
        </span>
        <span className="whitespace-nowrap">
          {views === null ? "–" : `${formatCount(views)} views`}
        </span>
        <span className="whitespace-nowrap">
          {features === null ? "–" : `${formatCount(features)} features`}
        </span>
      </p>

      <div className="mt-auto flex flex-wrap items-center gap-1">
        <Button
          size="sm"
          variant="ghost"
          aria-label={`Move ${id} earlier`}
          title="Earlier in the set"
          disabled={index === 0}
          onClick={() => set.move(id, -1)}
        >
          <ArrowLeft aria-hidden className="size-3.5" />
        </Button>
        <Button
          size="sm"
          variant="ghost"
          aria-label={`Move ${id} later`}
          title="Later in the set"
          disabled={index === count - 1}
          onClick={() => set.move(id, 1)}
        >
          <ArrowRight aria-hidden className="size-3.5" />
        </Button>
        {!reference && (
          <Button
            size="sm"
            variant="ghost"
            title="Compare every other run against this one"
            onClick={() => set.move(id, -index)}
          >
            Make reference
          </Button>
        )}
        <Button
          size="sm"
          variant="secondary"
          disabled={readOnly || run === null}
          title="Star this run and unstar the others of the set"
          onClick={() => void keep()}
        >
          Keep
        </Button>
        <Button
          size="sm"
          variant="ghost"
          aria-label={`Remove ${id} from the comparison`}
          title="Remove from the comparison"
          onClick={() => set.remove(id)}
        >
          <X aria-hidden className="size-3.5" />
        </Button>
      </div>
    </div>
  );
}
