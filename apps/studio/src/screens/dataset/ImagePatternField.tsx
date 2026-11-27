import { SegmentedControl, Textarea } from "@vitavision/ui";
import { PathField } from "../../components/PathField";
import type { ImagePattern } from "../../api/generated/types";
import type { PatternMatch } from "./datasetModel";

/** Where a camera's images are: a glob pattern (with the folder picker
 * writing `<folder>/*.<ext>`) or an explicit list of paths, one a line.
 * Under it, what the pattern matches now, from the layout the library
 * paired: the file count and the first and last file, or that nothing
 * matched. Paths are relative to the data folder `dataRoot`. */
export function ImagePatternField({
  value,
  onChange,
  match,
  dataRoot,
  disabled,
  label,
}: {
  value: ImagePattern;
  onChange: (next: ImagePattern) => void;
  /** What it matches; `null` while the layout is not known yet, `undefined`
   * for a source the layout does not list (laser frames). */
  match?: PatternMatch | null | undefined;
  dataRoot: string;
  disabled: boolean;
  label: string;
}) {
  const kind = value.kind;
  return (
    <div className="flex flex-col gap-1.5">
      <SegmentedControl
        aria-label={`${label}: how the images are named`}
        value={kind}
        disabled={disabled}
        onValueChange={(k) =>
          onChange(
            k === "list"
              ? { kind: "list", paths: value.kind === "list" ? value.paths : [] }
              : { kind: "glob", pattern: value.kind === "glob" ? value.pattern : "" },
          )
        }
        options={[
          { value: "glob", label: "Pattern" },
          { value: "list", label: "List" },
        ]}
      />
      {value.kind === "glob" ? (
        <PathField
          aria-label={`${label} pattern`}
          kind="folder"
          base={dataRoot}
          value={value.pattern}
          disabled={disabled}
          placeholder="cam0/*.png"
          onValueChange={(pattern) => onChange({ kind: "glob", pattern })}
          // A folder picked is every image in it, in the extension the
          // pattern had (png when it had none).
          onPick={(folder) => {
            const ext = /\.[A-Za-z0-9]+$/.exec(value.pattern)?.[0] ?? ".png";
            const base = folder === "." ? "" : `${folder.replace(/\/$/, "")}/`;
            onChange({ kind: "glob", pattern: `${base}*${ext}` });
          }}
        />
      ) : (
        <Textarea
          aria-label={`${label} list`}
          rows={3}
          disabled={disabled}
          spellCheck={false}
          className="font-mono text-xs"
          value={value.paths.join("\n")}
          placeholder={"cam0/0001.png\ncam0/0002.png"}
          onChange={(e) =>
            onChange({
              kind: "list",
              paths: e.target.value.split("\n").map((p) => p.trim()),
            })
          }
        />
      )}
      {match !== undefined && <MatchLine match={match} />}
    </div>
  );
}

/** "12 files · first cam0/0001.png · last cam0/0012.png". */
export function MatchLine({ match }: { match: PatternMatch | null }) {
  if (match === null) {
    return <p className="text-[11px] text-fg-subtle">Looking for the files…</p>;
  }
  // Nothing matched: the warning on the pattern says so; no second line.
  if (match.files === 0) return null;
  return (
    <p className="truncate font-mono text-[11px] text-fg-muted" role="status">
      <span className="text-fg">{match.files}</span>{" "}
      {match.files === 1 ? "file" : "files"}
      {match.first && <> · first {match.first}</>}
      {match.last && match.last !== match.first && <> · last {match.last}</>}
    </p>
  );
}
