import { Button, Input } from "@vitavision/ui";
import { FolderOpen } from "lucide-react";
import { relativePath } from "../lib/paths";
import { pickFile, pickFolder } from "../shell/dialogs";

/** A path in a manifest as text, with "Browse…" opening the in-app picker
 * over the back end's folders. With `base`, a path picked under it is
 * written relative to it (a manifest's paths resolve against its data
 * folder); a path elsewhere stays absolute. */
export function PathField({
  value,
  onValueChange,
  onPick,
  kind,
  base,
  disabled,
  placeholder,
  "aria-label": ariaLabel,
}: {
  value: string;
  onValueChange: (value: string) => void;
  /** Called with a path chosen in the picker instead of `onValueChange`,
   * for a field that writes something of it (a folder as a pattern). */
  onPick?: ((path: string) => void) | undefined;
  kind: "file" | "folder";
  base?: string | undefined;
  disabled?: boolean | undefined;
  placeholder?: string | undefined;
  "aria-label": string;
}) {
  const browse = async () => {
    const picked =
      kind === "file" ? await pickFile("Pick a file") : await pickFolder("Pick a folder");
    if (picked === null) return;
    const rel = base !== undefined ? relativePath(base, picked) : picked;
    // A path that leaves the data folder is clearer absolute.
    const path = rel.startsWith("..") ? picked : rel;
    (onPick ?? onValueChange)(path);
  };
  return (
    <div className="flex items-center gap-2">
      <Input
        aria-label={ariaLabel}
        value={value}
        disabled={disabled}
        placeholder={placeholder}
        spellCheck={false}
        className="min-w-0 flex-1 font-mono text-xs"
        onChange={(e) => onValueChange(e.target.value)}
      />
      <Button
        size="sm"
        variant="secondary"
        disabled={disabled}
        onClick={() => void browse()}
        aria-label={`Browse for ${ariaLabel}`}
      >
        <FolderOpen aria-hidden className="size-3.5" />
        Browse…
      </Button>
    </div>
  );
}
