import { Button, toast } from "@vitavision/ui";
import { Copy } from "lucide-react";
import { patchToml } from "../../project/patchToml";

/** The patch from the library defaults that a run will record, as TOML:
 * what `vcal run --config` would take, read-only, with a copy button. */
export function PatchView({ patch }: { patch: Record<string, unknown> | null }) {
  if (patch === null) {
    return (
      <p className="text-xs text-fg-muted">
        No patch while the configuration does not check.
      </p>
    );
  }
  const lines = patchToml(patch);
  const text = lines.join("\n");
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs text-fg-muted">
          The changes from the library defaults that the run records
          {lines.length === 0 ? ": none, the defaults" : ""}.
        </p>
        <Button
          size="sm"
          variant="ghost"
          disabled={lines.length === 0}
          onClick={() => {
            void navigator.clipboard?.writeText(text).then(
              () => toast({ title: "Patch copied", tone: "success" }),
              () => toast({ title: "The patch could not be copied", tone: "error" }),
            );
          }}
        >
          <Copy aria-hidden className="size-3.5" />
          Copy
        </Button>
      </div>
      {lines.length > 0 && (
        <pre
          aria-label="Resulting patch"
          className="max-h-64 overflow-auto rounded-control border border-line bg-ground px-3 py-2 font-mono text-[11px] leading-relaxed text-fg"
        >
          {text}
        </pre>
      )}
    </div>
  );
}
