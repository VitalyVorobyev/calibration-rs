import { Textarea } from "@vitavision/ui";
import { useEffect, useMemo, useRef } from "react";

interface SetupJsonEditorProps {
  manifest: unknown;
  config: unknown;
  /** Called on blur with whichever top-level keys parsed. */
  onApply: (next: { manifest?: unknown; config?: unknown }) => void;
}

/** The merged `{ manifest, config }` as editable JSON, applied on blur.
 *
 * It sits in a `Disclosure` (a native `<details>`), which keeps it mounted
 * while collapsed, so an uncontrolled `defaultValue` would go stale after a
 * form edit. It re-syncs whenever the setup changes — except while it has
 * focus, so an edit in progress (applied on blur) is not clobbered. */
export function SetupJsonEditor({ manifest, config, onApply }: SetupJsonEditorProps) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const merged = useMemo(
    () => JSON.stringify({ manifest, config }, null, 2),
    [manifest, config],
  );

  useEffect(() => {
    const editor = ref.current;
    if (editor && editor.ownerDocument.activeElement !== editor) editor.value = merged;
  }, [merged]);

  const handleBlur = (text: string) => {
    try {
      const parsed = JSON.parse(text) as Record<string, unknown>;
      onApply({
        ...(parsed.manifest != null ? { manifest: parsed.manifest } : {}),
        ...(parsed.config != null ? { config: parsed.config } : {}),
      });
    } catch {
      // Keep the previous values: a half-typed edit is not an error yet.
    }
  };

  return (
    <>
      <Textarea
        ref={ref}
        mono
        aria-label="Merged manifest and config JSON"
        defaultValue={merged}
        onBlur={(e) => handleBlur(e.target.value)}
        rows={18}
        spellCheck={false}
        className="resize-y px-3 py-2 text-[11px] leading-relaxed"
      />
      <p className="mt-1.5 text-[11px] text-fg-muted">
        Edits applied on blur. Both <code className="font-mono">manifest</code> and{" "}
        <code className="font-mono">config</code> keys are required at the top level.
      </p>
    </>
  );
}
