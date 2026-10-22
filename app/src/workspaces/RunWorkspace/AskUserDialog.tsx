import { Button, Dialog, DialogClose, Input } from "@vitavision/ui";
import { useState } from "react";
import { hintFor } from "./manifestFields";

interface AskUserDialogProps {
  field: string;
  prompt: string;
  suggestions: string[];
  /** Apply a chosen value for `field` to the manifest. */
  onApply: (choice: string) => void;
  /** Close without applying. */
  onDismiss: () => void;
}

/** Dialog for a runner `ask_user` event (fail-fast on ambiguity).
 *
 * The dataset runner raises an `ask_user` event when it hits an ambiguity it
 * refuses to guess (e.g. how images pair into views). This surfaces the
 * field + prompt, renders each suggestion as a click-to-apply button, and
 * offers a free-text input for open-ended fields. Applying writes the choice
 * into the manifest (via `applyAskUserChoice`) and dismisses — the user
 * reviews the form and re-runs, staying in control. */
export function AskUserDialog({
  field,
  prompt,
  suggestions,
  onApply,
  onDismiss,
}: AskUserDialogProps) {
  const [freeText, setFreeText] = useState("");
  const hint = hintFor(field);
  const trimmed = freeText.trim();

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onDismiss();
      }}
      title="Input needed"
      description={prompt}
      footer={
        <DialogClose asChild>
          <Button variant="ghost">Dismiss</Button>
        </DialogClose>
      }
    >
      <div className="mt-3 flex flex-col gap-3">
        <code className="font-mono text-[11px] text-fg-muted">{field}</code>
        {hint && hint !== prompt && (
          <p className="text-xs leading-relaxed text-fg-muted">{hint}</p>
        )}

        {suggestions.length > 0 && (
          <div className="flex flex-col gap-1.5">
            <span className="text-xs font-medium text-fg-muted">Choose one:</span>
            <div className="flex flex-wrap gap-1.5">
              {suggestions.map((s) => (
                <Button
                  key={s}
                  size="sm"
                  className="font-mono"
                  onClick={() => onApply(s)}
                >
                  {s}
                </Button>
              ))}
            </div>
          </div>
        )}

        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-fg-muted">Or enter a value:</span>
          <div className="flex gap-1.5">
            <Input
              type="text"
              aria-label={`Value for ${field}`}
              value={freeText}
              onChange={(e) => setFreeText(e.target.value)}
              placeholder={field}
              className="min-w-0 flex-1 font-mono text-xs"
            />
            <Button
              variant="primary"
              disabled={trimmed === ""}
              onClick={() => onApply(trimmed)}
            >
              Apply
            </Button>
          </div>
        </div>
      </div>
    </Dialog>
  );
}
