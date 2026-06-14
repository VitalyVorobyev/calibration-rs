/** Modal for a runner `ask_user` event (ADR 0019 fail-fast).
 *
 * The dataset runner raises an `ask_user` event when it hits an ambiguity it
 * refuses to guess (e.g. how images pair into views). This modal surfaces the
 * field + prompt, renders each suggestion as a click-to-apply button, and
 * offers a free-text input for open-ended fields. Applying writes the choice
 * into the manifest (via `applyAskUserChoice`) and dismisses — the user
 * reviews the form and re-runs, staying in control. */
import { useState } from "react";

import { hintFor } from "./manifestFields";

interface AskUserModalProps {
  field: string;
  prompt: string;
  suggestions: string[];
  /** Apply a chosen value for `field` to the manifest. */
  onApply: (choice: string) => void;
  /** Close without applying. */
  onDismiss: () => void;
}

export function AskUserModal({ field, prompt, suggestions, onApply, onDismiss }: AskUserModalProps) {
  const [freeText, setFreeText] = useState("");
  const hint = hintFor(field);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="dialog"
      aria-modal="true"
      aria-label={`Input needed for ${field}`}
      onClick={onDismiss}
    >
      <div
        className="w-full max-w-md rounded-lg border border-border bg-bg p-4 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-2">
          <span
            className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[12px] font-bold"
            style={{
              color: "var(--brand)",
              backgroundColor: "color-mix(in srgb, var(--brand) 16%, transparent)",
            }}
            aria-hidden="true"
          >
            ?
          </span>
          <div className="min-w-0">
            <h3 className="text-[13px] font-semibold text-foreground">Input needed</h3>
            <code className="font-mono text-[11px] text-muted-foreground">{field}</code>
          </div>
        </div>

        <p className="mt-3 text-[12px] leading-relaxed text-foreground">{prompt}</p>
        {hint && hint !== prompt && (
          <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">{hint}</p>
        )}

        {suggestions.length > 0 && (
          <div className="mt-3 flex flex-col gap-1.5">
            <span className="text-[11px] font-medium text-muted-foreground">Choose one:</span>
            <div className="flex flex-wrap gap-1.5">
              {suggestions.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => onApply(s)}
                  className="rounded-md border border-brand/40 bg-brand/[0.06] px-2.5 py-1 font-mono text-[11px] text-foreground transition-colors hover:bg-brand/[0.14]"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="mt-3 flex flex-col gap-1.5">
          <span className="text-[11px] font-medium text-muted-foreground">
            Or enter a value:
          </span>
          <div className="flex gap-1.5">
            <input
              type="text"
              value={freeText}
              onChange={(e) => setFreeText(e.target.value)}
              placeholder={field}
              className="min-w-0 flex-1 rounded-md border border-border bg-bg-soft px-2 py-1 font-mono text-[11px] text-foreground outline-none focus:border-brand/60"
            />
            <button
              type="button"
              disabled={freeText.trim() === ""}
              onClick={() => onApply(freeText.trim())}
              className={[
                "shrink-0 rounded-md px-3 py-1 text-[11px] font-semibold transition-colors",
                freeText.trim() === ""
                  ? "cursor-not-allowed border border-border bg-bg-soft text-muted-foreground"
                  : "bg-brand text-white hover:opacity-90",
              ].join(" ")}
              style={freeText.trim() === "" ? undefined : { backgroundColor: "var(--brand)" }}
            >
              Apply
            </button>
          </div>
        </div>

        <div className="mt-4 flex justify-end">
          <button
            type="button"
            onClick={onDismiss}
            className="rounded-md border border-border bg-bg px-3 py-1 text-[11px] text-muted-foreground transition-colors hover:text-foreground"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}
