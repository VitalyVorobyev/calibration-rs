import { Button, Callout, toast } from "@vitavision/ui";
import { RotateCcw, Save } from "lucide-react";
import { useDraft } from "../../project/draftStore";
import { useReadOnly } from "../../session/jobs";

/** The draft's foot: whether it is saved, Save (`⌘S`; refused while there
 * are errors, a warning does not stop it) and Revert. A project file that
 * changed on disk under the edits says so above it, with what can be done:
 * put my edits back on the disk's version, or keep editing and overwrite it. */
export function SaveBar({
  dirty,
  errors,
  warnings,
  editable,
}: {
  dirty: boolean;
  errors: number;
  warnings: number;
  editable: boolean;
}) {
  const readOnly = useReadOnly();
  const disk = useDraft((s) => s.disk);
  const save = async () => {
    if (await useDraft.getState().save())
      toast({ title: "Dataset saved", tone: "success" });
  };
  return (
    <div className="flex flex-col gap-2 border-t border-line bg-surface p-3">
      {disk && (
        <Callout
          tone="warning"
          title="The project file changed on disk"
          actions={
            <>
              <Button size="sm" onClick={() => useDraft.getState().rebase()}>
                Reload and keep my edits
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => useDraft.getState().keepEditing()}
              >
                Keep editing
              </Button>
            </>
          }
        >
          Another tab or program saved it. Reloading puts your edits back on its version;
          a field changed in both keeps your value and is marked changed. Keeping on means
          your next save replaces it.
        </Callout>
      )}
      <div className="flex items-center gap-2">
        <span
          role="status"
          className="flex min-w-0 items-center gap-2 text-xs text-fg-muted"
          aria-live="polite"
        >
          {!editable ? (
            "Read-only: the manifest is a file of its own"
          ) : dirty ? (
            <>
              <span aria-hidden className="size-2 shrink-0 rounded-full bg-signal" />
              Unsaved changes
            </>
          ) : (
            "Saved"
          )}
          {errors > 0 && (
            <span className="text-defect">
              · {errors} {errors === 1 ? "error" : "errors"}
            </span>
          )}
          {errors === 0 && warnings > 0 && (
            <span className="text-warn">
              · {warnings} {warnings === 1 ? "warning" : "warnings"}
            </span>
          )}
        </span>
        {editable && (
          <span className="ml-auto flex items-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              disabled={!dirty}
              onClick={() => useDraft.getState().revert()}
            >
              <RotateCcw aria-hidden className="size-3.5" />
              Revert
            </Button>
            <Button
              size="sm"
              variant="primary"
              disabled={!dirty || errors > 0 || readOnly}
              title={errors > 0 ? "Fix the errors first" : "Save (⌘S)"}
              onClick={() => void save()}
            >
              <Save aria-hidden className="size-3.5" />
              Save
            </Button>
          </span>
        )}
      </div>
    </div>
  );
}
