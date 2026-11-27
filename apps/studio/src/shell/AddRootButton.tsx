/** "Add folder…": the desktop's one act of granting a folder — the native
 * dialog, then `add_root`. Shown only where the roots can change and a
 * native dialog exists (the desktop app); a server's roots are its own. */
import { Button } from "@vitavision/ui";
import { FolderPlus } from "lucide-react";
import { useState } from "react";
import { api, hasNativeFolderDialog, pickNativeFolder } from "../api";
import { useSession } from "./session";

export function AddRootButton({ onAdded }: { onAdded?: () => void }) {
  const editable = useSession(
    (s) => s.status.kind === "ready" && s.status.info.roots_editable,
  );
  const setInfo = useSession((s) => s.setInfo);
  const [error, setError] = useState<string | null>(null);
  if (!editable || !hasNativeFolderDialog()) return null;

  const add = async () => {
    try {
      const path = await pickNativeFolder();
      if (path === null) return;
      setInfo(await api.addRoot({ path }));
      setError(null);
      onAdded?.();
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <span className="flex flex-col items-start gap-1">
      <Button variant="primary" size="sm" onClick={() => void add()}>
        <FolderPlus aria-hidden className="size-3.5" />
        Add folder…
      </Button>
      {error && <span className="text-[11px] text-defect">{error}</span>}
    </span>
  );
}
