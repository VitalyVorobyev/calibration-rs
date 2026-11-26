/** Settings: the root folders — the only folders the back end reads and
 * writes, where the folder browser starts. The desktop adds them through
 * the native dialog and removes them here; a server's are fixed by its
 * command line and only listed. */
import { Badge, Button, Dialog, Empty } from "@vitavision/ui";
import { X } from "lucide-react";
import { useState } from "react";
import { api } from "../api";
import { AddRootButton } from "./AddRootButton";
import { useSession } from "./session";

export function SettingsDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const status = useSession((s) => s.status);
  const setInfo = useSession((s) => s.setInfo);
  const [error, setError] = useState<string | null>(null);
  const info = status.kind === "ready" ? status.info : null;

  const remove = async (path: string) => {
    try {
      setInfo(await api.removeRoot({ path }));
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Settings"
      description={
        info?.roots_editable
          ? "The folders Calibration Studio may read and write. It reads nothing else."
          : "The folders this server serves, fixed when it starts (its --root flags)."
      }
      footer={<AddRootButton />}
      className="w-[min(36rem,92vw)]"
    >
      {info === null ? (
        <Empty>Not connected to a back end.</Empty>
      ) : (
        <ul aria-label="Folders" className="flex flex-col gap-1">
          {info.roots.map((root) => (
            <li
              key={root.path}
              className="flex items-center justify-between gap-3 rounded-control border border-line px-3 py-1.5"
            >
              <span className="truncate font-mono text-[12px]" title={root.path}>
                {root.path}
              </span>
              {root.samples ? (
                <Badge tone="info">samples</Badge>
              ) : (
                info.roots_editable && (
                  <Button
                    size="sm"
                    variant="ghost"
                    aria-label={`Remove ${root.path}`}
                    onClick={() => void remove(root.path)}
                  >
                    <X aria-hidden className="size-3.5" />
                  </Button>
                )
              )}
            </li>
          ))}
        </ul>
      )}
      {error && <p className="mt-2 text-[11px] text-defect">{error}</p>}
      {info && <p className="mt-3 text-[11px] text-fg-muted">Version {info.version}</p>}
    </Dialog>
  );
}
