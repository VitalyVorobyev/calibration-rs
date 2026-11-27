import { StatusBar } from "@vitavision/workbench";
import { isDirty, useDraft } from "../project/draftStore";
import { useProject } from "../project/projectStore";
import { useJobs } from "../session/jobs";
import { useSession } from "../session/session";

/** The foot of the window: what the app talks to (the desktop, or a server
 * and whether it took the token), whether the open project has unsaved
 * edits, the connection, and the app's version. */
export function StatusLine() {
  const status = useSession((s) => s.status);
  const project = useProject((s) => s.project);
  const dirty = useDraft(isDirty);
  const connection = useJobs((s) => s.connection);
  const info = status.kind === "ready" ? status.info : null;

  const transport = info
    ? info.shell === "desktop"
      ? "Desktop"
      : `Server ${window.location.origin}`
    : status.kind === "login"
      ? `Server ${window.location.origin}`
      : null;

  return (
    <StatusBar
      live
      start={[
        { label: "transport", value: transport },
        {
          label: "session",
          value: info ? (info.shell === "desktop" ? "local" : "token ok") : null,
        },
      ]}
      end={[
        {
          label: "connection",
          value:
            status.kind === "ready"
              ? connection === "lost"
                ? "reconnecting"
                : connection === "open"
                  ? "connected"
                  : "connecting"
              : null,
        },
        { label: "version", value: info?.version ?? null },
      ]}
    >
      <span className="ml-3 border-l border-line pl-3 font-mono text-xs text-fg-muted">
        {project ? (dirty ? "unsaved edits" : "project saved") : "no project"}
      </span>
    </StatusBar>
  );
}
