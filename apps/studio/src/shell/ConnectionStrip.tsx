import { Button, Callout } from "@vitavision/ui";
import { Loader } from "lucide-react";
import { useJobs } from "../session/jobs";
import { useSession } from "../session/session";

/** The strip under the header that says the back end is not there: it
 * never answered (`No back end answers`, with Retry), or the connection
 * dropped mid-session (`Reconnecting…` with when it was last heard). It
 * clears itself when the stream is back. The screens under it stay
 * readable; what writes is disabled by `useReadOnly`. */
export function ConnectionStrip() {
  const status = useSession((s) => s.status);
  const refresh = useSession((s) => s.refresh);
  const lost = useJobs((s) => s.connection === "lost");
  const lastEventAt = useJobs((s) => s.lastEventAt);

  if (status.kind === "offline") {
    return (
      <Callout
        tone="error"
        title="No back end answers"
        className="mx-3 mt-1 mb-2"
        actions={
          <Button size="sm" onClick={() => void refresh()}>
            Retry
          </Button>
        }
      >
        {status.message}. Start the desktop app with{" "}
        <code className="font-mono">bun run tauri dev</code>, or a server with{" "}
        <code className="font-mono">bun run serve -- --root DIR</code>.
      </Callout>
    );
  }
  if (status.kind === "ready" && lost) {
    return (
      <Callout tone="warning" className="mx-3 mt-1 mb-2">
        <span className="inline-flex flex-wrap items-center gap-x-2">
          <strong className="font-semibold text-fg">Connection lost.</strong>
          <Loader
            aria-hidden
            className="size-3.5 animate-spin motion-reduce:animate-none"
          />
          Reconnecting… changes are paused until the server answers.
          <span className="font-mono text-[11px] text-fg-muted">
            {lastEventAt === null
              ? "nothing heard yet"
              : `last heard ${new Date(lastEventAt).toLocaleTimeString()}`}
          </span>
        </span>
      </Callout>
    );
  }
  return null;
}
