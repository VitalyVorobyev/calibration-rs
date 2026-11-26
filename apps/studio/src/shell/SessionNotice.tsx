/** What the shell shows of the session above the workspace: the token field
 * when the server wants it, the connection error when no back end answers,
 * and on the desktop's first run the "Add folder" prompt. */
import { Button, Callout, ErrorBox, Field, Input } from "@vitavision/ui";
import { useState, type FormEvent } from "react";
import { AddRootButton } from "./AddRootButton";
import { userRoots, useSession } from "./session";

export function SessionNotice() {
  const status = useSession((s) => s.status);
  const refresh = useSession((s) => s.refresh);

  if (status.kind === "login") return <LoginForm error={status.error} />;
  if (status.kind === "offline") {
    return (
      <Callout
        tone="error"
        title="No back end answers"
        className="mb-2"
        actions={
          <Button size="sm" onClick={() => void refresh()}>
            Retry
          </Button>
        }
      >
        {status.message}. Start the desktop app with `bun run tauri dev`, or a server with
        `bun run serve -- --root DIR` (Vite proxies `/api` to it).
      </Callout>
    );
  }
  if (
    status.kind === "ready" &&
    status.info.roots_editable &&
    userRoots(status.info).length === 0
  ) {
    return (
      <Callout title="Add a folder to start" className="mb-2" actions={<AddRootButton />}>
        Calibration Studio reads only the folders you add: your datasets and projects. Add
        one now, or later in Settings.
      </Callout>
    );
  }
  return null;
}

function LoginForm({ error }: { error: string | null }) {
  const submitToken = useSession((s) => s.submitToken);
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    await submitToken(token);
    setBusy(false);
  };

  return (
    <form
      aria-label="Log in"
      onSubmit={(e) => void submit(e)}
      className="mb-2 flex max-w-md flex-col gap-2 rounded-panel border border-line bg-surface p-4"
    >
      <p className="text-[12px] text-fg-muted">
        This server needs its token: open the address it printed when it started, or paste
        the token (`calibration-studio-server token` prints it again).
      </p>
      <Field label="Token">
        <Input
          type="password"
          autoComplete="off"
          value={token}
          onChange={(e) => setToken(e.target.value)}
        />
      </Field>
      {error && <ErrorBox>{error}</ErrorBox>}
      <Button
        type="submit"
        variant="primary"
        loading={busy}
        disabled={token.trim() === ""}
      >
        Log in
      </Button>
    </form>
  );
}
