import { Button, ErrorBox, Field, Input, Panel } from "@vitavision/ui";
import { useState, type FormEvent } from "react";
import { Logo } from "../components/Logo";
import { useSession } from "../session/session";

/** What the page shows instead of the app while the server wants its
 * token: one centred panel, no other chrome. The token field is a password
 * field; a wrong token is said under it and the field keeps what was
 * typed. */
export function LoginPanel({ error }: { error: string | null }) {
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
    <main className="grid min-h-dvh place-items-center bg-ground p-6 text-fg">
      <div className="flex w-full max-w-sm flex-col gap-5">
        <div className="flex items-center gap-3">
          <Logo size={30} className="text-fg" />
          <div className="flex flex-col leading-tight">
            <h1 className="text-lg font-semibold tracking-tight">Calibration Studio</h1>
            <span className="font-mono text-xs text-fg-muted">
              {window.location.host}
            </span>
          </div>
        </div>
        <Panel title="Sign in" bodyClassName="flex flex-col gap-4">
          <form
            aria-label="Sign in"
            onSubmit={(e) => void submit(e)}
            className="flex flex-col gap-4"
          >
            <Field label="Token">
              <Input
                type="password"
                autoComplete="off"
                autoFocus
                spellCheck={false}
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
              Sign in
            </Button>
          </form>
          <p className="text-xs text-fg-muted">
            The token is printed by
            <code className="mt-1 block font-mono text-fg">
              calibration-studio-server token
            </code>
          </p>
        </Panel>
      </div>
    </main>
  );
}
