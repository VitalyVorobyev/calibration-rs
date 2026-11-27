/** Settings: the root folders (the only folders the back end reads and
 * writes, where the folder browser starts), the theme, what the app is
 * talking to, and the shortcuts. The desktop adds roots through the native
 * dialog and removes them here; a server's are fixed by its command line
 * and only listed, with the flag named. */
import {
  Badge,
  Button,
  Dialog,
  Empty,
  SegmentedControl,
  ShortcutList,
  readThemeChoice,
  setThemeChoice,
  type ThemeChoice,
} from "@vitavision/ui";
import { X } from "lucide-react";
import { useState, useSyncExternalStore, type ReactNode } from "react";
import { api } from "../api";
import { THEME_STORAGE_KEY } from "../lib/theme";
import { useReadOnly } from "../session/jobs";
import { useSession } from "../session/session";
import { useUi } from "../session/ui";
import { AddRootButton } from "./AddRootButton";
import { GLOBAL_SHORTCUTS } from "./shortcuts";

const THEMES: { value: ThemeChoice; label: string }[] = [
  { value: "system", label: "System" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
];

/** The stored theme choice, re-read when another tab changes it. */
function useThemeChoice(): [ThemeChoice, (choice: ThemeChoice) => void] {
  const subscribe = (onChange: () => void) => {
    window.addEventListener("storage", onChange);
    return () => window.removeEventListener("storage", onChange);
  };
  const choice = useSyncExternalStore<ThemeChoice>(
    subscribe,
    () => readThemeChoice(THEME_STORAGE_KEY),
    () => "system",
  );
  const [, bump] = useState(0);
  return [
    choice,
    (next) => {
      setThemeChoice(next, THEME_STORAGE_KEY);
      bump((n) => n + 1);
    },
  ];
}

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section aria-label={title} className="flex flex-col gap-2">
      <h3 className="text-[10px] font-semibold tracking-wider text-fg-muted uppercase">
        {title}
      </h3>
      {children}
    </section>
  );
}

export function SettingsDialog() {
  const open = useUi((s) => s.dialog === "settings");
  const close = useUi((s) => s.close);
  const status = useSession((s) => s.status);
  const setInfo = useSession((s) => s.setInfo);
  const readOnly = useReadOnly();
  const [error, setError] = useState<string | null>(null);
  const [theme, setTheme] = useThemeChoice();
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
      onOpenChange={(next) => !next && close()}
      title="Settings"
      className="w-[min(38rem,92vw)]"
    >
      <div className="flex flex-col gap-6 pt-3">
        <Group title="Library roots">
          {info === null ? (
            <Empty>Not connected to a back end.</Empty>
          ) : (
            <>
              <p className="text-xs text-fg-muted">
                {info.roots_editable
                  ? "The folders Calibration Studio may read and write. It reads nothing else."
                  : "The folders this server serves, fixed when it starts (its --root flags). Restart the server with another --root to change them."}
              </p>
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
                          disabled={readOnly}
                          onClick={() => void remove(root.path)}
                        >
                          <X aria-hidden className="size-3.5" />
                        </Button>
                      )
                    )}
                  </li>
                ))}
              </ul>
              <div>
                <AddRootButton />
              </div>
              {error && <p className="text-[11px] text-defect">{error}</p>}
            </>
          )}
        </Group>

        <Group title="Appearance">
          <SegmentedControl
            aria-label="Theme"
            value={theme}
            onValueChange={(value) => setTheme(value as ThemeChoice)}
            options={THEMES}
          />
        </Group>

        {info && (
          <Group title="Server">
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 font-mono text-xs">
              <dt className="text-fg-subtle">address</dt>
              <dd className="text-fg">
                {info.shell === "desktop"
                  ? "this computer (desktop app)"
                  : window.location.origin}
              </dd>
              <dt className="text-fg-subtle">version</dt>
              <dd className="text-fg">{info.version}</dd>
              <dt className="text-fg-subtle">samples</dt>
              <dd className="text-fg">{info.samples.length} available</dd>
            </dl>
          </Group>
        )}

        <Group title="Shortcuts">
          <ShortcutList groups={GLOBAL_SHORTCUTS} className="sm:grid-cols-1" />
        </Group>
      </div>
    </Dialog>
  );
}
