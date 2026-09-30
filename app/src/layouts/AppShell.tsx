import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { Button, cn, ErrorBox, ThemeToggle } from "@vitavision/ui";
import { NavLink, Outlet } from "react-router";
import { Logo } from "../components/Logo";
import { isTauriContext } from "../lib/tauri";
import { THEME_STORAGE_KEY } from "../lib/theme";
import { useStore } from "../store";
import { exportKindLabel } from "../store/exportKind";

/** App-wide chrome: left workspace rail + top bar + outlet for the
 * active workspace. The Open Export button and theme toggle live here
 * so every workspace can see the loaded export and flip themes. */
export function AppShell() {
  const exportPath = useStore((s) => s.exportPath);
  const kind = useStore((s) => s.kind);
  const loadError = useStore((s) => s.loadError);
  const loadExport = useStore((s) => s.loadExport);
  const setLoadError = useStore((s) => s.setLoadError);
  const tauriOk = isTauriContext();

  const handleOpen = async () => {
    setLoadError(null);
    if (!tauriOk) {
      setLoadError(
        "Tauri runtime not detected. Launch the app with `bun run tauri dev` " +
          "(or the bundled binary). Plain `bun run dev` only starts Vite, so " +
          "the file-dialog and IPC commands aren't wired up.",
      );
      return;
    }
    let chosen: string | null;
    try {
      chosen = await openDialog({
        multiple: false,
        directory: false,
        filters: [{ name: "Calibration export", extensions: ["json"] }],
      });
    } catch (e) {
      setLoadError(`File dialog error: ${String(e)}`);
      return;
    }
    if (!chosen) return;
    await loadExport(chosen);
  };

  const exportLabel = exportPath ? basename(exportPath) : "No export loaded";

  return (
    <div className="grid h-full w-full grid-cols-[var(--rail-w)_1fr] grid-rows-[auto_1fr] [--rail-w:3.5rem]">
      <header className="col-span-2 flex items-center justify-between gap-4 border-b border-line bg-surface px-4 py-2.5">
        <div className="flex items-center gap-3">
          <Logo size={24} className="text-fg" />
          <div className="flex flex-col leading-tight">
            <h1 className="m-0 text-sm font-semibold tracking-tight">calibration-rs</h1>
            <span className="text-[10px] uppercase tracking-wider text-fg-muted">
              {kind ? exportKindLabel(kind) : "diagnose · 3d · epipolar · depth · run"}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span
            className="hidden max-w-[28rem] truncate font-mono text-[11px] text-fg-muted sm:inline"
            title={exportPath ?? ""}
          >
            {exportLabel}
          </span>
          <Button size="sm" onClick={() => void handleOpen()} disabled={!tauriOk}>
            Open Export…
          </Button>
          <ThemeToggle storageKey={THEME_STORAGE_KEY} />
        </div>
      </header>

      <nav className="row-start-2 flex flex-col items-stretch gap-1 border-r border-line bg-surface py-2">
        <RailLink to="/diagnose" label="Diagnose" hotkey="1">
          <DiagnoseIcon />
        </RailLink>
        <RailLink to="/viewer3d" label="3D viewer" hotkey="2">
          <CubeIcon />
        </RailLink>
        <RailLink to="/epipolar" label="Epipolar" hotkey="3">
          <EpipolarIcon />
        </RailLink>
        <RailLink to="/depth" label="Depth (dense stereo)" hotkey="4">
          <DepthIcon />
        </RailLink>
        <RailLink to="/run" label="Run calibration" hotkey="5">
          <RunIcon />
        </RailLink>
      </nav>

      <section className="row-start-2 flex min-h-0 flex-col overflow-hidden p-3">
        {loadError && <ErrorBox className="mb-2">{loadError}</ErrorBox>}
        <div className="flex min-h-0 flex-1 flex-col">
          <Outlet />
        </div>
      </section>
    </div>
  );
}

interface RailLinkProps {
  to: string;
  label: string;
  hotkey: string;
  children: React.ReactNode;
}

function RailLink({ to, label, children, hotkey }: RailLinkProps) {
  return (
    <NavLink
      to={to}
      title={`${label} (⌘${hotkey})`}
      aria-label={label}
      className={({ isActive }) =>
        cn(
          "mx-1 grid h-10 place-items-center rounded-control text-fg-muted transition-colors",
          "hover:bg-raised hover:text-fg",
          "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal",
          isActive && "bg-signal/12 text-signal hover:bg-signal/12 hover:text-signal",
        )
      }
    >
      {children}
    </NavLink>
  );
}

function basename(p: string): string {
  const idx = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  return idx >= 0 ? p.slice(idx + 1) : p;
}

const ICON_SIZE = 18;
const iconBase = {
  width: ICON_SIZE,
  height: ICON_SIZE,
  viewBox: "0 0 24 24",
  fill: "none" as const,
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true as const,
};

function DiagnoseIcon() {
  return (
    <svg {...iconBase}>
      <circle cx="12" cy="12" r="9" />
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
    </svg>
  );
}

function CubeIcon() {
  return (
    <svg {...iconBase}>
      <path d="M12 3 3 7.5v9L12 21l9-4.5v-9z" />
      <path d="M3 7.5 12 12l9-4.5M12 12v9" />
    </svg>
  );
}

function EpipolarIcon() {
  return (
    <svg {...iconBase}>
      <rect x="3" y="5" width="8" height="14" rx="1" />
      <rect x="13" y="5" width="8" height="14" rx="1" />
      <path d="M5 9 21 15M3 17l18-6" />
    </svg>
  );
}

function DepthIcon() {
  return (
    <svg {...iconBase}>
      <path d="M3 7.5 12 3l9 4.5-9 4.5z" />
      <path d="M3 12l9 4.5L21 12M3 16.5 12 21l9-4.5" />
    </svg>
  );
}

function RunIcon() {
  return (
    <svg {...iconBase}>
      <path d="M6 4l13 8-13 8z" />
    </svg>
  );
}
