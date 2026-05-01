import { useEffect, useState } from "react";
import { Logo } from "./components/Logo";
import { ResidualViewer } from "./components/ResidualViewer";

const THEME_STORAGE_KEY = "calib-theme";

function isDocumentDark(): boolean {
  return document.documentElement.classList.contains("dark");
}

export function App() {
  const [dark, setDark] = useState<boolean>(isDocumentDark);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem(THEME_STORAGE_KEY, dark ? "dark" : "light");
  }, [dark]);

  return (
    <main className="flex h-full w-full flex-col gap-3 p-4">
      <header className="flex items-center justify-between gap-4 border-b border-border pb-3">
        <div className="flex items-center gap-3">
          <Logo size={28} className="text-foreground" />
          <div className="flex flex-col">
            <h1 className="m-0 text-base font-semibold tracking-tight">
              calibration-rs · diagnose
            </h1>
            <span className="text-[11px] text-muted-foreground">
              v0 — passive residual viewer (ADR 0014)
            </span>
          </div>
        </div>
        <ThemeToggle dark={dark} onToggle={() => setDark((d) => !d)} />
      </header>
      <ResidualViewer />
    </main>
  );
}

interface ThemeToggleProps {
  dark: boolean;
  onToggle: () => void;
}

function ThemeToggle({ dark, onToggle }: ThemeToggleProps) {
  return (
    <button
      type="button"
      onClick={onToggle}
      title={`Switch to ${dark ? "light" : "dark"} theme`}
      aria-label="Toggle theme"
      className="grid h-8 w-8 place-items-center !p-0"
    >
      {dark ? <SunIcon /> : <MoonIcon />}
    </button>
  );
}

function SunIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
}
