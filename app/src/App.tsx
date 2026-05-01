import { ResidualViewer } from "./components/ResidualViewer";

export function App() {
  return (
    <main className="flex h-full w-full flex-col gap-3 p-4">
      <header className="flex items-baseline justify-between gap-3">
        <h1 className="m-0 text-lg font-semibold">
          calibration-rs · diagnose
        </h1>
        <span className="text-xs opacity-60">
          v0 — passive residual viewer (ADR 0014)
        </span>
      </header>
      <ResidualViewer />
    </main>
  );
}
