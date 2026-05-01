/** Placeholder for B3's in-app calibration runner. Picks a problem
 * type + Input JSON + Config JSON, dispatches to
 * `vision_calibration_pipeline::dispatch::run_from_json` via a Tauri
 * command, streams `calib::progress` events, and on completion lands
 * the user on /diagnose with the produced export loaded. */
export function RunWorkspace() {
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <header className="flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-tight">
          Run calibration
        </h2>
      </header>
      <div className="flex min-h-0 flex-1 items-center justify-center rounded-md border border-dashed border-border bg-bg-soft">
        <div className="max-w-[36rem] p-6 text-center">
          <h3 className="mb-2 text-sm font-semibold">
            Pre-detected runner ships in B3
          </h3>
          <p className="text-[12px] text-muted-foreground">
            Pick a problem type, Input JSON (already-detected 2D-3D
            correspondences), and a Config JSON. The runner streams
            progress per session step and hands the produced export to
            the diagnose viewer when finished.
          </p>
          <p className="mt-4 text-[11px] text-muted-foreground">
            B4 follows up with image-directory detection (chessboard /
            charuco / puzzleboard) so the workflow is image → calibration
            → diagnose without leaving the app.
          </p>
        </div>
      </div>
    </div>
  );
}
