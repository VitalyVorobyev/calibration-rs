import { lazy, Suspense } from "react";

// Three.js + R3F + drei are heavy (~900 KB gzipped). Lazy-load so the
// diagnose / epipolar / run paths don't pull them in.
const Viewer3DWorkspace = lazy(() =>
  import("./index").then((m) => ({
    default: m.Viewer3DWorkspace,
  })),
);

function ViewerFallback() {
  return (
    <div className="flex min-h-0 flex-1 items-center justify-center rounded-md border border-dashed border-border bg-bg-soft">
      <p className="text-[13px] text-muted-foreground">Loading 3D scene…</p>
    </div>
  );
}

/** The 3D viewer route element: the workspace loaded on demand behind a
 * placeholder. */
export function LazyViewer3DWorkspace() {
  return (
    <Suspense fallback={<ViewerFallback />}>
      <Viewer3DWorkspace />
    </Suspense>
  );
}
