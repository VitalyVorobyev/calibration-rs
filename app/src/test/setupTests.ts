/** Shared setup for jsdom component tests (B-QUAL3). Only wired for the
 * `jsdom` project via `vitest.config.ts`'s `environmentMatchGlobs` — the
 * guards below make it a no-op when it happens to load under the plain
 * `node` environment (the pure-logic unit tests), so one setup file
 * covers both without branching in `vitest.config.ts`. */
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

if (typeof window !== "undefined") {
  // React 19 requires this flag so `@testing-library/react`'s internal
  // `act()` wrapping doesn't warn about updates outside of `act`.
  (
    globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT?: boolean }
  ).IS_REACT_ACT_ENVIRONMENT = true;

  // jsdom has no layout engine, so ResizeObserver is unimplemented.
  // FrameCanvas (via DiagnoseWorkspace) observes its container on mount;
  // without a stub `new ResizeObserver(...)` throws a ReferenceError.
  if (typeof window.ResizeObserver === "undefined") {
    class StubResizeObserver {
      observe(): void {}
      unobserve(): void {}
      disconnect(): void {}
    }
    window.ResizeObserver = StubResizeObserver;
  }

  // jsdom implements <canvas> but not a 2D rendering context (that needs
  // the native `canvas` package). FrameCanvas already no-ops when
  // `getContext` returns null; stub it directly rather than pulling in
  // node-canvas, which keeps the component tests dependency-free and
  // silences jsdom's noisy "not implemented" console error.
  HTMLCanvasElement.prototype.getContext = (() =>
    null) as typeof HTMLCanvasElement.prototype.getContext;

  // Every test unmounts its own tree; DOM tests would otherwise leak
  // between `it()` blocks since `globals: true` isn't set.
  afterEach(() => {
    cleanup();
  });
}
