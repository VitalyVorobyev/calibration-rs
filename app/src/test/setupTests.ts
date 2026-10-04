/** Shared setup for jsdom component tests. Every test file loads it; a
 * component test opts into jsdom with a `// @vitest-environment jsdom`
 * pragma, and the guards below make this a no-op under the plain `node`
 * environment (the pure-logic unit tests), so one setup file covers both
 * without branching in `vitest.config.ts`. */
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

if (typeof window !== "undefined") {
  // React 19 requires this flag so `@testing-library/react`'s internal
  // `act()` wrapping doesn't warn about updates outside of `act`.
  (
    globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT?: boolean }
  ).IS_REACT_ACT_ENVIRONMENT = true;

  // jsdom has no layout engine, so ResizeObserver is unimplemented.
  // The frame viewer (stage2d's ImageStage, via DiagnoseWorkspace) observes its viewport on mount;
  // without a stub `new ResizeObserver(...)` throws a ReferenceError.
  if (typeof window.ResizeObserver === "undefined") {
    class StubResizeObserver {
      observe(): void {}
      unobserve(): void {}
      disconnect(): void {}
    }
    window.ResizeObserver = StubResizeObserver;
  }

  // Every test unmounts its own tree; DOM tests would otherwise leak
  // between `it()` blocks since `globals: true` isn't set.
  afterEach(() => {
    cleanup();
  });
}
