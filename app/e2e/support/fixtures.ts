/** Re-exports the canonical export fixtures for Playwright specs.
 *
 * The actual fixture data lives in `src/test/exportFixtures.ts` so
 * Vitest component tests and these e2e specs share exactly one copy
 * (see that module's header for why — it used to drift). `e2e/` sits
 * outside `tsconfig.json`'s `include: ["src"]` (see `eslint.config.js`),
 * but that only affects type-aware linting/`tsc -b`; Playwright's own
 * transpile-only test runner resolves this relative import at runtime
 * without needing `e2e` added to the tsconfig project.
 */
export {
  ONE_PX_PNG_DATA_URL,
  PLANAR_EXPORT_FIXTURE,
} from "../../src/test/exportFixtures";
