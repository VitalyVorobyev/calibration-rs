import type { Page } from "@playwright/test";

/** Installs a minimal `window.__TAURI_INTERNALS__` shim before any of
 * the page's own scripts run, so `@tauri-apps/api/core`'s `invoke()`
 * (and `@tauri-apps/plugin-dialog`'s `open()`, which itself calls
 * `invoke("plugin:dialog|open", …)`) resolve against canned responses
 * instead of throwing "not a Tauri context".
 *
 * This is a hand-rolled substitute for `@tauri-apps/api/mocks`'s
 * `mockIPC`: that helper patches IPC inside the page's own module graph,
 * which works when the test and the app share one JS realm (Vitest +
 * jsdom component tests). Playwright instead loads the real built app
 * into a separate browser page, so the mock has to be injected as a
 * plain init script — `page.addInitScript` runs in that page's context
 * before `index.html`'s own `<script>` tags execute.
 *
 * `responses` is keyed by Tauri command name (e.g. `"load_export"`,
 * `"plugin:dialog|open"`); every call to that command returns the same
 * canned value regardless of its arguments, which is enough for a smoke
 * test driving one fixed scenario per spec.
 *
 * `assetUrl` stands in for the asset protocol: every `convertFileSrc`
 * (an image the webview loads from disk) resolves to it.
 */
export async function installTauriMock(
  page: Page,
  responses: Record<string, unknown>,
  assetUrl = "",
): Promise<void> {
  await page.addInitScript(
    ({ responsesArg, assetUrlArg }) => {
      (
        window as unknown as { __TAURI_INTERNALS__: Record<string, unknown> }
      ).__TAURI_INTERNALS__ = {
        invoke: (cmd: string) => {
          if (Object.prototype.hasOwnProperty.call(responsesArg, cmd)) {
            return Promise.resolve((responsesArg as Record<string, unknown>)[cmd]);
          }
          return Promise.resolve(null);
        },
        transformCallback: () => 0,
        unregisterCallback: () => {},
        convertFileSrc: (p: string) => assetUrlArg || p,
      };
    },
    { responsesArg: responses, assetUrlArg: assetUrl },
  );
}
