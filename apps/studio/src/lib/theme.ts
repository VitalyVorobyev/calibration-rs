/** The `localStorage` key of the theme choice (`@vitavision/ui`'s `ThemeChoice`).
 * `index.html`'s no-flash script reads the same literal before the first paint;
 * `initTheme` (main.tsx) and `ThemeToggle` (AppShell) read and write it. */
export const THEME_STORAGE_KEY = "calib-theme";
