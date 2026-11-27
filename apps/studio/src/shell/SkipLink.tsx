/** The first stop of the page's Tab order: a link to the screen itself,
 * visible only while it has focus. The window's frame puts the context bar
 * and the rail before the screen, so a keyboard user would tab through the
 * seven screens and the bar's controls each time without it. */
export const SCREEN_ID = "screen";

export function SkipLink() {
  return (
    <button
      type="button"
      onClick={() => document.getElementById(SCREEN_ID)?.focus()}
      className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:rounded-control focus:bg-signal focus:px-3 focus:py-1.5 focus:text-xs focus:font-medium focus:text-signal-fg focus:outline-2 focus:outline-offset-2 focus:outline-signal"
    >
      Skip to the screen
    </button>
  );
}
