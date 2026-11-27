/** The back-end seam, enforced: outside `src/api/`, no module imports
 * `@tauri-apps/*`, calls a transport's `.invoke(` or `.resourceUrl(`, or
 * builds an `/api/` URL. So command names exist only in the generated
 * client, and every feature runs on both transports. (oxlint's
 * `no-restricted-imports` covers the imports too; it has no rule for the
 * calls, hence this test.) */
import { describe, expect, it } from "vitest";

const sources = import.meta.glob<string>(["../**/*.{ts,tsx}", "!../api/**"], {
  query: "?raw",
  import: "default",
  eager: true,
});

const FORBIDDEN: [RegExp, string][] = [
  [/from\s+["']@tauri-apps\//, "imports @tauri-apps"],
  [/\.invoke\(/, "calls .invoke("],
  [/\.resourceUrl\(/, "calls .resourceUrl("],
  [/["'`]\/api\//, "builds an /api/ URL"],
];

describe("the back-end seam", () => {
  it("is crossed only inside src/api", () => {
    const files = Object.keys(sources);
    expect(files.length).toBeGreaterThan(50);
    const offences: string[] = [];
    for (const [file, text] of Object.entries(sources)) {
      for (const [pattern, what] of FORBIDDEN) {
        if (pattern.test(text)) offences.push(`${file} ${what}`);
      }
    }
    expect(offences).toEqual([]);
  });
});
