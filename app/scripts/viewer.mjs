// `bun run viewer -- <run-dir | export.json> [--port N] [--no-open]`: open a
// calibration run in the browser. Builds the frontend when `dist/` is
// missing or older than its sources, then runs the `calib-viewer` binary of
// `src-tauri` with the arguments given.
import { spawnSync } from "node:child_process";
import { existsSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const app = join(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
if (args[0] === "--") args.shift();

/** The newest modification time under `path`. */
function newest(path) {
  if (!existsSync(path)) return 0;
  const stat = statSync(path);
  if (!stat.isDirectory()) return stat.mtimeMs;
  let latest = stat.mtimeMs;
  for (const entry of readdirSync(path))
    latest = Math.max(latest, newest(join(path, entry)));
  return latest;
}

function run(command, commandArgs) {
  const result = spawnSync(command, commandArgs, { cwd: app, stdio: "inherit" });
  if (result.error) {
    console.error(`viewer: ${command}: ${result.error.message}`);
    process.exit(1);
  }
  if (result.status !== 0) process.exit(result.status ?? 1);
}

const dist = join(app, "dist");
// The repository carries a placeholder `dist/index.html` without `assets/`.
const built = existsSync(join(dist, "assets")) ? newest(join(dist, "index.html")) : 0;
const sources = Math.max(
  ...["src", "index.html", "package.json", "vite.config.ts", "bun.lock"].map((p) =>
    newest(join(app, p)),
  ),
);
if (built < sources) {
  console.log("viewer: building the frontend (dist/ is missing or stale)");
  run("bun", ["run", "build"]);
}

run("cargo", [
  "run",
  "--quiet",
  "--manifest-path",
  join(app, "src-tauri", "Cargo.toml"),
  "--bin",
  "calib-viewer",
  "--",
  ...args,
]);
