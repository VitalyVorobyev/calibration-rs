// `bun scripts/e2e-server.mjs <port>`: the real calibration-studio-server for
// the Playwright specs. Copies the committed fixture library
// (`e2e/fixtures/library`) to a fresh temporary root, so a spec that writes
// (a calibration run) never touches the checkout, and serves it on
// 127.0.0.1:<port> with the token in `$STUDIO_E2E_TOKEN` (a loopback bind
// with a configured token still requires it). Builds the debug server and the
// frontend first when they are missing.
import { spawn, spawnSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const app = join(dirname(fileURLToPath(import.meta.url)), "..");
const port = process.argv[2] ?? "1431";
const token = process.env.STUDIO_E2E_TOKEN;
if (!token) {
  console.error("e2e-server: set STUDIO_E2E_TOKEN");
  process.exit(2);
}

function run(command, args) {
  const result = spawnSync(command, args, { cwd: app, stdio: "inherit" });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

const exe = process.platform === "win32" ? ".exe" : "";
const binary = join(app, "target", "debug", `calibration-studio-server${exe}`);
if (!existsSync(binary)) run("cargo", ["build", "-p", "studio-server"]);
if (!existsSync(join(app, "dist", "assets"))) run("bun", ["run", "build"]);

const base = join(tmpdir(), `calibration-studio-e2e-${port}`);
rmSync(base, { recursive: true, force: true });
mkdirSync(base, { recursive: true });
const library = join(base, "library");
cpSync(join(app, "e2e", "fixtures", "library"), library, { recursive: true });
const tokenFile = join(base, "token");
writeFileSync(tokenFile, `${token}\n`, { mode: 0o600 });

const server = spawn(
  binary,
  [
    "--root",
    library,
    "--bind",
    `127.0.0.1:${port}`,
    "--token-file",
    tokenFile,
    "--state-dir",
    join(base, "state"),
    "--cache-dir",
    join(base, "cache"),
    "--dist",
    join(app, "dist"),
  ],
  { cwd: app, stdio: "inherit" },
);
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => server.kill(signal));
}
server.on("exit", (code) => process.exit(code ?? 0));
