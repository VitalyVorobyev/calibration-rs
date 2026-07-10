// Convert the Rust-generated JSON Schema (schemas-generated/diagnose_wire.json)
// into TypeScript interfaces under src/types/generated/ (B-QUAL2).
//
// Stage 2 of the two-stage `generate:types` pipeline: stage 1 is the Rust
// emitter (`generate:schemas`, needs cargo), this stage is TS-only (needs
// bun). Splitting them lets each CI job run the drift check with the
// toolchain it already has.
//
// Run via `bun run generate:types:ts`; prettier normalization happens in the
// package.json script that calls this, so the committed output matches
// `format:check`.
import { compileFromFile } from "json-schema-to-typescript";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const appDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const schemaPath = resolve(appDir, "schemas-generated/diagnose_wire.json");
const outPath = resolve(appDir, "src/types/generated/diagnose-wire.ts");

const banner = `/**
 * DO NOT EDIT — generated from the Rust wire types by
 * \`bun run generate:types\` (B-QUAL2). The source of truth is the
 * \`#[derive(schemars::JsonSchema)]\` types in the calibration workspace and
 * the diagnose Tauri commands; edit those and regenerate.
 */`;

const ts = await compileFromFile(schemaPath, {
  bannerComment: banner,
  additionalProperties: false,
  declareExternallyReferenced: true,
  enableConstEnums: false,
  format: false,
  cwd: dirname(schemaPath),
});

await mkdir(dirname(outPath), { recursive: true });
await writeFile(outPath, ts);
console.log(`wrote ${outPath}`);
