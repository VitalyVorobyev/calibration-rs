// Generate the frontend's API layer from the back end's schema
// (`schema/studio-api.json`, written by studio-core's `studio-schema`):
//
// - src/api/generated/types.ts — every wire type, by json-schema-to-typescript;
// - src/api/generated/client.ts — `createClient(transport)`: one typed function
//   per command and job, one URL builder per resource (`urls`), documented
//   with the command's own doc,
//   and the `Commands` / `Jobs` / `Resources` maps the test transport is
//   typed by. It is the only file that names commands.
//
// Run via `bun run generate:api:ts` (or `bun run generate:api`, which writes
// the schema first); prettier normalizes the output in the package.json
// script, so the committed files match `format:check`.
import { compile } from "json-schema-to-typescript";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const appDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const schemaPath = resolve(appDir, "schema/studio-api.json");
const outDir = resolve(appDir, "src/api/generated");

const schema = JSON.parse(await readFile(schemaPath, "utf8"));
const definitions = { ...schema.definitions };

const banner = (what) => `/**
 * DO NOT EDIT — ${what}, generated from the back end's command table by
 * \`bun run generate:api\`. The source of truth is studio-core's
 * \`api::table\` and the \`#[derive(schemars::JsonSchema)]\` types; edit those
 * and regenerate.
 */`;

const refName = (s) =>
  typeof s?.$ref === "string" && s.$ref.startsWith("#/definitions/")
    ? s.$ref.slice("#/definitions/".length)
    : null;
const pascal = (name) =>
  name
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join("");
const camel = (name) => {
  const p = pascal(name);
  return p.charAt(0).toLowerCase() + p.slice(1);
};

/** The type name of `part`: its definition, or a new one named `fallback`
 * for an inline schema (a list, an optional, `()`, any JSON). */
function named(part, fallback) {
  const ref = refName(part);
  if (ref) return ref;
  const anyJson =
    part === true ||
    (typeof part === "object" && part !== null && Object.keys(part).length === 0);
  definitions[fallback] = anyJson ? { tsType: "unknown" } : part;
  return fallback;
}

/** Any JSON (`serde_json::Value`): a schema that constrains nothing. Left
 * alone, json-schema-to-typescript would type it as an object. */
const SHAPE_KEYS = [
  "type",
  "$ref",
  "allOf",
  "anyOf",
  "oneOf",
  "enum",
  "const",
  "items",
  "properties",
  "additionalProperties",
  "not",
  "tsType",
];
function markAnyJson(node) {
  if (Array.isArray(node)) {
    node.forEach(markAnyJson);
    return;
  }
  if (typeof node !== "object" || node === null) return;
  for (const key of ["properties", "definitions"]) {
    for (const [name, child] of Object.entries(node[key] ?? {})) {
      if (child === true) node[key][name] = { tsType: "unknown" };
      else if (typeof child === "object" && !SHAPE_KEYS.some((k) => k in child)) {
        child.tsType = "unknown";
      }
    }
  }
  for (const value of Object.values(node)) markAnyJson(value);
}
markAnyJson(definitions);

const commands = schema.commands.map((c) => {
  const args = refName(c.args);
  if (!args) throw new Error(`${c.name}: its arguments are not a named type`);
  const base = pascal(c.name);
  return {
    ...c,
    args,
    reply: c.reply !== undefined ? named(c.reply, `${base}Reply`) : null,
    event: c.event !== undefined ? named(c.event, `${base}Event`) : null,
    output: c.output !== undefined ? named(c.output, `${base}Output`) : null,
  };
});
for (const required of ["job_status", "cancel_job"]) {
  if (!commands.some((c) => c.name === required && c.kind === "plain")) {
    throw new Error(`the table has no plain command ${required}`);
  }
}

// --- types.ts --------------------------------------------------------------
// A wrapper whose properties reference every definition, so each becomes an
// exported type; its own interface is unused.
const names = Object.keys(definitions).sort();
const wrapper = {
  $schema: "http://json-schema.org/draft-07/schema#",
  title: "StudioApiTypes",
  type: "object",
  additionalProperties: false,
  properties: Object.fromEntries(names.map((n) => [n, { $ref: `#/definitions/${n}` }])),
  definitions,
};
const types = await compile(wrapper, "StudioApiTypes", {
  bannerComment: banner("the back end's wire types"),
  additionalProperties: false,
  declareExternallyReferenced: true,
  enableConstEnums: false,
  format: false,
});

// --- client.ts -------------------------------------------------------------
const jsdoc = (doc, indent) => {
  if (!doc) return "";
  const lines = doc.replaceAll("*/", "*\\/").split("\n");
  return `${indent}/**\n${lines.map((l) => `${indent} *${l ? ` ${l}` : ""}`).join("\n")}\n${indent} */\n`;
};
const T = (name) => `T.${name}`;
const noArgs = (c) => {
  const def = definitions[c.args];
  return !def?.properties || Object.keys(def.properties).length === 0;
};

const plain = commands.filter((c) => c.kind === "plain");
const jobs = commands.filter((c) => c.kind === "job");
const resources = commands.filter((c) => c.kind === "resource");

const lines = [];
lines.push(banner("the typed client"));
lines.push(`import type { Job, JobControl, JobHub } from "../jobs";`);
lines.push(`import { attachJob, startJob } from "../jobs";`);
lines.push(`import type { Transport } from "../transport";`);
lines.push(`import type * as T from "./types";`);
lines.push("");
lines.push("/** Every plain command: its argument object and its reply. */");
lines.push("export interface Commands {");
for (const c of plain)
  lines.push(`  ${c.name}: { args: ${T(c.args)}; reply: ${T(c.reply)} };`);
lines.push("}");
lines.push("");
lines.push("/** Every job: its argument object, its progress event and its output. */");
lines.push("export interface Jobs {");
for (const c of jobs)
  lines.push(
    `  ${c.name}: { args: ${T(c.args)}; event: ${T(c.event)}; output: ${T(c.output)} };`,
  );
lines.push("}");
lines.push("");
lines.push("/** Every resource: its query object. */");
lines.push("export interface Resources {");
for (const c of resources) lines.push(`  ${c.name}: ${T(c.args)};`);
lines.push("}");
lines.push("");
lines.push("/** The back end's API over `t`; `hub` follows its jobs. */");
lines.push("export function createClient(t: Transport, hub: JobHub) {");
lines.push("  const control: JobControl = {");
lines.push(`    status: (id) => t.invoke("job_status", { id }) as Promise<T.JobInfo>,`);
lines.push(`    cancel: (id) => t.invoke("cancel_job", { id }) as Promise<T.JobInfo>,`);
lines.push("  };");
lines.push("  return {");
for (const c of plain) {
  if (c.doc) lines.push(jsdoc(c.doc, "    ").trimEnd());
  const param = noArgs(c) ? `args: ${T(c.args)} = {}` : `args: ${T(c.args)}`;
  lines.push(
    `    ${camel(c.name)}: (${param}): Promise<${T(c.reply)}> =>`,
    `      t.invoke("${c.name}", args) as Promise<${T(c.reply)}>,`,
  );
}
lines.push("    jobs: {");
for (const c of jobs) {
  if (c.doc) lines.push(jsdoc(c.doc, "      ").trimEnd());
  const job = `Job<${T(c.event)}, ${T(c.output)}>`;
  lines.push(
    `      ${camel(c.name)}: (args: ${T(c.args)}): Promise<${job}> =>`,
    `        startJob<${T(c.event)}, ${T(c.output)}>(t, hub, control, "${c.name}", args),`,
  );
}
lines.push("      /** Follow the job `id` (started earlier, say before a reload). */");
lines.push(
  "      attach: <E, R>(id: string): Promise<Job<E, R>> => attachJob<E, R>(hub, control, id),",
);
lines.push("    },");
lines.push(
  "    /** The URL of each resource: for an `<img>`, a `fetch` or `fetchPlane`. */",
);
lines.push("    urls: {");
for (const c of resources) {
  if (c.doc) lines.push(jsdoc(c.doc, "      ").trimEnd());
  lines.push(
    `      ${camel(c.name)}: (query: ${T(c.args)}): string =>`,
    `        t.resourceUrl("${c.name}", query),`,
  );
}
lines.push("    },");
lines.push("  };");
lines.push("}");
lines.push("");
lines.push("export type Client = ReturnType<typeof createClient>;");
const client = lines.join("\n") + "\n";

await mkdir(outDir, { recursive: true });
await writeFile(resolve(outDir, "types.ts"), types);
await writeFile(resolve(outDir, "client.ts"), client);
console.log(`wrote ${outDir}/{types,client}.ts`);
