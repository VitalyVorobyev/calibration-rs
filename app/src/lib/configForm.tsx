/** Schema-driven form generator (ADR 0018) for the JSON Schema subset
 * `cargo xtask emit-schemas` produces. Handles primitives (string /
 * number / integer / boolean), enums, objects with `properties` +
 * `required`, `$ref` resolution, `oneOf` tagged unions on `kind`, and
 * string arrays. Anything outside that subset falls back to a raw JSON
 * textarea so the user is never blocked by a schema feature we haven't
 * polished yet. ~250 LoC; rich array / number-range / file-picker
 * widgets land in B3e.
 */
import { useMemo } from "react";

// ─── Schema type (loose) ────────────────────────────────────────────────────
export interface JsonSchema {
  $defs?: Record<string, JsonSchema>;
  $ref?: string;
  type?: string | string[];
  enum?: unknown[];
  const?: unknown;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  items?: JsonSchema;
  oneOf?: JsonSchema[];
  description?: string;
  format?: string;
  minimum?: number;
  maximum?: number;
  minItems?: number;
  maxItems?: number;
  default?: unknown;
}

interface FormCtx {
  defs: Record<string, JsonSchema>;
  path: string;
}

// Resolve `$ref` like "#/$defs/CameraSource" against the supplied defs.
function resolveRef(ref: string, defs: Record<string, JsonSchema>): JsonSchema | null {
  const prefix = "#/$defs/";
  if (!ref.startsWith(prefix)) return null;
  const key = ref.slice(prefix.length);
  return defs[key] ?? null;
}

function resolve(schema: JsonSchema, ctx: FormCtx): JsonSchema {
  if (schema.$ref) {
    const r = resolveRef(schema.$ref, ctx.defs);
    return r ? { ...r, description: schema.description ?? r.description } : schema;
  }
  return schema;
}

function hasType(schema: JsonSchema, t: string): boolean {
  if (Array.isArray(schema.type)) return schema.type.includes(t);
  return schema.type === t;
}

// ─── Public component ───────────────────────────────────────────────────────
export interface ConfigFormProps {
  schema: JsonSchema;
  value: unknown;
  onChange: (next: unknown) => void;
  /** Field path prefix shown to the user (e.g. "config" or "manifest"). */
  rootLabel?: string;
}

export function ConfigForm({ schema, value, onChange, rootLabel }: ConfigFormProps) {
  const defs = useMemo(() => schema.$defs ?? {}, [schema]);
  const ctx: FormCtx = { defs, path: rootLabel ?? "" };
  return (
    <div className="flex flex-col gap-3 text-[13px]">
      <SchemaField schema={schema} value={value} onChange={onChange} ctx={ctx} />
    </div>
  );
}

// ─── Field dispatch ─────────────────────────────────────────────────────────
interface FieldProps {
  schema: JsonSchema;
  value: unknown;
  onChange: (next: unknown) => void;
  ctx: FormCtx;
  /** Label text — falls back to the path tail. */
  label?: string;
}

function SchemaField(props: FieldProps) {
  const schema = resolve(props.schema, props.ctx);

  if (schema.oneOf && schema.oneOf.length > 0) {
    if (isExternalTaggedOneOf(schema, props.ctx)) {
      return <ExternalTaggedOneOfField {...props} schema={schema} />;
    }
    return <OneOfField {...props} schema={schema} />;
  }
  if (schema.enum && hasType(schema, "string")) {
    return <EnumField {...props} schema={schema} />;
  }
  if (hasType(schema, "object") && schema.properties) {
    return <ObjectField {...props} schema={schema} />;
  }
  if (hasType(schema, "array")) {
    return <ArrayField {...props} schema={schema} />;
  }
  if (hasType(schema, "boolean")) {
    return <BoolField {...props} schema={schema} />;
  }
  if (hasType(schema, "integer") || hasType(schema, "number")) {
    return <NumberField {...props} schema={schema} />;
  }
  if (hasType(schema, "string")) {
    return <StringField {...props} schema={schema} />;
  }

  // Unknown shape — JSON textarea fallback.
  return <JsonField {...props} schema={schema} />;
}

// ─── Object ────────────────────────────────────────────────────────────────
function ObjectField({ schema, value, onChange, ctx, label }: FieldProps) {
  const props = schema.properties ?? {};
  const required = schema.required ?? [];
  const obj =
    value && typeof value === "object" && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : ({} as Record<string, unknown>);
  return (
    <fieldset className="flex flex-col gap-2 rounded-md border border-border bg-bg-soft p-3">
      {(label || schema.description) && (
        <legend className="px-1 text-[12px] font-semibold tracking-tight">
          {label ?? ""}
        </legend>
      )}
      {schema.description && (
        <p className="text-[11px] text-muted-foreground">{schema.description}</p>
      )}
      {Object.entries(props).map(([key, sub]) => (
        <div key={key} className="flex flex-col gap-1">
          <label className="text-[11px] font-medium text-muted-foreground">
            {key}
            {required.includes(key) && <span className="ml-1 text-[color:var(--brand)]">*</span>}
          </label>
          <SchemaField
            schema={sub}
            value={obj[key]}
            onChange={(next) => onChange({ ...obj, [key]: next })}
            ctx={{ ...ctx, path: `${ctx.path}.${key}` }}
            label={key}
          />
        </div>
      ))}
    </fieldset>
  );
}

// ─── oneOf with serde externally tagged enum shape ─────────────────────────
interface ExternalVariant {
  tag: string;
  schema: JsonSchema | null;
}

function externalVariant(variant: JsonSchema, ctx: FormCtx): ExternalVariant | null {
  const v = resolve(variant, ctx);
  if (typeof v.const === "string" && hasType(v, "string")) {
    return { tag: v.const, schema: null };
  }
  const required = v.required ?? [];
  const props = v.properties ?? {};
  if (required.length === 1 && props[required[0]]) {
    return { tag: required[0], schema: props[required[0]] };
  }
  const keys = Object.keys(props);
  if (keys.length === 1) {
    return { tag: keys[0], schema: props[keys[0]] };
  }
  return null;
}

function isExternalTaggedOneOf(schema: JsonSchema, ctx: FormCtx): boolean {
  const variants = schema.oneOf ?? [];
  return variants.length > 0 && variants.every((v) => externalVariant(v, ctx) != null);
}

function ExternalTaggedOneOfField({ schema, value, onChange, ctx, label }: FieldProps) {
  const variants = (schema.oneOf ?? [])
    .map((v) => externalVariant(v, ctx))
    .filter((v): v is ExternalVariant => v != null);
  const currentTag = externalCurrentTag(value, variants);
  const active = variants.find((v) => v.tag === currentTag) ?? variants[0];
  const payload =
    active?.schema &&
    value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    active.tag in value
      ? (value as Record<string, unknown>)[active.tag]
      : active?.schema
        ? defaultValueForSchema(resolve(active.schema, ctx))
        : undefined;

  return (
    <fieldset className="flex flex-col gap-2 rounded-md border border-border bg-bg-soft p-3">
      {label && (
        <legend className="px-1 text-[12px] font-semibold tracking-tight">{label}</legend>
      )}
      {schema.description && (
        <p className="text-[11px] text-muted-foreground">{schema.description}</p>
      )}
      <select
        className="rounded border border-border bg-bg px-2 py-1 text-[12px]"
        value={active?.tag ?? ""}
        onChange={(e) => {
          const next = variants.find((v) => v.tag === e.target.value);
          if (!next) return;
          if (!next.schema) {
            onChange(next.tag);
          } else {
            onChange({ [next.tag]: defaultValueForSchema(resolve(next.schema, ctx)) });
          }
        }}
      >
        {variants.map((v) => (
          <option key={v.tag} value={v.tag}>
            {v.tag}
          </option>
        ))}
      </select>
      {active?.schema && (
        <SchemaField
          schema={active.schema}
          value={payload}
          onChange={(next) => onChange({ [active.tag]: next })}
          ctx={{ ...ctx, path: `${ctx.path}.${active.tag}` }}
          label={active.tag}
        />
      )}
    </fieldset>
  );
}

function externalCurrentTag(value: unknown, variants: ExternalVariant[]): string {
  if (typeof value === "string" && variants.some((v) => v.tag === value)) {
    return value;
  }
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>;
    const key = Object.keys(obj).find((k) => variants.some((v) => v.tag === k));
    if (key) return key;
  }
  return variants[0]?.tag ?? "";
}

function defaultValueForSchema(schema: JsonSchema): unknown {
  const resolved = schema;
  if (resolved.default !== undefined) return resolved.default;
  if (typeof resolved.const === "string") return resolved.const;
  if (resolved.enum && resolved.enum.length > 0) return resolved.enum[0];
  if (hasType(resolved, "object") && resolved.properties) {
    const out: Record<string, unknown> = {};
    const required = resolved.required ?? Object.keys(resolved.properties);
    for (const key of required) {
      const sub = resolved.properties[key];
      if (sub) out[key] = defaultValueForSchema(sub);
    }
    return out;
  }
  if (hasType(resolved, "array")) return [];
  if (hasType(resolved, "boolean")) return false;
  if (hasType(resolved, "integer") || hasType(resolved, "number")) {
    return resolved.minimum ?? 1;
  }
  if (hasType(resolved, "string")) return "";
  return {};
}

// ─── oneOf with `kind` discriminator ────────────────────────────────────────
function OneOfField({ schema, value, onChange, ctx, label }: FieldProps) {
  const variants = (schema.oneOf ?? []).map((variant) => resolve(variant, ctx));
  const variantTags = variants.map((v) => {
    const k = v.properties?.kind;
    return typeof k?.const === "string" ? (k.const as string) : null;
  });
  const obj =
    value && typeof value === "object" && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : ({} as Record<string, unknown>);
  const currentKind = typeof obj.kind === "string" ? obj.kind : variantTags[0] ?? "";
  const activeIdx = variantTags.findIndex((t) => t === currentKind);
  const active = activeIdx >= 0 ? variants[activeIdx] : variants[0];

  return (
    <fieldset className="flex flex-col gap-2 rounded-md border border-border bg-bg-soft p-3">
      {label && (
        <legend className="px-1 text-[12px] font-semibold tracking-tight">{label}</legend>
      )}
      {schema.description && (
        <p className="text-[11px] text-muted-foreground">{schema.description}</p>
      )}
      <select
        className="rounded border border-border bg-bg px-2 py-1 text-[12px]"
        value={currentKind}
        onChange={(e) => {
          const tag = e.target.value;
          // Reset to a fresh variant body, preserving only `kind`.
          onChange({ kind: tag });
        }}
      >
        {variantTags.map((t, i) =>
          t ? (
            <option key={t} value={t}>
              {t}
            </option>
          ) : (
            <option key={`v${i}`} value="">
              variant {i}
            </option>
          ),
        )}
      </select>
      {active.properties && (
        <ObjectField
          schema={{ ...active, properties: omitKey(active.properties, "kind") }}
          value={obj}
          onChange={(next) => {
            const merged = next as Record<string, unknown>;
            onChange({ ...merged, kind: currentKind });
          }}
          ctx={ctx}
        />
      )}
    </fieldset>
  );
}

function omitKey<T>(obj: Record<string, T>, drop: string): Record<string, T> {
  const out: Record<string, T> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (k !== drop) out[k] = v;
  }
  return out;
}

// ─── Enums ──────────────────────────────────────────────────────────────────
function EnumField({ schema, value, onChange }: FieldProps) {
  const options = (schema.enum ?? []).map((v) => String(v));
  const current = typeof value === "string" ? value : options[0] ?? "";
  return (
    <select
      className="rounded border border-border bg-bg px-2 py-1 text-[12px]"
      value={current}
      onChange={(e) => onChange(e.target.value)}
    >
      {options.map((opt) => (
        <option key={opt} value={opt}>
          {opt}
        </option>
      ))}
    </select>
  );
}

// ─── Primitives ─────────────────────────────────────────────────────────────
function StringField({ schema, value, onChange }: FieldProps) {
  const v = typeof value === "string" ? value : "";
  const placeholder = schema.description ? schema.description.split("\n")[0] : undefined;
  return (
    <input
      type="text"
      value={v}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
      className="rounded border border-border bg-bg px-2 py-1 text-[12px]"
    />
  );
}

function NumberField({ schema, value, onChange }: FieldProps) {
  const v = typeof value === "number" ? value : 0;
  const step = hasType(schema, "integer") ? 1 : "any";
  return (
    <input
      type="number"
      value={v}
      step={step}
      min={schema.minimum}
      max={schema.maximum}
      onChange={(e) => {
        const text = e.target.value;
        if (text === "") return; // leave previous value if cleared mid-edit
        const parsed = hasType(schema, "integer") ? parseInt(text, 10) : parseFloat(text);
        if (!Number.isNaN(parsed)) onChange(parsed);
      }}
      className="rounded border border-border bg-bg px-2 py-1 text-[12px]"
    />
  );
}

function BoolField({ value, onChange }: FieldProps) {
  const v = value === true;
  return (
    <input
      type="checkbox"
      checked={v}
      onChange={(e) => onChange(e.target.checked)}
      className="h-4 w-4"
    />
  );
}

// ─── Array (string-only fast path; JSON fallback otherwise) ────────────────
function ArrayField(props: FieldProps) {
  const { schema, ctx } = props;
  const items = schema.items ? resolve(schema.items, ctx) : null;
  if (items && hasType(items, "string")) {
    return <StringArrayField {...props} schema={schema} />;
  }
  // Numbers / nested objects in arrays: defer to JSON textarea for v0.
  return <JsonField {...props} schema={schema} />;
}

function StringArrayField({ value, onChange }: FieldProps) {
  const arr = Array.isArray(value) ? (value as unknown[]).map((v) => String(v ?? "")) : [];
  return (
    <div className="flex flex-col gap-1">
      {arr.map((entry, i) => (
        <div key={i} className="flex gap-1">
          <input
            type="text"
            value={entry}
            onChange={(e) => {
              const copy = arr.slice();
              copy[i] = e.target.value;
              onChange(copy);
            }}
            className="flex-1 rounded border border-border bg-bg px-2 py-1 text-[12px]"
          />
          <button
            type="button"
            onClick={() => onChange(arr.filter((_, j) => j !== i))}
            className="rounded border border-border px-2 text-[11px]"
          >
            ×
          </button>
        </div>
      ))}
      <button
        type="button"
        onClick={() => onChange([...arr, ""])}
        className="self-start rounded border border-border px-2 py-1 text-[11px]"
      >
        + add
      </button>
    </div>
  );
}

// ─── JSON textarea fallback ────────────────────────────────────────────────
function JsonField({ value, onChange }: FieldProps) {
  const text = useMemo(() => JSON.stringify(value, null, 2), [value]);
  return (
    <textarea
      defaultValue={text}
      onBlur={(e) => {
        try {
          onChange(JSON.parse(e.target.value));
        } catch {
          /* keep previous value; v0 doesn't surface parse errors yet */
        }
      }}
      rows={6}
      className="rounded border border-border bg-bg px-2 py-1 font-mono text-[11px]"
    />
  );
}
