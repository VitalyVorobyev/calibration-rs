//! Render JSON Schemas as Python `TypedDict` source.
//!
//! The Python package's payload types are generated from the same schemas the
//! xtask writes for the app and for external tools, so a Rust type is the one
//! source of truth for its JSON shape in every language. The mapping:
//!
//! - every `$defs` entry, and every root, becomes a named Python type under its
//!   Rust name (a root's name is given explicitly, since a generic or aliased
//!   Rust type has no name of its own in the schema);
//! - an object becomes a `TypedDict`; fields outside `required` (a serde
//!   default, or skipped when empty) are `total=False`, through a private
//!   `_<Name>Required` base when an object has both kinds;
//! - `oneOf` / `anyOf` becomes a union: string constants merge into one
//!   `Literal[...]`, an object variant becomes a `TypedDict` named
//!   `<Enum>_<Variant>` after its tag, and `null` becomes `None`;
//! - a nullable `type` (`["number", "null"]`) becomes `float | None`;
//! - descriptions become docstrings (attribute docstrings for fields and
//!   aliases).
//!
//! The same name with two different schemas is an error: two Rust types (or
//! two instantiations of one generic type) would otherwise share a Python
//! name. A `rename` gives one of them its own name.

use anyhow::{Context, Result, bail};
use regex::Regex;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::sync::LazyLock;

/// One root schema to render.
pub struct PySchema<'a> {
    /// Where the schema comes from, for error messages.
    pub source: &'a str,
    /// The Python name of the root type: the Rust name of the type.
    pub root: &'a str,
    /// The schema, with its `$defs`.
    pub schema: &'a Value,
    /// `$defs` names to replace in this schema, as `(def name, Python name)`.
    /// Every one must occur in the schema.
    pub renames: &'a [(&'a str, &'a str)],
}

/// Render `schemas` as one Python module: `header` (comment lines and the
/// module docstring), the imports, every type, and `__all__`.
///
/// `renames` apply to every schema that has the def; the per-schema renames
/// win over them.
pub fn render(header: &str, schemas: &[PySchema<'_>], renames: &[(&str, &str)]) -> Result<String> {
    let defs = collect_defs(schemas, renames)?;
    let mut generator = Generator {
        defs: &defs,
        items: BTreeMap::new(),
        typing: BTreeSet::new(),
    };
    for (name, schema) in &defs {
        generator.named(name, schema)?;
    }
    generator.module(header)
}

/// Every named schema, keyed by Python name, with its `$ref`s rewritten to
/// Python names.
fn collect_defs(
    schemas: &[PySchema<'_>],
    global_renames: &[(&str, &str)],
) -> Result<BTreeMap<String, Value>> {
    let mut defs: BTreeMap<String, (Value, String)> = BTreeMap::new();
    let mut insert = |name: String, schema: Value, source: &str| -> Result<()> {
        match defs.get(&name) {
            Some((existing, first)) if *existing != schema => bail!(
                "`{name}` has one schema in {first} and another in {source}; \
                 give one of them its own Python name"
            ),
            Some(_) => Ok(()),
            None => {
                defs.insert(name, (schema, source.to_string()));
                Ok(())
            }
        }
    };

    for entry in schemas {
        let local_defs = entry
            .schema
            .get("$defs")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        for (from, _) in entry.renames {
            if !local_defs.contains_key(*from) {
                bail!(
                    "{}: rename of `{from}`, which is not in `$defs`",
                    entry.source
                );
            }
        }
        let rename: BTreeMap<&str, &str> = global_renames
            .iter()
            .chain(entry.renames)
            .map(|&(from, to)| (from, to))
            .collect();
        let python_name = |name: &str| rename.get(name).copied().unwrap_or(name).to_string();

        for (name, schema) in &local_defs {
            insert(
                python_name(name),
                rewrite_refs(schema, &python_name),
                entry.source,
            )?;
        }
        let mut root = entry
            .schema
            .as_object()
            .with_context(|| format!("{}: the root schema is not an object", entry.source))?
            .clone();
        for key in ["$defs", "$schema", "title"] {
            root.remove(key);
        }
        insert(
            entry.root.to_string(),
            rewrite_refs(&Value::Object(root), &python_name),
            entry.source,
        )?;
    }
    Ok(defs
        .into_iter()
        .map(|(name, (schema, _))| (name, schema))
        .collect())
}

/// `schema` with every `#/$defs/<name>` reference pointing at the Python name.
fn rewrite_refs(schema: &Value, python_name: &dyn Fn(&str) -> String) -> Value {
    match schema {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| {
                    let value = match (key.as_str(), value) {
                        ("$ref", Value::String(target)) => match target.strip_prefix(DEFS) {
                            Some(name) => Value::String(format!("{DEFS}{}", python_name(name))),
                            None => value.clone(),
                        },
                        _ => rewrite_refs(value, python_name),
                    };
                    (key.clone(), value)
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| rewrite_refs(item, python_name))
                .collect(),
        ),
        other => other.clone(),
    }
}

const DEFS: &str = "#/$defs/";

/// One top-level Python definition.
struct Item {
    code: String,
    /// Names its code evaluates at import time: an alias's right-hand side, a
    /// functional `TypedDict`'s field types. Class annotations are lazy
    /// (`from __future__ import annotations`), so classes have none.
    eager: BTreeSet<String>,
}

struct Generator<'a> {
    defs: &'a BTreeMap<String, Value>,
    items: BTreeMap<String, Item>,
    /// Names imported from `typing`.
    typing: BTreeSet<&'static str>,
}

impl Generator<'_> {
    /// Define `name` as the Python type of `schema`.
    fn named(&mut self, name: &str, schema: &Value) -> Result<()> {
        if self.items.contains_key(name) {
            return Ok(());
        }
        if !is_identifier(name) {
            bail!("`{name}` is not a valid Python name");
        }
        // Reserve the name before descending, so a recursive type terminates.
        self.items.insert(
            name.to_string(),
            Item {
                code: String::new(),
                eager: BTreeSet::new(),
            },
        );
        let item = if let Some(object) = self.object_fields(schema)? {
            self.typed_dict(name, schema, &object)?
        } else {
            self.alias(name, schema)?
        };
        self.items.insert(name.to_string(), item);
        Ok(())
    }

    /// The fields of an object schema (merged with its `$ref` target, the
    /// form serde's internally tagged newtype variants take), or `None` when
    /// `schema` is not an object with properties.
    fn object_fields(&self, schema: &Value) -> Result<Option<ObjectFields>> {
        let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
            return Ok(None);
        };
        let mut fields = ObjectFields::default();
        if let Some(target) = schema.get("$ref").and_then(Value::as_str) {
            let base = self.resolve(target)?;
            match self.object_fields(base)? {
                Some(base_fields) => fields = base_fields,
                None => bail!("`{target}` is merged into an object but is not one"),
            }
        }
        fields.properties.extend(properties.clone());
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            fields.required.extend(
                required
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string),
            );
        }
        Ok(Some(fields))
    }

    fn resolve(&self, target: &str) -> Result<&Value> {
        let name = target
            .strip_prefix(DEFS)
            .with_context(|| format!("unsupported `$ref` `{target}`"))?;
        self.defs
            .get(name)
            .with_context(|| format!("`$ref` to `{name}`, which no schema defines"))
    }

    fn typed_dict(&mut self, name: &str, schema: &Value, object: &ObjectFields) -> Result<Item> {
        self.typing.insert("TypedDict");
        let mut fields = Vec::new();
        let mut eager = BTreeSet::new();
        for (key, property) in &object.properties {
            let hint = format!("{name}_{key}");
            let ty = self.type_expr(property, &hint, &mut eager)?;
            fields.push(Field {
                key: key.clone(),
                ty,
                required: object.required.contains(key),
                doc: description(property),
            });
        }
        let doc = description(schema);
        let all_required = fields.iter().all(|f| f.required);
        let none_required = fields.iter().all(|f| !f.required);

        if fields.iter().all(|f| is_identifier(&f.key)) {
            // Class form; annotations are lazy, so it has no eager names.
            let mut code = String::new();
            if all_required || none_required {
                let total = if all_required { "" } else { ", total=False" };
                class_block(
                    &mut code,
                    name,
                    &format!("TypedDict{total}"),
                    doc.as_deref(),
                    &fields,
                );
            } else {
                let base = format!("_{name}Required");
                let (required, optional): (Vec<_>, Vec<_>) =
                    fields.into_iter().partition(|f| f.required);
                let base_doc = format!("The keys of `{name}` that every payload has.");
                class_block(&mut code, &base, "TypedDict", Some(&base_doc), &required);
                code.push_str("\n\n");
                class_block(
                    &mut code,
                    name,
                    &format!("{base}, total=False"),
                    doc.as_deref(),
                    &optional,
                );
            }
            return Ok(Item {
                code,
                eager: BTreeSet::new(),
            });
        }

        // A key that is not a Python identifier (`lambda`) needs the
        // functional form, whose field types are evaluated at import time.
        if !(all_required || none_required) {
            bail!(
                "`{name}`: a non-identifier key in an object with both required and optional fields"
            );
        }
        let mut code = format!("{name} = TypedDict(\n    \"{name}\",\n    {{\n");
        for field in &fields {
            if let Some(field_doc) = &field.doc {
                for line in clean_doc(field_doc).lines() {
                    writeln!(code, "        # {line}").ok();
                    if line.is_empty() {
                        code.truncate(code.len() - 2);
                        code.push('\n');
                    }
                }
            }
            writeln!(code, "        {}: {},", py_str(&field.key), field.ty).ok();
        }
        code.push_str("    },\n");
        if none_required {
            code.push_str("    total=False,\n");
        }
        code.push_str(")\n");
        if let Some(doc) = &doc {
            code.push_str(&docstring(doc, ""));
        }
        Ok(Item { code, eager })
    }

    fn alias(&mut self, name: &str, schema: &Value) -> Result<Item> {
        self.typing.insert("TypeAlias");
        let mut eager = BTreeSet::new();
        let ty = self.type_expr(schema, name, &mut eager)?;
        let mut code = alias_line(name, &ty);
        let mut doc = description(schema)
            .map(|d| clean_doc(&d))
            .unwrap_or_default();
        // A string enum lists what each value means.
        let variants: Vec<(String, String)> = variants(schema)
            .iter()
            .filter_map(|v| {
                let value = v.get("const")?.as_str()?;
                Some((value.to_string(), clean_doc(&description(v)?)))
            })
            .collect();
        if !variants.is_empty() {
            if !doc.is_empty() {
                doc.push_str("\n\n");
            }
            for (value, meaning) in variants {
                writeln!(doc, "- ``\"{value}\"``: {}", meaning.replace('\n', "\n  ")).ok();
            }
        }
        if !doc.trim().is_empty() {
            code.push_str(&docstring(doc.trim_end(), ""));
        }
        Ok(Item { code, eager })
    }

    /// The Python type of `schema`. An anonymous object in it is defined as
    /// a named type, `hint` (or `<hint>_<Variant>` for a union member).
    fn type_expr(
        &mut self,
        schema: &Value,
        hint: &str,
        deps: &mut BTreeSet<String>,
    ) -> Result<String> {
        let Some(map) = schema.as_object() else {
            return Ok(match schema {
                Value::Bool(true) => self.any(),
                _ => bail!("`{hint}`: unsupported schema `{schema}`"),
            });
        };

        if map.contains_key("properties") {
            if self.defs.contains_key(hint) || self.items.contains_key(hint) {
                bail!("the anonymous object `{hint}` would take a name already in use");
            }
            self.named(hint, schema)?;
            deps.insert(hint.to_string());
            return Ok(hint.to_string());
        }
        if let Some(target) = map.get("$ref").and_then(Value::as_str) {
            self.resolve(target)?;
            let name = &target[DEFS.len()..];
            deps.insert(name.to_string());
            return Ok(name.to_string());
        }
        if let Some(members) = map.get("oneOf").or_else(|| map.get("anyOf")) {
            let members = members
                .as_array()
                .context("`oneOf`/`anyOf` is not an array")?;
            return self.union(members, hint, deps);
        }
        if let Some(value) = map.get("const") {
            self.typing.insert("Literal");
            return Ok(format!("Literal[{}]", py_literal(value)?));
        }
        if let Some(values) = map.get("enum").and_then(Value::as_array) {
            let members: Vec<Value> = values
                .iter()
                .map(|v| serde_json::json!({ "const": v }))
                .collect();
            return self.union(&members, hint, deps);
        }
        if let Some([only]) = map
            .get("allOf")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
        {
            return self.type_expr(only, hint, deps);
        }

        let types: Vec<&str> = match map.get("type") {
            None => return Ok(self.any()),
            Some(Value::String(t)) => vec![t.as_str()],
            Some(Value::Array(ts)) => ts.iter().filter_map(Value::as_str).collect(),
            Some(other) => bail!("`{hint}`: unsupported `type` {other}"),
        };
        let mut parts = Vec::new();
        for t in types.iter().filter(|t| **t != "null") {
            parts.push(match *t {
                "integer" => "int".to_string(),
                "number" => "float".to_string(),
                "string" => "str".to_string(),
                "boolean" => "bool".to_string(),
                "array" => {
                    let item = match map.get("items") {
                        Some(items) => self.type_expr(items, hint, deps)?,
                        None => self.any(),
                    };
                    match fixed_len(map) {
                        // A fixed-size Rust array is a serde tuple: the
                        // extension returns a Python tuple, and reads a list
                        // or a tuple.
                        Some(n) => format!(
                            "tuple[{}] | list[{item}]",
                            vec![item.as_str(); n].join(", ")
                        ),
                        None => format!("list[{item}]"),
                    }
                }
                "object" => match map.get("additionalProperties") {
                    Some(values @ Value::Object(_)) => {
                        format!("dict[str, {}]", self.type_expr(values, hint, deps)?)
                    }
                    _ => format!("dict[str, {}]", self.any()),
                },
                other => bail!("`{hint}`: unsupported type `{other}`"),
            });
        }
        if types.contains(&"null") {
            parts.push("None".to_string());
        }
        Ok(parts.join(" | "))
    }

    /// A union of `members`: string constants merge into one `Literal`, an
    /// object member becomes `<hint>_<Variant>`, and `None` goes last.
    fn union(
        &mut self,
        members: &[Value],
        hint: &str,
        deps: &mut BTreeSet<String>,
    ) -> Result<String> {
        let mut literals = Vec::new();
        let mut parts: Vec<Option<String>> = Vec::new();
        let mut nullable = false;
        for member in members {
            if let Some(value) = member.get("const") {
                if literals.is_empty() {
                    parts.push(None); // the merged `Literal` goes here
                }
                literals.push(py_literal(value)?);
            } else if member.get("type").and_then(Value::as_str) == Some("null") {
                nullable = true;
            } else if member.get("properties").is_some() {
                let tag = variant_tag(member).with_context(|| {
                    format!("`{hint}`: a union member with no tag to name it by")
                })?;
                parts.push(Some(self.type_expr(
                    member,
                    &format!("{hint}_{}", pascal(&tag)),
                    deps,
                )?));
            } else {
                parts.push(Some(self.type_expr(member, hint, deps)?));
            }
        }
        let mut rendered: Vec<String> = Vec::new();
        for part in parts {
            let part = match part {
                Some(part) => part,
                None => {
                    self.typing.insert("Literal");
                    format!("Literal[{}]", literals.join(", "))
                }
            };
            if !rendered.contains(&part) {
                rendered.push(part);
            }
        }
        if nullable {
            rendered.push("None".to_string());
        }
        if rendered.is_empty() {
            bail!("`{hint}`: an empty union");
        }
        Ok(rendered.join(" | "))
    }

    fn any(&mut self) -> String {
        self.typing.insert("Any");
        "Any".to_string()
    }

    /// The module: header, imports, every item (a name after the names its
    /// import-time code uses), and `__all__`.
    fn module(&self, header: &str) -> Result<String> {
        let mut out = String::new();
        out.push_str(header.trim_end());
        out.push_str("\n\nfrom __future__ import annotations\n\n");
        let typing: Vec<&str> = self.typing.iter().copied().collect();
        writeln!(out, "from typing import {}", typing.join(", ")).ok();

        let mut emitted = BTreeSet::new();
        let mut visiting = BTreeSet::new();
        for name in self.items.keys() {
            self.emit(name, &mut emitted, &mut visiting, &mut out)?;
        }

        out.push_str("\n\n__all__ = [\n");
        for name in self.items.keys() {
            writeln!(out, "    {},", py_str(name)).ok();
        }
        out.push_str("]\n");
        Ok(out)
    }

    fn emit(
        &self,
        name: &str,
        emitted: &mut BTreeSet<String>,
        visiting: &mut BTreeSet<String>,
        out: &mut String,
    ) -> Result<()> {
        if emitted.contains(name) {
            return Ok(());
        }
        if !visiting.insert(name.to_string()) {
            bail!("`{name}` refers to itself at import time");
        }
        let item = &self.items[name];
        for dep in &item.eager {
            if self.items.contains_key(dep) {
                self.emit(dep, emitted, visiting, out)?;
            }
        }
        out.push_str("\n\n");
        out.push_str(&item.code);
        emitted.insert(name.to_string());
        Ok(())
    }
}

#[derive(Default)]
struct ObjectFields {
    properties: Map<String, Value>,
    required: BTreeSet<String>,
}

struct Field {
    key: String,
    ty: String,
    required: bool,
    doc: Option<String>,
}

/// `class <name>(<bases>):` with its docstring and annotated fields.
fn class_block(out: &mut String, name: &str, bases: &str, doc: Option<&str>, fields: &[Field]) {
    writeln!(out, "class {name}({bases}):").ok();
    if let Some(doc) = doc {
        out.push_str(&docstring(&clean_doc(doc), "    "));
    }
    for field in fields {
        // A blank line after each docstring keeps a field with its own.
        if out.ends_with("\"\"\"\n") {
            out.push('\n');
        }
        writeln!(out, "    {}: {}", field.key, field.ty).ok();
        if let Some(field_doc) = &field.doc {
            out.push_str(&docstring(&clean_doc(field_doc), "    "));
        }
    }
    if fields.is_empty() && doc.is_none() {
        out.push_str("    pass\n");
    }
}

/// Longest line before an alias is wrapped, as Python formatters default to.
const LINE_WIDTH: usize = 88;

/// `<name>: TypeAlias = <ty>`, wrapped one member per line when too long.
fn alias_line(name: &str, ty: &str) -> String {
    let line = format!("{name}: TypeAlias = {ty}\n");
    if line.len() <= LINE_WIDTH + 1 {
        return line;
    }
    let members = split_top_level(ty, " | ");
    if members.len() > 1 {
        return format!(
            "{name}: TypeAlias = (\n    {}\n)\n",
            members.join("\n    | ")
        );
    }
    match ty
        .strip_prefix("Literal[")
        .and_then(|rest| rest.strip_suffix(']'))
    {
        Some(values) => {
            let mut out = format!("{name}: TypeAlias = Literal[\n");
            for value in split_top_level(values, ", ") {
                writeln!(out, "    {value},").ok();
            }
            out.push_str("]\n");
            out
        }
        None => line,
    }
}

/// `text` split at each `separator` outside brackets and string literals.
fn split_top_level<'t>(text: &'t str, separator: &str) -> Vec<&'t str> {
    let bytes = text.as_bytes();
    let mut parts = Vec::new();
    let (mut depth, mut in_string, mut start, mut i) = (0i32, false, 0, 0);
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if in_string => i += 1,
            b'"' => in_string = !in_string,
            b'[' | b'(' if !in_string => depth += 1,
            b']' | b')' if !in_string => depth -= 1,
            _ if !in_string && depth == 0 && text[i..].starts_with(separator) => {
                parts.push(&text[start..i]);
                i += separator.len();
                start = i;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    parts.push(&text[start..]);
    parts
}

/// The length of an array schema whose `minItems` equals its `maxItems`.
fn fixed_len(array: &Map<String, Value>) -> Option<usize> {
    let min = array.get("minItems")?.as_u64()?;
    let max = array.get("maxItems")?.as_u64()?;
    (min == max && min > 0)
        .then(|| usize::try_from(min).ok())
        .flatten()
}

/// The members of a union schema (`oneOf` / `anyOf`).
fn variants(schema: &Value) -> &[Value] {
    schema
        .get("oneOf")
        .or_else(|| schema.get("anyOf"))
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

/// The tag that names an object variant: the value of its string-constant
/// property (serde's internal tag), or its only key (serde's external tag).
fn variant_tag(variant: &Value) -> Option<String> {
    let properties = variant.get("properties")?.as_object()?;
    let internal = properties
        .values()
        .find_map(|p| p.get("const").and_then(Value::as_str));
    match internal {
        Some(tag) => Some(tag.to_string()),
        None if properties.len() == 1 => properties.keys().next().cloned(),
        None => None,
    }
}

fn description(schema: &Value) -> Option<String> {
    schema
        .get("description")
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// Rustdoc markup rewritten for a docstring: an intra-doc link
/// (`` [`Foo`] ``, `` [`Foo`](path) ``) keeps only its text.
fn clean_doc(doc: &str) -> String {
    static LINK: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\[(`[^`\]]+`)\](\([^)\s]*\))?").expect("static regex"));
    LINK.replace_all(doc, "$1").into_owned()
}

/// A triple-quoted docstring, its lines indented by `indent`.
fn docstring(doc: &str, indent: &str) -> String {
    let escaped = doc.replace('\\', "\\\\").replace("\"\"\"", "\\\"\\\"\\\"");
    let escaped = escaped.trim();
    let mut out = String::new();
    let lines: Vec<&str> = escaped.lines().collect();
    if lines.len() <= 1 {
        let body = if escaped.ends_with('"') {
            format!("{escaped} ")
        } else {
            escaped.to_string()
        };
        writeln!(out, "{indent}\"\"\"{body}\"\"\"").ok();
        return out;
    }
    writeln!(out, "{indent}\"\"\"{}", lines[0]).ok();
    for line in &lines[1..] {
        if line.trim().is_empty() {
            out.push('\n');
        } else {
            writeln!(out, "{indent}{}", line.trim_end()).ok();
        }
    }
    writeln!(out, "{indent}\"\"\"").ok();
    out
}

fn py_literal(value: &Value) -> Result<String> {
    Ok(match value {
        Value::String(s) => py_str(s),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => n.to_string(),
        other => bail!("unsupported constant `{other}`"),
    })
}

/// A Python string literal (JSON escapes are valid Python escapes).
fn py_str(s: &str) -> String {
    serde_json::to_string(s).expect("a string serialises to JSON")
}

/// `brown_conrady5` → `BrownConrady5`; `Huber` stays `Huber`.
fn pascal(tag: &str) -> String {
    tag.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            chars.next().map_or_else(String::new, |first| {
                first.to_ascii_uppercase().to_string() + chars.as_str()
            })
        })
        .collect()
}

fn is_identifier(name: &str) -> bool {
    const KEYWORDS: &[&str] = &[
        "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
        "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
        "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return",
        "try", "while", "with", "yield",
    ];
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !KEYWORDS.contains(&name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn render_one(root: &str, schema: &Value) -> String {
        render(
            "\"\"\"Test module.\"\"\"",
            &[PySchema {
                source: "test.json",
                root,
                schema,
                renames: &[],
            }],
            &[],
        )
        .unwrap()
    }

    #[test]
    fn object_with_defs_unions_and_nullable_fields() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Settings",
            "description": "Solver settings.",
            "type": "object",
            "properties": {
                "iters": { "description": "Iteration cap.", "type": "integer", "minimum": 0 },
                "fix": { "description": "Fixed names; `None` keeps the default.", "type": ["array", "null"], "items": { "type": "string" } },
                "loss": { "$ref": "#/$defs/Loss", "default": "None" },
                "mode": { "$ref": "#/$defs/Mode" },
                "pt": { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2 }
            },
            "required": ["iters", "mode"],
            "$defs": {
                "Mode": {
                    "description": "How to run.",
                    "oneOf": [
                        { "const": "fast", "description": "Quick." },
                        { "const": "exact", "description": "Slow but exact." }
                    ]
                },
                "Loss": {
                    "oneOf": [
                        { "const": "None" },
                        {
                            "type": "object",
                            "properties": { "Huber": { "type": "object", "properties": { "scale": { "type": "number" } }, "required": ["scale"] } },
                            "required": ["Huber"]
                        }
                    ]
                }
            }
        });
        let expected = r#""""Test module."""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict


class Loss_Huber(TypedDict):
    Huber: Loss_Huber_Huber


Loss: TypeAlias = Literal["None"] | Loss_Huber


class Loss_Huber_Huber(TypedDict):
    scale: float


Mode: TypeAlias = Literal["fast", "exact"]
"""How to run.

- ``"fast"``: Quick.
- ``"exact"``: Slow but exact.
"""


class _SettingsRequired(TypedDict):
    """The keys of `Settings` that every payload has."""

    iters: int
    """Iteration cap."""

    mode: Mode


class Settings(_SettingsRequired, total=False):
    """Solver settings."""

    fix: list[str] | None
    """Fixed names; `None` keeps the default."""

    loss: Loss
    pt: tuple[float, float] | list[float]


__all__ = [
    "Loss",
    "Loss_Huber",
    "Loss_Huber_Huber",
    "Mode",
    "Settings",
]
"#;
        assert_eq!(render_one("Settings", &schema), expected);
    }

    #[test]
    fn internally_tagged_variants_and_keyword_keys() {
        let schema = json!({
            "description": "A distortion model with its [`Coefficients`](crate::Coefficients).",
            "oneOf": [
                { "type": "object", "properties": { "type": { "const": "none", "type": "string" } }, "required": ["type"] },
                {
                    "type": "object",
                    "properties": {
                        "type": { "const": "division", "type": "string" },
                        "lambda": { "type": "number", "description": "Division coefficient." }
                    },
                    "required": ["type", "lambda"]
                }
            ]
        });
        let expected = r#""""Test module."""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict


Distortion_Division = TypedDict(
    "Distortion_Division",
    {
        # Division coefficient.
        "lambda": float,
        "type": Literal["division"],
    },
)


class Distortion_None(TypedDict):
    type: Literal["none"]


Distortion: TypeAlias = Distortion_None | Distortion_Division
"""A distortion model with its `Coefficients`."""


__all__ = [
    "Distortion",
    "Distortion_Division",
    "Distortion_None",
]
"#;
        assert_eq!(render_one("Distortion", &schema), expected);
    }

    #[test]
    fn ref_merged_variant_takes_the_target_fields() {
        let schema = json!({
            "oneOf": [
                { "$ref": "#/$defs/Cov", "type": "object", "properties": { "status": { "const": "available" } }, "required": ["status"] },
                { "type": "object", "properties": { "status": { "const": "unavailable" } }, "required": ["status"] }
            ],
            "$defs": {
                "Cov": { "type": "object", "properties": { "labels": { "type": "array", "items": { "type": "string" } } }, "required": ["labels"] }
            }
        });
        let out = render_one("Uncertainty", &schema);
        assert!(out.contains(
            "class Uncertainty_Available(TypedDict):\n    labels: list[str]\n    status: Literal[\"available\"]\n"
        ));
        assert!(out.contains(
            "Uncertainty: TypeAlias = Uncertainty_Available | Uncertainty_Unavailable\n"
        ));
    }

    #[test]
    fn renames_and_conflicts() {
        let view = |meta: &str| {
            json!({
                "type": "object",
                "properties": { "views": { "type": "array", "items": { "$ref": "#/$defs/View" } } },
                "required": ["views"],
                "$defs": {
                    "View": { "type": "object", "properties": { "meta": { "$ref": format!("#/$defs/{meta}") } }, "required": ["meta"] },
                    "A": { "type": "null" },
                    "B": { "type": "string" },
                }
            })
        };
        let (a, b) = (view("A"), view("B"));
        let entry = |root, schema, renames| PySchema {
            source: root,
            root,
            schema,
            renames,
        };

        let err = render("", &[entry("InA", &a, &[]), entry("InB", &b, &[])], &[]).unwrap_err();
        assert!(
            err.to_string()
                .contains("`View` has one schema in InA and another in InB")
        );

        let out = render(
            "",
            &[
                entry("InA", &a, &[("View", "ViewA")]),
                entry("InB", &b, &[("View", "ViewB")]),
            ],
            &[],
        )
        .unwrap();
        assert!(out.contains("class InA(TypedDict):\n    views: list[ViewA]\n"));
        assert!(out.contains("class ViewB(TypedDict):\n    meta: B\n"));
        assert!(out.contains("A: TypeAlias = None\n"));

        let err = render("", &[entry("InA", &a, &[("Missing", "X")])], &[]).unwrap_err();
        assert!(err.to_string().contains("rename of `Missing`"));
    }

    #[test]
    fn docstrings_escape_and_indent() {
        assert_eq!(docstring("one line", "    "), "    \"\"\"one line\"\"\"\n");
        assert_eq!(
            docstring("ends \"quoted\"", ""),
            "\"\"\"ends \"quoted\" \"\"\"\n"
        );
        assert_eq!(
            docstring("a \\ b \"\"\" c\n\nmore", "    "),
            "    \"\"\"a \\\\ b \\\"\\\"\\\" c\n\n    more\n    \"\"\"\n"
        );
        assert_eq!(pascal("brown_conrady5"), "BrownConrady5");
        assert_eq!(
            alias_line("Long", &format!("Literal[{}]", ["\"a, b\""; 12].join(", "))),
            format!(
                "Long: TypeAlias = Literal[\n{}]\n",
                "    \"a, b\",\n".repeat(12)
            )
        );
        assert_eq!(
            alias_line("U", &["list[A | None]"; 6].join(" | ")),
            format!(
                "U: TypeAlias = (\n    {}\n)\n",
                ["list[A | None]"; 6].join("\n    | ")
            )
        );
        assert!(!is_identifier("lambda"));
        assert!(is_identifier("type"));
    }
}
