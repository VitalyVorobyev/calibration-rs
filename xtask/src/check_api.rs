//! `check-api`: a public type that can grow is `#[non_exhaustive]`.
//!
//! A `pub struct` whose named fields are all `pub` can be built by struct
//! literal and destructured, and a `pub enum` can be matched exhaustively;
//! either one then cannot gain a field or variant without breaking its users. The check
//! walks the library module tree of every published crate and reports each
//! such type that is neither `#[non_exhaustive]` nor `#[doc(hidden)]` (on the
//! item or an enclosing module) nor named in [`ALLOWLIST`], which lists the
//! types whose shape is fixed by definition, each with its reason.
//!
//! Skipped: items under `#[cfg(test)]`, restricted visibility (`pub(crate)`,
//! `pub(super)`, `pub(in …)`), unit and tuple structs, structs with a private
//! field (already opaque to other crates, so growable), and binary targets.
//! A type in a private module whose only re-export from a public module is
//! `#[doc(hidden)]` counts as hidden. Re-exports are matched by name within
//! the crate (globs are not resolved), so a `pub` type in a private module
//! that nothing re-exports still counts: narrow it to `pub(crate)`.
//!
//! An allowlist entry that matches no reported type, matches several, or has
//! an empty reason fails the check too.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use syn::punctuated::Punctuated;
use syn::{Attribute, Fields, Item, Meta, Token, UseTree, Visibility};

use crate::check_docs::{published_crates, rel};

/// The exhaustive public types, keyed `crate::Type` (or
/// `crate::module::Type` when a crate has two of the name), with the reason
/// their shape cannot grow.
const ALLOWLIST: &str = "xtask/api_exhaustive.toml";

/// An exhaustive public type.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Hit {
    /// The crate's library name (`vision_calibration_core`).
    krate: String,
    /// Module path inside the crate.
    module: Vec<String>,
    name: String,
    file: PathBuf,
    line: usize,
}

impl Hit {
    fn short_key(&self) -> String {
        format!("{}::{}", self.krate, self.name)
    }

    fn full_key(&self) -> String {
        let mut parts = vec![self.krate.as_str()];
        parts.extend(self.module.iter().map(String::as_str));
        parts.push(&self.name);
        parts.join("::")
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    reason: String,
}

pub fn run(root: &Path) -> Result<()> {
    let crates = published_crates(root)?;
    let mut hits = Vec::new();
    for krate in &crates {
        hits.extend(scan_crate(krate)?);
    }
    hits.sort_by(|a, b| (&a.file, a.line).cmp(&(&b.file, b.line)));

    let text =
        fs::read_to_string(root.join(ALLOWLIST)).with_context(|| format!("reading {ALLOWLIST}"))?;
    let allow = parse_allowlist(&text)?;
    let (unlisted, problems) = check(&hits, &allow);

    if unlisted.is_empty() && problems.is_empty() {
        let todo = allow
            .values()
            .filter(|e| e.reason.starts_with("TODO(review)"))
            .count();
        println!(
            "check-api: {} crates, {} exhaustive public types, all allowlisted ({todo} TODO(review))",
            crates.len(),
            hits.len()
        );
        return Ok(());
    }
    for hit in &unlisted {
        eprintln!(
            "{}:{}  {}  — add #[non_exhaustive] with a constructor, or allowlist it with a reason",
            rel(root, &hit.file),
            hit.line,
            hit.full_key()
        );
    }
    for problem in &problems {
        eprintln!("{ALLOWLIST}: {problem}");
    }
    bail!(
        "check-api: {} exhaustive public type(s) not allowlisted, {} allowlist problem(s)",
        unlisted.len(),
        problems.len()
    )
}

fn parse_allowlist(text: &str) -> Result<BTreeMap<String, Entry>> {
    toml::from_str(text).with_context(|| format!("parsing {ALLOWLIST}"))
}

/// The hits no allowlist entry covers, and the allowlist's own problems.
fn check<'a>(hits: &'a [Hit], allow: &BTreeMap<String, Entry>) -> (Vec<&'a Hit>, Vec<String>) {
    let mut problems = Vec::new();
    let mut covered = vec![false; hits.len()];
    for (key, entry) in allow {
        if entry.reason.trim().is_empty() {
            problems.push(format!("[empty reason] `{key}`"));
        }
        let matching: Vec<usize> = (0..hits.len())
            .filter(|&i| hits[i].full_key() == *key || hits[i].short_key() == *key)
            .collect();
        match matching.as_slice() {
            [] => problems.push(format!("[stale entry] `{key}` matches no reported type")),
            [i] => covered[*i] = true,
            many => problems.push(format!(
                "[ambiguous entry] `{key}` matches {}; key it by module path",
                many.iter()
                    .map(|&i| hits[i].full_key())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }
    let unlisted = hits
        .iter()
        .zip(&covered)
        .filter(|(_, c)| !**c)
        .map(|(h, _)| h)
        .collect();
    (unlisted, problems)
}

/// Every exhaustive public type of the crate at `dir`, walking the module
/// tree from `src/lib.rs`.
fn scan_crate(dir: &Path) -> Result<Vec<Hit>> {
    let lib = dir.join("src/lib.rs");
    if !lib.is_file() {
        return Ok(Vec::new());
    }
    let name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .context("crate directory name")?
        .replace('-', "_");
    let scope = Scope {
        krate: &name,
        module: Vec::new(),
        hidden: false,
        reachable: true,
        file: &lib,
        child_dir: Some(dir.join("src")),
    };
    let mut crawl = Crawl::default();
    scan_file(&lib, &scope, &mut crawl)?;
    Ok(crawl.finish())
}

/// Where the scanner is in a crate.
struct Scope<'a> {
    krate: &'a str,
    module: Vec<String>,
    /// Inside a `#[doc(hidden)]` module.
    hidden: bool,
    /// Every enclosing module is `pub`: the items here are public at their
    /// own path.
    reachable: bool,
    file: &'a Path,
    /// Where `mod x;` looks for `x.rs` / `x/mod.rs`; `None` scans one file
    /// alone (tests).
    child_dir: Option<PathBuf>,
}

impl Scope<'_> {
    fn enter<'b>(&self, m: &syn::ItemMod, file: &'b Path, child_dir: Option<PathBuf>) -> Scope<'b>
    where
        Self: 'b,
    {
        let mut module = self.module.clone();
        module.push(m.ident.to_string());
        Scope {
            krate: self.krate,
            module,
            hidden: self.hidden || is_doc_hidden(&m.attrs),
            reachable: self.reachable && matches!(m.vis, Visibility::Public(_)),
            file,
            child_dir,
        }
    }
}

/// What a walk of one crate collects.
#[derive(Default)]
struct Crawl {
    /// Exhaustive `pub` types, each with whether its module chain is `pub`.
    hits: Vec<(Hit, bool)>,
    /// Names a `#[doc(hidden)] pub use` in a public module re-exports.
    hidden_uses: BTreeSet<String>,
    /// Names a visible `pub use` in a public module re-exports.
    open_uses: BTreeSet<String>,
}

impl Crawl {
    /// The hits, less those whose only public path is a hidden re-export
    /// (a type in a private module, re-exported `#[doc(hidden)]`).
    fn finish(self) -> Vec<Hit> {
        let Crawl {
            hits,
            hidden_uses,
            open_uses,
        } = self;
        hits.into_iter()
            .filter(|(h, reachable)| {
                *reachable || !hidden_uses.contains(&h.name) || open_uses.contains(&h.name)
            })
            .map(|(h, _)| h)
            .collect()
    }
}

fn scan_file(path: &Path, scope: &Scope, crawl: &mut Crawl) -> Result<()> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let file = syn::parse_file(&text).with_context(|| format!("parsing {}", path.display()))?;
    scan_items(&file.items, scope, crawl)
}

fn scan_items(items: &[Item], scope: &Scope, crawl: &mut Crawl) -> Result<()> {
    for item in items {
        match item {
            Item::Struct(s) => {
                // A private field already keeps other crates from building
                // the struct by literal or destructuring it without `..`.
                let open = matches!(&s.fields, Fields::Named(f)
                    if f.named.iter().all(|f| matches!(f.vis, Visibility::Public(_))));
                if open && reportable(&s.vis, &s.attrs, scope) {
                    crawl.hits.push((hit(scope, &s.ident), scope.reachable));
                }
            }
            Item::Enum(e) => {
                if reportable(&e.vis, &e.attrs, scope) {
                    crawl.hits.push((hit(scope, &e.ident), scope.reachable));
                }
            }
            Item::Use(u) => {
                if matches!(u.vis, Visibility::Public(_))
                    && scope.reachable
                    && !is_cfg_test(&u.attrs)
                {
                    let names = if scope.hidden || is_doc_hidden(&u.attrs) {
                        &mut crawl.hidden_uses
                    } else {
                        &mut crawl.open_uses
                    };
                    use_names(&u.tree, names);
                }
            }
            Item::Mod(m) => {
                if is_cfg_test(&m.attrs) {
                    continue;
                }
                let name = m.ident.to_string();
                match &m.content {
                    Some((_, inner)) => {
                        let child_dir = scope.child_dir.as_ref().map(|d| d.join(&name));
                        let child = scope.enter(m, scope.file, child_dir);
                        scan_items(inner, &child, crawl)?;
                    }
                    None => {
                        let Some(dir) = &scope.child_dir else {
                            continue;
                        };
                        let (path, child_dir) = module_file(&m.attrs, scope.file, dir, &name)?;
                        let child = scope.enter(m, &path, Some(child_dir));
                        scan_file(&path, &child, crawl)?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// The original names a `use` tree imports (globs are not resolved).
fn use_names(tree: &UseTree, out: &mut BTreeSet<String>) {
    match tree {
        UseTree::Path(p) => use_names(&p.tree, out),
        UseTree::Name(n) => {
            out.insert(n.ident.to_string());
        }
        UseTree::Rename(r) => {
            out.insert(r.ident.to_string());
        }
        UseTree::Group(g) => g.items.iter().for_each(|t| use_names(t, out)),
        UseTree::Glob(_) => {}
    }
}

/// The file of `mod name;` and the directory its own `mod` items look in.
fn module_file(
    attrs: &[Attribute],
    current: &Path,
    dir: &Path,
    name: &str,
) -> Result<(PathBuf, PathBuf)> {
    if let Some(path) = path_attr(attrs) {
        let base = current.parent().context("source file has no parent")?;
        let file = base.join(path);
        let child_dir = file
            .parent()
            .context("module file has no parent")?
            .to_path_buf();
        return Ok((file, child_dir));
    }
    let flat = dir.join(format!("{name}.rs"));
    if flat.is_file() {
        return Ok((flat, dir.join(name)));
    }
    let nested = dir.join(name).join("mod.rs");
    if nested.is_file() {
        return Ok((nested, dir.join(name)));
    }
    bail!("module `{name}` not found under {}", dir.display())
}

fn path_attr(attrs: &[Attribute]) -> Option<String> {
    attrs.iter().find_map(|a| match &a.meta {
        Meta::NameValue(nv) if nv.path.is_ident("path") => match &nv.value {
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(s),
                ..
            }) => Some(s.value()),
            _ => None,
        },
        _ => None,
    })
}

fn hit(scope: &Scope, ident: &syn::Ident) -> Hit {
    Hit {
        krate: scope.krate.to_string(),
        module: scope.module.clone(),
        name: ident.to_string(),
        file: scope.file.to_path_buf(),
        line: ident.span().start().line,
    }
}

/// A plainly `pub`, visible, exhaustive item outside test code.
fn reportable(vis: &Visibility, attrs: &[Attribute], scope: &Scope) -> bool {
    matches!(vis, Visibility::Public(_))
        && !scope.hidden
        && !is_cfg_test(attrs)
        && !is_doc_hidden(attrs)
        && !metas(attrs)
            .iter()
            .any(|m| m.path().is_ident("non_exhaustive"))
}

/// The attributes' metas, with each `cfg_attr(pred, a, b)` expanded to
/// `a, b`.
fn metas(attrs: &[Attribute]) -> Vec<Meta> {
    let mut out = Vec::new();
    for attr in attrs {
        expand(attr.meta.clone(), &mut out);
    }
    out
}

fn expand(meta: Meta, out: &mut Vec<Meta>) {
    if let Meta::List(list) = &meta
        && list.path.is_ident("cfg_attr")
        && let Ok(args) = list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
    {
        for inner in args.into_iter().skip(1) {
            expand(inner, out);
        }
        return;
    }
    out.push(meta);
}

fn is_doc_hidden(attrs: &[Attribute]) -> bool {
    metas(attrs).iter().any(|m| match m {
        Meta::List(l) if l.path.is_ident("doc") => {
            idents(l.tokens.clone()).iter().any(|i| i == "hidden")
        }
        _ => false,
    })
}

/// `#[cfg(test)]`, or a `cfg` that requires `test` (`all(test, …)`).
fn is_cfg_test(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| match &a.meta {
        Meta::List(l) if l.path.is_ident("cfg") => {
            let ids = idents(l.tokens.clone());
            ids.iter().any(|i| i == "test") && !ids.iter().any(|i| i == "not" || i == "any")
        }
        _ => false,
    })
}

fn idents(tokens: proc_macro2::TokenStream) -> Vec<String> {
    let mut out = Vec::new();
    for tt in tokens {
        match tt {
            proc_macro2::TokenTree::Ident(i) => out.push(i.to_string()),
            proc_macro2::TokenTree::Group(g) => out.extend(idents(g.stream())),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(src: &str) -> Vec<String> {
        hits(src).iter().map(Hit::full_key).collect()
    }

    #[test]
    fn plain_pub_struct_with_pub_field_is_reported() {
        let src =
            "/// Doc.\n#[derive(Debug)]\npub struct Point {\n    pub x: f64,\n    pub y: f64,\n}\n";
        assert_eq!(scan(src), ["k::Point"]);
    }

    #[test]
    fn exhaustive_pub_enum_is_reported_even_with_non_exhaustive_variants() {
        let src = "pub enum Mode { #[non_exhaustive] A { x: u8 }, B }\n";
        assert_eq!(scan(src), ["k::Mode"]);
    }

    #[test]
    fn growable_hidden_and_private_shapes_are_skipped() {
        let src = r#"
            #[non_exhaustive]
            pub struct A { pub x: f64 }
            #[derive(
                Debug,
                Clone,
            )]
            #[cfg_attr(
                feature = "serde",
                derive(Serialize),
                non_exhaustive
            )]
            pub struct B { pub x: f64 }
            #[doc(hidden)]
            pub struct C { pub x: f64 }
            pub struct D { x: f64 }
            pub struct D2 { pub x: f64, y: f64 }
            pub(crate) struct E { pub x: f64 }
            pub(super) enum F { A }
            pub struct G(pub f64);
            pub struct H;
            #[non_exhaustive]
            pub enum I { A }
            #[cfg(test)]
            mod tests {
                pub struct J { pub x: f64 }
            }
            #[doc(hidden)]
            pub mod internal {
                pub struct K { pub x: f64 }
            }
            #[cfg(test)]
            pub struct L { pub x: f64 }
        "#;
        assert!(scan(src).is_empty(), "{:?}", scan(src));
    }

    #[test]
    fn items_in_inline_modules_carry_their_path() {
        let src = "mod private { pub enum E { A } }\npub mod m { pub(crate) mod n { pub struct S { pub a: u8 } } }\n";
        assert_eq!(scan(src), ["k::private::E", "k::m::n::S"]);
    }

    fn allow(entries: &[(&str, &str)]) -> BTreeMap<String, Entry> {
        entries
            .iter()
            .map(|(k, r)| {
                (
                    k.to_string(),
                    Entry {
                        reason: r.to_string(),
                    },
                )
            })
            .collect()
    }

    fn hits(src: &str) -> Vec<Hit> {
        let file = syn::parse_file(src).expect("fixture parses");
        let scope = Scope {
            krate: "k",
            module: Vec::new(),
            hidden: false,
            reachable: true,
            file: Path::new("lib.rs"),
            child_dir: None,
        };
        let mut crawl = Crawl::default();
        scan_items(&file.items, &scope, &mut crawl).expect("scan");
        crawl.finish()
    }

    #[test]
    fn a_hidden_re_export_hides_a_type_of_a_private_module() {
        let src = r#"
            mod inner {
                pub struct A { pub x: u8 }
                pub struct B { pub x: u8 }
                pub struct C { pub x: u8 }
            }
            #[doc(hidden)]
            pub use inner::{A, B as Bee};
            #[doc(hidden)]
            pub use inner::C;
            pub use inner::C as See;
            pub mod open {
                pub struct D { pub x: u8 }
            }
            #[doc(hidden)]
            pub use open::D;
        "#;
        assert_eq!(scan(src), ["k::inner::C", "k::open::D"]);
    }

    #[test]
    fn allowlisted_types_pass() {
        let hits = hits("pub struct P { pub x: f64 }\nmod m { pub enum E { A } }\n");
        let (unlisted, problems) = check(
            &hits,
            &allow(&[("k::P", "two coordinates"), ("k::m::E", "one case")]),
        );
        assert!(unlisted.is_empty() && problems.is_empty(), "{problems:?}");
    }

    #[test]
    fn stale_empty_and_ambiguous_entries_fail() {
        let hits = hits(
            "pub struct P { pub x: f64 }\nmod a { pub enum E { A } }\nmod b { pub enum E { A } }\n",
        );
        let (unlisted, problems) = check(
            &hits,
            &allow(&[("k::P", " "), ("k::Gone", "was here"), ("k::E", "one case")]),
        );
        assert_eq!(unlisted.len(), 2, "the ambiguous key covers neither E");
        assert_eq!(problems.len(), 3, "{problems:?}");
        assert!(problems.iter().any(|p| p.contains("[empty reason] `k::P`")));
        assert!(
            problems
                .iter()
                .any(|p| p.contains("[stale entry] `k::Gone`"))
        );
        assert!(
            problems
                .iter()
                .any(|p| p.contains("[ambiguous entry] `k::E`"))
        );
    }

    #[test]
    fn allowlist_needs_a_reason_field() {
        assert!(parse_allowlist("[\"k::P\"]\nreason = \"two coordinates\"\n").is_ok());
        assert!(parse_allowlist("[\"k::P\"]\n").is_err());
        assert!(parse_allowlist("[\"k::P\"]\nreason = \"x\"\nwhy = \"y\"\n").is_err());
    }
}
