//! `check-docs`: user-facing documentation carries no internal references.
//!
//! User-facing means the root README and CHANGELOG, the app README, the
//! book, the tutorials, and — for every crate that is published — its
//! README, its rustdoc (`//!` and `///` lines) and, for the Python package,
//! its Python sources. Dev-facing material (ADRs, proof notes, the roadmap,
//! the backlog, agent files) may be cited only from dev-facing docs.
//!
//! The check also keeps install snippets (`name = "X.Y"`) on the workspace's
//! current minor version.

use anyhow::{Context, Result, bail};
use regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};

/// One forbidden pattern and the reason it is forbidden.
struct Rule {
    name: &'static str,
    pattern: Regex,
}

fn rules() -> Vec<Rule> {
    let rule = |name, pattern: &str| Rule {
        name,
        pattern: Regex::new(pattern).expect("static regex"),
    };
    vec![
        rule("ADR reference", r"\bADRs?\b"),
        rule(
            "dev-docs path",
            r"(docs/|\.\./)(adrs|notes|internal)\b|\bbacklog\.md|\bROADMAP\.md",
        ),
        rule("private dataset", r"privatedata|rtv3d|examples-private"),
        rule(
            "backlog id",
            r"\b[A-Z][A-Z0-9]*-[A-Z][A-Z0-9]+(-[A-Z0-9]+)*\b",
        ),
        rule(
            "agent workflow",
            r"AGENTS\.md|CLAUDE\.md|\.claude/|\bcodex\b",
        ),
        rule("PR number", r"\(#\d+\)|\bPR ?#\d+|/pull/\d+"),
        rule("track id", r"\bTrack [A-Z]\b"),
    ]
}

/// A user-facing file and which of its lines are documentation.
struct DocFile {
    path: PathBuf,
    /// Rust sources: only `//!` and `///` lines are user-facing.
    rustdoc_only: bool,
}

pub fn run(root: &Path) -> Result<()> {
    let files = user_facing_files(root)?;
    let rules = rules();
    let mut findings = Vec::new();

    for file in &files {
        let text = fs::read_to_string(&file.path)
            .with_context(|| format!("reading {}", file.path.display()))?;
        for (idx, line) in text.lines().enumerate() {
            if file.rustdoc_only && !is_rustdoc(line) {
                continue;
            }
            for rule in &rules {
                if rule.pattern.is_match(line) {
                    findings.push(format!(
                        "{}:{}: [{}] {}",
                        rel(root, &file.path),
                        idx + 1,
                        rule.name,
                        line.trim()
                    ));
                }
            }
        }
    }

    findings.extend(check_install_versions(root, &files)?);

    if findings.is_empty() {
        println!("check-docs: {} user-facing files clean", files.len());
        return Ok(());
    }
    for finding in &findings {
        eprintln!("{finding}");
    }
    bail!(
        "check-docs: {} finding(s); user-facing docs must not cite internal material",
        findings.len()
    )
}

fn is_rustdoc(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("//!") || t.starts_with("///")
}

fn user_facing_files(root: &Path) -> Result<Vec<DocFile>> {
    let mut files = Vec::new();
    let mut push_md = |path: PathBuf| {
        files.push(DocFile {
            path,
            rustdoc_only: false,
        })
    };
    for name in ["README.md", "CHANGELOG.md", "app/README.md"] {
        push_md(root.join(name));
    }
    for dir in ["book/src", "docs/tutorials"] {
        for path in walk(&root.join(dir), &["md"])? {
            push_md(path);
        }
    }

    for krate in published_crates(root)? {
        let readme = krate.join("README.md");
        if readme.is_file() {
            files.push(DocFile {
                path: readme,
                rustdoc_only: false,
            });
        }
        for path in walk(&krate.join("src"), &["rs"])? {
            files.push(DocFile {
                path,
                rustdoc_only: true,
            });
        }
        for path in walk(&krate.join("python"), &["py", "pyi"])? {
            files.push(DocFile {
                path,
                rustdoc_only: false,
            });
        }
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(files)
}

/// Crates under `crates/` whose manifest does not say `publish = false`.
fn published_crates(root: &Path) -> Result<Vec<PathBuf>> {
    let mut crates = Vec::new();
    for entry in fs::read_dir(root.join("crates")).context("reading crates/")? {
        let dir = entry?.path();
        let manifest = dir.join("Cargo.toml");
        if !manifest.is_file() {
            continue;
        }
        let text = fs::read_to_string(&manifest)?;
        if !text.lines().any(|l| l.trim() == "publish = false") {
            crates.push(dir);
        }
    }
    crates.sort();
    Ok(crates)
}

/// Every file under `dir` with one of `exts`; empty if `dir` is absent.
fn walk(dir: &Path, exts: &[&str]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if !dir.is_dir() {
        return Ok(out);
    }
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in fs::read_dir(&d).with_context(|| format!("reading {}", d.display()))? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| exts.contains(&e))
            {
                out.push(path);
            }
        }
    }
    Ok(out)
}

/// Install snippets for workspace crates (`vision-… = "X.Y"`) must name the
/// workspace's current `X.Y`.
fn check_install_versions(root: &Path, files: &[DocFile]) -> Result<Vec<String>> {
    let manifest = fs::read_to_string(root.join("Cargo.toml"))?;
    let version = workspace_minor(&manifest).context("[workspace.package] version not found")?;
    let snippet = install_snippet();

    let mut findings = Vec::new();
    for file in files.iter().filter(|f| !f.rustdoc_only) {
        if file.path.ends_with("CHANGELOG.md") {
            continue;
        }
        let text = fs::read_to_string(&file.path)?;
        for (idx, line) in text.lines().enumerate() {
            if let Some(c) = snippet.captures(line)
                && c[2] != version
            {
                findings.push(format!(
                    "{}:{}: [stale version] `{}` pins {}, workspace is {version}",
                    rel(root, &file.path),
                    idx + 1,
                    &c[1],
                    &c[2]
                ));
            }
        }
    }
    Ok(findings)
}

/// `X.Y` of `[workspace.package] version = "X.Y.Z"`.
fn workspace_minor(manifest: &str) -> Option<String> {
    Regex::new(r#"(?m)^\[workspace\.package\][^\[]*?^version = "(\d+)\.(\d+)\."#)
        .expect("static regex")
        .captures(manifest)
        .map(|c| format!("{}.{}", &c[1], &c[2]))
}

/// A `vision-… = "X.Y"` (or `{ version = "X.Y" … }`) install line; captures
/// the crate name and the pinned `X.Y`.
fn install_snippet() -> Regex {
    Regex::new(r#"^\s*(vision-[a-z-]+)\s*=\s*(?:\{\s*version\s*=\s*)?"(\d+\.\d+)"#)
        .expect("static regex")
}

fn rel(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flagged_by(line: &str) -> Vec<&'static str> {
        rules()
            .into_iter()
            .filter(|r| r.pattern.is_match(line))
            .map(|r| r.name)
            .collect()
    }

    #[test]
    fn each_rule_flags_its_leak() {
        let cases = [
            ("See ADR 0012 for the schema.", "ADR reference"),
            ("[proof](../notes/rig-extrinsics.md)", "dev-docs path"),
            ("details in docs/internal/plan.md", "dev-docs path"),
            ("measured on privatedata/rtv3d", "private dataset"),
            ("tracked as P3-BACKEND-COST", "backlog id"),
            ("blocked by CHORE-DEPS", "backlog id"),
            ("see AGENTS.md for the workflow", "agent workflow"),
            ("landed in (#123)", "PR number"),
            ("part of Track B", "track id"),
        ];
        for (line, rule) in cases {
            assert!(flagged_by(line).contains(&rule), "{rule} missed: {line}");
        }
    }

    #[test]
    fn ordinary_prose_is_clean() {
        for line in [
            "Levenberg-Marquardt with a Brown-Conrady model and SE(3) poses.",
            "The `puzzle_130x130` layout and N-view triangulation.",
            "UTF-8 paths, an RGB-D sensor, and the Gauss-Newton step.",
            "Release notes live in the CHANGELOG.",
        ] {
            assert!(flagged_by(line).is_empty(), "false positive: {line}");
        }
    }

    #[test]
    fn install_snippets_carry_the_workspace_minor() {
        let manifest = "[workspace]\nmembers = []\n\n[workspace.package]\nversion = \"0.9.0\"\n";
        assert_eq!(workspace_minor(manifest).as_deref(), Some("0.9"));

        let snippet = install_snippet();
        let pin = |line: &str| snippet.captures(line).map(|c| c[2].to_string());
        assert_eq!(pin(r#"vision-calibration = "0.8""#).as_deref(), Some("0.8"));
        assert_eq!(
            pin(r#"vision-mvg = { version = "0.9", features = ["refine"] }"#).as_deref(),
            Some("0.9")
        );
        assert_eq!(pin(r#"nalgebra = "0.34""#), None);
    }
}
