//! With `embed-dist`, the built frontend (`apps/studio/dist`) goes into the
//! binary. A release build without it fails: a server that ships without
//! its page is broken. A debug build does not (the repository carries a
//! placeholder `index.html`, so `cargo test` and `cargo clippy` work before
//! `bun run build`); it serves what is on disk then.

fn main() {
    println!("cargo:rerun-if-changed=../../dist/index.html");
    println!("cargo:rerun-if-env-changed=PROFILE");
    if std::env::var_os("CARGO_FEATURE_EMBED_DIST").is_none() {
        return;
    }
    let dist = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../dist");
    if dist.join("index.html").is_file() && dist.join("assets").is_dir() {
        return;
    }
    let message = format!(
        "the frontend is not built ({} has no assets/): run `bun run build` in apps/studio/",
        dist.display()
    );
    if std::env::var("PROFILE").as_deref() == Ok("release") {
        panic!("{message}");
    }
    // A debug build serves the placeholder; the server says so when it
    // starts.
}
