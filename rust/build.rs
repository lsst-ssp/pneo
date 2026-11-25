//! Bake the package version into the extension as `pneo._core.__version__`.
//!
//! The crate carries no version of its own (`publish = false`); the package
//! version is the git tag, resolved by setuptools-scm.  maturin cannot run
//! setuptools-scm, so the release workflow stamps the resolved version into
//! `pyproject.toml` as a static `[project] version = "..."` before building
//! the wheel.  This script reads that line, so the wheel's extension reports
//! the same version as the Python package and the release smoke test can
//! check that the two came from the same build.
//!
//! Builds from a checkout (`rust/build.sh`, `cargo test`) see
//! `dynamic = ["version"]` instead and report `0+unknown`.

use std::fs;
use std::path::Path;

fn project_version(pyproject: &str) -> Option<String> {
    let mut in_project = false;
    for line in pyproject.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_project = t == "[project]";
            continue;
        }
        if in_project {
            if let Some(rest) = t.strip_prefix("version") {
                let rest = rest.trim_start();
                if let Some(val) = rest.strip_prefix('=') {
                    let val = val.trim().trim_matches('"').trim_matches('\'');
                    if !val.is_empty() {
                        return Some(val.to_string());
                    }
                }
            }
        }
    }
    None
}

fn main() {
    let pyproject = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("pyproject.toml");
    println!("cargo:rerun-if-changed={}", pyproject.display());
    let version = fs::read_to_string(&pyproject)
        .ok()
        .and_then(|s| project_version(&s))
        .unwrap_or_else(|| "0+unknown".to_string());
    println!("cargo:rustc-env=PNEO_VERSION={version}");
}
