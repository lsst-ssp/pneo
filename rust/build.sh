#!/usr/bin/env bash
# Build the pneo Rust core kernel and drop the extension module into the
# Python package.
#
# Layout choice (documented in rust/README.md): the crate is a plain
# cargo/pyo3 `cdylib` and this script copies the shared library to
# `src/pneo/_core.abi3.so`.  The repository's Python package is installed
# editable with setuptools, so `src/pneo/` IS the import location and a
# rebuild is picked up with no reinstall.  A maturin `develop`-style install
# would have to own the whole `pneo` package and would fight the existing
# editable install, hence this two-line copy instead.
#
# maturin is deliberately NOT used: driven from this crate alone it names the
# wheel's module `pneo_core`, not `pneo._core`, and configuring it to
# emit `pneo/_core` would make the wheel claim the whole `pneo` package
# and collide with the editable install.  A `pip`-installable wheel is a
# packaging task for the repo root, not for this crate.
#
# Usage:
#   rust/build.sh              # release build (default)
#   rust/build.sh --debug      # debug build (much slower kernel)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"

profile="release"
for arg in "$@"; do
    case "$arg" in
        --debug) profile="debug" ;;
        --release) profile="release" ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

# Platform differences.  On macOS the shared library is a .dylib, and a Python
# extension must leave the interpreter's symbols unresolved at link time
# (Python provides them when it loads the module), which needs
# `-undefined dynamic_lookup`.  maturin passes that flag itself; a plain
# `cargo build` does not.  Linux needs neither.
case "$(uname -s)" in
    Darwin)
        libname="libpneo_core.dylib"
        export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C link-arg=-undefined -C link-arg=dynamic_lookup"
        ;;
    *)
        libname="libpneo_core.so"
        ;;
esac

if [[ "$profile" == "release" ]]; then
    cargo build --release --manifest-path "$here/Cargo.toml"
else
    cargo build --manifest-path "$here/Cargo.toml"
fi

src="$here/target/$profile/$libname"
dst="$root/src/pneo/_core.abi3.so"
# Install by rename, not by overwrite.  `cp` truncates and rewrites the target
# in place, which corrupts the mapping of any process that already has the old
# library loaded (a long test run, a batch job).  `mv` within the same
# directory is an atomic rename: new arrivals get the new inode, existing
# mappings keep the old one until they exit.
tmp="$(mktemp "$dst.XXXXXX")"
trap 'rm -f "$tmp"' EXIT   # no stray temp file if the copy fails
cp -f "$src" "$tmp"
chmod 755 "$tmp"
mv -f "$tmp" "$dst"
echo "installed $dst"

python3 -c "
import pneo._core as c
print('pneo._core', c.__version__, '| COND_MAX =', c.COND_MAX)
"
