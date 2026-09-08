#!/usr/bin/env bash
# Build the ds42 Rust core kernel and drop the extension module into the
# Python package.
#
# Layout choice (documented in rust/README.md): the crate is a plain
# cargo/pyo3 `cdylib` and this script copies the shared library to
# `src/ds42/_core.abi3.so`.  The repository's Python package is installed
# editable with setuptools, so `src/ds42/` IS the import location and a
# rebuild is picked up with no reinstall.  A maturin `develop`-style install
# would have to own the whole `ds42` package and would fight the existing
# editable install, hence this two-line copy instead.
#
# maturin is deliberately NOT used: driven from this crate alone it names the
# wheel's module `ds42_core`, not `ds42._core`, and configuring it to
# emit `ds42/_core` would make the wheel claim the whole `ds42` package
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

if [[ "$profile" == "release" ]]; then
    cargo build --release --manifest-path "$here/Cargo.toml"
else
    cargo build --manifest-path "$here/Cargo.toml"
fi

src="$here/target/$profile/libds42_core.so"
dst="$root/src/ds42/_core.abi3.so"
# Install by rename, not by overwrite.  `cp` truncates and rewrites the target
# in place, which corrupts the mapping of any process that already has the old
# library loaded (a long test run, a batch job).  `mv` within the same
# directory is an atomic rename: new arrivals get the new inode, existing
# mappings keep the old one until they exit.
tmp="$(mktemp "$dst.XXXXXX")"
cp -f "$src" "$tmp"
chmod 755 "$tmp"
mv -f "$tmp" "$dst"
echo "installed $dst"

python3 -c "
import ds42._core as c
print('ds42._core', c.__version__, '| COND_MAX =', c.COND_MAX)
"
