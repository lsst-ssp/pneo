# Releasing pneo

`pneo` is published to PyPI by `.github/workflows/release.yml`, triggered by
**pushing a tag matching `vX.Y.Z`** — `git push origin v0.1.0` is the release
action; no GitHub Release object is required (create one afterward for notes if
you like; it does not re-trigger the workflow). Publishing uses PyPI
**Trusted Publishing** (OIDC): there is no API token in this repository, its
secrets, or anyone's keychain.

- [What gets published](#what-gets-published)
- [Cutting a release](#cutting-a-release)
- [Rehearsing without publishing](#rehearsing-without-publishing)
- [The version rule](#the-version-rule)
- [Platform coverage, and how to extend it](#platform-coverage-and-how-to-extend-it)
- [The one-time PyPI setup (to do before the first release)](#the-one-time-pypi-setup-to-do-before-the-first-release)
- [Reproducing the release build locally](#reproducing-the-release-build-locally)
- [Known gaps to close before a 1.0](#known-gaps-to-close-before-a-10)

## What gets published

Two kinds of artifact, built by two tools. The split is the packaging decision
of record, and it is written down in `pyproject.toml` next to the tables that
implement it:

| artifact | built by | contains | who gets it |
| --- | --- | --- | --- |
| `pneo-<v>.tar.gz` (sdist) | setuptools — the repo's own PEP 517 backend | the whole source: the Python package, the Rust crate, `tests/`, `scripts/`. Installing it **compiles nothing**. | every platform with no wheel below; `pip install` there needs no Rust toolchain and yields the numpy reference implementation |
| `pneo-<v>-cp311-abi3-<platform>.whl` | `maturin build` over `[tool.maturin]` | the Python package **plus** the compiled `pneo._core` | linux x86_64, macOS arm64, Windows x64 |

Two consequences, both deliberate:

- **The extension is optional at install time.** `pip install pneo` always
  succeeds. Where a wheel exists the fast kernel comes with it; where none does,
  pip falls back to the sdist and the user gets the reference implementation,
  which is exactly what `backend="auto"` degrades to at run time (see
  `src/pneo/backend.py`). Nothing hard-fails for want of `cargo`. The
  trade-off is that a user on an uncovered platform gets the 40×-slower path
  *silently* — `pneo.backend.core_available()` is how they find out, and
  extending the matrix is how we fix it.
- **`pip install -e .` in a checkout stays pure Python** and does not invoke
  cargo. The dev flow is unchanged: install editable once, then `rust/build.sh`
  after every crate edit (it drops `src/pneo/_core.abi3.so` into the import
  location). This is also what keeps the `python-backend` job in `ci.yml` an
  honest test of the no-extension install; `release.yml` never changes `ci.yml`'s
  premises.

maturin is used as a **CLI only**, never as the PEP 517 build backend. That is
what lets one `[project]` table feed both artifacts — so the sdist and the
wheels cannot disagree about name, version, dependencies or metadata — while
leaving `pip install -e .` free of a Rust dependency.

Because the crate is built against the stable ABI (`abi3-py311`), one wheel per
platform serves **every CPython ≥ 3.11**; the matrix is per platform, not per
platform × interpreter, and a new Python release needs no new wheel.

## Cutting a release

1. **There is nothing to bump.** The version is the git tag (setuptools-scm):
   pyproject declares `dynamic = ["version"]`, `__init__.__version__` reads the
   installed metadata, and the Rust crate carries no version at all
   (`publish = false`). Between tags, dev installs report
   `X.Y.(Z+1).devN+g<hash>`. The `version-check` job fails the release if any
   file grows a competing static version, or if the checkout does not resolve
   to exactly the pushed tag.
2. **Get CI green on `main`** — the release workflow does *not* re-run the test
   suite; it smoke-tests the built wheel. `ci.yml` on the release commit is the
   evidence that both backends pass.
3. **Rehearse** if the packaging changed at all: run the workflow by hand
   (see [below](#rehearsing-without-publishing)) and look at the artifacts.
4. **Commit, tag, push**:
   ```sh
   git tag -a v0.1.0 -m "pneo 0.1.0"
   git push origin v0.1.0        # <- this publishes to PyPI
   ```
5. **The tag push in step 4 is the release** — the workflow fires on it
   directly. Optionally add a GitHub release for human-readable notes
   afterward (`gh release create v0.1.0 --generate-notes`); it does not
   re-trigger the workflow.
6. **Watch the run.** In order: `version-check` (~20 s, fails fast on a
   version/tag mismatch), `sdist` and `wheels` in parallel, then `smoke-test`
   (installs the linux wheel on 3.11 and 3.13) and `metadata` (the sdist and
   every wheel must describe the same distribution), then `publish-pypi`.
   Nothing is uploaded unless every one of those passes, and PyPI uploads all
   files in one atomic call.
7. **Verify from the index**, in a fresh venv on a machine that is not this one:
   ```sh
   python -m venv /tmp/v && /tmp/v/bin/pip install pneo
   /tmp/v/bin/python -c "import pneo; from pneo.backend import core_available; print(pneo.__version__, core_available())"
   ```
   Expect `True` on the three covered platforms.

A failed upload is not partially applied, so the fix is: correct the problem,
bump to the next version (PyPI never allows re-uploading a version, even a
deleted one), and cut a new release. `release.yml` deliberately does **not**
pass `skip-existing`, so a duplicate upload fails loudly instead of looking
like a successful release.

## Rehearsing without publishing

`workflow_dispatch` on `.github/workflows/release.yml` (Actions → Release → Run
workflow) runs the whole pipeline **except** the publish job: sdist, three
wheels, both smoke tests, and every artifact attached to the run page for
inspection. There is no TestPyPI leg — by decision, pneo publishes directly
to pypi.org, and a dispatch run is the rehearsal.

The build steps are also runnable locally; see
[below](#reproducing-the-release-build-locally).

## The version rule

**The git tag is the single source of truth** (setuptools-scm). There is no
hand-maintained version string anywhere:

| place | what it does now |
| --- | --- |
| `pyproject.toml` | `dynamic = ["version"]` + `[tool.setuptools_scm]` (fallback `0.0.0+unknown` for tarball installs without `.git`); setuptools resolves the version at build time for the sdist and editable installs |
| `src/pneo/__init__.py` | `__version__` reads `importlib.metadata` (falls back to `0.0.0+uninstalled` in an uninstalled source tree) |
| `rust/Cargo.toml` | carries **no version** (`publish = false`; Cargo ≥ 1.75 defaults it); the crate is an implementation detail |
| `release.yml` wheels job | maturin cannot run setuptools-scm, so the resolved version is **stamped** into the runner-workspace pyproject before `maturin build`; the `metadata` job diffs sdist vs wheel METADATA so a stamping bug cannot ship |

Dev installs (`pip install -e .`) report `X.Y.(Z+1).devN+g<hash>` between tags;
the reported version is frozen at install time (reinstall to refresh). CI
checkouts use `fetch-depth: 0` so tags are visible to setuptools-scm.

`version-check` enforces, on every run:

- on a tag push, setuptools-scm must resolve to exactly the tag (clean, no
  `.dev`/local suffix — otherwise the checkout is not the tag);
- no file has grown a competing static version (pyproject, `__init__`, crate);
- `[tool.maturin] module-name == "pneo._core"` and
  `python-source == "src"`, matching `[tool.setuptools.packages.find] where` —
  lose the first and the wheel would silently ship a top-level `pneo_core`
  module that `pneo.backend` never looks for;
- exactly one `abi3-pyXY` feature on pyo3, and `requires-python` equal to the
  floor it implies — the "one wheel serves 3.11+" claim in this document is only
  true while that holds.

Nothing in the release machinery *writes* `rust/Cargo.toml`; it is read-only to
CI and to the workflow, and it never needs bumping.

## Platform coverage, and how to extend it

Shipped as wheels (one row of the `wheels` matrix each):

| wheel | runner | notes |
| --- | --- | --- |
| `manylinux2014_x86_64` | `ubuntu-latest` in `quay.io/pypa/manylinux2014_x86_64` | glibc ≥ 2.17 — deliberately the oldest we support, which includes RHEL/CentOS 7-era cluster nodes. The `manylinux: "2_17"` input names the container version explicitly so a future default change fails instead of narrowing the wheel silently. |
| `macosx_11_0_arm64` | `macos-14` | Apple silicon |
| `win_amd64` | `windows-latest` | |

Everything else — linux aarch64, musl linux (Alpine), macOS x86_64, ppc64le,
s390x, PyPy — installs from the sdist and runs the numpy reference. To promote
one to a wheel:

1. add a matrix row to the `wheels` job: `name`, `runner`, `target`,
   `interpreter`, `plat-tag`, and for linux an explicit `manylinux`;
2. bump the wheel count asserted in the `publish-pypi` job's "What is about to
   be uploaded" step;
3. for a linux target the host cannot execute natively, add
   `docker/setup-qemu-action` before the maturin step, or use the
   `ghcr.io/rust-cross/manylinux2014-cross:<arch>` container that
   `maturin-action` picks by default for `aarch64`/`armv7`/`ppc64le`/`s390x`
   (cross-compiling, no emulation, and much faster than QEMU);
4. consider whether the new wheel can be smoke-tested. The `smoke-test` job
   only exercises the linux x86_64 wheel, because that is the only one a GitHub
   runner in this matrix can *run*; a cross-compiled aarch64 wheel would ship
   with its contents checked (`check_wheel.py`) but never executed. That is the
   real cost of extending the matrix, and it is why coverage starts at three
   platforms rather than ten.

Concrete recipes for the common two:

```yaml
# linux aarch64, cross-compiled in the rust-cross container (no QEMU needed)
- name: linux-aarch64
  runner: ubuntu-latest
  target: aarch64-unknown-linux-gnu
  manylinux: "2_17"
  interpreter: python3.11
  plat-tag: manylinux2014_aarch64

# macOS x86_64 (Intel); or use target: universal2-apple-darwin for one fat wheel
- name: macos-x86_64
  runner: macos-14
  target: x86_64-apple-darwin
  interpreter: python
  plat-tag: macosx
```

## The one-time PyPI setup (to do before the first release)

Recorded so it can be audited or re-created. The maintainer configures a
**pending publisher** on pypi.org (Account → Publishing) with exactly these
values:

| field | value |
| --- | --- |
| PyPI project name | `pneo` |
| owner | `lsst-ssp` |
| repository | `pneo` |
| workflow filename | `release.yml` |
| environment | `pypi` |

> **These five strings are load-bearing.** PyPI matches the OIDC token against
> all of them. Renaming `.github/workflows/release.yml`, or changing
> `environment: name: pypi` in it, breaks publishing with an opaque
> `invalid-publisher` error and needs the publisher config updated on pypi.org
> first. Treat both names as API.

Because it is a *pending* publisher, the first successful upload creates the
`pneo` project on PyPI and converts the pending publisher into a normal
trusted publisher for it. Nothing else is needed on the PyPI side — in
particular no API token, no `~/.pypirc`, and no manual name reservation.

On the GitHub side, one thing is worth doing by hand: create the **`pypi`
environment** (Settings → Environments → New environment → `pypi`) so that
protection rules can be attached to it. A job referencing a missing environment
creates it implicitly with no rules, which works but gives up the one place
where a release can be gated. Recommended rules:

- **Required reviewers** = the maintainer: turns `publish-pypi` into a manual
  approval step *after* the wheels are built and smoke-tested — the last chance
  to stop a release.
- **Deployment branches and tags**: restrict to tags matching `v*`.

No secrets belong in that environment. `id-token: write` is granted to the
`publish-pypi` job alone (the workflow's top-level permission is
`contents: read`), and `pypa/gh-action-pypi-publish` is pinned by commit SHA, as
is `PyO3/maturin-action`; the maturin version itself is pinned in the
workflow's `env.MATURIN_VERSION`, because the wheel's platform tag and stripped
contents are properties of that version. The publish action also emits PEP 740
attestations by default, which is why it needs the OIDC token even beyond the
upload itself.

## Reproducing the release build locally

These are the commands the workflow runs, minus the manylinux container (a
build on any glibc ≤ 2.17 host produces the same tag; on a newer host the tag
will be `manylinux_2_XX` and PyPI's audience narrows accordingly — that is what
the container is for):

```sh
pip install build twine maturin==1.15.0

# sdist, as the `sdist` job builds and checks it
python -m build --sdist --outdir dist
python .github/scripts/check_sdist.py dist --version 0.0.1
python -m twine check --strict dist/*

# wheel, as the `wheels` job builds and checks it
maturin build --release --locked --out dist --interpreter python3.11
python .github/scripts/check_wheel.py dist --version 0.0.1 \
    --plat-tag manylinux2014_x86_64

# smoke-test the wheel exactly as the `smoke-test` job does: clean venv,
# working directory outside the repo, tests copied out of the tree
python -m venv /tmp/v && /tmp/v/bin/pip install dist/*.whl pytest
cd /tmp && PNEO_BACKEND=rust /tmp/v/bin/python \
    "$OLDPWD/.github/scripts/smoke_classify.py" rust
mkdir -p /tmp/wt && cp tests/test_imports.py tests/test_ranging.py \
    tests/test_transforms.py /tmp/wt/ && cd /tmp/wt && \
    PNEO_BACKEND=rust /tmp/v/bin/python -m pytest -q *.py

# the two tools must describe the same distribution (the `metadata` job)
python .github/scripts/check_metadata.py dist

# and the sdist's half of the contract: installs with no Rust in sight,
# degrades to the reference
python -m venv /tmp/vs
env -u CARGO -u CARGO_HOME -u RUSTUP_HOME PATH=/usr/bin:/bin \
    /tmp/vs/bin/pip install dist/pneo-*.tar.gz
cd /tmp && /tmp/vs/bin/python "$OLDPWD/.github/scripts/smoke_classify.py" python
```

The sdist is a *complete* source release, which is checked rather than assumed:
unpacking `pneo-<v>.tar.gz` and running `maturin build --release` inside it
produces the same wheel as the checkout does. That is what a downstream
packager (conda-forge, a distro, a site build) needs, and
`.github/scripts/check_sdist.py` fails the release if the crate, the lockfile or
`rust/build.sh` ever fall out of the tarball.

## Known gaps to close before a 1.0

Not blockers for publishing, but each one is visible to a PyPI user:

- ~~**No `LICENSE` file in the repository.**~~ Closed: `LICENSE` (BSD-3-Clause)
  is at the root and ships in both artifacts via the default `license-files` globs.
- ~~**A pip-installed user has no way to get the population model.**~~ Closed:
  the `pneo-fetch-model` console script (and `pneo fetch-model`
  subcommand) ships in both artifacts; the wheels and sdist still deliberately
  do not carry `data/digest2.model.csv` itself (~4 MB of MPC data, fetched at
  use time, not ours to redistribute).
- ~~**`README.md` has no installation section.**~~ Closed: the README carries
  an Installation section and a Command line section (it is the PyPI landing
  page). Original note, for context: worth saying that a wheel carries the Rust kernel, that
  other platforms get the numpy reference from the sdist, and how to check which
  one you have (`pneo.backend.core_available()`).
