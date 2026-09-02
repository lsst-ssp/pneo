"""WP4 unit tests — digest2 population histogram.

Acceptance criteria (IMPLEMENTATION.md WP4):
  1. integrated counts reproduce the file totals,
  2. NEO/other partition is exactly complementary at q = 1.3 AU,
  3. sample() histograms converge to the model,
  4. density is exactly 0 outside the declared support,
  5. both All and Unk variants load, with Unk <= All bin-wise.

Several tests deliberately re-derive quantities *independently* of the
implementation (a naive transcription of muk.c's scaling loop, a naive
transcription of digest2's binning loop, an independent CSV reader) so that a
bug in the vectorized implementation cannot hide behind the same bug in the
test.
"""

from __future__ import annotations

import csv
import math
import os

import numpy as np
import pytest

from digest42.constants import Q_NEO
from digest42.population import (
    BIN_VOLUME,
    D2_CLASSES,
    D2_VARIANTS,
    EPART,
    HPART,
    IPART_DEG,
    MODEL_SHAPE,
    QPART,
    Digest2Histogram,
    PopulationDensity,
    read_digest2_model,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "data", "digest2.model.csv")

ANGLE_VOLUME = (2.0 * math.pi) ** 3

# ---------------------------------------------------------------------------
# The four partition arrays, transcribed by hand from d2model.c (upstream
# digest2, github.com/Smithsonian/mpc-public):
#
#   double qpart[QX] = { .4, .7, .8, .9, 1., 1.1, 1.2, 1.3, 1.4, 1.5,
#     1.67, 1.8, 2., 2.2, 2.4, 2.6, 2.8, 3., 3.2, 3.5,
#     4., 4.5, 5., 5.5, 10., 20., 30., 40., 100. };
#   double epart[EX] = { .1, .2, .3, .4, .5, .7, .9, 1.1 };
#   double ipart[IX] = { 2, 5, 10, 15, 20, 25, 30, 40, 60, 90, 180 };
#   double hpart[HX] = { 6, 8, 10, 11, 12, 13, 14, 15, 16, 17,
#     18, 19, 20, 21, 22, 23, 24, 25.5 };
#
# The independent re-derivations below use THESE, never the module's own
# constants, so that a corrupted partition array in population.py cannot
# make the naive cross-check agree with the implementation by construction.
# ---------------------------------------------------------------------------
REF_QPART = np.array(
    [0.4, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
     1.67, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.5,
     4.0, 4.5, 5.0, 5.5, 10.0, 20.0, 30.0, 40.0, 100.0]
)
REF_EPART = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1])
REF_IPART_DEG = np.array(
    [2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 90.0, 180.0]
)
REF_HPART = np.array(
    [6.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
     18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.5]
)
REF_SHAPE = (len(REF_QPART), len(REF_EPART), len(REF_IPART_DEG), len(REF_HPART))

# Reference totals, independently established from the file by un-doing
# muk.c's 1/sqrt(pseudo-volume) scaling.  All/SS is the model file's own
# total, NOT S3M's raw object count: s3mbin.c drops out-of-model orbits and
# muk.c raises cells where the known catalogue exceeds the model.
REFERENCE_TOTALS = {
    ("All", "SS"): 14542147.0,
    ("All", "NEO"): 277469.0,
    ("All", "N22"): 38358.0,
    ("All", "N18"): 2339.0,
    ("All", "MB1"): 3687118.0,
    ("Unk", "SS"): 12998504.0,
    ("Unk", "NEO"): 235675.0,
}

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason=f"{MODEL_PATH} not present; run scripts/fetch_digest2_model.py",
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tables():
    return read_digest2_model(MODEL_PATH)


@pytest.fixture(scope="module")
def pair_all():
    return Digest2Histogram.neo_other_pair(MODEL_PATH, variant="All")


@pytest.fixture(scope="module")
def pair_unk():
    return Digest2Histogram.neo_other_pair(MODEL_PATH, variant="Unk")


@pytest.fixture(scope="module")
def samples(pair_all):
    """1e6 draws from each component, with the sampled q recovered."""
    neo, other = pair_all
    rng = np.random.default_rng(20260831)
    out = {}
    for comp in (neo, other):
        e6, h = comp.sample(1_000_000, rng)
        out[comp.label] = (e6, h)
    return out


# ---------------------------------------------------------------------------
# independent re-derivations used by the tests
# ---------------------------------------------------------------------------


def naive_muk_pseudo_volume():
    """Line-by-line transcription of muk.c's scaling loop (no numpy tricks).

    Uses the hand-transcribed REF_* partitions, not the module's.
    """
    V = np.zeros(REF_SHAPE)
    q0 = 0.0
    for iq in range(len(REF_QPART)):
        q1 = REF_QPART[iq]
        dq = q1 - q0
        q0 = q1
        e0 = 0.0
        d1 = 1.0
        for ie in range(len(REF_EPART)):
            e1 = REF_EPART[ie]
            d0 = d1
            d1 = 1.0 - e1 if e1 < 1.0 else 0.0
            dae = dq * (e1 - e0) / (d0 + d1)
            e0 = e1
            i0 = 0.0
            for ii in range(len(REF_IPART_DEG)):
                i1 = REF_IPART_DEG[ii]
                daei = dae * (i1 - i0)
                i0 = i1
                h0 = 0.0
                for ih in range(len(REF_HPART)):
                    h1 = REF_HPART[ih]
                    V[iq, ie, ii, ih] = daei * (h1 - h0)
                    h0 = h1
    return V


def naive_bin(part, x):
    """digest2's `for (b = 0; x >= part[b]; ++b)` rule, honestly."""
    b = 0
    while b < len(part) and x >= part[b]:
        b += 1
    return None if b == len(part) else b


def populated_bin(comp, ih=None):
    """(iq, ie, ii, ih) of the most populated bin, optionally in H bin ``ih``."""
    counts = np.asarray(comp.counts)
    if ih is not None:
        sub = counts[:, :, :, ih]
        assert sub.max() > 0, f"H bin {ih} is empty in {comp.label}"
        iq, ie, ii = np.unravel_index(int(np.argmax(sub)), sub.shape)
        return int(iq), int(ie), int(ii), int(ih)
    return tuple(int(k) for k in
                 np.unravel_index(int(np.argmax(counts)), MODEL_SHAPE))


def bin_centre(comp, idx):
    iq, ie, ii, ih = idx
    el = np.array([
        0.5 * (comp.q_edges[iq] + comp.q_edges[iq + 1]),
        0.5 * (comp.e_edges[ie] + comp.e_edges[ie + 1]),
        0.5 * (comp.i_edges[ii] + comp.i_edges[ii + 1]),
    ])
    return el, 0.5 * (comp.h_edges[ih] + comp.h_edges[ih + 1])


# ---------------------------------------------------------------------------
# 5. both variants load; Unk <= All
# ---------------------------------------------------------------------------


def test_all_tables_present(tables):
    expected = {(v, c) for v in D2_VARIANTS for c in ("SS",) + D2_CLASSES}
    assert set(tables) == expected
    for k, arr in tables.items():
        assert arr.shape == MODEL_SHAPE, k
        assert np.all(arr >= 0.0), k
        assert np.all(np.isfinite(arr)), k


def test_counts_are_exact_integers(tables):
    """The load-time proof that the 1/sqrt(V) inversion is the right one."""
    for k, arr in tables.items():
        assert np.array_equal(arr, np.round(arr)), k


def test_reference_totals(tables):
    for k, want in REFERENCE_TOTALS.items():
        assert tables[k].sum() == pytest.approx(want, rel=0, abs=0.5), k


def test_unk_le_all_binwise(tables):
    for cls in ("SS",) + D2_CLASSES:
        assert np.all(tables[("Unk", cls)] <= tables[("All", cls)] + 1e-9), cls
    # and strictly smaller in aggregate (some objects are known)
    assert tables[("Unk", "SS")].sum() < tables[("All", "SS")].sum()


def test_both_variants_build_components(pair_all, pair_unk):
    for pair, variant in ((pair_all, "All"), (pair_unk, "Unk")):
        neo, other = pair
        assert (neo.label, other.label) == ("NEO", "other")
        assert neo.variant == other.variant == variant
        assert neo.parameterization == other.parameterization == "qei"
        assert isinstance(neo, PopulationDensity)
        assert neo.total_count > 0 and other.total_count > 0


def test_unk_components_le_all_components(pair_all, pair_unk):
    for a, u in zip(pair_all, pair_unk):
        assert np.all(np.asarray(u.counts) <= np.asarray(a.counts) + 1e-9)
        assert u.total_count < a.total_count


def test_bad_variant_rejected():
    with pytest.raises(ValueError):
        Digest2Histogram.neo_other_pair(MODEL_PATH, variant="Known")


# ---------------------------------------------------------------------------
# the muk.c un-scaling, re-derived independently
# ---------------------------------------------------------------------------


def test_module_partitions_match_d2model_c():
    """The module's constants must equal the hand-transcribed d2model.c ones.

    Without this the independent re-derivations below would silently inherit
    any corruption of population.py's partition arrays.
    """
    np.testing.assert_array_equal(QPART, REF_QPART)
    np.testing.assert_array_equal(EPART, REF_EPART)
    np.testing.assert_array_equal(IPART_DEG, REF_IPART_DEG)
    np.testing.assert_array_equal(HPART, REF_HPART)
    assert MODEL_SHAPE == REF_SHAPE == (29, 8, 11, 18)

    # ... and the derived edge arrays: leading 0, degrees -> radians.
    from digest42.population import (
        E_EDGES, H_EDGES, I_EDGES, I_EDGES_DEG, Q_EDGES,
    )

    for edges, part in ((Q_EDGES, REF_QPART), (E_EDGES, REF_EPART),
                        (I_EDGES_DEG, REF_IPART_DEG), (H_EDGES, REF_HPART)):
        assert edges[0] == 0.0
        np.testing.assert_array_equal(edges[1:], part)
    np.testing.assert_allclose(I_EDGES, np.radians(I_EDGES_DEG), rtol=0, atol=0)

    # BIN_VOLUME is dq * de * di[rad] * dH -- spot-check the first bin by hand
    assert BIN_VOLUME[0, 0, 0, 0] == pytest.approx(
        0.4 * 0.1 * math.radians(2.0) * 6.0, rel=1e-15
    )


def test_pseudo_volume_matches_naive_transcription():
    from digest42.population import _MUK_PSEUDO_VOLUME

    np.testing.assert_allclose(
        _MUK_PSEUDO_VOLUME, naive_muk_pseudo_volume(), rtol=1e-15, atol=0.0
    )


def test_counts_match_independent_csv_reader(tables):
    """Independent reader: raw cell * sqrt(pseudo-volume), CELL BY CELL.

    Not just per-table totals: a sign-flipped or transposed cell would cancel
    in a sum.  Every one of the 32 x 45 936 cells is compared.
    """
    V = naive_muk_pseudo_volume()
    qidx = {round(float(v), 9): k for k, v in enumerate(REF_QPART)}
    eidx = {round(float(v), 9): k for k, v in enumerate(REF_EPART)}
    iidx = {round(float(v), 9): k for k, v in enumerate(REF_IPART_DEG)}
    ref: dict[tuple[str, str], np.ndarray] = {}
    nrows = 0
    with open(MODEL_PATH, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            iq = qidx[round(float(row[2]), 9)]
            ie = eidx[round(float(row[3]), 9)]
            ii = iidx[round(float(row[4]), 9)]
            arr = ref.setdefault((row[0], row[1]), np.zeros(REF_SHAPE))
            fields = row[5:]
            assert len(fields) == REF_SHAPE[3]
            for ih, field in enumerate(fields):
                if field.strip():
                    arr[iq, ie, ii, ih] = (
                        float(field) * math.sqrt(V[iq, ie, ii, ih])
                    )
            nrows += 1
    assert nrows == 32 * 29 * 8 * 11
    assert set(ref) == set(tables)
    for k, arr in ref.items():
        # implementation rounds to integers; the raw inversion must already
        # be within 1e-6 of those integers, cell by cell.
        assert np.abs(arr - tables[k]).max() < 1e-6, k
        assert np.array_equal(np.round(arr), tables[k]), k


def test_raw_cells_are_not_counts(tables):
    """Guard against a regression that drops the un-scaling entirely."""
    raw_sum = 0.0
    with open(MODEL_PATH, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            if row[0] == "All" and row[1] == "SS":
                raw_sum += sum(float(f) for f in row[5:] if f.strip())
    # ~6.35e7 raw vs 1.4542147e7 objects: a factor 4.4 apart.
    assert raw_sum > 6.0e7
    assert tables[("All", "SS")].sum() == pytest.approx(1.4542147e7, abs=0.5)


# ---------------------------------------------------------------------------
# 1. integrated counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["All", "Unk"])
def test_integrated_density_reproduces_total_count(variant):
    """sum over bins of density(bin centre) * bin volume * (2 pi)^3 == N."""
    neo, other = Digest2Histogram.neo_other_pair(MODEL_PATH, variant=variant)
    qc = 0.5 * (neo.q_edges[:-1] + neo.q_edges[1:])
    ec = 0.5 * (neo.e_edges[:-1] + neo.e_edges[1:])
    ic = 0.5 * (neo.i_edges[:-1] + neo.i_edges[1:])
    hc = 0.5 * (neo.h_edges[:-1] + neo.h_edges[1:])
    QQ, EE, II, HH = np.meshgrid(qc, ec, ic, hc, indexing="ij")
    elements3 = np.stack([QQ, EE, II], axis=-1)

    for comp in (neo, other):
        rho = comp.density(elements3, HH)
        assert rho.shape == MODEL_SHAPE
        integral = float((rho * BIN_VOLUME * ANGLE_VOLUME).sum())
        assert integral == pytest.approx(comp.total_count, rel=1e-9)


def test_total_counts_match_file(pair_all, pair_unk, tables):
    neo, other = pair_all
    assert neo.total_count == pytest.approx(tables[("All", "NEO")].sum(), abs=0.5)
    assert other.total_count == pytest.approx(
        tables[("All", "SS")].sum() - tables[("All", "NEO")].sum(), abs=0.5
    )
    assert neo.total_count == pytest.approx(277469.0, abs=0.5)
    assert other.total_count == pytest.approx(14264678.0, abs=0.5)

    uneo, uother = pair_unk
    assert uneo.total_count == pytest.approx(235675.0, abs=0.5)
    assert uother.total_count == pytest.approx(12762829.0, abs=0.5)


def test_density_units_single_bin(pair_all):
    """Hand-check one bin: density = count / (dq de di_rad dH) / (2 pi)^3."""
    neo, _ = pair_all
    counts = np.asarray(neo.counts)
    iq, ie, ii, ih = np.unravel_index(int(np.argmax(counts)), MODEL_SHAPE)
    dq = neo.q_edges[iq + 1] - neo.q_edges[iq]
    de = neo.e_edges[ie + 1] - neo.e_edges[ie]
    di = math.radians(neo.i_edges_deg[ii + 1] - neo.i_edges_deg[ii])
    dh = neo.h_edges[ih + 1] - neo.h_edges[ih]
    want = counts[iq, ie, ii, ih] / (dq * de * di * dh) / ANGLE_VOLUME
    centre = np.array([
        0.5 * (neo.q_edges[iq] + neo.q_edges[iq + 1]),
        0.5 * (neo.e_edges[ie] + neo.e_edges[ie + 1]),
        0.5 * (neo.i_edges[ii] + neo.i_edges[ii + 1]),
    ])
    got = float(neo.density(centre, 0.5 * (neo.h_edges[ih] + neo.h_edges[ih + 1])))
    assert got == pytest.approx(want, rel=1e-13)
    assert got > 0.0


# ---------------------------------------------------------------------------
# 2. partition / complementarity at q = 1.3 AU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["All", "Unk"])
def test_partition_residual_is_exactly_zero(variant, tables):
    """neo + other == SS, cell by cell, with zero residual."""
    neo, other = Digest2Histogram.neo_other_pair(MODEL_PATH, variant=variant)
    ss = tables[(variant, "SS")]
    residual = np.asarray(neo.counts) + np.asarray(other.counts) - ss
    assert np.abs(residual).max() == 0.0
    assert neo.total_count + other.total_count == pytest.approx(
        float(ss.sum()), rel=0, abs=0.0
    )


@pytest.mark.parametrize("variant", ["All", "Unk"])
def test_neo_class_is_exactly_the_q_lt_1p3_part_of_ss(variant, tables):
    """digest2's NEO test is q < 1.3 and 1.3 is the qpart[7] bin edge."""
    ss = tables[(variant, "SS")]
    neo = tables[(variant, "NEO")]
    iq13 = int(np.searchsorted(REF_QPART, Q_NEO))  # == 7, upper edge 1.3
    assert REF_QPART[iq13] == Q_NEO
    assert np.abs(ss[: iq13 + 1] - neo[: iq13 + 1]).max() == 0.0
    assert neo[iq13 + 1:].sum() == 0.0


@pytest.mark.parametrize("variant", ["All", "Unk"])
def test_complementarity_at_q_1p3(variant):
    """NEO density vanishes for q >= 1.3; other density vanishes for q < 1.3."""
    neo, other = Digest2Histogram.neo_other_pair(MODEL_PATH, variant=variant)
    rng = np.random.default_rng(7)
    n = 200_000
    q = rng.uniform(0.0, 100.0, n)
    e = rng.uniform(0.0, 1.1, n)
    i = rng.uniform(0.0, np.pi, n)
    h = rng.uniform(0.0, 25.5, n)
    # pin a slab of points onto the boundary itself
    q[:1000] = Q_NEO
    q[1000:2000] = np.nextafter(Q_NEO, 0.0)
    q[2000:3000] = np.nextafter(Q_NEO, 2.0)
    el = np.stack([q, e, i], axis=-1)

    d_neo = neo.density(el, h)
    d_oth = other.density(el, h)
    assert np.all(d_neo[q >= Q_NEO] == 0.0)
    assert np.all(d_oth[q < Q_NEO] == 0.0)
    # and neither component is trivially zero on its own side
    assert np.any(d_neo[q < Q_NEO] > 0.0)
    assert np.any(d_oth[q >= Q_NEO] > 0.0)
    # exactly one component can be nonzero anywhere
    assert not np.any((d_neo > 0.0) & (d_oth > 0.0))


def test_named_classes_do_not_partition_ss(tables):
    """Documents *why* other == SS - NEO rather than a union of classes.

    The 11 nominally non-NEO classes over-count SS - NEO by ~1.2% (they
    overlap) while also failing to cover it.  If this ever becomes an exact
    partition the choice in neo_other_pair could be revisited.
    """
    named = ("MC", "Hun", "Pho", "MB1", "Pal", "Han", "MB2", "MB3",
             "Hil", "JTr", "JFC")
    union = sum(tables[("All", c)] for c in named)
    other = tables[("All", "SS")] - tables[("All", "NEO")]
    excess = union.sum() / other.sum() - 1.0
    assert 0.005 < excess < 0.05, excess          # overlapping: over-counts
    assert np.any(union < other - 0.5)            # and does not cover


def test_n18_n22_are_subsets_of_neo(tables):
    """The trap the supervisor flagged: N18/N22 are H-limited NEO subsets.

    Exact in the "All" model.  In "Unk" the nesting is violated in a handful
    of bins by <= 4 objects: muk.c's "expand all where known > all" fix-up is
    applied to each class array independently, so the per-class differences
    all - known need not nest.  Recorded here as a known model artifact; it
    does not touch the NEO/other split, which uses only SS and NEO.
    """
    assert np.all(tables[("All", "N18")] <= tables[("All", "N22")])
    assert np.all(tables[("All", "N22")] <= tables[("All", "NEO")])
    for a, b in (("N18", "N22"), ("N22", "NEO")):
        excess = (tables[("Unk", a)] - tables[("Unk", b)]).max()
        assert excess <= 4.0, (a, b, excess)
    # ... and "Int" (MPC-interesting) overlaps NEO, so it is not disjoint
    assert tables[("All", "Int")].sum() > tables[("All", "NEO")].sum()


# ---------------------------------------------------------------------------
# 4. support truncation
# ---------------------------------------------------------------------------


def test_support_declaration(pair_all):
    for comp in pair_all:
        s = comp.support
        assert set(s) == {"q", "e", "i", "H"}
        assert s["q"] == (0.0, 100.0)
        assert s["e"] == (0.0, 1.1)
        assert s["i"] == pytest.approx((0.0, math.pi))
        assert s["H"] == (0.0, 25.5)


@pytest.mark.parametrize("comp_index", [0, 1])
def test_density_zero_outside_support(pair_all, comp_index):
    comp = pair_all[comp_index]
    base, base_h = bin_centre(comp, populated_bin(comp))
    assert comp.density(base, base_h) > 0.0

    def d(q=None, e=None, i=None, h=None):
        el = base.copy()
        if q is not None:
            el[0] = q
        if e is not None:
            el[1] = e
        if i is not None:
            el[2] = i
        return float(comp.density(el, base_h if h is None else h))

    # H support: [0, 25.5)
    assert d(h=25.5) == 0.0
    assert d(h=26.0) == 0.0
    assert d(h=1e3) == 0.0
    assert d(h=-1e-9) == 0.0
    assert d(h=-30.0) == 0.0
    assert d(h=np.nan) == 0.0
    # q support: [0, 100)
    assert d(q=100.0) == 0.0
    assert d(q=1e6) == 0.0
    assert d(q=-1e-9) == 0.0
    # e support: [0, 1.1)
    assert d(e=1.1) == 0.0
    assert d(e=5.0) == 0.0
    assert d(e=-1e-9) == 0.0
    # i support: [0, pi)
    assert d(i=math.pi) == 0.0
    assert d(i=4.0) == 0.0
    assert d(i=-1e-9) == 0.0
    # inclination is in RADIANS: 90 (as if degrees) is far outside
    assert d(i=90.0) == 0.0


def test_last_h_bin_is_truncated_not_clamped(pair_all):
    """D-4 deviation from digest2: hToBin() clamps H >= 24 into the top bin.

    digest42 truncates instead: [24, 25.5) carries density, >= 25.5 is zero.
    """
    _, other = pair_all
    el, _h = bin_centre(other, populated_bin(other, ih=len(REF_HPART) - 1))
    assert other.density(el, 24.0) > 0.0
    assert other.density(el, 25.4999) > 0.0
    assert other.density(el, 25.5) == 0.0
    assert other.density(el, 25.6) == 0.0
    assert other.density(el, 1e6) == 0.0


def test_first_h_bin_lower_edge_is_zero(pair_all, tables):
    """The H < 6 bin is populated (3218 objects in All/SS), so its lower
    edge matters; muk.c's volume loop starts at h0 = 0."""
    assert tables[("All", "SS")][:, :, :, 0].sum() == pytest.approx(3218.0, abs=0.5)
    _, other = pair_all
    # find a populated (q, e, i) cell in the first H bin
    idx = np.argwhere(np.asarray(other.counts)[:, :, :, 0] > 0)
    assert len(idx) > 0
    iq, ie, ii = idx[0]
    el = np.array([
        0.5 * (other.q_edges[iq] + other.q_edges[iq + 1]),
        0.5 * (other.e_edges[ie] + other.e_edges[ie + 1]),
        0.5 * (other.i_edges[ii] + other.i_edges[ii + 1]),
    ])
    assert other.density(el, 3.0) > 0.0
    assert other.density(el, 0.0) > 0.0
    assert other.density(el, -0.001) == 0.0


def test_binning_rule_matches_digest2(pair_all):
    """Compare the vectorized lookup with a naive transcription of digest2's
    `for (b = 0; x >= part[b]; ++b)` loop (H excluded: digest2 clamps)."""
    neo, _ = pair_all
    rng = np.random.default_rng(11)
    n = 3000
    q = rng.uniform(-0.5, 105.0, n)
    e = rng.uniform(-0.05, 1.2, n)
    ideg = rng.uniform(-1.0, 190.0, n)
    h = rng.uniform(-1.0, 27.0, n)
    # sprinkle exact edges
    q[:29] = REF_QPART
    e[:8] = REF_EPART
    ideg[:11] = REF_IPART_DEG
    h[:18] = REF_HPART

    iq, ie, ii, ih, ok = neo.bin_indices(
        np.stack([q, e, np.radians(ideg)], axis=-1), h
    )
    for k in range(n):
        bq = naive_bin(REF_QPART, q[k])
        be = naive_bin(REF_EPART, e[k])
        bi = naive_bin(REF_IPART_DEG, ideg[k])
        bh = naive_bin(REF_HPART, h[k])
        negative = q[k] < 0 or e[k] < 0 or ideg[k] < 0 or h[k] < 0
        want_ok = (bq is not None and be is not None and bi is not None
                   and bh is not None and not negative)
        assert bool(ok[k]) == want_ok, (k, q[k], e[k], ideg[k], h[k])
        if want_ok:
            assert (iq[k], ie[k], ii[k], ih[k]) == (bq, be, bi, bh)


def test_density_broadcasting_and_shapes(pair_all):
    neo, _ = pair_all
    el = np.array([[1.0, 0.15, 0.1], [2.5, 0.15, 0.1], [50.0, 0.05, 0.02]])
    out = neo.density(el, np.array([18.0, 18.0, 18.0]))
    assert out.shape == (3,)
    # scalar H broadcasts
    out2 = neo.density(el, 18.0)
    np.testing.assert_array_equal(out, out2)
    # single element vector -> 0-d result
    out3 = neo.density(np.array([1.0, 0.15, 0.1]), 18.0)
    assert np.ndim(out3) == 0
    assert float(out3) == pytest.approx(float(out[0]))
    # higher rank
    el4 = np.broadcast_to(el, (4, 5, 3, 3))
    assert neo.density(el4, np.full((4, 5, 3), 18.0)).shape == (4, 5, 3)
    with pytest.raises(ValueError):
        neo.density(np.array([1.0, 0.15]), 18.0)


# ---------------------------------------------------------------------------
# 3. sample()
# ---------------------------------------------------------------------------


def test_sample_shapes_and_ranges(pair_all, samples):
    for comp in pair_all:
        e6, h = samples[comp.label]
        assert e6.shape == (1_000_000, 6)
        assert h.shape == (1_000_000,)
        e = e6[:, 1]
        i = e6[:, 2]
        assert np.all((e >= 0.0) & (e < 1.1))
        assert np.all((i >= 0.0) & (i < math.pi))
        assert np.all((h >= 0.0) & (h < 25.5))
        assert np.all(np.isfinite(e6[:, 1:]))


def test_sample_a_is_q_over_one_minus_e(pair_all, samples):
    """a = q/(1-e); recover q and check it lies in the component's support
    and in a bin with nonzero density."""
    for comp in pair_all:
        e6, h = samples[comp.label]
        a, e, i = e6[:, 0], e6[:, 1], e6[:, 2]
        q = a * (1.0 - e)
        assert np.all(np.isfinite(q))
        assert np.all(q >= 0.0) and np.all(q < 100.0)
        if comp.label == "NEO":
            assert np.all(q < Q_NEO)
        else:
            assert np.all(q >= Q_NEO)
        # every sample must land in a populated bin
        d = comp.density(np.stack([q, e, i], axis=-1), h)
        assert np.all(d > 0.0)
        # bound orbits have a > 0; unbound only from the e >= 1 top bin
        assert np.all((a > 0.0) == (e < 1.0))


def _h_of(comp, samples):
    return samples[comp.label][1][:, None]


def test_sample_angles_uniform(pair_all, samples):
    for comp in pair_all:
        e6, _ = samples[comp.label]
        n = e6.shape[0]
        for col in (3, 4, 5):
            ang = e6[:, col]
            assert np.all((ang >= 0.0) & (ang < 2 * math.pi))
            # mean and variance of U(0, 2pi)
            assert ang.mean() == pytest.approx(math.pi, abs=6 * (2 * math.pi)
                                               / math.sqrt(12 * n))
            assert ang.var() == pytest.approx((2 * math.pi) ** 2 / 12, rel=0.02)
            # 20-bin chi^2
            counts, _ = np.histogram(ang, bins=20, range=(0.0, 2 * math.pi))
            exp = n / 20.0
            chi2 = ((counts - exp) ** 2 / exp).sum()
            assert chi2 < 43.8, chi2      # chi2_19, p = 0.001
        # angles independent of each other and of the elements
        # (e and i are genuinely correlated in the population, so only pairs
        # involving at least one angle are checked)
        cols = np.column_stack([e6[:, 3:6], e6[:, 1:3], _h_of(comp, samples)])
        c = np.corrcoef(cols.T)
        for r in range(3):                       # Omega, omega, M
            for s in range(cols.shape[1]):
                if r != s:
                    assert abs(c[r, s]) < 0.01, (comp.label, r, s, c[r, s])


@pytest.mark.parametrize("label", ["NEO", "other"])
def test_sample_histogram_converges(pair_all, samples, label):
    """Re-bin 1e6 samples on the model's own bins and compare to the model."""
    comp = {c.label: c for c in pair_all}[label]
    e6, h = samples[label]
    a, e, i = e6[:, 0], e6[:, 1], e6[:, 2]
    q = a * (1.0 - e)
    iq, ie, ii, ih, ok = comp.bin_indices(np.stack([q, e, i], axis=-1), h)
    assert ok.all()

    n = e6.shape[0]
    obs = np.zeros(MODEL_SHAPE)
    np.add.at(obs, (iq, ie, ii, ih), 1.0)
    assert obs.sum() == n

    p = np.asarray(comp.counts) / comp.total_count
    exp = p * n

    # zero-probability bins must never be drawn
    assert obs[exp == 0.0].sum() == 0.0

    # marginals: each within 5 binomial sigma
    for axis in range(4):
        axes = tuple(k for k in range(4) if k != axis)
        o = obs.sum(axis=axes)
        ex = exp.sum(axis=axes)
        sel = ex > 30
        sigma = np.sqrt(ex[sel] * (1 - ex[sel] / n))
        z = np.abs(o[sel] - ex[sel]) / sigma
        assert z.max() < 5.0, (axis, z.max())

    # chi^2 over the well-populated bins
    sel = exp >= 25
    dof = int(sel.sum())
    assert dof > 100
    chi2 = (((obs[sel] - exp[sel]) ** 2) / exp[sel]).sum()
    # 6-sigma band on chi2 with `dof` degrees of freedom
    assert abs(chi2 - dof) < 6.0 * math.sqrt(2.0 * dof), (chi2, dof)


def test_sample_within_bin_is_uniform(pair_all):
    """Within the single most populated bin, (q, e, i, H) must be uniform."""
    _, other = pair_all
    counts = np.asarray(other.counts)
    iq, ie, ii, ih = np.unravel_index(int(np.argmax(counts)), MODEL_SHAPE)
    rng = np.random.default_rng(4242)
    e6, h = other.sample(2_000_000, rng)
    q = e6[:, 0] * (1.0 - e6[:, 1])
    jq, je, ji, jh, _ = other.bin_indices(
        np.stack([q, e6[:, 1], e6[:, 2]], axis=-1), h
    )
    m = (jq == iq) & (je == ie) & (ji == ii) & (jh == ih)
    assert m.sum() > 5000
    for vals, lo, hi in (
        (q[m], other.q_edges[iq], other.q_edges[iq + 1]),
        (e6[m, 1], other.e_edges[ie], other.e_edges[ie + 1]),
        (e6[m, 2], other.i_edges[ii], other.i_edges[ii + 1]),
        (h[m], other.h_edges[ih], other.h_edges[ih + 1]),
    ):
        u = (vals - lo) / (hi - lo)
        assert u.min() >= 0.0 and u.max() < 1.0
        assert u.mean() == pytest.approx(0.5, abs=5 / math.sqrt(12 * m.sum()))
        assert u.var() == pytest.approx(1 / 12, rel=0.06)


def test_sample_reproducible_and_edge_cases(pair_all):
    neo, _ = pair_all
    a1 = neo.sample(1000, np.random.default_rng(1))
    a2 = neo.sample(1000, np.random.default_rng(1))
    np.testing.assert_array_equal(a1[0], a2[0])
    np.testing.assert_array_equal(a1[1], a2[1])
    b = neo.sample(1000, np.random.default_rng(2))
    assert not np.array_equal(a1[0], b[0])

    e6, h = neo.sample(0, np.random.default_rng(3))
    assert e6.shape == (0, 6) and h.shape == (0,)
    with pytest.raises(ValueError):
        neo.sample(-1, np.random.default_rng(3))

    empty = Digest2Histogram(np.zeros(MODEL_SHAPE), "other")
    assert empty.total_count == 0.0
    with pytest.raises(ValueError):
        empty.sample(10, np.random.default_rng(3))


def test_sample_hits_the_first_bin_of_each_axis(pair_all):
    """Regression guard for an off-by-one that would make bin 0 unreachable."""
    _, other = pair_all
    rng = np.random.default_rng(99)
    e6, h = other.sample(500_000, rng)
    q = e6[:, 0] * (1.0 - e6[:, 1])
    iq, ie, ii, ih, _ = other.bin_indices(
        np.stack([q, e6[:, 1], e6[:, 2]], axis=-1), h
    )
    counts = np.asarray(other.counts)
    for idx, axis in ((iq, 0), (ie, 1), (ii, 2), (ih, 3)):
        axes = tuple(k for k in range(4) if k != axis)
        if counts.sum(axis=axes)[0] / other.total_count > 1e-4:
            assert (idx == 0).any(), axis


# ---------------------------------------------------------------------------
# interpolation is not implemented (D-10)
# ---------------------------------------------------------------------------


def test_interpolation_raises():
    with pytest.raises(NotImplementedError):
        Digest2Histogram(np.zeros(MODEL_SHAPE), "other", interpolate=True)


# ---------------------------------------------------------------------------
# constructor validation
# ---------------------------------------------------------------------------


def test_constructor_validation():
    with pytest.raises(ValueError):
        Digest2Histogram(np.zeros((2, 2)), "other")
    bad = np.zeros(MODEL_SHAPE)
    bad[0, 0, 0, 0] = -1.0
    with pytest.raises(ValueError):
        Digest2Histogram(bad, "other")
    bad2 = np.zeros(MODEL_SHAPE)
    bad2[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        Digest2Histogram(bad2, "other")


def test_counts_view_is_read_only(pair_all):
    neo, _ = pair_all
    with pytest.raises(ValueError):
        neo.counts[0, 0, 0, 0] = 1.0


def test_from_classes_diagnostic(tables):
    h = Digest2Histogram.from_classes(MODEL_PATH, ("N22",), "NEO")
    assert h.total_count == pytest.approx(tables[("All", "N22")].sum(), abs=0.5)
    assert h.classes == "N22"


def test_read_returns_isolated_dict(tables):
    """A caller mutating the returned dict must not poison the cache."""
    d1 = read_digest2_model(MODEL_PATH)
    d1.pop(("All", "SS"))
    d1[("All", "NEO")] = "corrupted"
    d2 = read_digest2_model(MODEL_PATH)
    assert ("All", "SS") in d2
    assert isinstance(d2[("All", "NEO")], np.ndarray)
    assert d2[("All", "NEO")].sum() == pytest.approx(277469.0, abs=0.5)
    # the arrays themselves are read-only
    with pytest.raises(ValueError):
        d2[("All", "SS")][0, 0, 0, 0] = 1.0


def test_unknown_class_and_variant_raise_value_error():
    with pytest.raises(ValueError, match="unknown model variant"):
        Digest2Histogram.neo_other_pair(MODEL_PATH, variant="nope")
    with pytest.raises(ValueError, match="unknown model variant"):
        Digest2Histogram.from_classes(MODEL_PATH, ("SS",), "other",
                                      variant="nope")
    with pytest.raises(ValueError, match="unknown digest2 class"):
        Digest2Histogram.from_classes(MODEL_PATH, ("MBA",), "other")
    with pytest.raises(ValueError, match="unknown digest2 class"):
        Digest2Histogram.from_classes(MODEL_PATH, ("SS",), "other",
                                      subtract=("TNO",))


def test_bin_indices_validates_shape_and_flags_out_of_support(pair_all):
    neo, _ = pair_all
    with pytest.raises(ValueError):
        neo.bin_indices(np.array([1.0, 0.15]), 18.0)
    with pytest.raises(ValueError):
        neo.bin_indices(np.array(1.0), 18.0)
    # out-of-support inputs still return clamped, plausible-looking indices
    el = np.array([[1e6, 0.15, 0.1], [-5.0, 0.15, 0.1]])
    iq, ie, ii, ih, ok = neo.bin_indices(el, np.array([18.0, 18.0]))
    assert not ok.any()
    assert np.all((iq >= 0) & (iq < 29))       # hence the docstring warning


def test_sample_within_bin_never_reaches_upper_edge(pair_all):
    """NOTE-2 guard: lo + u*(hi-lo) must not round up to hi."""
    neo, other = pair_all
    rng = np.random.default_rng(31337)
    for comp in (neo, other):
        e6, h = comp.sample(200_000, rng)
        q = e6[:, 0] * (1.0 - e6[:, 1])
        iq, ie, ii, ih, ok = comp.bin_indices(
            np.stack([q, e6[:, 1], e6[:, 2]], axis=-1), h
        )
        assert ok.all()
        for vals, edges, idx in ((q, comp.q_edges, iq),
                                 (e6[:, 1], comp.e_edges, ie),
                                 (e6[:, 2], comp.i_edges, ii),
                                 (h, comp.h_edges, ih)):
            assert np.all(vals >= edges[idx])
            assert np.all(vals < edges[idx + 1])


def test_sample_unbound_fraction_is_documented(pair_all, pair_unk):
    """MAJOR-1: ~2% of NEO draws are unbound; the docstring must not
    understate it and callers must be able to reproduce the number."""
    for pair, want_neo in ((pair_all, 0.01915), (pair_unk, 0.02228)):
        neo, other = pair
        # exact expectation: half of the [0.9, 1.1) top e bin
        exact = 0.5 * np.asarray(neo.counts)[:, 7].sum() / neo.total_count
        assert exact == pytest.approx(want_neo, abs=5e-5)
        e6, _ = neo.sample(500_000, np.random.default_rng(8))
        frac = float((e6[:, 1] >= 1.0).mean())
        assert frac == pytest.approx(exact, abs=0.002)
        assert np.all(e6[e6[:, 1] > 1.0, 0] < 0.0)      # a < 0 when e > 1
        assert np.all(e6[e6[:, 1] < 1.0, 0] > 0.0)
        # the "other" component is unaffected at the 1e-5 level
        oexact = 0.5 * np.asarray(other.counts)[:, 7].sum() / other.total_count
        assert oexact < 1e-5
    assert "unbound" in Digest2Histogram.sample.__doc__.lower()
    assert "1.92" in Digest2Histogram.sample.__doc__
    assert "2.23" in Digest2Histogram.sample.__doc__


def test_read_rejects_bad_header(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("not,a,digest2,header\n")
    with pytest.raises(ValueError):
        read_digest2_model(p)


def test_read_rejects_incomplete_file(tmp_path):
    p = tmp_path / "short.csv"
    with open(MODEL_PATH) as fh:
        head = [next(fh) for _ in range(50)]
    p.write_text("".join(head))
    with pytest.raises(ValueError, match="missing"):
        read_digest2_model(p)


# ---------------------------------------------------------------------------
# WP7 addendum — the digest2-emulation knob (h_top_bin_infinite)
#
# digest2's ``hToBin`` clamps every H >= 24 into the final bin, i.e. it treats
# the top H bin as [24, inf); digest42 truncates at 25.5 per D-4.  The knob
# reinstates digest2's clamp so that the DESIGN §4.2 head-to-head can be run
# with the two codes consuming genuinely identical effective models.  These
# tests pin down: what changes (H >= 25.5 only), what does not (everything
# else, bit-for-bit), and the guard rails that keep the mode out of the gate.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pair_all_clamped():
    return Digest2Histogram.neo_other_pair(MODEL_PATH, variant="All",
                                           h_top_bin_infinite=True)


@pytest.fixture(scope="module")
def pair_unk_clamped():
    return Digest2Histogram.neo_other_pair(MODEL_PATH, variant="Unk",
                                           h_top_bin_infinite=True)


#: (component index, name) — both components of the pair get tested.
_COMPS = [(0, "NEO"), (1, "other")]


def _top_h_bin_density(comp):
    """(elements at a populated top-H-bin centre, that bin's density value)."""
    idx = populated_bin(comp, ih=len(REF_HPART) - 1)
    el, _h = bin_centre(comp, idx)
    # The bin's own density value, taken from the [24, 25.5) evaluation, which
    # both modes agree on.
    return el, float(comp.density(el, 24.5))


@pytest.mark.parametrize("variant", ["All", "Unk"])
@pytest.mark.parametrize("comp_index, name", _COMPS)
def test_clamped_top_bin_extends_to_infinity(variant, comp_index, name,
                                             pair_all, pair_unk,
                                             pair_all_clamped,
                                             pair_unk_clamped):
    """H = 25.5-eps, 25.5, 26, 30: clamped keeps the top-bin value, the
    default returns 0 for the last three."""
    plain = (pair_all if variant == "All" else pair_unk)[comp_index]
    clamped = (pair_all_clamped if variant == "All"
               else pair_unk_clamped)[comp_index]
    assert plain.label == clamped.label == name

    el, top = _top_h_bin_density(plain)
    assert top > 0.0
    # ...and the clamped component agrees with it inside the bin.
    assert float(clamped.density(el, 24.5)) == top

    below = np.nextafter(25.5, 0.0)          # 25.5 - eps
    assert float(plain.density(el, below)) == top
    assert float(clamped.density(el, below)) == top

    for h in (25.5, 26.0, 30.0):
        assert float(plain.density(el, h)) == 0.0
        assert float(clamped.density(el, h)) == top, f"H = {h}"

    # The value is a density *per unit H* and the clamp does not rescale it.
    assert float(clamped.density(el, 1e4)) == top
    assert float(clamped.density(el, np.inf)) == top
    # nan is not a magnitude, in either mode.
    assert float(clamped.density(el, np.nan)) == 0.0
    assert float(plain.density(el, np.nan)) == 0.0


@pytest.mark.parametrize("comp_index, name", _COMPS)
def test_clamped_does_not_emulate_the_negative_H_half_of_the_clamp(
    pair_all, pair_all_clamped, comp_index, name
):
    """digest2 also puts H < 0 in bin 0; we deliberately do not (module
    docstring).  The clamp is one-sided."""
    plain = pair_all[comp_index]
    clamped = pair_all_clamped[comp_index]
    el, _ = _top_h_bin_density(plain)
    for h in (-1e-9, -1.0, -30.0):
        assert float(clamped.density(el, h)) == 0.0
        assert float(plain.density(el, h)) == 0.0
    import digest42.population as _pop
    assert "deliberately NOT emulated" in _pop.__doc__


@pytest.mark.parametrize("comp_index, name", _COMPS)
def test_clamped_and_plain_are_bit_identical_below_25p5(
    pair_all, pair_all_clamped, comp_index, name
):
    """Random sweep over the whole (q, e, i) box with H < 25.5: the two modes
    must return the *same float*, not merely a close one.  This is what makes
    the clamped-vs-unclamped score difference attributable to the top bin."""
    plain = pair_all[comp_index]
    clamped = pair_all_clamped[comp_index]
    rng = np.random.default_rng(20260901)
    n = 200_000
    # Spread beyond the support on every axis so the truncation branches are
    # exercised too (negative values, above the top edge, nan).
    # Most of the model's mass sits at q < 6 AU and i < 35 deg, so a sweep
    # uniform over the whole box would land almost entirely in empty bins;
    # concentrate 90% of the draws where the counts are and send the rest out
    # to the far corners.
    q = np.where(rng.random(n) < 0.9,
                 rng.uniform(-0.2, 6.0, n), rng.uniform(-1.0, 110.0, n))
    e = rng.uniform(-0.05, 1.2, n)
    i = np.where(rng.random(n) < 0.9,
                 rng.uniform(-0.02, 0.6, n),
                 rng.uniform(-0.05, np.pi + 0.05, n))
    h = rng.uniform(-2.0, 25.5, n)
    h[rng.random(n) < 0.01] = np.nan
    q[rng.random(n) < 0.01] = np.nan
    el = np.stack([q, e, i], axis=-1)

    a = plain.density(el, h)
    b = clamped.density(el, h)
    assert np.array_equal(a, b)                     # bit-for-bit, nan-free
    # The NEO component covers only q < 1.3 AU, so its hit rate is a few
    # percent; a thousand non-zero evaluations is ample evidence that the
    # sweep exercises the lookup rather than just the truncation branches.
    assert (a > 0).sum() > 1000                     # the sweep hits the model
    assert (a == 0.0).sum() > 0.05 * n              # ...and the truncations

    # And the bin lookup agrees too, below 25.5.
    ia = plain.bin_indices(el, h)
    ib = clamped.bin_indices(el, h)
    for x, y in zip(ia, ib):
        assert np.array_equal(x, y)


@pytest.mark.parametrize("comp_index, name", _COMPS)
def test_clamped_sweep_above_25p5_is_the_top_bin(pair_all, pair_all_clamped,
                                                 comp_index, name):
    """Above 25.5 the clamped density equals the same (q, e, i) evaluated in
    the top H bin — for every point, not just the one hand-picked bin."""
    plain = pair_all[comp_index]
    clamped = pair_all_clamped[comp_index]
    rng = np.random.default_rng(4242)
    n = 50_000
    q = rng.uniform(0.0, 100.0, n)
    e = rng.uniform(0.0, 1.1, n)
    i = rng.uniform(0.0, np.pi, n)
    el = np.stack([q, e, i], axis=-1)
    h_hi = rng.uniform(25.5, 60.0, n)
    ref = plain.density(el, np.full(n, 24.75))      # inside [24, 25.5)
    assert np.array_equal(clamped.density(el, h_hi), ref)
    assert np.all(plain.density(el, h_hi) == 0.0)
    assert (ref > 0).any()


@pytest.mark.parametrize("comp_index, name", _COMPS)
def test_clamped_support_reports_infinite_H(pair_all, pair_all_clamped,
                                            comp_index, name):
    plain = pair_all[comp_index]
    clamped = pair_all_clamped[comp_index]
    assert plain.support["H"] == (0.0, 25.5)
    assert clamped.support["H"] == (0.0, float("inf"))
    # every other axis is untouched
    for axis in ("q", "e", "i"):
        assert clamped.support[axis] == plain.support[axis]
    assert plain.h_top_bin_infinite is False
    assert clamped.h_top_bin_infinite is True


@pytest.mark.parametrize("comp_index, name", _COMPS)
def test_clamped_sample_raises(pair_all_clamped, comp_index, name):
    comp = pair_all_clamped[comp_index]
    with pytest.raises(NotImplementedError, match="h_top_bin_infinite"):
        comp.sample(10, np.random.default_rng(0))
    with pytest.raises(NotImplementedError) as exc:
        comp.sample(0, np.random.default_rng(0))     # not even the n == 0 path
    msg = str(exc.value)
    assert "diverges" in msg and "gate-ineligible" in msg


@pytest.mark.parametrize("comp_index, name", _COMPS)
def test_clamped_total_count_is_the_finite_bin_sum_and_says_so(
    pair_all, pair_all_clamped, comp_index, name
):
    plain = pair_all[comp_index]
    clamped = pair_all_clamped[comp_index]
    assert clamped.total_count == plain.total_count
    assert clamped.total_count == pytest.approx(float(clamped.counts.sum()))
    doc = Digest2Histogram.total_count.__doc__
    assert "not** the integral" in doc or "not the integral" in doc
    assert "diverges" in doc


def test_clamped_flag_passes_through_the_factories():
    """neo_other_pair and from_classes must both carry the flag, and
    neo_other_pair must clamp BOTH components (a mixed pair is meaningless)."""
    for variant in D2_VARIANTS:
        pair = Digest2Histogram.neo_other_pair(MODEL_PATH, variant,
                                               h_top_bin_infinite=True)
        assert [c.h_top_bin_infinite for c in pair] == [True, True]
        assert [c.label for c in pair] == ["NEO", "other"]
        assert [c.variant for c in pair] == [variant, variant]
        default = Digest2Histogram.neo_other_pair(MODEL_PATH, variant)
        assert [c.h_top_bin_infinite for c in default] == [False, False]

    c = Digest2Histogram.from_classes(MODEL_PATH, ["MB1"], "other",
                                      h_top_bin_infinite=True)
    assert c.h_top_bin_infinite is True
    assert Digest2Histogram.from_classes(
        MODEL_PATH, ["MB1"], "other"
    ).h_top_bin_infinite is False
    assert "h_top_bin_infinite=True" in repr(c)


def test_clamped_mode_is_documented_as_gate_ineligible():
    """A comparison mode that leaked into the gate would silently invalidate
    DESIGN §4.1; the class docstring must say so in as many words."""
    doc = Digest2Histogram.__doc__
    assert "gate-ineligible" in doc
    assert "§4.1" in doc
    assert "normalizable" in doc


# ---------------------------------------------------------------------------
# Smooth-in-H reconstruction (h_interpolation="pchip")
#
# The construction is: interpolate the CUMULATIVE count over the 19 H edges
# with a monotone cubic, evaluate its derivative as the density.  Every test
# below pins one of the properties that makes that safe — per-bin mass
# conservation, non-negativity, continuity across interior edges, D-4
# truncation, and no mass invented outside the bin it came from (the
# WP4-review objection).  Where a quantity can be re-derived independently
# (the PCHIP slopes, the per-bin mass by quadrature *through* density()) it is,
# so a bug in the coefficient construction cannot hide behind the same bug in
# the test.
# ---------------------------------------------------------------------------

GL8_X, GL8_W = np.polynomial.legendre.leggauss(8)


@pytest.fixture(scope="module")
def pair_pchip():
    return Digest2Histogram.neo_other_pair(MODEL_PATH, variant="All",
                                           h_interpolation="pchip")


def cell_centres():
    """(n_cells, 3) centres of every (q, e, i) cell, in the qei parameterization."""
    q = 0.5 * (np.concatenate(([0.0], REF_QPART[:-1])) + REF_QPART)
    e = 0.5 * (np.concatenate(([0.0], REF_EPART[:-1])) + REF_EPART)
    i = np.radians(
        0.5 * (np.concatenate(([0.0], REF_IPART_DEG[:-1])) + REF_IPART_DEG)
    )
    Q, E, I = np.meshgrid(q, e, i, indexing="ij")
    return np.stack((Q, E, I), axis=-1).reshape(-1, 3)


def qei_volume_flat():
    """dq * de * di[rad] per cell, flattened to match :func:`cell_centres`."""
    dq = np.diff(np.concatenate(([0.0], REF_QPART)))
    de = np.diff(np.concatenate(([0.0], REF_EPART)))
    di = np.diff(np.radians(np.concatenate(([0.0], REF_IPART_DEG))))
    return (dq[:, None, None] * de[None, :, None] * di[None, None, :]).reshape(-1)


def naive_pchip_slopes(x, y):
    """Scalar, loop-based Fritsch-Carlson transcription (no numpy tricks).

    Written from the published construction, not from population.py:
    interior slopes are the weighted harmonic mean of the neighbouring
    secants (zero where they change sign or either vanishes), end slopes are
    the one-sided three-point estimate with the sign and 3-delta limiters.
    """
    n = len(x) - 1
    h = [x[k + 1] - x[k] for k in range(n)]
    d = [(y[k + 1] - y[k]) / h[k] for k in range(n)]
    m = [0.0] * (n + 1)
    for k in range(1, n):
        if d[k - 1] * d[k] > 0.0:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
        else:
            m[k] = 0.0

    def end(h0, h1, d0, d1):
        v = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if math.copysign(1.0, v) != math.copysign(1.0, d0):
            return 0.0
        if (math.copysign(1.0, d0) != math.copysign(1.0, d1)
                and abs(v) > 3.0 * abs(d0)):
            return 3.0 * d0
        return v

    m[0] = end(h[0], h[1], d[0], d[1])
    m[n] = end(h[n - 1], h[n - 2], d[n - 1], d[n - 2])
    return m


def test_pchip_slopes_match_an_independent_transcription():
    from digest42.population import _pchip_slopes

    rng = np.random.default_rng(4242)
    x = np.array([0.0, 0.3, 1.0, 1.1, 4.0, 9.0, 9.5])
    for trial in range(200):
        # Non-decreasing data, including flats and hard steps — exactly the
        # shapes a sparse cumulative count produces.
        inc = rng.choice([0.0, 0.0, 1e-9, 1.0, 1e4], size=x.size - 1)
        y = np.concatenate(([0.0], np.cumsum(inc)))
        m, _ = _pchip_slopes(x, y)
        ref = naive_pchip_slopes(list(x), list(y))
        # rtol, not exact equality: the reference evaluates the harmonic mean
        # as (w1+w2)/(w1/d0 + w2/d1) and the implementation as its reciprocal
        # form 1/whmean (which is what scipy does, and what the next test
        # pins).  Algebraically the same, different last bit.
        assert np.allclose(m, ref, rtol=1e-14, atol=0.0), (trial, m, ref)


def test_pchip_slopes_match_scipy_when_scipy_is_installed():
    """Cross-check against the canonical implementation, if it is importable.

    scipy is NOT a digest42 dependency [D-9] — hence the hand transcription in
    the test above, which is the load-bearing check.  When scipy happens to be
    present this pins us to the same construction the rest of the world calls
    PCHIP.
    """
    interp = pytest.importorskip("scipy.interpolate")
    from digest42.population import _pchip_slopes

    rng = np.random.default_rng(99)
    x = np.array([0.0, 6.0, 8.0, 10.0, 11.0, 12.0, 25.5])
    for _ in range(50):
        y = np.concatenate(
            ([0.0], np.cumsum(rng.choice([0.0, 1.0, 700.0], size=x.size - 1)))
        )
        m, _ = _pchip_slopes(x, y)
        sp = interp.PchipInterpolator(x, y)
        # PPoly coefficients: p(x) = c0 s^3 + c1 s^2 + c2 s + c3, so c[2] IS
        # scipy's slope at each interval's left knot.  Compared bit-exactly.
        assert np.array_equal(m[:-1], sp.c[2]), (m[:-1], sp.c[2])
        # The right-hand end knot has no interval of its own; scipy can only
        # be asked for it by evaluating the last segment's derivative at its
        # right end, which costs a couple of ulp of the segment's scale.
        scale = max(1e-300, float(np.abs(np.diff(y) / np.diff(x)).max()))
        assert abs(m[-1] - float(sp(x[-1], 1))) < 1e-12 * scale


def test_pchip_slopes_obey_the_monotonicity_bound(pair_pchip):
    """0 <= m_k <= 3 min(delta_{k-1}, delta_k) — the Fritsch-Carlson condition.

    This is the *mechanism* behind non-negativity and mass containment, so it
    is asserted directly on the shipped model rather than only through its
    consequences.  ``alpha, beta in [0, 3]`` puts every interval inside the
    monotonicity region, which is what makes C' >= 0.
    """
    from digest42.population import _pchip_slopes
    from digest42.population import H_EDGES

    for comp in pair_pchip:
        cum = np.zeros(comp.counts.shape[:3] + (19,))
        cum[..., 1:] = np.cumsum(comp.counts, axis=-1)
        m, delta = _pchip_slopes(H_EDGES, cum)
        assert m.min() >= 0.0
        assert delta.min() >= 0.0
        # every interval: both end slopes within 3x its own secant
        for k in range(18):
            lim = 3.0 * delta[..., k]
            assert np.all(m[..., k] <= lim + 1e-9 * np.maximum(lim, 1.0))
            assert np.all(m[..., k + 1] <= lim + 1e-9 * np.maximum(lim, 1.0))


def test_pchip_mass_is_conserved_in_every_h_bin_of_every_cell(pair_pchip):
    """(a) EXACT per-bin mass conservation, measured through density().

    8-node Gauss-Legendre is exact for the quadratic that density() evaluates,
    so this integrates the *reconstructed density itself* — not the
    coefficients' antiderivative — over all 2552 x 18 (cell, H bin) pairs of
    both components and compares to the stored object counts.
    """
    el = cell_centres()
    vol = qei_volume_flat() * ANGLE_VOLUME
    edges = np.concatenate(([0.0], REF_HPART))
    for comp in pair_pchip:
        counts = np.asarray(comp.counts).reshape(-1, len(REF_HPART))
        worst_rel = 0.0
        worst_abs = 0.0
        for k in range(len(REF_HPART)):
            lo, hi = edges[k], edges[k + 1]
            half = 0.5 * (hi - lo)
            acc = np.zeros(el.shape[0])
            for xg, wg in zip(GL8_X, GL8_W):
                h = half * xg + 0.5 * (hi + lo)
                acc += wg * comp.density(el, np.full(el.shape[0], h))
            mass = half * acc * vol
            dev = np.abs(mass - counts[:, k])
            nz = counts[:, k] > 0
            worst_abs = max(worst_abs, float(dev.max()))
            if nz.any():
                worst_rel = max(
                    worst_rel, float((dev[nz] / counts[nz, k]).max())
                )
            # empty bins must be EXACTLY empty, not almost
            assert np.all(mass[~nz] == 0.0)
        assert worst_rel < 1e-12, (comp.label, worst_rel)
        assert worst_abs < 1e-6, (comp.label, worst_abs)


def test_pchip_mass_identity_holds_from_the_coefficients(pair_pchip):
    """The same guarantee straight off the coefficient table, to 1e-12 rel.

    ``int_0^dH (a + b s + c s^2) ds = a dH + b dH^2/2 + c dH^3/3`` must be the
    bin's object count.  Cheap enough to run over every cell of both
    components, which the quadrature version above also does — the two agree
    because the quadrature is exact for a quadratic.
    """
    dH = np.diff(np.concatenate(([0.0], REF_HPART)))
    for comp in pair_pchip:
        a, b, c = (comp.h_coeffs[..., i] for i in range(3))
        integral = a * dH + b * dH**2 / 2.0 + c * dH**3 / 3.0
        mass = integral * (
            np.asarray(comp.qei_volume)[..., None] * ANGLE_VOLUME
        )
        counts = np.asarray(comp.counts)
        nz = counts > 0
        rel = np.abs(mass[nz] - counts[nz]) / counts[nz]
        assert rel.max() < 1e-12, (comp.label, rel.max())
        assert np.all(mass[~nz] == 0.0)


def test_pchip_density_is_non_negative_on_a_dense_sweep(pair_pchip):
    """(b) C' >= 0 everywhere, checked on 200 points per H bin x every cell."""
    el = cell_centres()
    edges = np.concatenate(([0.0], REF_HPART))
    frac = np.linspace(0.0, 1.0, 201)[:-1]      # [0, 1): half-open, like the bin
    worst = np.inf
    for comp in pair_pchip:
        for k in range(len(REF_HPART)):
            for f in frac:
                h = edges[k] + f * (edges[k + 1] - edges[k])
                d = comp.density(el, np.full(el.shape[0], h))
                worst = min(worst, float(d.min()))
        assert worst >= 0.0, (comp.label, worst)
    # 0.0 is attained (empty cells, and the flat ends of sparse ones), so the
    # bound is tight, not vacuous.
    assert worst == 0.0


def test_pchip_is_continuous_across_every_interior_h_edge(pair_pchip):
    """(c) both one-sided limits agree at all 17 interior edges, every cell.

    Measured relative to each cell's own peak density, because the absolute
    scale spans ~15 decades across cells.  Continuity is structural: PCHIP is
    C1, so its derivative is C0 — the coefficient identity ``a_{k} =
    (a + s(b + sc))|_{s = dH}`` of interval k-1 is what is checked here.
    """
    el = cell_centres()
    edges = np.concatenate(([0.0], REF_HPART))
    centres = 0.5 * (edges[:-1] + edges[1:])
    for comp in pair_pchip:
        scale = np.maximum(
            np.max([comp.density(el, np.full(el.shape[0], h))
                    for h in centres], axis=0),
            np.finfo(float).tiny,
        )
        for k in range(1, len(REF_HPART)):
            e = edges[k]
            left = comp.density(el, np.full(el.shape[0], np.nextafter(e, -1.0)))
            right = comp.density(el, np.full(el.shape[0], e))
            rel = np.abs(left - right) / scale
            assert rel.max() < 1e-12, (comp.label, e, rel.max())


def test_pchip_is_exactly_zero_outside_the_h_support(pair_pchip):
    """(d) D-4 unchanged: 0.0 below 0 and at/above 25.5, bit-exactly."""
    el = cell_centres()
    for comp in pair_pchip:
        for h in (-np.finfo(float).tiny, -1e-9, -1.0, -1e9,
                  25.5, np.nextafter(25.5, 100.0), 26.0, 1e9,
                  np.inf, -np.inf, np.nan):
            d = comp.density(el, np.full(el.shape[0], h))
            assert np.all(d == 0.0), (comp.label, h, d.max())


def test_pchip_density_is_discontinuous_at_the_support_endpoints(pair_pchip):
    """(d, second half) The endpoint jumps are REAL and documented.

    The density outside is exactly 0 (D-4), and just inside it is *generally*
    non-zero, so both support endpoints carry a real jump.  "Generally", not
    "always": at a cell whose next bin is very much more populated the
    one-sided three-point end slope comes out with the wrong sign and PCHIP's
    shape-preserving limiter sets it to exactly 0 — which is the interpolant
    doing its job, and leaves that cell's endpoint value continuous with the
    zero outside.  So the assertion is that the jump happens for a
    substantial fraction of populated cells, plus a deterministic synthetic
    case at each end.  Either way it is a documented property, not a defect:
    the histogram mode has the very same two jumps.
    """
    el = cell_centres()
    for comp in pair_pchip:
        counts = np.asarray(comp.counts)
        for k, h_in in ((0, 0.0), (-1, np.nextafter(25.5, 0.0))):
            pop = counts[..., k].reshape(-1) > 0
            if not pop.any():
                # The NEO component has no object with H < 6; only "other"
                # (268 of them) exercises the bright end on the real file.
                assert comp.label == "NEO" and k == 0
                continue
            d_in = comp.density(el, np.full(el.shape[0], h_in))
            assert np.all(d_in[pop] >= 0.0)
            # A real jump at a good fraction of the populated cells.
            # Measured on the All variant: bright end (H = 0) 15.7% of
            # "other"'s 159 populated cells — the size distribution rises so
            # steeply out of [0, 6) that the limiter zeroes most of them;
            # faint end (H -> 25.5) 100% of NEO's and 100% of other's, because
            # the top bin is where the mass is.
            assert (d_in[pop] > 0.0).mean() > 0.1, (comp.label, k)
        # exactly zero a hair outside, on every cell
        assert comp.density(el, np.full(el.shape[0],
                                        -np.finfo(float).tiny)).max() == 0.0
        assert comp.density(el, np.full(el.shape[0], 25.5)).max() == 0.0

    # Deterministic single-bin cells: mass only in the first (resp. last) H
    # bin, so the end slope has nothing to be limited against and the jump is
    # unambiguous.
    hot = np.array([[0.85, 0.25, np.radians(17.0)]])
    for k, h_in, h_out in ((0, 0.0, -np.finfo(float).tiny),
                           (17, np.nextafter(25.5, 0.0), 25.5)):
        counts = np.zeros(MODEL_SHAPE)
        counts[3, 2, 4, k] = 500.0
        comp = Digest2Histogram(counts, "other", h_interpolation="pchip")
        assert comp.density(hot, np.array([h_in]))[0] > 0.0
        assert comp.density(hot, np.array([h_out]))[0] == 0.0

    for doc in (Digest2Histogram.__doc__,
                __import__("digest42.population", fromlist=["x"]).__doc__):
        assert "discontinuous" in doc


def test_pchip_leaves_totals_support_and_qei_lookup_untouched(pair_all, pair_pchip):
    """(e) total_count, support and the q/e/i/H bin lookup are bit-identical."""
    rng = np.random.default_rng(2026)
    n = 200_000
    el = np.stack((rng.uniform(-1.0, 110.0, n),
                   rng.uniform(-0.05, 1.2, n),
                   rng.uniform(-0.05, 3.2, n)), axis=-1)
    h = rng.uniform(-1.0, 27.0, n)
    for hist, smooth in zip(pair_all, pair_pchip):
        assert smooth.total_count == hist.total_count
        assert smooth.support == hist.support
        assert smooth.support["H"] == (0.0, 25.5)
        a = hist.bin_indices(el, h)
        b = smooth.bin_indices(el, h)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)
        # in-support decisions, hence the entire zero pattern of density(),
        # must be identical too
        assert np.array_equal(hist.density(el, h) == 0.0,
                              smooth.density(el, h) == 0.0)
        assert np.asarray(smooth.counts).tobytes() == np.asarray(hist.counts).tobytes()


def test_pchip_reduces_to_the_histogram_when_the_cell_is_flat_in_h():
    """A cell whose counts are proportional to dH is *already* flat in H.

    Then the cumulative count is exactly linear, PCHIP reproduces the straight
    line, and the reconstructed density must agree with the piecewise-constant
    one.  This is the one configuration where the two modes are required to
    coincide, and it catches a wrong bin volume, a wrong dH, or a mis-scaled
    coefficient — none of which the mass-conservation tests can see, because
    they compare against the same counts either way.
    """
    dH = np.diff(np.concatenate(([0.0], REF_HPART)))
    counts = np.zeros(MODEL_SHAPE)
    counts[5, 3, 6, :] = 1000.0 * dH
    counts[0, 0, 0, :] = 7.0 * dH
    hist = Digest2Histogram(counts, "other")
    smooth = Digest2Histogram(counts, "other", h_interpolation="pchip")
    el = cell_centres()
    for h in np.linspace(0.0, 25.49, 400):
        a = hist.density(el, np.full(el.shape[0], h))
        b = smooth.density(el, np.full(el.shape[0], h))
        assert np.allclose(a, b, rtol=1e-13, atol=0.0), h


def test_pchip_confines_a_sparse_cell_to_its_own_bin():
    """The WP4-review "interpolation invents density" objection, answered.

    A cell with counts in ONE H bin keeps all of its mass in that bin and has
    identically zero density in every other bin — including when the populated
    bin is the first or the last, where the end-slope formula is in play.  The
    mechanism is monotonicity plus interpolation of the exact cumulative
    count; nothing here is a tuned tolerance.
    """
    el = np.array([[0.85, 0.25, np.radians(17.0)]])
    edges = np.concatenate(([0.0], REF_HPART))
    vol = float(
        (0.9 - 0.8) * (0.3 - 0.2) * np.radians(20.0 - 15.0)
    ) * ANGLE_VOLUME
    for populated in range(len(REF_HPART)):
        counts = np.zeros(MODEL_SHAPE)
        counts[3, 2, 4, populated] = 1234.0
        comp = Digest2Histogram(counts, "other", h_interpolation="pchip")
        for k in range(len(REF_HPART)):
            lo, hi = edges[k], edges[k + 1]
            half = 0.5 * (hi - lo)
            acc = 0.0
            for xg, wg in zip(GL8_X, GL8_W):
                acc += wg * float(
                    comp.density(el, np.array([half * xg + 0.5 * (hi + lo)]))[0]
                )
            mass = half * acc * vol
            if k == populated:
                assert abs(mass - 1234.0) < 1e-9, (populated, k, mass)
            else:
                # exactly zero: the interpolant is CONSTANT on an empty
                # interval (both end slopes vanish), so the density is 0.0
                assert mass == 0.0, (populated, k, mass)


def test_pchip_empty_cell_invents_nothing():
    """A cell with zero total has zero density at every H.  No exceptions."""
    comp = Digest2Histogram(np.zeros(MODEL_SHAPE), "other",
                            h_interpolation="pchip")
    el = cell_centres()
    for h in np.linspace(0.0, 25.4999, 500):
        assert np.all(comp.density(el, np.full(el.shape[0], h)) == 0.0)
    assert comp.total_count == 0.0
    assert np.all(np.asarray(comp.h_coeffs) == 0.0)


def test_pchip_sample_stays_inside_the_drawn_bin_and_the_support(pair_pchip):
    for comp in pair_pchip:
        rng = np.random.default_rng(31337)
        el6, h = comp.sample(300_000, rng)
        assert np.all(np.isfinite(h))
        assert h.min() >= 0.0 and h.max() < 25.5
        # the drawn H must fall in a bin the cell actually has counts in
        q = el6[:, 0] * (1.0 - el6[:, 1])
        iq, ie, ii, ih, ok = comp.bin_indices(
            np.stack((q, el6[:, 1], el6[:, 2]), axis=-1), h
        )
        bound = el6[:, 1] < 1.0
        assert np.all(ok[bound])
        assert np.all(np.asarray(comp.counts)[iq[bound], ie[bound],
                                              ii[bound], ih[bound]] > 0.0)


def test_pchip_sample_leaves_the_elements_bit_identical(pair_all, pair_pchip):
    """Only the within-bin H draw changes; (a, e, i, Omega, omega, M) do not.

    The pchip sampler consumes exactly one uniform per draw for H, in the same
    position in the stream as the histogram sampler's, so with the same seed
    every other column comes out bit-for-bit identical — which is the
    strongest available statement of "elements handling unchanged".
    """
    for hist, smooth in zip(pair_all, pair_pchip):
        a6, ah = hist.sample(50_000, np.random.default_rng(7))
        b6, bh = smooth.sample(50_000, np.random.default_rng(7))
        assert np.array_equal(a6, b6)
        assert not np.array_equal(ah, bh)          # H really did change
        # and the H bin each draw lands in is the same one
        el = np.stack((a6[:, 0] * (1.0 - a6[:, 1]), a6[:, 1], a6[:, 2]), -1)
        assert np.array_equal(hist.bin_indices(el, ah)[3],
                              smooth.bin_indices(el, bh)[3])


@pytest.mark.parametrize("comp_index,name", [(0, "NEO"), (1, "other")])
def test_pchip_sample_reproduces_the_analytic_density(pair_pchip, comp_index,
                                                      name):
    """chi-square of 2e6 draws against the analytic mass on FINE sub-bins.

    Eight sub-bins per H bin (144 total): far finer than the histogram the
    bin *selection* reproduces, so this tests the inverse-CDF draw inside the
    bin, i.e. that sample() and density() describe the same distribution.
    The expected masses are the coefficients' exact antiderivative summed over
    cells, so a mis-inverted CDF cannot cancel against a matching bug.
    """
    comp = pair_pchip[comp_index]
    assert comp.label == name
    m = 8
    edges = np.concatenate(([0.0], REF_HPART))
    vol = (np.asarray(comp.qei_volume) * ANGLE_VOLUME)
    co = comp.h_coeffs
    sub_edges = []
    expected = []
    for k in range(len(REF_HPART)):
        lo, hi = edges[k], edges[k + 1]
        s = np.linspace(0.0, hi - lo, m + 1)
        a, b, c = co[..., k, 0], co[..., k, 1], co[..., k, 2]

        def anti(t):
            return t * (a + t * (0.5 * b + t * c / 3.0))

        for j in range(m):
            expected.append(float(((anti(s[j + 1]) - anti(s[j])) * vol).sum()))
            sub_edges.append(lo + s[j])
    sub_edges.append(25.5)
    expected = np.array(expected)
    # sanity: the sub-bin masses must sum to the component's total count
    assert abs(expected.sum() - comp.total_count) < 1e-6

    n = 2_000_000
    _, h = comp.sample(n, np.random.default_rng(20260902))
    obs, _ = np.histogram(h, bins=np.array(sub_edges))
    exp = expected / expected.sum() * n
    use = exp > 20.0
    chi2 = float((((obs[use] - exp[use]) ** 2) / exp[use]).sum())
    dof = int(use.sum()) - 1
    # chi2/dof ~ 1 +- sqrt(2/dof); 1.45 is ~5 sigma at dof = 93
    assert chi2 / dof < 1.45, (name, chi2 / dof, dof)
    assert np.abs((obs[use] - exp[use]) / np.sqrt(exp[use])).max() < 5.0


def test_pchip_sample_inversion_is_converged(pair_pchip):
    """The safeguarded Newton solve reaches machine precision.

    Re-derives the residual of the inverse-CDF solve for 2e5 random
    (cell, bin, u) triples using the same coefficients the sampler uses:
    |C(s) - u C(dH)| / C(dH) must be at the 1e-15 level, so the drawn H is the
    exact quantile to double precision and the chi-square test above is not
    passing on a wide tolerance.
    """
    comp = pair_pchip[1]
    co = comp.h_coeffs
    counts = np.asarray(comp.counts)
    rng = np.random.default_rng(5)
    nz = np.argwhere(counts > 0)
    sel = nz[rng.integers(0, len(nz), 200_000)]
    iq, ie, ii, ih = (sel[:, k] for k in range(4))
    a, b, c = (co[iq, ie, ii, ih, k] for k in range(3))
    w = (comp.h_edges[ih + 1] - comp.h_edges[ih]).astype(float)

    def cdf(s):
        return s * (a + s * (0.5 * b + s * c / 3.0))

    total = cdf(w)
    target = rng.random(len(iq)) * total
    lo = np.zeros_like(w)
    hi = w.copy()
    s = w * (target / total)
    for it in range(60):
        f = cdf(s) - target
        lo = np.where(f <= 0.0, s, lo)
        hi = np.where(f > 0.0, s, hi)
        mid = 0.5 * (lo + hi)
        d = a + s * (b + s * c)
        with np.errstate(divide="ignore", invalid="ignore"):
            cand = s - np.where(d > 0.0, f / d, np.nan)
        take = np.isfinite(cand) & (cand > lo) & (cand < hi) & (it % 5 != 4)
        s = np.where(take, cand, mid)
    res = np.abs(cdf(s) - target) / total
    assert res.max() < 1e-13, res.max()


def test_pchip_and_the_d14_clamp_are_mutually_exclusive():
    counts = np.zeros(MODEL_SHAPE)
    with pytest.raises(ValueError, match="h_top_bin_infinite"):
        Digest2Histogram(counts, "other", h_interpolation="pchip",
                         h_top_bin_infinite=True)
    with pytest.raises(ValueError, match="h_top_bin_infinite"):
        Digest2Histogram.neo_other_pair(MODEL_PATH, "All",
                                        h_interpolation="pchip",
                                        h_top_bin_infinite=True)
    with pytest.raises(ValueError, match="h_top_bin_infinite"):
        Digest2Histogram.from_classes(MODEL_PATH, ["MB1"], "other",
                                      h_interpolation="pchip",
                                      h_top_bin_infinite=True)
    # each alone is fine
    assert Digest2Histogram(counts, "other",
                            h_interpolation="pchip").h_coeffs is not None
    assert Digest2Histogram(counts, "other",
                            h_top_bin_infinite=True).h_coeffs is None


@pytest.mark.parametrize("bad", ["linear", "cubic", "spline", "None", "",
                                 "PCHIP", "pchip "])
def test_unknown_h_interpolation_names_the_valid_options(bad):
    from digest42.population import H_INTERPOLATIONS

    with pytest.raises(ValueError) as ei:
        Digest2Histogram(np.zeros(MODEL_SHAPE), "other", h_interpolation=bad)
    msg = str(ei.value)
    assert repr(bad) in msg
    for opt in H_INTERPOLATIONS:
        assert repr(opt) in msg


def test_h_interpolation_passes_through_the_factories():
    for mode in ("none", "pchip"):
        pair = Digest2Histogram.neo_other_pair(MODEL_PATH, "All",
                                               h_interpolation=mode)
        assert [c.h_interpolation for c in pair] == [mode, mode]
        assert all((c.h_coeffs is None) == (mode == "none") for c in pair)
        one = Digest2Histogram.from_classes(MODEL_PATH, ["MB1"], "other",
                                            h_interpolation=mode)
        assert one.h_interpolation == mode
    # the default is the D-10 histogram everywhere
    assert [c.h_interpolation for c in
            Digest2Histogram.neo_other_pair(MODEL_PATH, "All")] \
        == ["none", "none"]
    assert Digest2Histogram(np.zeros(MODEL_SHAPE), "x").h_interpolation == "none"
    assert "h_interpolation='pchip'" in repr(
        Digest2Histogram(np.zeros(MODEL_SHAPE), "x", h_interpolation="pchip")
    )
    assert "h_interpolation" not in repr(Digest2Histogram(np.zeros(MODEL_SHAPE), "x"))


def test_pchip_coefficient_table_is_read_only_and_small(pair_pchip):
    for comp in pair_pchip:
        co = comp.h_coeffs
        assert co.shape == MODEL_SHAPE + (3,)
        assert not co.flags.writeable
        with pytest.raises(ValueError):
            co[0, 0, 0, 0, 0] = 1.0
        # 2552 cells x 18 intervals x 3 coefficients: ~1.1 MB per component.
        assert co.nbytes < 2 * 1024 * 1024


def test_full_qei_interpolation_is_still_refused_and_points_at_the_h_knob():
    with pytest.raises(NotImplementedError) as ei:
        Digest2Histogram(np.zeros(MODEL_SHAPE), "other", interpolate=True)
    assert "h_interpolation='pchip'" in str(ei.value)


def test_pchip_mode_documents_what_it_does_not_fix():
    """DESIGN §2.5 has two separate quadrature problems; this fixes one.

    The docstrings must not let a reader believe the smooth mode also cures
    the 1/(e sin i) integrable singularity — that lives on the e and i bin
    edges and is a property of the measure, not of the H reconstruction.
    """
    doc = __import__("digest42.population", fromlist=["x"]).__doc__
    assert "e sin i" in doc
    assert "staircase" in doc
    assert "cumulative" in doc
    assert "invents density" in doc
