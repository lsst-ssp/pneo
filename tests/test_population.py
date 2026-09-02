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
