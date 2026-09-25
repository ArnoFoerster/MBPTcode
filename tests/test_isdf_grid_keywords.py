"""Every grid keyword of `isdf_jk.isdf_grid` reaches the grid, or the call is refused.

The interpolation grid is ONE object -- the shell counts, the radii on those
shells, and whether the nucleus itself is sampled -- exactly as for
`space_time.separable_factors` (see `test_separable_factors_grid_keywords.py`,
the reference this mirrors). A keyword silently dropped here does not make the
SCF's J/K slower, it makes it a DIFFERENT factorization: `isdf_jk.ISDFJK.build`
calls `isdf_grid` for every mean field it fits, so a dropped keyword changes
the exchange matrix and, through it, every energy and gradient built on top,
while looking like an ordinary run.

THE SWEEP WAS SHOWN TO FAIL ONCE. `isdf_grid` never checked its explicit-radii
branch against the shipped table: it set `origins = {el: False for el in
radii}` unconditionally, so a caller who passed the published carbon radii
alongside their own counts got a grid with the nuclear point silently
dropped. Reinstating that branch on a backup copy of `isdf_jk.py` (`cp`,
overwrite the fixed `isdf_grid` with the old one, `pytest`, restore, `cmp`
against the backup to confirm the restore) makes
`test_a_matching_row_brings_its_nuclear_point` fail: the counts-only lookup
returns 307 points (the published row carries `origin`) while the explicit-
radii call returns 306, one point short at the nucleus. The same reinstated
copy also has no `grid_accuracy` keyword at all, so every `grid_accuracy` test
below fails with `TypeError: unexpected keyword argument`.

Each refusal is paired with the request it must NOT refuse, because a check
that fires on everything is not a check either.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto

from src.Base import isdf_jk
from src.Base.constants import ISDF_GRID_ACCURACY, ISDF_RADII_MATCH_TOL
from src.Base.isdf_jk import DEFAULT_COUNTS, isdf_grid
from src.Base.separable_ri import (PUBLISHED_COUNTS, _SHELL_ORDER, atomic_grid,
                                   shipped_radii_lookup)

BASIS = 'cc-pvtz'
AUXBASIS = BASIS + '-ri'
ETHYLENE = ('C 0.0 0.0 0.667; C 0.0 0.0 -0.667; H 0.0 0.923 1.238; '
            'H 0.0 -0.923 1.238; H 0.0 0.923 -1.238; H 0.0 -0.923 -1.238')
# Three sizes the shipped table holds for BOTH C and H at this basis, so the
# sweep exercises the lookup and never the run-time optimizer.
G2_COUNTS = dict(zip(_SHELL_ORDER, ISDF_GRID_ACCURACY[BASIS]['G2']))
G3_COUNTS = dict(zip(_SHELL_ORDER, ISDF_GRID_ACCURACY[BASIS]['G3']))
SWEEP = (DEFAULT_COUNTS, G2_COUNTS, G3_COUNTS)
# The published carbon grid is the one row that carries `origin`, and its
# counts appear nowhere else in the table.
CARBON_PUBLISHED_COUNTS = dict(zip(_SHELL_ORDER, PUBLISHED_COUNTS['C']))


@pytest.fixture(scope='module')
def ethylene():
    """The bare Mole -- `isdf_grid` reads only `mol`, no mean field needed."""
    return gto.M(atom=ETHYLENE, basis=BASIS, verbose=0)


@pytest.fixture(scope='module')
def carbon():
    """A single carbon atom, the only way to ask for the published carbon
    counts without dragging in an element that has no row there."""
    return gto.M(atom='C 0 0 0', basis=BASIS, spin=2, verbose=0)


def untabulated_counts(elements):
    """Counts the table holds for none of `elements`, found by walking A1 up.

    Derived rather than spelled out so the control cannot quietly become a
    tabulated row when the table grows.
    """
    counts = dict(DEFAULT_COUNTS)
    while any(shipped_radii_lookup(el, BASIS, AUXBASIS, counts) is not None
              for el in elements):
        counts['A1'] += 1
    return counts


def reoptimization_warnings(caught):
    """The warnings that say a grid was re-optimized at run time."""
    return [w for w in caught if 're-optimized at run time' in str(w.message)]


def test_a_counts_sweep_moves_the_grid(ethylene):
    """Three tabulated sizes, three different grids -- the point of asking."""
    mol = ethylene
    grids = []
    for counts in SWEEP:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pts = isdf_grid(mol, counts=counts)
        assert not reoptimization_warnings(caught), (
            f'counts {counts} re-optimized instead of reading its row: '
            f'{[str(w.message) for w in reoptimization_warnings(caught)]}')
        grids.append((counts, pts))
    sizes = [len(pts) for _, pts in grids]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes), (
        f'point counts {sizes} for {[c for c, _ in grids]}')
    for i, (counts_i, pts_i) in enumerate(grids):
        for counts_j, pts_j in grids[i + 1:]:
            assert pts_i.shape != pts_j.shape or not np.array_equal(
                pts_i, pts_j), f'{counts_i} and {counts_j} share a grid'


def test_nothing_asked_for_is_the_default_counts_lookup(ethylene):
    """A caller with no opinion gets `DEFAULT_COUNTS`, through the same path
    as a caller who names them -- the bitwise anchor for every bare call."""
    bare = isdf_grid(ethylene)
    named = isdf_grid(ethylene, counts=DEFAULT_COUNTS)
    assert np.array_equal(bare, named)


def test_grid_accuracy_agreeing_with_counts_proceeds(ethylene):
    """Two spellings of ONE grid are not a contradiction. Bitwise, because an
    accuracy level resolves to counts and must then take exactly the path
    those counts take."""
    level = isdf_grid(ethylene, grid_accuracy='G2')
    both = isdf_grid(ethylene, grid_accuracy='G2', counts=G2_COUNTS)
    assert np.array_equal(level, both)


def test_grid_accuracy_contradicting_counts_is_refused(ethylene):
    """Without this check the level's counts overwrote the caller's silently,
    so the grid returned was one the caller had explicitly asked against."""
    with pytest.raises(ValueError) as excinfo:
        isdf_grid(ethylene, grid_accuracy='G2', counts=G3_COUNTS)
    message = str(excinfo.value)
    assert str(dict(sorted(G3_COUNTS.items()))) in message
    assert str(dict(sorted(G2_COUNTS.items()))) in message


def test_explicit_radii_equal_to_the_row_are_the_row(ethylene):
    """Handing over the row's own radii with the row's own counts is one
    request, not two, and must reproduce the lookup path bitwise."""
    radii = {el: atomic_grid(el, BASIS, AUXBASIS, G2_COUNTS)[0]
             for el in ('C', 'H')}
    lookup = isdf_grid(ethylene, counts=G2_COUNTS)
    explicit = isdf_grid(ethylene, radii=radii, counts=G2_COUNTS)
    assert np.array_equal(lookup, explicit)


def test_a_matching_row_brings_its_nuclear_point(carbon):
    """`origin` belongs to the GRID, not to the recipe that found it.

    The published carbon row places one point at the nucleus, so honouring its
    radii while dropping the flag builds 306 points where the row describes
    307 -- a different grid, reported under the published grid's name. This is
    the defect the sweep was shown to fail on; see the module docstring.
    """
    radii, origin = atomic_grid('C', BASIS, AUXBASIS, CARBON_PUBLISHED_COUNTS)
    assert origin, 'the published carbon row is the case this test exists for'
    lookup = isdf_grid(carbon, counts=CARBON_PUBLISHED_COUNTS)
    explicit = isdf_grid(carbon, radii={'C': radii},
                         counts=CARBON_PUBLISHED_COUNTS)
    assert np.array_equal(lookup, explicit)
    assert len(lookup) == 307
    # The same radii where no row claims them: the nuclear point is the
    # caller's to add, so the grid is one point smaller.
    no_row = isdf_grid(carbon, radii={'C': radii},
                       counts=untabulated_counts(('C',)))
    assert len(explicit) == len(no_row) + 1


def test_radii_contradicting_the_row_are_refused(ethylene):
    """A perturbation of 1e-3 Bohr is 1e7 match tolerances and a different
    grid; resolving to either side would hand back a factorization one of the
    two keywords asked against."""
    radii = {el: atomic_grid(el, BASIS, AUXBASIS, G2_COUNTS)[0]
             for el in ('C', 'H')}
    table = np.atleast_1d(radii['C']['A1']).copy()
    bad = {el: dict(r) for el, r in radii.items()}
    bad['C']['A1'] = table + np.eye(len(table))[0] * 1e-3
    with pytest.raises(ValueError) as excinfo:
        isdf_grid(ethylene, radii=bad, counts=G2_COUNTS)
    message = str(excinfo.value)
    assert message.startswith('C ')
    assert np.array2string(bad['C']['A1'], precision=12) in message
    assert np.array2string(table, precision=12) in message
    assert str(dict(sorted(G2_COUNTS.items()))) in message


def test_radii_within_the_match_tolerance_are_the_row(ethylene):
    """The tolerance absorbs decimal round-tripping and nothing else: radii
    that agree to it ARE the row, nuclear point included."""
    radii = {el: atomic_grid(el, BASIS, AUXBASIS, G2_COUNTS)[0]
             for el in ('C', 'H')}
    nudged = {el: {shell: np.atleast_1d(r) + 0.5 * ISDF_RADII_MATCH_TOL
                   for shell, r in shells.items()}
              for el, shells in radii.items()}
    isdf_grid(ethylene, radii=nudged, counts=G2_COUNTS)


def test_explicit_radii_at_untabulated_counts_proceed(ethylene):
    """The positive control for the refusal above. No row describes these
    counts, so there is no second specification to contradict, and nothing is
    re-optimized, since the radii are given."""
    radii = {el: atomic_grid(el, BASIS, AUXBASIS, G2_COUNTS)[0]
             for el in ('C', 'H')}
    counts = untabulated_counts(('C', 'H'))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        explicit = isdf_grid(ethylene, radii=radii, counts=counts)
    assert not reoptimization_warnings(caught)
    alone = isdf_grid(ethylene, radii=radii)
    assert np.array_equal(explicit, alone)


def test_untabulated_counts_without_radii_name_what_the_table_holds(ethylene,
                                                                     monkeypatch):
    """Honouring the request means re-optimizing, and saying so: the miss is
    almost always a tuple nobody ever optimized, so the warning carries the
    element, the counts asked for and the sizes the table does hold.

    The optimizer is stubbed because it is minutes of multi-start descent and
    not what is under test; what is under test is that the request reaches it
    at the counts asked for, and that the caller is told.
    """
    counts = untabulated_counts(('C', 'H'))
    asked = []

    def stub(element, basis, auxbasis, counts=None, n_start=1, **kwargs):
        asked.append((element, dict(counts)))
        return atomic_grid(element, basis, auxbasis, G2_COUNTS)[0], 0.0

    monkeypatch.setattr(isdf_jk, 'optimize_atomic_radii', stub)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        isdf_grid(ethylene, counts=counts)
    assert sorted(el for el, _ in asked) == ['C', 'H']
    assert all(c == counts for _, c in asked)
    said = reoptimization_warnings(caught)
    assert said, 'a run-time re-optimization went unannounced'
    joined = ' '.join(str(w.message) for w in said)
    assert 'C' in joined and 'H' in joined
    assert str(sorted(counts.items())) in joined
    assert 'pts' in joined, 'the tabulated alternatives are missing'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
