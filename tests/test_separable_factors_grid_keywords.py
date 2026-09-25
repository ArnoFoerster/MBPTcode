"""Every grid keyword of `separable_factors` reaches the grid, or the call is refused.

The interpolation grid is ONE object -- the shell counts, the radii on those
shells, and whether the nucleus itself is sampled. Each keyword names part of
it, and a keyword that is silently dropped does not make a run slower, it makes
it a DIFFERENT factorization: the caller asks for one grid, gets another, and
the two are indistinguishable downstream because the answer looks perfectly
reasonable. Every quasiparticle energy, BSE root and gradient built on it is
then of a functional nobody requested, and a convergence study over the grid
measures the wrong axis -- which is exactly what a counts sweep that returns
the same grid at every count reports as "converged".

THE REGRESSION IS THE SWEEP. It was shown to fail under one perturbation:
reinstating the branch that substituted the published Duchemin-Blase cc-pVTZ
grids for any cc-pVTZ caller before the table lookup. With that branch back in
`separable_factors`, all three count sets return the identical 1282-point
ethylene grid and `test_a_counts_sweep_moves_the_grid` fails on the first pair.

Each refusal is paired with the request it must NOT refuse, because a check
that fires on everything is not a check either.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import ISDF_GRID_ACCURACY, ISDF_RADII_MATCH_TOL
from src.Base.separable_ri import (PUBLISHED_COUNTS, _SHELL_ORDER, atomic_grid,
                                   shipped_radii_lookup)
from src.SingleReference.GW import space_time
from src.SingleReference.GW.space_time import DEFAULT_COUNTS, separable_factors

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
    """(mol, mf) once: `separable_factors` reads `mf.mo_coeff` and the attached
    environment, so one converged mean field serves every grid below."""
    mol = gto.M(atom=ETHYLENE, basis=BASIS, verbose=0)
    return mol, scf.RHF(mol).run()


@pytest.fixture(scope='module')
def carbon():
    """(mol, mf) for a single carbon atom, the only way to ask for the
    published carbon counts without dragging an element that has no row there.
    ROHF because the ground state is a triplet and the factors need one C."""
    mol = gto.M(atom='C 0 0 0', basis=BASIS, spin=2, verbose=0)
    return mol, scf.ROHF(mol).run()


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
    """Three tabulated sizes, three different grids -- the point of asking.

    Point count, positions and the fitted factors all have to move: two counts
    returning the same coords is the published-grid substitution reappearing,
    and it would make every grid convergence study at this basis flat.
    """
    mol, mf = ethylene
    grids = []
    for counts in SWEEP:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            X_mo, D, _, coords = separable_factors(mf, mol, counts=counts)
        assert not reoptimization_warnings(caught), (
            f'counts {counts} re-optimized instead of reading its row: '
            f'{[str(w.message) for w in reoptimization_warnings(caught)]}')
        grids.append((counts, coords, X_mo, D))
    sizes = [len(coords) for _, coords, _, _ in grids]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes), (
        f'point counts {sizes} for {[c for c, _, _, _ in grids]}')
    for i, (counts_i, coords_i, X_i, D_i) in enumerate(grids):
        for counts_j, coords_j, X_j, D_j in grids[i + 1:]:
            assert coords_i.shape != coords_j.shape or not np.array_equal(
                coords_i, coords_j), f'{counts_i} and {counts_j} share a grid'
            assert X_i.shape != X_j.shape or not np.array_equal(X_i, X_j)
            assert D_i.shape != D_j.shape or not np.array_equal(D_i, D_j)


def test_nothing_asked_for_is_the_default_counts_lookup(ethylene):
    """A caller with no opinion gets `DEFAULT_COUNTS`, and gets it through the
    same path as a caller who names them. This is the bitwise anchor for every
    existing consumer of the bare call."""
    mol, mf = ethylene
    bare = separable_factors(mf, mol)
    named = separable_factors(mf, mol, counts=DEFAULT_COUNTS)
    for a, b in zip(bare, named):
        assert np.array_equal(a, b)


def test_grid_accuracy_agreeing_with_counts_proceeds(ethylene):
    """Two spellings of ONE grid are not a contradiction. Bitwise, because an
    accuracy level resolves to counts and must then take exactly the path
    those counts take."""
    mol, mf = ethylene
    level = separable_factors(mf, mol, grid_accuracy='G2')
    both = separable_factors(mf, mol, grid_accuracy='G2', counts=G2_COUNTS)
    for a, b in zip(level, both):
        assert np.array_equal(a, b)


def test_grid_accuracy_contradicting_counts_is_refused(ethylene):
    """The dropped keyword this pairs with: G2's counts used to overwrite the
    caller's silently, so the returned factorization was at a grid the caller
    had explicitly asked against."""
    mol, mf = ethylene
    with pytest.raises(ValueError) as excinfo:
        separable_factors(mf, mol, grid_accuracy='G2', counts=G3_COUNTS)
    message = str(excinfo.value)
    assert str(dict(sorted(G3_COUNTS.items()))) in message
    assert str(dict(sorted(G2_COUNTS.items()))) in message


def test_explicit_radii_equal_to_the_row_are_the_row(ethylene):
    """Handing over the row's own radii with the row's own counts is one
    request, not two, and must reproduce the lookup path bitwise."""
    mol, mf = ethylene
    radii = {el: atomic_grid(el, BASIS, AUXBASIS, G2_COUNTS)[0]
             for el in ('C', 'H')}
    lookup = separable_factors(mf, mol, counts=G2_COUNTS)
    explicit = separable_factors(mf, mol, radii=radii, counts=G2_COUNTS)
    for a, b in zip(lookup, explicit):
        assert np.array_equal(a, b)


def test_a_matching_row_brings_its_nuclear_point(carbon):
    """`origin` belongs to the GRID, not to the recipe that found it.

    The published carbon row places one point at the nucleus, so honouring its
    radii while dropping the flag builds 306 points where the row describes
    307 -- a different grid, reported under the published grid's name.
    """
    mol, mf = carbon
    radii, origin = atomic_grid('C', BASIS, AUXBASIS, CARBON_PUBLISHED_COUNTS)
    assert origin, 'the published carbon row is the case this test exists for'
    lookup = separable_factors(mf, mol, counts=CARBON_PUBLISHED_COUNTS)
    explicit = separable_factors(mf, mol, radii={'C': radii},
                                 counts=CARBON_PUBLISHED_COUNTS)
    for a, b in zip(lookup, explicit):
        assert np.array_equal(a, b)
    # The same radii where no row claims them: the nuclear point is the caller's
    # to add, so the grid is one point smaller.
    no_row = separable_factors(mf, mol, radii={'C': radii},
                               counts=untabulated_counts(('C',)))
    assert len(explicit[3]) == len(no_row[3]) + 1


def test_radii_contradicting_the_row_are_refused(ethylene):
    """A perturbation of 1e-3 Bohr is 1e7 match tolerances and a different
    grid; resolving to either side would hand back a factorization one of the
    two keywords asked against."""
    mol, mf = ethylene
    radii = {el: atomic_grid(el, BASIS, AUXBASIS, G2_COUNTS)[0]
             for el in ('C', 'H')}
    table = np.atleast_1d(radii['C']['A1']).copy()
    bad = {el: dict(r) for el, r in radii.items()}
    bad['C']['A1'] = table + np.eye(len(table))[0] * 1e-3
    with pytest.raises(ValueError) as excinfo:
        separable_factors(mf, mol, radii=bad, counts=G2_COUNTS)
    message = str(excinfo.value)
    assert message.startswith('C ')
    assert np.array2string(bad['C']['A1'], precision=12) in message
    assert np.array2string(table, precision=12) in message
    assert str(dict(sorted(G2_COUNTS.items()))) in message


def test_radii_within_the_match_tolerance_are_the_row(ethylene):
    """The tolerance absorbs decimal round-tripping and nothing else: radii
    that agree to it ARE the row, nuclear point included."""
    mol, mf = ethylene
    radii = {el: atomic_grid(el, BASIS, AUXBASIS, G2_COUNTS)[0]
             for el in ('C', 'H')}
    nudged = {el: {shell: np.atleast_1d(r) + 0.5 * ISDF_RADII_MATCH_TOL
                   for shell, r in shells.items()}
              for el, shells in radii.items()}
    separable_factors(mf, mol, radii=nudged, counts=G2_COUNTS)


def test_explicit_radii_at_untabulated_counts_proceed(ethylene):
    """The positive control for the refusal above. No row describes these
    counts, so there is no second specification to contradict and the radii
    stand on their own -- and nothing is re-optimized, since the radii are
    given."""
    mol, mf = ethylene
    radii = {el: atomic_grid(el, BASIS, AUXBASIS, G2_COUNTS)[0]
             for el in ('C', 'H')}
    counts = untabulated_counts(('C', 'H'))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        explicit = separable_factors(mf, mol, radii=radii, counts=counts)
    assert not reoptimization_warnings(caught)
    alone = separable_factors(mf, mol, radii=radii)
    for a, b in zip(explicit, alone):
        assert np.array_equal(a, b)


def test_untabulated_counts_without_radii_name_what_the_table_holds(ethylene,
                                                                    monkeypatch):
    """Honouring the request means re-optimizing, and saying so: the miss is
    almost always a tuple nobody ever optimized, so the warning carries the
    element, the counts asked for and the sizes the table does hold.

    The optimizer is stubbed because it is minutes of multi-start descent and
    not what is under test; what is under test is that the request reaches it
    at the counts asked for, and that the caller is told.
    """
    mol, mf = ethylene
    counts = untabulated_counts(('C', 'H'))
    asked = []

    def stub(element, basis, auxbasis, counts=None, n_start=1, **kwargs):
        asked.append((element, dict(counts)))
        return atomic_grid(element, basis, auxbasis, G2_COUNTS)[0], 0.0

    monkeypatch.setattr(space_time, 'optimize_atomic_radii', stub)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        separable_factors(mf, mol, counts=counts)
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
