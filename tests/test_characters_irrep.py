"""Gates for the point-group irrep selector in `src/properties/characters.py`.

If `orbital_irreps`/`root_irreps`/`select`/`impure` disagree with the labels a
direct product of the point group predicts, a caller selecting a root by
symmetry can silently report one state's energy under another state's label --
QUEST reports the lowest root OF A GIVEN SYMMETRY, not the lowest root outright.

The gate is not vacuous: on water/STO-3G/RHF with the four lowest TDA
singlets, replacing the direct-product XOR in `root_irreps`
(`ids[:nocc, None] ^ ids[None, nocc:]`) with a bitwise AND was checked once to
turn the correct labels ['B1', 'A2', 'A1', 'B2'] into ['A1', 'B1', 'A1', 'A1'],
drop root 2's purity from 1.0 to 0.9448, and move `select(..., 'A1')`'s answer
from root 2 to root 0 -- a different state reported under the right-looking
label. The tests below would fail on that perturbation through either the
label comparison, the purity comparison, or the selected index.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf, tdscf

# src/properties/__init__.py imports src.properties.nonadiabatic, which
# imports src.gradients.state_manifold; src.gradients's own __init__ imports
# excited_state, which imports back from src.properties.nonadiabatic -- a
# circular import that only resolves if src.gradients is already fully
# loaded first. this is not a cycle this file introduces or is scoped to fix.
import src.gradients  # noqa: F401,E402

from src.properties import characters                              # noqa: E402


def _water_sto3g_tda():
    """Water/STO-3G/RHF, the four lowest TDA singlets: the smallest case with
    a nontrivial (C2v) point group and more than one irrep among the roots.
    """
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
                basis='sto-3g', symmetry=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    nocc = mol.nelectron // 2
    td = tdscf.TDA(mf)
    td.singlet = True
    td.nstates = 4
    td.kernel()
    nvirt = mf.mo_coeff.shape[1] - nocc
    X = np.stack([np.asarray(x).reshape(nocc, nvirt).ravel()
                 for x, y in td.xy], axis=1)
    Y = np.zeros_like(X)
    return mol, mf.mo_coeff, nocc, np.asarray(td.e), X, Y


def _labels_equal(a, b):
    """True iff two `root_irreps` outputs agree bitwise, field by field."""
    return len(a) == len(b) and all(
        ra['irrep'] == rb['irrep'] and ra['occ'] == rb['occ']
        and ra['virt'] == rb['virt'] and ra['occ_irrep'] == rb['occ_irrep']
        and ra['virt_irrep'] == rb['virt_irrep']
        and ra['purity'] == rb['purity'] and ra['weight'] == rb['weight']
        for ra, rb in zip(a, b))


def test_moved_functions_match_recorded_baseline():
    """orbital_irreps/root_irreps/select/impure via `characters` reproduce,
    bitwise, the values recorded from the pre-move `state_symmetry.py`."""
    baseline_path = os.path.join(os.environ['TMPDIR'], 'irrep_before.npz')
    if not os.path.exists(baseline_path):
        pytest.skip(f'no recorded baseline at {baseline_path}')
    baseline = np.load(baseline_path, allow_pickle=True)

    mol, mo_coeff, nocc, omega, X, Y = _water_sto3g_tda()
    assert np.array_equal(mo_coeff, baseline['mo_coeff'])
    assert np.array_equal(X, baseline['X'])
    assert np.array_equal(Y, baseline['Y'])
    assert np.array_equal(omega, baseline['omega'])

    ids, names = characters.orbital_irreps(mol, mo_coeff)
    assert np.array_equal(ids, baseline['ids'])
    assert names == list(baseline['names'])

    labels = characters.root_irreps(mol, mo_coeff, nocc, X, Y)
    assert [r['irrep'] for r in labels] == list(baseline['irreps'])
    assert np.array_equal([r['purity'] for r in labels], baseline['purity'])
    assert np.array_equal([r['occ'] for r in labels], baseline['occ'])
    assert np.array_equal([r['virt'] for r in labels], baseline['virt'])
    assert [r['occ_irrep'] for r in labels] == list(baseline['occ_irrep'])
    assert [r['virt_irrep'] for r in labels] == list(baseline['virt_irrep'])
    assert np.array_equal([r['weight'] for r in labels], baseline['weight'])

    n, _ = characters.select(mol, mo_coeff, nocc, omega, X, Y, 'A1')
    assert n == int(baseline['n_a1'])

    imp = characters.impure(labels)
    assert [i for i, _ in imp] == list(baseline['impure_idx'])
    assert characters.PURITY_FLOOR == float(baseline['purity_floor'])


def test_select_by_irrep_picks_the_expected_root():
    """The lowest-of-symmetry root selected by irrep on water/STO-3G/RHF TDA
    is root 2 (A1, occ 3 -> virt 5), not root 0 (the lowest root outright)."""
    mol, mo_coeff, nocc, omega, X, Y = _water_sto3g_tda()
    n, labels = characters.select(mol, mo_coeff, nocc, omega, X, Y, 'A1')
    assert n == 2
    assert labels[n]['irrep'] == 'A1'
    assert labels[n]['occ'] == 3 and labels[n]['virt'] == 5
    assert n != int(np.argsort(np.asarray(omega))[0])


def test_select_missing_irrep_returns_none_not_raise():
    """A target irrep absent among the roots computed: `select` returns
    (None, labels) rather than raising. A caller that needs the available
    irreps named on a miss has to read them off `labels` itself.
    """
    mol, mo_coeff, nocc, omega, X, Y = _water_sto3g_tda()
    n, labels = characters.select(mol, mo_coeff, nocc, omega, X, Y, 'A1')
    assert n is not None

    n_missing, labels_missing = characters.select(
        mol, mo_coeff, nocc, omega, X, Y, 'NOSUCHIRREP')
    assert n_missing is None
    assert _labels_equal(labels_missing, labels)
    available = sorted({r['irrep'] for r in labels_missing})
    assert available == ['A1', 'A2', 'B1', 'B2']
