"""Gates for the one analytic-continuation call the adaptive explicit set
selects with (`qp_selection.ac_quasiparticle_shifts`).

The conventional space-time GW continues Sigma_c off the imaginary axis; the
chain solves its explicit roots by contour deformation with residues on the
pole model. For the selection to read AC minus explicit as the AC's error
alone, the AC must screen with the chain's own factors -- the bare gauge in a
continuum -- and solve with the chain's own static term, Eq. (18) added to
the root after the solve as the chain adds it (`reaction_field=`).
The gates:

  * `xc_diagonal=` is the static build where the two coincide, and
    `with_z=True` changes no root: each a solve of its own, so the roots
    agree to a few ulp (`assert_same` of tests/pins.py), not bitwise;
  * on formaldehyde/cc-pVDZ (HF, G1 ISDF, admitted window), in the gas phase
    and in water, the AC frontier shifts sit within the 0.1 eV calibration
    gate of the chain's explicit ones, and every AC pole strength is a
    quasiparticle's.
"""
import warnings

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics import Numerics
from src.Base.constants import HARTREE_TO_EV
from src.Base.declaration import QPStates
from src.Base.separable_ri import resolve_isdf_grid
from src.Base.solvent_screening import SolventScreening
from src.SingleReference.GW.qp_selection import ac_quasiparticle_shifts
from src.SingleReference.GW.qp_solve import static_exchange_diagonal
from src.SingleReference.GW.qp_states import resolve_qp_states
from src.SingleReference.GW.space_time import solve_qp_energy_space_time
from src.gradients.excited_state import ExcitedStateChain
from tests.isdf_test_grid import g1, g1_numerics
from tests.pins import assert_same

ATOM = ('C 0 0 0; O 0 0 1.205; H 0 0.9429 -0.5876; H 0 -0.9429 -0.5876')
#: The frontier calibration gate the selection falls back to the window at.
GATE_EV = 0.1


def factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol, mf.conv_tol_grad = 1e-11, 1e-8
    mf.kernel()
    return mf


@pytest.fixture(scope='module')
def formaldehyde():
    warnings.simplefilter('ignore')
    mol = gto.M(atom=ATOM, basis='cc-pvdz', verbose=0)
    return mol, factory(mol)


def test_the_handed_static_term_is_the_built_one(formaldehyde):
    mol, mf = formaldehyde
    nocc = mol.nelectron // 2
    states = np.arange(nocc - 2, nocc + 2)
    built = solve_qp_energy_space_time(mf, mol, nocc, states, distribute=False, counts=g1(mol))
    xc = static_exchange_diagonal(mf, mol, states)
    handed = solve_qp_energy_space_time(mf, mol, nocc, states, xc_diagonal=xc,
                                        distribute=False, counts=g1(mol))
    assert_same(handed, built, quantity='continued', label='handed')
    roots, z = solve_qp_energy_space_time(mf, mol, nocc, states,
                                          xc_diagonal=xc, with_z=True,
                                          distribute=False, counts=g1(mol))
    assert_same(roots, built, quantity='continued', label='with_z')
    assert np.all((z > 0) & (z <= 1))


@pytest.mark.parametrize('medium', ['gas', 'water'])
def test_ac_frontier_shifts_sit_within_the_gate(formaldehyde, medium):
    mol, mf = formaldehyde
    nocc = mol.nelectron // 2
    eps = np.asarray(mf.mo_energy, float)
    elements = sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})
    counts, n_start = resolve_isdf_grid('G1', 'cc-pvdz', elements,
                                        auxbasis='cc-pvdz-ri')
    window = list(resolve_qp_states(QPStates('admitted'), eps, nocc,
                                    degeneracy_tol=1e-4).explicit)
    env = None if medium == 'gas' else SolventScreening(mol, solvent='water')
    chain = ExcitedStateChain(
        mol, factory, mf=mf, environment=env,
        numerics=g1_numerics(mol, Numerics(isdf={'counts': counts, 'n_start': n_start},
                                           qp={'window': window, 'residue_route': 'sop', 'scissor': 'calibrate'},
                                           bse={'solver': 'dense'})))
    shared = chain._shared_forward(mol, mf)
    states = np.asarray(chain.resolved.qp_set)
    explicit = {int(p): float(shared.eps_qp[p] - shared.eps[p])
                for p in states}
    # Inside the window the root carries Eq. (18) whole, added after its
    # solve; the AC is handed the same array to add to its own.
    d_sigma = shared.d if shared.d_bare is None else shared.d_bare
    a, z = ac_quasiparticle_shifts(mf, mol, nocc, states,
                                   factors=(shared.x_mo, d_sigma),
                                   xc_diagonal=shared.xc_correction,
                                   numerics=Numerics(),
                                   reaction_field=shared.shift)
    assert set(a) == set(explicit)
    for p in (nocc - 1, nocc):
        err = abs(a[p] - explicit[p]) * HARTREE_TO_EV
        assert err < GATE_EV, (medium, p, err)
    assert all(0.0 < z[p] <= 1.0 for p in (nocc - 1, nocc))
