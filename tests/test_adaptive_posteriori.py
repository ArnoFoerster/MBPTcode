"""Gates for the adaptive explicit set's check at the end of a walk
(`properties.excitations.adaptive_check`, `ExcitedStateChain.
posteriori_check`).

The partition is frozen at the reference geometry; at the walk's last
geometry R* one analytic-continuation GW, calibrated on R*'s own explicit
roots, and R*'s exact weights give the followed state's first-order budget
with the frozen map. It records and warns; it never reselects.

Formaldehyde/cc-pVDZ (HF, G1 ISDF, SOP, admitted candidates, 1 meV): at a
geometry 0.01 Bohr off R0 the budget stays inside the tolerance and nothing
warns; the continuation at R* moved on the hole the followed state
weighs most, by three tolerances' worth of first-order error, makes the
check warn and say so, and the frozen partition and
the surface's energy at R* are untouched by either.
"""
import warnings

import numpy as np
import pytest
from pyscf import gto, scf

import src.gradients.excited_state as excited_state
from src.Base.constants import ADAPTIVE_QP_TOL_MEV, HARTREE_TO_MEV
from src.Base.declaration import QPStates
from src.Base.separable_ri import resolve_isdf_grid
from src.SingleReference.GW.qp_selection import first_order_weights
from src.SingleReference.GW.qp_states import resolve_qp_states
from src.gradients.excited_state import ExcitedStateChain
from src.properties.excitations import adaptive_check

ATOM = ('C 0 0 0; O 0 0 1.205; H 0 0.9429 -0.5876; H 0 -0.9429 -0.5876')
#: the first-order error, in units of the tolerance, that an AC error put
#: on the most heavily weighted hole at R* stands for
HOLE_KICK_TOLS = 3.0


def factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    return mf


@pytest.fixture(scope='module')
def walked():
    """(the adaptive chain frozen at R0, R*), the chain evaluated at R*."""
    warnings.simplefilter('ignore')
    mol = gto.M(atom=ATOM, basis='cc-pvdz', verbose=0)
    mf = factory(mol)
    eps = np.asarray(mf.mo_energy, float)
    elements = sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})
    counts, n_start = resolve_isdf_grid('G1', 'cc-pvdz', elements,
                                        auxbasis='cc-pvdz-ri')
    window = list(resolve_qp_states(QPStates('adaptive'), eps,
                                    mol.nelectron // 2,
                                    degeneracy_tol=1e-4).explicit)
    chain = ExcitedStateChain(
        mol, factory, qp_window=window, residue_route='sop',
        scissor='calibrate', outside='scissor', solver='dense',
        counts=counts, n_start=n_start, mf=mf,
        qp_select=QPStates('adaptive', targets=(('singlet', 0),
                                                ('triplet', 0))))
    chain.energy(mol, mf)
    end = mol.copy()
    end.set_geom_(mol.atom_coords() + 0.01 * np.array(
        [[0.3, -0.2, 0.1], [0.0, 0.4, -0.3], [0.2, 0.1, 0.0],
         [-0.1, 0.0, 0.2]]), unit='Bohr')
    return chain, end


def test_the_check_records_and_stays_quiet_inside_the_budget(walked):
    chain, end = walked
    part = chain.qp_partition
    e_before = chain.energy(end)[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        check = adaptive_check(chain, end)
    assert not any('over its' in str(w.message) for w in caught)
    assert not check['exceeds_tol']
    (row,) = check['targets']
    assert (row['spin'], row['root']) == ('singlet', 0)
    assert 0.0 < row['budget_meV'] <= ADAPTIVE_QP_TOL_MEV
    assert check['calibration']['passed']
    assert set(check['ac_shift_eV']) == set(part.candidates)
    assert check['orbital_map_changed'] == {}
    assert chain.qp_partition is part
    assert chain.energy(end)[0] == e_before


def test_posteriori_check_warns(walked, monkeypatch):
    chain, end = walked
    part = chain.qp_partition
    om, pieces = chain._forward(end, chain.mean_field(end)[1])
    n = first_order_weights(pieces[10][:, 0], pieces[11][:, 0], chain.nocc,
                            len(chain.mf0.mo_energy))
    hole = max(part.holes, key=lambda p: abs(n[p]))
    kick = HOLE_KICK_TOLS * ADAPTIVE_QP_TOL_MEV / HARTREE_TO_MEV / abs(n[hole])
    real = excited_state.ac_quasiparticle_shifts

    def kicked(*args, **kwargs):
        a, z = real(*args, **kwargs)
        a[hole] += kick
        return a, z
    monkeypatch.setattr(excited_state, 'ac_quasiparticle_shifts', kicked)
    e_before = chain.energy(end)[0]
    with pytest.warns(RuntimeWarning, match='over its'):
        check = adaptive_check(chain, end)
    assert check['exceeds_tol']
    (row,) = check['targets']
    assert row['budget_meV'] > ADAPTIVE_QP_TOL_MEV
    assert row['largest_terms_meV'][0][0] == [hole]
    assert chain.qp_partition is part
    assert chain.energy(end)[0] == e_before
