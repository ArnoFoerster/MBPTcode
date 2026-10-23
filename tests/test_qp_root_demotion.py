"""A quasiparticle root with a pole strength outside (0, 1] is kept, warned
about and flagged; it is neither refused nor demoted (it
can happen, rarely, and is often harmless -- the one exception to refusing every shortfall).

The declaration `QPStates()` (kind='admitted', threshold='gap') names its
orbitals from the mean-field spectrum by Eq. (27) before anything is solved,
so it cannot know whether the root the solve converges on is a quasiparticle.
On ethylene/cc-pVDZ, Hartree-Fock, the sum-over-poles route at M = 24 on the
G3 grid and the 64-point contour grid, it admits orbitals 3..17. With the
coalesced poles of an unmerged fit, the Newton from eps_17 climbs the falling
flank of a negative-weight pole to 19.286 eV with Z = -0.012; the same model's
quasiparticle is at 17.653 eV with Z = 0.947, where the explicit-residue route
puts it (17.652 eV).

NO FIT PRODUCES THE DEFECT. Coalesced poles are merged (`SOP_POLE_MERGE_TOL`)
and the pole model's quasiparticle is the largest-weight root of the pole-sum
equation, so the production fit places orbital 17 at its quasiparticle with
nothing flagged (`test_merged_poles_place_no_spurious_root`). The other gates
plant the Newton root (19.286 eV, Z = -0.0122) for orbital 17 and check:

- orbital 17 stays explicit with the planted root as solved, flagged in
  `qp_z_outside` with its Z, root and route, and in `qp_bookkeeping`'s
  'z_outside'; the reference warns;
- a displaced geometry keeps the set and the flags, and warns on its own
  roots without refusing;
- under ranks every rank keeps rank 0's verdict.

The demotion this replaced moved orbital 17 to the scissor of probe 16; the
number that moves with the change is recorded in
"""
import functools
import os
import sys

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.numerics.constants import (CD_NFREQ, HARTREE_TO_EV,
                                         SCF_DIFFERENTIABLE_CONV_TOL,
                                         SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.Base.numerics.numerics import numerics_of
from src.SingleReference.GW import contour_deformation, sum_over_poles
from src.SingleReference.GW.qp_states import (is_quasiparticle_root,
                                              resolve_qp_states)
from src.gradients.gwbse.conventions import FrozenConventions
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.properties.excitations import (SurfaceSpec, driven_chain,
                                        qp_bookkeeping, surface_of)

#: Ethylene, in Bohr.
ETHYLENE = [('C', (0.0, 0.0, 1.2604473250848962)),
            ('C', (0.0, 0.0, -1.2604473250848962)),
            ('H', (0.0, 1.7442172129735523, 2.3394809422115466)),
            ('H', (0.0, -1.7442172129735523, 2.3394809422115466)),
            ('H', (0.0, 1.7442172129735523, -2.3394809422115466)),
            ('H', (0.0, -1.7442172129735523, -2.3394809422115466))]
#: The pole count at which the fit places the spurious root.
N_POLES = 24
#: The contour grid that fit is made on, asked for explicitly: the fixed
#: pole-model grid (`CD_NFREQ_SOP`) fits poles that place no spurious root.
NFREQ_CD = CD_NFREQ
#: The orbital whose planted root is no quasiparticle.
FLAGGED = 17
#: Orbital 17's quasiparticle on the explicit-residue route, in eV, and how
#: far the merged pole model may put it (the spurious root is 1.6 eV away).
QP_17_EV = 17.652
QP_17_TOL_EV = 0.01


def factory(mol):
    mf = scf.RHF(mol)
    mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.kernel()
    return mf


def ethylene_case():
    """(mol, surface, chain) of the admitted-set SOP S1 at M = 24, not yet
    evaluated."""
    mol = gto.M(atom=ETHYLENE, unit='Bohr', basis='cc-pvdz', verbose=0,
                max_memory=4000)
    spec = SurfaceSpec(GroundState('rpa', 'hf'), chi0='space-time',
                       residues='sop', factorization='isdf',
                       qp_states=QPStates(),
                       numerics=numerics_of({'grid_accuracy': 'G3', 'n_poles': N_POLES,
                                 'nfreq_cd': NFREQ_CD}))
    surface = surface_of(spec, Excitation('singlet', root=1), mol, factory)
    return mol, surface, driven_chain(surface)


#: The root the Newton from eps_17 found on the unmerged fit, and its Z.
SPURIOUS_ROOT_HA = 19.286 / HARTREE_TO_EV
SPURIOUS_Z = -0.0122


def plant_spurious_root(real):
    """`qp_energy_sop` returning the spurious root and Z for orbital 17 alone."""
    @functools.wraps(real)
    def planted(p, *args, **kwargs):
        out = real(p, *args, **kwargs)
        return (SPURIOUS_ROOT_HA, SPURIOUS_Z) if p == FLAGGED else out
    return planted


@pytest.fixture(scope='module')
def ethylene():
    """The case evaluated once at the reference geometry with the spurious
    root planted for orbital 17."""
    with pytest.MonkeyPatch.context() as patch:
        planted = plant_spurious_root(sum_over_poles.qp_energy_sop)
        patch.setattr(sum_over_poles, 'qp_energy_sop', planted)
        patch.setattr(contour_deformation, 'qp_energy_sop', planted)
        mol, surface, chain = ethylene_case()
        with pytest.warns(RuntimeWarning, match='outside \\(0, 1\\]'):
            chain.excitation(mol, chain.mf0)
        yield mol, surface, chain


def test_merged_poles_place_no_spurious_root():
    """The production fit, coalesced poles merged: orbital 17 stays explicit
    with its quasiparticle root, and nothing is flagged."""
    mol, _, chain = ethylene_case()
    chain.excitation(mol, chain.mf0)
    z = chain.qp_diagnostics['z']
    assert FLAGGED in chain.resolved.qp_set and not chain.resolved.qp_z_outside
    assert all(is_quasiparticle_root(v) for v in z.values()), z
    root = chain.resolved.qp_seeds[FLAGGED] * HARTREE_TO_EV
    assert abs(root - QP_17_EV) < QP_17_TOL_EV, root


def test_the_bounds_are_the_definition():
    assert is_quasiparticle_root(1.0) and is_quasiparticle_root(0.947)
    for z in (0.0, -0.0122, np.nextafter(1.0, 2.0), np.inf, np.nan):
        assert not is_quasiparticle_root(z), z


def test_the_flagged_root_is_kept_and_recorded(ethylene):
    mol, surface, chain = ethylene
    eps = np.asarray(chain.mf0.mo_energy, float)
    declared = resolve_qp_states(QPStates(), eps, mol.nelectron // 2,
                                 degeneracy_tol=chain.degeneracy_tol).explicit
    assert FLAGGED in declared, 'the case does not declare the orbital'
    assert sorted(chain.resolved.qp_set) == sorted(declared)
    z = chain.qp_diagnostics['z']
    assert z[FLAGGED] == SPURIOUS_Z
    assert all(is_quasiparticle_root(v) for p, v in z.items() if p != FLAGGED)
    entry = chain.resolved.qp_z_outside[FLAGGED]
    assert entry['z'] == SPURIOUS_Z and entry['route'] == 'sop', entry
    assert entry['root'] == SPURIOUS_ROOT_HA
    # the root is kept as solved: no scissor for the orbital
    assert chain.resolved.qp_seeds[FLAGGED] == SPURIOUS_ROOT_HA
    assert FLAGGED not in chain.resolved.outside_shift
    record = qp_bookkeeping(surface, mol)
    assert list(record['z_outside']) == [FLAGGED]
    assert record['z_outside'][FLAGGED]['z'] == SPURIOUS_Z
    assert FLAGGED in record['explicit'] and FLAGGED not in record['outside']


def test_a_displaced_geometry_keeps_the_set_and_warns(ethylene):
    mol, _, chain = ethylene
    settled, flags = list(chain.resolved.qp_set), dict(chain.resolved.qp_z_outside)
    crd = mol.atom_coords().copy()
    crd[0, 2] += 1e-2
    with pytest.warns(RuntimeWarning, match='displaced geometry'):
        chain.excitation(mol.set_geom_(crd, unit='Bohr', inplace=False))
    assert list(chain.resolved.qp_set) == settled
    assert chain.resolved.qp_z_outside == flags


def test_a_root_outside_the_bounds_after_settling_is_not_refused(ethylene):
    _, _, chain = ethylene
    states = np.asarray(chain.resolved.qp_set)
    z = np.full(len(states), 0.95)
    z[-2] = -0.5
    route_out = {'z': z, 'roots': {int(p): 0.0 for p in states},
                 'routes': {int(p): 'sop' for p in states}}
    flags = dict(chain.resolved.qp_z_outside)
    with pytest.warns(RuntimeWarning, match='displaced geometry'):
        chain.quasiparticles._flag_z_outside(route_out, states)
    # the frozen conventions are not written at a displaced geometry
    assert chain.resolved.qp_z_outside == flags


def bare_chain(states):
    """An unsettled chain with only what `_flag_z_outside` reads."""
    chain = object.__new__(ExcitedStateChain)
    chain.conventions = FrozenConventions(qp_set=np.asarray(states))
    return chain


def flag_on_rank(comm, states, z_by_rank):
    chain = bare_chain(states)
    z = np.asarray(z_by_rank[comm.Get_rank()], float)
    route_out = {'z': z, 'roots': {int(p): 0.0 for p in states},
                 'routes': {int(p): 'sop' for p in states}}
    chain.quasiparticles._flag_z_outside(route_out, states)
    return list(chain.resolved.qp_set), sorted(chain.resolved.qp_z_outside)


@pytest.mark.filterwarnings('ignore:quasiparticle roots with a pole strength')
@pytest.mark.parametrize('rank0_flags', (True, False))
def test_every_rank_keeps_rank_zeros_verdict(rank0_flags):
    """Rank 1's copy of Z disagrees with rank 0's on the last state; both
    ranks keep the set whole and the flags rank 0's copy decides."""
    states = [15, 16, 17]
    outside, inside = [0.96, 0.95, -0.01], [0.96, 0.95, 0.93]
    z_by_rank = [outside, inside] if rank0_flags else [inside, outside]
    out = run_simulated(flag_on_rank, 2, states, z_by_rank)
    assert out[0] == out[1], out
    assert out[0] == (states, [17] if rank0_flags else [])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
