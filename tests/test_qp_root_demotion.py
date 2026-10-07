"""A quasiparticle root with a pole strength outside (0, 1] never reaches the BSE.

The declaration `QPStates()` (kind='admitted', threshold='gap') names its
orbitals from the mean-field spectrum by Eq. (27) before anything is solved,
so it cannot know whether the root the solve converges on is a quasiparticle.
On ethylene/cc-pVDZ, Hartree-Fock, the sum-over-poles route at M = 24 on the
G3 grid and the 64-point contour grid, it admits orbitals 3..17. Orbital 17
sits at reach 0.9989, and six of its fitted poles are clipped onto the fit's
lower bound E_g = 14.742 eV, which puts a pole of weight -5.6e-5 Ha into the
LUMO term of its Sigma at eps_LUMO + E_g = 19.308 eV, 15.7 meV above
eps_17 = 19.293 eV. The Newton from eps_17 climbs the falling flank of that
negative-weight pole to 19.286 eV with Z = -0.012; the same model's
quasiparticle is at 17.653 eV with Z = 0.947, where the explicit-residue route
puts it (17.652 eV). Accepted, the spurious root moves S1 by +17.0 meV, T1 by
+23.6 meV and the forces by 3-4e-4 Ha/Bohr.

The surface settles the set on its first solve (`_settle_qp_set`): an
orbital whose root has Z outside (0, 1] leaves the explicit set, carries the
outside scissor of the nearest explicit probe, and is recorded with its
reason. Here:

- every explicitly solved root has 0 < Z <= 1, and orbital 17, which the
  declaration made explicit, is demoted with its rejected root and Z on file,
  in `qp_bookkeeping` and in the scissor tier of probe 16;
- a displaced geometry evaluates on the settled set and demotes nothing;
- after settling, a rejected root is refused rather than demoted;
- under ranks every rank keeps rank 0's decision.

That a rejected root is demoted on the one fixed grid is gated in
tests/test_cd_grid_sizing.py.
"""
import copy
import os
import sys

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import (CD_NFREQ, SCF_DIFFERENTIABLE_CONV_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.GW.qp_states import (is_quasiparticle_root,
                                              resolve_qp_states)
from src.gradients.excited_state import ExcitedStateChain
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
#: The orbital whose root was no quasiparticle, and the explicit probe whose
#: shift it takes once outside: the nearest explicit orbital in energy.
DEMOTED = 17
PROBE = 16


def factory(mol):
    mf = scf.RHF(mol)
    mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.kernel()
    return mf


@pytest.fixture(scope='module')
def ethylene():
    """(mol, surface, chain) of the admitted-set SOP S1 at M = 24, evaluated
    once at the reference geometry."""
    mol = gto.M(atom=ETHYLENE, unit='Bohr', basis='cc-pvdz', verbose=0,
                max_memory=4000)
    spec = SurfaceSpec(GroundState('rpa', 'hf'), chi0='space-time',
                       residues='sop', factorization='isdf',
                       qp_states=QPStates(),
                       numerics={'grid_accuracy': 'G3', 'n_poles': N_POLES,
                                 'nfreq_cd': NFREQ_CD})
    surface = surface_of(spec, Excitation('singlet', root=1), mol, factory)
    chain = driven_chain(surface)
    with pytest.warns(RuntimeWarning, match='outside \\(0, 1\\]'):
        chain.excitation(mol, chain.mf0)
    return mol, surface, chain


def test_the_bounds_are_the_definition():
    assert is_quasiparticle_root(1.0) and is_quasiparticle_root(0.947)
    for z in (0.0, -0.0122, np.nextafter(1.0, 2.0), np.inf, np.nan):
        assert not is_quasiparticle_root(z), z


def test_every_explicit_root_is_a_quasiparticle(ethylene):
    _, _, chain = ethylene
    z = chain.qp_diagnostics['z']
    assert sorted(z) == sorted(int(p) for p in chain.qp_set)
    bad = {p: v for p, v in z.items() if not is_quasiparticle_root(v)}
    assert not bad, f'roots with Z outside (0, 1] on the BSE diagonal: {bad}'


def test_the_declared_orbital_is_demoted_and_recorded(ethylene):
    mol, surface, chain = ethylene
    eps = np.asarray(chain.mf0.mo_energy, float)
    declared = resolve_qp_states(QPStates(), eps, mol.nelectron // 2,
                                 degeneracy_tol=chain.degeneracy_tol).explicit
    assert DEMOTED in declared, 'the case does not declare the orbital'
    assert DEMOTED not in chain.qp_set
    assert sorted(chain.qp_set) == [p for p in declared if p != DEMOTED]
    entry = chain.qp_demoted[DEMOTED]
    assert entry['z'] <= 0.0 and entry['route'] == 'sop', entry
    # a demoted orbital leaves nothing of its branch on the chain
    assert DEMOTED not in chain.qp_seeds and DEMOTED not in chain.sop_poles
    assert chain.outside_shift[DEMOTED] == chain.qp_seeds[PROBE] - eps[PROBE]
    record = qp_bookkeeping(surface, mol)
    assert list(record['demoted']) == [DEMOTED]
    assert record['demoted'][DEMOTED]['z'] == entry['z']
    assert DEMOTED in record['outside'] and DEMOTED not in record['explicit']
    tier = [t for t in record['scissor_tiers'] if DEMOTED in t['orbitals']]
    assert len(tier) == 1 and tier[0]['probe'] == PROBE, tier


def test_a_displaced_geometry_keeps_the_settled_set(ethylene):
    mol, _, chain = ethylene
    settled, demoted = list(chain.qp_set), dict(chain.qp_demoted)
    crd = mol.atom_coords().copy()
    crd[0, 2] += 1e-2
    chain.excitation(mol.set_geom_(crd, unit='Bohr', inplace=False))
    assert list(chain.qp_set) == settled and chain.qp_demoted == demoted
    assert all(is_quasiparticle_root(v)
               for v in chain.qp_diagnostics['z'].values())


def test_a_rejected_root_after_settling_is_refused(ethylene):
    _, _, chain = ethylene
    view = copy.copy(chain)
    view.qp_demoted = dict(chain.qp_demoted)
    states = np.asarray(chain.qp_set)
    z = np.full(len(states), 0.95)
    z[-1] = -0.5
    route_out = {'z': z, 'roots': {int(p): 0.0 for p in states},
                 'routes': {int(p): 'sop' for p in states}}
    with pytest.raises(RuntimeError, match='displaced geometry'):
        view._settle_qp_set(route_out, states)
    assert list(view.qp_set) == list(chain.qp_set)


def bare_chain(states):
    """An unsettled chain with only what `_settle_qp_set` reads."""
    chain = object.__new__(ExcitedStateChain)
    chain.qp_set = np.asarray(states)
    chain.qp_demoted = {}
    chain.qp_set_settled = False
    chain.outside = 'scissor'
    return chain


def settle_on_rank(comm, states, z_by_rank):
    chain = bare_chain(states)
    z = np.asarray(z_by_rank[comm.Get_rank()], float)
    route_out = {'z': z, 'roots': {int(p): 0.0 for p in states},
                 'routes': {int(p): 'sop' for p in states}}
    keep = chain._settle_qp_set(route_out, states)
    return (None if keep is None else keep.tolist(), list(chain.qp_set),
            sorted(chain.qp_demoted))


@pytest.mark.filterwarnings('ignore:quasiparticle roots with a pole strength')
@pytest.mark.parametrize('rank0_rejects', (True, False))
def test_every_rank_keeps_rank_zeros_set(rank0_rejects):
    """Rank 1's copy of Z disagrees with rank 0's on the last state; both
    ranks keep the set rank 0's copy decides."""
    states = [15, 16, 17]
    rejected, accepted = [0.96, 0.95, -0.01], [0.96, 0.95, 0.93]
    z_by_rank = ([rejected, accepted] if rank0_rejects
                 else [accepted, rejected])
    out = run_simulated(settle_on_rank, 2, states, z_by_rank)
    assert out[0] == out[1], out
    if rank0_rejects:
        assert out[0] == ([True, True, False], [15, 16], [17])
    else:
        assert out[0] == (None, states, [])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
