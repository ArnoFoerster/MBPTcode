"""The fragment-diabatic A_eff and its gradient with the quasiparticle
Hamiltonian (`qp.hamiltonian`): a group of near-degenerate orbitals enters
the BSE as its H, so A_eff does not depend on how the group's orbitals rotate
into each other, and the gradient differentiates H.

An offset dimer of two DIFFERENT ethylenes (one C=C stretched to 1.38 A, 3.8 A
apart), cc-pVDZ, RHF, BSE@G0W0 on the ISDF/space-time chain with SOP residues,
singlet, one site state per monomer. The two pi orbitals (14, 15) form a group:
their Kohn-Sham energies are 0.80 eV apart, their own corrections differ by
92 meV, and H couples them by 17 meV (the monomers are not equivalent, so the
coupling does not cancel by symmetry).

WHAT EACH GATE IS FOR:

- INVARIANCE. With the pair made exactly degenerate (the second pi energy set
  to the first, an artificial but consistent mean field), every rotation of
  the pair is an equally canonical basis. A_eff in a basis mixed by 0.3 is
  A_eff unrotated to solver precision. On the diagonal route the pair keeps
  its own corrections, which follow the rotation (A_eff moves by about
  20 meV), so that route is refused.
- the root gradient through the partition equals the chain's own root
  gradient, which differentiates the same H: the pairs' adjoint on H's
  off-diagonal elements reaches the chain's reverse pass.
- each element's gradient along a random direction equals the relocalized
  finite difference, Tamm-Dancoff and full BSE.
"""
import warnings

import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics.constants import HARTREE_TO_MEV
from src.Base.numerics.numerics import numerics_of
from src.fragments.localization import FragmentOrbitals
from src.fragments.partition import FragmentPartition
from src.gradients.fragment_diabatic import DiabaticGradient
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.fragment_diabatic import BSEFragmentMethod
from tests.fragment_diabatic_fd import diabatic_derivative
from tests.isdf_test_grid import g1
from tests.test_fragment_diabatic import (AUX, BASIS, ETHYLENE, FRAGMENTS,
                                          SITES, factory, random_direction)

#: Angstrom: monomer B's half C=C length (ethylene's is 0.667) and its offset
HALF_CC_B = 0.69
OFFSET_B = (3.8, 0.3, 0.2)
#: the pi orbitals, one group of the quasiparticle Hamiltonian
PAIR = (14, 15)
#: Omega_0 below Q's lowest eigenvalue by the pole margin, both kernels (Q's
#: lowest is 0.3004 Ha Tamm-Dancoff and 0.3052 Ha full with one site state
#: each)
OMEGA0 = 0.28
#: two site states per monomer, for the root through the partition
TWO_PER_SITE = {0: 2, 1: 2}
#: the pair's mixing, sin^2 of the rotation angle
MIXING = 0.3
#: meV: how far A_eff may move under the rotation
INVARIANT_MEV = 1e-4


def stretched(atoms, half_cc):
    """Ethylene with its C=C set to 2 half_cc, the hydrogens moved along."""
    out = []
    for s, (x, y, z) in atoms:
        z = (z * half_cc / 0.667 if s == 'C'
             else z + (half_cc - 0.667) * np.sign(z))
        out.append((s, (x, y, z)))
    return out


def dimer():
    dx, dy, dz = OFFSET_B
    b = [(s, (x + dx, y + dy, z + dz))
         for s, (x, y, z) in stretched(ETHYLENE, HALF_CC_B)]
    return gto.M(atom=ETHYLENE + b, basis=BASIS, verbose=0)


def make_chain(mol, tda=True, hamiltonian=True, mf=None):
    warnings.simplefilter('ignore')
    return ExcitedStateChain(
        mol, factory, bse_tda=tda, mf=mf,
        numerics=numerics_of({'auxbasis': AUX, 'counts': g1(mol),
                              'residue_route': 'sop',
                              'qp_hamiltonian': hamiltonian}))


@pytest.fixture(scope='module')
def mol():
    return dimer()


@pytest.fixture(scope='module')
def chain(mol):
    return make_chain(mol)


@pytest.fixture(scope='module')
def full_chain(mol):
    return make_chain(mol, tda=False)


def a_eff(chain, mf):
    op = BSEFragmentMethod(chain).singles(chain.mol0, mf)[0]
    orbitals = FragmentOrbitals.from_mf(mf, FRAGMENTS)
    return FragmentPartition.build(op, orbitals, SITES, None,
                                   omega0=OMEGA0).a_eff


def degenerate(mf):
    """`mf` with the pair's second energy set to its first: every rotation of
    the pair is then as canonical as the unrotated one."""
    out = mf.copy()
    eps = np.array(mf.mo_energy, float)
    eps[PAIR[1]] = eps[PAIR[0]]
    out.mo_energy = eps
    return out


def rotated(mf, theta):
    out = mf.copy()
    c = np.array(mf.mo_coeff, float)
    p, q = PAIR
    cp, cq = c[:, p].copy(), c[:, q].copy()
    c[:, p] = np.cos(theta) * cp + np.sin(theta) * cq
    c[:, q] = -np.sin(theta) * cp + np.cos(theta) * cq
    out.mo_coeff = c
    return out


def test_the_pair_is_a_group_with_a_real_coupling(chain, mol):
    shared = chain.shared_forward(mol, chain.mf0)
    groups = [g for g in shared.qp_h.groups if tuple(g.orbitals) == PAIR]
    assert len(groups) == 1
    h = groups[0].hamiltonian
    own = np.diag(h) - shared.eps[list(PAIR)]
    assert abs(own[1] - own[0]) * HARTREE_TO_MEV > 50.0
    assert abs(h[0, 1]) * HARTREE_TO_MEV > 10.0


def test_a_eff_does_not_depend_on_the_basis_of_a_degenerate_group(mol, chain):
    mf = degenerate(chain.mf0)
    turned = rotated(mf, np.arcsin(np.sqrt(MIXING)))
    # one chain, frozen in the unrotated basis: the rotated orbitals are
    # matched to the group by its span and carry the same conventions
    frozen = make_chain(mol, mf=mf)
    reference = a_eff(frozen, mf)
    moved = np.abs(a_eff(frozen, turned) - reference).max() * HARTREE_TO_MEV
    assert moved < INVARIANT_MEV, moved


def test_a_degenerate_pair_solved_diagonally_is_refused(mol, chain):
    # the same pair on separate corrections: A_eff would follow the rotation
    frozen = make_chain(mol, hamiltonian=False, mf=degenerate(chain.mf0))
    with pytest.raises(ValueError, match='qp_hamiltonian') as refused:
        a_eff(frozen, frozen.mf0)
    assert str(list(PAIR)) in str(refused.value)


def test_root_gradient_is_the_chains_root_gradient(chain):
    ref, info = chain.excitation_gradient()
    omega = info['omega']
    grad = DiabaticGradient(BSEFragmentMethod(chain), FRAGMENTS, TWO_PER_SITE,
                            omega0=omega)
    g, d = grad.root_gradient(omega)
    assert abs(d['omega'] - omega) < 1e-9
    assert np.abs(g - ref).max() < 1e-8 * np.abs(ref).max()
    assert np.abs(d['canonical']).max() < 1e-10
    assert np.abs(d['localization']).max() < 1e-10


@pytest.mark.parametrize('tda', [True, False])
def test_element_gradients_match_finite_differences(chain, full_chain, tda):
    ch = chain if tda else full_chain
    direction = random_direction(ch.mol0.natm, 23)
    grad = DiabaticGradient(BSEFragmentMethod(ch), FRAGMENTS, SITES,
                            omega0=OMEGA0)
    fd = [diabatic_derivative(BSEFragmentMethod(ch), FRAGMENTS, SITES,
                              direction, omega0=OMEGA0, step=h)[0]['a_eff']
          for h in (1e-3, 5e-4)]
    fd = (4.0 * fd[1] - fd[0]) / 3.0
    scale = np.abs(fd).max()
    for a, b in ((0, 0), (1, 1), (0, 1)):
        an = float((grad.element(a, b)[0] * direction).sum())
        assert abs(an - fd[a, b]) < 2e-5 * scale, (a, b, an, fd[a, b])
