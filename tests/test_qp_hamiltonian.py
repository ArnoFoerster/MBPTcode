"""The quasiparticle Hamiltonian in frozen subspaces of the explicit set
(`ExcitedStateChain(qp_hamiltonian=True)`).

Diagonal G0W0 gives each orbital index its own Sigma_pp, so two orbitals that
rotate into each other across an avoided crossing trade their corrections
within the crossing's width and the BSE surface steps there (DMABN/cc-pVDZ S1:
LUMO and LUMO+1, 0.43 eV apart at R0, cross on the walk; corrections 0.11 eV
apart). Inside a subspace P frozen with the set, the option replaces them by
the subspace's quasiparticle Hamiltonian at self-consistency,

    H = e0 + 1/2 sum_x [(H - d_x)^-1 A_x + A_x (H - d_x)^-1],

e0 = F + Sigma_x - v_xc on P and A_x the pole model of every element of
Sigma^c on P's common poles, so that in H's eigenbasis element kl is
[Sigma^c_kl(e_k) + Sigma^c_kl(e_l)] / 2; put on the BSE as A_ia,jb +=
delta_ij H_ab - delta_ab H_ij: invariant to rotations inside P, so smooth
through a crossing (`test_qp_hamiltonian_rotation_invariance`).

Pinned here on formaldehyde/cc-pVDZ with C 0.1 Bohr out of plane, SOP on
orbitals 4-17. Orbitals 12 and 13 (4.2e-3 Ha apart, different irreps at C2v,
where their coupling vanishes) form the one subspace on Hartree-Fock; on PBE0
the pole model admits 4-10 alone and the test widens the window to join
4-6 and 9/10:
  - with no subspace of two the route is the diagonal one, energy and force
    to a few ulp (`assert_same` of tests/pins.py: two chains on two SCFs,
    so not bitwise); with the off-diagonal contractions zeroed H is diagonal and the
    energy is the diagonal route's but for the subspace's common poles;
  - the dense and the matrix-free Casida solves agree with the coupling on;
  - the force is the derivative of the energy, on Hartree-Fock (Sigma^c_pq
    alone) and on PBE0 (the static <p|Sigma_x - v_xc|q> as well);
  - in toluene (PCM) H carries <p|Sigma^env|q>, the first-order partner of
    Eq. (18) (the mean slope of Eq. (18) under p -> q and q -> p rotations
    is 2 Sigma^env_pq), and the force is still the derivative of the energy.
"""
import warnings

import numpy as np
import pytest
from pyscf import dft, gto, lib, scf

import src.gradients.adaptive_selection as adaptive_selection_module
import src.gradients.qp_set as qp_set_module
from src.Base.chain_settings import with_setting
from src.Base.numerics import Numerics
from src.Base.constants import (SCF_DIFFERENTIABLE_CONV_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.solvent_screening import SolventScreening
from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.GW.reaction_field import (offdiagonal_reaction_field,
                                                   pair_densities,
                                                   projected_quasiparticle_shift)
from src.gradients.excited_state import ExcitedStateChain
from tests.isdf_test_grid import g1_numerics
from tests.pins import assert_same

FORMALDEHYDE = 'O 0 0 1.205; C 0.053 0 0; H 0 0.943 -0.587; H 0 -0.943 -0.587'
BASIS = 'cc-pvdz'
WINDOW = list(range(4, 18))
#: the orbitals the pole model admits on PBE0, and a window joining some
PBE0_WINDOW = list(range(4, 11))
PBE0_SUBSPACE_WINDOW = 0.1
PBE0_SUBSPACES = ((4, 5, 6), (9, 10))
#: Bohr, and the force/energy agreement asked of the coupled surface
FD_STEP = 1e-4
FD_TOL = 1e-7
#: Ha: what the matrix-free roots are converged to against the dense ones
DAVIDSON_TOL = 1e-7
#: (atom, axis): C out of plane, an H in plane, O along the CO axis
COORDINATES = ((1, 0), (2, 1), (0, 2))
PAIR = (12, 13)
#: Ha: a coupling the subspaces carry at this geometry, at least
COUPLED = 1e-5
#: rad, and the relative agreement of the rotated Eq. (18)'s mean slope with
#: twice the off-diagonal element (central difference, O(theta^2))
ROTATION = 1e-3
ROTATION_TOL = 1e-4
#: simulated ranks, and how far the sliced layout may sit from the whole one
#: (summation order of the adjoints over the partitions)
RANKS = 2
LAYOUT_TOL = 1e-9
#: Ha: a subspace window no two orbitals of the set are closer than
NO_SUBSPACE = 1e-6
#: Ha: what one set of poles for a subspace, instead of one per orbital,
#: may move an uncoupled subspace's excitation energy by
COMMON_POLES = 1e-5


def rhf(mol):
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-jkfit')
    return converged(mf)


def pbe0(mol):
    mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis=BASIS + '-jkfit')
    return converged(mf)


def converged(mf):
    mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    return mf


def chain(factory=rhf, qp_hamiltonian=False, solver='dense',
          qp_window=WINDOW, numerics=None, qp_partition=None):
    warnings.simplefilter('ignore')
    mol = gto.M(atom=FORMALDEHYDE, basis=BASIS, verbose=0)
    numerics = Numerics() if numerics is None else numerics
    for field, value in (('qp.window', qp_window), ('qp.residue_route', 'sop'),
                         ('qp.hamiltonian', qp_hamiltonian),
                         ('bse.solver', solver)):
        numerics = with_setting(numerics, field, value)
    return ExcitedStateChain(
        mol, factory, mf=factory(mol), numerics=g1_numerics(mol, numerics),
        qp_partition=qp_partition)


def displaced(mol, atom, axis, step):
    out = mol.copy()
    coords = out.atom_coords().copy()
    coords[atom, axis] += step
    out.set_geom_(coords, unit='Bohr')
    out.build(False, False)
    return out


def coupling(ch):
    """The forward's QPHamiltonian at the reference geometry."""
    return ch._shared_forward(ch.mol0, ch.mf0).qp_h


def test_subspaces_are_frozen_with_the_set():
    ch = chain(qp_hamiltonian=True)
    ch.excitation()
    subspaces = ch.resolved.qp_subspaces[1]
    assert subspaces == (PAIR,)
    nocc = ch.nocc
    for b in subspaces:
        assert len(b) > 1
        assert all(p < nocc for p in b) or all(p >= nocc for p in b)
    h = coupling(ch)
    assert np.allclose(h.h, h.h.T, atol=0, rtol=0)
    assert not np.diag(h.h).any()
    assert np.abs(h.h).max() > COUPLED
    ch.excitation(displaced(ch.mol0, 1, 0, 0.05))
    assert ch.resolved.qp_subspaces[1] == subspaces


def test_without_a_subspace_the_route_is_the_diagonal_one(monkeypatch):
    diagonal = chain()
    om, grad = diagonal.excitation(), diagonal.excitation_gradient()[0]
    monkeypatch.setattr(qp_set_module, 'QP_HAMILTONIAN_WINDOW', NO_SUBSPACE)
    monkeypatch.setattr(adaptive_selection_module, 'QP_HAMILTONIAN_WINDOW',
                        NO_SUBSPACE)
    single = chain(qp_hamiltonian=True)
    # each chain converges its own SCF: the same numbers to a few ulp, never
    # bitwise across two SCF runs
    assert_same(single.excitation(), om, quantity='energy', label='excitation')
    assert single.resolved.qp_subspaces[1] == ()
    assert_same(single.excitation_gradient()[0], grad, quantity='gradient',
                label='excitation gradient')


def test_a_zero_coupling_is_the_diagonal_energy(monkeypatch):
    om = chain().excitation()
    monkeypatch.setattr(
        qp_set_module, 'pair_contractions',
        lambda tape, grid, pairs, tile_gb=None: [np.zeros_like(tape.wcs[0])
                                                 for _ in pairs])
    zeroed = chain(qp_hamiltonian=True)
    assert not coupling(zeroed).h.any()
    assert abs(zeroed.excitation() - om) < COMMON_POLES


def test_dense_and_matrix_free_casida_agree():
    dense = chain(qp_hamiltonian=True)
    free = chain(qp_hamiltonian=True, solver='davidson',
                 numerics=Numerics(bse={'nroots': 4, 'conv_tol': 1e-9}))
    assert abs(dense.excitation() - free.excitation()) < DAVIDSON_TOL
    assert abs(dense.excitation() - chain().excitation()) > 1e-5


@pytest.mark.parametrize('factory', [rhf, pbe0], ids=['hf', 'pbe0'])
def test_the_force_is_the_derivative_of_the_coupled_energy(factory,
                                                          monkeypatch):
    if factory is pbe0:
        monkeypatch.setattr(qp_set_module, 'QP_HAMILTONIAN_WINDOW',
                            PBE0_SUBSPACE_WINDOW)
        monkeypatch.setattr(adaptive_selection_module, 'QP_HAMILTONIAN_WINDOW',
                            PBE0_SUBSPACE_WINDOW)
        ch = chain(factory, qp_hamiltonian=True, qp_window=PBE0_WINDOW)
    else:
        ch = chain(factory, qp_hamiltonian=True)
    grad = ch.excitation_gradient()[0]
    assert ch.resolved.qp_subspaces[1] == (PBE0_SUBSPACES if factory is pbe0
                                  else (PAIR,))
    assert np.abs(coupling(ch).h).max() > COUPLED
    mol = ch.mol0
    for atom, axis in COORDINATES:
        fd = (ch.excitation(displaced(mol, atom, axis, FD_STEP))
              - ch.excitation(displaced(mol, atom, axis, -FD_STEP))) \
            / (2 * FD_STEP)
        assert abs(fd - grad[atom, axis]) < FD_TOL, (atom, axis, fd,
                                                     grad[atom, axis])


def solvated(qp_hamiltonian=False):
    """The chain in toluene (PCM, Eq. (18) on the diagonal)."""
    warnings.simplefilter('ignore')
    mol = gto.M(atom=FORMALDEHYDE, basis=BASIS, verbose=0)
    return ExcitedStateChain(
        mol, rhf, mf=rhf(mol),
        environment=SolventScreening(mol, solvent='toluene'),
        numerics=g1_numerics(mol, Numerics(qp={'window': WINDOW, 'residue_route': 'sop', 'hamiltonian': qp_hamiltonian},
                                           bse={'solver': 'dense'})))


def test_the_reaction_field_element_is_eq18_to_first_order():
    """Eq. (18) of p rotated by theta into q has slope 2 (pp|Delta W|pq), of
    q rotated into p 2 (qq|Delta W|qp) (occupancy signs aside); the element
    the chain puts on H is the mean of the two slopes over two."""
    ch = solvated(qp_hamiltonian=True)
    ch.excitation()
    fwd = ch._shared_forward(ch.mol0, ch.mf0)
    p, q = PAIR
    pair = [(p, q)]
    dens = [pair_densities(fwd.x_mo, f, pair) for f in (fwd.d, fwd.d_bare)]
    off = offdiagonal_reaction_field(fwd.screening, dens, pair, ch.nocc)[0]
    assert abs(off) > COUPLED

    def eq18(r, s, theta):
        x = fwd.x_mo[:, [r]] * np.cos(theta) + fwd.x_mo[:, [s]] * np.sin(theta)
        (_, wd), (_, wb) = fwd.screening
        return float(projected_quasiparticle_shift(
            fwd.d.T @ x ** 2, wd, fwd.d_bare.T @ x ** 2, wb, 0)[0])

    slopes = [(eq18(r, s, ROTATION) - eq18(r, s, -ROTATION)) / (2 * ROTATION)
              for r, s in ((p, q), (q, p))]
    assert abs(np.mean(slopes) - 2 * off) < ROTATION_TOL * abs(off), (
        slopes, 2 * off)


def test_the_force_is_the_derivative_in_a_continuum():
    ch = solvated(qp_hamiltonian=True)
    grad = ch.excitation_gradient()[0]
    assert ch.resolved.qp_subspaces[1] == (PAIR,)
    mol = ch.mol0
    for atom, axis in COORDINATES:
        fd = (ch.excitation(displaced(mol, atom, axis, FD_STEP))
              - ch.excitation(displaced(mol, atom, axis, -FD_STEP))) \
            / (2 * FD_STEP)
        assert abs(fd - grad[atom, axis]) < FD_TOL, (atom, axis, fd,
                                                     grad[atom, axis])


@pytest.mark.parametrize('environment', [False, True], ids=['gas', 'toluene'])
@pytest.mark.parametrize('coupled', [False, True], ids=['diagonal', 'coupled'])
def test_sliced_factors_over_ranks_are_the_whole_layout(environment, coupled):
    """Over two simulated ranks on sliced factors (the pair contractions over
    the frequency partition, the pair adjoints and Eq. (18)'s off-diagonal
    element on rows, the matrix-free one-body action), the force and Omega
    are the whole layout's on the same mean field, as they are on the
    diagonal route."""

    def rank(comm):
        converged = {}

        def factory(mol):
            key = np.asarray(mol.atom_coords()).tobytes()
            if key not in converged:
                converged[key] = rhf(mol)
            return converged[key]

        out = {}
        for tag, sliced in (('whole', None), ('sliced', True)):
            mol = gto.M(atom=FORMALDEHYDE, basis=BASIS, verbose=0)
            env = (SolventScreening(mol, solvent='toluene') if environment
                   else None)
            ch = ExcitedStateChain(
                mol, factory, mf=factory(mol), environment=env,
                numerics=g1_numerics(mol, Numerics(qp={'window': WINDOW, 'residue_route': 'sop', 'hamiltonian': coupled},
                                                   bse={'solver': 'davidson', 'nroots': 4, 'conv_tol': 1e-9},
                                                   layout={'sliced': sliced})))
            grad, diags = ch.excitation_gradient()
            out[tag] = (grad, diags['omega'])
        return out

    warnings.simplefilter('ignore')
    threads = lib.num_threads()
    lib.num_threads(1)
    try:
        results = run_simulated(rank, RANKS)
    finally:
        lib.num_threads(threads)
    for res in results:
        assert abs(res['sliced'][1] - res['whole'][1]) < LAYOUT_TOL
        assert np.abs(res['sliced'][0] - res['whole'][0]).max() < LAYOUT_TOL
