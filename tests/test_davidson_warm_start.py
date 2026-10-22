"""The Casida Davidson started from a previous solve's roots.

`_davidson_root(guess=(X, Y))` starts from the guess vectors followed by the
cold unit vectors, orthonormalized with their partners
(`trial_space.warm_trial_space`); `DavidsonWarmStart` holds a walk's last
roots and carries them into the next geometry's orbitals through the MO
overlap across the step. A guess moves where the iteration starts, not where
it ends. Gated:

  * the first batch is orthonormal together with its partners, the guess
    spanned first;
  * from its own converged roots and the tracked Ritz pairs above them
    (`_run_davidson`'s `tracked`) a solve ends in its first cycle, serially
    and at 2 and 3 simulated ranks (the pair-row loop), the roots within
    the Davidson's resolution (`roots_resolution`) of the cold ones and the
    same bits on every rank;
  * from a neighbouring problem's vectors it takes fewer cycles than cold,
    to the same roots;
  * the holder undoes orbital sign flips and rotations, and declines another
    spin, another pair space and a step beyond DAVIDSON_WARM_START_MAX_SHIFT;
  * on the production chain (formaldehyde/cc-pVDZ PBE0, ISDF-K G1,
    space-time, SOP, frontier set, Davidson, grid adjoint), the chain's own
    warm start (`davidson_warm_start`, seeded by `commit`): with an empty
    holder, and before a geometry is committed, the solve is the cold one;
    after the commit a repeat there ends in one cycle and a displaced
    geometry takes fewer cycles than cold, the roots
    within the Davidson's resolution of the cold ones and, with the state's
    X - Y, within the vector class bar (residual WARM_CONV_TOL), serially and
    at 2 simulated ranks, the same on every rank.
"""

import warnings

import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.constants import DAVIDSON_LINDEP, DAVIDSON_WARM_START_MAX_SHIFT
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.isdf_jk import isdf_jk
from src.Base.record_io import read_record, record_hash, write_record
from src.Base.separable_ri import resolve_isdf_grid
from src.Base.utils.mpi_grid import distributed, run_simulated
from src.SingleReference.LinearResponse.davidson import DavidsonWarmStart
from src.SingleReference.LinearResponse.davidson_shared import (_guess_indices,
                                                                _tracked_roots)
from src.SingleReference.LinearResponse.trial_space import warm_trial_space
from src.gradients.state_following import coupled
from src.properties.surfaces import potential_energy_surface
from tests.pins import align_signs, class_bar
from tests.test_distributed_fit_mpi import roots_resolution
from tests.test_distributed_trial_space import (NOCC, NROOTS, model_problem,
                                                root)

#: Orthonormality of a Gram-Schmidt batch of unit-norm vectors: eps times the
#: condition of the cold rows' Gram matrix after the projection off the guess
#: (676 on the model, smallest eigenvalue 1.48e-3), about 1.5e-13, which the
#: eigh-based orthonormalization leaves; measured 0.9e-13 to 1.8e-13 on the
#: workstation and 1.07e-13 on a cluster's zen3 nodes (MKL 2025.3), the same batch
#: to the last bits of the library. The bar is the class of vectors that two
#: libraries round differently (`class_bar('vector')`),
#: not a figure above the miss; a batch without the projection off the guess
#: leaves 0.67, one without the partners' overlap 5e-3.
ORTHONORMAL_TOL = class_bar('vector')
#: The coupling of a neighbouring model problem mixed into the model's.
NEIGHBOUR_MIX = 0.02
FORMALDEHYDE = ('C 0.0 0.0 -0.5296; O 0.0 0.0 0.6742; '
                'H 0.0 0.9429 -1.1123; H 0.0 -0.9429 -1.1123')
#: Bohr. The displaced geometry: a finite-difference step on the carbonyl.
STEP = 1e-3
#: The production chain's Davidson residual in the warm-start test, Ha. At the
#: default 1e-8 (BSE_DAVIDSON_CONV_TOL) two solves of one matrix stop up to
#: 2.2e-9 Ha apart: at 2 simulated ranks the cold solve at the displaced
#: geometry passes its residual check a cycle before the serial one (14 against
#: 15). The roots compared under the vector class bar are then the stopping
#: points' and not the warm start's. At 1e-10 every warm-vs-cold and
#: rank-vs-serial gap is at most 2.1e-12 Ha in the roots and 8e-13 in the
#: state's X - Y (workstation, 2 threads), the warm roots within 0.05
#: of the Davidson's resolution of the cold ones.
WARM_CONV_TOL = 1e-10


def neighbour():
    """(apply_AB, diag): the model with a second model's coupling mixed in."""
    apply_a, diag, _, _ = model_problem()
    apply_b, _, _, _ = model_problem(seed=12)

    def apply_AB(z):
        a, b = apply_a(z)
        a2, b2 = apply_b(z)
        return ((1 - NEIGHBOUR_MIX) * a + NEIGHBOUR_MIX * a2,
                (1 - NEIGHBOUR_MIX) * b + NEIGHBOUR_MIX * b2)

    apply_AB.screened_diagonal = lambda: diag
    return apply_AB, diag


def close(om, ref, eps, nocc, resolutions=1):
    """The roots within `resolutions` Davidson resolutions of the reference."""
    return (np.linalg.norm(om - ref) / np.linalg.norm(ref)
            < resolutions * roots_resolution(eps, nocc, ref))


def test_first_batch_is_orthonormal_with_partners():
    """[V, W] and its partners [W, V] are one orthonormal set, the guess
    spanned exactly by the first rows."""
    apply_AB, diag, _, _ = model_problem()
    _, X, Y, _, _, _ = root(apply_AB, diag)
    order = _guess_indices(diag, NROOTS)
    n = diag.size
    x0 = np.zeros((len(order), 2 * n))
    x0[np.arange(len(order)), order] = 1.0
    z = warm_trial_space(X, Y, x0, DAVIDSON_LINDEP)
    V, W = z[:, :n], z[:, n:]
    eye = np.eye(len(z))
    assert np.abs(V @ V.T + W @ W.T - eye).max() < ORTHONORMAL_TOL
    assert np.abs(V @ W.T + W @ V.T).max() < ORTHONORMAL_TOL
    assert NROOTS < len(z) <= NROOTS + len(order)
    # the guess vectors lie in the span of the first NROOTS rows and partners
    basis = np.vstack([z[:NROOTS], np.hstack([W[:NROOTS], V[:NROOTS]])])
    guess = np.vstack([X, Y]).T
    rest = guess - (guess @ basis.T) @ basis
    assert np.abs(rest).max() < ORTHONORMAL_TOL * np.abs(guess).max()


@pytest.mark.parametrize('size', [1, 2, 3])
def test_own_vectors_converge_in_one_cycle(size):
    """Started from its own converged roots and tracked pairs the solve
    ends in its first cycle, at the cold roots, the same on every rank."""
    apply_AB, diag, eps, _ = model_problem()
    held = {}
    om, X, Y, _, cycles, _ = root(apply_AB, diag, tracked=held)
    guess = (held['X'], held['Y'])
    assert held['X'].shape[1] == _tracked_roots(NROOTS, diag.size)
    if size == 1:
        out = [root(apply_AB, diag, guess=guess)]
    else:
        out = run_simulated(
            lambda comm: root(apply_AB, diag, comm=comm, guess=guess), size)
    for warm in out:
        assert warm[3].all() and warm[4] == 1 < cycles
        assert warm[5]['warm_start'] == held['X'].shape[1]
        assert np.array_equal(warm[0], out[0][0])
        assert close(warm[0], om, eps, NOCC)


@pytest.mark.parametrize('size', [1, 2])
def test_neighbouring_vectors_save_cycles(size):
    """From the model's roots, a neighbouring problem converges in fewer
    cycles than cold, to its own cold roots."""
    apply_AB, diag, eps, _ = model_problem()
    _, X, Y, _, _, _ = root(apply_AB, diag)
    moved, _ = neighbour()
    om, _, _, _, cold, _ = root(moved, diag)
    if size == 1:
        out = [root(moved, diag, guess=(X, Y))]
    else:
        out = run_simulated(
            lambda comm: root(moved, diag, comm=comm, guess=(X, Y)), size)
    for warm in out:
        assert warm[3].all() and warm[4] < cold
        assert close(warm[0], om, eps, NOCC)


def test_holder_carries_vectors_into_new_orbitals():
    """The guess at the same geometry in orbitals with flipped signs and a
    rotated occupied pair is the held vectors written in those orbitals;
    another spin, another pair space and a long step are declined."""
    mol = gto.M(atom=FORMALDEHYDE, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    nocc = mol.nelectron // 2
    nvir = mf.mo_coeff.shape[1] - nocc
    rng = np.random.default_rng(3)
    X = rng.standard_normal((nocc * nvir, 3))
    Y = 0.1 * rng.standard_normal((nocc * nvir, 3))
    holder = DavidsonWarmStart()
    holder.keep(mol, mf.mo_coeff, nocc, 'singlet', X, Y)
    # new orbitals C' = C U: signs flipped, occupied 2-3 rotated
    u = np.diag(rng.choice([-1.0, 1.0], mf.mo_coeff.shape[1]))
    c, s = np.cos(0.3), np.sin(0.3)
    u[2:4, 2:4] = u[2:4, 2:4] @ np.array([[c, -s], [s, c]])
    gx, gy = holder.guess(mol, mf.mo_coeff @ u, nocc, 'singlet')
    u_oo, u_vv = u[:nocc, :nocc], u[nocc:, nocc:]
    for held, got in ((X, gx), (Y, gy)):
        want = np.einsum('ij,iak,ab->jbk', u_oo,
                         held.reshape(nocc, nvir, -1), u_vv)
        assert np.abs(got - want.reshape(got.shape)).max() < 1e-12
    assert holder.guess(mol, mf.mo_coeff, nocc, 'triplet') is None
    assert holder.guess(mol, mf.mo_coeff, nocc - 1, 'singlet') is None
    coords = mol.atom_coords()
    coords[0, 2] += 2 * DAVIDSON_WARM_START_MAX_SHIFT
    far = mol.set_geom_(coords, unit='Bohr', inplace=False)
    assert holder.guess(far, mf.mo_coeff, nocc, 'singlet') is None
    assert (holder.used, holder.declined) == (1, 3)


def test_holder_holds_only_the_accepted_geometry():
    """Offered roots seed nothing until a walk accepts their geometry; then
    every spin offered there is held, and the offers elsewhere (a rejected
    trial step) are dropped."""
    mol = gto.M(atom=FORMALDEHYDE, basis='sto-3g', verbose=0)
    coords = mol.atom_coords()
    coords[0, 2] += 1e-3
    trial = mol.set_geom_(coords, unit='Bohr', inplace=False)
    mo = np.eye(mol.nao_nr())
    nocc = mol.nelectron // 2
    n_pair = nocc * (mol.nao_nr() - nocc)
    rng = np.random.default_rng(5)
    holder = DavidsonWarmStart()
    for spin in ('singlet', 'triplet'):
        holder.offer(mol, mo, nocc, spin, rng.standard_normal((n_pair, 2)),
                     rng.standard_normal((n_pair, 2)))
    holder.offer(trial, mo, nocc, 'singlet', rng.standard_normal((n_pair, 2)),
                 rng.standard_normal((n_pair, 2)))
    assert holder.guess(mol, mo, nocc, 'singlet') is None
    assert holder.accept(mol)
    assert set(holder.held) == {'singlet', 'triplet'} and not holder.offered
    held = holder.held['singlet'][3]
    assert not holder.accept(trial)
    assert holder.held['singlet'][3] is held


def test_holder_record_reads_back_to_the_bit(tmp_path):
    """A chain's record carries its warm start (`to_record`): read back,
    the holder guesses what the original guesses, bit for bit (a stored
    copy, not a second computation)."""
    mol = gto.M(atom=FORMALDEHYDE, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    nocc = mol.nelectron // 2
    n_pair = nocc * (mf.mo_coeff.shape[1] - nocc)
    rng = np.random.default_rng(7)
    holder = DavidsonWarmStart()
    holder.keep(mol, mf.mo_coeff, nocc, 'singlet',
                rng.standard_normal((n_pair, 3)),
                rng.standard_normal((n_pair, 3)))
    write_record(tmp_path / 'warm.json', holder.to_record())
    back = DavidsonWarmStart.from_record(
        read_record(tmp_path / 'warm.json'), mol)
    assert record_hash(back.to_record()) == record_hash(holder.to_record())
    for a, b in zip(holder.guess(mol, mf.mo_coeff, nocc, 'singlet'),
                    back.guess(mol, mf.mo_coeff, nocc, 'singlet')):
        assert np.array_equal(a, b)


def formaldehyde(shift=0.0):
    """Formaldehyde/cc-pVDZ, the carbon moved by `shift` Bohr along the
    carbonyl."""
    mol = gto.M(atom=FORMALDEHYDE, basis='cc-pvdz', verbose=0)
    if not shift:
        return mol
    coords = mol.atom_coords()
    coords[0, 2] += shift
    return mol.set_geom_(coords, unit='Bohr', inplace=False)


def chain_solves():
    """(solves, holder.used) on the production chain, each solve (roots,
    eps_qp, Davidson cycles, guess vectors kept, the state's X - Y) through
    the chain's own hook (`_casida(at=)`, `commit`): cold at the reference
    and at the displaced geometry (no holder); then with a fresh holder at
    the reference (empty: cold), at the displaced geometry before any commit
    (the reference's roots only offered: cold), at the reference once it is
    committed (from its own roots) and at the displaced geometry (from the
    reference committed again on those roots). Each geometry's forward is built once, so a warm
    and a cold solve read the same matrix."""
    warnings.simplefilter('ignore')
    counts, n_start = resolve_isdf_grid('G1', 'cc-pvdz', ['C', 'H', 'O'],
                                        auxbasis='cc-pvdz-ri')

    def factory(mol):
        mf = isdf_jk(dft.RKS(mol, xc='pbe0'), auxbasis='cc-pvdz-ri',
                     counts=counts, n_start=n_start)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-9, 200
        return mf

    chain = potential_energy_surface(
        formaldehyde(), factory, ground_state=GroundState('dft', 'pbe0'),
        excitation=Excitation('singlet', root=1, kernel='bse'),
        chi0='space-time', residues='sop', solver='davidson',
        factorization='isdf', qp_states=QPStates('frontier'),
        grid_accuracy='G1', bse_adjoint='grid', nroots=5,
        bse_conv_tol=WARM_CONV_TOL)
    assert isinstance(chain.davidson_warm_start, DavidsonWarmStart)
    chain.timer = {}
    at0 = chain.mean_field()
    at1 = chain.mean_field(formaldehyde(STEP))
    shared = [chain._shared_forward(*at) for at in (at0, at1)]

    def solve(s, at):
        om, xn, yn, _ = chain._casida(s.x_mo, s.d, s.eps_qp, s.w_aux,
                                      **coupled(s.qp_h), at=at)
        timings = chain.davidson_solves[-1]['timings']
        state = chain.state
        return (np.array(om), np.array(s.eps_qp),
                timings['davidson_iterations'],
                timings['davidson_warm_start'],
                np.array(xn[:, state] - yn[:, state]))

    chain.davidson_warm_start = None
    out = [solve(shared[0], at0), solve(shared[1], at1)]
    holder = chain.davidson_warm_start = DavidsonWarmStart()
    out += [solve(shared[0], at0), solve(shared[1], at1)]
    chain.commit(at0[0])
    out.append(solve(shared[0], at0))
    # committed again on its warm roots, as every step of a walk after the
    # first is seeded by a warm solve: seeded by the cold solve's Ritz pairs
    # the displaced geometry takes about as many cycles as cold (18 against
    # 18-20, workstation), by the warm one's 15
    chain.commit(at0[0])
    out.append(solve(shared[1], at1))
    return out, holder.used


def test_production_chain_warm_start():
    """Serially and at 2 simulated ranks, through the chain's own warm start:
    with an empty holder, and before the reference is committed, the solve
    is the cold one; once it is, the reference ends in one cycle and the
    displaced geometry takes fewer cycles than cold; every warm root is the
    cold one to the Davidson's resolution and within the vector class bar
    (`class_bar('vector')`, as the first batch's orthonormality), the state's
    X - Y within it too, and every rank's the same. No two solves are
    compared bit for bit."""
    nocc = 8
    vector_bar = class_bar('vector')
    with distributed(None):
        serial, used = chain_solves()
    ranked = run_simulated(lambda comm: chain_solves(), 2)
    for out, n_used in [(serial, used)] + ranked:
        cold0, cold1, first, early, again, warm = out
        assert n_used == 2 and first[3] == 0 and early[3] == 0
        assert first[2] == cold0[2] and early[2] == cold1[2]
        assert again[3] > 0 and again[2] == 1 < cold0[2]
        assert warm[3] > 0 and warm[2] < cold1[2]
        for got, ref in ((first, cold0), (early, cold1), (again, cold0),
                         (warm, cold1)):
            assert close(got[0], ref[0], ref[1], nocc)
            assert np.abs(got[0] - ref[0]).max() <= vector_bar
            gap = np.abs(align_signs(got[4], ref[4]) - ref[4]).max()
            assert gap <= vector_bar, (gap, vector_bar)
    for a, b in zip(ranked[0][0], ranked[1][0]):
        assert a[2] == b[2]
        assert close(a[0], b[0], b[1], nocc)
