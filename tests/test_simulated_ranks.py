"""Every distributed route reproduces the serial answer under simulated ranks.

`mpi_grid.simulated_world` runs each rank in a thread of this process and
reduces through shared memory, so the whole distributed code path -- the tau
partitions, the frequency partitions, the row split of Zt, every reduction
and its placement -- runs here, where MPI itself cannot start (the sandboxed
test runner refuses the socket MPI_Init binds). What it does not test is the
wire protocol; tests/test_mpi_routes.py under mpirun does.

Two and three ranks, because one of each kind of imbalance is enough to catch
a partition that only works when everything divides evenly: with 8 tau points
and 24 frequencies, three ranks leave remainders on both axes.

Gates, all against the serial call in the same process:
  * quasiparticle-set gradient (`qp_set_gradient`): roots bitwise, the three
    adjoints at 1e-11 relative -- the frequency-outside reverse pass sums the
    states' adjoints in a different order under a partition;
  * the SINGLE-state quasiparticle gradient (`qp_gradient_space_time`), the
    same partition with one state, on the contour-deformation and the
    pole-model route -- and again at 8 and 16 ranks, where the fixture has
    fewer tau points than ranks and the surplus own an empty block;
  * dRPA correlation energy and adjoint (`rpa_energy_and_adjoint`);
  * the BSE kernel's static W (`static_screening`);
  * the screened interaction in imaginary time and the self-energy matrix and
    diagonal with their adjoints (`screened_interaction_tau`,
    `selfenergy_block`, `selfenergy_diag`);
  * static W and the whole ISDF BSE (`isdf_bse_factors`, `solve_bse_isdf`),
    including the Davidson on the row-split block action;
  * the BSE@GW surface end to end (`ExcitedStateChain`), both forces at the
    routes test's anchored bar, `COMPOSED_GRAD_K` times what re-associating
    the serial force's sums on one BLAS thread moves it, floored at
    `COMPOSED_GRAD_FLOOR` for the excitation force and at the ISDF gradient
    reproducibility floor for the quasiparticle force;
  * that the ranks are made to hold ONE calculation and to follow ONE
    iteration: `replicate`, and the BSE run with rank 1 handed a perturbed
    spectrum and perturbed factors, which must come back as rank 0's roots,
    bitwise, with every rank's Davidson taking rank 0's iteration count; the
    GW window with rank 1
    handed a perturbed mean field, which its entry lockstep repairs, and
    perturbed factors, which it does NOT broadcast again -- factors are
    identical by construction, `separable_factors` locksteps its own -- and
    which an audited run reports instead (`mpi_grid.agreement`);
  * that they hold ONE GRID, which a chain decides for itself rather than
    receiving: a `FactorChain` built on every rank with rank 1's radii,
    frames and fit perturbed must end with rank 0's radii, clouds, frames,
    points, pair layout and fit, the radii search having run on every rank;
    and the points continued frames place at a displaced geometry are rank
    0's even where rank 1's frames moved;
  * that two ranks sharing a filesystem cannot tear the radii cache they both
    write;
  * the minimax transform's pseudo-inverse cutoff, which must leave every
    grid here bitwise and still refuse a NaN row.
Water/cc-pVDZ, Hartree-Fock, one factorization shared by every test.

WHICH GATES ARE BITWISE. A reduction over disjoint zero-padded slots --
proj(tau) and the branch sums of the self-energy, where a rank writes only the
tau points it owns -- adds exact zeros and is bitwise. Everything reduced as a
PARTIAL SUM is re-associated instead: chi0 accumulates its tau points as they
arrive, Wt(tau) and projbar sum over frequencies, so a partition sums the same
terms in a different order and the agreement is at the last bits, not in them.
Those gates are relative, and the tolerance says so -- `SIGMA_REL` where the
re-associated sum is a cancelling one and the last bits move further.

EVERY GATE HERE WAS SHOWN TO FAIL. A reduction turned into a no-op (the
collective still called, on a copy, so the ranks stay in step and the failure
is a wrong answer rather than a deadlock) breaks exactly the gate that covers
it:

  reduction made a no-op                 gates that then fail
  chi0 in `static_screening`             static_screening (both sizes) and
                                         both chain gradients, 1.2e-2 and
                                         1.6e-2 Ha/Bohr out
  Wt(tau) in `screened_interaction_tau`  screened_interaction_tau, and every
                                         self-energy forward and backward
  Wt_bar in selfenergy_BLOCK_backward    the block gate only; the diagonal
                                         one still passes
  Wt_bar in selfenergy_DIAG_backward     the diagonal gate only
  projbar in `qp_gradient_space_time`    both routes of the single-state
                                         gradient and both chain gradients,
                                         1.6e-3 and 2.2e-3 Ha/Bohr out

  replicate made a no-op                 test_replicate_overwrites_every_rank;
                                         and the
                                         perturbed-rank BSE, where the mixed
                                         partials put min eig(A-B) at
                                         -0.031 Ha and the driver refused the
                                         solve as an unstable REFERENCE -- the
                                         defect wearing the mask of physics
  the GW window's entry lockstep of the  the perturbed-mean-field GW window:
  mean field removed                     every rank 0.45 Ha off serial, the
                                         ranks' chi0 partials taken from two
                                         spectra
  TRANSFORM_FIT_RCOND raised to 1e-14    test_transform_pseudo_inverse: this
                                         grid's omega -> tau weights move in
                                         the sixth digit
  the construction lockstep of           the radii, 5% apart -- the first of
  `FrozenFactorization` removed          the six arrays that gate compares
  `coords`'s lockstep removed            test_placed_points_are_rank_zeros,
                                         rank 1's continued frames placing its
                                         own points; the frozen-frame gate
                                         still passes, its points being one
                                         function of the locked clouds
  `shareable_factors`'s lockstep         the fit M, rank 1's 1e-6 intact;
  removed                                X_ao and the layout are the same
                                         function of one grid and stay
                                         bitwise, which is what says it is
                                         the fit this one carries
  the cache temporary named on the       test_cache_write_survives_a_shared_path,
  pid again (`write_json_atomic`)        with a JSONDecodeError inside the file
                                         the rename published
"""
import copy
import json
import os
import sys
import threading
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base import separable_ri
from src.Base.constants import (COMPOSED_GRAD_K, ISDF_GRADIENT_FLOOR,
                                TRANSFORM_FIT_RCOND)
from src.Base.utils import time_frequency
from src.Base.utils.grids import gauss_legendre_grid
from src.Base.utils.mpi_grid import (distributed, lockstep_stats, replicate,
                                     run_simulated)
from src.Base.utils.time_frequency import (COSINE_TW, COSINE_WT,
                                           TimeFrequencyGrid,
                                           minimax_transform_weights)
from src.SingleReference.GW.space_time import (separable_factors,
                                               solve_qp_energy_space_time)
from src.SingleReference.LinearResponse.davidson import (isdf_bse_factors,
                                                         solve_bse_isdf)
from src.gradients import factor_chain
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.factor_chain import FactorChain, FrozenFactorization
from src.gradients.qp_space_time import (qp_gradient_space_time,
                                         qp_set_gradient)
from src.gradients.reaction_field_adjoint import static_grid, static_screening
from src.gradients.space_time_adjoint import (polarizability_tau,
                                              rpa_energy_and_adjoint,
                                              screened_interaction_tau,
                                              selfenergy_block,
                                              selfenergy_block_backward,
                                              selfenergy_diag,
                                              selfenergy_diag_backward,
                                              sigma_transforms)
from tests.test_mpi_routes import (COMPOSED_GRAD_FLOOR, one_thread,
                                   one_thread_scatter)

NTAU, NFREQ = 8, 24
REL = 1e-11
#: The self-energy routes re-associate a CANCELLING sum. Wt(tau) is
#: sum_w Ctw[t,w] (W_w - I) with minimax weights that alternate in sign and
#: are large against Wt itself, so a partition over frequency moves the last
#: bits by the cancellation factor of that sum -- measured 1e-12 here against
#: the 1e-16 of a benign one, and unchanged from 8 to 24 tau points, which is
#: what says it is the transform and not the grid's fit.
SIGMA_REL = 1e-10
SIZES = [2, 3]
#: More ranks than the fixture has items on an axis, where the surplus own an
#: EMPTY block: 8 ranks is one tau point each, 16 leaves eight ranks with no
#: tau point at all and every rank one or two frequencies. Run on the
#: single-state gradient alone -- it carries both partitions and every
#: reduction of the set route on one state, at a sixteenth of the cost.
MANY = [8, 16]
BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'


@pytest.fixture(scope='module')
def water():
    warnings.simplefilter('ignore')
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')
    X, D = factors[0], factors[1]
    eps = np.asarray(mf.mo_energy, float)
    gap = eps[nocc] - eps[nocc - 1]
    nu, wt = gauss_legendre_grid(NFREQ, w0=gap)
    grid = TimeFrequencyGrid.minimax_split(NTAU, 0.5 * gap, eps[-1] - eps[0],
                                           nu, wt, with_sine=False,
                                           with_inverse=False)
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])
    states = [nocc - 2, nocc - 1, nocc, nocc + 1]
    # The self-energy's own quadrature weights on this axis: the omega -> tau
    # half feeds Wt(tau), the two tau -> omega halves the continuation.
    transforms = sigma_transforms(eps, nocc, grid.tau_points, grid.omega_points,
                                  grid.omega_points, mu=mu)
    return dict(mf=mf, mol=mol, nocc=nocc, factors=factors, X=X, D=D, eps=eps,
                grid=grid, nu=nu, wt=wt, mu=mu, states=states,
                transforms=transforms, w_grid=static_grid(eps, nocc),
                sigma_states=states[1:3],
                weights=np.array([0.3, -1.1, 0.8, 0.45]))


def close(a, b, rel=REL):
    a, b = np.asarray(a), np.asarray(b)
    scale = max(np.abs(a).max(), np.abs(b).max())
    return scale == 0 or np.abs(a - b).max() <= rel * scale


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('route', ['explicit'])   # the backend every checkout has
def test_qp_set_gradient(water, size, route):
    w = water
    args = (w['X'], w['D'], w['eps'], w['nocc'], w['grid'], w['nu'], w['wt'],
            w['states'], w['weights'])
    kw = dict(mu=w['mu'], residue_route=route)
    serial = qp_set_gradient(*args, **kw)

    def one_rank(comm):
        return qp_set_gradient(*args, comm=comm, **kw)

    for roots, eps_bar, X_bar, D_bar in run_simulated(one_rank, size):
        assert np.array_equal(roots, serial[0])        # replicated, so bitwise
        assert close(eps_bar, serial[1])
        assert close(X_bar, serial[2])
        assert close(D_bar, serial[3])


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('route', ['explicit', 'sop'])
def test_qp_gradient_space_time(water, size, route):
    """The SINGLE-state route carries `qp_set_gradient`'s partition.

    Both branches of the reverse pass are exercised: 'explicit' takes the
    contour-deformation weights, where eps_bar picks up the Lorentzian's own
    derivative, and 'sop' takes the pole model, where `wc_bar` carries the
    whole adjoint and that term is absent. The root and the pole strength read
    the reduced wc, whose frequency partition writes one rank's rows and leaves
    the rest exact zeros, so under a reduction taken in rank order they are
    bitwise; the adjoints are partial sums over that partition and agree only
    to the last bits.
    """
    w = water
    args = (w['X'], w['D'], w['eps'], w['nocc'], w['grid'], w['nu'], w['wt'],
            w['nocc'] - 1)
    kw = dict(mu=w['mu'], residue_route=route)
    serial = qp_gradient_space_time(*args, **kw)

    def one_rank(comm):
        return qp_gradient_space_time(*args, comm=comm, **kw)

    for w_star, z, eps_bar, X_bar, D_bar in run_simulated(one_rank, size):
        assert w_star == serial[0] and z == serial[1]   # zero-padded, so bitwise
        assert close(eps_bar, serial[2])
        assert close(X_bar, serial[3])
        assert close(D_bar, serial[4])


@pytest.mark.parametrize('size', MANY)
@pytest.mark.parametrize('route', ['explicit', 'sop'])
def test_qp_gradient_space_time_empty_blocks(water, size, route):
    """The same route where a rank owns NOTHING on an axis.

    An empty block must contribute what the serial sum contributes for those
    indices -- nothing, in the same shape and dtype -- so eight tau points over
    sixteen ranks give the serial answer as exactly as two ranks do:
    `partition` hands back an empty index array, the sweep writes no slot, and
    the reduction adds exact zeros. The frequency axis is never empty here (24
    over 16) but drops to one row a rank, which is where a contraction written
    for a block rather than a row would break.

    BITWISE HERE IS A PROPERTY OF THIS COMMUNICATOR, not of the route: the
    simulated reduction adds the ranks' buffers in rank order. A real
    `Allreduce` adds them in whatever order its tree picks, which the
    zero-padded forward partitions survive and the reverse pass's partial sums
    do not, so tests/test_mpi_routes.py gates the same comparison on a number.
    """
    w = water
    args = (w['X'], w['D'], w['eps'], w['nocc'], w['grid'], w['nu'], w['wt'],
            w['nocc'] - 1)
    kw = dict(mu=w['mu'], residue_route=route)
    serial = qp_gradient_space_time(*args, **kw)

    def one_rank(comm):
        return qp_gradient_space_time(*args, comm=comm, **kw)

    for w_star, z, eps_bar, X_bar, D_bar in run_simulated(one_rank, size):
        assert w_star == serial[0] and z == serial[1]
        assert close(eps_bar, serial[2])
        assert close(X_bar, serial[3])
        assert close(D_bar, serial[4])


@pytest.mark.parametrize('size', SIZES)
def test_static_screening(water, size):
    """W(0) for the BSE kernel off the tau partition, the arithmetic
    `isdf_bse_factors(distribute=True)` runs for the same matrix.

    chi0 is accumulated over its tau points rather than kept per point, so the
    reduction re-associates that sum: relative, not bitwise. A is a local
    contraction of the factors and carries no partition at all.
    """
    w = water
    args = (w['X'], w['D'], w['eps'], w['nocc'], w['w_grid'])
    a0, w0 = static_screening(*args)

    for a, wmat in run_simulated(lambda c: static_screening(*args, comm=c), size):
        assert np.array_equal(a, a0)            # no sweep, so bitwise
        assert close(wmat, w0)


@pytest.mark.parametrize('size', SIZES)
def test_screened_interaction_tau(water, size):
    """Wt(tau) off the frequency partition, on a proj(tau) built serially.

    The naux^3 Dyson inversion is the per-frequency work and Wt(tau) is a sum
    over frequencies, so this is the one split here that both saves the
    compute and pays a reduction.
    """
    w = water
    proj = polarizability_tau(w['X'], w['D'], w['eps'], w['nocc'], w['grid'],
                              mu=w['mu'])
    ctw = w['transforms'][0]
    ref = screened_interaction_tau(proj, w['grid'], ctw)

    for got in run_simulated(
            lambda c: screened_interaction_tau(proj, w['grid'], ctw, comm=c),
            size):
        assert close(got, ref, rel=SIGMA_REL)


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('masked', [False, True])
def test_selfenergy_block_and_backward(water, size, masked):
    """Sigma^c_pq and its adjoint, forward and reverse on the same axes.

    The masked branch is a different code path -- `intermediate` restricts the
    summed orbital and narrows the fit range -- so it is partitioned here too.
    The reverse pass reads the forward's cache, which leaves reduced, so both
    ranks differentiate the same tape.
    """
    w = water
    nocc, states = w['nocc'], w['sigma_states']
    inter = list(range(nocc - 2)) + list(range(nocc + 2, len(w['eps']))) \
        if masked else None
    tr = sigma_transforms(w['eps'], nocc, w['grid'].tau_points,
                          w['grid'].omega_points, w['grid'].omega_points,
                          mu=w['mu'], intermediate=inter)
    args = (w['X'], w['D'], w['eps'], nocc, w['grid'], states, tr, w['mu'])
    sig0, cache0 = selfenergy_block(*args, intermediate=inter)
    rng = np.random.default_rng(11)
    shape = (len(w['grid'].omega_points), len(states), len(states))
    a_re, a_im = rng.normal(size=shape), rng.normal(size=shape)
    ref = selfenergy_block_backward(a_re, a_im, *args, cache0,
                                    intermediate=inter)

    def one_rank(comm):
        sig, cache = selfenergy_block(*args, intermediate=inter, comm=comm)
        bwd = selfenergy_block_backward(a_re, a_im, *args, cache,
                                        intermediate=inter, comm=comm)
        return sig, bwd

    for sig, bwd in run_simulated(one_rank, size):
        assert close(sig.real, sig0.real, rel=SIGMA_REL)
        assert close(sig.imag, sig0.imag, rel=SIGMA_REL)
        for got, want in zip(bwd, ref):
            assert close(got, want, rel=SIGMA_REL)


@pytest.mark.parametrize('size', SIZES)
def test_selfenergy_diag_and_backward(water, size):
    """Sigma^c_pp and its adjoint: `selfenergy_block`'s partition with one
    index of Sigma fixed."""
    w = water
    nocc, states = w['nocc'], w['sigma_states']
    args = (w['X'], w['D'], w['eps'], nocc, w['grid'], states, w['transforms'],
            w['mu'])
    sig0, cache0 = selfenergy_diag(*args)
    rng = np.random.default_rng(13)
    shape = (len(states), len(w['grid'].omega_points))
    a_re, a_im = rng.normal(size=shape), rng.normal(size=shape)
    ref = selfenergy_diag_backward(a_re, a_im, *args, cache0)

    def one_rank(comm):
        sig, cache = selfenergy_diag(*args, comm=comm)
        return sig, selfenergy_diag_backward(a_re, a_im, *args, cache,
                                             comm=comm)

    for sig, bwd in run_simulated(one_rank, size):
        assert close(sig.real, sig0.real, rel=SIGMA_REL)
        assert close(sig.imag, sig0.imag, rel=SIGMA_REL)
        for got, want in zip(bwd, ref):
            assert close(got, want, rel=SIGMA_REL)


@pytest.mark.parametrize('size', SIZES)
def test_rpa_energy_and_adjoint(water, size):
    w = water
    args = (w['X'], w['D'], w['eps'], w['nocc'], w['grid'])
    # (E_c, eps_bar, X_bar, D_bar) first; a fold adjoint may follow in trees
    # that carry the solvated dRPA fold, and is not this test's business
    e0, eps0, X0, D0 = rpa_energy_and_adjoint(*args, mu=w['mu'])[:4]

    def one_rank(comm):
        return rpa_energy_and_adjoint(*args, mu=w['mu'], comm=comm)[:4]

    for e, eps_bar, X_bar, D_bar in run_simulated(one_rank, size):
        assert abs(e - e0) <= REL * abs(e0)
        assert close(eps_bar, eps0) and close(X_bar, X0) and close(D_bar, D0)


@pytest.mark.parametrize('size', SIZES)
def test_static_w_and_bse(water, size):
    # probe=False here: this gate is the reduction, and the probe is a second
    # eigensolver in front of it. Replicated on every rank it is a Lanczos
    # (`davidson._lanczos_lowest`), not ARPACK -- scipy's eigsh holds one
    # process-wide lock across its whole iteration, so rank threads deadlock
    # in it -- and test_bse_is_rank_zeros_with_perturbed_ranks turns it on.
    w = water
    _, _, W0 = isdf_bse_factors(w['mf'], w['mol'], w['nocc'], factors=w['factors'])
    om0, _, _, info0 = solve_bse_isdf(w['mf'], w['mol'], w['nocc'], nroots=3,
                                      probe=False, progress=False)

    def one_rank(comm):
        _, _, W = isdf_bse_factors(w['mf'], w['mol'], w['nocc'],
                                   factors=w['factors'], distribute=True,
                                   comm=comm)
        om, _, _, info = solve_bse_isdf(w['mf'], w['mol'], w['nocc'], nroots=3,
                                        probe=False, progress=False,
                                        distribute=True, comm=comm)
        return W, om, info

    for W, om, info in run_simulated(one_rank, size):
        assert close(W, W0)
        assert close(om, om0, rel=1e-9)
        assert info['nranks'] == size
        assert close(info['eps'], info0['eps'])


@pytest.mark.parametrize('size', SIZES)
def test_blocked_gw_path(water, size):
    w = water
    homo = solve_qp_energy_space_time(w['mf'], w['mol'], w['nocc'], w['nocc'] - 1,
                                      factors=w['factors'])

    def one_rank(comm):
        return solve_qp_energy_space_time(w['mf'], w['mol'], w['nocc'],
                                          w['nocc'] - 1, factors=w['factors'],
                                          freq_block=3, distribute=True,
                                          comm=comm)

    for qp in run_simulated(one_rank, size):
        assert abs(qp - homo) < 1e-12


# ------------------------------------------------------- the whole surface
def chain_scf(mol):
    """A mean field converged for gradient work (conv_tol_grad 1e-11)."""
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    return mf


def own_chain():
    """A chain on this rank's OWN Mole.

    ONE MOLE PER RANK, AND NOT BECAUSE OF THE PHYSICS. The nuclear gradient
    reaches pyscf's `hcore_generator`, which evaluates the nuclear-attraction
    derivative inside `mol.with_rinv_at_nucleus(ia)` -- an IN-PLACE write to
    the shared `mol._env`. Real ranks are separate processes and never see
    each other's; two rank THREADS sharing one Mole overwrite each other's
    rinv origin, and the barriers here make them collide reliably (measured:
    the ranks then return gradients 0.5 Ha/Bohr apart, and disagree with each
    other, while the distributed sweeps themselves agree bitwise across
    ranks). That is an artifact of simulating ranks inside one process, not a
    property of the distribution; giving each rank its own Mole removes it.
    """
    mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
    return ExcitedStateChain(mol, chain_scf, mf=chain_scf(mol))


@pytest.mark.parametrize('size', SIZES)
def test_excited_state_chain_gradients(size):
    """Both of the surface's forces, end to end, within their anchored bars.

    Everything the chain distributes meets here at once: the static W of the
    BSE kernel, the quasiparticle set's gradient, the single-state
    quasiparticle gradient, and the Casida solve on top of them. A partition
    re-associates their sums, the orbital response amplifies the last bits of
    any re-association, and every rank converges its own SCF, so each bar is
    measured here as the routes test measures it: `COMPOSED_GRAD_K` times
    what the serial force moves on one BLAS thread, `COMPOSED_GRAD_FLOOR` at
    the least for the excitation force and `ISDF_GRADIENT_FLOOR` for the
    quasiparticle force, whose chain carries no Casida step.
    """
    chain = own_chain()
    ref_ex = chain.excitation_gradient()[0]
    qp_chain = own_chain()
    ref_qp = qp_chain.quasiparticle_gradient(0)[0]
    scatter = one_thread_scatter(chain.mol0, chain.mf0, ref_ex)
    bar = max(COMPOSED_GRAD_FLOOR, COMPOSED_GRAD_K * scatter)
    again_qp = one_thread(lambda: ExcitedStateChain(
        qp_chain.mol0, chain_scf, mf=qp_chain.mf0).quasiparticle_gradient(0)[0])
    scatter_qp = (0.0 if again_qp is None
                  else float(np.abs(again_qp - ref_qp).max()))
    bar_qp = max(ISDF_GRADIENT_FLOOR, COMPOSED_GRAD_K * scatter_qp)

    def one_rank(comm):
        return (own_chain().excitation_gradient()[0],
                own_chain().quasiparticle_gradient(0)[0])

    out = run_simulated(one_rank, size)
    d_ex = max(np.abs(g_ex - ref_ex).max() for g_ex, _ in out)
    d_qp = max(np.abs(g_qp - ref_qp).max() for _, g_qp in out)
    print(f'[info] {size} ranks: excitation force |d| {d_ex:.2e} = '
          f'{d_ex / bar:.3f} of the anchored bar {bar:.2e} = max('
          f'{COMPOSED_GRAD_FLOOR:.1e}, {COMPOSED_GRAD_K} x {scatter:.2e}) '
          f'Ha/Bohr; quasiparticle force |d| {d_qp:.2e} = '
          f'{d_qp / bar_qp:.3f} of {bar_qp:.2e} = max('
          f'{ISDF_GRADIENT_FLOOR:.1e}, {COMPOSED_GRAD_K} x {scatter_qp:.2e})')
    for g_ex, g_qp in out:
        assert np.abs(g_ex - ref_ex).max() < bar
        assert np.abs(g_qp - ref_qp).max() < bar_qp
    for g_ex, g_qp in out[1:]:                  # one surface on every rank
        assert np.array_equal(g_ex, out[0][0])
        assert np.array_equal(g_qp, out[0][1])
    assert np.abs(ref_ex).max() > 1e-3          # there is a force to compare


# ------------------------------------------- the ranks must hold one problem
def test_replicate_overwrites_every_rank(water):
    """`replicate` leaves rank 0's arrays, and rank 0's objects, everywhere.

    The contract the distributed entry points rest on: partials of two
    slightly different calculations are not partials of anything, so the
    inputs are made one calculation's before any of them is contracted.
    """
    ref = np.array([1.0, 2.0, 3.0])

    def one_rank(comm):
        rank = comm.Get_rank()
        mine = ref + (0.0 if rank == 0 else 1.0e-3 * rank)
        label = {'ntau': 18 if rank == 0 else 12}
        label = replicate(mine, label, comm=comm)[1]
        return mine, label

    for arr, label in run_simulated(one_rank, 2):
        assert np.array_equal(arr, ref)
        assert label == {'ntau': 18}


def test_frozen_conventions_are_rank_zeros(monkeypatch):
    """The chain's grid and its fit are rank 0's on every rank that decided them.

    THE ROUTES REPLICATE THEIR INPUTS; THE CHAIN DECIDES ITS OWN. Every
    distributed entry point starts by overwriting the ranks' mean field and
    factors with rank 0's, which is why they agree across nodes. A chain does
    not: it freezes radii, interpolation points, frames and a pair layout at
    the reference geometry, and across nodes each rank froze its own -- the
    route gates passed on every rank of two and four nodes while the
    end-to-end forces missed by 1.5e-3 and 5.7e-3 Ha/Bohr on the non-root
    ranks.

    Rank 1 is given an atomic optimizer that returns radii 5% larger, frames
    and a fit off in their last bits -- the three ways a node differs: a local
    descent on a multi-modal objective landing on another minimum, an `eigh`,
    and a dense solve. None may survive: the radii, the clouds, the frames,
    the points, the pair layout and the fit come back bitwise rank 0's. Every
    rank runs the search, as serial code, and the counter shows it did -- the
    lockstep after it is what makes the answer one.

    A perturbation is needed to see any of this at all. Two rank THREADS run
    the same arithmetic on the same libraries and agree bit for bit, which is
    exactly why the defect was invisible on a laptop and appeared only across
    nodes -- so the divergence is INJECTED here, and what is gated is that the
    lockstep erases it.
    """
    calls = []
    local = threading.local()
    real_radii = factor_chain.optimize_atomic_radii
    real_frames = factor_chain.atomic_frames
    real_fit = factor_chain.fit_M_stable

    def rank():
        return getattr(local, 'rank', 0)

    def optimizer(element, basis, auxbasis, **kw):
        calls.append(rank())
        radii, error = real_radii(element, basis, auxbasis, **kw)
        if rank() == 0:
            return radii, error
        return {k: 1.05 * np.asarray(v) for k, v in radii.items()}, error

    def frames(mol, **kw):
        axes, degenerate = real_frames(mol, **kw)
        return (axes if rank() == 0 else axes * (1.0 + 1e-12)), degenerate

    def fit(D, F, **kw):
        out = real_fit(D, F, **kw)
        return out if rank() == 0 else out * (1.0 + 1e-6)

    monkeypatch.setattr(factor_chain, 'optimize_atomic_radii', optimizer)
    monkeypatch.setattr(factor_chain, 'atomic_frames', frames)
    monkeypatch.setattr(factor_chain, 'fit_M_stable', fit)

    def one_rank(comm):
        local.rank = comm.Get_rank()
        mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
        chain = FactorChain(mol, chain_scf, mf=chain_scf(mol))
        crd = chain.coords(mol)
        x_ao, m_fit, _ = chain.factorization.shareable_factors(
            mol, chain.auxmol(mol), crd)
        return (chain.radii, np.vstack(chain.pts_local), chain.frames, crd,
                x_ao, m_fit, chain.layout)

    out = run_simulated(one_rank, 2)
    assert set(calls) == {0, 1}                 # every rank searched
    radii0, pts0, frames0, crd0, x0, m0, layout0 = out[0]
    for radii, pts, frames, crd, x_ao, m_fit, layout in out[1:]:
        for el in radii0:
            for shell in radii0[el]:
                assert np.array_equal(np.atleast_1d(radii[el][shell]),
                                      np.atleast_1d(radii0[el][shell]))
        # the atom-local clouds and the frames as well as the points they
        # place: the reverse pass reads them directly (`collocation_adjoint`)
        assert np.array_equal(pts, pts0) and np.array_equal(frames, frames0)
        assert np.array_equal(crd, crd0)
        assert np.array_equal(x_ao, x0)
        assert np.array_equal(m_fit, m0)
        for got, want in zip(layout, layout0):
            assert np.array_equal(got, want)


def test_placed_points_are_rank_zeros(monkeypatch):
    """The interpolation points placed at a displaced geometry are rank 0's.

    Frozen frames place the points by one small matmul of the locked clouds,
    which two rank threads compute bit for bit alike; CONTINUED frames are
    re-derived at every geometry (`continued_frames`, through the `eigh` of
    `atomic_frames`), which is where a node's own bits enter. Rank 1's
    continued frames are moved in their last bits here, and every rank must
    still place rank 0's points.
    """
    local = threading.local()
    real = factor_chain.continued_frames

    def continued(mol, frames, **kw):
        out = real(mol, frames, **kw)
        return out if getattr(local, 'rank', 0) == 0 else out * (1.0 + 1e-12)

    monkeypatch.setattr(factor_chain, 'continued_frames', continued)

    def one_rank(comm):
        local.rank = comm.Get_rank()
        mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
        shifted = gto.M(atom='O 0 0 0.117; H 0 0.787 -0.468; H 0 -0.757 -0.468',
                        basis=BASIS, verbose=0)
        return FrozenFactorization(mol, frames='continued').coords(shifted)

    out = run_simulated(one_rank, 2)
    assert np.array_equal(out[1], out[0])


def test_cache_write_survives_a_shared_path(tmp_path):
    """Two ranks writing one cache path leave a whole file, and the same one.

    The radii cache is written by whichever rank optimizes first, and a SLURM
    array optimizes the same element from every node at once onto one shared
    filesystem. A temporary named on the pid is not unique across NODES --
    pids repeat from one node to the next -- so the two writers open the same
    temporary, each truncating the other's bytes, and the rename then publishes
    a torn file that every reader takes for a cache hit. Two threads of one
    process reproduce it exactly, since they share a pid.

    The payload is large enough that one `json.dump` is several writes, which
    is what gives the interleaving something to tear.
    """
    path = str(tmp_path / 'O_0123456789ab.json')
    payload = [{'element': 'O', 'rank': r,
                'radii': {'A1': [r + 1e-3 * i for i in range(4000)]}}
               for r in range(2)]
    barrier = threading.Barrier(2)

    def one_rank(comm):
        seen = []
        for _ in range(20):
            barrier.wait(timeout=60)
            separable_ri.write_json_atomic(path, payload[comm.Get_rank()])
            with open(path) as fh:
                seen.append(json.load(fh))
        return seen

    for seen in run_simulated(one_rank, 2):
        assert len(seen) == 20
        for got in seen:
            assert got in payload              # whole, and one writer's own
    with open(path) as fh:
        assert json.load(fh) in payload
    # a private name is only unique while it exists, so it has to be cleaned up
    assert not [p for p in os.listdir(tmp_path) if p.endswith('.tmp')]


def _rank_inputs(w, comm, shift=1e-3, scale=1.0 + 1e-6, mean_field=True,
                 factors=True):
    """This rank's own mean field and factors; rank 1's are PERTURBED.

    The perturbation is enormous next to the last-bit drift it stands for --
    1 mHa on the virtuals and 1e-6 on the orbitals, 1e-6 on the collocation --
    so nothing that reads rank 1's data or follows rank 1's decisions can come
    back with rank 0's answer by accident. `mean_field` and `factors` choose
    which of the two rank 1 gets wrong.
    """
    mf = copy.copy(w['mf'])
    mf.mo_energy = np.asarray(w['mf'].mo_energy, float).copy()
    mf.mo_coeff = np.asarray(w['mf'].mo_coeff, float).copy()
    out = tuple(np.array(a, copy=True) for a in w['factors'])
    if comm.Get_rank() != 0:
        if mean_field:
            mf.mo_energy[w['nocc']:] += shift
            mf.mo_coeff *= scale
        if factors:
            out[0][...] *= scale
    return mf, out


@pytest.mark.parametrize('size', SIZES)
def test_bse_is_rank_zeros_with_perturbed_ranks(water, size):
    """The whole ISDF BSE returns RANK 0's roots when the other ranks arrive
    with different numbers -- and every rank takes rank 0's iteration.

    Two defects at once, both measured across two nodes at this very
    setting. (1) The ranks' data: rank 1's spectrum and factors are replaced
    by rank 0's, so the reduced exchange partials belong to one calculation.
    (2) The ranks' decisions: every rank runs the Davidson, on trial vectors
    the block action locksteps at entry and an action output identical on
    every rank, so every rank takes rank 0's decisions and the batch shapes
    cannot diverge -- under real ranks the divergence showed as
    MPI_ERR_TRUNCATE in the batch reduction and as a subspace Cholesky
    failing on one rank alone. The vind-call and iteration counts are
    therefore rank 0's on every rank, and the roots, locked to rank 0's on
    the way out, are rank 0's bits.

    Not bitwise against the serial call, and cannot be: the row split
    re-associates the sum over grid points inside the block action. 1e-10 is
    the last bits of that sum; the perturbation it must not admit is 1e-3.

    The (A-B) probe runs here, replicated as a Lanczos
    (`davidson._lanczos_lowest`): scipy's eigsh holds one process-wide lock
    across its whole iteration, so rank threads would deadlock inside it.
    """
    w = water
    om0, _, _, info0 = solve_bse_isdf(w['mf'], w['mol'], w['nocc'], nroots=3,
                                      probe=True, progress=False,
                                      factors=w['factors'])

    def one_rank(comm):
        mf, factors = _rank_inputs(w, comm)
        om, _, _, info = solve_bse_isdf(mf, w['mol'], w['nocc'], nroots=3,
                                        probe=True, progress=False,
                                        factors=factors, distribute=True,
                                        comm=comm)
        return om, info, np.abs(factors[0] - w['factors'][0]).max()

    out = run_simulated(one_rank, size)
    om_r0, info_r0 = out[0][0], out[0][1]
    assert info_r0['stats']['davidson_vind_calls'] > 0
    for om, info, dX in out:
        assert np.abs(om - om0).max() < 1e-10
        assert abs(info['min_eig_amb'] - info0['min_eig_amb']) < 1e-10
        assert dX == 0.0                       # the perturbation was overwritten
        assert np.asarray(om).tobytes() == np.asarray(om_r0).tobytes()
        assert (info['stats']['davidson_vind_calls']
                == info_r0['stats']['davidson_vind_calls'])
        assert (info['timings']['davidson_iterations']
                == info_r0['timings']['davidson_iterations'])


@pytest.mark.parametrize('size', SIZES)
def test_qp_window_is_rank_zeros_with_perturbed_mean_field(water, size):
    """The distributed GW window returns rank 0's quasiparticle energies when
    rank 1 arrives with another mean field.

    BITWISE, unlike the BSE: the mean-field arrays are a lockstep of rank 0's
    before the grid is sized from them, the reduction hands every rank the
    same chi0 and the same Sigma, and the Pade continuation and Newton that
    follow end in a lockstep of rank 0's roots. That matters downstream --
    these energies are the BSE diagonal, and a window that drifts between
    ranks is a kernel that drifts with it.
    """
    w = water
    window = np.array([w['nocc'] - 1, w['nocc']])
    ref = solve_qp_energy_space_time(w['mf'], w['mol'], w['nocc'], window,
                                     factors=w['factors'])

    def one_rank(comm):
        mf, factors = _rank_inputs(w, comm, factors=False)
        qp = solve_qp_energy_space_time(mf, w['mol'], w['nocc'], window,
                                        factors=factors, distribute=True,
                                        comm=comm)
        return qp, mf.mo_energy, mf.mo_coeff

    for qp, mo_energy, mo_coeff in run_simulated(one_rank, size):
        assert np.array_equal(qp, ref)
        assert np.array_equal(mo_energy, w['mf'].mo_energy)   # repaired
        assert np.array_equal(mo_coeff, w['mf'].mo_coeff)


@pytest.mark.parametrize('size', SIZES)
def test_qp_window_reports_perturbed_factors_under_audit(water, size):
    """Factors are identical by construction, so the window does not broadcast
    them again; an audited run SAYS when they are not.

    Rank 1's collocation is 1e-6 off. The window leaves it so -- it moves no
    factor bytes -- and every rank's audit counts the disagreement at the
    window's entry, where a production run would not look. The ranks still
    return one set of roots, rank 0's, since the window ends in a lockstep of
    them; that they are not the serial roots is the drift the audit reported.
    With the same inputs on every rank the audit finds nothing, which is what
    a cluster run shows to prove its kernels' inputs held one set of bits.
    """
    w = water
    window = np.array([w['nocc'] - 1, w['nocc']])

    def one_rank(comm, perturbed):
        mf, factors = _rank_inputs(w, comm, mean_field=False,
                                   factors=perturbed)
        with distributed(comm, audit=True):
            lockstep_stats(reset=True)
            qp = solve_qp_energy_space_time(mf, w['mol'], w['nocc'], window,
                                            factors=factors)
            stats = lockstep_stats()
        return qp, stats, np.abs(factors[0] - w['factors'][0]).max()

    for perturbed in (False, True):
        out = run_simulated(one_rank, size, perturbed)
        for r, (qp, stats, dX) in enumerate(out):
            assert np.array_equal(qp, out[0][0])
            assert stats['agreement_calls'] > 0
            if perturbed:
                assert stats['disagreements'] >= 1
                assert (stats['first_disagreement']
                        == 'solve_qp_energy_space_time inputs')
                assert (dX > 0.0) == (r != 0)          # left as it arrived
            else:
                assert stats['disagreements'] == 0
                assert stats['first_disagreement'] is None


# ------------------------------------------- the transform's pseudo-inverse
def _weights(kind, grid, e_min, e_max, rcond=None):
    """The transform, optionally with the pseudo-inverse cutoff DISABLED.

    A negative rcond keeps every singular value, which is the expression this
    routine carried before the cutoff existed -- so the two calls are the
    before and after of the change, on the same grid, in the same process.
    """
    saved = time_frequency.TRANSFORM_FIT_RCOND
    try:
        if rcond is not None:
            time_frequency.TRANSFORM_FIT_RCOND = rcond
        return minimax_transform_weights(kind, grid.tau_points,
                                         grid.omega_points, e_min, e_max)
    finally:
        time_frequency.TRANSFORM_FIT_RCOND = saved


def test_transform_pseudo_inverse(water):
    """A zero singular value contributes zero; everything else is untouched.

    The fit is a per-point least squares through an SVD, and below 20 points
    it carries no Tikhonov term: the filter is 1/S. A minimax tau point large
    enough that exp(-x tau) underflows over the whole node range leaves an
    exactly zero COLUMN in the design matrix, hence an exactly zero singular
    value, hence 0/0 -- a NaN row in the transform. It is silent twice over:
    the fit error of a NaN row is NaN, and `max(x, nan)` returns x, so the
    fit-error warning never sees it. That is the warning the cluster raised.

    BITWISE, and that is the point. The smallest relative singular value the
    grids here reach is 2.4e-17, and the cutoff sits at 1e-100: small values
    are still inverted, exactly as GreenX inverts them -- cutting at the
    SVD's backward error instead would have moved this grid's omega -> tau
    weights in the sixth digit, which is a change to the physics and not a
    NaN fix.
    """
    w = water
    grid = w['grid']
    e_min = 0.5 * (w['eps'][w['nocc']] - w['eps'][w['nocc'] - 1])
    e_max = w['eps'][-1] - w['eps'][0]
    for kind in (COSINE_TW, COSINE_WT):
        cut, err_cut = _weights(kind, grid, e_min, e_max)
        uncut, err_uncut = _weights(kind, grid, e_min, e_max, rcond=-1.0)
        assert np.array_equal(cut, uncut)
        assert err_cut == err_uncut

    # The two ways 1/S breaks, in the arithmetic itself, and why it is silent.
    with np.errstate(invalid='ignore', divide='ignore'):
        assert np.isnan(np.float64(0.0) / np.float64(0.0) ** 2)
        assert np.isinf(np.float64(5e-296) / np.float64(5e-296) ** 2)
    assert max(0.0, float('nan')) == 0.0       # a NaN never wins the max

    # A tau point so large that exp(-x tau) underflows for every node leaves
    # an exactly zero COLUMN, and a singular value whose square is zero.
    tau = np.array([0.1, 1.0, 10.0, 2.0e3])
    omega = np.array([0.5, 1.0, 2.0, 4.0])
    x = e_min * (e_max / e_min) ** (np.arange(400) / 399.0)
    psi, A = time_frequency._psi_and_matrix(COSINE_TW, tau, omega, 0, x)
    assert np.linalg.norm(A[:, -1]) == 0.0
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    assert S[-1] ** 2 == 0.0 and S[-1] < TRANSFORM_FIT_RCOND * S[0]
    with np.errstate(invalid='ignore', divide='ignore'):
        old = Vt.T @ ((S / S**2) * (U.T @ psi))
    assert not np.isfinite(old).any()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        W, err = minimax_transform_weights(COSINE_TW, tau, omega, e_min, e_max)
    assert np.isfinite(W).all()
    assert err > 0.0 and np.isfinite(err)      # the error is seen, and warned
    assert any(issubclass(c.category, RuntimeWarning) for c in caught)
    # Without the cutoff the same call is refused rather than returning NaN.
    saved = time_frequency.TRANSFORM_FIT_RCOND
    try:
        time_frequency.TRANSFORM_FIT_RCOND = -1.0
        with pytest.raises(FloatingPointError):
            minimax_transform_weights(COSINE_TW, tau, omega, e_min, e_max)
    finally:
        time_frequency.TRANSFORM_FIT_RCOND = saved


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
