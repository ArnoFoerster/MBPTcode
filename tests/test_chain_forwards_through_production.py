"""The gradient chains evaluate PRODUCTION's forward pass, not a copy of it.

Every adjoint in `src/gradients` differentiates some forward quantity, and the
only way an adjoint can be trusted is if the forward it reverses is the one the
energy routes evaluate. This file gates that seam by seam, to a few ulp
(`assert_same` of tests/pins.py: the test evaluates the production side again)
wherever the two sides are one implementation and a measured tolerance -- stated beside
the number -- wherever they are two constructions of one quantity:

  (a) E_c^dRPA: `rpa_energy_and_adjoint` calls
      `rpa_correlation_energy_space_time`, whose `want_tape=True` hands back
      proj(tau) and the partitions it was built on.
  (b) the BSE diagonal: the matrix-free route, the dense route and the
      gradient chain's blocks all place the quasiparticle energies through
      `block_actions.bse_pair_diagonal`.
  (c) the quasiparticle forward: `qp_gradient_space_time(want_grad=False)` is
      `solve_qp_energy_contour` on the same factors, on all three
      continuations, and the chain's contour grid is `cd_frequency_grid`.
  (d) Eq. (18): both constructions of Delta W contract through
      `reaction_field.projected_quasiparticle_shift`.

What breaks each gate:

  * the tape is the forward sweep: proj(tau) in the tape scaled by 1 + 1e-9
    gives tape != sweep, 3.4e-10 on a 0.339 peak, while E_c is untouched.
  * one diagonal for both BSE routes: davidson's diag_d + 1e-6 gives
    diag_d != the rule.
  * the dense and matrix-free routes agree: the same shift of 1e-6 puts the
    roots 1.0e-06 apart.
  * the chain's contour grid is production's: `_build_cd_grid` at
    nfreq_cd + 2 gives two extra nodes.
  * the quasiparticle forward is production's: Bp scaled by 1 + 1e-9 in the
    gradient breaks all three continuations.
  * one Eq. (18) contraction: the gradient's shift scaled by 1.000001 breaks
    the `==` and the 1.0e-06 relative check.

A gate with no such break is one where the two sides are literally the same
call, and the note on it says so: perturbing production moves both sides
together, which is what having one implementation means.
"""

import numpy as np
import pytest
from pyscf import df as pyscf_df, dft, gto, scf

from src.Base.numerics.constants import (CD_NTAU, HARTREE_TO_EV, ISDF_TILE_GB, KAPPA,
                                         SCF_DIFFERENTIABLE_GRAD_TOL,
                                         SCF_TIGHT_CONV_TOL)
from src.Base.environment_seam import attach_environment
from src.Base.numerics.numerics import Numerics
from src.environments.solvent_screening import SolventScreening, detach_solvent_screening
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.GW.contour_deformation import (cd_frequency_grid,
                                                        cd_grid_range,
                                                        solve_qp_energy_contour)
from src.SingleReference.GW.qp_solve import static_exchange_diagonal
from src.SingleReference.GW.reaction_field import (
    bare_gauge_transform, projected_quasiparticle_shift,
    separable_quasiparticle_shift)
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.BSE.solve import ISDF_KWARGS, solve_bse
from src.SingleReference.LinearResponse.block_actions import (block_action,
                                                              bse_pair_diagonal)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.space_time import (
    RPAEnergyTape, polarizability_projected_sweep,
    rpa_correlation_energy_space_time)
from src.gradients.gwbse.bse_isdf import bse_blocks
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.qp_space_time import qp_gradient_space_time
from src.gradients.gwbse.reaction_field_adjoint import (reaction_field_shift,
                                                        static_grid, static_screening)
from src.gradients.rpa.ground_state import RPAGroundStateChain
from src.gradients.rpa.space_time_adjoint import rpa_energy_and_adjoint
from tests.isdf_test_grid import g1, g1_numerics
from tests.pins import assert_same

BASIS, AUXBASIS = 'cc-pvdz', 'cc-pvdz-ri'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
SMALL, SMALL_AUX = 'cc-pvdz', 'cc-pvdz-ri'
#: Largest difference between the dense and the matrix-free BSE roots on one
#: quasiparticle diagonal, in Hartree. The two build the same operator from
#: different objects -- the dense route from the (naux, norb, norb) factor the
#: fit implies, the matrix-free one from (X, D) -- and the Davidson stops on a
#: 1e-5 residual, so the agreement is the eigensolver's, not the arithmetic's.
#: Measured 1.2e-08 on water/cc-pVDZ, three roots.
DENSE_DAVIDSON_TOL = 1e-7
#: What the two Delta W constructions may differ by, relative: one chi0 and a
#: congruence against two factorizations screened once each. Measured 3.5e-14
#: on water/sto-3g with a cc-pVDZ-ri auxiliary basis, both fed the same dressed
#: screening, so what is left is the bare partner alone.
DELTA_W_REL = 1e-11


def _scf(mol, basis=BASIS):
    mf = scf.RHF(mol).density_fit(auxbasis=basis + '-ri')
    mf.conv_tol = SCF_TIGHT_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def water():
    """One mean field and one factorization for every gate that needs them."""
    mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
    mf = _scf(mol)
    nocc = mol.nelectron // 2
    x_mo, d = separable_factors(mf, mol, auxbasis=AUXBASIS, counts=g1(mol))[:2]
    eps = np.asarray(mf.mo_energy, float)
    return dict(mol=mol, mf=mf, nocc=nocc, x_mo=x_mo, d=d, eps=eps,
                mu=0.5 * (eps[nocc - 1] + eps[nocc]))


@pytest.fixture(scope='module')
def solvated():
    """The two factorizations of one solvated molecule: X is shared, D is not."""
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis=SMALL, verbose=0)
    nocc = mol.nelectron // 2
    mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis=SMALL_AUX)
    mf.conv_tol = 1e-11
    mf.kernel()
    env = SolventScreening(mol, eps=1.78)
    detach_solvent_screening(mf)
    x_mo, d_bare = separable_factors(mf, mol, auxbasis=SMALL_AUX, counts=g1(mol))[:2]
    attach_environment(mf, env)
    _, d_dressed = separable_factors(mf, mol, auxbasis=SMALL_AUX, counts=g1(mol))[:2]
    detach_solvent_screening(mf)
    eps = np.asarray(mf.mo_energy, float)
    return dict(mol=mol, mf=mf, env=env, nocc=nocc, eps=eps, x_mo=x_mo,
                d_dressed=d_dressed, d_bare=d_bare,
                grid=static_grid(eps, nocc))


# ----------------------------------------------------------------- seam (a)
def test_the_chains_correlation_energy_is_the_production_routine(water):
    """E_c^dRPA of the chain is `rpa_correlation_energy_space_time`'s, to a
    few ulp (`assert_same`: the test computes it again).

    NO PERTURBATION EXISTS for this gate, and that is the point: the chain's
    forward pass CALLS the production routine, so any change to the energy
    moves both sides together. What a perturbation can still break is the tape
    the adjoint reads, which the next test gates.
    """
    mol, mf = water['mol'], water['mf']
    chain = RPAGroundStateChain(mol, _scf, mf=mf, numerics=Numerics(isdf={'counts': g1(mol)}))
    crd = chain.coords(mol)
    x_ao, d = chain.factors(mol, chain.auxmol(mol), crd)
    e_prod = rpa_correlation_energy_space_time(x_ao @ mf.mo_coeff, d,
                                               np.asarray(mf.mo_energy, float),
                                               chain.nocc, chain.grid)
    assert_same(chain.correlation_energy(), e_prod, label='chain E_c')
    # and the adjoint's own forward half is that same number
    assert_same(rpa_energy_and_adjoint(x_ao @ mf.mo_coeff, d,
                                       np.asarray(mf.mo_energy, float),
                                       chain.nocc, chain.grid,
                                       want_grad=False),
                e_prod, label='adjoint forward')
    assert_same(rpa_energy_and_adjoint(x_ao @ mf.mo_coeff, d,
                                       np.asarray(mf.mo_energy, float),
                                       chain.nocc, chain.grid,
                                       want_grad=True)[0],
                e_prod, label='adjoint forward with gradient')


def test_the_tape_is_the_forward_sweep_itself(water):
    """`want_tape=True` hands back proj(tau) as the sweep built it (to a few
    ulp: the test runs the sweep again).

    Breaks under `RPAEnergyTape(proj_tau * (1 + 1e-9), ...)` in
    `rpa_correlation_energy_space_time`: the tape differs from the sweep by
    3.4e-10 on a peak of 0.339, while the energy assertion below it still
    holds -- which is what makes this the gate on the tape and not on
    the energy.
    """
    w = water
    grid = RPAGroundStateChain(w['mol'], _scf, mf=w['mf'], numerics=Numerics(isdf={'counts': g1(w['mol'])})).grid
    e_c, tape = rpa_correlation_energy_space_time(
        w['x_mo'], w['d'], w['eps'], w['nocc'], grid, want_tape=True)
    assert isinstance(tape, RPAEnergyTape)
    assert tape.tau_indices is None and tape.freq_indices is None
    sweep = polarizability_projected_sweep(w['x_mo'], w['d'], w['eps'],
                                           w['nocc'], grid.tau_points,
                                           tile_memory_gb=ISDF_TILE_GB)
    assert_same(tape.proj_tau, sweep, unit='1', quantity='root',
                label='proj(tau)')
    assert_same(e_c, rpa_correlation_energy_space_time(
        w['x_mo'], w['d'], w['eps'], w['nocc'], grid), label='E_c')


def test_want_tape_false_returns_what_it_always_returned(water):
    """A bare float, the tape call's energy (to a few ulp: a second call),
    and no tuple in sight."""
    w = water
    grid = RPAGroundStateChain(w['mol'], _scf, mf=w['mf'], numerics=Numerics(isdf={'counts': g1(w['mol'])})).grid
    plain = rpa_correlation_energy_space_time(w['x_mo'], w['d'], w['eps'],
                                              w['nocc'], grid)
    assert isinstance(plain, float)
    assert_same(plain, rpa_correlation_energy_space_time(
        w['x_mo'], w['d'], w['eps'], w['nocc'], grid, want_tape=True)[0],
        label='E_c with the tape')


# ----------------------------------------------------------------- seam (b)
def test_both_bse_routes_place_the_quasiparticle_diagonal_by_one_rule(water):
    """The matrix-free action and the gradient chain's blocks read
    `bse_pair_diagonal` of the quasiparticle spectrum (to a few ulp: the
    test evaluates it again).

    The dense route reaches the same diagonal by the difference of two calls to
    the same routine, and its roots are gated against the matrix-free ones
    below at DENSE_DAVIDSON_TOL.

    Breaks under `diag_d = bse_pair_diagonal(...) + 1e-6` in
    `isdf_block_action`: the action's diagonal differs from the rule.
    """
    w = water
    eps_qp = w['eps'] + 0.05 * np.arange(len(w['eps']))   # not the mean field
    lr = LinearResponseSolver(eps_qp, spin_mode='restricted')
    naux = w['d'].shape[1]
    w_aux = np.eye(naux)
    _, diag_d = block_action(lr, w['nocc'], 'BSE', w_aux, (w['x_mo'], w['d']))
    assert_same(diag_d, bse_pair_diagonal(eps_qp, w['nocc']), quantity='root',
                label='action diagonal')

    # kappa = 0 and no screening leave the chain's A as the diagonal alone
    a, _, _ = bse_blocks(w['x_mo'], w['d'], eps_qp, np.zeros((naux, naux)),
                         w['nocc'], spin='triplet', bse_tda=True)
    assert KAPPA['triplet'] == 0.0
    assert_same(np.diag(a), bse_pair_diagonal(eps_qp, w['nocc']).ravel(),
                quantity='root', label='chain diagonal')


def test_the_dense_and_matrix_free_routes_agree_on_one_diagonal(water):
    """One quasiparticle array down both routes of `solve_bse`.

    The eigensolvers differ, not the operator: measured 1.2e-08 Ha on the three
    lowest roots of water/cc-pVDZ, against a Davidson residual tolerance of
    1e-5.

    Breaks under the same `+ 1e-6` as the gate above: the routes move
    1.01e-06 Ha apart.
    """
    w = water
    factors = (w['x_mo'], w['d'])
    qp = np.asarray(solve_bse(w['mf'], w['mol'], w['nocc'], solver='davidson',
                              integrals='isdf', numerics=Numerics(bse={'nroots': 3}), probe=False,
                              factors=factors)[3]['eps'], float)
    roots = {}
    for solver in ('davidson', 'dense'):
        roots[solver] = solve_bse(w['mf'], w['mol'], w['nocc'], solver=solver,
                                  integrals='isdf', numerics=Numerics(bse={'nroots': 3}), probe=False,
                                  qp=qp, factors=factors)[0]
    worst = np.abs(roots['davidson'] - roots['dense']).max()
    assert worst < DENSE_DAVIDSON_TOL, worst


def test_the_front_end_forwards_the_distributed_keywords(water):
    """`solve_bse(distribute=True, comm=)` reaches `solve_bse_isdf`.

    The two keywords are the matrix-free ISDF route's alone and the front end
    accepts them there, so a distributed BSE need not bypass `solve_bse`. Under
    simulated ranks every rank returns rank 0's roots.
    """
    assert {'distribute', 'comm', 'grid_accuracy'} <= set(ISDF_KWARGS)
    w = water
    serial = solve_bse(w['mf'], w['mol'], w['nocc'], solver='davidson',
                       integrals='isdf', numerics=Numerics(bse={'nroots': 3}), probe=False,
                       factors=(w['x_mo'], w['d']))[0]

    def one_rank(comm):
        om, _, _, info = solve_bse(w['mf'], w['mol'], w['nocc'],
                                   solver='davidson', integrals='isdf',
                                   numerics=Numerics(bse={'nroots': 3}), probe=False,
                                   factors=(w['x_mo'], w['d']),
                                   distribute=True, comm=comm)
        return om, info['nranks']

    for om, nranks in run_simulated(one_rank, 2):
        assert nranks == 2
        assert np.abs(om - serial).max() < 1e-9

    with pytest.raises(TypeError, match='distribute'):
        solve_bse(w['mf'], w['mol'], w['nocc'], solver='dense',
                  integrals='isdf', numerics=Numerics(bse={'nroots': 3}), probe=False, distribute=True)


# ----------------------------------------------------------------- seam (c)
@pytest.mark.parametrize('continuation,route',
                         [('cd', 'explicit'), ('laplace', 'laplace'),
                          ('sop', 'sop')])
def test_the_quasiparticle_forward_half_is_the_production_contour(
        water, continuation, route):
    """`qp_gradient_space_time(want_grad=False)` is `solve_qp_energy_contour`.

    The gradient's forward half is assembled entirely out of production
    objects -- the proj(tau) sweep, the three-index slice, the screening
    contraction, the real-frequency backends and the Newton itself -- so the
    energy and the pole strength agree on all three continuations, and no
    second forward exists to drift. They are two solves of one root (each
    with its own static exchange build), so they agree to a few ulp, not
    bitwise.

    Breaks under `Bp = three_index_slice(...) * (1 + 1e-9)` in
    `qp_gradient_space_time`, on all three continuations.
    """
    w = water
    p = w['nocc'] - 1
    factors = (w['x_mo'], w['d'])
    energies, pole_strengths, _ = solve_qp_energy_contour(
        w['mf'], w['mol'], w['nocc'], [p], continuation=continuation,
        factors=factors)
    nu, wt, grid, _ = cd_frequency_grid(w['eps'], w['nocc'], ntau=CD_NTAU)
    xc = float(static_exchange_diagonal(w['mf'], w['mol'], [p])[0])
    w_star, z = qp_gradient_space_time(w['x_mo'], w['d'], w['eps'], w['nocc'],
                                       grid, nu, wt, p, mu=w['mu'],
                                       xc_correction=xc, residue_route=route,
                                       want_grad=False)
    assert_same(w_star * HARTREE_TO_EV, energies[0], unit='eV',
                quantity='energy', label='root')
    assert_same(z, pole_strengths[0], unit='1', quantity='energy',
                label='pole strength')


def test_the_chains_contour_grid_is_the_production_one(water):
    """The surface's quadrature IS `cd_frequency_grid`, and the range it was
    fitted on IS `cd_grid_range`.

    Breaks under `_build_cd_grid` calling `cd_frequency_grid` at
    `nfreq_cd + 2`: the chain's quadrature carries two nodes production's does
    not.
    """
    w = water
    chain = ExcitedStateChain(w['mol'], _scf, mf=w['mf'], numerics=g1_numerics(w['mol']))
    nu, wt, grid, w0 = cd_frequency_grid(w['eps'], w['nocc'],
                                         ntau=chain.ntau_gw,
                                         nfreq_cd=chain.conventions.nfreq_cd)
    # the test builds the grid again: the chain's to a few ulp
    assert_same(nu, chain.conventions.nu, quantity='root', label='nu')
    assert_same(wt, chain.conventions.wt, quantity='root', label='wt')
    assert_same(w0, chain.conventions.w0_cd, quantity='root', label='w0')
    assert_same(grid.tau_points, chain.conventions.gw_grid.tau_points,
                unit='1', quantity='root', label='tau points')
    assert_same(grid.cosft_wt, chain.conventions.gw_grid.cosft_wt, unit='1',
                quantity='root', label='cosine weights')
    assert_same(cd_grid_range(w['eps'], w['nocc']),
                (chain.conventions.e_min_gw, chain.conventions.e_max),
                quantity='root', label='contour range')


# ----------------------------------------------------------------- seam (d)
def test_one_contraction_carries_both_delta_w_constructions(solvated):
    """Eq. (18) is `projected_quasiparticle_shift` on either route to Delta W.

    Fed the SAME projected densities and screenings the two are the same call,
    so they agree to a few ulp (the test calls it again); fed their own --
    one chi0 and a congruence against two
    factorizations screened once each -- they agree to DELTA_W_REL, which is
    the ISDF reproducibility floor and not a difference of formula.

    Breaks under the gradient's `reaction_field_shift` scaled by 1.000001: the
    few-ulp check above and the 1.0e-06 relative check below.
    """
    s = solvated
    pair = (static_screening(s['x_mo'], s['d_dressed'], s['eps'], s['nocc'],
                             s['grid']),
            static_screening(s['x_mo'], s['d_bare'], s['eps'], s['nocc'],
                             s['grid']))
    (a_d, w_d), (a_b, w_b) = pair
    two = reaction_field_shift(s['x_mo'], s['d_dressed'], s['d_bare'], s['eps'],
                               s['nocc'], grid=s['grid'], screening=pair)
    assert_same(two, projected_quasiparticle_shift(a_d, w_d, a_b, w_b,
                                                   s['nocc']),
                label='Delta W, two factorizations')

    auxmol = pyscf_df.addons.make_auxmol(s['mol'], auxbasis=SMALL_AUX)
    t = bare_gauge_transform(auxmol, s['env'])
    one = separable_quasiparticle_shift(s['x_mo'], s['d_dressed'], w_d, t,
                                        s['nocc'])
    assert_same(
        one, projected_quasiparticle_shift(a_d, w_d, t.T @ a_d,
                                           np.linalg.inv(np.eye(w_d.shape[0])
                                                         - t.T @ (np.eye(w_d.shape[0])
                                                                  - np.linalg.inv(w_d)) @ t),
                                           s['nocc']),
        label='Delta W, separable')
    rel = np.abs(two - one).max() / np.abs(one).max()
    assert rel < DELTA_W_REL, rel


# ----------------------------------------------------------------- seam (e)

