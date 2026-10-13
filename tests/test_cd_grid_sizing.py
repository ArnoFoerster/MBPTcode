"""How the quasiparticle set's contour-deformation grid is fixed and solved.

Water/cc-pVDZ (HF, DF) on the chain's own ISDF factors. Gated here:

- the grid is fixed once, before the first pass, and never resized against
  the roots: `CD_NFREQ_SOP` points for a set the pole model carries whole
  (its grid only feeds the fits), `CD_NFREQ` for a set with a state on the
  quadrature routes, an asked `nfreq_cd` a floor under both. A root beside
  an orbital energy (a neighbour's or its own) needs no finer grid, because
  the Lorentzian of that pole of G is integrated in closed form
  (`contour_deformation.cd_integral_weights`).
- [1 - chi0(i.nu)] is factorized by Cholesky (`screening_solver`), which
  matches LU to rounding and falls back to LU, with a warning, on a matrix
  that is not positive definite.
- a root rejected as no quasiparticle is demoted and the set re-solved on
  the same grid. The quadrature Newton is scripted there, so which root is
  where is fixed by the test.
"""
import os
import sys
import warnings

import numpy as np
import pytest
import scipy.linalg
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import (CD_NFREQ, CD_NFREQ_SOP, ISDF_TILE_GB,
                                SCF_DIFFERENTIABLE_CONV_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import QPStates
from src.SingleReference.GW.contour_deformation import (
    cd_screening_contraction_multi, screening_solver)
from src.SingleReference.GW.qp_states import resolve_qp_states
from src.SingleReference.LinearResponse.space_time import (
    owned_frequency_blocks, polarizability_projected_rows, three_index_slice)
import src.gradients.excited_state as excited_state
import src.gradients.qp_space_time as qp_space_time
from src.gradients.excited_state import ExcitedStateChain

#: Cholesky against LU on the same chi0(i.nu): both backward stable on a
#: matrix with eigenvalues in [1, ~2], so wc agrees to rounding, relative to
#: its largest entry. Measured 2.0e-15 on this case (naux 84).
CHOLESKY_VS_LU_REL = 1e-12
#: The scripted quadrature set: three valence orbitals of water.
STATES = [2, 3, 4]


def factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.kernel()
    return mf


@pytest.fixture(scope='module')
def water():
    warnings.simplefilter('ignore')
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='cc-pvdz', verbose=0)
    return mol, factory(mol)


def chain_on(water, **kw):
    """(chain, the arguments of a forward set solve on its factors)."""
    mol, mf = water
    chain = ExcitedStateChain(mol, factory, mf=mf, **kw)
    x_mo, d, eps, mu = chain._factors_for(mol, mf)[:4]
    return chain, (x_mo, d, eps, mu, 0.0, np.zeros(len(chain.qp_set)))


def recorded_solves(monkeypatch):
    """[nfreq] per `qp_set_gradient` call of the chain."""
    calls, real = [], excited_state.qp_set_gradient

    def solve(*a, **k):
        calls.append(len(a[5]))
        return real(*a, **k)
    monkeypatch.setattr(excited_state, 'qp_set_gradient', solve)
    return calls


# ------------------------------------------------------------------ Cholesky
def test_cholesky_matches_lu_on_the_frequency_pass(water, monkeypatch):
    """wc from the Cholesky pass against the same contraction by LU, block
    by block on the same chi0(i.nu); and no LU is taken on the way."""
    chain, (x_mo, d, eps, mu, _, _) = chain_on(water, qp_window=STATES)
    grid = chain.gw_grid
    proj = polarizability_projected_rows(x_mo, d, eps, chain.nocc,
                                         grid.tau_points, mu=mu)
    Bps = [three_index_slice(x_mo, d, p) for p in STATES]
    naux = d.shape[1]
    ref = [np.zeros((len(chain.nu), B.shape[1])) for B in Bps]
    for ks, blk in owned_frequency_blocks(proj, grid.cosft_wt, ISDF_TILE_GB):
        for m, k in enumerate(ks):
            lu = scipy.linalg.lu_factor(np.eye(naux) - blk[m])
            for B, wc in zip(Bps, ref):
                wc[k] = np.einsum('Pq,Pq->q', B,
                                  scipy.linalg.lu_solve(lu, B) - B)

    def refused(*a, **k):
        raise AssertionError('an LU in the frequency pass')
    monkeypatch.setattr(scipy.linalg, 'lu_factor', refused)
    got = cd_screening_contraction_multi(proj, grid.cosft_wt, Bps)
    for g, r in zip(got, ref):
        assert np.abs(g - r).max() <= CHOLESKY_VS_LU_REL * np.abs(r).max()


def test_a_matrix_that_is_not_positive_definite_falls_back_to_lu():
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    a = q @ np.diag(np.linspace(-0.5, 2.0, 12)) @ q.T
    b = rng.standard_normal((12, 3))
    with pytest.warns(RuntimeWarning, match='not positive definite'):
        solve = screening_solver(a)
    assert np.allclose(solve(b), np.linalg.solve(a, b), rtol=0, atol=1e-12)
    spd = q @ np.diag(np.linspace(1.0, 2.0, 12)) @ q.T
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert np.allclose(screening_solver(spd)(b), np.linalg.solve(spd, b),
                           rtol=0, atol=1e-12)


# ------------------------------------------------------ the pole-model grid
def test_a_pole_model_set_takes_the_fixed_grid_once(water, monkeypatch):
    mol, mf = water
    eps = np.asarray(mf.mo_energy, float)
    admitted = list(resolve_qp_states(QPStates(), eps, mol.nelectron // 2,
                                      degeneracy_tol=1e-6).explicit)
    chain, args = chain_on(water, qp_window=admitted, residue_route='sop')
    assert chain.nfreq_cd == CD_NFREQ
    calls = recorded_solves(monkeypatch)
    chain._qp_set_solve(*args)
    assert set(chain.qp_diagnostics['routes'].values()) == {'sop'}
    # the freezing solve and its repeat from the roots it froze, one grid
    assert calls == [CD_NFREQ_SOP] * 2, calls
    assert chain.nfreq_cd == CD_NFREQ_SOP and chain.cd_sized


def test_an_asked_grid_is_a_floor_under_the_pole_model_grid(water,
                                                            monkeypatch):
    """An `nfreq_cd` asked for above `CD_NFREQ_SOP` is kept (a refrozen chain
    passes its sized grid), and one below it is raised to it."""
    mol, mf = water
    eps = np.asarray(mf.mo_energy, float)
    admitted = list(resolve_qp_states(QPStates(), eps, mol.nelectron // 2,
                                      degeneracy_tol=1e-6).explicit)
    for asked, solved in ((4 * CD_NFREQ_SOP, 4 * CD_NFREQ_SOP),
                          (CD_NFREQ_SOP // 2, CD_NFREQ_SOP)):
        chain, args = chain_on(water, qp_window=admitted, residue_route='sop',
                               nfreq_cd=asked)
        calls = recorded_solves(monkeypatch)
        chain._qp_set_solve(*args)
        assert calls == [solved] * 2, (asked, calls)


def test_a_set_on_the_quadrature_takes_the_plain_grid_once(water,
                                                           monkeypatch):
    """The O 1s is outside Eq. (27), so it falls back to a residue route: the
    set is solved on `CD_NFREQ` points, which it keeps, and solved again
    there from the roots it froze."""
    chain, args = chain_on(water, qp_window=[0, 3, 4], residue_route='sop')
    calls = recorded_solves(monkeypatch)
    chain._qp_set_solve(*args)
    assert chain.qp_diagnostics['routes'][0] not in ('sop', 'scissor')
    assert calls == [CD_NFREQ] * 2, calls
    assert chain.nfreq_cd == CD_NFREQ and chain.cd_sized


# ------------------------------------------ one grid, one verdict per root
def scripted(monkeypatch, chain, script):
    """Replace the quadrature Newton by `script[p]` = (root, Z); returns the
    [(p, nfreq)] it was asked for, in order."""
    asked = []

    def newton(p, Bp, eps, nocc, nu_points, *a, **k):
        asked.append((int(p), len(nu_points)))
        w, z = script[int(p)]
        return w, z, []
    monkeypatch.setattr(qp_space_time, 'qp_energy_cd', newton)
    return asked


def far(eps, p):
    """A root half-way to orbital p's upper neighbour."""
    return 0.5 * (eps[p] + eps[p + 1])


def near(eps, p):
    """A root 1e-7 Ha above eps_p, far inside the grid's smallest node."""
    return eps[p] + 1e-7


@pytest.fixture
def quadrature_set(water):
    chain, args = chain_on(water, qp_window=STATES, residue_route='explicit')
    return chain, args, args[2]


def test_a_root_beside_its_own_pole_is_solved_on_the_one_grid(quadrature_set,
                                                              monkeypatch):
    chain, args, eps = quadrature_set
    script = {p: (far(eps, p), 0.9) for p in STATES}
    script[2] = (near(eps, 2), 0.9)
    asked = scripted(monkeypatch, chain, script)
    chain._qp_set_solve(*args)
    assert asked == [(p, CD_NFREQ) for p in STATES] * 2
    assert chain.nfreq_cd == CD_NFREQ and chain.cd_sized


@pytest.mark.filterwarnings('ignore:quasiparticle roots with a pole strength')
def test_a_rejected_root_is_demoted_on_the_same_grid(quadrature_set,
                                                     monkeypatch):
    chain, args, eps = quadrature_set
    script = {p: (far(eps, p), 0.9) for p in STATES}
    script[2] = (far(eps, 2), -0.5)
    asked = scripted(monkeypatch, chain, script)
    chain._qp_set_solve(*args)
    assert list(chain.qp_set) == [3, 4] and list(chain.qp_demoted) == [2]
    assert chain.qp_demoted[2]['z'] == -0.5
    assert asked == ([(p, CD_NFREQ) for p in STATES]
                     + [(p, CD_NFREQ) for p in (3, 4)] * 2)
    assert chain.nfreq_cd == CD_NFREQ


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
