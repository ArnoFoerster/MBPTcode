"""The Casida/BSE Davidson's screened preconditioner, d - (ii|W|aa).

The Davidson divides its residuals by a diagonal: by default ('screened')
the pair energies d less the diagonal of the screened direct term, formed in
the fit's own gauge from the block action's factors,

    (ii|W|aa) = [D^T (X_o o X_o)]^T W [D^T (X_v o X_v)],

each fixed grid tile's fitted squares formed whole by one rank and the tiles
added in tile order (`ordered_chain_sum`), or on request ('bare') d alone.
Gated here:

  * THE DEFAULT IS SCREENED at every entry point, and the bare path, asked
    for explicitly, never forms the screened diagonal and times nothing for
    it;
  * THE DIAGONAL is the dense oracle's: water and benzene against the direct
    term built whole from `isdf_df_coefficients`, and against the triplet
    action's own diag(A), which carries no Hartree term; TDHF's bare-Coulomb
    diagonal likewise; the DF route's diagonal from C_oo and C_vv the ISDF
    one; RPA's d itself;
  * THE TILE ORDER: every rank's diagonal is the serial one bitwise at 2, 3,
    5 and 8 simulated ranks, whole and sliced factors; one tile's addend on
    rank 1 moved 1e-12 relative, or dropped, fails it;
  * THE ROOTS with it on lie within ROOT_TOL of the bare-preconditioned ones
    at 1e-5, every root converged, for the BSE singlet and triplet, TDHF and
    RPA (bitwise, d alone) on the ISDF action and the BSE singlet on the DF
    action, benzene BSE taking fewer block actions;
  * OVER RANKS with it on, every rank takes rank 0's decisions and cycle
    count on a split pair space at 2, 3 and 8 simulated ranks, its roots
    within the Davidson's resolution of the serial screened ones;
  * the switch reaches `solve_bse_isdf`, and an unknown value is refused at
    the entry.
"""
import inspect
import os
import sys
import warnings

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import (DAVIDSON_PRECONDITIONER,
                                DAVIDSON_PRECONDITIONERS)
from src.Base.sliced_factors import SlicedFactors
from src.Base.utils.mpi_grid import (contiguous_block, current_comm,
                                     run_simulated)
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse import davidson
from src.SingleReference.LinearResponse.davidson import (
    df_block_action, isdf_bse_factors, isdf_block_action,
    isdf_df_coefficients, solve_bse_df, solve_bse_isdf, solve_casida_davidson)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from tests.test_distributed_fit_mpi import relative, roots_resolution

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
BENZENE = ('C 1.396 0 0; C 0.698 1.209 0; C -0.698 1.209 0; C -1.396 0 0; '
           'C -0.698 -1.209 0; C 0.698 -1.209 0; H 2.479 0 0; '
           'H 1.240 2.147 0; H -1.240 2.147 0; H -2.479 0 0; '
           'H -1.240 -2.147 0; H 1.240 -2.147 0')
NROOTS = {'water': 3, 'benzene': 12}
CONV_TOL = 1e-5
#: Relative: the diagonal against the oracle's, both double precision.
ORACLE_TOL = 1e-13
#: Ha: the screened roots against the bare ones, both converged to CONV_TOL.
ROOT_TOL = 1e-10
#: Pairs of benzene's 1953 whose triplet diag(A) is read off unit vectors.
SAMPLED_PAIRS = 64
#: Grid rows per diagonal tile where a gate splits benzene's grid.
GATE_TILE = 64
#: Pair rows per trial-space tile where a gate splits benzene's pair space.
PAIR_TILE = 64
RANK_SIZES = [2, 3, 8]
#: Rank counts the diagonal is gated bitwise at, against one rank.
DIAGONAL_SIZES = [2, 3, 5, 8]
MODES = [('BSE', 'singlet'), ('BSE', 'triplet'), ('TDHF', 'singlet'),
         ('RPA', 'singlet')]


def _system(geometry):
    mol = gto.M(atom=geometry, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')
    W_aux = isdf_bse_factors(mf, mol, nocc, factors=factors)[2]
    return dict(mol=mol, mf=mf, nocc=nocc, factors=factors, W_aux=W_aux,
                eps=np.asarray(mf.mo_energy, float))


@pytest.fixture(scope='module')
def systems():
    warnings.simplefilter('ignore')
    return {'water': _system(WATER), 'benzene': _system(BENZENE)}


def lr_of(s):
    return LinearResponseSolver(s['eps'].copy(), spin_mode='restricted')


def direct_oracle(s, W):
    """(ii|W|aa) off the diagonal of the direct term built whole, (ij|W|ab)
    as an (n_occ^2, n_vir^2) matrix of the fit's DF-like coefficients."""
    nocc = s['nocc']
    C = isdf_df_coefficients(s['factors'][0], s['factors'][1])
    naux, nmo = C.shape[0], C.shape[1]
    nv = nmo - nocc
    C_oo = C[:, :nocc, :nocc].reshape(naux, -1)
    C_vv = C[:, nocc:, nocc:].reshape(naux, -1)
    Wd = C_oo.T @ (C_vv if W is None else W @ C_vv)
    Wd = Wd.reshape(nocc, nocc, nv, nv)
    i, a = np.arange(nocc), np.arange(nv)
    return Wd[i[:, None], i[:, None], a[None, :], a[None, :]], C


def triplet_diagonal(act, diag, pairs):
    """diag(A) of a triplet action on `pairs`, off unit vectors: d - (ii|W|aa)."""
    unit = np.zeros((len(pairs), diag.size))
    unit[np.arange(len(pairs)), pairs] = 1.0
    Az, _ = act(unit.reshape((len(pairs),) + diag.shape))
    return Az.reshape(len(pairs), -1)[np.arange(len(pairs)), pairs]


def solve(s, nroots, mode, spin, preconditioner, route='isdf'):
    """(omega, timings) of `solve_casida_davidson` on the ISDF or DF action."""
    t = {}
    W = s['W_aux'] if mode == 'BSE' else None
    if route == 'isdf':
        lr, factors = lr_of(s), s['factors']
    else:
        C = isdf_df_coefficients(s['factors'][0], s['factors'][1])
        lr = LinearResponseSolver(s['eps'].copy(), coeff_df=C,
                                  spin_mode='restricted')
        factors = None
    om, _, _ = solve_casida_davidson(lr, s['nocc'], nroots=nroots,
                                     polarizability=mode, W_aux=W,
                                     conv_tol=CONV_TOL, isdf_factors=factors,
                                     spin=spin, timings=t,
                                     refuse_unconverged=True,
                                     preconditioner=preconditioner)
    return om, t


def test_the_default_is_screened():
    """'screened' names the default, and every entry point takes it."""
    assert DAVIDSON_PRECONDITIONER == 'screened'
    assert set(DAVIDSON_PRECONDITIONERS) == {'bare', 'screened'}
    for fn in (solve_bse_isdf, solve_bse_df, solve_casida_davidson,
               davidson._run_davidson, davidson._davidson_root):
        default = inspect.signature(fn).parameters['preconditioner'].default
        assert default == DAVIDSON_PRECONDITIONER, fn.__name__


def test_the_bare_path_never_forms_the_screened_diagonal(systems):
    """Asked for 'bare' (the default is the screened diagonal), the action's
    screened diagonal is never called, the corrections read d, and nothing is
    timed for it."""
    s = systems['water']
    act, diag = isdf_block_action(lr_of(s), s['nocc'], True, s['W_aux'],
                                  s['factors'])
    ref = davidson._run_davidson(act, diag, 3, CONV_TOL, 100, None,
                                 preconditioner='bare')

    def refused():
        raise AssertionError('the bare path formed the screened diagonal')

    act.screened_diagonal = refused
    t = {}
    got = davidson._run_davidson(act, diag, 3, CONV_TOL, 100, None, timings=t,
                                 preconditioner='bare')
    for a, b in zip(ref, got):
        assert np.array_equal(a, b)
    assert t['davidson_precond_setup'] == 0.0


@pytest.mark.parametrize('name', ['water', 'benzene'])
def test_the_diagonal_is_the_dense_oracles(systems, name):
    """(ii|W|aa) of the ISDF action against the direct term built whole, and
    d - (ii|W|aa) against the triplet action's own diag(A); TDHF's with the
    bare Coulomb; the DF route's from C_oo and C_vv; RPA's d itself."""
    s = systems[name]
    nocc = s['nocc']
    for W in (s['W_aux'], None):          # BSE, then TDHF
        act, diag = isdf_block_action(lr_of(s), nocc, True, W, s['factors'])
        dressed = act.screened_diagonal()
        oracle, C = direct_oracle(s, W)
        assert relative(diag - dressed, oracle) <= ORACLE_TOL
        assert np.abs(diag - dressed - oracle).max() <= (
            ORACLE_TOL * np.abs(oracle).max())
        act_t, _ = isdf_block_action(lr_of(s), nocc, True, W, s['factors'],
                                     spin='triplet')
        pairs = (np.arange(diag.size) if diag.size <= 200 else
                 np.random.default_rng(0).choice(diag.size, SAMPLED_PAIRS,
                                                 replace=False))
        own = triplet_diagonal(act_t, diag, pairs)
        assert np.abs(own - dressed.ravel()[pairs]).max() <= (
            ORACLE_TOL * np.abs(oracle).max())
        assert np.array_equal(act_t.screened_diagonal(), dressed)
        lr_df = LinearResponseSolver(s['eps'].copy(), coeff_df=C,
                                     spin_mode='restricted')
        act_df, diag_df = df_block_action(lr_df, nocc, True, W)
        assert np.array_equal(diag_df, diag)
        assert np.abs(act_df.screened_diagonal() - dressed).max() <= (
            ORACLE_TOL * np.abs(oracle).max())
    act, diag = isdf_block_action(lr_of(s), nocc, False, None, s['factors'])
    assert act.screened_diagonal() is diag
    act_df, diag_df = df_block_action(lr_df, nocc, False, None)
    assert act_df.screened_diagonal() is diag_df


def rank_diagonal(comm, s, sliced):
    """This rank's screened diagonal on its grid rows, whole or sliced
    factors."""
    factors = tuple(np.array(a) for a in s['factors'])
    if sliced:
        r0, r1 = contiguous_block(len(factors[3]), comm.Get_rank(),
                                  comm.Get_size())
        factors = SlicedFactors(*(np.ascontiguousarray(a[r0:r1])
                                  for a in factors[:3]), factors[3], comm)
    act, _ = isdf_block_action(lr_of(s), s['nocc'], True, s['W_aux'].copy(),
                               factors, comm=comm)
    return act.screened_diagonal()


@pytest.mark.parametrize('sliced', [False, True], ids=['whole', 'sliced'])
@pytest.mark.parametrize('size', DIAGONAL_SIZES)
def test_the_diagonal_is_the_one_rank_bits(systems, monkeypatch, size,
                                           sliced):
    """Every rank's screened diagonal is the serial one bitwise: each fixed
    grid tile's addend formed whole by one rank, the addends added in tile
    order (`ordered_chain_sum`)."""
    s = systems['benzene']
    monkeypatch.setattr(davidson, 'DAVIDSON_DIAGONAL_TILE', GATE_TILE)
    assert len(s['factors'][0]) > 3 * size * GATE_TILE    # several per rank
    act, diag = isdf_block_action(lr_of(s), s['nocc'], True, s['W_aux'],
                                  s['factors'])
    serial = act.screened_diagonal()
    out = run_simulated(rank_diagonal, size, s, sliced)
    for r, got in enumerate(out):
        assert np.array_equal(got, serial), (size, r)
    print(f'[info] {size} ranks, {"sliced" if sliced else "whole"}: '
          'every rank the serial diagonal bitwise')


@pytest.mark.parametrize('plant', ['ulp', 'dropped'])
def test_a_moved_or_dropped_tile_fails_the_diagonal(systems, monkeypatch,
                                                    plant):
    """The gate can fail: one tile's addend on rank 1 moved 1e-12 relative, or
    dropped, and the diagonal is not the serial one."""
    s = systems['benzene']
    monkeypatch.setattr(davidson, 'DAVIDSON_DIAGONAL_TILE', GATE_TILE)
    act, _ = isdf_block_action(lr_of(s), s['nocc'], True, s['W_aux'],
                               s['factors'])
    serial = act.screened_diagonal()
    real = davidson.self_pair_densities

    def planted(*args, **kwargs):
        out = real(*args, **kwargs)
        comm = current_comm()
        if comm is not None and comm.Get_rank() == 1:
            if plant == 'dropped':
                return np.zeros_like(out)
            out.flat[0] *= 1 + 1e-12
        return out

    monkeypatch.setattr(davidson, 'self_pair_densities', planted)
    out = run_simulated(rank_diagonal, 3, s, False)
    assert not any(np.array_equal(got, serial) for got in out), plant


@pytest.mark.parametrize('mode,spin', MODES,
                         ids=['-'.join(m) for m in MODES])
@pytest.mark.parametrize('name', ['water', 'benzene'])
def test_the_screened_roots_sit_on_the_bare_ones(systems, name, mode, spin):
    """Every root converged, within ROOT_TOL of the bare-preconditioned
    ones; RPA's diagonal is d, so its solve is the bare one bit for bit."""
    s = systems[name]
    om_b, t_b = solve(s, NROOTS[name], mode, spin, 'bare')
    om_s, t_s = solve(s, NROOTS[name], mode, spin, 'screened')
    assert np.abs(om_s - om_b).max() <= ROOT_TOL
    if mode == 'RPA':
        assert np.array_equal(om_s, om_b)
        assert t_s['davidson_vectors_applied'] == t_b['davidson_vectors_applied']
    else:
        assert t_s['davidson_precond_setup'] > 0.0
    if (name, mode, spin) == ('benzene', 'BSE', 'singlet'):
        assert (t_s['davidson_vectors_applied']
                < t_b['davidson_vectors_applied'])
        assert t_s['davidson_iterations'] <= t_b['davidson_iterations']


def test_the_df_route_takes_it(systems):
    """The DF action's screened diagonal drives the same roots."""
    s = systems['water']
    om_b, _ = solve(s, NROOTS['water'], 'BSE', 'singlet', 'bare', route='df')
    om_s, t_s = solve(s, NROOTS['water'], 'BSE', 'singlet', 'screened',
                      route='df')
    assert np.abs(om_s - om_b).max() <= ROOT_TOL
    assert t_s['davidson_precond_setup'] > 0.0


def screened_root(comm, s, nroots):
    """`_davidson_root` preconditioned by the screened diagonal on this
    rank's ISDF action."""
    act, diag = isdf_block_action(lr_of(s), s['nocc'], True, s['W_aux'].copy(),
                                  tuple(np.array(a) for a in s['factors']),
                                  comm=comm)
    counters = davidson._davidson_counters()
    out = davidson._davidson_root(act, diag, nroots, CONV_TOL, 100, None,
                                  counters=counters, comm=comm,
                                  pair_tile=PAIR_TILE,
                                  preconditioner='screened')
    return out[:5] + (counters,)


def decisions(counters):
    """The per-cycle decisions of the pair-row loop, residual norms included."""
    return [(c['cycle'], c['fresh'], c['new'], c['applied'], c['m1'],
             c['conv'], c['kept'], c['r_norms'].tobytes())
            for c in counters['trace']]


@pytest.mark.parametrize('size', RANK_SIZES)
def test_every_rank_takes_the_same_path(systems, size):
    """Benzene BSE@HF, 12 roots, the screened diagonal on 31 pair tiles:
    every rank takes rank 0's decisions and cycle count, the serial count,
    its roots within the resolution of the serial screened ones."""
    s = systems['benzene']
    om_1, _, _, conv_1, cyc_1, _ = screened_root(None, s, 12)
    assert conv_1.all()
    out = run_simulated(screened_root, size, s, 12)
    ref = decisions(out[0][5])
    for om, X, Y, conv, cycles, counters in out:
        assert conv.all() and cycles == out[0][4] == len(ref)
        assert decisions(counters) == ref
        assert np.array_equal(om, out[0][0]) and np.array_equal(X, out[0][1])
    assert out[0][4] == cyc_1
    assert relative(out[0][0], om_1) <= roots_resolution(s['eps'], s['nocc'],
                                                         om_1)


def test_solve_bse_isdf_takes_the_switch(systems):
    """The keyword reaches the Davidson of the production entry point, whose
    default is the screened diagonal bit for bit; an unknown value is refused
    before anything is built."""
    s = systems['water']
    kw = dict(nroots=3, qp=False, probe=False, factors=s['factors'],
              progress=False)
    om_b, _, _, info_b = solve_bse_isdf(s['mf'], s['mol'], s['nocc'],
                                        preconditioner='bare', **kw)
    om_s, _, _, info_s = solve_bse_isdf(s['mf'], s['mol'], s['nocc'],
                                        preconditioner='screened', **kw)
    om_d, _, _, info_d = solve_bse_isdf(s['mf'], s['mol'], s['nocc'], **kw)
    assert np.abs(om_s - om_b).max() <= ROOT_TOL
    assert np.array_equal(om_d, om_s)
    assert info_b['timings']['davidson_precond_setup'] == 0.0
    assert info_s['timings']['davidson_precond_setup'] > 0.0
    assert info_d['timings']['davidson_precond_setup'] > 0.0
    with pytest.raises(ValueError, match='preconditioner'):
        solve_bse_isdf(s['mf'], s['mol'], s['nocc'], preconditioner='exact',
                       factors=None, progress=False)
    with pytest.raises(ValueError, match='preconditioner'):
        solve_casida_davidson(lr_of(s), s['nocc'], preconditioner='diag')


def test_an_action_without_the_diagonal_is_refused(systems):
    """'screened' on an action that cannot form it raises, never falls back
    to d silently."""
    s = systems['water']
    act, diag = isdf_block_action(lr_of(s), s['nocc'], True, s['W_aux'],
                                  s['factors'])

    def plain(z):
        return act(z)

    with pytest.raises(ValueError, match='screened diagonal'):
        davidson._run_davidson(plain, diag, 3, CONV_TOL, 100, None,
                               preconditioner='screened')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
