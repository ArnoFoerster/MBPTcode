"""The (A-B) probe after the BSE Davidson, on the Davidson's own block action.

`solve_bse_isdf` builds one block action, runs the Davidson on it, and only
then probes min eig(A-B) on the same action, in tiers:

  (i)   Rayleigh-Ritz of (A - B) on the span of the converged roots' X - Y,
        one block action per root: the lowest Ritz value theta and its
        residual r prove an eigenvalue of (A - B) within r of theta, so
        theta - r > 0 (or theta + r < 0) certifies the sign -- of an
        eigenvalue the span reaches, not of the minimum over pair irreps
        no root lies in;
  (ii)  otherwise a Lanczos started from the lowest root's X - Y plus
        PROBE_START_MIX of the cold start 1/d at 1e-2;
  (iii) otherwise the Lanczos from 1/d, which has weight in every pair
        irrep, at escalating tolerances.

`probe=True` converges the Lanczos from 1/d, the general probe, bitwise the
value it had before the probe moved. Nothing the probe computes reaches the
roots.

Gated here on cc-pVDZ Hartree-Fock, BSE@HF (`qp=False`), with the dense
(A - B) built from the fit's own three-index factor (`isdf_df_coefficients`)
as the oracle:

  * the certificate on water, ethylene and benzene: the roots' span proves
    the sign, `probe_matvecs` one per root and `probe_source` 'roots'; theta
    is at most the lowest root's own quotient omega / |X - Y|^2; the sign is
    the converged Lanczos's and the dense minimum's; and the minimum lies
    within r of theta. On benzene the lowest root alone could not bracket
    it: its X - Y lies along the (A - B) eigenvector at 0.26695 Ha (overlap
    0.9998), in another irrep than the minimum at 0.23880, which only the
    fourth root reaches (overlap 0.9958);
  * `probe=True` converges from 1/d to the dense minimum within the
    Lanczos's own tolerance, recorded 'lanczos-cold', serially bitwise the
    standalone probe's value;
  * 90-degree twisted ethene, whose TDHF (A - B) is indefinite (-0.0053 Ha):
    refused by the Davidson's own breakdown at 1, 2 and 3 ranks, the
    breakdown's probe carrying the dense minimum; and with the BSE solved and
    the probe handed that TDHF (A - B), the span of the BSE roots does NOT
    certify, the warm Lanczos proves the negative sign, and the driver
    refuses the roots on every rank;
  * planted fall-throughs: a lowest root contaminated by the highest pair
    fails tier (i) and tier (ii) certifies, from exactly the warm start; a
    Lanczos stalled at its first pass escalates to 1e-3 from 1/d; stalled at
    every pass, the sign is reported unproven;
  * the roots and vectors of the whole solve, preconditioned by the bare d
    the baseline knew, bitwise those of `BASELINE_COMMIT`,
    whose probe ran BEFORE the Davidson on an action of its own, serially,
    'sign' and True, water and ethylene; at 2, 3 and 8 simulated ranks, where
    the Davidson's trial space is cut by pair rows and its reduced sums
    re-associate, within the Davidson's resolution of the one-rank roots;
    and the converged probe's value bitwise at every size;
  * every rank returns rank 0's roots, vectors, value and probe record,
    bitwise, at 2, 3 and 8 ranks, and at 16 on a 12-point grid where four
    ranks own no grid row; and one rank's certificate planted to fail alone is
    handed rank 0's by the certificate's own lockstep -- the audit counts that
    one repair on that one rank -- where without it the ranks' collectives
    part and raise.
"""
import copy
import os
import subprocess
import sys
import tarfile
import threading
import warnings
from pathlib import Path

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import PROBE_START_MIX
from src.Base.utils.mpi_grid import distributed, lockstep_stats, run_simulated
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse import davidson, trial_space
from src.SingleReference.LinearResponse.davidson import (
    _CasidaBreakdown, _lowest_amb_from_action, isdf_bse_factors,
    isdf_block_action, isdf_df_coefficients, lowest_amb_eigenvalue,
    solve_bse_isdf, solve_casida_davidson)
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from tests.test_distributed_fit_mpi import relative, roots_resolution

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
ETHYLENE = ('C 0.0 0.0 0.667; C 0.0 0.0 -0.667; H 0.0 0.923 1.238; '
            'H 0.0 -0.923 1.238; H 0.0 0.923 -1.238; H 0.0 -0.923 -1.238')
BENZENE = ('C 1.396 0 0; C 0.698 1.209 0; C -0.698 1.209 0; C -1.396 0 0; '
           'C -0.698 -1.209 0; C 0.698 -1.209 0; H 2.479 0 0; '
           'H 1.240 2.147 0; H -1.240 2.147 0; H -2.479 0 0; '
           'H -1.240 -2.147 0; H 1.240 -2.147 0')
TWISTED_ETHENE = """C 0 0 0.669; C 0 0 -0.669;
                    H 0  0.923 1.238; H 0 -0.923 1.238;
                    H  0.923 0 -1.238; H -0.923 0 -1.238"""
#: (geometry, roots) of each molecule.
MOLECULES = {'water': (WATER, 3), 'ethylene': (ETHYLENE, 5),
             'benzene': (BENZENE, 12)}
SIZES = [2, 3, 8]
CONV_TOL = 1e-5
#: ARPACK's relative tolerance in `lowest_amb_eigenvalue`, whose residual
#: bounds the converged value's distance from the eigenvalue it found.
ARPACK_TOL = 1e-6
#: The replicated Lanczos's relative residual, which `_lowest_amb_local`
#: tightens the converged probe to.
REPLICATED_TOL = float(np.sqrt(np.finfo(float).eps))
#: Ha. The row split re-associates the action's reduced sums, which moves a
#: distributed root and the certificate from the serial ones in the last bits
#: (1e-13 measured, the bar of tests/test_davidson_lockstep.py).
ROOT_TOL = 1e-11
#: Ha. The dense (A - B) against the block action's, 1e-13 relative on these
#: systems; room for it in the comparisons of a Rayleigh quotient with the
#: dense minimum.
DENSE_TOL = 1e-10
#: Weight of the highest pair's unit vector planted on the lowest root's
#: normalized X - Y: the residual it adds, about 0.1 max(d), outgrows theta.
PLANT_WEIGHT = 0.1
#: Grid points water's fit is cut to, and the ranks that then leave four of
#: them without a row.
CUT_POINTS, CUT_RANKS = 12, 16
REPO = Path(__file__).resolve().parents[1]
#: A commit before the probe moved behind the Davidson: its probe built an
#: action of its own and ran first.
BASELINE_COMMIT = '8b5d2820f5ecb5920440048cbe307382c9040aa4'
THREAD_CAPS = {name: '2' for name in
               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
#: The same solves on the extracted baseline, in their own process.
BASELINE_SOLVES = '''
import copy
import sys
import warnings

sys.path.insert(0, {archive!r})

import numpy as np
from pyscf import gto, scf

from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse.davidson import solve_bse_isdf

warnings.simplefilter('ignore')
out = {{}}
for name, (atom, nroots) in {molecules!r}.items():
    mol = gto.M(atom=atom, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')

    def one_rank(comm, probe):
        rmf = copy.copy(mf)
        rmf.mo_energy = np.asarray(mf.mo_energy, float).copy()
        rmf.mo_coeff = np.asarray(mf.mo_coeff, float).copy()
        return solve_bse_isdf(rmf, mol, nocc, nroots=nroots, qp=False,
                              probe=probe, progress=False,
                              factors=tuple(np.array(a) for a in factors),
                              distribute=comm is not None, comm=comm)

    for probe in {probes!r}:
        for size in {sizes!r}:
            omega, X, Y, info = (one_rank(None, probe) if size == 1 else
                                 run_simulated(one_rank, size, probe)[0])
            key = f'{{name}}_{{probe}}_{{size}}'
            out['omega_' + key], out['X_' + key], out['Y_' + key] = omega, X, Y
            out['amb_' + key] = np.float64(info['min_eig_amb'])
np.savez({out!r}, **out)
'''
PINNED = ('water', 'ethylene')
PINNED_PROBES = ('sign', True)
PINNED_SIZES = [1, 2, 3, 8]
#: Pair rows per tile of the Davidson's distributed trial space in the pins
#: over ranks: water's 95 pairs and ethylene's 320 fall into 12 and 40 tiles,
#: so every rank of every size here holds some.
SPLIT_TILE = 8
#: The vectors against the one-rank ones over ranks, each root's sign fixed.
VECTOR_TOL = 1e-10


def _system(geometry, nroots):
    """Mean field, fit and static W of one molecule, cc-pVDZ Hartree-Fock."""
    mol = gto.M(atom=geometry, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')
    W_aux = isdf_bse_factors(mf, mol, nocc, factors=factors)[2]
    return dict(mol=mol, mf=mf, nocc=nocc, nroots=nroots, factors=factors,
                W_aux=W_aux, eps=np.asarray(mf.mo_energy, float))


@pytest.fixture(scope='module')
def systems():
    warnings.simplefilter('ignore')
    return {name: _system(*spec) for name, spec in MOLECULES.items()}


@pytest.fixture(scope='module')
def twisted():
    warnings.simplefilter('ignore')
    return _system(TWISTED_ETHENE, 3)


def dense_amb(s, W_aux):
    """Every eigenvalue of (A - B), dense, from the fit's own three-index
    factor; W_aux None is TDHF."""
    X_mo, D = s['factors'][:2]
    lr = LinearResponseSolver(s['eps'], coeff_df=isdf_df_coefficients(X_mo, D),
                              spin_mode='restricted')
    A, B = lr.build_casida_matrices(s['nocc'], lBSE=True, W_aux=W_aux)
    return np.linalg.eigvalsh(A - B)


def rank_copy(s):
    """This rank's own mean field and factors: a distributed solve replicates
    rank 0's over them in place, and simulated ranks share one process."""
    mf = copy.copy(s['mf'])
    mf.mo_energy = np.asarray(s['mf'].mo_energy, float).copy()
    mf.mo_coeff = np.asarray(s['mf'].mo_coeff, float).copy()
    return mf, tuple(np.array(a, copy=True) for a in s['factors'])


def solve(s, comm=None, probe='sign', qp=False, **kw):
    """BSE@HF on this rank's own copies, over `comm` where one is given."""
    mf, factors = rank_copy(s)
    kw.setdefault('factors', factors)
    return solve_bse_isdf(mf, s['mol'], s['nocc'], nroots=s['nroots'], qp=qp,
                          probe=probe, progress=False, conv_tol=CONV_TOL,
                          distribute=comm is not None, comm=comm, **kw)


def action(s, comm=None, W_aux='bse'):
    """(apply_AB, diag_d): the BSE block action, or TDHF's for W_aux None."""
    lr = LinearResponseSolver(s['eps'].copy(), spin_mode='restricted')
    return isdf_block_action(lr, s['nocc'], True,
                             s['W_aux'] if W_aux == 'bse' else W_aux,
                             s['factors'], comm=comm)


def ranks(fn, size, *args):
    """fn(comm, *args) on every rank: serially with no comm for size 1."""
    if size == 1:
        with distributed(None):
            return [fn(None, *args)]
    return run_simulated(fn, size, *args)


def bitwise(a, b):
    """True when two arrays agree bit for bit, shape and dtype included."""
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def sign_fixed(X, Y):
    """(X, Y) with each root's sign taken from the largest component of X:
    the Davidson fixes no phase."""
    sign = np.sign(X[np.abs(X).argmax(axis=0), np.arange(X.shape[1])])
    return X * sign, Y * sign


def record(info):
    """The probe's record, its wall time taken out."""
    return {k: v for k, v in info['stats'].items()
            if k.startswith('probe') and k != 'probe_action_s'}


def planted_start(s, X, Y):
    """The lowest root's normalized X - Y with PLANT_WEIGHT of the highest
    pair's unit vector added."""
    start = X[:, 0] - Y[:, 0]
    start = start / np.linalg.norm(start)
    d = davidson.bse_pair_diagonal(s['eps'], s['nocc']).ravel()
    start[np.argmax(d)] += PLANT_WEIGHT
    return start


def rayleigh(apply_AB, diag_d, z):
    """(theta, r) of (A - B) at the normalized z, by one block action."""
    z = z / np.linalg.norm(z)
    A, B = apply_AB(z.reshape(1, *diag_d.shape))
    w = (A - B).ravel()
    theta = float(z @ w)
    return theta, float(np.linalg.norm(w - theta * z))


@pytest.mark.parametrize('name', list(MOLECULES))
def test_the_roots_certify_the_sign(systems, name):
    s = systems[name]
    om, X, Y, info = solve(s)
    st = info['stats']
    theta, r = info['min_eig_amb'], st['probe_residual']
    assert st['probe_source'] == 'roots' and st['probe_matvecs'] == len(om)
    assert st['probe_sign_proven'] and st['probe_tol_used'] is None
    # The span's lowest Ritz value is at most the lowest root's own quotient,
    # omega / |X - Y|^2 up to the Casida residual, whose two halves differ by
    # at most sqrt(2) CONV_TOL.
    xmy = X[:, 0] - Y[:, 0]
    assert (theta <= (om[0] + np.sqrt(2.0) * CONV_TOL * np.linalg.norm(xmy))
            / (xmy @ xmy))
    lam = dense_amb(s, s['W_aux'])
    lanczos = float(lowest_amb_eigenvalue(
        LinearResponseSolver(s['eps'], spin_mode='restricted'), s['nocc'],
        'BSE', s['W_aux'], s['factors'])[0])
    assert abs(lanczos - lam[0]) <= ARPACK_TOL * abs(lanczos) + DENSE_TOL
    assert theta - r > 0.0 and lanczos > 0.0 and lam[0] > 0.0
    assert theta >= lam[0] - DENSE_TOL                 # a Rayleigh quotient
    assert theta - lam[0] <= r                         # the minimum bracketed


@pytest.mark.parametrize('size', [1, 2])
@pytest.mark.parametrize('name', list(MOLECULES))
def test_the_converged_probe_starts_cold(systems, name, size):
    """`probe=True` is the general probe, the Lanczos from 1/d: the dense
    minimum within its own residual bound, ARPACK's tolerance serially and
    sqrt(eps) replicated, and serially the standalone probe's value bitwise,
    on an action of its own."""
    s = systems[name]
    _, _, _, info = ranks(lambda comm: solve(s, comm, probe=True), size)[0]
    lam = dense_amb(s, s['W_aux'])
    value = info['min_eig_amb']
    assert info['stats']['probe_source'] == 'lanczos-cold'
    tol = ARPACK_TOL if size == 1 else REPLICATED_TOL
    assert abs(value - lam[0]) <= tol * abs(value) + DENSE_TOL
    if size == 1:
        assert value == float(lowest_amb_eigenvalue(
            LinearResponseSolver(s['eps'], spin_mode='restricted'),
            s['nocc'], 'BSE', s['W_aux'], s['factors'])[0])


@pytest.mark.parametrize('size', [1, 2, 3])
def test_twisted_ethene_tdhf_is_refused_by_the_davidson(twisted, size):
    """The TDHF Davidson breaks down on every rank, and the breakdown's probe
    names the reference: the dense minimum, rank 0's value everywhere."""
    s = twisted
    lam = dense_amb(s, None)[0]
    assert lam < 0.0

    def one_rank(comm):
        lr = LinearResponseSolver(s['eps'].copy(), spin_mode='restricted')
        try:
            solve_casida_davidson(lr, s['nocc'], nroots=3,
                                  polarizability='TDHF',
                                  isdf_factors=rank_copy(s)[1], comm=comm)
        except _CasidaBreakdown as exc:
            return str(exc), exc.min_eig_amb
        return None

    out = ranks(one_rank, size)
    for verdict in out:
        assert verdict is not None, 'the Davidson solved an unstable reference'
        message, amb = verdict
        assert 'instability of the mean-field reference' in message
        assert amb == out[0][1]
        assert abs(amb - lam) <= ARPACK_TOL * abs(amb) + DENSE_TOL


@pytest.mark.parametrize('size', [1, 2, 3])
def test_twisted_ethene_falls_through_and_is_refused(twisted, monkeypatch,
                                                    size):
    """The BSE of twisted ethene converges (its (A - B) is positive); its probe
    is handed the TDHF (A - B) of the same reference instead: the span of the
    BSE roots fails tier (i), the warm Lanczos proves the sign negative, and
    the driver refuses the roots on every rank."""
    s = twisted
    lam = dense_amb(s, None)[0]
    real = davidson._lowest_amb_from_action
    records = {}

    def on_tdhf(apply_AB, diag_d, **kwargs):
        comm = kwargs.get('comm')
        records[0 if comm is None else comm.Get_rank()] = kwargs['stats']
        return real(*action(s, comm, W_aux=None), **kwargs)

    monkeypatch.setattr(davidson, '_lowest_amb_from_action', on_tdhf)

    def one_rank(comm):
        try:
            solve(s, comm)
        except RuntimeError as exc:
            return str(exc)
        return None

    out = ranks(one_rank, size)
    assert len(records) == size
    for rank, message in enumerate(out):
        assert message is not None, f'rank {rank} returned the roots'
        assert message.startswith('BSE refused before the solve'), message
        st = record({'stats': records[rank]})
        assert st == record({'stats': records[0]})
        assert st['probe_source'] == 'lanczos-warm' and st['probe_sign_proven']
        assert st['probe_tol_used'] == 1e-2 and st['probe_matvecs'] > 2
    # the message prints the value to 1e-6 Ha
    value = float(out[0].split('min eig(A-B) = ')[1].split(' Ha')[0])
    assert value < 0.0
    assert abs(value - lam) <= records[0]['probe_residual'] + 1e-6


def cold_and_warm(s, start):
    """The Lanczos starts the probe builds: 1/d, and `start` with
    PROBE_START_MIX of it, each normalized."""
    cold = 1.0 / davidson.bse_pair_diagonal(s['eps'], s['nocc']).ravel()
    cold /= np.linalg.norm(cold)
    warm = start / np.linalg.norm(start) + PROBE_START_MIX * cold
    return cold, warm / np.linalg.norm(warm)


class _Lanczos:
    """The probe's Lanczos, serial (eigsh) or replicated (`_lanczos_lowest`),
    recording each rank thread's start vectors, and on its first `stalls`
    calls returning the `stalled` vector as the Ritz pair instead."""

    def __init__(self, stalls=0, stalled=None):
        self.stalls, self.stalled = stalls, stalled
        self.local = threading.local()
        self.real_eigsh = davidson.eigsh
        self.real_lanczos = davidson._lanczos_lowest

    def starts(self):
        """This rank thread's start vectors, in call order."""
        return getattr(self.local, 'starts', [])

    def _stall(self, v0):
        self.local.starts = self.starts() + [np.array(v0, copy=True)]
        return len(self.local.starts) <= self.stalls

    def eigsh(self, op, k, which, tol, v0, return_eigenvectors):
        if not self._stall(v0):
            return self.real_eigsh(op, k=k, which=which, tol=tol, v0=v0,
                                   return_eigenvectors=return_eigenvectors)
        x = self.stalled / np.linalg.norm(self.stalled)
        theta = np.array([x @ op.matvec(x)])
        return (theta, x[:, None]) if return_eigenvectors else theta

    def lanczos(self, matvec, v0, k, tol):
        if not self._stall(v0):
            return self.real_lanczos(matvec, v0, k, tol)
        x = self.stalled / np.linalg.norm(self.stalled)
        return np.array([x @ matvec(x)]), x[None, :]


@pytest.mark.parametrize('size', [1, 2, 3])
def test_a_root_that_does_not_certify_falls_through(systems, monkeypatch,
                                                    size):
    """The lowest root planted with the highest pair: tier (i) fails, and the
    Lanczos from exactly that root plus PROBE_START_MIX of 1/d certifies at
    its first tolerance."""
    s = systems['water']
    _, X, Y, _ = solve(s, probe=False)
    start = planted_start(s, X, Y)
    theta, r = rayleigh(*action(s), start)
    assert theta - r <= 0.0, 'the plant does not defeat tier (i)'
    lam = dense_amb(s, s['W_aux'])
    _, warm = cold_and_warm(s, start)
    lanczos = _Lanczos()
    monkeypatch.setattr(davidson, 'eigsh', lanczos.eigsh)
    monkeypatch.setattr(davidson, '_lanczos_lowest', lanczos.lanczos)

    def one_rank(comm):
        stats = {}
        value = _lowest_amb_from_action(*action(s, comm), sign_only=True,
                                        stats=stats, comm=comm, start=start)
        return float(value[0]), stats, lanczos.starts()

    out = ranks(one_rank, size)
    for value, stats, starts in out:
        assert value == out[0][0]
        assert stats['probe_source'] == 'lanczos-warm'
        assert stats['probe_sign_proven'] and stats['probe_tol_used'] == 1e-2
        assert stats['probe_matvecs'] > 2          # tier (i), Lanczos, residual
        assert len(starts) == 1
        assert np.abs(starts[0] - warm).max() <= 1e-15
        resid = stats['probe_residual']
        assert value - resid > 0.0
        assert np.abs(lam - value).min() <= resid + DENSE_TOL


@pytest.mark.parametrize('size', [1, 2, 3])
@pytest.mark.parametrize('stalls', [1, 4])
def test_a_lanczos_that_does_not_certify_escalates(systems, monkeypatch,
                                                   size, stalls):
    """Tier (i) planted to fail and the warm Lanczos stalled on the planted
    vector: the Lanczos from 1/d at the next tolerance, 1e-3, certifies.
    Stalled at all four passes, the sign is reported unproven at the probe's
    own tolerance."""
    s = systems['water']
    _, X, Y, _ = solve(s, probe=False)
    start = planted_start(s, X, Y)
    cold, warm = cold_and_warm(s, start)
    lanczos = _Lanczos(stalls, stalled=start)
    monkeypatch.setattr(davidson, 'eigsh', lanczos.eigsh)
    monkeypatch.setattr(davidson, '_lanczos_lowest', lanczos.lanczos)

    def one_rank(comm):
        stats = {}
        value = _lowest_amb_from_action(*action(s, comm), sign_only=True,
                                        stats=stats, comm=comm, start=start)
        return float(value[0]), record({'stats': stats}), lanczos.starts()

    out = ranks(one_rank, size)
    for value, stats, starts in out:
        assert value == out[0][0] and stats == out[0][1]
        assert stats['probe_source'] == 'lanczos-cold'
        assert len(starts) == (2 if stalls == 1 else 4)
        assert np.abs(starts[0] - warm).max() <= 1e-15
        assert all(bitwise(v0, cold) for v0 in starts[1:])
        if stalls == 1:
            assert stats['probe_sign_proven']
            assert stats['probe_tol_used'] == 1e-3
        else:
            assert not stats['probe_sign_proven']
            assert stats['probe_tol_used'] == 1e-6
            assert value - stats['probe_residual'] <= 0.0


@pytest.fixture(scope='session')
def baseline(tmp_path_factory):
    """`BASELINE_COMMIT`'s roots and vectors, per molecule, probe and size."""
    out = tmp_path_factory.mktemp('probe_after_davidson_baseline')
    tar = out.parent / f'{BASELINE_COMMIT}.tar'
    done = subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar',
                           '-o', str(tar), BASELINE_COMMIT],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    with tarfile.open(tar) as fh:
        fh.extractall(out)
    script, npz = out / 'baseline.py', out / 'roots.npz'
    script.write_text(BASELINE_SOLVES.format(
        archive=str(out), molecules={n: MOLECULES[n] for n in PINNED},
        probes=PINNED_PROBES, sizes=PINNED_SIZES, out=str(npz)))
    env = dict(os.environ, **THREAD_CAPS)
    env.pop('PYTHONPATH', None)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(out),
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    return dict(np.load(npz))


@pytest.mark.parametrize('size', PINNED_SIZES)
@pytest.mark.parametrize('probe', PINNED_PROBES)
@pytest.mark.parametrize('name', PINNED)
def test_the_roots_are_bitwise_the_parents(systems, baseline, name, probe,
                                           size, monkeypatch):
    """The probe moved behind the Davidson and onto its action and reaches no
    root: serially no bit of a root or a vector moves, and the converged
    probe, from 1/d on the action alone, returns the baseline's value bit for
    bit at every size.

    Over ranks the roots are NOT pinned bitwise: the Davidson's trial space
    is cut by pair rows there (`trial_space.real_eig_rows`), and its sums over
    the pair index -- projected blocks, norms, overlaps -- are reduced over
    the ranks' tiles, which re-associates them with the rank count. On
    SPLIT_TILE tiles, which give every rank here pair rows of its own, the
    roots are held to the Davidson's resolution of the one-rank roots and
    the vectors to VECTOR_TOL of them.

    The solves ask for preconditioner='bare': the baseline's Davidson divided
    by d, and the default preconditioner is now the screened diagonal, which
    takes another iteration path to the same roots: screened, the vectors
    sat up to 7.4e-7 and the roots 1.3e-11 relative from the baseline's, both
    within the solve's convergence to CONV_TOL but outside these gates."""
    s = systems[name]
    if size > 1:
        monkeypatch.setattr(trial_space, 'DAVIDSON_PAIR_TILE', SPLIT_TILE)
    omega, X, Y, info = ranks(lambda comm: solve(s, comm, probe=probe,
                                                 preconditioner='bare'),
                              size)[0]
    key = f'{name}_{probe}_{size}'
    one = f'{name}_{probe}_1'
    if size == 1:
        assert bitwise(omega, baseline['omega_' + key])
        assert bitwise(X, baseline['X_' + key])
        assert bitwise(Y, baseline['Y_' + key])
    else:
        assert relative(omega, baseline['omega_' + one]) <= roots_resolution(
            info['eps'], s['nocc'], baseline['omega_' + one])
        for got, want in zip(sign_fixed(X, Y),
                             sign_fixed(baseline['X_' + one],
                                        baseline['Y_' + one])):
            assert np.abs(got - want).max() <= VECTOR_TOL
    if probe is True:
        assert bitwise(np.float64(info['min_eig_amb']), baseline['amb_' + key])


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('name', PINNED)
def test_every_rank_returns_rank_zeros_probe(systems, name, size):
    """Every rank's roots, vectors, value and probe record are rank 0's,
    bitwise; rank 0's within the row split's last bits of the serial ones."""
    s = systems[name]
    om_s, _, _, info_s = solve(s)
    out = ranks(lambda comm: solve(s, comm), size)
    om0, X0, Y0, info0 = out[0]
    assert np.abs(om0 - om_s).max() <= ROOT_TOL
    assert abs(info0['min_eig_amb'] - info_s['min_eig_amb']) <= ROOT_TOL
    for key in ('probe_source', 'probe_matvecs', 'probe_sign_proven',
                'probe_tol_used'):
        assert info0['stats'][key] == info_s['stats'][key], key
    for om, X, Y, info in out:
        assert bitwise(om, om0) and bitwise(X, X0) and bitwise(Y, Y0)
        assert info['min_eig_amb'] == info0['min_eig_amb']
        assert record(info) == record(info0)


def test_ranks_that_own_no_grid_row(systems):
    """Water's fit cut to CUT_POINTS grid points at CUT_RANKS ranks, so four
    ranks hold no row of Zt: rank 0's solve and certificate everywhere."""
    s = systems['water']
    cut = tuple(np.ascontiguousarray(a[:CUT_POINTS]) for a in s['factors'])
    assert CUT_RANKS > CUT_POINTS

    def one_rank(comm):
        return solve(s, comm, qp=s['eps'].copy(), W_aux=s['W_aux'].copy(),
                     factors=tuple(a.copy() for a in cut))

    om_s, _, _, info_s = solve(s, qp=s['eps'].copy(), W_aux=s['W_aux'],
                               factors=cut)
    out = ranks(one_rank, CUT_RANKS)
    om0, X0, Y0, info0 = out[0]
    assert np.abs(om0 - om_s).max() <= ROOT_TOL
    assert info0['stats']['probe_source'] == 'roots'
    for om, X, Y, info in out:
        assert bitwise(om, om0) and bitwise(X, X0) and bitwise(Y, Y0)
        assert info['min_eig_amb'] == info0['min_eig_amb']
        assert record(info) == record(info0)


@pytest.mark.parametrize('size', SIZES)
def test_one_ranks_drifted_certificate_is_rank_zeros(systems, size):
    """The last rank's tier (i) action output moved by 1e3 Ha on one element
    that the lowest root hardly touches: alone it would not certify and would
    enter the Lanczos while the others returned. The certificate's lockstep
    hands it rank 0's, so every rank certifies from the roots with rank 0's
    value, and the audit counts that one repair on that one rank."""
    s = systems['water']
    om, X, Y, _ = solve(s, probe=False)
    start = X - Y
    quiet = int(np.argmin(np.abs(start[:, 0])))
    target = size - 1

    def one_rank(comm):
        act, diag_d = action(s, comm)
        calls = [0]

        def drifted(z):
            A, B = act(z)
            calls[0] += 1
            if comm.Get_rank() == target and calls[0] == 1:
                A.reshape(-1)[quiet] += 1e3
            return A, B

        stats = {}
        with distributed(comm, audit=True):
            lockstep_stats(reset=True)
            value = _lowest_amb_from_action(drifted, diag_d, sign_only=True,
                                            stats=stats, comm=comm,
                                            start=start)
            return float(value[0]), stats, lockstep_stats()

    out = run_simulated(one_rank, size)
    for rank, (value, stats, audit) in enumerate(out):
        assert value == out[0][0]
        assert stats['probe_source'] == 'roots'
        assert stats['probe_matvecs'] == len(om)
        assert audit['mismatched_calls'] == (1 if rank == target else 0)
        assert (audit['max_abs_diff'] > 1e2) == (rank == target)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
