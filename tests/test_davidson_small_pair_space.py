"""A Davidson whose pair space runs out is completed densely, and only then.

pyscf's `real_eig` adds each correction together with its x/y-swapped
partner, so m trial pairs span 2m of the 2 n_ov dimensions. Once one cycle's
corrections outnumber the pairs left, their span is the whole remainder,
every direction in it is its own partner (x = +-y), the partner test drops
them all, and the solve stops with "no new direction survived the
linear-dependence test" at any tolerance.

The reproduction is water/6-31G (40 pairs), BSE@G0W0 on PBE by the DF
route, 3 roots: at cycle 4 the subspace holds 34 of the 40 pairs (68 of 80
dimensions) and 20 corrections arrive for the 12 dimensions left; their part
outside the subspace has rank 12, none is kept, and the three roots stop at
|r| = 4.1e-3, 6.0e-6 and 1.2e-2, 1.5e-5 Ha off the dense ones, at conv_tol
1e-8 and 1e-5 alike. 2 roots on the same operator converge on their own at
the full 40.

Gated here:

  * the reproduction converges to |r| <= BSE_DAVIDSON_CONV_TOL and meets the
    dense roots to DENSE_TOL at 2 and 3 roots and guess factors 1 and 2, the
    3-root solves by the dense completion and the 2-root ones without it;
    `solve_bse_df` itself at 3 roots does the same;
  * at 2 and 3 simulated ranks every rank returns rank 0's roots and vectors
    bitwise, and rank 0 the serial ones, since the DF action is not divided;
  * the other pair-space failure, real_eig's projected (A-B) block breaking
    down after the partner test kept 3 pairs for the last one left (water/
    6-31G BSE@HF, 4 roots, conv_tol 1e-5), is completed densely too;
  * solves that converge are BITWISE those of `BASELINE_COMMIT`, the commit
    before the completion, each in its own process: water/cc-pVDZ (95
    pairs), 5 roots at the default conv_tol and at BSE_DAVIDSON_CONV_TOL,
    which fills its whole pair space and converges on it. The 500-pair model
    of test_davidson_residual_floor converges without the completion and its
    subspace stays under half of it, but is not pinned to that commit
    bitwise: the commit predates the residual-floor sizing of the Davidson's
    corrections, which moves the model's bits by design (measured 6.0e-14 Ha
    on the roots and one eigenvector's sign) and leaves water's alone. The
    completion acts on failed solves only, which is what keeps them;
  * with the completion unavailable (a pair space above BSE_DENSE_MAX_NOV)
    the warning names the pair space and advises fewer roots, a larger guess
    or the dense solver, and never a looser tolerance.

Disabling the completion (`completes` returning False) fails the 3-root
cases on their residuals, the two serial and the four at 2 and 3 ranks, the
`solve_bse_df` case on its warning and the breakdown case on its
RuntimeError, while the 2-root, bitwise and warning gates still pass: the
rule, not the tolerance, is what converges them.
"""
import inspect
import os
import re
import subprocess
import sys
import tarfile
import warnings
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.constants import BSE_DAVIDSON_CONV_TOL
from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_orbital_energies)
from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.LinearResponse import davidson
from src.SingleReference.LinearResponse.davidson import (solve_bse_df,
                                                         solve_casida_davidson)
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver)

WATER = 'O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24'
#: The geometry of the breakdown case and of the bitwise one.
WATER_C2V = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
CASES = [(nroots, guess_factor) for nroots in (2, 3) for guess_factor in (1, 2)]
SIZES = [2, 3]
#: Hartree. The Davidson's own default residual, the breakdown case's.
DEFAULT_CONV_TOL = inspect.signature(
    solve_casida_davidson).parameters['conv_tol'].default
#: Hartree. The roots against `build_casida_matrices`' dense ones, measured
#: 1.9e-15 completed and 3.6e-14 converged on the reproduction, 3.5e-14 on the
#: breakdown case.
DENSE_TOL = 1e-12
REPO = Path(__file__).resolve().parents[1]
#: A commit before the dense completion: the water solves are bitwise its
#: solves (see the module docstring for why the model is not).
BASELINE_COMMIT = '3ae688706f409591b2304d9a7ef653122aa36be6'
THREAD_CAPS = {name: '2' for name in
               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
#: Converging solves on one tree, in their own process, to an npz.
CONVERGING_PROBE = '''
import inspect
import sys
import warnings

sys.path.insert(0, {tree!r})

import numpy as np
from pyscf import gto, scf

from src.Base.constants import BSE_DAVIDSON_CONV_TOL
from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_orbital_energies)
from src.SingleReference.LinearResponse import davidson
from src.SingleReference.LinearResponse.davidson import (bse_pair_diagonal,
                                                         solve_casida_davidson)
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver)

warnings.simplefilter('ignore')
out = {{}}
# The baseline predates the Davidson's timings; the space is then unknown (-1)
# and only the values are compared.
TIMED = 'timings' in inspect.signature(solve_casida_davidson).parameters


def timed(t):
    return dict(timings=t) if TIMED else {{}}


def record(name, omega, X, Y, t, stats=None):
    out[name + '_omega'], out[name + '_X'], out[name + '_Y'] = omega, X, Y
    out[name + '_space'] = t.get('davidson_subspace_max', -1)
    out[name + '_dense'] = bool((stats or {{}}).get('davidson_dense_completion'))


mol = gto.M(atom={atom!r}, basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
mf.conv_tol = 1e-11
mf.kernel()
nocc = mol.nelectron // 2
lr = LinearResponseSolver(get_orbital_energies(mf, representation='spatial'),
                          coeff_df=get_density_fitting_coefficients(
                              mol, mf, representation='spatial'),
                          spin_mode='restricted')
w_aux = lr.static_screening_aux(nocc)
for name, tol in (('water_default', {{}}),
                  ('water_chain', {{'conv_tol': BSE_DAVIDSON_CONV_TOL}})):
    t, stats = {{}}, {{}}
    record(name, *solve_casida_davidson(lr, nocc, nroots=5,
                                        polarizability='BSE', W_aux=w_aux,
                                        stats=stats, **timed(t), **tol), t,
           stats)

# test_davidson_residual_floor's model: A = D + 2 V^T V, B = 2 V^T V.
rng = np.random.default_rng(7)
eps = np.concatenate([-np.sort(rng.uniform(0.3, 12.0, 10))[::-1],
                      np.sort(rng.uniform(0.05, 3.0, 50))])
diag = bse_pair_diagonal(eps, 10)
V = rng.standard_normal((20, diag.size)) / np.sqrt(diag.size) * 0.5


def apply_AB(z):
    v = (2 * (z.reshape(len(z), -1) @ V.T) @ V).reshape(z.shape)
    return diag[None] * z + v, v


t, stats = {{}}, {{}}
record('model', *davidson._run_davidson(apply_AB, diag, 5,
                                        BSE_DAVIDSON_CONV_TOL, 100, None,
                                        stats=stats, **timed(t)), t, stats)
np.savez({out!r}, **out)
'''


@pytest.fixture(scope='module')
def small():
    """The reproduction's operator: the G0W0 diagonal, DF factor and static W
    `solve_bse_df` builds on water/6-31G PBE, and its dense roots."""
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    mf = dft.RKS(mol, xc='pbe').density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    nocc = mol.nelectron // 2
    # 2 roots converge without the completion; the operator is what is wanted.
    _, _, _, info = solve_bse_df(mf, mol, nocc, nroots=2, probe=False,
                                 progress=False)
    lr = LinearResponseSolver(info['eps'], coeff_df=info['coeff_df'],
                              spin_mode='restricted')
    return dict(mol=mol, mf=mf, nocc=nocc, lr=lr, eps=info['eps'],
                coeff=info['coeff_df'], w_aux=info['W_aux'],
                dense=dense_roots(lr, nocc, info['W_aux']))


@pytest.fixture(scope='module')
def breakdown():
    """water/6-31G BSE@HF in the cc-pVDZ Coulomb-fitting basis, whose 4-root
    Davidson broke down at conv_tol 1e-5, and its dense roots."""
    mol = gto.M(atom=WATER_C2V, basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol = 1e-11
    mf.kernel()
    nocc = mol.nelectron // 2
    lr = LinearResponseSolver(get_orbital_energies(mf, representation='spatial'),
                              coeff_df=get_density_fitting_coefficients(
                                  mol, mf, representation='spatial'),
                              spin_mode='restricted')
    w_aux = lr.static_screening_aux(nocc)
    return dict(nocc=nocc, lr=lr, w_aux=w_aux,
                dense=dense_roots(lr, nocc, w_aux))


@pytest.fixture(scope='module')
def serial(small):
    """The serial reproduction per (nroots, guess_factor)."""
    return {case: solve(small, *case) for case in CASES}


@pytest.fixture(scope='module')
def converging(tmp_path_factory):
    """The converging solves on `BASELINE_COMMIT` and on this tree, each in
    its own process."""
    out = tmp_path_factory.mktemp('davidson_small_pair_space')
    tar = out / f'{BASELINE_COMMIT}.tar'
    done = subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar',
                           '-o', str(tar), BASELINE_COMMIT],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    with tarfile.open(tar) as fh:
        fh.extractall(out / 'baseline')
    env = dict(os.environ, **THREAD_CAPS)
    env.pop('PYTHONPATH', None)
    results = {}
    for name, tree in (('baseline', out / 'baseline'), ('current', REPO)):
        script, npz = out / f'{name}.py', out / f'{name}.npz'
        script.write_text(CONVERGING_PROBE.format(tree=str(tree),
                                                  atom=WATER_C2V,
                                                  out=str(npz)))
        proc = subprocess.run([sys.executable, str(script)], cwd=str(tree),
                              capture_output=True, text=True, env=env)
        assert proc.returncode == 0, proc.stderr
        results[name] = dict(np.load(npz))
    return results


def dense_roots(lr, nocc, w_aux):
    """Every Casida root of production's dense builder, which owes nothing to
    the block action: omega^2 from (A-B)^1/2 (A+B) (A-B)^1/2."""
    A, B = lr.build_casida_matrices(nocc, lBSE=True, W_aux=w_aux)
    w, U = np.linalg.eigh(A - B)
    s = (U * np.sqrt(w)) @ U.T
    return np.sqrt(np.linalg.eigvalsh(s @ (A + B) @ s))


def solve(small, nroots, guess_factor, comm=None, **kw):
    """(omega, X, Y, stats) of the reproduction on this rank's own copies."""
    lr = LinearResponseSolver(np.array(small['eps']),
                              coeff_df=np.array(small['coeff']),
                              spin_mode='restricted')
    stats = {}
    omega, X, Y = solve_casida_davidson(
        lr, small['nocc'], nroots=nroots, polarizability='BSE',
        W_aux=np.array(small['w_aux']), conv_tol=BSE_DAVIDSON_CONV_TOL,
        guess_factor=guess_factor, stats=stats, comm=comm, **kw)
    return omega, X, Y, stats


def residual(lr, nocc, w_aux, omega, X, Y):
    """|[A X + B Y - omega X; B X + A Y + omega Y]| per root, through the
    block action."""
    act, diag = davidson._block_action(lr, nocc, 'BSE', w_aux, None)
    shape = (-1,) + diag.shape
    Ax, Bx = act(X.T.reshape(shape))
    Ay, By = act(Y.T.reshape(shape))
    n = len(omega)
    top = (Ax + By).reshape(n, -1) - omega[:, None] * X.T
    bot = (Bx + Ay).reshape(n, -1) + omega[:, None] * Y.T
    return np.sqrt(np.sum(top**2, axis=1) + np.sum(bot**2, axis=1))


def bitwise(a, b):
    """True when two arrays agree bit for bit, shape and dtype included."""
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


@pytest.mark.parametrize('nroots,guess_factor', CASES)
def test_the_small_pair_space_converges_to_the_dense_roots(small, serial,
                                                           nroots,
                                                           guess_factor):
    """Every root at the tolerance and on the dense one; 3 roots get there by
    the completion, having stopped at 34 of the 40 pairs, and 2 roots on their
    own at the full 40."""
    omega, X, Y, stats = serial[(nroots, guess_factor)]
    r = residual(small['lr'], small['nocc'], small['w_aux'], omega, X, Y)
    assert r.max() <= BSE_DAVIDSON_CONV_TOL, r
    assert np.abs(omega - small['dense'][:nroots]).max() <= DENSE_TOL
    assert stats['davidson_dense_completion'] == (nroots == 3)


def test_solve_bse_df_completes_the_reproduction(small):
    """The entry point the stall was found through: no warning, the dense
    roots, and the record of how they were reached."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        omega, _, _, info = solve_bse_df(small['mf'], small['mol'],
                                         small['nocc'], nroots=3,
                                         conv_tol=BSE_DAVIDSON_CONV_TOL,
                                         progress=False)
    assert not [w for w in record if 'Davidson' in str(w.message)]
    assert np.abs(omega - small['dense'][:3]).max() <= DENSE_TOL
    assert info['stats']['davidson_dense_completion']
    assert info['timings']['davidson_subspace_max'] == small['dense'].size


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('nroots,guess_factor', CASES)
def test_every_rank_returns_rank_zeros_roots(small, serial, size, nroots,
                                             guess_factor):
    """Rank 0's roots and vectors on every rank, bitwise, and the serial ones
    on rank 0: the DF action runs whole on every rank on lockstepped inputs,
    and the dense completion builds its A and B through it."""
    out = run_simulated(lambda comm: solve(small, nroots, guess_factor, comm),
                        size)
    om0, X0, Y0, st0 = out[0]
    r = residual(small['lr'], small['nocc'], small['w_aux'], om0, X0, Y0)
    assert r.max() <= BSE_DAVIDSON_CONV_TOL, r
    assert np.abs(om0 - small['dense'][:nroots]).max() <= DENSE_TOL
    om_s, X_s, Y_s, st_s = serial[(nroots, guess_factor)]
    assert bitwise(om0, om_s) and bitwise(X0, X_s) and bitwise(Y0, Y_s)
    for omega, X, Y, stats in out:
        assert bitwise(omega, om0) and bitwise(X, X0) and bitwise(Y, Y0)
        for key in ('davidson_vind_calls', 'davidson_block_actions',
                    'davidson_dense_completion'):
            assert stats[key] == st0[key] == st_s[key], key


def test_a_breakdown_at_the_pair_space_limit_is_completed_densely(breakdown):
    """real_eig's projected (A-B) block lost definiteness after the partner
    test kept 3 pairs for the 1 left; (A-B) itself is positive definite, so
    the dense completion, not the instability error, is the answer."""
    b = breakdown
    stats = {}
    omega, X, Y = solve_casida_davidson(b['lr'], b['nocc'], nroots=4,
                                        polarizability='BSE', W_aux=b['w_aux'],
                                        guess_factor=1, stats=stats)
    assert stats['davidson_dense_completion']
    assert np.abs(omega - b['dense'][:4]).max() <= DENSE_TOL
    r = residual(b['lr'], b['nocc'], b['w_aux'], omega, X, Y)
    assert r.max() <= DEFAULT_CONV_TOL


@pytest.mark.parametrize('case', ['water_default', 'water_chain', 'model'])
def test_converging_solves_are_bitwise_the_baseline_commits(converging, case):
    """The completion acts on failed solves only. water/cc-pVDZ fills its 95
    pairs and converges on them, so a rule that acted whenever the space ran
    out would have moved it, and it is bitwise the baseline's; the 500-pair
    model stays under half its space and never reaches the completion."""
    base, cur = converging['baseline'], converging['current']
    assert not bool(cur[f'{case}_dense'])
    space = int(cur[f'{case}_space'])
    if case == 'model':
        assert 2 * space <= 500
        return
    for part in ('omega', 'X', 'Y'):
        assert bitwise(cur[f'{case}_{part}'], base[f'{case}_{part}']), part
    assert space == 95


def test_the_pair_space_limit_names_itself_when_not_completed(small,
                                                              monkeypatch):
    """Above BSE_DENSE_MAX_NOV the solve is not completed, and the stop is
    reported as the pair space's, with what to change -- a tolerance moves
    nothing there."""
    n_pair = small['dense'].size
    monkeypatch.setattr(davidson, 'BSE_DENSE_MAX_NOV', n_pair - 1)
    with pytest.warns(RuntimeWarning) as record:
        solve(small, 3, 2)
    message = ' '.join(str(w.message) for w in record
                       if 'Davidson left' in str(w.message))
    assert re.search(rf'held \d+ of the {n_pair} particle-hole pairs', message)
    assert 'fewer roots, raise guess_factor, or use the dense solver' in message
    assert 'conv_tol' not in message.split('happened to give.')[1]
    with pytest.raises(RuntimeError, match='Refused.*fewer roots'):
        solve(small, 3, 2, refuse_unconverged=True)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
