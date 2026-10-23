"""A Davidson whose pair space runs out is completed densely, and only then.

pyscf's `real_eig` adds each correction together with its x/y-swapped
partner, so m trial pairs span 2m of the 2 n_ov dimensions. Once one cycle's
corrections outnumber the pairs left, their span is the whole remainder,
every direction in it is its own partner (x = +-y), the partner test drops
them all, and the solve stops with "no new direction survived the
linear-dependence test" at any tolerance.

The reproduction is water/6-31G (40 pairs), BSE@G0W0 on PBE by the DF
route, 3 roots: at cycle 4 the subspace holds 33 of the 40 pairs (66 of 80
dimensions) and 20 corrections arrive for the 14 dimensions left; none is
kept, and the three roots stop at |r| = 8.8e-3, 4.8e-4 and 9.9e-3, 1.2e-5 Ha
off the dense ones, at conv_tol 1e-8 and 1e-5 alike (measured with the
completion disabled). 2 roots on the same operator converge on
their own at the full 40.

The stall needs an operator on which the last cycle's corrections exactly
outnumber the pairs left, and a meV move of the quasiparticle energies takes
it away: on the first geometry, 'O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24', the
3-root solve converges on its own. The geometry is therefore the second H moved
to (0.93, 0, -0.25) and the first to 0.98, where the stall is back on
today's operator. `test_the_pair_space_limit_names_itself_when_not_completed`
is the gate that the reproduction still stalls: it fails with "DID NOT WARN"
once it does not.

The stall needs real_eig's twenty corrections a cycle. The Davidson now
corrects only its tracked Ritz pairs (`davidson._davidson_root`), and on the
reproduction it no longer runs out of pairs at any of 2 to 15 roots, every
root on the dense one to 2e-13 Ha without the completion (workstation,
measured). The completion stays for a solve whose tracked pairs outnumber
the pairs left; the stall gates reach it by handing the loop real_eig's
width (`real_eig_width`), the path the completion was written for.

The stall is the iteration's path, not a guarantee of the solver: the last
cycles' partner test is degenerate (|w| = 1 to rounding), so under a cluster's MKL a
relative 1e-16 move of the operator's inputs decides whether it happens
(measured: 15 of 30 neighbours stalled at one noise size). The
solver guarantees the roots, to the tolerance and to the dense ones, by either
path. The stall gates therefore run on `stalling`, the first round-off
neighbour of the reproduction that stalls (the reproduction itself under the
recording BLAS), and fail if none of 40 does; the other gates assert what holds
on either path.

Gated here:

  * the reproduction converges to |r| <= BSE_DAVIDSON_CONV_TOL and meets the
    dense roots to DENSE_TOL at 2 and 3 roots and guess factors 1 and 2, the
    3-root solves (at real_eig's width) by the dense completion and the
    2-root ones without it;
    `solve_bse_df` itself at 3 roots does the same;
  * at 2 and 3 simulated ranks every rank returns rank 0's roots and vectors
    bitwise, from one iteration, the DF action not being divided, and rank 0
    the serial roots to the Davidson's resolution, its trial space being cut
    by pair rows over the ranks, which re-associates the degenerate partner
    test at the exhausted pair space and can change the path there;
  * the other pair-space failure, real_eig's projected (A-B) block breaking
    down after the partner test kept 3 pairs for the last one left (water/
    6-31G BSE@HF, 4 roots, conv_tol 1e-5), is completed densely too;
  * solves that converge are those of `BASELINE_COMMIT`, the commit before
    the completion, each in its own process and this tree's at real_eig's
    width (the baseline's path), uncompleted and within the
    class bars (`assert_within_class`): water/cc-pVDZ (95 pairs), 5 roots at
    the default conv_tol and at BSE_DAVIDSON_CONV_TOL, which fills its whole
    pair space and converges on it. They were bitwise until the trial space
    was projected twice (`trial_space.gram_schmidt_rows`), which moves the
    full-space roots by 7.3e-14 Ha and the vectors by 6.3e-13 (workstation,
    measured); with the single projection restored they are bitwise
    again. The 500-pair model
    of test_davidson_residual_floor converges without the completion and its
    subspace stays under half of it, but is not pinned to that commit
    bitwise: the commit predates the residual-floor sizing of the Davidson's
    corrections, which moves the model's bits by design (measured 6.0e-14 Ha
    on the roots and one eigenvector's sign) and leaves water's alone. The
    completion acts on failed solves only, which is what keeps them;
  * the reproductions above, the breakdown and the baseline's solves divide
    by the bare d they were found with, asked for: the default
    preconditioner is the screened diagonal, on which the reproduction
    and `solve_bse_df` converge to the dense roots without exhausting the
    pair space, and the breakdown reaches its dense roots too -- gated as
    well;
  * with the completion unavailable (a pair space above BSE_DENSE_MAX_NOV)
    the warning names the pair space and advises fewer roots, a larger guess
    or the dense solver, and never a looser tolerance.

Disabling the completion (`completes` returning False) fails the 3-root
cases on their residuals, the two serial and the four at 2 and 3 ranks, the
`solve_bse_df` case on its warning and the breakdown case on its
RuntimeError, while the 2-root, bitwise and warning gates still pass: the
rule, not the tolerance, is what converges them.
"""
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto, scf
from pyscf.lib import param

from tests.archived_tree import require_commit, unpack_commit
from tests.archive_radii import install_current_radii
from src.Base.numerics.constants import BSE_DAVIDSON_CONV_TOL
from tests.pins import THREAD_CAPS, assert_within_class, bitwise
from src.Base.numerics.numerics import Numerics
from src.Base.pyscf.pyscf_interface import (get_density_fitting_coefficients,
                                            get_orbital_energies)
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.LinearResponse import (block_actions, davidson,
                                                davidson_shared, trial_space)
from src.SingleReference.LinearResponse.davidson import solve_casida_davidson
from src.SingleReference.BSE.solve import solve_bse_df
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver)
from tests.test_distributed_fit_mpi import relative, roots_resolution

WATER = 'O 0 0 0; H 0 0 0.98; H 0.93 0 -0.25'
#: The geometry of the breakdown case and of the bitwise one.
WATER_C2V = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
CASES = [(nroots, guess_factor) for nroots in (2, 3) for guess_factor in (1, 2)]
SIZES = [2, 3]
#: Pair rows per tile of the Davidson's distributed trial space in the rank
#: gate: the reproduction's 40 pairs fall into 10 tiles.
SPLIT_TILE = 4
#: Hartree. The residual the breakdown case broke down at: the entry points'
#: own default until they took the chain's (BSE_DAVIDSON_CONV_TOL).
DEFAULT_CONV_TOL = 1e-5
#: Hartree. The roots against `build_casida_matrices`' dense ones, measured
#: 5.9e-14 completed and 2.7e-14 converged on the reproduction, 3.5e-14 on the
#: breakdown case.
DENSE_TOL = 1e-12
#: Hartree. The breakdown case's roots against the dense ones by the screened
#: default, which converges it at DEFAULT_CONV_TOL rather than completing it:
#: a root is off by the square of its residual over the gap, measured 1.1e-12
#: once the Davidson corrects only its tracked pairs (3.5e-14 before).
BREAKDOWN_DENSE_TOL = DEFAULT_CONV_TOL ** 2
REPO = Path(__file__).resolve().parents[1]
#: A commit before the dense completion: the water solves are bitwise its
#: solves (see the module docstring for why the model is not).
BASELINE_COMMIT = '3ae688706f409591b2304d9a7ef653122aa36be6'
#: Converging solves on one tree, in their own process, to an npz.
CONVERGING_PROBE = '''
import inspect
import sys
import warnings

sys.path.insert(0, {tree!r})

import numpy as np
from pyscf import gto, scf

try:
    from src.Base.numerics.constants import BSE_DAVIDSON_CONV_TOL
except ImportError:  # an older tree's module layout
    from src.Base.constants import BSE_DAVIDSON_CONV_TOL
try:
    from src.Base.pyscf.pyscf_interface import (get_density_fitting_coefficients,
                                          get_orbital_energies)
except ImportError:  # an older tree's module layout
    from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                          get_orbital_energies)
from src.SingleReference.LinearResponse import davidson
from src.SingleReference.LinearResponse.davidson import (bse_pair_diagonal,
                                                         solve_casida_davidson)
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver)

warnings.simplefilter('ignore')
out = {{}}
# A tree whose Davidson corrects only its tracked pairs is handed real_eig's
# width, the baseline's path (`real_eig_width`).
if hasattr(davidson, 'tracked_roots'):
    from pyscf.lib import param
    from src.SingleReference.LinearResponse import davidson_shared
    davidson.tracked_roots = (lambda nroots, n_pair, extra=None:
                              davidson_shared.real_eig_space(
                                  param.MAX_MEMORY, nroots, n_pair)[0])
# The baseline predates the Davidson's timings; the space is then unknown (-1)
# and only the values are compared.
TIMED = 'timings' in inspect.signature(solve_casida_davidson).parameters

# The parent's Davidson divided by d; this tree's default is the screened
# diagonal, so it is asked for d where it takes the keyword.
# the loop's name in the commit under test (private in older ones)
RUN = getattr(davidson, 'run_davidson', None) or davidson._run_davidson
BARE = ({{'preconditioner': 'bare'}} if 'preconditioner' in
        inspect.signature(RUN).parameters else {{}})

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
# The parent took the Davidson's choices as keywords, defaulting to a 1e-5
# residual and 100 cycles; this tree reads them off one Numerics, whose
# defaults are the chain's, so the parent's are named here.
if 'numerics' in inspect.signature(solve_casida_davidson).parameters:
    try:
        from src.Base.numerics.numerics import Numerics
    except ImportError:  # an older tree's module layout
        from src.Base.numerics import Numerics

    def solve_kw(tol):
        return dict(numerics=Numerics(bse=dict(
            {{'nroots': 5, 'conv_tol': 1e-5, 'max_cycle': 100,
              'preconditioner': 'bare'}}, **tol)))
else:
    def solve_kw(tol):
        return dict(nroots=5, **tol, **BARE)
for name, tol in (('water_default', {{}}),
                  ('water_chain', {{'conv_tol': BSE_DAVIDSON_CONV_TOL}})):
    t, stats = {{}}, {{}}
    record(name, *solve_casida_davidson(lr, nocc, polarizability='BSE',
                                        W_aux=w_aux, stats=stats, **timed(t),
                                        **solve_kw(tol)), t, stats)

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
record('model', *RUN(apply_AB, diag, 5,
                                        BSE_DAVIDSON_CONV_TOL, 100, None,
                                        stats=stats, **timed(t), **BARE), t,
                                        stats)
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
    _, _, _, info = solve_bse_df(mf, mol, nocc, numerics=Numerics(bse={'nroots': 2}), probe=False,
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


#: Relative noise on the operator's inputs, a round-off-sized move that no
#: physics sees (the G0W0 energies, DF factor and static W each carry 1e-10).
#: It keeps the symmetries the inputs carry, the DF factor's in p, q and W's:
#: elementwise noise breaks them at 1e-10, the block action and
#: `build_casida_matrices` read the broken parts differently, and the roots of
#: the two then sit up to 1.3e-12 Ha apart, outside DENSE_TOL (measured on the
#: neighbours 1 and 2; symmetric noise 3e-15 to 1.05e-13 Ha on the first 11).
STALL_NOISE = 1e-10
#: Operators tried, the unperturbed one first, for one that stalls.
STALL_CANDIDATES = 40


def real_eig_width(nroots, n_pair, extra=None):
    """real_eig's corrections a cycle, max(nroots, MAX_SPACE_INC), handed to
    the loop in place of its tracked pairs (`davidson.tracked_roots`): the
    width at which the reproduction's pair space runs out."""
    return davidson_shared.real_eig_space(param.MAX_MEMORY, nroots, n_pair)[0]


@pytest.fixture(scope='module')
def stalling(small, serial):
    """An operator on which the 3-root Davidson exhausts the pair space and is
    completed densely, with its dense roots and its serial 3-root solves.

    Whether the last cycle's corrections exactly outnumber the pairs left is
    decided by the degenerate partner test (|w| = 1 to rounding), so one
    operator stalls under one BLAS and not under another: on a cluster's MKL a
    relative 1e-16 move of the inputs toggles it (measured), and
    the solver guarantees the roots, not that path. The reproduction is
    therefore the first of its round-off neighbours that stalls at both guess
    factors, the unperturbed operator where it does. With the trial space
    projected twice (`trial_space.gram_schmidt_rows`) the unperturbed
    operator converges on the workstation without running out of pairs, and
    the neighbour 2 stalls. Every solve here runs at real_eig's width
    (`real_eig_width`)."""
    cases = [case for case in CASES if case[0] == 3]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(davidson, 'tracked_roots', real_eig_width)
        return _first_stall(small, cases)


def _first_stall(small, cases):
    for seed in range(STALL_CANDIDATES):
        op = dict(small)
        if seed:
            rng = np.random.default_rng(seed)
            for key in ('eps', 'coeff', 'w_aux'):
                op[key] = small[key] * (1 + symmetric_noise(
                    rng.standard_normal(small[key].shape)))
            op['lr'] = LinearResponseSolver(op['eps'], coeff_df=op['coeff'],
                                            spin_mode='restricted')
            op['dense'] = dense_roots(op['lr'], op['nocc'], op['w_aux'])
        runs = {case: solve(op, *case) for case in cases}
        if all(run[3]['davidson_dense_completion'] for run in runs.values()):
            return dict(op, serial=runs)
    pytest.fail(f'none of {STALL_CANDIDATES} neighbours of the reproduction '
                'stalls the 3-root Davidson: the stall this file gates is gone')


@pytest.fixture(scope='module')
def converging(tmp_path_factory):
    """The converging solves on `BASELINE_COMMIT` and on this tree, each in
    its own process."""
    out = tmp_path_factory.mktemp('davidson_small_pair_space')
    tar = out / f'{BASELINE_COMMIT}.tar'
    require_commit(REPO, BASELINE_COMMIT)
    unpack_commit(REPO, BASELINE_COMMIT, tar, out / 'baseline')
    install_current_radii(out / 'baseline')
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


def symmetric_noise(z):
    """STALL_NOISE times the standard normals `z`, symmetrized over the last
    two axes where the input is a matrix or a stack of them (W, the DF
    factor), so that the noise keeps the symmetry of the input it scales."""
    if z.ndim >= 2:
        z = 0.5 * (z + np.swapaxes(z, -1, -2))
    return STALL_NOISE * z


def dense_roots(lr, nocc, w_aux):
    """Every Casida root of production's dense builder, which owes nothing to
    the block action: omega^2 from (A-B)^1/2 (A+B) (A-B)^1/2."""
    A, B = lr.build_casida_matrices(nocc, lBSE=True, W_aux=w_aux)
    w, U = np.linalg.eigh(A - B)
    s = (U * np.sqrt(w)) @ U.T
    return np.sqrt(np.linalg.eigvalsh(s @ (A + B) @ s))


def solve(small, nroots, guess_factor, comm=None, **kw):
    """(omega, X, Y, stats) of the reproduction on this rank's own copies,
    divided by the bare d unless `kw` asks otherwise: the default
    preconditioner is the screened diagonal, on which the reproduction
    converges without exhausting its pair space."""
    numerics = Numerics(bse={'nroots': nroots,
                             'conv_tol': BSE_DAVIDSON_CONV_TOL,
                             'guess_factor': guess_factor,
                             'preconditioner': kw.pop('preconditioner',
                                                      'bare')})
    lr = LinearResponseSolver(np.array(small['eps']),
                              coeff_df=np.array(small['coeff']),
                              spin_mode='restricted')
    stats = {}
    omega, X, Y = solve_casida_davidson(
        lr, small['nocc'], numerics=numerics, polarizability='BSE',
        W_aux=np.array(small['w_aux']), stats=stats, comm=comm, **kw)
    return omega, X, Y, stats


def residual(lr, nocc, w_aux, omega, X, Y):
    """|[A X + B Y - omega X; B X + A Y + omega Y]| per root, through the
    block action."""
    act, diag = block_actions.block_action(lr, nocc, 'BSE', w_aux, None)
    shape = (-1,) + diag.shape
    Ax, Bx = act(X.T.reshape(shape))
    Ay, By = act(Y.T.reshape(shape))
    n = len(omega)
    top = (Ax + By).reshape(n, -1) - omega[:, None] * X.T
    bot = (Bx + Ay).reshape(n, -1) + omega[:, None] * Y.T
    return np.sqrt(np.sum(top**2, axis=1) + np.sum(bot**2, axis=1))


@pytest.mark.parametrize('nroots,guess_factor', CASES)
def test_the_small_pair_space_converges_to_the_dense_roots(small, serial,
                                                           stalling, nroots,
                                                           guess_factor):
    """Every root at the tolerance and on the dense one, whichever way the
    solve got there: 3 roots on `stalling`'s operator, which stops them at 33
    of the 40 pairs and is completed densely (the completion is asserted
    to have fired, or the case would test the plain path), 2 roots on the
    reproduction itself, where completed or not is the iteration's path and
    not a guarantee."""
    op = stalling if nroots == 3 else small
    omega, X, Y, stats = (stalling['serial'] if nroots == 3 else serial)[
        (nroots, guess_factor)]
    r = residual(op['lr'], op['nocc'], op['w_aux'], omega, X, Y)
    assert r.max() <= BSE_DAVIDSON_CONV_TOL, r
    assert np.abs(omega - op['dense'][:nroots]).max() <= DENSE_TOL
    if nroots == 3:
        assert stats['davidson_dense_completion']


def test_solve_bse_df_completes_the_reproduction(small):
    """The entry point that stalled on the reproduction at real_eig's width,
    at the Davidson's own: no warning, the dense
    roots, and a record of how they were reached that agrees with itself -- on
    the bare d, the default preconditioner being the screened diagonal."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        omega, _, _, info = solve_bse_df(small['mf'], small['mol'],
                                         small['nocc'], numerics=Numerics(bse={'nroots': 3, 'conv_tol': BSE_DAVIDSON_CONV_TOL, 'preconditioner': 'bare'}),
                                         progress=False)
    assert not [w for w in record if 'Davidson' in str(w.message)]
    assert np.abs(omega - small['dense'][:3]).max() <= DENSE_TOL
    # whether the solve stalled is the iteration's path under this BLAS (see
    # `stalling`); what it guarantees is that a completed solve says so and
    # holds the whole pair space, and an uncompleted one stays inside it
    completed = info['stats']['davidson_dense_completion']
    space = info['timings']['davidson_subspace_max']
    assert (space == small['dense'].size) if completed else (
        space <= small['dense'].size)


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('nroots,guess_factor', CASES)
def test_every_rank_returns_rank_zeros_roots(small, serial, size, nroots,
                                             guess_factor, monkeypatch):
    """Rank 0's roots and vectors on every rank, bitwise, from rank 0's
    iteration: the DF action runs whole on every rank on lockstepped inputs,
    and the dense completion builds its A and B through it.

    Rank 0's roots are held to the Davidson's resolution of the serial ones,
    not to their bits, and its path is not the serial one: over ranks the
    trial space is cut by pair rows (`trial_space.casida_rows`), here on
    SPLIT_TILE tiles that give every rank rows of its own, and its sums over
    the pair index re-associate with the rank count. Where the pair space is
    exhausted the partner test is degenerate -- the last corrections are
    their own partners, |w| = 1 to rounding -- so a re-associated overlap
    may keep one direction the serial solve drops, and the solve then fills
    the last pairs one a cycle and converges on the complete space instead
    of completing densely, on the same roots."""
    monkeypatch.setattr(trial_space, 'DAVIDSON_PAIR_TILE', SPLIT_TILE)
    out = run_simulated(lambda comm: solve(small, nroots, guess_factor, comm),
                        size)
    om0, X0, Y0, st0 = out[0]
    r = residual(small['lr'], small['nocc'], small['w_aux'], om0, X0, Y0)
    assert r.max() <= BSE_DAVIDSON_CONV_TOL, r
    assert np.abs(om0 - small['dense'][:nroots]).max() <= DENSE_TOL
    om_s, X_s, Y_s, st_s = serial[(nroots, guess_factor)]
    assert relative(om0, om_s) <= roots_resolution(small['eps'],
                                                   small['nocc'], om_s)
    for omega, X, Y, stats in out:
        assert bitwise(omega, om0) and bitwise(X, X0) and bitwise(Y, Y0)
        for key in ('davidson_vind_calls', 'davidson_block_actions',
                    'davidson_dense_completion'):
            assert stats[key] == st0[key], key


def test_a_breakdown_at_the_pair_space_limit_is_completed_densely(
        breakdown, monkeypatch):
    """The projected (A-B) block losing definiteness once the pair space ran
    out -- pyscf's real_eig did after the partner test kept 3 pairs for the 1
    left; the Casida loop restarts there on its tracked pairs instead, so the
    breakdown is planted at that cycle -- while (A-B) itself is positive
    definite: the dense completion, not the instability error, is the answer.
    On the bare d, asked for: the default preconditioner is the screened
    diagonal."""
    b = breakdown
    n_pair = b['nocc'] * (len(b['lr'].eps) - b['nocc'])
    solver = trial_space.TDDFT_subspace_eigen_solver

    def breaks_at_the_limit(a, *args):
        if len(a) >= n_pair - 1:
            raise np.linalg.LinAlgError('planted: projected (A-B) indefinite')
        return solver(a, *args)

    monkeypatch.setattr(trial_space, 'TDDFT_subspace_eigen_solver',
                        breaks_at_the_limit)
    stats = {}
    omega, X, Y = solve_casida_davidson(
        b['lr'], b['nocc'],
        numerics=Numerics(bse={'nroots': 4, 'guess_factor': 1,
                               'conv_tol': DEFAULT_CONV_TOL,
                               'preconditioner': 'bare'}),
        polarizability='BSE', W_aux=b['w_aux'], stats=stats)
    assert stats['davidson_dense_completion']
    assert np.abs(omega - b['dense'][:4]).max() <= DENSE_TOL
    r = residual(b['lr'], b['nocc'], b['w_aux'], omega, X, Y)
    assert r.max() <= DEFAULT_CONV_TOL


@pytest.mark.parametrize('nroots,guess_factor', CASES)
def test_the_screened_default_converges_to_the_dense_roots(small, breakdown,
                                                           nroots,
                                                           guess_factor):
    """By default -- the screened diagonal -- the reproduction and the
    breakdown converge to the dense roots, the reproduction without the
    dense completion, and `solve_bse_df` likewise."""
    omega, X, Y, stats = solve(small, nroots, guess_factor,
                               preconditioner=davidson.DAVIDSON_PRECONDITIONER)
    assert davidson.DAVIDSON_PRECONDITIONER == 'screened'
    r = residual(small['lr'], small['nocc'], small['w_aux'], omega, X, Y)
    assert r.max() <= BSE_DAVIDSON_CONV_TOL, r
    assert np.abs(omega - small['dense'][:nroots]).max() <= DENSE_TOL
    assert not stats['davidson_dense_completion']
    if (nroots, guess_factor) != CASES[-1]:
        return
    omega, _, _, info = solve_bse_df(small['mf'], small['mol'], small['nocc'],
                                     numerics=Numerics(bse={'nroots': 3, 'conv_tol': BSE_DAVIDSON_CONV_TOL}),
                                     progress=False)
    assert np.abs(omega - small['dense'][:3]).max() <= DENSE_TOL
    assert not info['stats']['davidson_dense_completion']
    b = breakdown
    omega, X, Y = solve_casida_davidson(
        b['lr'], b['nocc'],
        numerics=Numerics(bse={'nroots': 4, 'guess_factor': 1,
                               'conv_tol': DEFAULT_CONV_TOL}),
        polarizability='BSE', W_aux=b['w_aux'])
    assert np.abs(omega - b['dense'][:4]).max() <= BREAKDOWN_DENSE_TOL
    r = residual(b['lr'], b['nocc'], b['w_aux'], omega, X, Y)
    assert r.max() <= DEFAULT_CONV_TOL


@pytest.mark.parametrize('case', ['water_default', 'water_chain', 'model'])
def test_converging_solves_are_the_baseline_commits(converging, case):
    """The completion acts on failed solves only. water/cc-pVDZ fills its 95
    pairs and converges on them, so a rule that acted whenever the space ran
    out would have moved it, and it is the baseline's to the class bars (the
    projection twice reassociates the trial space's sums); the 500-pair
    model stays under half its space and never reaches the completion. Both
    trees divide by the bare d, this one asked for it: the default
    preconditioner is the screened diagonal. This tree runs at real_eig's
    width (`real_eig_width`), the baseline's path: at its own the roots are
    the same to the class bar, the vectors not (water's X 8.8e-7 at conv_tol
    1e-5 and 1.9e-10 at BSE_DAVIDSON_CONV_TOL, measured), a converged
    iteration's other path to them."""
    base, cur = converging['baseline'], converging['current']
    assert not bool(cur[f'{case}_dense'])
    space = int(cur[f'{case}_space'])
    if case == 'model':
        assert 2 * space <= 500
        return
    for part, quantity in (('omega', 'root'), ('X', 'vector'),
                           ('Y', 'vector')):
        assert_within_class(cur[f'{case}_{part}'], base[f'{case}_{part}'],
                            quantity=quantity,
                            unit='Ha' if quantity == 'root' else '1',
                            label=part)
    assert space == 95


def test_the_pair_space_limit_names_itself_when_not_completed(stalling,
                                                              monkeypatch):
    """Above BSE_DENSE_MAX_NOV the solve is not completed, and the stop is
    reported as the pair space's, with what to change -- a tolerance moves
    nothing there. On `stalling`'s operator at real_eig's width, the one that
    stops at the pair space under this BLAS: the loop is the same with and without the
    completion, which acts after it."""
    small = stalling
    n_pair = small['dense'].size
    monkeypatch.setattr(davidson, 'tracked_roots', real_eig_width)
    monkeypatch.setattr(davidson, 'BSE_DENSE_MAX_NOV', n_pair - 1)
    with pytest.warns(RuntimeWarning) as record:
        solve(small, 3, 2, refuse_unconverged=False)
    message = ' '.join(str(w.message) for w in record
                       if 'Davidson left' in str(w.message))
    assert re.search(rf'held \d+ of the {n_pair} particle-hole pairs', message)
    assert 'fewer roots, raise guess_factor, or use the dense solver' in message
    assert 'loosen conv_tol' not in message
    with pytest.raises(RuntimeError, match='Refused.*fewer roots'):
        solve(small, 3, 2, refuse_unconverged=True)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
