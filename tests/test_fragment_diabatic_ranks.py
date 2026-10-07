"""The fragment-diabatic BSE and the gradient of its diabatic matrix under real
ranks, on sliced factors with the grid BSE adjoint, against the serial run.

    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
        mpirun -n 2 python tests/test_fragment_diabatic_ranks.py
    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
        mpirun -n 2 python tests/test_fragment_diabatic_ranks.py --skew

Script, not a pytest module, like tests/test_mpi_routes.py: it initializes
MPI, which a sandboxed test runner cannot, and exits with a status.
`main(comm)` takes the communicator.

The system is tests/test_fragment_diabatic.py's offset ethylene dimer
(cc-pVDZ, RHF, BSE@G0W0 on the ISDF/space-time chain, singlet, one site state
per monomer), Tamm-Dancoff and full. Both dense limits of
`fragment_bse.dense_limit` are set below the smallest block, so every solve is
iterative: the Davidson diabats (Casida Davidson for the full kernel), the
pole guard's Newton on Q, the batched conjugate-gradient resolvent and the
MINRES diabat response. The BSE action and the reverse chain are divided
over the ranks (`sliced=True`, `bse_adjoint='grid'`); replicated steps take
rank 0's result through `lockstep`, and iterations and refusals stop where
rank 0 says (`krylov.root_driven_solve` for the localization, its response,
the Davidsons and the diabat responses; rank 0's Newton step and
conjugate-gradient mask).

Checks, per kernel:
  * every rank applied the collective BSE action the same number of times,
    in the same widths: the collectives paired
  * the partition -- A_eff, Sigma, dSigma/dOmega, the guard's lowest Q
    eigenvalue -- and the element gradients (both sites and their coupling)
    hold the same bits on every rank, and match rank 0's serial run (every
    rank computes the serial reference inside `distributed(None)`; rank 0's
    is broadcast, so the reference is one calculation)
  * the serial grid adjoint matches the serial explicit adjoint, the
    realization tests/test_fragment_diabatic.py gates against finite
    differences
and once, over both kernels: every `lockstep(...)` and `root_driven_solve(...)`
call in fragment_bse, fragment_diabatic and fragment_localization ran on this
rank (the call sites are read off the source, so a new one is covered or the
check fails; the guard Newton's and the Casida Davidson's are the full
kernel's alone).

Without `--skew` every rank holds rank 0's mean field (`lockstep_mean_field`)
and runs rank 0's code, which on one machine makes every replicated step
agree to the bit by construction. `--skew` makes every rank but 0 disagree on
purpose, in the two ways two nodes do:
  * its OWN mean field, the way two nodes' SCFs differ: a third of the
    orbitals change sign and the orbital energies move by one ulp. Every
    result must still be rank 0's, which holds only if everything downstream
    -- the localization, the canonical-gauge term, the localization
    response, the chain -- reads rank 0's mean field; and the audit has to
    SEE the repair (`mismatched_calls` > 0 on some rank).
  * its OWN stopping decisions, far apart rather than one ulp apart: every
    iterative step is handed a tolerance `DISAGREE_FACTOR` looser on rank 1
    (the localization, its MINRES response, the Davidsons, the diabat MINRES
    responses, the conjugate-gradient resolvent) and the guard Newton a step
    tolerance of `DISAGREE_NEWTON_TOL`. A replicated solver would stop rank 1
    early and leave rank 0 waiting in the next action, and a replicated
    residual check would refuse on rank 1 alone and abort the job. Here the
    action counts must still agree, the root-driven steps must never have run
    on rank 1, and the resolvent -- replicated, on rank 0's stopping mask --
    must have run there with its loose tolerance.

`--isdf-scf` runs instead the production chain row (`potential_energy_surface`:
LRC-wPBEh on an ISDF-K mean field, full BSE, SOP residues, the admitted
quasiparticle set, grid G1, sliced factors, row fit, grid adjoint) on the same
dimer, its mean field converged over the ranks: J/K answered from each rank's
tiles of the fit, and refused outside `distributed_fock`. The localization,
the partition and the element gradients run on that mean field; the checks
are that it is distributed on every rank (`distributed_handles`, the
ISDFJK's distributed twin), that every rank applied the BSE action the same
number of times, and that the partition and the gradients are one set of
bits on every rank and match rank 0's serial run, whose ISDF-K SCF is
converged serially: two SCFs, hence `ISDF_PARTITION_BAR` (Ha),
`ISDF_DSIGMA_BAR` (dSigma/dOmega, dimensionless) and the gradient bar.

`--tda-only` runs the Tamm-Dancoff kernel alone; `--watchdog SECONDS` sets
the wall time after which a hung run is aborted (default three hours).

Bars: rank 0's serial run and the distributed one differ by the reduction
order of the sliced BSE action and of the grid adjoint, which the iterative
solves (tolerance `FRAGMENT_SOLVE_TOL`) carry into their solutions. The
`reduced` standard of tests/test_mpi_routes.py: 1e-10 Ha on the partition,
1e-8 Ha/Bohr on a gradient (the ISDF gradient's reproducibility floor).
"""
import os
import sys

SKEW = '--skew' in sys.argv
TDA_ONLY = '--tda-only' in sys.argv
ISDF_SCF = '--isdf-scf' in sys.argv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ast
import copy
import hashlib
import threading
import time
import warnings

import numpy as np
from pyscf import dft, gto, lib

from src.Base import fragment_localization
from src.Base.constants import (SCF_DIFFERENTIABLE_CONV_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.distributed_df import distributed_handles
from src.Base.isdf_jk import isdf_jk
from src.Base.separable_ri import resolve_isdf_grid
from src.Base.utils.mpi_grid import (broadcast, current_comm, distributed,
                                     grid_comm, lockstep_mean_field,
                                     lockstep_stats)
from src.gradients import fragment_diabatic
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.fragment_diabatic import DiabaticGradient
from src.properties import fragment_bse
from src.properties.surfaces import potential_energy_surface
from tests.test_fragment_diabatic import (AUX, BASIS, DIMER, FRAGMENTS, SITES,
                                          factory)

#: The `reduced` bars (module docstring).
PARTITION_BAR = 1e-10
GRADIENT_BAR = 1e-8
#: `--isdf-scf`: the reference's ISDF-K SCF is converged serially and the
#: distributed one over the ranks, so the partition carries two SCFs'
#: convergence on top of the reduction order.
ISDF_PARTITION_BAR = 1e-9
#: dSigma/dOmega is dimensionless, -y^T S y of the resolvent vectors, whose
#: conjugate-gradient tolerance carries the two SCFs' difference into it at a
#: larger multiple than into Sigma itself.
ISDF_DSIGMA_BAR = 1e-8
ISDF_XC = 'lrc-wpbeh'
#: Serial grid against serial explicit adjoint: the fold carries the last
#: bits of its seeds through the fit adjoint (Gram condition ~1e8), see
#: tests/test_mpi_routes.py `grid_adjoint_routes`.
ADJOINT_BAR = 1e-8
#: Both dense limits, below every block of the dimer.
ITERATIVE_LIMIT = 10
#: The elements whose gradients are compared: both sites and their coupling.
ELEMENTS = ((0, 0), (1, 1), (0, 1))
#: Wall seconds after which a rank aborts the job: a replicated solver whose
#: ranks fall out of step waits in a collective forever.
WATCHDOG_SECONDS = (int(sys.argv[sys.argv.index('--watchdog') + 1])
                    if '--watchdog' in sys.argv else 3 * 3600)
MODULES = (fragment_bse, fragment_diabatic, fragment_localization)
#: What keeps the ranks in agreement: rank 0's data (`lockstep`), rank 0's
#: iterations and refusals (`root_driven_solve`).
RANK_AGREEMENT = ('lockstep', 'root_driven_solve')
#: How much looser rank 1's solver tolerances are under `--skew`: a replicated
#: solver would stop there many iterations before rank 0.
DISAGREE_FACTOR = 1e4
DISAGREE_NEWTON_TOL = 1e-2


class Gate:
    """Each rank's verdicts, gathered at the end (tests/test_mpi_routes.py)."""

    def __init__(self, comm):
        self.comm = comm
        self.rank = 0 if comm is None else comm.Get_rank()
        self.size = 1 if comm is None else comm.Get_size()
        self.verdicts = []

    def say(self, text):
        if self.rank == 0:
            print(text, flush=True)

    def info(self, text):
        self.say(f'  [info] {text}')

    def check(self, ok, label, detail=''):
        ok = bool(ok)
        self.verdicts.append((ok, label, detail))
        self.say(f"  [{'ok' if ok else 'FAIL'}] {label}"
                 + (f'  ({detail})' if detail else ''))
        return ok

    def everyone(self, obj):
        return self.comm.allgather(obj) if self.comm is not None else [obj]

    def distinct(self, *arrays):
        return len(set(self.everyone(digest(*arrays))))

    def finish(self):
        failures = [(r, label, detail)
                    for r, own in enumerate(self.everyone(self.verdicts))
                    for ok, label, detail in own if not ok]
        if self.rank == 0:
            for r, label, detail in failures:
                if r != 0:
                    print(f'  [FAIL on rank {r}] {label}'
                          + (f'  ({detail})' if detail else ''))
            print('\n' + ('All checks passed on every rank.' if not failures
                          else 'FAILURES above.'), flush=True)
        return 0 if not failures else 1


def digest(*arrays):
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, float)).tobytes())
    return h.hexdigest()[:12]


def lockstep_sites():
    """{(module file, first line, last line)} of every call of a
    RANK_AGREEMENT function in MODULES' source."""
    sites = set()
    for module in MODULES:
        with open(module.__file__) as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in RANK_AGREEMENT):
                sites.add((os.path.basename(module.__file__), node.lineno,
                           node.end_lineno))
    return sites


def sites_reached(sites, seen):
    """The call sites one of the `seen` (file, line) frames lies in."""
    return {s for s in sites
            if any(f == s[0] and s[1] <= n <= s[2] for f, n in seen)}


class LockstepRecorder:
    """Wraps the RANK_AGREEMENT functions in MODULES and records the calling
    line."""

    def __init__(self):
        self.seen = set()
        self._saved = []

    def __enter__(self):
        for module in MODULES:
            for name in RANK_AGREEMENT:
                if not hasattr(module, name):
                    continue
                real = getattr(module, name)
                self._saved.append((module, name, real))

                def wrapped(*args, _real=real, **kwargs):
                    frame = sys._getframe(1)
                    self.seen.add((os.path.basename(frame.f_code.co_filename),
                                   frame.f_lineno))
                    return _real(*args, **kwargs)
                setattr(module, name, wrapped)
        return self

    def __exit__(self, *exc):
        for module, name, real in self._saved:
            setattr(module, name, real)


class ActionCounter:
    """The widths of this rank's applications of the collective BSE action,
    in order (`BSEOperator.apply`)."""

    def __enter__(self):
        self.widths = []
        self._real = real = fragment_bse.BSEOperator.apply

        def counted(op, x):
            self.widths.append(int(np.shape(x)[1]) if np.ndim(x) > 1 else 1)
            return real(op, x)
        fragment_bse.BSEOperator.apply = counted
        return self

    def __exit__(self, *exc):
        fragment_bse.BSEOperator.apply = self._real


class SolverProbe:
    """Counts this rank's calls of every solver that iterates on the
    collective action; with `loosen`, hands each a tolerance DISAGREE_FACTOR
    looser and the guard Newton a step tolerance of DISAGREE_NEWTON_TOL.

    (module, name, the tolerance's keyword, its position): the keyword when
    the call site passes it by name, the position otherwise.
    """
    SOLVERS = ((fragment_bse, 'solve_symmetric', 'tol_residual', None),
               (fragment_bse, '_casida_davidson', 'tol', 4),
               (fragment_bse, '_batched_cg', 'tol', 3),
               (fragment_diabatic, 'minres', 'rtol', None),
               (fragment_localization, '_localize', 'conv_tol', None))
    #: the diabat and localization responses both call `minres`
    ROOT_DRIVEN = ('solve_symmetric', '_casida_davidson', 'minres',
                   '_localize')

    def __init__(self, loosen):
        self.loosen = loosen
        self.calls = {name: 0 for _, name, _, _ in self.SOLVERS}
        self._saved = []

    def __enter__(self):
        for module, name, key, index in self.SOLVERS:
            real = getattr(module, name)
            self._saved.append((module, name, real))

            def wrapped(*args, _real=real, _name=name, _key=key, _index=index,
                        **kwargs):
                self.calls[_name] += 1
                if self.loosen:
                    if _key in kwargs:
                        kwargs[_key] *= DISAGREE_FACTOR
                    else:
                        args = list(args)
                        args[_index] *= DISAGREE_FACTOR
                return _real(*args, **kwargs)
            setattr(module, name, wrapped)
        if self.loosen:
            self._saved.append((fragment_bse, 'FRAGMENT_GUARD_NEWTON_TOL',
                                fragment_bse.FRAGMENT_GUARD_NEWTON_TOL))
            fragment_bse.FRAGMENT_GUARD_NEWTON_TOL = DISAGREE_NEWTON_TOL
        return self

    def __exit__(self, *exc):
        for module, name, real in self._saved:
            setattr(module, name, real)


def watchdog(comm, seconds):
    """Abort every rank once `seconds` have passed: a hang is a failure."""
    if comm is None or comm.Get_size() == 1:
        return

    def fire():
        sys.stderr.write(f'rank {comm.Get_rank()}: watchdog after {seconds} s, '
                         'aborting every rank\n')
        sys.stderr.flush()
        comm.Abort(2)
    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()


def own_mean_field(gate, mf0):
    """A copy of rank 0's mean field for one kernel's runs; under `--skew`,
    on every rank but 0, a third of its orbitals change sign and its orbital
    energies move one ulp. A copy per kernel, because the chain repairs the
    mean field it is handed in place (`lockstep_mean_field`)."""
    mf = copy.copy(mf0)
    mf.mo_coeff = np.array(mf0.mo_coeff, copy=True)
    mf.mo_energy = np.array(mf0.mo_energy, copy=True)
    mf.mo_occ = np.array(mf0.mo_occ, copy=True)
    if SKEW and gate.rank > 0:
        flip = np.arange(mf.mo_coeff.shape[1]) % 3 == 1
        mf.mo_coeff[:, flip] *= -1.0
        mf.mo_energy += np.spacing(np.abs(mf.mo_energy))
    return mf


def evaluate(mol, mf, tda, adjoint, sliced):
    """The partition and the ELEMENTS' gradients of one run."""
    chain = ExcitedStateChain(mol, factory, bse_tda=tda, auxbasis=AUX, mf=mf,
                              sliced=sliced, bse_adjoint=adjoint)
    return measure(DiabaticGradient(chain, FRAGMENTS, SITES, mf=mf,
                                    route='isdf'))


def isdf_factory(mol):
    """An LRC-wPBEh ISDF-K mean field at grid G1: converged here serially,
    built and handed to the chain unconverged under ranks, which converges it
    over them (`factor_chain.converged_factory`)."""
    mf = dft.RKS(mol, xc=ISDF_XC)
    elements = sorted({mol.atom_symbol(i) for i in range(mol.natm)})
    counts, n_start = resolve_isdf_grid('G1', str(mol.basis), elements,
                                        auxbasis=AUX)
    mf = isdf_jk(mf, auxbasis=AUX, counts=counts, n_start=n_start)
    mf.conv_tol, mf.conv_tol_grad = (SCF_DIFFERENTIABLE_CONV_TOL,
                                     SCF_DIFFERENTIABLE_GRAD_TOL)
    mf.max_cycle, mf.verbose = 200, 0
    comm = current_comm()
    if comm is not None and comm.Get_size() > 1:
        return mf
    return mf.run()


def isdf_evaluate(mol):
    """`evaluate` on the production chain row over an ISDF-K mean field, the
    chain's own; with the gradient object, whose mean field is inspected."""
    chain = potential_energy_surface(
        mol, isdf_factory, ground_state=GroundState('dft', ISDF_XC),
        excitation=Excitation('singlet', root=1, kernel='bse'),
        chi0='space-time', residues='sop', solver='davidson',
        factorization='isdf', qp_states=QPStates('admitted'),
        grid_accuracy='G1', nroots=5, sliced=True, fit='rows',
        bse_adjoint='grid')
    g = DiabaticGradient(chain, FRAGMENTS, SITES, route='isdf')
    return measure(g), g


def measure(g):
    """The partition and the ELEMENTS' gradients of a `DiabaticGradient`."""
    part = g.partition
    out = dict(a_eff=np.array(part.a_eff), sigma=np.array(part.sigma),
               dsigma=np.array(part.dsigma), q_lowest=float(part.q_lowest),
               omega0=float(part.omega0), labels=list(part.labels))
    for ab in ELEMENTS:
        grad, diags = g.element(*ab)
        out[ab] = np.array(grad)
        out[ab, 'parts'] = {k: np.array(diags[k])
                            for k in ('chain', 'canonical', 'localization')}
    return out


def partition_arrays(out):
    return (out['a_eff'], out['sigma'], out['dsigma'],
            np.atleast_1d(out['q_lowest']))


def largest(ref, got):
    return max(float(np.abs(np.asarray(a) - np.asarray(b)).max())
               for a, b in zip(ref, got))


def run_kernel(gate, comm, mol, mf, tda):
    name = 'TDA' if tda else 'full BSE'
    gate.say(f'\n-- {name}, {gate.size} rank(s)'
             + (', a mean field per rank' if SKEW else ''))
    t0 = time.perf_counter()
    with distributed(None):
        grid_ref = evaluate(mol, mf, tda, 'grid', sliced=False)
        explicit_ref = evaluate(mol, mf, tda, 'explicit', sliced=False)
    grid_ref = broadcast(grid_ref, comm)
    explicit_ref = broadcast(explicit_ref, comm)
    t1 = time.perf_counter()

    lockstep_stats(reset=True)
    disagree = SKEW and gate.rank > 0
    with LockstepRecorder() as rec, ActionCounter() as actions, \
            SolverProbe(loosen=disagree) as probe, \
            distributed(comm, audit=True):
        out = evaluate(mol, mf, tda, 'grid', sliced=True)
    stats = lockstep_stats(reset=True)
    t2 = time.perf_counter()
    gate.info(f'serial references {t1 - t0:.0f} s, distributed run '
              f'{t2 - t1:.0f} s; labels {out["labels"]}')

    widths = gate.everyone(tuple(actions.widths))
    gate.check(len(set(widths)) == 1,
               f'{name}: every rank applied the collective BSE action the '
               'same number of times in the same widths',
               f'actions per rank {[len(w) for w in widths]}, columns per rank '
               f'{[sum(w) for w in widths]}')
    calls = gate.everyone(probe.calls)
    gate.info(f'{name}: solver calls per rank {calls}')
    if SKEW and gate.size > 1:
        ran_on_workers = {k: [c[k] for c in calls[1:]]
                          for k in SolverProbe.ROOT_DRIVEN}
        cg = [c['_batched_cg'] for c in calls]
        gate.check(not any(map(any, ran_on_workers.values()))
                   and calls[0]['solve_symmetric'] > 0
                   and calls[0]['minres'] > 0 and calls[0]['_localize'] > 0
                   and len(set(cg)) == 1 and cg[0] > 0,
                   f'{name}: rank 1 handed looser tolerances '
                   f'(x{DISAGREE_FACTOR:.0e}, Newton {DISAGREE_NEWTON_TOL:.0e})'
                   ' and still in step: the root-driven solvers and the '
                   'localization ran on rank 0 alone, the conjugate-gradient '
                   "resolvent on every rank on rank 0's mask",
                   f'root-driven solver calls on ranks > 0 {ran_on_workers}, '
                   f'resolvent calls per rank {cg}')

    reached = sites_reached(lockstep_sites(), rec.seen)
    gate.info(f'{name}: rank-agreement call sites reached '
              f'{sorted((f, a) for f, a, _ in reached)}')

    mism = gate.everyone(int(stats.get('mismatched_calls', 0)))
    audited = gate.everyone(int(stats.get('audited_calls', 0)))
    diff = gate.everyone(float(stats.get('max_abs_diff', 0.0)))
    detail = (f'audited calls per rank {audited}, rank copies repaired {mism}, '
              f'largest repair {max(diff):.2e}')
    if SKEW and gate.size > 1:
        gate.check(sum(mism) > 0, f'{name}: the ranks\' own mean fields '
                   'differed and the locksteps repaired it (the run can see a '
                   'cross-node divergence)', detail)
    else:
        gate.info(f'{name}: {detail}')

    n = gate.distinct(*partition_arrays(out))
    d = largest(partition_arrays(grid_ref), partition_arrays(out))
    gate.check(n == 1 and d < PARTITION_BAR,
               f'{name}: partition (A_eff, Sigma, dSigma, guard) the same bits '
               f"on every rank, == rank 0's serial run",
               f'{n} distinct, |d| {d:.2e} Ha, bar {PARTITION_BAR:.0e}')

    for ab in ELEMENTS:
        label = '/'.join(grid_ref['labels'][k] for k in ab)
        n = gate.distinct(out[ab])
        d = float(np.abs(out[ab] - grid_ref[ab]).max())
        parts = ', '.join(
            f"{k} {np.abs(out[ab, 'parts'][k] - grid_ref[ab, 'parts'][k]).max():.1e}"
            for k in ('chain', 'canonical', 'localization'))
        gate.check(n == 1 and d < GRADIENT_BAR,
                   f'{name}: dE[{label}]/dR the same bits on every rank, == '
                   "rank 0's serial run",
                   f'{n} distinct, |d| {d:.2e} Ha/Bohr ({parts}), |g| '
                   f'{np.abs(grid_ref[ab]).max():.2e}, bar {GRADIENT_BAR:.0e}')
        d = float(np.abs(grid_ref[ab] - explicit_ref[ab]).max())
        gate.check(d < ADJOINT_BAR,
                   f'{name}: dE[{label}]/dR serial, grid adjoint == explicit '
                   'adjoint', f'|d| {d:.2e} Ha/Bohr, bar {ADJOINT_BAR:.0e}')
    return reached


def run_isdf(gate, comm, mol):
    """`--isdf-scf`: the module on an ISDF-K mean field converged over the
    ranks, against rank 0's serial run."""
    gate.say(f'\n-- ISDF-K mean field over {gate.size} rank(s), full BSE')
    t0 = time.perf_counter()
    with distributed(None):
        ref = isdf_evaluate(mol)[0]
    ref = broadcast(ref, comm)
    t1 = time.perf_counter()
    with ActionCounter() as actions, distributed(comm):
        out, g = isdf_evaluate(mol)
    t2 = time.perf_counter()
    gate.info(f'serial reference {t1 - t0:.0f} s, distributed run '
              f'{t2 - t1:.0f} s; labels {out["labels"]}')

    flags = gate.everyone((
        distributed_handles(g.mf, comm) is not None,
        bool(getattr(getattr(g.mf, 'with_df', None), '_distributed_twin',
                     False))))
    gate.check(gate.size == 1 or all(h and t for h, t in flags),
               'the mean field is converged over the ranks on every rank: '
               'its distributed handles, and an ISDFJK that refuses a whole '
               'build', f'(handles, distributed twin) per rank {flags}')
    widths = gate.everyone(tuple(actions.widths))
    gate.check(len(set(widths)) == 1,
               'every rank applied the collective BSE action the same number '
               'of times in the same widths',
               f'actions per rank {[len(w) for w in widths]}, columns per rank '
               f'{[sum(w) for w in widths]}')

    n = gate.distinct(*partition_arrays(out))
    each = {k: largest([ref[k]], [out[k]])
            for k in ('a_eff', 'sigma', 'q_lowest', 'dsigma')}
    gate.check(n == 1 and max(each['a_eff'], each['sigma'], each['q_lowest'])
               < ISDF_PARTITION_BAR and each['dsigma'] < ISDF_DSIGMA_BAR,
               'partition (A_eff, Sigma, dSigma, guard) the same bits on every '
               "rank, == rank 0's serial run",
               f'{n} distinct, |d| A_eff {each["a_eff"]:.1e}, Sigma '
               f'{each["sigma"]:.1e}, guard {each["q_lowest"]:.1e} Ha (bar '
               f'{ISDF_PARTITION_BAR:.0e}); dSigma/dOmega {each["dsigma"]:.1e} '
               f'(bar {ISDF_DSIGMA_BAR:.0e})')
    for ab in ELEMENTS:
        label = '/'.join(ref['labels'][k] for k in ab)
        n = gate.distinct(out[ab])
        d = float(np.abs(out[ab] - ref[ab]).max())
        gate.check(n == 1 and d < GRADIENT_BAR,
                   f'dE[{label}]/dR the same bits on every rank, == rank 0\'s '
                   'serial run',
                   f'{n} distinct, |d| {d:.2e} Ha/Bohr, |g| '
                   f'{np.abs(ref[ab]).max():.2e}, bar {GRADIENT_BAR:.0e}')


def main(comm, kernels=None):
    """Every check on this rank of `comm`; 0 when every rank passed."""
    if kernels is None:
        kernels = (True,) if TDA_ONLY else (True, False)
    gate = Gate(comm)
    watchdog(comm, WATCHDOG_SECONDS)
    warnings.simplefilter('ignore')
    fragment_bse.FRAGMENT_DENSE_MAX = ITERATIVE_LIMIT
    fragment_bse.FRAGMENT_MATRIX_FREE_DENSE_MAX = ITERATIVE_LIMIT
    gate.say(f'fragment-diabatic BSE over {gate.size} rank(s), '
             f'{lib.num_threads()} OpenMP thread(s) per rank'
             + (', a mean field per rank' if SKEW else ''))
    mol = gto.M(atom=DIMER, basis=BASIS, verbose=0)
    if ISDF_SCF:
        run_isdf(gate, comm, mol)
        return gate.finish()
    with distributed(None):
        mf0 = factory(mol)
    with distributed(comm):
        lockstep_mean_field(mf0)
    reached = set()
    for tda in kernels:
        mf = own_mean_field(gate, mf0)
        gate.info('mean field digests per rank: '
                  f'{gate.everyone(digest(mf.mo_coeff, mf.mo_energy))}')
        reached |= run_kernel(gate, comm, mol, mf, tda)
    sites = lockstep_sites()
    missing = sorted(sites - reached)
    detail = (f'{len(reached)} of {len(sites)}'
              + (f', missing (file, first, last line) {missing}'
                 if missing else ''))
    if len(kernels) < 2:
        gate.info(f'rank-agreement call sites, one kernel only: {detail}')
    else:
        gate.say('\n-- coverage')
        gate.check(not missing, 'every lockstep and root_driven_solve call '
                   'site of the three modules ran under the ranks (both '
                   'kernels together)', detail)
    return gate.finish()


if __name__ == '__main__':
    sys.exit(main(grid_comm()[0]))
