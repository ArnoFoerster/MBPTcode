"""The GW/ISDF kernels read the distribution context and lockstep their inputs.

A driver carries no communicator: an entry point runs inside
`distributed(comm)` and every kernel called there without `comm=` finds it
through `current_comm()`. A kernel then makes its OUTPUT identical on every
rank in one of two ways, and these gates hold each to it:

  * inputs that can differ between ranks -- a mean field each rank converged,
    points a rank placed from its own radii and frames, the spectrum a grid
    is sized from -- are a `lockstep` of rank 0's at the kernel's entry, one
    packed call;
  * inputs produced by an upstream kernel (the factors, proj(tau), W) are
    identical by construction and are NOT broadcast again, 10 to 27 GB at the
    chlorophyllide hexamer; an audited run (`distributed(comm, audit=True)`)
    compares their 64-bit digests instead (`mpi_grid.agreement`), and those of
    the kernel's outputs, so a cluster run can show every kernel held one set
    of bits on every rank without moving them.

Gates, water/cc-pVDZ at 2, 3 and 8 simulated ranks:
  * each kernel called WITHOUT `comm` inside `run_simulated` returns the bits
    the same call returns with `comm=` passed -- it found the context rather
    than running serially on every rank -- and the serial answer at this
    suite's standard (bitwise where the route is an output partition);
  * a serial region, `distributed(None)`, inside the context is the serial
    call bitwise and communicates nothing;
  * rank 2's orbital energies and orbitals, each one ulp off, still give every
    rank the factors of the unperturbed run bitwise, and leave every rank
    holding rank 0's mean field; the audit counts exactly that one repair;
  * `agreement` is True on identical bytes and False, on every rank, when one
    rank's array is one ulp off; the digest sees one word anywhere, a swap and
    a +-1 ulp pair; audit-only calls cost nothing outside an audit;
  * an audited run of every kernel here finds nothing to report on identical
    inputs, and reports a factor one ulp off at the first kernel to read it;
  * `lockstep_mean_field` lives below GW: importing `mpi_grid` imports no GW
    module, which is what removes `replicate_mean_field` from the import
    cycle environment -> distributed_df -> GW.space_time -> ... ->
    environment.

SHOWN TO FAIL, each on a backup restored and `cmp`-verified afterwards:

  perturbation                               gates that then fail
  the entry lockstep in `separable_factors`  test_separable_factors_repairs_
  removed                                    a_one_ulp_mean_field at 3 and 8
                                             ranks: rank 2 keeps its own
                                             orbitals; the factors still agree,
                                             the exit lockstep repairs X_mo
  `agreement` returning True unconditionally 8: the three agreement gates at
                                             2, 3 and 8 ranks, the structure
                                             gate, both audit gates here and
                                             test_simulated_ranks' audited GW
                                             window at 2 and 3
  the context fallback in qp_set_gradient    test_kernels_find_the_context at
  removed                                    2, 3 and 8: each rank runs the
                                             serial call
  `replicate_factors` made a no-op           test_lockstep_mean_field_and_the_
                                             re_exports, and test_simulated_
                                             ranks' perturbed-rank BSE at 2
                                             and 3
"""
import copy
import os
import subprocess
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.utils import mpi_grid
from src.Base.utils.grids import gauss_legendre_grid
from src.Base.utils.mpi_grid import (agreement, distributed,
                                     lockstep_mean_field, lockstep_stats,
                                     run_simulated)
from src.Base.utils.time_frequency import TimeFrequencyGrid
from src.SingleReference.GW.space_time import (replicate_factors,
                                               replicate_mean_field,
                                               separable_factors,
                                               solve_qp_energy_space_time)
from src.gradients.qp_space_time import qp_gradient_space_time, qp_set_gradient
from src.gradients.reaction_field_adjoint import static_grid, static_screening
from src.gradients.space_time_adjoint import (polarizability_tau,
                                              rpa_energy_and_adjoint,
                                              screened_interaction_tau,
                                              screened_interaction_tau_backward,
                                              selfenergy_diag,
                                              selfenergy_diag_backward,
                                              sigma_transforms)
from tests.test_isdf_fit_ranks import FIT_TOL
from tests.test_simulated_ranks import NFREQ, NTAU, REL, SIGMA_REL, close

SIZES = [2, 3, 8]
#: Rank sizes with a rank 2 to perturb.
PERTURBED_SIZES = [3, 8]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def water():
    warnings.simplefilter('ignore')
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    factors = separable_factors(mf, mol, auxbasis='cc-pvdz-ri')
    eps = np.asarray(mf.mo_energy, float)
    gap = eps[nocc] - eps[nocc - 1]
    nu, wt = gauss_legendre_grid(NFREQ, w0=gap)
    grid = TimeFrequencyGrid.minimax_split(NTAU, 0.5 * gap, eps[-1] - eps[0],
                                           nu, wt, with_sine=False,
                                           with_inverse=False)
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])
    transforms = sigma_transforms(eps, nocc, grid.tau_points, grid.omega_points,
                                  grid.omega_points, mu=mu)
    return dict(mf=mf, mol=mol, nocc=nocc, factors=factors, X=factors[0],
                D=factors[1], eps=eps, grid=grid, nu=nu, wt=wt, mu=mu,
                transforms=transforms, w_grid=static_grid(eps, nocc),
                states=[nocc - 2, nocc - 1, nocc, nocc + 1],
                sigma_states=[nocc - 1, nocc],
                weights=np.array([0.3, -1.1, 0.8, 0.45]))


def _kernels(w):
    """{name: call(comm)} for every kernel gated here, on the shared fixture.

    Each call copies the arrays a kernel may lockstep in place, so a rank
    thread never writes into another rank's buffer through the fixture.
    """
    nocc = w['nocc']
    proj = polarizability_tau(w['X'], w['D'], w['eps'], nocc, w['grid'],
                              mu=w['mu'])
    ctw = w['transforms'][0]
    wt_bar = np.random.default_rng(3).normal(size=(ctw.shape[0],)
                                             + proj.shape[1:])
    sig_args = (w['X'], w['D'], w['eps'], nocc, w['grid'], w['sigma_states'],
                w['transforms'], w['mu'])
    a_shape = (len(w['sigma_states']), len(w['grid'].omega_points))
    a_re, a_im = np.random.default_rng(4).normal(size=(2,) + a_shape)

    def window(comm):
        t = {}
        qp = solve_qp_energy_space_time(
            w['mf'], w['mol'], nocc, np.array([nocc - 1, nocc]),
            factors=w['factors'], timings=t, comm=comm)
        return qp, t['nranks']

    def selfenergy(comm):
        sig, cache = selfenergy_diag(*sig_args, comm=comm)
        return (sig,) + selfenergy_diag_backward(a_re, a_im, *sig_args, cache,
                                                 comm=comm)

    return {
        'window': window,
        'qp_set_gradient': lambda comm: qp_set_gradient(
            w['X'], w['D'], w['eps'].copy(), nocc, w['grid'], w['nu'],
            w['wt'], w['states'], w['weights'].copy(), mu=w['mu'],
            residue_route='explicit', comm=comm),
        'qp_gradient_explicit': lambda comm: qp_gradient_space_time(
            w['X'], w['D'], w['eps'].copy(), nocc, w['grid'], w['nu'],
            w['wt'], nocc - 1, mu=w['mu'], residue_route='explicit',
            comm=comm),
        'qp_gradient_sop': lambda comm: qp_gradient_space_time(
            w['X'], w['D'], w['eps'].copy(), nocc, w['grid'], w['nu'],
            w['wt'], nocc - 1, mu=w['mu'], residue_route='sop', comm=comm),
        'rpa_energy_and_adjoint': lambda comm: rpa_energy_and_adjoint(
            w['X'], w['D'], w['eps'], nocc, w['grid'], mu=w['mu'],
            comm=comm)[:4],
        'screened_interaction_tau': lambda comm: (
            screened_interaction_tau(proj, w['grid'], ctw, comm=comm),
            screened_interaction_tau_backward(wt_bar, proj, w['grid'], ctw,
                                              comm=comm)),
        'selfenergy_diag': selfenergy,
        'static_screening': lambda comm: static_screening(
            w['X'], w['D'], w['eps'].copy(), nocc, w['w_grid'], comm=comm),
    }


def _leaves(out):
    """The arrays and numbers of a kernel's return value, flattened."""
    if isinstance(out, (tuple, list)):
        return [leaf for item in out for leaf in _leaves(item)]
    return [np.asarray(out)]


def _bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


#: Which leaves of each kernel are an output partition and so bitwise serial.
BITWISE_LEAVES = {'window': [0], 'qp_set_gradient': [0],
                  'qp_gradient_explicit': [0, 1], 'qp_gradient_sop': [0, 1],
                  'static_screening': [0]}
#: The relative standard of the other leaves: cancelling sums for the
#: self-energy and the screened interaction, benign ones elsewhere.
LEAF_REL = {'screened_interaction_tau': SIGMA_REL,
            'selfenergy_diag': SIGMA_REL}


@pytest.mark.parametrize('size', SIZES)
def test_kernels_find_the_context(water, size):
    """Without `comm=`, inside `run_simulated`: the explicit-comm call's bits
    on every rank, and the serial answer at this suite's standard."""
    kernels = _kernels(water)
    serial = {name: call(None) for name, call in kernels.items()}

    def one_rank(comm):
        return {name: (call(None), call(comm))
                for name, call in kernels.items()}

    for r, got in enumerate(run_simulated(one_rank, size)):
        for name, (found, explicit) in got.items():
            if name == 'window':
                assert found[1] == size, 'the window ran on one rank'
            for a, b in zip(_leaves(found), _leaves(explicit)):
                assert _bitwise(a, b), (name, r)
            for i, (a, b) in enumerate(zip(_leaves(found),
                                           _leaves(serial[name]))):
                if name == 'window' and i == 1:
                    continue                       # the rank count itself
                if i in BITWISE_LEAVES.get(name, []):
                    assert _bitwise(a, b), (name, i, r)
                else:
                    assert close(a, b, rel=LEAF_REL.get(name, REL)), (name, i)


@pytest.mark.parametrize('size', SIZES)
def test_fit_finds_the_context(water, size):
    """`separable_factors` without `comm=`: its three-centre pass is split
    (each rank walks a share of the blocks), every rank returns the explicit
    call's bits, and D is the serial fit to `FIT_TOL`."""
    w = water
    kw = dict(auxbasis='cc-pvdz-ri', block_memory_gb=2e-4)
    ser = separable_factors(w['mf'], w['mol'], **kw)

    def one_rank(comm):
        t = {}
        found = separable_factors(w['mf'], w['mol'], timings=t, **kw)
        return found, separable_factors(w['mf'], w['mol'], comm=comm, **kw), t

    out = run_simulated(one_rank, size)
    total = out[0][2]['fit_blocks_total']
    assert sum(t['fit_blocks'] for _, _, t in out) == total
    for found, explicit, _ in out:
        for a, b, c in zip(found, explicit, out[0][0]):
            assert _bitwise(a, b) and _bitwise(a, c)
        for j in (0, 2, 3):                        # X_mo, X_ao, coords
            assert _bitwise(found[j], ser[j])
        assert close(found[1], ser[1], rel=FIT_TOL)


@pytest.mark.parametrize('size', SIZES)
def test_serial_region_inside_the_context(water, size):
    """`distributed(None)` inside a rank is the serial call, bitwise, and no
    kernel in it communicates."""
    kernels = _kernels(water)
    serial = {name: call(None) for name, call in kernels.items()}

    def one_rank(comm):
        with distributed(None):
            lockstep_stats(reset=True)
            out = {name: call(None) for name, call in kernels.items()}
            return out, lockstep_stats()

    for got, stats in run_simulated(one_rank, size):
        assert stats['calls'] == 0 and 'agreement_calls' not in stats
        for name in kernels:
            for i, (a, b) in enumerate(zip(_leaves(got[name]),
                                           _leaves(serial[name]))):
                assert _bitwise(a, b), (name, i)


def _own_mean_field(w, comm, ulp_rank):
    """A copy of the mean field; rank `ulp_rank`'s is one ulp off in one
    orbital energy and one orbital coefficient."""
    mf = copy.copy(w['mf'])
    mf.mo_energy = np.array(w['mf'].mo_energy, copy=True)
    mf.mo_coeff = np.array(w['mf'].mo_coeff, copy=True)
    mf.mo_occ = np.array(w['mf'].mo_occ, copy=True)
    if comm.Get_rank() == ulp_rank:
        nocc = w['nocc']
        mf.mo_energy[nocc] = np.nextafter(mf.mo_energy[nocc], np.inf)
        mf.mo_coeff[3, nocc - 1] = np.nextafter(mf.mo_coeff[3, nocc - 1],
                                                np.inf)
    return mf


@pytest.mark.parametrize('size', PERTURBED_SIZES)
def test_separable_factors_repairs_a_one_ulp_mean_field(water, size):
    """Rank 2's mean field one ulp off: the factors of the unperturbed run on
    every rank, rank 0's mean field on every rank, one repair counted."""
    w = water
    kw = dict(auxbasis='cc-pvdz-ri')
    nocc = w['nocc']
    moved = (w['mf'].mo_energy[nocc], w['mf'].mo_coeff[3, nocc - 1])
    ulp = max(abs(np.nextafter(x, np.inf) - x) for x in moved)

    def one_rank(comm, ulp_rank):
        mf = _own_mean_field(w, comm, ulp_rank)
        with distributed(comm, audit=True):
            lockstep_stats(reset=True)
            factors = separable_factors(mf, w['mol'], **kw)
            return factors, mf.mo_energy, mf.mo_coeff, lockstep_stats()

    reference = run_simulated(one_rank, size, -1)            # nobody perturbed
    for _, _, _, stats in reference:
        assert stats['audited_calls'] > 0 and stats['mismatched_calls'] == 0
    for r, (factors, mo_energy, mo_coeff, stats) in enumerate(
            run_simulated(one_rank, size, 2)):
        for a, b in zip(factors, reference[0][0]):
            assert _bitwise(a, b), r
        assert _bitwise(mo_energy, w['mf'].mo_energy), r
        assert _bitwise(mo_coeff, np.asarray(w['mf'].mo_coeff)), r
        assert stats['mismatched_calls'] == (1 if r == 2 else 0), r
        assert stats['max_abs_diff'] == (ulp if r == 2 else 0.0), r


@pytest.mark.parametrize('size', SIZES)
def test_agreement_sees_one_ulp(size):
    """True on identical bytes, False on every rank when the last rank's
    array is one ulp off, and the counters say which call it was."""
    ref = np.random.default_rng(0).normal(size=(64, 9))

    def one_rank(comm):
        x = ref.copy()
        same = agreement((x, {'n': 3, 'w': 0.5}), label='identical')
        if comm.Get_rank() == comm.Get_size() - 1:
            x[17, 4] = np.nextafter(x[17, 4], -np.inf)
        apart = agreement(x, label='one ulp')
        return same, apart, lockstep_stats()

    for same, apart, stats in run_simulated(one_rank, size):
        assert same is True and apart is False
        assert stats['agreement_calls'] == 2 and stats['disagreements'] == 1
        assert stats['first_disagreement'] == 'one ulp'


def test_agreement_compares_values_structure_and_layout():
    """The logical C-ordered bytes: a Fortran copy agrees, another shape or
    dtype does not; serial is True and counts nothing; `audit_only` stays
    silent outside an audit block and checks inside one."""
    ref = np.random.default_rng(1).normal(size=(6, 5))

    def one_rank(comm):
        last = comm.Get_rank() == comm.Get_size() - 1
        layout = agreement(np.asfortranarray(ref) if last else ref)
        shape = agreement(ref.reshape(5, 6) if last else ref)
        dtype = agreement(ref.astype(np.float32) if last else ref)
        quiet = agreement(ref[::-1] if last else ref, audit_only=True)
        before = lockstep_stats()['agreement_calls']
        with distributed(comm, audit=True):
            loud = agreement(ref[::-1] if last else ref, audit_only=True)
        return layout, shape, dtype, quiet, loud, before

    for layout, shape, dtype, quiet, loud, before in run_simulated(one_rank,
                                                                   3):
        assert layout and not shape and not dtype
        assert quiet and before == 3 and not loud
    lockstep_stats(reset=True)
    assert agreement(ref) is True
    assert 'agreement_calls' not in lockstep_stats()


def test_digest_sees_every_word():
    """One ulp at any word -- first, last, either side of a block boundary --
    a swap, a +-1 ulp pair and a byte of a ragged tail all move the digest."""
    block = mpi_grid.AGREEMENT_DIGEST_BLOCK
    ref = np.random.default_rng(2).normal(size=3 * block + 5)
    d0 = mpi_grid._digest(ref)
    for i in (0, block - 1, block, 2 * block + 7, ref.size - 1):
        x = ref.copy()
        x[i] = np.nextafter(x[i], np.inf)
        assert mpi_grid._digest(x) != d0, i
    x = ref.copy()
    x[[3, 3 + block]] = x[[3 + block, 3]]
    assert mpi_grid._digest(x) != d0
    x = ref.copy()
    x[10], x[11] = np.nextafter(x[10], np.inf), np.nextafter(x[11], -np.inf)
    assert mpi_grid._digest(x) != d0
    tail = np.arange(13, dtype=np.int8)
    moved = tail.copy()
    moved[-1] += 1
    assert mpi_grid._digest(tail) != mpi_grid._digest(moved)
    body = ref[:-5]
    assert (mpi_grid._digest(np.asfortranarray(body.reshape(-1, 4)))
            == mpi_grid._digest(body))


@pytest.mark.parametrize('size', [3])
def test_audit_finds_every_kernel_in_agreement(water, size):
    """An audited run of every kernel here on identical inputs: each checks
    its inputs and outputs and none reports a difference, nor does any
    lockstep find one to repair."""
    kernels = _kernels(water)

    def one_rank(comm):
        with distributed(comm, audit=True):
            lockstep_stats(reset=True)
            for call in kernels.values():
                call(None)
            separable_factors(water['mf'], water['mol'],
                              auxbasis='cc-pvdz-ri')
            return lockstep_stats()

    for stats in run_simulated(one_rank, size):
        assert stats['agreement_calls'] >= 2 * len(kernels)
        assert stats['disagreements'] == 0
        assert stats['mismatched_calls'] == 0


@pytest.mark.parametrize('size', [3])
def test_audit_reports_a_drifted_factor(water, size):
    """Rank 1's collocation one ulp off, into the quasiparticle-set gradient:
    not repaired -- a factor is identical by construction -- and reported on
    every rank at that kernel's entry."""
    w = water

    def one_rank(comm):
        x = np.array(w['X'], copy=True)
        if comm.Get_rank() == 1:
            x[5, 2] = np.nextafter(x[5, 2], np.inf)
        with distributed(comm, audit=True):
            lockstep_stats(reset=True)
            qp_set_gradient(x, w['D'], w['eps'].copy(), w['nocc'], w['grid'],
                            w['nu'], w['wt'], w['states'], w['weights'].copy(),
                            mu=w['mu'], residue_route='explicit')
            stats = lockstep_stats()
        return stats, _bitwise(x, w['X'])

    for r, (stats, untouched) in enumerate(run_simulated(one_rank, size)):
        assert stats['disagreements'] >= 1
        assert stats['first_disagreement'] == 'qp_set_gradient inputs'
        assert untouched == (r != 1)


def test_lockstep_mean_field_and_the_re_exports(water):
    """`lockstep_mean_field` leaves rank 0's five attributes everywhere; the
    names the chains import keep an explicit comm, None staying serial."""
    w = water

    def one_rank(comm):
        mf = copy.copy(w['mf'])
        mf.mo_energy = np.array(w['mf'].mo_energy, copy=True)
        mf.mo_coeff = np.array(w['mf'].mo_coeff, copy=True)
        mf.mo_occ = np.array(w['mf'].mo_occ, copy=True)
        if comm.Get_rank():
            mf.mo_energy += 1e-3
            mf.mo_coeff *= 1.0 + 1e-6
            mf.e_tot, mf.converged = mf.e_tot + 1.0, False
        untouched = replicate_mean_field(mf, None).mo_energy.copy()
        lockstep_mean_field(mf)
        factors = tuple(np.array(a, copy=True) for a in w['factors'])
        if comm.Get_rank():
            factors[0][...] *= 1.0 + 1e-6
        replicate_factors(factors, comm)
        return mf, untouched, factors

    for r, (mf, untouched, factors) in enumerate(run_simulated(one_rank, 3)):
        assert _bitwise(mf.mo_energy, w['mf'].mo_energy)
        assert _bitwise(mf.mo_coeff, np.asarray(w['mf'].mo_coeff))
        assert mf.e_tot == w['mf'].e_tot and mf.converged == w['mf'].converged
        assert _bitwise(untouched, w['mf'].mo_energy) == (r == 0)
        for a, b in zip(factors, w['factors']):
            assert _bitwise(a, b)


def test_mpi_grid_imports_no_gw():
    """Replicating a mean field no longer reaches into GW: `mpi_grid` alone
    loads no `SingleReference` module, so `distributed_df` can take
    `lockstep_mean_field` from it outside the environment import cycle."""
    code = ('import sys; import src.Base.utils.mpi_grid; '
            'print(sorted(m for m in sys.modules if "SingleReference" in m '
            'or m.startswith("src.Base.environment")))')
    env = dict(os.environ, MBPT_USE_MPI='0')
    env.pop('PYTHONPATH', None)
    out = subprocess.run([sys.executable, '-c', code], cwd=REPO, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == '[]'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
