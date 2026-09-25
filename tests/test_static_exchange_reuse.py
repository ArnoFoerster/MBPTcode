"""<Sigma_x - v_xc> is built once per mean field and handed to every consumer.

The static term of every imaginary-axis quasiparticle route is
`mo.T (Sigma_x - v_xc) mo`: one exchange matrix and the exchange-correlation
potential on the DFT grid. It carries NO state index and NO spectrum -- K and
v_xc are functionals of the converged density and the orbitals -- so a
two-state window costs exactly what the whole BSE diagonal costs, and a driver
that runs a GW window and then a BSE used to pay for it twice. On
naphthalene/cc-pVDZ/PBE0 at two threads one build is 1.23 s, split 1.00 s for
`nr_rks` on the DFT grid, 0.08 s for K and 0.004 s for J; on
anthracene/cc-pVTZ/PBE0 it is the ~22 s of `t_qp_static` that no rank count
divided, because every rank builds the same matrix.

`static_exchange_mean_field_matrix` is that object, handed in through
`sigma_x_matrix=` the way `factors=` is. Indexing is all that is left to do
with it, so the gates here are BITWISE, not on a tolerance.

What is gated:

  * the handed-in matrix reproduces the built diagonal exactly, on HF and on
    PBE0, for a two-state window and for the whole diagonal;
  * `shifted_mean_field` -- the view each evGW cycle screens with -- builds the
    same matrix as the mean field it copies, which is why one build may serve
    every cycle;
  * the environment term stays OUTSIDE the cached matrix, so a reaction field
    still lands on top of it exactly as it lands on a freshly built one;
  * `solve_bse_isdf` builds it ONCE for a whole BSE@G0W0 and once for a whole
    evGW loop, and not at all when the caller hands one in, with the roots
    bitwise either way;
  * the window's quasiparticle energies are bitwise the built ones, serially
    and under simulated ranks;
  * `exchange='df-direct'` forms no such matrix and refuses a handed-in one.

EVERY GATE HERE WAS SHOWN TO FAIL, each perturbation applied to a backup of
the file, restored and `cmp`-verified afterwards:

  perturbation                           gates that then fail
  the handed-in matrix scaled by         9 of 14: the four
  1 + 1e-6, at the call sites in this    test_handed_matrix_is_the_built
  file (`sigma_x_matrix=(1 + 1e-6) *     _diagonal cases (2.7e-22 and 4.5e-21
  matrix`) and nowhere else, so the      Ha on HF, where the whole term is
  built side stays put                   1e-16; 1.9e-07 and 1.3e-06 Ha on
                                         PBE0), the PBE0 reaction-field case
                                         (1.9e-07 Ha), the window
                                         (1.7e-07 Ha) and the BSE roots
                                         (3.0e-07 Ha). At
                                         1 + 1e-12 instead, only the five
                                         diagonal gates fail: a 5e-13 Ha
                                         shift is below what a converged
                                         Newton root and a conv_tol=1e-5
                                         Davidson resolve, so the solved
                                         quantities need a perturbation above
                                         their own solver tolerance before a
                                         bitwise gate on them can see it
  `static_exchange_matrix` ignores its   test_bse_builds_the_static_matrix
  `sigma_x_matrix` and rebuilds          _once, 2 builds where 1 is expected.
  (`if True:` on the branch)             The value gates do NOT fail, and
                                         cannot: both sides then rebuild the
                                         same matrix. Counting the builds is
                                         what catches a hand-in silently
                                         dropped, and comparing the values is
                                         what catches a wrong one
  `solve_bse_isdf` drops the             test_bse_builds_the_static_matrix
  `gw_kw['sigma_x_matrix']` hand-off     _once, 2 builds where 1 is expected,
                                         and test_evgw_builds_the_static
                                         _matrix_once, 4 builds for 3 cycles.
                                         The evGW roots are the same to every
                                         printed digit either way, which is
                                         the point: the rebuild was pure cost
  `static_exchange_diagonal` accepts     test_df_direct_refuses_a_matrix
  a matrix on 'df-direct' (the raise
  removed)

The PBE0 reference of the shared fixture is itself a perturbation finding: on
Hartree-Fock the scaled hand-in moved the window and the BSE roots by nothing
at all, because <Sigma_x - v_xc> vanishes there by construction and the
quasiparticle energy never reads it.

Run as a script, this file hands itself to pytest and exits with its verdict.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.utils.mpi_grid import run_simulated
from src.SingleReference.GW import qp_solve
from src.SingleReference.GW.evGW import shifted_mean_field
from src.SingleReference.GW.qp_solve import (static_exchange_diagonal,
                                             static_exchange_matrix,
                                             static_exchange_mean_field_matrix)
from src.SingleReference.GW.space_time import (separable_factors,
                                               solve_qp_energy_space_time)
from src.SingleReference.LinearResponse.davidson import solve_bse_isdf

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
#: Both references the routes serve: on Hartree-Fock the static term is zero by
#: construction and on a hybrid it is the whole DFT starting-point correction,
#: and only the second one pays for the DFT grid.
REFERENCES = ('hf', 'pbe0')
SIZES = [2, 3]


def mean_field(xc):
    """Water/cc-pVDZ, density fitted, as every route here takes it."""
    warnings.simplefilter('ignore')
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    if xc == 'hf':
        mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    else:
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    return mf, mol, mol.nelectron // 2


@pytest.fixture(scope='module')
def water():
    """PBE0, NOT Hartree-Fock: <Sigma_x - v_xc> is zero by construction on a
    Hartree-Fock reference, so a defect in the term it carries could not move a
    quasiparticle energy there and every gate below would pass blind."""
    mf, mol, nocc = mean_field('pbe0')
    return dict(mf=mf, mol=mol, nocc=nocc,
                factors=separable_factors(mf, mol, auxbasis='cc-pvdz-ri'))


@pytest.mark.parametrize('xc', REFERENCES)
@pytest.mark.parametrize('window', ['window', 'full'])
def test_handed_matrix_is_the_built_diagonal(xc, window):
    """Handing the matrix in leaves the diagonal bit for bit where it was."""
    mf, mol, nocc = mean_field(xc)
    nmo = mf.mo_coeff.shape[1]
    states = (np.asarray([nocc - 1, nocc]) if window == 'window'
              else np.arange(nmo))
    built = static_exchange_diagonal(mf, mol, states)
    matrix = static_exchange_mean_field_matrix(mf, mol)
    handed = static_exchange_diagonal(mf, mol, states, sigma_x_matrix=matrix)
    assert np.array_equal(handed, built), 'the cached static term moved'
    # The full matrix route agrees with itself too, so a caller may keep either.
    assert np.array_equal(np.diag(static_exchange_matrix(
        mf, mol, sigma_x_matrix=matrix))[states], built)


@pytest.mark.parametrize('xc', REFERENCES)
def test_the_shifted_mean_field_builds_the_same_matrix(xc):
    """Only the eigenvalues move along an evGW loop, so the static term does
    not: one build serves every cycle."""
    mf, mol, _ = mean_field(xc)
    eps = np.asarray(mf.mo_energy, float)
    view = shifted_mean_field(mf, eps + 0.1)
    assert np.array_equal(static_exchange_mean_field_matrix(view, mol),
                          static_exchange_mean_field_matrix(mf, mol))


@pytest.mark.parametrize('xc', REFERENCES)
def test_the_reaction_field_stays_outside_the_cached_matrix(xc):
    """The cached matrix is the mean field's own, so a continuum's Eq. (18)
    shift lands on top of it exactly as it lands on a freshly built one."""
    mf, mol, nocc = mean_field(xc)
    nmo = mf.mo_coeff.shape[1]
    shift = 0.01 * np.arange(nmo)
    states = np.asarray([nocc - 1, nocc])
    built = static_exchange_diagonal(mf, mol, states, reaction_field=shift)
    matrix = static_exchange_mean_field_matrix(mf, mol)
    handed = static_exchange_diagonal(mf, mol, states, reaction_field=shift,
                                      sigma_x_matrix=matrix)
    assert np.array_equal(handed, built)
    assert not np.array_equal(built, static_exchange_diagonal(mf, mol, states))


def test_df_direct_refuses_a_matrix(water):
    """'df-direct' streams K against the state vectors and forms no matrix, so
    a handed-in one would name a different static term."""
    w = water
    matrix = static_exchange_mean_field_matrix(w['mf'], w['mol'])
    with pytest.raises(ValueError, match='df-direct'):
        static_exchange_diagonal(w['mf'], w['mol'], [w['nocc']],
                                 exchange='df-direct', sigma_x_matrix=matrix)


def test_window_is_bitwise_with_a_handed_matrix(water):
    """The space-time window takes the cached matrix without moving a root."""
    w = water
    window = np.asarray([w['nocc'] - 1, w['nocc']])
    ref = solve_qp_energy_space_time(w['mf'], w['mol'], w['nocc'], window,
                                     factors=w['factors'])
    matrix = static_exchange_mean_field_matrix(w['mf'], w['mol'])
    handed = solve_qp_energy_space_time(w['mf'], w['mol'], w['nocc'], window,
                                        factors=w['factors'],
                                        sigma_x_matrix=matrix)
    assert np.array_equal(handed, ref)


def test_window_times_the_static_term_apart(water):
    """`t_qp_static` is the exchange build and `t_qp_states` the per-state
    Pade and root search, the two halves of `t_qp`: a caller reading where a
    window's time went can tell a static term worth handing in from a root
    search that no hand-in would shorten. The clock reads move no root."""
    w = water
    window = np.asarray([w['nocc'] - 1, w['nocc']])
    timings = {}
    timed = solve_qp_energy_space_time(w['mf'], w['mol'], w['nocc'], window,
                                       factors=w['factors'], timings=timings)
    plain = solve_qp_energy_space_time(w['mf'], w['mol'], w['nocc'], window,
                                       factors=w['factors'])
    assert np.array_equal(timed, plain)
    for key in ('t_qp_static', 't_qp_states', 't_qp'):
        assert key in timings and timings[key] >= 0.0, key
    assert timings['t_qp_static'] + timings['t_qp_states'] <= timings['t_qp'] + 1e-3


@pytest.mark.parametrize('size', SIZES)
def test_simulated_ranks_take_the_handed_matrix(water, size):
    """A rank count still returns rank 0's window, matrix handed in or not."""
    w = water
    window = np.asarray([w['nocc'] - 1, w['nocc']])
    ref = solve_qp_energy_space_time(w['mf'], w['mol'], w['nocc'], window,
                                     factors=w['factors'])
    matrix = static_exchange_mean_field_matrix(w['mf'], w['mol'])

    def one_rank(comm):
        return solve_qp_energy_space_time(w['mf'], w['mol'], w['nocc'], window,
                                          factors=w['factors'],
                                          sigma_x_matrix=matrix,
                                          distribute=True, comm=comm)

    for qp in run_simulated(one_rank, size):
        assert np.array_equal(qp, ref)


def count_builds(monkeypatch):
    """Count every <Sigma_x - v_xc> build, wherever it is reached from."""
    calls = []
    built = qp_solve.static_exchange_mean_field_matrix

    def counted(*args, **kwargs):
        calls.append(1)
        return built(*args, **kwargs)

    monkeypatch.setattr(qp_solve, 'static_exchange_mean_field_matrix', counted)
    monkeypatch.setattr(
        'src.SingleReference.LinearResponse.davidson.'
        'static_exchange_mean_field_matrix', counted)
    return calls, built


def test_bse_builds_the_static_matrix_once(water, monkeypatch):
    """One BSE@G0W0 builds <Sigma_x - v_xc> once, and not at all when the
    caller hands one in."""
    w = water
    calls, built = count_builds(monkeypatch)
    solve_bse_isdf(w['mf'], w['mol'], w['nocc'], nroots=2,
                   factors=w['factors'], probe=False, progress=False)
    assert len(calls) == 1, f'{len(calls)} static builds for one BSE@G0W0'

    matrix = built(w['mf'], w['mol'])
    calls.clear()
    solve_bse_isdf(w['mf'], w['mol'], w['nocc'], nroots=2,
                   factors=w['factors'], probe=False, progress=False,
                   sigma_x_matrix=matrix)
    assert not calls, 'a handed-in matrix was rebuilt anyway'


def test_evgw_builds_the_static_matrix_once(water, monkeypatch):
    """An evGW loop screens each cycle with a `shifted_mean_field` view whose
    static term is the mean field's own, so the whole loop takes one build and
    not one per cycle."""
    w = water
    calls, _ = count_builds(monkeypatch)
    _, _, _, info = solve_bse_isdf(w['mf'], w['mol'], w['nocc'], nroots=2,
                                   factors=w['factors'], probe=False,
                                   progress=False, self_consistency='evGW',
                                   gw_kwargs={'max_cycle': 3})
    cycles = info['evgw']['cycles']
    assert cycles > 1, 'one cycle cannot show a per-cycle rebuild'
    assert len(calls) == 1, f'{len(calls)} static builds for {cycles} cycles'


def test_bse_roots_are_bitwise_with_a_handed_matrix(water):
    """The roots do not care where the static term came from."""
    w = water
    ref, _, _, _ = solve_bse_isdf(w['mf'], w['mol'], w['nocc'], nroots=2,
                                  factors=w['factors'], probe=False,
                                  progress=False)
    matrix = static_exchange_mean_field_matrix(w['mf'], w['mol'])
    handed, _, _, _ = solve_bse_isdf(w['mf'], w['mol'], w['nocc'], nroots=2,
                                     factors=w['factors'], probe=False,
                                     progress=False, sigma_x_matrix=matrix)
    assert np.array_equal(handed, ref)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
