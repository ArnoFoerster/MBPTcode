"""<Sigma_x - v_xc> is built once per mean field and handed to every consumer.

The static term of every imaginary-axis quasiparticle route is
`mo.T (Sigma_x - v_xc) mo`: one exchange matrix and the exchange-correlation
potential on the DFT grid. It carries NO state index and NO spectrum -- K and
v_xc are functionals of the converged density and the orbitals -- so a
two-state window costs exactly what the whole BSE diagonal costs, and a driver
that runs a GW window and then a BSE must not pay for it twice. On
naphthalene/cc-pVDZ/PBE0 at two threads one build is 1.23 s, split 1.00 s for
`nr_rks` on the DFT grid, 0.08 s for K and 0.004 s for J; on
anthracene/cc-pVTZ/PBE0 it is the ~22 s of `t_qp_static` that no rank count
divided, because every rank builds the same matrix.

`static_exchange_mean_field_matrix` is that object, handed in through
`sigma_x_matrix=` the way `factors=` is. Indexing is all that is left to do
with it, so the gates here are BITWISE, not on a tolerance.

What is gated:

  * the handed-in matrix reproduces the built diagonal to a few ulp, on HF and on
    PBE0, for a two-state window and for the whole diagonal;
  * `shifted_mean_field` -- the view each evGW cycle screens with -- builds the
    same matrix as the mean field it copies, which is why one build may serve
    every cycle;
  * the environment term stays OUTSIDE the cached matrix, so a reaction field
    still lands on top of it exactly as it lands on a freshly built one;
  * `solve_bse_isdf` builds it ONCE for a whole BSE@G0W0 and once for a whole
    evGW loop, and not at all when the caller hands one in, with the roots
    the same to a few ulp either way;
  * the window's quasiparticle energies are the built ones to a few ulp,
    serially and under simulated ranks (two computations,
    the few-ulp bar is floored at one Ha, so the HF diagonal
    cases below, whose whole term is 1e-16 Ha, no longer see a 1e-6
    relative perturbation);
  * `exchange='df-direct'` forms no such matrix and refuses a handed-in one;
  * the AO operator under the matrix is built once per mean field and
    density and read by the static term and by the orbital responses of the
    Kohn-Sham correction and the EXX double counting
    (`mean_field_static_operator`).

What breaks each gate:

  * The handed-in matrix scaled by 1 + 1e-6 at the call sites in this file
    (`sigma_x_matrix=(1 + 1e-6) * matrix`), the built side untouched: the four
    test_handed_matrix_is_the_built_diagonal cases (2.7e-22 and 4.5e-21 Ha on
    HF, where the whole term is 1e-16; 1.9e-07 and 1.3e-06 Ha on PBE0), the
    PBE0 reaction-field case (1.9e-07 Ha), the window (1.7e-07 Ha) and the BSE
    roots (3.0e-07 Ha). At 1 + 1e-12 only the five diagonal gates see it: a
    5e-13 Ha shift is below what a converged Newton root and a conv_tol=1e-5
    Davidson resolve, so the solved quantities need a perturbation above their
    own solver tolerance before a few-ulp gate on them can see it.
  * `static_exchange_matrix` ignoring its `sigma_x_matrix` and rebuilding:
    test_bse_builds_the_static_matrix_once, 2 builds where 1 is expected. The
    value gates cannot see it, since both sides then rebuild the same matrix.
    Counting the builds catches a hand-in silently dropped, comparing the
    values catches a wrong one.
  * `solve_bse_isdf` dropping the `gw_kw['sigma_x_matrix']` hand-off:
    test_bse_builds_the_static_matrix_once, 2 builds where 1 is expected, and
    test_evgw_builds_the_static_matrix_once, 4 builds for 3 cycles. The evGW
    roots are the same to every printed digit either way: the rebuild is pure
    cost.
  * `static_exchange_diagonal` accepting a matrix on 'df-direct' (the raise
    removed): test_df_direct_refuses_a_matrix.

The PBE0 reference of the shared fixture matters: on Hartree-Fock the scaled
hand-in moves the window and the BSE roots by nothing at all, because <Sigma_x
- v_xc> vanishes there by construction and the
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

from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.GW import qp_solve
from src.SingleReference.GW.evGW import shifted_mean_field
from src.SingleReference.GW.qp_solve import (static_exchange_diagonal,
                                             static_exchange_matrix,
                                             static_exchange_mean_field_matrix)
from src.SingleReference.GW.space_time import (separable_factors,
                                               solve_qp_energy_space_time)
from src.SingleReference.BSE.solve import solve_bse_isdf
from tests.isdf_test_grid import g1
from tests.pins import assert_same

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
                factors=separable_factors(mf, mol, auxbasis='cc-pvdz-ri', counts=g1(mol)))


@pytest.mark.parametrize('xc', REFERENCES)
@pytest.mark.parametrize('window', ['window', 'full'])
def test_handed_matrix_is_the_built_diagonal(xc, window):
    """Handing the matrix in leaves the diagonal where it was, to a few ulp."""
    mf, mol, nocc = mean_field(xc)
    nmo = mf.mo_coeff.shape[1]
    states = (np.asarray([nocc - 1, nocc]) if window == 'window'
              else np.arange(nmo))
    built = static_exchange_diagonal(mf, mol, states)
    matrix = static_exchange_mean_field_matrix(mf, mol)
    handed = static_exchange_diagonal(mf, mol, states, sigma_x_matrix=matrix)
    assert_same(handed, built, quantity='root',
                label='the cached static term moved')
    # The full matrix route agrees with itself too, so a caller may keep either.
    assert_same(np.diag(static_exchange_matrix(
        mf, mol, sigma_x_matrix=matrix))[states], built, quantity='root',
        label='the full matrix route')


@pytest.mark.parametrize('xc', REFERENCES)
def test_the_shifted_mean_field_builds_the_same_matrix(xc):
    """Only the eigenvalues move along an evGW loop, so the static term does
    not: one build serves every cycle."""
    mf, mol, _ = mean_field(xc)
    eps = np.asarray(mf.mo_energy, float)
    view = shifted_mean_field(mf, eps + 0.1)
    assert_same(static_exchange_mean_field_matrix(view, mol),
                static_exchange_mean_field_matrix(mf, mol), quantity='root',
                label='shifted view')


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
    assert_same(handed, built, quantity='root', label='with a reaction field')
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
    assert_same(handed, ref, quantity='continued', label='window')


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
    assert_same(timed, plain, quantity='continued', label='timed window')
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
        assert_same(qp, ref, quantity='continued', label='ranks')


def count_builds(monkeypatch):
    """Count every <Sigma_x - v_xc> build, wherever it is reached from."""
    calls = []
    built = qp_solve.static_exchange_mean_field_matrix

    def counted(*args, **kwargs):
        calls.append(1)
        return built(*args, **kwargs)

    monkeypatch.setattr(qp_solve, 'static_exchange_mean_field_matrix', counted)
    monkeypatch.setattr(
        'src.SingleReference.BSE.solve.'
        'static_exchange_mean_field_matrix', counted)
    return calls, built


def test_bse_builds_the_static_matrix_once(water, monkeypatch):
    """One BSE@G0W0 builds <Sigma_x - v_xc> once, and not at all when the
    caller hands one in."""
    w = water
    calls, built = count_builds(monkeypatch)
    solve_bse_isdf(w['mf'], w['mol'], w['nocc'], numerics=Numerics(bse={'nroots': 2}),
                   factors=w['factors'], probe=False, progress=False)
    assert len(calls) == 1, f'{len(calls)} static builds for one BSE@G0W0'

    matrix = built(w['mf'], w['mol'])
    calls.clear()
    solve_bse_isdf(w['mf'], w['mol'], w['nocc'], numerics=Numerics(bse={'nroots': 2}),
                   factors=w['factors'], probe=False, progress=False,
                   sigma_x_matrix=matrix)
    assert not calls, 'a handed-in matrix was rebuilt anyway'


def test_evgw_builds_the_static_matrix_once(water, monkeypatch):
    """An evGW loop screens each cycle with a `shifted_mean_field` view whose
    static term is the mean field's own, so the whole loop takes one build and
    not one per cycle."""
    w = water
    calls, _ = count_builds(monkeypatch)
    _, _, _, info = solve_bse_isdf(w['mf'], w['mol'], w['nocc'], numerics=Numerics(bse={'nroots': 2}),
                                   factors=w['factors'], probe=False,
                                   progress=False, self_consistency='evGW',
                                   gw_kwargs={'max_cycle': 3, 'tol': 1e-2})
    cycles = info['evgw']['cycles']
    assert cycles > 1, 'one cycle cannot show a per-cycle rebuild'
    assert len(calls) == 1, f'{len(calls)} static builds for {cycles} cycles'


def test_bse_roots_are_bitwise_with_a_handed_matrix(water):
    """The roots do not care where the static term came from."""
    w = water
    ref, _, _, _ = solve_bse_isdf(w['mf'], w['mol'], w['nocc'], numerics=Numerics(bse={'nroots': 2}),
                                  factors=w['factors'], probe=False,
                                  progress=False)
    matrix = static_exchange_mean_field_matrix(w['mf'], w['mol'])
    handed, _, _, _ = solve_bse_isdf(w['mf'], w['mol'], w['nocc'], numerics=Numerics(bse={'nroots': 2}),
                                     factors=w['factors'], probe=False,
                                     progress=False, sigma_x_matrix=matrix)
    assert_same(handed, ref, quantity='root', label='BSE roots')


@pytest.mark.parametrize('xc', REFERENCES)
def test_one_operator_serves_the_term_and_its_orbital_responses(xc):
    """The static term, the Kohn-Sham correction's orbital response and the
    EXX double counting's read ONE Sigma_x - v_xc
    (`mean_field_static_operator`), built once per mean field and density:
    one K among the three, each the operator built per consumer to a few ulp
    (the static term bitwise: the same calls under the same thread policy). A new
    density on the same mean field is a new operator, not the held one.

    Breaks if a consumer builds its own again (two or three K) or if the key
    misses the density (the held operator after the orbitals moved)."""
    from src.gradients.mean_field_response import (exx_double_counting_Y,
                                                   qp_xc_correction_Y)
    from src.Base.utils.parallel.threads import blas_single_threaded
    mf, mol, nocc = mean_field(xc)
    mo, dm = mf.mo_coeff, mf.make_rdm1()
    with blas_single_threaded():
        own = -0.5 * mf.get_k(mol, dm) - (mf.get_veff(mol, dm)
                                          - mf.get_j(mol, dm))
    weights = np.linspace(0.1, 1.0, mo.shape[1])
    qp_solve._STATIC_OPERATORS.clear()
    calls, get_k = [], mf.get_k
    mf.get_k = lambda *a, **k: calls.append(1) or get_k(*a, **k)
    matrix = static_exchange_mean_field_matrix(mf, mol)
    y_qp = qp_xc_correction_Y(mf, weights, nocc)
    y_exx = exx_double_counting_Y(mf, nocc)
    assert len(calls) == 1, f'{len(calls)} K builds for one mean field'
    o_mo = mo.T @ own @ mo
    assert np.array_equal(matrix, o_mo)
    assert_same(y_qp[:, nocc:], 2.0 * (o_mo @ np.diag(weights))[:, nocc:],
                quantity='root', label='qp_xc_correction_Y, virtual columns')
    if xc != 'hf':
        want = np.zeros_like(y_exx)
        want[:, :nocc] = 4.0 * o_mo[:, :nocc]
        assert_same(y_exx, want, quantity='root', label='exx_double_counting_Y')
    mf.mo_coeff = mo @ np.linalg.qr(np.eye(mo.shape[1]) + 1e-3 * np.tri(
        mo.shape[1], k=-1))[0]                 # another density, same object
    moved = static_exchange_mean_field_matrix(mf, mol)
    assert len(calls) == 2, 'the held operator was read at a new density'
    if xc != 'hf':                             # zero on HF at any density
        assert not np.allclose(moved, matrix)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
