"""The sum-over-poles route in production is its reference code, to a few ulp.

`SingleReference/GW/sum_over_poles.py` holds the forward physics -- the
auxiliary-pole fit, the closed-form Sigma^c and its slope, the compression
guards and the quasiparticle root -- and `gradients/gwbse/sum_over_poles_adjoint.py`
the reverse pass. Production must reproduce the reference numbers, so every
gate here is a few ulp (`same`), not a method tolerance, and the reference each one
compares against is a verbatim copy of the original code: recomputing it
through the production module would gate nothing.

The auxiliary-pole fit is not kept as the original code either: production
eliminates the amplitudes before the least squares, so its reference is the
full least squares, on blocks of full numerical rank where the two give one
answer, to REDUCED_FIT_REL_BAR rather than a few ulp.

The quasiparticle root of `qp_energy_sop` is the other piece NOT kept as the
original code: the original's seeded Newton is replaced by the one root
finder of `Solvers.qp_equation`,
which brackets every root between the poles and keeps the largest Z. Both
end on a Newton step below 1e-11 from different starts, so the roots agree to
one ULP and Z to its last bits (`test_the_quasiparticle_loop_is_the_old_loop`),
and the pointwise scan of the same finder lands on the same root
(`test_the_pole_sum_and_the_scan_agree`).

What breaks each gate:

  * the occupied sign dropped in production `denominators`, so every
    denominator reads omega - eps_q - Om_m. On water/cc-pVDZ it
    moves Sigma^c(eps_HOMO) from 0.048752188370 to -0.163108404978 Ha and the
    HOMO's quasiparticle root from -0.446839342814 to -0.665725250931 Ha,
    5.96 eV.
  * the sign of production `pole_amplitudes` flipped. It is the
    fit itself, so Sigma^c and every adjoint change sign, while the pole
    POSITIONS and `compressible` do not move at all -- neither ever sees an
    amplitude, which is what those two gates are separately for.
  * `sigma_sop_backward`'s wc_bar built on 1/D^2 instead of 1/D, the one
    plausible confusion with the eps_bar branch beside it: only
    the adjoint gate. |wc_bar|max goes from 1.02e-01 to 3.48e-01.
  * xc_correction dropped from the production equation. On
    water's HOMO-1 with a 0.013 Ha correction the loop then lands at
    -0.530410948641 Ha against the -0.518042446697 Ha of the equation it was
    supposed to solve, 1.2e-02 Ha apart -- which is also what makes the
    comparison against the pointwise scan a real check and not an identity.
  * one direction of the range of K left out of the vector fit's
    projection: on the exact model the poles move by 2.3 relative against
    the full least squares
    (`test_the_fit_is_the_full_least_squares_where_that_is_well_conditioned`).
  * the import gate cannot be broken from inside production -- importing
    anything of `src.gradients` there is circular and dies at import time --
    so its sensitivity is a positive control inside the same test: the
    adjoint module, which does import the gradient package, must leak exactly
    the modules the production module may not.
"""
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from src.Base.numerics.constants import (SOP_CLEARANCE_MIN, SOP_FIT_RCOND,
                                         SOP_FIT_STRIDE, SOP_N_POLES)
from src.Base.utils.grids.grids import gauss_legendre_grid, gap_scaled_w0
from src.SingleReference.GW import sum_over_poles as prod
from src.SingleReference.GW.contour_deformation import residue_set, wc_explicit
from src.SingleReference.GW.real_screening import screening_aux
from src.Base.orbital_spaces import ov_energies
from src.Base.orbital_spaces import get_occ_virt_indices
from src.Solvers.qp_equation import solve_qp_equation_scan
from src.gradients.gwbse import sum_over_poles_adjoint as adjoint
from src.gradients.gwbse.contour_deformation_adjoint import (integral_term_backward,
                                                             screening_chain)
from tests.pins import assert_same_tree


def same(new, old, label=None):
    """Production against the verbatim reference, two computations: every
    number to a few ulp (`assert_same_tree`)."""
    assert_same_tree(new, old, quantity='root', label=label)

#: The model spectrum and screening of tests/test_sum_over_poles.py, where the
#: fit is exact and every quantity is a closed form of known numbers.
EPS = np.array([-0.90, -0.62, -0.35, 0.18, 0.44, 0.83, 1.25, 1.70])
NOCC = 3
POLES = np.array([0.70, 1.50, 3.00])
WEIGHTS = np.array([0.050, 0.080, 0.120])
COUPLING = np.array([0.31, -0.24, 0.45, 0.18, -0.37, 0.22, 0.14, -0.29])


# ------------------------------------------------------- the reference code
# Verbatim copies of the original code. They are the reference: a gate that
# recomputed the reference through production would gate nothing.

def _old_pole_basis(nu_points, poles):
    nu = np.asarray(nu_points, float)[:, None]
    om = np.asarray(poles, float)[None, :]
    return -2.0 * om / (nu ** 2 + om ** 2)


def _old_pole_pseudoinverse(nu_points, poles, rcond=1e-12):
    return np.linalg.pinv(_old_pole_basis(nu_points, poles), rcond=rcond)


def _old_pole_amplitudes(wc, poles, nu_points, rcond=1e-12):
    return _old_pole_pseudoinverse(nu_points, poles, rcond) @ np.asarray(wc,
                                                                         float)


def _old_initial_poles(n_poles, gap, e_max):
    return np.logspace(np.log10(gap), np.log10(e_max), int(n_poles))


def _full_fit_poles(wc, nu_points, poles, n_iter=5, stride=1, bounds=None,
                    merge_tol=prod.SOP_POLE_MERGE_TOL, ranks=None):
    """Vector fitting with the full least squares: every column's amplitudes
    a_j and the weight b in one (n_col nfreq) x (n_col + 1) M block, solved by
    lstsq. Production eliminates the a_j first; where this block has full
    numerical rank the two give one b. ranks: a list that receives
    (rank, columns, condition number) of each pass's block."""
    om = np.asarray(poles, float).copy()
    lo, hi = bounds
    u_s = -np.asarray(nu_points, float) ** 2
    f = np.asarray(wc, float)[:, ::stride].T
    n_col, n_freq = f.shape
    for _ in range(n_iter):
        om = prod.merge_poles(om, merge_tol)
        m = om.size
        kern = 1.0 / (u_s[:, None] - (om ** 2)[None, :])
        block = np.zeros((n_freq * n_col, (n_col + 1) * m))
        for j in range(n_col):
            rows = slice(j * n_freq, (j + 1) * n_freq)
            block[rows, j * m:(j + 1) * m] = kern
            block[rows, -m:] = -f[j][:, None] * kern
        x, _, rank, sv = np.linalg.lstsq(block, f.reshape(-1), rcond=None)
        if ranks is not None:
            ranks.append((rank, block.shape[1], sv[0] / sv[-1]))
        zeros = np.linalg.eigvals(np.diag(om ** 2)
                                  - np.ones((m, 1)) @ x[-m:][None, :])
        om = np.sqrt(np.sort(np.clip(np.real(zeros), lo ** 2, hi ** 2)))
    return prod.merge_poles(om, merge_tol)


def _old_denominators(omega, poles, eps, nocc):
    sign = np.where(np.arange(len(eps)) < nocc, 1.0, -1.0)
    return (omega - np.asarray(eps, float))[:, None] + sign[:, None] * np.asarray(
        poles, float)[None, :]


def _old_pole_clearance(omega, poles, eps, nocc):
    return float(np.abs(_old_denominators(omega, poles, eps, nocc)).min()
                 / np.min(poles))


def _old_compressible(omega, eps, nocc):
    limit = float(ov_energies(eps, nocc).min())
    swept = residue_set(eps, nocc, omega)
    reach = max((abs(eps[q] - omega) for q, _ in swept), default=0.0) / limit
    return bool(reach < 1.0), float(reach)


def _old_sigma_sop(omega, amplitudes, poles, eps, nocc):
    return float(np.sum(np.asarray(amplitudes, float).T
                        / _old_denominators(omega, poles, eps, nocc)))


def _old_sigma_sop_slope(omega, amplitudes, poles, eps, nocc):
    return -float(np.sum(np.asarray(amplitudes, float).T
                         / _old_denominators(omega, poles, eps, nocc) ** 2))


def _old_qp_energy_sop(p, amplitudes, poles, eps, nocc, xc_correction=0.0,
                       tol=1e-11, max_iter=100, w0=None, trace=None):
    w = float(eps[p] if w0 is None else w0)
    for _ in range(int(max_iter)):
        if trace is not None:
            trace.append(w)
        s = _old_sigma_sop(w, amplitudes, poles, eps, nocc)
        slope = _old_sigma_sop_slope(w, amplitudes, poles, eps, nocc)
        step = (eps[p] + xc_correction + s - w) / (1.0 - slope)
        w += step
        if abs(step) < tol:
            break
    slope = _old_sigma_sop_slope(w, amplitudes, poles, eps, nocc)
    return w, 1.0 / (1.0 - slope)


def _old_sigma_sop_backward(omega, amplitudes, poles, eps, nocc, nu_points,
                            sigma_bar=1.0, rcond=1e-12):
    den = _old_denominators(omega, poles, eps, nocc)
    inv2 = 1.0 / den ** 2
    amp = np.asarray(amplitudes, float).T
    eps_bar = sigma_bar * np.sum(amp * inv2, axis=1)
    omega_bar = -sigma_bar * float(np.sum(amp * inv2))
    f_pinv = _old_pole_pseudoinverse(nu_points, poles, rcond)
    wc_bar = sigma_bar * (f_pinv.T @ (1.0 / den).T)
    return eps_bar, wc_bar, omega_bar


def _old_sop_partials(omega, amplitudes, poles, eps, nocc, nu_points, Bp, C_ov,
                      sigma_bar=1.0):
    eps_bar, wc_bar, _ = _old_sigma_sop_backward(omega, amplitudes, poles, eps,
                                                 nocc, nu_points, sigma_bar)
    d = ov_energies(eps, nocc)
    Bp_bar = np.zeros_like(Bp)
    Cov_bar = np.zeros_like(C_ov)
    d_bar = np.zeros_like(d)
    for k, nu in enumerate(nu_points):
        wtb = screening_aux(C_ov, d, nu, True)[1] @ Bp
        bb, _, _ = integral_term_backward(Bp, wtb, wc_bar[k], want_chi0=False)
        Bp_bar += bb
        cvb, db, _ = screening_chain(wtb, wc_bar[k], C_ov, d, nu, True)
        Cov_bar += cvb
        d_bar += db
    occ, virt = get_occ_virt_indices(eps, nocc)
    d4 = d_bar.reshape(len(occ), len(virt))
    eps_bar[virt] += d4.sum(axis=0)
    eps_bar[occ] -= d4.sum(axis=1)
    return eps_bar, Bp_bar, Cov_bar


# ------------------------------------------------------------------- the data

def model_grid(n=160):
    return gauss_legendre_grid(n, gap_scaled_w0(EPS, NOCC))


def model_wc(nu_points):
    """wc[k, q] = c_q^2 sum_m a_m F[k, m], the screening that IS M poles."""
    return np.outer(_old_pole_basis(nu_points, POLES) @ WEIGHTS,
                    COUPLING ** 2)


@pytest.fixture(scope='module')
def water():
    """(eps, nocc, B, C_ov, nu, wt) for RHF/cc-pVDZ water, density fitted."""
    pytest.importorskip('pyscf')
    from pyscf import gto, scf
    from src.Base.pyscf.pyscf_interface import get_density_fitting_coefficients
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-10)
    eps = np.asarray(mf.mo_energy, float)
    nocc = mol.nelectron // 2
    b = get_density_fitting_coefficients(mol, mf, representation='spatial')
    occ, virt = get_occ_virt_indices(eps, nocc)
    c_ov = b[:, occ, :][:, :, virt].reshape(b.shape[0], -1)
    nu, wt = gauss_legendre_grid(48, gap_scaled_w0(eps, nocc))
    return eps, nocc, b, c_ov, nu, wt


def states(nocc):
    """The core, the two states below the gap, and the first above it."""
    return (0, nocc - 2, nocc - 1, nocc)


def fitted(eps, nocc, b, c_ov, nu, p):
    """(wc, poles, amplitudes) of one state, through the production route."""
    bp = b[:, p, :]
    wc = wc_explicit(bp, c_ov, ov_energies(eps, nocc), nu)
    poles, amp = prod.sop_from_wc(wc, nu, eps, nocc)
    return bp, wc, poles, amp


# ------------------------------------------------------------------ the gates

def test_the_constants_kept_their_values():
    """The four numbers live in `Base.constants`; a move that changed one
    would change every guard and every fit built on it."""
    assert (SOP_CLEARANCE_MIN, SOP_FIT_RCOND, SOP_N_POLES, SOP_FIT_STRIDE) \
        == (0.05, 1e-12, 12, 8)
    assert (prod.SOP_CLEARANCE_MIN, prod.SOP_FIT_RCOND, prod.SOP_N_POLES,
            prod.SOP_FIT_STRIDE) == (SOP_CLEARANCE_MIN, SOP_FIT_RCOND,
                                     SOP_N_POLES, SOP_FIT_STRIDE)


#: Relative distance allowed between the reduced fit's poles and the full
#: least squares' where that block has full numerical rank (condition number
#: below FULL_BLOCK_COND_MAX). The two solve one problem by different
#: factorizations, so they part by about the condition number times the
#: rounding: measured 7e-15 to 9e-13 on the cases below (conditions 1e5 to
#: 2e7). A projection that misses part of the range of K moves the poles by
#: 1e-3 and more.
REDUCED_FIT_REL_BAR = 1e-10
FULL_BLOCK_COND_MAX = 1e8


def model_cases():
    """(name, wc, M, bounds) of the well-conditioned fits: the exact
    three-pole model, and screenings of 4 to 6 poles fitted by 3 and 4."""
    nu, _ = model_grid()
    yield 'exact model', model_wc(nu), 3, (0.05, 20.0)
    rng = np.random.default_rng(7)
    for n_true in (4, 5, 6):
        true = np.sort(rng.uniform(0.5, 4.0, n_true))
        wc = _old_pole_basis(nu, true) @ rng.uniform(0.01, 0.1, (n_true, 8))
        for m in (3, 4):
            yield f'{n_true} poles by {m}', wc, m, (0.3, 6.0)


def test_the_fit_is_the_full_least_squares_where_that_is_well_conditioned():
    """The reduced fit against the full one, on blocks of full rank.

    On water's screening the full block is NOT of full rank (46 of 48 columns
    at a condition of 1e14): lstsq's cut drops two directions and its b does
    not minimize the residual, so the two fits part there by design."""
    nu, _ = model_grid()
    for name, wc, m, bounds in model_cases():
        start = prod.initial_poles(m, 0.3, 6.0)
        ranks = []
        full = _full_fit_poles(wc, nu, start, bounds=bounds, ranks=ranks)
        assert all(r == n and c < FULL_BLOCK_COND_MAX for r, n, c in ranks), (
            name, ranks)
        got = prod.fit_poles(wc, nu, start, bounds=bounds)
        assert got.shape == full.shape, (name, got, full)
        rel = np.abs(got - full).max() / np.abs(full).max()
        assert rel < REDUCED_FIT_REL_BAR, (name, rel)


def test_the_basis_is_the_old_basis_on_the_model():
    """Exact-model screening: basis, pseudoinverse, amplitudes."""
    nu, _ = model_grid()
    wc = model_wc(nu)
    start = prod.initial_poles(3, 0.3, 6.0)
    same(start, _old_initial_poles(3, 0.3, 6.0), 'initial poles')
    same(prod.pole_basis(nu, POLES), _old_pole_basis(nu, POLES), 'basis')
    same(prod.pole_pseudoinverse(nu, POLES),
         _old_pole_pseudoinverse(nu, POLES), 'pseudoinverse')
    same(prod.pole_amplitudes(wc, POLES, nu),
         _old_pole_amplitudes(wc, POLES, nu), 'amplitudes')


def test_the_entry_point_is_the_fit_on_water(water):
    """The entry point's poles are the merged fit and its amplitudes the old
    map of them, where the fit is not exact (the fit itself is gated against
    the full least squares on the model cases)."""
    eps, nocc, b, c_ov, nu, _ = water
    d = ov_energies(eps, nocc)
    gap, top = float(d.min()), float(d.max())
    start = prod.initial_poles(SOP_N_POLES, gap, top)
    same(start, _old_initial_poles(SOP_N_POLES, gap, top), 'initial poles')
    for p in states(nocc):
        wc = wc_explicit(b[:, p, :], c_ov, d, nu)
        poles, amp = prod.sop_from_wc(wc, nu, eps, nocc)
        merged = prod.fit_poles(wc, nu, start, stride=SOP_FIT_STRIDE,
                                bounds=(gap, top))
        same(poles, merged, ('merged poles', p))
        same(amp, _old_pole_amplitudes(wc, merged, nu), ('amplitudes', p))


def test_the_closed_form_is_the_old_one(water):
    """Sigma^c, its slope, the denominators and both guards, on four states.

    The core state is in on purpose: it is the one `compressible` has to
    refuse, and the warning it raises is part of what moved.
    """
    eps, nocc, b, c_ov, nu, _ = water
    for p in states(nocc):
        _, _, poles, amp = fitted(eps, nocc, b, c_ov, nu, p)
        for shift in (-0.05, 0.02, 0.11):
            omega = float(eps[p] + shift)
            same(prod.denominators(omega, poles, eps, nocc),
                 _old_denominators(omega, poles, eps, nocc), 'denominators')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                s_new = prod.sigma_sop(omega, amp, poles, eps, nocc)
            s_old = _old_sigma_sop(omega, amp, poles, eps, nocc)
            same(s_new, s_old, ('Sigma^c', p, shift))
            same(prod.sigma_sop_slope(omega, amp, poles, eps, nocc),
                 _old_sigma_sop_slope(omega, amp, poles, eps, nocc), 'slope')
            same(prod.pole_clearance(omega, poles, eps, nocc),
                 _old_pole_clearance(omega, poles, eps, nocc), 'clearance')
            same(prod.compressible(omega, eps, nocc),
                 _old_compressible(omega, eps, nocc), 'compressible')
    # the core state is refused, with the words the callers match on
    _, _, poles, amp = fitted(eps, nocc, b, c_ov, nu, 0)
    with pytest.warns(RuntimeWarning, match='Eq. \\(27\\)'):
        prod.sigma_sop(float(eps[0]), amp, poles, eps, nocc)


def test_the_quasiparticle_loop_is_the_old_loop(water):
    """Same root, same Z: the bracketed pole-sum solve lands on the loop's
    root and Z by the pin rule (8 ulp on the recording BLAS, the energy class
    bar under another; the cluster's MKL: 2 ulp, 1.1e-16 Ha). Not bitwise: both end on a Newton step below 1e-11 from
    different starts (the loop from eps_p, the bracketed solve from inside
    its interval)."""
    eps, nocc, b, c_ov, nu, _ = water
    for p in (nocc - 2, nocc - 1, nocc):
        _, _, poles, amp = fitted(eps, nocc, b, c_ov, nu, p)
        for xc, w0 in ((0.0, None), (0.013, float(eps[p]) + 0.01)):
            trace = []
            w_old, z_old = _old_qp_energy_sop(p, amp, poles, eps, nocc,
                                              xc_correction=xc, w0=w0,
                                              trace=trace)
            w_new, z_new = prod.qp_energy_sop(p, amp, poles, eps, nocc,
                                              xc_correction=xc)
            # a quasiparticle energy and a pole strength not read off a
            # continuation: the energy class, Z dimensionless
            assert_same_tree(w_new, w_old, quantity='energy',
                             label=('root', p, xc))
            assert_same_tree(z_new, z_old, unit='1', quantity='energy',
                             label=('Z', p, xc))
            # the root really is one: the residual of the equation solved
            s = prod.sigma_sop(w_new, amp, poles, eps, nocc, check=False)
            assert abs(w_new - eps[p] - xc - s) < 1e-10, (p, xc)
            assert len(trace) > 1, 'the loop must take more than one step'


def test_the_pole_sum_and_the_scan_agree(water):
    """Two bracket sources of the one finder, one equation: the pole sum's
    intervals and the pointwise scan of Sigma with its closed-form slope
    return the same root and Z to 1e-14."""
    eps, nocc, b, c_ov, nu, _ = water
    for p in (nocc - 2, nocc - 1, nocc):
        _, _, poles, amp = fitted(eps, nocc, b, c_ov, nu, p)

        def residual(w, amp=amp, poles=poles, xc=0.0):
            return w - eps[p] - xc - prod.sigma_sop(w, amp, poles, eps, nocc,
                                                    check=False)

        def fprime(w, amp=amp, poles=poles):
            return 1.0 - prod.sigma_sop_slope(w, amp, poles, eps, nocc)

        for xc in (0.0, 0.013):
            w_poles, z_poles = prod.qp_energy_sop(p, amp, poles, eps, nocc,
                                                  xc_correction=xc)
            w_scan, z_scan = solve_qp_equation_scan(
                lambda w, xc=xc: residual(w, xc=xc), fprime,
                float(eps[p] + xc))
            assert abs(w_poles - w_scan) < 1e-14, (p, xc, w_poles, w_scan)
            assert abs(z_poles - z_scan) < 1e-14, (p, xc, z_poles, z_scan)


def test_the_adjoints_are_the_old_ones(water):
    """(eps_bar, wc_bar, omega_bar), and the whole way down to (Bp, C_ov)."""
    eps, nocc, b, c_ov, nu, _ = water
    for p in states(nocc):
        bp, _, poles, amp = fitted(eps, nocc, b, c_ov, nu, p)
        for shift in (-0.05, 0.02):
            omega = float(eps[p] + shift)
            for new, old in zip(adjoint.sigma_sop_backward(omega, amp, poles,
                                                           eps, nocc, nu),
                                _old_sigma_sop_backward(omega, amp, poles, eps,
                                                        nocc, nu)):
                same(new, old, ('sigma_sop_backward', p, shift))
        omega = float(eps[p] + 0.02)
        for new, old in zip(adjoint.sop_partials(omega, amp, poles, eps, nocc,
                                                 nu, bp, c_ov),
                            _old_sop_partials(omega, amp, poles, eps, nocc, nu,
                                              bp, c_ov)):
            same(new, old, ('sop_partials', p))


def test_the_model_screening_still_reproduces_the_contour_deformation():
    """The physics the move must not touch: on a screening that IS M poles,
    the closed form equals the integral-plus-residues split."""
    from src.SingleReference.GW.contour_deformation import sigma_cd

    class ModelScreening:
        def value(self, z):
            return 1.0 + float(np.sum(2.0 * WEIGHTS * POLES
                                      / (z ** 2 - POLES ** 2)))

        def apply(self, freq, b):
            return self.value(freq) * np.asarray(b, float)

    nu, wt = model_grid()
    wc = model_wc(nu)
    amp = np.outer(WEIGHTS, COUPLING ** 2)
    bp = COUPLING[None, :].copy()
    for p in (NOCC - 1, NOCC):
        omega = float(EPS[p] + 0.02)
        cd = sigma_cd(p, omega, bp, EPS, NOCC, nu, wt,
                      residues=residue_set(EPS, NOCC, omega), wc=wc,
                      real_screening=ModelScreening())
        sop = prod.sigma_sop(omega, amp, POLES, EPS, NOCC, check=False)
        assert abs(sop - cd) < 1e-8, f'{sop:.12f} vs {cd:.12f}'
    same(prod.pole_amplitudes(wc, POLES, nu),
         _old_pole_amplitudes(wc, POLES, nu), 'amplitudes')


def test_production_does_not_import_the_gradient_package():
    """The whole point of the move: `src.gradients` is the consumer, not the
    dependency. A fresh interpreter, because an already-imported module would
    make this pass for the wrong reason, and the adjoint module as the positive
    control -- it imports the gradient package, so the same probe must see it."""
    root = str(Path(__file__).resolve().parent.parent)
    probe = ('import sys; sys.path.insert(0, %r);'
             'import %s;'
             "leaked = [m for m in sys.modules if m.startswith('src.gradients')];"
             'sys.exit(1 if leaked else 0)')
    clean = subprocess.run(
        [sys.executable, '-c',
         probe % (root, 'src.SingleReference.GW.sum_over_poles')],
        capture_output=True, text=True)
    assert clean.returncode == 0, clean.stderr
    leaks = subprocess.run(
        [sys.executable, '-c',
         probe % (root, 'src.gradients.gwbse.sum_over_poles_adjoint')],
        capture_output=True, text=True)
    assert leaks.returncode == 1, 'the probe cannot see a leak it should see'
