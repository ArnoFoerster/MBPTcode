"""The sum-over-poles route moved to production, BITWISE.

`SingleReference/GW/sum_over_poles.py` now holds the forward physics -- the
auxiliary-pole fit, the closed-form Sigma^c and its slope, the compression
guards and the quasiparticle Newton -- `gradients/sum_over_poles_adjoint.py`
the reverse pass, and `gradients/sum_over_poles.py` is a re-export shim. A move
is only a move if the numbers do not change, so every gate here is
`array_equal`, not a tolerance, and the reference each one compares against is
a verbatim copy of the code that was replaced: recomputing it through the new
module would gate nothing.

The Newton of `qp_energy_sop` is the one piece NOT rerouted through
`Solvers.qp_equation.solve_qp_equation_newton_guarded`. The two are the same
iteration -- the guard switches off at pole_offset = 0, and the model has no
pole on the real axis to guard against -- but the residual is spelled in the
model's own order, (eps + xc + s - w), against the solver's -(w - eps - xc - s),
and that is a last-bit difference: measured on water/cc-pVDZ, the two iterate
lists separate by one ULP at the first step for the HOMO-1 (1.1e-16 Ha) and the
LUMO (2.8e-17 Ha), and with a 0.013 Ha exchange-correlation correction the
LUMO's converged root lands 2.8e-17 Ha apart. Z agrees bitwise in all six
cases and the roots in five of six. `test_the_loop_is_not_the_guarded_newton`
holds that equivalence at the level it really has.

Every gate below was shown to fail once, by breaking in the source what the
gate watches and running the 12 cases of this file against it; each run was
restored from a backup copy and the restore checked with `cmp`:

  * the occupied sign dropped in production `denominators`, so every
    denominator reads omega - eps_q - Om_m: 6 fail, 6 pass. On water/cc-pVDZ it
    moves Sigma^c(eps_HOMO) from 0.048752188370 to -0.163108404978 Ha and the
    HOMO's quasiparticle root from -0.446839342814 to -0.665725250931 Ha,
    5.96 eV.
  * the sign of production `pole_amplitudes` flipped: 3 fail, 9 pass. It is the
    fit itself, so Sigma^c and every adjoint change sign, while the pole
    POSITIONS and `compressible` do not move at all -- neither ever sees an
    amplitude, which is what those two gates are separately for.
  * `sigma_sop_backward`'s wc_bar built on 1/D^2 instead of 1/D, the one
    plausible confusion with the eps_bar branch beside it: 1 fail, 11 pass, and
    only the adjoint gate. |wc_bar|max goes from 1.02e-01 to 3.48e-01. The
    shim-vs-production gate CANNOT see it, because both names resolve to the
    same perturbed object -- which is the limit of every shim gate and why the
    reference here is the verbatim old code.
  * the shim re-exporting a sign-flipped wrapper of `sigma_sop` instead of the
    production object: 3 fail, 9 pass -- the identity gate, the old-code closed
    form, and the shim-vs-production comparison. This is the failure mode a
    shim actually has: a name that resolves to something merely resembling the
    moved object.
  * two names dropped from the shim's re-exports (`pole_basis`,
    `_denominators`): 3 fail, 9 pass.
  * the shim rebinding SOP_N_POLES to 13 instead of taking it from
    `Base.constants`: 2 fail, 10 pass. Two spellings of one default is how the
    fit a gradient froze stops matching the fit an energy used.
  * xc_correction dropped from the production Newton step: 2 fail, 10 pass. On
    water's HOMO-1 with a 0.013 Ha correction the loop then lands at
    -0.530410948641 Ha against the -0.518042446697 Ha of the equation it was
    supposed to solve, 1.2e-02 Ha apart -- which is also what makes the
    comparison against the guarded solver a real check and not an identity.
  * the import gate cannot be broken from inside production -- importing
    anything of `src.gradients` there is circular and dies at import time --
    so its sensitivity is a positive control inside the same test: the shim,
    which does import the gradient package, must leak exactly the modules the
    production module may not.
"""
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from src.Base.constants import (SOP_CLEARANCE_MIN, SOP_FIT_RCOND,
                                SOP_FIT_STRIDE, SOP_N_POLES)
from src.Base.utils.grids import gauss_legendre_grid, gap_scaled_w0
from src.SingleReference.GW import sum_over_poles as prod
from src.SingleReference.GW.contour_deformation import residue_set, wc_explicit
from src.SingleReference.GW.real_screening import ov_energies, screening_aux
from src.SingleReference.base import get_occ_virt_indices
from src.Solvers.qp_equation import solve_qp_equation_newton_guarded
from src.gradients import sum_over_poles as shim
from src.gradients import sum_over_poles_adjoint as adjoint
from src.gradients.contour_deformation_adjoint import (integral_term_backward,
                                                       screening_chain)

#: Every name the gradient module exported before the move.
OLD_NAMES = ['SOP_CLEARANCE_MIN', 'SOP_FIT_RCOND', 'SOP_N_POLES',
             'SOP_FIT_STRIDE', 'initial_poles', 'pole_basis',
             'pole_pseudoinverse', 'pole_amplitudes', 'fit_poles',
             '_denominators', 'pole_clearance', 'compressible', 'sigma_sop',
             'sigma_sop_slope', 'qp_energy_sop', 'sigma_sop_backward',
             'sop_from_wc', 'sop_partials', '_ov_energies', 'residue_set',
             'screening_aux', 'integral_term_backward', 'screening_chain']

#: The model spectrum and screening of tests/test_sum_over_poles.py, where the
#: fit is exact and every quantity is a closed form of known numbers.
EPS = np.array([-0.90, -0.62, -0.35, 0.18, 0.44, 0.83, 1.25, 1.70])
NOCC = 3
POLES = np.array([0.70, 1.50, 3.00])
WEIGHTS = np.array([0.050, 0.080, 0.120])
COUPLING = np.array([0.31, -0.24, 0.45, 0.18, -0.37, 0.22, 0.14, -0.29])


# --------------------------------------------------------------- the old code
# Verbatim copies of what was replaced. They are the reference: a gate that
# recomputed the reference through the new code would gate nothing.

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


def _old_fit_poles(wc, nu_points, poles, n_iter=5, stride=1, bounds=None):
    wc = np.asarray(wc, float)
    om = np.asarray(poles, float).copy()
    n_poles = om.size
    lo, hi = bounds if bounds is not None else (om[0], om[-1])
    u_s = -np.asarray(nu_points, float) ** 2
    cols = range(0, wc.shape[1], max(int(stride), 1))
    f_cols = [wc[:, q] for q in cols]
    n_col = len(f_cols)
    for _ in range(int(n_iter)):
        kern = 1.0 / (u_s[:, None] - (om ** 2)[None, :])
        block = np.zeros((len(u_s) * n_col, n_col * n_poles + n_poles))
        rhs = np.concatenate(f_cols)
        for j, f in enumerate(f_cols):
            rows = slice(j * len(u_s), (j + 1) * len(u_s))
            block[rows, j * n_poles:(j + 1) * n_poles] = kern
            block[rows, -n_poles:] = -f[:, None] * kern
        x, *_ = np.linalg.lstsq(block, rhs, rcond=None)
        zeros = np.linalg.eigvals(np.diag(om ** 2)
                                  - np.ones((n_poles, 1)) @ x[-n_poles:][None, :])
        om = np.sqrt(np.sort(np.clip(np.real(zeros), lo ** 2, hi ** 2)))
    return om


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
    from src.Base.pyscf_interface import get_density_fitting_coefficients
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

def test_every_name_the_gradient_module_exported_still_resolves():
    for name in OLD_NAMES:
        assert hasattr(shim, name), name


def test_the_shim_and_production_are_the_same_objects():
    """A shim that re-implemented anything would be a second route."""
    for old, new in (('initial_poles', prod.initial_poles),
                     ('pole_basis', prod.pole_basis),
                     ('pole_pseudoinverse', prod.pole_pseudoinverse),
                     ('pole_amplitudes', prod.pole_amplitudes),
                     ('fit_poles', prod.fit_poles),
                     ('_denominators', prod.denominators),
                     ('pole_clearance', prod.pole_clearance),
                     ('compressible', prod.compressible),
                     ('sigma_sop', prod.sigma_sop),
                     ('sigma_sop_slope', prod.sigma_sop_slope),
                     ('qp_energy_sop', prod.qp_energy_sop),
                     ('sop_from_wc', prod.sop_from_wc),
                     ('sigma_sop_backward', adjoint.sigma_sop_backward),
                     ('sop_partials', adjoint.sop_partials),
                     ('_ov_energies', ov_energies),
                     ('screening_aux', screening_aux),
                     ('residue_set', residue_set),
                     ('integral_term_backward', integral_term_backward),
                     ('screening_chain', screening_chain)):
        assert getattr(shim, old) is new, old
    for name, value in (('SOP_CLEARANCE_MIN', SOP_CLEARANCE_MIN),
                        ('SOP_FIT_RCOND', SOP_FIT_RCOND),
                        ('SOP_N_POLES', SOP_N_POLES),
                        ('SOP_FIT_STRIDE', SOP_FIT_STRIDE)):
        assert getattr(shim, name) == value, name


def test_the_constants_kept_their_values():
    """The four numbers moved to `Base.constants`; a move that changed one
    would change every guard and every fit built on it, and a caller importing
    them from the gradient module still has to get those objects."""
    assert (SOP_CLEARANCE_MIN, SOP_FIT_RCOND, SOP_N_POLES, SOP_FIT_STRIDE) \
        == (0.05, 1e-12, 12, 8)
    assert (shim.SOP_CLEARANCE_MIN, shim.SOP_FIT_RCOND, shim.SOP_N_POLES,
            shim.SOP_FIT_STRIDE) == (SOP_CLEARANCE_MIN, SOP_FIT_RCOND,
                                     SOP_N_POLES, SOP_FIT_STRIDE)
    assert (prod.SOP_CLEARANCE_MIN, prod.SOP_FIT_RCOND, prod.SOP_N_POLES,
            prod.SOP_FIT_STRIDE) == (SOP_CLEARANCE_MIN, SOP_FIT_RCOND,
                                     SOP_N_POLES, SOP_FIT_STRIDE)


def test_the_fit_is_the_old_fit_on_the_model():
    """Exact-model screening: poles, basis, pseudoinverse, amplitudes."""
    nu, _ = model_grid()
    wc = model_wc(nu)
    start = prod.initial_poles(3, 0.3, 6.0)
    assert np.array_equal(start, _old_initial_poles(3, 0.3, 6.0))
    assert np.array_equal(prod.pole_basis(nu, POLES),
                          _old_pole_basis(nu, POLES))
    assert np.array_equal(prod.pole_pseudoinverse(nu, POLES),
                          _old_pole_pseudoinverse(nu, POLES))
    assert np.array_equal(prod.pole_amplitudes(wc, POLES, nu),
                          _old_pole_amplitudes(wc, POLES, nu))
    assert np.array_equal(
        prod.fit_poles(wc, nu, start, bounds=(0.05, 20.0)),
        _old_fit_poles(wc, nu, start, bounds=(0.05, 20.0)))


def test_the_fit_is_the_old_fit_on_water(water):
    """The vector fitting and its entry point, where the fit is not exact."""
    eps, nocc, b, c_ov, nu, _ = water
    d = ov_energies(eps, nocc)
    gap, top = float(d.min()), float(d.max())
    start = prod.initial_poles(SOP_N_POLES, gap, top)
    assert np.array_equal(start, _old_initial_poles(SOP_N_POLES, gap, top))
    for p in states(nocc):
        wc = wc_explicit(b[:, p, :], c_ov, d, nu)
        assert np.array_equal(
            prod.fit_poles(wc, nu, start, stride=SOP_FIT_STRIDE,
                           bounds=(gap, top)),
            _old_fit_poles(wc, nu, start, stride=SOP_FIT_STRIDE,
                           bounds=(gap, top))), p
        poles, amp = prod.sop_from_wc(wc, nu, eps, nocc)
        old_poles = _old_fit_poles(wc, nu, start, stride=SOP_FIT_STRIDE,
                                   bounds=(gap, top))
        assert np.array_equal(poles, old_poles), p
        assert np.array_equal(amp, _old_pole_amplitudes(wc, old_poles, nu)), p


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
            assert np.array_equal(prod.denominators(omega, poles, eps, nocc),
                                  _old_denominators(omega, poles, eps, nocc))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                s_new = prod.sigma_sop(omega, amp, poles, eps, nocc)
                s_shim = shim.sigma_sop(omega, amp, poles, eps, nocc)
            s_old = _old_sigma_sop(omega, amp, poles, eps, nocc)
            assert s_new == s_old == s_shim, (p, shift)
            assert (prod.sigma_sop_slope(omega, amp, poles, eps, nocc)
                    == _old_sigma_sop_slope(omega, amp, poles, eps, nocc))
            assert (prod.pole_clearance(omega, poles, eps, nocc)
                    == _old_pole_clearance(omega, poles, eps, nocc))
            assert (prod.compressible(omega, eps, nocc)
                    == _old_compressible(omega, eps, nocc))
    # the core state is refused, with the words the callers match on
    _, _, poles, amp = fitted(eps, nocc, b, c_ov, nu, 0)
    with pytest.warns(RuntimeWarning, match='Eq. \\(27\\)'):
        prod.sigma_sop(float(eps[0]), amp, poles, eps, nocc)


def test_the_quasiparticle_loop_is_the_old_loop(water):
    """Same iterates, same root, same Z -- the loop was kept, not rerouted."""
    eps, nocc, b, c_ov, nu, _ = water
    for p in (nocc - 2, nocc - 1, nocc):
        _, _, poles, amp = fitted(eps, nocc, b, c_ov, nu, p)
        for xc, w0 in ((0.0, None), (0.013, float(eps[p]) + 0.01)):
            trace = []
            w_old, z_old = _old_qp_energy_sop(p, amp, poles, eps, nocc,
                                              xc_correction=xc, w0=w0,
                                              trace=trace)
            w_new, z_new = prod.qp_energy_sop(p, amp, poles, eps, nocc,
                                              xc_correction=xc, w0=w0)
            assert (w_new, z_new) == (w_old, z_old), (p, xc)
            assert (w_new, z_new) == shim.qp_energy_sop(
                p, amp, poles, eps, nocc, xc_correction=xc, w0=w0), (p, xc)
            # the root really is one: the residual of the equation solved
            s = prod.sigma_sop(w_new, amp, poles, eps, nocc, check=False)
            assert abs(w_new - eps[p] - xc - s) < 1e-10, (p, xc)
            assert len(trace) > 1, 'the loop must take more than one step'


def test_the_loop_is_not_the_guarded_newton(water):
    """The repo's guarded Newton solves the SAME equation, to the last bits.

    With pole_offset = 0 the guard never fires -- the pole model has no pole on
    the real axis to hold the iterate away from -- so the only difference is
    the order the residual is spelled in: (eps + xc + s - w) here against
    -(w - eps - xc - s) there. That is a rounding, not a method, and this gate
    says so quantitatively: the roots agree to 1e-15, which is why the loop
    stays where the model's own algebra is written and no guard is imported
    into a route that has nothing to guard.
    """
    eps, nocc, b, c_ov, nu, _ = water
    for p in (nocc - 2, nocc - 1, nocc):
        _, _, poles, amp = fitted(eps, nocc, b, c_ov, nu, p)

        def sigma(w, amp=amp, poles=poles):
            return prod.sigma_sop(w, amp, poles, eps, nocc, check=False)

        def slope(w, amp=amp, poles=poles):
            return prod.sigma_sop_slope(w, amp, poles, eps, nocc)

        for xc in (0.0, 0.013):
            w_loop, z_loop = prod.qp_energy_sop(p, amp, poles, eps, nocc,
                                                xc_correction=xc)
            w_g, z_g = solve_qp_equation_newton_guarded(
                sigma, slope, eps, p, nocc, xc_correction=xc,
                w0=float(eps[p]), pole_offset=0.0, relax_offset=False,
                linearize_on_capture=False, offset_min=0.0, z_min=0.0,
                linear_offset=0.0)
            assert abs(w_loop - w_g) < 1e-15, (p, xc, w_loop, w_g)
            assert abs(z_loop - z_g) < 1e-15, (p, xc, z_loop, z_g)


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
                assert np.array_equal(new, old), (p, shift)
        omega = float(eps[p] + 0.02)
        for new, old in zip(adjoint.sop_partials(omega, amp, poles, eps, nocc,
                                                 nu, bp, c_ov),
                            _old_sop_partials(omega, amp, poles, eps, nocc, nu,
                                              bp, c_ov)):
            assert np.array_equal(new, old), p


def test_the_shim_path_and_the_production_path_are_bitwise_equal(water):
    """Every recorded quantity, computed live through both import paths."""
    eps, nocc, b, c_ov, nu, _ = water
    d = ov_energies(eps, nocc)
    assert np.array_equal(d, shim._ov_energies(eps, nocc))
    gap, top = float(d.min()), float(d.max())
    start = shim.initial_poles(SOP_N_POLES, gap, top)
    assert np.array_equal(start, prod.initial_poles(SOP_N_POLES, gap, top))
    assert np.array_equal(shim.pole_basis(nu, start),
                          prod.pole_basis(nu, start))
    for p in states(nocc):
        bp = b[:, p, :]
        wc = wc_explicit(bp, c_ov, d, nu)
        assert np.array_equal(shim.pole_amplitudes(wc, start, nu),
                              prod.pole_amplitudes(wc, start, nu))
        poles_s, amp_s = shim.sop_from_wc(wc, nu, eps, nocc)
        poles_p, amp_p = prod.sop_from_wc(wc, nu, eps, nocc)
        assert np.array_equal(poles_s, poles_p) and np.array_equal(amp_s, amp_p)
        for shift in (-0.05, 0.02, 0.11):
            omega = float(eps[p] + shift)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                assert (shim.sigma_sop(omega, amp_s, poles_s, eps, nocc)
                        == prod.sigma_sop(omega, amp_p, poles_p, eps, nocc))
            assert (shim.sigma_sop_slope(omega, amp_s, poles_s, eps, nocc)
                    == prod.sigma_sop_slope(omega, amp_p, poles_p, eps, nocc))
            assert np.array_equal(shim._denominators(omega, poles_s, eps, nocc),
                                  prod.denominators(omega, poles_p, eps, nocc))
            assert (shim.pole_clearance(omega, poles_s, eps, nocc)
                    == prod.pole_clearance(omega, poles_p, eps, nocc))
            assert (shim.compressible(omega, eps, nocc)
                    == prod.compressible(omega, eps, nocc))
            for a, c in zip(shim.sigma_sop_backward(omega, amp_s, poles_s, eps,
                                                    nocc, nu),
                            adjoint.sigma_sop_backward(omega, amp_p, poles_p,
                                                       eps, nocc, nu)):
                assert np.array_equal(a, c), (p, shift)
        if p != 0:
            assert (shim.qp_energy_sop(p, amp_s, poles_s, eps, nocc)
                    == prod.qp_energy_sop(p, amp_p, poles_p, eps, nocc))
        omega = float(eps[p] + 0.02)
        for a, c in zip(shim.sop_partials(omega, amp_s, poles_s, eps, nocc, nu,
                                          bp, c_ov),
                        adjoint.sop_partials(omega, amp_p, poles_p, eps, nocc,
                                             nu, bp, c_ov)):
            assert np.array_equal(a, c), p


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
    assert np.array_equal(prod.pole_amplitudes(wc, POLES, nu),
                          _old_pole_amplitudes(wc, POLES, nu))


def test_production_does_not_import_the_gradient_package():
    """The whole point of the move: `src.gradients` is the consumer, not the
    dependency. A fresh interpreter, because an already-imported module would
    make this pass for the wrong reason, and the shim as the positive control
    -- it imports the gradient package, so the same probe must see it."""
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
        [sys.executable, '-c', probe % (root, 'src.gradients.sum_over_poles')],
        capture_output=True, text=True)
    assert leaks.returncode == 1, 'the probe cannot see a leak it should see'
