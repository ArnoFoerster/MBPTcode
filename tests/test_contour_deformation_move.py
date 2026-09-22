"""The contour deformation's forward half moved to production, BITWISE.

`SingleReference/GW/contour_deformation.py` now holds the forward physics,
`SingleReference/GW/real_screening.py` the explicit real-frequency screening,
`Solvers/qp_equation.solve_qp_equation_newton_guarded` the pole-guarded Newton,
and `gradients/contour_deformation.py` is a shim. A move is only a move if the
numbers do not change, so every gate here is `array_equal`, not a tolerance.

Every gate below was shown to fail once, by breaking in the source what the
gate watches and running the 16 cases of this file against it:

  * the Newton step, -(w - eps_p - xc - s)/(1 - sp) signed + in
    `Solvers.qp_equation`: 7 fail, 9 pass -- both flat-screening guards, the
    refusal, the capture fallback, the frozen guard, water, and the
    shim-against-production comparison. One step from the oxygen 1s start of
    water/cc-pVDZ goes to -21.141598284733 Ha correctly and to
    -19.956969423548 Ha flipped, 1.18 Ha apart, and the iterate lists then
    share nothing past their first entry.
  * the residue term of Sigma dropped in `sigma_cd`: 2 fail. It is worth
    8.1e-3 Ha on that oxygen 1s, 0.519747765714 against 0.511673049707 Ha,
    and without it the capture model is no longer captured at all -- no
    Newton of it ever reaches the pole of the neighbouring orbital.
  * the imaginary-axis d-slope of `frequency_factor`, -2 (w^2 - d^2)/den^2
    signed +: 1 fail. On the pair energies that gate samples, 0.05 to 3 Ha,
    the flip moves the slope by 1.0e+03 at w = 0.017, 3.8e+01 at w = 0.31 and
    6.4e-01 at w = 2.5 -- it diverges as the pair energy meets the frequency,
    which is where the integrand of Sigma^int is built.
  * the frequency-block split reading cosft_wt at the block-local row instead
    of the global one: 1 fail. It shows only when a block carries SEVERAL
    frequencies, which is why that gate picks a tile budget of two
    frequencies per block: at one per block the two indices coincide and the
    broken split passes.
  * one name dropped from the shim's re-exports (`screening_applied`): 2 fail.
  * `_f_rpa` delegating its real-axis eta = 0 branch as well: 1 fail. That
    branch is a/(a^2+0) - b/(b^2+0) against `frequency_factor`'s 1/a - 1/b,
    the same function up to 5.7e-14 apart, and the RPA numbers are pinned to
    its own spelling.
  * the shim not handing its own QP_POLE_OFFSET_MIN to the solver: 1 fail.
    tests/test_cd_pole_guard.py rebinds that module global and the Newton has
    to see it.
  * the import gate cannot be broken from inside production -- importing
    anything of `src.gradients` there is circular and dies at import time --
    so its sensitivity is shown on `Embedding.masked_self_energy`, which does
    import the shim: the same probe returns the leaked `src.gradients.*` and
    fails, while the two production modules leave sys.modules clean.
"""
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from src.Base.constants import (QP_POLE_OFFSET, QP_POLE_OFFSET_MIN,
                                QP_POLE_STRENGTH_MIN)
from src.Base.utils.grids import gauss_legendre_grid, gap_scaled_w0
from src.SingleReference.GW import contour_deformation as prod
from src.SingleReference.GW.real_screening import (ExplicitRealScreening,
                                                   ov_energies, screening_aux)
from src.SingleReference.LinearResponse.imaginary_frequency import (
    _f_rpa, frequency_factor)
from src.SingleReference.LinearResponse.space_time import (
    frequency_blocks, owned_frequency_blocks)
from src.SingleReference.base import get_occ_virt_indices
from src.Solvers.qp_equation import solve_qp_equation_newton_guarded
from src.gradients import contour_deformation as shim
from src.gradients import contour_deformation_adjoint as adjoint
from src.gradients import qp_space_time as qst
from src.gradients import space_time_adjoint as sta

#: Every name the gradient module exported before the move.
OLD_NAMES = ['_f_imaginary', '_f_real', '_ov_energies', '_need',
             '_wc_explicit', '_residue_backend', 'ExplicitRealScreening',
             'screening_applied', 'screening_aux', 'screening_contraction',
             'residue_set', 'residue_pole_distance', 'root_pole_distance',
             'sigma_cd', 'sigma_cd_slope', 'screening_chain',
             'integral_term_backward', 'residue_terms_backward',
             'sigma_cd_backward', 'qp_energy_cd', 'qp_energy_cd_backward']


# --------------------------------------------------------------- the old code
# Verbatim copies of what was replaced. They are the reference: a gate that
# recomputed the reference through the new code would gate nothing.

def _old_f_imaginary(d, nu):
    den = d ** 2 + nu ** 2
    return -2.0 * d / den, -2.0 * (nu ** 2 - d ** 2) / den ** 2


def _old_f_real(d, omega, eta=0.0):
    if eta == 0.0:
        a, b = omega - d, omega + d
        return 1.0 / a - 1.0 / b, 1.0 / a ** 2 + 1.0 / b ** 2, \
            -1.0 / a ** 2 + 1.0 / b ** 2
    a, b = omega - d, omega + d
    da, db = a ** 2 + eta ** 2, b ** 2 + eta ** 2
    f = a / da - b / db
    dfd = -(eta ** 2 - a ** 2) / da ** 2 - (eta ** 2 - b ** 2) / db ** 2
    dfw = (eta ** 2 - a ** 2) / da ** 2 - (eta ** 2 - b ** 2) / db ** 2
    return f, dfd, dfw


def _old_f_rpa(lr, d, w, is_imaginary):
    if is_imaginary:
        return -2.0 * d / (d**2 + w**2)
    else:
        if np.abs(w) < 1e-12:
            return -2.0 * d / (d**2 + lr.eta**2)
        else:
            return (w - d) / ((w - d)**2 + lr.eta**2) \
                - (w + d) / ((w + d)**2 + lr.eta**2)


def _reference_newton(p, Bp, eps, nocc, nu_points, nu_weights, wc,
                      xc_correction=0.0, tol=1e-11, max_iter=100, w0=None,
                      C_ov=None, pole_offset=QP_POLE_OFFSET, eta=0.0,
                      real_screening=None, linearize_on_capture=True,
                      relax_offset=True, guard_out=None, trace=None):
    """The quasiparticle loop of `qp_energy_cd` exactly as it was before the
    move, with one `trace.append` per self-energy evaluation."""
    w = float(eps[p] + (pole_offset if p < nocc else -pole_offset)
              if w0 is None else w0)
    common = dict(eta=eta, C_ov=C_ov, real_screening=real_screening)
    offset = float(pole_offset)
    frozen = not relax_offset
    if guard_out is not None:
        guard_out['pole_offset'] = offset
    last_push = None
    blocker = None
    was_pinned = False
    for _ in range(max_iter):
        gaps = np.abs(eps - w)
        pinned = gaps.min() < offset
        if pinned:
            q = int(np.argmin(gaps))
            w_push = float(eps[q] + np.sign(w - eps[q] or 1.0) * offset)
            if (last_push is not None and abs(w_push - last_push) < tol
                    and offset > QP_POLE_OFFSET_MIN):
                if frozen:
                    warnings.warn(
                        f'the frozen {offset:.1e} Ha pole guard of orbital {p} '
                        f'does not reach its quasiparticle root, so this '
                        f'geometry is not on the frozen Newton path: the guard '
                        f'is relaxed from here as an unpinned one would be.',
                        RuntimeWarning, stacklevel=2)
                    frozen = False
                offset = max(0.5 * offset, QP_POLE_OFFSET_MIN)
                if guard_out is not None:
                    guard_out['pole_offset'] = offset
                w_push = float(eps[q] + np.sign(w - eps[q] or 1.0) * offset)
            last_push = w_push
            blocker = q
            w = w_push
        was_pinned = pinned
        residues = prod.residue_set(eps, nocc, w)
        if trace is not None:
            trace.append(w)
        s = prod.sigma_cd(p, w, Bp, eps, nocc, nu_points, nu_weights,
                          residues=residues, wc=wc, **common)
        sp = prod.sigma_cd_slope(p, w, Bp, eps, nocc, nu_points, nu_weights,
                                 residues, wc, **common)
        step = -(w - eps[p] - xc_correction - s) / (1.0 - sp)
        w += step
        if abs(step) < tol and not was_pinned:
            break
    else:
        where = ('' if blocker is None else
                 f' The iterate was pinned against orbital {blocker} at '
                 f'eps={eps[blocker]:.6f} Ha with the root near {w:.6f} Ha, '
                 f'{abs(w - eps[blocker]):.1e} Ha away'
                 + (' -- the same orbital.' if blocker == p else
                    f', not p={p} (eps={eps[p]:.6f}).'))
        if blocker is not None and blocker != p and linearize_on_capture:
            w_lin = float(eps[p] + (QP_POLE_OFFSET if p < nocc
                                    else -QP_POLE_OFFSET))
            res_lin = prod.residue_set(eps, nocc, w_lin)
            if trace is not None:
                trace.append(w_lin)
            s_lin = prod.sigma_cd(p, w_lin, Bp, eps, nocc, nu_points,
                                  nu_weights, residues=res_lin, wc=wc, **common)
            sp_lin = prod.sigma_cd_slope(p, w_lin, Bp, eps, nocc, nu_points,
                                         nu_weights, res_lin, wc, **common)
            w = w_lin - (w_lin - eps[p] - xc_correction - s_lin) / (1.0 - sp_lin)
            warnings.warn(
                f'the CD Newton for orbital {p} was captured by the pole of '
                f'the self-energy at orbital {blocker} (eps='
                f'{eps[blocker]:.6f} Ha), not by a root of its own. Falling '
                f'back to the LINEARIZED quasiparticle energy '
                f'{w:.6f} Ha, one step from eps_{p}+/-{QP_POLE_OFFSET:.0e}. That '
                f'is a different approximation from the self-consistent root '
                f'and is not interchangeable with the other orbitals here.',
                RuntimeWarning, stacklevel=2)
            residues = prod.residue_set(eps, nocc, w)
            sp = prod.sigma_cd_slope(p, w, Bp, eps, nocc, nu_points, nu_weights,
                                     residues, wc, **common)
            return w, 1.0 / (1.0 - sp), residues
        raise RuntimeError(
            f"CD quasiparticle Newton for p={p} did not converge in "
            f"{max_iter} steps; the last pole guard was {offset:.1e} Ha."
            + where +
            f" A root within {QP_POLE_OFFSET_MIN:.0e} Ha of an orbital energy "
            f"cannot be resolved by this quadrature and no offset resolves "
            f"both.")
    if offset != pole_offset:
        warnings.warn(
            f'the quasiparticle root of orbital {p} lies inside the '
            f'{pole_offset:.0e} Ha pole guard, so the guard was relaxed to '
            f'{offset:.1e} Ha to reach it. The self-energy quadrature is less '
            f'accurate that close to the pole; the root is still converged to '
            f'{tol:.0e}.', RuntimeWarning, stacklevel=2)
    residues = prod.residue_set(eps, nocc, w)
    sp = prod.sigma_cd_slope(p, w, Bp, eps, nocc, nu_points, nu_weights,
                             residues, wc, **common)
    z = 1.0 / (1.0 - sp)
    if z < QP_POLE_STRENGTH_MIN and linearize_on_capture:
        w_lin = float(eps[p] + (QP_POLE_OFFSET if p < nocc
                                else -QP_POLE_OFFSET))
        res_lin = prod.residue_set(eps, nocc, w_lin)
        if trace is not None:
            trace.append(w_lin)
        s_lin = prod.sigma_cd(p, w_lin, Bp, eps, nocc, nu_points, nu_weights,
                              residues=res_lin, wc=wc, **common)
        sp_lin = prod.sigma_cd_slope(p, w_lin, Bp, eps, nocc, nu_points,
                                     nu_weights, res_lin, wc, **common)
        w_new = w_lin - (w_lin - eps[p] - xc_correction - s_lin) / (1.0 - sp_lin)
        warnings.warn(
            f'the CD Newton for orbital {p} converged on a root with pole '
            f'strength Z={z:.3f}, below {QP_POLE_STRENGTH_MIN}: that is a '
            f'satellite at {w:.6f} Ha, not the quasiparticle. Falling back to '
            f'the LINEARIZED quasiparticle energy {w_new:.6f} Ha. That is a '
            f'different approximation from the self-consistent root and is '
            f'not interchangeable with the other orbitals here.',
            RuntimeWarning, stacklevel=2)
        w = w_new
        residues = prod.residue_set(eps, nocc, w)
        sp = prod.sigma_cd_slope(p, w, Bp, eps, nocc, nu_points, nu_weights,
                                 residues, wc, **common)
        z = 1.0 / (1.0 - sp)
    return w, z, residues


# ------------------------------------------------------------------- the data

#: The synthetic spectra of tests/test_cd_pole_guard.py: a flat screening puts
#: the root inside the guard (2e-3), outside it (5e-3) or exactly on the pole
#: (0.0), and the second spectrum drags orbital 4 onto the pole at orbital 3.
GUARD_EPS = np.array([-0.9, -0.6, -0.35, 0.15, 0.4, 0.8])
GUARD_NOCC = 3
GUARD_NU = np.array([0.05, 0.3, 1.0, 4.0])
GUARD_WT = np.array([0.1, 0.3, 0.8, 3.0])
CAPTURE_EPS = np.array([-0.9, -0.6, -0.35, 0.15, 0.20, 0.8])


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


def capture_model():
    """(p, Bp, C_ov, wc) whose Newton is captured by the pole at orbital 3."""
    rng = np.random.default_rng(7)
    b = rng.normal(scale=0.15, size=(3, len(CAPTURE_EPS), len(CAPTURE_EPS)))
    b = 0.5 * (b + b.transpose(0, 2, 1))
    c_ov = b[:, :GUARD_NOCC, GUARD_NOCC:].reshape(
        3, GUARD_NOCC * (len(CAPTURE_EPS) - GUARD_NOCC))
    wc = np.full((len(GUARD_NU), len(CAPTURE_EPS)), 0.16)
    return 4, b[:, 4, :], c_ov, wc


def states(nocc):
    """The core, the two states below the gap, and the first above it."""
    return (0, nocc - 2, nocc - 1, nocc)


def _traced_solve(p, Bp, eps, nocc, nu_points, nu_weights, wc, trace,
                  C_ov=None, **kwargs):
    """`qp_energy_cd`'s two callbacks, recording every omega Sigma is asked for.

    The production Newton takes Sigma through callbacks, so this is where its
    iterates are visible; `qp_energy_cd` builds exactly these two.
    """
    common = dict(eta=0.0, C_ov=C_ov, real_screening=None)

    def sigma(w):
        trace.append(w)
        return prod.sigma_cd(p, w, Bp, eps, nocc, nu_points, nu_weights,
                             residues=prod.residue_set(eps, nocc, w), wc=wc,
                             **common)

    def slope(w):
        return prod.sigma_cd_slope(p, w, Bp, eps, nocc, nu_points, nu_weights,
                                   prod.residue_set(eps, nocc, w), wc, **common)

    w, z = solve_qp_equation_newton_guarded(sigma, slope, eps, p, nocc,
                                            **kwargs)
    return w, z, prod.residue_set(eps, nocc, w)


# ------------------------------------------------------------------- the gates

def test_every_name_the_gradient_module_exported_still_resolves():
    for name in OLD_NAMES:
        assert hasattr(shim, name), name
    assert shim.ExplicitRealScreening is adjoint.ExplicitRealScreeningAdjoint
    assert issubclass(shim.ExplicitRealScreening, ExplicitRealScreening)
    # and the three names other modules import from the moved qp_space_time
    for name in ('cd_screening_contraction', 'cd_screening_contraction_multi',
                 'residue_route_auto'):
        assert getattr(qst, name) is getattr(prod, name), name


def test_the_shim_and_production_are_the_same_objects():
    """A shim that re-implemented anything would be a second route. Every old
    name is the one moved object, `_f_imaginary` and `_f_real` excepted: those
    two are the only ones the shim spells out, and they are gated on the old
    formulas below."""
    for old, new in (('sigma_cd', prod.sigma_cd),
                     ('sigma_cd_slope', prod.sigma_cd_slope),
                     ('residue_set', prod.residue_set),
                     ('residue_pole_distance', prod.residue_pole_distance),
                     ('root_pole_distance', prod.root_pole_distance),
                     ('screening_applied', prod.screening_applied),
                     ('screening_contraction', prod.screening_contraction),
                     ('_need', prod._need),
                     ('_wc_explicit', prod.wc_explicit),
                     ('_ov_energies', ov_energies),
                     ('screening_aux', screening_aux),
                     ('_residue_backend', adjoint._residue_backend),
                     ('screening_chain', adjoint.screening_chain),
                     ('integral_term_backward', adjoint.integral_term_backward),
                     ('residue_terms_backward', adjoint.residue_terms_backward),
                     ('sigma_cd_backward', adjoint.sigma_cd_backward),
                     ('qp_energy_cd_backward', adjoint.qp_energy_cd_backward)):
        assert getattr(shim, old) is new, old
    assert sta.owned_frequency_blocks is owned_frequency_blocks
    rng = np.random.default_rng(3)
    d = rng.uniform(0.05, 3.0, size=11)
    for w in (0.017, 0.31, 2.5):
        for a, b in zip(shim._f_imaginary(d, w), _old_f_imaginary(d, w)):
            assert np.array_equal(a, b), w
        for eta in (0.0, 0.05):
            for a, b in zip(shim._f_real(d, w, eta), _old_f_real(d, w, eta)):
                assert np.array_equal(a, b), (w, eta)


def test_frequency_factor_reproduces_the_old_formulas():
    """The dedup onto one function has to be a rounding-for-rounding copy."""
    rng = np.random.default_rng(20250918)
    d = rng.uniform(0.05, 3.0, size=37)
    for w in (0.0, 1e-13, 0.017, 0.31, 2.5, -0.44):
        f, dfd, dfw = frequency_factor(d, w, True, slopes=True)
        f_old, dfd_old = _old_f_imaginary(d, w)
        assert np.array_equal(f, f_old) and np.array_equal(dfd, dfd_old)
        assert dfw is None
        assert np.array_equal(frequency_factor(d, w, True), f_old)
        for eta in (0.0, 1e-3, 0.05):
            got = frequency_factor(d, w, False, eta, slopes=True)
            for a, b in zip(got, _old_f_real(d, w, eta)):
                assert np.array_equal(a, b), (w, eta)
            assert np.array_equal(frequency_factor(d, w, False, eta),
                                  _old_f_real(d, w, eta)[0])


def test_the_rpa_frequency_factor_is_untouched():
    """`_f_rpa` now delegates on the imaginary axis and for eta != 0 on the
    real one. Its two other branches keep their own spelling: w = 0 is the
    static limit, and eta = 0 is a/(a^2+0) - b/(b^2+0), which differs from
    1/a - 1/b in the last bits (5.7e-14 on these pair energies).
    """
    rng = np.random.default_rng(20250918)
    d = rng.uniform(0.05, 3.0, size=37)
    for eta in (0.0, 1e-3, 0.05):
        lr = type('LR', (), {'eta': eta})()
        for w in (0.0, 1e-13, 0.017, 0.31, 2.5, -0.44):
            for axis in (True, False):
                assert np.array_equal(_f_rpa(lr, d, w, axis),
                                      _old_f_rpa(lr, d, w, axis)), (eta, w, axis)
    lr = type('LR', (), {'eta': 0.0})()
    shared = frequency_factor(d, 0.31, False, 0.0)
    own = _f_rpa(lr, d, 0.31, False)
    assert not np.array_equal(shared, own), \
        'the two real-axis eta = 0 spellings coincide; then they should be one'
    assert np.abs(shared - own).max() < 1e-12


@pytest.mark.parametrize('screening', [2e-3, 5e-3])
def test_the_guarded_newton_reproduces_the_old_loop_on_the_guard(screening):
    """Same iterates, same root, same Z, same warnings -- the flat-screening
    models where the root sits inside and outside the pole guard."""
    Bp = np.zeros((3, len(GUARD_EPS)))
    wc = np.full((len(GUARD_NU), len(GUARD_EPS)), screening)
    ref_trace, new_trace = [], []
    with warnings.catch_warnings(record=True) as ref_caught:
        warnings.simplefilter('always')
        w_ref, z_ref, res_ref = _reference_newton(
            GUARD_NOCC - 1, Bp, GUARD_EPS, GUARD_NOCC, GUARD_NU, GUARD_WT, wc,
            trace=ref_trace)
    with warnings.catch_warnings(record=True) as new_caught:
        warnings.simplefilter('always')
        w_new, z_new, res_new = _traced_solve(
            GUARD_NOCC - 1, Bp, GUARD_EPS, GUARD_NOCC, GUARD_NU, GUARD_WT, wc,
            new_trace)
    assert prod.qp_energy_cd(GUARD_NOCC - 1, Bp, GUARD_EPS, GUARD_NOCC,
                             GUARD_NU, GUARD_WT, wc=wc)[:2] == (w_new, z_new)
    assert new_trace == ref_trace, (new_trace, ref_trace)
    assert w_new == w_ref and z_new == z_ref and res_new == res_ref
    assert ([str(c.message) for c in new_caught]
            == [str(c.message) for c in ref_caught])


def test_a_root_on_the_pole_is_refused_with_the_same_words():
    Bp = np.zeros((3, len(GUARD_EPS)))
    wc = np.zeros((len(GUARD_NU), len(GUARD_EPS)))
    with pytest.raises(RuntimeError) as ref:
        _reference_newton(GUARD_NOCC - 1, Bp, GUARD_EPS, GUARD_NOCC, GUARD_NU,
                          GUARD_WT, wc)
    with pytest.raises(RuntimeError) as new:
        prod.qp_energy_cd(GUARD_NOCC - 1, Bp, GUARD_EPS, GUARD_NOCC, GUARD_NU,
                          GUARD_WT, wc=wc)
    assert str(new.value) == str(ref.value)


def test_the_capture_fallback_reproduces_the_old_loop():
    """The branch that ends on a POLE of another orbital, not on a root."""
    p, Bp, c_ov, wc = capture_model()
    ref_trace, new_trace = [], []
    with warnings.catch_warnings(record=True) as ref_caught:
        warnings.simplefilter('always')
        w_ref, z_ref, res_ref = _reference_newton(
            p, Bp, CAPTURE_EPS, GUARD_NOCC, GUARD_NU, GUARD_WT, wc, C_ov=c_ov,
            trace=ref_trace)
    with warnings.catch_warnings(record=True) as new_caught:
        warnings.simplefilter('always')
        w_new, z_new, res_new = _traced_solve(
            p, Bp, CAPTURE_EPS, GUARD_NOCC, GUARD_NU, GUARD_WT, wc, new_trace,
            C_ov=c_ov)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        assert prod.qp_energy_cd(p, Bp, CAPTURE_EPS, GUARD_NOCC, GUARD_NU,
                                 GUARD_WT, wc=wc,
                                 C_ov=c_ov)[:2] == (w_new, z_new)
    assert new_trace == ref_trace
    assert w_new == w_ref and z_new == z_ref and res_new == res_ref
    assert ([str(c.message) for c in new_caught]
            == [str(c.message) for c in ref_caught])
    assert any('captured by the pole' in str(c.message) for c in new_caught)


def test_the_frozen_guard_warns_exactly_as_it_did():
    Bp = np.zeros((3, len(GUARD_EPS)))
    wc = np.full((len(GUARD_NU), len(GUARD_EPS)), 2e-3)
    ref_guard, new_guard = {}, {}
    with warnings.catch_warnings(record=True) as ref_caught:
        warnings.simplefilter('always')
        w_ref = _reference_newton(GUARD_NOCC - 1, Bp, GUARD_EPS, GUARD_NOCC,
                                  GUARD_NU, GUARD_WT, wc, relax_offset=False,
                                  guard_out=ref_guard)[0]
    with warnings.catch_warnings(record=True) as new_caught:
        warnings.simplefilter('always')
        w_new = prod.qp_energy_cd(GUARD_NOCC - 1, Bp, GUARD_EPS, GUARD_NOCC,
                                  GUARD_NU, GUARD_WT, wc=wc,
                                  relax_offset=False, guard_out=new_guard)[0]
    assert w_new == w_ref and new_guard == ref_guard
    assert ([str(c.message) for c in new_caught]
            == [str(c.message) for c in ref_caught])
    assert any('frozen' in str(c.message) for c in new_caught)


def test_the_guarded_newton_reproduces_the_old_loop_on_water(water):
    """The real thing: the oxygen 1s, both states below the gap, the first
    above it. The core state sweeps every occupied residue, so this exercises
    the explicit real-frequency backend as well as the loop."""
    eps, nocc, b, c_ov, nu, wt = water
    d = ov_energies(eps, nocc)
    for p in states(nocc):
        bp = b[:, p, :]
        wc = prod.wc_explicit(bp, c_ov, d, nu)
        ref_trace, new_trace = [], []
        with warnings.catch_warnings(record=True) as ref_caught:
            warnings.simplefilter('always')
            w_ref, z_ref, res_ref = _reference_newton(
                p, bp, eps, nocc, nu, wt, wc, C_ov=c_ov, trace=ref_trace)
        with warnings.catch_warnings(record=True) as new_caught:
            warnings.simplefilter('always')
            w_new, z_new, res_new = _traced_solve(
                p, bp, eps, nocc, nu, wt, wc, new_trace, C_ov=c_ov)
        assert shim.qp_energy_cd(p, bp, eps, nocc, nu, wt, wc=wc,
                                 C_ov=c_ov)[:2] == (w_new, z_new)
        assert new_trace == ref_trace, p
        assert w_new == w_ref and z_new == z_ref and res_new == res_ref, p
        assert ([str(c.message) for c in new_caught]
                == [str(c.message) for c in ref_caught]), p


def test_the_shim_path_and_the_production_path_are_bitwise_equal(water):
    """Sigma, its slope, the pole sets and the distances, on four states."""
    eps, nocc, b, c_ov, nu, wt = water
    d = ov_energies(eps, nocc)
    assert np.array_equal(d, shim._ov_energies(eps, nocc))
    for p in states(nocc):
        bp = b[:, p, :]
        wc_new = prod.wc_explicit(bp, c_ov, d, nu)
        assert np.array_equal(wc_new, shim._wc_explicit(bp, c_ov, d, nu))
        for shift in (-0.05, 0.02, 0.11):
            omega = float(eps[p] + shift)
            res = prod.residue_set(eps, nocc, omega)
            assert res == shim.residue_set(eps, nocc, omega)
            s_new = prod.sigma_cd(p, omega, bp, eps, nocc, nu, wt,
                                  residues=res, wc=wc_new, C_ov=c_ov)
            s_shim = shim.sigma_cd(p, omega, bp, eps, nocc, nu, wt,
                                   residues=res, wc=wc_new, C_ov=c_ov)
            assert s_new == s_shim
            assert (prod.sigma_cd_slope(p, omega, bp, eps, nocc, nu, wt, res,
                                        wc_new, C_ov=c_ov)
                    == shim.sigma_cd_slope(p, omega, bp, eps, nocc, nu, wt,
                                           res, wc_new, C_ov=c_ov))
            assert np.array_equal(
                prod.residue_pole_distance(eps, nocc, omega, res),
                shim.residue_pole_distance(eps, nocc, omega, res))
    w = [prod.qp_energy_cd(p, b[:, p, :], eps, nocc, nu, wt, C_ov=c_ov)[0]
         for p in (nocc - 1, nocc)]
    assert (prod.root_pole_distance(eps, w, [nocc - 1, nocc])
            == shim.root_pole_distance(eps, w, [nocc - 1, nocc]))


def test_the_residue_term_is_what_the_gate_is_worth(water):
    """The gate above compares numbers the residue term moves by 8.1e-3 Ha on
    the oxygen 1s; without that term it would compare the integral alone."""
    eps, nocc, b, c_ov, nu, wt = water
    d = ov_energies(eps, nocc)
    bp = b[:, 0, :]
    wc = prod.wc_explicit(bp, c_ov, d, nu)
    omega = float(eps[0] + QP_POLE_OFFSET)
    res = prod.residue_set(eps, nocc, omega)
    full = prod.sigma_cd(0, omega, bp, eps, nocc, nu, wt, residues=res, wc=wc,
                         C_ov=c_ov)
    dropped = prod.sigma_cd(0, omega, bp, eps, nocc, nu, wt, residues=[],
                            wc=wc, C_ov=c_ov)
    assert abs(full - dropped) > 1e-3, full - dropped


def test_a_flipped_newton_step_would_not_survive_the_gate(water):
    """The iterate comparison is sensitive to the step itself, not only to the
    root: one step from the core start goes 1.18 Ha the other way."""
    eps, nocc, b, c_ov, nu, wt = water
    d = ov_energies(eps, nocc)
    bp = b[:, 0, :]
    wc = prod.wc_explicit(bp, c_ov, d, nu)
    w = float(eps[0] + QP_POLE_OFFSET)
    res = prod.residue_set(eps, nocc, w)
    s = prod.sigma_cd(0, w, bp, eps, nocc, nu, wt, residues=res, wc=wc,
                      C_ov=c_ov)
    sp = prod.sigma_cd_slope(0, w, bp, eps, nocc, nu, wt, res, wc, C_ov=c_ov)
    good = w - (w - eps[0] - s) / (1.0 - sp)
    flipped = w + (w - eps[0] - s) / (1.0 - sp)
    assert abs(good - flipped) > 1.0, abs(good - flipped)


def test_the_shim_still_carries_a_rebound_guard_floor(water):
    """tests/test_cd_pole_guard.py sets `contour_deformation.QP_POLE_OFFSET_MIN`
    and expects the Newton to see it. The floor is a solver argument now, so
    the shim has to read its own module global at call time."""
    p, Bp, c_ov, wc = capture_model()
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError('stop')

    original = shim._qp_energy_cd
    shim._qp_energy_cd = spy
    shim.QP_POLE_OFFSET_MIN = 1e-9
    try:
        with pytest.raises(RuntimeError, match='stop'):
            shim.qp_energy_cd(p, Bp, CAPTURE_EPS, GUARD_NOCC, GUARD_NU,
                              GUARD_WT, wc=wc, C_ov=c_ov)
    finally:
        shim._qp_energy_cd = original
        shim.QP_POLE_OFFSET_MIN = QP_POLE_OFFSET_MIN
    assert seen['offset_min'] == 1e-9
    assert seen['z_min'] == QP_POLE_STRENGTH_MIN
    assert seen['linear_offset'] == QP_POLE_OFFSET


def test_the_frequency_blocks_are_the_same_generator():
    """`owned_frequency_blocks` moved to production; the gradient package
    imports it back, and both walk the same blocks.

    The tile budget is picked so the axis really is split -- more than one
    block and more than one frequency in a block -- because a one-frequency
    block hides every index a split can get wrong: the block-local row is the
    global row there, and the frequencies a block names are trivially the ones
    it transformed.
    """
    rng = np.random.default_rng(11)
    a = rng.normal(size=(6, 12, 12))
    proj_tau = 0.02 * (a + a.transpose(0, 2, 1))
    cosft_wt = rng.normal(size=(5, 6))
    tile = 8e-6
    blocks = frequency_blocks(len(cosft_wt), proj_tau.shape[-1], tile, 3)
    assert len(blocks) > 1 and max(k1 - k0 for k0, k1 in blocks) > 1, blocks
    for kwargs in ({}, dict(freq_indices=[1, 3]),
                   dict(freq_indices=[1, 3], live=4)):
        one = list(owned_frequency_blocks(proj_tau, cosft_wt, tile, **kwargs))
        two = list(sta.owned_frequency_blocks(proj_tau, cosft_wt, tile,
                                              **kwargs))
        assert [k for k, _ in one] == [k for k, _ in two]
        for (_, x), (_, y) in zip(one, two):
            assert np.array_equal(x, y)
        # each block is the transform of exactly the frequencies it names,
        # and the blocks together carry every owned frequency once
        for ks, blk in one:
            assert np.array_equal(blk, np.tensordot(cosft_wt[ks], proj_tau,
                                                    axes=(1, 0))), (kwargs, ks)
        carried = [k for ks, _ in one for k in ks]
        assert carried == sorted(kwargs.get('freq_indices',
                                            range(len(cosft_wt)))), kwargs


def test_production_does_not_import_the_gradient_package():
    """The whole point of the move: `src.gradients` is the consumer, not the
    dependency. A fresh interpreter, because an already-imported module would
    make this pass for the wrong reason."""
    root = str(Path(__file__).resolve().parent.parent)
    code = ('import sys; sys.path.insert(0, %r);'
            'import src.SingleReference.GW.contour_deformation;'
            'import src.SingleReference.GW.real_screening;'
            "leaked = [m for m in sys.modules if m.startswith('src.gradients')];"
            'assert not leaked, leaked' % root)
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stderr
