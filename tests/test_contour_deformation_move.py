"""The contour deformation in production equals its reference copy, to a few
ulp (two computations of each number: `assert_same` of tests/pins.py).

`SingleReference/GW/contour_deformation.py` holds the forward physics,
`SingleReference/GW/real_screening.py` the explicit real-frequency screening,
and `gradients/contour_deformation_adjoint.py` the adjoints. Production must
reproduce the reference numbers, so every gate here is a few ulp, not a
physics tolerance -- except the root itself: the reference's pole-guarded Newton is
replaced by the one root finder of `Solvers.qp_equation` (one quasiparticle
root finder everywhere), which lands on the reference's root wherever
that root is a quasiparticle, to the finder's tolerance, from another path.

What breaks each gate:

  * the residual of `qp_energy_cd`, w - eps_p - xc - s, signed + on s:
    the flat-screening models and water land on another root. One Newton
    step from the oxygen 1s start of water/cc-pVDZ goes to -20.117866411565
    Ha correctly and to -20.980701296715 Ha flipped, 0.86 Ha apart.
  * the residue term of Sigma dropped in `sigma_cd`. It is worth
    8.1e-3 Ha on that oxygen 1s, 0.520811747276 against 0.512737031268 Ha.
  * the imaginary-axis d-slope of `frequency_factor`, -2 (w^2 - d^2)/den^2
    signed +. On the pair energies that gate samples, 0.05 to 3 Ha,
    the flip moves the slope by 1.0e+03 at w = 0.017, 3.8e+01 at w = 0.31 and
    6.4e-01 at w = 2.5 -- it diverges as the pair energy meets the frequency,
    which is where the integrand of Sigma^int is built.
  * the frequency-block split reading cosft_wt at the block-local row instead
    of the global one. It shows only when a block carries SEVERAL
    frequencies, which is why that gate picks a tile budget of two
    frequencies per block: at one per block the two indices coincide and the
    broken split passes.
  * the split transforming only the rows it owns instead of its serial
    blocks: the same gate, on the row-count-sensitive tensordot.
  * `f_rpa` delegating its real-axis eta = 0 branch as well. That
    branch is a/(a^2+0) - b/(b^2+0) against `frequency_factor`'s 1/a - 1/b,
    the same function up to 5.7e-14 apart, and the RPA numbers are pinned to
    its own spelling.
  * the import gate cannot be broken from inside production -- importing
    anything of `src.gradients` there is circular and dies at import time --
    so its sensitivity is shown on `gradients.contour_deformation_adjoint`:
    the same probe returns the leaked `src.gradients.*` and fails, while the
    two production modules leave sys.modules clean.
"""
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from src.Base.numerics.constants import QP_POLE_OFFSET, QP_WINDOW_Z_MIN
from src.Base.utils.grids.grids import gauss_legendre_grid, gap_scaled_w0
from src.SingleReference.GW import contour_deformation as prod
from src.SingleReference.GW.real_screening import ExplicitRealScreening
from src.Base.orbital_spaces import ov_energies
from src.SingleReference.LinearResponse.imaginary_frequency import (
    f_rpa, frequency_factor)
from src.SingleReference.LinearResponse import space_time as ls_space_time
from src.SingleReference.LinearResponse.space_time import (
    frequency_blocks, owned_frequency_blocks)
from src.Base.orbital_spaces import get_occ_virt_indices
from src.gradients.gwbse import contour_deformation_adjoint as adjoint
from src.gradients.gwbse import qp_space_time as qst
from src.gradients.rpa import space_time_adjoint as sta
from tests.test_frequency_rows_serial_shaped import RowCountSensitiveNumpy
from tests.pins import assert_same

# --------------------------------------------------------- the reference code
# Verbatim reference implementations: a gate that recomputed the reference
# through production would gate nothing. The retired guard's floor, from the
# constants it was removed from with it.
QP_POLE_OFFSET_MIN = 1e-9

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
    """The reference quasiparticle loop of `qp_energy_cd`, with one
    `trace.append` per self-energy evaluation."""
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
        # the words follow production's: w here is the last iterate
        where = ('' if blocker is None else
                 f' The iterate was pinned against orbital {blocker} at '
                 f'eps={eps[blocker]:.6f} Ha and ended at {w:.6f} Ha, '
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
            f'{offset:.1e} Ha to reach it; the root is converged to '
            f'{tol:.0e}.', RuntimeWarning, stacklevel=2)
    residues = prod.residue_set(eps, nocc, w)
    sp = prod.sigma_cd_slope(p, w, Bp, eps, nocc, nu_points, nu_weights,
                             residues, wc, **common)
    # a converged root is returned with its Z, the quasiparticle threshold
    # read by the caller (production's rule)
    z = 1.0 / (1.0 - sp)
    return w, z, residues


# ------------------------------------------------------------------- the data

#: The synthetic spectrum of tests/test_cd_root_beside_an_orbital_energy.py: a
#: flat screening puts the root inside the retired guard (-4e-3) or outside it
#: (-1e-2).
GUARD_EPS = np.array([-0.9, -0.6, -0.35, 0.15, 0.4, 0.8])
GUARD_NOCC = 3
GUARD_NU = np.array([0.05, 0.3, 1.0, 4.0])
GUARD_WT = np.array([0.1, 0.3, 0.8, 3.0])


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


# ------------------------------------------------------------------- the gates

def test_the_gradient_side_holds_no_second_copy():
    """A re-implementation on the gradient side would be a second route: the
    adjoint screening subclasses the forward one, and what qp_space_time and
    space_time_adjoint import back from production is production's object."""
    assert issubclass(adjoint.ExplicitRealScreeningAdjoint,
                      ExplicitRealScreening)
    for name in ('cd_screening_contraction_multi', 'residue_route_auto'):
        assert getattr(qst, name) is getattr(prod, name), name
    assert sta.owned_frequency_blocks is owned_frequency_blocks


def test_frequency_factor_reproduces_the_old_formulas():
    """The dedup onto one function has to be a rounding-for-rounding copy (to
    a few ulp: the old formula is evaluated again)."""
    rng = np.random.default_rng(20250918)
    d = rng.uniform(0.05, 3.0, size=37)
    for w in (0.0, 1e-13, 0.017, 0.31, 2.5, -0.44):
        f, dfd, dfw = frequency_factor(d, w, True, slopes=True)
        f_old, dfd_old = _old_f_imaginary(d, w)
        assert_same(f, f_old, unit='1', quantity='root', label=('f', w))
        assert_same(dfd, dfd_old, unit='1', quantity='root', label=('df/dd', w))
        assert dfw is None
        assert_same(frequency_factor(d, w, True), f_old, unit='1',
                    quantity='root', label=('f alone', w))
        for eta in (0.0, 1e-3, 0.05):
            got = frequency_factor(d, w, False, eta, slopes=True)
            for a, b in zip(got, _old_f_real(d, w, eta)):
                assert_same(a, b, unit='1', quantity='root', label=(w, eta))
            assert_same(frequency_factor(d, w, False, eta),
                        _old_f_real(d, w, eta)[0], unit='1', quantity='root',
                        label=(w, eta, 'f alone'))


def test_the_rpa_frequency_factor_is_untouched():
    """`f_rpa` delegates on the imaginary axis and for eta != 0 on the
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
                assert_same(f_rpa(lr, d, w, axis), _old_f_rpa(lr, d, w, axis),
                            unit='1', quantity='root', label=(eta, w, axis))
    lr = type('LR', (), {'eta': 0.0})()
    shared = frequency_factor(d, 0.31, False, 0.0)
    own = f_rpa(lr, d, 0.31, False)
    assert not np.array_equal(shared, own), \
        'the two real-axis eta = 0 spellings coincide; then they should be one'
    assert np.abs(shared - own).max() < 1e-12


def assert_same_root(new, ref, where):
    """The production root is the reference loop's where that one is a
    quasiparticle: both converged to QP_ROOT_TOL on one equation, from
    different paths, and the same residue set frozen at it."""
    (w_new, z_new, res_new), (w_ref, z_ref, res_ref) = new, ref
    assert z_ref > QP_WINDOW_Z_MIN, (where, z_ref)
    assert abs(w_new - w_ref) < 1e-10, (where, w_new - w_ref)
    assert abs(z_new - z_ref) < 1e-8, (where, z_new - z_ref)
    assert res_new == res_ref, where


@pytest.mark.parametrize('screening', [-4e-3, -1e-2])
def test_the_root_is_the_old_loops_on_the_flat_models(screening):
    """The flat-screening models whose root sits inside and outside the
    retired pole guard (physical sign, W^c(0) < 0;
    tests/test_cd_root_beside_an_orbital_energy.py)."""
    Bp = np.zeros((3, len(GUARD_EPS)))
    wc = np.full((len(GUARD_NU), len(GUARD_EPS)), screening)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = _reference_newton(GUARD_NOCC - 1, Bp, GUARD_EPS, GUARD_NOCC,
                                GUARD_NU, GUARD_WT, wc)
    new = prod.qp_energy_cd(GUARD_NOCC - 1, Bp, GUARD_EPS, GUARD_NOCC,
                            GUARD_NU, GUARD_WT, wc=wc)
    assert_same_root(new, ref, screening)


def test_the_root_is_the_old_loops_on_water(water):
    """The real thing: the oxygen 1s, both states below the gap, the first
    above it. The core state sweeps every occupied residue, so this exercises
    the explicit real-frequency backend as well as the root search."""
    eps, nocc, b, c_ov, nu, wt = water
    d = ov_energies(eps, nocc)
    for p in states(nocc):
        bp = b[:, p, :]
        wc = prod.wc_explicit(bp, c_ov, d, nu)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ref = _reference_newton(p, bp, eps, nocc, nu, wt, wc, C_ov=c_ov)
        new = prod.qp_energy_cd(p, bp, eps, nocc, nu, wt, wc=wc, C_ov=c_ov)
        assert_same_root(new, ref, p)


def test_the_residue_term_is_what_the_gate_is_worth(water):
    """The water gate above compares numbers the residue term moves by 8.1e-3
    Ha on the oxygen 1s; without that term it would compare the integral
    alone."""
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
    """The residual's sign is worth a root: one Newton step from the core
    start goes 0.86 Ha the other way."""
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
    assert abs(good - flipped) > 0.5, abs(good - flipped)


def test_the_frequency_blocks_are_the_same_generator(monkeypatch):
    """`owned_frequency_blocks` lives in production; the gradient package
    imports it, and both walk the same blocks.

    The tile budget is picked so the axis really is split -- more than one
    block and more than one frequency in a block -- because a one-frequency
    block hides every index a split can get wrong: the block-local row is the
    global row there, and the frequencies a block names are trivially the ones
    it transformed.

    Each row is pinned to the transform of its whole SERIAL block, not to a
    transform of the owned rows alone. The owned-rows call is the defect, not
    a gauge the split may choose: a GEMM's rows depend on the call's row count
    on OpenBLAS, and that broke the bitwise quasiparticle roots over 8 ranks.
    MKL keeps such rows bitwise, so the transform runs on a tensordot whose
    rows depend on the row count (tests/test_frequency_rows_serial_shaped.py),
    or the pin could not tell the two calls apart.
    """
    monkeypatch.setattr(ls_space_time, 'np', RowCountSensitiveNumpy())
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
            assert_same(x, y, unit='1', quantity='root', label=kwargs)
        # each row is its serial block's row, and the blocks together carry
        # every owned frequency once
        serial = {}
        for k0, k1 in frequency_blocks(len(cosft_wt), proj_tau.shape[-1], tile,
                                       kwargs.get('live', 3)):
            ks = list(range(k0, k1))
            serial.update(zip(ks, ls_space_time.np.tensordot(
                cosft_wt[ks], proj_tau, axes=(1, 0))))
        for ks, blk in one:
            for k, row in zip(ks, blk):
                assert_same(row, serial[k], unit='1', quantity='root',
                            label=(kwargs, k))
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
