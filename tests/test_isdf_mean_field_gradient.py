"""The force of a mean field whose exchange comes from ISDF factors.

THE GATE IS THE POINT. An analytic gradient is only correct against the energy
its own route reports, so every comparison here is with a CENTRAL FINITE
DIFFERENCE of that energy, swept in h. The signature of a correct pairing is
that the residual falls as h^2 -- the finite difference's own truncation error
-- rather than settling on a constant, which is what a missing term looks like
on a smooth surface. pyscf's gradient on this mean field sits 4.0e-4 Ha/Bohr
away and does NOT fall (tests/test_isdf_jk_gradient_guard.py keeps that
measurement).

What the residual falls TO is set by the conditioning of the fit's Gram matrix,
which the adjoint inverts twice: 1.1e-8 Ha/Bohr on Hartree-Fock at the
production regularization of 4e-7 (cond(G) = 2e8), 1.0e-9 on B3LYP where a_x is
0.2, and 5e-12 once the Gram is floored at 1e-2. That is the numerics of
`fit_adjoint`, not an omission, and the sweep below says so: a missing term
would not care about the regularization.

Translational invariance is asserted everywhere it is free. It catches a
dropped skeleton term instantly -- the interpolation points' own translation is
worth 9.7e-4 Ha/Bohr here and breaks it -- but it is blind to the frames
turning with the environment, which is worth 5.0e-5 and leaves it at 1e-16.
"""
import numpy as np
import pytest
from pyscf import df, dft, gto, scf

from src.Base.constants import BOHR_TO_ANGSTROM
from src.Base.isdf_jk import ISDFJK, isdf_jk
from src.Base.separable_ri import DEFAULT_REGULARIZATION
from src.gradients.isdf_derivatives import (fock_partial_skeleton_df,
                                            xc_hybrid_coeff)
from src.gradients.isdf_mean_field import (exchange_free_reference,
                                           isdf_exchange_skeleton,
                                           isdf_mean_field_gradient,
                                           require_isdf_gradient_support)

AUXBASIS = 'cc-pvdz-ri'
GEOMETRY = (('O', (0.0, 0.0, 0.1173)), ('H', (0.0, 0.7572, -0.4692)),
            ('H', (0.0, -0.7572, -0.4692)))
#: Angstrom. Three halvings, so h^2 (a factor of 4 each time) is distinguishable
#: from a constant, which is the whole content of the gate.
STEPS = (4e-3, 2e-3, 1e-3, 5e-4)
#: The displaced coordinate: H(1) along z, where the defect was measured.
MOVED = (1, 2)


def water(shift=None):
    """The reference geometry, or one atom displaced by `shift` Angstrom."""
    atom = [[s, list(c)] for s, c in GEOMETRY]
    if shift is not None:
        ia, x, d = shift
        atom[ia][1][x] += d
    return gto.M(atom=atom, basis='cc-pvdz', verbose=0)


def mean_field(xc, shift=None, route='isdf', regularization=None):
    """A converged SCF, exchange from the interpolation or from the fit.

    conv_tol_grad is 1e-11: the force assumes the occupied-virtual Fock block
    vanishes, and at 1e-10 the analytic gradient here moves by 7e-9 Ha/Bohr,
    which is the size of everything this file measures.
    """
    mol = water(shift)
    mf = dft.RKS(mol, xc=xc) if xc else scf.RHF(mol)
    mf = (isdf_jk(mf, auxbasis=AUXBASIS) if route == 'isdf'
          else mf.density_fit(auxbasis=AUXBASIS))
    if regularization is not None:
        mf.with_df.regularization = regularization
    if xc:
        mf.grids.prune = None
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-11
    mf.kernel()
    assert mf.converged
    return mf


def central_difference(energy, ia, x, h):
    """dE/dR in Hartree/Bohr from a two-point difference of `energy(shift)`."""
    return ((energy((ia, x, h)) - energy((ia, x, -h)))
            / (2.0 * h / BOHR_TO_ANGSTROM))


def sweep(analytic, energy, ia, x, steps=STEPS):
    """[analytic - finite difference] at each step, largest step first."""
    return [analytic - central_difference(energy, ia, x, h) for h in steps]


def richardson(residuals):
    """The h -> 0 limit of a residual falling as h^2, from the two finest steps.

    r(h) = e + c h^2 with e the gradient's own error, so (4 r(h/2) - r(h))/3
    removes the finite difference's truncation and leaves e.
    """
    return (4.0 * residuals[-1] - residuals[-2]) / 3.0


def exchange_energy(mf, dm):
    """-(a_x/4) Tr[dm K] with K from this mean field's ISDF factors."""
    a_x = xc_hybrid_coeff(mf)[1]
    vk = mf.with_df.get_jk(dm, hermi=1, with_j=False)[1]
    return -0.25 * a_x * float(np.einsum('ij,ji->', dm, vk))


def test_the_exchange_energy_is_one_scalar_in_the_factors():
    """E_K = -(a_x/4) sum_PQ Z_PQ W_PQ^2 with W = X dm X^T, and K is exactly
    its derivative in the density -- which is why the force needs no orbital
    response beyond the Pulay term pyscf already carries."""
    mf = mean_field('b3lyp')
    w = mf.with_df
    dm = mf.make_rdm1()
    a_x = xc_hybrid_coeff(mf)[1]
    Z = w.M.T @ (w.auxmol.intor('int2c2e', aosym='s1') @ w.M)
    W = w.X @ dm @ w.X.T
    assert abs(-0.25 * a_x * np.einsum('PQ,PQ,PQ->', Z, W, W)
               - exchange_energy(mf, dm)) < 1e-12

    # dE_K/dD = -(a_x/2) K, along a random symmetric direction
    rng = np.random.default_rng(0)
    s = rng.standard_normal(dm.shape)
    s = s + s.T
    h = 1e-5
    slope = (exchange_energy(mf, dm + h * s)
             - exchange_energy(mf, dm - h * s)) / (2.0 * h)
    predicted = -0.5 * a_x * float(np.einsum('ij,ij->', mf.with_df.get_jk(
        dm, hermi=1, with_j=False)[1], s))
    assert abs(slope - predicted) < 1e-6 * max(abs(predicted), 1.0)


def test_the_exchange_skeleton_follows_its_own_energy():
    """The new branch on its own, at a FROZEN density matrix, so nothing of
    pyscf's is in the comparison: the fit, the collocation, the interpolation
    points and the frames against a difference of the exchange energy alone."""
    mf = mean_field('b3lyp')
    dm = mf.make_rdm1()
    a_x = xc_hybrid_coeff(mf)[1]
    g = isdf_exchange_skeleton(mf)
    assert np.abs(g.sum(axis=0)).max() < 1e-10

    def energy(shift):
        w = ISDFJK(water(shift), auxbasis=AUXBASIS, check_tol=None)
        w.build()
        vk = w.get_jk(dm, hermi=1, with_j=False)[1]
        return -0.25 * a_x * float(np.einsum('ij,ji->', dm, vk))

    ia, x = MOVED
    res = sweep(g[ia, x], energy, ia, x, steps=STEPS[:3])
    for coarse, fine in zip(res, res[1:]):
        assert abs(coarse) > 3.0 * abs(fine), res
    assert abs(richardson(res)) < 1e-8, res


@pytest.mark.parametrize('xc', ['b3lyp', None], ids=['b3lyp', 'hartree-fock'])
def test_the_force_falls_as_h_squared_onto_its_own_energy(xc):
    """THE GATE. Analytic against a central difference of the ISDF route's own
    SCF energy, swept in h, with the fitted route as the control that says what
    a correct pairing looks like on the same molecule."""
    out = {}
    for route in ('isdf', 'df'):
        mf = mean_field(xc, route=route)
        if route == 'isdf':
            g = isdf_mean_field_gradient(mf)
        else:
            grad = mf.Gradients()
            grad.grid_response = bool(xc)
            g = np.asarray(grad.kernel())
        assert np.abs(g.sum(axis=0)).max() < 1e-10, route
        ia, x = MOVED
        out[route] = sweep(
            g[ia, x], lambda s: mean_field(xc, s, route=route).e_tot, ia, x)
    for route, res in out.items():
        # h^2: each halving of the step cuts the residual by about four
        for coarse, fine in zip(res, res[1:]):
            assert abs(coarse) > 3.0 * abs(fine), (route, res)
        assert abs(res[-1]) < 1e-7, (route, res)
    # what the interpolated route's residual falls TO, with the truncation
    # extrapolated away: 1.0e-09 on B3LYP and 1.0e-08 on Hartree-Fock, where
    # a_x is five times larger. The fitted control reaches 3e-13, and the gap
    # is the fit's conditioning -- see the regularization sweep below.
    assert abs(richardson(out['isdf'])) < 1e-7, out


def test_the_remaining_error_is_the_fits_conditioning():
    """What the residual falls TO is the Gram matrix the fit adjoint inverts
    twice, not a missing term: flooring it at 1e-2 takes cond(G) from 2e8 to
    7.9e3 and the error from 1e-8 to 1e-11. A missing term would not care."""
    ia, x = MOVED
    errors = {}
    for reg in (DEFAULT_REGULARIZATION, 1e-2):
        mf = mean_field(None, regularization=reg)
        g = isdf_mean_field_gradient(mf)
        errors[reg] = richardson(sweep(
            g[ia, x], lambda s: mean_field(None, s, regularization=reg).e_tot,
            ia, x, steps=STEPS[1:]))
    assert abs(errors[DEFAULT_REGULARIZATION]) < 1e-7, errors
    assert abs(errors[1e-2]) < 0.1 * abs(errors[DEFAULT_REGULARIZATION]), errors


def test_the_reference_is_missing_exactly_the_exchange():
    """The half of the assembly that is not new, pinned without a finite
    difference: put the FITTED exchange skeleton where the ISDF one goes and
    pyscf's own density-fitted gradient comes back, to 1e-11. So
    `exchange_free_reference` carries the one-electron, Coulomb,
    exchange-correlation and Pulay terms and nothing else."""
    mf = mean_field('b3lyp', route='df')
    a_x = xc_hybrid_coeff(mf)[1]
    grad = mf.Gradients()
    grad.grid_response = True
    ref = exchange_free_reference(mf).Gradients()
    ref.grid_response = True
    auxmol = df.addons.make_auxmol(mf.mol, auxbasis=AUXBASIS)
    # E_2 of the folded Fock partial at gamma = D is twice the energy, so the
    # exchange half of the skeleton is halved to become dE_K/dR.
    fitted = 0.5 * fock_partial_skeleton_df(mf, auxmol, np.diag(mf.mo_occ),
                                            mf.mol.nelectron // 2,
                                            channels=[(0.0, a_x)],
                                            coulomb=False)
    assert np.abs(np.asarray(grad.kernel())
                  - np.asarray(ref.kernel()) - fitted).max() < 1e-11


def test_what_is_not_covered_refuses_by_name():
    """Everything outside the restricted closed-shell hybrid on df-direct J."""
    mol = water()
    cases = {
        'interpolated Coulomb': isdf_jk(dft.RKS(mol, xc='b3lyp'),
                                        auxbasis=AUXBASIS, j_route='isdf'),
        'unrestricted': isdf_jk(dft.UKS(mol, xc='b3lyp'), auxbasis=AUXBASIS),
        'restricted open shell': isdf_jk(dft.ROKS(mol, xc='b3lyp'),
                                         auxbasis=AUXBASIS),
    }
    for label, mf in cases.items():
        with pytest.raises(NotImplementedError):
            require_isdf_gradient_support(mf, label)

    # a grid handed in from outside: the rows are points, but which atom owns
    # which is exactly what the point chain needs and cannot recover
    built = ISDFJK(mol, auxbasis=AUXBASIS, check_tol=None)
    built.build()
    injected = isdf_jk(dft.RKS(mol, xc='b3lyp'), auxbasis=AUXBASIS)
    injected.with_df.coords = built.coords
    with pytest.raises(NotImplementedError):
        require_isdf_gradient_support(injected, 'injected coords')

    # a fitted mean field is pyscf's business, not this module's
    with pytest.raises(TypeError):
        require_isdf_gradient_support(
            dft.RKS(mol, xc='b3lyp').density_fit(auxbasis=AUXBASIS), 'fitted')

    # and the covered cases are not refused. A range-separated hybrid is one of
    # them now: its exchange is one channel per operator sharing the bare fit,
    # and the exchange-free reference zeroes the coefficients instead of
    # subtracting HF by name, which is what could not be written down before.
    # Gated in test_range_separated_exchange.py.
    for label, xc in (('covered', 'b3lyp'), ('range-separated', 'lrc-wpbeh')):
        require_isdf_gradient_support(
            isdf_jk(dft.RKS(mol, xc=xc), auxbasis=AUXBASIS), label)
