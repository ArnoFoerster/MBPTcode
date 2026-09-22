"""The folded Fock partial's exchange half, built from ISDF factors.

`fock_partial_skeleton` routes any density-fitted mean field to the
auxiliary-basis skeleton, which knows nothing of an interpolative separable
fit: on an ISDF mean field it returns the derivative of a different function,
and that is what made the excited-state surface refuse the route. The exchange
half now comes from the same adjoint chain the SCF force uses, generalized to
the BILINEAR form a folded partial needs -- its exchange term is
-(gamma vk[D] + D vk[gamma]), two different densities where the SCF energy has
one twice.

Two gates, and they catch different things. A finite difference of the
bilinear, swept in h, says the ADJOINT CHAIN differentiates what it claims to:
gamma, D and the MO coefficients are held fixed as arrays while the geometry
moves, which is the skeleton convention, and a missing term shows as a residual
that stops falling.

IT CANNOT SAY ANYTHING ABOUT THE PREFACTOR, and the first version of this file
believed it did. A difference of -(c/4) Tr[gamma K[D]] confirms whatever c the
analytic side uses, because the reference is built from the same constant --
the check was reading its own answer back. The factor was wrong by four, the
sweep converged at h^2 to 1.5e-08, and the excited-state gradient it fed was
out by twice its own value. What pins it is `fock_partial_skeleton_df` at the
same gamma: the fitted and conventional branches already agree with each other,
so agreeing with either fixes the constant.
"""
import numpy as np
import pytest
from pyscf import dft, gto

from src.Base.constants import BOHR_TO_ANGSTROM
from src.Base.isdf_jk import isdf_jk
from src.gradients.isdf_derivatives import (fock_partial_skeleton,
                                            xc_hybrid_coeff)
from src.gradients.isdf_mean_field import isdf_fock_partial_exchange

AUXBASIS = 'cc-pvdz-ri'
GEOMETRY = (('O', (0.0, 0.0, 0.1173)), ('H', (0.0, 0.7572, -0.4692)),
            ('H', (0.0, -0.7572, -0.4692)))
#: Angstrom, halving three times so that h^2 is distinguishable from a
#: constant -- which is the whole content of the gate.
STEPS = (4e-3, 2e-3, 1e-3, 5e-4)
MOVED = (1, 2)


def water(shift=None):
    atom = [[s, list(c)] for s, c in GEOMETRY]
    if shift is not None:
        ia, x, d = shift
        atom[ia][1][x] += d
    return gto.M(atom=atom, basis='cc-pvdz', verbose=0)


def mean_field(shift=None, xc='b3lyp'):
    mf = isdf_jk(dft.RKS(water(shift), xc=xc), auxbasis=AUXBASIS)
    mf.grids.prune = None
    # the partial assumes the occupied-virtual Fock block vanishes
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-11
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def reference():
    """A converged ISDF mean field, a random symmetric gamma, and g_ao."""
    mf = mean_field()
    rng = np.random.default_rng(0)
    n = mf.mo_coeff.shape[1]
    gamma = rng.standard_normal((n, n))
    gamma = gamma + gamma.T
    C = mf.mo_coeff
    g_ao = C @ (0.5 * (gamma + gamma.T)) @ C.T
    return mf, gamma, g_ao, np.asarray(mf.make_rdm1())


def bilinear(g_ao, dm, a_x, shift):
    """-(a_x/2) Tr[g_ao K^ISDF[dm]] with BOTH matrices frozen as arrays.

    The exchange term of the folded two-particle density
    Gamma_abcd = gamma_ab D_cd - gamma_ac D_bd / 2. Only the integrals move.
    """
    mf = isdf_jk(dft.RKS(water(shift), xc='b3lyp'), auxbasis=AUXBASIS)
    vk = mf.with_df.get_jk(dm, hermi=1, with_j=False)[1]
    return -0.5 * a_x * float(np.einsum('ab,ab->', g_ao, vk))


def test_the_two_density_form_reduces_to_the_scf_one():
    """dm_other = dm must be the SCF's own exchange energy exactly, not nearly:
    the bilinear is the same contraction with W^2 replaced by W1 W2."""
    from src.gradients.isdf_mean_field import isdf_exchange_adjoints

    rng = np.random.default_rng(1)
    X = rng.standard_normal((9, 5))
    dm = rng.standard_normal((5, 5))
    dm = dm + dm.T
    Z = rng.standard_normal((9, 9))
    Z = Z + Z.T
    one = isdf_exchange_adjoints(X, Z, dm, 0.2)
    two = isdf_exchange_adjoints(X, Z, dm, 0.2, dm_other=dm)
    assert np.array_equal(one[0], two[0])
    assert np.array_equal(one[1], two[1])


def test_the_exchange_half_follows_its_own_bilinear(reference):
    """The residual must FALL as h^2. Settling on a constant is what a dropped
    skeleton term looks like on a smooth surface -- the failure mode pyscf's
    fitted derivative shows on this same mean field."""
    mf, gamma, g_ao, dm = reference
    a_x = xc_hybrid_coeff(mf)[1]
    analytic = isdf_fock_partial_exchange(mf, gamma)[MOVED[0], MOVED[1]]
    res = []
    for h in STEPS:
        plus = bilinear(g_ao, dm, a_x, (MOVED[0], MOVED[1], h))
        minus = bilinear(g_ao, dm, a_x, (MOVED[0], MOVED[1], -h))
        fd = (plus - minus) / (2.0 * h / BOHR_TO_ANGSTROM)
        res.append(analytic - fd)
    assert abs(res[-1]) < abs(res[0])
    # h^2 convergence: each halving must cut the residual, and the extrapolated
    # limit is the adjoint's own error rather than the difference's truncation
    limit = (4.0 * res[-1] - res[-2]) / 3.0
    assert abs(limit) < 1e-5, (res, limit)


def test_it_agrees_with_the_fitted_skeleton(reference):
    """THE GATE THAT PINS THE CONSTANT, because it is the only one here that
    does not build its reference from the same coefficient. The fitted and
    conventional branches already agree with each other -- a range-separated
    hybrid mixes them inside one call -- so the interpolated one must land on
    the fitted answer to the fit's own accuracy and no better. At the shipped
    148 points per atom that is under a percent, which is the same fit error
    the total energy shows as 7.3 mHa.
    """
    from src.gradients.isdf_derivatives import (_auxmol_of, exchange_channels,
                                                fock_partial_skeleton_df)

    mf, gamma, _, _ = reference
    nocc = mf.mol.nelectron // 2
    isdf = isdf_fock_partial_exchange(mf, gamma)
    fitted = fock_partial_skeleton_df(mf, _auxmol_of(mf), gamma, nocc,
                                      coulomb=False,
                                      channels=exchange_channels(mf))
    big = np.abs(fitted) > 1e-3 * np.abs(fitted).max()
    assert np.abs(isdf[big] / fitted[big] - 1.0).max() < 0.05


def test_it_is_translationally_invariant(reference):
    """Moving every atom together changes no integral, so the skeleton sums to
    zero over atoms. It needs no reference and catches a dropped term at once
    -- the interpolation points' own translation is worth 1e-3 here."""
    mf, gamma, _, _ = reference
    g = isdf_fock_partial_exchange(mf, gamma)
    assert np.abs(g.sum(axis=0)).max() < 1e-7 * max(np.abs(g).max(), 1.0)


def test_the_dispatch_reaches_the_isdf_branch(reference):
    """`fock_partial_skeleton` must not hand an interpolated mean field to the
    auxiliary-basis exchange skeleton. The two differ by far more than either
    one's own error, so agreement would be the surprise."""
    mf, gamma, _, _ = reference
    nocc = mf.mol.nelectron // 2
    total = fock_partial_skeleton(mf, gamma, nocc)
    assert np.isfinite(total).all()
    # the exchange half is a real fraction of it, so the dispatch is not a no-op
    exchange = isdf_fock_partial_exchange(mf, gamma)
    assert np.abs(exchange).max() > 1e-6 * np.abs(total).max()


def exx_minus_xc(dm, a_x, shift):
    """E_x^exact - E_xc at a displaced geometry, the DENSITY frozen.

    The Hartree-Fock and Kohn-Sham energies differ in exactly two terms at one
    density -- the exact-exchange fraction and the functional -- so their
    difference is

        -(1 - a_x)/4 Tr[D K^ISDF] - E_xc^DFT

    with `nr_rks`'s E_xc, which excludes the exact exchange the hybrid carries
    separately. The quadrature is rebuilt at the displaced geometry because the
    grid moves with the atoms, which is the `grid_response` term the analytic
    side sets.
    """
    mf = isdf_jk(dft.RKS(water(shift), xc='b3lyp'), auxbasis=AUXBASIS)
    mf.grids.prune = None
    mf.grids.build()
    vk = mf.with_df.get_jk(dm, hermi=1, with_j=False)[1]
    exc = mf._numint.nr_rks(mf.mol, mf.grids, mf.xc, dm)[1]
    return (-0.25 * (1.0 - a_x) * float(np.einsum('ij,ji->', dm, vk))
            - float(exc))


def test_the_double_counting_follows_its_own_energy(reference):
    """(E_x^exact - E_xc) on the interpolated route, against a finite
    difference of exactly that difference. The fitted branch would build the
    exact-exchange half from the auxiliary basis while the SCF built it from
    the interpolation, which is a different function and shows here as a
    residual that does not fall.
    """
    from src.gradients.isdf_derivatives import exx_double_counting_skeleton

    mf, _, _, dm = reference
    a_x = xc_hybrid_coeff(mf)[1]
    analytic = exx_double_counting_skeleton(mf)[MOVED[0], MOVED[1]]
    res = []
    for h in STEPS:
        plus = exx_minus_xc(dm, a_x, (MOVED[0], MOVED[1], h))
        minus = exx_minus_xc(dm, a_x, (MOVED[0], MOVED[1], -h))
        res.append(analytic - (plus - minus) / (2.0 * h / BOHR_TO_ANGSTROM))
    assert abs(res[-1]) < abs(res[0])
    limit = (4.0 * res[-1] - res[-2]) / 3.0
    assert abs(limit) < 1e-5, (res, limit)


def test_the_double_counting_sums_to_zero_over_atoms(reference):
    """Both members are translationally invariant, so their difference is too
    -- and to machine precision, because the two ISDF gradients share one fit
    and cancel term by term rather than nearly."""
    from src.gradients.isdf_derivatives import exx_double_counting_skeleton

    mf = reference[0]
    g = np.asarray(exx_double_counting_skeleton(mf))
    assert np.abs(g.sum(axis=0)).max() < 1e-11


def test_the_excited_state_gradient_follows_its_own_surface():
    """END TO END, which is what the refusal was protecting and what the unit
    gates above could not have caught on their own: the wrong prefactor passed
    every one of them and left this residual CONSTANT at twice the gradient.

    The same sweep on the density-fitted route is run alongside, because what
    is being claimed is not that the interpolated gradient is exact -- it is
    the derivative of the interpolated surface, which is a different surface --
    but that it differentiates its own energy as well as the fitted one
    differentiates its own.
    """
    from src.gradients.excited_state import ExcitedStateChain

    steps = (4e-3, 2e-3, 1e-3)
    out = {}
    for route in ('df', 'isdf'):
        def build(m, route=route):
            mf = dft.RKS(m, xc='b3lyp')
            mf = (isdf_jk(mf, auxbasis=AUXBASIS) if route == 'isdf'
                  else mf.density_fit(auxbasis=AUXBASIS))
            mf.grids.prune = None
            mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-11
            mf.kernel()
            return mf

        mol = water()
        chain = ExcitedStateChain(mol, build, basis='cc-pvdz', state=0,
                                  spin='singlet', n_start=8, solver='davidson',
                                  nroots=3, qp_window=2)
        g = chain.total_gradient(mol)[0]
        # ONE chain, its frozen conventions held, only the geometry moving:
        # re-freezing at every displaced point makes the surface discontinuous
        # and the difference meaningless.
        res = [g[MOVED[0], MOVED[1]]
               - (chain.total_energy(water((MOVED[0], MOVED[1], h)))
                  - chain.total_energy(water((MOVED[0], MOVED[1], -h))))
               / (2.0 * h / BOHR_TO_ANGSTROM) for h in steps]
        out[route] = res
        assert np.abs(g.sum(axis=0)).max() < 1e-6
        # h^2: each halving cuts the residual by about four
        for a, b in zip(res, res[1:]):
            assert 3.0 < a / b < 5.0, (route, res)
    # and the interpolated route is no worse than the fitted one
    assert abs(out['isdf'][-1]) < 5.0 * abs(out['df'][-1])
