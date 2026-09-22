"""ONE ISDF fit M, built two ways: how far apart are they, and why.

The separable (ISDF) fit solves one least-squares problem -- Duchemin and
Blase, JCP 150, 174120 (2019) eqs 8-9, balanced and Tikhonov-regularized --
and this tree realizes it twice:

  production   `GW.space_time.separable_factors` -> `build_separable_ri` ->
               `fit_M_streaming`. The Gram matrix never sees D: it is the
               elementwise product (A A^T) .* (B B^T) of two small collocation
               products, and F D^T is accumulated over shell-pair blocks, so
               the n_k x n_rho array is never held. LAPACK `posv` solves.
  the chain    `gradients.factor_chain.FrozenFactorization.shareable_factors`
               -> `fit_M_stable`, on a FROZEN pair layout: D explicit,
               G = D D^T, `cho_factor`/`cho_solve`.

Both are handed the same grid. Measured on the shipped radii at
`DEFAULT_COUNTS`, the chain's run-time `optimize_atomic_radii` returns the
tabulated row for H, C and O, so the interpolation points are identical point
for point -- (a) is not where they differ.

WHAT DIFFERS, cc-pVDZ, relative to the largest entry:

    quantity                                   water        formaldehyde
    M           (naux, nk)                     1.11e-08     8.01e-06
    D = M^T V^(1/2)                            3.76e-08     2.30e-05
    M D, the fitted (mu nu|P) V^-1             4.63e-13     4.48e-08
    |M D - F|, each fit's OWN residual         8.1e-13      5.8e-09
                                               (4.5235e-03) (8.6794e-03)
    E_J + E_K from the fitted ERI              5.3e-12 Ha   --

THE CAUSE IS NOT THE ESTIMATOR. The regularization (`DEFAULT_REGULARIZATION`),
the auxiliary metric, the angular weights, the pair layout and the grid are the
same object on both sides; every sub-route was bisected and none of them is an
algebraic difference:

    balanced Gram, production factorized vs chain explicit   7.1e-16 / 5.9e-13
    row scale, diag(S) vs |D_k|                              1.0e-15 / 6.4e-11
    F D^T, np.linalg.solve(V, .) vs lu_factor/lu_solve(V)    8.2e-14 / 1.1e-13

It is CONDITIONING. The balanced Gram matrix is numerically singular -- its
smallest eigenvalue is -2e-15 and 169 of 444 eigenvalues (162 of 592 at
formaldehyde) lie BELOW the 4e-07 Tikhonov shift -- so M is fixed only up to
the near-null space of D, and cond(G + 4e-07) = 1.58e+08. A 1e-16 relative
perturbation of the Gram alone moves M by 1.01e-08 relative, which is the
water difference to a factor 1.1. That is the whole of it at water.

Formaldehyde adds a second, larger term: PRODUCTION BUILDS THE GRAM OVER THE
UNSCREENED PAIR SET and its right-hand side over the screened one, where the
chain screens both. `fit_M_streaming` says so and argues the dropped columns
are "orders below the Tikhonov shift", and for the Gram they are (5.9e-13
relative at the 6 columns of 1444 dropped here). Amplified by 1.5e+08 that is
8e-06 in M. The two are then least-squares solutions of DIFFERENT problems, and
the gap widens with the tolerance: at `pair_tol` = 1e-04 on water, 8 columns of
576 dropped, M differs by 0.94 and production's residual on the full test set
is 7.4e-02 against the chain's 4.5e-03. This is pinned below.

WHAT SURVIVES. The near-null directions are invisible to a pair density, which
is what the fit is for: the fitted three-centre integrals agree 4-5 orders
better than M does, both fits report the same residual, and the two-electron
energy from the two factorizations differs by 5e-12 Ha on water. The gradient
chain differentiates its OWN fit for both the energy and the force, so the
1e-08 Ha/Bohr reproducibility floor is untouched by this; what the numbers say
is that a chain surface and a production `calc_qp_energy` at one geometry are
not the same number below these levels.

This test PINS the measurement. It does not unify the fits: which realization
is the one, and what the other's callers pay, is the owner's call.

Every pin here fires. Giving the chain's `fit_M_stable` a Tikhonov shift 0.1%
larger than production's -- one realization made a different estimator, nothing
else touched -- moves M by 4.7e-04, D by 4.4e-04, the fitted integrals by
1.2e-07, the residual gap to 1.2e-06 and the two-electron energy by 8.7e-07 Ha,
each past its bound above. That is also the size to keep in mind: the
regularization constant is worth four orders more than the choice of
realization.
"""
import numpy as np
import pytest
import scipy.linalg
from pyscf import df as pyscf_df, gto, scf

import src.Base.separable_ri as separable_ri
import src.gradients.factor_chain as factor_chain
from src.Base.separable_ri import (DEFAULT_PAIR_TOL, DEFAULT_REGULARIZATION,
                                   atomic_grid, aux_metric_sqrt, fit_M_stable,
                                   fit_M_streaming, molecular_points_covariant)
from src.SingleReference.GW.space_time import DEFAULT_COUNTS
from src.gradients.factor_chain import FrozenFactorization

# `test_set_layout`, `test_set_D` and `test_set_three_center` stay
# module-qualified: imported by name, pytest collects them as test functions.

BASIS, AUX = 'cc-pvdz', 'cc-pvdz-ri'
MOLECULES = {
    'water': 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
    'formaldehyde': ('C 0 0 -0.5395; O 0 0 0.6636; '
                     'H 0 0.9445 -1.1090; H 0 -0.9445 -1.1090'),
}

#: Measured relative differences, times a small margin, as (M, D, M D, residual).
#: A future drift past these is a change in one of the two realizations.
PINNED = {'water': (3e-8, 1e-7, 2e-12, 5e-12),
          'formaldehyde': (2.5e-5, 7e-5, 1.5e-7, 2e-8)}

#: A pair tolerance at which screening BITES on water -- 8 of 576 columns go.
#: The default drops nothing there, which is exactly how the asymmetry between
#: production's unscreened Gram and its screened right-hand side stays invisible
#: on the smallest test molecule.
SCREENING_BITES = 1e-4

#: |E_J + E_K| from the two factorizations, water, with margin (measured 5.3e-12).
ENERGY_TOL = 1e-10


def molecule(name):
    return gto.M(atom=MOLECULES[name], basis=BASIS, verbose=0)


def shipped_grid(mol):
    """The interpolation points `separable_factors` places at `DEFAULT_COUNTS`."""
    radii, origins = {}, {}
    for el in sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)}):
        radii[el], origins[el] = atomic_grid(el, mol.basis, AUX, DEFAULT_COUNTS)
    return molecular_points_covariant(mol, radii, origin_by_element=origins)


def two_fits(name, pair_tol=DEFAULT_PAIR_TOL):
    """Both realizations of M on one grid, with the pieces each is made of."""
    mol = molecule(name)
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=AUX)
    crd = shipped_grid(mol)
    naux = auxmol.nao_nr()
    V = auxmol.intor('int2c2e', aosym='s1')
    layout = separable_ri.test_set_layout(mol, crd, pair_tol=pair_tol)
    mu_i, nu_i, wc_l = layout
    D_test = separable_ri.test_set_D(mol, auxmol, crd, layout)
    e3 = factor_chain.test_set_three_center(mol, auxmol, mu_i, nu_i)
    F = np.hstack([np.linalg.solve(V, e3.T) * wc_l[None, :], np.eye(naux)])
    M_chain = fit_M_stable(D_test, F)
    M_prod = fit_M_streaming(mol, auxmol, crd, pair_tol=pair_tol)
    V_half = aux_metric_sqrt(auxmol, None, V=V)
    return dict(mol=mol, auxmol=auxmol, coords=crd, naux=naux, D_test=D_test,
                F=F, M_chain=M_chain, M_prod=M_prod,
                D_chain=M_chain.T @ V_half, D_prod=M_prod.T @ V_half)


@pytest.fixture(scope='module')
def fits():
    return {name: two_fits(name) for name in MOLECULES}


def relative(a, b):
    """max |a - b| against the larger of the two, which is how M is judged."""
    scale = max(np.abs(a).max(), np.abs(b).max())
    return float(np.abs(a - b).max() / scale) if scale else 0.0


def balanced_gram(D):
    """G = Dt Dt^T on unit rows, and the inverse row norms -- eqs 8-9's balancing."""
    s = np.sqrt(np.einsum('kr,kr->k', D, D))
    d = 1.0 / np.where(s == 0.0, 1.0, s)
    Dt = D * d[:, None]
    return Dt @ Dt.T, d


def solve_balanced(G, FD_balanced, d):
    """The regularized Cholesky solve both realizations end in."""
    G = G.copy()
    G[np.diag_indices_from(G)] += DEFAULT_REGULARIZATION
    cho = scipy.linalg.cho_factor(G, lower=True)
    return scipy.linalg.cho_solve(cho, FD_balanced.T).T * d[None, :]


# ------------------------------------------------------- (a) the same grid
@pytest.mark.parametrize('name', list(MOLECULES))
def test_both_realizations_start_from_the_same_interpolation_points(name):
    """The chain's re-optimized radii ARE the shipped row at these counts.

    Nothing below means anything otherwise: two fits on two grids are two
    functionals, and their difference would be the grid's, not the fit's.
    """
    mol = molecule(name)
    chain = FrozenFactorization(mol, auxbasis=AUX).coords(mol)
    assert np.array_equal(chain, shipped_grid(mol))


# ----------------------------------------------- (b) how far apart they are
@pytest.mark.parametrize('name', list(MOLECULES))
def test_the_two_fits_agree_at_the_measured_level(fits, name):
    """M, D, the fitted integrals and the residual, each against its pin."""
    w = fits[name]
    m_tol, d_tol, md_tol, res_tol = PINNED[name]
    assert relative(w['M_chain'], w['M_prod']) < m_tol
    assert relative(w['D_chain'], w['D_prod']) < d_tol
    # the object the fit is FOR, sum_k rho_r(k) M[P,k] ~ (r|P) V^-1
    assert relative(w['M_chain'] @ w['D_test'],
                    w['M_prod'] @ w['D_test']) < md_tol
    residuals = [np.abs(M @ w['D_test'] - w['F']).max()
                 for M in (w['M_chain'], w['M_prod'])]
    assert abs(residuals[0] - residuals[1]) < res_tol
    assert min(residuals) > 1e-3        # neither fit is exact; this is a fit


# -------------------------------------------------------- (c) the cause
@pytest.mark.parametrize('name', list(MOLECULES))
def test_the_fit_is_determined_only_up_to_the_conditioning(fits, name):
    """G is singular and the shift is what makes M exist at all.

    Most of the spectrum lies UNDER the Tikhonov shift, so the fit has a large
    near-null space and cond(G + shift) is set by the shift itself.
    """
    G, _ = balanced_gram(fits[name]['D_test'])
    ev = np.linalg.eigvalsh(G)
    assert ev.min() < 1e-12                       # numerically singular
    assert (ev < DEFAULT_REGULARIZATION).sum() > 0.25 * len(ev)
    shifted = ev + DEFAULT_REGULARIZATION
    assert 1e8 < shifted.max() / shifted.min() < 1e9


def test_the_water_difference_is_the_amplified_last_bit(fits):
    """A 1e-16 relative perturbation of the Gram moves M as far as the two
    realizations are apart.

    Water drops no test-set column, so the two Gram matrices agree to rounding
    and this is the WHOLE difference: 1.01e-08 against the measured 1.11e-08.
    """
    w = fits['water']
    D_test, F = w['D_test'], w['F']
    G, d = balanced_gram(D_test)
    FD = (F @ D_test.T) * d[None, :]
    rng = np.random.default_rng(0)
    noise = rng.normal(size=G.shape)
    noise = 0.5 * (noise + noise.T)
    perturbed = solve_balanced(G + 1e-16 * np.abs(G).max() * noise, FD, d)
    amplified = relative(solve_balanced(G, FD, d), perturbed)
    measured = relative(w['M_chain'], w['M_prod'])
    assert 0.2 < measured / amplified < 5.0


def test_production_builds_its_gram_over_the_unscreened_pair_set(fits):
    """The second term, and the one that is not rounding.

    `fit_M_streaming` forms S = (A A^T) .* (B B^T) over EVERY pair and F D^T
    over the screened ones; `fit_M_stable` screens both. At the default
    tolerance the two Gram matrices still agree to 1e-13 or better, but the
    solutions are of different problems, and raising the tolerance separates
    them completely: production's residual on the FULL test set is then more
    than an order worse than the chain's, which is what says the mixture is
    not the least-squares solution of either problem.
    """
    w = fits['water']
    mol, auxmol, crd = w['mol'], w['auxmol'], w['coords']
    full = separable_ri.test_set_layout(mol, crd, pair_tol=0.0)
    D_full = separable_ri.test_set_D(mol, auxmol, crd, full)
    V = auxmol.intor('int2c2e', aosym='s1')
    e3 = factor_chain.test_set_three_center(mol, auxmol, full[0], full[1])
    F_full = np.hstack([np.linalg.solve(V, e3.T) * full[2][None, :],
                        np.eye(w['naux'])])
    assert D_full.shape == w['D_test'].shape      # the default drops nothing here

    bites = two_fits('water', pair_tol=SCREENING_BITES)
    assert bites['D_test'].shape[1] < D_full.shape[1]
    assert relative(bites['M_chain'], bites['M_prod']) > 0.1
    chain = np.abs(bites['M_chain'] @ D_full - F_full).max()
    production = np.abs(bites['M_prod'] @ D_full - F_full).max()
    assert production > 10 * chain


# --------------------------------------------- (d) what it is worth downstream
def test_the_two_electron_energy_barely_moves(fits):
    """E_J + E_K from the fitted ERI of each factorization, water.

    D differs by 3.8e-08 relative and the energy by 5e-12 Ha: the difference
    lives in the near-null space of D, where no pair density reaches it. This
    is the number that says the two fits are one functional in practice even
    though M is not one array.
    """
    mol = molecule('water')
    mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    mf.kernel()
    dm = mf.make_rdm1()
    X = mol.eval_gto('GTOval_sph', fits['water']['coords'])
    energies = []
    for D in (fits['water']['D_chain'], fits['water']['D_prod']):
        B = np.einsum('kP,km,kn->Pmn', D, X, X, optimize=True)
        J = np.einsum('Pmn,Pls,ls->mn', B, B, dm, optimize=True)
        K = np.einsum('Pms,Pln,ls->mn', B, B, dm, optimize=True)
        energies.append(float(0.5 * np.einsum('ij,ij', dm, J)
                              - 0.25 * np.einsum('ij,ij', dm, K)))
    assert abs(energies[0] - energies[1]) < ENERGY_TOL
    assert min(energies) > 1.0                    # it is the real two-electron energy
