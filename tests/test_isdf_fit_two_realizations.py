"""One ISDF fit M, and the realizations that solve it.

The separable (ISDF) fit is one least-squares problem (Duchemin and Blase,
JCP 150, 174120 (2019) eqs 8-9, balanced and Tikhonov-regularized) with the
estimator of `separable_ri.fit_M_streaming`: the Gram matrix over every
product pair of the test set, (A A^T) o (B B^T) + P P^T, and the right-hand
side F D^T over the pairs the screen keeps, so a screened pair keeps its row
of the Gram matrix and is fitted to zero. Three realizations:

  replicated   `FrozenFactorization._fit` is `fit_M_streaming` on the frozen
               pair layout, the same bits on the same points and pairs;
  rows         `fit_rows`, the grid index in fixed tiles over the ranks;
  whole form   `fit_M_stable` on D over every product pair and F zero on the
               screened ones, what the whole-form fit adjoint rebuilds.

The last two are gated within `FIT_REASSOCIATION_K` times what one
reassociation of the replicated fit moves D (its three-centre blocks cut per
shell and accumulated in reverse). A screened-Gram estimator (Gram and
right-hand side over the screened pairs alone) is a different least-squares
problem wherever the screen drops a pair; it is kept as proof that the gate
can fail. Water keeps every pair and cannot tell the two apart.

The fit is determined only up to the conditioning of its balanced Gram
matrix: most of the spectrum lies under the 4e-7 Tikhonov shift, so two
realizations differ in D's near-null space, where no pair density reaches;
their two-electron energies agree to 1e-10 Ha.
"""
import numpy as np
import pytest
from pyscf import df as pyscf_df, gto, scf

from src.Base import separable_ri
from src.Base.constants import FIT_REASSOCIATION_K
from src.Base.separable_ri import (DEFAULT_REGULARIZATION, atomic_grid,
                                   aux_metric_sqrt, fit_M_stable,
                                   fit_M_streaming, molecular_points_covariant)
from src.SingleReference.GW.space_time import DEFAULT_COUNTS
from src.gradients.factor_chain import FrozenFactorization
from src.gradients.isdf_derivatives import pair_positions, product_pairs

# `test_set_layout` and `test_set_D` stay module-qualified: imported by name,
# pytest collects them as test functions.

BASIS, AUX = 'cc-pvdz', 'cc-pvdz-ri'
MOLECULES = {
    'water': 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
    'ethylene': ('C 0.0 0.0 0.667; C 0.0 0.0 -0.667; H 0.0 0.923 1.238; '
                 'H 0.0 -0.923 1.238; H 0.0 0.923 -1.238; '
                 'H 0.0 -0.923 -1.238'),
    'formaldehyde': ('C 0 0 -0.5395; O 0 0 0.6636; '
                     'H 0 0.9445 -1.1090; H 0 -0.9445 -1.1090'),
}
#: Pairs the default screen drops; water drops none, so it cannot tell a
#: screened Gram from the one fit.
SCREENED = {'water': 0, 'ethylene': 72, 'formaldehyde': 6}
#: One hydrogen 0.03 A along y, a geometry the frozen layout is read at.
DISPLACED = {'ethylene': ('C 0.0 0.0 0.667; C 0.0 0.0 -0.667; '
                          'H 0.0 0.953 1.238; H 0.0 -0.923 1.238; '
                          'H 0.0 0.923 -1.238; H 0.0 -0.923 -1.238')}
#: |E_J + E_K| difference of two realizations' factorizations, Ha.
ENERGY_TOL = 1e-10
#: How many bars the screened-Gram estimator must sit away for the gate to
#: have failed it.
OTHER_PROBLEM = 1e3


def molecule(atom):
    return gto.M(atom=atom, basis=BASIS, verbose=0)


def shipped_grid(mol):
    """The interpolation points `separable_factors` places at `DEFAULT_COUNTS`."""
    radii, origins = {}, {}
    for el in sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)}):
        radii[el], origins[el] = atomic_grid(el, mol.basis, AUX, DEFAULT_COUNTS)
    return molecular_points_covariant(mol, radii, origin_by_element=origins)


def per_shell_reversed(mol, nk, n2, naux, block_memory_gb):
    """Every shell its own block of the three-centre pass, last first."""
    return [(s, s + 1) for s in reversed(range(mol.nbas))]


def whole_form(mol, auxmol, crd, layout, fit=fit_M_stable):
    """The estimator formed whole: D over every product pair, F zero on the
    screened ones (`dfactor_adjoint_gauges`' rebuild), solved by `fit`."""
    gram = product_pairs(mol)
    V = auxmol.intor('int2c2e', aosym='s1')
    naux = auxmol.nao_nr()
    mu, nu, wc = layout
    e3c = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1')
    e3c = e3c.reshape(mol.nao_nr(), mol.nao_nr(), naux)
    F = np.zeros((naux, len(gram[0]) + naux))
    F[:, pair_positions(layout, gram, mol.nao_nr())] = (
        np.linalg.solve(V, e3c[mu, nu, :].T) * wc[None, :])
    F[:, len(gram[0]):] = np.eye(naux)
    return fit(separable_ri.test_set_D(mol, auxmol, crd, gram), F)


def screened_gram(mol, auxmol, crd, layout):
    """The screened-Gram estimator: Gram and right-hand side over the
    screened pairs alone."""
    V = auxmol.intor('int2c2e', aosym='s1')
    naux = auxmol.nao_nr()
    mu, nu, wc = layout
    e3c = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1')
    e3c = e3c.reshape(mol.nao_nr(), mol.nao_nr(), naux)
    F = np.hstack([np.linalg.solve(V, e3c[mu, nu, :].T) * wc[None, :],
                   np.eye(naux)])
    return fit_M_stable(separable_ri.test_set_D(mol, auxmol, crd, layout), F)


def realizations(name):
    """Every realization of M on the chain's frozen grid, and the bar."""
    mol = molecule(MOLECULES[name])
    fac = FrozenFactorization(mol, auxbasis=AUX)
    auxmol, crd = fac.auxmol(mol), fac.coords(mol)
    rows = fit_M_streaming(mol, auxmol, crd, fit='rows', layout=fac.layout)
    out = dict(mol=mol, auxmol=auxmol, coords=crd, layout=fac.layout,
               chain=fac.shareable_factors(mol, auxmol, crd)[1],
               streaming=fit_M_streaming(mol, auxmol, crd),
               rows=np.vstack([rows.mt[t] for t in sorted(rows.mt)]).T,
               whole=whole_form(mol, auxmol, crd, fac.layout),
               screened=screened_gram(mol, auxmol, crd, fac.layout))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(separable_ri, 'ao_blocks', per_shell_reversed)
        out['bar'] = fit_M_streaming(mol, auxmol, crd, layout=fac.layout)
    V_half = aux_metric_sqrt(auxmol, None)
    out['D'] = {k: out[k].T @ V_half for k in ('chain', 'rows', 'whole',
                                                'screened', 'bar')}
    return out


@pytest.fixture(scope='module')
def fits():
    return {name: realizations(name) for name in MOLECULES}


def relative(a, b):
    """||a - b|| / ||b||."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ------------------------------------------------------- (a) the same grid
@pytest.mark.parametrize('name', ['water', 'formaldehyde'])
def test_the_chain_and_production_place_the_same_points(name):
    """The chain's re-optimized radii are the shipped row at these counts,
    so the chain and production fit on one grid."""
    mol = molecule(MOLECULES[name])
    chain = FrozenFactorization(mol, auxbasis=AUX).coords(mol)
    assert np.array_equal(chain, shipped_grid(mol))


# ------------------------------------------- (b) the chain runs production
@pytest.mark.parametrize('name', list(MOLECULES))
def test_the_chains_fit_is_production_bit_for_bit(fits, name):
    """At the reference the frozen layout is the geometry's own screen, and
    the chain's M is `fit_M_streaming`'s on the same points, bitwise."""
    w = fits[name]
    npairs = len(product_pairs(w['mol'])[0])
    assert npairs - len(w['layout'][0]) == SCREENED[name]
    assert np.array_equal(w['chain'], w['streaming'])


def test_the_frozen_layout_is_what_the_fit_keeps():
    """At a displaced geometry the replicated fit on the geometry's own
    layout is the unfrozen call's bits, and on a layout short of one pair it
    is another fit: the layout is read, not re-screened."""
    mol = molecule(DISPLACED['ethylene'])
    fac = FrozenFactorization(molecule(MOLECULES['ethylene']), auxbasis=AUX)
    auxmol, crd = fac.auxmol(mol), fac.coords(mol)
    own = separable_ri.test_set_layout(mol, crd)
    free = fit_M_streaming(mol, auxmol, crd)
    assert np.array_equal(fit_M_streaming(mol, auxmol, crd, layout=own), free)
    short = tuple(np.delete(a, len(a) // 2) for a in own)
    assert not np.array_equal(
        fit_M_streaming(mol, auxmol, crd, layout=short), free)


# --------------------------------- (c) the other realizations, anchored
@pytest.mark.parametrize('name', list(MOLECULES))
def test_every_realization_is_the_one_fit(fits, name):
    """The row fit and the whole form within `FIT_REASSOCIATION_K` times
    what reversing the replicated fit's three-centre sum moves D."""
    w = fits[name]
    D = w['D']
    bar = relative(D['bar'], D['chain'])
    assert bar > 0.0, 'the reassociation moved nothing: no bar'
    for key in ('rows', 'whole'):
        ratio = relative(D[key], D['chain']) / bar
        print(f'{name}: {key} {ratio:.2f} x the bar {bar:.2e}')
        assert ratio <= FIT_REASSOCIATION_K, (key, ratio)


@pytest.mark.parametrize('name', ['ethylene', 'formaldehyde'])
def test_a_screened_gram_is_another_problem(fits, name):
    """The screened-Gram estimator sits more than `OTHER_PROBLEM` bars from
    the one fit wherever the screen drops a pair."""
    w = fits[name]
    D = w['D']
    bar = relative(D['bar'], D['chain'])
    assert relative(D['screened'], D['chain']) > OTHER_PROBLEM * bar


def test_water_cannot_tell_the_two_problems_apart(fits):
    """Every pair kept: the screened-Gram estimator is the whole form,
    bitwise."""
    w = fits['water']
    assert np.array_equal(w['screened'], w['whole'])


# ------------------------------------------------ (d) the conditioning
@pytest.mark.parametrize('name', list(MOLECULES))
def test_the_fit_is_determined_only_up_to_the_conditioning(fits, name):
    """The balanced Gram matrix over every product pair is numerically
    singular: most of its spectrum lies under the Tikhonov shift, which sets
    cond(G + shift) between 1e8 and 1e10."""
    w = fits[name]
    D = separable_ri.test_set_D(w['mol'], w['auxmol'], w['coords'],
                                product_pairs(w['mol']))
    s = np.sqrt(np.einsum('kr,kr->k', D, D))
    Dt = D / np.where(s == 0.0, 1.0, s)[:, None]
    ev = np.linalg.eigvalsh(Dt @ Dt.T)
    assert ev.min() < 1e-12
    assert (ev < DEFAULT_REGULARIZATION).sum() > 0.25 * len(ev)
    shifted = ev + DEFAULT_REGULARIZATION
    assert 1e8 < shifted.max() / shifted.min() < 1e10


# ----------------------------------------- (e) what it is worth downstream
def test_two_realizations_carry_one_two_electron_energy(fits):
    """E_J + E_K of water from the replicated and the row fit's factors
    agree to `ENERGY_TOL`: the realizations differ where no pair density
    reaches."""
    w = fits['water']
    mol = w['mol']
    mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    mf.kernel()
    dm = mf.make_rdm1()
    X = mol.eval_gto('GTOval_sph', w['coords'])
    energies = []
    for D in (w['D']['chain'], w['D']['rows']):
        B = np.einsum('kP,km,kn->Pmn', D, X, X, optimize=True)
        J = np.einsum('Pmn,Pls,ls->mn', B, B, dm, optimize=True)
        K = np.einsum('Pms,Pln,ls->mn', B, B, dm, optimize=True)
        energies.append(float(0.5 * np.einsum('ij,ij', dm, J)
                              - 0.25 * np.einsum('ij,ij', dm, K)))
    assert abs(energies[0] - energies[1]) < ENERGY_TOL
    assert min(energies) > 1.0
