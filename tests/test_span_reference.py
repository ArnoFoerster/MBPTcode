"""The mean field of an augmented basis solved in the canonical span of a
DECLARED cut (here 1e-4, `numerics.span.cut`; no cut is ever chosen by
refusal), and nothing changed for a basis with no overlap eigenvalue below it.

The virtual orbitals that carry near-null AO combinations are beyond the
separable fit; in the span above 1e-4 the ISDF BSE roots of CBP gas
aug-cc-pVDZ agree with DF in the same span to 1 meV. Gated here:

  * `canonical_span_size` is None for cc-pVDZ water and benzene and for
    water aug-cc-pVDZ (nothing dropped, so `span_confined` hands the mean
    field back untouched) and a count for benzene aug-cc-pVDZ (179 of 192 at
    1e-4);
  * `span_confined` solves in the span of that many overlap eigenvectors,
    records the cut and the boundary, and survives the continuum's PCM
    re-convergence;
  * a chain freezes its reference's span (`frozen_span`): every other
    geometry's SCF is solved in it and the force is the finite difference of
    that energy; a mean field of another span dimension is refused;
  * the factors of benzene aug-cc-pVDZ in the span pass the orbital refusal,
    and those of the diffuse water (four extra diffuse s and p shells) are
    still refused there: the refusal fires on what the cut leaves, and names
    the density-fitted route in the full basis as the way on.
"""
import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.numerics import Numerics
from src.Base.isdf_jk import (canonical_span_size, frozen_span, solve_in_span,
                              span_confined)
from src.Base.separable_ri import UnrepresentedOrbitals, shipped_radii_lookup
from src.Base.solvent_screening import SolventScreening
from src.SingleReference.GW.space_time import separable_factors
from src.gradients.excited_state import ExcitedStateChain
from tests.isdf_test_grid import g1_numerics

H2O = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
BENZENE = ('C 0 1.3915 0; C 1.2051 0.6958 0; C 1.2051 -0.6958 0; '
           'C 0 -1.3915 0; C -1.2051 -0.6958 0; C -1.2051 0.6958 0; '
           'H 0 2.4715 0; H 2.1404 1.2358 0; H 2.1404 -1.2358 0; '
           'H 0 -2.4715 0; H -2.1404 -1.2358 0; H -2.1404 1.2358 0')
#: The extra diffuse shells (first exponent, ratio, count).
EXTRA = (0.03, 1.15, 4)
COUNTS = {'A1': 24, 'A2': 15, 'A3': 9, 'B1': 3}
#: The declared cut whose spans these tests build by hand.
CUT = 1e-4


def _diffuse(element):
    base = gto.basis.load('aug-cc-pvdz', element)
    a0, ratio, n = EXTRA
    for k in range(n):
        base = base + [[0, [a0 * ratio**k, 1.0]], [1, [a0 * ratio**k, 1.0]]]
    return base


def _rhf(mol, n_kept, auxbasis):
    mf = span_confined(scf.RHF(mol).density_fit(auxbasis=auxbasis), n_kept,
                       CUT)
    mf.conv_tol, mf.conv_tol_grad = 1e-10, 1e-8
    return mf


def _radii(elements):
    return {el: shipped_radii_lookup(el, 'aug-cc-pvdz', 'aug-cc-pvdz-ri',
                                     COUNTS)[0] for el in elements}


@pytest.mark.parametrize('atom, basis', [(H2O, 'cc-pvdz'),
                                         (BENZENE, 'cc-pvdz'),
                                         (H2O, 'aug-cc-pvdz')])
def test_nothing_is_cut_where_nothing_lies_below(atom, basis):
    """Where no overlap eigenvalue is below the cut the full basis is the
    span and the mean field is handed back as it came."""
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    assert canonical_span_size(mol, CUT) is None
    mf = scf.RHF(mol)
    assert span_confined(mf, None) is mf
    assert not hasattr(mf, 'span') and '_eigh' not in mf.__dict__


def test_the_span_of_an_augmented_basis():
    """Benzene aug-cc-pVDZ: 13 of 192 directions below 1e-4, the SCF in the
    remaining 179, its record naming the boundary."""
    mol = gto.M(atom=BENZENE, basis='aug-cc-pvdz', verbose=0)
    n = canonical_span_size(mol, CUT)
    assert n == 179 and mol.nao_nr() == 192
    mf = _rhf(mol, n, 'aug-cc-pvdz-jkfit')
    mf.kernel()
    assert mf.converged and mf.mo_coeff.shape[1] == n
    span = mf.span
    assert span['cut'] == CUT
    assert (span['n_kept'], span['n_dropped'], span['nao']) == (179, 13, 192)
    assert (span['largest_dropped_eigenvalue'] < CUT
            < span['smallest_kept_eigenvalue'])
    # the factors represent every orbital of the span (worst 0.026 measured)
    separable_factors(mf, mol, auxbasis='aug-cc-pvdz-ri',
                      radii=_radii(('C', 'H')))


def test_the_refusal_fires_on_what_the_cut_leaves():
    """The diffuse water keeps 62 of 89 directions at 1e-4, and the factors
    still miss three of its orbital densities there (worst 0.17 on the
    small-box aug-cc-pVDZ rows; the larger-box rows miss five, worst 0.31)."""
    mol = gto.M(atom=H2O, basis={'O': _diffuse('O'), 'H': _diffuse('H')},
                verbose=0)
    n = canonical_span_size(mol, CUT)
    assert n == 62
    mf = _rhf(mol, n, 'aug-cc-pvdz-jkfit')
    mf.kernel()
    assert mf.converged
    with pytest.raises(UnrepresentedOrbitals,
                       match='density-fitted route in the full basis') as refused:
        separable_factors(mf, mol, auxbasis='aug-cc-pvdz-ri',
                          radii=_radii(('O', 'H')))
    assert len(refused.value.orbitals) == 3


def test_the_continuum_keeps_the_span():
    """The continuum's PCM re-convergence of a span-confined mean field stays
    in its span."""
    mol = gto.M(atom=H2O, basis='aug-cc-pvdz', verbose=0)
    n = mol.nao_nr() - 3

    def factory(m):
        mf = dft.RKS(m, xc='pbe0').density_fit(auxbasis='aug-cc-pvdz-jkfit')
        mf = span_confined(mf, n, CUT)
        mf.conv_tol, mf.conv_tol_grad = 1e-10, 1e-8
        mf.kernel()
        return mf

    mf = factory(mol)
    assert mf.mo_coeff.shape[1] == n and mf.span['n_dropped'] == 3
    assert mf.span['cut'] == CUT
    solvated = SolventScreening(mol, eps=2.4, eps_static=2.4).mean_field(
        mol, factory)
    assert solvated.converged and solvated.mo_coeff.shape[1] == n
    assert solvated.span['n_kept'] == n


def _displaced(mol, ia, x, h):
    c = mol.atom_coords().copy()
    c[ia, x] += h
    return mol.set_geom_(c, unit='Bohr', inplace=False)


def test_the_chain_freezes_its_span():
    """Water aug-cc-pVDZ with three directions cut: the chain's displaced SCF
    stands on its reference's span, and the Hartree-Fock force there is the
    central difference of the frozen-span energy."""
    mol = gto.M(atom=H2O, basis='aug-cc-pvdz', verbose=0)
    n = mol.nao_nr() - 3

    def factory(m):
        mf = span_confined(scf.RHF(m).density_fit(auxbasis='aug-cc-pvdz-jkfit'),
                           n, CUT)
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
        mf.kernel()
        return mf

    chain = ExcitedStateChain(mol, factory, mf=factory(mol),
                              numerics=g1_numerics(mol, Numerics(bse={'solver': 'dense'})))
    assert chain.span_basis is chain.mf0.span_basis
    assert not chain.mf0.span['frozen']
    _, mf = chain.mean_field(_displaced(mol, 0, 2, 0.02))
    assert mf.span['frozen'] and mf.span_basis is chain.span_basis
    g = chain.mf0.nuc_grad_method().kernel()
    h = 1e-3
    with frozen_span(chain.span_basis):
        e = [factory(_displaced(mol, 1, 1, s * h)).e_tot for s in (-1, 1)]
    assert abs(g[1, 1] - (e[1] - e[0]) / (2 * h)) < 1e-6


def test_a_chain_refuses_another_span_dimension():
    """A walk whose SCF lands on another span is on another surface."""
    mol = gto.M(atom=H2O, basis='cc-pvdz', verbose=0)
    nao = mol.nao_nr()

    def factory(m):
        n = nao - 1 if m is not mol else nao - 2
        w, U = np.linalg.eigh(m.intor('int1e_ovlp'))
        mf = solve_in_span(scf.RHF(m).density_fit(auxbasis='cc-pvdz-jkfit'),
                           U[:, nao - n:] / np.sqrt(w[nao - n:]))
        mf.conv_tol = 1e-10
        mf.kernel()
        return mf

    chain = ExcitedStateChain(mol, factory, mf=factory(mol), numerics=g1_numerics(mol))
    moved = mol.copy()
    moved.set_geom_(mol.atom_coords() + 1e-3, unit='Bohr')
    moved.build(False, False)
    with pytest.raises(RuntimeError, match='another span'):
        chain.mean_field(moved)
