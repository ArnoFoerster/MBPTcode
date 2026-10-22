"""The aug-cc-pVDZ rows of the radii table put the outer shells in a small box.

Rows cut with the outer shells in a larger box fit near-null AO directions
10-100 times worse: on CBP gas aug-cc-pVDZ they break (A - B) at every count.
The aug-cc-pVDZ rows say in their provenance that they are the small-box
table's. Gated here:
  * the (24, 15, 9, 3) row of H, C, N, O and S has the outermost radius of
    the small-box table (the larger box puts C at 10.59, N at 12.88, O at
    7.08, H at 7.75 bohr), and every such row says where it comes from;
  * water with four extra diffuse s and p shells, Hartree-Fock in the canonical
    span above 1e-4 (62 orbitals): the worst virtual density residual of the
    factors is 0.166 on the small-box row and 0.312 on the larger-box one; the
    bound sits between them;
  * rows of other bases carry no such stamp (the cc-pVDZ rows are not
    touched).
"""
import json
import pathlib

import numpy as np
import pytest
from pyscf import df as pyscf_df, gto, scf

from src.Base.isdf_jk import canonical_basis, solve_in_span
from src.Base.separable_ri import (orbital_density_residual,
                                   shipped_radii, shipped_radii_lookup)
from src.SingleReference.GW.space_time import unrefused_separable_factors

TABLE = pathlib.Path(__file__).resolve().parent.parent / 'src' / 'Base' / \
    'data' / 'optimized_radii.json'
G3 = {'A1': 24, 'A2': 15, 'A3': 9, 'B1': 3}
#: Outermost radius (bohr) of the small-box (24, 15, 9, 3) aug-cc-pVDZ row.
OUTER_SMALL_BOX = {'H': 4.999999999999999, 'C': 9.259174525425674,
                 'N': 7.480591801832869, 'O': 9.813809344748067,
                 'S': 10.187078608256567}
#: Worst virtual density residual measured 0.166 (small box) and 0.312
#: (larger).
WATER_BOUND = 0.2


@pytest.mark.parametrize('element', sorted(OUTER_SMALL_BOX))
def test_outer_shells_are_the_small_box_ones(element):
    radii = shipped_radii_lookup(element, 'aug-cc-pvdz', 'aug-cc-pvdz-ri',
                                 G3)[0]
    outer = max(float(np.max(v)) for v in radii.values())
    assert outer == pytest.approx(OUTER_SMALL_BOX[element], rel=1e-12)


def test_every_small_box_row_names_its_source():
    raw = json.loads(TABLE.read_text())
    assert any(r['basis'] == 'aug-cc-pvdz' and r['from'] == 'small-box table'
               for r in raw['restorations'])
    for key, row in raw['rows'].items():
        stamped = row.get('provenance', '').startswith('small-box table:')
        if key.split('|')[1] != 'aug-cc-pvdz':
            assert not stamped, key
    stamped = [k for k, r in raw['rows'].items()
               if r.get('provenance', '').startswith('small-box table:')]
    assert len(stamped) == 40
    assert set(shipped_radii()) >= set(stamped)


def test_water_with_diffuse_shells_is_represented_in_the_span():
    def basis(element):
        base = gto.basis.load('aug-cc-pvdz', element)
        for k in range(4):
            a = 0.03 * 1.15**k
            base = base + [[0, [a, 1.0]], [1, [a, 1.0]]]
        return base
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis={'O': basis('O'), 'H': basis('H')}, verbose=0)
    auxmol = pyscf_df.addons.make_auxmol(mol, 'aug-cc-pvdz-ri')
    radii = {el: shipped_radii_lookup(el, 'aug-cc-pvdz', 'aug-cc-pvdz-ri',
                                      G3)[0] for el in ('O', 'H')}
    mf = solve_in_span(scf.RHF(mol).density_fit(auxbasis='aug-cc-pvdz-jkfit'),
                       canonical_basis(mol.intor('int1e_ovlp'), 1e-4))
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged
    _, D, X_ao, _ = unrefused_separable_factors(
        mf, mol, auxbasis='aug-cc-pvdz-ri', radii=radii)
    rel, _ = orbital_density_residual(mol, auxmol, mf.mo_coeff, X_ao, D)
    assert rel[mol.nelectron // 2:].max() < WATER_BOUND
