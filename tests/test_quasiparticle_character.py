"""Where a quasiparticle's orbital lives, and whether a scan follows the same one.

  1. Pipek-Mezey character on Li+(EC): the LUMO is lithium's (2s-like), the
     first carbonate pi* is EC's, the HOMO is EC's, and the Loewdin population
     of each canonical orbital -- the cross-check -- names the same fragment.
     The weights over the localized window sum to one.
  2. A geometry track follows the orbital's localized character, not its
     index: with two canonical columns swapped at a displaced geometry (a
     planted level crossing), `track_orbital` returns the moved column.
  3. `quasiparticle_order` flags a requested orbital that is not the
     lowest-energy attachment or removal, and stays silent for one that is;
     `surface_order` reads the surface's own mean field and declared orbital.
  4. An unrestricted reference is labelled per spin channel; a fragment list
     that does not partition the atoms is refused.
  5. A charged surface with `track=True` follows its orbital through a planted
     crossing: handed the displaced geometry's mean field with the LUMO and
     LUMO+1 columns swapped, it returns the energy the untracked surface gives
     on the unswapped one, where the untracked surface steps onto the other
     state.

Run as a script (`python tests/test_quasiparticle_character.py`) or under
pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.declaration import ChargedExcitation
from src.gradients.rpa_bse_surface import MeanFieldQPSurface
from src.properties.characters import (orbital_fingerprint,
                                       quasiparticle_character,
                                       quasiparticle_order, surface_order,
                                       track_orbital)

#: Li+ bound to the carbonyl oxygen of a planar ethylene carbonate
EC_LI = '''C 0 0 0; O 0 0 1.195; O 1.09 0 -0.763; O -1.09 0 -0.763;
C 0.77 0 -2.12; C -0.77 0 -2.12; H 1.25 0.89 -2.55; H 1.25 -0.89 -2.55;
H -1.25 0.89 -2.55; H -1.25 -0.89 -2.55; Li 0 0 {z}'''
FRAGMENTS = {'Li': [10], 'EC': list(range(10))}
WATER = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'


def _li_ec(z=3.05):
    mol = gto.M(atom=EC_LI.format(z=z), basis='6-31g*', charge=1, verbose=0)
    return scf.RHF(mol).run(conv_tol=1e-10)


def _dominant(weights):
    return max(weights, key=weights.get)


def test_li_ec_character():
    mf = _li_ec()
    nocc = mf.mol.nelectron // 2
    lumo = quasiparticle_character(mf, nocc, FRAGMENTS)
    assert lumo['dominant'] == 'Li' and lumo['weights']['Li'] > 0.95, lumo
    assert _dominant(lumo['lowdin']) == 'Li'
    assert abs(sum(lumo['weights'].values()) - 1.0) < 1e-10
    # the first virtual the localized orbitals give to the carbonate: its pi*
    ec = next(p for p in range(nocc, nocc + 6)
              if quasiparticle_character(mf, p, FRAGMENTS)['dominant'] == 'EC')
    pi_star = quasiparticle_character(mf, ec, FRAGMENTS)
    assert pi_star['weights']['EC'] > 0.95, pi_star
    assert _dominant(pi_star['lowdin']) == 'EC', pi_star['lowdin']
    homo = quasiparticle_character(mf, nocc - 1, FRAGMENTS)
    assert homo['dominant'] == 'EC' and _dominant(homo['lowdin']) == 'EC'
    assert homo['window'][1] == nocc, 'an occupied orbital is localized among occupied'


def test_track_follows_a_planted_crossing():
    ref = _li_ec()
    nocc = ref.mol.nelectron // 2
    ec = next(p for p in range(nocc, nocc + 6)
              if quasiparticle_character(ref, p, FRAGMENTS)['dominant'] == 'EC')
    fingerprint = orbital_fingerprint(ref, ec)
    moved = _li_ec(z=3.10)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)       # unambiguous
        same, _ = track_orbital(fingerprint, moved)
    assert same == ec, same
    other = ec - 1                                           # a Li-like level
    swapped = moved.mo_coeff.copy()
    swapped[:, [ec, other]] = swapped[:, [other, ec]]
    followed, score = track_orbital(fingerprint, moved, mo_coeff=swapped)
    assert followed == other, (followed, score)
    assert score[other] > 0.9 and score[ec] < 0.5, score


def test_order_audit():
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri').run(conv_tol=1e-10)
    nocc = mol.nelectron // 2
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        lumo = quasiparticle_order(mf, nocc)
        homo = quasiparticle_order(mf, nocc - 1)
    assert not lumo['reordered'] and not homo['reordered']
    with pytest.warns(RuntimeWarning, match='not the lowest-energy attachment'):
        second = quasiparticle_order(mf, nocc + 1)
    assert second['reordered'] and second['lowest_process'] == nocc
    with pytest.warns(RuntimeWarning, match='not the lowest-energy removal'):
        quasiparticle_order(mf, nocc - 2)

    class Surface:
        """The two things `surface_order` reads of a charged surface."""
        physics_excitation = ChargedExcitation(nocc + 1, +1)

        def mean_field(self, mol=None):
            return mol, mf

    with pytest.warns(RuntimeWarning, match='not the lowest-energy attachment'):
        assert surface_order(Surface())['lowest_process'] == nocc


def test_unrestricted_and_refusals():
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='6-31g', spin=1, verbose=0)
    uhf = scf.UHF(mol).run(conv_tol=1e-10)
    fragments = {'O': [0], 'H': [1]}
    for spin, n in enumerate(uhf.nelec):
        out = quasiparticle_character(uhf, n - 1, fragments, spin=spin)
        assert abs(sum(out['weights'].values()) - 1.0) < 1e-10
        assert out['dominant'] == 'O'
    with pytest.raises(ValueError, match='spin=0'):
        quasiparticle_character(uhf, 3, fragments)
    with pytest.raises(ValueError, match='partition'):
        quasiparticle_character(uhf, 3, {'O': [0]}, spin=0)


def test_a_tracked_surface_follows_its_orbital():
    """Measured: tracked-on-swapped minus untracked-on-unswapped 0.0 Ha; the
    untracked surface on the swapped columns lands 0.072 Ha away."""
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)

    def factory(m):
        mf = scf.RHF(m)
        mf.conv_tol = 1e-12
        mf.kernel()
        return mf

    tracked = MeanFieldQPSurface(mol, factory, state=1, track=True,
                                 solver='dense')
    plain = MeanFieldQPSurface(mol, factory, state=1, solver='dense')
    assert tracked.total_energy(mol) == plain.total_energy(mol)
    moved = mol.copy()
    moved.set_geom_(mol.atom_coords() + np.array([[0, 0, 0.02], [0, 0.01, 0],
                                                  [0, 0, 0]]), unit='Bohr')
    moved.build(False, False)
    mf = factory(moved)
    nocc = mol.nelectron // 2
    swapped = mf.copy()
    order = np.arange(mf.mo_energy.size)
    order[[nocc, nocc + 1]] = order[[nocc + 1, nocc]]
    swapped.mo_coeff, swapped.mo_energy = mf.mo_coeff[:, order], mf.mo_energy[order]
    right = plain.total_energy(moved, mf)
    assert abs(tracked.total_energy(moved, swapped) - right) < 1e-10
    assert tracked.follow_log[-1]['orbital'] == nocc + 1
    assert abs(plain.total_energy(moved, swapped) - right) > 1e-2


TESTS = [test_li_ec_character, test_track_follows_a_planted_crossing,
         test_order_audit, test_unrestricted_and_refusals,
         test_a_tracked_surface_follows_its_orbital]


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f'[OK  ] {test.__name__}')
        except (Exception, pytest.fail.Exception) as exc:   # noqa: BLE001
            failed += 1
            print(f'[FAIL] {test.__name__} -- {type(exc).__name__}: {exc}')
    print('\nALL PASSED' if not failed else '\nFAILURES DETECTED')
    sys.exit(0 if not failed else 1)
