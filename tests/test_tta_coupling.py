"""The configuration model of triplet-triplet annihilation
(`src.fragments.tta_coupling`) against full configuration interaction.

Two closed-shell H4 rectangles 3.2 A apart, 6-31G, density-fitted RHF
(cc-pVDZ-JKFIT), fragment-localized orbitals: 8 electrons in 16 orbitals,
small enough that every state of the model is an explicit FCI vector,
built by applying spin-orbital creation and annihilation operators to the
reference determinant, and every element is <X|H|Y> with the same fitted
integrals. Random unit amplitudes on every pair of each fragment, so no
HOMO-LUMO structure or symmetry makes a term vanish.

WHAT EACH GATE IS FOR:

- the spin-free 1(TT) of the module is a unit singlet (S^2 = 0) orthogonal
  to the singlet-singlet pair on the same orbitals: the pairing
  coefficients (1, 2) / (2 sqrt 3).
- <S(x)|H|1(TT)> of the closed form equals the FCI element, for a site
  singlet and for a charge-transfer configuration, on the Hartree-Fock
  determinant and on a determinant rotated off it (occupied-virtual mixing,
  so f_vo is not zero and its term is tested; without it the element is
  off by more than the tolerance by orders of magnitude).
- the bare singles block <S|H|S'> equals the FCI elements.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import fci, gto, scf
from pyscf.fci import addons, direct_spin1
from scipy.linalg import expm

from src.fragments.localization import FragmentOrbitals
from src.fragments.tta_coupling import local_factors, triplet_pair_elements

RECTANGLE = [(0, 0, 0), (0.74, 0, 0), (0, 1.6, 0), (0.74, 1.6, 0.1)]
ATOMS = ([('H', p) for p in RECTANGLE]
         + [('H', (x + 0.3, y + 0.2, z + 3.2)) for x, y, z in RECTANGLE])
AUX = 'cc-pvdz-jkfit'


class Oracle:
    """FCI vectors and the Hamiltonian in the orbitals [c_occ, c_vir]."""

    def __init__(self, mol, mf, c_occ, c_vir):
        self.no = c_occ.shape[1]
        c = np.hstack([c_occ, c_vir])
        self.norb = c.shape[1]
        self.nelec = (self.no, self.no)
        b = np.concatenate(list(local_factors(mol, c, AUX)))
        eri = np.einsum('Qij,Qkl->ijkl', b, b)
        h1 = c.T @ mf.get_hcore() @ c
        self.h2 = direct_spin1.absorb_h1e(h1, eri, self.norb, self.nelec, 0.5)
        na = fci.cistring.num_strings(self.norb, self.no)
        self.ref = np.zeros((na, na))
        self.ref[0, 0] = 1.0          # the first no orbitals, both spins

    def e(self, psi, p, q, ne):
        """E_pq psi = sum_sigma a+_p sigma a_q sigma psi."""
        out = 0.0
        for cre, des, ne1 in ((addons.cre_a, addons.des_a, (ne[0] - 1, ne[1])),
                              (addons.cre_b, addons.des_b, (ne[0], ne[1] - 1))):
            out = out + cre(des(psi, self.norb, ne, q), self.norb, ne1, p)
        return out

    def single(self, x):
        no, out = self.no, 0.0
        for i, a in zip(*np.nonzero(x)):
            out = out + x[i, a] * self.e(self.ref, no + a, i, self.nelec)
        return out / np.sqrt(2.0)

    def pair(self, ta, tb, exchange):
        """sum ta_ia tb_jb E_ai E_bj |0> (exchange: E_aj E_bi)."""
        no, out = self.no, 0.0
        for i, a in zip(*np.nonzero(ta)):
            for j, b in zip(*np.nonzero(tb)):
                h1, h2 = (j, i) if exchange else (i, j)
                out = out + ta[i, a] * tb[j, b] * self.e(
                    self.e(self.ref, no + b, h2, self.nelec), no + a, h1,
                    self.nelec)
        return out

    def h(self, psi):
        return direct_spin1.contract_2e(self.h2, psi, self.norb, self.nelec)


@pytest.fixture(scope='module')
def system():
    mol = gto.M(atom=ATOMS, basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit(AUX)
    mf.conv_tol = 1e-12
    mf.run()
    orb = FragmentOrbitals.from_mf(mf, [range(4), range(4, 8)])
    return mol, mf, orb


def amplitudes(orb, seed):
    """Unit triplets on fragments 0 and 1, a unit site singlet on 0 and a
    unit charge-transfer configuration 0 -> 1, random."""
    rng = np.random.default_rng(seed)
    ol, vl = orb.occ_labels, orb.vir_labels

    def block(k, l):
        x = np.zeros((orb.nocc, orb.nvir))
        x[np.ix_(ol == k, vl == l)] = rng.normal(size=((ol == k).sum(),
                                                       (vl == l).sum()))
        return x / np.linalg.norm(x)
    return block(0, 0), block(1, 1), np.array([block(0, 0), block(0, 1)])


def rotated(orb, angle, seed):
    """[c_occ, c_vir] rotated off the Hartree-Fock determinant."""
    no, nv = orb.nocc, orb.nvir
    kappa = np.zeros((no + nv, no + nv))
    kappa[:no, no:] = angle * np.random.default_rng(seed).normal(size=(no, nv))
    u = expm(kappa - kappa.T)
    c = np.hstack([orb.c_occ, orb.c_vir]) @ u
    return c[:, :no], c[:, no:]


def test_the_triplet_pair_is_a_unit_singlet(system):
    mol, mf, orb = system
    oracle = Oracle(mol, mf, orb.c_occ, orb.c_vir)
    ta, tb, _ = amplitudes(orb, 3)
    p, r = oracle.pair(ta, tb, False), oracle.pair(ta, tb, True)
    tt = (p + 2.0 * r) / (2.0 * np.sqrt(3.0))
    assert abs(np.sum(tt * tt) - 1.0) < 1e-12
    s2 = fci.spin_op.spin_square0(tt, oracle.norb, oracle.nelec)[0]
    assert abs(s2) < 1e-12
    assert abs(np.sum(tt * p)) < 1e-12        # orthogonal to S S
    # the other pairing coefficient gives no singlet triplet pair
    wrong = (p + r) / np.linalg.norm(p + r)
    assert abs(np.sum(wrong * tt)) < 0.99


@pytest.mark.parametrize('angle', [0.0, 0.05])
def test_triplet_pair_elements_are_the_fci_elements(system, angle):
    mol, mf, orb = system
    c_occ, c_vir = ((orb.c_occ, orb.c_vir) if angle == 0.0
                    else rotated(orb, angle, 9))
    oracle = Oracle(mol, mf, c_occ, c_vir)
    ta, tb, singles = amplitudes(orb, 5)
    tt = (oracle.pair(ta, tb, False) + 2.0 * oracle.pair(ta, tb, True)) \
        / (2.0 * np.sqrt(3.0))
    htt = oracle.h(tt)
    states = [oracle.single(x) for x in singles]
    ref_tt = np.array([np.sum(s * htt) for s in states])
    ref_ss = np.array([[np.sum(s * oracle.h(t)) for t in states]
                       for s in states])
    el = triplet_pair_elements(mol, mf.get_hcore(), c_occ, c_vir, AUX, ta, tb,
                               singles)
    assert np.abs(el['tt'] - ref_tt).max() < 1e-12
    off = ~np.eye(len(states), dtype=bool)
    assert np.abs(el['ss'][off] - ref_ss[off]).max() < 1e-12
    fov = el['fock'][:orb.nocc, orb.nocc:]
    if angle == 0.0:
        assert np.abs(fov).max() < 1e-6                     # Brillouin
    else:
        # the f_vo term is what a non-Hartree-Fock reference adds
        term = np.sqrt(1.5) * np.einsum(
            'mia,ia->m', singles, tb @ fov.T @ ta + ta @ fov.T @ tb)
        assert np.abs(term).max() > 1e-6
    assert np.abs(ref_tt).min() > 1e-6


def test_the_triplets_must_sit_on_two_fragments(system):
    mol, mf, orb = system
    ta, tb, singles = amplitudes(orb, 5)
    with pytest.raises(ValueError, match='different fragments'):
        triplet_pair_elements(mol, mf.get_hcore(), orb.c_occ, orb.c_vir, AUX,
                              ta, ta, singles)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
