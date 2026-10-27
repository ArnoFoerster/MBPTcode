"""ActiveSpace (src/SingleReference/common/active_space.py): the window-selection rules and the
index views built on top of them.

Run: python tests/test_active_space.py, or under pytest.

Checks:
  1. from_counts selects the contiguous frontier window; from_indices
     validates a window spelt out by hand
  2. from_irreps selects N2's valence CAS(6,6) by character
  3. from_population selects ethylene's pi/pi* window by Loewdin population
  4. the spin-orbital and spatial views of a window agree with each other and
     with the permuted mean field they describe
  5. frozen_core counts the core shells of the run drivers' convention
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.SingleReference.common.active_space import ActiveSpace, frozen_core

#: N2 valence CAS(6,6): 3sigma_g + the 1pi_u pair occupied, the 1pi_g* pair +
#: 3sigma_u* virtual. Stable in character all the way from 0.8 to 4.0 A, unlike
#: the energy ordering.
N2_VALENCE_66 = ({'A1g': 1, 'E1ux': 1, 'E1uy': 1},
                 {'E1gx': 1, 'E1gy': 1, 'A1u': 1})


ETHYLENE = """
C   0.0000000000   0.0000000000   0.6695000000
C   0.0000000000   0.0000000000  -0.6695000000
H   0.0000000000   0.9289000000   1.2321000000
H   0.0000000000  -0.9289000000   1.2321000000
H   0.0000000000   0.9289000000  -1.2321000000
H   0.0000000000  -0.9289000000  -1.2321000000
"""


def check_from_counts_and_from_indices():
    mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run()
    eps, nocc = mf.mo_energy, mol.nelectron // 2

    sp = ActiveSpace.from_counts(eps, nocc, 2, 2)
    as_tuple_ok = (np.array_equal(sp.as_tuple()[0], sp.occ_active)
                  and np.array_equal(sp.as_tuple()[1], sp.virt_active))
    window_ok = (np.array_equal(sp.occ_active, [nocc - 2, nocc - 1])
                and np.array_equal(sp.virt_active, [nocc, nocc + 1])
                and sp.is_contiguous and sp.n_act == 4 and sp.n_act_occ == 2
                and sp.nelec_active == 4 and as_tuple_ok)
    try:
        ActiveSpace.from_counts(eps, nocc, 999, 2)
        oversized_raises = False
    except ValueError:
        oversized_raises = True

    ActiveSpace.from_indices(len(eps), nocc, [nocc - 2, nocc - 1], [nocc])
    try:
        ActiveSpace.from_indices(len(eps), nocc, [2, 1], [nocc])
        malformed_raises = False
    except ValueError:
        malformed_raises = True

    ok = window_ok and oversized_raises and malformed_raises
    print(f"from_counts window={list(sp.occ_active)}+{list(sp.virt_active)} "
          f"contiguous={sp.is_contiguous} oversized/malformed raise "
          f"{oversized_raises}/{malformed_raises}: {'OK' if ok else 'FAIL'}")
    return ok


def check_from_irreps_selects_n2_cas66():
    mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pvdz', symmetry=True, verbose=0)
    mf = scf.RHF(mol).run()
    occ_irreps, vir_irreps = N2_VALENCE_66
    sp = ActiveSpace.from_irreps(mol, mf, occ_irreps, vir_irreps)
    nocc = mol.nelectron // 2
    window_ok = (sp.n_act_occ == 3 and sp.n_act == 6
                and np.array_equal(sp.occ_active, [nocc - 3, nocc - 2, nocc - 1])
                and np.array_equal(sp.virt_active, [nocc, nocc + 1, nocc + 2]))
    try:
        ActiveSpace.from_irreps(mol, mf, {'A1g': 10}, vir_irreps)
        insufficient_raises = False
    except ValueError:
        insufficient_raises = True
    ok = window_ok and insufficient_raises
    print(f"from_irreps N2 CAS(6,6): occ={list(sp.occ_active)} "
          f"virt={list(sp.virt_active)} insufficient-irrep raises "
          f"{insufficient_raises}: {'OK' if ok else 'FAIL'}")
    return ok


def check_from_population_selects_ethylene_pi_window():
    mol = gto.M(atom=ETHYLENE, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    nocc = mol.nelectron // 2

    sp = ActiveSpace.from_population(mf, atoms=[0, 1], ao_type='p', threshold=0.7,
                                     n_occ=1, n_virt=1)
    window_ok = np.array_equal(sp.occ_active, [nocc - 1]) and np.array_equal(
        sp.virt_active, [nocc])

    try:
        ActiveSpace.from_population(mf, atoms=[0, 1], ao_type='p', threshold=1.5,
                                    n_occ=1, n_virt=1)
        empty_raises = False
    except ValueError:
        empty_raises = True

    try:
        ActiveSpace.from_population(mf, atoms=[0, 1], ao_type='p', threshold=0.7,
                                    n_occ=1, n_virt=3)
        insufficient_raises = False
    except ValueError:
        insufficient_raises = True

    ok = window_ok and empty_raises and insufficient_raises
    print(f"from_population ethylene pi window: occ={list(sp.occ_active)} "
          f"virt={list(sp.virt_active)} empty/insufficient raise "
          f"{empty_raises}/{insufficient_raises}: {'OK' if ok else 'FAIL'}")
    return ok


def check_permuted_views_are_self_consistent():
    mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run()
    eps, nocc = mf.mo_energy, mol.nelectron // 2
    sp = ActiveSpace.from_counts(eps, nocc, 2, 2)

    core, act, ext = sp.spatial_partition()
    core_so, act_so, ext_so = sp.spin_orbital()
    doubled = (len(core_so) == 2 * len(core) and len(act_so) == 2 * len(act)
              and len(ext_so) == 2 * len(ext))
    interleaved = np.array_equal(core_so, np.arange(2 * len(core)))

    rmf = sp.permuted_mf(mf)
    n_core, n_act = sp.contiguous_counts()
    counts_ok = (rmf.n_occ_spatial == n_core and rmf.n_act_spatial == n_act
                and rmf.n_act_occ == sp.n_act_occ)

    ok = doubled and interleaved and counts_ok
    print(f"spin_orbital doubles spatial_partition={doubled} interleaved="
          f"{interleaved} permuted_mf counts match contiguous_counts="
          f"{counts_ok}: {'OK' if ok else 'FAIL'}")
    return ok


def run():
    all_ok = True
    for check in (check_from_counts_and_from_indices,
                  check_from_irreps_selects_n2_cas66,
                  check_from_population_selects_ethylene_pi_window,
                  check_permuted_views_are_self_consistent):
        all_ok &= bool(check())
    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    return all_ok


def test_active_space_selection():
    assert run()


def test_frozen_core_counts_the_core_shells():
    """1s past He, 2s2p past Ne, 3s3p past Ar, 3d past Zn: H 0, Li 1, Na 5,
    K 9, Ga 14 (atoms of the molecule summed)."""
    def core(atoms, spin=0):
        return frozen_core(gto.M(atom=atoms, basis='sto-3g', spin=spin, verbose=0))
    assert core('O 0 0 0; H 0 0 1; H 0 1 0') == 1
    assert core('Li 0 0 0; H 0 0 1.6') == 1
    assert core('Na 0 0 0; Cl 0 0 2.4') == 10
    assert core('K 0 0 0; Br 0 0 2.8') == 23
    assert core('Ga 0 0 0', spin=1) == 14


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
