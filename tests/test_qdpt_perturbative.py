"""The QDPT downfolding driver (src/MultiReference/QDPT/perturbative.py) and
its solver dispatcher (src/Solvers/active_space.py).

Run: python tests/test_qdpt_perturbative.py, or under pytest.

Checks:
  1. run_pipeline agrees between the 'pyscf_fci' and default ('openfermion')
     backends on the same downfolded g_eff (to round-off), and carries nroots
     through; 'pyscf_fci' refuses a three-body vertex, a non-Hermitian g_eff
     without symmetrize and a non-spin-adapted one; with a frozen core and
     several external orbitals the spin-free and spin-orbital downfoldings
     and every backend agree
  2. build_effective_hamiltonian's active_space= keyword reproduces the same
     g_eff as the equivalent manual window/permuted mean field, including a
     scattered window only reachable via permutation
  3. the unavailable solvers ('pyhast', 'pdaggerq', 'ccsdt') refuse cleanly
  4. the 'dmrg' backend, run in its own subprocess (see qdpt_dmrg_check.py
     for why), agrees with 'pyscf_fci' on the same downfolded g_eff
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.active_space import ActiveSpace
from src.MultiReference.QDPT.perturbative import build_effective_hamiltonian, run_pipeline
from src.Solvers.active_space import solve

H4 = 'H 0 0 0; H 1.8 0 0; H 0.54 2.34 0; H 2.52 1.62 0.9'


def _h4():
    mol = gto.M(atom=H4, basis='sto-3g', unit='Bohr', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12, conv_tol_grad=1e-9)
    return mol, mf


def check_pyscf_fci_and_openfermion_agree():
    mol, mf = _h4()
    kw = dict(max_pt_order=2, max_vertex=2, mol=mol, mf=mf, n_occ_spatial=0,
              n_act_spatial=3)

    e_fci = run_pipeline(solver='pyscf_fci', nroots=2, **kw)
    nroots_ok = len(e_fci) == 2 and e_fci[1] > e_fci[0]

    e_fci_ground = run_pipeline(solver='pyscf_fci', **kw)
    repeat_ok = abs(e_fci_ground[0] - e_fci[0]) < 1e-12

    e_ed = run_pipeline(solver='openfermion', **kw)
    dim_ok = len(e_ed) == 9  # C(3, 2)^2 determinants at Sz = 0
    d = abs(e_ed[0].real - e_fci[0])
    cross_ok = d < 1e-10  # both solve the Hermitian part of the same g_eff

    ok = nroots_ok and repeat_ok and dim_ok and cross_ok
    print(f"pyscf_fci vs openfermion on H4/sto-3g QDPT(2,2): nroots carried="
          f"{nroots_ok}, repeatable={repeat_ok}, determinant dim={len(e_ed)}, "
          f"|dE|={d:.2e}: {'OK' if ok else 'FAIL'}")
    return ok


def check_two_body_backends_refuse_what_they_cannot_carry():
    """'pyscf_fci' and 'dmrg' take a Hermitian, spin-adapted one- and two-body
    Hamiltonian: a three-body vertex, a non-Hermitian g_eff without
    symmetrize, or a spin-orbital g_eff that is not spin-adapted is an error,
    never silently dropped or averaged."""
    mol, mf = _h4()
    kw = dict(max_pt_order=2, mol=mol, mf=mf, n_occ_spatial=0, n_act_spatial=3)

    def refused(**extra):
        try:
            run_pipeline(solver='pyscf_fci', **kw, **extra)
        except ValueError:
            return True
        return False

    three_body = refused(max_vertex=3) and refused(max_vertex=3, spin_free=True)
    non_hermitian = refused(max_vertex=2, symmetrize=False)
    g_eff, space = build_effective_hamiltonian(mol, mf, 0, 3, max_pt_order=2,
                                               max_vertex=2)
    broken = dict(g_eff)
    broken[2] = g_eff[2].copy()
    broken[2][0, 2, 0, 2] += 1e-3                     # an alpha-alpha element alone
    broken[2][2, 0, 2, 0] += 1e-3
    try:
        solve('pyscf_fci', broken, space['n_orbs'], space['n_elec_active'],
              space['act_idx'], space['act_occ_idx'], space['act_vir_idx'])
        not_adapted = False
    except ValueError:
        not_adapted = True
    ok = three_body and non_hermitian and not_adapted
    print(f"pyscf_fci refuses a three-body vertex={three_body}, a non-Hermitian "
          f"g_eff without symmetrize={non_hermitian}, a non-spin-adapted one="
          f"{not_adapted}: {'OK' if ok else 'FAIL'}")
    return ok


def check_routes_agree_with_a_frozen_core():
    """H4/6-31G, one core orbital, three active, four external: the core
    energy is there, the spin-free and spin-orbital downfoldings give the same
    spectrum with the three-body vertex, and every backend the same two-body
    ground state."""
    mol = gto.M(atom=H4, basis='6-31g', unit='Bohr', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12, conv_tol_grad=1e-9)
    kw = dict(max_pt_order=2, mol=mol, mf=mf, n_occ_spatial=1, n_act_spatial=3)
    g_eff, _ = build_effective_hamiltonian(mol, mf, 1, 3, max_pt_order=2, max_vertex=2)
    core_ok = abs(g_eff[0]) > 1e-3
    e3 = run_pipeline(solver='openfermion', max_vertex=3, **kw)[0].real
    e3_sf = run_pipeline(solver='openfermion', max_vertex=3, spin_free=True, **kw)[0].real
    e2 = run_pipeline(solver='openfermion', max_vertex=2, **kw)[0].real
    e2_fci = run_pipeline(solver='pyscf_fci', max_vertex=2, **kw)[0]
    e2_fci_sf = run_pipeline(solver='pyscf_fci', max_vertex=2, spin_free=True, **kw)[0]
    d3, d2 = abs(e3_sf - e3), max(abs(e2_fci - e2), abs(e2_fci_sf - e2))
    ok = core_ok and d3 < 1e-10 and d2 < 1e-10
    print(f"H4/6-31G frozen core: core energy {g_eff[0]:+.6f}; spin-free vs "
          f"spin-orbital with g3 |dE|={d3:.1e}; pyscf_fci (both) vs determinant "
          f"basis |dE|={d2:.1e}: {'OK' if ok else 'FAIL'}")
    return ok


def check_active_space_keyword_matches_manual_window():
    mol, mf = _h4()
    nocc, norb = mol.nelectron // 2, mol.nao
    kw = dict(max_pt_order=2, max_vertex=2)

    ref, _ = build_effective_hamiltonian(mol, mf, 0, 3, **kw)
    space = ActiveSpace.from_indices(norb, nocc, [0, 1], [2])
    got, info = build_effective_hamiltonian(mol, mf, active_space=space, **kw)
    contiguous_ok = (info['active_space'] is space
                     and all(np.array_equal(np.asarray(got[n]), np.asarray(ref[n]))
                            for n in (0, 1, 2)))

    # a window the energy ordering cannot produce: only the permutation reaches it
    scattered = ActiveSpace.from_indices(norb, nocc, [0, 1], [3])
    shim = scattered.permuted_mf(mf)
    manual, _ = build_effective_hamiltonian(mol, shim, *scattered.contiguous_counts(), **kw)
    keyed, _ = build_effective_hamiltonian(mol, mf, active_space=scattered, **kw)
    scattered_ok = all(np.array_equal(np.asarray(keyed[n]), np.asarray(manual[n]))
                      for n in (0, 1, 2))

    ok = contiguous_ok and scattered_ok
    print(f"active_space= reproduces a manual contiguous window="
          f"{contiguous_ok}, and a scattered one reachable only by "
          f"permutation={scattered_ok}: {'OK' if ok else 'FAIL'}")
    return ok


def check_unavailable_solvers_refuse_cleanly():
    mol, mf = _h4()
    g_eff, space = build_effective_hamiltonian(mol, mf, 0, 3, max_pt_order=2,
                                               max_vertex=2)
    ok = True
    for name in ('pyhast', 'pdaggerq', 'ccsdt', 'CCSDT'):
        try:
            solve(name, g_eff, space['n_orbs'], space['n_elec_active'],
                 space['act_idx'], space['act_occ_idx'], space['act_vir_idx'])
            ok = False
        except NotImplementedError:
            pass
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"  {name} raised {type(exc).__name__}, not NotImplementedError")
            ok = False
    print(f"pyhast/pdaggerq/ccsdt/CCSDT all raise NotImplementedError: "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def check_dmrg_backend():
    probe = subprocess.run([sys.executable, '-c', 'import block2'],
                          capture_output=True)
    if probe.returncode != 0:
        print("dmrg backend: block2 not importable in this environment, "
              "skipping (not a failure): OK")
        return True

    script = os.path.join(os.path.dirname(__file__), 'qdpt_dmrg_check.py')
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    verified = 'QDPT DMRG BACKEND VERIFIED' in result.stdout
    ok = result.returncode == 0 and verified
    print(f"dmrg backend check (subprocess): {'OK' if ok else 'FAIL'}")
    if not ok:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
    return ok


def run():
    all_ok = True
    for check in (check_pyscf_fci_and_openfermion_agree,
                  check_two_body_backends_refuse_what_they_cannot_carry,
                  check_routes_agree_with_a_frozen_core,
                  check_active_space_keyword_matches_manual_window,
                  check_unavailable_solvers_refuse_cleanly,
                  check_dmrg_backend):
        all_ok &= bool(check())
    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    return all_ok


def test_qdpt_downfolding():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
