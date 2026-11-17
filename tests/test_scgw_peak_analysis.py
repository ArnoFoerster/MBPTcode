"""analysis='sum' against analysis='separate' in solve_qp_energy_scgw.

The scGW cycle runs once per molecule, with its Pade fit saved
(save_pade); both analyses then redo the real axis from that same fit
(ref_file), so they see the same Sigma(w) and any difference comes from
the analysis alone. Checked:

- the total A: Tr G against the sum of the diagonal of G in the
  eigenbasis of F (diag_Greal_from_pade) must agree to rounding;
- every kept peak of the total A is the quasiparticle peak of some orbital
  group, at the same position up to one frequency step: in the total A the
  tails of neighbouring levels shift the parabola refinement slightly
  (3e-4 Ha on H2O at eta = 1e-2, step 1e-3), within one A_i they do not;
- its weight is the group's (Z x degeneracy) up to WEIGHT_TOL: the total
  A's boundaries also take in the tails of other levels (on H2O up to
  0.055 apart, e.g. 0.898 against 0.953 at 1.137 Ha), so the tolerance
  only asks for the same peak, not the same number;
- HOMO and LUMO, which the GW100 runs report, agree up to one step;
- a degenerate level is one group with one energy, and its Z is per
  orbital (Ne 2p: weight ~2.9 in the total A, Z ~0.97);
- an unknown analysis is refused.

    python tests/test_scgw_peak_analysis.py
"""
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf

import src.SingleReference.GW.scgw as scgw
from src.Base.pyscf.pyscf_interface import freeze_core

ETA = 1e-2
W_STEP = ETA / 10                  # the frequency step _finish_qp_from_pade uses
A_TOL = 1e-10                      # |A_sum - A_separate|, relative to max |A|
WEIGHT_TOL = 0.1                   # |weight_sum - weight_group|


def scgw_ref_file(atom, tmpdir, name):
    """(mf, mol, nocc, ref_file): frozen-core RHF and one scGW run (RI,
    cc-pVDZ) whose Pade fit is saved to ref_file."""
    mol = gto.M(atom=atom, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.run()
    nocc = mol.nelectron // 2
    mf, mol = freeze_core(mf)
    nocc -= mf.frozen_core

    save_file = os.path.join(tmpdir, name)
    scgw.solve_qp_energy_scgw(mf, mol, nocc, method='RI', auxbasis='cc-pvdz-ri',
                              save_file=save_file, save_pade=True, eta=ETA)
    return mf, mol, nocc, save_file + '.h5'


def both_analyses(mf, mol, nocc, ref_file):
    """The real axis redone from the saved fit, once per analysis."""
    return {an: scgw.solve_qp_energy_scgw(mf, mol, nocc, ref_file=ref_file, eta=ETA,
                                          analysis=an, return_peak_info=True)
            for an in ('sum', 'separate')}


def homo_lumo(qpes, mu):
    occ, virt = qpes[qpes < mu], qpes[qpes > mu]
    return (occ.max() if len(occ) else np.nan), (virt.min() if len(virt) else np.nan)


def check_sum_vs_separate(out, name):
    """The checks of the module docstring that hold for any molecule."""
    q_sum, A_sum, w_sum, _, mu, info_sum = out['sum']
    q_sep, A_sep, w_sep, _, mu_sep, info_sep = out['separate']
    ok = True
    print(f'\n=== {name} ===')

    # Same axis, same mu, same total A.
    same_axis = np.array_equal(w_sum, w_sep) and mu == mu_sep
    dA = np.abs(A_sum - A_sep).max() / np.abs(A_sum).max()
    print(f'same axis and mu: {same_axis}; max |A_sum - A_separate| / max |A| = {dA:.1e} '
          f'(tol {A_TOL:.0e})')
    ok &= same_axis and dA < A_TOL

    # Every kept peak of the total A against the closest group peak.
    weight_group = info_sep['Z'] * info_sep['degeneracy']
    print(f'  {"w_sum (Ha)":>11} {"w_sep (Ha)":>11} {"|dw|":>8} {"wt_sum":>7} {"wt_grp":>7}')
    for q, wt in zip(q_sum, info_sum['weights']):
        if np.all(np.isnan(q_sep)):
            print(f'  {q:11.6f}  no kept peak in separate')
            ok = False
            continue
        i = np.nanargmin(np.abs(q_sep - q))
        dw, dwt = abs(q_sep[i] - q), abs(weight_group[i] - wt)
        good = dw <= W_STEP and dwt <= WEIGHT_TOL
        print(f'  {q:11.6f} {q_sep[i]:11.6f} {dw:8.1e} {wt:7.3f} {weight_group[i]:7.3f}'
              f'{"" if good else "  <- off"}')
        ok &= good

    # And the other way round: no kept group without a kept peak in the sum.
    n_groups_kept = len(np.unique(info_sep['group'][~np.isnan(q_sep)]))
    print(f'kept peaks: {len(q_sum)} (sum), {n_groups_kept} groups (separate)')
    ok &= n_groups_kept == len(q_sum)

    # HOMO / LUMO as the GW100 runs take them.
    (h_sum, l_sum), (h_sep, l_sep) = homo_lumo(q_sum, mu), homo_lumo(q_sep, mu)
    print(f'HOMO {h_sum:.6f} / {h_sep:.6f} Ha, LUMO {l_sum:.6f} / {l_sep:.6f} Ha (sum / separate)')
    ok &= abs(h_sum - h_sep) <= W_STEP and abs(l_sum - l_sep) <= W_STEP

    print(f'{name}: {"PASS" if ok else "FAIL"}')
    return bool(ok)


def check_degenerate_level(out, name, members):
    """The orbitals `members` (a degenerate level of F) form one group with
    one energy, and their Z is the total A's weight / degeneracy."""
    q_sum, _, _, _, _, info_sum = out['sum']
    q_sep, _, _, _, _, info_sep = out['separate']
    members = np.asarray(members)
    ok = True
    print(f'\n=== {name} ===')

    group = info_sep['group'][members]
    one_group = len(np.unique(group)) == 1 and info_sep['degeneracy'][members[0]] == len(members)
    print(f'orbitals {members.tolist()}: groups {group.tolist()}, degeneracy '
          f'{info_sep["degeneracy"][members].tolist()}')
    ok &= one_group

    q = q_sep[members]
    one_energy = not np.isnan(q).any() and np.ptp(q) == 0
    print(f'energies {q.tolist()}')
    ok &= one_energy

    if one_energy:
        i = np.argmin(np.abs(q_sum - q[0]))
        wt_sum, Z = info_sum['weights'][i], info_sep['Z'][members[0]]
        print(f'weight in the total A {wt_sum:.3f}, Z per orbital {Z:.3f}, '
              f'Z x {len(members)} = {Z * len(members):.3f} (tol {WEIGHT_TOL})')
        ok &= 0 < Z <= 1 and abs(Z * len(members) - wt_sum) <= WEIGHT_TOL

    print(f'{name}: {"PASS" if ok else "FAIL"}')
    return bool(ok)


def check_unknown_analysis_refused(mf, mol, nocc, ref_file, name):
    print(f'\n=== {name} ===')
    try:
        scgw.solve_qp_energy_scgw(mf, mol, nocc, ref_file=ref_file, eta=ETA,
                                  analysis='seperate')
    except ValueError as err:
        print(f'refused: {err}\n{name}: PASS')
        return True
    print(f'not refused\n{name}: FAIL')
    return False


def run():
    results = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        # H2O: no degeneracy, close-lying virtual levels.
        mf, mol, nocc, ref = scgw_ref_file('O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
                                           tmpdir, 'h2o')
        out = both_analyses(mf, mol, nocc, ref)
        results['H2O: sum vs separate'] = check_sum_vs_separate(out, 'H2O: sum vs separate')
        results['unknown analysis refused'] = check_unknown_analysis_refused(
            mf, mol, nocc, ref, 'unknown analysis refused')

        # Ne (frozen 1s): 2s, then the threefold 2p (orbitals 1-3) as HOMO.
        mf, mol, nocc, ref = scgw_ref_file('Ne 0 0 0', tmpdir, 'ne')
        out = both_analyses(mf, mol, nocc, ref)
        results['Ne: sum vs separate'] = check_sum_vs_separate(out, 'Ne: sum vs separate')
        results['Ne: 2p one group'] = check_degenerate_level(out, 'Ne: 2p one group', [1, 2, 3])

    print('\n=== Summary ===')
    for name, ok in results.items():
        print(f'{name}: {"PASS" if ok else "FAIL"}')
    return all(results.values())


def test_scgw_peak_analysis():
    assert run()


if __name__ == '__main__':
    ok = run()
    if not ok:
        print('FAILURES DETECTED')
    sys.exit(0 if ok else 1)
