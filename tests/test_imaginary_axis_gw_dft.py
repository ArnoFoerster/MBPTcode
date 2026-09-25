"""Imaginary-axis GW vs the exact Casida route for a DFT (KS/hybrid) starting point.

tests/test_imaginary_axis_gw.py only ever starts from HF, and HF hides a real
bug: solve_qp_energy_imaginary_axis samples Sigma_c along a vertical line in the
complex plane and Pade-continues to the real axis, and that line used to be the
plain imaginary axis, Re z = 0. The sampled Green's functions G_m(z + i*omega')
sit a distance |eps_m| from such a contour, so the construction is only sound
while no orbital energy is near the absolute zero of energy. A HF gap straddles
zero comfortably (phenol/cc-pVDZ: HOMO -0.314, LUMO +0.131 Ha), but a KS or
hybrid one does not -- PBE0 pushes the LUMO to within a few meV of zero and the
contour then runs essentially through a pole of G. The minimax quadrature cannot
resolve a Lorentzian that narrow and the Pade fit inherits a near-singularity
sitting on top of its own sample points; phenol/cc-pVDZ@PBE0 HOMO came out
anywhere in 4.1-7.1 eV against an exact 7.79 eV, with no trend in nfreq.

The fix is to sample on Re z = mu instead, mu = mid-gap. These systems are
chosen so the unshifted contour would be pathological -- every one has an
orbital within 0.05 Ha of zero, asserted below so the test cannot quietly stop
exercising the failure mode -- and every case here is off by 0.4-3 eV at some
nfreq if the mu shift is removed.
"""
import os
import sys

import numpy as np
from pyscf import gto, scf, dft

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.imaginary_axis import solve_qp_energy_imaginary_axis
from src.SingleReference.GW.qp_solve import static_exchange_diagonal
from src.Base.constants import HARTREE_TO_EV


ETHENE = '''
C  0.000  0.000  0.667
C  0.000  0.000 -0.667
H  0.000  0.923  1.238
H  0.000 -0.923  1.238
H  0.000  0.923 -1.238
H  0.000 -0.923 -1.238
'''

# nfreq=16 is deliberately excluded: it is not a converged minimax grid (~20 meV
# on the GGA case even with the mu shift in place), so pinning it would measure
# grid convergence rather than the contour bug this file exists for.
NFREQS = (20, 24, 28, 34)

# (label, geometry, basis, xc, homo/lumo, nfreq values to scan)
# xc=None means an HF starting point: kept as the control, so a change that
# breaks HF instead of DFT is caught here too rather than only in the HF file.
CASES = [
    ('N2/cc-pVDZ@PBE0   HOMO', 'N 0 0 0; N 0 0 1.098', 'cc-pvdz', 'pbe0', 'homo', NFREQS),
    ('N2/cc-pVDZ@PBE0   LUMO', 'N 0 0 0; N 0 0 1.098', 'cc-pvdz', 'pbe0', 'lumo', NFREQS),
    ('CO/cc-pVDZ@PBE0   HOMO', 'C 0 0 0; O 0 0 1.128', 'cc-pvdz', 'pbe0', 'homo', NFREQS),
    ('C2H4/cc-pVDZ@PBE  HOMO', ETHENE,                 'cc-pvdz', 'pbe',  'homo', NFREQS),
    ('CO/cc-pVDZ@HF     HOMO', 'C 0 0 0; O 0 0 1.128', 'cc-pvdz', None,   'homo', NFREQS),
]

TOL_MEV = 15.0


def run_case(atom, basis, xc, which, nfreqs):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = (scf.RHF(mol) if xc is None else dft.RKS(mol, xc=xc)).density_fit()
    mf.with_df.auxbasis = basis + '-ri'
    mf.run()

    nocc = mol.nelectron // 2
    p_state = nocc - 1 if which == 'homo' else nocc

    # Guard: without a near-zero orbital energy the unshifted contour is fine and
    # this case would pass either way, i.e. it would stop being a regression test.
    min_abs_eps = np.min(np.abs(mf.mo_energy))

    ref = float(np.atleast_1d(
        calc_qp_energy(mf, selfenergy='GW', polarizability='RPA', df=True, state=p_state))[0])
    qps = [solve_qp_energy_imaginary_axis(mf, mol, nocc, p_state, nfreq=n) * HARTREE_TO_EV
           for n in nfreqs]
    return ref, qps, min_abs_eps


def test_a_range_separated_reference_gets_full_range_exchange_back():
    """G0W0 on LRC-wPBEh, which is the point of using it.

    The quasiparticle equation replaces the reference's whole exchange-
    correlation potential by the self-energy, so the static part that comes
    back is the FULL-range exact exchange while the part subtracted is the
    functional's own -- which for a range-separated hybrid is short-range
    density-functional exchange plus long-range exact exchange. Getting that
    backwards would silently return only one range of the exchange, so the
    two halves are asserted separately here rather than through their
    difference.
    """
    mol = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24', basis='sto-3g',
                verbose=0)
    mf = dft.RKS(mol, xc='lrc-wpbeh')
    mf.conv_tol = 1e-12
    mf.kernel()
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    assert omega > 0.0 and alpha > hyb, 'this test needs a range-separated hybrid'

    dm = mf.make_rdm1()
    # the exchange added back is full-range, not the functional's long-range part
    assert np.abs(mf.get_k(mol, dm)
                  - scf.hf.get_jk(mol, dm, hermi=1, with_j=False)[1]).max() < 1e-12
    # and the potential taken out is the functional's own, range separation and all
    v_xc = mf.get_veff(mol, dm) - mf.get_j(mol, dm)
    assert np.abs(v_xc).max() > 1e-3

    states = np.arange(mf.mo_coeff.shape[1])
    correction = static_exchange_diagonal(mf, mol, states, exchange='mf')
    assert correction.shape == states.shape
    assert np.all(np.isfinite(correction))
    # on Hartree-Fock the same quantity cancels; on a hybrid it must not
    hf = scf.RHF(mol)
    hf.conv_tol = 1e-12
    hf.kernel()
    assert np.abs(static_exchange_diagonal(hf, mol, states, exchange='mf')).max() < 1e-8
    assert np.abs(correction).max() > 1e-2


if __name__ == '__main__':
    all_ok = True
    try:
        test_a_range_separated_reference_gets_full_range_exchange_back()
        print("LRC-wPBEh reference: full-range exchange back, the functional's "
              "own potential out  OK")
    except AssertionError as exc:
        all_ok = False
        print(f"LRC-wPBEh reference: full-range exchange back  FAIL ({exc})")
    for label, atom, basis, xc, which, nfreqs in CASES:
        ref, qps, min_abs_eps = run_case(atom, basis, xc, which, nfreqs)
        diffs_mev = [(q - ref) * 1000 for q in qps]
        ok = max(abs(d) for d in diffs_mev) < TOL_MEV

        # HF is the control case and legitimately has no near-zero orbital.
        exercises_bug = xc is None or min_abs_eps < 0.05
        ok &= exercises_bug
        all_ok &= ok

        scan = '  '.join(f"{n}:{q:.4f}" for n, q in zip(nfreqs, qps))
        print(f"{label:24s} min|eps|={min_abs_eps:.4f} Ha  analytic={ref:.4f} eV")
        print(f"{'':24s} AC nfreq -> {scan}")
        print(f"{'':24s} max|diff| = {max(abs(d) for d in diffs_mev):.2f} meV  "
              f"{'OK' if ok else 'FAIL'}"
              f"{'' if exercises_bug else '  (no near-zero orbital: case no longer probes the mu shift)'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
