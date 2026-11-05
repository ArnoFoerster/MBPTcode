"""The Ms = +-1 triplets of spin-component scaled ADC(2) on the density-fitted
fold (eeADC.ee_ri_flip).

Water, 6-31G, DF-RHF on cc-pvdz-jkfit, the fold on cc-pvdz-ri, all electrons
correlated. Checks: the Ms = +-1 fold has the spin-flip roots of the
spin-orbital supermatrix built from the same factors
(tests.test_ee_adc_scs.arbiter_roots) to 1e-9 Ha, for SCS, SOS and an
arbitrary (c_os, c_ss); unscaled it has the Ms = 0 triplet's roots; scaled,
T1's Ms = +-1 and Ms = 0 components differ (SCS: by 0.18 eV here).

Run: python tests/test_ee_ri_flip.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.numerics.constants import ADC2_SPIN_SCALING
from src.SingleReference.ADC.eeADC import ee_fold, ee_ri_flip, ee_ri_fold
from src.SingleReference.ADC.eeADC import ee_transition_density as etd
from tests.adc_common import AUX, WATER_ORIGIN, check, rhf_df
from tests.test_ee_adc_scs import arbiter_roots

TILE_GB = 1e-9


def check_flip(mol, mf):
    """Every check."""
    ok = True
    f = ee_ri_fold.bare_factors(mol, mf, frozen=0, auxbasis=AUX)
    eps = np.concatenate([f.eo, f.ev])
    B = etd.factors_of(ee_ri_fold.RIFoldADC2(f, 'singlet'))
    for scaling in (None, ADC2_SPIN_SCALING['scs'], ADC2_SPIN_SCALING['sos'], (0.7, 1.9)):
        op = ee_ri_flip.RIFoldSpinFlip(f, scaling=scaling, tile_gb=TILE_GB)
        flip = np.asarray(ee_fold.solve_folded(op.pieces(), 2, spin='triplet',
                                               tol_omega=1e-11, tol_residual=1e-9).omega)
        ms0 = np.asarray(ee_fold.solve_folded(
            ee_ri_fold.RIFoldADC2(f, 'triplet', scaling=scaling).pieces(), 2,
            spin='triplet', tol_omega=1e-11, tol_residual=1e-9).omega)
        if scaling is None:
            ok &= check(float(np.abs(flip - ms0).max()) < 1e-9,
                        'unscaled, the Ms = +-1 roots are the Ms = 0 triplet\'s',
                        f'{np.abs(flip - ms0).max():.1e} Ha')
            continue
        ws, weight = arbiter_roots(eps, B, f.no, scaling)
        arb = ws[weight > 0.5]
        dev = max(float(np.min(np.abs(arb - w))) for w in flip)
        ok &= check(dev < 1e-9, f'{scaling}: the Ms = +-1 fold has the spin-flip roots '
                    'of the spin-orbital matrix', f'{dev:.1e} Ha')
        ok &= check(abs(flip[0] - ms0[0]) > 1e-4, f'{scaling}: T1 Ms = +-1 and Ms = 0 '
                    'differ', f'{(flip[0] - ms0[0]) * 27.211386:+.4f} eV')
    return ok



def run():
    """Every check; True when all passed."""
    mf = rhf_df(WATER_ORIGIN, '6-31g')
    return check_flip(mf.mol, mf)


def test_ee_ri_flip():
    assert run()


if __name__ == '__main__':
    ok = run()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    sys.exit(0 if ok else 1)
