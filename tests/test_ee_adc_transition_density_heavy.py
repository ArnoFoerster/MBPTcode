"""ISR-ADC(2) transition densities against the exact ISR on a split-valence
basis (eeADC/ee_transition_density, tests/exact_isr_pp_oracle).

The checks 1-3 of test_ee_adc_transition_density.py on LiH/6-31G: nine virtual
spatial orbitals, so every state-to-state term meets three distinct virtuals
in one contraction (the slow-tier checks run on minimal bases). The truncated
exact ISR is the ADC(2) secular matrix; gamma^lr of the 14 lowest roots
(14 x 14 pairs) and gamma^0r equal the oracle's to 1e-9; l = r is ee_density's
difference density; leaving out any one term misses the oracle. About 15
minutes on a workstation (7315 determinants).

Run: python tests/test_ee_adc_transition_density_heavy.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.test_ee_adc_transition_density import LIH, check_oracle


def run():
    """Every check; True when all passed."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return check_oracle(LIH, basis='6-31g')


def test_ee_adc_transition_density_heavy():
    assert run()


if __name__ == '__main__':
    ok = run()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    sys.exit(0 if ok else 1)
