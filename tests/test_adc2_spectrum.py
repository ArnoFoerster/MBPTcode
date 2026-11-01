"""The ADC(2) spectrum of one geometry and the rate layer's ingredients from
it (gradients.adc.ee_spectrum, properties.transition_moments
.spectrum_properties).

Formaldehyde, cc-pVDZ, DF-RHF on cc-pvdz-jkfit, ADC(2) on cc-pvdz-ri, the
two 1s cores frozen. Checks:

  1. E_0 + omega of the two lowest roots of each spin equal ADC2State's
     energies of those roots (follow='index', the gas phase) to 1e-9 Ha.
  2. every transition density is zero on the frozen core, and the table,
     dipoles and oscillator strengths equal the active-space contraction of
     ee_transition_density on the fold's own factors (h_SO and r restricted
     to the active orbitals) to 1e-14.
  3. symmetry (C2v): S1 (n -> pi*, A2) has no transition dipole (f < 1e-20),
     and its coupling to T1 (the same n -> pi* configuration) vanishes
     (El-Sayed, < 1e-10 Ha) while its coupling to T2 (pi -> pi*) does not
     (> 1e-5 Ha).
  4. refusals: a root the solve did not return, a negative root index, a
     triplet bra, a ground ket.
  5. an SCS or SOS MP2 state model is refused: its E_0 is spin-component
     scaled, while these roots and ISR transition densities are ADC(2)'s.

Run: python tests/test_adc2_spectrum.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.SingleReference.ADC.eeADC import ee_transition_density as etd
from src.gradients.adc.ee_spectrum import adc2_spectrum
from src.gradients.adc.ee_state_model import ADC2State
from src.Base.numerics.numerics import Numerics
from src.gradients.state_models import HFState, MP2State
from src.properties import spin_orbit, transition_moments
from tests.adc_common import H2CO, check

FROZEN = 2
N_STATES = 3


def make_scf(m):
    mf = scf.RHF(m).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol, mf.conv_tol_grad, mf.verbose = 1e-12, 1e-9, 0
    return mf


def mp2_model(mol):
    return MP2State(HFState(mol, make_scf), frozen=FROZEN, auxbasis='cc-pvdz-ri',
                    numerics=Numerics())


def check_energies(mol, spec):
    """Check 1."""
    h = np.zeros((mol.nao, mol.nao))
    err = 0.0
    for spin in ('singlet', 'triplet'):
        for k in range(2):
            r = ADC2State(mp2_model(mol), spin, k, N_STATES, follow='index')(h, 0.0)
            err = max(err, abs(r.energy - spec.energy(spin, k)))
    return check(err < 1e-9, 'E_0 + omega equal ADC2State\'s energies', f'{err:.1e} Ha')


def check_active(mol, spec, props):
    """Check 2."""
    ok = True
    nf = FROZEN
    op = spec._roots['singlet'][0]
    f = op.f
    ref = etd.reference_from_factors(np.concatenate([f.eo, f.ev]), etd.factors_of(op),
                                     f.no, ground=True)

    def state(spin, k):
        o, out = spec._roots[spin]
        return etd.fold_state(o, np.asarray(out['y'])[:, k], float(out['omega'][k]),
                              float(out['t1'][k]))

    h = props['h_mo'][:, nf:, nf:]
    C = np.asarray(spec.mf.mo_coeff)[:, nf:]
    ground = [etd.ground_transition_density(ref, state('triplet', j))[1]
              for j in range(N_STATES)]
    pairs = [[etd.transition_density(ref, state('singlet', i), state('triplet', j))[1]
              for j in range(N_STATES)] for i in range(N_STATES)]
    charge = [etd.ground_transition_density(ref, state('singlet', i))[0]
              for i in range(N_STATES)]
    table = spin_orbit.socme_table_from_densities(h, ground, pairs)
    mu = transition_moments.transition_dipoles(mol, C, charge)
    err = max(float(np.abs(table - props['socme']).max()),
              float(np.abs(mu - props['dipole']).max()),
              float(np.abs(transition_moments.oscillator_strengths(
                  spec.omega['singlet'][:N_STATES], mu) - props['f_osc']).max()))
    core = 0.0
    for bra, ket in ((('ground', 0), ('singlet', 0)), (('ground', 0), ('triplet', 0)),
                     (('singlet', 0), ('triplet', 1))):
        for g in spec.transition_density(bra, ket):
            core = max(core, float(np.abs(g[:nf]).max()), float(np.abs(g[:, :nf]).max()))
    ok &= check(core == 0.0, 'the frozen core carries no transition density')
    ok &= check(err < 1e-14, 'table, dipoles and f equal the active-space contraction',
                f'{err:.1e}')
    return ok


def check_symmetry(props):
    """Check 3."""
    ok = check(props['f_osc'][0] < 1e-20, 'S1 (n -> pi*) is dipole-forbidden',
               f"f {props['f_osc'][0]:.1e}")
    v = props['socme']
    ok &= check(v[1, 0] < 1e-10 and v[1, 1] > 1e-5,
                'El-Sayed: S1-T1 vanishes, S1-T2 does not',
                f'{v[1, 0]:.1e} Ha, {v[1, 1]:.1e} Ha')
    return ok


def check_refusals(spec):
    """Check 4."""
    ok = True
    for bra, ket, what in ((('ground', 0), ('singlet', 99), 'a root not returned'),
                           (('ground', 0), ('triplet', -1), 'a negative root index'),
                           (('triplet', 0), ('singlet', 0), 'a triplet bra'),
                           (('singlet', 0), ('ground', 0), 'a ground ket')):
        try:
            spec.transition_density(bra, ket)
            ok &= check(False, f'{what} is refused')
        except ValueError as exc:
            ok &= check(True, f'{what} is refused', str(exc)[:60])
    return ok


def check_scaled(mol):
    """Check 5."""
    ok = True
    for scaling in ('scs', 'sos'):
        mp2 = MP2State(HFState(mol, make_scf), frozen=FROZEN, auxbasis='cc-pvdz-ri',
                       numerics=Numerics(), spin_scaling=scaling)
        try:
            adc2_spectrum(mp2, N_STATES)
            ok &= check(False, f'a {scaling} state model is refused')
        except ValueError as exc:
            ok &= check('spin_scaling' in str(exc), f'a {scaling} state model is refused',
                        str(exc)[:60])
    return ok


def run():
    """Every check; True when all passed."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mol = gto.M(atom=H2CO, basis='cc-pvdz', verbose=0)
        spec = adc2_spectrum(mp2_model(mol), N_STATES)
        props = transition_moments.spectrum_properties(spec, N_STATES)
        ok = check(spec.flags == [] and props['flags'] == [],
                   'a clean spectrum carries no flags, and its properties none',
                   str(spec.flags))
        ok &= check_energies(mol, spec)
        ok &= check_active(mol, spec, props)
        ok &= check_symmetry(props)
        ok &= check_refusals(spec)
        ok &= check_scaled(mol)
    return ok


def test_adc2_spectrum():
    assert run()


if __name__ == '__main__':
    ok = run()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    sys.exit(0 if ok else 1)
