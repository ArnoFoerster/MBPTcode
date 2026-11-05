"""The ADC(2) spectrum of one geometry and the rate layer's ingredients from
it (gradients.adc.ee_spectrum, properties.transition_moments
.spectrum_properties).

Formaldehyde, cc-pVDZ, DF-RHF on cc-pvdz-jkfit, ADC(2) on cc-pvdz-ri, the
two 1s cores frozen, the roots solved to 1e-12 Ha and the residual 1e-11
(the models' Numerics, so the symmetry zeros below hold to round-off).
Checks:

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
     triplet bra with a singlet ket, a ground ket.
  5. a triplet-triplet pair has charge density only; the SCS-ADC(2) spectrum
     of the same molecule (MP2State spin_scaling 'scs') is labelled
     'scs-adc2' and its roots are the scaled fold's (its E_0 + omega of S1
     equals the SCS ADC2State's to 1e-9 Ha); the rate layer takes
     its energies (`from_adc2`, method 'scs-adc2') but refuses its
     spin-orbit couplings (`from_adc2_spectrum`, or a table handed to
     `from_adc2`), naming the sublevel splitting; the ADC(2) spectrum's
     `StateMinimum` carries its own table, dipoles and energies.

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
from src.photophysics.minimum import from_adc2, from_adc2_spectrum
from src.properties import spin_orbit, transition_moments

# the exact SOMF with X2C, which the tables here were recorded with
SOMF = Numerics().derive('the exact operator the tables were recorded with',
                         properties={'soc_two_electron': 'somf'})
from tests.adc_common import H2CO, check

FROZEN = 2
N_STATES = 3


def make_scf(m):
    mf = scf.RHF(m).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol, mf.conv_tol_grad, mf.verbose = 1e-12, 1e-9, 0
    return mf


# tight roots: a symmetry-forbidden f grows as the roots' residual squared
TIGHT = Numerics(adc={'conv_tol': 1e-12, 'residual_tol': 1e-11})


def mp2_model(mol):
    return MP2State(HFState(mol, make_scf), frozen=FROZEN, auxbasis='cc-pvdz-ri',
                    numerics=TIGHT)


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
                           (('triplet', 0), ('singlet', 0), 'a triplet bra with a singlet ket'),
                           (('singlet', 0), ('ground', 0), 'a ground ket')):
        try:
            spec.transition_density(bra, ket)
            ok &= check(False, f'{what} is refused')
        except ValueError as exc:
            ok &= check(True, f'{what} is refused', str(exc)[:60])
    return ok


def check_scs(mol, spec):
    """Check 5."""
    charge, spin = spec.transition_density(('triplet', 0), ('triplet', 1))
    ok = check(np.abs(spin).max() == 0.0 and np.abs(charge).max() > 1e-6,
               'a triplet-triplet pair carries charge density only')
    def scs_model():
        return MP2State(HFState(mol, make_scf), frozen=FROZEN, auxbasis='cc-pvdz-ri',
                        spin_scaling='scs', numerics=TIGHT)

    scs = adc2_spectrum(scs_model(), N_STATES)
    r = ADC2State(scs_model(), 'singlet', 0, N_STATES, follow='index')(
        np.zeros((mol.nao, mol.nao)), 0.0)
    ok &= check(abs(r.energy - scs.energy('singlet', 0)) < 1e-9,
                'on an SCS model the spectrum\'s S1 is the SCS state model\'s root',
                f"{abs(r.energy - scs.energy('singlet', 0)):.1e} Ha")
    ok &= check(spec.method == 'adc2' and scs.method == 'scs-adc2',
                'the spectra are labelled by method')
    ok &= check(abs(scs.omega['triplet'][0] - spec.omega['triplet'][0]) > 1e-3,
                'SCS moves T1', f"{(scs.omega['triplet'][0] - spec.omega['triplet'][0]) * 27.211386:+.3f} eV")
    m = from_adc2('R0', mol.atom_coords(), scs.e0, scs.omega['singlet'][:N_STATES],
                  scs.omega['triplet'][:N_STATES], method='scs-adc2')
    ok &= check(m.method == 'scs-adc2' and m.socme is None
                and m.energy['T1'] == scs.energy('triplet', 0),
                'an SCS-ADC(2) minimum carries its energies and no spin-orbit table')
    for what, call in (('an SCS-ADC(2) spectrum to from_adc2_spectrum',
                        lambda: from_adc2_spectrum(scs, 'R0', N_STATES, numerics=SOMF)),
                       ('an SCS-ADC(2) minimum with a spin-orbit table',
                        lambda: from_adc2('R0', mol.atom_coords(), scs.e0, [0.1], [0.1],
                                          socme=np.zeros((2, 1)), method='scs-adc2'))):
        try:
            call()
            ok &= check(False, f'{what} is refused')
        except ValueError as exc:
            ok &= check('not well-defined for spin-orbit' in str(exc) and 'splits the sublevels' in str(exc), f'{what} is refused', str(exc)[:60])
    props = transition_moments.spectrum_properties(spec, N_STATES, numerics=SOMF)
    m = from_adc2_spectrum(spec, 'R0', N_STATES, numerics=SOMF)
    ok &= check(m.method == 'adc2' and np.array_equal(m.socme, props['socme'])
                and np.array_equal(m.dipole, props['dipole'])
                and m.energy['T1'] == spec.energy('triplet', 0),
                'from_adc2_spectrum carries the method\'s own table and energies',
                m.provenance['soc_sublevels'])
    return ok


def run():
    """Every check; True when all passed."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mol = gto.M(atom=H2CO, basis='cc-pvdz', verbose=0)
        spec = adc2_spectrum(mp2_model(mol), N_STATES)
        props = transition_moments.spectrum_properties(spec, N_STATES, numerics=SOMF)
        ok = check(spec.flags == [] and props['flags'] == [],
                   'a clean spectrum carries no flags, and its properties none',
                   str(spec.flags))
        ok &= check_energies(mol, spec)
        ok &= check_active(mol, spec, props)
        ok &= check_symmetry(props)
        ok &= check_refusals(spec)
        ok &= check_scs(mol, spec)
    return ok


def test_adc2_spectrum():
    assert run()


if __name__ == '__main__':
    ok = run()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    sys.exit(0 if ok else 1)
