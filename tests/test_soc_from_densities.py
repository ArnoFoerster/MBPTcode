"""Spin-orbit couplings and oscillator strengths from transition densities
(properties.spin_orbit.density_vector, socme_table_from_densities;
properties.transition_moments).

Checks:

  1. CIS limit: on the ADC(1) roots (CIS, the Tamm-Dancoff Casida problem of
     the Hartree-Fock reference) the spin transition densities give the
     Casida route's vectors, <S|H_SO|T> through `interstate_element` to
     1e-15 Ha and the ground element through `ground_state_element` with
     its sqrt 2 spin factor to 1e-12 relative; their charge densities give
     `BSE.solve.oscillator_strengths` of the same vectors to 1e-12.
  2. Spin algebra: for ADC(1) and ADC(2), the sum over the three triplet
     sublevels of |<S|H_SO|T,M>|^2, with the two-component operator
     i sum_eta h^eta sigma_eta in spin orbitals and the spin-orbital dense
     route's eigenvectors (each triplet's three degenerate roots, any mixing
     of them), equals |density_vector|^2 of the spin-adapted singlet and
     Ms = 0 triplet to 1e-10 relative (formaldehyde, sto-3g; S0, S1-S3 by
     T1-T3); the ADC(2) table differs from the ADC(1) one by more than 1e-3
     relative somewhere (the second-order terms reach the coupling).
  3. socme_table_from_densities is the table of those norms, row 0 the
     ground state.

Run: python tests/test_soc_from_densities.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import ao2mo, gto, scf

from src.Base.pyscf.pyscf_interface import (get_antisymmetrized_spin_integrals,
                                            get_two_electron_integrals_chemist)
from src.Base.numerics.constants import GROUND_SPIN_FACTOR
from src.SingleReference.ADC.eeADC import (ee_equations, ee_transition_density,
                                           ee_u_dense_full, ee_utils)
from src.SingleReference.ADC.eeADC.solve import solve_ee_adc
from src.SingleReference.BSE.solve import oscillator_strengths as casida_f
from src.properties import spin_orbit, transition_moments
from tests.adc_common import H2CO, check

etd = ee_transition_density
PAULI = (np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]),
         np.array([[1, 0], [0, -1]], complex))
NROOTS = 3


def setup():
    mol = gto.M(atom=H2CO, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
    mf.kernel()
    return mol, mf


def spin_adapted(mf, mol, level):
    """(omega, states) per spin of the dense spin-free route, and its
    reference."""
    no = mol.nelectron // 2
    eps = np.asarray(mf.mo_energy, float)
    nv = len(eps) - no
    V = get_two_electron_integrals_chemist(mol, mf, representation='spatial'
                                           ).transpose(0, 2, 1, 3)
    out = {}
    for spin, parity in (('singlet', 1), ('triplet', -1)):
        e, Z = solve_ee_adc(mf, level=level, spin=spin, nroots=NROOTS,
                            conv_tol=1e-11)
        out[spin] = (e, Z, [etd.flat_state(Z[:, k], no, nv, level, parity)
                            for k in range(NROOTS)])
    return out, etd.reference_from_integrals(eps, V, no, level, ground=True)


def densities(states, ref):
    """(ground spin densities of the triplets, singlet-triplet spin
    densities, ground charge densities of the singlets)."""
    s, t = states['singlet'][2], states['triplet'][2]
    ground = [etd.ground_transition_density(ref, x)[1] for x in t]
    pairs = [[etd.transition_density(ref, a, b)[1] for b in t] for a in s]
    charge = [etd.ground_transition_density(ref, x)[0] for x in s]
    return ground, pairs, charge


def check_cis(mol, mf, h):
    """Check 1."""
    ok = True
    no = mol.nelectron // 2
    states, _ = spin_adapted(mf, mol, 'adc1')
    zeroth = etd.Reference(None, None, None, None)
    (es, Zs, s), (et, Zt, t) = states['singlet'], states['triplet']
    # the ADC(1) singles blocks are x / sqrt 2 per spin: X = sqrt 2 x
    Xs, Xt = np.sqrt(2.0) * Zs[:Zs.shape[0] // 2], np.sqrt(2.0) * Zt[:Zt.shape[0] // 2]
    err = 0.0
    for i in range(NROOTS):
        for j in range(NROOTS):
            a = spin_orbit.density_vector(h, etd.transition_density(zeroth, s[i], t[j])[1])
            b = spin_orbit.interstate_element(h, no, Xs[:, i], Xt[:, j])
            err = max(err, float(np.abs(a - b).max()))
    ok &= check(err < 1e-15, 'CIS: singlet-triplet elements equal the Casida route',
                f'{err:.1e} Ha')
    err = 0.0
    for j in range(NROOTS):
        a = np.linalg.norm(spin_orbit.density_vector(
            h, etd.ground_transition_density(zeroth, t[j])[1]))
        b = spin_orbit.coupling(spin_orbit.ground_state_element(h, no, Xt[:, j]),
                                GROUND_SPIN_FACTOR)
        err = max(err, abs(a - b) / b)
    ok &= check(err < 1e-12, 'CIS: ground-triplet couplings equal the Casida route',
                f'{err:.1e} relative')
    charge = [etd.ground_transition_density(zeroth, x)[0] for x in s]
    mu = transition_moments.transition_dipoles(mol, mf.mo_coeff, charge)
    f = transition_moments.oscillator_strengths(es, mu)
    f_ref, _ = casida_f(mf, mol, no, es, Xs, np.zeros_like(Xs))
    err = float(np.abs(f - f_ref).max())
    ok &= check(err < 1e-12, 'CIS: oscillator strengths equal the Casida route',
                f'{err:.1e}')
    return ok


def spin_orbital_table(mol, mf, h, level, omega_s, omega_t):
    """The (n_s + 1, n_t) |V| table from spin-orbital eigenvectors, the
    triplet sublevels summed in quadrature."""
    no = mol.nelectron // 2
    eps = np.asarray(mf.mo_energy, float)
    nmo = len(eps)
    eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape((nmo,) * 4)
    _, g = get_antisymmetrized_spin_integrals(
        mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff, eri)
    NO, N = 2 * no, 2 * nmo
    w, U = np.linalg.eigh(ee_u_dense_full.build_supermatrix(
        np.repeat(eps, 2), g, NO, level=level))
    amps = ee_utils.mp_amplitudes(np.repeat(eps, 2), g, NO, N, order=3)
    t2, p0 = amps['t2_1'], ee_utils.mp_density2(amps)
    ns = NO * (N - NO)
    H = 1j * sum(np.kron(h[k], PAULI[k]) for k in range(3))
    be = ee_equations.NUMPY

    def state(k):
        u1 = U[:ns, k].reshape(NO, N - NO)
        u2 = None if level == 'adc1' else ee_utils.unfold_doubles(U[ns:, k], NO, N) / 2
        return u1, u2

    def roots(e0, count):
        idx = np.where(np.abs(w - e0) < 1e-7)[0]
        if len(idx) != count:
            raise AssertionError(f'{len(idx)} spin-orbital roots at {e0}, not {count}')
        return idx

    out = np.zeros((NROOTS + 1, NROOTS))
    for j, et in enumerate(omega_t):
        for k in roots(et, 3):
            u1, u2 = state(k)
            terms = (etd.ground_transition_terms(be, u1, t2=t2) if u2 is None else
                     etd.ground_transition_terms(be, u1, u2, t2, amps['t2_2'],
                                                 amps['t1_2'], p0))
            out[0, j] += abs(np.sum(H * etd.assemble(terms, NO, N - NO))) ** 2
            for i, es in enumerate(omega_s):
                l1, l2 = state(roots(es, 1)[0])
                terms = (etd.transition_density_terms(be, l1, u1) if u2 is None else
                         etd.transition_density_terms(be, l1, u1, l2, u2, t2, p0))
                out[i + 1, j] += abs(np.sum(H * etd.assemble(terms, NO, N - NO))) ** 2
    return np.sqrt(out)


def check_spin_algebra(mol, mf, h):
    """Checks 2 and 3."""
    ok = True
    tables = {}
    for level in ('adc1', 'adc2'):
        states, ref = spin_adapted(mf, mol, level)
        ground, pairs, _ = densities(states, ref)
        table = spin_orbit.socme_table_from_densities(h, ground, pairs)
        manual = np.array([[np.linalg.norm(spin_orbit.density_vector(h, g))
                            for g in ground]]
                          + [[np.linalg.norm(spin_orbit.density_vector(h, g))
                              for g in row] for row in pairs])
        ok &= check(table.shape == (NROOTS + 1, NROOTS)
                    and np.array_equal(table, manual),
                    f'{level}: the table holds the norms, row 0 the ground state')
        ref_table = spin_orbital_table(mol, mf, h, level, states['singlet'][0],
                                       states['triplet'][0])
        big = ref_table > 1e-8
        err = float(np.max(np.abs(table - ref_table)[big] / ref_table[big]))
        small = float(np.max(np.abs(table - ref_table)[~big], initial=0.0))
        ok &= check(err < 1e-10 and small < 1e-12,
                    f'{level}: |V| equals the spin-orbital sum over the three '
                    'sublevels', f'{err:.1e} relative, {small:.1e} Ha on '
                    'the forbidden pairs')
        tables[level] = table
    big = tables['adc1'] > 1e-8
    moved = float(np.max(np.abs(tables['adc2'] - tables['adc1'])[big]
                         / tables['adc1'][big]))
    ok &= check(moved > 1e-3, 'ADC(2) moves the couplings off ADC(1)',
                f'{moved:.1e} relative')
    return ok


def run():
    """Every check; True when all passed."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mol, mf = setup()
        h = spin_orbit.soc_operator_mo(mf, mol)
        ok = check_cis(mol, mf, h)
        ok &= check_spin_algebra(mol, mf, h)
    return ok


def test_soc_from_densities():
    assert run()


if __name__ == '__main__':
    ok = run()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    sys.exit(0 if ok else 1)
