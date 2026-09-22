"""The spin-adaptation factors of the spin-orbit matrix element, measured.

Every other gate on `spin_orbit.py` is invariant to a global scale on |V|:
El-Sayed is a ratio, the sublevel sum rule closes against the same factor it
tests, and the four solver routes share it. So the two constants that turn a
spatial-orbital Casida vector into a physical matrix element -- 1/2 between two
excited states, 1/sqrt(2) from the ground state -- rest on a derivation and on
nothing measured.

Here they are measured. The states are built as explicit determinants over spin
orbitals, the operator is applied with fermionic signs, and

    |V|^2 = sum_M |<A|H_SO|T,M>|^2

is evaluated by second quantization alone. Nothing below imports the factor it
checks, and a rate goes as |V|^2, so sqrt(2) astray in either constant is a
factor of two in every k_ISC.

TAMM-DANCOFF ONLY. With Y = 0 an excited state IS a linear combination of
determinants and the comparison is exact; outside it, (X + Y) against (X - Y)
against (XX + YY) is a choice of transition density rather than an identity,
and no determinant expansion arbitrates between them.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.x2c import sfx2c1e, x2c

from src.properties.spin_orbit import (EXCITED_SPIN_FACTOR,
                                       GROUND_SPIN_FACTOR, coupling,
                                       ground_state_element,
                                       interstate_element, soc_operator,
                                       soc_operator_mo)

# The PAULI matrices. h_SO's alpha^2/4 is the prefactor of sigma, not of the
# spin operator s = sigma/2: Breit-Pauli's (alpha^2/2)(r x p).s is
# (alpha^2/4)(r x p).sigma. Using s here halves every element below and the
# comparison would then confirm a module that is itself halved.
SPIN_MATRICES = np.array([[[0.0, 1.0], [1.0, 0.0]],
                          [[0.0, -1.0j], [1.0j, 0.0]],
                          [[1.0, 0.0], [0.0, -1.0]]], complex)

# (hole spin, particle spin, weight) per spin-adapted single, alpha = 0.
CSF_TERMS = {'singlet': (((0, 0), 2 ** -0.5), ((1, 1), 2 ** -0.5)),
             'triplet0': (((0, 0), 2 ** -0.5), ((1, 1), -2 ** -0.5)),
             'triplet+': (((1, 0), 1.0),),
             'triplet-': (((0, 1), 1.0),)}


def spin_orbital_operator(h_mo):
    """O_PQ = i sum_eta h^eta_pq S^eta_st over spin orbitals P = 2p + s.

    h_SO is stored as the real antisymmetric imaginary part of a Hermitian
    operator, so the i belongs here, with the spin algebra.
    """
    h = np.asarray(h_mo, float)
    return 1j * sum(np.kron(h[eta], SPIN_MATRICES[eta]) for eta in range(3))


def annihilate(det, q):
    """a_q on a determinant held as an ascending tuple of spin orbitals."""
    if q not in det:
        return None, 0.0
    k = det.index(q)
    return det[:k] + det[k + 1:], (-1.0) ** k


def create(det, p):
    """a^dagger_p, with the sign from the operators it anticommutes past."""
    if p in det:
        return None, 0.0
    k = sum(1 for r in det if r < p)
    return det[:k] + (p,) + det[k:], (-1.0) ** k


def apply_one_body(o, ket):
    """O|ket> as {determinant: amplitude}, O in the spin-orbital basis."""
    out = {}
    for det, c in ket.items():
        for q in det:
            mid, s1 = annihilate(det, q)
            for p in np.flatnonzero(np.abs(o[:, q]) > 0.0):
                new, s2 = create(mid, int(p))
                if new is None:
                    continue
                out[new] = out.get(new, 0j) + c * o[p, q] * s1 * s2
    return out


def overlap(bra, ket):
    """<bra|ket> for two determinant expansions."""
    return sum(np.conj(c) * ket[d] for d, c in bra.items() if d in ket)


def excite(ref, i, a, spin_hole, spin_particle):
    """a^dagger_{a,particle} a_{i,hole}|ref>, determinant and fermion sign."""
    mid, s1 = annihilate(ref, 2 * i + spin_hole)
    if mid is None:
        return None, 0.0
    new, s2 = create(mid, 2 * a + spin_particle)
    return new, s1 * s2


def csf(ref, x, nocc, kind):
    """A spin-adapted single as determinants, from a spatial <X|X> = 1 vector."""
    x = np.reshape(x, (nocc, -1))
    out = {}
    for i in range(x.shape[0]):
        for a_off, c in enumerate(x[i]):
            for (hole, particle), w in CSF_TERMS[kind]:
                det, sign = excite(ref, i, nocc + a_off, hole, particle)
                out[det] = out.get(det, 0j) + c * w * sign
    return out


def explicit_coupling(h_mo, nocc, x_t, x_s=None):
    """sqrt(sum_M |<A|H_SO|T,M>|^2) from determinants alone, in Hartree.

    A is the ground state when `x_s` is None, else the singlet root it carries.
    """
    o = spin_orbital_operator(h_mo)
    ref = tuple(range(2 * nocc))
    bra = {ref: 1.0 + 0j} if x_s is None else csf(ref, x_s, nocc, 'singlet')
    total = 0.0
    for kind in ('triplet-', 'triplet0', 'triplet+'):
        total += abs(overlap(bra, apply_one_body(o, csf(ref, x_t, nocc, kind)))) ** 2
    return float(np.sqrt(total))


def normalized_vectors(rng, nocc, nvir, count=2):
    """Casida vectors in this repo's <X|X> = 1, which is what the factors assume."""
    out = []
    for _ in range(count):
        x = rng.standard_normal((nocc, nvir))
        out.append(x / np.linalg.norm(x))
    return out


@pytest.fixture(scope='module')
def water():
    """(mol, mf, h_mo, nocc). Oxygen carries the whole coupling at this size."""
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mol, mf, soc_operator_mo(mf, mol), mol.nelectron // 2


def test_the_determinant_states_are_orthonormal(water):
    """The reference construction must be right before it can judge anything:
    each spin-adapted single is normalized and the singlet is orthogonal to all
    three triplet sublevels."""
    _, mf, _, nocc = water
    nvir = np.asarray(mf.mo_coeff).shape[1] - nocc
    ref = tuple(range(2 * nocc))
    x, = normalized_vectors(np.random.default_rng(0), nocc, nvir, 1)
    states = {k: csf(ref, x, nocc, k) for k in CSF_TERMS}
    for kind, vec in states.items():
        assert abs(overlap(vec, vec)) == pytest.approx(1.0, rel=1e-12), kind
    for kind in ('triplet-', 'triplet0', 'triplet+'):
        assert abs(overlap(states['singlet'], states[kind])) < 1e-12


def test_the_ground_state_spin_factor_is_measured(water):
    """<S_0|H_SO|T,M> summed over M, against 1/sqrt(2) times the ov trace."""
    _, mf, h_mo, nocc = water
    nvir = np.asarray(mf.mo_coeff).shape[1] - nocc
    rng = np.random.default_rng(11)
    for x_t in normalized_vectors(rng, nocc, nvir, 4):
        v_code = coupling(ground_state_element(h_mo, nocc, x_t),
                          GROUND_SPIN_FACTOR)
        assert explicit_coupling(h_mo, nocc, x_t) == pytest.approx(v_code,
                                                                  rel=1e-10)


def test_the_interstate_spin_factor_is_measured(water):
    """<S_I|H_SO|T_J,M> summed over M, against 1/2 times Tr[h gamma^IJ].

    This is the one the rate is built from, and it also fixes the sign of the
    occupied block against the virtual one: flip either and the sum moves.
    """
    _, mf, h_mo, nocc = water
    nvir = np.asarray(mf.mo_coeff).shape[1] - nocc
    rng = np.random.default_rng(23)
    for _ in range(4):
        x_s, x_t = normalized_vectors(rng, nocc, nvir, 2)
        v_code = coupling(interstate_element(h_mo, nocc, x_s, x_t),
                          EXCITED_SPIN_FACTOR)
        assert explicit_coupling(h_mo, nocc, x_t, x_s) == pytest.approx(
            v_code, rel=1e-10)


def test_a_factor_of_root_two_would_be_caught(water):
    """THE DISCRIMINATING CHECK. Both constants are the single most likely
    thing to be wrong by sqrt(2), and every other gate on this module passes
    either way, so the comparison above has to be shown to separate them."""
    _, mf, h_mo, nocc = water
    nvir = np.asarray(mf.mo_coeff).shape[1] - nocc
    rng = np.random.default_rng(5)
    x_s, x_t = normalized_vectors(rng, nocc, nvir, 2)

    ground = explicit_coupling(h_mo, nocc, x_t)
    excited = explicit_coupling(h_mo, nocc, x_t, x_s)
    assert ground > 0.0 and excited > 0.0
    for factor in (np.sqrt(2.0), 1.0 / np.sqrt(2.0), 2.0):
        assert ground != pytest.approx(
            coupling(ground_state_element(h_mo, nocc, x_t),
                     GROUND_SPIN_FACTOR * factor), rel=1e-3)
        assert excited != pytest.approx(
            coupling(interstate_element(h_mo, nocc, x_s, x_t),
                     EXCITED_SPIN_FACTOR * factor), rel=1e-3)


@pytest.mark.parametrize('one_electron, bound',
                         [('x2c', 1e-7), ('breit-pauli', 1e-2)])
def test_against_pyscfs_two_component_operator(water, one_electron, bound):
    """G6, AND THE ONE THAT WOULD HAVE CAUGHT THE HALVING.

    Everything above pairs the module's own h with a spin matrix chosen to
    match it, so a consistent pair is confirmed whether or not it is the right
    pair. pySCF's X2C spin-orbital Hamiltonian settles that from outside: the
    SOC remainder IS the one-body matrix in the spin-orbital basis, so it is
    contracted as it stands and no convention of ours enters it.

    On the 'x2c' route the module's own one-electron operator IS this matrix,
    so the comparison isolates the spin algebra alone. It is not exact to
    machine precision, and the reason is worth knowing: the SOC remainder of
    X2C also carries a small SPIN-FREE term, from the spinor and spin-free
    transformations decoupling slightly differently -- 4e-4 of the operator,
    with the two spin-diagonal blocks agreeing to 2e-11, so it is scalar. A
    singlet-triplet element cannot see a scalar operator, which is why the
    residual is 1e-8 rather than 4e-4, and the component extraction that drops
    it is right to.

    On 'breit-pauli' the comparison also carries the physics difference between
    the leading term and the exact one, a fraction of a per cent for first-row
    atoms -- still an order of magnitude below the factor of two it excludes.
    """
    mol, mf, _, nocc = water
    nao, c = mol.nao_nr(), np.asarray(mf.mo_coeff)
    soc = (x2c.SpinOrbitalX2CHelper(mol).get_hcore(mol)
           - np.kron(np.eye(2), sfx2c1e.SpinFreeX2CHelper(mol).get_hcore(mol)))
    o = np.zeros((2 * c.shape[1],) * 2, complex)          # re-index to 2p + s
    for s in (0, 1):
        for t in (0, 1):
            o[s::2, t::2] = c.T @ soc[s * nao:(s + 1) * nao,
                                      t * nao:(t + 1) * nao] @ c

    ref = tuple(range(2 * nocc))
    nvir = c.shape[1] - nocc
    h = soc_operator_mo(mf, mol, two_electron=None,       # x2c-1e has no 2e SOC
                        one_electron=one_electron)
    for x_t in normalized_vectors(np.random.default_rng(31), nocc, nvir, 3):
        two_c = np.sqrt(sum(
            abs(overlap({ref: 1.0 + 0j},
                        apply_one_body(o, csf(ref, x_t, nocc, kind)))) ** 2
            for kind in ('triplet-', 'triplet0', 'triplet+')))
        ours = coupling(ground_state_element(h, nocc, x_t), GROUND_SPIN_FACTOR)
        assert abs(two_c / ours - 1.0) < bound, (
            f'the module is {two_c / ours:.4f} times pySCF\'s two-component '
            f'operator, which is a convention error rather than a basis one')


def test_the_two_one_electron_routes_agree_on_a_first_row_molecule(water):
    """X2C is the default because it is exact within its own decoupling where
    `int1e_pnucxp` is only the leading term, and because it carries its own
    normalization -- there is no alpha^2/4 to pair with a spin matrix, which is
    the convention that halved this module. On a first-row system the two must
    still agree to well under a per cent, or one of them is not what it says."""
    mol, mf, _, nocc = water
    a, b = (soc_operator(mol, mf.make_rdm1(), one_electron=k)
            for k in ('x2c', 'breit-pauli'))
    assert np.array_equal(soc_operator(mol, mf.make_rdm1()), a), 'x2c is default'
    dev = abs(a - b).max() / abs(b).max()
    assert dev < 0.01, f'x2c and Breit-Pauli differ by {dev:.2%} on a C/O/H system'
    assert dev > 1e-6, 'the two routes are suspiciously identical'
