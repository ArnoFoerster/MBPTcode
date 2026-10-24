"""The analytic second-order spin-orbit derivative against the element's own
finite difference, on formaldehyde.

BSE@G0W0 on Hartree-Fock, cc-pVDZ, the space-time chain with a Davidson Casida,
at a twisted, pyramidalized geometry with no symmetry, so every coupling is
non-zero. Stretching C=O brings the 3(pi pi*) state down onto 3(n pi*): at
r_CO = 1.45 A T2 sits 66 meV above T1, at 1.38 A 0.52 eV (G1, outside
scissor).

The reference differences the SINGLE element <S1|H_SO|T1> along one fixed
direction, both roots followed by their overlap with the reference and their
signs carried, Richardson-extrapolated over two steps. The analytic route is
`spin_vibronic.spin_vibronic_coupling` with T2 and its singlet partner as the
only paths, off one `StateManifold.evaluate`. The partner is chosen by orbital
character, not by index: on the scissor surface S2 and S3 swap at 1.45 A
(S2 = 5->8 at 5.89 eV is T3's partner, S3 = 7->8 at 6.40 eV is T2's), and
T2 + S2 then keeps a path without the opposite one that cancels it (25%).

WHAT EACH GATE IS FOR:

- near the T1/T2 degeneracy the T2 + partner truncation IS the derivative: the
  T2 path grows as 1/(E_T2 - E_T1) and everything it leaves out is bounded;
- that remainder is bounded, measured: 2.3e-6 Ha/Bohr at 0.52 eV, where it
  is a fifth of the answer rather than a percent;
- the trap on a molecule: differencing the norm over S1 x {T1, T2} cancels
  the T1-T2 mixing and reads the variation of the large S1-T2 element
  instead: 1.08 times the S1-T1 derivative at 66 meV, 9.9 times at 0.52 eV.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics.numerics import Numerics
from src.Base.numerics.constants import (SCF_DIFFERENTIABLE_CONV_TOL,
                                         SCF_DIFFERENTIABLE_GRAD_TOL)
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.state_manifold import StateManifold
from src.gradients.state_identity import mo_overlap, state_overlap
from src.properties.spin_orbit import soc_operator_mo, soc_vectors
from src.properties.spin_vibronic import spin_vibronic_coupling
from tests.isdf_test_grid import g1_numerics

BASIS, AUX = 'cc-pvdz', 'cc-pvdz-ri'
S1, S2, S3, T1, T2 = (('singlet', 0), ('singlet', 1), ('singlet', 2),
                      ('triplet', 0), ('triplet', 1))
#: Cartesian steps in Bohr, a factor of two apart for the extrapolation
STEPS = (2e-3, 1e-3)


def twisted(r_co):
    return (f'C 0 0 0.0; O 0.05 -0.03 {r_co}; H 0.12 0.943 -0.588; '
            f'H -0.20 -0.943 -0.588')


def rhf(mol):
    mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.kernel()
    return mf


def direction(natm):
    """A fixed unit Cartesian direction with no net translation."""
    u = np.random.default_rng(7).normal(size=(natm, 3))
    u -= u.mean(axis=0)
    return u / np.linalg.norm(u)


def displaced_elements(man, ev, mol, u, h):
    """(followed single element (3,), block norm over S1 x {T1, T2}) at x + h u."""
    m = mol.copy()
    m.set_geom_(mol.atom_coords() + h * u, unit='Bohr')
    m.build(False, False)
    e = man.evaluate(mol=m, states=(S1, T1, T2))
    nocc = mol.nelectron // 2
    t = mo_overlap(mol, ev.mf.mo_coeff, m, e.mf.mo_coeff)
    followed = []
    for spin in ('singlet', 'triplet'):
        _, x0, y0 = ev.spectrum[spin]
        _, x, y = e.spectrum[spin]
        ov = state_overlap(t, nocc, x0[:, :1], y0[:, :1], x, y)[1, 1:]
        j = int(np.argmax(np.abs(ov)))
        assert abs(ov[j]) > 0.99, (spin, ov)
        followed.append((j, np.sign(ov[j])))
    (js, ss), (jt, st) = followed
    h_mo = soc_operator_mo(e.mf, m)
    _, xs, ys = e.spectrum['singlet']
    _, xt, yt = e.spectrum['triplet']
    single = ss * st * soc_vectors(h_mo, nocc, (xs[:, js], ys[:, js]),
                                  (xt[:, jt], yt[:, jt]))
    block = sum(float((soc_vectors(h_mo, nocc, (xs[:, 0], ys[:, 0]),
                                  (xt[:, j], yt[:, j])) ** 2).sum())
                for j in (0, 1))
    return single, block


def t2_partner(ev):
    """The singlet of T2's orbital character among S2 and S3."""
    _, xt, _ = ev.spectrum['triplet']
    _, xs, _ = ev.spectrum['singlet']
    ov = {key: abs(float(xs[:, key[1]] @ xt[:, T2[1]])) for key in (S2, S3)}
    key = max(ov, key=ov.get)
    assert ov[key] > 0.9, ov
    return key


@pytest.fixture(scope='module', params=(1.45, 1.38))
def case(request):
    mol = gto.M(atom=twisted(request.param), basis=BASIS, verbose=0)
    chain = ExcitedStateChain(
        mol, rhf, mf=rhf(mol), numerics=g1_numerics(mol, Numerics(bse={'solver': 'davidson', 'adjoint': 'grid', 'nroots': 3})))
    man = StateManifold(chain, states=(S1, S2, S3, T1, T2))
    ev = man.evaluate(couplings=((S2, S1), (S3, S1), (T2, T1)))
    out = spin_vibronic_coupling(ev, soc_operator_mo(ev.mf, ev.mol), S1, T1,
                                 (t2_partner(ev),), (T2,))
    u = direction(mol.natm)
    b0 = float(out['v0_vector'] @ out['v0_vector']
               + out['soc']['triplet2'] @ out['soc']['triplet2'])
    fd, old = {}, {}
    for h in STEPS:
        (vp, bp), (vm, bm) = (displaced_elements(man, ev, mol, u, s * h)
                              for s in (1.0, -1.0))
        fd[h] = (vp - vm) / (2.0 * h)
        old[h] = np.sqrt(max(bp + bm - 2.0 * b0, 0.0) / (2.0 * h ** 2))
    fine, coarse = STEPS[1], STEPS[0]
    return {'r': request.param,
            'gap_tt': ev.omega[T2] - ev.omega[T1],
            'analytic': np.einsum('ax,axe->e', u, out['dv_cart']),
            'single': (4.0 * fd[fine] - fd[coarse]) / 3.0,
            # the coarse step: a second difference of the block norm squared,
            # 5.7e-14 on 4.5e-8 at the fine one, where the Davidson
            # stops moves it by 1e-15 (|block - single| / single 0.045 to 0.061
            # at 1.45 A); 2.3e-13 at the coarse one (0.0704, 0.0703 at
            # conv_tol 1e-8, 1e-10)
            'block': old[coarse]}


def test_t2_truncation_is_the_derivative_near_the_degeneracy(case):
    """At 66 meV the T2 + partner paths reproduce the element's difference."""
    res = np.linalg.norm(case['analytic'] - case['single'])
    rel = res / np.linalg.norm(case['single'])
    if case['r'] == 1.45:
        assert case['gap_tt'] < 0.004
        assert rel < 0.03
    else:
        # the remainder is the same size where T2 does not dominate
        assert case['gap_tt'] > 0.015
        assert 0.05 < rel < 0.5
        assert res < 3e-6


def test_block_norm_is_another_quantity(case):
    """The block norm reads the large S1-T2 element's variation."""
    single = np.linalg.norm(case['single'])
    off_old = abs(case['block'] - single) / single
    off_new = np.linalg.norm(case['analytic'] - case['single']) / single
    assert off_old > 0.05 and off_new < 0.5 * off_old
    if case['r'] == 1.38:
        # where T2 does not dominate, the block norm is ten times the element
        assert case['block'] > 3.0 * single
