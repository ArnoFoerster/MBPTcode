"""Gates for the classical soft-mode block of the Duschinsky bath
(`duschinsky.split_soft_modes`, `RateModel.classical_below_cm`).

1. A soft mode that keeps its frequency and does not mix: the quantum band is
   the stiff band convolved with the soft mode's displaced-oscillator band,
   and the classical block (a Gaussian of mean lambda_soft and variance
   2 lambda_soft kT, Ulstrup and Jortner, J. Chem. Phys. 63, 4358 (1975))
   reproduces it at 300 K to the order (hbar w / kT)^2.
2. With no mode below the cutoff the rate chain is bitwise the one with every
   mode quantum; at an orthogonal J the stiff model's J is orthogonal.
3. 4-(dimethylamino)benzonitrile S1/T1 (TDA-wB97X-D3/def2-SVP Hessians, gas):
   with the modes below 100 cm^-1 classical, RISC passes the guard at 300 K
   with its band centred near lambda, ISC stays accepted, and the rate chain
   runs the full model.
"""
import os
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
sys.path.insert(0, HERE)

import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics.constants import (BOLTZMANN_HARTREE_PER_KELVIN,
                                         DUSCHINSKY_CLASSICAL_BELOW_CM,
                                         HARTREE_TO_CM, HARTREE_TO_EV)
from src.photophysics.rate_chain import RateInputs, RateModel, rate_chain
from src.properties.duschinsky import (DuschinskyModel, duschinsky_density,
                                       duschinsky_guard, split_soft_modes,
                                       sum_rule_lambda)
from test_curvilinear_duschinsky import models, t1_mode_lambda
from test_duschinsky import rotation
from test_duschinsky_guard import dmabn, minimum  # noqa
import test_rate_chain as trc

CM = 1.0 / HARTREE_TO_CM
EV = 1.0 / HARTREE_TO_EV
T = 300.0


def three_modes(w_soft):
    """A soft mode at w_soft (cm^-1, the same in both states, unmixed) and
    two stiff modes rotated into each other, all displaced."""
    w_g = np.array([w_soft, 900.0, 1500.0]) * CM
    w_e = np.array([w_soft, 820.0, 1600.0]) * CM
    j = np.eye(3)
    j[1:, 1:] = rotation(0.3)
    k = np.array([1000.0, 4.0, -2.0])
    return DuschinskyModel(w_g, w_e, j, k)


@pytest.mark.parametrize('direction', ['absorption', 'emission'])
def test_a_classical_soft_mode_is_its_quantum_band_at_room_temperature(
        direction):
    """10 cm^-1 at 300 K, lambda_soft 0.028 eV: the stiff band with the soft
    block's Gaussian against the full quantum band, to 1e-3 of the peak."""
    full = three_modes(10.0)
    stiff, block = split_soft_modes(full, 100.0 * CM)
    assert block.lambda_ground == pytest.approx(block.lambda_excited,
                                                rel=1e-12)
    assert block.coupling < 1e-14
    lam = block.lam(direction)
    kt = BOLTZMANN_HARTREE_PER_KELVIN * T
    sigma = 15.0 * CM
    de = np.linspace(-0.08, 0.02, 41)
    ref = duschinsky_density(full, de, T, direction, broadening=sigma)
    got = duschinsky_density(stiff, de + lam, T, direction,
                             broadening=np.sqrt(sigma ** 2 + 2 * lam * kt))
    assert lam * HARTREE_TO_EV == pytest.approx(0.0282, abs=1e-4)
    assert np.abs(got - ref).max() < 1e-3 * ref.max()


def test_an_orthogonal_j_leaves_an_orthogonal_stiff_j():
    full = DuschinskyModel(np.array([40.0, 900.0, 1500.0]) * CM,
                           np.array([70.0, 820.0, 1600.0]) * CM,
                           _turn3(0.2, 0.3),
                           np.array([30.0, 4.0, -2.0]))
    stiff, block = split_soft_modes(full, 100.0 * CM)
    assert np.abs(stiff.J.T @ stiff.J - np.eye(2)).max() < 1e-12
    assert len(block.omega_ground) == 1
    assert split_soft_modes(full, 30.0 * CM) == (full, None)
    with pytest.raises(ValueError, match='no quantum modes'):
        split_soft_modes(full, 2000.0 * CM)


def _turn3(a, b):
    """A rotation of three modes mixing the first with the second by a and
    the second with the third by b."""
    r1, r2 = np.eye(3), np.eye(3)
    r1[:2, :2] = rotation(a)
    r2[1:, 1:] = rotation(b)
    return r1 @ r2


def test_the_block_keeps_its_zero_point_difference_and_its_own_lambdas():
    """A soft mode whose frequency doubles (20 -> 40 cm^-1), unmixed: the
    stiff model's zero-point shift plus the block's is the full model's, and
    the block's is its own mode's (w_e - w_g)/2; each direction reads the
    initial state's own reorganization."""
    w_g = np.array([20.0, 900.0, 1500.0]) * CM
    w_e = np.array([40.0, 820.0, 1600.0]) * CM
    j = np.eye(3)
    j[1:, 1:] = rotation(0.3)
    full = DuschinskyModel(w_g, w_e, j, np.array([300.0, 4.0, -2.0]))
    stiff, block = split_soft_modes(full, 100.0 * CM)
    for direction, sign in (('absorption', 1.0), ('emission', -1.0)):
        assert (sign * stiff.zero_point_shift()
                + block.zero_point_shift(direction)) == pytest.approx(
            sign * full.zero_point_shift(), rel=1e-12)
    assert block.zero_point == pytest.approx(10.0 * CM, rel=1e-9)
    assert block.lam('absorption') == pytest.approx(
        0.5 * (20.0 * CM) ** 2 * (j[0, 0] * 300.0) ** 2, rel=1e-12)
    assert block.lam('emission') == pytest.approx(
        0.5 * (40.0 * CM) ** 2 * 300.0 ** 2, rel=1e-12)


def test_no_mode_below_the_cutoff_is_bitwise_the_quantum_bath():
    """Synthetic formaldehyde (every mode above 800 cm^-1): the production
    cutoff changes nothing, bit for bit."""
    mol = gto.M(atom=trc.GEO, basis='sto-3g', verbose=0)
    inp = trc.build(mol)
    a = rate_chain(inp, trc.FULL, T)
    b = rate_chain(inp, replace(trc.FULL, classical_below_cm=0.0), T)
    assert trc.FULL.classical_below_cm == DUSCHINSKY_CLASSICAL_BELOW_CM
    assert a['soft_modes'] is None
    for side in ('ISC', 'RISC'):
        for key in ('k_pair', 'k_condon', 'k_ht', 'k_eff'):
            assert a[side][key] == b[side][key]
    assert a['duschinsky_guard'] == b['duschinsky_guard']


def dmabn_inputs(hs, ht, e_s1, e_t1, gap=None):
    """RateInputs of DMABN, each state's energy at the other minimum from its
    own sum rule; with `gap` (Hartree) the singlets lowered at both minima to
    that adiabatic S1 - T1 gap (ORCA's 1.55 eV puts RISC at e^-60, below the
    real-time integral's floor)."""
    if gap is not None:
        e_s1 = e_t1 + gap
    m = models(hs, ht)['curvilinear']
    lam_s = sum_rule_lambda(m, 'absorption')
    lam_t = t1_mode_lambda(hs, ht)
    return RateInputs(
        s1=minimum('S1', hs.mol.atom_coords(),
                   {'S1': e_s1, 'T1': e_t1 + lam_t}),
        t1=minimum('T1', ht.mol.atom_coords(),
                   {'S1': e_s1 + lam_s, 'T1': e_t1}),
        hessians={'S1': hs, 'T1': ht}, gradients={}, solvent={})


def test_dmabn_risc_passes_with_the_soft_modes_classical(dmabn):
    """RISC at 300 K: 2.21 eV above its 0-0 line with every mode quantum,
    0.50 eV (lambda 0.45 eV) with the three T1 modes below 100 cm^-1
    (9.8, 33.4, 53.2) classical, the block at T1's own reorganization; ISC
    0.70 eV against 0.54 eV, the block at S1's. Both pass at 50 and 200
    cm^-1 too."""
    _, hs, ht, _, _ = dmabn
    m = models(hs, ht)['curvilinear']
    lam_e = {'absorption': sum_rule_lambda(m, 'absorption'),
             'emission': t1_mode_lambda(hs, ht)}
    stiff, block = split_soft_modes(m, 100.0 * CM)
    assert np.round(block.omega_ground * HARTREE_TO_CM, 1).tolist() == \
        [9.8, 33.4, 53.2]
    centre = {}
    for direction in ('absorption', 'emission'):
        rec = duschinsky_guard(stiff, T, direction, lam_e[direction],
                               block.lam(direction))
        assert rec['status'] == 'computed', rec
        centre[direction] = rec['first_moment_eV'] - rec['zero_point_eV']
    assert centre['absorption'] == pytest.approx(0.4997, abs=0.002)
    assert centre['emission'] == pytest.approx(0.7028, abs=0.002)
    for cut in (50.0, 200.0):
        stiff, block = split_soft_modes(m, cut * CM)
        for direction in ('absorption', 'emission'):
            assert duschinsky_guard(stiff, T, direction, lam_e[direction],
                                    block.lam(direction))['status'] == \
                'computed'


def test_dmabn_rate_chain_runs_the_full_model(dmabn):
    _, hs, ht, e_s1, e_t1 = dmabn
    inp = dmabn_inputs(hs, ht, e_s1, e_t1, gap=0.2 * EV)
    full = replace(RateModel.full('curvilinear', 0.0), coupling='condon',
                   triplets='T1')
    out = rate_chain(inp, full, T)
    assert out['soft_modes']['omega_ground_cm'] == pytest.approx(
        [9.8, 33.4, 53.2], abs=0.05)
    for side in ('ISC', 'RISC'):
        assert out['duschinsky_guard'][side]['status'] == 'computed'
        assert out[side]['k_pair'] > 0.0
    # the gap is shifted by the two Hessians' whole zero-point difference,
    # and the block reads each direction's initial state
    zpe = out['zero_point_shift_eV']
    assert out['RISC']['zero_point_used_eV'] == pytest.approx(zpe, abs=1e-12)
    assert out['ISC']['zero_point_used_eV'] == pytest.approx(-zpe, abs=1e-12)
    soft = out['soft_modes']
    assert out['RISC']['lambda_soft_block_eV'] == soft['lambda_ground_eV']
    assert out['ISC']['lambda_soft_block_eV'] == soft['lambda_excited_eV']
    # the gas-phase density takes its width from the classical block alone
    assert out['RISC']['lambda_solvent_eV'] == 0.0


def test_refusals():
    with pytest.raises(ValueError, match='classical_below_cm'):
        replace(trc.FULL, classical_below_cm=-1.0)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
