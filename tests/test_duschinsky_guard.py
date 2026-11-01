"""Gates for the Duschinsky guard (`duschinsky.duschinsky_guard`) and the
rate chain's refusal of a Duschinsky model that does not describe a crossing.

4-(dimethylamino)benzonitrile (DMABN), gas: TDA-wB97X-D3/def2-SVP Hessians of
S1 and T1, each at its own minimum (tests/data/dmabn_s1_t1_hessians.npz). The
two minima differ by a large amino twist, and the rectilinear Duschinsky
matrix of the two Cartesian mode sets turns the soft T1 and S1 modes into the
C-H and skeletal stretches: the band's closed-form first moment comes out at
tens of eV at 300 K. The guard refuses that model; the rate chain then
refuses its full rung and the ladder goes on to the displaced oscillators,
which run. A rigid molecule (synthetic formaldehyde in test_rate_chain.py,
water here with energies of its own) passes.
"""
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto

from src.Base.geometry.geometry import molecule_at
from src.Base.numerics.constants import (DUSCHINSKY_SOFT_STIFF_WEIGHT_MAX,
                                         HARTREE_TO_CM, HARTREE_TO_EV)
from src.photophysics.minimum import StateMinimum
from src.photophysics.rate_chain import RateInputs, RateModel, rate_chain
from src.photophysics.rate_models import RateRefused
from src.properties.duschinsky import (CartesianHessian, DuschinskyModel,
                                       displacement_from_minima,
                                       duschinsky_guard, guard_message,
                                       mean_energy_transfer, mode_sets,
                                       soft_stiff_weight, sum_rule_lambda)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                    'dmabn_s1_t1_hessians.npz')
EV = 1.0 / HARTREE_TO_EV
CM = 1.0 / HARTREE_TO_CM
T = 300.0


@pytest.fixture(scope='module')
def dmabn():
    """(template, S1 Hessian, T1 Hessian, S1 energy, T1 energy), each state
    at its own minimum."""
    d = np.load(DATA)
    mol = gto.M(atom=[(str(e), tuple(r)) for e, r in
                      zip(d['symbols'], d['coords_s1'])],
                unit='Bohr', basis='sto-3g', verbose=0)
    hs = CartesianHessian(mol, d['hessian_s1'], 'TDA-wB97X-D3 S1')
    ht = CartesianHessian(molecule_at(mol, d['coords_t1']), d['hessian_t1'],
                          'TDA-wB97X-D3 T1')
    return mol, hs, ht, float(d['energy_s1']), float(d['energy_t1'])


def rectilinear(hs, ht):
    """The rate chain's model: T1 'ground', S1 'excited', the rectilinear
    J of the two mode sets and the curvilinear K of the minima."""
    sg, se, j, dev = mode_sets(ht, hs)
    k = displacement_from_minima(sg, se, ht.mol, hs.mol, 'curvilinear')
    return DuschinskyModel(sg.omega, se.omega, j, k, orthogonality=dev), sg


def test_dmabn_rectilinear_band_moments_are_tens_of_ev(dmabn):
    """The failure the guard exists for, as reported: <x> 46.7 eV for RISC
    and 16.6 eV for ISC at 300 K, against a reorganization below 1 eV."""
    _, hs, ht, _, _ = dmabn
    model, _ = rectilinear(hs, ht)
    assert mean_energy_transfer(model, T, 'absorption') * HARTREE_TO_EV == \
        pytest.approx(46.69, abs=0.01)
    assert mean_energy_transfer(model, T, 'emission') * HARTREE_TO_EV == \
        pytest.approx(16.63, abs=0.01)
    assert sum_rule_lambda(model, 'absorption') * HARTREE_TO_EV < 1.0
    assert soft_stiff_weight(model, 'absorption') > 0.2
    assert soft_stiff_weight(model, 'emission') > 0.19


def test_dmabn_rectilinear_model_is_refused(dmabn):
    """Refused in both directions even where the energies are granted the
    model's own sum rule: the soft modes sit in the stretches and the band
    is centred tens of eV away from lambda."""
    _, hs, ht, _, _ = dmabn
    model, _ = rectilinear(hs, ht)
    for direction in ('absorption', 'emission'):
        rec = duschinsky_guard(model, T, direction,
                               sum_rule_lambda(model, direction))
        assert rec['status'] == 'refused'
        assert rec['refused_by'] == ['mixing', 'first_moment']
        assert rec['soft_stiff_weight'] > DUSCHINSKY_SOFT_STIFF_WEIGHT_MAX
        assert rec['first_moment_ratio'] > 2.0
        assert 'stiff modes' in guard_message(rec)


def test_the_guard_refuses_each_test_alone():
    """One clean two-mode model, then each failure: a lambda from the
    energies far from the sum rule, a frequency change whose distortion
    energy dwarfs lambda, a soft mode turned into a stiff one, no energies
    at all."""
    w_g = np.array([500.0, 1500.0]) * CM
    w_e = np.array([480.0, 1450.0]) * CM
    k = np.array([2.0, 1.0])
    clean = DuschinskyModel(w_g, w_e, np.eye(2), k)
    lam = sum_rule_lambda(clean, 'absorption')
    assert duschinsky_guard(clean, T, 'absorption', lam)['status'] == \
        'computed'
    rec = duschinsky_guard(clean, T, 'absorption', 3.0 * lam)
    assert 'sum_rule' in rec['refused_by'] and 'mixing' not in \
        rec['refused_by']
    # a stiff mode that more than doubles: the distortion energy alone
    stiffened = DuschinskyModel(np.array([1000.0, 1100.0]) * CM,
                                np.array([2500.0, 1100.0]) * CM,
                                np.eye(2), np.ones(2))
    rec = duschinsky_guard(stiffened, T, 'absorption',
                           sum_rule_lambda(stiffened, 'absorption'))
    assert rec['refused_by'] == ['first_moment']
    assert duschinsky_guard(clean, T, 'absorption', None)['refused_by'] == \
        ['sum_rule', 'first_moment']
    c, s = np.cos(0.2), np.sin(0.2)
    soft = DuschinskyModel(np.array([20.0, 1500.0]) * CM,
                           np.array([25.0, 1450.0]) * CM,
                           np.array([[c, -s], [s, c]]), np.zeros(2))
    assert soft_stiff_weight(soft, 'absorption') == pytest.approx(s * s,
                                                                rel=1e-12)
    rec = duschinsky_guard(soft, T, 'absorption', 0.1 * EV)
    assert 'mixing' in rec['refused_by']


def minimum(where, coords, energy):
    """A Condon-only `StateMinimum` of DMABN with S1 and T1 energies."""
    socme = np.array([[0.0], [1.0]]) * CM            # rows S0, S1; column T1
    return StateMinimum(
        method='TDA-wB97X-D3', where=where, coords_bohr=np.asarray(coords),
        e0=energy['T1'] - 3.0 * EV, energy=dict(energy),
        omega={k: v - energy['T1'] + 3.0 * EV for k, v in energy.items()},
        emitter='S1', reservoir='T1', socme=socme, soc_dv_cart=None,
        soc_dv_paths=None, dipole=None, f_osc=None, d_ground=None,
        solvation='gas')


def test_dmabn_rate_chain_refuses_duschinsky_and_runs_displaced(dmabn):
    """The rate chain on DMABN: the rectilinear Duschinsky model is refused
    with the guard's reason, the displaced-oscillator rung runs. Each state's energy
    at the other minimum is its own minimum's plus the curvilinear sum rule
    on its own modes (the energies are not part of the data set)."""
    mol, hs, ht, e_s1, e_t1 = dmabn
    model, sg = rectilinear(hs, ht)
    lam_s = sum_rule_lambda(model, 'absorption')
    sg_s, se_t, _, _ = mode_sets(hs, ht)
    k_t = displacement_from_minima(sg_s, se_t, hs.mol, ht.mol, 'curvilinear')
    lam_t = 0.5 * float(np.sum((se_t.omega * k_t) ** 2))
    s1 = minimum('S1', hs.mol.atom_coords(), {'S1': e_s1, 'T1': e_t1 + lam_t})
    t1 = minimum('T1', ht.mol.atom_coords(), {'S1': e_s1 + lam_s, 'T1': e_t1})
    inp = RateInputs(s1=s1, t1=t1, hessians={'S1': hs, 'T1': ht},
                     gradients={}, solvent={})
    full = replace(RateModel.full('curvilinear', 0.01), coupling='condon',
                   triplets='T1', duschinsky_coordinates='rectilinear')
    with pytest.raises(RateRefused, match='does not describe ISC'):
        rate_chain(inp, full, T)
    off = rate_chain(inp, replace(full, duschinsky=False), T)
    assert off['duschinsky_guard'] is None
    for side in ('ISC', 'RISC'):
        assert off[side]['k_pair'] > 0.0


def test_water_two_bases_pass_with_their_own_energies():
    """A rigid molecule: water HF/STO-3G and HF/6-31G as the two 'states',
    each at its own minimum, the energy lambda of each direction from the
    accepting basis' energy at both minima. Every test passes."""
    from pyscf import scf
    from pyscf.geomopt import geometric_solver
    geo = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'
    hess, mols = {}, {}
    for basis in ('sto-3g', '6-31g'):
        mf = scf.RHF(gto.M(atom=geo, basis=basis, verbose=0)).run(
            conv_tol=1e-12)
        mols[basis] = geometric_solver.optimize(mf, convergence_gmax=1e-7)
        mf = scf.RHF(mols[basis]).run(conv_tol=1e-12)
        hess[basis] = CartesianHessian(mols[basis], mf.Hessian().kernel())

    def energy(basis, geometry):
        mol = molecule_at(mols[basis], mols[geometry].atom_coords())
        return scf.RHF(mol).run(conv_tol=1e-12).e_tot

    lam = {b: energy(b, a) - energy(b, b)
           for a, b in (('sto-3g', '6-31g'), ('6-31g', 'sto-3g'))}
    sg, se, j, dev = mode_sets(hess['sto-3g'], hess['6-31g'])
    k = displacement_from_minima(sg, se, mols['sto-3g'], mols['6-31g'],
                                 'cartesian')
    model = DuschinskyModel(sg.omega, se.omega, j, k, orthogonality=dev)
    for direction, accepting in (('absorption', '6-31g'),
                                 ('emission', 'sto-3g')):
        rec = duschinsky_guard(model, T, direction, lam[accepting])
        assert rec['status'] == 'computed', rec
        assert rec['soft_stiff_weight'] == 0.0


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
