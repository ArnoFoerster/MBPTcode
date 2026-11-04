"""Gates for the curvilinear Duschinsky matrix (`curvilinear.
curvilinear_duschinsky`) and the Franck-Condon band of a non-orthogonal J
(`duschinsky`, coordinates='curvilinear').

1. The band of a non-orthogonal J is the sum over vibronic states whose
   overlaps are integrated on a grid (with the Jacobian |det J| of the map
   between the two coordinate sets), in both directions; its mean is the
   closed form.
2. J is the Wilson GF route's L_e^-1 L_g (each state's own G and F matrices
   in non-redundant internals) for water, HF/STO-3G and HF/6-31G as the two
   states (Wilson, Decius and Cross, Molecular Vibrations, 1955; Reimers,
   J. Chem. Phys. 115, 9103 (2001)).
3. At one geometry it is the Cartesian overlap L_e^T L_g, and it departs
   from it linearly with the distance between the minima.
4. A methyl turned by 60 degrees (ethane, valence force field) stays on the
   torsion: nothing of it reaches the stretches, where the Cartesian J puts
   a quarter of it.
5. 4-(dimethylamino)benzonitrile S1/T1 (TDA-wB97X-D3/def2-SVP Hessians):
   the soft modes stay out of the stretches, the T1-mode reorganization of
   the S1 minimum is the curvilinear projection's, and the band moments come
   down from tens of eV to the reorganization plus the distortion of the
   soft modes' frequency change, which at 300 K still refuses RISC.
"""
import os
import sys
import warnings
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
sys.path.insert(0, HERE)

import numpy as np
import pytest
from pyscf import gto
from scipy.stats import norm

from src.Base.geometry.geometry import molecule_at
from src.Base.numerics.constants import (DUSCHINSKY_SOFT_STIFF_WEIGHT_MAX,
                                         HARTREE_TO_CM, HARTREE_TO_EV)
from src.photophysics.rate_chain import RateInputs, RateModel, rate_chain
from src.photophysics.rate_models import RateRefused
from src.photophysics.rates import ManifoldGaps
from src.properties import curvilinear as cv
from src.properties.duschinsky import (CartesianHessian, DuschinskyModel,
                                       DuschinskyOscillators,
                                       displacement_from_minima,
                                       duschinsky_density, duschinsky_guard,
                                       mean_energy_transfer, mode_sets,
                                       soft_stiff_weight, sum_rule_lambda)
from src.properties.optimize import cartesian_hessian
from src.properties.vibronic import align_to
from test_curvilinear import ethane, force_field, rotated
from test_duschinsky import overlap_lines, rotation, water_hessians  # noqa
from test_duschinsky_guard import dmabn, minimum  # noqa

CM = 1.0 / HARTREE_TO_CM
EV = 1.0 / HARTREE_TO_EV
STACKED = ManifoldGaps('stacked', None)

# a strongly rotated, frequency-changed two-mode model whose J is not
# orthogonal (singular values 1.15 and 0.9)
W_G = np.array([400.0, 1100.0]) * CM
W_E = np.array([330.0, 1250.0]) * CM
J_CURV = rotation(0.35) @ np.diag([1.15, 0.9]) @ rotation(-0.2)
CURVED = DuschinskyModel(W_G, W_E, J_CURV,
                         np.array([1.8 / np.sqrt(W_E[0]),
                                   -1.1 / np.sqrt(W_E[1])]),
                         coordinates='curvilinear')


# ------------------------------------------------- the non-orthogonal band
@pytest.mark.parametrize('temperature, direction, n_init',
                         [(1.0, 'absorption', 1), (1.0, 'emission', 1),
                          (150.0, 'absorption', 9), (150.0, 'emission', 9)])
def test_non_orthogonal_band_matches_the_sum_over_states(temperature,
                                                         direction, n_init):
    """The density from G against sum_n P_n sum_m |<m|n>|^2 Gaussian, the
    final states on the grid normalized with |det J| (the position states of
    the two coordinate sets differ by that Jacobian)."""
    wi, wf, j, k = CURVED.oriented(direction)
    x, w = overlap_lines(wi, wf, j, k, temperature, n_init, 40)
    w = w * abs(np.linalg.det(j))
    assert w.sum() == pytest.approx(1.0, abs=2e-9)       # the basis is complete
    sigma = 60.0 * CM
    de = np.linspace(-0.04, 0.04, 41)
    ref = np.array([(w * norm.pdf(-d - x, scale=sigma)).sum() for d in de])
    got = duschinsky_density(CURVED, de, temperature, direction,
                             broadening=sigma)
    assert np.abs(got - ref).max() < 1e-8 * ref.max()
    assert (w * x).sum() == pytest.approx(
        mean_energy_transfer(CURVED, temperature, direction), rel=1e-7)


def test_non_orthogonal_first_moment_is_the_closed_form():
    for direction in ('absorption', 'emission'):
        exact = mean_energy_transfer(CURVED, 300.0, direction)
        k1, _ = DuschinskyOscillators(CURVED).cumulants(300.0, direction)
        assert k1 == pytest.approx(exact, rel=1e-8)


def test_an_orthogonal_curvilinear_model_is_the_rectilinear_one():
    """The same orthogonal J declared curvilinear: the general route (J^-T,
    the inverse for emission) gives the rectilinear band to rounding."""
    rect = DuschinskyModel(W_G, W_E, rotation(0.35), CURVED.K)
    curv = replace(rect, coordinates='curvilinear')
    de = np.linspace(-0.04, 0.04, 21)
    for direction in ('absorption', 'emission'):
        a = duschinsky_density(rect, de, 150.0, direction, broadening=60 * CM)
        b = duschinsky_density(curv, de, 150.0, direction, broadening=60 * CM)
        assert np.abs(a - b).max() < 1e-11 * a.max()


# ------------------------------------------------------------ J from Hessians
def gf_modes(mol, hess, masses, prims):
    """Wilson GF: (frequencies, L_s) of a Cartesian Hessian at a minimum in
    the non-redundant internals `prims`, L_s normalized as L^T G^-1 L = 1."""
    b = cv.wilson_b(mol.atom_coords(), prims)
    minv = np.repeat(1.0 / masses, 3)
    g = (b * minv) @ b.T
    bp = np.linalg.pinv(b)
    f = bp.T @ cartesian_hessian(hess, mol.natm) @ bp
    ev, vec = np.linalg.eig(g @ f)
    order = np.argsort(ev.real)
    lam, ls = ev.real[order], vec.real[:, order]
    ls = ls / np.sqrt(np.einsum('ik,ij,jk->k', ls, np.linalg.inv(g), ls))
    return np.sqrt(lam), ls


def test_water_j_is_the_wilson_gf_route(water_hessians):
    hg, he = water_hessians['sto-3g'], water_hessians['6-31g']
    sg, se, j, dev = mode_sets(hg, he, 'curvilinear')
    assert dev > 1e-3                          # two G matrices, not orthogonal
    prims = cv.common_primitives(sg.mol, se.mol)
    assert [len(p) for p in prims] == [2, 1, 0]   # 3 internals, 3 modes
    wg, lsg = gf_modes(sg.mol, hg.hessian, sg.masses, prims)
    rot = align_to(hg.mol, he.mol, return_rotation=True)[1]
    from src.properties.duschinsky import rotated_hessian
    we, lse = gf_modes(se.mol, rotated_hessian(
        cartesian_hessian(he.hessian, 3), rot), sg.masses, prims)
    assert wg == pytest.approx(sg.omega, rel=1e-8)
    assert we == pytest.approx(se.omega, rel=1e-8)
    ref = np.linalg.solve(lse, lsg)
    # the GF eigenvectors carry their own signs
    w = np.repeat(sg.masses, 3) ** -0.5
    a_g = cv.wilson_b(sg.mol.atom_coords(), prims) * w
    a_e = cv.wilson_b(se.mol.atom_coords(), prims) * w
    sgn_g = np.sign(np.einsum('ik,ik->k', a_g @ sg.modes, lsg))
    sgn_e = np.sign(np.einsum('ik,ik->k', a_e @ se.modes, lse))
    assert np.abs(j - sgn_e[:, None] * ref * sgn_g[None, :]).max() < 1e-8


def test_at_one_geometry_j_is_the_cartesian_overlap(water_hessians):
    """Two Hessians at ONE geometry: the curvilinear J is L_e^T L_g; with the
    second Hessian at a geometry moved by t along a mode, J departs from it
    linearly in t."""
    hg, he = water_hessians['sto-3g'], water_hessians['6-31g']
    same = CartesianHessian(hg.mol, he.hessian)
    sg, se, j, _ = mode_sets(hg, same, 'curvilinear')
    assert np.abs(j - se.modes.T @ sg.modes).max() < 1e-12
    gaps = []
    for t in (1e-3, 2e-3):
        dx = t * (sg.modes[:, 0] * np.repeat(sg.masses, 3) ** -0.5)
        moved = CartesianHessian(molecule_at(hg.mol, hg.mol.atom_coords()
                                             + dx.reshape(-1, 3)), he.hessian)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sg, se, j, _ = mode_sets(hg, moved, 'curvilinear')
        gaps.append(np.abs(j - se.modes.T @ sg.modes).max())
    assert gaps[1] / gaps[0] == pytest.approx(2.0, rel=2e-2)


def test_a_turned_methyl_stays_on_the_torsion():
    """Ethane, its second methyl turned by 60 degrees, one valence force
    field: the torsion's curvilinear column has nothing on the stretches,
    the Cartesian one a quarter of its weight."""
    m0 = ethane()
    m1 = align_to(m0, rotated(m0, 60.0))
    w0, l0, masses, _ = force_field(m0)
    w1, l1, _, _ = force_field(m1)
    j = cv.curvilinear_duschinsky(m0, l0, m1, l1, masses)
    torsion = np.argmin(w0)
    stiff = w1 > 1000.0 * CM
    assert (j[stiff, torsion] ** 2).sum() < 1e-12
    assert ((l1.T @ l0)[stiff, torsion] ** 2).sum() > 0.2


# ----------------------------------------------------------------- DMABN
def models(hs, ht):
    out = {}
    for c in ('rectilinear', 'curvilinear'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sg, se, j, dev = mode_sets(ht, hs, c)
        k = displacement_from_minima(sg, se, ht.mol, hs.mol, 'curvilinear')
        out[c] = DuschinskyModel(sg.omega, se.omega, j, k, orthogonality=dev,
                                 coordinates=c)
    return out


def t1_mode_lambda(hs, ht):
    """The S1 minimum's curvilinear displacement projected directly on the
    T1 modes: its T1 reorganization."""
    ss, st, _, _ = mode_sets(hs, ht, 'rectilinear')
    k = displacement_from_minima(ss, st, hs.mol, ht.mol, 'curvilinear')
    return 0.5 * float(np.sum((st.omega * k) ** 2))


def test_dmabn_curvilinear_j_keeps_the_soft_modes_soft(dmabn):
    _, hs, ht, _, _ = dmabn
    m = models(hs, ht)
    for direction in ('absorption', 'emission'):
        assert soft_stiff_weight(m['rectilinear'], direction) > 0.19
        assert soft_stiff_weight(m['curvilinear'], direction) < \
            0.5 * DUSCHINSKY_SOFT_STIFF_WEIGHT_MAX
    # the S1 minimum on the T1 modes: through the rectilinear J the torsion
    # lands on the stretches (8.1 eV); through the curvilinear J it is the
    # direct projection's 0.54 eV
    lam = t1_mode_lambda(hs, ht)
    assert lam * HARTREE_TO_EV == pytest.approx(0.542, abs=1e-3)
    assert sum_rule_lambda(m['rectilinear'], 'emission') > 10.0 * lam
    assert sum_rule_lambda(m['curvilinear'], 'emission') == pytest.approx(
        lam, rel=2e-3)


def test_dmabn_curvilinear_band_moments(dmabn):
    """ISC: the band's first moment less the zero-point shift is 0.76 eV
    at 300 K against lambda 0.54 eV, accepted. RISC: 2.21 eV against 0.45
    eV at 300 K, refused by the first moment alone: the T1 mode at 9.8
    cm^-1 maps onto S1 modes at 76-86 cm^-1, and the harmonic thermal
    energy of that frequency change, kT (w_f^2 / w_i^2 - 1) / 2, is about
    1 eV. At 1 K both directions pass; the rectilinear model fails at every
    temperature."""
    _, hs, ht, _, _ = dmabn
    m = models(hs, ht)
    lam_e = {'emission': t1_mode_lambda(hs, ht),
             'absorption': sum_rule_lambda(m['curvilinear'], 'absorption')}
    isc = duschinsky_guard(m['curvilinear'], 300.0, 'emission',
                           lam_e['emission'])
    assert isc['status'] == 'computed'
    assert isc['first_moment_eV'] - isc['zero_point_eV'] == pytest.approx(
        0.757, abs=0.002)
    risc = duschinsky_guard(m['curvilinear'], 300.0, 'absorption',
                            lam_e['absorption'])
    assert risc['refused_by'] == ['first_moment']
    assert risc['first_moment_eV'] == pytest.approx(2.195, abs=0.002)
    for direction in ('absorption', 'emission'):
        cold = duschinsky_guard(m['curvilinear'], 1.0, direction,
                                lam_e[direction])
        assert cold['status'] == 'computed', cold
        assert duschinsky_guard(m['rectilinear'], 1.0, direction,
                                lam_e[direction])['status'] == 'refused'


def test_dmabn_rate_chain_records_the_curvilinear_refusal(dmabn):
    """The curvilinear J with every mode quantum at 300 K refuses DMABN by
    the RISC first moment, not by mixing; the rectilinear rung still names the
    mixing; the displaced oscillators run."""
    _, hs, ht, e_s1, e_t1 = dmabn
    m = models(hs, ht)
    lam_s = sum_rule_lambda(m['curvilinear'], 'absorption')
    lam_t = t1_mode_lambda(hs, ht)
    inp = RateInputs(
        s1=minimum('S1', hs.mol.atom_coords(),
                   {'S1': e_s1, 'T1': e_t1 + lam_t}),
        t1=minimum('T1', ht.mol.atom_coords(),
                   {'S1': e_s1 + lam_s, 'T1': e_t1}),
        hessians={'S1': hs, 'T1': ht}, gradients={}, solvent={})
    full = replace(RateModel.full('curvilinear', 0.01), coupling='condon',
                   triplets='T1', classical_below_cm=0.0)
    with pytest.raises(RateRefused, match='RISC: band first moment'):
        rate_chain(inp, full, 300.0, STACKED)
    with pytest.raises(RateRefused, match='ISC: a soft mode'):
        rate_chain(inp, replace(full, duschinsky_coordinates='rectilinear'),
                   300.0, STACKED)
    off = rate_chain(inp, replace(full, duschinsky=False), 300.0, STACKED)
    assert off['huang_rhys_sum_t1_modes'] < 10.0
    for side in ('ISC', 'RISC'):
        assert off[side]['k_pair'] > 0.0


def test_refusals():
    with pytest.raises(ValueError, match='outside'):
        DuschinskyModel(W_G, W_E, np.diag([1.0, 0.3]), CURVED.K,
                        coordinates='curvilinear')
    with pytest.raises(ValueError, match='coordinates are'):
        DuschinskyModel(W_G, W_E, np.eye(2), CURVED.K, coordinates='internal')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
