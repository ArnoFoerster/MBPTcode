"""Gates for src/properties/duschinsky.py: the Franck-Condon band with
Duschinsky rotation and frequency change.

The references are independent of the correlation function: the displaced
oscillator's own generating function (`franck_condon`) at J = 1 and equal
frequencies, a brute-force sum over vibronic states whose overlaps are
integrated on a quadrature grid of Hermite functions, the closed-form mean
energy transfer, and a block-diagonal model whose correlation function is the
product of its blocks'. The Hessian gates use water HF/STO-3G (and HF/6-31G as
a second 'state') through pyscf's analytic Hessian.
"""
import os
import sys
import warnings

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.numerics.constants import (BOLTZMANN_HARTREE_PER_KELVIN,
                                         GAUSSIAN_FWHM_PER_SIGMA,
                                         HARTREE_TO_CM, HARTREE_TO_EV)
from src.properties import duschinsky as dm
from src.properties.band_shape import (DisplacedOscillators, VibronicModel,
                                       band_shape, vibronic_band,
                                       vibronic_band_grid)
from src.properties.duschinsky import (CartesianHessian, DuschinskyModel,
                                       DuschinskyOscillators,
                                       duschinsky_density, duschinsky_model,
                                       log_correlation, mean_energy_transfer,
                                       read_orca_hessian)
from src.properties.franck_condon import (TRAPEZOID, log_generating_function,
                                          bose, fc_weighted_dos)

CM = 1.0 / HARTREE_TO_CM
EV = 1.0 / HARTREE_TO_EV
MODES = np.array([300.0, 800.0, 1500.0]) * CM
DQ = np.array([3.0, -1.5, 0.8])          # signed, Bohr sqrt(m_e)
S_MODES = MODES * DQ ** 2 / 2.0


def rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


# a strongly rotated, frequency-changed two-mode model
W_G = np.array([400.0, 1100.0]) * CM
W_E = np.array([330.0, 1250.0]) * CM
TWO_MODE = DuschinskyModel(W_G, W_E, rotation(0.35),
                           np.array([1.8 / np.sqrt(W_E[0]),
                                     -1.1 / np.sqrt(W_E[1])]))


# ---------------------------------------------- the displaced-oscillator limit
@pytest.mark.parametrize('temperature', [1.0, 300.0])
def test_unit_rotation_and_equal_frequencies_is_the_generating_function(
        temperature):
    """ln G at J = 1, omega_e = omega_g = the displaced oscillator's, every t."""
    model = DuschinskyModel.from_displacement(MODES, DQ)
    t = np.linspace(0.0, 4000.0, 3001)
    got = log_correlation(model, t, temperature, 'emission')
    n = bose(MODES, BOLTZMANN_HARTREE_PER_KELVIN * temperature)
    with np.errstate(divide='ignore'):
        ref = log_generating_function(t, S_MODES, MODES, n, np.log(n), 0.0,
                                       0.0)
    assert np.abs(np.exp(got) - np.exp(ref)).max() < 1e-13
    # absorption reads the same density at J = 1: one spectrum, one G
    assert np.abs(np.exp(log_correlation(model, t, temperature, 'absorption'))
                  - np.exp(ref)).max() < 1e-13


@pytest.mark.parametrize('g_fwhm, l_fwhm', [(0.04, 0.0), (0.0, 0.03),
                                            (0.03, 0.02)])
def test_unit_rotation_band_is_the_displaced_oscillator_band(g_fwhm, l_fwhm):
    """vibronic_band on the J = 1 model = band_shape on S_k, every key.

    The Lorentzian runs both through one real-axis integrator (agreement at
    the roundoff of G); the bare Gaussian runs the displaced oscillator on its
    saddle-shifted contour, a different quadrature."""
    g, l = g_fwhm * EV, l_fwhm * EV
    model = DuschinskyOscillators(DuschinskyModel.from_displacement(MODES, DQ))
    assert isinstance(model, VibronicModel)
    e = vibronic_band_grid(DisplacedOscillators(S_MODES, MODES), 3.0 * EV,
                           300.0, gaussian_fwhm=g, lorentzian_fwhm=l,
                           points_per_width=4)
    ref = band_shape(S_MODES, MODES, 3.0 * EV, 300.0, e, gaussian_fwhm=g,
                     lorentzian_fwhm=l)
    got = vibronic_band(model, 3.0 * EV, 300.0, e, gaussian_fwhm=g,
                        lorentzian_fwhm=l)
    # one integrator: roundoff; two (the bare Gaussian): the real-axis
    # integral stops where G has fallen to 1e-10 of its peak
    bar = 1e-12 if l_fwhm else 1e-9
    for key in ('fc_absorption', 'fc_emission', 'absorption', 'emission',
                'emission_per_nm'):
        assert np.abs(got[key] - ref[key]).max() < bar * ref[key].max(), key
    for key in ('peak_absorption', 'peak_emission', 'fwhm_absorption',
                'fwhm_emission', 'fwhm_emission_nm', 'stokes_shift'):
        assert got[key] == pytest.approx(ref[key], rel=1e-10), key


def test_a_reviving_fast_mode_does_not_end_the_time_grid():
    """A 3000 cm^-1 mode with S = 9 beside soft displaced ones: |G| dips
    below 1e-10 inside the first period and revives after it. The density at
    J = 1 is still the displaced oscillator's (it was 1.1% low at -0.6 eV
    when the grid stopped at the dip)."""
    w = np.array([85., 212., 494., 1130., 1193., 1698., 3032., 3185.]) * CM
    s = np.array([5.55, 2.93, 0.98, 0.61, 0.87, 1.57, 9.08, 1.77])
    model = DuschinskyModel(w, w.copy(), np.eye(len(w)), np.sqrt(2 * s / w))
    lam = 0.1 * EV
    kt = BOLTZMANN_HARTREE_PER_KELVIN * 300.0
    for de in np.array([-0.6, -1.0, -2.0]) * EV:
        rho = duschinsky_density(model, de + lam, 300.0, 'emission',
                                 broadening=np.sqrt(2 * lam * kt))
        ref = fc_weighted_dos(de, s, w, 300.0, lambda_classical=lam)
        assert rho == pytest.approx(ref, rel=1e-8)


def test_band_grid_of_the_displaced_model_is_band_grid():
    from src.properties.band_shape import band_grid
    kw = dict(gaussian_fwhm=0.03 * EV, lorentzian_fwhm=0.01 * EV)
    a = band_grid(S_MODES, MODES, 3.0 * EV, 300.0, **kw)
    b = vibronic_band_grid(DisplacedOscillators(S_MODES, MODES), 3.0 * EV,
                           300.0, **kw)
    assert a[0] == pytest.approx(b[0], rel=1e-12)
    assert a[-1] == pytest.approx(b[-1], rel=1e-12)


# ------------------------------------------------- brute force: two modes
def hermite_functions(q, omega, nmax):
    """psi_n(q), n < nmax, of a unit-mass oscillator of frequency omega."""
    x = np.sqrt(omega) * q
    psi = np.zeros((nmax,) + q.shape)
    psi[0] = (omega / np.pi) ** 0.25 * np.exp(-x ** 2 / 2.0)
    if nmax > 1:
        psi[1] = np.sqrt(2.0) * x * psi[0]
    for n in range(2, nmax):
        psi[n] = (np.sqrt(2.0 / n) * x * psi[n - 1]
                  - np.sqrt((n - 1) / n) * psi[n - 2])
    return psi


def overlap_lines(wi, wf, J, K, temperature, n_init, n_final, operator=None,
                  span=9.0, points=301):
    """(x, weight) of every line: x = E_f(m) - E_i(n) with zero-point
    energies, weight = P_n |<m_f| O(q) |n_i>|^2, the overlaps integrated on a
    grid in the initial coordinates; O = 1 (Condon) or a callable of the grid
    returning one array per Cartesian component."""
    axes = [np.linspace(-span / np.sqrt(w), span / np.sqrt(w), points)
            for w in wi]
    q = np.stack(np.meshgrid(*axes, indexing='ij'))
    da = np.prod([a[1] - a[0] for a in axes])
    qf = np.einsum('ij,jab->iab', J, q) + K[:, None, None]
    pi = [hermite_functions(q[k], wi[k], n_init) for k in range(2)]
    pf = [hermite_functions(qf[k], wf[k], n_final) for k in range(2)]
    ops = [np.ones(q.shape[1:])] if operator is None else operator(q)
    kt = BOLTZMANN_HARTREE_PER_KELVIN * temperature
    with np.errstate(under='ignore'):
        z = np.prod(1.0 / (1.0 - np.exp(-wi / kt)))
    m = np.arange(n_final)
    ef = (m[:, None] + 0.5) * wf[0] + (m[None, :] + 0.5) * wf[1]
    xs, ws = [], []
    for a in range(n_init):
        for b in range(n_init):
            with np.errstate(under='ignore'):
                p = np.exp(-(a * wi[0] + b * wi[1]) / kt) / z
            if p < 1e-16:
                continue
            weight = 0.0
            for op in ops:
                weight = weight + np.einsum('mab,nab,ab->mn', pf[0], pf[1],
                                            pi[0][a] * pi[1][b] * op) ** 2
            xs.append(ef - ((a + 0.5) * wi[0] + (b + 0.5) * wi[1]))
            ws.append(p * weight * da ** 2)
    return np.concatenate([x.ravel() for x in xs]), np.concatenate(
        [w.ravel() for w in ws])


@pytest.mark.parametrize('temperature, direction, n_init',
                         [(1.0, 'absorption', 1), (1.0, 'emission', 1),
                          (150.0, 'absorption', 9), (150.0, 'emission', 9)])
def test_two_mode_model_matches_the_sum_over_states(temperature, direction,
                                                    n_init):
    """Rotation 20 deg, frequencies -17 %/+14 %, both displaced: the density
    from G against sum_n P_n sum_m |<m|n>|^2 Gaussian(dE + x_mn)."""
    wi, wf, J, K = TWO_MODE.oriented(direction)
    x, w = overlap_lines(wi, wf, J, K, temperature, n_init, 40)
    assert w.sum() == pytest.approx(1.0, abs=2e-9)       # the basis is complete
    sigma = 60.0 * CM
    de = np.linspace(-0.04, 0.04, 41)
    ref = np.array([(w * norm.pdf(-d - x, scale=sigma)).sum() for d in de])
    got = duschinsky_density(TWO_MODE, de, temperature, direction,
                             broadening=sigma)
    assert np.abs(got - ref).max() < 1e-8 * ref.max()


def test_the_rotation_and_the_frequency_change_both_matter():
    """The same model with J = 1, or with the excited frequencies set to the
    ground ones, misses the sum over states by far more than the gate bar."""
    x, w = overlap_lines(*TWO_MODE.oriented('absorption'), 1.0, 1, 40)
    sigma = 60.0 * CM
    de = np.linspace(-0.04, 0.04, 41)
    ref = np.array([(w * norm.pdf(-d - x, scale=sigma)).sum() for d in de])
    for broken in (DuschinskyModel(W_G, W_E, np.eye(2), TWO_MODE.K),
                   DuschinskyModel(W_G, W_G, TWO_MODE.J, TWO_MODE.K)):
        got = duschinsky_density(broken, de, 1.0, 'absorption',
                                 broadening=sigma)
        assert np.abs(got - ref).max() > 1e-2 * ref.max()


@pytest.mark.parametrize('temperature', [1.0, 300.0])
def test_first_moment_is_the_closed_form(temperature):
    """<x> of the band = <H_f - H_i> by the virial-free closed form, and the
    band integrates to one."""
    for direction in ('absorption', 'emission'):
        sigma = 50.0 * CM
        exact = mean_energy_transfer(TWO_MODE, temperature, direction)
        # 24 standard deviations of the band (sigma included) either side
        _, var = DuschinskyOscillators(TWO_MODE).cumulants(temperature,
                                                           direction)
        reach = 24.0 * np.sqrt(var + sigma ** 2)
        de = np.arange(-exact - reach, -exact + reach, 0.5 * sigma)
        rho = duschinsky_density(TWO_MODE, de, temperature, direction,
                                 broadening=sigma)
        assert TRAPEZOID(rho, de) == pytest.approx(1.0, abs=1e-9)
        mean = -TRAPEZOID(de * rho, de)
        assert mean == pytest.approx(exact, rel=1e-8)
        k1, _ = DuschinskyOscillators(TWO_MODE).cumulants(temperature,
                                                          direction)
        assert k1 == pytest.approx(exact, rel=1e-8)
    # the 0-0 line moved by the zero-point difference: the means differ
    # from the equal-frequency lambda
    assert TWO_MODE.zero_point_shift() == pytest.approx(
        0.5 * (W_E.sum() - W_G.sum()), rel=1e-14)


# ------------------------------------------------- the branch of sqrt(det)
def block_model(n_blocks, seed=7):
    """n_blocks independent rotated two-mode blocks, frequencies changed by
    -40 %..+60 %, the modes then shuffled: G is the product of the blocks' G,
    whatever order the modes are in."""
    rng = np.random.default_rng(seed)
    wg = rng.uniform(150.0, 1700.0, 2 * n_blocks) * CM
    we = wg * rng.uniform(0.6, 1.6, 2 * n_blocks)
    k = rng.normal(0.0, 0.6, 2 * n_blocks) / np.sqrt(we)
    blocks = [rotation(a) for a in rng.uniform(-0.6, 0.6, n_blocks)]
    J = np.zeros((2 * n_blocks, 2 * n_blocks))
    for b, r in enumerate(blocks):
        J[2 * b:2 * b + 2, 2 * b:2 * b + 2] = r
    perm = rng.permutation(2 * n_blocks)
    whole = DuschinskyModel(wg[perm], we[perm], J[np.ix_(perm, perm)], k[perm])
    parts = [DuschinskyModel(wg[2 * b:2 * b + 2], we[2 * b:2 * b + 2],
                             blocks[b], k[2 * b:2 * b + 2])
             for b in range(n_blocks)]
    return whole, parts


#: 120 modes on a grid of 0.8 rad per step on the fastest mode: det N1 N2
#: turns by more than pi on some steps
BLOCKS, COARSE = 60, np.linspace(0.0, 3000.0, 31)


@pytest.fixture(scope='module')
def blocks():
    whole, parts = block_model(BLOCKS)
    ref = np.prod([np.exp(log_correlation(p, COARSE, 300.0, 'emission'))
                   for p in parts], axis=0)
    return whole, ref


def test_branch_is_continued_through_many_modes(blocks):
    """G of the 120 shuffled modes = the product of the blocks' G."""
    whole, ref = blocks
    got = np.exp(log_correlation(whole, COARSE, 300.0, 'emission'))
    assert np.abs(got - ref).max() < 1e-11


def test_branch_continuation_is_what_holds_it(blocks, monkeypatch):
    """With one principal logarithm per step instead, the 120-mode G jumps
    sign: the gate above can fail."""
    whole, ref = blocks

    def principal(self, log_det, ta, pa, tb, depth):
        pb = self.point(tb)
        d = pb[0] - pa[0]
        return log_det + d.real + 1j * np.angle(np.exp(1j * d.imag)), pb

    monkeypatch.setattr(dm._Correlation, '_advance', principal)
    got = np.exp(log_correlation(whole, COARSE, 300.0, 'emission'))
    assert np.abs(got - ref).max() > 1e-3


# ---------------------------------------------------- the model from Hessians
@pytest.fixture(scope='module')
def water_hessians():
    from pyscf import gto, scf
    from pyscf.geomopt import geometric_solver
    geo = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'
    out = {}
    for basis in ('sto-3g', '6-31g'):
        mol = gto.M(atom=geo, basis=basis, verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-12)
        mol_opt = geometric_solver.optimize(mf, convergence_gmax=1e-7)
        mf = scf.RHF(mol_opt).run(conv_tol=1e-12)
        out[basis] = CartesianHessian(mol_opt, mf.Hessian().kernel(),
                                      f'pyscf RHF/{basis}')
    return out


def turned(hess, seed=3):
    """The same state in another frame: geometry rotated and shifted, the
    Hessian's blocks turned with it."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    from src.Base.geometry.geometry import molecule_at
    from src.properties.optimize import cartesian_hessian
    x = hess.mol.atom_coords() @ q + np.array([0.3, -1.2, 2.0])
    h = cartesian_hessian(hess.hessian, hess.mol.natm)
    return CartesianHessian(molecule_at(hess.mol, x),
                            dm.rotated_hessian(h, q), hess.source)


def test_one_state_against_itself_is_the_identity(water_hessians):
    g = water_hessians['sto-3g']
    m = duschinsky_model(g, turned(g))
    assert np.abs(np.abs(m.J) - np.eye(3)).max() < 1e-9
    assert np.abs(m.K).max() < 1e-7
    assert np.abs(m.omega_e - m.omega_g).max() < 1e-12


def test_the_model_does_not_depend_on_the_excited_frame(water_hessians):
    g, e = water_hessians['sto-3g'], water_hessians['6-31g']
    a = duschinsky_model(g, e)
    b = duschinsky_model(g, turned(e))
    # equal up to the signs of the modes, which are the eigensolver's
    assert np.abs(np.abs(a.J) - np.abs(b.J)).max() < 1e-8
    assert np.abs(np.abs(a.K) - np.abs(b.K)).max() < 1e-7
    assert np.abs(a.omega_e - b.omega_e).max() < 1e-12
    # the excited Hessian NOT turned with its geometry gives another model
    t = turned(e)
    bad = CartesianHessian(t.mol, e.hessian, e.source)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        c = duschinsky_model(g, bad)
    assert np.abs(np.abs(c.J) - np.abs(a.J)).max() > 1e-3


def test_water_duschinsky_keeps_the_symmetry_blocks(water_hessians):
    """C2v: the b2 stretch does not mix with the two a1 modes, and the
    displacement is totally symmetric; the a1 pair does rotate."""
    m = duschinsky_model(water_hessians['sto-3g'], water_hessians['6-31g'])
    b2 = int(np.argmax(np.abs(m.J[:, 2])))     # the b2 mode is the highest
    a1 = [k for k in range(3) if k != 2]
    assert b2 == 2
    assert np.abs(m.J[np.ix_(a1, [2])]).max() < 1e-8
    assert abs(m.K[2]) < 1e-7
    assert np.abs(m.J[np.ix_(a1, a1)] - np.diag(np.diag(
        m.J[np.ix_(a1, a1)]))).max() > 1e-2
    # the two minima differ by 0.04 A and 11 deg: axis switching leaves J
    # orthogonal to 5e-3 only (warned above DUSCHINSKY_ORTHOGONALITY_TOL)
    assert 1e-4 < m.orthogonality < dm.DUSCHINSKY_ORTHOGONALITY_TOL


def write_orca_hessian(path, hess, columns=5):
    """An ORCA .hess file (ORCA's layout: 5-column blocks, $atoms in Bohr)."""
    from src.properties.optimize import cartesian_hessian
    h = cartesian_hessian(hess.hessian, hess.mol.natm)
    n = h.shape[0]
    lines = ['', '$orca_hessian_file', '', '$act_atom', '  0', '',
             '$hessian', str(n)]
    for c0 in range(0, n, columns):
        cols = range(c0, min(n, c0 + columns))
        lines.append(' ' * 12 + ''.join(f'{c:>19d}' for c in cols))
        for r in range(n):
            lines.append(f'{r:>6d}    ' + ''.join(f'{h[r, c]:19.10E}'
                                                  for c in cols))
    lines += ['', '$atoms', str(hess.mol.natm)]
    mass = hess.mol.atom_mass_list(isotope_avg=True)
    for i, (x, y, z) in enumerate(hess.mol.atom_coords()):
        lines.append(f' {hess.mol.atom_symbol(i):<3s}{mass[i]:12.5f}'
                     f'{x:15.6f}{y:15.6f}{z:15.6f}')
    lines += ['', '$end']
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def test_orca_hessian_is_read_as_data(water_hessians, tmp_path):
    e = turned(water_hessians['6-31g'])
    path = tmp_path / 'water.hess'
    write_orca_hessian(path, e)
    read = read_orca_hessian(path, water_hessians['sto-3g'].mol,
                             source='file')
    from src.properties.optimize import cartesian_hessian
    assert np.abs(read.hessian - cartesian_hessian(e.hessian, 3)).max() < 1e-9
    assert np.abs(read.mol.atom_coords() - e.mol.atom_coords()).max() < 1e-6
    a = duschinsky_model(water_hessians['sto-3g'], water_hessians['6-31g'])
    b = duschinsky_model(water_hessians['sto-3g'], read)
    assert np.abs(b.omega_e - a.omega_e).max() < 1e-9
    assert np.abs(np.abs(b.K) - np.abs(a.K)).max() < 1e-5
    from pyscf import gto
    other = gto.M(atom='S 0 0 0; H 0 1 0; H 0 0 1', basis='sto-3g',
                  verbose=0)
    with pytest.raises(ValueError, match='template'):
        read_orca_hessian(path, other)


# ------------------------------------------------------------------ refusals
def test_refusals():
    with pytest.raises(ValueError, match='broadening'):
        duschinsky_density(TWO_MODE, 0.0, 300.0, 'emission')
    with pytest.raises(ValueError, match='direction'):
        duschinsky_density(TWO_MODE, 0.0, 300.0, 'fluorescence',
                           broadening=1e-3)
    with pytest.raises(ValueError, match='orthogonal'):
        DuschinskyModel(W_G, W_E, 1.1 * np.eye(2), TWO_MODE.K)
    with pytest.raises(ValueError, match='VIBRONIC_MODE_FLOOR_CM'):
        DuschinskyModel(W_G, -W_E, np.eye(2), TWO_MODE.K)
    with pytest.raises(ValueError, match='temperature'):
        duschinsky_density(TWO_MODE, 0.0, 0.0, 'emission', broadening=1e-3)
    with pytest.raises(TypeError, match='Hessian'):
        duschinsky_model(object(), object())


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
