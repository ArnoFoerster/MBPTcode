"""The UISC rate model hands its decay rates the downhill gap.

The rates in `properties.rates` take Delta_E = E_final - E_initial, the
reaction energy's sign: `fc_weighted_dos` is the Fourier transform of a
generating function whose zero-temperature lines sit at dE = -n omega -
lambda, so a step that RELEASES energy has dE < 0. The UISC driver once passed
E(S1) - E(S0) to the S1 -> S0 internal conversion and E(T1) - E(S0) to the
T1 -> S0 decay, both positive, which reads the uphill (absorption) side of
the density of states: on a toy spectrum k_IC came out 14 s^-1 where the decay
is 7e14 s^-1. The decays are now `rates.rate_table`'s, the table the TADF
model reads too. These tests drive it, directly and through
`uisc.uisc_rate_table`, on a toy spectrum and require the downhill rate,
which a positive gap misses by many orders of magnitude. S0 at each excited
minimum stands the toy displacement's sum rule above E_0, so the decays pass
their guard (`rates.fc_lambda_guard`).
"""
import numpy as np
import pytest
from pyscf import gto

from src.Base.constants import (ATOMIC_TIME_SECONDS, HARTREE_TO_EV,
                                TRIPLET_SUBLEVELS)
from src.Base.geometry import molecule_at
from src.photophysics import uisc
from src.photophysics.declaration import UiscModel
from src.photophysics.minimum import GroundModes, StateMinimum
from src.properties.rates import (Minimum, fc_weighted_dos,
                                  internal_conversion_rate, rate_table)
from src.properties.curvilinear import huang_rhys_curvilinear
from src.properties.vibronic import align_to, project_coupling
from tests.photophysics_cases import FORMALDEHYDE_TOY, internal_modes_of

#: The toy spectrum: Huang-Rhys factors and frequencies (Hartree), every
#: mode a vibration (an imaginary one is refused), and the matrix's
#: reorganization.
S_K = np.array([0.8, 0.5, 0.1])
OMEGA = np.array([0.006, 0.0075, 0.002])
LAMBDA_ENV = 0.05 / HARTREE_TO_EV
T = 300.0
#: Hartree: E(S0) at its minimum and the two states above it.
E_0, E_S1, E_T1 = -100.0, -99.97, -99.95
#: Hartree: the toy displacement's sum rule sum_k S_k omega_k, S0's relaxation.
SUM_RULE = float((S_K * OMEGA).sum())
MODEL = UiscModel(n_modes=2, dq=0.1, s_block=1, t_block=1,
                  lambda_samanta=(0.10, 0.20))


def table(d_k=None, v_t0=0.0):
    """`rate_table` with the toy spectrum as every displacement, S1 and T1
    at E_S1 and E_T1, and <S0|H_SO|T1> = v_t0."""
    socme = np.zeros((2, 1))
    socme[0, 0] = v_t0
    s1 = Minimum('S1', 'T1', {'S1': E_S1, 'T1': E_T1 + 0.01}, socme,
                 e_ground=E_0 + SUM_RULE)
    t1 = Minimum('S1', 'T1', {'S1': E_S1 + 0.01, 'T1': E_T1}, socme,
                 e_ground=E_0 + SUM_RULE)
    return rate_table(s1, t1, E_0, OMEGA, S_K, T, LAMBDA_ENV, s_ic=S_K,
                      d_ic=d_k, s_t=S_K)


def rho(delta_e, s=S_K):
    return fc_weighted_dos(delta_e, s, OMEGA, T, lambda_classical=LAMBDA_ENV)


def test_internal_conversion_is_the_downhill_rate():
    d_k = np.array([0.1, 0.2, 0.05])
    k_ic = table(d_k=d_k)['k_ic']
    down, per_mode = internal_conversion_rate(E_0 - E_S1, d_k, OMEGA, rho, T)
    up = internal_conversion_rate(E_S1 - E_0, d_k, OMEGA, rho, T)[0]
    assert down > 1e10 * up
    assert k_ic == pytest.approx(down, rel=1e-12)
    assert (per_mode > 0).all()


def test_triplet_decay_is_the_downhill_rate():
    v_t0 = 1e-6
    k_nr = table(v_t0=v_t0)['k_nr_t1']
    # out of a triplet: the average over its three sublevels
    v2 = v_t0 ** 2 / TRIPLET_SUBLEVELS
    down = 2.0 * np.pi * v2 * rho(E_0 - E_T1) / ATOMIC_TIME_SECONDS
    up = 2.0 * np.pi * v2 * rho(E_T1 - E_0) / ATOMIC_TIME_SECONDS
    assert down > 1e10 * up
    assert k_nr == pytest.approx(down, rel=1e-12)


# ------------------------------------------------ the same through UISC
#: six internal modes of formaldehyde: the toy spectrum's three and three
#: more, one of them undisplaced
OMEGA_6 = np.array([0.006, 0.0075, 0.002, 0.001, 0.004, 0.012])
S_6 = np.array([0.8, 0.5, 0.1, 0.0, 0.3, 0.1])


def uisc_table(v_t0, d_ground):
    """The UISC table of a formaldehyde-shaped toy: S1 and T1 minima displaced
    along the internal modes by the toy Huang-Rhys factors, S1 and T1 at the
    toy energies, <S0|H_SO|T1> = v_t0."""
    template = gto.M(atom=FORMALDEHYDE_TOY, basis='sto-3g', verbose=0)
    modes_, masses = internal_modes_of(template)
    modes = GroundModes(OMEGA_6, modes_, masses, 'df')
    w = np.repeat(masses, 3) ** -0.5
    a = np.sqrt(2.0 * S_6 / OMEGA_6)
    x0 = template.atom_coords()

    def at(scale):
        return x0 + (w * (modes_ @ (scale * a))).reshape(-1, 3)

    socme = np.zeros((2, 1))
    socme[0, 0] = v_t0
    socme[1, 0] = 1e-6

    def minimum(where, coords, energy, omega, scale=0.0):
        # S0 there: the sum rule of a displacement `scale` times the toy's
        e0 = E_0 + scale ** 2 * float((S_6 * OMEGA_6).sum())
        return StateMinimum(
            method='gw-bse', where=where, coords_bohr=coords, e0=e0,
            energy=energy, omega=omega, emitter='S1', reservoir='T1',
            socme=socme, soc_dv_cart=None, soc_dv_paths=None,
            dipole=np.array([[0.5, 0.0, 0.1]]), f_osc=np.array([0.1]),
            d_ground=d_ground, solvation='gas')

    vert = minimum('R0', x0, {'S1': E_S1 + 0.01, 'T1': E_T1 + 0.01},
                   {'S1': 0.04, 'T1': 0.035})
    s1 = minimum('S1', at(1.0), {'S1': E_S1, 'T1': E_T1 + 0.01},
                 {'S1': 0.03, 'T1': 0.035}, 1.0)
    t1 = minimum('T1', at(0.7), {'S1': E_S1 + 0.01, 'T1': E_T1},
                 {'S1': 0.035, 'T1': 0.025}, 0.7)
    out = uisc.uisc_rate_table(modes, vert, s1, t1, None, template, T,
                               LAMBDA_ENV * HARTREE_TO_EV, MODEL, 'stacked')
    mol0 = molecule_at(template, x0)
    s1_geom, rot = align_to(mol0, molecule_at(template, s1.coords_bohr),
                            return_rotation=True)
    t1_geom = align_to(mol0, molecule_at(template, t1.coords_bohr))
    # the table's factors: the displacements in valence internals at R0
    s_ic = huang_rhys_curvilinear(mol0, mol0, s1_geom, OMEGA_6, modes_,
                                  masses)[0]
    s_t = huang_rhys_curvilinear(mol0, mol0, t1_geom, OMEGA_6, modes_,
                                 masses)[0]
    d_k = project_coupling(d_ground @ rot, modes_, masses)
    return out, s_ic, s_t, d_k


def test_uisc_internal_conversion_is_the_downhill_rate():
    d_ground = 0.1 * np.random.default_rng(5).standard_normal((4, 3))
    got, s_ic, _, d_k = uisc_table(1e-6, d_ground)
    down, _ = internal_conversion_rate(E_0 - E_S1, d_k, OMEGA_6,
                                       lambda e: rho6(e, s_ic), T)
    up = internal_conversion_rate(E_S1 - E_0, d_k, OMEGA_6,
                                  lambda e: rho6(e, s_ic), T)[0]
    assert down > 1e10 * up > 0.0
    assert got['k_ic'] == pytest.approx(down, rel=1e-12)


def test_uisc_triplet_decay_is_the_downhill_rate():
    v_t0 = 1e-6
    got, _, s_t, _ = uisc_table(v_t0, 0.1 * np.ones((4, 3)))
    v2 = v_t0 ** 2 / TRIPLET_SUBLEVELS
    down = 2.0 * np.pi * v2 * rho6(E_0 - E_T1, s_t) / ATOMIC_TIME_SECONDS
    up = 2.0 * np.pi * v2 * rho6(E_T1 - E_0, s_t) / ATOMIC_TIME_SECONDS
    assert down > 1e10 * up
    assert got['k_nr_t1'] == pytest.approx(down, rel=1e-12)


def rho6(delta_e, s):
    return fc_weighted_dos(delta_e, s, OMEGA_6, T, lambda_classical=LAMBDA_ENV)
