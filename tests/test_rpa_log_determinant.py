"""The dRPA correlation energy warns and flags a dielectric whose determinant is not positive.

E_c = (1/2pi) int dw Tr{log(1 - chi0(iw)) + chi0(iw)} needs det(1 - chi0) > 0,
which a negative semidefinite chi0 guarantees at every imaginary frequency.
`np.linalg.slogdet` returns (sign, log|det|), and both molecular routes used
to drop the sign, so a chi0 with one eigenvalue above 1 at one frequency gave
a finite number for the energy of another functional, silently. Now one helper
(`space_time.dielectric_log_determinant`, which the periodic route uses too) warns
with `NonPositiveDielectric`, naming the frequency, and records it under the
'nonpositive_det' flag.

Both checks plant chi0 in place of the screening build: negative definite on
every frequency but one, which carries a single eigenvalue at 1.5, so the
determinant there is negative. The gas-phase energies themselves are pinned
bitwise in tests/test_refactor_baseline.py (e0_terms) and to the Casida route
in tests/test_rpa_correlation_energy.py.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from src.Base.numerics.numerics import Numerics
from src.SingleReference.LinearResponse import rpa_energy, space_time

NAUX, NFREQ, BAD = 6, 8, 3


def planted_chi0(nfreq):
    """(nfreq, NAUX, NAUX) symmetric chi0, negative definite except at BAD."""
    q = np.linalg.qr(np.random.default_rng(0).standard_normal((NAUX, NAUX)))[0]
    out = np.empty((nfreq, NAUX, NAUX))
    for k in range(nfreq):
        lam = -np.linspace(0.05, 0.5, NAUX)
        if k == BAD:
            lam[0] = 1.5
        out[k] = (q * lam) @ q.T
    return out


def test_imaginary_axis_route_warns_at_the_planted_frequency(monkeypatch):
    monkeypatch.setattr(rpa_energy, 'solve_polarizability_imaginary_axis',
                        lambda lr, nocc, freq: planted_chi0(len(freq)))
    lr = SimpleNamespace(eps=np.array([-1.0, -0.5, 0.3, 1.0]))
    with pytest.warns(space_time.NonPositiveDielectric, match=f'not positive at {BAD} '):
        rpa_energy.rpa_correlation_energy_imaginary_axis(
            lr, 2, numerics=Numerics(quadrature={'nfreq_rpa': NFREQ}))


def test_space_time_route_warns_at_the_planted_frequency(monkeypatch):
    chi0 = planted_chi0(NFREQ)
    monkeypatch.setattr(space_time, 'polarizability_projected_sweep',
                        lambda *a, **k: np.zeros((2, NAUX, NAUX)))
    monkeypatch.setattr(space_time, 'owned_frequency_blocks',
                        lambda *a, **k: iter([(list(range(NFREQ)), chi0)]))
    grid = SimpleNamespace(ntau=2, nfreq=NFREQ, tau_points=np.ones(2),
                           cosft_wt=np.zeros((NFREQ, 2)),
                           omega_weights=np.ones(NFREQ))
    with pytest.warns(space_time.NonPositiveDielectric, match=f'not positive at {BAD} '):
        space_time.rpa_correlation_energy_space_time(
            None, np.zeros((1, NAUX)), np.array([-1.0, 1.0]), 1, grid)


def test_a_positive_determinant_is_the_plain_log_determinant():
    """Where it exists the check is invisible: slogdet's own log|det|, no flag."""
    chi0 = planted_chi0(BAD)                  # every frequency before BAD
    for k, c in enumerate(chi0):
        sign, logdet = np.linalg.slogdet(np.eye(NAUX) - c)
        assert sign > 0
        flags = {}
        assert space_time.dielectric_log_determinant(np.eye(NAUX) - c, k, flags) == logdet
        assert flags == {}


def test_a_nonpositive_determinant_is_flagged_by_where():
    c = planted_chi0(NFREQ)[BAD]
    flags = {}
    with pytest.warns(space_time.NonPositiveDielectric):
        out = space_time.dielectric_log_determinant(np.eye(NAUX) - c, BAD, flags)
    assert out == np.linalg.slogdet(np.eye(NAUX) - c)[1]
    assert flags == {'nonpositive_det': [BAD]}
