"""The RPA downfolding of an unrestricted (UHF/UKS) reference on the
imagfrequency route against the casida route.

The mean fields are density-fitted UHF calculations, so the three-index
factors are the mean field's own fit, a real one with a fitting error: the
OH radical (93 auxiliary functions) and triplet CH2 (116) in 6-31G. casida
with df=True runs on the same factors, so the two routes solve the same
problem and differ only by what the route adds: the frequency quadrature of
the correlation energies and the continuation of Sigma^E. The spaces have
different, non-contiguous orbitals per spin, N^A_alpha != N^A_beta, unequal
cores, a channel without an active pair, and no active pair at all.

mode='imagfrequency' keeps the static polarizability P(0) = chi0_alpha(0) +
chi0_beta(0) of the whole system in the fitting basis, takes E_c from the
two-spin imaginary-axis energy and keeps its frequency grid, which covers
the transitions of both spins.
"""
from functools import cache

import numpy as np
import pytest

from src.MultiReference.Downfolding import run_low_level
from src.SingleReference.common.active_space import ActiveSpace
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver
from src.SingleReference.LinearResponse.rpa_energy import (
    imaginary_axis_energy_grid, rpa_correlation_energy_imaginary_axis)
from tests.downfolding_oracle import (CH2, OH, drpa_so,
                                      spin_orbital_integrals, unrestricted)

REFS = ('oh', 'ch2')
MOLECULES = {'oh': (OH, 1), 'ch2': (CH2, 2)}
# (occupied, virtual) active orbitals of alpha, then of beta. OH has 5 alpha
# and 4 beta occupied orbitals of 11, CH2 5 and 3 of 13.
SPACES = {
    'oh': {
        'asymmetric': (([2, 4], [5, 7]), ([3], [4, 6, 7])),
        'unequal_cores': (([3, 4], [5, 6]), ([2, 3], [4, 5])),
        'no_beta_pair': (([3, 4], [5]), ([], [4, 5, 6])),
        'occupied_only': (([3, 4], []), ([2, 3], [])),
    },
    'ch2': {
        'asymmetric': (([2, 4], [5, 7]), ([2], [3, 5, 6])),
        'unequal_cores': (([3, 4], [5, 6]), ([1, 2], [3, 4])),
        'no_beta_pair': (([3, 4], [5]), ([], [3, 4, 5])),
        'occupied_only': (([3, 4], []), ([1, 2], [])),
    },
}


@cache
def _mean_field(ref):
    """A density-fitted UHF of the OH radical or of triplet CH2 in 6-31G."""
    atom, spin = MOLECULES[ref]
    return unrestricted(atom, spin, density_fit=True)


@cache
def _low(ref, mode):
    """The RPA low-level result of `ref` on `mode`; casida with df=True runs
    on the mean field's own fit, the three-index factors of imagfrequency."""
    return run_low_level(_mean_field(ref), method='rpa', mode=mode)


@pytest.fixture(scope='module', autouse=True)
def _release_cached_mean_fields():
    """Empty the caches when the module is done (a mean field held at
    interpreter shutdown leaves pyscf's temporary chkfile behind)."""
    yield
    for cached in (_low, _mean_field):
        cached.cache_clear()


def _spaces(ref, which):
    """The pair (alpha, beta) of ActiveSpace objects of SPACES[ref][which]."""
    mf = _mean_field(ref)
    nmo = mf.mo_coeff.shape[-1]
    nocc = [int(np.count_nonzero(occ)) for occ in mf.mo_occ]
    return tuple(ActiveSpace.from_indices(nmo, n, occ, virt)
                 for n, (occ, virt) in zip(nocc, SPACES[ref][which]))


@pytest.mark.parametrize('ref', REFS)
def test_full_rpa_is_the_imaginary_axis_energy(ref):
    """E_c of the low-level result is the two-spin imaginary-axis energy on
    the mean field's factors, and equals the casida route's plasmon formula
    on the same factors to the quadrature error. Kept for the active spaces:
    the energy's own frequency grid, over the transitions of both spins, and
    P(0), the sum of the two spins' static responses,
    -2 sum_ia B_ia B_ia^T / delta_ia each. Measured: E_c against casida
    4.8e-13 (OH) and 3.6e-13 (CH2) on the 20-point grid, P(0) 6.9e-16."""
    low, cas = _low(ref, 'imagfrequency'), _low(ref, 'casida')
    assert low.unrestricted and low.mode == 'imagfrequency'
    assert low.omega is None and low.xpy is None and low.eri is None
    assert len(low.factors) == 2
    assert all(np.array_equal(x, y) for x, y in zip(low.factors, cas.factors))
    lr = LinearResponseSolver(low.eps, coeff_df=low.factors,
                              spin_mode='unrestricted')
    assert low.e_c_full == rpa_correlation_energy_imaginary_axis(lr, low.nocc)
    assert abs(low.e_c_full - cas.e_c_full) < 1e-9
    points, weights = imaginary_axis_energy_grid(low.eps, low.nocc)
    assert np.array_equal(low.cache['grid'][0], points)
    assert np.array_equal(low.cache['grid'][1], weights)
    chi0 = 0.0
    for b, eps, nocc in zip(low.factors, low.eps, low.nocc):
        b_ov = b[:, :nocc, nocc:].reshape(b.shape[0], -1)
        delta = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()
        chi0 = chi0 - 2.0 * (b_ov / delta) @ b_ov.T
    assert low.cache['p0'].shape == (low.factors[0].shape[0],) * 2
    assert np.abs(low.cache['p0'] - chi0).max() < 1e-12


def test_numerical_routes_refuse_a_spin_without_occupied_or_virtual_orbitals():
    """Triplet H2 in 6-31G has both electrons in alpha orbitals: the beta
    channel has no occupied orbital, so no transition and no mid-gap. casida
    takes it, the RPA then holding the alpha pairs alone (E_c against the
    oracle's spin-orbital RPA on exact integrals, measured 4.4e-16 on
    E_c = -0.0081); the numerical routes refuse it, naming the spin and
    casida."""
    mf = unrestricted('H 0 0 0; H 0 0 0.74', 2)
    assert [int(np.count_nonzero(occ)) for occ in mf.mo_occ] == [2, 0]
    low = run_low_level(mf, method='rpa', df=False)
    assert low.nocc == (2, 0) and len(low.omega) == 4
    nmo = mf.mo_coeff.shape[-1]
    _, eri_so = spin_orbital_integrals(mf.mol, *mf.mo_coeff)
    e_c = drpa_so(eri_so, np.concatenate(mf.mo_energy), [0, 1])[0]
    assert e_c < -1e-3 and abs(low.e_c_full - e_c) < 1e-10
    for mode in ('imagfrequency', 'space-time'):
        with pytest.raises(NotImplementedError,
                           match="beta channel has 0 occupied of 4.*"
                                 "mode='casida'"):
            run_low_level(mf, method='rpa', mode=mode)
