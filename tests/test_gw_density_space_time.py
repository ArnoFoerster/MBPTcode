"""The GW density on the space-time route (mode='space-time', N^3).

- `correlation_self_energy_matrix_space_time`: the full MO Sigma_c,pq(mu + i w)
  of the space-time QP route on any query frequencies; its diagonal at the
  Pade nodes is the QP route's `_sigma_mo_diagonal` on the same factors.
- No ISDF grid (a bare Numerics()): the exception `separable_factors` raises.

Water/cc-pVDZ, HF, DF on cc-pvdz-ri (the factors' auxiliary basis too);
default quadratures (W grid minimax 20, outer Gauss-Legendre 48, the
space-time route's 'auto' time grid).

The density itself, space-time against casida per block of Delta gamma_MO
and on the natural occupations, per relaxation, is in test_gw_density_none,
test_gw_density_dyson and test_gw_density_zvector.

Recorded when that comparison lived here (space-time against imagfrequency,
water): Sigma_c,pq(mu + i w) on the outer nodes against the imagfrequency
route's: max |diff| 2-3e-6 (max |Sigma| 8e-2) on both G1 and GH, the finest
ISDF level for cc-pVDZ. GH is no closer than G1, so the floor is not the
ISDF grid's density; the time grid does not move it either (HF 'dyson' on GH unchanged to 3 digits from 16 to 26 tau points,
tau_target 1e-8 .. 1e-14).
"""
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.factorization.sliced_factors import unpack_factors
from src.Base.numerics.numerics import Numerics
from src.Base.orbital_spaces import particle_hole_window
from src.Base.pyscf.pyscf_interface import get_orbital_energies
from src.Base.utils.grids.analytic_continuation import pade_nodes
from src.SingleReference.GW import space_time
from src.SingleReference.GW.density_matrix import compute_gw_density_matrix
from src.SingleReference.GW.space_time import (
    correlation_self_energy_matrix_space_time, separable_factors)
from src.SingleReference.LinearResponse.space_time import chi0_imaginary_frequency
from tests.isdf_test_grid import g1_numerics
from tests.molecules import WATER

AUXBASIS = 'cc-pvdz-ri'

def water(basis='cc-pvdz'):
    return gto.M(atom=WATER, basis=basis, verbose=0)


def fitted(mf):
    mf = mf.density_fit(auxbasis=AUXBASIS)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 200
    return mf.run()


def report(name, ok, detail=''):
    print(f"{name}: {'OK' if ok else 'FAIL'} {detail}")
    return ok


def check_matrix_diagonal_matches_qp(mf):
    """The diagonal of the full Sigma_c,pq at the Pade nodes is the QP
    route's `_sigma_mo_diagonal` for every state on the same factors."""
    mol = mf.mol
    nocc = mol.nelectron // 2
    numerics = g1_numerics(mf)
    pade = pade_nodes(numerics.quadrature)
    sigma, mu = correlation_self_energy_matrix_space_time(mf, mol, nocc, pade, numerics)

    eps = get_orbital_energies(mf, representation='spatial')
    q = numerics.quadrature
    axes = space_time._space_time_grids(
        [eps], [nocc], [eps], dict(ntau=q.ntau_st, nfreq=q.nfreq_st,
                                   pade_freq=pade, tau_target=q.tau_target),
        q.w0_gl, None)
    X_mo, D, X_ao, coords = unpack_factors(separable_factors(mf, mol, numerics=numerics))
    W = chi0_imaginary_frequency(X_mo, D, eps, nocc, axes['grid'], mu=mu)
    space_time._dyson_in_place(W, range(W.shape[0]))
    states = np.arange(len(eps))
    qp = space_time._sigma_mo_diagonal(
        X_mo, D, W, mf, eps, nocc, axes['tau_points'], axes['freq_points'],
        pade, mu, states, X_ao=X_ao, coords=coords)
    diag = np.einsum('wpp->pw', sigma)
    err = np.abs(diag - qp).max()
    ok = (sigma.shape == (len(pade), len(eps), len(eps))
          and mu == particle_hole_window(eps, nocc)[2] and err < 1e-10)
    return report('Sigma_c,pq diagonal at the Pade nodes == _sigma_mo_diagonal',
                  ok, f'(max |diff| {err:.2e}, max |Sigma| {np.abs(qp).max():.2e})')


def check_space_time_needs_an_isdf_grid(mf):
    """A Numerics() naming no ISDF grid: the exception separable_factors
    raises, with its message."""
    try:
        separable_factors(mf, mf.mol, numerics=Numerics())
    except Exception as err:  # noqa: BLE001 -- whatever the factors raise
        expected = err
    else:
        return report('separable_factors refuses Numerics() without a grid', False)
    try:
        compute_gw_density_matrix(mf, mode='space-time', relaxation='none',
                                  numerics=Numerics())
    except Exception as err:  # noqa: BLE001 -- compared below
        ok = type(err) is type(expected) and str(err) == str(expected)
        return report('space-time without an ISDF grid raises as separable_factors',
                      ok, f'({type(err).__name__}: {err})')
    return report('space-time without an ISDF grid raises as separable_factors', False,
                  '(nothing raised)')


def check_open_shell_refused():
    """An ROHF reference (OH radical): mode='space-time' raises the named
    closed-shell error mode='imagfrequency' raises."""
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='cc-pvdz', spin=1, verbose=0)
    rohf = scf.ROHF(mol).density_fit(auxbasis=AUXBASIS).run()

    def raised(mode, numerics):
        try:
            compute_gw_density_matrix(rohf, mode=mode, relaxation='none',
                                      numerics=numerics)
        except Exception as err:  # noqa: BLE001 -- compared below
            return err
        return None
    imag = raised('imagfrequency', Numerics())
    st = raised('space-time', g1_numerics(rohf))
    ok = (isinstance(imag, NotImplementedError) and type(st) is type(imag)
          and 'closed-shell' in str(st))
    return report('ROHF refused on space-time as on imagfrequency', ok,
                  f'({type(st).__name__}: {st})')


def check_smeared_refused(hf):
    """The matrix Sigma itself refuses a thermally smeared RHF, before any
    ISDF work (a bare Numerics() has no grid, so reaching it would raise the
    other error)."""
    name = 'smeared RHF refused by the space-time matrix Sigma'
    smeared = copy.copy(hf)
    smeared.sigma, smeared.smearing_method = 0.01, 'fermi'
    try:
        correlation_self_energy_matrix_space_time(
            smeared, smeared.mol, 5, np.array([0.5]), Numerics())
    except NotImplementedError as err:
        return report(name, 'T = 0' in str(err), f'({err})')
    except Exception as err:  # noqa: BLE001 -- wrong type
        return report(name, False, f'({type(err).__name__}: {err})')
    return report(name, False)


def run():
    hf = fitted(scf.RHF(water()))
    results = [check_matrix_diagonal_matches_qp(hf),
               check_space_time_needs_an_isdf_grid(hf),
               check_open_shell_refused(),
               check_smeared_refused(hf)]
    return all(results)


def test_gw_density_space_time():
    assert run()


if __name__ == '__main__':
    ok = run()
    print("\nALL PASSED" if ok else "\nFAILURES DETECTED")
    sys.exit(0 if ok else 1)
