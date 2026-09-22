"""`eta` reaches every route that has one, and is refused by the routes that do not.

`eta` is a NAMED argument of `calc_qp_energy`, so `**route_kwargs` cannot carry
it: whatever the dispatcher does not pass on by hand is dropped without a word.
It was dropped for both imaginary-axis modes, and a caller writing
`calc_qp_energy(mf, eta=1e-7, mode='imagfrequency')` got a number computed at
the default broadening under the name of its own.

The fix is a refusal rather than a forward, because neither imaginary-axis route
has a broadening to honour. chi0(i.omega) has the real denominator
-2d/(d^2 + w^2), chi0(i.tau) is a separable factorization with no denominator at
all, and Sigma_c reaches the real axis by Pade rather than at w + i.eta.
`solve_qp_energy_imaginary_axis` used to take an `eta` and hand it to the
solver, but only the real-axis branch of `_f_rpa` reads it, so forwarding would
have looked like a fix while changing nothing. That parameter is gone: the
signature no longer advertises a broadening the route does not have, which
leaves the refusal below as the only spelling a caller can reach.

Checks:
  1. eta is INERT on the imaginary-frequency route, measured at the root cause:
     W(i.omega) from two solvers whose eta differs by four orders of magnitude
     is bitwise identical, while the real-axis branch of the same solver moves.
     This is what rules out forwarding, and it is measured rather than asserted
     from reading the denominator.
  2. The route itself no longer accepts an eta to be inert to. The guard in
     `calc_qp_energy` only covers callers coming through the dispatcher, and
     `solve_qp_energy_imaginary_axis` is exported and called directly.
  3. eta is LIVE on the Casida route: the same two values move the energy. A
     refusal is only meaningful if the parameter does something somewhere.
  4. Both imaginary-axis modes raise on a non-default eta, under every spelling
     of the mode, and name eta in the message.
  5. The default is untouched: omitting eta, and passing the default
     explicitly, both still run and agree bitwise on both routes.
  6. The evGW branch forwards eta too -- it is the other named argument the
     same dispatcher used to drop.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf, df

from src.Base.constants import DEFAULT_BROADENING_ETA
from src.Base.pyscf_interface import get_orbital_energies, get_df_coefficients_ov
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.imaginary_axis import solve_qp_energy_imaginary_axis
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

# Far enough from DEFAULT_BROADENING_ETA that no route could reach the same
# number twice by rounding: 1e-7 is the tight value the acene timings ask for,
# 5.0 Ha is unphysical on purpose, so a route that ignores it says so loudly.
TIGHT_ETA = 1e-7
ABSURD_ETA = 5.0

NFREQ = 20


def check(ok, label, detail=''):
    """Report a condition AND enforce it.

    pytest DISCARDS whatever a test function returns, so a suite built out of
    `return ok` passes whether ok is True or False. Asserting here fixes every
    call site at once.
    """
    print(f"[{'OK  ' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    assert ok, f"{label}{(' -- ' + detail) if detail else ''}"
    return ok


@pytest.fixture(scope='module')
def hf_mf():
    """HF/6-31g, small enough that every check below is a few seconds."""
    mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.run()
    return mf


def test_eta_is_inert_on_the_imaginary_frequency_route(hf_mf):
    """Why the fix is a refusal: the screening this route builds has no eta in it.

    Measured on W itself rather than on the quasiparticle energy, because that
    is where the inertness comes from: the whole eta dependence of the route
    was `LinearResponseSolver.eta`, read only by the real-axis branch of
    `_f_rpa`.
    """
    mol = hf_mf.mol
    nocc = mol.nelectron // 2
    eps = get_orbital_energies(hf_mf, representation='spatial')
    occ, virt = get_occ_virt_indices(eps, nocc)
    C_ov, _ = get_df_coefficients_ov(mol, hf_mf, occ, virt, rows=[nocc - 1])

    # Includes w = 0, where the real-axis branch takes its own limit, so a
    # forwarded eta could not hide in the endpoint either.
    grid = np.array([0.0, 0.3, 1.7])
    solvers = [LinearResponseSolver(eps, coeff_ov=C_ov, spin_mode='restricted',
                                    eta=e)
               for e in (DEFAULT_BROADENING_ETA, ABSURD_ETA)]

    W_imag = [lr.solve_rpa_screening(grid, nocc, is_imaginary=True)
              for lr in solvers]
    check(np.array_equal(W_imag[0], W_imag[1]),
          'W(i.omega) is bitwise identical across four orders of eta',
          f'max|diff| = {np.max(np.abs(W_imag[0] - W_imag[1])):.3e}')

    # The same solvers on the real axis, so the check above is a statement about
    # the imaginary branch and not about a solver that ignores eta everywhere.
    W_real = [lr.solve_rpa_screening(grid, nocc, is_imaginary=False)
              for lr in solvers]
    check(np.max(np.abs(W_real[0] - W_real[1])) > 1e-6,
          'while the real-axis branch of the same solver moves with eta',
          f'max|diff| = {np.max(np.abs(W_real[0] - W_real[1])):.3e}')


def test_the_imaginary_axis_route_takes_no_eta(hf_mf):
    """The root cause, not just the guard: the signature stops offering one.

    `calc_qp_energy`'s refusal only covers callers that come through the
    dispatcher. `solve_qp_energy_imaginary_axis` is exported from
    `src.SingleReference` and called directly by tools and tests, which the
    guard never sees.
    """
    params = inspect.signature(solve_qp_energy_imaginary_axis).parameters
    check('eta' not in params,
          'solve_qp_energy_imaginary_axis advertises no broadening',
          f'signature is ({", ".join(params)})')

    mol = hf_mf.mol
    nocc = mol.nelectron // 2
    with pytest.raises(TypeError, match='eta'):
        solve_qp_energy_imaginary_axis(hf_mf, mol, nocc, nocc - 1,
                                       nfreq=NFREQ, eta=ABSURD_ETA)


def test_eta_is_live_on_the_casida_route(hf_mf):
    """The refusal means something only because eta acts on the other route."""
    at_default = calc_qp_energy(hf_mf, mode='casida',
                                eta=DEFAULT_BROADENING_ETA)
    at_absurd = calc_qp_energy(hf_mf, mode='casida', eta=ABSURD_ETA)

    check(at_default != at_absurd,
          'casida broadening reaches the self-energy',
          f'{at_default:.6f} eV vs {at_absurd:.6f} eV')


@pytest.mark.parametrize('mode', ['imagfrequency', 'imag-frequency',
                                  'imag_frequency', 'space-time', 'space_time'])
def test_non_default_eta_is_refused(hf_mf, mode):
    """The drift itself: this used to return a number computed at the default."""
    with pytest.raises(ValueError, match='eta'):
        calc_qp_energy(hf_mf, mode=mode, eta=TIGHT_ETA)


def test_the_default_still_runs_and_is_unchanged(hf_mf):
    """A guard that refuses the default would break every existing caller."""
    for mode in ('imagfrequency', 'casida'):
        implicit = calc_qp_energy(hf_mf, mode=mode)
        explicit = calc_qp_energy(hf_mf, mode=mode, eta=DEFAULT_BROADENING_ETA)
        check(implicit == explicit,
              f"mode={mode!r} takes the default eta either way",
              f'{implicit!r} vs {explicit!r}')


def test_evgw_forwards_eta(monkeypatch):
    """The same dispatcher dropped eta on the way into the evGW loop."""
    seen = {}

    def fake_loop(mf, mol=None, mode='space-time', **route_kw):
        seen.update(route_kw)
        raise RuntimeError('stop: the keywords are all this check needs')

    import src.SingleReference.GW.evGW as evgw
    monkeypatch.setattr(evgw, 'evgw_eigenvalues', fake_loop)

    mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.run()

    with pytest.raises(RuntimeError):
        calc_qp_energy(mf, mode='casida', self_consistency='evGW',
                       eta=TIGHT_ETA)

    check(seen.get('eta') == TIGHT_ETA,
          'evGW receives the caller eta',
          f'route keywords were {seen}')


if __name__ == '__main__':
    mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.run()

    print('\n-- 1. eta is inert on the imaginary-frequency route')
    test_eta_is_inert_on_the_imaginary_frequency_route(mf)
    test_the_imaginary_axis_route_takes_no_eta(mf)

    print('\n-- 2. eta is live on the Casida route')
    test_eta_is_live_on_the_casida_route(mf)

    print('\n-- 3. a non-default eta is refused by both imaginary-axis modes')
    for mode in ('imagfrequency', 'imag-frequency', 'imag_frequency',
                 'space-time', 'space_time'):
        try:
            calc_qp_energy(mf, mode=mode, eta=TIGHT_ETA)
        except ValueError as exc:
            check('eta' in str(exc), f'mode={mode!r} refuses eta={TIGHT_ETA}')
        else:
            check(False, f'mode={mode!r} ACCEPTED eta={TIGHT_ETA}')

    print('\n-- 4. the default is untouched')
    test_the_default_still_runs_and_is_unchanged(mf)

    print('\nALL PASSED')
