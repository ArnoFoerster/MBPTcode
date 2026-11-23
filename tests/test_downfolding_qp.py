"""The quasiparticle spectrum of the RPA low-level run (qp=True) and the
static environment self-energy at those energies (sigma_energies='qp').

`run_low_level(mf, method='rpa', qp=True)` solves one-shot diagonal G0W0 for
every orbital of every spin on the run's own W; the numbers are those of
`calc_qp_energy(mf, selfenergy='GW', polarizability='RPA')` on the same mean
field and route (casida at its default broadening). An orbital without a QP
root is NaN and not converged, flagged once on the run ('qp_no_root').
"""
import warnings

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics.constants import HARTREE_TO_EV
from src.EffectiveHamiltonian.Downfolding import run_low_level
from src.EffectiveHamiltonian.Downfolding.methods import rpa
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.Solvers.qp_equation import QPNoRoot, QPWeightOutside
from tests.downfolding_oracle import OH, unrestricted, water


def _calc(mf, nmo, **kw):
    """calc_qp_energy for every orbital, Hartree; NaN where it has no root."""
    ref = calc_qp_energy(mf, selfenergy='GW', polarizability='RPA',
                         state=list(range(nmo)), **kw)
    # casida: {p: {method: eV}}; the imaginary-axis routes: one eV per state
    ev = ([ref[p]['GW'] for p in range(nmo)] if isinstance(ref, dict)
          else list(ref))
    return np.array(ev, dtype=float) / HARTREE_TO_EV


def _assert_converged_match(qp_e, ref, tol_ev):
    ok = np.isfinite(qp_e) & np.isfinite(ref)
    assert ok.sum() >= 4
    assert np.max(np.abs(qp_e[ok] - ref[ok])) * HARTREE_TO_EV < tol_ev


def test_default_run_has_no_qp():
    assert run_low_level(water(), method='rpa').qp is None


def test_casida_qp_matches_calc_qp_energy():
    mf = water()
    low = run_low_level(mf, method='rpa', mode='casida', qp=True)
    nmo = mf.mo_coeff.shape[1]
    qp = low.qp
    assert len(qp.energies) == 1 and qp.energies[0].shape == (nmo,)
    _assert_converged_match(qp.energies[0], _calc(mf, nmo, mode='casida'),
                            1e-8)
    nocc = mf.mol.nelectron // 2
    frontier = slice(nocc - 1, nocc + 3)              # HOMO .. LUMO+2
    assert qp.converged[0][frontier].all()
    z = qp.z[0][frontier]
    assert np.all((z > 0) & (z <= 1))


def test_casida_qp_unrestricted():
    mf = unrestricted(OH, 1)
    low = run_low_level(mf, method='rpa', mode='casida', qp=True)
    nmo = mf.mo_coeff.shape[-1]
    assert len(low.qp.energies) == 2
    for s, spin in enumerate(('alpha', 'beta')):
        _assert_converged_match(
            low.qp.energies[s],
            _calc(mf, nmo, mode='casida', spin_channel=spin), 1e-8)


def test_degenerate_partners_equal():
    mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-11
    mf.run()
    low = run_low_level(mf, method='rpa', mode='casida', qp=True)
    eps, e = mf.mo_energy, low.qp.energies[0]
    nocc = mol.nelectron // 2
    pairs = [(p, p + 1) for p in range(nocc - 3, nocc + 3)
             if abs(eps[p] - eps[p + 1]) < 1e-10]
    assert pairs                                       # 1pi_u, 1pi_g
    for p, q in pairs:
        assert abs(e[p] - e[q]) < 1e-8


def test_unconverged_orbital_is_flagged_not_fatal(monkeypatch):
    mf = water()
    target = float(mf.mo_energy[2])
    solve = rpa.solve_qp_equation_pole_strength

    def no_root_at_target(func, e0, **kw):
        return np.nan if abs(e0 - target) < 1e-12 else solve(func, e0, **kw)

    monkeypatch.setattr(rpa, 'solve_qp_equation_pole_strength',
                        no_root_at_target)
    with pytest.warns(RuntimeWarning, match='qp'):
        low = run_low_level(mf, method='rpa', mode='casida', qp=True)
    assert np.isnan(low.qp.energies[0][2]) and not low.qp.converged[0][2]
    assert np.isnan(low.qp.z[0][2])
    assert low.qp.converged[0][3]
    assert 'qp_no_root' in [f['flag'] for f in low.numerical_flags]


def test_hf_refuses_qp():
    with pytest.raises(TypeError, match='qp'):
        run_low_level(water(), method='hf', qp=True)


def test_qp_arrays_are_read_only():
    low = run_low_level(water(), method='rpa', qp=True)
    with pytest.raises(ValueError):
        low.qp.energies[0][0] = 0.0


# --- the imaginary-axis routes: the GW drivers on the run's settings --------

def _fitted_water():
    from tests.gw_density_routes import fitted, water as gw_water
    return fitted(scf.RHF(gw_water()))


def test_imagfrequency_qp_matches_calc_qp_energy():
    from src.Base.numerics.numerics import Numerics
    mf, num = _fitted_water(), Numerics()
    nmo = mf.mo_coeff.shape[1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        low = run_low_level(mf, method='rpa', mode='imagfrequency',
                            numerics=num, qp=True)
    _assert_converged_match(
        low.qp.energies[0],
        _calc(mf, nmo, mode='imagfrequency', numerics=num), 1e-10)
    # the root finder's per-orbital warnings are summed up in named flags
    assert not [w for w in caught
                if issubclass(w.category, (QPNoRoot, QPWeightOutside))]
    z = low.qp.z[0]
    outside = np.isfinite(z) & ((z <= 0) | (z > 1))
    flags = [f['flag'] for f in low.numerical_flags]
    assert ('qp_z_outside' in flags) == bool(outside.any())


def test_space_time_qp_matches_calc_qp_energy():
    from tests.isdf_test_grid import g1_numerics
    mf = _fitted_water()
    num = g1_numerics(mf)
    nmo = mf.mo_coeff.shape[1]
    low = run_low_level(mf, method='rpa', mode='space-time', numerics=num,
                        qp=True)
    _assert_converged_match(
        low.qp.energies[0], _calc(mf, nmo, mode='space-time', numerics=num),
        1e-10)


def test_imagfrequency_qp_unrestricted():
    from src.Base.numerics.numerics import Numerics
    mf, num = unrestricted(OH, 1, density_fit=True), Numerics()
    nmo = mf.mo_coeff.shape[-1]
    low = run_low_level(mf, method='rpa', mode='imagfrequency', numerics=num,
                        qp=True)
    for s, spin in enumerate(('alpha', 'beta')):
        _assert_converged_match(
            low.qp.energies[s],
            _calc(mf, nmo, mode='imagfrequency', numerics=num,
                  spin_channel=spin), 1e-10)
