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


# --- sigma_energies: Sigma^E at the QP energies -----------------------------

from src.EffectiveHamiltonian.Downfolding import build_effective_hamiltonian
from src.EffectiveHamiltonian.active_space import (ActiveSpace,
                                                   unrestricted_from_counts)
from src.Base.numerics.numerics import numerics_or_default


def _water_qp():
    mf = water()
    low = run_low_level(mf, method='rpa', mode='casida', qp=True)
    aspace = ActiveSpace.from_counts(mf.mo_energy, mf.mol.nelectron // 2,
                                     2, 2)
    return mf, low, aspace


def _quiet(fun, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fun(*args, **kw)


def test_mf_is_the_default_bitwise():
    _, low, aspace = _water_qp()
    for sigma in ('srg', 'averaged'):
        t0 = _quiet(rpa.downfold, low, aspace, sigma=sigma)
        t1 = _quiet(rpa.downfold, low, aspace, sigma=sigma,
                    sigma_energies='mf')
        assert np.array_equal(t0.sigma_bar[0], t1.sigma_bar[0])
        assert np.array_equal(t0.vtilde[0], t1.vtilde[0])
        assert t0.e_c_active == t1.e_c_active


def test_qp_is_the_formula_at_qp_energies():
    _, low, aspace = _water_qp()
    w = rpa.screened_residues(low, aspace)
    env = aspace.env
    args = (w, low.qp.energies[0][aspace.act], low.eps[env], env < low.nocc,
            low.omega)
    s = numerics_or_default(low.numerics).downfolding.srg_s
    expected = {'averaged': rpa.averaged_self_energy(*args),
                'srg': rpa.srg_self_energy(*args, s)}
    for sigma, ref in expected.items():
        got = _quiet(rpa.downfold, low, aspace, sigma=sigma,
                     sigma_energies='qp').sigma_bar[0]
        assert np.max(np.abs(got - ref)) < 1e-12
        mf_bar = _quiet(rpa.downfold, low, aspace, sigma=sigma).sigma_bar[0]
        assert np.max(np.abs(got - mf_bar)) > 1e-6


def test_qp_through_build_effective_hamiltonian():
    _, low, aspace = _water_qp()
    h_qp = _quiet(build_effective_hamiltonian, low, aspace, sigma='averaged',
                  sigma_energies='qp')
    h_mf = _quiet(build_effective_hamiltonian, low, aspace, sigma='averaged')
    assert not np.allclose(h_qp.g_eff[1], h_mf.g_eff[1])


def test_qp_runs_on_every_route_and_spin():
    from src.Base.numerics.numerics import Numerics
    from tests.isdf_test_grid import g1_numerics
    mf = _fitted_water()
    nocc = mf.mol.nelectron // 2
    aspace = ActiveSpace.from_counts(mf.mo_energy, nocc, 2, 2)
    for mode, num in (('imagfrequency', Numerics()),
                      ('space-time', g1_numerics(mf))):
        low = run_low_level(mf, method='rpa', mode=mode, numerics=num,
                            qp=True)
        qp = _quiet(rpa.downfold, low, aspace, sigma='averaged',
                    sigma_energies='qp').sigma_bar[0]
        m = _quiet(rpa.downfold, low, aspace, sigma='averaged').sigma_bar[0]
        assert np.all(np.isfinite(qp)) and np.max(np.abs(qp - m)) > 1e-6
    mf = unrestricted(OH, 1)
    low = run_low_level(mf, method='rpa', mode='casida', qp=True)
    nocc = tuple(int(np.count_nonzero(o)) for o in mf.mo_occ)
    pair = unrestricted_from_counts(mf.mo_energy, nocc, 4, (2, 1))
    qp = _quiet(rpa.downfold, low, pair, sigma='averaged',
                sigma_energies='qp').sigma_bar
    m = _quiet(rpa.downfold, low, pair, sigma='averaged').sigma_bar
    assert len(qp) == 2
    assert all(np.all(np.isfinite(a)) for a in qp)
    assert max(np.max(np.abs(a - b)) for a, b in zip(qp, m)) > 1e-6


def test_refusals():
    mf, low, aspace = _water_qp()
    plain = run_low_level(mf, method='rpa', mode='casida')
    with pytest.raises(ValueError, match='qp=True'):
        rpa.downfold(plain, aspace, sigma='srg', sigma_energies='qp')
    with pytest.raises(ValueError, match='sigma'):
        rpa.downfold(low, aspace, sigma=None, sigma_energies='qp')
    with pytest.raises(ValueError, match="'mf', 'qp'"):
        rpa.downfold(low, aspace, sigma='srg', sigma_energies='quasiparticle')
    _quiet(rpa.downfold, low, aspace, sigma='srg', sigma_energies='QP')
    t = int(aspace.act[0])
    energies = low.qp.energies[0].copy()
    energies[t] = np.nan
    low.qp = rpa.QuasiParticles(energies=(energies,), z=low.qp.z,
                                converged=(np.isfinite(energies),))
    with pytest.raises(ValueError, match=str(t)):
        rpa.downfold(low, aspace, sigma='srg', sigma_energies='qp')


def test_pole_check_uses_qp_energies():
    _, low, aspace = _water_qp()
    lo, hi = rpa.pole_window(low, aspace)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        rpa.downfold(low, aspace, sigma='averaged')
    assert not [w for w in caught
                if issubclass(w.category, rpa.SigmaNearPoleWarning)]
    t = int(aspace.act[-1])
    energies = low.qp.energies[0].copy()
    energies[t] = hi - 1e-6
    low.qp = rpa.QuasiParticles(energies=(energies,), z=low.qp.z,
                                converged=low.qp.converged)
    with pytest.warns(rpa.SigmaNearPoleWarning, match=str(t)):
        rpa.downfold(low, aspace, sigma='averaged', sigma_energies='qp')
    rec = _quiet(rpa.downfold, low, aspace, sigma='averaged',
                 sigma_energies='qp').diagnostics['sigma_poles']
    assert abs(rec['pole_distance'][-1] - 1e-6) < 1e-9


def test_qp_computed_once(monkeypatch):
    calls = []
    real = rpa.quasiparticles

    def counted(low):
        calls.append(1)
        return real(low)

    monkeypatch.setattr(rpa, 'quasiparticles', counted)
    mf, low, aspace = _water_qp()
    other = ActiveSpace.from_counts(mf.mo_energy, mf.mol.nelectron // 2, 1, 1)
    for space in (aspace, other):
        _quiet(rpa.downfold, low, space, sigma='srg', sigma_energies='qp')
    assert len(calls) == 1


def test_qp_diagnostic():
    mf, low, aspace = _water_qp()
    rec = rpa.downfold(low, aspace).diagnostics['qp_energies']
    assert np.array_equal(rec['energies'], low.qp.energies[0][aspace.act])
    assert np.array_equal(rec['z'], low.qp.z[0][aspace.act])
    plain = run_low_level(mf, method='rpa', mode='casida')
    assert 'qp_energies' not in rpa.downfold(plain, aspace).diagnostics
