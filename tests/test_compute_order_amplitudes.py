"""The order-general MPn amplitude recursion,
MPnDensityDriver.compute_order_amplitudes_general.

Orders 1-3 come from the committed mpn_density_pieces; order 4 from
mpn_amp_numerators_o4, which GenerateADC writes on each machine (never
committed). The pdaggerq-dependent checks generate it into a temporary
directory and point the generated_mpn package at that directory alone.

  1. Orders <= 3 reproduce the hand-wired chain bit for bit (t4_3 included,
     which the hand chain never materializes), and an ungenerated order
     raises naming GenerateADC.
  2. Order 4: T1^(4) equals the hand-wired compute_t1_4 bit for bit, and the
     MP5 energy E^(5) = 1/4 <ij||ab> t2^(4) (and E^(3), E^(4) from the lower
     orders) equals the lambda-Taylor coefficient of the exact ground-state
     energy of H(lambda) = F + lambda (H - F), extracted on a complex-lambda
     circle (exact_oracle; H4 and LiH, sto-3g).
  3. Rank truncation: on BeH2/sto-3g (6 occupied, 8 virtual spin orbitals: a
     quintuple manifold) t3_4 and t4_4 lack their rank-5/6 rungs and the
     request raises; max_rank_top=2 does not; allow_rank_truncation warns.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import ao2mo, gto, scf

from src.CodeGeneration.pdaggerq_interface import pdaggerq_available
from src.Base.pyscf.pyscf_interface import (get_antisymmetrized_spin_integrals,
                                            get_orbital_energies)
from src.CodeGeneration.diagrammatic import exact_oracle as EO
from src.SingleReference.DensityMatrix import generated_mpn
from src.SingleReference.DensityMatrix.generated_mpn import (
    mpn_density_pieces as gen)
from src.SingleReference.DensityMatrix.mpn_density_driver import MPnDensityDriver
from src.SingleReference.common.mp_amplitudes import orbital_denominator

SYSTEMS = {'H4/sto-3g': 'H 0 0 0; H 0 0 0.80; H 0 0 1.75; H 0 0 2.60',
           'LiH/sto-3g': 'Li 0 0 0; H 0 0 1.6'}
BEH2 = 'Be 0 0 0; H 0 0 1.3; H 0 0 -1.3'
ENERGY_TOL = 1e-11
needs_pdaggerq = pytest.mark.skipif(not pdaggerq_available(),
                                    reason='order 4 is generated with pdaggerq (optional)')


def _scf(atom):
    mol = gto.M(atom=atom, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 300
    mf.run()
    return mol, mf


def _driver(mol, mf):
    n = mf.mo_coeff.shape[1]
    eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape(n, n, n, n)
    h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    _, g = get_antisymmetrized_spin_integrals(h1_mo, eri)
    eps = get_orbital_energies(mf, representation='spin')
    return MPnDensityDriver(eps, g, mol.nelectron)


def _only_modules_in(monkeypatch, path):
    """generated_mpn's per-order numerator modules are read from `path` only
    (the committed mpn_density_pieces is imported directly, not through it)."""
    monkeypatch.setattr(generated_mpn, '__path__', [str(path)])
    for name in [m for m in sys.modules
                 if m.startswith(generated_mpn.__name__ + '.mpn_amp_numerators_')]:
        monkeypatch.delitem(sys.modules, name)


@pytest.fixture(scope='module')
def o4_dir(tmp_path_factory):
    from src.CodeGeneration import generate_mpn_density
    out = tmp_path_factory.mktemp('generated_mpn')
    mp = pytest.MonkeyPatch()
    mp.setattr(generate_mpn_density, 'OUT_DIR', str(out))
    try:
        generate_mpn_density.generate_amplitude_numerator_module(
            [4], 'mpn_amp_numerators_o4')
    finally:
        mp.undo()
    return out


@pytest.mark.parametrize('name', list(SYSTEMS))
def test_orders_up_to_three_are_the_hand_chain(name, tmp_path, monkeypatch):
    _only_modules_in(monkeypatch, tmp_path)
    mpn = _driver(*_scf(SYSTEMS[name]))
    amps = mpn.compute_order_amplitudes_general(3)
    t2_1 = mpn.compute_t2_1()
    o2 = mpn.compute_order2_amplitudes(t2_1)
    E2 = mpn.compute_E(t2_1)
    o3 = mpn.compute_order3_amplitudes(*o2, t2_1, E2=E2)
    t4_3 = gen.t4_3_numerator(**mpn._args(), t2_2=o2[1], t3_2=o2[2],
                              t4_2=o2[3]) / orbital_denominator(mpn.eps[mpn.o], mpn.eps[mpn.v], 4)
    legacy = {(2, 1): t2_1,
              (1, 2): o2[0], (2, 2): o2[1], (3, 2): o2[2], (4, 2): o2[3],
              (1, 3): o3[0], (2, 3): o3[1], (3, 3): o3[2], (4, 3): t4_3}
    assert set(amps) == set(legacy)
    for key, ref in legacy.items():
        assert np.max(np.abs(amps[key] - ref)) == 0.0, key
    with pytest.raises(RuntimeError, match=r'orders \[4\].*GenerateADC'):
        mpn.compute_order_amplitudes_general(4)


def _exact_energy_series(mol, mf, orders, radius=0.3, n_points=32):
    """{n: E^(n)} of the exact ground state of H(lambda), FFT on a circle."""
    eps, h1, g, nocc = EO.spinorb_system(mol, mf)
    es = EO.ExactSelfEnergy(eps, h1, g, nocc)
    h_full = EO._sector_h(es.dets_n, h1, g)
    f_diag = np.diag([sum(eps[p] for p in d) for d in es.dets_n])
    lams = radius * np.exp(2j * np.pi * np.arange(n_points) / n_points)
    vals = []
    for lam in lams:
        es._cache[lam] = ((1 - lam) * f_diag + lam * h_full, None, None)
        vals.append(es._ground_state(lam)[0])
    coefs = np.fft.fft(np.array(vals)) / n_points
    return {n: (coefs[n] / radius ** n).real for n in orders}


@needs_pdaggerq
@pytest.mark.parametrize('name', list(SYSTEMS))
def test_order_four_is_exact(name, o4_dir, monkeypatch):
    _only_modules_in(monkeypatch, o4_dir)
    mol, mf = _scf(SYSTEMS[name])
    mpn = _driver(mol, mf)
    amps = mpn.compute_order_amplitudes_general(4)
    t1_4 = mpn.compute_t1_4(amps[(1, 3)], amps[(2, 3)], amps[(3, 3)], amps[(1, 2)],
                            E2=mpn.compute_E(amps[(2, 1)]))
    assert np.max(np.abs(amps[(1, 4)] - t1_4)) == 0.0
    no, nv = mpn.no, mpn.nv
    for r in (1, 2, 3, 4):
        assert amps[(r, 4)].shape == (nv,) * r + (no,) * r

    exact = _exact_energy_series(mol, mf, (3, 4, 5))
    for n in (3, 4, 5):
        e = mpn.compute_E(amps[(2, n - 1)])
        d = abs(e - exact[n])
        print(f"{name}: E^({n}) from t2^({n - 1}) {e:.12e}, exact {exact[n]:.12e}, |d| {d:.1e}")
        assert d < ENERGY_TOL, (n, d)
    with pytest.raises(RuntimeError, match=r'orders \[5\]'):
        mpn.compute_order_amplitudes_general(5)


@needs_pdaggerq
def test_rank_truncation_raises_on_a_quintuple_manifold(o4_dir, monkeypatch):
    _only_modules_in(monkeypatch, o4_dir)
    mpn = _driver(*_scf(BEH2))
    assert min(mpn.no, mpn.nv) >= 5
    with pytest.raises(RuntimeError, match=r'rank truncation.*t3_4, t4_4'):
        mpn.compute_order_amplitudes_general(4)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        amps = mpn.compute_order_amplitudes_general(4, max_rank_top=2)
    assert (2, 4) in amps and (3, 4) not in amps
    with pytest.warns(UserWarning, match='rank truncation'):
        mpn.compute_order_amplitudes_general(4, allow_rank_truncation=True)
