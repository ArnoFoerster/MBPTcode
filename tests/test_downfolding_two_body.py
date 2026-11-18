"""The exact active-space integrals and the packing of g_eff.

active_eri reproduces the active block of the exact MO integrals of the
oracle. Packed spin-free and spin-orbital, the same restricted Hamiltonian of
H2/6-31G (all four spatial orbitals active, two electrons) gives the same
ground state energy from the openfermion and the pyscf_fci backends, and both
equal pyscf's FCI energy when the constant is the nuclear repulsion. An
empty active space gives zero-size blocks and a g_eff of the constant alone,
and the spin-free g_eff never aliases its inputs.
"""
import numpy as np
import pytest
from pyscf import fci

from src.MultiReference.Downfolding.two_body import active_eri, pack
from src.Solvers.active_space import solve
from tests.downfolding_oracle import h2, mo_integrals, water


def test_active_eri_is_exact():
    mf = water()
    idx = [2, 4, 5, 6]
    act = mf.mo_coeff[:, idx]
    _, eri = mo_integrals(mf)
    ref = eri[np.ix_(idx, idx, idx, idx)]
    aa, ab, bb = active_eri(mf.mol, (act, act))
    for block in (aa, ab, bb):
        assert block.shape == (4,) * 4
        assert np.max(np.abs(block - ref)) < 1e-10


def test_active_eri_ab_index_convention():
    """ab[t,u,v,w] = (t_a u_a | v_b w_b): alpha pair first."""
    mf = water()
    ia, ib = [2, 4, 5, 6], [3, 4, 5, 6]
    _, eri = mo_integrals(mf)
    _, ab, _ = active_eri(mf.mol, (mf.mo_coeff[:, ia], mf.mo_coeff[:, ib]))
    assert np.max(np.abs(ab - eri[np.ix_(ia, ia, ib, ib)])) < 1e-10


def test_pack_spin_free_contract():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(3, 3))
    h = h + h.T
    v = rng.normal(size=(3,) * 4)
    g = pack((h, h), (v, v, v), 1.5, spin_free=True)
    assert g[0] == 1.5
    assert np.array_equal(g[1], h)
    assert np.array_equal(g[2], v.transpose(0, 2, 1, 3))
    assert g[2][0, 1, 2, 0] == v[0, 2, 1, 0]


def test_pack_spin_orbital_solves_the_same():
    mf = h2()
    h, eri = mo_integrals(mf)
    n = h.shape[0]
    e0 = mf.energy_nuc()
    args = ((h, h), (eri, eri, eri), e0)
    g_so = pack(*args, spin_free=False)
    g_sf = pack(*args, spin_free=True)
    assert g_so[0] == e0
    e_so = solve('openfermion', g_so, 2 * n, 2)
    e_sf = solve('pyscf_fci', g_sf, n, 2, spin_free=True)
    e_fci = fci.FCI(mf).kernel()[0]
    assert abs(e_so - e_sf) < 1e-9
    assert abs(e_sf - e_fci) < 1e-9


def test_pack_refuses_different_channels():
    rng = np.random.default_rng(1)
    h = np.eye(2)
    v = rng.normal(size=(2,) * 4)
    with pytest.raises(NotImplementedError, match='unrestricted'):
        pack((h, h + 0.1), (v, v, v), 0.0, spin_free=True)
    with pytest.raises(NotImplementedError, match='unrestricted'):
        pack((h, h), (v, v + 0.1, v), 0.0, spin_free=False)
    with pytest.raises(NotImplementedError, match='unrestricted'):
        pack((h, h), (v, v, v + 0.1), 0.0, spin_free=False)


def test_empty_active_space():
    """n_act = 0: zero-size integrals without an integral transformation, and
    a g_eff that carries only the constant."""
    mf = water()
    empty = mf.mo_coeff[:, []]
    blocks = active_eri(mf.mol, (empty, empty))
    assert [b.shape for b in blocks] == [(0, 0, 0, 0)] * 3
    h = np.zeros((0, 0))
    for spin_free in (True, False):
        g = pack((h, h), blocks, -76.0, spin_free=spin_free)
        assert g[0] == -76.0
        assert g[1].shape == (0, 0) and g[2].shape == (0, 0, 0, 0)


def test_pack_spin_free_does_not_alias():
    h = np.eye(2)
    v = np.ones((2,) * 4)
    g = pack((h, h), (v, v, v), 0.0, spin_free=True)
    g[1][0, 0] = 5.0
    g[2][0, 0, 0, 0] = 5.0
    assert h[0, 0] == 1.0 and v[0, 0, 0, 0] == 1.0
