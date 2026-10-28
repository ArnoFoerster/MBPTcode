"""The four-centre exchange skeleton in one pass, its atoms split over ranks.

`mean_field_response.exchange_channel_skeleton` (`skeleton_tiles.exchange_skeleton_by_atoms`) differentiates one exchange
channel (the erf-attenuated one of a range-separated hybrid is never density
fitted) with pyscf's direct derivative integrals, nao^4 work. It runs both
densities through ONE pass (`derivative_exchange_rows`) and, under ranks,
each rank only the rows of its own atoms (`atom_share`); the (natm, 3) rows
are summed.

Gates, formaldehyde/aug-cc-pVDZ, random symmetric densities, omega 0 and 0.3:
  * one rank: the skeleton of two separate `pyscf.grad.rhf.get_jk` passes
    within 1e-12 relative (one pass screens on both densities at once, so a
    quartet near the screening bound can move the last bits);
  * 2 and 3 simulated ranks: every rank the same bits, the one-rank result
    within 1e-12 relative, and
    each rank asks libcint only for its own atoms' shells, together every
    shell once;
  * the atom shares cover every atom once, contiguous and balanced by AO
    count.
"""
import threading

import numpy as np
import pytest
from pyscf import gto
from pyscf.grad import rhf as grad_rhf
from pyscf.scf import _vhf

from src.Base.utils.parallel.mpi_grid import distributed
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients.mean_field_response import exchange_channel_skeleton
from src.gradients.skeleton_tiles import atom_share

FORMALDEHYDE = 'C 0 0 0; O 0 0 1.205; H 0 0.94 -0.59; H 0.02 -0.94 -0.6'


class _Mf:
    def __init__(self, mol):
        self.mol = mol


def molecule():
    return gto.M(atom=FORMALDEHYDE, basis='aug-cc-pvdz', verbose=0)


def densities(nao):
    rng = np.random.default_rng(11)
    a, b = rng.standard_normal((2, nao, nao))
    return a + a.T, b + b.T


def two_passes(mol, g_ao, D, omega, weight):
    """The skeleton as two separate pyscf derivative K builds give it."""
    with mol.with_range_coulomb(omega):
        vk_D = grad_rhf.get_jk(mol, D)[1]
        vk_g = grad_rhf.get_jk(mol, g_ao)[1]
    t = -weight * (np.einsum('ac,xac->xa', g_ao, vk_D)
                   + np.einsum('ad,xad->xa', D, vk_g))
    return np.array([t[:, p0:p1].sum(axis=1) for _, _, p0, p1 in mol.aoslice_by_atom()])


@pytest.mark.parametrize('omega', [0.0, 0.3])
def test_one_pass_is_two_passes(omega):
    mol = molecule()
    D, g_ao = densities(mol.nao)
    with distributed(None):
        got = exchange_channel_skeleton(_Mf(mol), g_ao, D, omega, 0.7)
    ref = two_passes(mol, g_ao, D, omega, 0.7)
    assert np.abs(got - ref).max() <= 1e-12 * np.abs(ref).max()


@pytest.mark.parametrize('size', [2, 3])
@pytest.mark.parametrize('omega', [0.0, 0.3])
def test_ranks_differentiate_their_atoms(size, omega, monkeypatch):
    mol = molecule()
    D, g_ao = densities(mol.nao)
    with distributed(None):
        serial = exchange_channel_skeleton(_Mf(mol), g_ao, D, omega, 0.7)
    asked, lock = {}, threading.Lock()
    own = _vhf.direct_mapdm

    def counted(*args, **kwargs):
        with lock:
            asked.setdefault(threading.current_thread().name, []).append(kwargs['shls_slice'][:2])
        return own(*args, **kwargs)
    monkeypatch.setattr(_vhf, 'direct_mapdm', counted)

    def one_rank(comm):
        return (threading.current_thread().name,
                exchange_channel_skeleton(_Mf(molecule()), g_ao, D, omega, 0.7))

    out = run_simulated(one_rank, size)
    slices = mol.aoslice_by_atom()
    shells = []
    for rank, (name, grad) in enumerate(out):
        np.testing.assert_array_equal(grad, out[0][1])
        assert np.abs(grad - serial).max() <= 1e-12 * np.abs(serial).max()
        a0, a1 = atom_share(mol, type('C', (), {'Get_size': lambda s: size,
                                                 'Get_rank': lambda s, r=rank: r})())
        expect = [] if a1 == a0 else [(int(slices[a0][0]), int(slices[a1 - 1][1]))]
        assert [tuple(int(x) for x in sl) for sl in asked.get(name, [])] == expect
        shells.extend(range(*expect[0]) if expect else [])
    assert sorted(shells) == list(range(mol.nbas))


@pytest.mark.parametrize('size', [1, 2, 3, 4, 7])
def test_atom_shares_cover_and_balance(size):
    mol = gto.M(atom='\n'.join(f'C {1.4 * i} 0 0' if i % 3 else f'H {1.4 * i} 0 0'
                               for i in range(12)), basis='aug-cc-pvdz', verbose=0)
    nao = np.array([p1 - p0 for _, _, p0, p1 in mol.aoslice_by_atom()])

    class Ranks:
        def __init__(self, r):
            self.r = r

        def Get_size(self):
            return size

        def Get_rank(self):
            return self.r
    shares = [atom_share(mol, Ranks(r)) for r in range(size)]
    covered = [a for a0, a1 in shares for a in range(a0, a1)]
    assert covered == list(range(mol.natm))
    loads = [nao[a0:a1].sum() for a0, a1 in shares]
    assert max(loads) <= nao.sum() / size + nao.max()
