"""The four-centre exchange skeleton in one pass, its rows in shell tiles dealt
over the ranks.

`mean_field_response.exchange_channel_skeleton` (`skeleton_tiles.exchange_skeleton_tiled`) differentiates one exchange
channel (the erf-attenuated one of a range-separated hybrid is never density
fitted) with pyscf's direct derivative integrals, nao^4 work. It runs both
densities through ONE pass (`derivative_exchange_rows`) on fixed tiles of
consecutive shells (`exchange_skeleton_tiles`), dealt to the ranks by their
estimated cost (`exchange_tile_costs`, `balanced_owners`: costliest first onto
the least loaded rank), and adds the tiles' (natm, 3) addends in tile order.

WHAT BREAKS IF THIS FAILS. Contiguous atom shares balanced by AO count left
ranks idle for most of the pass on CzBN1 (2 nodes: 105 s mean, 177 s max
waiting in the sum): a screened row's cost differs between atoms far more
than its AO count says. Interleaved tiles (t % size) still left 129-197 s per
rank on 4 nodes (32 ranks, 172 tiles); the tiles now go by their screened
quartet count.

Gates, formaldehyde/aug-cc-pVDZ, random symmetric densities, omega 0 and 0.3:
  * one rank: the skeleton of two separate `pyscf.grad.rhf.get_jk` passes
    within 1e-12 relative (one pass screens on both densities at once, so a
    quartet near the screening bound can move the last bits);
  * 2 and 3 simulated ranks: the one-rank result bit for bit on every rank,
    each rank asking libcint for exactly its tiles (`balanced_owners`), and
    together for every shell once;
  * the longest-first owners hold no rank more than a fair share of the
    estimated cost plus one tile, on a chain whose tiles differ in cost;
  * the tiles cover the shells once, in order, none wider than the tile
    width unless a single shell is.
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
from src.Base.numerics.constants import SKELETON_EXCHANGE_TILE
from src.gradients.skeleton_tiles import (_derivative_exchange_screen, balanced_owners,
                                          exchange_skeleton_tiles, exchange_tile_costs)

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
def test_ranks_take_interleaved_tiles(size, omega, monkeypatch):
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
    tiles = exchange_skeleton_tiles(mol)
    assert len(tiles) > 2 * size
    with mol.with_range_coulomb(omega):
        owners = balanced_owners(exchange_tile_costs(mol, tiles, _derivative_exchange_screen(mol)),
                                 size)
    shells = []
    for rank, (name, grad) in enumerate(out):
        np.testing.assert_array_equal(grad, serial)
        mine = [tuple(int(x) for x in sl) for sl in asked.get(name, [])]
        assert mine == [tiles[t] for t in range(len(tiles)) if owners[t] == rank]
        shells.extend(s for s0, s1 in mine for s in range(s0, s1))
    assert sorted(shells) == list(range(mol.nbas))


def test_tiles_cover_the_shells():
    mol = gto.M(atom='\n'.join(f'C {1.4 * i} 0 0' if i % 3 else f'H {1.4 * i} 0 0'
                               for i in range(12)), basis='aug-cc-pvdz', verbose=0)
    ao_loc = mol.ao_loc_nr()
    tiles = exchange_skeleton_tiles(mol)
    assert tiles[0][0] == 0 and tiles[-1][1] == mol.nbas
    assert all(a[1] == b[0] for a, b in zip(tiles, tiles[1:]))
    assert all(s1 - s0 == 1 or ao_loc[s1] - ao_loc[s0] <= SKELETON_EXCHANGE_TILE
               for s0, s1 in tiles)


@pytest.mark.parametrize('size', [3, 5])
def test_tiles_go_to_the_ranks_by_their_cost(size):
    """A chain of carbons and hydrogens in aug-cc-pVDZ: its tiles' estimated
    costs differ over twofold, and the longest-first owners leave no rank
    above a fair share plus the costliest tile, while every tile has one
    owner."""
    mol = gto.M(atom='\n'.join(f'C {1.4 * i} 0 0' if i % 3 else f'H {1.4 * i} 0 0'
                               for i in range(12)), basis='aug-cc-pvdz', verbose=0)
    tiles = exchange_skeleton_tiles(mol)
    costs = exchange_tile_costs(mol, tiles, _derivative_exchange_screen(mol))
    assert max(costs) > 2 * min(costs)
    owners = balanced_owners(costs, size)
    assert sorted(set(owners)) == list(range(size)) and len(owners) == len(tiles)
    loads = [sum(c for c, o in zip(costs, owners) if o == r) for r in range(size)]
    assert max(loads) <= sum(costs) / size + max(costs)
