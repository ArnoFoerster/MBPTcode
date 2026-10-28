"""One factorization of the PCM matrix per geometry, not two solves per cycle.

pyscf's polarizable continuum solves K q = R v and K^T x = v afresh on every
SCF iteration, each a dense O(n^3) in the number of surface points. K is built
from the cavity and the dielectric constant alone, so it does not change while
the density iterates: the factorization belongs outside the loop and only the
triangular solves belong inside.

WHAT THIS IS AND IS NOT WORTH. The triangular solves are 80 times cheaper than
the dense ones at anthracene's size and more above it, but measured against a
whole iteration they are a small part of it: on anthracene at Lebedev order 29,
a cycle spends 28 s in the exchange-correlation potential, 21 s building the
surface potential of the density and putting the charges back on the basis,
and 1.4 s in these two solves. So this is a few percent, taken because it is
free and exact, and not a cure.

The cost that dominates the continuum is the SURFACE POTENTIAL, which is
n_ao^2 x n_surf, and it shares its cubic growth with everything else. Both it
and these solves fall with the number of surface points, which is the knob that
actually moves: on formaldehyde the solvation energy changes by 0.4 meV between
Lebedev order 11 and order 29 while the potential costs four times more at 29.

THE SURFACE POTENTIAL ON TILES, SPLIT OVER RANKS. pyscf evaluates it, and the
reaction-field matrix `_get_vmat`, as (nao, nao, k) blocks of three-centre
integrals with k sized by `max_memory` alone, so a large memory share puts the
whole (nao, nao, n_surf) array in one block: 16 GB for a 62-atom emitter in
aug-cc-pVDZ, on every rank, in every SCF cycle. `tile_cavity` gives the PCM
object versions of both on `cavity_blocks` tiles instead. Where the SCF runs
over ranks, `distributed_fock` attaches its communicator as `_cavity_comm` for
as long as it locks the continuum; each rank then evaluates only a contiguous
range of the tiles, so it holds one tile at a time and the integrals cost
about 1 / n of a cycle. The tiles depend on the cavity and the basis alone
(ISDF_TILE_GB, libcint's index bound; a cavity past one tile is cut into at
least PCM_CAVITY_MIN_TILES), and the
reaction-field matrix is added tile by tile in tile order (`ordered_sum`), so
the result is the same bits at every rank count. Every rank enters both calls
at the same point of the loop there, which is what the reduction needs;
outside that block the tiles run replicated.
"""
import numpy as np
import scipy.linalg
from pyscf import df as pyscf_df
from pyscf import gto

from src.Base.numerics.constants import ISDF_TILE_GB, PCM_CAVITY_MIN_TILES
from src.Base.utils.parallel.mpi_grid import contiguous_block, ordered_sum, reduce_sum
from src.environments.pcm_derivatives import cavity_blocks

#: Attribute the factorization is cached under, on the PCM object itself.
_CACHE = '_cached_pcm_lu'


def _factorization(obj, K):
    """(lu, luT) for this K, rebuilt whenever `build` replaces the matrix."""
    cached = getattr(obj, _CACHE, None)
    if cached is not None and cached[0] is K:
        return cached[1], cached[2]
    lu, lu_t = scipy.linalg.lu_factor(K), scipy.linalg.lu_factor(K.T)
    setattr(obj, _CACHE, (K, lu, lu_t))
    return lu, lu_t


def _get_vind(self, dms):
    """pyscf's own reaction-field step with the two dense solves factorized.

    The body follows `pyscf.solvent.pcm.PCM._get_vind` exactly; only the two
    `numpy.linalg.solve` calls become triangular solves against a cached LU.
    A gate asserts this against the unpatched routine, so a change upstream
    shows up as a disagreement rather than as silence.
    """
    if not self._intermediates:
        self.build()

    nao = dms.shape[-1]
    dms = dms.reshape(-1, nao, nao)
    if dms.shape[0] == 2:
        dms = (dms[0] + dms[1]).reshape(-1, nao, nao)

    K = self._intermediates['K']
    R = self._intermediates['R']
    lu, lu_t = _factorization(self, K)
    v_grids_e = self._get_v(dms)
    v_grids = self.v_grids_n - v_grids_e

    b = np.dot(R, v_grids.T)
    q = scipy.linalg.lu_solve(lu, b).T

    vK_1 = scipy.linalg.lu_solve(lu_t, v_grids.T)
    qt = np.dot(R.T, vK_1).T
    q_sym = (q + qt) / 2.0

    vmat = self._get_vmat(q_sym)
    epcm = 0.5 * np.dot(q_sym[0], v_grids[0])

    self._intermediates['q'] = q[0]
    self._intermediates['q_sym'] = q_sym[0]
    self._intermediates['v_grids'] = v_grids[0]
    self._intermediates['dm'] = dms
    return epcm, vmat[0]


def cavity_tiles(nao, ngrids):
    """[(p0, p1)] tiles of the surface points: `cavity_blocks` (ISDF_TILE_GB,
    libcint's index bound); a cavity that needs more than one is cut finer, to
    at least PCM_CAVITY_MIN_TILES, so that many ranks can share it. One tile
    is pyscf's own block and its bits. A function of the cavity and the basis
    only."""
    tiles = cavity_blocks(nao, ngrids, comp=1, budget_gb=ISDF_TILE_GB)
    if len(tiles) == 1:
        return tiles
    width = min(tiles[0][1] - tiles[0][0], -(-ngrids // PCM_CAVITY_MIN_TILES))
    width = max(width, 1)
    return [(p0, min(p0 + width, ngrids)) for p0 in range(0, ngrids, width)]


def _cavity_share(self, nao):
    """(tiles, comm): the [(index, (p0, p1))] tiles this rank evaluates, a
    contiguous range of `cavity_tiles`, and the communicator their partials
    are summed over (None: all of them, here)."""
    ngrids = self.surface['grid_coords'].shape[0]
    tiles = list(enumerate(cavity_tiles(nao, ngrids)))
    comm = getattr(self, '_cavity_comm', None)
    if comm is None or comm.Get_size() == 1:
        return tiles, None
    start, stop = contiguous_block(len(tiles), comm.Get_rank(), comm.Get_size())
    return tiles[start:stop], comm


def _cavity_integrals(self, tiles):
    """(index, (p0, p1), (nao, nao, p1 - p0) potential integrals) for each
    tile: pyscf's `_get_v` block, one tile at a time."""
    mol = self.mol
    coords = self.surface['grid_coords']
    exponents = self.surface['charge_exp']
    int3c2e = mol._add_suffix('int3c2e')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
    for index, (p0, p1) in tiles:
        fakemol = gto.fakemol_for_charges(coords[p0:p1], expnt=exponents[p0:p1] ** 2)
        yield index, (p0, p1), pyscf_df.incore.aux_e2(mol, fakemol, intor=int3c2e,
                                                      aosym='s1', cintopt=cintopt)


def _get_v(self, dms):
    """pyscf's `PCM._get_v` (the electronic potential on the surface points,
    (nset, n_surf)) on this rank's tiles; the other ranks' points are zero
    here until the sum, so it is exact."""
    nao = dms.shape[-1]
    ngrids = self.surface['grid_coords'].shape[0]
    v_grids_e = np.zeros((dms.shape[0], ngrids))
    tiles, comm = _cavity_share(self, nao)
    for _, (p0, p1), v_nj in _cavity_integrals(self, tiles):
        for i in range(dms.shape[0]):
            v_grids_e[i, p0:p1] = np.einsum('ijL,ij->L', v_nj, dms[i])
    return reduce_sum(v_grids_e, comm)


def _get_vmat(self, q):
    """pyscf's `PCM._get_vmat` (the charges' potential on the basis,
    (nset, nao, nao)): each tile's term, added onto zero in tile order over
    all ranks' tiles (`ordered_sum`), which with one tile is pyscf's bits."""
    nao = self.mol.nao
    ngrids = self.surface['grid_coords'].shape[0]
    q = q.reshape(-1, ngrids)
    tiles, comm = _cavity_share(self, nao)
    terms = []
    for index, (p0, p1), v_nj in _cavity_integrals(self, tiles):
        terms.append((index, np.stack([-np.einsum('ijL,L->ij', v_nj, q[i, p0:p1])
                                       for i in range(q.shape[0])])))
    shape = (q.shape[0], nao, nao)
    return ordered_sum(terms, comm, shape=shape, onto=np.zeros(shape))


def tile_cavity(with_solvent):
    """Give one PCM object the tiled surface potential and reaction-field
    matrix (`_get_v`, `_get_vmat`), and return it. Idempotent."""
    if with_solvent is None or getattr(with_solvent, '_tiled_cavity', False):
        return with_solvent
    with_solvent._get_v = _get_v.__get__(with_solvent)
    with_solvent._get_vmat = _get_vmat.__get__(with_solvent)
    with_solvent._tiled_cavity = True
    return with_solvent


def factorize_once(with_solvent):
    """Give one PCM object the cached factorization and the tiled cavity
    integrals (`tile_cavity`), and return it.

    Idempotent, and safe on an object that is rebuilt: the cache is keyed on
    the identity of K, so a `build()` that replaces the matrix invalidates it.
    """
    if with_solvent is None or getattr(with_solvent, '_cached_factorization', False):
        return with_solvent
    with_solvent._get_vind = _get_vind.__get__(with_solvent)
    with_solvent._cached_factorization = True
    return tile_cavity(with_solvent)
