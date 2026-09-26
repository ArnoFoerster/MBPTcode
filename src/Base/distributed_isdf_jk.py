"""The ISDF-K SCF divided over the ranks: the interpolation points in fixed
tiles, and the auxiliary shells of the integral-direct Coulomb matrix.

WHY. At a six-chlorophyllide hexamer model / cc-pVTZ (534 atoms, nao 12204, naux
31260) the fitted three-centre tensor is 18.6 TB, 37 TB with the second one a
range-separated functional builds, so the mean field's exchange is the ISDF
one (`isdf_jk.ISDFJK`),

    K = X^T [Z o (X Dm X^T)] X,     Z = M^T V M,     X[P, mu] = chi_mu(r_P),

three M^2 x nao contractions on M = 117762 points, and its Coulomb matrix is
pyscf's integral-direct DF-J, which stores no tensor. On one rank that is the
whole SCF, and Z alone is 111 GB.

WHAT A RANK HOLDS. The grid index is cut into the fixed tiles of the
row-distributed fit (`separable_ri.fit_rows`, `FIT_CHOLESKY_BLOCK` points),
owned block-cyclically -- tile t by rank t % size, the fit's own layout, so
M^T never moves. A rank holds, for its tiles: X_t, M^T_t, and per operator
either Z's rows (Z_t = G_t M^T, G_t = M^T_t V, 'dense') or G_t alone, Z's
blocks then formed per build ('factored'). No rank holds X, M, Z or
X Dm X^T whole. V is evaluated by every rank in fixed slabs of auxiliary
shells and never held whole either.

HOW K IS BUILT. The density enters through its occupied factor, since
pyscf tags every density it builds with its orbitals: Dm = C~ C~^T with
C~ = C_occ sqrt(n), so X Dm X^T = Y Y^T with Y = X C~, M^2 nocc and not
M^2 nao (an untagged density takes Y = X Dm, the general form). For every
column tile j in tile order its owner broadcasts [Y_j | X_j (| M^T_j)], and
each rank adds, for each of its row tiles i,

    T_i += [(Y_i Y_j^T) o Z_ij] X_j,         then  K_r = sum_i X_i^T T_i,

so every GEMM has a shape fixed by the tiles, T_i is summed over j in one
order at every rank count, and the rank's partial K_r goes into ONE
reduction with J -- a sum over the grid, re-associated at the rank
boundaries like every Fock partial of the distributed DF SCF.

J. `df_jk.get_j`'s two integral passes, the auxiliary shells cut into
contiguous blocks of about naux / size functions: pass 1 gives each rank
the exact rows of (P|mu nu) Dm_{mu nu} for its shells, gathered verbatim;
the metric solve rho = V^-1 j is every rank's own, locked to rank 0's; pass
2 is each rank's shells' partial of sum_P (P|mu nu) rho_P, reduced with K.

THE DRIVER IS `distributed_df.distributed_mean_field`, unchanged:
`distributed_df_jk` hands an ISDF mean field this handle, which locksteps the
density and its orbitals at entry and reduces its partials, so pyscf's loop
runs on every rank on the same numbers.

THREADS. The driver holds BLAS at one thread for pyscf's OpenMP (`nr_rks` on
the DFT grid, the DF-J passes). The row fit, G and Z's rows and the K builds
are BLAS GEMMs, and the metric's factor and solve LAPACK: each takes back the
pool the driver's wrap took (`threads.blas_full_pool`), so it runs on the
count the process had before the SCF -- the count a one-rank handle built
outside it runs on, which keeps the tiles its tiles bit for bit -- while the
libcint passes inside it (the metric slabs, the fit's three-centre integrals)
drop to one again. `scf_isdf_blas_fit`, `scf_isdf_blas_kernel` and
`scf_isdf_blas_k` record the counts they ran on.

WHAT THE NUMBERS DO. The row fit is another realization of the serial fit's
estimator (not its bits), and the reductions re-associate J and K, so the
distributed SCF lands on the serial ISDF-K SCF within the SCF's own
convergence threshold, the same on every rank.

MEASURED, anthracene/cc-pVDZ/LRC-wPBEh on two laptop threads (M 3552 in 7
tiles, nao 246, naux 924, nocc 47), the one-rank handle: the row fit 12.8
s; one operator's Z rows 0.75 s (31 GF/s); a K build 0.23 s, 34 GF/s on
2 M^2 (nocc + nao) + 2 M nao^2, against 0.32 s for the serial ISDFJK's K on
the density itself; J 1.7 s, pyscf's integral-direct DF-J either way.

SCALED TO THE HEXAMER / cc-pVTZ (M 117762 in 230 tiles of 512, nao 12204,
naux 31260, nocc 1062), per rank, 8 / 16 ranks, Z held for the bare and the
long-range operator (a rank's ~29 / ~15 tiles, ~14720 / ~7360 rows):

  array, GB                                     8 ranks   16 ranks
  X tiles                                       1.44      0.72
  M^T tiles                                     3.68      1.84
  Z rows, two operators                         27.7      13.9
  G tiles, while one operator's Z forms         3.68      1.84
  T rows, one per density                       1.44      0.72
  Y tiles; a streamed tile; a Hadamard block    0.13; 0.05; 0.002
  metric slab, while G forms                    0.13      0.13
  K and J partials                              2.38      2.38
  DF-J's metric factor (twice that as it forms) 7.8       7.8
  held through the SCF                          ~45       ~27
  the row fit's own peak, in the build          ~23       ~17

  flops per rank                                8 ranks   16 ranks
  a K build, 2 M^2 (nocc + nao) + 2 M nao^2     5.0e13    2.5e13
  Z's rows once per operator, 2 M^2 naux        1.1e14    5.4e13
  G once per operator, 2 M naux^2               2.9e13    1.4e13

At the measured 17 GF/s a core, 16 cores a rank, a K build is ~190 s at 8
ranks and ~95 s at 16, each operator's Z ~500 s / ~250 s once; the DF-J,
11289 s a build on one core, ~90 s / ~45 s where it divides over the 128 /
256 cores. Every K build receives the other ranks' Y and X tiles, 12.5 GB a
rank. Factored, a build pays 2 M^2 naux / size again and holds no Z rows.
"""
import contextlib
import time

import numpy as np
import scipy.linalg
from pyscf import df, lib
from pyscf.df import addons
from pyscf.lib import logger
from pyscf.scf import _vhf, jk

from src.Base.constants import (FIT_CHOLESKY_BLOCK,
                                ISDF_SCF_KERNEL_MEMORY_FRACTION,
                                ISDF_SCF_METRIC_SLAB)
from src.Base.distributed_df import (_counted, _lockstep_density,
                                     _reduce_parts, _spent)
from src.Base.isdf_jk import ISDFJK, isdf_grid, range_coulomb
from src.Base.separable_ri import fit_M_streaming
from src.Base.utils.mpi_grid import (allgather_ranges, broadcast,
                                     broadcast_rows, contiguous_block,
                                     current_comm, lockstep, partition)
from src.Base.utils.threads import blas_full_pool, blas_single_threaded

#: The BLAS threads the row fit, the interaction's rows and the K builds ran
#: on, in `mf._distributed_timings`: the fewest over the calls, 0 where the
#: count cannot be read.
ISDF_BLAS_KEYS = ('scf_isdf_blas_fit', 'scf_isdf_blas_kernel',
                  'scf_isdf_blas_k')
#: What the ISDF handle adds to `mf._distributed_timings`, beside the DF
#: SCF's keys it fills the same way (`scf_build_slices` is this build,
#: `scf_fock_jk` this rank's partials, `scf_reduce` their reduction):
#: `scf_isdf_fit`, the row fit inside the build; `scf_isdf_kernel`, the
#: interaction's rows (G, and Z's rows where they are held), per operator;
#: `scf_fock_j` and `scf_fock_k`, the two partials; `scf_isdf_stream`, the
#: column tiles' broadcasts inside the K builds; `scf_requests_k`, the K
#: builds this rank contributed to; and `ISDF_BLAS_KEYS`.
ISDF_TIMING_KEYS = ('scf_isdf_fit', 'scf_isdf_kernel', 'scf_fock_j',
                    'scf_fock_k', 'scf_isdf_stream',
                    'scf_requests_k') + ISDF_BLAS_KEYS


class _DirectCoulomb:
    """pyscf's integral-direct DF-J (`df_jk.get_j`) for one operator, the
    auxiliary shells cut into one contiguous block per rank.

    The setup is `df_jk.get_j`'s own, inside the attenuated operator where
    `omega` asks for one: the Schwarz and density screening, the metric's
    Cholesky factor (or the metric itself where it is not positive definite)
    and the concatenated molecule the three-centre passes run on.
    """

    def __init__(self, mol, auxbasis, omega, direct_scf_tol, comm):
        self.comm = comm
        size = 1 if comm is None else comm.Get_size()
        self.rank = 0 if comm is None else comm.Get_rank()
        with (range_coulomb(mol, None, omega) if omega
              else contextlib.nullcontext()):
            auxmol = addons.make_auxmol(mol, auxbasis)
            opt = _vhf._VHFOpt(mol, 'int3c2e', 'CVHFnr3c2e_schwarz_cond',
                               dmcondname='CVHFnr_dm_cond',
                               direct_scf_tol=direct_scf_tol)
            opt.init_cvhf_direct(mol, 'int2e', 'CVHFnr_int2e_q_cond')
            j2c = auxmol.intor('int2c2e', hermi=1)
            aux_loc = auxmol.ao_loc
            aux_q_cond = [np.sqrt(abs(j2c.diagonal()))[i0:i1].max()
                          for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])]
            opt.q_cond = np.hstack((opt.q_cond.ravel(), aux_q_cond))
            try:
                with blas_full_pool():
                    self.j2c, self.j2c_type = scipy.linalg.cho_factor(
                        j2c, lower=True), 'cd'
            except scipy.linalg.LinAlgError:
                self.j2c, self.j2c_type = j2c, 'regular'
            # the fourth index of jk.get_jk, one s function standing in
            bas_placeholder = np.array([0, 0, 1, 1, 0, 0, 0, 0],
                                       dtype=np.int32)
            fakemol = mol + auxmol
            fakemol._bas = np.vstack((fakemol._bas, bas_placeholder))
        self.opt, self.fakemol, self.aux_loc = opt, fakemol, aux_loc
        self.nbas, self.nao = mol.nbas, mol.nao_nr()
        self.naux = int(aux_loc[-1])
        # Contiguous shells of about naux / size functions each, the blocks
        # tiling the auxiliary shells.
        starts = [int(np.searchsorted(aux_loc, contiguous_block(
            self.naux, r, size)[0])) for r in range(size)]
        self.shells = list(zip(starts, starts[1:] + [auxmol.nbas]))

    def __call__(self, dms):
        """This rank's partial of J for a stack of density matrices."""
        n = len(dms)
        nbas, aux_loc, opt = self.nbas, self.aux_loc, self.opt
        nbas1 = nbas + len(aux_loc) - 1
        k0, k1 = self.shells[self.rank]
        p0, p1 = int(aux_loc[k0]), int(aux_loc[k1])
        shls_slice = (0, nbas, 0, nbas, nbas + k0, nbas + k1, nbas1,
                      nbas1 + 1)
        # pass 1: (P|mu nu) Dm_{mu nu} for this rank's P, exact rows
        jaux = np.zeros((self.naux, n))
        if k1 > k0:
            with lib.temporary_env(
                    opt, prescreen='CVHFnr3c2e_vj_pass1_prescreen'):
                part = jk.get_jk(self.fakemol, dms, ['ijkl,ji->kl'] * n,
                                 'int3c2e', aosym='s2ij', hermi=0,
                                 shls_slice=shls_slice, vhfopt=opt)
            jaux[p0:p1] = np.array(part)[:, :, 0].T
        allgather_ranges(jaux, [[(int(aux_loc[a]), int(aux_loc[b]))]
                                for a, b in self.shells], self.comm)
        with blas_full_pool():
            if self.j2c_type == 'cd':
                rho = scipy.linalg.cho_solve(self.j2c, jaux)
            else:
                rho = scipy.linalg.solve(self.j2c, jaux)
        # each rank's own solve: rank 0's is the one every partial reads
        rho = lockstep(np.ascontiguousarray(rho.T), self.comm, check=True)
        rho = rho[:, None, :]
        vj = np.zeros((n, self.nao, self.nao))
        if k1 > k0:
            # pass 2: sum over this rank's P of (P|mu nu) rho_P
            with lib.temporary_env(
                    opt, prescreen='CVHFnr3c2e_vj_pass2_prescreen',
                    _dmcondname=None):
                opt.dm_cond = np.array([abs(rho[:, :, i0:i1]).max()
                                        for i0, i1 in zip(aux_loc[:-1],
                                                          aux_loc[1:])])
                vj = np.asarray(jk.get_jk(
                    self.fakemol, np.ascontiguousarray(rho[:, :, p0:p1]),
                    ['ijkl,lk->ij'] * n, 'int3c2e', aosym='s2ij', hermi=1,
                    shls_slice=shls_slice, vhfopt=opt)).reshape(
                        n, self.nao, self.nao)
        return vj


class DistributedISDFJK(df.df.DF):
    """A `with_df` answering J and K of an ISDF mean field from this rank's
    tiles of the interpolation grid and block of the auxiliary shells.

    Built from the mean field's own `ISDFJK`, whose grid settings, fit
    settings and Coulomb route it takes; every rank calls `get_jk` at the
    same point of the same SCF loop, locksteps the density, adds this rank's
    partials to the others' and returns the total.

    tile: grid points per tile, `FIT_CHOLESKY_BLOCK` when None -- the fit's
          tiles and the kernel's are the same ones.
    """

    _keys = {'comm', 'timings', 'source', 'tile', 'tiles', 'mine', 'coords',
             'X', 'MT', 'held', 'fit_held', 'z_mode', 'omega_kernels'}

    def __init__(self, source, comm=None, timings=None, tile=None):
        if not isinstance(source, ISDFJK):
            raise TypeError(f'DistributedISDFJK divides an ISDFJK, not a '
                            f'{type(source).__name__}')
        super().__init__(source.mol, auxbasis=source.auxbasis)
        comm = current_comm() if comm is None else comm
        self.comm, self.timings, self.source = comm, timings, source
        self.max_memory = source.max_memory
        self.stdout, self.verbose = source.stdout, source.verbose
        self.tile = FIT_CHOLESKY_BLOCK if tile is None else int(tile)
        if source.j_route not in ('df-direct', 'isdf'):
            raise ValueError(f'j_route={source.j_route!r}')
        if source.j_route == 'isdf':
            raise NotImplementedError(
                "j_route='isdf' builds J from the interpolation, which "
                'ISDFJK itself warns is unsafe in an SCF; over ranks the '
                "Coulomb matrix is the integral-direct DF-J ('df-direct').")
        if source.refit_omega:
            raise NotImplementedError(
                'refit_omega fits a second M against the attenuated metric; '
                'over ranks the attenuated Z reuses the one row fit.')
        self.tiles = self.mine = self.coords = None
        self.X = self.MT = None
        #: {array: most bytes of it this rank held at once}, read off arrays.
        self.held = {}
        #: The row fit's own ledger (`RowFit.held`).
        self.fit_held = {}
        #: 'dense' (Z's rows held per operator) or 'factored' (G held, Z's
        #: blocks formed per build); rank 0's decision.
        self.z_mode = None
        #: {'%.6f' % omega: {tile: Z rows or G rows}}
        self.omega_kernels = {}
        self._coulomb = {}
        self._replaced = None

    # -- construction --------------------------------------------------------

    def build(self):
        """This rank's tiles: the grid on every rank (rank 0's), the row fit,
        the collocation of the tiles this rank owns. Collective."""
        if self.MT is not None:
            return self
        t_build = time.time()
        log = logger.new_logger(self)
        t0 = (logger.process_clock(), logger.perf_counter())
        source, mol, comm = self.source, self.mol, self.comm
        rank, size = ((0, 1) if comm is None
                      else (comm.Get_rank(), comm.Get_size()))
        self.auxmol = addons.make_auxmol(mol, self.auxbasis)
        # the grid's frames and the fit are BLAS: the pool a wrap took
        with blas_full_pool() as pool:
            coords = source.coords
            if coords is None:
                # `ISDFJK.build`'s own grid call, so the two handles fit one
                # grid
                counts = source.counts if source._named_counts else None
                coords = isdf_grid(mol, counts=counts, radii=source.radii,
                                   auxbasis=source.auxbasis,
                                   n_start=source.n_start)
            coords = lockstep(np.array(coords, dtype=float, order='C'), comm,
                              check=True)
            self.coords = coords
            nk = len(coords)
            self.tiles = [(t0_, min(t0_ + self.tile, nk))
                          for t0_ in range(0, nk, self.tile)]
            self.mine = [int(t) for t in partition(len(self.tiles), rank,
                                                   size)]
            t_fit = time.time()
            fit = fit_M_streaming(mol, self.auxmol, coords,
                                  l_max_second=source.l_max_second,
                                  regularization=source.regularization,
                                  block_memory_gb=source.block_memory_gb,
                                  progress=source.progress, comm=comm,
                                  fit='rows', block=self.tile)
            _spent(self.timings, 'scf_isdf_fit', t_fit)
        _ran_on(self.timings, 'scf_isdf_blas_fit', pool)
        self.MT = fit.mt
        self.fit_held = dict(fit.held)
        del fit
        self.X = {t: mol.eval_gto('GTOval_sph', coords[slice(*self.tiles[t])])
                  for t in self.mine}
        self._hold('X_tiles', self.X.values())
        self._hold('MT_tiles', self.MT.values())
        _spent(self.timings, 'scf_build_slices', t_build)
        log.timer('ISDF tiles %d of %d (M = %d, nao = %d, naux = %d)'
                  % (len(self.mine), len(self.tiles), nk, mol.nao_nr(),
                     self.auxmol.nao_nr()), *t0)
        return self

    def _hold(self, name, arrays):
        """`held[name]`: the most bytes the arrays named so have summed to."""
        nbytes = sum(int(a.nbytes) for a in arrays)
        self.held[name] = max(self.held.get(name, 0), nbytes)

    def _resolve_z_mode(self):
        """Rank 0's choice, the same on every rank: Z's rows held where the
        two operators a range-separated SCF asks for fit in
        `ISDF_SCF_KERNEL_MEMORY_FRACTION` of what max_memory leaves."""
        if self.z_mode is not None:
            return self.z_mode
        mode = self.source.z_mode
        if mode == 'auto':
            rows = sum(t1 - t0 for t0, t1 in (self.tiles[t]
                                              for t in self.mine))
            mine = 2 * rows * len(self.coords) * 8
            free = (self.max_memory - lib.current_memory()[0]) * 1e6
            mode = ('dense' if mine < ISDF_SCF_KERNEL_MEMORY_FRACTION * free
                    else 'factored')
        self.z_mode = broadcast(mode, self.comm)
        return self.z_mode

    def interaction(self, omega):
        """This rank's rows of the interpolated interaction for `omega`
        (pyscf's `range_coulomb` convention), built on first request --
        every rank at the same point, since Z's rows read every rank's M^T.

        G_t = M^T_t V, V in fixed slabs of auxiliary shells, each rank its
        own; dense, Z_t = G_t M^T with M^T's tiles broadcast in tile order.
        """
        key = '%.6f' % (0.0 if omega is None else omega)
        if key in self.omega_kernels:
            return self.omega_kernels[key]
        if self.MT is None:
            self.build()
        t0 = time.time()
        with blas_full_pool() as pool:
            kernel = self._interaction_rows(omega)
        self.omega_kernels[key] = kernel
        _spent(self.timings, 'scf_isdf_kernel', t0)
        _ran_on(self.timings, 'scf_isdf_blas_kernel', pool)
        return kernel

    def _interaction_rows(self, omega):
        """G's rows for `omega`, or Z's where they are held: the metric slabs
        libcint's (BLAS at one thread), the GEMMs on the caller's pool."""
        auxmol = self.auxmol
        naux = auxmol.nao_nr()
        aux_loc = auxmol.ao_loc
        G = {t: np.empty((len(mt), naux)) for t, mt in self.MT.items()}
        om = 0.0 if omega is None else float(omega)
        with (range_coulomb(self.mol, auxmol, om) if om
              else contextlib.nullcontext()):
            for s0, s1 in _shell_slabs(aux_loc, ISDF_SCF_METRIC_SLAB):
                c0, c1 = int(aux_loc[s0]), int(aux_loc[s1])
                with blas_single_threaded():
                    slab = auxmol.intor('int2c2e', shls_slice=(s0, s1, 0,
                                                               auxmol.nbas))
                self.held['metric_slab'] = max(self.held.get('metric_slab', 0),
                                               int(slab.nbytes))
                for t, mt in self.MT.items():
                    G[t][:, c0:c1] = np.dot(mt, slab.T)
                del slab
        self._hold('G_tiles', G.values())
        if self._resolve_z_mode() != 'dense':
            return G
        nk = len(self.coords)
        Z = {t: np.empty((len(g), nk)) for t, g in G.items()}
        for j, (j0, j1) in enumerate(self.tiles):
            MT_j = self._column_tile(j, lambda t: self.MT[t], naux)
            for t, g in G.items():
                Z[t][:, j0:j1] = np.dot(g, MT_j.T)
        del G
        self._hold('Z_rows', Z.values())
        return Z

    def _column_tile(self, j, pack, ncol):
        """Tile j of a grid-row array, on every rank: `pack(j)` on its owner,
        broadcast verbatim to the others."""
        j0, j1 = self.tiles[j]
        size = 1 if self.comm is None else self.comm.Get_size()
        owner = j % size
        if (0 if self.comm is None else self.comm.Get_rank()) == owner:
            buf = np.ascontiguousarray(pack(j))
        else:
            buf = np.empty((j1 - j0, ncol))
        t0 = time.time()
        broadcast_rows(buf, owner, self.comm)
        _spent(self.timings, 'scf_isdf_stream', t0)
        self.held['stream_tile'] = max(self.held.get('stream_tile', 0),
                                       int(buf.nbytes))
        return buf

    # -- the interface _DFHF calls ------------------------------------------

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=1e-13, omega=None):
        """The whole J and K of rank 0's density with the operator `omega`,
        from every rank's partials in one reduction. Collective."""
        dms = _lockstep_density(dm, self.comm, self.timings)
        if np.iscomplexobj(dms):
            raise NotImplementedError('ISDF J/K takes real density matrices')
        if self.MT is None:
            self.build()
        factors = _occupied_factors(dms)
        shape = np.shape(dms)
        nao = shape[-1]
        stack = np.asarray(dms).reshape(-1, nao, nao)
        t0 = time.time()
        vj = vk = None
        if with_j:
            t_j = time.time()
            vj = self._coulomb_engine(omega, direct_scf_tol)(
                np.ascontiguousarray(stack))
            _spent(self.timings, 'scf_fock_j', t_j)
        if with_k:
            t_k = time.time()
            vk = self.exchange_partial(stack, factors, omega)
            _spent(self.timings, 'scf_fock_k', t_k)
            _counted(self.timings, 'scf_requests_k')
        _spent(self.timings, 'scf_fock_jk', t0)
        _counted(self.timings, 'scf_requests_jk')
        t0 = time.time()
        out = iter(_reduce_parts([x for x in (vj, vk) if x is not None],
                                 self.comm))
        _spent(self.timings, 'scf_reduce', t0)
        vj = next(out).reshape(shape) if with_j else None
        vk = next(out).reshape(shape) if with_k else None
        return vj, vk

    def _coulomb_engine(self, omega, direct_scf_tol):
        """The integral-direct J for `omega`, built on first request."""
        key = '%.6f' % (0.0 if omega is None else omega)
        if key not in self._coulomb:
            self._coulomb[key] = _DirectCoulomb(
                self.mol, self.auxbasis, omega, direct_scf_tol, self.comm)
        return self._coulomb[key]

    def exchange_partial(self, dms, factors, omega=None):
        """This rank's partial of K for a stack of densities: its row tiles'
        X_i^T sum_j [(Y_i Y_j^T) o Z_ij] X_j, the column tiles in order.

        factors: the occupied factors C~ (Dm = C~ C~^T) per density, or None
            for the general form Y = X Dm, the column side X itself.
        """
        with blas_full_pool() as pool:
            vk = self._exchange_rows(dms, factors, omega)
        _ran_on(self.timings, 'scf_isdf_blas_k', pool)
        return vk

    def _exchange_rows(self, dms, factors, omega):
        """`exchange_partial` on the caller's BLAS pool."""
        kernel = self.interaction(omega)
        dense = self.z_mode == 'dense'
        nset, nao = len(dms), dms.shape[-1]
        naux = self.auxmol.nao_nr()
        if factors is None:
            Y = {t: [np.dot(x, dm) for dm in dms] for t, x in self.X.items()}
        else:
            Y = {t: [np.dot(x, c) for c in factors] for t, x in self.X.items()}
        widths = ([] if factors is None else [c.shape[1] for c in factors])
        cols = np.cumsum([0] + widths + [nao] + ([] if dense else [naux]))

        def pack(j):
            parts = ([] if factors is None else Y[j]) + [self.X[j]]
            return np.hstack(parts + ([] if dense else [self.MT[j]]))

        T = {t: [np.zeros((len(x), nao)) for _ in range(nset)]
             for t, x in self.X.items()}
        self._hold('Y_tiles', [y for ys in Y.values() for y in ys])
        self._hold('T_rows', [a for ts in T.values() for a in ts])
        for j, (j0, j1) in enumerate(self.tiles):
            buf = self._column_tile(j, pack, int(cols[-1]))
            X_j = buf[:, cols[nset if factors is not None else 0]:
                      cols[(nset if factors is not None else 0) + 1]]
            for i in self.mine:
                Z_ij = (kernel[i][:, j0:j1] if dense
                        else np.dot(kernel[i], buf[:, cols[-2]:].T))
                for s in range(nset):
                    right = (X_j if factors is None
                             else buf[:, cols[s]:cols[s + 1]])
                    A = np.dot(Y[i][s], right.T)
                    A *= Z_ij
                    T[i][s] += np.dot(A, X_j)
                self.held['A_block'] = max(self.held.get('A_block', 0),
                                           int(Z_ij.nbytes))
            del buf
        vk = np.zeros((nset, nao, nao))
        for i in self.mine:
            for s in range(nset):
                vk[s] += np.dot(self.X[i].T, T[i][s])
        return vk

    # -- bookkeeping ---------------------------------------------------------

    def held_bytes(self):
        """{array: bytes} this rank holds now, read off the arrays: the X and
        M^T tiles and every operator's interaction rows."""
        out = {'coords': 0 if self.coords is None else int(self.coords.nbytes),
               'X_tiles': sum(int(a.nbytes) for a in (self.X or {}).values()),
               'MT_tiles': sum(int(a.nbytes)
                               for a in (self.MT or {}).values())}
        for key, rows in self.omega_kernels.items():
            out[f'interaction_{key}'] = sum(int(a.nbytes)
                                            for a in rows.values())
        return out

    def get_naoaux(self):
        return 0 if self.coords is None else len(self.coords)

    def reset(self, mol=None):
        super().reset(mol)
        if mol is not None:
            self.release()
        return self

    def loop(self, blksize=None):
        raise NotImplementedError(
            'DistributedISDFJK stores no three-index tensor, and a slice of '
            'one would be read as the whole: it answers J and K only.')

    # -- installation --------------------------------------------------------

    def install(self, mf):
        """Answer this mean field's J/K until `uninstall`."""
        if mf.with_df is not self:
            self._replaced = (mf, mf.with_df)
            mf.with_df = self
        return self

    def uninstall(self):
        """Put the mean field's own ISDFJK back, KEEPING the tiles: a
        downstream reader of `mf.with_df` (a gradient, the factors a GW run
        takes from the SCF) would read one rank's tiles as the grid."""
        pair, self._replaced = self._replaced, None
        if pair is not None:
            mf, with_df = pair
            mf.with_df = with_df
        return self

    def release(self):
        """Drop this rank's tiles and interaction rows."""
        self.uninstall()
        self.X = self.MT = None
        self.omega_kernels, self._coulomb = {}, {}
        self.z_mode = None
        return self


def _shell_slabs(ao_loc, width):
    """(s0, s1) runs of consecutive shells of at most `width` functions (one
    shell where a shell alone is wider): fixed by the basis, so a slab is the
    same GEMM at every rank count."""
    out, s0 = [], 0
    for s in range(1, len(ao_loc)):
        if ao_loc[s] - ao_loc[s0] > width and s - 1 > s0:
            out.append((s0, s - 1))
            s0 = s - 1
    out.append((s0, len(ao_loc) - 1))
    return out


def _ran_on(timings, key, threads):
    """`key`: the fewest BLAS threads a stage has run on, 0 unread."""
    if timings is not None:
        threads = int(threads or 0)
        timings[key] = min(timings.get(key, threads), threads)


def _occupied_factors(dms):
    """C~ with Dm = C~ C~^T per density, C~ = C[:, n > 0] sqrt(n), from the
    orbitals pyscf tags a density with; None where the tag is missing or an
    occupation is negative, which leaves the general form."""
    mo_coeff = getattr(dms, 'mo_coeff', None)
    mo_occ = getattr(dms, 'mo_occ', None)
    if mo_coeff is None or mo_occ is None:
        return None
    nao = np.shape(dms)[-1]
    nset = int(np.prod(np.shape(dms)[:-2], dtype=int))
    mo_coeff = np.asarray(mo_coeff, dtype=float).reshape(nset, nao, -1)
    mo_occ = np.asarray(mo_occ, dtype=float).reshape(nset, -1)
    if np.any(mo_occ < 0):
        return None
    return [np.ascontiguousarray(c[:, n > 0] * np.sqrt(n[n > 0]))
            for c, n in zip(mo_coeff, mo_occ)]


def distributed_isdf_jk(mf, comm=None, timings=None, tile=None, build=True):
    """Give an ISDF mean field a `with_df` holding this rank's tiles.

    Returns the handle, installed, or None on a one-rank world, where `mf` is
    left exactly as it came in -- its own `ISDFJK`, the serial path,
    untouched. `uninstall()` puts the ISDFJK back while the tiles stay.
    build: False leaves the build (collective) to the first J/K request, so
    that a distributed SCF started next times it as its own stage.

    The build runs on the caller's BLAS pool, never one dropped to a thread
    -- inside `blas_single_threaded`, the pool that wrap took: the row fit
    is GEMM-bound and holds only its libcint passes at one thread itself,
    and a BLAS's bits follow its thread count, so tiles fitted at another
    count than a one-rank handle's are not its tiles -- at 16 threads a rank
    K moved by 1e2 of its reassociation response.
    """
    comm = current_comm() if comm is None else comm
    if comm is None or comm.Get_size() == 1:
        return None
    with_df = getattr(mf, 'with_df', None)
    if isinstance(with_df, DistributedISDFJK):
        return with_df
    dist = DistributedISDFJK(with_df, comm=comm, timings=timings, tile=tile)
    dist.install(mf)
    if build:
        dist.build()
    return dist


def distributed_isdf_storage(mf, comm=None):
    """What one rank holds of the ISDF SCF's factors, in bytes, read off the
    handle a distributed SCF left on `mf` (`held_bytes`, `held`, the fit's
    `fit_held`), with the whole arrays' sizes beside them; None where the
    mean field carries no such handle."""
    comm = current_comm() if comm is None else comm
    handles = getattr(mf, '_distributed', None)
    if handles is None or not isinstance(handles[0], DistributedISDFJK):
        return None
    dist = handles[0]
    nk, nao = len(dist.coords), dist.mol.nao_nr()
    naux = dist.auxmol.nao_nr()
    return dict(ranks=1 if comm is None else comm.Get_size(), M=nk, nao=nao,
                naux=naux, tile=dist.tile, tiles=len(dist.tiles),
                tiles_here=len(dist.mine),
                rows_here=sum(len(x) for x in dist.X.values()),
                z_mode=dist.z_mode,
                held_now=dist.held_bytes(), held_peak=dict(dist.held),
                fit_held=dict(dist.fit_held),
                whole=dict(X=nk * nao * 8, MT=nk * naux * 8, Z=nk * nk * 8))
