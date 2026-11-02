"""The mean-field skeleton derivatives of a nuclear gradient in fixed tiles
over the ranks: the fitted Fock skeleton by auxiliary tiles, the interpolated
exchange skeleton by the row fit's grid-row tiles.

A correlated force on a fitted or interpolated mean field
needs the skeleton (coefficients held fixed) of three two-electron terms: the
folded Fock partial Tr[g F] of the relaxed density, the Sigma_x - v_xc
correction of the quasiparticle energies, and, on an ISDF-K reference, the
mean field's own exchange force. Formed whole, each holds (mu nu|P) and its
adjoint as (nao, nao, naux) arrays on every rank, and the interpolated one
(M, M) arrays and the whole fit's adjoint beside them, whatever the rank
count: nao^2 naux and M^2 doubles, hundreds of GB each on a system of a few
hundred atoms.

Fitted Fock skeleton. With (ab|cd) = sum_PQ J[ab,P] Vinv[P,Q] J[cd,Q] and
the folded density Gamma_abcd = g_ab D_cd - g_ac D_bd / 2 (D = Cd Cd^T), every
three-centre term reads the adjoint on J only symmetrized in its AO pair,
S = Jbar + Jbar^T:

    sum_{m n P} S[m,n,P] (grad_m m n|P) + (1/2) S[m,n,P] (m n|grad_P P)
    S_c = 2 (g (V^-1 b)_P + D (V^-1 a)_P),    a_P = g.J^P,  b_P = D.J^P
    S_x = -w (g Kt^P Cd^T + Cd Kt^P^T g),      Kt^P = sum_Q Vinv_PQ J^Q Cd

with the two-centre term of Vbar = -sym((V^-1 a)(V^-1 b)^T) + (w/2) Vinv T
Vinv, T_PQ = tr[g J^P D J^Q] (the Coulomb half, and the full-range exchange
at weight w). The auxiliary index is cut into fixed tiles of consecutive
shells (`SKELETON_AUX_TILE` functions), tile t rank t % size's, and inside a
tile the first AO index into slabs whose derivative block stays under
`THREE_CENTER_BLOCK_BYTES`: each (slab, tile) block of (mn|P), its two
derivative integrals and S is made, contracted and dropped. Pass 1 gives a
tile's rows of a, b and L^P = J^P Cd; a and b are gathered verbatim and
solved on every rank (locked to rank 0's); for the exchange, L moves in one
exchange from the auxiliary tiles to fixed blocks of whole occupied orbitals
(`SKELETON_EXCHANGE_OCC_BLOCK`, a rank's blocks one contiguous run), where
Kt = V^-1 L is solved on the metric's Cholesky factor V = Lc Lc^T, At = Kt g
is formed and each block's (naux, naux) term of Vinv T Vinv =
<g Kt^P, Kt^Q> is taken; the terms add in block order (`ordered_chain_sum`)
and At moves back to the tiles in a second exchange. Neither V^-1 nor the
product Vinv T Vinv is
formed: on a near-singular metric (DMABN/aug-cc-pVDZ-JKFIT, smallest
eigenvalue 7e-8) they lose digits that the solves keep, about 1e-6 Ha/Bohr
of a force. Pass 2 is each tile's (natm, 3) addend: its AO-slab
contractions, its auxiliary centres and its rows of the two-centre
derivative. Every tile's (natm, 3) addend reaches every rank and is added in
tile order (`ordered_sum`), so the skeleton is the one-rank bits at every
rank count.

Interpolated exchange. E = -(p/4) sum_w w sum_PQ Z^w_PQ W1_PQ W2_PQ with
Z^w = M^T V_w M and W = X dm X^T. Every channel shares M, so the adjoints
collapse onto Vsum = sum_w w V_w and H = W1 o W2:

    MT_bar = -(p/2) (H M^T) Vsum,    V_bar^w = -(p w/4) M H M^T,
    X_bar  = -(p/2) [(Zs o W2) X dm + (Zs o W1) X dm_other],  Zs = M^T Vsum M

In the row fit's tiles (tile t rank t % size's) each column tile's owner
broadcasts [X_j | M^T_j | X_j dm | X_j dm_other] in tile order; every rank
forms the (tile, tile) blocks of W1, W2 and Zs for its row tiles and
accumulates A_i = sum_j H_ij M^T_j and X_bar_i; M H M^T = sum_i M^T_i^T A_i
reaches rank 0 tile by tile in tile order. `isdf_fit_adjoint.fit_rows_adjoint`
carries MT_bar and X_bar to the nuclei in the same tiles; `isdf_exchange_seeds`
stops before it, so a force contracts every skeleton's seeds with its own in
one `isdf_fit_adjoint.fit_rows_adjoints` call. Nothing is summed across ranks, so
the result is the same bits at every rank count, one rank included. The
estimator is the row fit's, the one the distributed ISDF-K SCF minimized; its
M^T and collocation tiles are reused when they are handed in. The AO-product
test set takes the same seeds on its own row fit and adjoint
(`ao_product_fit_rows`, `ao_product_fit_rows_adjoints`), so no rank holds an
(M, M) array there either.

XC skeleton on a moving grid. T = Tr[g v_xc[D]] = sum_k w_k f(r_k) with
f = sum_i v_i rho^g_i (v = de/drho_i; rho_i = rho, grad rho, tau) is the
g-directional derivative of E_xc[D], and the energy's Becke grid moves with
the atoms: each point r_k rides with its owner o(k), and its weight
w_k = vol_k P_o(r_k) / sum_b P_b(r_k) changes with every atom. So

    dT/dR_A = sum_k w_k d_A f(r_k)|_grid fixed          (the AOs move)
            + sum_{k: o(k)=A} w_k grad_r f(r_k)          (the points move)
            + sum_k (dw_k/dR_A) f(r_k)                   (the weights move)

where d_A f needs the kernel, u_j = sum_i f_ij rho^g_i, for the density's
response. The AO translation identity grad_r f = -sum_B d_B f turns the
second line into minus the first line's atom sum over the owner's points, so
the three terms together sum to zero over atoms, point by point. The weights'
derivative is Becke's cell function differentiated as Johnson, Gill and
Pople (J. Chem. Phys. 98, 5612 (1993), appendix B) with every pair's three
moving centres, contracted with w_k f_k without forming dw/dR. The mean
field's own sorted grid is cut into fixed runs of `XC_SKELETON_TILE` points,
tile t rank t % size's; each gives one (natm, 3) addend, the addends are
gathered verbatim and summed in tile order on every rank, so the result is
the same bits at every rank count, one rank included.

The mean field's own force. At fixed D the energy of a `j_route='df-direct'`
SCF differentiates term by term, each in fixed tiles gathered verbatim
(`tile_sum`): Tr[D dh] - Tr[W dS] by atoms, E_J = Tr[D J[D]] / 2 as the
Coulomb half of the fitted skeleton at g = D by auxiliary tiles, and E_xc on
the moving grid by grid tiles, the three terms above with f = e_xc and v in
place of the kernel (the energy is first order in rho). A density-fitted
SCF's exact exchange is the same skeleton's exchange half at g = D, one
`FittedFockSkeleton` per operator: the bare one, and a range-separated
hybrid's erf(omega r)/r on its own attenuated fit, whose singular metric is
solved on the eigenvalue pseudo-inverse as pyscf's own gradient solves it
(`fitted_energy_skeleton`).
"""
import ctypes

import numpy as np
import scipy.linalg
from pyscf import gto, lib
from pyscf.df import incore
from pyscf.df.incore import LINEAR_DEP_THR
from pyscf.dft import gen_grid, numint, radi
from pyscf.scf import _vhf

from src.Base.numerics.constants import (ISDF_TEST_SETS,
                                         SKELETON_AUX_TILE, SKELETON_EXCHANGE_OCC_BLOCK,
                                         SKELETON_EXCHANGE_TILE,
                                         THREE_CENTER_BLOCK_BYTES, XC_SKELETON_TILE)
from src.Base.factorization.ao_product_fit import ao_product_fit_rows
from src.Base.factorization.separable_ri import (Held, fit_rows, row_layout,
                                                 tile_collocation, tile_edge)
from src.Base.factorization.shell_runs import shell_runs
from src.gradients.isdf_fit_adjoint import (AdjointSeeds, tiles_at_root, two_centre_rows_adjoint,
                                            fit_rows_adjoint)
from src.Base.utils.parallel.mpi_grid import (allgather_ranges, broadcast_rows,
                                              contiguous_block, current_comm,
                                              exchange_blocks, lockstep,
                                              ordered_chain_sum, ordered_sum,
                                              partition)


class FittedFockSkeleton:
    """The fitted Fock skeleton of one folded partial g over auxiliary tiles.

    `prepare(tiles, comm)` runs pass 1 on the tiles this rank owns and the
    collectives after it; `addend(t)` is then tile t's (natm, 3), the same
    bits whichever rank computes it. `fitted_fock_skeleton` is the sum.

    g_ao, dm: the folded partial and the SCF density, AO basis, symmetric.
    occ: Cd with dm = Cd Cd^T, needed only for the exchange.
    coulomb: whether the Coulomb half is taken; exchange: the summed weight
    of the exchange channels of operator `omega` (0 for none).
    omega: the operator, 0 the bare Coulomb one and a nonzero value the
    erf(omega r)/r of a range-separated hybrid, on the three- and
    two-centre integrals alike (pyscf's `range_coulomb` fit).

    The metric is solved as pyscf's density-fitted gradient solves it
    (`pyscf/df/grad/rhf.py::_gen_metric_solver`): through its Cholesky
    factor, and where that factorization fails -- an attenuated metric is
    singular -- through the pseudo-inverse of its eigenvalues above
    LINEAR_DEP_THR (`metric` says which, rank 0's choice on every rank).
    The pseudo-inverse is applied as its eigenvector and eigenvalue factors
    and never formed: its entries reach 1/LINEAR_DEP_THR, and a product
    with it turns a last-bit change of the density into ~1e-8 Ha/Bohr of
    the force.
    """

    def __init__(self, mol, auxmol, g_ao, dm, occ=None, coulomb=True,
                 exchange=0.0, tile=None, max_bytes=THREE_CENTER_BLOCK_BYTES,
                 omega=0.0):
        self.omega = float(omega)
        if self.omega:
            # private copies: the attenuation lives in the molecules' env
            mol, auxmol = mol.copy(), auxmol.copy()
            mol.omega = auxmol.omega = self.omega
        self.mol, self.auxmol = mol, auxmol
        self.metric = None
        self.g = np.ascontiguousarray(g_ao, dtype=float)
        self.D = np.ascontiguousarray(dm, dtype=float)
        self.coulomb, self.w = bool(coulomb), float(exchange)
        if self.w != 0.0 and occ is None:
            raise ValueError('the exchange half needs the occupied factor Cd '
                             'of the density, dm = Cd Cd^T')
        self.Cdt = (None if occ is None
                    else np.ascontiguousarray(np.asarray(occ, float).T))
        self.max_bytes = int(max_bytes)
        self.tiles = shell_runs(auxmol.ao_loc_nr(),
                                SKELETON_AUX_TILE if tile is None else tile)
        self.aux_loc = auxmol.ao_loc_nr()
        self.ao_loc = mol.ao_loc_nr()
        self.ao_slices = [(int(p0), int(p1))
                          for _, _, p0, p1 in mol.aoslice_by_atom()]
        self.aux_atom = np.empty(auxmol.nao_nr(), dtype=int)
        for ia, (_, _, q0, q1) in enumerate(auxmol.aoslice_by_atom()):
            self.aux_atom[q0:q1] = ia
        self.Va = self.Vb = None
        self.At, self.VTV = {}, {}
        #: {array: most bytes of it this rank held at once}
        self.held = {}

    def _hold(self, name, nbytes):
        self.held[name] = max(self.held.get(name, 0), int(nbytes))

    def functions(self, t):
        """[p0, p1) of tile t's auxiliary functions."""
        k0, k1 = self.tiles[t]
        return int(self.aux_loc[k0]), int(self.aux_loc[k1])

    def slabs(self, t, comp):
        """AO shell slabs of tile t's `comp`-component block under the cap."""
        p0, p1 = self.functions(t)
        per_ao = comp * self.mol.nao_nr() * (p1 - p0) * 8
        return shell_runs(self.ao_loc, self.max_bytes // int(per_ao))

    def _block(self, intor, comp, s0, s1, t):
        """(comp, a, nao, n_t) of `intor` on AO shells [s0, s1) and tile t."""
        k0, k1 = self.tiles[t]
        nb = self.mol.nbas
        out = incore.aux_e2(self.mol, self.auxmol, intor=intor, aosym='s1',
                            comp=comp, shls_slice=(s0, s1, 0, nb, k0, k1))
        self._hold('integral_block', out.nbytes)
        return out.reshape(comp, int(self.ao_loc[s1] - self.ao_loc[s0]),
                           self.mol.nao_nr(), -1)

    def prepare(self, owned, comm=None):
        """Pass 1 on the `owned` tiles, then the gathers, the metric solves
        and the exchange's column blocks: every rank calls it,
        collectively."""
        rank, size = ((0, 1) if comm is None
                      else (comm.Get_rank(), comm.Get_size()))
        owners = [[int(t) for t in partition(len(self.tiles), r, size)]
                  for r in range(size)]
        nao, naux = self.mol.nao_nr(), self.auxmol.nao_nr()
        a, b = np.zeros(naux), np.zeros(naux)
        L = {}
        nocc = 0 if self.Cdt is None else self.Cdt.shape[0]
        for t in owned:
            p0, p1 = self.functions(t)
            if self.w:
                L[t] = np.empty((p1 - p0, nocc, nao))
            for s0, s1 in self.slabs(t, 1):
                a0, a1 = int(self.ao_loc[s0]), int(self.ao_loc[s1])
                JT = self._block('int3c2e', 1, s0, s1, t)[0].transpose(2, 1, 0)
                if self.coulomb:
                    flat = JT.reshape(p1 - p0, -1)
                    a[p0:p1] += flat @ np.ascontiguousarray(
                        self.g[:, a0:a1]).ravel()
                    b[p0:p1] += flat @ np.ascontiguousarray(
                        self.D[:, a0:a1]).ravel()
                if self.w:
                    # L^P as (nocc, nao) = (J^P Cd)^T, the slab's columns
                    L[t][:, :, a0:a1] = np.matmul(self.Cdt, JT)
                del JT
        self._hold('L_tiles', sum(x.nbytes for x in L.values()))
        ranges = [[self.functions(t) for t in own] for own in owners]
        V = self.auxmol.intor('int2c2e', aosym='s1')
        # pyscf's rule: the Cholesky factor where it exists, else the
        # eigenvectors U and eigenvalues e above LINEAR_DEP_THR, the
        # pseudo-inverse U e^-1 U^T applied as three factors and never
        # formed; rank 0's on every rank. The bare Coulomb half alone solves
        # V directly and needs neither.
        Lc = U = e = None
        if self.w or self.omega:
            try:
                Lc = scipy.linalg.cholesky(V, lower=True)
            except np.linalg.LinAlgError:
                w, v = scipy.linalg.eigh(V)
                keep = w > LINEAR_DEP_THR
                U, e = np.ascontiguousarray(v[:, keep]), w[keep]
            Lc, U, e = lockstep((Lc, U, e), comm, check=True)
        self.metric = 'eigen' if U is not None else 'cholesky'
        if self.coulomb:
            allgather_ranges(a, ranges, comm)
            allgather_ranges(b, ranges, comm)
            # each rank's own solve, rank 0's the one every addend reads
            self.Va, self.Vb = lockstep(
                (np.linalg.solve(V, a), np.linalg.solve(V, b)) if U is None
                else (U @ ((U.T @ a) / e), U @ ((U.T @ b) / e)), comm,
                check=True)
        if not self.w:
            return self
        del V
        # Kt = V^-1 L acts on each column of L alone, and At^P = Kt^P g mixes
        # the columns of one occupied orbital only, so the solve runs on
        # fixed blocks of whole occupied orbitals (`occupied_column_blocks`),
        # a rank's blocks one contiguous run: one exchange moves L from the
        # auxiliary tiles to the blocks, each block is solved, multiplied by
        # g and contracted into its (naux, naux) term of <g Kt^P, Kt^Q> on
        # its own, the terms add in block order (`ordered_chain_sum`), and
        # one exchange moves At back to the tiles. Each byte of L and At
        # crosses once, where a stream of every tile to every rank moved the
        # whole tensor to each rank three times. Every block is the same
        # arithmetic on whichever rank holds it, so the skeleton is the same
        # bits at every rank count.
        blocks = occupied_column_blocks(nocc)
        block_owner = [r for r in range(size)
                       for _ in range(*contiguous_block(len(blocks), r, size))]
        runs = [contiguous_block(len(blocks), r, size) for r in range(size)]
        cols = [(blocks[b0][0], blocks[b1 - 1][1]) if b1 > b0 else (0, 0)
                for b0, b1 in runs]
        rows = [[self.functions(t) for t in own] for own in owners]
        nrows = [sum(p1 - p0 for p0, p1 in own) for own in rows]
        i0, i1 = cols[rank]
        mine = i1 - i0
        # tiles to blocks: this rank's tiles' rows, each rank's columns
        stacked = (np.concatenate([L.pop(t) for t in owned]) if owned
                   else np.empty((0, nocc, nao)))
        got = exchange_blocks(
            [np.ascontiguousarray(stacked[:, c0:c1, :]).reshape(
                len(stacked), (c1 - c0) * nao) for c0, c1 in cols],
            [(nrows[r], mine * nao) for r in range(size)], comm)
        stacked = None
        Lb = np.empty((naux, mine, nao))
        for r in range(size):
            at = 0
            for p0, p1 in rows[r]:
                Lb[p0:p1] = got[r][at:at + p1 - p0].reshape(p1 - p0, mine, nao)
                at += p1 - p0
        got = None
        self._hold('L_columns', Lb.nbytes)
        At = np.empty_like(Lb)
        terms = {}
        for b in range(*runs[rank]):
            b0, b1 = blocks[b][0] - i0, blocks[b][1] - i0
            X = Lb[:, b0:b1, :].reshape(naux, -1)
            if U is None:
                # V = Lc Lc^T, by substitution, never by an explicit inverse:
                # on a near-singular metric the inverse's product loses
                # digits that the solves keep
                Kb = scipy.linalg.solve_triangular(Lc, X, lower=True)
                Kb = scipy.linalg.solve_triangular(Lc, Kb, lower=True, trans='T')
            else:
                # the pseudo-inverse as its factors, U (e^-1 (U^T L)): the
                # explicit one carries round-off of 1/LINEAR_DEP_THR into
                # every direction of Kt
                Kb = U @ ((U.T @ X) / e[:, None])
            Kb = Kb.reshape(naux, b1 - b0, nao)
            # (g Kt^P), the only form of Kt pass 2 reads
            At[:, b0:b1, :] = np.matmul(Kb, self.g)
            terms[b] = (At[:, b0:b1, :].reshape(naux, -1)
                        @ Kb.reshape(naux, -1).T)
            del Kb, X
        Lb = None
        self._hold('VTV_terms', sum(x.nbytes for x in terms.values()))
        # Vinv T Vinv = <g Kt^P, Kt^Q>: the product of the inverses would lose
        # digits as the square of the metric's condition number
        VTV = ordered_chain_sum(terms, block_owner, comm, np.zeros((naux, naux)))
        terms = None
        for t in owned:
            p0, p1 = self.functions(t)
            self.VTV[t] = np.ascontiguousarray(VTV[p0:p1])
        VTV = None
        # blocks back to tiles: each rank's tiles' rows of every rank's columns
        back = exchange_blocks(
            [np.ascontiguousarray(np.concatenate(
                [At[p0:p1] for p0, p1 in rows[r]]) if rows[r]
                else np.empty((0, mine, nao))).reshape(nrows[r], mine * nao)
             for r in range(size)],
            [(nrows[rank], (cols[r][1] - cols[r][0]) * nao) for r in range(size)],
            comm)
        At = None
        at = 0
        for t in owned:
            p0, p1 = self.functions(t)
            self.At[t] = np.empty((p1 - p0, nocc, nao))
            for r in range(size):
                c0, c1 = cols[r]
                self.At[t][:, c0:c1, :] = back[r][at:at + p1 - p0].reshape(
                    p1 - p0, c1 - c0, nao)
            at += p1 - p0
        back = None
        self._hold('At_tiles', sum(x.nbytes for x in self.At.values()))
        return self

    def _pair_adjoint(self, t, a0, a1):
        """S[m, n, P] of tile t for m in [a0, a1), laid out [P, n, m]."""
        p0, p1 = self.functions(t)
        nao = self.mol.nao_nr()
        ST = np.zeros((p1 - p0, nao, a1 - a0))
        if self.coulomb:
            ST += self.g[None, :, a0:a1] * (2.0 * self.Vb[p0:p1, None, None])
            ST += self.D[None, :, a0:a1] * (2.0 * self.Va[p0:p1, None, None])
        if self.w:
            At = self.At[t]
            x = np.matmul(At.transpose(0, 2, 1), self.Cdt[:, a0:a1])
            x += np.matmul(self.Cdt.T, At[:, :, a0:a1])
            x *= self.w
            ST -= x
            del x
        self._hold('pair_adjoint', ST.nbytes)
        return ST

    def _metric_rows(self, t):
        """Tile t's rows of the symmetric two-centre adjoint Vbar."""
        p0, p1 = self.functions(t)
        rows = np.zeros((p1 - p0, self.auxmol.nao_nr()))
        if self.coulomb:
            rows -= 0.5 * (self.Va[p0:p1, None] * self.Vb[None, :]
                           + self.Vb[p0:p1, None] * self.Va[None, :])
        if self.w:
            rows += (0.5 * self.w) * self.VTV[t]
        return rows

    def addend(self, t):
        """(natm, 3): tile t's share of the skeleton, pass 2."""
        mol, auxmol = self.mol, self.auxmol
        nao = mol.nao_nr()
        p0, p1 = self.functions(t)
        t_ao = np.zeros((3, nao))
        t_aux = np.zeros((3, p1 - p0))
        for s0, s1 in self.slabs(t, 3):
            a0, a1 = int(self.ao_loc[s0]), int(self.ao_loc[s1])
            ST = self._pair_adjoint(t, a0, a1)
            d = self._block('int3c2e_ip1', 3, s0, s1, t)
            for x in range(3):
                # +(grad m n|P): the nuclear derivative carries the minus
                t_ao[x, a0:a1] = -np.einsum('Pnm,Pnm->m',
                                            d[x].transpose(2, 1, 0), ST)
            del d
            d = self._block('int3c2e_ip2', 3, s0, s1, t)
            for x in range(3):
                t_aux[x] -= 0.5 * np.einsum('Pnm,Pnm->P',
                                            d[x].transpose(2, 1, 0), ST)
            del d, ST
        k0, k1 = self.tiles[t]
        v1 = auxmol.intor('int2c2e_ip1', comp=3,
                          shls_slice=(k0, k1, 0, auxmol.nbas))
        t_aux -= 2.0 * np.einsum('xPQ,PQ->xP', v1, self._metric_rows(t))
        del v1
        out = np.zeros((mol.natm, 3))
        for ia, (q0, q1) in enumerate(self.ao_slices):
            out[ia] += t_ao[:, q0:q1].sum(axis=1)
        np.add.at(out, self.aux_atom[p0:p1], t_aux.T)
        return out


def occupied_column_blocks(nocc):
    """(i0, i1) blocks of SKELETON_EXCHANGE_OCC_BLOCK consecutive occupied
    orbitals: the column blocks of the fitted exchange skeleton's solve,
    fixed by nocc, never by the rank count."""
    w = SKELETON_EXCHANGE_OCC_BLOCK
    return [(i, min(i + w, int(nocc))) for i in range(0, int(nocc), w)]


def fitted_fock_skeleton(mol, auxmol, g_ao, dm, occ=None, coulomb=True,
                         exchange=0.0, comm=None, tile=None):
    """(natm, 3) fitted skeleton of Tr[g F] (Coulomb half, full-range exchange
    at weight `exchange`), the same bits on every rank and at every rank
    count: a `FittedFockSkeleton` over this rank's tiles, its addends added
    in tile order (`ordered_sum`)."""
    comm = current_comm() if comm is None else comm
    rank, size = ((0, 1) if comm is None
                  else (comm.Get_rank(), comm.Get_size()))
    if not coulomb and exchange == 0.0:
        return np.zeros((mol.natm, 3))
    if size > 1:
        g_ao, dm, occ = lockstep((g_ao, dm, occ), comm, check=True)
    skeleton = FittedFockSkeleton(mol, auxmol, g_ao, dm, occ=occ,
                                  coulomb=coulomb, exchange=exchange,
                                  tile=tile)
    mine = [int(t) for t in partition(len(skeleton.tiles), rank, size)]
    skeleton.prepare(mine, comm)
    # every tile's (natm, 3) addend, added in tile order: the one-rank bits
    return ordered_sum([(t, skeleton.addend(t)) for t in mine], comm,
                       onto=np.zeros((mol.natm, 3)))


def _derivative_exchange_screen(mol):
    """pyscf's prescreening of the derivative J/K (`pyscf.grad.rhf.get_jk`),
    built the same way, for a direct pass over a shell slice."""
    libcvhf = _vhf.libcvhf
    vhfopt = _vhf._VHFOpt(mol, 'int2e_ip1', 'CVHFgrad_jk_prescreen',
                          dmcondname='CVHFnr_dm_cond1')
    ao_loc = mol.ao_loc_nr()
    nbas = mol.nbas
    q_cond = np.empty((2, nbas, nbas))
    with mol.with_integral_screen(vhfopt.direct_scf_tol ** 2):
        for out, intor, cond in ((q_cond[0], 'int2e_ip1ip2', libcvhf.CVHFnr_int2e_pp_q_cond),
                                 (q_cond[1], 'int2e', libcvhf.CVHFnr_int2e_q_cond)):
            cond(getattr(libcvhf, mol._add_suffix(intor)), lib.c_null_ptr(),
                 out.ctypes, ao_loc.ctypes, mol._atm.ctypes, ctypes.c_int(mol.natm),
                 mol._bas.ctypes, ctypes.c_int(nbas), mol._env.ctypes)
    vhfopt.q_cond = q_cond
    return vhfopt


def derivative_exchange_rows(mol, dms, shells, vhfopt=None):
    """(ndm, 3, rows, nao): the K half of `pyscf.grad.rhf.get_jk`, signs
    included, for every density of `dms` in ONE direct pass over the
    four-centre derivative integrals, the rows of the AO shells
    [shells[0], shells[1]) only. The integrals and their screening are
    pyscf's; J is not formed. vhfopt: the screening of
    `_derivative_exchange_screen`, built once by a caller passing several
    slices."""
    sh0, sh1 = shells
    nbas = mol.nbas
    vk = _vhf.direct_mapdm(mol._add_suffix('int2e_ip1'), 's2kl', 'jk->s1il',
                           np.asarray(dms, float), 3, mol._atm, mol._bas, mol._env,
                           vhfopt=(_derivative_exchange_screen(mol) if vhfopt is None
                                   else vhfopt),
                           shls_slice=(sh0, sh1, 0, nbas, 0, nbas, 0, nbas))
    return -np.asarray(vk)


def exchange_skeleton_tiles(mol):
    """(s0, s1) runs of consecutive shells of at most SKELETON_EXCHANGE_TILE
    functions: the tiles of the exchange skeleton's derivative pass."""
    return shell_runs(mol.ao_loc_nr(), SKELETON_EXCHANGE_TILE)


def exchange_skeleton_by_atoms(mol, g_ao, D, omega, weight, comm=None):
    """(natm, 3) of weight * d/dR of one four-centre exchange channel
    (`mean_field_response.exchange_channel_skeleton`): both densities through
    one direct pass (`derivative_exchange_rows`) under the attenuation
    `omega`. The pass costs nao^4; its rows are cut into fixed shell tiles
    (`exchange_skeleton_tiles`), tile t rank t % size's over the ranks of
    `comm` (default `current_comm()`, every rank calling), so each rank's rows
    interleave the whole molecule and the ranks finish together (a screened
    row's cost differs between atoms far more than its AO count says); each
    tile gives a (natm, 3) addend, added in tile order (`tile_sum`): the same
    bits at every rank count.
    """
    comm = current_comm() if comm is None else comm
    tiles = exchange_skeleton_tiles(mol)
    ao_loc = mol.ao_loc_nr()
    ao_atom = np.empty(mol.nao_nr(), dtype=int)
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        ao_atom[p0:p1] = ia
    held = {}

    def addend(t):
        sh0, sh1 = tiles[t]
        p0, p1 = int(ao_loc[sh0]), int(ao_loc[sh1])
        with mol.with_range_coulomb(omega):
            if 'screen' not in held:
                held['screen'] = _derivative_exchange_screen(mol)
            vk_D, vk_g = derivative_exchange_rows(mol, [D, g_ao], (sh0, sh1),
                                                  held['screen'])
        rows = -weight * (np.einsum('ac,xac->xa', g_ao[p0:p1], vk_D, optimize=True)
                          + np.einsum('ad,xad->xa', D[p0:p1], vk_g, optimize=True))
        out = np.zeros((mol.natm, 3))
        np.add.at(out, ao_atom[p0:p1], rows.T)
        return out

    return tile_sum(addend, len(tiles), mol.natm, comm)


def isdf_exchange_rows(mol, auxmol, coords, layout, dm, dm_other=None,
                       prefactor=1.0, channels=((0.0, 1.0),), mt=None, X=None,
                       block=None, numerics=None, comm=None,
                       test_set=ISDF_TEST_SETS[0]):
    """(centre, points, held) of d/dR E_K^ISDF at fixed densities on the row
    fit: the (natm, 3) centre terms of every integral and collocation, and
    the (nk, 3) adjoint on the points for the point and frame chain.

    E_K = -(prefactor/4) sum_w w sum_PQ Z^w_PQ (X dm X^T)_PQ (X o X^T)_PQ,
    o = dm_other (dm when None), channels [(omega, w)]. mt, X: this rank's
    tiles of the row fit's M^T and of the collocation, {tile: rows}, the
    SCF's own where it holds them; None fits and collocates here. layout:
    the pair columns the fit screened (`separable_ri.screened_layout`), None
    on the AO-product test set (`test_set` 'ao_products', whose row fit and
    adjoint are `ao_product_fit_rows` and `ao_product_fit_rows_adjoints`).
    `isdf_exchange_seeds`, then the fit adjoint of its seeds alone.
    """
    comm = current_comm() if comm is None else comm
    seeds, two, held = isdf_exchange_seeds(
        mol, auxmol, coords, layout, dm, dm_other=dm_other,
        prefactor=prefactor, channels=channels, mt=mt, X=X, block=block,
        numerics=numerics, comm=comm, test_set=test_set)
    if test_set == 'ao_products':
        # cycle: that module builds on the pieces of this one
        from src.gradients.ao_product_fit_adjoint import (
            ao_product_fit_rows_adjoints)
        adjoint = ao_product_fit_rows_adjoints(
            mol, auxmol, coords, [seeds], numerics=numerics, comm=comm,
            block=block)[0]
    else:
        adjoint = fit_rows_adjoint(mol, auxmol, coords, None, layout,
                                   numerics=numerics, comm=comm,
                                   block=block, mt_bar=seeds.mt_bar,
                                   x_bar_ao=seeds.x_bar_ao,
                                   metric_bar=seeds.metric_bar)
    held.update({f'adjoint_{k}': v for k, v in adjoint.held.items()})
    centre = adjoint.fit_centre + adjoint.coll_centre + two
    return centre, adjoint.fit_points + adjoint.coll_points, held


def isdf_exchange_seeds(mol, auxmol, coords, layout, dm, dm_other=None,
                        prefactor=1.0, channels=((0.0, 1.0),), mt=None,
                        X=None, block=None, numerics=None, comm=None,
                        test_set=ISDF_TEST_SETS[0]):
    """(seeds, centre, held): what d/dR E_K^ISDF of `isdf_exchange_rows`
    hands the row fit's adjoint -- `isdf_fit_adjoint.AdjointSeeds` with MT_bar
    and X_bar on this rank's tiles and the bare metric's V_bar on rank 0 --
    and the (natm, 3) of the attenuated metrics, which reach the nuclei
    outside the fit. The fit adjoint is linear in the seeds, so a force
    gathering several of these contracts their sum once
    (`isdf_fit_adjoint.fit_rows_adjoints`). Without `mt` the row fit of
    `test_set` is made here: `fit_rows` on `layout`, or the AO-product one
    (`ao_product_fit_rows`)."""
    comm = current_comm() if comm is None else comm
    rank, size = ((0, 1) if comm is None
                  else (comm.Get_rank(), comm.Get_size()))
    if size > 1:
        coords, dm, dm_other = lockstep((coords, dm, dm_other), comm,
                                        check=True)
    block = tile_edge(block)
    nao, naux, natm = mol.nao_nr(), auxmol.nao_nr(), mol.natm
    nk = len(coords)
    tiles, owners, _ = row_layout(nk, block, comm)
    mine = owners[rank]
    hold = held = Held()

    if mt is None and test_set == 'ao_products':
        fit = ao_product_fit_rows(mol, auxmol, coords, numerics=numerics,
                                  comm=comm, block=block)
        mt = fit.mt
        held.update({f'fit_{k}': v for k, v in fit.held.items()})
        del fit
    elif mt is None:
        fit = fit_rows(mol, auxmol, coords, numerics=numerics, comm=comm,
                       block=block, layout=layout)
        mt = fit.mt
        held.update({f'fit_{k}': v for k, v in fit.held.items()})
        del fit
    if X is None:
        X = {t: tile_collocation(mol, coords[slice(*tiles[t])])
             for t in mine}
    # one metric per operator, every channel sharing M
    Vsum = np.zeros((naux, naux))
    bare, attenuated, metrics = 0.0, [], {}
    for omega, weight in channels:
        if weight == 0.0:
            continue
        if omega not in metrics:
            aux_w = auxmol
            if omega != 0.0:
                # a copy carries the operator, the caller's molecule none
                aux_w = auxmol.copy()
                aux_w.omega = omega
            metrics[omega] = aux_w.intor('int2c2e', aosym='s1')
        Vsum += weight * metrics[omega]
        if omega == 0.0:
            bare += weight
        else:
            attenuated.append((omega, weight))
    del metrics
    two_density = dm_other is not None
    Y1 = {t: X[t] @ dm for t in mine}
    Y2 = {t: X[t] @ dm_other for t in mine} if two_density else Y1
    G = {t: mt[t] @ Vsum for t in mine}
    A = {t: np.zeros((len(mt[t]), naux)) for t in mine}
    Xb = {t: np.zeros((len(mt[t]), nao)) for t in mine}
    hold('row_tiles', sum(a.nbytes for d in (X, mt, Y1, G, A, Xb)
                          for a in d.values())
         + (sum(a.nbytes for a in Y2.values()) if two_density else 0))
    cols = [0, nao, nao + naux, 2 * nao + naux] + ([3 * nao + naux]
                                                    if two_density else [])
    for j, (j0, j1) in enumerate(tiles):
        owner = j % size
        if rank == owner:
            buf = np.hstack([X[j], mt[j], Y1[j]] + ([Y2[j]] if two_density
                                                    else []))
        else:
            buf = np.empty((j1 - j0, cols[-1]))
        broadcast_rows(buf, owner, comm)
        hold('column_tile', buf.nbytes)
        X_j, MT_j = buf[:, :nao], buf[:, nao:nao + naux]
        Y1_j = buf[:, cols[2]:cols[3]]
        Y2_j = buf[:, cols[3]:cols[4]] if two_density else Y1_j
        for i in mine:
            W1 = Y1[i] @ X_j.T
            W2 = Y2[i] @ X_j.T if two_density else W1
            Zs = G[i] @ MT_j.T
            H = W1 * W2
            A[i] += H @ MT_j
            del H
            Zs2 = Zs * W2
            Xb[i] += Zs2 @ Y1_j
            if two_density:
                Zs *= W1
                Xb[i] += Zs @ Y2_j
            hold('tile_blocks', W1.nbytes * 4)
            del W1, W2, Zs, Zs2
        del buf
    del Y1, Y2, G
    # one density enters twice: its two slots are the same product
    x_scale = -0.5 * prefactor if two_density else -prefactor
    MTb = {}
    for t in mine:
        MTb[t] = A[t] @ Vsum
        MTb[t] *= -0.5 * prefactor
        Xb[t] *= x_scale
    C = np.zeros((naux, naux)) if rank == 0 else None
    for t, got in tiles_at_root([mt, A], tiles, rank, size, comm, hold):
        if got is not None:
            C += got[0].T @ got[1]
    del A
    two = np.zeros(natm * 3)
    for omega, weight in attenuated:
        Vb = None if rank != 0 else (-0.25 * prefactor * weight) * C
        two += two_centre_rows_adjoint(auxmol, Vb, block, rank, size, comm,
                                        hold, omega=omega)
    metric_bar = (None if rank != 0 or bare == 0.0
                  else (-0.25 * prefactor * bare) * C)
    del C
    seeds = AdjointSeeds(mt_bar=MTb, x_bar_ao=Xb, metric_bar=metric_bar)
    return seeds, two.reshape(natm, 3), dict(held)


def xc_grid_tiles(npoints, tile=None):
    """[start, stop) runs of a grid's points in its own order,
    `XC_SKELETON_TILE` each: fixed by the grid, never by the rank count."""
    tile = XC_SKELETON_TILE if tile is None else int(tile)
    return [(p0, min(p0 + tile, npoints)) for p0 in range(0, npoints, tile)]


def becke_adjustment(mol, grids):
    """(natm, natm) a_ij of the grid's cell coordinate nu = mu + a_ij (1 - mu^2)
    (Becke's size adjustment, a = 0 without one), refusing a partition whose
    weights are not Becke's original cell function of that nu."""
    if grids.becke_scheme is not gen_grid.original_becke:
        raise NotImplementedError(
            f'the xc skeleton differentiates the original Becke partition, '
            f'not {grids.becke_scheme!r}')
    adjust = grids.radii_adjust
    if not callable(adjust) or grids.atomic_radii is None:
        return np.zeros((mol.natm, mol.natm))
    if adjust not in (radi.treutler_atomic_radii_adjust,
                      radi.becke_atomic_radii_adjust):
        raise NotImplementedError(
            f'the xc skeleton knows the Treutler and Becke radii adjustments, '
            f'not {adjust!r}')
    f = adjust(mol, grids.atomic_radii)
    # g + a (1 - g^2) at g = 0 is a, the table get_partition hands its C kernel
    return np.array([[f(i, j, 0.0) for j in range(mol.natm)]
                     for i in range(mol.natm)])


def becke_weight_response(mol, coords, owner, phi, adjust):
    """(natm, 3) of sum_k phi_k d ln w_k / dR for the Becke weights of points
    `coords`, point k riding with atom owner[k]; phi_k = w_k f_k gives
    sum_k f_k dw_k/dR.

    ln w = ln P_o - ln sum_b P_b with P_i = prod_{j != i} s(nu_ij), so
    d ln w = sum_{i>j} [lam_ij - (P_i pt_ij + P_j pt_ji) / Z] d mu_ij, where
    pt_ij = d ln s(nu_ij) / d mu_ij, pt_ji = d ln s(nu_ji) / d mu_ij and lam
    keeps the owner's own pairs. mu_ij = (r_i - r_j) / R_ij moves with atom i,
    atom j and the point: (u_i - mu e_ij) / R_ij, -(u_j - mu e_ij) / R_ij and
    (u_j - u_i) / R_ij, with u_a the unit vector from the point to atom a and
    e_ij that from j to i.
    """
    natm, n = mol.natm, len(coords)
    grad = np.zeros((natm, 3))
    if natm == 1 or n == 0:
        return grad
    R = mol.atom_coords()
    Rab = gto.inter_distance(mol)
    to_atom = R[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.einsum('akx,akx->ak', to_atom, to_atom))
    unit = to_atom / dist[:, :, None]

    def cells(i):
        mu = (dist[i] - dist[:i]) / Rab[i, :i, None]
        nu = mu + adjust[i, :i, None] * (1.0 - mu * mu)
        p1 = 0.5 * (3.0 - nu * nu) * nu
        p2 = 0.5 * (3.0 - p1 * p1) * p1
        p3 = 0.5 * (3.0 - p2 * p2) * p2
        return mu, nu, p1, p2, p3

    P = np.ones((natm, n))
    for i in range(1, natm):
        p3 = cells(i)[4]
        P[i] *= np.prod(0.5 * (1.0 - p3), axis=0)
        P[:i] *= 0.5 * (1.0 + p3)
    Z = P.sum(axis=0)
    at_point = np.zeros((3, n))
    for i in range(1, natm):
        mu, nu, p1, p2, p3 = cells(i)
        # ds/dnu dnu/dmu of the three-fold iterated cell function
        t = ((27.0 / 16.0) * (1.0 - p2 * p2) * (1.0 - p1 * p1)
             * (1.0 - nu * nu) * (1.0 - 2.0 * adjust[i, :i, None] * mu))
        s_ij, s_ji = 0.5 * (1.0 - p3), 0.5 * (1.0 + p3)
        # where a cell function vanishes so does t: nothing to divide
        pt_ij = -t / np.where(s_ij > 0.0, s_ij, 1.0)
        pt_ji = t / np.where(s_ji > 0.0, s_ji, 1.0)
        lam = ((owner == i) * pt_ij
               + (owner[None, :] == np.arange(i)[:, None]) * pt_ji)
        eta = lam - (P[i] * pt_ij + P[:i] * pt_ji) / Z
        psi = phi * eta / Rab[i, :i, None]
        e = (R[i] - R[:i]) / Rab[i, :i, None]
        s_mu = (psi * mu).sum(axis=1)
        s_uj = np.einsum('jk,jkx->jx', psi, unit[:i])
        grad[i] += psi.sum(axis=0) @ unit[i] - s_mu @ e
        grad[:i] += s_mu[:, None] * e - s_uj
        at_point += (np.einsum('jk,jkx->xk', psi, unit[:i])
                     - psi.sum(axis=0) * unit[i].T)
    np.add.at(grad, owner, at_point.T)
    return grad


def _xc_tile_addend(mol, ni, xc_code, xctype, coords, weight, owner, g_ao,
                    dm, adjust, cutoff):
    """(natm, 3) of one tile's points: the fixed-grid AO term, the points
    riding with their owners, and the weights' response (module docstring).
    g_ao None: of E_xc[dm] itself, f = e_xc and v in place of the kernel."""
    natm = mol.natm
    out = np.zeros((natm, 3))
    real = owner >= 0
    coords, weight, owner = coords[real], weight[real], owner[real]
    if len(weight) == 0:
        return out
    mask = gen_grid.make_mask(mol, coords)
    ao = ni.eval_ao(mol, coords, deriv=1 if xctype == 'LDA' else 2,
                    non0tab=mask, cutoff=cutoff)
    shls, loc = (0, mol.nbas), mol.ao_loc_nr()
    ncomp = 1 if xctype == 'LDA' else 4
    cD = [numint._dot_ao_dm(mol, ao[i], dm, mask, shls, loc)
          for i in range(ncomp)]
    cg = (None if g_ao is None
          else [numint._dot_ao_dm(mol, ao[i], g_ao, mask, shls, loc)
                for i in range(ncomp)])

    def density(c):
        """rho_i of a symmetric AO matrix from its c_i = ao_i X (pyscf's
        'eff' layout: rho, grad rho, tau = |grad phi|^2 X / 2)."""
        rows = [np.einsum('km,km->k', ao[0], c[0])]
        if xctype != 'LDA':
            rows += [2.0 * np.einsum('km,km->k', ao[i], c[0])
                     for i in (1, 2, 3)]
        if xctype == 'MGGA':
            rows.append(0.5 * sum(np.einsum('km,km->k', ao[i], c[i])
                                  for i in (1, 2, 3)))
        return np.array(rows)

    rho = density(cD)
    nvar, n = rho.shape
    rho_in = rho[0] if xctype == 'LDA' else rho
    if cg is None:
        exc, v = ni.eval_xc_eff(xc_code, rho_in, deriv=1, xctype=xctype,
                                spin=0)[:2]
        f = exc * rho[0]
        # the energy is first order in rho: v where T has its kernel
        wu, wv = weight * np.reshape(v, (nvar, n)), None
    else:
        rho_g = density(cg)
        v, fxc = ni.eval_xc_eff(xc_code, rho_in, deriv=2, xctype=xctype,
                                spin=0)[1:3]
        v = np.reshape(v, (nvar, n))
        fxc = np.reshape(fxc, (nvar, nvar, n))
        f = np.einsum('ik,ik->k', v, rho_g)
        # the density's response to the AOs moving goes through the kernel
        wu = weight * np.einsum('ijk,ik->jk', fxc, rho_g)
        wv = weight * v

    def mix(j, i):
        """wu_j c^D_i + wv_j c^g_i: one (coefficient, c) pair per density."""
        t = wu[j][:, None] * cD[i]
        if wv is not None:
            t += wv[j][:, None] * cg[i]
        return t

    # q_x[k, m] = w [v_0 d_x phi_m c_0 + v_i (d_x phi_m c_i + d_x d_i phi_m c_0)
    # + (v_tau / 2) d_x d_i phi_m c_i], one (v, c) pair per density: (u, D)
    # and (v, g); -2 sum_{k, m on B} q_x is the fixed-grid d/dR_B
    b = mix(0, 0)
    lean = []
    if xctype != 'LDA':
        for i in (1, 2, 3):
            b += mix(i, i)
            e_i = mix(i, 0)
            if xctype == 'MGGA':
                e_i += 0.5 * mix(4, i)
            lean.append(e_i)
    # pyscf's deriv=2 AO rows of d_x d_i: xx xy xz yy yz zz from row 4
    hessian_rows = ((4, 5, 6), (5, 7, 8), (6, 8, 9))
    by_ao = np.zeros((3, mol.nao_nr()))
    by_point = np.zeros((3, len(weight)))
    for x in range(3):
        q = ao[1 + x] * b
        for i, e_i in enumerate(lean):
            q += ao[hessian_rows[x][i]] * e_i
        by_ao[x] = q.sum(axis=0)
        by_point[x] = q.sum(axis=1)
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        out[ia] -= 2.0 * by_ao[:, p0:p1].sum(axis=1)
    # grad_r f = -sum_B d_B f at a point: the point term is +2 sum_m q_x
    np.add.at(out, owner, 2.0 * by_point.T)
    return out + becke_weight_response(mol, coords, owner, weight * f, adjust)


def xc_grid_skeleton(mol, grids, ni, xc_code, g_ao, dm, comm=None,
                     tile=None):
    """(natm, 3) of d/dR Tr[g_ao v_xc^DFT[dm]] with both AO matrices held and
    the Becke grid `grids` moving with the atoms, rank 0's on every rank
    (module docstring). Restricted densities; the functional part of
    `xc_code` only (exact exchange and a VV10 kernel are not here)."""
    return _xc_moving_grid(mol, grids, ni, xc_code, g_ao, dm, comm, tile)


def xc_energy_grid_skeleton(mol, grids, ni, xc_code, dm, comm=None,
                            tile=None):
    """(natm, 3) of d/dR E_xc^DFT[dm] = d/dR sum_k w_k e_xc(r_k) with dm held
    and the Becke grid `grids` moving with the atoms, rank 0's on every rank:
    `xc_grid_skeleton`'s three terms with f = e_xc (module docstring)."""
    return _xc_moving_grid(mol, grids, ni, xc_code, None, dm, comm, tile)


def _xc_moving_grid(mol, grids, ni, xc_code, g_ao, dm, comm, tile):
    """The grid tiles, their addends and the verbatim gather of both xc
    skeletons; g_ao None is the energy's."""
    comm = current_comm() if comm is None else comm
    rank, size = ((0, 1) if comm is None
                  else (comm.Get_rank(), comm.Get_size()))
    natm = mol.natm
    xctype = ni._xc_type(xc_code)
    if xctype == 'HF':
        return np.zeros((natm, 3))
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        raise NotImplementedError(f'the xc skeleton of a {xctype} functional')
    if grids.coords is None:
        grids.build()
    owner = getattr(grids, 'atm_idx', None)
    if owner is None:
        raise RuntimeError(
            "the grid does not record each point's atom (pyscf's "
            "Grids.atm_idx), which the points' motion needs")
    coords, weights = grids.coords, grids.weights
    if size > 1:
        g_ao, dm, coords, weights = lockstep((g_ao, dm, coords, weights),
                                             comm, check=True)
    adjust = becke_adjustment(mol, grids)
    tiles = xc_grid_tiles(len(weights), tile)

    def addend(t):
        p0, p1 = tiles[t]
        return _xc_tile_addend(
            mol, ni, xc_code, xctype, coords[p0:p1], weights[p0:p1],
            owner[p0:p1], g_ao, dm, adjust, grids.cutoff)

    return tile_sum(addend, len(tiles), natm, comm)


def tile_sum(addend, ntiles, natm, comm=None):
    """(natm, 3) sum over tiles t of `addend(t)`, tile t rank t % size's:
    the addends are gathered verbatim and summed in tile order on every rank,
    so the same bits at every rank count, ranks owning no tile included."""
    comm = current_comm() if comm is None else comm
    rank, size = ((0, 1) if comm is None
                  else (comm.Get_rank(), comm.Get_size()))
    addends = np.zeros((ntiles, natm * 3))
    for t in partition(ntiles, rank, size):
        addends[t] = addend(int(t)).ravel()
    allgather_ranges(addends, [[(int(t), int(t) + 1)
                                for t in partition(ntiles, r, size)]
                               for r in range(size)], comm)
    out = np.zeros(natm * 3)
    for row in addends:
        out += row
    return out.reshape(natm, 3)


def fitted_energy_skeleton(mol, auxmol, dm, occ=None, channels=(), comm=None,
                           tile=None):
    """(natm, 3) of d/dR (1/2) Tr[dm (J - (1/2) sum_w w K_omega)[dm]], the
    density-fitted Coulomb and exchange energy of a restricted SCF with dm
    held, rank 0's on every rank.

    channels: [(omega, w)] of the exchange (`exchange_channels`), each
    operator fitted on its own metric; occ: Cd with dm = Cd Cd^T, needed with
    any exchange. One `FittedFockSkeleton` at g = dm takes the Coulomb half
    and the bare channels' summed weight, one more each attenuated channel;
    their addends are summed over the auxiliary tiles by `tile_sum`.
    """
    comm = current_comm() if comm is None else comm
    rank, size = ((0, 1) if comm is None
                  else (comm.Get_rank(), comm.Get_size()))
    if size > 1:
        dm, occ = lockstep((dm, occ), comm, check=True)
    bare = sum(w for omega, w in channels if omega == 0.0)
    parts = [(0.0, True, bare)] + [(float(omega), False, float(w))
                                   for omega, w in channels
                                   if omega != 0.0 and w != 0.0]
    out = np.zeros((mol.natm, 3))
    for omega, coulomb, w in parts:
        skeleton = FittedFockSkeleton(mol, auxmol, dm, dm, occ=occ if w else None,
                                      coulomb=coulomb, exchange=w, tile=tile,
                                      omega=omega)
        skeleton.prepare([int(t) for t in partition(len(skeleton.tiles), rank,
                                                    size)], comm)
        # Tr[g F[dm]] at g = dm is twice the energy
        out += 0.5 * tile_sum(skeleton.addend, len(skeleton.tiles), mol.natm,
                              comm)
    return out


def one_electron_energy_skeleton(mol, hcore_deriv, dm, dme, comm=None):
    """(natm, 3) of Tr[dm dh/dR_A] - Tr[dme dS/dR_A], the atoms A over the
    ranks (`tile_sum`), rank 0's on every rank. hcore_deriv: pyscf's
    `hcore_generator` (kinetic, nuclear attraction, ECP); dme the
    energy-weighted density C n eps C^T."""
    comm = current_comm() if comm is None else comm
    if comm is not None and comm.Get_size() > 1:
        dm, dme = lockstep((dm, dme), comm, check=True)
    aoslices = mol.aoslice_by_atom()

    def addend(ia):
        s0, s1, p0, p1 = (int(x) for x in aoslices[ia])
        out = np.zeros((mol.natm, 3))
        out[ia] = np.einsum('xij,ij->x', hcore_deriv(ia), dm)
        # <d mu|nu> rows of the atom's own AOs, both slots by symmetry
        ds = mol.intor('int1e_ipovlp', comp=3,
                       shls_slice=(s0, s1, 0, mol.nbas))
        out[ia] += 2.0 * np.einsum('xij,ij->x', ds, dme[p0:p1])
        return out

    return tile_sum(addend, mol.natm, mol.natm, comm)
