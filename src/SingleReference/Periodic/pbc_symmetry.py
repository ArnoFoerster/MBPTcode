"""
Point-group and time-reversal symmetry of a Gamma-centred k-mesh, for every
k-point route that sums over momentum transfers.

An operation S = {R|t} of the crystal (or R followed by time reversal) maps
the Bloch AOs at k onto those at Sk by a unitary U(S, k), so a one-body
quantity of a symmetric state obeys

    A^{Sk} = U(S, k) A^k U(S, k)^dagger           (conjugated for time reversal)

and the same holds for the share X_q^k of any transfer q in a sum over q of
products of covariant factors (Sigma^k = sum_q X_q^k, K^k = sum_q K_q^k):
X_{Sq}^{Sk} = U X_q^k U^dagger. Summing only the irreducible transfers qbar,
each weighted by its star, and averaging the result over the group gives the
full sum exactly,

    sum_q X_q = (1/|G|) sum_S R_S [ sum_qbar |star(qbar)| X_qbar ],

because each q of a star is reached |G|/|star| times. A scalar of q (an RPA
energy, the dielectric head) needs only the weighted sum over qbar.

The operations are pyscf's (`KPoints`): the space group of the cell, plus
time reversal when it has no inversion, restricted to those mapping the mesh
onto itself; U(S, k) is pyscf's Wigner-D rotation of each atom's shells times
the Bloch phase of the lattice vector the atom lands on. Every operation is
checked on the overlap, S^{Sk} = U S^k U^dagger, before it is used.

The group acts on transfers as on k-points, which a Gamma-centred mesh shares
with its transfers (`get_momentum_transfer_map`). The trivial group (identity
only) makes every symmetric route its plain form.
"""
import numpy as np

from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.symm.symmetry import _get_phase, _get_rotation_mat

from src.Base.numerics.constants import KMESH_SYMMETRY_TOL
from src.Base.utils.parallel.mpi_grid import allgather_rows, contiguous_block, current_comm


class KMeshSymmetry:
    """
    The operations of a crystal that map its Gamma-centred k-mesh onto itself.

    image[s, k]: index of S_s k. time_reversal[s]: S_s includes time reversal.
    irreducible: one k-point per star, the lowest index; rep[k]: the
    irreducible k-point of k's star; star_size[k]: size of k's star.
    """

    def __init__(self, cell, kpts):
        kpts = np.asarray(kpts)
        if np.abs(cell.get_scaled_kpts(kpts[:1])).max() > 1e-12:
            raise ValueError('KMeshSymmetry: the mesh must be Gamma-centred '
                             '(kpts[0] = 0), so that transfers are k-points')
        kp = KPoints(cell, kpts).build(space_group_symmetry=True,
                                       time_reversal_symmetry=True)
        nop = len(kp.ops)
        nrot = kp.k2opk.shape[1]
        on_mesh = [s for s in range(nrot) if (kp.k2opk[:, s] >= 0).all()]
        self.cell, self.kpts, self.nk = cell, kpts, len(kpts)
        self._ops = [kp.ops[s % nop] for s in on_mesh]
        self.time_reversal = np.array([s >= nop for s in on_mesh])
        self.image = np.ascontiguousarray(kp.k2opk[:, on_mesh].T)
        scaled = kp.kpts_scaled
        # U(S, k) = M_S diag(phase of the atom each AO column sits on)
        nao = cell.nao_nr()
        self._block = [_get_rotation_mat(cell, scaled[0], np.empty((nao, nao)), op,
                                         kp.Dmats[s % nop], ignore_phase=True).real
                       for s, op in zip(on_mesh, self._ops)]
        self._atom_of_ao = np.repeat(np.arange(cell.natm),
                                     np.diff(cell.aoslice_by_atom()[:, 2:4]).ravel())
        self._phase = np.array([[_get_phase(cell, op, scaled[k])[1]
                                 for k in range(self.nk)] for op in self._ops])
        self._index_stars()
        self._check_on_overlap()

    @classmethod
    def trivial(cls, nk):
        """The identity alone: every k-point its own star."""
        self = cls.__new__(cls)
        self.cell, self.kpts, self.nk = None, None, int(nk)
        self.time_reversal = np.zeros(1, dtype=bool)
        self.image = np.arange(nk)[None, :]
        self._index_stars()
        return self

    @property
    def nop(self):
        return len(self.image)

    def _index_stars(self):
        self.rep = -np.ones(self.nk, dtype=int)
        for k in range(self.nk):
            if self.rep[k] < 0:
                self.rep[self.image[:, k]] = k
        self.irreducible = np.flatnonzero(self.rep == np.arange(self.nk))
        self.star_size = np.bincount(self.rep, minlength=self.nk)[self.rep]

    def rotation(self, s, k):
        """U(S_s, k): Bloch AOs at k onto those at S_s k, (nao, nao)."""
        if self.cell is None:
            raise ValueError('the trivial group has no AO rotation')
        return self._block[s] * self._phase[s, k][self._atom_of_ao][None, :]

    def rotate(self, s, k, A):
        """A^k (..., nao, nao) carried to S_s k."""
        if self.cell is None:
            return A
        U = self.rotation(s, k)
        out = U @ A @ U.conj().T
        return out.conj() if self.time_reversal[s] else out

    def symmetrize(self, field):
        """(1/|G|) sum_S R_S[field] of a k-field (nk, ..., nao, nao): the full
        transfer sum from its star-weighted irreducible part, on every rank of
        `current_comm()`. A field with an axis after k (tau of Sigma and G)
        is averaged by `contiguous_block`s of that axis, one per rank, and the
        blocks are gathered verbatim (`allgather_rows`), so every rank holds
        the serial average bit for bit after one gather of the field; a field
        (nk, nao, nao) is averaged whole on each rank, which costs less than
        that gather."""
        if self.nop == 1:
            return field
        comm = current_comm()
        rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
        if field.ndim == 3 or size == 1:
            out = np.zeros(field.shape, dtype=complex)
            for s in range(self.nop):
                for k in range(self.nk):
                    out[self.image[s, k]] += self.rotate(s, k, field[k])
            return out / self.nop
        # rows of the gather: the split axis first, (n1, nk, ...)
        start, stop = contiguous_block(field.shape[1], rank, size)
        out = np.zeros((field.shape[1], self.nk) + field.shape[2:], dtype=complex)
        if stop > start:
            mine = out[start:stop].swapaxes(0, 1)
            for s in range(self.nop):
                for k in range(self.nk):
                    mine[self.image[s, k]] += self.rotate(s, k, field[k, start:stop])
            mine /= self.nop
        allgather_rows(out.view(np.float64), comm)
        return np.ascontiguousarray(out.swapaxes(0, 1))

    def unfold(self, field):
        """A k-field known at the irreducible k-points (others ignored) on the
        whole mesh, each k from its star's representative."""
        out = np.array(field, dtype=complex)
        for s in range(self.nop):
            for k in self.irreducible:
                out[self.image[s, k]] = self.rotate(s, k, field[k])
        return out

    def _check_on_overlap(self):
        """Refuse the group if an operation does not carry S^k onto S^{Sk}."""
        S = np.asarray(self.cell.pbc_intor('int1e_ovlp', hermi=1, kpts=self.kpts))
        worst = max(np.abs(self.rotate(s, k, S[k]) - S[self.image[s, k]]).max()
                    for s in range(self.nop) for k in range(self.nk))
        if worst > KMESH_SYMMETRY_TOL:
            raise RuntimeError(f'KMeshSymmetry: an operation misses the overlap by '
                               f'{worst:.2e} > {KMESH_SYMMETRY_TOL:.0e}; the AO rotation '
                               f'convention does not match this pyscf')


def kmesh_symmetry(cell, kpts, use_symmetry):
    """KMeshSymmetry of the mesh, or the trivial group."""
    return KMeshSymmetry(cell, kpts) if use_symmetry else KMeshSymmetry.trivial(len(kpts))
