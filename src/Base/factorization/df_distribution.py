"""The small pieces a distributed J/K build and its ISDF-K twin share: timing
counters, the lockstepped density, the one-reduction sum of a Fock piece, the
install / uninstall of a distributed `with_df` handle, and the handles a
converged distributed SCF leaves on its mean field.

A leaf below `distributed_df` and `distributed_isdf_jk`, so that each imports
it at module level.
"""
import time

import numpy as np

from src.Base.utils.parallel.mpi_grid import current_comm, lockstep, reduce_sum


def add_seconds(timings, key, since):
    """Add the seconds since `since` to `key`. No dict, no timing."""
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + (time.time() - since)


def add_count(timings, key, n=1):
    """`n` more of whatever `key` counts."""
    if timings is not None:
        timings[key] = timings.get(key, 0) + n


def lockstep_density(dms, comm, timings=None):
    """Rank 0's density matrix, and the orbitals and occupations tagging it,
    written into this rank's own arrays; the same object back.

    The tag travels with the density because both `df_jk.get_jk` and
    `numint._gen_rho_evaluator` read it -- K and rho from the orbitals, not
    from D -- so a density locked without its orbitals would still hand the
    partial this rank's own. In place, so the SCF loop that owns these arrays
    decides its next step from rank 0's copy too.

    Checked (`mpi_grid.lockstep(check=True)`): every rank diagonalized the
    same all-reduced Fock matrix, so digests prove the density rank 0's and
    only a drifted array is broadcast.
    """
    if np.iscomplexobj(dms):
        raise NotImplementedError(
            'the reduction moves float64; a complex density matrix would need '
            'its halves reduced separately')
    t0 = time.time()
    lockstep((dms, getattr(dms, 'mo_coeff', None),
              getattr(dms, 'mo_occ', None),
              getattr(dms, 'read_columns', None)), comm, check=True)
    add_seconds(timings, 'scf_lockstep', t0)
    return dms


def reduce_parts(arrays, comm):
    """The ranks' partials added, in one reduction for the whole Fock piece.

    The sum is re-associated (rank boundaries first, then whatever order a
    rank used inside its own block), so the result differs from the serial
    one in the last bits. `reduce_sum` adds in rank order, so it differs in
    the same last bits on every rank and in every repeat.
    """
    shapes = [np.shape(a) for a in arrays]
    buf = np.concatenate([np.ravel(np.asarray(a, dtype=float))
                          for a in arrays])
    reduce_sum(buf, comm)
    out, off = [], 0
    for shape in shapes:
        n = int(np.prod(shape, dtype=int))
        out.append(buf[off:off + n].reshape(shape))
        off += n
    return out


class InstalledWithDF:
    """`install` / `uninstall` of a distributed `with_df` handle.

    A rank's share is not the whole: a downstream caller that reads
    `mf.with_df` (iterates `loop()` for the GW fit or the solvent screening,
    takes the grid a gradient differentiates) would read one rank's rows or
    tiles and call them the tensor. So the handle answers J/K only while a
    distributed stage is running; `uninstall` puts the caller's own object
    back and keeps this rank's share.
    """

    def install(self, mf):
        """Answer this mean field's J/K until `uninstall`."""
        if mf.with_df is not self:
            self._replaced = (mf, mf.with_df)
            mf.with_df = self
        return self

    def uninstall(self):
        """Put the mean field's own `with_df` back, keeping this rank's share."""
        pair, self._replaced = self._replaced, None
        if pair is not None:
            mf, with_df = pair
            mf.with_df = with_df
        return self


def distributed_handles(mf, comm=None):
    """`mf`'s (J/K, quadrature, comm) handles under `comm`, or None.

    What a converged distributed SCF leaves on `mf._distributed`: the objects
    holding this rank's slice of the fitted tensor and of the grid, so a later
    stage that wants the same Fock pieces (the static exchange the
    quasiparticle routes build once) reuses them instead of paying the slice
    build again. They are not installed on the mean field between stages;
    `distributed_fock` is the bracket that installs them.

    None where the handles belong to another communicator, which is a mean
    field carried over from a run with a different rank count: rows cut for
    four ranks are not a partition over eight.
    """
    comm = current_comm() if comm is None else comm
    handles = getattr(mf, '_distributed', None)
    if comm is None or comm.Get_size() == 1 or handles is None:
        return None
    return handles if handles[2] is comm else None
