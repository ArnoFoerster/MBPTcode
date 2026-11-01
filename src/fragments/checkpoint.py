"""Checkpoint of the fragment partition's three long solves.

`FragmentPartition.build` spends nearly all its time in three stages: each
diabat block's eigenpairs, the pole guard on Q, and the resolvent vectors y.
A `PartitionCheckpoint` saves each stage's result in a directory and hands it
back to a later build at the same problem, so a rerun, or a second job at the
identical geometry, skips the solve.

WHAT A FILE IS KEYED BY. Every file holds a key (JSON): the format version, a
caller's `tag` (the code commit, the basis: whatever the arrays below cannot
see), the kernel (Tamm-Dancoff or full), the energy an energy-dependent
operator K(Omega) is taken at (None for a static one), the dimensions, the
sites and the charge-transfer diabats, the solver tolerances and constants, the
dense limit, the fragment labels of every orbital and, for the guard and the
resolvent, Omega_0, the margin and a digest of the diabat vectors they were
computed from. It also holds a FINGERPRINT of the arrays the discrete key
cannot name: the operator's orbital energies (quasiparticle energies for
GW-BSE), the local orbital rotations projected on fixed random vectors, and the
operator's action on a fixed random block at FRAGMENT_CHECKPOINT_PROBE_ROWS
fixed rows.
A different geometry, basis, mean field, kernel route, localization or screening
moves the fingerprint far above FRAGMENT_CHECKPOINT_MATCH_TOL (a 5e-4 Bohr step
moves it by ~1e-4); the same problem rerun on another node moves it by at most
the run-to-run noise, far below. A file whose key or fingerprint differs is
ignored (and overwritten when the stage is solved again).

WHAT A RESTORE RETURNS. The saved arrays, bit for bit. A restored partition of
the same problem therefore equals the fresh one exactly in the process that
saved it, and to the solver tolerance when the fingerprints agree to the match
tolerance in another run.

A stage that raised a numerical flag (`FragmentPartition.guard_flags`: an
unconverged solve, an unconverged guard) is not saved, so the next run solves
it again and flags it again instead of restoring a result without its warning.

RANKS. Rank 0 reads, compares and writes (a write goes through a temporary
file and `os.replace`); its verdict, then its arrays, reach every rank through
`lockstep`, so every rank skips or runs the same solve. The directory need only
exist on rank 0.
"""
import hashlib
import json
import os

import numpy as np

from src.Base.numerics import constants
from src.Base.utils.parallel.mpi_grid import current_comm, lockstep

VERSION = 1

_SOLVER_CONSTANTS = ('FRAGMENT_DAVIDSON_EXTRA_ROOTS',
                     'FRAGMENT_DAVIDSON_MAX_CYCLE', 'FRAGMENT_CG_MAX_ITER',
                     'FRAGMENT_PRECONDITIONER_FLOOR', 'FRAGMENT_GUARD_MAX_CYCLE',
                     'FRAGMENT_GUARD_NEWTON_TOL', 'FRAGMENT_GUARD_TOL',
                     'FRAGMENT_POLE_NEWTON_STEPS')


def digest(array):
    """Hex digest of an array's exact bytes."""
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest() + f'{a.shape}{a.dtype}'


def _is_root():
    comm = current_comm()
    return comm is None or comm.Get_rank() == 0


class PartitionCheckpoint:
    """Directory of stage files for one `FragmentPartition.build` problem.

    `tag`: anything that identifies the run beyond the arrays (a mapping or
    string; the code commit and basis belong here). Call `bind` once per
    build, then `load` / `save` per stage.
    """

    def __init__(self, directory, tag=None):
        self.directory = os.fspath(directory)
        self.tag = json.dumps(tag, sort_keys=True, default=str)
        self.key = self.fingerprint = None

    def bind(self, operator, orbitals, eps_qp, sites, ct, settings):
        """Fix the key and the fingerprint of this build (a collective: the
        operator acts once on the probe block). `settings`: the build's
        `tol`, `margin` and `dense_max`, recorded in the key as given."""
        probe_rng = np.random.default_rng(
            constants.FRAGMENT_CHECKPOINT_PROBE_SEED)
        dim = operator.dim
        n_rows = min(constants.FRAGMENT_CHECKPOINT_PROBE_ROWS, dim)
        rows = np.sort(probe_rng.choice(dim, size=n_rows, replace=False))
        x = probe_rng.standard_normal(
            (dim, constants.FRAGMENT_CHECKPOINT_PROBE_COLUMNS))
        y = np.asarray(operator.apply(x), float)
        proj_o = probe_rng.standard_normal(orbitals.u_occ.shape[1])
        proj_v = probe_rng.standard_normal(orbitals.u_vir.shape[1])
        self.fingerprint = dict(
            eps_qp=np.asarray(eps_qp, float),
            u_occ=orbitals.u_occ @ proj_o, u_vir=orbitals.u_vir @ proj_v,
            probe=y[rows])
        self.key = dict(
            version=VERSION, tag=self.tag, tda=bool(operator.tda),
            omega=(None if operator.omega is None
                   else float(operator.omega)),
            dim=int(dim), nocc=int(operator.nocc), nvir=int(operator.nvir),
            sites=sorted((int(k), int(n)) for k, n in sites.items()),
            ct=sorted((int(k), int(l), int(n)) for (k, l), n in ct.items()),
            tol=float(settings['tol']), margin=float(settings['margin']),
            dense_max=int(settings['dense_max']),
            solver={n: float(getattr(constants, n))
                    for n in _SOLVER_CONSTANTS},
            occ_labels=digest(np.asarray(orbitals.occ_labels)),
            vir_labels=digest(np.asarray(orbitals.vir_labels)))

    def _path(self, stage):
        return os.path.join(self.directory, f'{stage}.npz')

    def load(self, stage, **extra):
        """The saved arrays of `stage` (a dict) when its key, with `extra`
        added, and its fingerprint match this build's, else None. Every rank
        gets the same answer."""
        root = _is_root()
        found = self._read(stage, extra) if root else None
        # the verdict, with the arrays' shapes and dtypes, as one string ...
        layout = lockstep(json.dumps(
            None if found is None
            else {n: [list(a.shape), a.dtype.str] for n, a in found.items()}))
        layout = json.loads(layout)
        if layout is None:
            return None
        # ... then the arrays, into buffers of that layout on the other ranks
        if not root:
            found = {n: np.empty(tuple(shape), dtype=np.dtype(dtype))
                     for n, (shape, dtype) in layout.items()}
        return lockstep(found)

    def _read(self, stage, extra):
        path = self._path(stage)
        if not os.path.exists(path):
            return None
        try:
            with np.load(path, allow_pickle=False) as f:
                key = json.loads(str(f['key']))
                if key != self._key(extra):
                    return None
                for name, ref in self.fingerprint.items():
                    saved = f[f'fp_{name}']
                    scale = max(1.0, float(np.abs(ref).max()))
                    if (saved.shape != ref.shape
                            or np.abs(saved - ref).max()
                            > constants.FRAGMENT_CHECKPOINT_MATCH_TOL * scale):
                        return None
                return {n[4:]: f[n] for n in f.files if n.startswith('out_')}
        except (OSError, ValueError, KeyError):
            return None     # a torn or foreign file is a miss, not an error

    def _key(self, extra):
        return json.loads(json.dumps({**self.key, **extra}, sort_keys=True))

    def save(self, stage, extra=None, **arrays):
        """Write `stage`'s arrays (rank 0 only)."""
        if not _is_root():
            return
        os.makedirs(self.directory, exist_ok=True)
        path = self._path(stage)
        tmp = f'{path}.tmp{os.getpid()}.npz'
        np.savez(tmp, key=json.dumps(self._key(extra or {}), sort_keys=True),
                 **{f'fp_{n}': a for n, a in self.fingerprint.items()},
                 **{f'out_{n}': np.asarray(a) for n, a in arrays.items()})
        os.replace(tmp, path)
