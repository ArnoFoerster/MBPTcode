"""The checkpoint store of a photophysics run: one directory, one record per
stage.

    <root>/<system>/<environment>/
        s0, hess, vert, s1, t1, rate          the stage records
        s1_walked, s1_progress, ...           inside a stage (`stages`)
        s0.xyz, s1.xyz, t1.xyz                the minima, for a reader

Each record is written atomically (`record_io.write_record`: a JSON tree
beside an .npz of its arrays, read back to the bit) and carries

    stamp        what the record is a function of: the declaration's fields
                 that stage reads, the grids by name AND content, the span
                 (`stages.Stamps`), and their sha256 under 'identity'
    provenance   the commit of the checkout, the BLAS and its threads, the
                 host, the ranks, the time
    medium       the continuum the environment names, resolved: both
                 dielectric constants and the cavity convention
                 (`src.environments.medium`), None in the gas phase; the
                 environment in the stamp already fixes it

A record is READ only under the stamp it was written with: another stamp
refuses the run (`StoreRefused`), since every later stage would build on a
record of another surface. A stamp whose scf sections still name the retired
fields at the values they always held is that stamp without them
(`stamp_as_read`). The provenance is recorded and not compared: a
resume on another commit or BLAS is allowed and visible.

Under ranks the store reads the communicator of the enclosing
`distributed(comm)` block: rank 0 alone touches the disk, to write (a barrier
follows every write) and to read or look: what it finds is broadcast, so every
rank holds the same record whatever its view of the filesystem, and no rank
reads a file that rank 0 is replacing. Every call of `has`, `names`, `held`,
`read`, `reusable`, `update`, `write`, `write_xyz`, `set_aside` and `workdir`
is therefore collective: every rank makes it. The stages themselves hold no
communicator.
"""
import datetime
import functools
import json
import os
import pathlib
import platform

from src.Base.numerics.constants import BOHR_TO_ANGSTROM
from src.Base.numerics.numerics import without_retired
from src.Base.utils.record_io import git_commit, read_record, record_hash, write_record
from src.Base.utils.parallel.mpi_grid import broadcast, current_comm, is_root, n_ranks
from src.Base.utils.parallel.threads import blas_libraries
from src.environments.medium import parse as parse_medium


class StoreRefused(RuntimeError):
    """A record written under another stamp than the reader's."""


@functools.lru_cache(maxsize=1)
def commit():
    """The checkout's commit (`git_commit`), asked once per process."""
    return git_commit()


def provenance():
    """Where and on what a record was computed: not part of the stamp."""
    return {'commit': commit(), 'blas': blas_libraries(),
            'host': platform.node(), 'ranks': n_ranks(),
            'slurm_job': os.environ.get('SLURM_JOB_ID'),
            'written': datetime.datetime.now().isoformat(timespec='seconds')}


def _without_retired_scf(tree):
    """`tree` with every scf section in it (a stage's, and the Hessian
    stage's inside a later stage's stamp) read through `without_retired`."""
    out = {}
    for key, value in tree.items():
        if key == 'scf' and isinstance(value, dict):
            value = without_retired({'scf': value})['scf']
        elif isinstance(value, dict):
            value = _without_retired_scf(value)
        out[key] = value
    return out


def stamp_as_read(stamp):
    """The held `stamp` as the current code writes it: the retired fields
    (`numerics.IDENTITY_RETIRED`) at the values they always held left out
    and its identity hashed again; a retired field at another value refuses.
    A stamp that names no retired field is returned as it is."""
    bare = {k: v for k, v in stamp.items() if k != 'identity'}
    try:
        now = _without_retired_scf(bare)
    except ValueError as exc:
        raise StoreRefused(str(exc)) from exc
    if now == bare:
        return stamp
    return dict(now, identity=record_hash(now))


def _sync(path):
    """Force the file or directory `path` to stable storage."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sync_record(path):
    """The record at `path` and the arrays file it names, then their
    directory, forced to stable storage."""
    path = pathlib.Path(path)
    for name in (path.name, json.loads(path.read_text())['arrays']):
        _sync(path.with_name(name))
    _sync(path.parent)


class Store:
    """One run's records (module docstring). `root` is the directory the
    caller names; the run's own directory is root/<system>/<environment>."""

    def __init__(self, root, declaration):
        self.dir = (pathlib.Path(root).resolve() / declaration.system.name
                    / declaration.environment_dir)
        medium = None if declaration.gas else parse_medium(declaration.environment)
        self.medium = None if medium is None else medium.record()
        if self.root_rank:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.barrier()

    # ------------------------------------------------------------ ranks
    @property
    def root_rank(self):
        """Whether this rank writes (rank 0, or no distribution)."""
        return is_root()

    @staticmethod
    def barrier():
        """Wait for every rank of the distribution."""
        comm = current_comm()
        if comm is None:
            return
        if hasattr(comm, 'Barrier'):
            comm.Barrier()
        else:                   # a simulated communicator has no Barrier
            comm.allgather(None)

    # ------------------------------------------------------------- paths
    def path(self, name):
        """The file of record `name` (a stage record is name.json)."""
        return self.dir / (name if '.' in name else f'{name}.json')

    def on_root(self, look):
        """`look()` made on rank 0 alone and its value, or the exception it
        raised, on every rank: what the disk shows is one fact, however each
        rank's view of the filesystem is cached or delayed."""
        comm = current_comm()
        if comm is None:
            return look()
        verdict = None
        if self.root_rank:
            try:
                verdict = ('value', look())
            except Exception as exc:
                verdict = ('raised', exc)
        kind, got = broadcast(verdict, comm)
        if kind == 'raised':
            raise got
        return got

    def has(self, name):
        """Whether record `name` is in the store."""
        return self.on_root(lambda: self.path(name).exists())

    def names(self):
        """Every file in the run's directory, sorted."""
        return self.on_root(lambda: sorted(p.name for p in self.dir.iterdir())
                            if self.dir.is_dir() else [])

    # ----------------------------------------------------------- records
    def write(self, name, record, stamp):
        """`record` under `name`, stamped with `stamp` and the provenance;
        rank 0 writes, every rank waits for it."""
        if self.root_rank:
            write_record(self.path(name), {
                'stamp': dict(stamp, identity=record_hash(stamp)),
                'provenance': provenance(), 'medium': self.medium,
                'record': record})
            _sync_record(self.path(name))
        self.barrier()

    def held(self, name):
        """The whole stored tree of `name`: stamp, provenance, record."""
        return self.on_root(lambda: read_record(self.path(name)))

    def read(self, name, stamp):
        """The record `name` written under `stamp`; another stamp refuses."""
        held = self.held(name)
        want = dict(stamp, identity=record_hash(stamp))
        try:
            was = stamp_as_read(held['stamp'])
        except StoreRefused as exc:
            raise StoreRefused(f'{self.path(name)}: {exc}') from exc
        if was != want:
            differ = sorted(k for k in set(was) | set(want)
                            if was.get(k) != want.get(k)
                            and k != 'identity')
            raise StoreRefused(
                f'{self.path(name)} was computed under another declaration '
                f'(differs in {differ}): it belongs to another surface. Point '
                f'the store at a fresh directory, or declare what it used.')
        return held['record']

    def reusable(self, name, stamp):
        """Whether `name` is on disk for THIS declaration; on disk under
        another stamp refuses (`read`)."""
        if not self.has(name):
            return False
        self.read(name, stamp)
        return True

    def update(self, name, stamp, **fields):
        """Record `name` with `fields` set, written again under its stamp."""
        record = dict(self.read(name, stamp), **fields)
        self.write(name, record, stamp)
        return record

    # ------------------------------------------------------------ others
    def write_xyz(self, name, mol, comment):
        """`mol` as an xyz file in Angstrom, for a reader."""
        if self.root_rank:
            lines = [str(mol.natm), comment]
            for ia, c in enumerate(mol.atom_coords() * BOHR_TO_ANGSTROM):
                lines.append(f'{mol.atom_pure_symbol(ia):2s} {c[0]:20.14f} '
                             f'{c[1]:20.14f} {c[2]:20.14f}')
            path = self.path(name)
            tmp = path.with_name(path.name + '.tmp')
            tmp.write_text('\n'.join(lines) + '\n')
            _sync(tmp)
            os.replace(tmp, path)
            _sync(path.parent)
        self.barrier()

    def set_aside(self, names, under):
        """Move the files and directories `names` of the run's directory
        into its subdirectory `under` (made by rank 0, every rank waits),
        where the store no longer finds them: nothing is deleted. Returns
        the names moved."""
        moved = self.on_root(
            lambda: [n for n in names if (self.dir / n).exists()])
        if self.root_rank and moved:
            target = self.dir / under
            target.mkdir(parents=True, exist_ok=True)
            for n in moved:
                os.replace(self.dir / n, target / n)
            _sync(self.dir)
        self.barrier()
        return moved

    def workdir(self, name):
        """A subdirectory for a program's own files (geomeTRIC's, a force
        cache); made by rank 0, None on the others."""
        path = self.dir / name
        if self.root_rank:
            path.mkdir(exist_ok=True)
        self.barrier()
        return path
