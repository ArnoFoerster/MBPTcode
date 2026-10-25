# Threads and MPI

## Threads

Set `OMP_NUM_THREADS` to the number of physical cores the run may use. It sets
the OpenMP and BLAS threads of PySCF and NumPy, and the thread pool of the
per-state quasiparticle root scan (`calc_qp_energy(n_workers=...)`). The pool
takes `n_workers`, else `OMP_NUM_THREADS`, else one thread per CPU the process
may run on, and never exceeds those CPUs; `n_workers=1` gives the serial scan.

Leave `OMP_PROC_BIND` unset. Pool threads inherit the main thread's CPU mask, so
a bound main thread confines the pool to its own CPUs, often a single core; the
scan then warns and shrinks the pool. With the pip wheels, export
`OMP_WAIT_POLICY=PASSIVE`: otherwise PySCF's OpenMP threads and NumPy's OpenBLAS
threads spin against each other on small calls.

Several processes on one machine each take `OMP_NUM_THREADS=T` and their own
T cores, bound by the launcher.

## Distributed eigensolve

The dense eigensolves of the Casida route (`CasidaSolver.solve`, behind the RPA,
GW and dense BSE routes) and of ADC (`solve_dense`) can run on
[ELPA](https://elpa.mpcdf.mpg.de/) over MPI ranks. They do so for a matrix of
dimension 5000 or more (their `threshold` argument) whenever `mpi4py` and
ELPA's Python binding `pyelpa` both import, on one rank or on several.
Otherwise, or with `MBPT_USE_ELPA=0`, they call `scipy.linalg.eigh`, as they do
for every complex Hermitian matrix.

Rank 0 runs the script alone; the other ranks wait to serve its eigensolves.
Open and close the script with

```python
import sys

from src.Base.utils.linear_algebra.diagonalization import (serve_distributed_solves,
                                                          release_workers)

if serve_distributed_solves():
    sys.exit(0)                     # a worker rank, released by rank 0 at the end

...                                 # the calculation, on rank 0 only

release_workers()
```

On one rank, or without `mpi4py` and `pyelpa`, both calls do nothing, so the
same script runs serially. Call
`serve_distributed_solves()` before anything imports `mpi4py`: `pyelpa` refuses
to load after it, and every solve then stays on `eigh`. A failure inside a
distributed solve, on any rank, aborts the job.

Rank 0 still builds and holds each matrix and its eigenvectors, and runs
everything outside the eigensolve on its own threads. ELPA spreads the
eigensolver's work and workspace over the ranks, not the matrices' memory.

R ranks of T threads each:

```bash
export OMP_NUM_THREADS=T
export ELPA_DEFAULT_omp_threads=$OMP_NUM_THREADS
mpirun -n R --bind-to core --map-by slot:PE=T python run.py
```

ELPA's own OpenMP threads default to one per rank, and `OMP_NUM_THREADS` does
not reach them; `ELPA_DEFAULT_omp_threads` does, if `pyelpa` is linked against
the OpenMP build of ELPA (`libelpa_openmp`). Bind the ranks to cores: unbound
ranks run the solve far more slowly.

`pyelpa` is on neither PyPI nor conda-forge. Build it from `python/pyelpa` in
the source of the installed ELPA version: with ELPA's own
`./configure --enable-python`, or against the installed library with this
`setup.py` in `python/`, ELPA's `.pc` file on `PKG_CONFIG_PATH`, Cython
installed, `CC=mpicc`, and `pip install --no-build-isolation .`:

```python
import subprocess

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup


def pkg(flag):
    return subprocess.check_output(['pkg-config', flag, 'elpa_openmp'],
                                   text=True).split()


ext = Extension(
    'pyelpa.wrapper',
    sources=['pyelpa/wrapper.pyx'],
    include_dirs=[numpy.get_include()] + [f[2:] for f in pkg('--cflags-only-I')],
    extra_compile_args=[f for f in pkg('--cflags') if not f.startswith('-I')],
    extra_link_args=pkg('--libs'),
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
)

setup(name='pyelpa', version='2025.01.002',     # the ELPA release
      packages=['pyelpa'], ext_modules=cythonize([ext], language_level=3))
```

Write `elpa` for `elpa_openmp` where ELPA was built without OpenMP. To check
the setup, run `tests/test_elpa_casida.py` under `mpirun`: on more than one
rank it fails if a solve fell back to `eigh`.

## Running under MPI

The space-time GW routes, the ISDF fit, the BSE Davidson and the
density-fitted SCF divide their work over MPI ranks. Every rank runs the whole
script -- the SCF loop, the Davidson, the gradient, the geometry optimization --
and MPI lives only inside the numerical kernels: the time and frequency sweeps
of the GW self-energy, the ISDF fit, the BSE matrix-vector products, the SCF's
Coulomb, exchange and exchange-correlation builds, and the pieces of an
analytic force. A driver needs no communicator: the kernels read it from the
region the script opens once. (`solve_bse_isdf` also takes `comm=` and
`distribute=`, for a caller that wants a communicator other than the region's.)

```python
from pyscf import dft, gto

from src.Base.factorization.distributed_df import distributed_mean_field
from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.mpi_grid import distributed, grid_comm
from src.SingleReference.BSE.solve import solve_bse_isdf

comm = grid_comm()[0]                  # COMM_WORLD, or None without mpi4py
with distributed(comm):
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
                basis='cc-pvdz')
    mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis='cc-pvdz-ri')
    distributed_mean_field(mf)         # J/K and the xc grid split over ranks
    numerics = Numerics(isdf={'level': 'G1'}, bse={'nroots': 5})
    omega, X, Y, info = solve_bse_isdf(mf, mol, mol.nelectron // 2,
                                       numerics=numerics)
    if comm is None or comm.Get_rank() == 0:
        print(omega)                   # the same bits on every rank
```

Launch it with `mpirun -n R python run.py`, with `OMP_NUM_THREADS` set to the
cores each rank may use (see [Threads](#threads)). Without `mpi4py`, or with
`MBPT_USE_MPI=0`, `grid_comm` returns None and the same script runs serially,
bit for bit the serial code.
`mpi4py` is imported on first use, never at module import.

A solvent continuum works the same way. `SolventScreening.mean_field`
converges the solvated SCF through `distributed_mean_field`; the reaction
field of the cavity is cheap, so every rank computes it from the shared
density, and the code checks that it matches rank 0's. A gradient chain in a
continuum also runs on sliced factors (`sliced=True`) and on the
row-distributed fit (`fit='rows'`): every interaction it reads (the dressed
one, the bare one the self-energy screens with, and the static one of a split
equilibrium) is built from the same fit as grid rows, so no rank holds a
factor whole.

Two ranks need not agree in the last bits of an orbital energy or a Davidson
residual, and a decision taken from such a number -- a grid size, when to stop
-- would then differ between them. So every kernel copies rank 0's version of
the inputs that can differ to all ranks on entry, and combines what it computes
so that its output is the same bits on every rank. Every decision a driver
takes is then the same on every rank, and the drivers stay in step without
talking to each other. `with distributed(comm, audit=True):` checks at run time
that the kernels saw identical inputs on every rank.

Two rules follow. A computation meant for one rank only, or a serial reference
inside the region, runs under `with distributed(None):`, since a kernel that
found the region's communicator would wait in a collective the other ranks
never enter. And an exception that leaves the region on any rank prints its
traceback and calls `comm.Abort(1)`, instead of leaving the other ranks waiting
until the wall clock ends the job.

The distributed eigensolve above is a different mode -- rank 0 runs the script
alone and the other ranks serve its eigensolves -- and returns the
eigenvectors on rank 0 only, so it does not combine with a distributed region:
in a script that opens one, set `MBPT_USE_ELPA=0`. `tests/test_mpi_routes.py`
under `mpirun` checks every distributed route against its serial reference;
`tests/README.md` lists the tests that gate each split.
