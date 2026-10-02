# Installation

Requires Python 3.10+, NumPy, SciPy, PySCF, opt_einsum and threadpoolctl:

```bash
pip install numpy scipy pyscf opt_einsum threadpoolctl
```

`threadpoolctl` pins BLAS to one thread inside the quasiparticle root scan's
thread pool and is imported at module load by `GW/qp_energy.py`. Thread
settings are under [Threads](parallel.md#threads).

`opt_einsum` is imported at module load by `CC/cached_einsum.py`, which most of
the tree pulls in, so it is not optional.

The coupled-cluster integral path additionally needs `openfermion` and
`openfermionpyscf`, and the determinant-basis active-space solver
(`Solvers.active_space_solver.ExactDiagonalizationSolver`, the default
backend of `Solvers.active_space.solve`) needs `openfermion`; both are
imported only when they are reached:

```bash
pip install openfermion openfermionpyscf
```

Three more optional dependencies, each imported only when the feature is
reached:

```bash
pip install geometric        # geomeTRIC relaxation, in properties.optimize
pip install cppe             # polarizable embedding from a real potential file
pip install pyscf-dispersion # D3/D4 empirical dispersion
```

`Solvers.active_space_solver.Block2DMRGSolver` additionally needs `block2`:

```bash
pip install block2
```

block2's wheel links its own OpenMP runtime. With an MKL build of NumPy
(Intel's runtime), importing block2 after NumPy aborts the process (`OMP:
Error #15`) instead of raising a catchable error, while importing block2
first works; so `block2` must be imported before `numpy` in any process that
uses both. `KMP_DUPLICATE_LIB_OK=TRUE` suppresses the abort but can deadlock
the DMRG sweep.

`PolarizableSites`' own hand-rolled coupled-dipole response and PCM solvation
need nothing beyond pyscf.

The distributed eigensolve needs `mpi4py` and ELPA's `pyelpa`, both
optional; see [Distributed eigensolve](parallel.md#distributed-eigensolve).

There is no build step. Run from the repository root so that `src` is
importable.
