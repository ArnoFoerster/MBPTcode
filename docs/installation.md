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
`openfermionpyscf`, imported only when they are reached:

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

`PolarizableSites`' own hand-rolled coupled-dipole response and PCM solvation
need nothing beyond pyscf.

The distributed eigensolve needs `mpi4py` and ELPA's `pyelpa`, both
optional; see [Distributed eigensolve](parallel.md#distributed-eigensolve).

There is no build step. Run from the repository root so that `src` is
importable.
