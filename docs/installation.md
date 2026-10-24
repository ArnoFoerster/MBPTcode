# Installation

Requires Python 3.10+, NumPy, SciPy, PySCF, opt_einsum and threadpoolctl:

```bash
pip install numpy scipy pyscf opt_einsum threadpoolctl
```

`threadpoolctl` pins BLAS to one thread inside the thread pools of the
quasiparticle root scan and similar per-state loops
(`Base/utils/parallel/threads.py`). Without it the code runs and gives the
same results, but those loops may oversubscribe the cores. Thread settings are
under [Threads](parallel.md#threads).

`opt_einsum` is imported at module load by `CC/cached_einsum.py`, which most of
the tree pulls in, so it is not optional.

The restricted coupled-cluster integral path and the determinant-basis
active-space solver (`Solvers.active_space_solver.ExactDiagonalizationSolver`,
the default backend of `Solvers.active_space.solve`) need `openfermion`, which
is imported only when they are reached:

```bash
pip install openfermion
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

`psutil` is optional: off Linux, where pyscf cannot read a process's resident
size, the memory budgets of the ISDF and DF builds and of the EE-ADC Davidson
(`Base.utils.parallel.memory.current_memory_mb`) subtract what the process
already holds only when it is installed.

```bash
pip install psutil
```

`PolarizableSites`' own hand-rolled coupled-dipole response and PCM solvation
need nothing beyond pyscf.

The distributed eigensolve needs `mpi4py` and ELPA's `pyelpa`, both
optional; see [Distributed eigensolve](parallel.md#distributed-eigensolve).

The code generators in `src/CodeGeneration` (they derive the ADC, MPn
density, coupled-cluster and QDPT equations and write them out as NumPy
modules) need `pdaggerq`. The hand-written methods do not; the generated
charged-ADC levels (`ADCSolver(mf, level=..., pdaggerq=True)`: ADC(2) to
ADC(5) for ionization potentials and electron affinities) need the modules the
generators write, and those are not part of the repository: you produce them
on your own machine.
pdaggerq is not on PyPI; it builds from source with CMake:

```bash
git clone https://github.com/edeprince3/pdaggerq.git && cd pdaggerq && git checkout f3a1f29 && python -m pip install .
```

Then, from the repository root, `python src/CodeGeneration/generate_adcn.py`
writes the ADC(2), ADC(3) and ADC(4) base blocks to
`src/SingleReference/ADC/generatedADC/` (about a minute); `python
src/CodeGeneration/generate_adc.py` reports what each ADC order still lacks, and
`generate_adc.GenerateADC()` generates every registered module through ADC(5)
(the ADC(5) U^(4) derivation alone runs well over half an hour).
`generate_adc.GenerateRuntime(orders=(2, 3, 4))` writes everything the
generated charged-ADC solvers import at those orders (the dense blocks, the
matrix-free apply modules, their density-fitted twins, the Laplace and
Goldstone-diagram modules); add 5 for ADC(5), whose Laplace modules run to
hundreds of MB. A level whose modules are missing raises an ImportError
naming the call that writes them, and its tests are skipped.

```python
from pyscf import gto, scf
from src.CodeGeneration.generate_adc import GenerateRuntime
from src.SingleReference.ADC import ADCSolver

GenerateRuntime(orders=(2, 3, 4))          # once per machine, about 15 minutes
mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467', basis='sto-3g')
mf = scf.RHF(mol).run()
e, Z = ADCSolver(mf, level='adc4', pdaggerq=True, ntau=12,
                 construction='diagrammatic').solve()   # ADC(4)+ names its construction
print(e, Z)                                # the HOMO ionization pole (Hartree) and its weight
```
The generator tests skip when pdaggerq is missing. The diagram engine
(`src/CodeGeneration/diagrammatic`) and the QDPT generator also need `networkx`
(`pip install networkx`; openfermion already pulls it in). The MPn-density, CC
and QDPT generators rewrite modules that are part of the repository; a
regeneration differs from them in the header comments at least, so compare
before keeping one.

There is no build step. Run from the repository root so that `src` is
importable.
