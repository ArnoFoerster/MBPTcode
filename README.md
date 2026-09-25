# MBPTcode

Many-body perturbation theory for molecular systems, on top of
[PySCF](https://pyscf.org/): Dyson IP/EA-ADC, MPn density matrices, coupled
cluster, GW and linear response, and the analytic nuclear gradients and
potential-energy-surface properties built on top of them. Co-authored by
Claude.

## Methods

**ADC** — Dyson IP/EA-ADC in the algebraic-diagrammatic-construction
hierarchy of Schirmer, Cederbaum and Walter,
[Phys. Rev. A 28, 1237 (1983)](https://doi.org/10.1103/PhysRevA.28.1237).

| branch | levels |
|---|---|
| RHF, spin-adapted (CSF basis) | ADC(2)-X, ADC(3) |
| UHF / spin-orbital | ADC(3) |

Both branches have dense and matrix-free (Davidson) routes, density
fitting, Epstein-Nesbet denominator dressing, and static self-energy
corrections.

**Density matrices** — MP2, MP3 and MP4 correlated 1-RDMs, restricted and
unrestricted, dense and DF, with Laplace-fused amplitude routes. GW and
CCSD/CCSDT density matrices too. Generated with pdaggerq.

**Coupled cluster** — CCSD and CCSDT amplitudes and lambda equations,
restricted and spin-orbital, plus EOM-CC (IP/EA/EE). Generated with pdaggerq.

**Screening** — static RPA screened Coulomb interaction W, and the screened
C^(1) block of the screened multichannel Dyson equation, following
Romaniello and Berger, [arXiv:2603.27329](https://arxiv.org/abs/2603.27329).

**GW / linear response** — G0W0 and eigenvalue-self-consistent GW on the
real and imaginary axes, Casida/RPA/BSE, RPA correlation energies.

Three routes reach the same quasiparticle energy and differ only in cost:

| function | how W and Sigma are built | cost |
|---|---|---|
| `calc_qp_energy` | Casida problem solved explicitly | O(N⁶) |
| `solve_qp_energy_imaginary_axis` | quadrature on an imaginary-frequency grid | O(N⁴) |
| `solve_qp_energy_space_time` | pointwise product in imaginary time, on a separable (ISDF) factorization of the ERIs | O(N³) |

A correlated density matrix is passed to any of them as
`dm_correction=`. The `dm_ccsd=` alias for that argument has been **removed**;
it was already marked deprecated, and callers that still use it now raise
`TypeError`.

The imaginary-axis routes reach the real axis by one of four continuations,
`calc_qp_energy(..., continuation=...)`: `'pade'` (Thiele-Pade of
Sigma_c(i.omega), the default, no analytic gradient); `'cd'` (contour
deformation — the omega' contour rotated onto the imaginary axis, the poles of
G it sweeps over collected as residues, no continuation at all); `'sop'` (W
modelled by M poles fit on the imaginary axis, so Sigma_c is closed-form and
never evaluated off it, valence states only); and `'spectral'` (the Lehmann
sum at omega + i.eta, `mode='casida'` only). `MODE_CONTINUATIONS` is the
validity table pairing modes with the continuations they accept; a keyword
another continuation reads raises `TypeError` naming both.

A dense route (`GW.quasi_boson`, `LinearResponse.quasi_boson_bse`) builds the
same dRPA/BSE amplitudes as an explicit auxiliary-boson diagonalization rather
than a Davidson iteration — bitwise the physics of `casida(eta=0)`, at O(N^6),
and what the dense analytic gradients below differentiate.

**evGW** — `calc_qp_energy(..., self_consistency='evGW')` drives any of those
three routes to a fixed point: the quasiparticle energies are reinjected into
G and P₀ until the spectrum stops moving, with every eigenvalue updated, DIIS
acceleration and convergence decided on the HOMO and LUMO. The equation stays
anchored on the mean field while the screening follows the iterate — anchored
on the iterate instead, each cycle adds its own correction a second time and
the gap runs away without ever converging. `evgw_eigenvalues` returns the whole
converged spectrum and a record of how it got there. See `examples/12_evgw.py`.
`self_consistency='evGW0'` reinjects them into G alone: the mean field's P₀ and
W stay and only the poles of Σ_c move, on the Casida route.

**qsGW** — `self_consistency='qsGW'` (or `'qsGW0'`, W kept at the mean field)
runs the quasiparticle-self-consistent loop on the Casida route: a static
Hermitian self-energy replaces v_xc, h + J + K + Σ̃ is diagonalized, and the
orbitals and eigenvalues are reinjected until the density and the frontier
eigenvalues stop moving. Σ̃ is the SRG-regularized form of Marie and Loos
([J. Chem. Theory Comput. 19, 3943 (2023)](https://doi.org/10.1021/acs.jctc.3c00281))
at flow s = 100 Ha⁻², since Kotani's mode A on the pole sum has no fixed
point. `qsgw_eigenvalues` returns the spectrum, the orbitals and, for a BSE on
top, the DF factors and static W of the result. Restricted closed shell, gas
phase.

**Low-scaling factorization** — the separable RI of Duchemin and Blase
([J. Chem. Phys. 150, 174120 (2019)](https://doi.org/10.1063/1.5090605)),
with optimized atomic interpolation grids. It backs the space-time GW route,
`solve_bse_isdf` (a BSE on the same factors), and ISDF-J/K for the SCF, which
replaces the density-fitted `cderi` and so removes the three-index tensor from
the memory budget.

**BSE** — the iterative (Davidson) Bethe-Salpeter equation, singlet or
triplet, in two interchangeable flavours that share every convention:
`solve_bse_isdf` on ISDF factors (one factorization for the GW that feeds it
and the kernel) and `solve_bse_df` on pyscf's own Coulomb fit. Both take
`self_consistency='evGW'`.

Which eigenvalues build the static W is set by the LEVEL OF THEORY: G0W0
screens W₀ at the mean-field eigenvalues, which is the standard split, and evGW
at its converged ones. An explicit `qp=` array therefore follows the G0W0
convention unless `screen_at='qp'` says the array is itself a self-consistent
spectrum.

**Solvent** — polarizable-continuum screening in the style of Duchemin,
Jacquemin and Blase, [J. Chem. Phys. 144, 164106 (2016)](https://doi.org/10.1063/1.4946778).
Non-equilibrium solvation is two dielectric constants and needs both: the
ground state relaxes inside PCM(ε_static), since the solvent nuclei have had
time to reorient around it, and only the *response* is optical. One
`SolventScreening` carries both — `env.mean_field(mol, factory)` applies the
first, `attach_environment(mf, env)` the second, which substitutes v → v + ṽ
at the integral chokepoints.

Inside GW the reaction field is then Duchemin, Guido, Jacquemin and Blase,
Chem. Sci. 9, 4430 (2018), Eq. (18): the
self-polarization of the orbital carrying the added charge in the *screened*
reaction field, on every route, with Σ itself screened by the bare interaction
— screening Σ dynamically as well counts the same polarization twice. The
static COHSEX operator remains the fallback for the routes that never form W
(ADC, an unrestricted reference); the two differ by 0.39 eV of quasiparticle
gap on water in water. See `examples/13_solvated_gw_bse.py`.

Both halves of the reaction field's nuclear derivative are analytic —
`aux_kernel_adjoint` for the dressed metric, `static_self_energy_adjoint` for
the static term — on the bilinear cavity derivative of `pcm_derivatives.py`;
a solvated excitation gradient reproduces finite differences at the same
8e-8-relative floor as the gas-phase chain.

**Empirical dispersion** — a semi-classical -C6/R^6 correction (D3/D4, via
pyscf's own `xc='pbe0-d4'`-style functional names) for a hybrid, which has no
long-range correlation of its own. It is a function of the nuclear
coordinates alone, so it moves the surface without moving the spectrum: every
orbital, and therefore every excitation energy, is bitwise unchanged by it.
It must never sit under a direct-RPA ground state, whose correlation energy
already contains dispersion, nor be added twice to a functional (r2SCAN-3c,
wB97X-V, ...) that already carries a correction of its own — both are refused
rather than silently double-counted.

**Polarizable embedding** — classical QM/MMPol, in the style of Li, D'Avino,
Duchemin, Beljonne and Blase,
[Phys. Rev. B 97, 035108 (2018)](https://doi.org/10.1103/PhysRevB.97.035108).
A site's induced dipole couples to every other site's,
`mu = (alpha^-1 - T)^-1 E`, giving the classical response matrix B that
dresses the interaction exactly as the PCM continuum's v -> v + ṽ does, field
(dipole) response in place of charge (surface) response. `PolarizableSites`
builds B from a hand-rolled, isotropic, uniformly Thole-damped coupling;
`cppe_interface.py` sources it instead from a real force-field potential file
(PyFraME, DALTON's PE library) through [cppe](https://github.com/maxscheurer/cppe),
with per-atom anisotropic tensors and 1-2/1-3 exclusions. `composite_environment.py`
carries permanent charges and polarizable sites as ONE environment with
neither channel leaking into the other's: permanent multipoles reach the
mean field and contribute nothing to the reaction-field kernel, sites screen
and leave the ground state alone — the split the paper's Sec. II A and II D
draw.

**Analytic nuclear gradients** — `src/gradients/` differentiates production's
own forward objects (`src.SingleReference`, `src.Base`) rather than a second
copy of them: the Lagrangian of Toelle,
[arXiv:2412.17085](https://arxiv.org/abs/2412.17085), and Toelle, Kitsaras and
Loos, [arXiv:2507.02160](https://arxiv.org/abs/2507.02160), with the papers'
iterative BCH/truncated-Taylor machinery replaced by exact closed forms.
`RPAGroundStateChain` carries the cubic-scaling dRPA ground-state gradient and
`ExcitedStateChain` the cubic-scaling BSE@GW one, both on a frozen ISDF
factorization; `DenseRPASurface` and `DenseBSESurface` carry the same two
gradients at O(N^6), on the dense quasi-boson layer.
Every continuation above (Pade, contour deformation, sum-over-poles) and every
environment above (PCM, dispersion, polarizable sites) has its own adjoint,
so a quasiparticle or excitation gradient is exact for whichever route and
surroundings computed the energy, not a finite difference of a different one.
One orbital-response (Z-vector) solve is shared by every energy target a
chain carries. See `examples/16_numerical_hessian_from_gradient.py` for a
property built directly from the gradient: a vibrational analysis by central
differences of the analytic force, exact against pyscf's own analytic Hessian
where that exists, and the only route where it does not (an ISDF mean field).

**Potential-energy surfaces** — `src/properties/` computes FROM a surface
rather than BY one. `potential_energy_surface(ground_state, excitation=None,
charge=None, ...)` dispatches on the DECLARED physics (`src.Base.declaration`:
`GroundState`, `Excitation`, `ChargedExcitation`, `QPStates`) to the gradient
chain that realizes it and records the realization; `compare_surfaces` refuses
to difference two surfaces that do not share a ground-state functional and
environment. `calc_vertical_excitation`, `calc_emission_energy`,
`calc_adiabatic_excitation` and `calc_adiabatic_gap` build both surfaces of a
difference from one `SurfaceSpec`, so the functional under an excited state
and under its ground state cannot silently be two different functionals.
`optimize`/`relax` walk Cartesian RFO/BFGS with the rigid-body directions
projected out, or [geomeTRIC](https://github.com/leeping/geomeTRIC) when it is
installed; `vibronic` gives normal modes and Huang-Rhys factors by two
independent routes; `conformers` the Boltzmann-weighted average over torsional
minima a soft emitter has; `spin_orbit` and `nonadiabatic` the two couplings a
`rates` Marcus-Levich-Jortner or golden-rule rate needs; `characters` labels a
BSE root by its charge-transfer weight; `hessian` the nuclear Hessian by
central differences of the analytic gradient, for a route pyscf's own
analytic Hessian cannot serve. See
`examples/15_excited_state_geometry_optimization.py` for the one entry point,
`calc_adiabatic_excitation`, driving an excited-state relaxation end to end,
and `examples/17_spin_orbit_coupling.py` for `spin_orbit` read at the two
minima `calc_adiabatic_gap` already relaxes (El-Sayed's rule, computed rather
than assumed) plus the Herzberg-Teller dV/dq scan over `vibronic`'s
ground-state modes that `rates.spin_vibronic_rate` needs for an
El-Sayed-forbidden pair, where the Condon term alone is not the whole story.

**Finite temperature** — Matsubara-axis grids via the intermediate
representation, for systems where the T = 0 grids (which key on the HOMO-LUMO
gap) are undefined.

## Install

Requires Python 3.10+, NumPy, SciPy, PySCF, opt_einsum and threadpoolctl:

```bash
pip install numpy scipy pyscf opt_einsum threadpoolctl
```

`threadpoolctl` pins BLAS to one thread inside the quasiparticle root scan's
thread pool and is imported at module load by `GW/qp_energy.py`. Thread
settings are under [Threads](#threads).

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
optional; see [Distributed eigensolve](#distributed-eigensolve).

There is no build step. Run from the repository root so that `src` is
importable.

## Quick start

```python
from pyscf import gto, scf
from src.SingleReference.ADC import ADCSolver

mol = gto.M(atom='O 0 0 0; H 0 0 0.958; H 0.926 0 -0.240', basis='cc-pVDZ')
mf = scf.RHF(mol).run()

e, Z = ADCSolver(mf, level='adc3').solve()
print(f"ADC(3) IP = {-e[0] * 27.2114:.3f} eV   Z = {Z[0]:.3f}")
```

See `examples/` for density fitting, Epstein-Nesbet variants, open-shell
references, several ionization states, and screened singles.

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

In a Slurm job:

```bash
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread       # 16 cores; without it, 8 cores and their SMT siblings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_WAIT_POLICY=PASSIVE
unset OMP_PROC_BIND
```

Several processes in one job step each get their own cores with
`srun --ntasks=R --cpus-per-task=T --hint=nomultithread` and `OMP_NUM_THREADS=T`.

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

from src.Base.utils.linearAlgebra.diagonalization import (serve_distributed_solves,
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

In a Slurm job, one node as 8 ranks of 24 cores:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export ELPA_DEFAULT_omp_threads=$OMP_NUM_THREADS
srun --mpi=pmix --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=cores python run.py
```

ELPA's own OpenMP threads default to one per rank, and `OMP_NUM_THREADS` does
not reach them; `ELPA_DEFAULT_omp_threads` does, if `pyelpa` is linked against
the OpenMP build of ELPA (`libelpa_openmp`). Bind the ranks to cores, with
`--cpu-bind=cores` or `mpirun --bind-to core --map-by slot:PE=24`: unbound, a
test solve ran at least ten times slower. `--mpi=pmix` suits Open MPI 5;
`srun --mpi=list` shows what your Slurm offers.

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
the setup, run `tests/test_elpa_casida.py` under `srun` or `mpirun`: on more
than one rank it fails if a solve fell back to `eigh`.

## Running under MPI

The space-time GW routes, the ISDF fit, the BSE Davidson and the
density-fitted SCF divide their work over MPI ranks. Every rank runs the whole
script -- the SCF loop, the Davidson, the gradient chains, the geometry walk --
and MPI lives only inside the kernels that realize the physics: the tau and
frequency sweeps of the GW self-energy and its adjoints, the three-centre pass
of the ISDF fit, the rows of the screened kernel in the BSE block action, and
the auxiliary rows and grid points of the SCF's J, K and exchange-correlation
potential. A driver never takes a communicator; the kernels read it from the
region the script opens once:

```python
from pyscf import dft, gto

from src.Base.distributed_df import distributed_mean_field
from src.Base.utils.mpi_grid import distributed, grid_comm
from src.SingleReference.LinearResponse.davidson import solve_bse_isdf

comm = grid_comm()[0]                  # COMM_WORLD, or None without mpi4py
with distributed(comm):
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
                basis='cc-pvdz')
    mf = dft.RKS(mol, xc='pbe0').density_fit()
    distributed_mean_field(mf)         # J/K and the xc grid split over ranks
    omega, X, Y, info = solve_bse_isdf(mf, mol, mol.nelectron // 2, nroots=5)
    if comm is None or comm.Get_rank() == 0:
        print(omega)                   # the same bits on every rank
```

Launch it with `mpirun -n R python run.py`, or `srun --mpi=pmix` in a Slurm
job, with `OMP_NUM_THREADS` set to the cores each rank may use (see
[Threads](#threads)). Without `mpi4py`, or with `MBPT_USE_MPI=0`, `grid_comm`
returns None and the same script runs serially, bit for bit the serial code.
`mpi4py` is imported on first use, never at module import.

WHY THE RANKS AGREE. Each rank converges its own arithmetic, and two nodes do
not repeat each other's last bits: an orbital energy, an interpolation point
or a Davidson residual can differ, and a discrete decision taken from it -- a
grid size, a trial-vector count, when to stop -- then differs outright. So
every kernel `lockstep`s at its entry the inputs that can differ between ranks
(rank 0's copy is written into every rank's buffers), and gathers or
all-reduces what it computes. Its output is then the same bits on every rank,
every decision a driver takes from it is the same, and the replicated drivers
stay in step without any protocol between them. Inputs one kernel hands the
next are identical by construction and are not broadcast again;
`with distributed(comm, audit=True):` makes the kernels compare 64-bit digests
of them (`mpi_grid.agreement`) and count the repairs their locksteps made
(`mpi_grid.lockstep_stats()`), which is how a run shows that every kernel held
one set of bits on every rank.

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

## Basis sets from CP2K

CP2K's aug-MOLOPT families, all-electron bases with tiered RI sets built for GW
and BSE, are read from CP2K at run time and registered as PySCF basis names:

```python
from src.Base.basis.cp2k_basis import register

basis, aux = register('aug-SZV-MOLOPT-ae', max_error=1e-4)
mol = gto.M(atom='O 0 0 0; H 0 0 0.958; H 0.926 0 -0.240', basis=basis)
```

Without `max_error` you get the tightest RI tiers: converged, but possibly far
larger than needed. Choose the tier deliberately; the RI name then carries the
largest Delta-I among the tiers it holds, so one name means one set. Data
sources, offline use and the trade-off are in `src/Base/basis/README.md`.

## Tests

Two styles coexist. The ADC/CC/density-matrix/GW-core suite are standalone
scripts that print their own verdict and exit non-zero on failure:

```bash
python tests/test_adc3.py
```

The gradients/properties/environment suite (`src/gradients`, `src/properties`,
`src/Base/{declaration,dispersion,polarizable_sites,composite_environment,
cppe_interface,pcm_factorization,pcm_derivatives}.py` and their
`SingleReference` dependents) is ordinary pytest:

```bash
pytest tests/test_excited_state.py
```

`tests/README.md` maps which tests cover what.

## Layout

```
src/Base/               PySCF interface, constants, linear algebra
    declaration.py      the GroundState/Excitation/QPStates vocabulary a
                        surface is built from and validated against
    separable_ri.py     ISDF / separable-RI factorization of the ERIs
    isdf_jk.py          ISDF Coulomb and exchange for the SCF
    environment.py      what the surroundings do, in one contract
    solvent_screening.py  PCM reaction field, with its analytic adjoint
    dispersion.py        the empirical D3/D4 correction
    composite_environment.py, polarizable_sites.py, cppe_interface.py
                        QM/MMPol: permanent charges plus induced dipoles
    pcm_factorization.py, pcm_derivatives.py
                        the PCM cavity's cached solve and its nuclear
                        derivative primitives
    basis/cp2k_basis.py CP2K's aug-MOLOPT orbital and RI sets, read at run time
    utils/grids.py      minimax and Gauss-Legendre imaginary-axis grids
    utils/time_frequency.py  one grid object carrying both axes
    utils/matsubara.py  finite-temperature (IR) grids
src/SingleReference/
    ADC/                the ADC solvers (see ADC/__init__.py for the map)
    CC/                 CCSD/CCSDT amplitudes, lambda, EOM
    DensityMatrix/      MPn / GW / CC correlated 1-RDMs
    EpsteinNesbet/      EN denominators and shifts
    GW/                 self-energy, QP equation, imaginary axis/time,
                        the reaction field's shift, the evGW and qsGW
                        loops, contour deformation and sum-over-poles
                        continuations, the dense quasi-boson route
    LinearResponse/     Casida, RPA, BSE, Davidson, the dense quasi-boson BSE
    BSE/                the upfolded (non-perturbative) BSE
src/Solvers/            quasiparticle root finders, including the
                        pole-guarded Newton solve contour deformation uses,
                        and a matrix-free Davidson eigensolver
src/gradients/          analytic nuclear gradients: one adjoint module per
                        forward one, differentiating production's own objects
src/properties/         the ONE surface dispatcher, geometry optimization,
                        vibronic analysis, conformers, rates and the
                        couplings they need
```

## License

MIT — see [LICENSE](LICENSE). Free for any use, including commercial and
closed-source; the only condition is that the copyright notice is kept.

A few files are third-party components under the Apache License 2.0 —
the CC DIIS routine and CCSDT amplitude equations (from
[pdaggerq](https://github.com/edeprince3/pdaggerq)), and the minimax
quadrature tables plus the imaginary time/frequency transformation weights
ported from Fortran in `src/Base/utils/time_frequency.py` (from
[GreenX](https://github.com/nomad-coe/greenX)).
They are listed in [NOTICE](NOTICE), which must be retained in
redistributions.
