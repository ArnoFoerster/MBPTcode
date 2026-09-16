# MBPTcode

Many-body perturbation theory for molecular systems, on top of
[PySCF](https://pyscf.org/): Dyson IP/EA-ADC, MPn density matrices, coupled
cluster, GW and linear response. Co-authored by Claude.

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

**evGW** — `calc_qp_energy(..., self_consistency='evGW')` drives any of those
three routes to a fixed point: the quasiparticle energies are reinjected into
G and P₀ until the spectrum stops moving, with every eigenvalue updated, DIIS
acceleration and convergence decided on the HOMO and LUMO. The equation stays
anchored on the mean field while the screening follows the iterate — anchored
on the iterate instead, each cycle adds its own correction a second time and
the gap runs away without ever converging. `evgw_eigenvalues` returns the whole
converged spectrum and a record of how it got there. See `examples/12_evgw.py`.

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

**Finite temperature** — Matsubara-axis grids via the intermediate
representation, for systems where the T = 0 grids (which key on the HOMO-LUMO
gap) are undefined.

## Install

Requires Python 3.10+, NumPy, SciPy, PySCF and opt_einsum:

```bash
pip install numpy scipy pyscf opt_einsum
```

`opt_einsum` is imported at module load by `CC/cached_einsum.py`, which most of
the tree pulls in, so it is not optional.

The coupled-cluster integral path additionally needs `openfermion` and
`openfermionpyscf`, imported only when they are reached:

```bash
pip install openfermion openfermionpyscf
```

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

The tests are standalone scripts that print their own verdicts and exit
non-zero on failure:

```bash
python tests/test_adc3.py
```

`pytest` collects almost nothing here — the checks live under
`if __name__ == '__main__'` — so run them as scripts and read the exit code.
`tests/README.md` maps which tests cover what.

## Layout

```
src/Base/               PySCF interface, constants, linear algebra
    separable_ri.py     ISDF / separable-RI factorization of the ERIs
    isdf_jk.py          ISDF Coulomb and exchange for the SCF
    environment.py      what the surroundings do, in one contract
    solvent_screening.py  PCM reaction field
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
                        the reaction field's shift, the evGW loop
    LinearResponse/     Casida, RPA, BSE, Davidson
src/Solvers/            quasiparticle root finders
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
