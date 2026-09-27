# MBPTcode

Many-body perturbation theory for molecular systems, on top of
[PySCF](https://pyscf.org/): Dyson IP/EA-ADC, MPn density matrices, coupled
cluster, GW and linear response, and the analytic nuclear gradients and
potential-energy-surface properties built on top of them. GW, BSE and RPA also
run with k-point sampling, for crystals and slabs.

**Documentation: [docs/index.md](docs/index.md).**

## Methods

- **[ADC](docs/methods/adc.md)** — Dyson IP/EA-ADC(2)-X and ADC(3), RHF
  spin-adapted and UHF/spin-orbital, dense and matrix-free.
- **[GW and linear response](docs/methods/gw.md)** — G0W0, evGW/evGW0 and
  qsGW/qsGW0, restricted or unrestricted, on three routes from O(N⁶) to O(N³);
  Casida/RPA and RPA correlation energies.
- **[BSE](docs/methods/bse.md)** — the Davidson Bethe-Salpeter equation,
  singlet or triplet, on ISDF or density-fitted factors.
- **[Low-scaling factorization](docs/methods/isdf-low-scaling.md)** — the
  separable RI (ISDF) behind the O(N³) routes, and ISDF-J/K for the SCF.
- **[Environments](docs/methods/environments.md)** — polarizable-continuum
  screening, equilibrium solvation of charged states, an explicit polarizable
  shell, QM/MMPol embedding, empirical dispersion.
- **[Analytic nuclear gradients](docs/methods/gradients.md)** — cubic-scaling
  dRPA and BSE@GW gradients on a frozen ISDF factorization and their dense
  counterparts, with an adjoint for every continuation and environment.
- **[Potential-energy surfaces](docs/methods/properties.md)** — one surface
  dispatcher, geometry optimization, vibronic analysis, conformers,
  spin-orbit and nonadiabatic couplings, rates.
- **[Density matrices](docs/methods/density-matrices.md)** — MP2/MP3/MP4, GW
  and CC correlated 1-RDMs.
- **[Coupled cluster](docs/methods/coupled-cluster.md)** — CCSD/CCSDT and
  EOM-CC (IP/EA/EE).
- **[Finite temperature](docs/methods/finite-temperature.md)** — Matsubara-axis
  grids via the intermediate representation.
- **[Periodic systems](docs/methods/periodic.md)** — k-point RPA, BSE and GW
  for crystals, metals and slabs.

## Install

Requires Python 3.10+, NumPy, SciPy, PySCF, opt_einsum and threadpoolctl:

```bash
pip install numpy scipy pyscf opt_einsum threadpoolctl
```

There is no build step. Run from the repository root so that `src` is
importable. Optional dependencies (coupled cluster, geomeTRIC, cppe, D3/D4
dispersion, the distributed ELPA eigensolve) are in
[docs/installation.md](docs/installation.md).

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

Threads and MPI, basis sets from CP2K, the tests and the source layout are
in [docs/](docs/index.md).

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
