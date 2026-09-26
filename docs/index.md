# MBPTcode documentation

## Using it

- [Installation](installation.md) — dependencies, required and optional
- [Quick start](getting-started.md) — a first ADC(3) run
- [Threads and MPI](parallel.md) — thread settings, the distributed ELPA
  eigensolve, and the routes that divide their work over MPI ranks
- [Basis sets from CP2K](basis-sets.md) — the aug-MOLOPT families and their RI
  tiers
- [Tests](testing.md) — the two test styles and how to run them
- [Source layout](layout.md) — where each part of the code lives

## Methods

- [ADC](methods/adc.md) — Dyson IP/EA-ADC(2)-X and ADC(3)
- [GW and linear response](methods/gw.md) — G0W0, evGW and qsGW on three routes
  from O(N⁶) to O(N³), Casida/RPA, RPA correlation energies, static screening
- [BSE](methods/bse.md) — the Davidson Bethe-Salpeter equation on ISDF or
  density-fitted factors
- [Low-scaling factorization](methods/isdf-low-scaling.md) — the separable RI
  (ISDF) behind the O(N³) routes and ISDF-J/K for the SCF
- [Environments](methods/environments.md) — polarizable-continuum screening,
  equilibrium solvation of charged states, an explicit polarizable shell,
  QM/MMPol embedding, empirical dispersion
- [Analytic nuclear gradients](methods/gradients.md) — dRPA and BSE@GW
  gradients, with an adjoint for every continuation and environment
- [Potential-energy surfaces and properties](methods/properties.md) — the
  surface dispatcher, geometry optimization, vibronic analysis, couplings and
  rates
- [Density matrices](methods/density-matrices.md) — MPn, GW and CC correlated
  1-RDMs
- [Coupled cluster](methods/coupled-cluster.md) — CCSD/CCSDT and EOM-CC
- [Finite temperature](methods/finite-temperature.md) — Matsubara-axis grids
- [Periodic systems](methods/periodic.md) — k-point RPA, BSE and GW for
  crystals, metals and slabs
