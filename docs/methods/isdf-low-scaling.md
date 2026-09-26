# Low-scaling factorization

**Low-scaling factorization** — the separable RI of Duchemin and Blase
([J. Chem. Phys. 150, 174120 (2019)](https://doi.org/10.1063/1.5090605)),
with optimized atomic interpolation grids. It backs the space-time GW route,
`solve_bse_isdf` (a BSE on the same factors), and ISDF-J/K for the SCF, which
replaces the density-fitted `cderi` and so removes the three-index tensor from
the memory budget.
