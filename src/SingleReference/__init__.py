"""Standalone single-reference MBPT backend: quasiparticle energies, excitation
energies, RPA correlation energies, and 1-RDMs for a HF/KS reference. Each entry
point is imported from the module that defines it; this package re-exports
nothing, so importing one of its modules loads only what that module needs.

Density matrices from here
(`SingleReference.DensityMatrix.density_matrix.compute_gw_density_matrix`) and
from src/SingleReference/CC (compute_ccsd/ccsdt_density_matrix) are both
AO-basis 1-RDMs, usable interchangeably as calc_qp_energy(...,
dm_correction=...).

Three routes reach the same quasiparticle energy and differ only in cost:
`SingleReference.GW.qp_energy.calc_qp_energy` solves the Casida problem
explicitly (O(N^6)), `SingleReference.GW.imaginary_axis.solve_qp_energy_imaginary_axis`
integrates the self-energy on an imaginary frequency grid (O(N^4)), and
`SingleReference.GW.space_time.solve_qp_energy_space_time` forms it as a
pointwise product in imaginary time on a separable (ISDF) factorization of
the ERIs (O(N^3)). `SingleReference.BSE.solve.solve_bse_isdf` is the BSE built
on the same factors, and `SingleReference.BSE.solve.solve_bse_df` its twin on
pyscf's own density fitting; `SingleReference.BSE.solve.solve_bse` is the one
front end over every BSE route, choosing the solver and the integrals. Every
one of them evaluates the self-energy once, on the mean field's eigenvalues;
`SingleReference.GW.evGW.evgw_eigenvalues` reinjects the quasiparticle
energies into G and P0 to a fixed point, and is reached from the same front
door as self_consistency='evGW'.
"""
