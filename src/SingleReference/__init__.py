"""Standalone single-reference MBPT backend: quasiparticle energies, excitation energies, RPA correlation energies, and 1-RDMs for a HF/KS reference.

Density matrices from here (compute_gw_density_matrix) and from
src/SingleReference/CC (compute_ccsd/ccsdt_density_matrix) are both AO-basis
1-RDMs, usable interchangeably as calc_qp_energy(..., dm_correction=...).

Three routes reach the same quasiparticle energy and differ only in cost:
calc_qp_energy solves the Casida problem explicitly (O(N^6)),
solve_qp_energy_imaginary_axis integrates the self-energy on an imaginary
frequency grid (O(N^4)), and solve_qp_energy_space_time forms it as a
pointwise product in imaginary time on a separable (ISDF) factorization of
the ERIs (O(N^3)). solve_bse_isdf is the BSE built on the same factors, and
solve_bse_df its twin on pyscf's own density fitting. Every one of them
evaluates the self-energy once, on the mean field's eigenvalues; evgw_eigen-
values reinjects the quasiparticle energies into G and P0 to a fixed point,
and is reached from the same front door as self_consistency='evGW'.
"""
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.GW.transition_amplitudes import AmplitudeGenerator
from src.SingleReference.LinearResponse.casida import CasidaSolver, CasidaResult
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.evGW import evgw_eigenvalues
from src.SingleReference.LinearResponse.davidson import (solve_casida_davidson,
                                                        solve_bse_isdf,
                                                        solve_bse_df,
                                                        isdf_bse_factors,
                                                        isdf_df_coefficients,
                                                        lowest_amb_eigenvalue,
                                                        oscillator_strengths)
from src.SingleReference.DensityMatrix.density_matrix import (GWDensityMatrixSolver,
                                                 compute_gw_density_matrix,
                                                 MP2DensityMatrixSolver,
                                                 compute_mp2_density_matrix,
                                                 compute_mp2_density_matrix_ao,
                                                 MP3DensityMatrixSolver,
                                                 compute_mp3_density_matrix)
from src.SingleReference.GW.imaginary_axis import (
    solve_screening_imaginary_axis, self_energy_imaginary_axis,
    solve_qp_energy_imaginary_axis)
from src.SingleReference.GW.space_time import (solve_qp_energy_space_time,
                                               separable_factors)
from src.SingleReference.LinearResponse.rpa_energy import (
    rpa_correlation_energy_casida, rpa_correlation_energy_imaginary_axis)
from src.SingleReference.ADC import (
    ADCSolver, ADCSolverRestricted, ADCSolverUnrestricted, build_static_correction)
