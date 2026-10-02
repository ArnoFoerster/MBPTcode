"""Solver layer: root finders for the quasiparticle equation, a matrix-free
symmetric eigensolver, and the active-space many-body solvers.

Self-energies live in src/SingleReference/. qp_equation.py holds every root
search in the codebase, shared by the GW and ADC front ends, including the
pole-guarded Newton solve the contour-deformation continuation uses and the
fixed-point driver. davidson.py holds the one matrix-free symmetric
eigensolver both ADC routes -- charged (IP/EA) and neutral (ee) -- iterate on.
active_space_solver.py holds the ActiveSpaceSolver interface and its FCI,
determinant-basis and DMRG adapters; exact_diagonalization.py builds the
determinant-basis matrix ExactDiagonalizationSolver diagonalizes. active_space.py
dispatches a downfolded effective Hamiltonian (as built by
src/MultiReference/QDPT/perturbative.py) over those same adapters by name.
"""
from src.Solvers.qp_equation import (
    solve_qp_equation, solve_qp_equation_graphical, solve_qp_equation_newton,
    solve_qp_equation_newton_batch, solve_qp_equation_newton_guarded,
    solve_qp_equation_bisection, solve_fixed_point, calculate_z_factor,
    spectral_function)
from src.Solvers.davidson import (solve_symmetric, overlap_pick,
                                  jacobi_davidson_precond, diagonal_seeds)
from src.Solvers.active_space_solver import (ActiveSpaceSolver,
                                             Block2DMRGSolver,
                                             ExactDiagonalizationSolver,
                                             PyscfFCISolver, spin_sector)
from src.Solvers.exact_diagonalization import (build_explicit_hamiltonian,
                                               spatial_to_spin_orbitals)
from src.Solvers.active_space import solve
