"""Solver layer: root finders for the quasiparticle equation and a matrix-free
symmetric eigensolver.

Self-energies live in src/SingleReference/. qp_equation.py holds every root
search in the codebase, shared by the GW and ADC front ends, including the
pole-guarded Newton solve the contour-deformation continuation uses and the
fixed-point driver. davidson.py holds a generic matrix-free symmetric
Davidson-Liu eigensolver.
"""
from src.Solvers.qp_equation import (
    solve_qp_equation, solve_qp_equation_graphical, solve_qp_equation_newton,
    solve_qp_equation_newton_batch, solve_qp_equation_newton_guarded,
    solve_qp_equation_bisection, solve_fixed_point, calculate_z_factor,
    spectral_function)
from src.Solvers.davidson import davidson
