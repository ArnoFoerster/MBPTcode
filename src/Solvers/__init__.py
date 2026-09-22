"""Solver layer: root finders for the quasiparticle equation.

Self-energies live in src/SingleReference/. qp_equation.py holds every root
search in the codebase, including the pole-guarded Newton solve the
contour-deformation continuation uses.
"""
from src.Solvers.qp_equation import (
    solve_qp_equation, solve_qp_equation_graphical, solve_qp_equation_newton,
    solve_qp_equation_newton_batch, solve_qp_equation_newton_guarded,
    solve_qp_equation_bisection, solve_fixed_point, calculate_z_factor,
    spectral_function)
