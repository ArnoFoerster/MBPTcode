"""Solver layer: everything that *solves* a Hamiltonian or dynamical equation.

Self-energies live in src/SingleReference/. qp_equation.py holds the shared
quasiparticle root finders used by the GW and ADC front ends. davidson.py
holds a generic matrix-free symmetric eigensolver.
"""
from src.Solvers.qp_equation import (
    solve_qp_equation, solve_qp_equation_graphical, solve_qp_equation_newton,
    solve_qp_equation_newton_batch, solve_qp_equation_bisection, calculate_z_factor,
    spectral_function)
from src.Solvers.davidson import davidson
