"""Bethe-Salpeter equation beyond the static kernel.

bse_upfolded.py carries the frequency-free (upfolded) dynamical BSE of
Bintrim and Berkelbach, J. Chem. Phys. 156, 044114 (2022): the singles space
is augmented by the doubles that the dynamical kernel resums, so the
eigenvalue problem is linear and the frequency dependence is recovered
exactly on downfolding.

The static BSE lives in LinearResponse/ -- Casida (casida.py, dense) and the
matrix-free Davidson routes (davidson.py), including the ISDF one.
"""
from src.SingleReference.BSE.bse_upfolded import (build_hamiltonian, downfold,
                                                  dynamical_bse_matrix,
                                                  solve_bse_upfolded,
                                                  solve_dense)
