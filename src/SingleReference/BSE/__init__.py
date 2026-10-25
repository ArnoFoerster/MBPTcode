"""Bethe-Salpeter equation beyond the static kernel.

bse_upfolded.py carries the frequency-free (upfolded) dynamical BSE of
Bintrim and Berkelbach, J. Chem. Phys. 156, 044114 (2022): the singles space
is augmented by the doubles that the dynamical kernel resums, so the
eigenvalue problem is linear and the frequency dependence is recovered
exactly on downfolding.

solve.py is the static BSE@GW: the front end `solve_bse` and its drivers,
on the Casida eigensolvers of LinearResponse/ (casida.py, dense; davidson.py,
matrix-free). quasi_boson_bse.py is the dense quasi-boson route.
"""
