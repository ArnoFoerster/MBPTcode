"""Solver layer: root finders for the quasiparticle equation, a matrix-free
symmetric eigensolver, and DIIS. The active-space many-body solvers live in
src/EffectiveHamiltonian/solvers/, beside the effective Hamiltonians they
solve.

Self-energies live in src/SingleReference/. qp_equation.py holds the one
quasiparticle root finder of the code base (`find_qp_root`: bracket, refine,
keep the largest valid pole strength, or along a walk the root continuing
the previous geometry's), shared by every GW and ADC front end. davidson.py
holds the one matrix-free symmetric eigensolver (`solve_symmetric`): both ADC
routes, the Tamm-Dancoff BSE and the (A - B) probe iterate on it. The full
Casida/BSE problem (paired, row-distributed) and the upfolded BSE
(non-symmetric) keep loops of their own; davidson.py says why. diis.py holds
the DIIS extrapolation of the CC, GW and ADC fixed-point loops.
"""
