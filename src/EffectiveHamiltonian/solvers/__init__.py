"""The many-body solvers of an active-space Hamiltonian g_eff, as the two
builders of src/EffectiveHamiltonian produce it: solve.py dispatches by name
(`solve`, `SOLVERS`) over the engines of backends.py (FCI, the determinant
basis, DMRG, CCSDT/CCSDTQ behind the ActiveSpaceSolver interface);
determinants.py builds the explicit determinant-basis matrix and the
spin-orbital form of a spatial g_eff. Each entry point is imported from the
module that defines it; this package re-exports nothing.
"""
