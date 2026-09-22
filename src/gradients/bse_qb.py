"""Compatibility shim: the BSE@G0W0 eigenproblem in the four
Toelle/Kitsaras/Loos variants lives in
`SingleReference.LinearResponse.quasi_boson_bse`, and `BSEqb` here is its
adjoint subclass `gradients.quasi_boson_adjoint.BSEqbAdjoint` -- the same
eigenpairs, plus `partials` and the two chains it drives.
"""
from src.gradients.quasi_boson_adjoint import (  # noqa: F401
    BSEqbAdjoint as BSEqb)
