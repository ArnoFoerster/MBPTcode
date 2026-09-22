"""Compatibility shim: the diagonal quasi-boson EOM G0W0 solve lives in
`SingleReference.GW.quasi_boson`, and `QPqb` here is its adjoint subclass
`gradients.quasi_boson_adjoint.QPqbAdjoint` -- the same arithmetic on bosons
that carry the Frechet maps a gradient chains through.

`qp_energy_general` has no reverse half and is production's own.
"""
from src.SingleReference.GW.quasi_boson import qp_energy_general  # noqa: F401
from src.gradients.quasi_boson_adjoint import (  # noqa: F401
    QPqbAdjoint as QPqb)
