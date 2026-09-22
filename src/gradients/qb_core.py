"""Compatibility shim: the quasi-boson layer's forward half lives in
`SingleReference.GW.quasi_boson` and its reverse half in
`gradients.quasi_boson_adjoint`; every name this module exports resolves to
one of those two, so a caller reaches the same objects under the names it has
always used.

`RPA` is the ADJOINT subclass, which is what this name has always meant on the
gradient side: the forward-only class of the same name is
`GW.quasi_boson.RPA`, and it carries no Frechet map.
"""
from src.SingleReference.GW.quasi_boson import (build_rpa_AB,  # noqa: F401
                                                couplings_V, eigh_sym,
                                                funm_sym, invsqrtm_sym,
                                                sqrtm_sym)
from src.gradients.quasi_boson_adjoint import (  # noqa: F401
    RPAAdjoint as RPA, frechet_funm_sym)
