"""The Casida-level inputs of an `ExcitedStateChain` reverse pass, in the order
the BSE adjoints (`bse_backward`, `isdf_bse_backward`, ...) take them."""


def casida_args(chain, shared, solution):
    """(X_mo, D, eps_qp, W_aux, nocc, cache, Xn, Yn) of one forward
    (`SharedForward`) and its Casida step (`CasidaSolution`)."""
    return (shared.x_mo, shared.d, shared.eps_qp, shared.w_aux, chain.nocc,
            solution.cache, solution.x, solution.y)
