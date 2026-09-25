"""The nuclear force of a mean field whose exchange comes from ISDF factors.

WHAT IS WRONG WITHOUT THIS
--------------------------
`src.Base.isdf_jk.ISDFJK` gives an SCF a `with_df` that answers `get_jk` with
integral-direct DF Coulomb and exchange built from the interpolative separable
density fit,

    K = X^T [Z .* (X D X^T)] X,   X[P,mu] = chi_mu(r_P),   Z = M^T V M.

pyscf's own `mf.Gradients()` differentiates the FITTED interaction and knows
nothing of the interpolation points or of M, so the force it returns is the
gradient of a different function than the energy just reported: on
water/cc-pVDZ/B3LYP it misses 4.0e-4 Ha/Bohr, and the miss is CONSTANT as the
finite-difference step shrinks while the difference itself converges as h^2.

THE ONE TERM THAT IS WRONG, AND THE ASSEMBLY THAT REPLACES IT
-------------------------------------------------------------
Everything else in that mean field is pyscf's and correct: the one-electron and
nuclear terms, the overlap (Pulay) term, the exchange-correlation functional,
and -- on the default `j_route='df-direct'` -- the Coulomb term, which comes
from pyscf's integral-direct DF-J on the same auxiliary basis. So the force is

    dE/dR = g[reference functional MINUS its exact-exchange fraction]
          + d/dR E_K^ISDF at fixed density,

with the first member ONE pyscf gradient at the ISDF orbitals rather than a
hand-assembled sum: `exchange_free_reference` removes the exact exchange by
subtracting `a_x*HF` from the functional, which leaves the exchange-correlation
grid term, the Coulomb term, the one-electron term and the energy-weighted
overlap term bit-identical to what the ISDF route actually minimized. Building
those by hand instead is where the coefficients go wrong, and the check that
the two halves are consistent is exact: putting the FITTED exchange skeleton
(`fock_partial_skeleton_df` at gamma = D, halved) in place of the ISDF one
reproduces pyscf's own density-fitted gradient to 1e-12.

No orbital-response term appears, and that is not an approximation. The ISDF
exchange matrix is EXACTLY the functional derivative of the ISDF exchange
energy, dE_K/dD = -(a_x/2) K, so the SCF is variational for the energy this
route reports and the Hellmann-Feynman-plus-Pulay structure holds unchanged.

THE ENERGY, AS A SCALAR IN THE FACTORS
--------------------------------------
K's four collocations collapse onto one symmetric (M, M) object:

    E_K = -(a_x/4) Tr[D K] = -(a_x/4) sum_PQ Z_PQ W_PQ^2,   W = X D X^T

(closed shell, Tr D = N). Its reverse pass is therefore two adjoints, one on
the collocation and one on Z, and from there the chain is the one the
correlated route already runs: the fit's Z-vector, the two- and three-centre
derivative integrals, the AO and auxiliary centres, and the interpolation
points translating and turning with their atoms.

X CARRIES BOTH GEOMETRY DEPENDENCES and both are carried: the basis functions
differentiated at a fixed point (`basis_centre_forces`) and the points
themselves moving with the atoms that own them (`point_chain`, with the frame
derivative, since `isdf_grid` places the shells in covariant atomic frames that
turn as the environment does). Dropping either one leaves a force that still
looks plausible, and only one of them is caught for free: the points' own
translation breaks |grad.sum|, while the frames turning leaves it near round-off
because a frame rotation is translation-invariant.

WHAT THIS DOES NOT COVER
------------------------
The analytic HESSIAN of a correlated energy on such a mean field, which
`refuse_isdf_jk_gradient` still refuses (`src.properties.vibronic`): a second
derivative of the factors is not built. The first-derivative correlated force
IS covered: the folded Fock partial's exchange half comes from
`isdf_fock_partial_exchange`, built from the ISDF factors rather than the
auxiliary basis, and the exact-exchange double counting is built on the mean
field's own `ISDFJK` (`exx_double_counting_skeleton`, `reference_energy`)
rather than a fresh fit, so a GW, BSE or dRPA force reads the same exchange
the energy did. The dRPA ground-state force is gated against a
Richardson-extrapolated finite difference of its own reported energy to
1.5e-8 Ha/Bohr on PBE0 and 1.7e-8 on LRC-wPBEh.

`require_isdf_gradient_support` refuses, by name: `j_route='isdf'` (the Coulomb
term would then be interpolated too, and the reference gradient's DF-J
derivative would be the wrong function), an unrestricted reference, a mean
field whose exchange is not interpolated at all, and a factorization whose
interpolation points were injected from outside, whose decomposition into
atom-local clouds is then unknown. A range-separated hybrid is covered: its
exchange is one channel per operator sharing the bare fit, and each attenuated
metric carries its own two-centre derivative.

WHERE THE PIECES LIVE
---------------------
Only the assembly is here. `exchange_free_reference` is a forward mean field
and lives in production (`src.Base.isdf_jk`); `isdf_exchange_skeleton` and its
adjoints live with every other adjoint of the factorization
(`src.gradients.isdf_derivatives`). Both are re-exported from here, so this
module still names the whole force.
"""
import types

import numpy as np
from pyscf import scf as pyscf_scf

from src.Base.dispersion import dispersion_gradient
from src.Base.isdf_jk import ISDFJK, exchange_free_reference  # noqa: F401
from src.SingleReference.LinearResponse.rpa_energy import xc_hybrid_coeff
from src.gradients.isdf_derivatives import (  # noqa: F401
    isdf_exchange_adjoints, isdf_exchange_skeleton, isdf_fock_partial_exchange)


def require_isdf_gradient_support(mf, what):
    """Raise unless this ISDF mean field is one whose force is built here."""
    with_df = getattr(mf, 'with_df', None)
    if not isinstance(with_df, ISDFJK):
        raise TypeError(
            f'{what} needs a mean field whose exchange comes from ISDF factors '
            f'(with_df an ISDFJK, got {type(with_df).__name__}); pyscf\'s own '
            f'gradient is correct for a fitted one.')
    if not isinstance(mf, pyscf_scf.hf.RHF) or isinstance(mf,
                                                          pyscf_scf.rohf.ROHF):
        raise NotImplementedError(
            f'{what} covers the restricted closed-shell case only, and this is '
            f'a {type(mf).__name__}: an open-shell exchange energy is a sum '
            f'over spins of -(a_x/2) Tr[D_s K(D_s)], a different scalar with a '
            f'different adjoint.')
    if with_df.j_route != 'df-direct':
        raise NotImplementedError(
            f'{what} takes the Coulomb derivative from pyscf\'s density-fitted '
            f'gradient, which is the right function only for '
            f"j_route='df-direct'; this mean field builds J from the "
            f'interpolation (j_route={with_df.j_route!r}), whose derivative is '
            'not built. That route is measurably unsafe in an SCF anyway -- '
            'see ISDFJK.build.')
    if with_df.coords is not None and with_df.grid_radii is None:
        raise NotImplementedError(
            f'{what} needs the interpolation points as atom-local clouds, and '
            f'this factorization was handed its `coords` from outside, so the '
            f'radii that placed them -- and which atom owns which point -- are '
            f'unknown. Let ISDFJK.build pick the grid.')


def attach_isdf_gradient(mf):
    """Give an ISDF mean field a `nuc_grad_method` that differentiates ITS energy.

    WITHOUT THIS, ANY OPTIMIZER WALKS DOWNHILL ON THE WRONG SURFACE. pyscf's
    own gradient differentiates the FITTED interaction and knows nothing of the
    interpolation, so it returns the force of a different function than the
    energy the SCF just reported. Both geomeTRIC and pyberny take the mean
    field and ask IT for a gradient, so no dispatch at the call site can reach
    them; the object itself has to answer correctly.

    Subclassing the gradient pyscf would have built keeps `as_scanner` and
    everything the geometry optimizers need, and replaces only the number.
    """
    base_cls = mf.nuc_grad_method().__class__

    class _ISDFGradients(base_cls):
        """pyscf's gradient with the ISDF force in place of the fitted one."""

        def kernel(self, *args, **kwargs):
            self.de = isdf_mean_field_gradient(self.base)
            return self.de

    mf.nuc_grad_method = types.MethodType(lambda m: _ISDFGradients(m), mf)
    mf.Gradients = mf.nuc_grad_method
    return mf


def isdf_mean_field_gradient(mf):
    """(natm, 3) force of a converged SCF whose exchange is from ISDF factors.

    `exchange_free_reference`'s gradient carries every term the interpolation
    does not touch, including the Pulay term, `isdf_exchange_skeleton` supplies
    the one it does, and the empirical dispersion correction -- which the
    reference's name cannot keep -- is added from the mean field's own name.
    `grid_response=True` because the quadrature moves with the atoms and the
    finite difference this is gated against sees that.

    A pure functional never asks ISDFJK for K -- with `j_route='df-direct'` the
    interpolation then enters the energy nowhere at all -- so the reference IS
    the whole force, and no factorization is built to find that out.
    """
    require_isdf_gradient_support(mf, 'the ISDF mean-field gradient')
    g0 = exchange_free_reference(mf).Gradients()
    g0.grid_response = True
    grad = np.asarray(g0.kernel()) + dispersion_gradient(mf)
    if xc_hybrid_coeff(mf)[1] == 0.0:
        return grad
    if not mf.with_df._built:
        mf.with_df.build()
    return grad + isdf_exchange_skeleton(mf)
