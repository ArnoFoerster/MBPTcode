"""The nuclear force of a mean field whose exchange comes from ISDF factors.

`src.Base.isdf_jk.ISDFJK` gives an SCF integral-direct DF Coulomb and an
exchange built from the interpolative separable density fit,

    K = X^T [Z .* (X D X^T)] X,   X[P,mu] = chi_mu(r_P),   Z = M^T V M.

pyscf's `mf.Gradients()` differentiates the fitted interaction and knows
nothing of the interpolation points or of M, so its force belongs to a
different function (on water/cc-pVDZ/B3LYP it misses 4.0e-4 Ha/Bohr, constant
as the finite-difference step shrinks). Only the exchange term is replaced:

    dE/dR = dE_nuc + Tr[D dh] - Tr[W dS] + dE_J|_D + dE_xc|_D + dE_K^ISDF|_D
          + dE_disp,        W = C n eps C^T,

every two-electron and grid term at fixed density, each in fixed tiles over
the ranks (`src.Base.skeleton_tiles`): the one-electron and overlap terms by
atoms, the Coulomb term (the Coulomb half of the fitted Fock skeleton at
g = D) by auxiliary tiles, the xc energy on the moving Becke grid by grid
tiles, and the exchange on the row fit's tiles. Tile addends are summed in
tile order, so the force is the same bits at every rank count. The non-exchange
terms agree with pyscf's gradient of the same functional with exact exchange
zeroed (`exchange_free_reference`) to 1e-13 on the unpruned grid pyscf's
full response differentiates; the SCF's own grid is pruned by density, and
this force is the derivative of that grid's energy.

No orbital-response term appears: the ISDF exchange matrix is the functional
derivative of the ISDF exchange energy, dE_K/dD = -(a_x/2) K, so the SCF is
variational for the reported energy. That energy is a scalar in one symmetric
(M, M) object,

    E_K = -(a_x/4) Tr[D K] = -(a_x/4) sum_PQ Z_PQ W_PQ^2,   W = X D X^T

(closed shell), so its reverse pass is two adjoints, on the collocation and on
Z, followed by the fit's chain shared with the correlated route. X depends on
geometry twice: through the basis functions at fixed points
(`basis_centre_forces`) and through the points moving and turning with their
atoms (`point_chain`, including the derivative of `isdf_grid`'s covariant
atomic frames). Dropping the point translation breaks |grad.sum|; dropping the
frame rotation does not, since a frame rotation is translation-invariant.

Not covered: the analytic Hessian of a correlated energy on such a mean field
(refused by `refuse_isdf_jk_gradient` in `src.properties.vibronic`). Correlated
first-derivative forces are covered: the folded Fock partial's exchange half
is `isdf_fock_partial_exchange`, and the exact-exchange double counting is
built on the mean field's own `ISDFJK`, so a GW, BSE or dRPA force reads the
same exchange the energy did. `require_isdf_gradient_support` refuses
`j_route='isdf'`, an unrestricted reference, a mean field without interpolated
exchange, and injected interpolation points (no atom-local decomposition).
Range-separated hybrids are covered: one exchange channel per operator, each
attenuated metric with its own two-centre derivative.

This module only assembles the force; it re-exports the exchange pieces from
`src.gradients.isdf_derivatives` and `exchange_free_reference`.
"""
import types

import numpy as np
from pyscf import scf as pyscf_scf
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import rks as rks_grad

from src.Base.dispersion import dispersion_gradient
from src.Base.isdf_jk import (  # noqa: F401
    ISDFJK, _base_functional, exchange_free_reference,
    mean_field_skeleton_force)
from src.Base.skeleton_tiles import (fitted_coulomb_energy_skeleton,
                                     one_electron_energy_skeleton,
                                     xc_energy_grid_skeleton)
from src.gradients.isdf_derivatives import (  # noqa: F401
    _auxmol_of, exchange_fit, isdf_exchange_adjoints, isdf_exchange_skeleton,
    isdf_fock_partial_exchange, isdf_scf_handle, mean_field_exchange_wanted)


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
            f'{what} takes the Coulomb derivative from the density-fitted '
            f'skeleton, which is the right function only for '
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
    """Give an ISDF mean field a `nuc_grad_method` that differentiates its energy.

    Geometry optimizers (geomeTRIC, pyberny) ask the mean field itself for a
    gradient, so the object must return the ISDF force rather than pyscf's
    fitted one. The class subclasses the gradient pyscf would have built,
    keeping `as_scanner`, and replaces only `kernel`.

    A continuum wrapped around the mean field later needs this called again on
    the wrapped object: `solvent.PCM` copies the instance dictionary, so it
    would inherit a gradient bound to the unwrapped, unconverged one. Stale
    bindings are dropped here, `kernel` refuses an unconverged base, and
    `mean_field_skeleton_force` adds the reaction field's fixed-density term.
    """
    for stale in ('nuc_grad_method', 'Gradients'):
        mf.__dict__.pop(stale, None)
    # A PCM mean field's gradient class is pyscf's solvent mixin, which wraps
    # another gradient object; subclass the plain SCF's gradient and keep the
    # wrapped mean field as the base (`mean_field_skeleton_force` adds the
    # reaction field).
    plain = mf.undo_solvent() if getattr(mf, 'with_solvent', None) is not None \
        else mf
    for stale in ('nuc_grad_method', 'Gradients'):
        plain.__dict__.pop(stale, None)
    base_cls = plain.nuc_grad_method().__class__

    class _ISDFGradients(base_cls):
        """pyscf's gradient with the ISDF force in place of the fitted one."""

        def kernel(self, *args, **kwargs):
            if getattr(self.base, 'mo_occ', None) is None:
                raise RuntimeError(
                    'this ISDF gradient belongs to a mean field that was never '
                    'converged -- typically the object a continuum was wrapped '
                    'around afterwards. Call attach_isdf_gradient on the '
                    'wrapped mean field.')
            self.de = mean_field_skeleton_force(self.base)
            return self.de

    mf.nuc_grad_method = types.MethodType(lambda m: _ISDFGradients(m), mf)
    mf.Gradients = mf.nuc_grad_method
    return mf


def isdf_mean_field_gradient(mf, fit=None):
    """(natm, 3) force of a converged SCF whose exchange is from ISDF factors.

    The module docstring's sum, each term at fixed density in fixed tiles over
    the ranks, plus the empirical dispersion correction named by the mean
    field. The xc quadrature moves with the atoms; a VV10 kernel's term is
    pyscf's full response, whole.

    A pure functional with `j_route='df-direct'` has no interpolation in its
    energy, so no factorization is built. With the row fit (`fit`,
    `exchange_fit`'s choice when None) the whole factorization is not built
    either; its skeleton reads the tiles alone.
    """
    require_isdf_gradient_support(mf, 'the ISDF mean-field gradient')
    mol = mf.mol
    dm = np.asarray(mf.make_rdm1(mf.mo_coeff, mf.mo_occ))
    dme = rhf_grad.make_rdm1e(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    handle = isdf_scf_handle(mf)
    auxmol = handle.auxmol if handle is not None else _auxmol_of(mf)
    grad = rhf_grad.grad_nuc(mol)
    # the generator's with_rinv_at_nucleus writes its molecule: a private one
    grad = grad + one_electron_energy_skeleton(
        mol, rhf_grad.Gradients(mf).hcore_generator(mol.copy()), dm, dme)
    grad = grad + fitted_coulomb_energy_skeleton(mol, auxmol, dm)
    if hasattr(mf, 'xc'):
        grad = grad + xc_energy_grid_skeleton(
            mol, mf.grids, mf._numint, _base_functional(mf.xc), dm)
        if mf.do_nlc():
            grad = grad + _nlc_gradient(mf, dm)
    grad = grad + dispersion_gradient(mf)
    if not mean_field_exchange_wanted(mf):
        return grad
    fit = exchange_fit(mf, fit)
    if fit == 'replicated' and not mf.with_df._built:
        mf.with_df.build()
    return grad + isdf_exchange_skeleton(mf, fit=fit)


def _nlc_gradient(mf, dm):
    """(natm, 3) of d/dR E_VV10[dm] on the moving grid: pyscf's full
    response, assembled as its `grad_elec` does, whole on every rank."""
    mol, ni = mf.mol, mf._numint
    xc = mf.xc if ni.libxc.is_nlc(mf.xc) else mf.nlc
    if mf.nlcgrids.coords is None:
        mf.nlcgrids.build(with_non0tab=True)
    enlc, vnlc = rks_grad.get_nlc_vxc_full_response(ni, mol, mf.nlcgrids, xc,
                                                    dm)
    out = np.asarray(enlc, float).copy()
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        out[ia] += 2.0 * np.einsum('xij,ij->x', vnlc[:, p0:p1], dm[p0:p1])
    return out
