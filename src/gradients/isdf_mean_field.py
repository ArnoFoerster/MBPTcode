"""The nuclear force of a mean field whose exchange comes from ISDF factors.

`src.Base.factorization.isdf_jk.ISDFJK` gives an SCF integral-direct DF Coulomb and an
exchange built from the interpolative separable density fit,

    K = X^T [Z .* (X D X^T)] X,   X[P,mu] = chi_mu(r_P),   Z = M^T V M.

pyscf's `mf.Gradients()` differentiates the fitted interaction and knows
nothing of the interpolation points or of M, so its force belongs to a
different function (on water/cc-pVDZ/B3LYP it misses 4.0e-4 Ha/Bohr, constant
as the finite-difference step shrinks). Only the exchange term is replaced:

    dE/dR = dE_nuc + Tr[D dh] - Tr[W dS] + dE_J|_D + dE_xc|_D + dE_K^ISDF|_D
          + dE_disp,        W = C n eps C^T,

every two-electron and grid term at fixed density, each in fixed tiles over
the ranks (`src.gradients.skeleton_tiles`): the one-electron and overlap terms by
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

This module assembles the force from the exchange pieces of
`src.gradients.isdf_derivatives`, and holds the exchange-free reference
(`exchange_free_reference`) and the refusal of pySCF's own force on an ISDF
mean field (`refuse_isdf_jk_gradient`).
"""
import types

import numpy as np
from pyscf import dft
from pyscf import scf as pyscf_scf
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import rks as rks_grad
from pyscf.scf.dispersion import parse_disp
from pyscf.solvent import pcm as pyscf_pcm

from src.Base.pyscf.dispersion import dispersion_gradient
from src.Base.pyscf.pyscf_interface import auxmol_of
from src.environments.pcm_derivatives import solvation_gradient
from src.Base.factorization.isdf_jk import ISDFJK, isdf_jk
from src.gradients.skeleton_tiles import (fitted_coulomb_energy_skeleton,
                                          one_electron_energy_skeleton,
                                          xc_energy_grid_skeleton)
from src.gradients.isdf_derivatives import (
    exchange_fit, isdf_exchange_skeleton, isdf_scf_handle,
    mean_field_exchange_wanted, require_fitted_coulomb)
from src.gradients.one_electron import one_electron_gradient


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
    require_fitted_coulomb(mf, what)
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
    # the hook a continuum wrapped around this mean field rebinds the force
    # with (`SolventScreening`), without importing this layer
    mf.force_hook = attach_isdf_gradient
    return mf


def isdf_jk_with_force(mf, **kwargs):
    """`isdf_jk(mf, **kwargs)` with the ISDF force attached: the mean field
    every route that takes a nuclear force from it builds."""
    return attach_isdf_gradient(isdf_jk(mf, **kwargs))


def mean_field_skeleton_force(mf):
    """dE/dR of the energy this mean field reported, whatever built its exchange.

    One call site for every correlated chain that adds a Lagrangian on top of
    the mean-field force. pyscf's gradient is right for a fitted reference and
    wrong for an interpolated one, and the correct interpolated force exists,
    so the choice is a dispatch and not a refusal.
    """
    if isinstance(getattr(mf, 'with_df', None), ISDFJK):
        return np.asarray(isdf_mean_field_gradient(mf))
    g0 = mf.Gradients()
    if hasattr(mf, 'xc'):
        g0.grid_response = True
    return np.asarray(g0.kernel())


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
    auxmol = handle.auxmol if handle is not None else auxmol_of(mf)
    one_electron = one_electron_gradient(mf)
    grad = one_electron.grad_nuc(mol)
    # the generator's with_rinv_at_nucleus writes its molecule: a private one
    grad = grad + one_electron_energy_skeleton(
        mol, one_electron.hcore_generator(mol.copy()), dm, dme)
    grad = grad + fitted_coulomb_energy_skeleton(mol, auxmol, dm)
    if hasattr(mf, 'xc'):
        grad = grad + xc_energy_grid_skeleton(
            mol, mf.grids, mf._numint, base_functional(mf.xc), dm)
        if mf.do_nlc():
            grad = grad + _nlc_gradient(mf, dm)
    grad = grad + dispersion_gradient(mf)
    if getattr(mf, 'with_solvent', None) is not None:
        grad = grad + reaction_field_gradient(mf.with_solvent, dm)
    if not mean_field_exchange_wanted(mf):
        return grad
    fit = exchange_fit(fit)
    if fit == 'replicated' and not mf.with_df.built:
        mf.with_df.build()
    return grad + isdf_exchange_skeleton(mf, fit=fit)


def reaction_field_gradient(solvent, dm):
    """(natm, 3) the explicit nuclear derivative of a PCM reaction-field
    energy at fixed density: the cavity's surface charges against the nuclei,
    against the electrons, and the response matrix moving with the cavity
    (pyscf's own PCM gradient terms). The orbital response is in the energy-
    weighted density already, the reaction field being in the converged Fock.

    Without it the force of a PCM-wrapped ISDF-K mean field is the gas-phase
    skeleton at the solvated density: on water/cc-pVDZ PBE0 in toluene,
    4.3e-3 Ha/Bohr from a central difference of the reported energy, and
    6e-8 with it.

    The potential-integral term runs on `cavity_blocks`, not on pyscf's
    `grad_qv`, whose blocks overrun libcint's int-sized index from ~90 atoms
    in cc-pVDZ at 120 GB a rank (a segfault in the first force).
    """
    if not isinstance(solvent, pyscf_pcm.PCM):
        raise NotImplementedError(
            f'the reaction-field force is built for pyscf PCM continua, and '
            f'this mean field carries a {type(solvent).__name__}')
    return solvation_gradient(solvent, dm)


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


#: Hartree-Fock minus its own exchange, for `exchange_free_reference`: the
#: empty functional, so a pure Hartree-Fock mean field keeps none of it.
_NO_FUNCTIONAL = '0*LDA'


class _NoExactExchange(dft.numint.NumInt):
    """A numint reporting no exact exchange, with the functional untouched.

    pyscf's Kohn-Sham `get_veff` and its gradient both read the exchange
    fractions from `rsh_and_hybrid_coeff` and never by re-parsing the string,
    so zeroing them here removes the exact exchange and leaves `eval_xc` as it
    was, which for a range-separated functional is the short-range DFT
    exchange libxc holds under the same name.

    Appending "- a_x*HF" to the functional instead could not express a
    range-separated hybrid and could not survive a dispersion suffix
    (`b3lyp-d3bj - 0.2*HF` is not a name pyscf can split).
    """

    def rsh_and_hybrid_coeff(self, xc_code, spin=0):
        return 0.0, 0.0, 0.0

    def hybrid_coeff(self, xc_code, spin=0):
        return 0.0

    def rsh_coeff(self, xc_code):
        return 0.0, 0.0, 0.0


def base_functional(xc):
    """`xc` with any empirical-dispersion suffix removed.

    Keyed on the dispersion version, not on `parse_disp`'s first return: that
    element is "xc_code_for_dftd3", the name to hand the DFTD3 library, and
    can be None for a plain functional; with no dispersion suffix `xc` itself
    is returned.

    The dispersion is a function of the geometry alone and the caller that
    takes this reference's gradient adds its force back separately.
    """
    base, disp, _ = parse_disp(xc)
    out = xc if disp is None else base
    if not out:
        raise ValueError(
            f'could not read a functional out of {xc!r}: parse_disp gave '
            f'{(base, disp)}. The exchange-free reference cannot be built '
            f'without one, and pyscf fails far from here if it is handed None.')
    return out


def exchange_free_reference(mf):
    """`mf` at its own orbitals with the exact-exchange fraction taken out.

    Everything the ISDF route did not interpolate, in one pyscf mean field: the
    functional minus its a_x*HF term, density-fitted on the same auxiliary basis
    (which is where `j_route='df-direct'` takes J from) and carrying the
    converged coefficients, orbital energies and occupations, so that its
    gradient's one-electron, Coulomb, exchange-correlation and energy-weighted
    overlap terms are the ISDF route's own.

    The same grid objects as the mean field's: a Kohn-Sham energy is a property
    of its quadrature, and a finite difference taken against a different grid
    would be differencing two functionals.

    An empirical dispersion correction is left out of the name (see
    `base_functional`); the caller adds its force back separately.
    """
    # Kohn-Sham iff the reference carries a functional, the same test
    # `mean_field_skeleton_force` makes below.
    is_ks = hasattr(mf, 'xc')
    # Hartree-Fock minus its exchange is the empty functional, which is what
    # `_NO_FUNCTIONAL` already is. A Kohn-Sham reference keeps its own
    # functional and has the exact-exchange fractions zeroed instead of
    # subtracted by name (`_NoExactExchange`).
    xc = base_functional(mf.xc) if is_ks else _NO_FUNCTIONAL
    ref = dft.RKS(mf.mol, xc=xc).density_fit(auxbasis=mf.with_df.auxbasis)
    if is_ks:
        ref._numint = _NoExactExchange()
        ref.grids, ref.nlcgrids = mf.grids, mf.nlcgrids
    ref.mo_coeff, ref.mo_energy = mf.mo_coeff, mf.mo_energy
    ref.mo_occ, ref.converged = mf.mo_occ, True
    return ref


def refuse_isdf_jk_gradient(mf, what):
    """Raise if `mf` answers its exchange from ISDF factors and a nuclear force
    is about to be taken from pyscf's own gradient.

    pyscf's density-fitted gradient differentiates the integrals of a fitted
    interaction and knows nothing of the interpolation points or the fit
    matrix this route builds K from, so it returns a gradient of a different
    function than the energy reported.

    The ground-state force itself is built elsewhere
    (`src.gradients.isdf_mean_field.isdf_mean_field_gradient`). What remains
    fitted, and why every caller of this guard is refused, is the Lagrangian a
    correlated chain adds on top: `fock_partial_skeleton` routes a
    density-fitted mean field to `fock_partial_skeleton_df`, whose exchange
    half is built from the auxiliary basis, and `exx_double_counting_skeleton`
    differences two fitted pyscf gradients. Energies are unaffected.
    """
    with_df = getattr(mf, 'with_df', None)
    if not isinstance(with_df, ISDFJK):
        return
    raise NotImplementedError(
        f'{what} takes a nuclear force from a mean field whose exchange comes '
        f'from ISDF factors, and pyscf\'s gradient does not know about them: '
        f'it differentiates the FITTED interaction rather than the '
        f'interpolated one this route actually built. Use '
        f'`src.gradients.isdf_mean_field.isdf_mean_field_gradient` for the SCF '
        f'force; a correlated (GW/BSE/dRPA) force on this reference is not '
        f'built at all.')
