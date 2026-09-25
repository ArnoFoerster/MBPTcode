"""A force must not be taken from the ISDF exchange route by PYSCF's gradient.

pyscf's density-fitted gradient differentiates a FITTED interaction. It knows
nothing of the interpolation points or the fit matrix this route builds K from,
so what it returns on such a mean field is the force of one functional
evaluated on another: 4.0e-4 Ha/Bohr away from a difference of this route's own
energy, and CONSTANT in the step while the difference itself converges.

That force is now available -- `isdf_mean_field_gradient` replaces the fitted
exchange derivative by the ISDF one and lands on the same energy -- so this
file gates the contrast rather than the defect: the guard still fires, because
it protects callers that take pyscf's gradient, and those are the correlated
chains whose remaining skeletons are fitted too. The gate on the route that
works is tests/test_isdf_mean_field_gradient.py.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto

from src.Base.constants import BOHR_TO_ANGSTROM
from src.Base.isdf_jk import ISDFJK, isdf_jk, refuse_isdf_jk_gradient
from src.gradients.isdf_mean_field import isdf_mean_field_gradient

ATOM = 'O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24'
#: Angstrom. The disagreement is CONSTANT in h, so any step shows it; this one
#: is small enough that the difference's own truncation error, which falls as
#: h^2, is three orders below it and the two forces can be told apart.
STEP_ANGSTROM = 2.5e-4


def mean_field(route, atom=ATOM):
    mf = dft.RKS(gto.M(atom=atom, basis='cc-pvdz', verbose=0), xc='b3lyp')
    mf = isdf_jk(mf) if route == 'isdf' else mf.density_fit(auxbasis='cc-pvdz-ri')
    mf.grids.prune = None
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-11
    mf.kernel()
    return mf


def finite_difference(route):
    """dE/dz of H(1) from a central difference of `route`'s own SCF energy."""
    moved = []
    for sign in (+1, -1):
        atom = [['O', [0.0, 0.0, 0.0]], ['H', [0.0, 0.0, 0.96]],
                ['H', [0.93, 0.0, -0.24]]]
        atom[1][1][2] += sign * STEP_ANGSTROM
        moved.append(mean_field(route, atom).e_tot)
    return (moved[0] - moved[1]) / (2.0 * STEP_ANGSTROM / BOHR_TO_ANGSTROM)


def test_the_guard_fires_only_on_the_isdf_route():
    assert isinstance(mean_field('isdf').with_df, ISDFJK)
    with pytest.raises(NotImplementedError, match='ISDF factors'):
        refuse_isdf_jk_gradient(mean_field('isdf'), 'a caller')
    refuse_isdf_jk_gradient(mean_field('df'), 'a caller')       # fitted: fine
    from pyscf import scf
    refuse_isdf_jk_gradient(scf.RHF(gto.M(atom=ATOM, basis='sto-3g', verbose=0)),
                            'a caller')                          # no with_df


def test_the_surfaces_that_move_nuclei_now_dispatch_instead():
    """THE REFUSAL HERE IS RETIRED, and the assertion is inverted rather than
    deleted so that reinstating one is a decision and not a drift. Both entries
    took pyscf's gradient and had to refuse while their Lagrangians built the
    folded Fock partial and the exact-exchange double counting from the FITTED
    skeletons; both now come from the interpolation, so the correct force
    exists and the choice is a dispatch.

    `factor_chain.FactorChain.mean_field_gradient` is where the dispatch
    itself lives, on `with_df`'s type: `mean_field_skeleton_force` for an
    ISDF one, pyscf's own fitted gradient otherwise. `excited_state`'s chain
    does not override that method, so its own source calls
    `mean_field_gradient(` and never spells `mean_field_skeleton_force(` --
    the dispatch is reached through inheritance, not repeated.

    A source scan says only that the call is present, which is why
    tests/test_isdf_fock_partial.py carries the finite difference that says the
    dispatched force is right.
    """
    import inspect

    from src.gradients import excited_state, factor_chain
    for module in (excited_state, factor_chain):
        assert 'refuse_isdf_jk_gradient(' not in inspect.getsource(module), \
            module.__name__
    assert 'mean_field_skeleton_force(' in inspect.getsource(factor_chain)
    assert 'mean_field_gradient(' in inspect.getsource(excited_state)


def test_the_guard_sends_the_caller_to_the_force_that_exists():
    """A refusal that does not name the route that works is how a built
    gradient stays unused."""
    with pytest.raises(NotImplementedError, match='isdf_mean_field_gradient'):
        refuse_isdf_jk_gradient(mean_field('isdf'), 'a caller')


def test_pyscfs_gradient_misses_what_the_isdf_one_hits():
    """The measurement the guard exists for, and its resolution, at one step.

    pyscf's force on this mean field is 4e-4 from a difference of the mean
    field's OWN energy, where the ISDF force sits at the finite difference's
    truncation error -- the same place the fitted route's force sits.

    REACHED THROUGH THE PARENT CLASS ON PURPOSE. `isdf_jk` now attaches
    `attach_isdf_gradient`, so `mf.Gradients()` answers correctly and the
    hazard is no longer reachable the way a caller would hit it -- which is
    the point of attaching it. The hazard is still REAL for anyone who builds
    pyscf's gradient directly, so it is measured here by doing exactly that.
    """
    fd = {route: finite_difference(route) for route in ('isdf', 'df')}

    fitted = mean_field('df').Gradients()
    fitted.grid_response = True
    assert abs(fitted.kernel()[1, 2] - fd['df']) < 1e-5

    mf = mean_field('isdf')
    # the gradient pyscf would have built, one class up from the attached one
    pyscf_force = type(mf.Gradients()).__mro__[1](mf)
    pyscf_force.grid_response = True
    wrong = abs(pyscf_force.kernel()[1, 2] - fd['isdf'])
    right = abs(isdf_mean_field_gradient(mf)[1, 2] - fd['isdf'])
    # pyscf's force is one functional evaluated on another, so its error IS
    # the interpolation error and shrinks as the grid improves: 3.9e-4 on the
    # grids before the re-gate, 8.4e-5 after it. What has to hold is that it
    # stays resolvable above the difference, not that the grid stays bad.
    assert right < 1e-6, right
    assert wrong > 100.0 * right, (right, wrong)
    # and what a caller actually gets is the right one
    attached = mf.Gradients()
    attached.grid_response = True
    assert abs(attached.kernel()[1, 2] - fd['isdf']) < 1e-6


def test_normal_modes_refuses_to_build_a_fitted_hessian():
    """THE HESSIAN HAD NO GUARD AT ALL, and it is the same defect one
    derivative further: pyscf differentiates the fitted interaction twice and
    returns force constants for a functional the SCF never minimized. Nothing
    raised, so a rate run produced modes and Huang-Rhys factors that looked
    ordinary -- and S_k goes as 1/omega_k, so the softest modes, which the
    fitted exchange derivative misplaces most, are the ones carrying the
    weight.

    Passing a Hessian in is still allowed: a caller that built one correctly
    is not what this protects against.
    """
    from src.properties.vibronic import normal_modes

    isdf = mean_field('isdf')
    with pytest.raises(NotImplementedError, match='ISDF factors'):
        normal_modes(isdf)
    # A Hessian the caller supplies passes straight through. It must be the
    # (natm, natm, 3, 3) one pyscf returns: `normal_modes` hands BACK the
    # (3N, 3N) reshape under the same name, so feeding its own fourth return
    # value in raises "axes don't match array" from the transpose.
    df = mean_field('df')
    raw = df.Hessian().kernel()
    omega = normal_modes(df, hess=raw)[0]
    again = normal_modes(isdf, hess=raw)[0]
    assert np.allclose(again, omega, atol=0, rtol=0)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
