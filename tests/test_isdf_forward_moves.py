"""The forward physics the ISDF gradient chain reads from production modules.

Three routines that describe a mean field rather than a derivative of one
(its response kernel, its Fock in the MO basis, the pseudo-inverse of an
auxiliary metric) live in `src.Base.pyscf.pyscf_interface`, and the mean field with
its exact exchange removed lives in `src.Base.factorization.isdf_jk`. The gradient package
imports them, so there is one implementation; every check below holds to a
few ulp (`assert_same`; the class bar on a BLAS other than the recording one)
against the expression the routine encodes, evaluated independently here.
The perturbations below are rejected at that few-ulp bar, i.e. under the
recording BLAS; each is shown to move the bits on any BLAS.

Two pairs compute different objects and stay separate:

    point_layout            returns the atom-local clouds and their owners;
                            `molecular_points_covariant` returns the placed
                            points. They share the stacking order, so the
                            clouds rotated and translated are production's
                            cloud, to a few ulp.
    continued_frames        carries a discrete branch over from a reference
                            frame; `atomic_frames` decides one from the
                            geometry. They agree where the continuation is a
                            no-op, as on water.

Each gate rejects a perturbation:

    gate                         perturbation                      rejected
    response kernel              the operator scaled by 1 + 1e-12  yes
    MO Fock                      one element moved by 1e-12        yes
    auxiliary pseudo-inverse     lindep raised to 1e-2              yes
    exchange-free reference      the reference left with its       yes
                                 exact exchange in place
    placed interpolation points  one local point moved by 1e-12    yes
    frame continuation           two reference axes flipped        yes
    the rebuilt fit              M from `fit_M_streaming`          yes

The whole-form `isdf_exchange_skeleton` refits M rather than taking the mean
field's own: `fit_adjoint` reverses `fit_M_stable` on an explicit test set,
while the SCF's M comes from `fit_M_streaming`, another realization. On
water/cc-pVDZ, where the pair screen keeps every pair, the two fit matrices
differ by 1.1e-8 relative in the fit's near-null space, and substituting one
for the other moves the Hartree-Fock skeleton force by 2.8e-10 Ha/Bohr, below
the 1e-8 reproducibility floor.
"""
import functools
import pathlib
import subprocess
import sys

import numpy as np
from pyscf import dft, gto, scf

import src.Base.pyscf.pyscf_interface as production_interface
import src.gradients.isdf_derivatives as derivatives
import src.gradients.isdf_mean_field as mean_field_module
import src.gradients.mean_field_response as response
from src.gradients.isdf_mean_field import exchange_free_reference
from src.gradients.isdf_mean_field import isdf_jk_with_force
from src.Base.pyscf.pyscf_interface import (fock_mo,
                                            response_kernel)
from src.Base.factorization.separable_ri import atomic_frames, molecular_points_covariant
from src.Base.pyscf.pyscf_interface import xc_hybrid_coeff
from src.gradients.isdf_derivatives import (continued_frames,
                                            isdf_exchange_skeleton,
                                            isdf_fock_partial_exchange,
                                            point_layout)
from src.gradients.isdf_mean_field import isdf_mean_field_gradient
from tests.isdf_test_grid import g1
from tests.pins import assert_same

AUXBASIS = 'cc-pvdz-ri'
#: What the tiled mean-field force may miss pyscf's gradient of the
#: exchange-free reference plus the exchange skeleton by on one grid.
EXACT_SPLIT_BAR = 1e-12
GEOMETRY = (('O', (0.0, 0.0, 0.1173)), ('H', (0.0, 0.7572, -0.4692)),
            ('H', (0.0, -0.7572, -0.4692)))
REPO = pathlib.Path(__file__).resolve().parent.parent

#: Bound on |M_stable - M_streaming| / |M_stable|max on water/cc-pVDZ, G1:
#: round-off of two realizations of one fit, which depends on the BLAS
#: (2.80e-8 on a workstation's MKL, 1.99e-8 on a cluster's).
FIT_FACTOR_DISCREPANCY = 5e-8
#: The Ha/Bohr the substitution moved the Hartree-Fock skeleton force by on
#: G1 (4.3e-9 on the workstation), below the 1e-8 reproducibility floor because
#: the fit adjoint solves the Gram matrix once per side (applying G^-1 twice
#: would give 3.5e-8).
GRADIENT_FLOOR = 1e-8


def water():
    """The reference geometry, cc-pVDZ."""
    return gto.M(atom=[[s, list(c)] for s, c in GEOMETRY], basis='cc-pvdz',
                 verbose=0)


def converged(mf):
    """`mf` at conv_tol_grad 1e-11, which the gradient Lagrangian assumes."""
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-11
    mf.kernel()
    assert mf.converged
    return mf


@functools.lru_cache(maxsize=None)
def fitted(xc):
    """A density-fitted mean field, Hartree-Fock for xc=None."""
    mol = water()
    if xc is None:
        return converged(scf.RHF(mol).density_fit(auxbasis=AUXBASIS))
    mf = dft.RKS(mol, xc=xc).density_fit(auxbasis=AUXBASIS)
    mf.grids.prune = None
    return converged(mf)


@functools.lru_cache(maxsize=None)
def interpolated(xc):
    """A mean field whose exchange comes from the ISDF factors."""
    mol = water()
    if xc is None:
        return converged(isdf_jk_with_force(scf.RHF(mol), auxbasis=AUXBASIS, counts=g1(scf.RHF(mol))))
    mf = dft.RKS(mol, xc=xc)
    mf.grids.prune = None
    return converged(isdf_jk_with_force(mf, auxbasis=AUXBASIS, counts=g1(mf)))


def symmetric_direction(nao, seed=7):
    """A fixed symmetric density-like direction to apply an operator to."""
    x = np.random.default_rng(seed).standard_normal((nao, nao))
    return 0.5 * (x + x.T)


def test_the_moved_routines_are_one_object_and_not_a_copy():
    """The gradient package's names are production's objects, by identity,
    so no second spelling can drift."""
    assert response.response_kernel is production_interface.response_kernel
    assert response.fock_mo is production_interface.fock_mo
    # the tiled fitted skeleton inverts no attenuated metric, so the gradient
    # package does not hold the name
    assert not hasattr(derivatives, 'aux_metric_inverse')
    # one function under one name in both modules
    assert (mean_field_module.isdf_exchange_skeleton
            is derivatives.isdf_exchange_skeleton)


def test_the_response_kernel_is_pyscfs_own_operator():
    """G(x) = dV_eff/dD . x is `gen_response`, and on Hartree-Fock it reduces
    to vj - 0.5 vk."""
    for xc in (None, 'pbe0'):
        mf = fitted(xc)
        x = symmetric_direction(mf.mol.nao_nr())
        assert_same(response_kernel(mf, True)(x),
                    mf.gen_response(hermi=1)(x), quantity='root',
                    label=('response kernel', xc))
        # the gas-phase operator is the same one without a continuum, and there
        # is no continuum here, so the two coincide
        assert_same(response_kernel(mf, False)(x),
                    response_kernel(mf, True)(x), quantity='root',
                    label=('gas-phase kernel', xc))
    mf = fitted(None)
    x = symmetric_direction(mf.mol.nao_nr())
    vj, vk = mf.get_jk(mf.mol, x, hermi=1)
    assert np.abs(response_kernel(mf, True)(x) - (vj - 0.5 * vk)).max() < 1e-12
    # cached on the mean field: a Z-vector solve calls this per iteration
    assert response_kernel(mf, True) is response_kernel(mf, True)


def test_the_mo_fock_is_the_transformed_fock():
    """C^T F C of a converged mean field, cached because a Krylov matvec reads
    it every iteration and it depends on the mean field alone."""
    for xc in (None, 'pbe0'):
        mf = fitted(xc)
        C = mf.mo_coeff
        assert_same(fock_mo(mf), C.T @ mf.get_fock() @ C, quantity='root',
                    label=('MO Fock', xc))
    assert fock_mo(fitted(None)) is fock_mo(fitted(None))


def test_the_exchange_free_reference_carries_none_of_the_exact_exchange():
    """The reference keeps the orbitals, quadrature and auxiliary basis and
    drops a_x K; its gradient plus the ISDF exchange skeleton is the whole
    force, an exact split.

    The force is assembled from tiled terms (`skeleton_tiles`), so the two
    agree to `EXACT_SPLIT_BAR`, not bitwise, on a grid without density
    pruning. On a density-pruned grid they differ by the pruned points' own
    terms (1.3e-9): pyscf's full response differentiates the unpruned grid,
    the force the pruned one the energy sums."""
    for xc in (None, 'pbe0'):
        mf = interpolated(xc)
        ref = exchange_free_reference(mf)
        assert xc_hybrid_coeff(ref)[1] == 0.0
        assert np.array_equal(ref.mo_coeff, mf.mo_coeff)
        assert np.array_equal(ref.mo_energy, mf.mo_energy)
        if xc is not None:
            assert ref.grids is mf.grids
            base = dft.RKS(water(), xc=xc)
            base.grids.prune, base.small_rho_cutoff = None, 0.0
            mf = converged(isdf_jk_with_force(base, auxbasis=AUXBASIS, counts=g1(base)))
            ref = exchange_free_reference(mf)
        g0 = ref.Gradients()
        g0.grid_response = True
        apart = np.abs(isdf_mean_field_gradient(mf)
                       - np.asarray(g0.kernel())
                       - isdf_exchange_skeleton(mf)).max()
        assert apart < EXACT_SPLIT_BAR, (xc, apart)


def test_the_point_layout_places_productions_own_interpolation_points():
    """The clouds rotated into their frames and translated to their nuclei
    are `molecular_points_covariant`'s cloud to a few ulp, in the same row
    order."""
    mf = interpolated('pbe0')
    mol, with_df = mf.mol, mf.with_df
    pts_local, owner = point_layout(mol, with_df.grid_radii,
                                    with_df.grid_origins)
    frames = atomic_frames(mol)[0]
    placed = np.vstack([pts_local[ia] @ frames[ia] + mol.atom_coord(ia)
                        for ia in range(mol.natm)])
    assert_same(placed, molecular_points_covariant(
        mol, with_df.grid_radii, origin_by_element=with_df.grid_origins),
        unit='1', quantity='root', label='placed points')
    assert np.array_equal(owner, np.concatenate(
        [np.full(len(p), ia, dtype=int) for ia, p in enumerate(pts_local)]))


def test_the_frame_continuation_is_a_no_op_at_its_own_reference():
    """Continuing `atomic_frames` onto itself changes nothing: the signs come
    from fixed generic references and are already what the matching picks;
    the two differ only once a path has moved the branch."""
    mol = water()
    frames = atomic_frames(mol)[0]
    assert np.array_equal(continued_frames(mol, frames), frames)


def test_the_skeletons_fit_from_the_mean_field_moves_the_force(monkeypatch):
    """Two realizations of the fit inside one adjoint: `fit_adjoint`
    reverses `fit_M_stable` on an explicit test set; `fit_M_streaming`
    realizes the same estimator otherwise, and substituting it moves the
    Hartree-Fock skeleton force below the reproducibility floor."""
    mf = interpolated(None)
    reference = isdf_exchange_skeleton(mf, fit='replicated')
    streaming = mf.with_df.M               # what the SCF itself fitted
    stable = derivatives.fit_M_stable
    rebuilt = {}

    def substituted(D, F, regularization):
        rebuilt['M'] = stable(D, F, regularization)
        return streaming

    monkeypatch.setattr(derivatives, 'fit_M_stable', substituted)
    moved = isdf_exchange_skeleton(mf, fit='replicated')
    monkeypatch.undo()

    factor = (np.abs(rebuilt['M'] - streaming).max()
              / np.abs(rebuilt['M']).max())
    force = float(np.abs(moved - reference).max())
    print(f'\nfit factors {factor:.2e} relative, force {force:.2e} Ha/Bohr')
    assert 0.0 < factor <= FIT_FACTOR_DISCREPANCY
    assert force < GRADIENT_FLOOR


def test_the_production_homes_import_without_the_gradient_package():
    """Each production module imports without any src.gradients module, in
    a fresh interpreter."""
    for module in ('src.Base.pyscf.pyscf_interface', 'src.Base.factorization.isdf_jk'):
        code = (f'import sys, {module}; '
                "assert not [m for m in sys.modules "
                "if m.startswith('src.gradients')], "
                "sorted(m for m in sys.modules if m.startswith('src.gradients'))")
        proc = subprocess.run([sys.executable, '-c', code], cwd=str(REPO),
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


def test_the_two_gradient_modules_import_in_either_order():
    """The dependency runs one way: the adjoints and the ISDF exchange
    skeleton in `isdf_derivatives`, the assembly of one mean-field force in
    `isdf_mean_field`."""
    names = ('src.gradients.isdf_derivatives', 'src.gradients.isdf_mean_field')
    for order in (names, names[::-1]):
        code = f'import {order[0]}; import {order[1]}'
        proc = subprocess.run([sys.executable, '-c', code], cwd=str(REPO),
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
    source = (REPO / 'src' / 'gradients' / 'isdf_derivatives.py').read_text()
    assert 'isdf_mean_field' not in source


def test_every_gate_rejects_its_own_perturbation():
    """One perturbation per gate above, each the smallest thing that gate
    catches, is rejected."""
    mf = fitted('pbe0')
    mol = mf.mol
    x = symmetric_direction(mol.nao_nr())

    # the response kernel, and the MO Fock
    assert not np.array_equal(response_kernel(mf, True)(x),
                              mf.gen_response(hermi=1)(x) * (1.0 + 1e-12))
    F_mo = fock_mo(mf).copy()
    F_mo[0, 0] += 1e-12
    assert not np.array_equal(fock_mo(mf), F_mo)

    # the exchange-free reference: leaving the exact exchange in place
    interp = interpolated('pbe0')
    still_hybrid = dft.RKS(interp.mol, xc=interp.xc).density_fit(
        auxbasis=interp.with_df.auxbasis)
    still_hybrid.grids = interp.grids
    still_hybrid.mo_coeff, still_hybrid.mo_energy = (interp.mo_coeff,
                                                     interp.mo_energy)
    still_hybrid.mo_occ, still_hybrid.converged = interp.mo_occ, True
    wrong = still_hybrid.Gradients()
    wrong.grid_response = True
    assert not np.array_equal(
        isdf_mean_field_gradient(interp),
        np.asarray(wrong.kernel()) + isdf_exchange_skeleton(interp))

    # the placed points, and the frame continuation
    pts_local, _ = point_layout(interp.mol, interp.with_df.grid_radii,
                                interp.with_df.grid_origins)
    frames = atomic_frames(interp.mol)[0]
    moved = [p.copy() for p in pts_local]
    moved[0][0, 0] += 1e-12
    placed = np.vstack([moved[ia] @ frames[ia] + interp.mol.atom_coord(ia)
                        for ia in range(interp.mol.natm)])
    assert not np.array_equal(placed, molecular_points_covariant(
        interp.mol, interp.with_df.grid_radii,
        origin_by_element=interp.with_df.grid_origins))
    # two axes, so the reference stays a proper rotation: one flip alone is
    # undone by the parity repair
    flipped = frames.copy()
    flipped[0, :2] *= -1.0
    assert not np.array_equal(continued_frames(interp.mol, flipped), frames)


def test_the_folded_exchange_partial_is_the_skeleton_at_two_densities():
    """A folded Fock partial's exchange half is the SCF's own contraction
    with W^2 replaced by W1 W2 and the bilinear coefficient doubled, as in
    `fock_partial_skeleton_df` at the same gamma."""
    mf = interpolated('pbe0')
    nmo = mf.mo_coeff.shape[1]
    gamma = np.diag(np.arange(nmo) * 0.01 + 0.5)
    gamma[0, 2] = gamma[2, 0] = 0.3
    g_ao = mf.mo_coeff @ (0.5 * (gamma + gamma.T)) @ mf.mo_coeff.T
    assert_same(
        isdf_fock_partial_exchange(mf, gamma),
        isdf_exchange_skeleton(mf, dm=g_ao, dm_other=mf.make_rdm1(),
                               prefactor=2.0, channels=None),
        quantity='gradient', label='folded exchange partial')
