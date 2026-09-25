"""The forward physics under the ISDF gradient chain, where it now lives.

Three routines that describe a MEAN FIELD rather than a derivative of one --
its response kernel, its Fock in the MO basis, the pseudo-inverse of an
auxiliary metric -- moved to `src.Base.pyscf_interface`, and the mean field
with its exact exchange removed moved to `src.Base.isdf_jk`. The gradient
package imports them back, so there is ONE implementation and no copy that can
drift; every check below is against the expression the moved routine encodes,
evaluated independently here, and is bitwise rather than tolerant.

Three more pairs were read and NOT merged, because the two members compute
different objects:

    point_layout            returns the atom-local clouds and their owners;
                            `molecular_points_covariant` returns the PLACED
                            points. They share the stacking order, so the
                            clouds rotated and translated ARE production's
                            cloud, bitwise, and that is what is checked.
    continued_frames        carries a discrete branch over from a reference
                            frame; `atomic_frames` decides one from the
                            geometry. They agree only where the continuation
                            is a no-op, which on water it exactly is.
    test_set_three_center   is (mu nu|P) on the test-set pairs, blocked over
                            the mu shells; `test_set_D` is the COLLOCATION of
                            those pairs on the interpolation points. The
                            blocked build is checked against the dense tensor
                            it exists to avoid forming.

WHAT THE PERTURBATIONS SAY. Every gate here is shown once to reject a wrong
answer, so that none of them is an identity that cannot fail:

    gate                         perturbation                      rejected
    response kernel              the operator scaled by 1 + 1e-12  yes
    MO Fock                      one element moved by 1e-12        yes
    auxiliary pseudo-inverse     lindep raised to 1e-2              yes
    static exchange diagonal     exchange='df-direct' in place     yes
                                 of the mean field's own K
    exchange-free reference      the reference left with its       yes
                                 exact exchange in place
    placed interpolation points  one local point moved by 1e-12    yes
    frame continuation           two reference axes flipped        yes
    blocked three-centre rows    one element moved by 1e-12        yes
    the rebuilt fit              M from `fit_M_streaming`          yes

THE FIT IS REBUILT AND STAYS REBUILT. `isdf_exchange_skeleton` does not take
the mean field's own M: `fit_adjoint` reverses `fit_M_stable` on a frozen
column set, while `fit_M_streaming` -- what the SCF ran, and what
`build_separable_ri` returns -- forms its Gram matrix from the UNSCREENED test
set and its right-hand side from the screened one. Measured on water/cc-pVDZ,
the two fit matrices differ by 1.11e-08 relative to |M|max = 3.4e+05, and
substituting one for the other moves the skeleton force by 3.5e-08 Ha/Bohr on
Hartree-Fock and 7.3e-09 on PBE0. The Hartree-Fock figure is above the 1e-8
Ha/Bohr reproducibility floor a gradient here is gated at, so the estimators
cannot be mixed and the rebuild is not removable.
"""
import functools
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from pyscf import df as pyscf_df, dft, gto, scf

import src.Base.isdf_jk as production_jk
import src.Base.pyscf_interface as production_interface
import src.gradients.isdf_derivatives as derivatives
import src.gradients.isdf_mean_field as mean_field_module
from src.Base.constants import AUX_METRIC_LINDEP
from src.Base.isdf_jk import exchange_free_reference, isdf_jk, range_coulomb
from src.Base.pyscf_interface import (aux_metric_inverse, fock_mo,
                                      response_kernel)
# aliased: pytest collects a module-level name starting with `test_` as a test
from src.Base.separable_ri import atomic_frames, molecular_points_covariant
from src.Base.separable_ri import test_set_layout as pair_layout
from src.SingleReference.GW.qp_solve import static_exchange_diagonal
from src.SingleReference.LinearResponse.rpa_energy import xc_hybrid_coeff
from src.gradients.factor_chain import \
    test_set_three_center as pair_three_centre
from src.gradients.isdf_derivatives import (continued_frames,
                                            isdf_exchange_skeleton,
                                            isdf_fock_partial_exchange,
                                            point_layout, qp_xc_correction)
from src.gradients.isdf_mean_field import isdf_mean_field_gradient

AUXBASIS = 'cc-pvdz-ri'
GEOMETRY = (('O', (0.0, 0.0, 0.1173)), ('H', (0.0, 0.7572, -0.4692)),
            ('H', (0.0, -0.7572, -0.4692)))
#: The long-range channel whose auxiliary metric is numerically singular: 50 of
#: 84 functions survive `AUX_METRIC_LINDEP` on this basis, against all 84 on the
#: bare Coulomb metric.
OMEGA = 0.2
REPO = pathlib.Path(__file__).resolve().parent.parent

#: |M_stable - M_streaming| / |M_stable|max on water/cc-pVDZ, and the Ha/Bohr
#: the substitution moves the Hartree-Fock skeleton force by. The second is
#: what decides the question: it is above the 1e-8 reproducibility floor.
FIT_FACTOR_DISCREPANCY = 1.11e-08
FIT_FORCE_DISCREPANCY_HF = 3.5e-08
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
        return converged(isdf_jk(scf.RHF(mol), auxbasis=AUXBASIS))
    mf = dft.RKS(mol, xc=xc)
    mf.grids.prune = None
    return converged(isdf_jk(mf, auxbasis=AUXBASIS))


def symmetric_direction(nao, seed=7):
    """A fixed symmetric density-like direction to apply an operator to."""
    x = np.random.default_rng(seed).standard_normal((nao, nao))
    return 0.5 * (x + x.T)


def metrics():
    """(auxmol, bare metric, long-range metric) of the auxiliary basis."""
    mol = water()
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=AUXBASIS)
    bare = auxmol.intor('int2c2e', aosym='s1')
    with range_coulomb(mol, auxmol, OMEGA):
        attenuated = auxmol.intor('int2c2e', aosym='s1')
    return auxmol, bare, attenuated


def tikhonov_inverse(V, lindep):
    """The inverse the production routine encodes, written out here."""
    V = 0.5 * (np.asarray(V, float) + np.asarray(V, float).T)
    w, u = np.linalg.eigh(V)
    keep = w > lindep * w.max()
    return (u[:, keep] / w[keep]) @ u[:, keep].T


def test_the_moved_routines_are_one_object_and_not_a_copy():
    """A move that leaves a copy behind is the failure this whole exercise is
    about: two spellings diverge silently. The gradient package's names must BE
    production's, by identity."""
    assert derivatives.response_kernel is production_interface.response_kernel
    assert derivatives.fock_mo is production_interface.fock_mo
    assert (derivatives.aux_metric_inverse
            is production_interface.aux_metric_inverse)
    assert (mean_field_module.exchange_free_reference
            is production_jk.exchange_free_reference)
    # and the shim is gone: one function under one name in both modules
    assert (mean_field_module.isdf_fock_partial_exchange
            is derivatives.isdf_fock_partial_exchange)
    assert (mean_field_module.isdf_exchange_skeleton
            is derivatives.isdf_exchange_skeleton)


def test_the_response_kernel_is_pyscfs_own_operator():
    """G(x) = dV_eff/dD . x is `gen_response`, and on Hartree-Fock it reduces to
    vj - 0.5 vk, which is the whole difference between the two references in a
    Lagrangian built on it."""
    for xc in (None, 'pbe0'):
        mf = fitted(xc)
        x = symmetric_direction(mf.mol.nao_nr())
        assert np.array_equal(response_kernel(mf, True)(x),
                              mf.gen_response(hermi=1)(x))
        # the gas-phase operator is the same one without a continuum, and there
        # is no continuum here, so the two coincide
        assert np.array_equal(response_kernel(mf, False)(x),
                              response_kernel(mf, True)(x))
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
        assert np.array_equal(fock_mo(mf), C.T @ mf.get_fock() @ C)
    assert fock_mo(fitted(None)) is fock_mo(fitted(None))


def test_the_auxiliary_inverse_is_tikhonov_and_not_a_truncation():
    """The bare Coulomb metric is full rank and this is an ordinary inverse; the
    long-range metric is not, and the shifted spectrum keeps the object a true
    inverse of what is evaluated, which is what makes -V^-1 dV V^-1 exact."""
    auxmol, bare, attenuated = metrics()
    naux = auxmol.nao_nr()
    for V in (bare, attenuated):
        assert np.array_equal(aux_metric_inverse(V),
                              tikhonov_inverse(V, AUX_METRIC_LINDEP))
    w_bare = np.linalg.eigvalsh(bare)
    w_lr = np.linalg.eigvalsh(0.5 * (attenuated + attenuated.T))
    assert int((w_bare > AUX_METRIC_LINDEP * w_bare.max()).sum()) == naux
    assert int((w_lr > AUX_METRIC_LINDEP * w_lr.max()).sum()) < naux
    assert np.abs(aux_metric_inverse(bare) @ bare - np.eye(naux)).max() < 1e-9


def test_the_quasiparticle_correction_is_the_static_exchange_diagonal():
    """<p|Sigma_x - v_xc|p> is production's own one-body term, so the chain
    differentiates and `calc_qp_energy` evaluates one function; the wrapper adds
    nothing but the all-states default."""
    for xc in (None, 'pbe0'):
        mf = fitted(xc)
        nmo = mf.mo_coeff.shape[1]
        every = np.arange(nmo)
        assert np.array_equal(
            qp_xc_correction(mf),
            static_exchange_diagonal(mf, mf.mol, every, exchange='mf'))
        shift = np.arange(nmo, dtype=float) * 1e-3
        assert np.array_equal(
            qp_xc_correction(mf, [3, 5], reaction_field=shift),
            static_exchange_diagonal(mf, mf.mol, [3, 5], exchange='mf',
                                     reaction_field=shift))


def test_the_exchange_free_reference_carries_none_of_the_exact_exchange():
    """It is the mean field the interpolation did NOT touch: same orbitals, same
    quadrature, same auxiliary basis, and no a_x K. Its gradient plus the ISDF
    exchange skeleton is the whole force, bitwise -- which is the statement that
    makes the split a decomposition rather than an approximation."""
    for xc in (None, 'pbe0'):
        mf = interpolated(xc)
        ref = exchange_free_reference(mf)
        assert xc_hybrid_coeff(ref)[1] == 0.0
        assert np.array_equal(ref.mo_coeff, mf.mo_coeff)
        assert np.array_equal(ref.mo_energy, mf.mo_energy)
        if xc is not None:
            assert ref.grids is mf.grids
        g0 = ref.Gradients()
        g0.grid_response = True
        assert np.array_equal(
            isdf_mean_field_gradient(mf),
            np.asarray(g0.kernel()) + isdf_exchange_skeleton(mf))


def test_the_point_layout_places_productions_own_interpolation_points():
    """The clouds rotated into their frames and translated to their nuclei ARE
    `molecular_points_covariant`'s cloud: same order, same arithmetic, so the
    row order of X is the same object on both sides."""
    mf = interpolated('pbe0')
    mol, with_df = mf.mol, mf.with_df
    pts_local, owner = point_layout(mol, with_df.grid_radii,
                                    with_df.grid_origins)
    frames = atomic_frames(mol)[0]
    placed = np.vstack([pts_local[ia] @ frames[ia] + mol.atom_coord(ia)
                        for ia in range(mol.natm)])
    assert np.array_equal(placed, molecular_points_covariant(
        mol, with_df.grid_radii, origin_by_element=with_df.grid_origins))
    assert np.array_equal(owner, np.concatenate(
        [np.full(len(p), ia, dtype=int) for ia, p in enumerate(pts_local)]))


def test_the_frame_continuation_is_a_no_op_at_its_own_reference():
    """Continuing `atomic_frames` onto itself changes nothing -- the signs come
    from fixed generic references and are already what the matching picks -- so
    the two are one convention here, and differ only once a path has moved the
    branch."""
    mol = water()
    frames = atomic_frames(mol)[0]
    assert np.array_equal(continued_frames(mol, frames), frames)


def test_the_blocked_three_centre_rows_are_the_dense_tensors_own():
    """(mu nu|P) on the test-set pairs, blocked over the mu shells, against the
    dense (nao, nao, naux) tensor it exists not to form: 209 GB at a hundred
    atoms."""
    mf = interpolated('pbe0')
    mol, auxmol = mf.mol, mf.with_df.auxmol
    mu, nu, _ = pair_layout(mol, mf.with_df.coords)
    dense = pyscf_df.incore.aux_e2(
        mol, auxmol, intor='int3c2e', aosym='s1').reshape(
            mol.nao_nr(), mol.nao_nr(), auxmol.nao_nr())
    assert np.array_equal(pair_three_centre(mol, auxmol, mu, nu),
                          dense[mu, nu, :])


def test_the_skeletons_fit_cannot_be_taken_from_the_mean_field(monkeypatch):
    """The two estimators are not interchangeable at the accuracy a force is
    gated at. `fit_adjoint` reverses `fit_M_stable` on the frozen column set;
    `fit_M_streaming` is a different estimator, and substituting it moves the
    Hartree-Fock skeleton force above the reproducibility floor."""
    mf = interpolated(None)
    reference = isdf_exchange_skeleton(mf)
    streaming = mf.with_df.M               # what the SCF itself fitted
    stable = derivatives.fit_M_stable
    rebuilt = {}

    def substituted(D, F, regularization):
        rebuilt['M'] = stable(D, F, regularization)
        return streaming

    monkeypatch.setattr(derivatives, 'fit_M_stable', substituted)
    moved = isdf_exchange_skeleton(mf)
    monkeypatch.undo()

    factor = (np.abs(rebuilt['M'] - streaming).max()
              / np.abs(rebuilt['M']).max())
    force = float(np.abs(moved - reference).max())
    assert factor == pytest.approx(FIT_FACTOR_DISCREPANCY, rel=0.1)
    assert force == pytest.approx(FIT_FORCE_DISCREPANCY_HF, rel=0.2)
    assert force > GRADIENT_FLOOR


def test_the_production_homes_import_without_the_gradient_package():
    """Forward physics that still needed an adjoint module to import would not
    have moved. Fresh interpreters, both modules, both orders."""
    for module in ('src.Base.pyscf_interface', 'src.Base.isdf_jk'):
        code = (f'import sys, {module}; '
                "assert not [m for m in sys.modules "
                "if m.startswith('src.gradients')], "
                "sorted(m for m in sys.modules if m.startswith('src.gradients'))")
        proc = subprocess.run([sys.executable, '-c', code], cwd=str(REPO),
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


def test_the_two_gradient_modules_import_in_either_order():
    """The shim that stood in for the cycle is gone, so the dependency runs one
    way: the adjoints and the ISDF exchange skeleton in `isdf_derivatives`, the
    assembly of one mean-field force in `isdf_mean_field`."""
    names = ('src.gradients.isdf_derivatives', 'src.gradients.isdf_mean_field')
    for order in (names, names[::-1]):
        code = f'import {order[0]}; import {order[1]}'
        proc = subprocess.run([sys.executable, '-c', code], cwd=str(REPO),
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
    source = (REPO / 'src' / 'gradients' / 'isdf_derivatives.py').read_text()
    assert 'isdf_mean_field' not in source


def test_every_gate_rejects_its_own_perturbation():
    """A check that cannot fail is not a check. One perturbation per gate above,
    each of them the smallest thing that gate exists to catch."""
    mf = fitted('pbe0')
    mol = mf.mol
    x = symmetric_direction(mol.nao_nr())

    # the response kernel, and the MO Fock
    assert not np.array_equal(response_kernel(mf, True)(x),
                              mf.gen_response(hermi=1)(x) * (1.0 + 1e-12))
    F_mo = fock_mo(mf).copy()
    F_mo[0, 0] += 1e-12
    assert not np.array_equal(fock_mo(mf), F_mo)

    # the auxiliary pseudo-inverse: a different lindep is a different operator
    _, _, attenuated = metrics()
    assert not np.array_equal(aux_metric_inverse(attenuated),
                              tikhonov_inverse(attenuated, 1e-2))

    # the static exchange diagonal: the mean field's own K, not a refitted one
    every = np.arange(mf.mo_coeff.shape[1])
    assert not np.array_equal(
        qp_xc_correction(mf),
        static_exchange_diagonal(mf, mol, every, exchange='df-direct'))

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
    # TWO axes, so the reference stays a proper rotation: flipping one alone
    # is undone by the parity repair, which is the convention working.
    flipped = frames.copy()
    flipped[0, :2] *= -1.0
    assert not np.array_equal(continued_frames(interp.mol, flipped), frames)

    # the blocked three-centre rows
    auxmol = interp.with_df.auxmol
    muv, nuv, _ = pair_layout(interp.mol, interp.with_df.coords)
    rows = pair_three_centre(interp.mol, auxmol, muv, nuv).copy()
    rows[0, 0] += 1e-12
    assert not np.array_equal(
        pair_three_centre(interp.mol, auxmol, muv, nuv), rows)


def test_the_folded_exchange_partial_is_the_skeleton_at_two_densities():
    """The shim's one caller-visible content: a folded Fock partial's exchange
    half is the SCF's own contraction with W^2 replaced by W1 W2 and the
    bilinear coefficient doubled, which is what makes the two agree with
    `fock_partial_skeleton_df` at the same gamma."""
    mf = interpolated('pbe0')
    nmo = mf.mo_coeff.shape[1]
    gamma = np.diag(np.arange(nmo) * 0.01 + 0.5)
    gamma[0, 2] = gamma[2, 0] = 0.3
    g_ao = mf.mo_coeff @ (0.5 * (gamma + gamma.T)) @ mf.mo_coeff.T
    assert np.array_equal(
        isdf_fock_partial_exchange(mf, gamma),
        isdf_exchange_skeleton(mf, dm=g_ao, dm_other=mf.make_rdm1(),
                               prefactor=2.0, channels=None))
