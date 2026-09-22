"""
The reaction field's shift of every quasiparticle energy.

Duchemin, Guido, Jacquemin and Blase, Chem. Sci. 9, 4430 (2018) Eq. (18):

    Delta eps_i = -(1/2) <ii|Delta W|ii>        occupied
    Delta eps_a = +(1/2) <aa|Delta W|aa>        virtual
"""
from contextlib import contextmanager

import numpy as np
from pyscf import df as pyscf_df
from pyscf import scf as pyscf_scf

from src.Base.environment import (attached_environment, dresses_interaction,
                                  environment_of)
from src.Base.pyscf_interface import get_orbital_energies
from src.Base.separable_ri import aux_metric_sqrt
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver


def bare_gauge_transform(auxmol, environment, V=None):
    """T with V^(1/2) = Vt^(1/2) T: the map from the dressed auxiliary gauge to
    the bare one, or None when nothing screens.

    A pseudo-inverse because `aux_metric_sqrt` drops the metric's null space,
    and the dressed metric drops it in the same place -- v + vtilde is a
    positive kernel on the same range.

    `dresses_interaction` is what decides: an environment that dresses v is
    exactly one that carries the Eq. (18) shift, so the two gauges exist
    together or not at all.
    """
    if not dresses_interaction(environment, auxmol):
        return None
    V = auxmol.intor('int2c2e', aosym='s1') if V is None else V
    return np.linalg.pinv(aux_metric_sqrt(auxmol, environment, V=V)) @ \
        aux_metric_sqrt(auxmol, None, V=V)


def screened_interaction_difference(w_aux, d_mo, auxmol, environment, V=None):
    """Delta W(0) on the interpolation grid, (M, M), from ONE chi0.

    w_aux is [1 - chi~]^-1 in the gauge `d_mo` was built in, i.e. the dressed
    one; the bare partner is reached by congruence rather than by screening a
    second time. Zero in the gas phase.

    The explicit matrix the compact forms below avoid, kept as the reference
    they are checked against (tests/test_reaction_field.py).
    """
    t = bare_gauge_transform(auxmol, environment, V=V)
    if t is None:
        return None
    n = w_aux.shape[0]
    chi_dressed = np.eye(n) - np.linalg.inv(w_aux)
    chi_bare = t.T @ chi_dressed @ t
    w_bare = np.linalg.inv(np.eye(n) - chi_bare)
    d_bare = d_mo @ t
    return d_mo @ w_aux @ d_mo.T - d_bare @ w_bare @ d_bare.T


def quasiparticle_shift(x_mo, delta_w, nocc):
    """Eq. (18) for every orbital from an explicit Delta W, in Hartree.

    x_mo is the interpolation-grid collocation in the MO basis, so
    <pp|Delta W|pp> is a quadratic form in the orbital density on that grid.
    The reference form of `separable_quasiparticle_shift`, which reaches the
    same numbers without building Delta W; zero without a reaction field.
    """
    if delta_w is None:
        return np.zeros(x_mo.shape[1])
    density = x_mo ** 2
    shift = 0.5 * np.einsum('kp,kl,lp->p', density, delta_w, density,
                            optimize=True)
    shift[:nocc] *= -1.0
    return shift


def bare_screening(w_dressed, transform):
    """[1 - chi~_bare]^-1 from its dressed partner, by congruence.

    chi0 is a property of the solute, so the two screenings differ only in the
    gauge their auxiliary index sits in; screening a second time from scratch
    would build the same chi0 twice and let the two drift apart numerically.
    """
    n = w_dressed.shape[0]
    chi = transform.T @ (np.eye(n) - np.linalg.inv(w_dressed)) @ transform
    return np.linalg.inv(np.eye(n) - chi)


def separable_gauge_transform(mol, environment, auxbasis=None):
    """`bare_gauge_transform` on the auxiliary basis the separable fit uses.

    NOT mf.with_df's. `separable_factors` builds its own auxmol from the
    route's `auxbasis`, and a transform built on a different one is a different
    gauge -- which would make Delta W the difference of two unrelated
    screenings instead of two factorizations of one chi0.

    None means `dresses_interaction` is False, which is why a caller reading
    `transform is not None` is reading the one decision and not a second one:
    the environments that dress the interaction are the environments that
    carry the Eq. (18) shift.
    """
    auxmol = pyscf_df.addons.make_auxmol(
        mol, auxbasis=auxbasis or (str(mol.basis) + '-ri'))
    return bare_gauge_transform(auxmol, environment)


def separable_quasiparticle_shift(x_mo, d_mo, w_dressed, transform, nocc):
    """Eq. (18) for every orbital from the separable factors, in Hartree.

    Delta W on the interpolation grid is (M, M) in a rank that grows with the
    system -- 1.4e9 entries at a 500-atom solute. Nothing needs that matrix:
    Eq. (18) contracts it twice against the SAME orbital density, so projecting
    the density onto the auxiliary index first leaves nothing bigger than
    (naux, nmo).

    Zero when `transform` is None, which is the gas phase.
    """
    if transform is None:
        return np.zeros(x_mo.shape[1])
    a = d_mo.T @ (x_mo ** 2)
    a_bare = transform.T @ a
    shift = 0.5 * (
        np.einsum('Qp,QR,Rp->p', a, w_dressed, a, optimize=True)
        - np.einsum('Qp,QR,Rp->p', a_bare, bare_screening(w_dressed, transform),
                    a_bare, optimize=True))
    shift[:nocc] *= -1.0
    return shift


def environment_quasiparticle_shift(mf, mol=None, nocc=None, auxbasis=None):
    """
    Eq. (18) for every orbital of `mf`, or None in the gas phase.
    """
    mol = mf.mol if mol is None else mol

    # Unrestricted falls back to COHSEX
    if isinstance(mf, pyscf_scf.uhf.UHF):
        return None

    # The gas-phase exit comes FIRST
    environment = environment_of(mf)
    if not getattr(environment, 'screens', True):
        return None
    nocc = mol.nelectron // 2 if nocc is None else nocc
    auxmol = pyscf_df.addons.make_auxmol(
        mol, auxbasis=auxbasis or getattr(getattr(mf, 'with_df', None),
                                          'auxbasis', None)
        or (str(mol.basis) + '-ri'))
    V = auxmol.intor('int2c2e', aosym='s1')
    if not dresses_interaction(environment, auxmol):
        return None
    t = bare_gauge_transform(auxmol, environment, V=V)
    key = np.asarray(mf.mo_energy, float).tobytes()
    cached = getattr(mf, '_reaction_field_shift', None)
    if cached is not None and cached[0] == key and cached[1] is environment:
        return cached[2]

    # B = Vt^(1/2) V^-1 (P|pq): the Coulomb-metric fit, dressed in the gauge
    # where the interaction is B^T B and the bare partner is T^T B.
    mo = np.asarray(mf.mo_coeff, float)
    naux = V.shape[0]
    three = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1')
    fit = np.linalg.solve(V, three.reshape(-1, naux).T).reshape(naux, *mo.shape[:1] * 2)
    coeff = np.einsum('PQ,Qmn,mp,nq->Ppq', aux_metric_sqrt(auxmol, environment,
                                                           V=V),
                      fit, mo, mo, optimize=True)
    del three, fit

    eps = get_orbital_energies(mf, representation='spatial')
    w_dressed = np.asarray(LinearResponseSolver(
        eps, coeff_df=coeff, spin_mode='restricted').static_screening_aux(nocc))
    # the orbital densities (P|pp), the only columns a self-element needs
    rho = np.einsum('Qpp->Qp', coeff, optimize=True)
    rho_bare = t.T @ rho
    shift = 0.5 * (
        np.einsum('Qp,QR,Rp->p', rho, w_dressed, rho, optimize=True)
        - np.einsum('Qp,QR,Rp->p', rho_bare, bare_screening(w_dressed, t),
                    rho_bare, optimize=True))
    shift[:nocc] *= -1.0
    mf._reaction_field_shift = (key, environment, shift)
    return shift


@contextmanager
def bare_self_energy(mf, shift):
    """
    The environment off `mf` while a self-energy's integrals are built.

    The continuum is carried by `shift` instead, and screening Sigma with the
    dressed interaction as well would count the same polarization twice: Eq.
    (18) IS the static approximation to Sigma[W_solv] - Sigma[W_gas]. A route
    with no shift -- the gas phase, or the unrestricted COHSEX fallback --
    keeps its environment and the dressed screening that goes with it.
    """
    if shift is None:
        yield mf
    else:
        with attached_environment(mf, None):
            yield mf
