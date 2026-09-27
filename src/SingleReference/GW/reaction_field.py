"""
The reaction field's shift of every quasiparticle energy.

Duchemin, Guido, Jacquemin and Blase, Chem. Sci. 9, 4430 (2018) Eq. (18):

    Delta eps_i = -(1/2) <ii|Delta W|ii>        occupied
    Delta eps_a = +(1/2) <aa|Delta W|aa>        virtual
"""
import hashlib
from contextlib import contextmanager

import numpy as np
from pyscf import df as pyscf_df
from pyscf import scf as pyscf_scf

from src.Base.constants import REACTION_FIELD_CACHE_SIZE

from src.Base.environment import (attached_environment, dresses_interaction,
                                  environment_of)
from src.Base.pyscf_interface import (get_orbital_energies,
                                      require_closed_shell_or_unrestricted)
from src.Base.separable_ri import aux_metric_sqrt, default_auxbasis
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
        mol, auxbasis=auxbasis or default_auxbasis(mol.basis))
    return bare_gauge_transform(auxmol, environment)


def projected_quasiparticle_shift(a_dressed, w_dressed, a_bare, w_bare, nocc):
    """Eq. (18) for every orbital from the PROJECTED orbital densities, in Hartree.

    THE ONE CONTRACTION OF EQ. (18). Delta W never forms: the shift contracts
    it twice against the same orbital density, so projecting that density onto
    the auxiliary index first -- a[Q,p] = sum_k D[k,Q] X[k,p]^2 -- leaves
    nothing bigger than (naux, nmo), against the (M, M) of Delta W itself.

    How the BARE partner is reached is the caller's: by congruence from one
    chi0 (`separable_quasiparticle_shift`, a forward pass's cheapest route) or
    by screening a second factorization
    (`gradients.reaction_field_adjoint.reaction_field_shift`, whose D branch
    carries a nuclear derivative that a pseudo-inverse of a metric root does
    not). The two agree to the ISDF reproducibility floor and differ nowhere
    else, because the contraction below is the same object for both.

    The sign is the occupancy's: the added charge is stabilized either way, so
    both branches close the quasiparticle gap.
    """
    shift = 0.5 * (
        np.einsum('Qp,QR,Rp->p', a_dressed, w_dressed, a_dressed, optimize=True)
        - np.einsum('Qp,QR,Rp->p', a_bare, w_bare, a_bare, optimize=True))
    shift[:nocc] *= -1.0
    return shift


def separable_quasiparticle_shift(x_mo, d_mo, w_dressed, transform, nocc):
    """Eq. (18) for every orbital from the separable factors, in Hartree.

    ONE chi0 and a congruence: the bare screening is `bare_screening` of the
    dressed one and the bare density its congruent partner, which is what a
    forward pass wants. The contraction itself is
    `projected_quasiparticle_shift`, so Delta W on the interpolation grid --
    (M, M) in a rank that grows with the system -- never forms.

    Zero when `transform` is None, which is the gas phase.
    """
    if transform is None:
        return np.zeros(x_mo.shape[1])
    a = d_mo.T @ (x_mo ** 2)
    return projected_quasiparticle_shift(a, w_dressed, transform.T @ a,
                                         bare_screening(w_dressed, transform),
                                         nocc)


def _shift_key(mf, nocc, auxbasis):
    """Everything the cached Eq. (18) array depends on besides the environment.

    The spectrum alone is not enough: the shift also reads the orbitals, the
    occupation it splits them by and the auxiliary basis the screening is built
    in, and two calls that differ in any of them must not share an array.
    """
    orbitals = hashlib.sha1(np.ascontiguousarray(mf.mo_coeff, float)).hexdigest()
    return (np.asarray(mf.mo_energy, float).tobytes(), orbitals,
            repr(nocc), repr(auxbasis))


def environment_quasiparticle_shift(mf, mol=None, nocc=None, auxbasis=None,
                                    environment=None):
    """
    Eq. (18) for every orbital of `mf`, or None in the gas phase.

    Restricted: an (nmo,) array. Unrestricted: a (2, nmo) array, alpha then
    beta. The screened interaction is ONE object, built from the spin-summed
    chi0 = chi0_alpha + chi0_beta, and each spin orbital's shift contracts it
    against that orbital's own density with its own occupancy sign -- so a
    closed shell run unrestricted gives the restricted shift in both rows.

    nocc: the occupation, (nalpha, nbeta) when unrestricted; by default the
    mean field's own.

    environment: the continuum to form it in, by default the one attached to
    `mf`. Passing another -- `SolventScreening.static_partner()` -- gives the
    same contraction in the static response on the same cavity, without
    re-attaching anything.
    """
    mol = mf.mol if mol is None else mol
    unrestricted = isinstance(mf, pyscf_scf.uhf.UHF)

    # The gas-phase exit comes FIRST
    environment = environment_of(mf) if environment is None else environment
    if not getattr(environment, 'screens', True):
        return None
    require_closed_shell_or_unrestricted(mf, 'the Eq. (18) shift', mol=mol)
    if unrestricted:
        nocc = tuple(int(n) for n in (mf.nelec if nocc is None else nocc))
    else:
        nocc = mol.nelectron // 2 if nocc is None else nocc
    auxbasis = (auxbasis or getattr(getattr(mf, 'with_df', None), 'auxbasis', None)
                or default_auxbasis(mol.basis))
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=auxbasis)
    V = auxmol.intor('int2c2e', aosym='s1')
    if not dresses_interaction(environment, auxmol):
        return None
    t = bare_gauge_transform(auxmol, environment, V=V)
    # One entry per (orbitals, continuum): the optical and the static response
    # of one mean field are asked for side by side (`equilibrium_level_shift`),
    # and a single resident entry would rebuild each on every alternation.
    key = (_shift_key(mf, nocc, auxbasis), id(environment))
    cache = mf.__dict__.get('_reaction_field_shift')
    if not isinstance(cache, dict):
        cache = mf._reaction_field_shift = {}
    hit = cache.get(key)
    if hit is not None and hit[0] is environment:
        return hit[1]

    mos = ([np.asarray(c, float) for c in mf.mo_coeff] if unrestricted
           else [np.asarray(mf.mo_coeff, float)])
    coeffs = _dressed_fit(mol, auxmol, V, environment, mos)

    eps = get_orbital_energies(mf, representation='spatial')
    if unrestricted:
        solver = LinearResponseSolver(tuple(eps), coeff_df=tuple(coeffs),
                                      spin_mode='unrestricted')
    else:
        solver = LinearResponseSolver(eps, coeff_df=coeffs[0],
                                      spin_mode='restricted')
    w_dressed = np.asarray(solver.static_screening_aux(nocc))
    w_bare = bare_screening(w_dressed, t)
    shifts = []
    for coeff, n in zip(coeffs, nocc if unrestricted else (nocc,)):
        # the orbital densities (P|pp), the only columns a self-element needs
        rho = np.einsum('Qpp->Qp', coeff, optimize=True)
        rho_bare = t.T @ rho
        s = 0.5 * (np.einsum('Qp,QR,Rp->p', rho, w_dressed, rho, optimize=True)
                   - np.einsum('Qp,QR,Rp->p', rho_bare, w_bare, rho_bare,
                               optimize=True))
        s[:n] *= -1.0
        shifts.append(s)
    shift = np.array(shifts) if unrestricted else shifts[0]
    while len(cache) >= REACTION_FIELD_CACHE_SIZE:
        cache.pop(next(iter(cache)))
    cache[key] = (environment, shift)
    return shift


def _dressed_fit(mol, auxmol, V, environment, mos):
    """B = Vt^(1/2) V^-1 (P|pq) for each set of orbitals in `mos`.

    The Coulomb-metric fit, dressed in the gauge where the interaction is
    B^T B and the bare partner is T^T B. One fit serves both spins; only the
    MO transform is per spin.
    """
    naux, nao = V.shape[0], mol.nao_nr()
    three = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1')
    fit = np.linalg.solve(V, three.reshape(-1, naux).T).reshape(naux, nao, nao)
    dressed_fit = np.einsum('PQ,Qmn->Pmn',
                            aux_metric_sqrt(auxmol, environment, V=V), fit,
                            optimize=True)
    del three, fit
    return [np.einsum('Pmn,mp,nq->Ppq', dressed_fit, mo, mo, optimize=True)
            for mo in mos]


def dressed_factors(mf, mol=None, environment=None, auxbasis=None):
    """(B in the dressed gauge, one per spin, and T), or None where nothing
    dresses the interaction.

    The factors Eq. (18) is built from, in the mean field's own orbitals:
    B_s = Vt^(1/2) V^-1 (P|pq) and the bare partner T^T B_s, on the mean
    field's auxiliary basis. A loop that rotates its orbitals (qsGW) rotates
    these, so its continuum operator and the Eq. (18) of every other route
    read one gauge.
    """
    mol = mf.mol if mol is None else mol
    environment = environment_of(mf) if environment is None else environment
    auxbasis = (auxbasis or getattr(getattr(mf, 'with_df', None), 'auxbasis', None)
                or default_auxbasis(mol.basis))
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=auxbasis)
    V = auxmol.intor('int2c2e', aosym='s1')
    if not dresses_interaction(environment, auxmol):
        return None
    mos = ([np.asarray(c, float) for c in mf.mo_coeff]
           if isinstance(mf, pyscf_scf.uhf.UHF)
           else [np.asarray(mf.mo_coeff, float)])
    return (_dressed_fit(mol, auxmol, V, environment, mos),
            bare_gauge_transform(auxmol, environment, V=V))


def continuum_operator(coeffs, t, w_dressed, noccs):
    """The static continuum operator of a quasiparticle-self-consistent loop,
    one (nmo, nmo) matrix per spin, in Hartree.

    STATIC COHSEX IN Delta W, in the approximation that gives Eq. (18). With
    Delta W = W_dressed - W_bare the static screened reaction field,

        Sigma^COH_pq = 1/2 sum_m (pm|Delta W|mq),
        Sigma^SEX_pq = - sum_i^occ (pi|Delta W|iq).

    Duchemin et al. reach Eq. (18) by keeping, of the Coulomb hole's
    completeness sum and of the screened exchange's density matrix, the
    orbital that carries the charge alone -- the self-polarization of |p>^2 in
    its own reaction field, 1/2 (pp|Delta W|pp) with the occupancy sign.
    Keeping the orbital of each index in the same way, and symmetrizing,

        Sigma^solv_pq = 1/4 [ s_p (pp|Delta W|pq) + s_q (pq|Delta W|qq) ],

    s = -1 occupied and +1 virtual: Hermitian, and its diagonal IS Eq. (18),
    so a loop's first cycle adds exactly what the one-shot routes add. Where
    Delta W is constant over the molecule (a Born sphere) it is also the whole
    COHSEX operator: the completeness and the density matrix then close
    exactly and nothing was dropped.

    coeffs: the dressed factors in the loop's current orbitals, one per spin
    (`dressed_factors`, rotated); t the bare gauge transform; w_dressed the
    static screening of all spins' polarizability in the dressed gauge.
    """
    w_bare = bare_screening(w_dressed, t)
    out = []
    for coeff, nocc in zip(coeffs, noccs):
        rho = np.einsum('Qpp->Qp', coeff, optimize=True)
        bare = np.einsum('QR,Qpq->Rpq', t, coeff, optimize=True)
        m = (np.einsum('Rp,Rpq->pq', w_dressed @ rho, coeff, optimize=True)
             - np.einsum('Rp,Rpq->pq', w_bare @ (t.T @ rho), bare,
                         optimize=True))
        sign = np.ones(m.shape[0])
        sign[:nocc] = -1.0
        quarter = 0.25 * sign[:, None] * m
        out.append(quarter + quarter.T)
    return out


def equilibrium_level_shift(mf, mol=None, nocc=None, auxbasis=None):
    """Eq. (18) in the static response less Eq. (18) in the optical one, per
    orbital in Hartree: how much further a level moves once the solvent has
    reoriented around the charged state.

    THE SCREENED FORM. Both terms are Duchemin et al.'s self-polarization of
    the orbital carrying the charge, one at eps_static and one at eps_inf on
    the SAME cavity (`SolventScreening.static_partner`), each screened by the
    solute's own chi0:

        Delta eps_p^eq = Eq18_p(eps_s) - Eq18_p(eps_inf),

    with Eq. (18)'s occupancy sign, so an occupied level rises further and a
    virtual one falls further. The ion's energy E^(N-/+1) = E_0 -/+ eps_p
    therefore drops by

        Delta E_ss = -/+ Delta eps_p^eq <= 0,

    its equilibrium solvation beyond the vertical (non-equilibrium) one, and
    lambda_s = -Delta E_ss is the solvent (outer-sphere) reorganization
    energy of the charge transfer. It is a classical relaxation added to the
    ion's energy, not a self-energy term: it enters AFTER the quasiparticle
    equation, unrenormalized by Z.

    Zero when eps_static = eps_inf. Raises when the environment has no static
    response to pair with -- a continuum without eps_static, or none at all.
    Restricted: (nmo,); unrestricted: (2, nmo), alpha then beta.
    """
    environment = environment_of(mf)
    if not hasattr(environment, 'static_partner'):
        raise ValueError(
            f'equilibrium solvation needs a continuum with a static response; '
            f'this mean field carries {type(environment).__name__}')
    static = environment.static_partner()
    optical = environment_quasiparticle_shift(mf, mol, nocc, auxbasis)
    relaxed = environment_quasiparticle_shift(mf, mol, nocc, auxbasis,
                                              environment=static)
    return relaxed - optical


def equilibrium_solvation_energy(level_shift, orbital, nocc):
    """Delta E_ss <= 0 of the ion made by removing (orbital < nocc) or adding
    an electron in `orbital`, from one channel's `equilibrium_level_shift`."""
    sign = -1.0 if int(orbital) < int(nocc) else 1.0
    return sign * float(np.asarray(level_shift)[int(orbital)])


@contextmanager
def bare_self_energy(mf, shift):
    """
    The environment off `mf` while a self-energy's integrals are built.

    The continuum is carried by `shift` instead, and screening Sigma with the
    dressed interaction as well would count the same polarization twice: Eq.
    (18) IS the static approximation to Sigma[W_solv] - Sigma[W_gas]. A route
    with no shift -- the gas phase, or an environment that dresses no
    interaction -- keeps its environment and the screening that goes with it.
    """
    if shift is None:
        yield mf
    else:
        with attached_environment(mf, None):
            yield mf
