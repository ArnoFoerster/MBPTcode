"""
The reaction field's shift of every quasiparticle energy.

Duchemin, Guido, Jacquemin and Blase, Chem. Sci. 9, 4430 (2018) Eq. (18):

    Delta eps_i = -(1/2) <ii|Delta W|ii>        occupied
    Delta eps_a = +(1/2) <aa|Delta W|aa>        virtual

WHERE IT ENTERS (`shift_after_solve`). In G0W0 the shift is added to the
quasiparticle root after the solve, whole. Inside the G0W0 equation the fixed
Sigma_c(w) renormalizes it to Z x shift, while the reaction-field dressing of
the BSE direct kernel, which Eqs. (18)-(20) of the paper pair it with, enters
whole: the (1 - Z)(P_e + P_h) left over is a spurious blue shift of 130-550 meV
in any continuum (2026-10-19). On an evGW cycle (an `eps_anchor` handed
to the route) the shift stays inside the equation, as in the paper: G and W
are rebuilt on levels that already carry it, so Sigma_c moves with them and
the fixed point realizes the whole shift. Adding it after the solve there as
well would count it again, once through the levels and once on top.
"""
import hashlib
from contextlib import contextmanager

import numpy as np
from pyscf import df as pyscf_df
from pyscf import scf as pyscf_scf

from src.Base.basis.ri_fallback import paired_auxbasis
from src.Base.constants import (FIT_CHOLESKY_BLOCK, ISDF_TILE_GB,
                                REACTION_FIELD_CACHE_SIZE)
from src.Base.environment import (attached_environment, dresses_interaction,
                                  environment_of)
from src.Base.pyscf_interface import (get_orbital_energies,
                                      require_closed_shell_or_unrestricted)
from src.Base.separable_ri import (aux_metric_sqrt,
                                   rows_transpose_product)
from src.Base.sliced_factors import GridTileRows, SlicedFactors
from src.Base.utils.mpi_grid import current_comm
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.LinearResponse.space_time import dyson_screening


def shift_after_solve(eps_anchor):
    """Whether a quasiparticle route adds Eq. (18) to its roots after the
    solve (True: G0W0) or carries it in the static term of the equation
    (False: an evGW cycle, the one caller that anchors the equation on a
    spectrum other than the one screening it; see the module docstring)."""
    return eps_anchor is None


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


def bare_screening(w_dressed, transform):
    """[1 - chi~_bare]^-1 from its dressed partner, by congruence.

    chi0 is a property of the solute, so the two screenings differ only in the
    gauge their auxiliary index sits in; screening a second time from scratch
    would build the same chi0 twice and let the two drift apart numerically.
    """
    n = w_dressed.shape[0]
    return dyson_screening(np.eye(n) - np.linalg.inv(w_dressed), transform)


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
        mol, auxbasis=auxbasis or paired_auxbasis(mol.basis))
    return bare_gauge_transform(auxmol, environment)


def sliced_density(x_mo, d, rho_rows, what, block=None):
    """A[Q, k] = sum_g D[g, Q] rho[g, k] from this rank's grid rows `rho_rows`
    of a density and of D (`SlicedFactors` of the layout of `x_mo`): a sum
    over the grid of row products in fixed tiles of `block` points
    (`separable_ri.rows_transpose_product`), so no rank holds X_mo or D whole
    and A is the same bits at every rank count."""
    if not isinstance(d, SlicedFactors) or d.rows != x_mo.rows:
        raise ValueError(f'the {what} on sliced factors need D as rows of the '
                         f'same layout as X_mo')
    comm = x_mo.require(current_comm()).comm
    block = FIT_CHOLESKY_BLOCK if block is None else int(block)
    rho = GridTileRows.from_blocks(rho_rows, x_mo.npts, block, comm)
    return rows_transpose_product(d.D, d.rows[0], rho, block=block, comm=comm)


def density_projection(x_mo, d, block=None):
    """A[Q, p] = sum_k D[k, Q] X_mo[k, p]^2 from this rank's rows of the two
    `SlicedFactors` (one gauge's D, the shared X_mo), by `sliced_density`."""
    return sliced_density(x_mo, d, x_mo.X_mo ** 2, 'orbital densities', block)


def orbital_densities(x_mo, d_mo):
    """A = D^T X_mo^2, (naux, nmo), the orbital densities Eq. (18) contracts:
    one product of whole arrays, or the tiled grid sum on `SlicedFactors`."""
    if isinstance(x_mo, SlicedFactors):
        return density_projection(x_mo, d_mo)
    return d_mo.T @ (x_mo ** 2)


def pair_densities(x_mo, d_mo, pairs, block=None):
    """A[Q, k] = sum_g D[g, Q] X_mo[g, p_k] X_mo[g, q_k] for the orbital pairs
    (p_k, q_k): `orbital_densities` of the overlap densities, whole or as the
    tiled grid sum on `SlicedFactors`."""
    p = [int(a) for a, _ in pairs]
    q = [int(b) for _, b in pairs]
    if not isinstance(x_mo, SlicedFactors):
        return d_mo.T @ (x_mo[:, p] * x_mo[:, q])
    return sliced_density(x_mo, d_mo, x_mo.X_mo[:, p] * x_mo.X_mo[:, q],
                          'pair densities', block)


def offdiagonal_reaction_field(screening, pair_dens, pairs, nocc):
    """<p|Sigma^env|q> for the pairs, p != q on one side of the gap, in Hartree.

    Eq. (18) is the self-polarization of one orbital's density, quartic in
    its coefficients, so no one-body operator has it as its diagonal in every
    basis. The element taken is the one that reproduces it to first order in
    a rotation of p into q, symmetrized:

        Sigma^env_pq = +-(1/2) [(pp|Delta W|pq) + (qq|Delta W|qp)],

    the sign the occupancy's as on the diagonal. screening: ((A, W) dressed,
    (A, W) bare); pair_dens: the matching pair densities (`pair_densities`).
    """
    out = np.zeros(len(pairs))
    for ((a, w), pd), sign in zip(zip(screening, pair_dens), (1.0, -1.0)):
        for k, (p, q) in enumerate(pairs):
            out[k] += sign * 0.5 * float((a[:, p] + a[:, q]) @ w @ pd[:, k])
    occupied = np.array([p < nocc for p, _ in pairs])
    out[occupied] *= -1.0
    return out


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


def congruent_quasiparticle_shift(a, w_dressed, transform, nocc):
    """Eq. (18) from the dressed orbital densities a and W(0), the bare partner
    by congruence (`bare_screening`): THE ONE SHIFT of every forward route."""
    return projected_quasiparticle_shift(a, w_dressed, transform.T @ a,
                                         bare_screening(w_dressed, transform),
                                         nocc)


def separable_quasiparticle_shift(x_mo, d_mo, w_dressed, transform, nocc):
    """Eq. (18) for every orbital from the separable factors, in Hartree.

    ONE chi0 and a congruence: the bare screening is `bare_screening` of the
    dressed one and the bare density its congruent partner, which is what a
    forward pass wants. The contraction itself is
    `projected_quasiparticle_shift`, so Delta W on the interpolation grid --
    (M, M) in a rank that grows with the system -- never forms.

    Zero when `transform` is None, which is the gas phase. On `SlicedFactors`
    d_mo is the rows of the same layout (`density_projection`).
    """
    if transform is None:
        return np.zeros(x_mo.nmo if isinstance(x_mo, SlicedFactors)
                        else x_mo.shape[1])
    return congruent_quasiparticle_shift(orbital_densities(x_mo, d_mo),
                                         w_dressed, transform, nocc)


def _shift_key(mf, mol, nocc, auxmol):
    """Everything the cached Eq. (18) array depends on besides the environment:
    the spectrum, the orbitals, the occupation it splits them by, the geometry,
    and the orbital and auxiliary basis sets. Two calls that differ in any of
    them must not share an array (a second auxiliary basis, an occupation or a
    copy at a displaced geometry with unchanged eigenvalues is rebuilt).
    """
    orbitals = hashlib.sha1(np.ascontiguousarray(mf.mo_coeff, float)).hexdigest()
    return (np.asarray(mf.mo_energy, float).tobytes(), orbitals, repr(nocc),
            np.asarray(mol.atom_coords(), float).tobytes(), repr(mol._basis),
            repr(auxmol._basis))


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

    One array for every route on these factors: the separable routes take the
    same `congruent_quasiparticle_shift` on their own factors
    (`separable_quasiparticle_shift`) and agree with it to the fit. Cached on
    the mean field under every input of the shift -- spectrum, orbitals,
    occupation, geometry, orbital and auxiliary basis -- and the environment,
    so an evGW copy, a second auxiliary basis or a displaced geometry is
    rebuilt rather than served a resident array.
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
                or paired_auxbasis(mol.basis))
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=auxbasis)
    V = auxmol.intor('int2c2e', aosym='s1')
    if not dresses_interaction(environment, auxmol):
        return None
    t = bare_gauge_transform(auxmol, environment, V=V)
    # One entry per (inputs, continuum): the optical and the static response
    # of one mean field are asked for side by side (`equilibrium_level_shift`),
    # and a single resident entry would rebuild each on every alternation.
    key = (_shift_key(mf, mol, nocc, auxmol), id(environment))
    cache = mf.__dict__.get('_reaction_field_shift')
    if not isinstance(cache, dict):
        cache = mf._reaction_field_shift = {}
    hit = cache.get(key)
    if hit is not None and hit[0] is environment:
        return hit[1]

    # cycle: LinearResponseSolver reaches back into this package's screening
    from src.SingleReference.LinearResponse.linear_response import \
        LinearResponseSolver

    # B = Vt^(1/2) V^-1 (P|pq): the Coulomb-metric fit, dressed in the gauge
    # where the interaction is B^T B and the bare partner is T^T B. Only two
    # slices of B enter per spin -- the occupied-virtual block chi0 is built
    # from and the orbital densities (P|pp) a self-element reads -- so (P|mn)
    # is never held whole.
    eps = get_orbital_energies(mf, representation='spatial')
    mos = ([np.asarray(c, float) for c in mf.mo_coeff] if unrestricted
           else [np.asarray(mf.mo_coeff, float)])
    spectra = tuple(eps) if unrestricted else (eps,)
    noccs = nocc if unrestricted else (nocc,)
    root = aux_metric_sqrt(auxmol, environment, V=V)
    coeff_ov, rho = [], []
    for (pair_ov, pair_pp) in _three_center_ov_and_diagonal(
            mol, auxmol, [(mo, *get_occ_virt_indices(e, n))
                          for mo, e, n in zip(mos, spectra, noccs)]):
        coeff_ov.append(root @ np.linalg.solve(V, pair_ov))
        rho.append(root @ np.linalg.solve(V, pair_pp))
        del pair_ov, pair_pp

    if unrestricted:
        solver = LinearResponseSolver(spectra, coeff_ov=tuple(coeff_ov),
                                      spin_mode='unrestricted')
    else:
        solver = LinearResponseSolver(eps, coeff_ov=coeff_ov[0],
                                      spin_mode='restricted')
    w_dressed = np.asarray(solver.static_screening_aux(nocc))
    shifts = [congruent_quasiparticle_shift(r, w_dressed, t, n)
              for r, n in zip(rho, noccs)]
    shift = np.array(shifts) if unrestricted else shifts[0]
    while len(cache) >= REACTION_FIELD_CACHE_SIZE:
        cache.pop(next(iter(cache)))
    cache[key] = (environment, shift)
    return shift


def _three_center_ov_and_diagonal(mol, auxmol, orbital_sets,
                                  tile_gb=ISDF_TILE_GB):
    """For each (mo, occ, virt) of `orbital_sets`: ((P|ia) as
    (naux, nocc*nvirt), (P|pp) as (naux, nmo)), the two MO slices of the
    unfitted three-centre integrals, built over blocks of auxiliary shells of
    at most `tile_gb` of (P|mn) each; every set reads the same blocks."""
    nao = mol.nao_nr()
    naux = auxmol.nao_nr()
    out, plans = [], []
    for mo, occ, virt in orbital_sets:
        mo = np.asarray(mo, float)
        plans.append((mo, np.ascontiguousarray(mo[:, occ]),
                      np.ascontiguousarray(mo[:, virt])))
        out.append((np.empty((naux, len(occ) * len(virt))),
                    np.empty((naux, mo.shape[1]))))
    ao_loc = auxmol.ao_loc_nr()
    per_function = 8.0 * nao * nao
    s0 = 0
    while s0 < auxmol.nbas:
        s1 = s0 + 1
        while (s1 < auxmol.nbas and (ao_loc[s1 + 1] - ao_loc[s0]) * per_function
               <= tile_gb * 1e9):
            s1 += 1
        p0, p1 = ao_loc[s0], ao_loc[s1]
        three = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1',
                                       shls_slice=(0, mol.nbas, 0, mol.nbas,
                                                   s0, s1))
        for (mo, mo_o, mo_v), (pair_ov, pair_pp) in zip(plans, out):
            half = np.tensordot(three, mo_o, axes=(0, 0))      # (n, P, i)
            pair_ov[p0:p1] = np.tensordot(half, mo_v,
                                          axes=(0, 0)).reshape(p1 - p0, -1)
            half = np.tensordot(three, mo, axes=(0, 0))        # (n, P, p)
            pair_pp[p0:p1] = np.einsum('nPp,np->Pp', half, mo, optimize=True)
            del half
        del three
        s0 = s1
    return out


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
                or paired_auxbasis(mol.basis))
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
