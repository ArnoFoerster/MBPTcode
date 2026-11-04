"""Exciton descriptors of Casida/BSE excitations from their (X, Y) eigenvectors:
electron-hole distance, electron and hole sizes, their covariance, and the size
of the excitation, as expectation values over the exciton wavefunction."""
import numpy as np
from pyscf import scf

from src.Base.pyscf.pyscf_interface import get_orbital_energies
from src.Base.orbital_spaces import get_occ_virt_indices


def exciton_descriptors(mf, mol, nocc, X, Y):
    """Spatial descriptors of every root, in bohr, from the exciton wavefunction.

    Parameters
    ----------
    mf : PySCF restricted mean field (RHF or RKS); `mo_coeff` shape (nao, nmo)
    mol : PySCF Mole the mean field was built on
    nocc : int
        number of doubly occupied orbitals
    X, Y : ndarray, shape (nocc*nvirt, nroots), pair index ia = i*nvirt + a
        Casida/BSE eigenvectors with the spatial normalization
        sum_ia X_ia^2 - Y_ia^2 = 1 of `CasidaSolver` and `solve_casida_davidson`;
        Y = 0 in the Tamm-Dancoff approximation.

    Returns
    -------
    dict of ndarray, one entry per root, atomic units (bohr):
        'c_n'     shape (nroots,)    norm <Psi_n|Psi_n> of the exciton wavefunction
        'r_e'     shape (nroots, 3)  <r_e>, mean electron position, index (n, mu)
        'r_h'     shape (nroots, 3)  <r_h>, mean hole position, index (n, mu)
        'd_eh'    shape (nroots,)    |<r_h> - <r_e>|, electron-hole distance
        'sigma_e' shape (nroots,)    sqrt(<r_e^2> - <r_e>^2), electron size
        'sigma_h' shape (nroots,)    sqrt(<r_h^2> - <r_h>^2), hole size
        'cov_eh'  shape (nroots,)    sum_mu <r_e^mu r_h^mu> - <r_e^mu><r_h^mu>
        'R_eh'    shape (nroots,)    cov_eh / (sigma_e sigma_h), correlation
        'd_exc'   shape (nroots,)    sqrt(<|r_h - r_e|^2>), size of the excitation
      and the same projected on the Cartesian components, index (n, x):
        'd_eh_dir', 'sigma_e_dir', 'sigma_h_dir', 'd_exc_dir', 'R_eh_dir'
                  shape (nroots, 3)
        'cov_eh_mat' shape (nroots, 3, 3)  <r_e^x r_h^y> - <r_e^x><r_h^y>,
                  index (n, x, y), x the electron and y the hole component
        'R_eh_mat'   shape (nroots, 3, 3)  cov_eh_mat / (sigma_e^x sigma_h^y)

    Notes
    -----
    The exciton wavefunction of root n, with ψ_i occupied and ψ_a empty orbitals,

        Ψ_n(r_e, r_h) = Σ_ia X_ia ψ_a(r_e) ψ_i(r_h) + Y_ia ψ_i(r_e) ψ_a(r_h),

    is not normalized beyond TDA: c_n = ⟨Ψ_n|Ψ_n⟩ = Σ_ia X_ia² + Y_ia², and every
    expectation value below is ⟨O⟩ = ⟨Ψ_n|O|Ψ_n⟩ / c_n. With the dipole matrix
    µ_pq = ⟨ψ_p|r|ψ_q⟩ and the diagonal second moments M_pq = ⟨ψ_p|r_µ²|ψ_q⟩
    in the orbital basis, the two terms of Ψ_n contribute separately to the
    one-particle moments (the hole sits on i in the X term and on a in the Y term)
    and couple through µ_ia in the two-particle moment:

        ⟨r_h⟩  = [Σ_ija X_ia µ_ij X_ja + Σ_iab Y_ia µ_ab Y_ib] / c_n
        ⟨r_e⟩  = [Σ_iab X_ia µ_ab X_ib + Σ_ija Y_ia µ_ij Y_ja] / c_n
        ⟨r_h²⟩, ⟨r_e²⟩ likewise with M in place of µ
        ⟨r_e^µ r_h^ν⟩ = [Σ_ijab X_ia µ_ab X_jb µ'_ji + Σ_ijab Y_ia µ_ij Y_jb µ'_ab
                         + 2 Σ_ijab X_ia µ_aj Y_jb µ'_ib] / c_n

    where µ carries the electron component and µ' the hole component; the X-Y
    cross term appears twice in Ψ_n² with the same operator, so its factor is 2
    and not a sum with the (µ ↔ ν) transpose, which would differ off the
    diagonal. Then

        d_eh   = |⟨r_h⟩ - ⟨r_e⟩|
        σ_e    = √(⟨r_e²⟩ - ⟨r_e⟩²),   σ_h = √(⟨r_h²⟩ - ⟨r_h⟩²)
        COV_eh = Σ_µ ⟨r_e^µ r_h^µ⟩ - ⟨r_e^µ⟩⟨r_h^µ⟩,   R_eh = COV_eh / (σ_e σ_h)
        d_exc  = √⟨|r_h - r_e|²⟩ = √(d_eh² + σ_e² + σ_h² - 2 COV_eh).

    The directional descriptors project the same moments on one Cartesian
    component x: σ_e^x = √(⟨x_e²⟩ - ⟨x_e⟩²), d_eh^x = |⟨x_h⟩ - ⟨x_e⟩|,
    d_exc^x = √⟨(x_h - x_e)²⟩ and R_eh^x = COV_eh^xx / (σ_e^x σ_h^x), with
    COV_eh^xy = ⟨r_e^x r_h^y⟩ - ⟨r_e^x⟩⟨r_h^y⟩ the full covariance. The totals are
    the quadrature sums, σ_e² = Σ_x (σ_e^x)², d_exc² = Σ_x (d_exc^x)², and
    COV_eh = Σ_x COV_eh^xx.

    d_eh, σ_e, σ_h, COV_eh, R_eh and d_exc are origin independent; ⟨r_e⟩ and
    ⟨r_h⟩ are reported relative to the coordinate origin of `mol`. The
    contraction order follows CP2K's `get_exciton_descriptors`
    (bse_properties.F), so the two implementations agree by construction.

    References
    ----------
    M. Graml and J. Wilhelm, Optical excitations in nanographenes from the
    Bethe-Salpeter equation and time-dependent density functional theory,
    2026, Eqs. (26)-(36); definitions after Mewes, Plasser, Krylov and Dreuw,
    J. Chem. Theory Comput. 14, 710 (2018), Eqs. (15)-(22).
    """
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError("exciton_descriptors is restricted-only; an "
                                  "unrestricted X carries two spin blocks")
    eps = get_orbital_energies(mf, representation='spatial')
    occ, virt = get_occ_virt_indices(eps, nocc)
    nvirt = len(virt)
    nroots = X.shape[1]
    mo = mf.mo_coeff
    ao_r, ao_r2 = _ao_moments(mol)

    # Orbital-basis blocks: block_pq = sum_mn C_mp ao_mn C_nq, index (x, p, q).
    def mo_block(ao, p, q):
        return np.einsum('xmn,mp,nq->xpq', ao, mo[:, p], mo[:, q], optimize=True)

    d_oo, d_vv, d_ov = (mo_block(ao_r, occ, occ), mo_block(ao_r, virt, virt),
                        mo_block(ao_r, occ, virt))
    q_oo, q_vv = mo_block(ao_r2, occ, occ), mo_block(ao_r2, virt, virt)

    # Eigenvectors as (n, i, a); pair index ia = i*nvirt + a is row-major.
    T = np.ascontiguousarray(X.T).reshape(nroots, len(occ), nvirt)
    U = np.ascontiguousarray(Y.T).reshape(nroots, len(occ), nvirt)

    # c_n = sum_ia X_ia X_ia + Y_ia Y_ia
    c_n = np.einsum('nia,nia->n', T, T) + np.einsum('nia,nia->n', U, U)

    def one_particle(block_oo, block_vv):
        # hole:     sum_ija X_ia b_ij X_ja + sum_iab Y_ia b_ab Y_ib
        # electron: sum_iab X_ia b_ab X_ib + sum_ija Y_ia b_ij Y_ja
        h = (np.einsum('nia,xij,nja->nx', T, block_oo, T, optimize=True)
             + np.einsum('nia,xab,nib->nx', U, block_vv, U, optimize=True))
        e = (np.einsum('nia,xab,nib->nx', T, block_vv, T, optimize=True)
             + np.einsum('nia,xij,nja->nx', U, block_oo, U, optimize=True))
        return h / c_n[:, None], e / c_n[:, None]

    r_h, r_e = one_particle(d_oo, d_vv)          # <r_h>, <r_e>       (n, mu)
    r_h2, r_e2 = one_particle(q_oo, q_vv)        # <r_h^2>, <r_e^2>   (n, mu)

    # <r_e^x r_h^y>, electron component x, hole component y, index (n, x, y),
    # contracted pairwise so no intermediate exceeds (n, 3, nocc, nvirt): handed
    # the four-index form whole, einsum's optimizer caps intermediates at the
    # largest input and falls back to one O(nocc^2 nvirt^2) loop per term.
    #   XX: sum_ijab X_ia mu_ab X_jb mu_ji
    w = np.einsum('nia,xab->nxib', T, d_vv)              # sum_a X_ia mu_ab
    v = np.einsum('nxib,njb->nxij', w, T)                # sum_b w_ib X_jb
    xx = np.einsum('nxij,yji->nxy', v, d_oo)             # sum_ij v_ij mu_ji
    #   YY: sum_ijab Y_ia mu_ij Y_jb mu_ab
    w = np.einsum('nia,xij->nxja', U, d_oo)              # sum_i Y_ia mu_ij
    v = np.einsum('nxja,njb->nxab', w, U)                # sum_j w_ja Y_jb
    yy = np.einsum('nxab,yab->nxy', v, d_vv)             # sum_ab v_ab mu_ab
    #   XY: sum_ijab X_ia mu_aj Y_jb mu_ib, entering twice: Psi^2 holds the X-Y
    #   product twice with the same operator, so the second copy is not the
    #   transpose in (x, y) (the transpose would differ off the diagonal).
    w = np.einsum('nia,xja->nxij', T, d_ov)              # sum_a X_ia mu_ja
    v = np.einsum('nxij,njb->nxib', w, U)                # sum_j w_ij Y_jb
    xy = np.einsum('nxib,yib->nxy', v, d_ov)             # sum_ib v_ib mu_ib
    r_eh = (xx + yy + 2.0 * xy) / c_n[:, None, None]
    return _descriptors_from_moments(c_n, r_h, r_e, r_h2, r_e2, r_eh)


def _ao_moments(mol):
    """(r, r_µ²) AO integrals about the origin, each shape (3, nao, nao),
    index (x, m, n)."""
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        ao_r = mol.intor_symmetric('int1e_r', comp=3)      # (3, nao, nao)
        ao_rr = mol.intor_symmetric('int1e_rr', comp=9)    # (9, nao, nao)
    # xx, yy, zz of the (xx xy xz yx yy yz zx zy zz) order
    return ao_r, ao_rr[[0, 4, 8]]


def _descriptors_from_moments(c_n, r_h, r_e, r_h2, r_e2, r_eh):
    """The descriptor dict of `exciton_descriptors` from the normalized moments
    ⟨r_h⟩, ⟨r_e⟩, ⟨r_h²⟩, ⟨r_e²⟩, each shape (nroots, 3), index (n, x), and
    ⟨r_e^x r_h^y⟩, shape (nroots, 3, 3), index (n, x, y)."""
    # COV_eh^xy = <r_e^x r_h^y> - <r_e^x><r_h^y>; the scalar COV_eh is its trace
    cov_mat = r_eh - np.einsum('nx,ny->nxy', r_e, r_h)
    cov_eh = np.einsum('nxx->n', cov_mat)
    # per component: sigma^x = sqrt(<x^2> - <x>^2), d_eh^x = |<x_h> - <x_e>|
    sigma_e_dir = np.sqrt(r_e2 - r_e**2)
    sigma_h_dir = np.sqrt(r_h2 - r_h**2)
    d_eh_dir = np.abs(r_h - r_e)
    cov_diag = np.einsum('nxx->nx', cov_mat)
    d_exc_dir = np.sqrt(d_eh_dir**2 + sigma_e_dir**2 + sigma_h_dir**2 - 2.0 * cov_diag)
    # totals: sigma^2 = sum_x (sigma^x)^2, d_eh^2 = sum_x (d_eh^x)^2, likewise d_exc
    sigma_e = np.sqrt(np.einsum('nx,nx->n', sigma_e_dir, sigma_e_dir))
    sigma_h = np.sqrt(np.einsum('nx,nx->n', sigma_h_dir, sigma_h_dir))
    d_eh = np.linalg.norm(r_h - r_e, axis=1)
    d_exc = np.sqrt(np.einsum('nx,nx->n', d_exc_dir, d_exc_dir))
    return {'c_n': c_n, 'r_e': r_e, 'r_h': r_h, 'd_eh': d_eh, 'sigma_e': sigma_e,
            'sigma_h': sigma_h, 'cov_eh': cov_eh, 'R_eh': cov_eh / (sigma_e * sigma_h),
            'd_exc': d_exc,
            'd_eh_dir': d_eh_dir, 'sigma_e_dir': sigma_e_dir,
            'sigma_h_dir': sigma_h_dir, 'd_exc_dir': d_exc_dir,
            'R_eh_dir': cov_diag / (sigma_e_dir * sigma_h_dir),
            'cov_eh_mat': cov_mat,
            'R_eh_mat': cov_mat / np.einsum('nx,ny->nxy', sigma_e_dir, sigma_h_dir)}


def exciton_descriptors_tdm(mol, mo_coeff, gamma):
    """Spatial and natural-transition-orbital descriptors of every root from its
    full one-particle transition density matrix (1TDM), in bohr.

    Parameters
    ----------
    mol : PySCF Mole the orbitals are expanded on
    mo_coeff : ndarray, shape (nao, norb), index (m, p)
        orthonormal orbitals ψ_p the 1TDM is written in; with a frozen core, the
        active columns of the mean field's `mo_coeff`
    gamma : ndarray, shape (nroots, norb, norb), index (n, p, q)
        γ^0n_pq = ⟨Ψ_0|a†_p a_q|Ψ_n⟩ of one spin block, p the hole and q the
        electron index; all four blocks (oo, ov, vo, vv) enter

    Returns
    -------
    dict of ndarray, the keys of `exciton_descriptors` with 'c_n' the squared
    norm Ω_n = Σ_pq γ_pq², and, index n the root and i the NTO pair,
        'nto_weights' shape (nroots, norb)  λ_ni, descending
        'pr_nto'      shape (nroots,)       Ω² / Σ_i λ_i²
        's_he'        shape (nroots,)       -Σ_i λ̂_i log2 λ̂_i, λ̂_i = λ_i / Ω
        's_he_raw'    shape (nroots,)       -Σ_i λ_i log2 λ_i
        'z_he'        shape (nroots,)       2^s_he

    Notes
    -----
    The exciton wavefunction of root n, ψ_p(r_h) a hole and ψ_q(r_e) an
    electron orbital,

        χ_n(r_h, r_e) = Σ_pq γ_pq ψ_p(r_h) ψ_q(r_e),

    has the squared norm Ω = Σ_pq γ_pq², and ⟨O⟩ = ⟨χ_n|O|χ_n⟩ / Ω. With µ_pq
    and M_pq the dipole and diagonal second-moment matrices of
    `exciton_descriptors`, x the electron and y the hole component,

        ⟨r_h⟩         = Σ_pqr γ_pq µ_pr γ_rq / Ω
        ⟨r_e⟩         = Σ_pqs γ_pq µ_qs γ_ps / Ω
        ⟨r_e^x r_h^y⟩ = Σ_pqrs γ_pq µ^y_pr γ_rs µ^x_qs / Ω

    and ⟨r_h²⟩, ⟨r_e²⟩ with M in place of µ; d_eh, σ_e, σ_h, COV_eh, R_eh and
    d_exc follow from these moments as in `exciton_descriptors`. With
    γ_ia = X_ia, γ_ai = Y_ia and the oo and vv blocks zero they are its terms,
    so the (X, Y) form is the special case of this one.

    The natural transition orbital (NTO) weights λ_i are the squared singular
    values of γ, Σ_i λ_i = Ω. Then

        PR_NTO = (Σ_i λ_i)² / Σ_i λ_i²
        S_H|E  = -Σ_i λ̂_i log2 λ̂_i,   λ̂_i = λ_i / Ω,   Z_HE = 2^S_H|E

    with zero weights dropped from the sum. λ̂ has unit sum, as the worked
    cases of Plasser's definition assume, so one NTO pair gives S_H|E = 0 and
    Z_HE = PR_NTO = 1 at any Ω. The raw form on λ_i,
    S_raw = Ω S_H|E - Ω log2 Ω, is returned beside it. Every descriptor but Ω
    and S_raw is unchanged under γ → cγ.

    References
    ----------
    F. Plasser, J. Chem. Phys. 144, 194107 (2016), Eqs. (11), (12), (15);
    S. A. Bäppler, F. Plasser, M. Wormit and A. Dreuw, Phys. Rev. A 90, 052521
    (2014), the exciton wavefunction as the full 1TDM; definitions after Mewes,
    Plasser, Krylov and Dreuw, J. Chem. Theory Comput. 14, 710 (2018),
    Eqs. (13)-(22).
    """
    gamma = np.asarray(gamma, dtype=float)
    ao_r, ao_r2 = _ao_moments(mol)
    # Orbital-basis blocks: mu_xpq = sum_mn C_mp ao_xmn C_nq, index (x, p, q).
    mu = np.einsum('xmn,mp,nq->xpq', ao_r, mo_coeff, mo_coeff, optimize=True)
    m2 = np.einsum('xmn,mp,nq->xpq', ao_r2, mo_coeff, mo_coeff, optimize=True)

    # c_n = sum_pq gamma_npq gamma_npq
    c_n = np.einsum('npq,npq->n', gamma, gamma)
    if np.any(c_n == 0.0):
        raise ValueError('a root with a zero 1TDM: the charge part of an Ms = 0 '
                         'triplet and the spin part of a singlet vanish')

    def one_particle(b):
        # hole:     sum_pqr gamma_npq b_xpr gamma_nrq
        # electron: sum_pqs gamma_npq b_xqs gamma_nps
        h = np.einsum('npq,xpr,nrq->nx', gamma, b, gamma, optimize=True)
        e = np.einsum('npq,xqs,nps->nx', gamma, b, gamma, optimize=True)
        return h / c_n[:, None], e / c_n[:, None]

    r_h, r_e = one_particle(mu)                  # <r_h>, <r_e>       (n, x)
    r_h2, r_e2 = one_particle(m2)                # <r_h^2>, <r_e^2>   (n, x)

    # <r_e^x r_h^y> = sum_pqrs gamma_npq mu_ypr gamma_nrs mu_xqs, index (n, x, y)
    w = np.einsum('ypr,nrs->nyps', mu, gamma)            # sum_r mu_ypr gamma_nrs
    v = np.einsum('npq,nyps->nyqs', gamma, w)            # sum_p gamma_npq w_nyps
    # r_eh_nxy = sum_qs v_nyqs mu_xqs / c_n
    r_eh = np.einsum('nyqs,xqs->nxy', v, mu) / c_n[:, None, None]

    # lambda_ni = s_ni^2, s_ni the singular values of gamma_n, descending
    lam = np.linalg.svd(gamma, compute_uv=False) ** 2
    lam_hat = lam / c_n[:, None]
    # S = -sum_i lam_ni log2 lam_ni over the nonzero weights, normalized and raw
    s_he = -np.einsum('ni,ni->n', lam_hat,
                      np.log2(np.where(lam_hat > 0.0, lam_hat, 1.0)))
    s_raw = -np.einsum('ni,ni->n', lam, np.log2(np.where(lam > 0.0, lam, 1.0)))
    out = _descriptors_from_moments(c_n, r_h, r_e, r_h2, r_e2, r_eh)
    out.update(nto_weights=lam,
               pr_nto=c_n**2 / np.einsum('ni,ni->n', lam, lam),
               s_he=s_he, s_he_raw=s_raw, z_he=2.0**s_he)
    return out
