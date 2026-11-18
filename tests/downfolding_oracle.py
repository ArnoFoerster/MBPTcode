"""Brute-force numpy reference for the downfolding tests.

Every function here is written from the spin-orbital formulas of the
downfolding spec (§3.3, "Every formula in both forms"), literally and with
plain loops where that reads closest to the formula, on exact 4-centre
integrals taken straight from `mol.intor`. Nothing in `src/` is used, so the
production code in `src/MultiReference/Downfolding/` is checked against an
independent implementation (tests/test_downfolding_one_body.py,
tests/test_downfolding_energy.py, tests/test_downfolding_two_body.py).
The direct RPA (`drpa`, `rpa_energy`) is written from its restricted form
(spec §3.3, row E_RPA,corr).

Spin orbitals are numbered alpha first, then beta: with n spatial orbitals,
spin orbital p < n is alpha orbital p and spin orbital n + p is beta orbital p.
"""
import numpy as np
from pyscf import gto, scf


def water():
    """Converged RHF of H2O in 6-31G (spherical), at a fixed standard geometry."""
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='6-31g', cart=False, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.verbose = 0
    return mf.run()


def h2():
    """Converged RHF of H2 at 0.74 Angstrom in 6-31G (spherical)."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='6-31g', cart=False,
                verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.verbose = 0
    return mf.run()


def _transform(mol, c1, c2, c3, c4):
    """Chemist (pq|rs) with the four indices on the orbitals c1..c4, from the
    exact AO integrals."""
    eri_ao = mol.intor('int2e')
    return np.einsum('abcd,ap,bq,cr,ds->pqrs', eri_ao, c1, c2, c3, c4,
                     optimize=True)


def _hcore(mol):
    return mol.intor('int1e_kin') + mol.intor('int1e_nuc')


def mo_integrals(mf, mo_coeff=None):
    """(h, eri): the core Hamiltonian and the chemist (pq|rs) on the spatial
    MOs (mf.mo_coeff unless `mo_coeff` is given), from exact AO integrals."""
    c = mf.mo_coeff if mo_coeff is None else mo_coeff
    return c.T @ _hcore(mf.mol) @ c, _transform(mf.mol, c, c, c, c)


def spin_orbital_integrals(mol, mo_a, mo_b):
    """(h_so, eri_so) over the 2n spin orbitals, alpha orbitals `mo_a` first,
    then beta orbitals `mo_b`.

    eri_so[p, q, r, s] = (pq|rs) is zero unless p and q have the same spin
    and r and s have the same spin; mo_a and mo_b may be different orbitals.
    """
    n = mo_a.shape[1]
    hcore = _hcore(mol)
    mos = (mo_a, mo_b)
    h_so = np.zeros((2 * n, 2 * n))
    eri_so = np.zeros((2 * n,) * 4)
    for s, cs in enumerate(mos):
        bs = slice(s * n, (s + 1) * n)
        h_so[bs, bs] = cs.T @ hcore @ cs
        for r, cr in enumerate(mos):
            br = slice(r * n, (r + 1) * n)
            eri_so[bs, bs, br, br] = _transform(mol, cs, cs, cr, cr)
    return h_so, eri_so


def spin_orbital_one_body(h_so, eri_so, occ_env, act, vtilde_so, occ_act, dc):
    """(f, t_dc) on the active spin orbitals, from spec §3.3's spin-orbital column.

    f_tu    = h_tu + sum_{i in E} (v_tu,ii - v_ti,iu) rho_ii
    t^DC_tu = sum_{vw in A} (vt_tu,vw - vt_tw,vu) rho_vw     (dc='full')
    t^DC_tu = sum_{vw in A} vt_tu,vw rho_vw                  (dc='hartree',
                                                              the form with Sigma^E)

    h_so, eri_so: one-body and chemist two-body integrals over all spin orbitals.
    occ_env: the occupied environment spin orbitals (rho_ii = 1; the empty
        environment orbitals have rho_ii = 0 and drop out).
    act: the active spin orbitals; f and t_dc are indexed locally in this order.
    vtilde_so: (len(act),)*4 chemist effective interaction on the active
        spin orbitals, local indices.
    occ_act: local indices of the occupied active spin orbitals, where the
        reference density rho^A,0 is 1.
    """
    n = len(act)
    rho = np.zeros((n, n))
    for k in occ_act:
        rho[k, k] = 1.0
    f = np.zeros((n, n))
    t_dc = np.zeros((n, n))
    for t in range(n):
        for u in range(n):
            p, q = act[t], act[u]
            f[t, u] = h_so[p, q]
            for i in occ_env:
                f[t, u] += eri_so[p, q, i, i] - eri_so[p, i, i, q]
            for v in range(n):
                for w in range(n):
                    term = vtilde_so[t, u, v, w]
                    if dc == 'full':
                        term -= vtilde_so[t, w, v, u]
                    elif dc != 'hartree':
                        raise ValueError(dc)
                    t_dc[t, u] += term * rho[v, w]
    return f, t_dc


def spin_orbital_energies(h_so, eri_so, occ_env, vtilde_so, occ_act,
                          sigma_bar_so=None):
    """The E_0 terms of spec §3.3's spin-orbital column, as a dict.

    e_core        = sum_{i in E} h_ii rho_ii
    e_ee          = 1/2 sum_{ij in E} (v_ii,jj - v_ij,ji) rho_ii rho_jj
    hartree       = 1/2 sum_{tu in A} vt_tt,uu rho_tt rho_uu
    exchange      = -1/2 sum_{tu in A} vt_tu,ut rho_tt rho_uu
                    (hartree + exchange = E_HF,ee[vt])
    exchange_term = sum_{t,v occ. active} vt_tv,vt      (the term with Sigma^E)
    sigma_trace   = sum_{t occ. active} Sigma_bar_tt    (enters E_0 with a minus)

    Arguments as in `spin_orbital_one_body`; sigma_bar_so is the static
    self-energy on the active spin orbitals (local indices), or None for 0.
    """
    e_core = sum(h_so[i, i] for i in occ_env)
    e_ee = 0.0
    for i in occ_env:
        for j in occ_env:
            e_ee += 0.5 * (eri_so[i, i, j, j] - eri_so[i, j, j, i])
    hartree = exchange = exchange_term = 0.0
    for t in occ_act:
        for u in occ_act:
            hartree += 0.5 * vtilde_so[t, t, u, u]
            exchange -= 0.5 * vtilde_so[t, u, u, t]
            exchange_term += vtilde_so[t, u, u, t]
    trace = 0.0
    if sigma_bar_so is not None:
        trace = sum(sigma_bar_so[t, t] for t in occ_act)
    return {'e_core': e_core, 'e_ee': e_ee, 'hartree': hartree,
            'exchange': exchange, 'exchange_term': exchange_term,
            'sigma_trace': trace}


def reference_energy(g_eff, n_act_occ):
    """The energy of the closed-shell reference determinant of a spin-free
    g_eff (keys 0, 1, 2; g_eff[2] is physicist <tu|vw>, so its chemist form is
    g_eff[2].transpose(0, 2, 1, 3)) with the first n_act_occ active orbitals
    doubly occupied:

    E = g_eff[0] + 2 sum_k h_kk + sum_kl [2 (kk|ll) - (kl|lk)].
    """
    h = g_eff[1]
    eri = g_eff[2].transpose(0, 2, 1, 3)
    e = g_eff[0]
    for k in range(n_act_occ):
        e += 2 * h[k, k]
        for l in range(n_act_occ):
            e += 2 * eri[k, k, l, l] - eri[k, l, l, k]
    return e


def _pair_space(eri, eps, nocc):
    """(delta, K) on the nocc*nvirt (i, a) pairs in C order: delta_ia =
    eps_a - eps_i and K_ia,jb = (ia|jb), a counted from nocc."""
    delta = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()
    n = delta.size
    return delta, eri[:nocc, nocc:, :nocc, nocc:].reshape(n, n)


def drpa(eri, eps, nocc, keep=None):
    """(omega, xpy): the restricted singlet direct RPA of chemist `eri`.

    M = delta^1/2 (delta + 4K) delta^1/2 = Z omega^2 Z^T and
    xpy = delta^1/2 Z omega^-1/2, one excitation per column, rows the (i, a)
    pairs in C order. keep: a bool mask over those pairs; the problem is then
    solved on the kept pairs alone and xpy has one row per kept pair.
    """
    delta, K = _pair_space(eri, eps, nocc)
    if keep is not None:
        delta, K = delta[keep], K[np.ix_(keep, keep)]
    root = np.sqrt(delta)
    M = root[:, None] * (np.diag(delta) + 4.0 * K) * root[None, :]
    omega2, Z = np.linalg.eigh(M)
    omega = np.sqrt(omega2)
    return omega, root[:, None] * Z / np.sqrt(omega)[None, :]


def rpa_energy(eri, eps, nocc):
    """The restricted direct RPA correlation energy of chemist `eri`,
    1/2 sum_n omega_n - 1/2 sum_ia (delta_ia + 2 K_ia,ia)."""
    delta, K = _pair_space(eri, eps, nocc)
    omega, _ = drpa(eri, eps, nocc)
    return 0.5 * omega.sum() - 0.5 * np.sum(delta + 2.0 * np.diag(K))
