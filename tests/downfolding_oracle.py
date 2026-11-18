"""Brute-force numpy reference for the downfolding tests.

Every function here is written from the spin-orbital formulas of the
downfolding spec (§3.3, "Every formula in both forms"), literally and with
plain loops where that reads closest to the formula, on exact 4-centre
integrals taken straight from `mol.intor`. Nothing in `src/` is used, so the
production code in `src/MultiReference/Downfolding/` is checked against an
independent implementation (tests/test_downfolding_one_body.py).

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
