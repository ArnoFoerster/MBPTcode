"""Brute-force numpy reference for the downfolding tests.

Every function here is written from the spin-orbital form of the
downfolding formulas, literally and with plain loops where that reads
closest to the formula, on exact 4-centre integrals taken straight from
`mol.intor`. Nothing in `src/` is used, so the production
code in `src/EffectiveHamiltonian/Downfolding/` is checked against an
independent implementation (tests/test_downfolding_one_body.py,
tests/test_downfolding_energy.py, tests/test_downfolding_two_body.py).
The direct RPA (`drpa`, `rpa_energy`) is written from its restricted form;
the cRPA correction (`vtilde_crpa` and the two zeroing variants) is the
pole sum of its modes, and `vtilde_crpa_so` the same correction built in
spin orbitals without spin adaptation; `w_static_active` is the full W(0) on the active orbitals.
`vtilde_mrpa` is the mRPA interaction (Scott and Booth's Eq. 5) in spin
orbitals, from its own solve of every singlet and triplet mode.
The environment self-energy has its own residues of the full W
(`env_residues`), the exact pole sum S^E(z) at a complex frequency
(`sigma_env`) and the static reductions written from Marie and Loos's
Eq. 12 at eta = 0 (`sigma_mode_a`) and Eq. 44 (`sigma_srg`)
(tests/test_downfolding_sigma.py).

The two-spin functions at the end (`drpa_so`, `w0_minus_v_so`,
`vtilde_crpa_uhf`, `vtilde_mrpa_uhf`, `env_residues_uhf`, `sigma_env_uhf`,
`sigma_mode_a_uhf`, `sigma_srg_uhf`, `reference_energy_uhf`) are the same
formulas for an unrestricted reference, where the alpha and beta orbitals,
orbital energies, occupied counts and active sets differ. They work on the
spin-orbital integrals of `spin_orbital_integrals`, the spin-orbital
energies, the list of occupied spin orbitals and the list of active spin
orbitals (alpha active orbitals, then beta), with no spin adaptation
anywhere (tests/test_downfolding_unrestricted.py).

Spin orbitals are numbered alpha first, then beta: with n spatial orbitals,
spin orbital p < n is alpha orbital p and spin orbital n + p is beta orbital p.
"""
import numpy as np
from pyscf import df, dft, gto, lib, scf


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


OH = 'O 0 0 0; H 0 0 0.97'
CH2 = 'C 0 0 0; H 0 0.9931 0.4226; H 0 -0.9931 0.4226'
NH = 'N 0 0 0; H 0 0 1.04'


def unrestricted(atom, spin, basis='6-31g', xc=None, density_fit=False):
    """Converged UHF (UKS with the functional `xc`) of `atom` with
    `spin` = N_alpha - N_beta, spherical basis; density fitted on request."""
    mol = gto.M(atom=atom, basis=basis, spin=spin, cart=False, verbose=0)
    mf = scf.UHF(mol) if xc is None else dft.UKS(mol, xc=xc)
    if density_fit:
        mf = mf.density_fit()
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    assert mf.converged
    return mf


def h2_broken(basis='6-31g'):
    """Converged broken-symmetry UHF of H2 at 2.5 Angstrom: one electron on
    each atom, N_alpha = N_beta, the alpha and beta orbitals different."""
    mol = gto.M(atom='H 0 0 0; H 0 0 2.5', basis=basis, cart=False, verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    dm = mf.init_guess_by_atom()
    first = mol.aoslice_by_atom()[0]
    dm[0][first[2]:first[3]] = dm[0][:, first[2]:first[3]] = 0.0  # alpha on atom 2
    dm[1][first[3]:] = dm[1][:, first[3]:] = 0.0                  # beta on atom 1
    mf.kernel(dm0=dm)
    assert mf.converged and mf.spin_square()[0] > 0.5
    return mf


def fitted_spin_orbital_integrals(mf):
    """eri_so over the 2n spin orbitals (alpha first), as
    `spin_orbital_integrals` gives it, but of the density-fitted interaction
    of the mean field `mf`: (pq|rs) = sum_Q L_Q,pq L_Q,rs with L the
    Cholesky-decomposed three-index integrals of mf's auxiliary basis,
    computed here by pyscf from the auxiliary molecule, in each spin's
    orbitals."""
    mol = mf.mol
    l_ao = lib.unpack_tril(df.incore.cholesky_eri(mol,
                                                  auxmol=mf.with_df.auxmol))
    l_mo = [np.einsum('Qmn,mp,nq->Qpq', l_ao, c, c, optimize=True)
            for c in mf.mo_coeff]
    n = l_mo[0].shape[1]
    eri_so = np.zeros((2 * n,) * 4)
    for s in (0, 1):
        bs = slice(s * n, (s + 1) * n)
        for r in (0, 1):
            br = slice(r * n, (r + 1) * n)
            eri_so[bs, bs, br, br] = np.einsum('Qpq,Qrs->pqrs', l_mo[s],
                                               l_mo[r], optimize=True)
    return eri_so


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
    """(f, t_dc) on the active spin orbitals, in spin-orbital form.

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
    """The E_0 terms in spin-orbital form, as a dict.

    e_core        = sum_{i in E} h_ii rho_ii
    e_hartree     = 1/2 sum_{ij in E} v_ii,jj rho_ii rho_jj
    e_exchange    = -1/2 sum_{ij in E} v_ij,ji rho_ii rho_jj
    e_ee          = e_hartree + e_exchange
    hartree       = 1/2 sum_{tu in A} vt_tt,uu rho_tt rho_uu
    exchange      = -1/2 sum_{tu in A} vt_tu,ut rho_tt rho_uu
                    (hartree + exchange = E_HF,ee[vt])
    exchange_term = sum_{t,v occ. active} vt_tv,vt      (the term with Sigma^E)
    sigma_trace   = sum_{t occ. active} Sigma_bar_tt    (enters E_0 with a minus)

    Arguments as in `spin_orbital_one_body`; sigma_bar_so is the static
    self-energy on the active spin orbitals (local indices), or None for 0.
    """
    e_core = sum(h_so[i, i] for i in occ_env)
    e_hartree = e_exchange = 0.0
    for i in occ_env:
        for j in occ_env:
            e_hartree += 0.5 * eri_so[i, i, j, j]
            e_exchange -= 0.5 * eri_so[i, j, j, i]
    hartree = exchange = exchange_term = 0.0
    for t in occ_act:
        for u in occ_act:
            hartree += 0.5 * vtilde_so[t, t, u, u]
            exchange -= 0.5 * vtilde_so[t, u, u, t]
            exchange_term += vtilde_so[t, u, u, t]
    trace = 0.0
    if sigma_bar_so is not None:
        trace = sum(sigma_bar_so[t, t] for t in occ_act)
    return {'e_core': e_core, 'e_hartree': e_hartree,
            'e_exchange': e_exchange, 'e_ee': e_hartree + e_exchange,
            'hartree': hartree, 'exchange': exchange,
            'exchange_term': exchange_term, 'sigma_trace': trace}


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


def reference_energy_uhf(g_eff, n_act_occ):
    """The energy of the reference determinant of a spin-resolved g_eff,
    g_eff[1] = (h_a, h_b) and g_eff[2] = (aa, ab, bb) physicist, with the
    first n_act_occ[s] active orbitals of spin s occupied:

    E = g_eff[0] + sum_s sum_k h^s_kk
        + 1/2 sum_s sum_kl [(kk|ll)^ss - (kl|lk)^ss] + sum_{k in a, l in b} (kk|ll)^ab.
    """
    h = g_eff[1]
    aa, ab, bb = (x.transpose(0, 2, 1, 3) for x in g_eff[2])
    e = g_eff[0]
    for s, same in enumerate((aa, bb)):
        for k in range(n_act_occ[s]):
            e += h[s][k, k]
            for l in range(n_act_occ[s]):
                e += 0.5 * (same[k, k, l, l] - same[k, l, l, k])
    for k in range(n_act_occ[0]):
        for l in range(n_act_occ[1]):
            e += ab[k, k, l, l]
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


def _active_pairs(eps, nocc, part):
    """Bool mask over the (i, a) pairs in C order, True where i is an active
    occupied and a an active virtual orbital (written out here so the oracle
    does not lean on the production mask)."""
    occ = {int(i) for i in part.occ_active}
    virt = {int(a) for a in part.virt_active}
    return np.array([i in occ and a in virt for i in range(nocc)
                     for a in range(nocc, len(eps))], dtype=bool)


def _pole_sum(eri, nocc, act, omega, xpy, pairs):
    """W(0) - v on the orbitals `act`, chemist order, from restricted singlet
    RPA modes (omega, xpy) whose xpy rows are the (i, a) pairs `pairs`.

    Spin orbitals (paper Eq. 2.12): W(0) - v = sum_n w_n w_n^T (-2/Omega_n),
    w_n,pq = sum_ia (pq|ia) (X+Y)_ia,n over the spin-orbital pairs. A
    restricted singlet mode has (X+Y)_{ia sigma} = xpy_ia / sqrt(2) on both
    spins (same norm X^TX - Y^TY = 1), so for either spin of p, q its
    spin-summed residue is w_n,pq = sqrt(2) sum_ia (pq|ia) xpy_ia,n; triplet
    modes have opposite signs on the two spins and give w = 0. Hence the
    restricted residue factor is -4/Omega_n on r_n,pq = sum_ia (pq|ia) xpy_ia,n:
        W(0) - v = -4 sum_n r_n r_n^T / Omega_n,
    i.e. -4 v (A+B)^-1 v for xpy diag(1/Omega) xpy^T = (A+B)^-1.
    """
    n = len(act)
    occ, virt = zip(*pairs) if len(pairs) else ((), ())
    v = eri[np.ix_(act, act)][:, :, list(occ), list(virt)].reshape(n * n, -1)
    r = v @ xpy
    out = -4.0 * (r / omega[None, :]) @ r.T
    return out.reshape((n,) * 4)


def _pair_labels(eps, nocc):
    return [(i, a) for i in range(nocc) for a in range(nocc, len(eps))]


def w0_minus_v(eri, eps, nocc, act, keep=None):
    """W(0) - v on the orbitals `act` (chemist order) of the restricted direct
    RPA of chemist `eri`, solved on the (i, a) pairs `keep` (bool mask, None:
    all), by the pole sum of `_pole_sum`."""
    omega, xpy = drpa(eri, eps, nocc, keep=keep)
    labels = _pair_labels(eps, nocc)
    if keep is not None:
        labels = [p for p, k in zip(labels, keep) if k]
    return _pole_sum(eri, nocc, act, omega, xpy, labels)


def vtilde_crpa(eri, eps, nocc, part):
    """The cRPA correction on part.act: the active pairs (i and a both
    active) deleted from the response, the rest solved and pole-summed."""
    keep = ~_active_pairs(eps, nocc, part)
    return w0_minus_v(eri, eps, nocc, part.act, keep=keep)


def vtilde_crpa_zeroing(eri, eps, nocc, part):
    """The cRPA correction by the paper's literal Eq. 2.14.

    (A+B)^R = A+B and (A-B)^R = A-B = delta, both set to zero on the block
    where both pairs are active (i, a, j, b all in the active space), then
    the full-size M = (A-B)^1/2 (A+B) (A-B)^1/2 diagonalized. (A-B)^1/2 is
    zero on the active rows, so M is block diagonal: the active block gives
    Omega = 0 modes, which are dropped, and the residual block is the deleted
    problem. X+Y = (A-B)^1/2 Z Omega^-1/2 as in `drpa`.
    """
    delta, K = _pair_space(eri, eps, nocc)
    active = _active_pairs(eps, nocc, part)
    apb = np.diag(delta) + 4.0 * K
    apb[np.ix_(active, active)] = 0.0
    amb = delta.copy()
    amb[active] = 0.0
    root = np.sqrt(amb)
    omega2, Z = np.linalg.eigh(root[:, None] * apb * root[None, :])
    modes = omega2 > 1e-10 * omega2.max()
    assert np.count_nonzero(~modes) == np.count_nonzero(active)
    omega = np.sqrt(omega2[modes])
    xpy = root[:, None] * Z[:, modes] / np.sqrt(omega)[None, :]
    return _pole_sum(eri, nocc, part.act, omega, xpy,
                     _pair_labels(eps, nocc))


def vtilde_zero_apb_only(eri, eps, nocc, part):
    """The wrong variant: only A+B zeroed on the active block, A-B = delta
    kept, then the full-size M = delta^1/2 (A+B) delta^1/2 solved and
    pole-summed. With the cross blocks of A+B left in, M is indefinite, so
    some Omega_n = sqrt(Omega_n^2) are imaginary; the pole sum is taken in
    complex arithmetic, where the residue r_n r_n^T / Omega_n is
    (delta^1/2 Z)_n (delta^1/2 Z)_n^T / Omega_n^2 and real."""
    delta, K = _pair_space(eri, eps, nocc)
    active = _active_pairs(eps, nocc, part)
    apb = np.diag(delta) + 4.0 * K
    apb[np.ix_(active, active)] = 0.0
    root = np.sqrt(delta)
    omega2, Z = np.linalg.eigh(root[:, None] * apb * root[None, :])
    omega = np.sqrt(omega2.astype(complex))
    xpy = root[:, None] * Z / np.sqrt(omega)[None, :]
    out = _pole_sum(eri, nocc, part.act, omega, xpy, _pair_labels(eps, nocc))
    assert np.abs(out.imag).max() < 1e-10
    return out.real


def w_static_active(eri, eps, nocc, part):
    """W(0) on part.act (chemist order) of the full restricted direct RPA of
    chemist `eri`: the bare active integrals plus the pole sum over every
    (i, a) pair, nothing deleted."""
    act = part.act
    return (eri[np.ix_(act, act, act, act)]
            + w0_minus_v(eri, eps, nocc, act))


def vtilde_crpa_so(eri, eps, nocc, part):
    """The cRPA correction written in spin orbitals, its alpha-alpha block.

    The spin-orbital direct RPA has A = delta + K and B = K with
    K_(i s a s),(j t b t) = (i a|j b) for any two spins s, t (Coulomb only;
    spin-flip pairs have no Coulomb coupling and are left out), so
    (A+B)_so = delta + 2K over the 2 nocc nvirt same-spin pairs. The pairs
    whose i and a are both active are deleted, for both spins, and
        vt_so = -2 V [(A+B)_so,rr]^-1 V^T,
    V_(t u),(i a) = (t s u s|i t a t) over every active spin orbital t, u,
    the static (omega = 0) form of the pole sum -2/Omega_n of paper Eq. 2.12.
    No restricted spin adaptation is used, so this checks the restricted
    factor -4 on delta + 4K independently. Spin orbitals as at the top of
    this module: p < nmo alpha, nmo + p beta."""
    nmo = len(eps)
    active = _active_pairs(eps, nocc, part)
    labels = _pair_labels(eps, nocc)
    pairs = [(s * nmo + i, s * nmo + a) for s in (0, 1)
             for (i, a), act in zip(labels, active) if not act]
    spin_orb = [(p % nmo, p // nmo) for p in range(2 * nmo)]

    def coulomb(p, q, r, s):
        (p, sp), (q, sq), (r, sr), (s, ss) = (spin_orb[x] for x in (p, q, r, s))
        return eri[p, q, r, s] if sp == sq and sr == ss else 0.0

    apb = np.zeros((len(pairs), len(pairs)))
    for x, (i, a) in enumerate(pairs):
        apb[x, x] = eps[a % nmo] - eps[i % nmo]
        for y, (j, b) in enumerate(pairs):
            apb[x, y] += 2.0 * coulomb(i, a, j, b)
    act_so = [p for p in part.act] + [nmo + p for p in part.act]
    m = len(act_so)
    v = np.array([[coulomb(t, u, i, a) for (i, a) in pairs]
                  for t in act_so for u in act_so])
    vt = (-2.0 * v @ np.linalg.solve(apb, v.T)).reshape((m,) * 4)
    n = len(part.act)
    return vt[:n, :n, :n, :n]


def vtilde_mrpa(eri, eps, nocc, part):
    """The mRPA correction written in spin orbitals: (vt_aa, vt_ab).

    Scott and Booth, PRL 132, 076401 (2024), Eq. 5, on the spin-blocked
    particle-hole space: the 2 nocc nvirt same-spin pairs (i s, a s), alpha
    then beta, with A - B = delta and A + B = delta + 2K_so,
    K_(i s a s),(j t b t) = (i a|j b) for any two spins s, t, as in
    `vtilde_crpa_so`. Its own direct RPA of every mode, singlet and triplet
    alike: M = delta^1/2 (A+B) delta^1/2 = Z Omega^2 Z^T and X+Y =
    delta^1/2 Z Omega^-1/2 (X^T X - Y^T Y = 1). The zeroth moment
    eta0 = (X+Y)(X+Y)^T and eta1 = A - B = delta are projected on the pairs
    whose i and a are both active, for both spins, and Eq. 5 gives the
    interaction on that block,
        v_eff = 1/2 [eta0_A^-1 delta_A eta0_A^-1 - delta_A],
    and vt = v_eff - K_so on it. vt is set on (ia|jb), (ai|jb), (ia|bj) and
    (ai|bj) of the active spin orbitals (real orbitals) and is zero on every
    other element. Returned: the (n_act,)*4 chemist blocks with both pairs
    alpha (vt_aa) and with the first pair alpha, the second beta (vt_ab),
    local indices on part.act. Spin orbitals as at the top of this module."""
    nmo = len(eps)
    labels = _pair_labels(eps, nocc)
    pairs = [(s * nmo + i, s * nmo + a) for s in (0, 1) for (i, a) in labels]
    act_so = [p for p in part.act] + [nmo + p for p in part.act]
    keep = [x for x, (i, a) in enumerate(pairs)
            if i in act_so and a in act_so]
    delta = np.array([eps[a % nmo] - eps[i % nmo] for (i, a) in pairs])
    k_so = np.array([[eri[i % nmo, a % nmo, j % nmo, b % nmo]
                      for (j, b) in pairs] for (i, a) in pairs])
    root = np.sqrt(delta)
    omega2, z = np.linalg.eigh(root[:, None] * (np.diag(delta) + 2.0 * k_so)
                               * root[None, :])
    xpy = root[:, None] * z / np.sqrt(np.sqrt(omega2))[None, :]
    eta0 = (xpy @ xpy.T)[np.ix_(keep, keep)]
    d_a = np.diag(delta[keep])
    inv = np.linalg.inv(eta0)
    v_eff = 0.5 * (inv @ d_a @ inv - d_a)
    vt_ph = v_eff - k_so[np.ix_(keep, keep)]
    m = len(act_so)
    local = {p: x for x, p in enumerate(act_so)}
    vt = np.zeros((m,) * 4)
    for x, px in enumerate(keep):
        i, a = (local[p] for p in pairs[px])
        for y, py in enumerate(keep):
            j, b = (local[p] for p in pairs[py])
            for t, u in ((i, a), (a, i)):
                for v, w in ((j, b), (b, j)):
                    vt[t, u, v, w] = vt_ph[x, y]
    n = len(part.act)
    return vt[:n, :n, :n, :n], vt[:n, :n, n:, n:]


def env_residues(eri, eps, nocc, part):
    """(omega, w, c) of the full restricted direct RPA of chemist `eri` for
    the environment self-energy of part.act.

    w[n, t, r] = sqrt(2) sum_ia (t r|i a) xpy_ia,n, t over part.act and r over
    the environment orbitals (inactive, then environment virtual): the
    spin-orbital residue of W between t and r for either spin (the restricted
    singlet mode puts xpy/sqrt(2) on both spins of each pair, see
    `_pole_sum`), so that sum_r w_tr w_ur over spatial r is the spin-orbital
    sum over the r of t's spin. c[n, r] = eps_r + sgn(eps_r - eps_F) omega_n,
    the pole of S^E: eps_r - omega_n for an occupied r, eps_r + omega_n for an
    empty one.
    """
    omega, xpy = drpa(eri, eps, nocc)
    occ, virt = zip(*_pair_labels(eps, nocc))
    env = list(part.env_occ) + list(part.env_virt)
    act = list(part.act)
    v = eri[np.ix_(act, env)][:, :, list(occ), list(virt)]
    w = np.sqrt(2.0) * np.einsum('trP,Pn->ntr', v, xpy)
    sign = np.array([-1.0 if r < nocc else 1.0 for r in env])
    c = np.asarray(eps)[env][None, :] + sign[None, :] * omega[:, None]
    return omega, w, c


def sigma_env(z, eri, eps, nocc, part):
    """S^E_tu(z) = sum_{r in E} sum_n w_tr w_ur / (z - eps_r - sgn(eps_r - eps_F) omega_n),
    the correlation part of G^E (W - v) at the complex frequency z, on
    part.act (local indices); `env_residues` gives w and the poles."""
    _, w, c = env_residues(eri, eps, nocc, part)
    n = len(part.act)
    out = np.zeros((n, n), dtype=complex)
    for t in range(n):
        for u in range(n):
            out[t, u] = np.sum(w[:, t, :] * w[:, u, :] / (z - c))
    return out


def sigma_mode_a(eri, eps, nocc, part):
    """Eq. 12 of Marie and Loos at eta = 0: 1/2 [S^E_tu(eps_t) + S^E_tu(eps_u)],
    the pole sum `sigma_env` on the real axis at the mean-field energies."""
    act = list(part.act)
    n = len(act)
    out = np.zeros((n, n))
    for t in range(n):
        for u in range(n):
            out[t, u] = 0.5 * (sigma_env(eps[act[t]], eri, eps, nocc, part)[t, u]
                               + sigma_env(eps[act[u]], eri, eps, nocc, part)[t, u]).real
    return out


def sigma_srg(eri, eps, nocc, part, s):
    """Eq. 44 of Marie and Loos on the environment orbitals r:

        Sigma_tu = sum_rn (D_tr + D_ur) / (D_tr^2 + D_ur^2) w_tr w_ur
                   [1 - exp(-(D_tr^2 + D_ur^2) s)],

    D_tr = eps_t - eps_r - sgn(eps_r - eps_F) omega_n; s = None is s = inf
    (Eq. 42), the bracket 1."""
    _, w, c = env_residues(eri, eps, nocc, part)
    act = list(part.act)
    n = len(act)
    out = np.zeros((n, n))
    for t in range(n):
        for u in range(n):
            d_t, d_u = eps[act[t]] - c, eps[act[u]] - c
            lam = d_t**2 + d_u**2
            damp = 1.0 if s is None else 1.0 - np.exp(-lam * s)
            out[t, u] = np.sum((d_t + d_u) / lam * w[:, t, :] * w[:, u, :]
                               * damp)
    return out


# --- two spins: an unrestricted reference, everything in spin orbitals --------

def so_pairs(eps_so, occ_so):
    """The same-spin particle-hole pairs (i, a) of the spin orbitals, stacked:
    the alpha pairs in C order over (i, a), then the beta pairs. Spin-flip
    pairs have no Coulomb coupling and are left out."""
    n = len(eps_so) // 2
    occ = {int(i) for i in occ_so}
    pairs = []
    for s in (0, 1):
        spin = range(s * n, (s + 1) * n)
        pairs += [(i, a) for i in spin if i in occ
                  for a in spin if a not in occ]
    return pairs


def drpa_so(eri_so, eps_so, occ_so, keep=None):
    """(e_c, omega, xpy, pairs): the direct RPA in spin orbitals.

    A = delta + K and B = K on the pairs of `so_pairs`, K_ia,jb = (ia|jb)
    for any two spins, so A - B = delta and A + B = delta + 2K:
    M = delta^1/2 (delta + 2K) delta^1/2 = Z omega^2 Z^T,
    X+Y = delta^1/2 Z omega^-1/2 (X^T X - Y^T Y = 1), one mode per column,
    and e_c = 1/2 (sum_n omega_n - Tr A). keep: a bool mask over the pairs;
    the problem is then solved on the kept pairs alone, which `pairs` lists
    and the rows of xpy follow.
    """
    pairs = so_pairs(eps_so, occ_so)
    if keep is not None:
        pairs = [pair for pair, kept in zip(pairs, keep) if kept]
    m = len(pairs)
    delta = np.array([eps_so[a] - eps_so[i] for i, a in pairs])
    k = np.zeros((m, m))
    for x, (i, a) in enumerate(pairs):
        for y, (j, b) in enumerate(pairs):
            k[x, y] = eri_so[i, a, j, b]
    root = np.sqrt(delta)
    omega2, z = np.linalg.eigh(root[:, None] * (np.diag(delta) + 2.0 * k)
                               * root[None, :])
    omega = np.sqrt(omega2)
    xpy = root[:, None] * z / np.sqrt(omega)[None, :]
    e_c = 0.5 * omega.sum() - 0.5 * np.sum(delta + np.diag(k))
    return e_c, omega, xpy, pairs


def w0_minus_v_so(eri_so, eps_so, occ_so, orbs, keep=None):
    """W(0) - v between the spin orbitals `orbs`, chemist (len(orbs),)*4, of
    the direct RPA of `drpa_so` (on the pairs `keep`, None: all), by the pole
    sum of the paper's Eq. 2.12:

        W(0) - v = -2 sum_n w_n w_n^T / omega_n,
        w_n,pq = sum_ia (pq|ia) (X+Y)_ia,n.
    """
    _, omega, xpy, pairs = drpa_so(eri_so, eps_so, occ_so, keep=keep)
    m = len(orbs)
    v = np.zeros((m * m, len(pairs)))
    for x, (p, q) in enumerate((p, q) for p in orbs for q in orbs):
        for y, (i, a) in enumerate(pairs):
            v[x, y] = eri_so[p, q, i, a]
    w = v @ xpy
    return (-2.0 * (w / omega[None, :]) @ w.T).reshape((m,) * 4)


def active_pairs_so(pairs, act_so):
    """Bool mask over `pairs`, True where i and a are both active."""
    act = {int(p) for p in act_so}
    return np.array([i in act and a in act for i, a in pairs], dtype=bool)


def spin_blocks(x, n):
    """(aa, ab, bb) of a chemist array over 2n local spin orbitals, alpha
    first: both pairs alpha; the first pair alpha and the second beta; both
    beta."""
    return x[:n, :n, :n, :n], x[:n, :n, n:, n:], x[n:, n:, n:, n:]


def vtilde_crpa_uhf(eri_so, eps_so, occ_so, act_so):
    """(aa, ab, bb): the cRPA correction on the active spin orbitals. The
    pairs whose i and a are both active are deleted, for both spins, and the
    rest is solved and pole-summed (`w0_minus_v_so`). Local indices in the
    order of act_so, the alpha active orbitals first."""
    keep = ~active_pairs_so(so_pairs(eps_so, occ_so), act_so)
    vt = w0_minus_v_so(eri_so, eps_so, occ_so, list(act_so), keep=keep)
    return spin_blocks(vt, len(act_so) // 2)


def vtilde_mrpa_uhf(eri_so, eps_so, occ_so, act_so):
    """(aa, ab, bb): the mRPA correction on the active spin orbitals.

    Scott and Booth, PRL 132, 076401 (2024), Eq. 5, with the full direct RPA
    of `drpa_so`: eta0 = (X+Y)(X+Y)^T and eta1 = A - B = delta are projected
    on the pairs whose i and a are both active, of both spins, and

        v_eff = 1/2 [eta0_A^-1 delta_A eta0_A^-1 - delta_A],  vt = v_eff - K

    on that block. vt is set on (ia|jb), (ai|jb), (ia|bj) and (ai|bj) of the
    active spin orbitals (real orbitals) and is zero on every other element.
    """
    _, _, xpy, pairs = drpa_so(eri_so, eps_so, occ_so)
    keep = np.flatnonzero(active_pairs_so(pairs, act_so))
    m = len(act_so)
    vt = np.zeros((m,) * 4)
    if len(keep) == 0:
        return spin_blocks(vt, m // 2)
    d_a = np.diag([eps_so[pairs[x][1]] - eps_so[pairs[x][0]] for x in keep])
    inv = np.linalg.inv((xpy @ xpy.T)[np.ix_(keep, keep)])
    v_eff = 0.5 * (inv @ d_a @ inv - d_a)
    local = {int(p): x for x, p in enumerate(act_so)}
    for x, px in enumerate(keep):
        i_so, a_so = pairs[px]
        for y, py in enumerate(keep):
            j_so, b_so = pairs[py]
            value = v_eff[x, y] - eri_so[i_so, a_so, j_so, b_so]
            i, a, j, b = (local[q] for q in (i_so, a_so, j_so, b_so))
            for t, u in ((i, a), (a, i)):
                for v, w in ((j, b), (b, j)):
                    vt[t, u, v, w] = value
    return spin_blocks(vt, m // 2)


def env_residues_uhf(eri_so, eps_so, occ_so, act_so):
    """(omega, w, c, env) of the full direct RPA of `drpa_so` for the
    environment self-energy of the active spin orbitals.

    w[n, t, r] = sum_ia (t r|i a) (X+Y)_ia,n, the sum over the pairs of both
    spins, t over act_so and r over `env`, every spin orbital outside act_so
    in ascending order; it is zero unless r has the spin of t.
    c[n, r] = eps_r - omega_n for an occupied r and eps_r + omega_n for an
    empty one, the pole of S^E.
    """
    _, omega, xpy, pairs = drpa_so(eri_so, eps_so, occ_so)
    act = [int(p) for p in act_so]
    occ = {int(i) for i in occ_so}
    env = [r for r in range(len(eps_so)) if r not in set(act)]
    v = np.zeros((len(act), len(env), len(pairs)))
    for x, t in enumerate(act):
        for y, r in enumerate(env):
            for z, (i, a) in enumerate(pairs):
                v[x, y, z] = eri_so[t, r, i, a]
    w = np.einsum('trP,Pn->ntr', v, xpy)
    sign = np.array([-1.0 if r in occ else 1.0 for r in env])
    c = np.asarray(eps_so)[env][None, :] + sign[None, :] * omega[:, None]
    return omega, w, c, env


def _sigma_env_from(z, w, c):
    """S^E_tu(z) = sum_rn w_tr w_ur / (z - c) from the residues and poles of
    `env_residues_uhf`."""
    n = w.shape[1]
    out = np.zeros((n, n), dtype=complex)
    for t in range(n):
        for u in range(n):
            out[t, u] = np.sum(w[:, t, :] * w[:, u, :] / (z - c))
    return out


def sigma_env_uhf(z, eri_so, eps_so, occ_so, act_so):
    """S^E_tu(z) = sum_{r in E} sum_n w_tr w_ur / (z - eps_r -+ omega_n) on
    the active spin orbitals (local indices, alpha first), at the complex
    frequency z; its alpha-beta blocks are zero."""
    _, w, c, _ = env_residues_uhf(eri_so, eps_so, occ_so, act_so)
    return _sigma_env_from(z, w, c)


def _spin_diagonal(sigma, n):
    """(alpha, beta) blocks of a one-body array over 2n local spin orbitals,
    whose alpha-beta blocks must vanish."""
    assert np.abs(sigma[:n, n:]).max(initial=0.0) < 1e-12
    assert np.abs(sigma[n:, :n]).max(initial=0.0) < 1e-12
    return sigma[:n, :n], sigma[n:, n:]


def sigma_mode_a_uhf(eri_so, eps_so, occ_so, act_so):
    """(alpha, beta): Eq. 12 of Marie and Loos at eta = 0 per spin,
    1/2 [S^E_tu(eps_t) + S^E_tu(eps_u)], with each spin orbital's own
    energy."""
    _, w, c, _ = env_residues_uhf(eri_so, eps_so, occ_so, act_so)
    act = [int(p) for p in act_so]
    m = len(act)
    out = np.zeros((m, m))
    for t in range(m):
        at_t = _sigma_env_from(eps_so[act[t]], w, c)
        for u in range(m):
            at_u = _sigma_env_from(eps_so[act[u]], w, c)
            out[t, u] = 0.5 * (at_t[t, u] + at_u[t, u]).real
    return _spin_diagonal(out, m // 2)


def sigma_srg_uhf(eri_so, eps_so, occ_so, act_so, s):
    """(alpha, beta): Eq. 44 of Marie and Loos per spin, on the environment
    spin orbitals r,

        Sigma_tu = sum_rn (D_tr + D_ur) / (D_tr^2 + D_ur^2) w_tr w_ur
                   [1 - exp(-(D_tr^2 + D_ur^2) s)],

    D_tr = eps_t - c_rn with the poles c of `env_residues_uhf`; s = None is
    s = inf (Eq. 42), the bracket 1."""
    _, w, c, _ = env_residues_uhf(eri_so, eps_so, occ_so, act_so)
    act = [int(p) for p in act_so]
    m = len(act)
    out = np.zeros((m, m))
    for t in range(m):
        for u in range(m):
            d_t, d_u = eps_so[act[t]] - c, eps_so[act[u]] - c
            lam = d_t**2 + d_u**2
            damp = 1.0 if s is None else 1.0 - np.exp(-lam * s)
            out[t, u] = np.sum((d_t + d_u) / lam * w[:, t, :] * w[:, u, :]
                               * damp)
    return _spin_diagonal(out, m // 2)
