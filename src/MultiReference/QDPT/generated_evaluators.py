"""QDPT active-space effective Hamiltonian g_eff (sectors 0 to max_vertex), spin-orbital.

Machine-generated from the Wick-contracted MP2/MP3 downfolding diagrams, for
max vertex rank 2 or 3. Do not edit by hand. Called from the QDPT driver,
perturbative.build_effective_hamiltonian.
"""
import numpy as np

def eval_pt2_v2(g_anti_spin, h1_spin, eps_spin, occ_idx, act_idx, virt_idx, act_occ_idx, act_vir_idx, n_spin_orbs, shift=None):
    import numpy as np
    g_eff = {n: np.zeros((len(act_idx),) * (2*n)) for n in range(1, 3)}
    g_eff[0] = 0.0
    _cache = {}
    _use_count = {
        'int_0': 1,
        'int_1': 1,
        'int_2': 1,
        'int_3': 1,
        'int_4': 1,
        'int_5': 1,
    }

    # Sector: 0-Body | Term: +1.00 * h1(i,i)
    b0 = h1_spin[np.ix_(occ_idx, occ_idx)]
    g_eff[0] += 1.0 * np.einsum('AA->', b0, optimize=True)

    # Sector: 0-Body | Term: +0.50 * g(j,i,j,i)
    b0 = g_anti_spin[np.ix_(occ_idx, occ_idx, occ_idx, occ_idx)]
    g_eff[0] += 0.5 * np.einsum('BABA->', b0, optimize=True)

    # Sector: 1-Body | Term: +1.00 * h1(p,q)
    b0 = h1_spin[np.ix_(act_idx, act_idx)]
    g_eff[1] += 1.0 * np.einsum('AB->AB', b0, optimize=True)

    # Sector: 1-Body | Term: -1.00 * g(i,p,q,i)
    b0 = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, occ_idx)]
    g_eff[1] += -1.0 * np.einsum('ABCA->BC', b0, optimize=True)

    # Sector: 2-Body | Term: +1.00 * g(q,p,s,r)
    b0 = g_anti_spin[np.ix_(act_idx, act_idx, act_idx, act_idx)]
    g_eff[2] += 1.0 * np.einsum('BADC->ABCD', b0, optimize=True)

    # --- Skeleton 94bc3a09 contains 1 permutations ---
    # Base Term: +0.250 * g(a,b,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_a - e_b)]
    g_temp = {0: np.zeros((len(act_idx),) * 0)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    if 'int_0' not in _cache:
        I1 = np.zeros(())
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            b0 = b0 / d0
            num = np.einsum('BDC,DCB->BCD', b0, b1, optimize=True)
            I1 += np.einsum('BCD->', num, optimize=True)
        _cache['int_0'] = I1
    I1 = _cache['int_0']
    g_temp[0] += 0.25 * np.einsum('->', I1, optimize=True)
    _use_count['int_0'] -= 1
    if _use_count['int_0'] == 0:
        del _cache['int_0']
    # Projecting permutation: +0.250 * g(a,b,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_a - e_b)]
    g_eff[0] += 1.000000000000 * g_temp[0]

    # --- Skeleton 75ad85ad contains 1 permutations ---
    # Base Term: -0.50 * g(a,p,j,i) * g(j,i,a,q) / [(E_corr + e_j + e_i - e_a - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0 = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_1' not in _cache:
        b0 = b0
        b1 = b1
        d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[act_idx][None, :, None, None]
        d0[np.abs(d0) < 1e-12] = 1e-12
        b0 = b0 / d0
        num = np.einsum('ADCB,CBAE->ABCDE', b0, b1, optimize=True)
        I1 = np.einsum('ABCDE->DE', num, optimize=True)
        _cache['int_1'] = I1
    I1 = _cache['int_1']
    g_temp[1][:, :] += -0.5 * np.einsum('DE->DE', I1, optimize=True)
    _use_count['int_1'] -= 1
    if _use_count['int_1'] == 0:
        del _cache['int_1']
    # Projecting permutation: -0.50 * g(a,p,j,i) * g(j,i,a,q) / [(E_corr + e_j + e_i - e_a - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 3421b2e9 contains 1 permutations ---
    # Base Term: -0.50 * g(a,b,q,i) * g(i,p,a,b) / [(E_corr + e_q + e_i - e_a - e_b)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_2' not in _cache:
        I1 = np.zeros((len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            b0 = b0 / d0
            num = np.einsum('BEC,CDB->BCDE', b0, b1, optimize=True)
            I1 += np.einsum('BCDE->DE', num, optimize=True)
        _cache['int_2'] = I1
    I1 = _cache['int_2']
    g_temp[1][:, :] += -0.5 * np.einsum('DE->DE', I1, optimize=True)
    _use_count['int_2'] -= 1
    if _use_count['int_2'] == 0:
        del _cache['int_2']
    # Projecting permutation: -0.50 * g(a,b,q,i) * g(i,p,a,b) / [(E_corr + e_q + e_i - e_a - e_b)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton b2162412 contains 1 permutations ---
    # Base Term: +0.50 * g(j,i,s,r) * g(q,p,j,i) / [(E_corr + e_j + e_i - e_q - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, act_idx)]
    b1 = g_anti_spin[np.ix_(act_idx, act_idx, occ_idx, occ_idx)]
    if 'int_3' not in _cache:
        b0 = b0
        b1 = b1
        num = np.einsum('BAFE,DCBA->ABCDEF', b0, b1, optimize=True)
        d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[act_idx][None, None, :, None, None, None]
        d0[np.abs(d0) < 1e-12] = 1e-12
        num = num / d0
        I1 = np.einsum('ABCDEF->CDEF', num, optimize=True)
        _cache['int_3'] = I1
    I1 = _cache['int_3']
    g_temp[2][:, :, :, :] += 0.5 * np.einsum('CDEF->CDEF', I1, optimize=True)
    _use_count['int_3'] -= 1
    if _use_count['int_3'] == 0:
        del _cache['int_3']
    # Projecting permutation: +0.50 * g(j,i,s,r) * g(q,p,j,i) / [(E_corr + e_j + e_i - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton fb0c734b contains 4 permutations ---
    # Base Term: +1.00 * g(a,p,s,i) * g(i,q,a,r) / [(E_corr + e_s + e_i - e_a - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0 = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_4' not in _cache:
        b0 = b0
        b1 = b1
        d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[act_idx][None, :, None, None]
        d0[np.abs(d0) < 1e-12] = 1e-12
        b0 = b0 / d0
        num = np.einsum('ACFB,BDAE->ABCDEF', b0, b1, optimize=True)
        I1 = np.einsum('ABCDEF->CDEF', num, optimize=True)
        _cache['int_4'] = I1
    I1 = _cache['int_4']
    g_temp[2][:, :, :, :] += 1.0 * np.einsum('CDEF->CDEF', I1, optimize=True)
    _use_count['int_4'] -= 1
    if _use_count['int_4'] == 0:
        del _cache['int_4']
    # Projecting permutation: +1.00 * g(a,p,s,i) * g(i,q,a,r) / [(E_corr + e_s + e_i - e_a - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -1.00 * g(a,p,r,i) * g(i,q,a,s) / [(E_corr + e_r + e_i - e_a - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: -1.00 * g(a,q,s,i) * g(i,p,a,r) / [(E_corr + e_s + e_i - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: +1.00 * g(a,q,r,i) * g(i,p,a,s) / [(E_corr + e_r + e_i - e_a - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton d6997bae contains 1 permutations ---
    # Base Term: +0.50 * g(a,b,s,r) * g(q,p,a,b) / [(E_corr + e_s + e_r - e_a - e_b)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, virt_idx)]
    if 'int_5' not in _cache:
        I1 = np.zeros((len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            b0 = b0 / d0
            num = np.einsum('BFE,DCB->BCDEF', b0, b1, optimize=True)
            I1 += np.einsum('BCDEF->CDEF', num, optimize=True)
        _cache['int_5'] = I1
    I1 = _cache['int_5']
    g_temp[2][:, :, :, :] += 0.5 * np.einsum('CDEF->CDEF', I1, optimize=True)
    _use_count['int_5'] -= 1
    if _use_count['int_5'] == 0:
        del _cache['int_5']
    # Projecting permutation: +0.50 * g(a,b,s,r) * g(q,p,a,b) / [(E_corr + e_s + e_r - e_a - e_b)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    return g_eff

def eval_pt2_v3(g_anti_spin, h1_spin, eps_spin, occ_idx, act_idx, virt_idx, act_occ_idx, act_vir_idx, n_spin_orbs, shift=None):
    import numpy as np
    g_eff = {n: np.zeros((len(act_idx),) * (2*n)) for n in range(1, 4)}
    g_eff[0] = 0.0
    _cache = {}
    _use_count = {
        'int_0': 1,
        'int_1': 1,
        'int_2': 1,
        'int_3': 1,
        'int_4': 1,
        'int_5': 1,
        'int_6': 1,
        'int_7': 1,
    }

    # Sector: 0-Body | Term: +1.00 * h1(i,i)
    b0 = h1_spin[np.ix_(occ_idx, occ_idx)]
    g_eff[0] += 1.0 * np.einsum('AA->', b0, optimize=True)

    # Sector: 0-Body | Term: +0.50 * g(j,i,j,i)
    b0 = g_anti_spin[np.ix_(occ_idx, occ_idx, occ_idx, occ_idx)]
    g_eff[0] += 0.5 * np.einsum('BABA->', b0, optimize=True)

    # Sector: 1-Body | Term: +1.00 * h1(p,q)
    b0 = h1_spin[np.ix_(act_idx, act_idx)]
    g_eff[1] += 1.0 * np.einsum('AB->AB', b0, optimize=True)

    # Sector: 1-Body | Term: -1.00 * g(i,p,q,i)
    b0 = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, occ_idx)]
    g_eff[1] += -1.0 * np.einsum('ABCA->BC', b0, optimize=True)

    # Sector: 2-Body | Term: +1.00 * g(q,p,s,r)
    b0 = g_anti_spin[np.ix_(act_idx, act_idx, act_idx, act_idx)]
    g_eff[2] += 1.0 * np.einsum('BADC->ABCD', b0, optimize=True)

    # --- Skeleton 94bc3a09 contains 1 permutations ---
    # Base Term: +0.250 * g(a,b,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_a - e_b)]
    g_temp = {0: np.zeros((len(act_idx),) * 0)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    if 'int_0' not in _cache:
        I1 = np.zeros(())
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            b0 = b0 / d0
            num = np.einsum('BDC,DCB->BCD', b0, b1, optimize=True)
            I1 += np.einsum('BCD->', num, optimize=True)
        _cache['int_0'] = I1
    I1 = _cache['int_0']
    g_temp[0] += 0.25 * np.einsum('->', I1, optimize=True)
    _use_count['int_0'] -= 1
    if _use_count['int_0'] == 0:
        del _cache['int_0']
    # Projecting permutation: +0.250 * g(a,b,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_a - e_b)]
    g_eff[0] += 1.000000000000 * g_temp[0]

    # --- Skeleton 75ad85ad contains 1 permutations ---
    # Base Term: -0.50 * g(a,p,j,i) * g(j,i,a,q) / [(E_corr + e_j + e_i - e_a - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0 = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_1' not in _cache:
        b0 = b0
        b1 = b1
        d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[act_idx][None, :, None, None]
        d0[np.abs(d0) < 1e-12] = 1e-12
        b0 = b0 / d0
        num = np.einsum('ADCB,CBAE->ABCDE', b0, b1, optimize=True)
        I1 = np.einsum('ABCDE->DE', num, optimize=True)
        _cache['int_1'] = I1
    I1 = _cache['int_1']
    g_temp[1][:, :] += -0.5 * np.einsum('DE->DE', I1, optimize=True)
    _use_count['int_1'] -= 1
    if _use_count['int_1'] == 0:
        del _cache['int_1']
    # Projecting permutation: -0.50 * g(a,p,j,i) * g(j,i,a,q) / [(E_corr + e_j + e_i - e_a - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 3421b2e9 contains 1 permutations ---
    # Base Term: -0.50 * g(a,b,q,i) * g(i,p,a,b) / [(E_corr + e_q + e_i - e_a - e_b)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_2' not in _cache:
        I1 = np.zeros((len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            b0 = b0 / d0
            num = np.einsum('BEC,CDB->BCDE', b0, b1, optimize=True)
            I1 += np.einsum('BCDE->DE', num, optimize=True)
        _cache['int_2'] = I1
    I1 = _cache['int_2']
    g_temp[1][:, :] += -0.5 * np.einsum('DE->DE', I1, optimize=True)
    _use_count['int_2'] -= 1
    if _use_count['int_2'] == 0:
        del _cache['int_2']
    # Projecting permutation: -0.50 * g(a,b,q,i) * g(i,p,a,b) / [(E_corr + e_q + e_i - e_a - e_b)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton b2162412 contains 1 permutations ---
    # Base Term: +0.50 * g(j,i,s,r) * g(q,p,j,i) / [(E_corr + e_j + e_i - e_q - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, act_idx)]
    b1 = g_anti_spin[np.ix_(act_idx, act_idx, occ_idx, occ_idx)]
    if 'int_3' not in _cache:
        b0 = b0
        b1 = b1
        num = np.einsum('BAFE,DCBA->ABCDEF', b0, b1, optimize=True)
        d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[act_idx][None, None, :, None, None, None]
        d0[np.abs(d0) < 1e-12] = 1e-12
        num = num / d0
        I1 = np.einsum('ABCDEF->CDEF', num, optimize=True)
        _cache['int_3'] = I1
    I1 = _cache['int_3']
    g_temp[2][:, :, :, :] += 0.5 * np.einsum('CDEF->CDEF', I1, optimize=True)
    _use_count['int_3'] -= 1
    if _use_count['int_3'] == 0:
        del _cache['int_3']
    # Projecting permutation: +0.50 * g(j,i,s,r) * g(q,p,j,i) / [(E_corr + e_j + e_i - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton fb0c734b contains 4 permutations ---
    # Base Term: +1.00 * g(a,p,s,i) * g(i,q,a,r) / [(E_corr + e_s + e_i - e_a - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0 = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_4' not in _cache:
        b0 = b0
        b1 = b1
        d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[act_idx][None, :, None, None]
        d0[np.abs(d0) < 1e-12] = 1e-12
        b0 = b0 / d0
        num = np.einsum('ACFB,BDAE->ABCDEF', b0, b1, optimize=True)
        I1 = np.einsum('ABCDEF->CDEF', num, optimize=True)
        _cache['int_4'] = I1
    I1 = _cache['int_4']
    g_temp[2][:, :, :, :] += 1.0 * np.einsum('CDEF->CDEF', I1, optimize=True)
    _use_count['int_4'] -= 1
    if _use_count['int_4'] == 0:
        del _cache['int_4']
    # Projecting permutation: +1.00 * g(a,p,s,i) * g(i,q,a,r) / [(E_corr + e_s + e_i - e_a - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -1.00 * g(a,p,r,i) * g(i,q,a,s) / [(E_corr + e_r + e_i - e_a - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: -1.00 * g(a,q,s,i) * g(i,p,a,r) / [(E_corr + e_s + e_i - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: +1.00 * g(a,q,r,i) * g(i,p,a,s) / [(E_corr + e_r + e_i - e_a - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton d6997bae contains 1 permutations ---
    # Base Term: +0.50 * g(a,b,s,r) * g(q,p,a,b) / [(E_corr + e_s + e_r - e_a - e_b)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, virt_idx)]
    if 'int_5' not in _cache:
        I1 = np.zeros((len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            b0 = b0 / d0
            num = np.einsum('BFE,DCB->BCDEF', b0, b1, optimize=True)
            I1 += np.einsum('BCDEF->CDEF', num, optimize=True)
        _cache['int_5'] = I1
    I1 = _cache['int_5']
    g_temp[2][:, :, :, :] += 0.5 * np.einsum('CDEF->CDEF', I1, optimize=True)
    _use_count['int_5'] -= 1
    if _use_count['int_5'] == 0:
        del _cache['int_5']
    # Projecting permutation: +0.50 * g(a,b,s,r) * g(q,p,a,b) / [(E_corr + e_s + e_r - e_a - e_b)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton 9e1ea51e contains 9 permutations ---
    # Base Term: -1.00 * g(i,r,t,s) * g(q,p,u,i) / [(E_corr + e_u + e_i - e_q - e_p)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, act_idx, occ_idx)]
    if 'int_6' not in _cache:
        I1 = np.zeros((len(act_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_i in range(len(occ_idx)):
            b0 = b0_full[i_i, :, :, :]
            b1 = b1_full[:, :, :, i_i]
            num = np.einsum('DFE,CBG->BCDEFG', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx[i_i]] + (shift / 2 if shift is not None else 0.0)) - eps_spin[act_idx][None, :, None, None, None, None] - eps_spin[act_idx][:, None, None, None, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            I1 += num
        _cache['int_6'] = I1
    I1 = _cache['int_6']
    g_temp[3][:, :, :, :, :, :] += -1.0 * np.einsum('BCDEFG->BCDEFG', I1, optimize=True)
    _use_count['int_6'] -= 1
    if _use_count['int_6'] == 0:
        del _cache['int_6']
    # Projecting permutation: -1.00 * g(i,r,t,s) * g(q,p,u,i) / [(E_corr + e_u + e_i - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(i,r,u,s) * g(q,p,t,i) / [(E_corr + e_t + e_i - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 5, 4))
    # Projecting permutation: -1.00 * g(i,r,u,t) * g(q,p,s,i) / [(E_corr + e_s + e_i - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 3, 4))
    # Projecting permutation: +1.00 * g(i,q,t,s) * g(r,p,u,i) / [(E_corr + e_u + e_i - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 4, 5))
    # Projecting permutation: -1.00 * g(i,q,u,s) * g(r,p,t,i) / [(E_corr + e_t + e_i - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 5, 4))
    # Projecting permutation: +1.00 * g(i,q,u,t) * g(r,p,s,i) / [(E_corr + e_s + e_i - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 5, 3, 4))
    # Projecting permutation: -1.00 * g(i,p,t,s) * g(r,q,u,i) / [(E_corr + e_u + e_i - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 4, 5))
    # Projecting permutation: +1.00 * g(i,p,u,s) * g(r,q,t,i) / [(E_corr + e_t + e_i - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 5, 4))
    # Projecting permutation: -1.00 * g(i,p,u,t) * g(r,q,s,i) / [(E_corr + e_s + e_i - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 5, 3, 4))

    # --- Skeleton d29ea506 contains 9 permutations ---
    # Base Term: -1.00 * g(a,p,u,t) * g(r,q,a,s) / [(E_corr + e_u + e_t - e_a - e_p)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, act_idx)]
    if 'int_7' not in _cache:
        I1 = np.zeros((len(act_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            b0 = b0 / d0
            num = np.einsum('BGF,DCE->BCDEFG', b0, b1, optimize=True)
            I1 += num
        _cache['int_7'] = I1
    I1 = _cache['int_7']
    g_temp[3][:, :, :, :, :, :] += -1.0 * np.einsum('BCDEFG->BCDEFG', I1, optimize=True)
    _use_count['int_7'] -= 1
    if _use_count['int_7'] == 0:
        del _cache['int_7']
    # Projecting permutation: -1.00 * g(a,p,u,t) * g(r,q,a,s) / [(E_corr + e_u + e_t - e_a - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,p,u,s) * g(r,q,a,t) / [(E_corr + e_u + e_s - e_a - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,p,t,s) * g(r,q,a,u) / [(E_corr + e_t + e_s - e_a - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: +1.00 * g(a,q,u,t) * g(r,p,a,s) / [(E_corr + e_u + e_t - e_a - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,q,u,s) * g(r,p,a,t) / [(E_corr + e_u + e_s - e_a - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 3, 5))
    # Projecting permutation: +1.00 * g(a,q,t,s) * g(r,p,a,u) / [(E_corr + e_t + e_s - e_a - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 5, 3))
    # Projecting permutation: -1.00 * g(a,r,u,t) * g(q,p,a,s) / [(E_corr + e_u + e_t - e_a - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,r,u,s) * g(q,p,a,t) / [(E_corr + e_u + e_s - e_a - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,r,t,s) * g(q,p,a,u) / [(E_corr + e_t + e_s - e_a - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 5, 3))
    return g_eff

def eval_pt3_v2(g_anti_spin, h1_spin, eps_spin, occ_idx, act_idx, virt_idx, act_occ_idx, act_vir_idx, n_spin_orbs, shift=None):
    import numpy as np
    g_eff = {n: np.zeros((len(act_idx),) * (2*n)) for n in range(1, 3)}
    g_eff[0] = 0.0
    _cache = {}
    _use_count = {
        'int_0': 1,
        'int_1': 1,
        'int_2': 1,
        'int_3': 1,
        'int_4': 1,
        'int_5': 1,
        'int_6': 1,
        'int_7': 1,
        'int_8': 1,
        'int_9': 1,
        'int_10': 1,
        'int_11': 1,
        'int_12': 1,
        'int_13': 1,
        'int_14': 1,
        'int_15': 1,
        'int_16': 1,
        'int_17': 1,
        'int_18': 1,
        'int_19': 1,
        'int_20': 1,
        'int_21': 1,
        'int_22': 1,
        'int_23': 1,
        'int_24': 1,
        'int_25': 1,
        'int_26': 1,
        'int_27': 1,
        'int_28': 1,
        'int_29': 1,
        'int_30': 1,
        'int_31': 1,
        'int_32': 1,
        'int_33': 1,
        'int_34': 1,
        'int_35': 1,
        'int_36': 1,
        'int_37': 1,
        'int_38': 1,
        'int_39': 1,
        'int_40': 1,
        'int_41': 1,
        'int_42': 1,
        'int_43': 1,
        'int_44': 1,
        'int_45': 1,
        'int_46': 1,
        'int_47': 1,
        'int_48': 1,
        'int_49': 1,
        'int_50': 1,
        'int_51': 1,
        'int_52': 1,
        'int_53': 1,
        'int_54': 1,
        'int_55': 1,
        'int_56': 1,
        'int_57': 1,
        'int_58': 1,
        'int_59': 1,
        'int_60': 1,
    }

    # Sector: 0-Body | Term: +1.00 * h1(i,i)
    b0 = h1_spin[np.ix_(occ_idx, occ_idx)]
    g_eff[0] += 1.0 * np.einsum('AA->', b0, optimize=True)

    # Sector: 0-Body | Term: +0.50 * g(j,i,j,i)
    b0 = g_anti_spin[np.ix_(occ_idx, occ_idx, occ_idx, occ_idx)]
    g_eff[0] += 0.5 * np.einsum('BABA->', b0, optimize=True)

    # Sector: 1-Body | Term: +1.00 * h1(p,q)
    b0 = h1_spin[np.ix_(act_idx, act_idx)]
    g_eff[1] += 1.0 * np.einsum('AB->AB', b0, optimize=True)

    # Sector: 1-Body | Term: -1.00 * g(i,p,q,i)
    b0 = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, occ_idx)]
    g_eff[1] += -1.0 * np.einsum('ABCA->BC', b0, optimize=True)

    # Sector: 2-Body | Term: +1.00 * g(q,p,s,r)
    b0 = g_anti_spin[np.ix_(act_idx, act_idx, act_idx, act_idx)]
    g_eff[2] += 1.0 * np.einsum('BADC->ABCD', b0, optimize=True)

    # --- Skeleton 94bc3a09 contains 1 permutations ---
    # Base Term: +0.250 * g(a,b,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_a - e_b)]
    g_temp = {0: np.zeros((len(act_idx),) * 0)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    if 'int_0' not in _cache:
        I1 = np.zeros(())
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            b0 = b0 / d0
            num = np.einsum('BDC,DCB->BCD', b0, b1, optimize=True)
            I1 += np.einsum('BCD->', num, optimize=True)
        _cache['int_0'] = I1
    I1 = _cache['int_0']
    g_temp[0] += 0.25 * np.einsum('->', I1, optimize=True)
    _use_count['int_0'] -= 1
    if _use_count['int_0'] == 0:
        del _cache['int_0']
    # Projecting permutation: +0.250 * g(a,b,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_a - e_b)]
    g_eff[0] += 1.000000000000 * g_temp[0]

    # --- Skeleton b3d347c4 contains 1 permutations ---
    # Base Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    g_temp = {0: np.zeros((len(act_idx),) * 0)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, occ_idx, occ_idx)]
    if 'int_1' not in _cache:
        I1 = np.zeros((len(occ_idx), len(occ_idx), len(occ_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BFE,DCB->BCDEF', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None, None, None]
            num = num / d0
            I1 += np.einsum('BCDEF->CDEF', num, optimize=True)
        _cache['int_1'] = I1
    I1 = _cache['int_1']
    b2 = b2_full
    num = np.einsum('CDEF,FEDC->CDEF', I1, b2, optimize=True)
    I2 = np.einsum('CDEF->', num, optimize=True)
    g_temp[0] += 0.125 * np.einsum('->', I2, optimize=True)
    _use_count['int_1'] -= 1
    if _use_count['int_1'] == 0:
        del _cache['int_1']
    # Projecting permutation: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    g_eff[0] += 1.000000000000 * g_temp[0]

    # --- Skeleton ccea8972 contains 1 permutations ---
    # Base Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    g_temp = {0: np.zeros((len(act_idx),) * 0)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    if 'int_2' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[:, i_a, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('ED,CFE->CDEF', b0, b1, optimize=True)
                d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx][:, None, None, None] - eps_spin[virt_idx[i_a]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :] += np.einsum('CDEF->CDF', num, optimize=True)
        _cache['int_2'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_2'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDF,FDC->CDF', I1, b2, optimize=True)
        I2 = np.einsum('CDF->', num, optimize=True)
        g_temp[0] += 1.0 * np.einsum('->', I2, optimize=True)
    _use_count['int_2'] -= 1
    if _use_count['int_2'] == 0:
        del _cache['int_2']
    # Projecting permutation: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    g_eff[0] += 1.000000000000 * g_temp[0]

    # --- Skeleton fa4ffacc contains 1 permutations ---
    # Base Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    g_temp = {0: np.zeros((len(act_idx),) * 0)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, virt_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    if 'int_3' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                for i_c in range(len(virt_idx)):
                    b0 = b0_full[i_a, i_b, i_c, :]
                    b1 = b1_full[i_c, :, :, :]
                    num = np.einsum('D,DFE->DEF', b0, b1, optimize=True)
                    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] - eps_spin[virt_idx[i_c]] - eps_spin[virt_idx][:, None, None]
                    d0[np.abs(d0) < 1e-12] = 1e-12
                    num = num / d0
                    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                    d1[np.abs(d1) < 1e-12] = 1e-12
                    num = num / d1
                    I1[i_a, i_b, :, :] += np.einsum('DEF->EF', num, optimize=True)
        _cache['int_3'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_3'][i_a, i_b, :, :]
            b2 = b2_full[:, :, i_a, i_b]
            num = np.einsum('EF,FE->EF', I1, b2, optimize=True)
            I2 = np.einsum('EF->', num, optimize=True)
            g_temp[0] += 0.125 * np.einsum('->', I2, optimize=True)
    _use_count['int_3'] -= 1
    if _use_count['int_3'] == 0:
        del _cache['int_3']
    # Projecting permutation: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    g_eff[0] += 1.000000000000 * g_temp[0]

    # --- Skeleton 75ad85ad contains 1 permutations ---
    # Base Term: -0.50 * g(a,p,j,i) * g(j,i,a,q) / [(E_corr + e_j + e_i - e_a - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0 = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_4' not in _cache:
        b0 = b0
        b1 = b1
        d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[act_idx][None, :, None, None]
        d0[np.abs(d0) < 1e-12] = 1e-12
        b0 = b0 / d0
        num = np.einsum('ADCB,CBAE->ABCDE', b0, b1, optimize=True)
        I1 = np.einsum('ABCDE->DE', num, optimize=True)
        _cache['int_4'] = I1
    I1 = _cache['int_4']
    g_temp[1][:, :] += -0.5 * np.einsum('DE->DE', I1, optimize=True)
    _use_count['int_4'] -= 1
    if _use_count['int_4'] == 0:
        del _cache['int_4']
    # Projecting permutation: -0.50 * g(a,p,j,i) * g(j,i,a,q) / [(E_corr + e_j + e_i - e_a - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 3421b2e9 contains 1 permutations ---
    # Base Term: -0.50 * g(a,b,q,i) * g(i,p,a,b) / [(E_corr + e_q + e_i - e_a - e_b)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_5' not in _cache:
        I1 = np.zeros((len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            b0 = b0 / d0
            num = np.einsum('BEC,CDB->BCDE', b0, b1, optimize=True)
            I1 += np.einsum('BCDE->DE', num, optimize=True)
        _cache['int_5'] = I1
    I1 = _cache['int_5']
    g_temp[1][:, :] += -0.5 * np.einsum('DE->DE', I1, optimize=True)
    _use_count['int_5'] -= 1
    if _use_count['int_5'] == 0:
        del _cache['int_5']
    # Projecting permutation: -0.50 * g(a,b,q,i) * g(i,p,a,b) / [(E_corr + e_q + e_i - e_a - e_b)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton dfca8151 contains 1 permutations ---
    # Base Term: -0.250 * g(a,p,l,k) * g(j,i,a,q) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_p)] * [(E_corr + e_j + e_i - e_a - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0 = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_6' not in _cache:
        b0 = b0
        b1 = b1
        d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[act_idx][None, :, None, None]
        d1[np.abs(d1) < 1e-12] = 1e-12
        b0 = b0 / d1
        num = np.einsum('AFED,EDCB->ABCDEF', b0, b1, optimize=True)
        d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, None, :]
        num = num / d0
        I1 = np.einsum('ABCDEF->ABCF', num, optimize=True)
        _cache['int_6'] = I1
    I1 = _cache['int_6']
    num = np.einsum('ABCF,CBAG->ABCFG', I1, b2, optimize=True)
    I2 = np.einsum('ABCFG->FG', num, optimize=True)
    g_temp[1][:, :] += -0.25 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_6'] -= 1
    if _use_count['int_6'] == 0:
        del _cache['int_6']
    # Projecting permutation: -0.250 * g(a,p,l,k) * g(j,i,a,q) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_p)] * [(E_corr + e_j + e_i - e_a - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton ff24b53d contains 1 permutations ---
    # Base Term: -0.50 * g(k,a,j,i) * g(b,p,q,k) * g(j,i,a,b) / [(E_corr + e_q + e_k - e_b - e_p)] * [(E_corr + e_q + e_j + e_i - e_b - e_p - e_a)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    if 'int_7' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, i_a, :]
            num = np.einsum('EDC,DCB->BCDE', b0, b1, optimize=True)
            I1[i_a, :, :, :, :] += num
        _cache['int_7'] = I1
    b2 = b2_full
    d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_7'][i_a, :, :, :, :]
        num = np.einsum('BCDE,BFGE->BCDEFG', I1, b2, optimize=True)
        num = num / d0
        d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[virt_idx[i_a]]
        d1[np.abs(d1) < 1e-12] = 1e-12
        num = num / d1
        I2 = np.einsum('BCDEFG->FG', num, optimize=True)
        g_temp[1][:, :] += -0.5 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_7'] -= 1
    if _use_count['int_7'] == 0:
        del _cache['int_7']
    # Projecting permutation: -0.50 * g(k,a,j,i) * g(b,p,q,k) * g(j,i,a,b) / [(E_corr + e_q + e_k - e_b - e_p)] * [(E_corr + e_q + e_j + e_i - e_b - e_p - e_a)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 0ab601c6 contains 1 permutations ---
    # Base Term: -0.250 * g(a,b,q,k) * g(k,p,j,i) * g(j,i,a,b) / [(E_corr + e_q + e_k - e_a - e_b)] * [(E_corr + e_q + e_j + e_i - e_a - e_b - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    if 'int_8' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BGE,EFDC->BCDEFG', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None]
            num = num / d0
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEFG->BCDFG', num, optimize=True)
        _cache['int_8'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_8'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BCDFG,DCB->BCDFG', I1, b2, optimize=True)
        I2 = np.einsum('BCDFG->FG', num, optimize=True)
        g_temp[1][:, :] += -0.25 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_8'] -= 1
    if _use_count['int_8'] == 0:
        del _cache['int_8']
    # Projecting permutation: -0.250 * g(a,b,q,k) * g(k,p,j,i) * g(j,i,a,b) / [(E_corr + e_q + e_k - e_a - e_b)] * [(E_corr + e_q + e_j + e_i - e_a - e_b - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 1d17f101 contains 1 permutations ---
    # Base Term: -1.00 * g(j,a,b,i) * g(b,p,k,j) * g(k,i,a,q) / [(E_corr + e_k + e_j - e_b - e_p)] * [(E_corr + e_k + e_i - e_p - e_a)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_9' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('DBC,BFED->BCDEF', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[act_idx][None, None, None, None, :] - eps_spin[virt_idx[i_a]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :] += np.einsum('BCDEF->CEF', num, optimize=True)
        _cache['int_9'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_9'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CEF,ECG->CEFG', I1, b2, optimize=True)
        I2 = np.einsum('CEFG->FG', num, optimize=True)
        g_temp[1][:, :] += -1.0 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_9'] -= 1
    if _use_count['int_9'] == 0:
        del _cache['int_9']
    # Projecting permutation: -1.00 * g(j,a,b,i) * g(b,p,k,j) * g(k,i,a,q) / [(E_corr + e_k + e_j - e_b - e_p)] * [(E_corr + e_k + e_i - e_p - e_a)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton d70ea4ec contains 1 permutations ---
    # Base Term: +1.00 * g(j,a,q,i) * g(b,p,k,j) * g(k,i,a,b) / [(E_corr + e_k + e_j - e_b - e_p)] * [(E_corr + e_k + e_q + e_i - e_b - e_p - e_a)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    if 'int_10' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('DGC,BFED->BCDEFG', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[virt_idx[i_a]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEFG->BCEFG', num, optimize=True)
        _cache['int_10'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_10'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BCEFG,ECB->BCEFG', I1, b2, optimize=True)
        I2 = np.einsum('BCEFG->FG', num, optimize=True)
        g_temp[1][:, :] += 1.0 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_10'] -= 1
    if _use_count['int_10'] == 0:
        del _cache['int_10']
    # Projecting permutation: +1.00 * g(j,a,q,i) * g(b,p,k,j) * g(k,i,a,b) / [(E_corr + e_k + e_j - e_b - e_p)] * [(E_corr + e_k + e_q + e_i - e_b - e_p - e_a)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton f95c0dde contains 1 permutations ---
    # Base Term: +1.00 * g(a,b,k,j) * g(j,p,a,i) * g(k,i,b,q) / [(E_corr + e_k + e_j - e_a - e_b)] * [(E_corr + e_k + e_i - e_b - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_11' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BED,DFC->BCDEF', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :]
            num = num / d0
            I1 += np.einsum('BCDEF->BCEF', num, optimize=True)
        _cache['int_11'] = I1
    I1 = _cache['int_11']
    b2 = b2_full
    num = np.einsum('BCEF,ECBG->BCEFG', I1, b2, optimize=True)
    I2 = np.einsum('BCEFG->FG', num, optimize=True)
    g_temp[1][:, :] += 1.0 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_11'] -= 1
    if _use_count['int_11'] == 0:
        del _cache['int_11']
    # Projecting permutation: +1.00 * g(a,b,k,j) * g(j,p,a,i) * g(k,i,b,q) / [(E_corr + e_k + e_j - e_a - e_b)] * [(E_corr + e_k + e_i - e_b - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton eb8ae6c7 contains 1 permutations ---
    # Base Term: +0.50 * g(a,b,k,j) * g(j,p,q,i) * g(k,i,a,b) / [(E_corr + e_k + e_j - e_a - e_b)] * [(E_corr + e_k + e_q + e_i - e_a - e_b - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, occ_idx)]
    if 'int_12' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BED,ECB->BCDE', b0, b1, optimize=True)
            I1[i_a, :, :, :, :] += num
        _cache['int_12'] = I1
    b2 = b2_full
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_12'][i_a, :, :, :, :]
        num = np.einsum('BCDE,DFGC->BCDEFG', I1, b2, optimize=True)
        d0 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None]
        d0[np.abs(d0) < 1e-12] = 1e-12
        num = num / d0
        I2 = np.einsum('BCDEFG->FG', num, optimize=True)
        g_temp[1][:, :] += 0.5 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_12'] -= 1
    if _use_count['int_12'] == 0:
        del _cache['int_12']
    # Projecting permutation: +0.50 * g(a,b,k,j) * g(j,p,q,i) * g(k,i,a,b) / [(E_corr + e_k + e_j - e_a - e_b)] * [(E_corr + e_k + e_q + e_i - e_a - e_b - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 05800a39 contains 1 permutations ---
    # Base Term: -0.50 * g(a,b,c,i) * g(c,p,q,j) * g(j,i,a,b) / [(E_corr + e_q + e_j - e_c - e_p)] * [(E_corr + e_q + e_j + e_i - e_p - e_a - e_b)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    if 'int_13' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, i_b, :, :]
                b1 = b1_full[:, :, i_a, i_b]
                num = np.einsum('CD,ED->CDE', b0, b1, optimize=True)
                I1[i_a, i_b, :, :, :] += num
        _cache['int_13'] = I1
    b2 = b2_full
    d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_13'][i_a, i_b, :, :, :]
            num = np.einsum('CDE,CFGE->CDEFG', I1, b2, optimize=True)
            num = num / d0
            d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[act_idx][None, None, None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I2 = np.einsum('CDEFG->FG', num, optimize=True)
            g_temp[1][:, :] += -0.5 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_13'] -= 1
    if _use_count['int_13'] == 0:
        del _cache['int_13']
    # Projecting permutation: -0.50 * g(a,b,c,i) * g(c,p,q,j) * g(j,i,a,b) / [(E_corr + e_q + e_j - e_c - e_p)] * [(E_corr + e_q + e_j + e_i - e_p - e_a - e_b)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 3d7c4c8a contains 1 permutations ---
    # Base Term: +1.00 * g(a,p,b,i) * g(b,c,q,j) * g(j,i,a,c) / [(E_corr + e_q + e_j - e_b - e_c)] * [(E_corr + e_q + e_j + e_i - e_c - e_a - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    if 'int_14' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('FD,CGE->CDEFG', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, :, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :, :, :] += num
        _cache['int_14'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_14'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDEFG,EDC->CDEFG', I1, b2, optimize=True)
        I2 = np.einsum('CDEFG->FG', num, optimize=True)
        g_temp[1][:, :] += 1.0 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_14'] -= 1
    if _use_count['int_14'] == 0:
        del _cache['int_14']
    # Projecting permutation: +1.00 * g(a,p,b,i) * g(b,c,q,j) * g(j,i,a,c) / [(E_corr + e_q + e_j - e_b - e_c)] * [(E_corr + e_q + e_j + e_i - e_c - e_a - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 3c92a970 contains 1 permutations ---
    # Base Term: -0.250 * g(a,b,c,q) * g(c,p,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_p)] * [(E_corr + e_j + e_i + e_q - e_p - e_a - e_b)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    if 'int_15' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, i_b, :, :]
                b1 = b1_full[:, :, :, :]
                num = np.einsum('CG,CFED->CDEFG', b0, b1, optimize=True)
                d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] - eps_spin[act_idx][None, None, None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, i_b, :, :, :, :] += np.einsum('CDEFG->DEFG', num, optimize=True)
        _cache['int_15'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_15'][i_a, i_b, :, :, :, :]
            b2 = b2_full[:, :, i_a, i_b]
            num = np.einsum('DEFG,ED->DEFG', I1, b2, optimize=True)
            I2 = np.einsum('DEFG->FG', num, optimize=True)
            g_temp[1][:, :] += -0.25 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_15'] -= 1
    if _use_count['int_15'] == 0:
        del _cache['int_15']
    # Projecting permutation: -0.250 * g(a,b,c,q) * g(c,p,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_p)] * [(E_corr + e_j + e_i + e_q - e_p - e_a - e_b)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton d8e26961 contains 1 permutations ---
    # Base Term: -0.250 * g(a,p,b,c) * g(b,c,j,i) * g(j,i,a,q) / [(E_corr + e_j + e_i - e_b - e_c)] * [(E_corr + e_j + e_i - e_a - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, virt_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_16' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('FC,CED->CDEF', b0, b1, optimize=True)
                d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, :]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :] += np.einsum('CDEF->DEF', num, optimize=True)
        _cache['int_16'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_16'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('DEF,EDG->DEFG', I1, b2, optimize=True)
        I2 = np.einsum('DEFG->FG', num, optimize=True)
        g_temp[1][:, :] += -0.25 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_16'] -= 1
    if _use_count['int_16'] == 0:
        del _cache['int_16']
    # Projecting permutation: -0.250 * g(a,p,b,c) * g(b,c,j,i) * g(j,i,a,q) / [(E_corr + e_j + e_i - e_b - e_c)] * [(E_corr + e_j + e_i - e_a - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 24b668bc contains 1 permutations ---
    # Base Term: +0.50 * g(a,p,b,q) * g(b,c,j,i) * g(j,i,a,c) / [(E_corr + e_j + e_i - e_b - e_c)] * [(E_corr + e_j + e_i + e_q - e_c - e_a - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, act_idx)]
    if 'int_17' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_b, :, :, :]
                b1 = b1_full[:, :, i_a, :]
                d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                b0 = b0 / d1
                num = np.einsum('CED,EDC->CDE', b0, b1, optimize=True)
                I1[i_a, i_b, :, :, :] += num
        _cache['int_17'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_17'][i_a, i_b, :, :, :]
            b2 = b2_full[i_a, :, i_b, :]
            num = np.einsum('CDE,FG->CDEFG', I1, b2, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, :, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            I2 = np.einsum('CDEFG->FG', num, optimize=True)
            g_temp[1][:, :] += 0.5 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_17'] -= 1
    if _use_count['int_17'] == 0:
        del _cache['int_17']
    # Projecting permutation: +0.50 * g(a,p,b,q) * g(b,c,j,i) * g(j,i,a,c) / [(E_corr + e_j + e_i - e_b - e_c)] * [(E_corr + e_j + e_i + e_q - e_c - e_a - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 899ed865 contains 1 permutations ---
    # Base Term: -0.50 * g(a,b,k,j) * g(i,p,b,q) * g(k,j,a,i) / [(E_corr + e_k + e_j - e_a - e_b)] * [(E_corr + e_i - e_b)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_18' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BED,EDC->BCDE', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 1 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx][:, None, None, None]
            num = num / d0
            I1 += np.einsum('BCDE->BC', num, optimize=True)
        _cache['int_18'] = I1
    I1 = _cache['int_18']
    b2 = b2_full
    num = np.einsum('BC,CFBG->BCFG', I1, b2, optimize=True)
    I2 = np.einsum('BCFG->FG', num, optimize=True)
    g_temp[1][:, :] += -0.5 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_18'] -= 1
    if _use_count['int_18'] == 0:
        del _cache['int_18']
    # Projecting permutation: -0.50 * g(a,b,k,j) * g(i,p,b,q) * g(k,j,a,i) / [(E_corr + e_k + e_j - e_a - e_b)] * [(E_corr + e_i - e_b)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 2330cab2 contains 1 permutations ---
    # Base Term: -0.250 * g(a,b,k,j) * g(i,p,a,b) * g(k,j,q,i) / [(E_corr + e_k + e_j - e_a - e_b)] * [(E_corr + e_q + e_i - e_a - e_b)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_19' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BED,EDGC->BCDEG', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None, None, None]
            num = num / d0
            I1[i_a, :, :, :] += np.einsum('BCDEG->BCG', num, optimize=True)
        _cache['int_19'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_19'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BCG,CFB->BCFG', I1, b2, optimize=True)
        I2 = np.einsum('BCFG->FG', num, optimize=True)
        g_temp[1][:, :] += -0.25 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_19'] -= 1
    if _use_count['int_19'] == 0:
        del _cache['int_19']
    # Projecting permutation: -0.250 * g(a,b,k,j) * g(i,p,a,b) * g(k,j,q,i) / [(E_corr + e_k + e_j - e_a - e_b)] * [(E_corr + e_q + e_i - e_a - e_b)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 4aefda64 contains 1 permutations ---
    # Base Term: -1.00 * g(j,a,b,i) * g(b,c,q,j) * g(i,p,a,c) / [(E_corr + e_q + e_j - e_b - e_c)] * [(E_corr + e_q + e_i - e_c - e_a)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_20' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[:, i_a, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('ED,CGE->CDEG', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx][:, None, None, None] - eps_spin[virt_idx[i_a]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :] += np.einsum('CDEG->CDG', num, optimize=True)
        _cache['int_20'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_20'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDG,DFC->CDFG', I1, b2, optimize=True)
        I2 = np.einsum('CDFG->FG', num, optimize=True)
        g_temp[1][:, :] += -1.0 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_20'] -= 1
    if _use_count['int_20'] == 0:
        del _cache['int_20']
    # Projecting permutation: -1.00 * g(j,a,b,i) * g(b,c,q,j) * g(i,p,a,c) / [(E_corr + e_q + e_j - e_b - e_c)] * [(E_corr + e_q + e_i - e_c - e_a)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 1a911f5e contains 1 permutations ---
    # Base Term: -0.50 * g(i,a,b,c) * g(b,c,j,i) * g(j,p,a,q) / [(E_corr + e_j + e_i - e_b - e_c)] * [(E_corr + e_j - e_a)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, virt_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_21' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[:, i_a, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('DC,CED->CDE', b0, b1, optimize=True)
                d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[occ_idx] + (shift / 1 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :] += np.einsum('CDE->E', num, optimize=True)
        _cache['int_21'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_21'][i_a, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('E,EFG->EFG', I1, b2, optimize=True)
        I2 = np.einsum('EFG->FG', num, optimize=True)
        g_temp[1][:, :] += -0.5 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_21'] -= 1
    if _use_count['int_21'] == 0:
        del _cache['int_21']
    # Projecting permutation: -0.50 * g(i,a,b,c) * g(b,c,j,i) * g(j,p,a,q) / [(E_corr + e_j + e_i - e_b - e_c)] * [(E_corr + e_j - e_a)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton ae3833b9 contains 1 permutations ---
    # Base Term: +1.00 * g(i,a,b,q) * g(b,c,j,i) * g(j,p,a,c) / [(E_corr + e_j + e_i - e_b - e_c)] * [(E_corr + e_j + e_q - e_c - e_a)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_22' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[:, i_a, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('DG,CED->CDEG', b0, b1, optimize=True)
                d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[virt_idx[i_a]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :] += np.einsum('CDEG->CEG', num, optimize=True)
        _cache['int_22'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_22'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CEG,EFC->CEFG', I1, b2, optimize=True)
        I2 = np.einsum('CEFG->FG', num, optimize=True)
        g_temp[1][:, :] += 1.0 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_22'] -= 1
    if _use_count['int_22'] == 0:
        del _cache['int_22']
    # Projecting permutation: +1.00 * g(i,a,b,q) * g(b,c,j,i) * g(j,p,a,c) / [(E_corr + e_j + e_i - e_b - e_c)] * [(E_corr + e_j + e_q - e_c - e_a)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton b4276dfe contains 1 permutations ---
    # Base Term: -0.250 * g(a,b,c,d) * g(c,d,q,i) * g(i,p,a,b) / [(E_corr + e_q + e_i - e_c - e_d)] * [(E_corr + e_q + e_i - e_a - e_b)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, virt_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_23' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                for i_c in range(len(virt_idx)):
                    b0 = b0_full[i_a, i_b, i_c, :]
                    b1 = b1_full[i_c, :, :, :]
                    num = np.einsum('D,DGE->DEG', b0, b1, optimize=True)
                    d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] - eps_spin[virt_idx[i_c]] - eps_spin[virt_idx][:, None, None]
                    d0[np.abs(d0) < 1e-12] = 1e-12
                    num = num / d0
                    d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                    d1[np.abs(d1) < 1e-12] = 1e-12
                    num = num / d1
                    I1[i_a, i_b, :, :] += np.einsum('DEG->EG', num, optimize=True)
        _cache['int_23'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_23'][i_a, i_b, :, :]
            b2 = b2_full[:, :, i_a, i_b]
            num = np.einsum('EG,EF->EFG', I1, b2, optimize=True)
            I2 = np.einsum('EFG->FG', num, optimize=True)
            g_temp[1][:, :] += -0.25 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_23'] -= 1
    if _use_count['int_23'] == 0:
        del _cache['int_23']
    # Projecting permutation: -0.250 * g(a,b,c,d) * g(c,d,q,i) * g(i,p,a,b) / [(E_corr + e_q + e_i - e_c - e_d)] * [(E_corr + e_q + e_i - e_a - e_b)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton b2162412 contains 1 permutations ---
    # Base Term: +0.50 * g(j,i,s,r) * g(q,p,j,i) / [(E_corr + e_j + e_i - e_q - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, act_idx)]
    b1 = g_anti_spin[np.ix_(act_idx, act_idx, occ_idx, occ_idx)]
    if 'int_24' not in _cache:
        b0 = b0
        b1 = b1
        num = np.einsum('BAFE,DCBA->ABCDEF', b0, b1, optimize=True)
        d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[act_idx][None, None, :, None, None, None]
        d0[np.abs(d0) < 1e-12] = 1e-12
        num = num / d0
        I1 = np.einsum('ABCDEF->CDEF', num, optimize=True)
        _cache['int_24'] = I1
    I1 = _cache['int_24']
    g_temp[2][:, :, :, :] += 0.5 * np.einsum('CDEF->CDEF', I1, optimize=True)
    _use_count['int_24'] -= 1
    if _use_count['int_24'] == 0:
        del _cache['int_24']
    # Projecting permutation: +0.50 * g(j,i,s,r) * g(q,p,j,i) / [(E_corr + e_j + e_i - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton fb0c734b contains 4 permutations ---
    # Base Term: +1.00 * g(a,p,s,i) * g(i,q,a,r) / [(E_corr + e_s + e_i - e_a - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0 = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_25' not in _cache:
        b0 = b0
        b1 = b1
        d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[act_idx][None, :, None, None]
        d0[np.abs(d0) < 1e-12] = 1e-12
        b0 = b0 / d0
        num = np.einsum('ACFB,BDAE->ABCDEF', b0, b1, optimize=True)
        I1 = np.einsum('ABCDEF->CDEF', num, optimize=True)
        _cache['int_25'] = I1
    I1 = _cache['int_25']
    g_temp[2][:, :, :, :] += 1.0 * np.einsum('CDEF->CDEF', I1, optimize=True)
    _use_count['int_25'] -= 1
    if _use_count['int_25'] == 0:
        del _cache['int_25']
    # Projecting permutation: +1.00 * g(a,p,s,i) * g(i,q,a,r) / [(E_corr + e_s + e_i - e_a - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -1.00 * g(a,p,r,i) * g(i,q,a,s) / [(E_corr + e_r + e_i - e_a - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: -1.00 * g(a,q,s,i) * g(i,p,a,r) / [(E_corr + e_s + e_i - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: +1.00 * g(a,q,r,i) * g(i,p,a,s) / [(E_corr + e_r + e_i - e_a - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton d6997bae contains 1 permutations ---
    # Base Term: +0.50 * g(a,b,s,r) * g(q,p,a,b) / [(E_corr + e_s + e_r - e_a - e_b)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, virt_idx)]
    if 'int_26' not in _cache:
        I1 = np.zeros((len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            b0 = b0 / d0
            num = np.einsum('BFE,DCB->BCDEF', b0, b1, optimize=True)
            I1 += np.einsum('BCDEF->CDEF', num, optimize=True)
        _cache['int_26'] = I1
    I1 = _cache['int_26']
    g_temp[2][:, :, :, :] += 0.5 * np.einsum('CDEF->CDEF', I1, optimize=True)
    _use_count['int_26'] -= 1
    if _use_count['int_26'] == 0:
        del _cache['int_26']
    # Projecting permutation: +0.50 * g(a,b,s,r) * g(q,p,a,b) / [(E_corr + e_s + e_r - e_a - e_b)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton 0ca95203 contains 1 permutations ---
    # Base Term: +0.250 * g(j,i,s,r) * g(l,k,j,i) * g(q,p,l,k) / [(E_corr + e_l + e_k - e_q - e_p)] * [(E_corr + e_j + e_i - e_q - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0 = g_anti_spin[np.ix_(occ_idx, occ_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_idx, act_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, act_idx)]
    if 'int_27' not in _cache:
        b0 = b0
        b1 = b1
        num = np.einsum('DCBA,FEDC->ABCDEF', b0, b1, optimize=True)
        d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_idx][None, None, None, None, None, :] - eps_spin[act_idx][None, None, None, None, :, None]
        d0[np.abs(d0) < 1e-12] = 1e-12
        num = num / d0
        d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, None, :] - eps_spin[act_idx][None, None, None, None, :, None]
        d1[np.abs(d1) < 1e-12] = 1e-12
        num = num / d1
        I1 = np.einsum('ABCDEF->ABEF', num, optimize=True)
        _cache['int_27'] = I1
    I1 = _cache['int_27']
    num = np.einsum('ABEF,BAHG->ABEFGH', I1, b2, optimize=True)
    I2 = np.einsum('ABEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += 0.25 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_27'] -= 1
    if _use_count['int_27'] == 0:
        del _cache['int_27']
    # Projecting permutation: +0.250 * g(j,i,s,r) * g(l,k,j,i) * g(q,p,l,k) / [(E_corr + e_l + e_k - e_q - e_p)] * [(E_corr + e_j + e_i - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton eb51cbc6 contains 2 permutations ---
    # Base Term: -0.50 * g(k,a,j,i) * g(j,i,a,r) * g(q,p,s,k) / [(E_corr + e_s + e_k - e_q - e_p)] * [(E_corr + e_s + e_j + e_i - e_q - e_p - e_a)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_28' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('DCB,FEHD->BCDEFH', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[act_idx][None, None, None, :, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[virt_idx[i_a]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEFH->BCEFH', num, optimize=True)
        _cache['int_28'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_28'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BCEFH,CBG->BCEFGH', I1, b2, optimize=True)
        I2 = np.einsum('BCEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_28'] -= 1
    if _use_count['int_28'] == 0:
        del _cache['int_28']
    # Projecting permutation: -0.50 * g(k,a,j,i) * g(j,i,a,r) * g(q,p,s,k) / [(E_corr + e_s + e_k - e_q - e_p)] * [(E_corr + e_s + e_j + e_i - e_q - e_p - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +0.50 * g(k,a,j,i) * g(j,i,a,s) * g(q,p,r,k) / [(E_corr + e_r + e_k - e_q - e_p)] * [(E_corr + e_r + e_j + e_i - e_q - e_p - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton 36618024 contains 4 permutations ---
    # Base Term: +0.50 * g(a,p,s,k) * g(j,i,a,r) * g(k,q,j,i) / [(E_corr + e_s + e_k - e_a - e_p)] * [(E_corr + e_s + e_j + e_i - e_a - e_p - e_q)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_29' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('EHD,DFCB->BCDEFH', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[:, None, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[act_idx][None, None, None, None, :, None]
            num = num / d0
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEFH->BCEFH', num, optimize=True)
        _cache['int_29'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_29'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BCEFH,CBG->BCEFGH', I1, b2, optimize=True)
        I2 = np.einsum('BCEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += 0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_29'] -= 1
    if _use_count['int_29'] == 0:
        del _cache['int_29']
    # Projecting permutation: +0.50 * g(a,p,s,k) * g(j,i,a,r) * g(k,q,j,i) / [(E_corr + e_s + e_k - e_a - e_p)] * [(E_corr + e_s + e_j + e_i - e_a - e_p - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -0.50 * g(a,p,r,k) * g(j,i,a,s) * g(k,q,j,i) / [(E_corr + e_r + e_k - e_a - e_p)] * [(E_corr + e_r + e_j + e_i - e_a - e_p - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: -0.50 * g(a,q,s,k) * g(j,i,a,r) * g(k,p,j,i) / [(E_corr + e_s + e_k - e_a - e_q)] * [(E_corr + e_s + e_j + e_i - e_a - e_q - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: +0.50 * g(a,q,r,k) * g(j,i,a,s) * g(k,p,j,i) / [(E_corr + e_r + e_k - e_a - e_q)] * [(E_corr + e_r + e_j + e_i - e_a - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 36883c82 contains 2 permutations ---
    # Base Term: +1.00 * g(j,a,s,i) * g(k,i,a,r) * g(q,p,k,j) / [(E_corr + e_k + e_j - e_q - e_p)] * [(E_corr + e_k + e_s + e_i - e_q - e_p - e_a)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_30' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('CHB,FEDC->BCDEFH', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[act_idx][None, None, None, :, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[virt_idx[i_a]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEFH->BDEFH', num, optimize=True)
        _cache['int_30'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_30'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BDEFH,DBG->BDEFGH', I1, b2, optimize=True)
        I2 = np.einsum('BDEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += 1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_30'] -= 1
    if _use_count['int_30'] == 0:
        del _cache['int_30']
    # Projecting permutation: +1.00 * g(j,a,s,i) * g(k,i,a,r) * g(q,p,k,j) / [(E_corr + e_k + e_j - e_q - e_p)] * [(E_corr + e_k + e_s + e_i - e_q - e_p - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -1.00 * g(j,a,r,i) * g(k,i,a,s) * g(q,p,k,j) / [(E_corr + e_k + e_j - e_q - e_p)] * [(E_corr + e_k + e_r + e_i - e_q - e_p - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton 337e69f1 contains 2 permutations ---
    # Base Term: +1.00 * g(a,p,k,j) * g(k,i,s,r) * g(j,q,a,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_k + e_i - e_p - e_q)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0 = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, act_idx)]
    if 'int_31' not in _cache:
        b0 = b0
        b1 = b1
        d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[act_idx][None, :, None, None]
        d1[np.abs(d1) < 1e-12] = 1e-12
        b0 = b0 / d1
        num = np.einsum('AEDC,CFAB->ABCDEF', b0, b1, optimize=True)
        d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[act_idx][None, None, None, None, None, :]
        num = num / d0
        I1 = np.einsum('ABCDEF->BDEF', num, optimize=True)
        _cache['int_31'] = I1
    I1 = _cache['int_31']
    num = np.einsum('BDEF,DBHG->BDEFGH', I1, b2, optimize=True)
    I2 = np.einsum('BDEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += 1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_31'] -= 1
    if _use_count['int_31'] == 0:
        del _cache['int_31']
    # Projecting permutation: +1.00 * g(a,p,k,j) * g(k,i,s,r) * g(j,q,a,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_k + e_i - e_p - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -1.00 * g(a,q,k,j) * g(k,i,s,r) * g(j,p,a,i) / [(E_corr + e_k + e_j - e_a - e_q)] * [(E_corr + e_k + e_i - e_q - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))

    # --- Skeleton 2bad60a4 contains 4 permutations ---
    # Base Term: -1.00 * g(a,p,k,j) * g(k,i,a,r) * g(j,q,s,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_k + e_s + e_i - e_a - e_p - e_q)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_32' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('EDC,CFHB->BCDEFH', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[:, None, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[act_idx][None, None, None, None, :, None]
            num = num / d0
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEFH->BDEFH', num, optimize=True)
        _cache['int_32'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_32'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BDEFH,DBG->BDEFGH', I1, b2, optimize=True)
        I2 = np.einsum('BDEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_32'] -= 1
    if _use_count['int_32'] == 0:
        del _cache['int_32']
    # Projecting permutation: -1.00 * g(a,p,k,j) * g(k,i,a,r) * g(j,q,s,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_k + e_s + e_i - e_a - e_p - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(a,p,k,j) * g(k,i,a,s) * g(j,q,r,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_k + e_r + e_i - e_a - e_p - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: +1.00 * g(a,q,k,j) * g(k,i,a,r) * g(j,p,s,i) / [(E_corr + e_k + e_j - e_a - e_q)] * [(E_corr + e_k + e_s + e_i - e_a - e_q - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: -1.00 * g(a,q,k,j) * g(k,i,a,s) * g(j,p,r,i) / [(E_corr + e_k + e_j - e_a - e_q)] * [(E_corr + e_k + e_r + e_i - e_a - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton f522c72e contains 2 permutations ---
    # Base Term: -1.00 * g(a,q,j,i) * g(b,p,s,r) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_a - e_q)] * [(E_corr + e_j + e_i + e_s + e_r - e_a - e_q - e_b - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, act_idx)]
    if 'int_33' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, :, :]
                b1 = b1_full[:, :, i_a, i_b]
                d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                b0 = b0 / d1
                num = np.einsum('FDC,DC->CDF', b0, b1, optimize=True)
                I1[i_a, i_b, :, :, :] += num
        _cache['int_33'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_33'][i_a, i_b, :, :, :]
            b2 = b2_full[i_b, :, :, :]
            num = np.einsum('CDF,EHG->CDEFGH', I1, b2, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 4 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[occ_idx] + (shift / 4 if shift is not None else 0.0))[:, None, None, None, None, None] + (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[act_idx][None, None, :, None, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
            g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_33'] -= 1
    if _use_count['int_33'] == 0:
        del _cache['int_33']
    # Projecting permutation: -1.00 * g(a,q,j,i) * g(b,p,s,r) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_a - e_q)] * [(E_corr + e_j + e_i + e_s + e_r - e_a - e_q - e_b - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(a,p,j,i) * g(b,q,s,r) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_a - e_p)] * [(E_corr + e_j + e_i + e_s + e_r - e_a - e_p - e_b - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))

    # --- Skeleton 1515b1e3 contains 1 permutations ---
    # Base Term: +0.50 * g(a,b,s,r) * g(j,i,a,b) * g(q,p,j,i) / [(E_corr + e_s + e_r - e_a - e_b)] * [(E_corr + e_s + e_r + e_j + e_i - e_a - e_b - e_q - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(act_idx, act_idx, occ_idx, occ_idx)]
    if 'int_34' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, i_b, :, :]
                b1 = b1_full[:, :, i_a, i_b]
                d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[:, None] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                b0 = b0 / d1
                num = np.einsum('HG,DC->CDGH', b0, b1, optimize=True)
                I1[i_a, i_b, :, :, :, :] += num
        _cache['int_34'] = I1
    b2 = b2_full
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_34'][i_a, i_b, :, :, :, :]
            num = np.einsum('CDGH,FEDC->CDEFGH', I1, b2, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 4 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[occ_idx] + (shift / 4 if shift is not None else 0.0))[:, None, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[act_idx][None, None, :, None, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
            g_temp[2][:, :, :, :] += 0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_34'] -= 1
    if _use_count['int_34'] == 0:
        del _cache['int_34']
    # Projecting permutation: +0.50 * g(a,b,s,r) * g(j,i,a,b) * g(q,p,j,i) / [(E_corr + e_s + e_r - e_a - e_b)] * [(E_corr + e_s + e_r + e_j + e_i - e_a - e_b - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton 00869b9c contains 2 permutations ---
    # Base Term: -1.00 * g(a,b,r,i) * g(j,i,a,b) * g(q,p,s,j) / [(E_corr + e_r + e_i - e_a - e_b)] * [(E_corr + e_r + e_i + e_s + e_j - e_a - e_b - e_q - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(act_idx, act_idx, act_idx, occ_idx)]
    if 'int_35' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, i_b, :, :]
                b1 = b1_full[:, :, i_a, i_b]
                d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[:, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                b0 = b0 / d1
                num = np.einsum('GC,DC->CDG', b0, b1, optimize=True)
                I1[i_a, i_b, :, :, :] += num
        _cache['int_35'] = I1
    b2 = b2_full
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_35'][i_a, i_b, :, :, :]
            num = np.einsum('CDG,FEHD->CDEFGH', I1, b2, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 4 if shift is not None else 0.0))[:, None, None, None, None, None] + (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 4 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[act_idx][None, None, :, None, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
            g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_35'] -= 1
    if _use_count['int_35'] == 0:
        del _cache['int_35']
    # Projecting permutation: -1.00 * g(a,b,r,i) * g(j,i,a,b) * g(q,p,s,j) / [(E_corr + e_r + e_i - e_a - e_b)] * [(E_corr + e_r + e_i + e_s + e_j - e_a - e_b - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(a,b,s,i) * g(j,i,a,b) * g(q,p,r,j) / [(E_corr + e_s + e_i - e_a - e_b)] * [(E_corr + e_s + e_i + e_r + e_j - e_a - e_b - e_q - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton f02d16b9 contains 4 permutations ---
    # Base Term: -1.00 * g(a,q,b,i) * g(b,p,s,j) * g(j,i,a,r) / [(E_corr + e_s + e_j - e_b - e_p)] * [(E_corr + e_s + e_j + e_i - e_p - e_a - e_q)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_36' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('FBC,BEHD->BCDEFH', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, None, :, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEFH->CDEFH', num, optimize=True)
        _cache['int_36'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_36'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDEFH,DCG->CDEFGH', I1, b2, optimize=True)
        I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_36'] -= 1
    if _use_count['int_36'] == 0:
        del _cache['int_36']
    # Projecting permutation: -1.00 * g(a,q,b,i) * g(b,p,s,j) * g(j,i,a,r) / [(E_corr + e_s + e_j - e_b - e_p)] * [(E_corr + e_s + e_j + e_i - e_p - e_a - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(a,q,b,i) * g(b,p,r,j) * g(j,i,a,s) / [(E_corr + e_r + e_j - e_b - e_p)] * [(E_corr + e_r + e_j + e_i - e_p - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: +1.00 * g(a,p,b,i) * g(b,q,s,j) * g(j,i,a,r) / [(E_corr + e_s + e_j - e_b - e_q)] * [(E_corr + e_s + e_j + e_i - e_q - e_a - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: -1.00 * g(a,p,b,i) * g(b,q,r,j) * g(j,i,a,s) / [(E_corr + e_r + e_j - e_b - e_q)] * [(E_corr + e_r + e_j + e_i - e_q - e_a - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 9685ee7d contains 2 permutations ---
    # Base Term: +2.00 * g(a,q,r,i) * g(b,p,s,j) * g(j,i,a,b) / [(E_corr + e_r + e_i - e_a - e_q)] * [(E_corr + e_r + e_i + e_s + e_j - e_a - e_q - e_b - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    if 'int_37' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, :, :]
                b1 = b1_full[:, :, i_a, i_b]
                d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                b0 = b0 / d1
                num = np.einsum('FGC,DC->CDFG', b0, b1, optimize=True)
                I1[i_a, i_b, :, :, :, :] += num
        _cache['int_37'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_37'][i_a, i_b, :, :, :, :]
            b2 = b2_full[i_b, :, :, :]
            num = np.einsum('CDFG,EHD->CDEFGH', I1, b2, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 4 if shift is not None else 0.0))[:, None, None, None, None, None] + (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 4 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[act_idx][None, None, :, None, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
            g_temp[2][:, :, :, :] += 2.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_37'] -= 1
    if _use_count['int_37'] == 0:
        del _cache['int_37']
    # Projecting permutation: +2.00 * g(a,q,r,i) * g(b,p,s,j) * g(j,i,a,b) / [(E_corr + e_r + e_i - e_a - e_q)] * [(E_corr + e_r + e_i + e_s + e_j - e_a - e_q - e_b - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -2.00 * g(a,q,s,i) * g(b,p,r,j) * g(j,i,a,b) / [(E_corr + e_s + e_i - e_a - e_q)] * [(E_corr + e_s + e_i + e_r + e_j - e_a - e_q - e_b - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton 10d2307f contains 2 permutations ---
    # Base Term: -1.00 * g(a,b,s,j) * g(j,i,b,r) * g(q,p,a,i) / [(E_corr + e_s + e_j - e_a - e_b)] * [(E_corr + e_s + e_j + e_i - e_b - e_q - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_38' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BHD,FEC->BCDEFH', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[act_idx][None, None, None, :, None, None]
            num = num / d0
            I1 += num
        _cache['int_38'] = I1
    I1 = _cache['int_38']
    b2 = b2_full
    num = np.einsum('BCDEFH,DCBG->BCDEFGH', I1, b2, optimize=True)
    I2 = np.einsum('BCDEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_38'] -= 1
    if _use_count['int_38'] == 0:
        del _cache['int_38']
    # Projecting permutation: -1.00 * g(a,b,s,j) * g(j,i,b,r) * g(q,p,a,i) / [(E_corr + e_s + e_j - e_a - e_b)] * [(E_corr + e_s + e_j + e_i - e_b - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(a,b,r,j) * g(j,i,b,s) * g(q,p,a,i) / [(E_corr + e_r + e_j - e_a - e_b)] * [(E_corr + e_r + e_j + e_i - e_b - e_q - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton ab6757bb contains 4 permutations ---
    # Base Term: -0.50 * g(a,q,b,s) * g(b,p,j,i) * g(j,i,a,r) / [(E_corr + e_j + e_i - e_b - e_p)] * [(E_corr + e_j + e_i + e_s - e_p - e_a - e_q)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_39' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('FBH,BEDC->BCDEFH', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, None, :, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEFH->CDEFH', num, optimize=True)
        _cache['int_39'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_39'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDEFH,DCG->CDEFGH', I1, b2, optimize=True)
        I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_39'] -= 1
    if _use_count['int_39'] == 0:
        del _cache['int_39']
    # Projecting permutation: -0.50 * g(a,q,b,s) * g(b,p,j,i) * g(j,i,a,r) / [(E_corr + e_j + e_i - e_b - e_p)] * [(E_corr + e_j + e_i + e_s - e_p - e_a - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +0.50 * g(a,q,b,r) * g(b,p,j,i) * g(j,i,a,s) / [(E_corr + e_j + e_i - e_b - e_p)] * [(E_corr + e_j + e_i + e_r - e_p - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: +0.50 * g(a,p,b,s) * g(b,q,j,i) * g(j,i,a,r) / [(E_corr + e_j + e_i - e_b - e_q)] * [(E_corr + e_j + e_i + e_s - e_q - e_a - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: -0.50 * g(a,p,b,r) * g(b,q,j,i) * g(j,i,a,s) / [(E_corr + e_j + e_i - e_b - e_q)] * [(E_corr + e_j + e_i + e_r - e_q - e_a - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton aaa5540f contains 1 permutations ---
    # Base Term: +0.50 * g(a,b,j,i) * g(j,i,s,r) * g(q,p,a,b) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_j + e_i - e_q - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, act_idx)]
    if 'int_40' not in _cache:
        I1 = np.zeros((len(occ_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BDC,FEB->BCDEF', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[act_idx][None, None, None, None, :] - eps_spin[act_idx][None, None, None, :, None]
            num = num / d0
            I1 += np.einsum('BCDEF->CDEF', num, optimize=True)
        _cache['int_40'] = I1
    I1 = _cache['int_40']
    b2 = b2_full
    num = np.einsum('CDEF,DCHG->CDEFGH', I1, b2, optimize=True)
    I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += 0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_40'] -= 1
    if _use_count['int_40'] == 0:
        del _cache['int_40']
    # Projecting permutation: +0.50 * g(a,b,j,i) * g(j,i,s,r) * g(q,p,a,b) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_j + e_i - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton e0fc6e6d contains 2 permutations ---
    # Base Term: -1.00 * g(a,b,j,i) * g(j,i,b,r) * g(q,p,a,s) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_j + e_i + e_s - e_b - e_q - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_41' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BDC,FEH->BCDEFH', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[act_idx][None, None, None, :, None, None]
            num = num / d0
            I1 += num
        _cache['int_41'] = I1
    I1 = _cache['int_41']
    b2 = b2_full
    num = np.einsum('BCDEFH,DCBG->BCDEFGH', I1, b2, optimize=True)
    I2 = np.einsum('BCDEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_41'] -= 1
    if _use_count['int_41'] == 0:
        del _cache['int_41']
    # Projecting permutation: -1.00 * g(a,b,j,i) * g(j,i,b,r) * g(q,p,a,s) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_j + e_i + e_s - e_b - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(a,b,j,i) * g(j,i,b,s) * g(q,p,a,r) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_j + e_i + e_r - e_b - e_q - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton c69b52d3 contains 2 permutations ---
    # Base Term: -0.50 * g(a,p,k,j) * g(i,q,s,r) * g(k,j,a,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_i - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0 = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, act_idx)]
    if 'int_42' not in _cache:
        b0 = b0
        b1 = b1
        d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[act_idx][None, :, None, None]
        d1[np.abs(d1) < 1e-12] = 1e-12
        b0 = b0 / d1
        num = np.einsum('AEDC,DCAB->ABCDE', b0, b1, optimize=True)
        d0 = (eps_spin[occ_idx] + (shift / 1 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[act_idx][None, None, None, None, :]
        num = num / d0
        I1 = np.einsum('ABCDE->BE', num, optimize=True)
        _cache['int_42'] = I1
    I1 = _cache['int_42']
    num = np.einsum('BE,BFHG->BEFGH', I1, b2, optimize=True)
    I2 = np.einsum('BEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += -0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_42'] -= 1
    if _use_count['int_42'] == 0:
        del _cache['int_42']
    # Projecting permutation: -0.50 * g(a,p,k,j) * g(i,q,s,r) * g(k,j,a,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_i - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +0.50 * g(a,q,k,j) * g(i,p,s,r) * g(k,j,a,i) / [(E_corr + e_k + e_j - e_a - e_q)] * [(E_corr + e_i - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))

    # --- Skeleton 3d619463 contains 4 permutations ---
    # Base Term: +0.50 * g(a,p,k,j) * g(i,q,a,r) * g(k,j,s,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_s + e_i - e_a - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0 = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_43' not in _cache:
        b0 = b0
        b1 = b1
        d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[act_idx][None, :, None, None]
        d1[np.abs(d1) < 1e-12] = 1e-12
        b0 = b0 / d1
        num = np.einsum('AEDC,DCHB->ABCDEH', b0, b1, optimize=True)
        d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None]
        num = num / d0
        I1 = np.einsum('ABCDEH->ABEH', num, optimize=True)
        _cache['int_43'] = I1
    I1 = _cache['int_43']
    num = np.einsum('ABEH,BFAG->ABEFGH', I1, b2, optimize=True)
    I2 = np.einsum('ABEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += 0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_43'] -= 1
    if _use_count['int_43'] == 0:
        del _cache['int_43']
    # Projecting permutation: +0.50 * g(a,p,k,j) * g(i,q,a,r) * g(k,j,s,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_s + e_i - e_a - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -0.50 * g(a,p,k,j) * g(i,q,a,s) * g(k,j,r,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_r + e_i - e_a - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: -0.50 * g(a,q,k,j) * g(i,p,a,r) * g(k,j,s,i) / [(E_corr + e_k + e_j - e_a - e_q)] * [(E_corr + e_s + e_i - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: +0.50 * g(a,q,k,j) * g(i,p,a,s) * g(k,j,r,i) / [(E_corr + e_k + e_j - e_a - e_q)] * [(E_corr + e_r + e_i - e_a - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 626d96f0 contains 4 permutations ---
    # Base Term: +1.00 * g(j,a,b,i) * g(b,p,s,j) * g(i,q,a,r) / [(E_corr + e_s + e_j - e_b - e_p)] * [(E_corr + e_s + e_i - e_p - e_a)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_44' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('DBC,BEHD->BCDEH', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[act_idx][None, None, None, :, None] - eps_spin[virt_idx[i_a]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :] += np.einsum('BCDEH->CEH', num, optimize=True)
        _cache['int_44'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_44'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CEH,CFG->CEFGH', I1, b2, optimize=True)
        I2 = np.einsum('CEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += 1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_44'] -= 1
    if _use_count['int_44'] == 0:
        del _cache['int_44']
    # Projecting permutation: +1.00 * g(j,a,b,i) * g(b,p,s,j) * g(i,q,a,r) / [(E_corr + e_s + e_j - e_b - e_p)] * [(E_corr + e_s + e_i - e_p - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -1.00 * g(j,a,b,i) * g(b,p,r,j) * g(i,q,a,s) / [(E_corr + e_r + e_j - e_b - e_p)] * [(E_corr + e_r + e_i - e_p - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: -1.00 * g(j,a,b,i) * g(b,q,s,j) * g(i,p,a,r) / [(E_corr + e_s + e_j - e_b - e_q)] * [(E_corr + e_s + e_i - e_q - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: +1.00 * g(j,a,b,i) * g(b,q,r,j) * g(i,p,a,s) / [(E_corr + e_r + e_j - e_b - e_q)] * [(E_corr + e_r + e_i - e_q - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 0a3b7670 contains 4 permutations ---
    # Base Term: -1.00 * g(j,a,r,i) * g(b,p,s,j) * g(i,q,a,b) / [(E_corr + e_s + e_j - e_b - e_p)] * [(E_corr + e_s + e_r + e_i - e_b - e_p - e_a)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_45' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('DGC,BEHD->BCDEGH', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[virt_idx[i_a]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEGH->BCEGH', num, optimize=True)
        _cache['int_45'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_45'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BCEGH,CFB->BCEFGH', I1, b2, optimize=True)
        I2 = np.einsum('BCEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_45'] -= 1
    if _use_count['int_45'] == 0:
        del _cache['int_45']
    # Projecting permutation: -1.00 * g(j,a,r,i) * g(b,p,s,j) * g(i,q,a,b) / [(E_corr + e_s + e_j - e_b - e_p)] * [(E_corr + e_s + e_r + e_i - e_b - e_p - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(j,a,s,i) * g(b,p,r,j) * g(i,q,a,b) / [(E_corr + e_r + e_j - e_b - e_p)] * [(E_corr + e_r + e_s + e_i - e_b - e_p - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: +1.00 * g(j,a,r,i) * g(b,q,s,j) * g(i,p,a,b) / [(E_corr + e_s + e_j - e_b - e_q)] * [(E_corr + e_s + e_r + e_i - e_b - e_q - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: -1.00 * g(j,a,s,i) * g(b,q,r,j) * g(i,p,a,b) / [(E_corr + e_r + e_j - e_b - e_q)] * [(E_corr + e_r + e_s + e_i - e_b - e_q - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 8c2824b9 contains 4 permutations ---
    # Base Term: -1.00 * g(a,b,s,j) * g(i,q,b,r) * g(j,p,a,i) / [(E_corr + e_s + e_j - e_a - e_b)] * [(E_corr + e_s + e_i - e_b - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_46' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BHD,DEC->BCDEH', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None]
            num = num / d0
            I1 += np.einsum('BCDEH->BCEH', num, optimize=True)
        _cache['int_46'] = I1
    I1 = _cache['int_46']
    b2 = b2_full
    num = np.einsum('BCEH,CFBG->BCEFGH', I1, b2, optimize=True)
    I2 = np.einsum('BCEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_46'] -= 1
    if _use_count['int_46'] == 0:
        del _cache['int_46']
    # Projecting permutation: -1.00 * g(a,b,s,j) * g(i,q,b,r) * g(j,p,a,i) / [(E_corr + e_s + e_j - e_a - e_b)] * [(E_corr + e_s + e_i - e_b - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(a,b,r,j) * g(i,q,b,s) * g(j,p,a,i) / [(E_corr + e_r + e_j - e_a - e_b)] * [(E_corr + e_r + e_i - e_b - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: +1.00 * g(a,b,s,j) * g(i,p,b,r) * g(j,q,a,i) / [(E_corr + e_s + e_j - e_a - e_b)] * [(E_corr + e_s + e_i - e_b - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: -1.00 * g(a,b,r,j) * g(i,p,b,s) * g(j,q,a,i) / [(E_corr + e_r + e_j - e_a - e_b)] * [(E_corr + e_r + e_i - e_b - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 61db6856 contains 4 permutations ---
    # Base Term: -0.50 * g(a,b,s,j) * g(i,q,a,b) * g(j,p,r,i) / [(E_corr + e_s + e_j - e_a - e_b)] * [(E_corr + e_s + e_r + e_i - e_a - e_b - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_47' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BHD,DEGC->BCDEGH', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None]
            num = num / d0
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEGH->BCEGH', num, optimize=True)
        _cache['int_47'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_47'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BCEGH,CFB->BCEFGH', I1, b2, optimize=True)
        I2 = np.einsum('BCEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_47'] -= 1
    if _use_count['int_47'] == 0:
        del _cache['int_47']
    # Projecting permutation: -0.50 * g(a,b,s,j) * g(i,q,a,b) * g(j,p,r,i) / [(E_corr + e_s + e_j - e_a - e_b)] * [(E_corr + e_s + e_r + e_i - e_a - e_b - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +0.50 * g(a,b,r,j) * g(i,q,a,b) * g(j,p,s,i) / [(E_corr + e_r + e_j - e_a - e_b)] * [(E_corr + e_r + e_s + e_i - e_a - e_b - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: +0.50 * g(a,b,s,j) * g(i,p,a,b) * g(j,q,r,i) / [(E_corr + e_s + e_j - e_a - e_b)] * [(E_corr + e_s + e_r + e_i - e_a - e_b - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: -0.50 * g(a,b,r,j) * g(i,p,a,b) * g(j,q,s,i) / [(E_corr + e_r + e_j - e_a - e_b)] * [(E_corr + e_r + e_s + e_i - e_a - e_b - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton f4f231ee contains 4 permutations ---
    # Base Term: -1.00 * g(i,a,b,s) * g(b,p,j,i) * g(j,q,a,r) / [(E_corr + e_j + e_i - e_b - e_p)] * [(E_corr + e_j + e_s - e_p - e_a)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_48' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('CBH,BEDC->BCDEH', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] - eps_spin[act_idx][None, None, None, :, None] - eps_spin[virt_idx[i_a]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :] += np.einsum('BCDEH->DEH', num, optimize=True)
        _cache['int_48'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_48'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('DEH,DFG->DEFGH', I1, b2, optimize=True)
        I2 = np.einsum('DEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_48'] -= 1
    if _use_count['int_48'] == 0:
        del _cache['int_48']
    # Projecting permutation: -1.00 * g(i,a,b,s) * g(b,p,j,i) * g(j,q,a,r) / [(E_corr + e_j + e_i - e_b - e_p)] * [(E_corr + e_j + e_s - e_p - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(i,a,b,r) * g(b,p,j,i) * g(j,q,a,s) / [(E_corr + e_j + e_i - e_b - e_p)] * [(E_corr + e_j + e_r - e_p - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: +1.00 * g(i,a,b,s) * g(b,q,j,i) * g(j,p,a,r) / [(E_corr + e_j + e_i - e_b - e_q)] * [(E_corr + e_j + e_s - e_q - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: -1.00 * g(i,a,b,r) * g(b,q,j,i) * g(j,p,a,s) / [(E_corr + e_j + e_i - e_b - e_q)] * [(E_corr + e_j + e_r - e_q - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 42c2dc6b contains 2 permutations ---
    # Base Term: -1.00 * g(i,a,s,r) * g(b,p,j,i) * g(j,q,a,b) / [(E_corr + e_j + e_i - e_b - e_p)] * [(E_corr + e_j + e_s + e_r - e_b - e_p - e_a)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, act_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_49' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('CHG,BEDC->BCDEGH', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[virt_idx[i_a]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEGH->BDEGH', num, optimize=True)
        _cache['int_49'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_49'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BDEGH,DFB->BDEFGH', I1, b2, optimize=True)
        I2 = np.einsum('BDEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_49'] -= 1
    if _use_count['int_49'] == 0:
        del _cache['int_49']
    # Projecting permutation: -1.00 * g(i,a,s,r) * g(b,p,j,i) * g(j,q,a,b) / [(E_corr + e_j + e_i - e_b - e_p)] * [(E_corr + e_j + e_s + e_r - e_b - e_p - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(i,a,s,r) * g(b,q,j,i) * g(j,p,a,b) / [(E_corr + e_j + e_i - e_b - e_q)] * [(E_corr + e_j + e_s + e_r - e_b - e_q - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))

    # --- Skeleton f4126c60 contains 1 permutations ---
    # Base Term: -1.00 * g(a,b,j,i) * g(i,p,a,b) * g(j,q,s,r) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_i + e_s + e_r - e_a - e_b - e_q)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_50' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BDC,DFHG->BCDFGH', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None]
            num = num / d0
            I1[i_a, :, :, :, :, :] += np.einsum('BCDFGH->BCFGH', num, optimize=True)
        _cache['int_50'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_50'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BCFGH,CEB->BCEFGH', I1, b2, optimize=True)
        I2 = np.einsum('BCEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_50'] -= 1
    if _use_count['int_50'] == 0:
        del _cache['int_50']
    # Projecting permutation: -1.00 * g(a,b,j,i) * g(i,p,a,b) * g(j,q,s,r) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_i + e_s + e_r - e_a - e_b - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton f4126c60 contains 1 permutations ---
    # Base Term: -1.00 * g(a,b,j,i) * g(i,p,s,r) * g(j,q,a,b) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_i - e_q)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, act_idx)]
    if 'int_51' not in _cache:
        I1 = np.zeros((len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BDC,DFB->BCDF', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 1 if shift is not None else 0.0))[None, :, None, None] - eps_spin[act_idx][None, None, None, :]
            num = num / d0
            I1 += np.einsum('BCDF->CF', num, optimize=True)
        _cache['int_51'] = I1
    I1 = _cache['int_51']
    b2 = b2_full
    num = np.einsum('CF,CEHG->CEFGH', I1, b2, optimize=True)
    I2 = np.einsum('CEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_51'] -= 1
    if _use_count['int_51'] == 0:
        del _cache['int_51']
    # Projecting permutation: -1.00 * g(a,b,j,i) * g(i,p,s,r) * g(j,q,a,b) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_i - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton 0eba25b9 contains 2 permutations ---
    # Base Term: +2.00 * g(a,b,j,i) * g(i,p,a,s) * g(j,q,b,r) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_i + e_r - e_a - e_q)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_52' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BDC,DFBG->BCDFG', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, :, None]
            num = num / d0
            I1[i_a, :, :, :] += np.einsum('BCDFG->CFG', num, optimize=True)
        _cache['int_52'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_52'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CFG,CEH->CEFGH', I1, b2, optimize=True)
        I2 = np.einsum('CEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += 2.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_52'] -= 1
    if _use_count['int_52'] == 0:
        del _cache['int_52']
    # Projecting permutation: +2.00 * g(a,b,j,i) * g(i,p,a,s) * g(j,q,b,r) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_i + e_r - e_a - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -2.00 * g(a,b,j,i) * g(i,p,a,r) * g(j,q,b,s) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_i + e_s - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton 6af87110 contains 2 permutations ---
    # Base Term: -0.50 * g(a,b,c,i) * g(c,p,s,r) * g(i,q,a,b) / [(E_corr + e_s + e_r - e_c - e_p)] * [(E_corr + e_s + e_r + e_i - e_p - e_a - e_b)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_53' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, i_b, :, :]
                b1 = b1_full[:, :, :, :]
                num = np.einsum('CD,CEHG->CDEGH', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, :, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[act_idx][None, None, :, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, i_b, :, :, :, :] += np.einsum('CDEGH->DEGH', num, optimize=True)
        _cache['int_53'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_53'][i_a, i_b, :, :, :, :]
            b2 = b2_full[:, :, i_a, i_b]
            num = np.einsum('DEGH,DF->DEFGH', I1, b2, optimize=True)
            I2 = np.einsum('DEFGH->EFGH', num, optimize=True)
            g_temp[2][:, :, :, :] += -0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_53'] -= 1
    if _use_count['int_53'] == 0:
        del _cache['int_53']
    # Projecting permutation: -0.50 * g(a,b,c,i) * g(c,p,s,r) * g(i,q,a,b) / [(E_corr + e_s + e_r - e_c - e_p)] * [(E_corr + e_s + e_r + e_i - e_p - e_a - e_b)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +0.50 * g(a,b,c,i) * g(c,q,s,r) * g(i,p,a,b) / [(E_corr + e_s + e_r - e_c - e_q)] * [(E_corr + e_s + e_r + e_i - e_q - e_a - e_b)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))

    # --- Skeleton 8a96a8aa contains 2 permutations ---
    # Base Term: +1.00 * g(a,p,b,i) * g(b,c,s,r) * g(i,q,a,c) / [(E_corr + e_s + e_r - e_b - e_c)] * [(E_corr + e_s + e_r + e_i - e_c - e_a - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_54' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('ED,CHG->CDEGH', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, :, None, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :, :, :] += num
        _cache['int_54'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_54'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDEGH,DFC->CDEFGH', I1, b2, optimize=True)
        I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += 1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_54'] -= 1
    if _use_count['int_54'] == 0:
        del _cache['int_54']
    # Projecting permutation: +1.00 * g(a,p,b,i) * g(b,c,s,r) * g(i,q,a,c) / [(E_corr + e_s + e_r - e_b - e_c)] * [(E_corr + e_s + e_r + e_i - e_c - e_a - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -1.00 * g(a,q,b,i) * g(b,c,s,r) * g(i,p,a,c) / [(E_corr + e_s + e_r - e_b - e_c)] * [(E_corr + e_s + e_r + e_i - e_c - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))

    # --- Skeleton 7bb85e53 contains 4 permutations ---
    # Base Term: +0.50 * g(a,b,c,r) * g(c,p,s,i) * g(i,q,a,b) / [(E_corr + e_s + e_i - e_c - e_p)] * [(E_corr + e_s + e_i + e_r - e_p - e_a - e_b)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_55' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, i_b, :, :]
                b1 = b1_full[:, :, :, :]
                num = np.einsum('CG,CEHD->CDEGH', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, :, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] - eps_spin[act_idx][None, None, :, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, i_b, :, :, :, :] += np.einsum('CDEGH->DEGH', num, optimize=True)
        _cache['int_55'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_55'][i_a, i_b, :, :, :, :]
            b2 = b2_full[:, :, i_a, i_b]
            num = np.einsum('DEGH,DF->DEFGH', I1, b2, optimize=True)
            I2 = np.einsum('DEFGH->EFGH', num, optimize=True)
            g_temp[2][:, :, :, :] += 0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_55'] -= 1
    if _use_count['int_55'] == 0:
        del _cache['int_55']
    # Projecting permutation: +0.50 * g(a,b,c,r) * g(c,p,s,i) * g(i,q,a,b) / [(E_corr + e_s + e_i - e_c - e_p)] * [(E_corr + e_s + e_i + e_r - e_p - e_a - e_b)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -0.50 * g(a,b,c,s) * g(c,p,r,i) * g(i,q,a,b) / [(E_corr + e_r + e_i - e_c - e_p)] * [(E_corr + e_r + e_i + e_s - e_p - e_a - e_b)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: -0.50 * g(a,b,c,r) * g(c,q,s,i) * g(i,p,a,b) / [(E_corr + e_s + e_i - e_c - e_q)] * [(E_corr + e_s + e_i + e_r - e_q - e_a - e_b)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: +0.50 * g(a,b,c,s) * g(c,q,r,i) * g(i,p,a,b) / [(E_corr + e_r + e_i - e_c - e_q)] * [(E_corr + e_r + e_i + e_s - e_q - e_a - e_b)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 2c175111 contains 4 permutations ---
    # Base Term: +0.50 * g(a,p,b,c) * g(b,c,s,i) * g(i,q,a,r) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s + e_i - e_a - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, virt_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_56' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('EC,CHD->CDEH', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, :, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :] += np.einsum('CDEH->DEH', num, optimize=True)
        _cache['int_56'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_56'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('DEH,DFG->DEFGH', I1, b2, optimize=True)
        I2 = np.einsum('DEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += 0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_56'] -= 1
    if _use_count['int_56'] == 0:
        del _cache['int_56']
    # Projecting permutation: +0.50 * g(a,p,b,c) * g(b,c,s,i) * g(i,q,a,r) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s + e_i - e_a - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -0.50 * g(a,p,b,c) * g(b,c,r,i) * g(i,q,a,s) / [(E_corr + e_r + e_i - e_b - e_c)] * [(E_corr + e_r + e_i - e_a - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: -0.50 * g(a,q,b,c) * g(b,c,s,i) * g(i,p,a,r) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s + e_i - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: +0.50 * g(a,q,b,c) * g(b,c,r,i) * g(i,p,a,s) / [(E_corr + e_r + e_i - e_b - e_c)] * [(E_corr + e_r + e_i - e_a - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton e1b90331 contains 4 permutations ---
    # Base Term: -1.00 * g(a,p,b,r) * g(b,c,s,i) * g(i,q,a,c) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s + e_i + e_r - e_c - e_a - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_57' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('EG,CHD->CDEGH', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, :, None, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :, :, :] += num
        _cache['int_57'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_57'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDEGH,DFC->CDEFGH', I1, b2, optimize=True)
        I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_57'] -= 1
    if _use_count['int_57'] == 0:
        del _cache['int_57']
    # Projecting permutation: -1.00 * g(a,p,b,r) * g(b,c,s,i) * g(i,q,a,c) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s + e_i + e_r - e_c - e_a - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(a,p,b,s) * g(b,c,r,i) * g(i,q,a,c) / [(E_corr + e_r + e_i - e_b - e_c)] * [(E_corr + e_r + e_i + e_s - e_c - e_a - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: +1.00 * g(a,q,b,r) * g(b,c,s,i) * g(i,p,a,c) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s + e_i + e_r - e_c - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: -1.00 * g(a,q,b,s) * g(b,c,r,i) * g(i,p,a,c) / [(E_corr + e_r + e_i - e_b - e_c)] * [(E_corr + e_r + e_i + e_s - e_c - e_a - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 3838303e contains 2 permutations ---
    # Base Term: -0.50 * g(i,a,b,c) * g(b,c,s,i) * g(q,p,a,r) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s - e_a)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, virt_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, act_idx)]
    if 'int_58' not in _cache:
        I1 = np.zeros((len(virt_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[:, i_a, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('DC,CHD->CDH', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 1 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :] += np.einsum('CDH->H', num, optimize=True)
        _cache['int_58'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_58'][i_a, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('H,FEG->EFGH', I1, b2, optimize=True)
        I2 = num
        g_temp[2][:, :, :, :] += -0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_58'] -= 1
    if _use_count['int_58'] == 0:
        del _cache['int_58']
    # Projecting permutation: -0.50 * g(i,a,b,c) * g(b,c,s,i) * g(q,p,a,r) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +0.50 * g(i,a,b,c) * g(b,c,r,i) * g(q,p,a,s) / [(E_corr + e_r + e_i - e_b - e_c)] * [(E_corr + e_r - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton 6164c799 contains 2 permutations ---
    # Base Term: +1.00 * g(i,a,b,r) * g(b,c,s,i) * g(q,p,a,c) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s + e_r - e_c - e_a)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, virt_idx)]
    if 'int_59' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[:, i_a, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('DG,CHD->CDGH', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] - eps_spin[virt_idx][:, None, None, None] - eps_spin[virt_idx[i_a]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :] += np.einsum('CDGH->CGH', num, optimize=True)
        _cache['int_59'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_59'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CGH,FEC->CEFGH', I1, b2, optimize=True)
        I2 = np.einsum('CEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += 1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_59'] -= 1
    if _use_count['int_59'] == 0:
        del _cache['int_59']
    # Projecting permutation: +1.00 * g(i,a,b,r) * g(b,c,s,i) * g(q,p,a,c) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s + e_r - e_c - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -1.00 * g(i,a,b,s) * g(b,c,r,i) * g(q,p,a,c) / [(E_corr + e_r + e_i - e_b - e_c)] * [(E_corr + e_r + e_s - e_c - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton 557bd504 contains 1 permutations ---
    # Base Term: +0.250 * g(a,b,c,d) * g(c,d,s,r) * g(q,p,a,b) / [(E_corr + e_s + e_r - e_c - e_d)] * [(E_corr + e_s + e_r - e_a - e_b)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, virt_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, virt_idx)]
    if 'int_60' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                for i_c in range(len(virt_idx)):
                    b0 = b0_full[i_a, i_b, i_c, :]
                    b1 = b1_full[i_c, :, :, :]
                    num = np.einsum('D,DHG->DGH', b0, b1, optimize=True)
                    d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] - eps_spin[virt_idx[i_c]] - eps_spin[virt_idx][:, None, None]
                    d0[np.abs(d0) < 1e-12] = 1e-12
                    num = num / d0
                    d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                    d1[np.abs(d1) < 1e-12] = 1e-12
                    num = num / d1
                    I1[i_a, i_b, :, :] += np.einsum('DGH->GH', num, optimize=True)
        _cache['int_60'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_60'][i_a, i_b, :, :]
            b2 = b2_full[:, :, i_a, i_b]
            num = np.einsum('GH,FE->EFGH', I1, b2, optimize=True)
            I2 = num
            g_temp[2][:, :, :, :] += 0.25 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_60'] -= 1
    if _use_count['int_60'] == 0:
        del _cache['int_60']
    # Projecting permutation: +0.250 * g(a,b,c,d) * g(c,d,s,r) * g(q,p,a,b) / [(E_corr + e_s + e_r - e_c - e_d)] * [(E_corr + e_s + e_r - e_a - e_b)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # --- Missing MPn Shift ---
    missing_shift = 0.0
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_occ_idx, occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx', 'l': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx', 'l': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_occ_idx, act_occ_idx)]
    num = 0.125 * np.einsum('ABFE,DCAB,FEDC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, virt_idx, virt_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, virt_idx, virt_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, virt_idx, virt_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, virt_idx, virt_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, virt_idx, virt_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, virt_idx, virt_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, virt_idx, virt_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, virt_idx, virt_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, virt_idx, act_vir_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, virt_idx, act_vir_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, virt_idx, act_vir_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, virt_idx, act_vir_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, virt_idx, act_vir_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, virt_idx, act_vir_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, virt_idx, act_vir_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, virt_idx, act_vir_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, virt_idx, act_vir_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, virt_idx, act_vir_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, virt_idx, act_vir_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, virt_idx, act_vir_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, virt_idx, act_vir_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, virt_idx, act_vir_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, virt_idx, act_vir_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, virt_idx, act_vir_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, act_vir_idx, virt_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, act_vir_idx, virt_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, act_vir_idx, virt_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, act_vir_idx, virt_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, act_vir_idx, virt_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, act_vir_idx, virt_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, act_vir_idx, virt_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, act_vir_idx, virt_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, act_vir_idx, virt_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, act_vir_idx, virt_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, act_vir_idx, virt_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, act_vir_idx, virt_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, act_vir_idx, virt_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, act_vir_idx, virt_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, act_vir_idx, virt_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, act_vir_idx, virt_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, act_vir_idx, act_vir_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, act_vir_idx, act_vir_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, act_vir_idx, act_vir_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, act_vir_idx, act_vir_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, act_vir_idx, act_vir_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, act_vir_idx, act_vir_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, act_vir_idx, act_vir_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, act_vir_idx, act_vir_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[virt_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, act_vir_idx, act_vir_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, act_vir_idx, act_vir_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, act_vir_idx, act_vir_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, act_vir_idx, act_vir_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, act_vir_idx, act_vir_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(occ_idx, act_vir_idx, act_vir_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, act_vir_idx, act_vir_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx', 'k': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_occ_idx, act_vir_idx, act_vir_idx, act_occ_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    num = 1.0 * np.einsum('EABD,BCFE,FDAC->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, :, None, None, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][:, None, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'd': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'd': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'd': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'd': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'd': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'd': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'd': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'd': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_vir_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'd': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_vir_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'd': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_vir_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'd': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_vir_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'd': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_vir_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'd': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_vir_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'd': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_vir_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'd': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_vir_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'd': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, virt_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'd': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, virt_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'd': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, virt_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'd': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, virt_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'd': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, virt_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'd': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, virt_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'd': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, virt_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'd': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, virt_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'd': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_vir_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'd': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_vir_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'd': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_vir_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'd': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_vir_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'd': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_vir_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'd': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_vir_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, virt_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'd': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_vir_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'virt_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'd': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_vir_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, virt_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'd': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, virt_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'd': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, virt_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'd': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, virt_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'd': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, virt_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'd': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, virt_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'd': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, virt_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'd': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, virt_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'virt_idx', 'd': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, virt_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'd': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_vir_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'd': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_vir_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'd': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_vir_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'd': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_vir_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'd': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_vir_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'd': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_vir_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'd': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_vir_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'virt_idx', 'c': 'act_vir_idx', 'd': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_vir_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, virt_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[virt_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'd': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, virt_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'd': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, virt_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'd': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, virt_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'd': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, virt_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'd': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, virt_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'd': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, virt_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'd': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, virt_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'virt_idx', 'd': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, virt_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(virt_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'd': 'virt_idx', 'i': 'occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_vir_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'd': 'virt_idx', 'i': 'occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_vir_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'd': 'virt_idx', 'i': 'act_occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_vir_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'd': 'virt_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_vir_idx, virt_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, virt_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[virt_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'd': 'act_vir_idx', 'i': 'occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_vir_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'd': 'act_vir_idx', 'i': 'occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_vir_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, occ_idx, act_vir_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'd': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_vir_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    # Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    # Assignment: {'a': 'act_vir_idx', 'b': 'act_vir_idx', 'c': 'act_vir_idx', 'd': 'act_vir_idx', 'i': 'act_occ_idx', 'j': 'act_occ_idx'}
    b0 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_vir_idx, act_vir_idx)]
    b1 = g_anti_spin[np.ix_(act_vir_idx, act_vir_idx, act_occ_idx, act_occ_idx)]
    b2 = g_anti_spin[np.ix_(act_occ_idx, act_occ_idx, act_vir_idx, act_vir_idx)]
    num = 0.125 * np.einsum('ABCD,CDFE,FEAB->ABCDEF', b0, b1, b2, optimize=True)
    d0 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][None, None, :, None, None, None] - eps_spin[act_vir_idx][None, None, None, :, None, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    num = num / d0
    d1 = (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[act_vir_idx][:, None, None, None, None, None] - eps_spin[act_vir_idx][None, :, None, None, None, None]
    d1[np.abs(d1) < 1e-12] = 1e-12
    num = num / d1
    missing_shift += np.einsum('ABCDEF->', num, optimize=True)
    g_eff[0] += missing_shift
    return g_eff

def eval_pt3_v3(g_anti_spin, h1_spin, eps_spin, occ_idx, act_idx, virt_idx, act_occ_idx, act_vir_idx, n_spin_orbs, shift=None):
    import numpy as np
    g_eff = {n: np.zeros((len(act_idx),) * (2*n)) for n in range(1, 4)}
    g_eff[0] = 0.0
    _cache = {}
    _use_count = {
        'int_0': 1,
        'int_1': 1,
        'int_2': 1,
        'int_3': 1,
        'int_4': 1,
        'int_5': 1,
        'int_6': 1,
        'int_7': 1,
        'int_8': 1,
        'int_9': 1,
        'int_10': 1,
        'int_11': 1,
        'int_12': 1,
        'int_13': 1,
        'int_14': 1,
        'int_15': 1,
        'int_16': 1,
        'int_17': 1,
        'int_18': 1,
        'int_19': 1,
        'int_20': 1,
        'int_21': 1,
        'int_22': 1,
        'int_23': 1,
        'int_24': 1,
        'int_25': 1,
        'int_26': 1,
        'int_27': 1,
        'int_28': 1,
        'int_29': 1,
        'int_30': 1,
        'int_31': 1,
        'int_32': 1,
        'int_33': 1,
        'int_34': 1,
        'int_35': 1,
        'int_36': 1,
        'int_37': 1,
        'int_38': 1,
        'int_39': 1,
        'int_40': 1,
        'int_41': 1,
        'int_42': 1,
        'int_43': 1,
        'int_44': 1,
        'int_45': 1,
        'int_46': 1,
        'int_47': 1,
        'int_48': 1,
        'int_49': 1,
        'int_50': 1,
        'int_51': 1,
        'int_52': 1,
        'int_53': 1,
        'int_54': 1,
        'int_55': 1,
        'int_56': 1,
        'int_57': 1,
        'int_58': 1,
        'int_59': 1,
        'int_60': 1,
        'int_61': 1,
        'int_62': 1,
        'int_63': 1,
        'int_64': 1,
        'int_65': 1,
        'int_66': 1,
        'int_67': 1,
        'int_68': 1,
        'int_69': 1,
        'int_70': 1,
        'int_71': 1,
        'int_72': 1,
        'int_73': 1,
        'int_74': 1,
        'int_75': 1,
        'int_76': 1,
        'int_77': 1,
        'int_78': 1,
        'int_79': 1,
        'int_80': 1,
        'int_81': 1,
        'int_82': 1,
        'int_83': 1,
        'int_84': 1,
        'int_85': 1,
        'int_86': 1,
        'int_87': 1,
    }

    # Sector: 0-Body | Term: +1.00 * h1(i,i)
    b0 = h1_spin[np.ix_(occ_idx, occ_idx)]
    g_eff[0] += 1.0 * np.einsum('AA->', b0, optimize=True)

    # Sector: 0-Body | Term: +0.50 * g(j,i,j,i)
    b0 = g_anti_spin[np.ix_(occ_idx, occ_idx, occ_idx, occ_idx)]
    g_eff[0] += 0.5 * np.einsum('BABA->', b0, optimize=True)

    # Sector: 1-Body | Term: +1.00 * h1(p,q)
    b0 = h1_spin[np.ix_(act_idx, act_idx)]
    g_eff[1] += 1.0 * np.einsum('AB->AB', b0, optimize=True)

    # Sector: 1-Body | Term: -1.00 * g(i,p,q,i)
    b0 = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, occ_idx)]
    g_eff[1] += -1.0 * np.einsum('ABCA->BC', b0, optimize=True)

    # Sector: 2-Body | Term: +1.00 * g(q,p,s,r)
    b0 = g_anti_spin[np.ix_(act_idx, act_idx, act_idx, act_idx)]
    g_eff[2] += 1.0 * np.einsum('BADC->ABCD', b0, optimize=True)

    # --- Skeleton 94bc3a09 contains 1 permutations ---
    # Base Term: +0.250 * g(a,b,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_a - e_b)]
    g_temp = {0: np.zeros((len(act_idx),) * 0)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    if 'int_0' not in _cache:
        I1 = np.zeros(())
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            b0 = b0 / d0
            num = np.einsum('BDC,DCB->BCD', b0, b1, optimize=True)
            I1 += np.einsum('BCD->', num, optimize=True)
        _cache['int_0'] = I1
    I1 = _cache['int_0']
    g_temp[0] += 0.25 * np.einsum('->', I1, optimize=True)
    _use_count['int_0'] -= 1
    if _use_count['int_0'] == 0:
        del _cache['int_0']
    # Projecting permutation: +0.250 * g(a,b,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_a - e_b)]
    g_eff[0] += 1.000000000000 * g_temp[0]

    # --- Skeleton b3d347c4 contains 1 permutations ---
    # Base Term: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    g_temp = {0: np.zeros((len(act_idx),) * 0)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, occ_idx, occ_idx)]
    if 'int_1' not in _cache:
        I1 = np.zeros((len(occ_idx), len(occ_idx), len(occ_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BFE,DCB->BCDEF', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None, None, None]
            num = num / d0
            I1 += np.einsum('BCDEF->CDEF', num, optimize=True)
        _cache['int_1'] = I1
    I1 = _cache['int_1']
    b2 = b2_full
    num = np.einsum('CDEF,FEDC->CDEF', I1, b2, optimize=True)
    I2 = np.einsum('CDEF->', num, optimize=True)
    g_temp[0] += 0.125 * np.einsum('->', I2, optimize=True)
    _use_count['int_1'] -= 1
    if _use_count['int_1'] == 0:
        del _cache['int_1']
    # Projecting permutation: +0.1250 * g(a,b,l,k) * g(j,i,a,b) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_b)] * [(E_corr + e_j + e_i - e_a - e_b)]
    g_eff[0] += 1.000000000000 * g_temp[0]

    # --- Skeleton ccea8972 contains 1 permutations ---
    # Base Term: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    g_temp = {0: np.zeros((len(act_idx),) * 0)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    if 'int_2' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[:, i_a, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('ED,CFE->CDEF', b0, b1, optimize=True)
                d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx][:, None, None, None] - eps_spin[virt_idx[i_a]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :] += np.einsum('CDEF->CDF', num, optimize=True)
        _cache['int_2'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_2'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDF,FDC->CDF', I1, b2, optimize=True)
        I2 = np.einsum('CDF->', num, optimize=True)
        g_temp[0] += 1.0 * np.einsum('->', I2, optimize=True)
    _use_count['int_2'] -= 1
    if _use_count['int_2'] == 0:
        del _cache['int_2']
    # Projecting permutation: +1.00 * g(j,a,b,i) * g(b,c,k,j) * g(k,i,a,c) / [(E_corr + e_k + e_j - e_b - e_c)] * [(E_corr + e_k + e_i - e_c - e_a)]
    g_eff[0] += 1.000000000000 * g_temp[0]

    # --- Skeleton fa4ffacc contains 1 permutations ---
    # Base Term: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    g_temp = {0: np.zeros((len(act_idx),) * 0)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, virt_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    if 'int_3' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                for i_c in range(len(virt_idx)):
                    b0 = b0_full[i_a, i_b, i_c, :]
                    b1 = b1_full[i_c, :, :, :]
                    num = np.einsum('D,DFE->DEF', b0, b1, optimize=True)
                    d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] - eps_spin[virt_idx[i_c]] - eps_spin[virt_idx][:, None, None]
                    d0[np.abs(d0) < 1e-12] = 1e-12
                    num = num / d0
                    d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                    d1[np.abs(d1) < 1e-12] = 1e-12
                    num = num / d1
                    I1[i_a, i_b, :, :] += np.einsum('DEF->EF', num, optimize=True)
        _cache['int_3'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_3'][i_a, i_b, :, :]
            b2 = b2_full[:, :, i_a, i_b]
            num = np.einsum('EF,FE->EF', I1, b2, optimize=True)
            I2 = np.einsum('EF->', num, optimize=True)
            g_temp[0] += 0.125 * np.einsum('->', I2, optimize=True)
    _use_count['int_3'] -= 1
    if _use_count['int_3'] == 0:
        del _cache['int_3']
    # Projecting permutation: +0.1250 * g(a,b,c,d) * g(c,d,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_d)] * [(E_corr + e_j + e_i - e_a - e_b)]
    g_eff[0] += 1.000000000000 * g_temp[0]

    # --- Skeleton 75ad85ad contains 1 permutations ---
    # Base Term: -0.50 * g(a,p,j,i) * g(j,i,a,q) / [(E_corr + e_j + e_i - e_a - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0 = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_4' not in _cache:
        b0 = b0
        b1 = b1
        d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[act_idx][None, :, None, None]
        d0[np.abs(d0) < 1e-12] = 1e-12
        b0 = b0 / d0
        num = np.einsum('ADCB,CBAE->ABCDE', b0, b1, optimize=True)
        I1 = np.einsum('ABCDE->DE', num, optimize=True)
        _cache['int_4'] = I1
    I1 = _cache['int_4']
    g_temp[1][:, :] += -0.5 * np.einsum('DE->DE', I1, optimize=True)
    _use_count['int_4'] -= 1
    if _use_count['int_4'] == 0:
        del _cache['int_4']
    # Projecting permutation: -0.50 * g(a,p,j,i) * g(j,i,a,q) / [(E_corr + e_j + e_i - e_a - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 3421b2e9 contains 1 permutations ---
    # Base Term: -0.50 * g(a,b,q,i) * g(i,p,a,b) / [(E_corr + e_q + e_i - e_a - e_b)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_5' not in _cache:
        I1 = np.zeros((len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            b0 = b0 / d0
            num = np.einsum('BEC,CDB->BCDE', b0, b1, optimize=True)
            I1 += np.einsum('BCDE->DE', num, optimize=True)
        _cache['int_5'] = I1
    I1 = _cache['int_5']
    g_temp[1][:, :] += -0.5 * np.einsum('DE->DE', I1, optimize=True)
    _use_count['int_5'] -= 1
    if _use_count['int_5'] == 0:
        del _cache['int_5']
    # Projecting permutation: -0.50 * g(a,b,q,i) * g(i,p,a,b) / [(E_corr + e_q + e_i - e_a - e_b)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton dfca8151 contains 1 permutations ---
    # Base Term: -0.250 * g(a,p,l,k) * g(j,i,a,q) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_p)] * [(E_corr + e_j + e_i - e_a - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0 = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_6' not in _cache:
        b0 = b0
        b1 = b1
        d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[act_idx][None, :, None, None]
        d1[np.abs(d1) < 1e-12] = 1e-12
        b0 = b0 / d1
        num = np.einsum('AFED,EDCB->ABCDEF', b0, b1, optimize=True)
        d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, None, :]
        num = num / d0
        I1 = np.einsum('ABCDEF->ABCF', num, optimize=True)
        _cache['int_6'] = I1
    I1 = _cache['int_6']
    num = np.einsum('ABCF,CBAG->ABCFG', I1, b2, optimize=True)
    I2 = np.einsum('ABCFG->FG', num, optimize=True)
    g_temp[1][:, :] += -0.25 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_6'] -= 1
    if _use_count['int_6'] == 0:
        del _cache['int_6']
    # Projecting permutation: -0.250 * g(a,p,l,k) * g(j,i,a,q) * g(l,k,j,i) / [(E_corr + e_l + e_k - e_a - e_p)] * [(E_corr + e_j + e_i - e_a - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton ff24b53d contains 1 permutations ---
    # Base Term: -0.50 * g(k,a,j,i) * g(b,p,q,k) * g(j,i,a,b) / [(E_corr + e_q + e_k - e_b - e_p)] * [(E_corr + e_q + e_j + e_i - e_b - e_p - e_a)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    if 'int_7' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, i_a, :]
            num = np.einsum('EDC,DCB->BCDE', b0, b1, optimize=True)
            I1[i_a, :, :, :, :] += num
        _cache['int_7'] = I1
    b2 = b2_full
    d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_7'][i_a, :, :, :, :]
        num = np.einsum('BCDE,BFGE->BCDEFG', I1, b2, optimize=True)
        num = num / d0
        d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[virt_idx[i_a]]
        d1[np.abs(d1) < 1e-12] = 1e-12
        num = num / d1
        I2 = np.einsum('BCDEFG->FG', num, optimize=True)
        g_temp[1][:, :] += -0.5 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_7'] -= 1
    if _use_count['int_7'] == 0:
        del _cache['int_7']
    # Projecting permutation: -0.50 * g(k,a,j,i) * g(b,p,q,k) * g(j,i,a,b) / [(E_corr + e_q + e_k - e_b - e_p)] * [(E_corr + e_q + e_j + e_i - e_b - e_p - e_a)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 0ab601c6 contains 1 permutations ---
    # Base Term: -0.250 * g(a,b,q,k) * g(k,p,j,i) * g(j,i,a,b) / [(E_corr + e_q + e_k - e_a - e_b)] * [(E_corr + e_q + e_j + e_i - e_a - e_b - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    if 'int_8' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BGE,EFDC->BCDEFG', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None]
            num = num / d0
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEFG->BCDFG', num, optimize=True)
        _cache['int_8'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_8'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BCDFG,DCB->BCDFG', I1, b2, optimize=True)
        I2 = np.einsum('BCDFG->FG', num, optimize=True)
        g_temp[1][:, :] += -0.25 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_8'] -= 1
    if _use_count['int_8'] == 0:
        del _cache['int_8']
    # Projecting permutation: -0.250 * g(a,b,q,k) * g(k,p,j,i) * g(j,i,a,b) / [(E_corr + e_q + e_k - e_a - e_b)] * [(E_corr + e_q + e_j + e_i - e_a - e_b - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 1d17f101 contains 1 permutations ---
    # Base Term: -1.00 * g(j,a,b,i) * g(b,p,k,j) * g(k,i,a,q) / [(E_corr + e_k + e_j - e_b - e_p)] * [(E_corr + e_k + e_i - e_p - e_a)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_9' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('DBC,BFED->BCDEF', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[act_idx][None, None, None, None, :] - eps_spin[virt_idx[i_a]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :] += np.einsum('BCDEF->CEF', num, optimize=True)
        _cache['int_9'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_9'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CEF,ECG->CEFG', I1, b2, optimize=True)
        I2 = np.einsum('CEFG->FG', num, optimize=True)
        g_temp[1][:, :] += -1.0 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_9'] -= 1
    if _use_count['int_9'] == 0:
        del _cache['int_9']
    # Projecting permutation: -1.00 * g(j,a,b,i) * g(b,p,k,j) * g(k,i,a,q) / [(E_corr + e_k + e_j - e_b - e_p)] * [(E_corr + e_k + e_i - e_p - e_a)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton d70ea4ec contains 1 permutations ---
    # Base Term: +1.00 * g(j,a,q,i) * g(b,p,k,j) * g(k,i,a,b) / [(E_corr + e_k + e_j - e_b - e_p)] * [(E_corr + e_k + e_q + e_i - e_b - e_p - e_a)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    if 'int_10' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('DGC,BFED->BCDEFG', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[virt_idx[i_a]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEFG->BCEFG', num, optimize=True)
        _cache['int_10'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_10'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BCEFG,ECB->BCEFG', I1, b2, optimize=True)
        I2 = np.einsum('BCEFG->FG', num, optimize=True)
        g_temp[1][:, :] += 1.0 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_10'] -= 1
    if _use_count['int_10'] == 0:
        del _cache['int_10']
    # Projecting permutation: +1.00 * g(j,a,q,i) * g(b,p,k,j) * g(k,i,a,b) / [(E_corr + e_k + e_j - e_b - e_p)] * [(E_corr + e_k + e_q + e_i - e_b - e_p - e_a)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton f95c0dde contains 1 permutations ---
    # Base Term: +1.00 * g(a,b,k,j) * g(j,p,a,i) * g(k,i,b,q) / [(E_corr + e_k + e_j - e_a - e_b)] * [(E_corr + e_k + e_i - e_b - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_11' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BED,DFC->BCDEF', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :]
            num = num / d0
            I1 += np.einsum('BCDEF->BCEF', num, optimize=True)
        _cache['int_11'] = I1
    I1 = _cache['int_11']
    b2 = b2_full
    num = np.einsum('BCEF,ECBG->BCEFG', I1, b2, optimize=True)
    I2 = np.einsum('BCEFG->FG', num, optimize=True)
    g_temp[1][:, :] += 1.0 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_11'] -= 1
    if _use_count['int_11'] == 0:
        del _cache['int_11']
    # Projecting permutation: +1.00 * g(a,b,k,j) * g(j,p,a,i) * g(k,i,b,q) / [(E_corr + e_k + e_j - e_a - e_b)] * [(E_corr + e_k + e_i - e_b - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton eb8ae6c7 contains 1 permutations ---
    # Base Term: +0.50 * g(a,b,k,j) * g(j,p,q,i) * g(k,i,a,b) / [(E_corr + e_k + e_j - e_a - e_b)] * [(E_corr + e_k + e_q + e_i - e_a - e_b - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, occ_idx)]
    if 'int_12' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BED,ECB->BCDE', b0, b1, optimize=True)
            I1[i_a, :, :, :, :] += num
        _cache['int_12'] = I1
    b2 = b2_full
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_12'][i_a, :, :, :, :]
        num = np.einsum('BCDE,DFGC->BCDEFG', I1, b2, optimize=True)
        d0 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None]
        d0[np.abs(d0) < 1e-12] = 1e-12
        num = num / d0
        I2 = np.einsum('BCDEFG->FG', num, optimize=True)
        g_temp[1][:, :] += 0.5 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_12'] -= 1
    if _use_count['int_12'] == 0:
        del _cache['int_12']
    # Projecting permutation: +0.50 * g(a,b,k,j) * g(j,p,q,i) * g(k,i,a,b) / [(E_corr + e_k + e_j - e_a - e_b)] * [(E_corr + e_k + e_q + e_i - e_a - e_b - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 05800a39 contains 1 permutations ---
    # Base Term: -0.50 * g(a,b,c,i) * g(c,p,q,j) * g(j,i,a,b) / [(E_corr + e_q + e_j - e_c - e_p)] * [(E_corr + e_q + e_j + e_i - e_p - e_a - e_b)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    if 'int_13' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, i_b, :, :]
                b1 = b1_full[:, :, i_a, i_b]
                num = np.einsum('CD,ED->CDE', b0, b1, optimize=True)
                I1[i_a, i_b, :, :, :] += num
        _cache['int_13'] = I1
    b2 = b2_full
    d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None]
    d0[np.abs(d0) < 1e-12] = 1e-12
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_13'][i_a, i_b, :, :, :]
            num = np.einsum('CDE,CFGE->CDEFG', I1, b2, optimize=True)
            num = num / d0
            d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[act_idx][None, None, None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I2 = np.einsum('CDEFG->FG', num, optimize=True)
            g_temp[1][:, :] += -0.5 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_13'] -= 1
    if _use_count['int_13'] == 0:
        del _cache['int_13']
    # Projecting permutation: -0.50 * g(a,b,c,i) * g(c,p,q,j) * g(j,i,a,b) / [(E_corr + e_q + e_j - e_c - e_p)] * [(E_corr + e_q + e_j + e_i - e_p - e_a - e_b)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 3d7c4c8a contains 1 permutations ---
    # Base Term: +1.00 * g(a,p,b,i) * g(b,c,q,j) * g(j,i,a,c) / [(E_corr + e_q + e_j - e_b - e_c)] * [(E_corr + e_q + e_j + e_i - e_c - e_a - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    if 'int_14' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('FD,CGE->CDEFG', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, :, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :, :, :] += num
        _cache['int_14'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_14'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDEFG,EDC->CDEFG', I1, b2, optimize=True)
        I2 = np.einsum('CDEFG->FG', num, optimize=True)
        g_temp[1][:, :] += 1.0 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_14'] -= 1
    if _use_count['int_14'] == 0:
        del _cache['int_14']
    # Projecting permutation: +1.00 * g(a,p,b,i) * g(b,c,q,j) * g(j,i,a,c) / [(E_corr + e_q + e_j - e_b - e_c)] * [(E_corr + e_q + e_j + e_i - e_c - e_a - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 3c92a970 contains 1 permutations ---
    # Base Term: -0.250 * g(a,b,c,q) * g(c,p,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_p)] * [(E_corr + e_j + e_i + e_q - e_p - e_a - e_b)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    if 'int_15' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, i_b, :, :]
                b1 = b1_full[:, :, :, :]
                num = np.einsum('CG,CFED->CDEFG', b0, b1, optimize=True)
                d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] - eps_spin[act_idx][None, None, None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, i_b, :, :, :, :] += np.einsum('CDEFG->DEFG', num, optimize=True)
        _cache['int_15'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_15'][i_a, i_b, :, :, :, :]
            b2 = b2_full[:, :, i_a, i_b]
            num = np.einsum('DEFG,ED->DEFG', I1, b2, optimize=True)
            I2 = np.einsum('DEFG->FG', num, optimize=True)
            g_temp[1][:, :] += -0.25 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_15'] -= 1
    if _use_count['int_15'] == 0:
        del _cache['int_15']
    # Projecting permutation: -0.250 * g(a,b,c,q) * g(c,p,j,i) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_c - e_p)] * [(E_corr + e_j + e_i + e_q - e_p - e_a - e_b)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton d8e26961 contains 1 permutations ---
    # Base Term: -0.250 * g(a,p,b,c) * g(b,c,j,i) * g(j,i,a,q) / [(E_corr + e_j + e_i - e_b - e_c)] * [(E_corr + e_j + e_i - e_a - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, virt_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_16' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('FC,CED->CDEF', b0, b1, optimize=True)
                d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, :]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :] += np.einsum('CDEF->DEF', num, optimize=True)
        _cache['int_16'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_16'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('DEF,EDG->DEFG', I1, b2, optimize=True)
        I2 = np.einsum('DEFG->FG', num, optimize=True)
        g_temp[1][:, :] += -0.25 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_16'] -= 1
    if _use_count['int_16'] == 0:
        del _cache['int_16']
    # Projecting permutation: -0.250 * g(a,p,b,c) * g(b,c,j,i) * g(j,i,a,q) / [(E_corr + e_j + e_i - e_b - e_c)] * [(E_corr + e_j + e_i - e_a - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 24b668bc contains 1 permutations ---
    # Base Term: +0.50 * g(a,p,b,q) * g(b,c,j,i) * g(j,i,a,c) / [(E_corr + e_j + e_i - e_b - e_c)] * [(E_corr + e_j + e_i + e_q - e_c - e_a - e_p)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, act_idx)]
    if 'int_17' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_b, :, :, :]
                b1 = b1_full[:, :, i_a, :]
                d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                b0 = b0 / d1
                num = np.einsum('CED,EDC->CDE', b0, b1, optimize=True)
                I1[i_a, i_b, :, :, :] += num
        _cache['int_17'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_17'][i_a, i_b, :, :, :]
            b2 = b2_full[i_a, :, i_b, :]
            num = np.einsum('CDE,FG->CDEFG', I1, b2, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, :, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            I2 = np.einsum('CDEFG->FG', num, optimize=True)
            g_temp[1][:, :] += 0.5 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_17'] -= 1
    if _use_count['int_17'] == 0:
        del _cache['int_17']
    # Projecting permutation: +0.50 * g(a,p,b,q) * g(b,c,j,i) * g(j,i,a,c) / [(E_corr + e_j + e_i - e_b - e_c)] * [(E_corr + e_j + e_i + e_q - e_c - e_a - e_p)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 899ed865 contains 1 permutations ---
    # Base Term: -0.50 * g(a,b,k,j) * g(i,p,b,q) * g(k,j,a,i) / [(E_corr + e_k + e_j - e_a - e_b)] * [(E_corr + e_i - e_b)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_18' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BED,EDC->BCDE', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 1 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx][:, None, None, None]
            num = num / d0
            I1 += np.einsum('BCDE->BC', num, optimize=True)
        _cache['int_18'] = I1
    I1 = _cache['int_18']
    b2 = b2_full
    num = np.einsum('BC,CFBG->BCFG', I1, b2, optimize=True)
    I2 = np.einsum('BCFG->FG', num, optimize=True)
    g_temp[1][:, :] += -0.5 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_18'] -= 1
    if _use_count['int_18'] == 0:
        del _cache['int_18']
    # Projecting permutation: -0.50 * g(a,b,k,j) * g(i,p,b,q) * g(k,j,a,i) / [(E_corr + e_k + e_j - e_a - e_b)] * [(E_corr + e_i - e_b)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 2330cab2 contains 1 permutations ---
    # Base Term: -0.250 * g(a,b,k,j) * g(i,p,a,b) * g(k,j,q,i) / [(E_corr + e_k + e_j - e_a - e_b)] * [(E_corr + e_q + e_i - e_a - e_b)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_19' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BED,EDGC->BCDEG', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None, None, None]
            num = num / d0
            I1[i_a, :, :, :] += np.einsum('BCDEG->BCG', num, optimize=True)
        _cache['int_19'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_19'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BCG,CFB->BCFG', I1, b2, optimize=True)
        I2 = np.einsum('BCFG->FG', num, optimize=True)
        g_temp[1][:, :] += -0.25 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_19'] -= 1
    if _use_count['int_19'] == 0:
        del _cache['int_19']
    # Projecting permutation: -0.250 * g(a,b,k,j) * g(i,p,a,b) * g(k,j,q,i) / [(E_corr + e_k + e_j - e_a - e_b)] * [(E_corr + e_q + e_i - e_a - e_b)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 4aefda64 contains 1 permutations ---
    # Base Term: -1.00 * g(j,a,b,i) * g(b,c,q,j) * g(i,p,a,c) / [(E_corr + e_q + e_j - e_b - e_c)] * [(E_corr + e_q + e_i - e_c - e_a)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_20' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[:, i_a, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('ED,CGE->CDEG', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx][:, None, None, None] - eps_spin[virt_idx[i_a]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :] += np.einsum('CDEG->CDG', num, optimize=True)
        _cache['int_20'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_20'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDG,DFC->CDFG', I1, b2, optimize=True)
        I2 = np.einsum('CDFG->FG', num, optimize=True)
        g_temp[1][:, :] += -1.0 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_20'] -= 1
    if _use_count['int_20'] == 0:
        del _cache['int_20']
    # Projecting permutation: -1.00 * g(j,a,b,i) * g(b,c,q,j) * g(i,p,a,c) / [(E_corr + e_q + e_j - e_b - e_c)] * [(E_corr + e_q + e_i - e_c - e_a)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton 1a911f5e contains 1 permutations ---
    # Base Term: -0.50 * g(i,a,b,c) * g(b,c,j,i) * g(j,p,a,q) / [(E_corr + e_j + e_i - e_b - e_c)] * [(E_corr + e_j - e_a)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, virt_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_21' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[:, i_a, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('DC,CED->CDE', b0, b1, optimize=True)
                d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[occ_idx] + (shift / 1 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :] += np.einsum('CDE->E', num, optimize=True)
        _cache['int_21'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_21'][i_a, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('E,EFG->EFG', I1, b2, optimize=True)
        I2 = np.einsum('EFG->FG', num, optimize=True)
        g_temp[1][:, :] += -0.5 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_21'] -= 1
    if _use_count['int_21'] == 0:
        del _cache['int_21']
    # Projecting permutation: -0.50 * g(i,a,b,c) * g(b,c,j,i) * g(j,p,a,q) / [(E_corr + e_j + e_i - e_b - e_c)] * [(E_corr + e_j - e_a)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton ae3833b9 contains 1 permutations ---
    # Base Term: +1.00 * g(i,a,b,q) * g(b,c,j,i) * g(j,p,a,c) / [(E_corr + e_j + e_i - e_b - e_c)] * [(E_corr + e_j + e_q - e_c - e_a)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_22' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[:, i_a, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('DG,CED->CDEG', b0, b1, optimize=True)
                d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[virt_idx[i_a]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :] += np.einsum('CDEG->CEG', num, optimize=True)
        _cache['int_22'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_22'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CEG,EFC->CEFG', I1, b2, optimize=True)
        I2 = np.einsum('CEFG->FG', num, optimize=True)
        g_temp[1][:, :] += 1.0 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_22'] -= 1
    if _use_count['int_22'] == 0:
        del _cache['int_22']
    # Projecting permutation: +1.00 * g(i,a,b,q) * g(b,c,j,i) * g(j,p,a,c) / [(E_corr + e_j + e_i - e_b - e_c)] * [(E_corr + e_j + e_q - e_c - e_a)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton b4276dfe contains 1 permutations ---
    # Base Term: -0.250 * g(a,b,c,d) * g(c,d,q,i) * g(i,p,a,b) / [(E_corr + e_q + e_i - e_c - e_d)] * [(E_corr + e_q + e_i - e_a - e_b)]
    g_temp = {1: np.zeros((len(act_idx),) * 2)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, virt_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_23' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                for i_c in range(len(virt_idx)):
                    b0 = b0_full[i_a, i_b, i_c, :]
                    b1 = b1_full[i_c, :, :, :]
                    num = np.einsum('D,DGE->DEG', b0, b1, optimize=True)
                    d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] - eps_spin[virt_idx[i_c]] - eps_spin[virt_idx][:, None, None]
                    d0[np.abs(d0) < 1e-12] = 1e-12
                    num = num / d0
                    d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                    d1[np.abs(d1) < 1e-12] = 1e-12
                    num = num / d1
                    I1[i_a, i_b, :, :] += np.einsum('DEG->EG', num, optimize=True)
        _cache['int_23'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_23'][i_a, i_b, :, :]
            b2 = b2_full[:, :, i_a, i_b]
            num = np.einsum('EG,EF->EFG', I1, b2, optimize=True)
            I2 = np.einsum('EFG->FG', num, optimize=True)
            g_temp[1][:, :] += -0.25 * np.einsum('FG->FG', I2, optimize=True)
    _use_count['int_23'] -= 1
    if _use_count['int_23'] == 0:
        del _cache['int_23']
    # Projecting permutation: -0.250 * g(a,b,c,d) * g(c,d,q,i) * g(i,p,a,b) / [(E_corr + e_q + e_i - e_c - e_d)] * [(E_corr + e_q + e_i - e_a - e_b)]
    g_eff[1] += 1.000000000000 * np.transpose(g_temp[1], (0, 1))

    # --- Skeleton b2162412 contains 1 permutations ---
    # Base Term: +0.50 * g(j,i,s,r) * g(q,p,j,i) / [(E_corr + e_j + e_i - e_q - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, act_idx)]
    b1 = g_anti_spin[np.ix_(act_idx, act_idx, occ_idx, occ_idx)]
    if 'int_24' not in _cache:
        b0 = b0
        b1 = b1
        num = np.einsum('BAFE,DCBA->ABCDEF', b0, b1, optimize=True)
        d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[act_idx][None, None, :, None, None, None]
        d0[np.abs(d0) < 1e-12] = 1e-12
        num = num / d0
        I1 = np.einsum('ABCDEF->CDEF', num, optimize=True)
        _cache['int_24'] = I1
    I1 = _cache['int_24']
    g_temp[2][:, :, :, :] += 0.5 * np.einsum('CDEF->CDEF', I1, optimize=True)
    _use_count['int_24'] -= 1
    if _use_count['int_24'] == 0:
        del _cache['int_24']
    # Projecting permutation: +0.50 * g(j,i,s,r) * g(q,p,j,i) / [(E_corr + e_j + e_i - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton fb0c734b contains 4 permutations ---
    # Base Term: +1.00 * g(a,p,s,i) * g(i,q,a,r) / [(E_corr + e_s + e_i - e_a - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0 = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_25' not in _cache:
        b0 = b0
        b1 = b1
        d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[act_idx][None, :, None, None]
        d0[np.abs(d0) < 1e-12] = 1e-12
        b0 = b0 / d0
        num = np.einsum('ACFB,BDAE->ABCDEF', b0, b1, optimize=True)
        I1 = np.einsum('ABCDEF->CDEF', num, optimize=True)
        _cache['int_25'] = I1
    I1 = _cache['int_25']
    g_temp[2][:, :, :, :] += 1.0 * np.einsum('CDEF->CDEF', I1, optimize=True)
    _use_count['int_25'] -= 1
    if _use_count['int_25'] == 0:
        del _cache['int_25']
    # Projecting permutation: +1.00 * g(a,p,s,i) * g(i,q,a,r) / [(E_corr + e_s + e_i - e_a - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -1.00 * g(a,p,r,i) * g(i,q,a,s) / [(E_corr + e_r + e_i - e_a - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: -1.00 * g(a,q,s,i) * g(i,p,a,r) / [(E_corr + e_s + e_i - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: +1.00 * g(a,q,r,i) * g(i,p,a,s) / [(E_corr + e_r + e_i - e_a - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton d6997bae contains 1 permutations ---
    # Base Term: +0.50 * g(a,b,s,r) * g(q,p,a,b) / [(E_corr + e_s + e_r - e_a - e_b)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, virt_idx)]
    if 'int_26' not in _cache:
        I1 = np.zeros((len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            b0 = b0 / d0
            num = np.einsum('BFE,DCB->BCDEF', b0, b1, optimize=True)
            I1 += np.einsum('BCDEF->CDEF', num, optimize=True)
        _cache['int_26'] = I1
    I1 = _cache['int_26']
    g_temp[2][:, :, :, :] += 0.5 * np.einsum('CDEF->CDEF', I1, optimize=True)
    _use_count['int_26'] -= 1
    if _use_count['int_26'] == 0:
        del _cache['int_26']
    # Projecting permutation: +0.50 * g(a,b,s,r) * g(q,p,a,b) / [(E_corr + e_s + e_r - e_a - e_b)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton 0ca95203 contains 1 permutations ---
    # Base Term: +0.250 * g(j,i,s,r) * g(l,k,j,i) * g(q,p,l,k) / [(E_corr + e_l + e_k - e_q - e_p)] * [(E_corr + e_j + e_i - e_q - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0 = g_anti_spin[np.ix_(occ_idx, occ_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(act_idx, act_idx, occ_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, act_idx)]
    if 'int_27' not in _cache:
        b0 = b0
        b1 = b1
        num = np.einsum('DCBA,FEDC->ABCDEF', b0, b1, optimize=True)
        d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_idx][None, None, None, None, None, :] - eps_spin[act_idx][None, None, None, None, :, None]
        d0[np.abs(d0) < 1e-12] = 1e-12
        num = num / d0
        d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, None, :] - eps_spin[act_idx][None, None, None, None, :, None]
        d1[np.abs(d1) < 1e-12] = 1e-12
        num = num / d1
        I1 = np.einsum('ABCDEF->ABEF', num, optimize=True)
        _cache['int_27'] = I1
    I1 = _cache['int_27']
    num = np.einsum('ABEF,BAHG->ABEFGH', I1, b2, optimize=True)
    I2 = np.einsum('ABEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += 0.25 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_27'] -= 1
    if _use_count['int_27'] == 0:
        del _cache['int_27']
    # Projecting permutation: +0.250 * g(j,i,s,r) * g(l,k,j,i) * g(q,p,l,k) / [(E_corr + e_l + e_k - e_q - e_p)] * [(E_corr + e_j + e_i - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton eb51cbc6 contains 2 permutations ---
    # Base Term: -0.50 * g(k,a,j,i) * g(j,i,a,r) * g(q,p,s,k) / [(E_corr + e_s + e_k - e_q - e_p)] * [(E_corr + e_s + e_j + e_i - e_q - e_p - e_a)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_28' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('DCB,FEHD->BCDEFH', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[act_idx][None, None, None, :, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[virt_idx[i_a]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEFH->BCEFH', num, optimize=True)
        _cache['int_28'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_28'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BCEFH,CBG->BCEFGH', I1, b2, optimize=True)
        I2 = np.einsum('BCEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_28'] -= 1
    if _use_count['int_28'] == 0:
        del _cache['int_28']
    # Projecting permutation: -0.50 * g(k,a,j,i) * g(j,i,a,r) * g(q,p,s,k) / [(E_corr + e_s + e_k - e_q - e_p)] * [(E_corr + e_s + e_j + e_i - e_q - e_p - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +0.50 * g(k,a,j,i) * g(j,i,a,s) * g(q,p,r,k) / [(E_corr + e_r + e_k - e_q - e_p)] * [(E_corr + e_r + e_j + e_i - e_q - e_p - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton 36618024 contains 4 permutations ---
    # Base Term: +0.50 * g(a,p,s,k) * g(j,i,a,r) * g(k,q,j,i) / [(E_corr + e_s + e_k - e_a - e_p)] * [(E_corr + e_s + e_j + e_i - e_a - e_p - e_q)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_29' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('EHD,DFCB->BCDEFH', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[:, None, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[act_idx][None, None, None, None, :, None]
            num = num / d0
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEFH->BCEFH', num, optimize=True)
        _cache['int_29'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_29'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BCEFH,CBG->BCEFGH', I1, b2, optimize=True)
        I2 = np.einsum('BCEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += 0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_29'] -= 1
    if _use_count['int_29'] == 0:
        del _cache['int_29']
    # Projecting permutation: +0.50 * g(a,p,s,k) * g(j,i,a,r) * g(k,q,j,i) / [(E_corr + e_s + e_k - e_a - e_p)] * [(E_corr + e_s + e_j + e_i - e_a - e_p - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -0.50 * g(a,p,r,k) * g(j,i,a,s) * g(k,q,j,i) / [(E_corr + e_r + e_k - e_a - e_p)] * [(E_corr + e_r + e_j + e_i - e_a - e_p - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: -0.50 * g(a,q,s,k) * g(j,i,a,r) * g(k,p,j,i) / [(E_corr + e_s + e_k - e_a - e_q)] * [(E_corr + e_s + e_j + e_i - e_a - e_q - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: +0.50 * g(a,q,r,k) * g(j,i,a,s) * g(k,p,j,i) / [(E_corr + e_r + e_k - e_a - e_q)] * [(E_corr + e_r + e_j + e_i - e_a - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 36883c82 contains 2 permutations ---
    # Base Term: +1.00 * g(j,a,s,i) * g(k,i,a,r) * g(q,p,k,j) / [(E_corr + e_k + e_j - e_q - e_p)] * [(E_corr + e_k + e_s + e_i - e_q - e_p - e_a)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_30' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('CHB,FEDC->BCDEFH', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[act_idx][None, None, None, :, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[virt_idx[i_a]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEFH->BDEFH', num, optimize=True)
        _cache['int_30'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_30'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BDEFH,DBG->BDEFGH', I1, b2, optimize=True)
        I2 = np.einsum('BDEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += 1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_30'] -= 1
    if _use_count['int_30'] == 0:
        del _cache['int_30']
    # Projecting permutation: +1.00 * g(j,a,s,i) * g(k,i,a,r) * g(q,p,k,j) / [(E_corr + e_k + e_j - e_q - e_p)] * [(E_corr + e_k + e_s + e_i - e_q - e_p - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -1.00 * g(j,a,r,i) * g(k,i,a,s) * g(q,p,k,j) / [(E_corr + e_k + e_j - e_q - e_p)] * [(E_corr + e_k + e_r + e_i - e_q - e_p - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton 337e69f1 contains 2 permutations ---
    # Base Term: +1.00 * g(a,p,k,j) * g(k,i,s,r) * g(j,q,a,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_k + e_i - e_p - e_q)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0 = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, act_idx)]
    if 'int_31' not in _cache:
        b0 = b0
        b1 = b1
        d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[act_idx][None, :, None, None]
        d1[np.abs(d1) < 1e-12] = 1e-12
        b0 = b0 / d1
        num = np.einsum('AEDC,CFAB->ABCDEF', b0, b1, optimize=True)
        d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[act_idx][None, None, None, None, None, :]
        num = num / d0
        I1 = np.einsum('ABCDEF->BDEF', num, optimize=True)
        _cache['int_31'] = I1
    I1 = _cache['int_31']
    num = np.einsum('BDEF,DBHG->BDEFGH', I1, b2, optimize=True)
    I2 = np.einsum('BDEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += 1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_31'] -= 1
    if _use_count['int_31'] == 0:
        del _cache['int_31']
    # Projecting permutation: +1.00 * g(a,p,k,j) * g(k,i,s,r) * g(j,q,a,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_k + e_i - e_p - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -1.00 * g(a,q,k,j) * g(k,i,s,r) * g(j,p,a,i) / [(E_corr + e_k + e_j - e_a - e_q)] * [(E_corr + e_k + e_i - e_q - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))

    # --- Skeleton 2bad60a4 contains 4 permutations ---
    # Base Term: -1.00 * g(a,p,k,j) * g(k,i,a,r) * g(j,q,s,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_k + e_s + e_i - e_a - e_p - e_q)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_32' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('EDC,CFHB->BCDEFH', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[:, None, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[act_idx][None, None, None, None, :, None]
            num = num / d0
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEFH->BDEFH', num, optimize=True)
        _cache['int_32'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_32'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BDEFH,DBG->BDEFGH', I1, b2, optimize=True)
        I2 = np.einsum('BDEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_32'] -= 1
    if _use_count['int_32'] == 0:
        del _cache['int_32']
    # Projecting permutation: -1.00 * g(a,p,k,j) * g(k,i,a,r) * g(j,q,s,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_k + e_s + e_i - e_a - e_p - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(a,p,k,j) * g(k,i,a,s) * g(j,q,r,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_k + e_r + e_i - e_a - e_p - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: +1.00 * g(a,q,k,j) * g(k,i,a,r) * g(j,p,s,i) / [(E_corr + e_k + e_j - e_a - e_q)] * [(E_corr + e_k + e_s + e_i - e_a - e_q - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: -1.00 * g(a,q,k,j) * g(k,i,a,s) * g(j,p,r,i) / [(E_corr + e_k + e_j - e_a - e_q)] * [(E_corr + e_k + e_r + e_i - e_a - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton f522c72e contains 2 permutations ---
    # Base Term: -1.00 * g(a,q,j,i) * g(b,p,s,r) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_a - e_q)] * [(E_corr + e_j + e_i + e_s + e_r - e_a - e_q - e_b - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, act_idx)]
    if 'int_33' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, :, :]
                b1 = b1_full[:, :, i_a, i_b]
                d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                b0 = b0 / d1
                num = np.einsum('FDC,DC->CDF', b0, b1, optimize=True)
                I1[i_a, i_b, :, :, :] += num
        _cache['int_33'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_33'][i_a, i_b, :, :, :]
            b2 = b2_full[i_b, :, :, :]
            num = np.einsum('CDF,EHG->CDEFGH', I1, b2, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 4 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[occ_idx] + (shift / 4 if shift is not None else 0.0))[:, None, None, None, None, None] + (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[act_idx][None, None, :, None, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
            g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_33'] -= 1
    if _use_count['int_33'] == 0:
        del _cache['int_33']
    # Projecting permutation: -1.00 * g(a,q,j,i) * g(b,p,s,r) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_a - e_q)] * [(E_corr + e_j + e_i + e_s + e_r - e_a - e_q - e_b - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(a,p,j,i) * g(b,q,s,r) * g(j,i,a,b) / [(E_corr + e_j + e_i - e_a - e_p)] * [(E_corr + e_j + e_i + e_s + e_r - e_a - e_p - e_b - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))

    # --- Skeleton 1515b1e3 contains 1 permutations ---
    # Base Term: +0.50 * g(a,b,s,r) * g(j,i,a,b) * g(q,p,j,i) / [(E_corr + e_s + e_r - e_a - e_b)] * [(E_corr + e_s + e_r + e_j + e_i - e_a - e_b - e_q - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(act_idx, act_idx, occ_idx, occ_idx)]
    if 'int_34' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, i_b, :, :]
                b1 = b1_full[:, :, i_a, i_b]
                d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[:, None] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                b0 = b0 / d1
                num = np.einsum('HG,DC->CDGH', b0, b1, optimize=True)
                I1[i_a, i_b, :, :, :, :] += num
        _cache['int_34'] = I1
    b2 = b2_full
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_34'][i_a, i_b, :, :, :, :]
            num = np.einsum('CDGH,FEDC->CDEFGH', I1, b2, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 4 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[occ_idx] + (shift / 4 if shift is not None else 0.0))[:, None, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[act_idx][None, None, :, None, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
            g_temp[2][:, :, :, :] += 0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_34'] -= 1
    if _use_count['int_34'] == 0:
        del _cache['int_34']
    # Projecting permutation: +0.50 * g(a,b,s,r) * g(j,i,a,b) * g(q,p,j,i) / [(E_corr + e_s + e_r - e_a - e_b)] * [(E_corr + e_s + e_r + e_j + e_i - e_a - e_b - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton 00869b9c contains 2 permutations ---
    # Base Term: -1.00 * g(a,b,r,i) * g(j,i,a,b) * g(q,p,s,j) / [(E_corr + e_r + e_i - e_a - e_b)] * [(E_corr + e_r + e_i + e_s + e_j - e_a - e_b - e_q - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(act_idx, act_idx, act_idx, occ_idx)]
    if 'int_35' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, i_b, :, :]
                b1 = b1_full[:, :, i_a, i_b]
                d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[:, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                b0 = b0 / d1
                num = np.einsum('GC,DC->CDG', b0, b1, optimize=True)
                I1[i_a, i_b, :, :, :] += num
        _cache['int_35'] = I1
    b2 = b2_full
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_35'][i_a, i_b, :, :, :]
            num = np.einsum('CDG,FEHD->CDEFGH', I1, b2, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 4 if shift is not None else 0.0))[:, None, None, None, None, None] + (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 4 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[act_idx][None, None, :, None, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
            g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_35'] -= 1
    if _use_count['int_35'] == 0:
        del _cache['int_35']
    # Projecting permutation: -1.00 * g(a,b,r,i) * g(j,i,a,b) * g(q,p,s,j) / [(E_corr + e_r + e_i - e_a - e_b)] * [(E_corr + e_r + e_i + e_s + e_j - e_a - e_b - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(a,b,s,i) * g(j,i,a,b) * g(q,p,r,j) / [(E_corr + e_s + e_i - e_a - e_b)] * [(E_corr + e_s + e_i + e_r + e_j - e_a - e_b - e_q - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton f02d16b9 contains 4 permutations ---
    # Base Term: -1.00 * g(a,q,b,i) * g(b,p,s,j) * g(j,i,a,r) / [(E_corr + e_s + e_j - e_b - e_p)] * [(E_corr + e_s + e_j + e_i - e_p - e_a - e_q)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_36' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('FBC,BEHD->BCDEFH', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, None, :, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEFH->CDEFH', num, optimize=True)
        _cache['int_36'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_36'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDEFH,DCG->CDEFGH', I1, b2, optimize=True)
        I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_36'] -= 1
    if _use_count['int_36'] == 0:
        del _cache['int_36']
    # Projecting permutation: -1.00 * g(a,q,b,i) * g(b,p,s,j) * g(j,i,a,r) / [(E_corr + e_s + e_j - e_b - e_p)] * [(E_corr + e_s + e_j + e_i - e_p - e_a - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(a,q,b,i) * g(b,p,r,j) * g(j,i,a,s) / [(E_corr + e_r + e_j - e_b - e_p)] * [(E_corr + e_r + e_j + e_i - e_p - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: +1.00 * g(a,p,b,i) * g(b,q,s,j) * g(j,i,a,r) / [(E_corr + e_s + e_j - e_b - e_q)] * [(E_corr + e_s + e_j + e_i - e_q - e_a - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: -1.00 * g(a,p,b,i) * g(b,q,r,j) * g(j,i,a,s) / [(E_corr + e_r + e_j - e_b - e_q)] * [(E_corr + e_r + e_j + e_i - e_q - e_a - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 9685ee7d contains 2 permutations ---
    # Base Term: +2.00 * g(a,q,r,i) * g(b,p,s,j) * g(j,i,a,b) / [(E_corr + e_r + e_i - e_a - e_q)] * [(E_corr + e_r + e_i + e_s + e_j - e_a - e_q - e_b - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    if 'int_37' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, :, :]
                b1 = b1_full[:, :, i_a, i_b]
                d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                b0 = b0 / d1
                num = np.einsum('FGC,DC->CDFG', b0, b1, optimize=True)
                I1[i_a, i_b, :, :, :, :] += num
        _cache['int_37'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_37'][i_a, i_b, :, :, :, :]
            b2 = b2_full[i_b, :, :, :]
            num = np.einsum('CDFG,EHD->CDEFGH', I1, b2, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 4 if shift is not None else 0.0))[:, None, None, None, None, None] + (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 4 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[act_idx][None, None, :, None, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
            g_temp[2][:, :, :, :] += 2.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_37'] -= 1
    if _use_count['int_37'] == 0:
        del _cache['int_37']
    # Projecting permutation: +2.00 * g(a,q,r,i) * g(b,p,s,j) * g(j,i,a,b) / [(E_corr + e_r + e_i - e_a - e_q)] * [(E_corr + e_r + e_i + e_s + e_j - e_a - e_q - e_b - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -2.00 * g(a,q,s,i) * g(b,p,r,j) * g(j,i,a,b) / [(E_corr + e_s + e_i - e_a - e_q)] * [(E_corr + e_s + e_i + e_r + e_j - e_a - e_q - e_b - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton 10d2307f contains 2 permutations ---
    # Base Term: -1.00 * g(a,b,s,j) * g(j,i,b,r) * g(q,p,a,i) / [(E_corr + e_s + e_j - e_a - e_b)] * [(E_corr + e_s + e_j + e_i - e_b - e_q - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_38' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BHD,FEC->BCDEFH', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[act_idx][None, None, None, :, None, None]
            num = num / d0
            I1 += num
        _cache['int_38'] = I1
    I1 = _cache['int_38']
    b2 = b2_full
    num = np.einsum('BCDEFH,DCBG->BCDEFGH', I1, b2, optimize=True)
    I2 = np.einsum('BCDEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_38'] -= 1
    if _use_count['int_38'] == 0:
        del _cache['int_38']
    # Projecting permutation: -1.00 * g(a,b,s,j) * g(j,i,b,r) * g(q,p,a,i) / [(E_corr + e_s + e_j - e_a - e_b)] * [(E_corr + e_s + e_j + e_i - e_b - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(a,b,r,j) * g(j,i,b,s) * g(q,p,a,i) / [(E_corr + e_r + e_j - e_a - e_b)] * [(E_corr + e_r + e_j + e_i - e_b - e_q - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton ab6757bb contains 4 permutations ---
    # Base Term: -0.50 * g(a,q,b,s) * g(b,p,j,i) * g(j,i,a,r) / [(E_corr + e_j + e_i - e_b - e_p)] * [(E_corr + e_j + e_i + e_s - e_p - e_a - e_q)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_39' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('FBH,BEDC->BCDEFH', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, None, :, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEFH->CDEFH', num, optimize=True)
        _cache['int_39'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_39'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDEFH,DCG->CDEFGH', I1, b2, optimize=True)
        I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_39'] -= 1
    if _use_count['int_39'] == 0:
        del _cache['int_39']
    # Projecting permutation: -0.50 * g(a,q,b,s) * g(b,p,j,i) * g(j,i,a,r) / [(E_corr + e_j + e_i - e_b - e_p)] * [(E_corr + e_j + e_i + e_s - e_p - e_a - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +0.50 * g(a,q,b,r) * g(b,p,j,i) * g(j,i,a,s) / [(E_corr + e_j + e_i - e_b - e_p)] * [(E_corr + e_j + e_i + e_r - e_p - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: +0.50 * g(a,p,b,s) * g(b,q,j,i) * g(j,i,a,r) / [(E_corr + e_j + e_i - e_b - e_q)] * [(E_corr + e_j + e_i + e_s - e_q - e_a - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: -0.50 * g(a,p,b,r) * g(b,q,j,i) * g(j,i,a,s) / [(E_corr + e_j + e_i - e_b - e_q)] * [(E_corr + e_j + e_i + e_r - e_q - e_a - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton aaa5540f contains 1 permutations ---
    # Base Term: +0.50 * g(a,b,j,i) * g(j,i,s,r) * g(q,p,a,b) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_j + e_i - e_q - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, act_idx)]
    if 'int_40' not in _cache:
        I1 = np.zeros((len(occ_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BDC,FEB->BCDEF', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[act_idx][None, None, None, None, :] - eps_spin[act_idx][None, None, None, :, None]
            num = num / d0
            I1 += np.einsum('BCDEF->CDEF', num, optimize=True)
        _cache['int_40'] = I1
    I1 = _cache['int_40']
    b2 = b2_full
    num = np.einsum('CDEF,DCHG->CDEFGH', I1, b2, optimize=True)
    I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += 0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_40'] -= 1
    if _use_count['int_40'] == 0:
        del _cache['int_40']
    # Projecting permutation: +0.50 * g(a,b,j,i) * g(j,i,s,r) * g(q,p,a,b) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_j + e_i - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton e0fc6e6d contains 2 permutations ---
    # Base Term: -1.00 * g(a,b,j,i) * g(j,i,b,r) * g(q,p,a,s) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_j + e_i + e_s - e_b - e_q - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    if 'int_41' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BDC,FEH->BCDEFH', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None] - eps_spin[act_idx][None, None, None, :, None, None]
            num = num / d0
            I1 += num
        _cache['int_41'] = I1
    I1 = _cache['int_41']
    b2 = b2_full
    num = np.einsum('BCDEFH,DCBG->BCDEFGH', I1, b2, optimize=True)
    I2 = np.einsum('BCDEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_41'] -= 1
    if _use_count['int_41'] == 0:
        del _cache['int_41']
    # Projecting permutation: -1.00 * g(a,b,j,i) * g(j,i,b,r) * g(q,p,a,s) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_j + e_i + e_s - e_b - e_q - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(a,b,j,i) * g(j,i,b,s) * g(q,p,a,r) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_j + e_i + e_r - e_b - e_q - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton c69b52d3 contains 2 permutations ---
    # Base Term: -0.50 * g(a,p,k,j) * g(i,q,s,r) * g(k,j,a,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_i - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0 = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, act_idx)]
    if 'int_42' not in _cache:
        b0 = b0
        b1 = b1
        d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[act_idx][None, :, None, None]
        d1[np.abs(d1) < 1e-12] = 1e-12
        b0 = b0 / d1
        num = np.einsum('AEDC,DCAB->ABCDE', b0, b1, optimize=True)
        d0 = (eps_spin[occ_idx] + (shift / 1 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[act_idx][None, None, None, None, :]
        num = num / d0
        I1 = np.einsum('ABCDE->BE', num, optimize=True)
        _cache['int_42'] = I1
    I1 = _cache['int_42']
    num = np.einsum('BE,BFHG->BEFGH', I1, b2, optimize=True)
    I2 = np.einsum('BEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += -0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_42'] -= 1
    if _use_count['int_42'] == 0:
        del _cache['int_42']
    # Projecting permutation: -0.50 * g(a,p,k,j) * g(i,q,s,r) * g(k,j,a,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_i - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +0.50 * g(a,q,k,j) * g(i,p,s,r) * g(k,j,a,i) / [(E_corr + e_k + e_j - e_a - e_q)] * [(E_corr + e_i - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))

    # --- Skeleton 3d619463 contains 4 permutations ---
    # Base Term: +0.50 * g(a,p,k,j) * g(i,q,a,r) * g(k,j,s,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_s + e_i - e_a - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0 = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1 = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, occ_idx)]
    b2 = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_43' not in _cache:
        b0 = b0
        b1 = b1
        d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] - eps_spin[virt_idx][:, None, None, None] - eps_spin[act_idx][None, :, None, None]
        d1[np.abs(d1) < 1e-12] = 1e-12
        b0 = b0 / d1
        num = np.einsum('AEDC,DCHB->ABCDEH', b0, b1, optimize=True)
        d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, None, :, None]
        num = num / d0
        I1 = np.einsum('ABCDEH->ABEH', num, optimize=True)
        _cache['int_43'] = I1
    I1 = _cache['int_43']
    num = np.einsum('ABEH,BFAG->ABEFGH', I1, b2, optimize=True)
    I2 = np.einsum('ABEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += 0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_43'] -= 1
    if _use_count['int_43'] == 0:
        del _cache['int_43']
    # Projecting permutation: +0.50 * g(a,p,k,j) * g(i,q,a,r) * g(k,j,s,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_s + e_i - e_a - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -0.50 * g(a,p,k,j) * g(i,q,a,s) * g(k,j,r,i) / [(E_corr + e_k + e_j - e_a - e_p)] * [(E_corr + e_r + e_i - e_a - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: -0.50 * g(a,q,k,j) * g(i,p,a,r) * g(k,j,s,i) / [(E_corr + e_k + e_j - e_a - e_q)] * [(E_corr + e_s + e_i - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: +0.50 * g(a,q,k,j) * g(i,p,a,s) * g(k,j,r,i) / [(E_corr + e_k + e_j - e_a - e_q)] * [(E_corr + e_r + e_i - e_a - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 626d96f0 contains 4 permutations ---
    # Base Term: +1.00 * g(j,a,b,i) * g(b,p,s,j) * g(i,q,a,r) / [(E_corr + e_s + e_j - e_b - e_p)] * [(E_corr + e_s + e_i - e_p - e_a)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_44' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('DBC,BEHD->BCDEH', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[act_idx][None, None, None, :, None] - eps_spin[virt_idx[i_a]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :] += np.einsum('BCDEH->CEH', num, optimize=True)
        _cache['int_44'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_44'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CEH,CFG->CEFGH', I1, b2, optimize=True)
        I2 = np.einsum('CEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += 1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_44'] -= 1
    if _use_count['int_44'] == 0:
        del _cache['int_44']
    # Projecting permutation: +1.00 * g(j,a,b,i) * g(b,p,s,j) * g(i,q,a,r) / [(E_corr + e_s + e_j - e_b - e_p)] * [(E_corr + e_s + e_i - e_p - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -1.00 * g(j,a,b,i) * g(b,p,r,j) * g(i,q,a,s) / [(E_corr + e_r + e_j - e_b - e_p)] * [(E_corr + e_r + e_i - e_p - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: -1.00 * g(j,a,b,i) * g(b,q,s,j) * g(i,p,a,r) / [(E_corr + e_s + e_j - e_b - e_q)] * [(E_corr + e_s + e_i - e_q - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: +1.00 * g(j,a,b,i) * g(b,q,r,j) * g(i,p,a,s) / [(E_corr + e_r + e_j - e_b - e_q)] * [(E_corr + e_r + e_i - e_q - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 0a3b7670 contains 4 permutations ---
    # Base Term: -1.00 * g(j,a,r,i) * g(b,p,s,j) * g(i,q,a,b) / [(E_corr + e_s + e_j - e_b - e_p)] * [(E_corr + e_s + e_r + e_i - e_b - e_p - e_a)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_45' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('DGC,BEHD->BCDEGH', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[virt_idx[i_a]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEGH->BCEGH', num, optimize=True)
        _cache['int_45'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_45'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BCEGH,CFB->BCEFGH', I1, b2, optimize=True)
        I2 = np.einsum('BCEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_45'] -= 1
    if _use_count['int_45'] == 0:
        del _cache['int_45']
    # Projecting permutation: -1.00 * g(j,a,r,i) * g(b,p,s,j) * g(i,q,a,b) / [(E_corr + e_s + e_j - e_b - e_p)] * [(E_corr + e_s + e_r + e_i - e_b - e_p - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(j,a,s,i) * g(b,p,r,j) * g(i,q,a,b) / [(E_corr + e_r + e_j - e_b - e_p)] * [(E_corr + e_r + e_s + e_i - e_b - e_p - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: +1.00 * g(j,a,r,i) * g(b,q,s,j) * g(i,p,a,b) / [(E_corr + e_s + e_j - e_b - e_q)] * [(E_corr + e_s + e_r + e_i - e_b - e_q - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: -1.00 * g(j,a,s,i) * g(b,q,r,j) * g(i,p,a,b) / [(E_corr + e_r + e_j - e_b - e_q)] * [(E_corr + e_r + e_s + e_i - e_b - e_q - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 8c2824b9 contains 4 permutations ---
    # Base Term: -1.00 * g(a,b,s,j) * g(i,q,b,r) * g(j,p,a,i) / [(E_corr + e_s + e_j - e_a - e_b)] * [(E_corr + e_s + e_i - e_b - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_46' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BHD,DEC->BCDEH', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None]
            num = num / d0
            I1 += np.einsum('BCDEH->BCEH', num, optimize=True)
        _cache['int_46'] = I1
    I1 = _cache['int_46']
    b2 = b2_full
    num = np.einsum('BCEH,CFBG->BCEFGH', I1, b2, optimize=True)
    I2 = np.einsum('BCEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_46'] -= 1
    if _use_count['int_46'] == 0:
        del _cache['int_46']
    # Projecting permutation: -1.00 * g(a,b,s,j) * g(i,q,b,r) * g(j,p,a,i) / [(E_corr + e_s + e_j - e_a - e_b)] * [(E_corr + e_s + e_i - e_b - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(a,b,r,j) * g(i,q,b,s) * g(j,p,a,i) / [(E_corr + e_r + e_j - e_a - e_b)] * [(E_corr + e_r + e_i - e_b - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: +1.00 * g(a,b,s,j) * g(i,p,b,r) * g(j,q,a,i) / [(E_corr + e_s + e_j - e_a - e_b)] * [(E_corr + e_s + e_i - e_b - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: -1.00 * g(a,b,r,j) * g(i,p,b,s) * g(j,q,a,i) / [(E_corr + e_r + e_j - e_a - e_b)] * [(E_corr + e_r + e_i - e_b - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 61db6856 contains 4 permutations ---
    # Base Term: -0.50 * g(a,b,s,j) * g(i,q,a,b) * g(j,p,r,i) / [(E_corr + e_s + e_j - e_a - e_b)] * [(E_corr + e_s + e_r + e_i - e_a - e_b - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_47' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BHD,DEGC->BCDEGH', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None]
            num = num / d0
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEGH->BCEGH', num, optimize=True)
        _cache['int_47'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_47'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BCEGH,CFB->BCEFGH', I1, b2, optimize=True)
        I2 = np.einsum('BCEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_47'] -= 1
    if _use_count['int_47'] == 0:
        del _cache['int_47']
    # Projecting permutation: -0.50 * g(a,b,s,j) * g(i,q,a,b) * g(j,p,r,i) / [(E_corr + e_s + e_j - e_a - e_b)] * [(E_corr + e_s + e_r + e_i - e_a - e_b - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +0.50 * g(a,b,r,j) * g(i,q,a,b) * g(j,p,s,i) / [(E_corr + e_r + e_j - e_a - e_b)] * [(E_corr + e_r + e_s + e_i - e_a - e_b - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: +0.50 * g(a,b,s,j) * g(i,p,a,b) * g(j,q,r,i) / [(E_corr + e_s + e_j - e_a - e_b)] * [(E_corr + e_s + e_r + e_i - e_a - e_b - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: -0.50 * g(a,b,r,j) * g(i,p,a,b) * g(j,q,s,i) / [(E_corr + e_r + e_j - e_a - e_b)] * [(E_corr + e_r + e_s + e_i - e_a - e_b - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton f4f231ee contains 4 permutations ---
    # Base Term: -1.00 * g(i,a,b,s) * g(b,p,j,i) * g(j,q,a,r) / [(E_corr + e_j + e_i - e_b - e_p)] * [(E_corr + e_j + e_s - e_p - e_a)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_48' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('CBH,BEDC->BCDEH', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] - eps_spin[act_idx][None, None, None, :, None] - eps_spin[virt_idx[i_a]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :] += np.einsum('BCDEH->DEH', num, optimize=True)
        _cache['int_48'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_48'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('DEH,DFG->DEFGH', I1, b2, optimize=True)
        I2 = np.einsum('DEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_48'] -= 1
    if _use_count['int_48'] == 0:
        del _cache['int_48']
    # Projecting permutation: -1.00 * g(i,a,b,s) * g(b,p,j,i) * g(j,q,a,r) / [(E_corr + e_j + e_i - e_b - e_p)] * [(E_corr + e_j + e_s - e_p - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(i,a,b,r) * g(b,p,j,i) * g(j,q,a,s) / [(E_corr + e_j + e_i - e_b - e_p)] * [(E_corr + e_j + e_r - e_p - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: +1.00 * g(i,a,b,s) * g(b,q,j,i) * g(j,p,a,r) / [(E_corr + e_j + e_i - e_b - e_q)] * [(E_corr + e_j + e_s - e_q - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: -1.00 * g(i,a,b,r) * g(b,q,j,i) * g(j,p,a,s) / [(E_corr + e_j + e_i - e_b - e_q)] * [(E_corr + e_j + e_r - e_q - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 42c2dc6b contains 2 permutations ---
    # Base Term: -1.00 * g(i,a,s,r) * g(b,p,j,i) * g(j,q,a,b) / [(E_corr + e_j + e_i - e_b - e_p)] * [(E_corr + e_j + e_s + e_r - e_b - e_p - e_a)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, act_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_49' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('CHG,BEDC->BCDEGH', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None] - eps_spin[virt_idx[i_a]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :, :, :] += np.einsum('BCDEGH->BDEGH', num, optimize=True)
        _cache['int_49'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_49'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BDEGH,DFB->BDEFGH', I1, b2, optimize=True)
        I2 = np.einsum('BDEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_49'] -= 1
    if _use_count['int_49'] == 0:
        del _cache['int_49']
    # Projecting permutation: -1.00 * g(i,a,s,r) * g(b,p,j,i) * g(j,q,a,b) / [(E_corr + e_j + e_i - e_b - e_p)] * [(E_corr + e_j + e_s + e_r - e_b - e_p - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(i,a,s,r) * g(b,q,j,i) * g(j,p,a,b) / [(E_corr + e_j + e_i - e_b - e_q)] * [(E_corr + e_j + e_s + e_r - e_b - e_q - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))

    # --- Skeleton f4126c60 contains 1 permutations ---
    # Base Term: -1.00 * g(a,b,j,i) * g(i,p,a,b) * g(j,q,s,r) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_i + e_s + e_r - e_a - e_b - e_q)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_50' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BDC,DFHG->BCDFGH', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None, None]
            num = num / d0
            I1[i_a, :, :, :, :, :] += np.einsum('BCDFGH->BCFGH', num, optimize=True)
        _cache['int_50'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_50'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('BCFGH,CEB->BCEFGH', I1, b2, optimize=True)
        I2 = np.einsum('BCEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_50'] -= 1
    if _use_count['int_50'] == 0:
        del _cache['int_50']
    # Projecting permutation: -1.00 * g(a,b,j,i) * g(i,p,a,b) * g(j,q,s,r) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_i + e_s + e_r - e_a - e_b - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton f4126c60 contains 1 permutations ---
    # Base Term: -1.00 * g(a,b,j,i) * g(i,p,s,r) * g(j,q,a,b) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_i - e_q)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, act_idx)]
    if 'int_51' not in _cache:
        I1 = np.zeros((len(occ_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BDC,DFB->BCDF', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 1 if shift is not None else 0.0))[None, :, None, None] - eps_spin[act_idx][None, None, None, :]
            num = num / d0
            I1 += np.einsum('BCDF->CF', num, optimize=True)
        _cache['int_51'] = I1
    I1 = _cache['int_51']
    b2 = b2_full
    num = np.einsum('CF,CEHG->CEFGH', I1, b2, optimize=True)
    I2 = np.einsum('CEFGH->EFGH', num, optimize=True)
    g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_51'] -= 1
    if _use_count['int_51'] == 0:
        del _cache['int_51']
    # Projecting permutation: -1.00 * g(a,b,j,i) * g(i,p,s,r) * g(j,q,a,b) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_i - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton 0eba25b9 contains 2 permutations ---
    # Base Term: +2.00 * g(a,b,j,i) * g(i,p,a,s) * g(j,q,b,r) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_i + e_r - e_a - e_q)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_52' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, :, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BDC,DFBG->BCDFG', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, None, :, None]
            num = num / d0
            I1[i_a, :, :, :] += np.einsum('BCDFG->CFG', num, optimize=True)
        _cache['int_52'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_52'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CFG,CEH->CEFGH', I1, b2, optimize=True)
        I2 = np.einsum('CEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += 2.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_52'] -= 1
    if _use_count['int_52'] == 0:
        del _cache['int_52']
    # Projecting permutation: +2.00 * g(a,b,j,i) * g(i,p,a,s) * g(j,q,b,r) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_i + e_r - e_a - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -2.00 * g(a,b,j,i) * g(i,p,a,r) * g(j,q,b,s) / [(E_corr + e_j + e_i - e_a - e_b)] * [(E_corr + e_i + e_s - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton 6af87110 contains 2 permutations ---
    # Base Term: -0.50 * g(a,b,c,i) * g(c,p,s,r) * g(i,q,a,b) / [(E_corr + e_s + e_r - e_c - e_p)] * [(E_corr + e_s + e_r + e_i - e_p - e_a - e_b)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_53' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, i_b, :, :]
                b1 = b1_full[:, :, :, :]
                num = np.einsum('CD,CEHG->CDEGH', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, :, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[act_idx][None, None, :, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, i_b, :, :, :, :] += np.einsum('CDEGH->DEGH', num, optimize=True)
        _cache['int_53'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_53'][i_a, i_b, :, :, :, :]
            b2 = b2_full[:, :, i_a, i_b]
            num = np.einsum('DEGH,DF->DEFGH', I1, b2, optimize=True)
            I2 = np.einsum('DEFGH->EFGH', num, optimize=True)
            g_temp[2][:, :, :, :] += -0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_53'] -= 1
    if _use_count['int_53'] == 0:
        del _cache['int_53']
    # Projecting permutation: -0.50 * g(a,b,c,i) * g(c,p,s,r) * g(i,q,a,b) / [(E_corr + e_s + e_r - e_c - e_p)] * [(E_corr + e_s + e_r + e_i - e_p - e_a - e_b)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +0.50 * g(a,b,c,i) * g(c,q,s,r) * g(i,p,a,b) / [(E_corr + e_s + e_r - e_c - e_q)] * [(E_corr + e_s + e_r + e_i - e_q - e_a - e_b)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))

    # --- Skeleton 8a96a8aa contains 2 permutations ---
    # Base Term: +1.00 * g(a,p,b,i) * g(b,c,s,r) * g(i,q,a,c) / [(E_corr + e_s + e_r - e_b - e_c)] * [(E_corr + e_s + e_r + e_i - e_c - e_a - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_54' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('ED,CHG->CDEGH', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, :, None, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :, :, :] += num
        _cache['int_54'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_54'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDEGH,DFC->CDEFGH', I1, b2, optimize=True)
        I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += 1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_54'] -= 1
    if _use_count['int_54'] == 0:
        del _cache['int_54']
    # Projecting permutation: +1.00 * g(a,p,b,i) * g(b,c,s,r) * g(i,q,a,c) / [(E_corr + e_s + e_r - e_b - e_c)] * [(E_corr + e_s + e_r + e_i - e_c - e_a - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -1.00 * g(a,q,b,i) * g(b,c,s,r) * g(i,p,a,c) / [(E_corr + e_s + e_r - e_b - e_c)] * [(E_corr + e_s + e_r + e_i - e_c - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))

    # --- Skeleton 7bb85e53 contains 4 permutations ---
    # Base Term: +0.50 * g(a,b,c,r) * g(c,p,s,i) * g(i,q,a,b) / [(E_corr + e_s + e_i - e_c - e_p)] * [(E_corr + e_s + e_i + e_r - e_p - e_a - e_b)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_55' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, i_b, :, :]
                b1 = b1_full[:, :, :, :]
                num = np.einsum('CG,CEHD->CDEGH', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, :, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] - eps_spin[act_idx][None, None, :, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, i_b, :, :, :, :] += np.einsum('CDEGH->DEGH', num, optimize=True)
        _cache['int_55'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_55'][i_a, i_b, :, :, :, :]
            b2 = b2_full[:, :, i_a, i_b]
            num = np.einsum('DEGH,DF->DEFGH', I1, b2, optimize=True)
            I2 = np.einsum('DEFGH->EFGH', num, optimize=True)
            g_temp[2][:, :, :, :] += 0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_55'] -= 1
    if _use_count['int_55'] == 0:
        del _cache['int_55']
    # Projecting permutation: +0.50 * g(a,b,c,r) * g(c,p,s,i) * g(i,q,a,b) / [(E_corr + e_s + e_i - e_c - e_p)] * [(E_corr + e_s + e_i + e_r - e_p - e_a - e_b)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -0.50 * g(a,b,c,s) * g(c,p,r,i) * g(i,q,a,b) / [(E_corr + e_r + e_i - e_c - e_p)] * [(E_corr + e_r + e_i + e_s - e_p - e_a - e_b)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: -0.50 * g(a,b,c,r) * g(c,q,s,i) * g(i,p,a,b) / [(E_corr + e_s + e_i - e_c - e_q)] * [(E_corr + e_s + e_i + e_r - e_q - e_a - e_b)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: +0.50 * g(a,b,c,s) * g(c,q,r,i) * g(i,p,a,b) / [(E_corr + e_r + e_i - e_c - e_q)] * [(E_corr + e_r + e_i + e_s - e_q - e_a - e_b)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 2c175111 contains 4 permutations ---
    # Base Term: +0.50 * g(a,p,b,c) * g(b,c,s,i) * g(i,q,a,r) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s + e_i - e_a - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, virt_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_56' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('EC,CHD->CDEH', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, :, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :] += np.einsum('CDEH->DEH', num, optimize=True)
        _cache['int_56'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_56'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('DEH,DFG->DEFGH', I1, b2, optimize=True)
        I2 = np.einsum('DEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += 0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_56'] -= 1
    if _use_count['int_56'] == 0:
        del _cache['int_56']
    # Projecting permutation: +0.50 * g(a,p,b,c) * g(b,c,s,i) * g(i,q,a,r) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s + e_i - e_a - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -0.50 * g(a,p,b,c) * g(b,c,r,i) * g(i,q,a,s) / [(E_corr + e_r + e_i - e_b - e_c)] * [(E_corr + e_r + e_i - e_a - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: -0.50 * g(a,q,b,c) * g(b,c,s,i) * g(i,p,a,r) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s + e_i - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: +0.50 * g(a,q,b,c) * g(b,c,r,i) * g(i,p,a,s) / [(E_corr + e_r + e_i - e_b - e_c)] * [(E_corr + e_r + e_i - e_a - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton e1b90331 contains 4 permutations ---
    # Base Term: -1.00 * g(a,p,b,r) * g(b,c,s,i) * g(i,q,a,c) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s + e_i + e_r - e_c - e_a - e_p)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    if 'int_57' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('EG,CHD->CDEGH', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[None, :, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, :, None, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :, :, :] += num
        _cache['int_57'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_57'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDEGH,DFC->CDEFGH', I1, b2, optimize=True)
        I2 = np.einsum('CDEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += -1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_57'] -= 1
    if _use_count['int_57'] == 0:
        del _cache['int_57']
    # Projecting permutation: -1.00 * g(a,p,b,r) * g(b,c,s,i) * g(i,q,a,c) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s + e_i + e_r - e_c - e_a - e_p)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +1.00 * g(a,p,b,s) * g(b,c,r,i) * g(i,q,a,c) / [(E_corr + e_r + e_i - e_b - e_c)] * [(E_corr + e_r + e_i + e_s - e_c - e_a - e_p)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))
    # Projecting permutation: +1.00 * g(a,q,b,r) * g(b,c,s,i) * g(i,p,a,c) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s + e_i + e_r - e_c - e_a - e_q)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (1, 0, 2, 3))
    # Projecting permutation: -1.00 * g(a,q,b,s) * g(b,c,r,i) * g(i,p,a,c) / [(E_corr + e_r + e_i - e_b - e_c)] * [(E_corr + e_r + e_i + e_s - e_c - e_a - e_q)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (1, 0, 3, 2))

    # --- Skeleton 3838303e contains 2 permutations ---
    # Base Term: -0.50 * g(i,a,b,c) * g(b,c,s,i) * g(q,p,a,r) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s - e_a)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, virt_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, act_idx)]
    if 'int_58' not in _cache:
        I1 = np.zeros((len(virt_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[:, i_a, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('DC,CHD->CDH', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 1 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :] += np.einsum('CDH->H', num, optimize=True)
        _cache['int_58'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_58'][i_a, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('H,FEG->EFGH', I1, b2, optimize=True)
        I2 = num
        g_temp[2][:, :, :, :] += -0.5 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_58'] -= 1
    if _use_count['int_58'] == 0:
        del _cache['int_58']
    # Projecting permutation: -0.50 * g(i,a,b,c) * g(b,c,s,i) * g(q,p,a,r) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: +0.50 * g(i,a,b,c) * g(b,c,r,i) * g(q,p,a,s) / [(E_corr + e_r + e_i - e_b - e_c)] * [(E_corr + e_r - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton 6164c799 contains 2 permutations ---
    # Base Term: +1.00 * g(i,a,b,r) * g(b,c,s,i) * g(q,p,a,c) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s + e_r - e_c - e_a)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, virt_idx)]
    if 'int_59' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[:, i_a, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('DG,CHD->CDGH', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] - eps_spin[virt_idx][:, None, None, None] - eps_spin[virt_idx[i_a]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :] += np.einsum('CDGH->CGH', num, optimize=True)
        _cache['int_59'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_59'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CGH,FEC->CEFGH', I1, b2, optimize=True)
        I2 = np.einsum('CEFGH->EFGH', num, optimize=True)
        g_temp[2][:, :, :, :] += 1.0 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_59'] -= 1
    if _use_count['int_59'] == 0:
        del _cache['int_59']
    # Projecting permutation: +1.00 * g(i,a,b,r) * g(b,c,s,i) * g(q,p,a,c) / [(E_corr + e_s + e_i - e_b - e_c)] * [(E_corr + e_s + e_r - e_c - e_a)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))
    # Projecting permutation: -1.00 * g(i,a,b,s) * g(b,c,r,i) * g(q,p,a,c) / [(E_corr + e_r + e_i - e_b - e_c)] * [(E_corr + e_r + e_s - e_c - e_a)]
    g_eff[2] += -1.000000000000 * np.transpose(g_temp[2], (0, 1, 3, 2))

    # --- Skeleton 557bd504 contains 1 permutations ---
    # Base Term: +0.250 * g(a,b,c,d) * g(c,d,s,r) * g(q,p,a,b) / [(E_corr + e_s + e_r - e_c - e_d)] * [(E_corr + e_s + e_r - e_a - e_b)]
    g_temp = {2: np.zeros((len(act_idx),) * 4)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, virt_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, virt_idx)]
    if 'int_60' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                for i_c in range(len(virt_idx)):
                    b0 = b0_full[i_a, i_b, i_c, :]
                    b1 = b1_full[i_c, :, :, :]
                    num = np.einsum('D,DHG->DGH', b0, b1, optimize=True)
                    d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] - eps_spin[virt_idx[i_c]] - eps_spin[virt_idx][:, None, None]
                    d0[np.abs(d0) < 1e-12] = 1e-12
                    num = num / d0
                    d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                    d1[np.abs(d1) < 1e-12] = 1e-12
                    num = num / d1
                    I1[i_a, i_b, :, :] += np.einsum('DGH->GH', num, optimize=True)
        _cache['int_60'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_60'][i_a, i_b, :, :]
            b2 = b2_full[:, :, i_a, i_b]
            num = np.einsum('GH,FE->EFGH', I1, b2, optimize=True)
            I2 = num
            g_temp[2][:, :, :, :] += 0.25 * np.einsum('EFGH->EFGH', I2, optimize=True)
    _use_count['int_60'] -= 1
    if _use_count['int_60'] == 0:
        del _cache['int_60']
    # Projecting permutation: +0.250 * g(a,b,c,d) * g(c,d,s,r) * g(q,p,a,b) / [(E_corr + e_s + e_r - e_c - e_d)] * [(E_corr + e_s + e_r - e_a - e_b)]
    g_eff[2] += 1.000000000000 * np.transpose(g_temp[2], (0, 1, 2, 3))

    # --- Skeleton 9e1ea51e contains 9 permutations ---
    # Base Term: -1.00 * g(i,r,t,s) * g(q,p,u,i) / [(E_corr + e_u + e_i - e_q - e_p)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, act_idx, occ_idx)]
    if 'int_61' not in _cache:
        I1 = np.zeros((len(act_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_i in range(len(occ_idx)):
            b0 = b0_full[i_i, :, :, :]
            b1 = b1_full[:, :, :, i_i]
            num = np.einsum('DFE,CBG->BCDEFG', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx[i_i]] + (shift / 2 if shift is not None else 0.0)) - eps_spin[act_idx][None, :, None, None, None, None] - eps_spin[act_idx][:, None, None, None, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            I1 += num
        _cache['int_61'] = I1
    I1 = _cache['int_61']
    g_temp[3][:, :, :, :, :, :] += -1.0 * np.einsum('BCDEFG->BCDEFG', I1, optimize=True)
    _use_count['int_61'] -= 1
    if _use_count['int_61'] == 0:
        del _cache['int_61']
    # Projecting permutation: -1.00 * g(i,r,t,s) * g(q,p,u,i) / [(E_corr + e_u + e_i - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(i,r,u,s) * g(q,p,t,i) / [(E_corr + e_t + e_i - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 5, 4))
    # Projecting permutation: -1.00 * g(i,r,u,t) * g(q,p,s,i) / [(E_corr + e_s + e_i - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 3, 4))
    # Projecting permutation: +1.00 * g(i,q,t,s) * g(r,p,u,i) / [(E_corr + e_u + e_i - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 4, 5))
    # Projecting permutation: -1.00 * g(i,q,u,s) * g(r,p,t,i) / [(E_corr + e_t + e_i - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 5, 4))
    # Projecting permutation: +1.00 * g(i,q,u,t) * g(r,p,s,i) / [(E_corr + e_s + e_i - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 5, 3, 4))
    # Projecting permutation: -1.00 * g(i,p,t,s) * g(r,q,u,i) / [(E_corr + e_u + e_i - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 4, 5))
    # Projecting permutation: +1.00 * g(i,p,u,s) * g(r,q,t,i) / [(E_corr + e_t + e_i - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 5, 4))
    # Projecting permutation: -1.00 * g(i,p,u,t) * g(r,q,s,i) / [(E_corr + e_s + e_i - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 5, 3, 4))

    # --- Skeleton d29ea506 contains 9 permutations ---
    # Base Term: -1.00 * g(a,p,u,t) * g(r,q,a,s) / [(E_corr + e_u + e_t - e_a - e_p)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, act_idx)]
    if 'int_62' not in _cache:
        I1 = np.zeros((len(act_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            b0 = b0 / d0
            num = np.einsum('BGF,DCE->BCDEFG', b0, b1, optimize=True)
            I1 += num
        _cache['int_62'] = I1
    I1 = _cache['int_62']
    g_temp[3][:, :, :, :, :, :] += -1.0 * np.einsum('BCDEFG->BCDEFG', I1, optimize=True)
    _use_count['int_62'] -= 1
    if _use_count['int_62'] == 0:
        del _cache['int_62']
    # Projecting permutation: -1.00 * g(a,p,u,t) * g(r,q,a,s) / [(E_corr + e_u + e_t - e_a - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,p,u,s) * g(r,q,a,t) / [(E_corr + e_u + e_s - e_a - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,p,t,s) * g(r,q,a,u) / [(E_corr + e_t + e_s - e_a - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: +1.00 * g(a,q,u,t) * g(r,p,a,s) / [(E_corr + e_u + e_t - e_a - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,q,u,s) * g(r,p,a,t) / [(E_corr + e_u + e_s - e_a - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 3, 5))
    # Projecting permutation: +1.00 * g(a,q,t,s) * g(r,p,a,u) / [(E_corr + e_t + e_s - e_a - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 5, 3))
    # Projecting permutation: -1.00 * g(a,r,u,t) * g(q,p,a,s) / [(E_corr + e_u + e_t - e_a - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,r,u,s) * g(q,p,a,t) / [(E_corr + e_u + e_s - e_a - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,r,t,s) * g(q,p,a,u) / [(E_corr + e_t + e_s - e_a - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 5, 3))

    # --- Skeleton c7f89938 contains 9 permutations ---
    # Base Term: -0.50 * g(j,i,t,s) * g(k,r,j,i) * g(q,p,u,k) / [(E_corr + e_u + e_k - e_q - e_p)] * [(E_corr + e_u + e_j + e_i - e_q - e_p - e_r)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(occ_idx, act_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, act_idx)]
    if 'int_63' not in _cache:
        I1 = np.zeros((len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_i in range(len(occ_idx)):
            for i_j in range(len(occ_idx)):
                b0 = b0_full[:, :, i_j, i_i]
                b1 = b1_full[:, :, :, :]
                num = np.einsum('CF,EDIC->CDEFI', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[:, None, None, None, None] - eps_spin[act_idx][None, None, :, None, None] - eps_spin[act_idx][None, :, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx[i_j]] + (shift / 3 if shift is not None else 0.0)) + (eps_spin[occ_idx[i_i]] + (shift / 3 if shift is not None else 0.0)) - eps_spin[act_idx][None, None, :, None, None] - eps_spin[act_idx][None, :, None, None, None] - eps_spin[act_idx][None, None, None, :, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_i, i_j, :, :, :, :] += np.einsum('CDEFI->DEFI', num, optimize=True)
        _cache['int_63'] = I1
    for i_i in range(len(occ_idx)):
        for i_j in range(len(occ_idx)):
            I1 = _cache['int_63'][i_i, i_j, :, :, :, :]
            b2 = b2_full[i_j, i_i, :, :]
            num = np.einsum('DEFI,HG->DEFGHI', I1, b2, optimize=True)
            I2 = num
            g_temp[3][:, :, :, :, :, :] += -0.5 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_63'] -= 1
    if _use_count['int_63'] == 0:
        del _cache['int_63']
    # Projecting permutation: -0.50 * g(j,i,t,s) * g(k,r,j,i) * g(q,p,u,k) / [(E_corr + e_u + e_k - e_q - e_p)] * [(E_corr + e_u + e_j + e_i - e_q - e_p - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +0.50 * g(j,i,u,s) * g(k,r,j,i) * g(q,p,t,k) / [(E_corr + e_t + e_k - e_q - e_p)] * [(E_corr + e_t + e_j + e_i - e_q - e_p - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 5, 4))
    # Projecting permutation: -0.50 * g(j,i,u,t) * g(k,r,j,i) * g(q,p,s,k) / [(E_corr + e_s + e_k - e_q - e_p)] * [(E_corr + e_s + e_j + e_i - e_q - e_p - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 3, 4))
    # Projecting permutation: +0.50 * g(j,i,t,s) * g(k,q,j,i) * g(r,p,u,k) / [(E_corr + e_u + e_k - e_r - e_p)] * [(E_corr + e_u + e_j + e_i - e_r - e_p - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 4, 5))
    # Projecting permutation: -0.50 * g(j,i,u,s) * g(k,q,j,i) * g(r,p,t,k) / [(E_corr + e_t + e_k - e_r - e_p)] * [(E_corr + e_t + e_j + e_i - e_r - e_p - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 5, 4))
    # Projecting permutation: +0.50 * g(j,i,u,t) * g(k,q,j,i) * g(r,p,s,k) / [(E_corr + e_s + e_k - e_r - e_p)] * [(E_corr + e_s + e_j + e_i - e_r - e_p - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 5, 3, 4))
    # Projecting permutation: -0.50 * g(j,i,t,s) * g(k,p,j,i) * g(r,q,u,k) / [(E_corr + e_u + e_k - e_r - e_q)] * [(E_corr + e_u + e_j + e_i - e_r - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 4, 5))
    # Projecting permutation: +0.50 * g(j,i,u,s) * g(k,p,j,i) * g(r,q,t,k) / [(E_corr + e_t + e_k - e_r - e_q)] * [(E_corr + e_t + e_j + e_i - e_r - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 5, 4))
    # Projecting permutation: -0.50 * g(j,i,u,t) * g(k,p,j,i) * g(r,q,s,k) / [(E_corr + e_s + e_k - e_r - e_q)] * [(E_corr + e_s + e_j + e_i - e_r - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 5, 3, 4))

    # --- Skeleton 33430693 contains 9 permutations ---
    # Base Term: +1.00 * g(k,i,t,s) * g(j,r,u,i) * g(q,p,k,j) / [(E_corr + e_k + e_j - e_q - e_p)] * [(E_corr + e_k + e_u + e_i - e_q - e_p - e_r)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, act_idx)]
    if 'int_64' not in _cache:
        I1 = np.zeros((len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_i in range(len(occ_idx)):
            for i_j in range(len(occ_idx)):
                b0 = b0_full[i_j, :, :, i_i]
                b1 = b1_full[:, :, :, i_j]
                num = np.einsum('FI,EDC->CDEFI', b0, b1, optimize=True)
                d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[:, None, None, None, None] + (eps_spin[occ_idx[i_j]] + (shift / 2 if shift is not None else 0.0)) - eps_spin[act_idx][None, None, :, None, None] - eps_spin[act_idx][None, :, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[:, None, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx[i_i]] + (shift / 3 if shift is not None else 0.0)) - eps_spin[act_idx][None, None, :, None, None] - eps_spin[act_idx][None, :, None, None, None] - eps_spin[act_idx][None, None, None, :, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_i, :, :, :, :, :] += num
        _cache['int_64'] = I1
    for i_i in range(len(occ_idx)):
        I1 = _cache['int_64'][i_i, :, :, :, :, :]
        b2 = b2_full[:, i_i, :, :]
        num = np.einsum('CDEFI,CHG->CDEFGHI', I1, b2, optimize=True)
        I2 = np.einsum('CDEFGHI->DEFGHI', num, optimize=True)
        g_temp[3][:, :, :, :, :, :] += 1.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_64'] -= 1
    if _use_count['int_64'] == 0:
        del _cache['int_64']
    # Projecting permutation: +1.00 * g(k,i,t,s) * g(j,r,u,i) * g(q,p,k,j) / [(E_corr + e_k + e_j - e_q - e_p)] * [(E_corr + e_k + e_u + e_i - e_q - e_p - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: -1.00 * g(k,i,u,s) * g(j,r,t,i) * g(q,p,k,j) / [(E_corr + e_k + e_j - e_q - e_p)] * [(E_corr + e_k + e_t + e_i - e_q - e_p - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 5, 4))
    # Projecting permutation: +1.00 * g(k,i,u,t) * g(j,r,s,i) * g(q,p,k,j) / [(E_corr + e_k + e_j - e_q - e_p)] * [(E_corr + e_k + e_s + e_i - e_q - e_p - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 3, 4))
    # Projecting permutation: -1.00 * g(k,i,t,s) * g(j,q,u,i) * g(r,p,k,j) / [(E_corr + e_k + e_j - e_r - e_p)] * [(E_corr + e_k + e_u + e_i - e_r - e_p - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 4, 5))
    # Projecting permutation: +1.00 * g(k,i,u,s) * g(j,q,t,i) * g(r,p,k,j) / [(E_corr + e_k + e_j - e_r - e_p)] * [(E_corr + e_k + e_t + e_i - e_r - e_p - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 5, 4))
    # Projecting permutation: -1.00 * g(k,i,u,t) * g(j,q,s,i) * g(r,p,k,j) / [(E_corr + e_k + e_j - e_r - e_p)] * [(E_corr + e_k + e_s + e_i - e_r - e_p - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 5, 3, 4))
    # Projecting permutation: +1.00 * g(k,i,t,s) * g(j,p,u,i) * g(r,q,k,j) / [(E_corr + e_k + e_j - e_r - e_q)] * [(E_corr + e_k + e_u + e_i - e_r - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 4, 5))
    # Projecting permutation: -1.00 * g(k,i,u,s) * g(j,p,t,i) * g(r,q,k,j) / [(E_corr + e_k + e_j - e_r - e_q)] * [(E_corr + e_k + e_t + e_i - e_r - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 5, 4))
    # Projecting permutation: +1.00 * g(k,i,u,t) * g(j,p,s,i) * g(r,q,k,j) / [(E_corr + e_k + e_j - e_r - e_q)] * [(E_corr + e_k + e_s + e_i - e_r - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 5, 3, 4))

    # --- Skeleton fcc354d9 contains 9 permutations ---
    # Base Term: -1.00 * g(a,p,u,t) * g(j,i,a,s) * g(r,q,j,i) / [(E_corr + e_u + e_t - e_a - e_p)] * [(E_corr + e_u + e_t + e_j + e_i - e_a - e_p - e_r - e_q)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, act_idx)]
    if 'int_65' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_i in range(len(occ_idx)):
                for i_j in range(len(occ_idx)):
                    b0 = b0_full[i_j, i_i, i_a, :]
                    b1 = b1_full[:, :, i_j, i_i]
                    num = np.einsum('G,FE->EFG', b0, b1, optimize=True)
                    I1[i_a, i_i, i_j, :, :, :] += num
        _cache['int_65'] = I1
    for i_a in range(len(virt_idx)):
        for i_i in range(len(occ_idx)):
            for i_j in range(len(occ_idx)):
                I1 = _cache['int_65'][i_a, i_i, i_j, :, :, :]
                b2 = b2_full[i_a, :, :, :]
                num = np.einsum('EFG,DIH->DEFGHI', I1, b2, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None, None, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, :, None] + (eps_spin[occ_idx[i_j]] + (shift / 4 if shift is not None else 0.0)) + (eps_spin[occ_idx[i_i]] + (shift / 4 if shift is not None else 0.0)) - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None, None, None, None, None] - eps_spin[act_idx][None, None, :, None, None, None] - eps_spin[act_idx][None, :, None, None, None, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I2 = num
                g_temp[3][:, :, :, :, :, :] += -1.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_65'] -= 1
    if _use_count['int_65'] == 0:
        del _cache['int_65']
    # Projecting permutation: -1.00 * g(a,p,u,t) * g(j,i,a,s) * g(r,q,j,i) / [(E_corr + e_u + e_t - e_a - e_p)] * [(E_corr + e_u + e_t + e_j + e_i - e_a - e_p - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,p,u,s) * g(j,i,a,t) * g(r,q,j,i) / [(E_corr + e_u + e_s - e_a - e_p)] * [(E_corr + e_u + e_s + e_j + e_i - e_a - e_p - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,p,t,s) * g(j,i,a,u) * g(r,q,j,i) / [(E_corr + e_t + e_s - e_a - e_p)] * [(E_corr + e_t + e_s + e_j + e_i - e_a - e_p - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: +1.00 * g(a,q,u,t) * g(j,i,a,s) * g(r,p,j,i) / [(E_corr + e_u + e_t - e_a - e_q)] * [(E_corr + e_u + e_t + e_j + e_i - e_a - e_q - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,q,u,s) * g(j,i,a,t) * g(r,p,j,i) / [(E_corr + e_u + e_s - e_a - e_q)] * [(E_corr + e_u + e_s + e_j + e_i - e_a - e_q - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 3, 5))
    # Projecting permutation: +1.00 * g(a,q,t,s) * g(j,i,a,u) * g(r,p,j,i) / [(E_corr + e_t + e_s - e_a - e_q)] * [(E_corr + e_t + e_s + e_j + e_i - e_a - e_q - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 5, 3))
    # Projecting permutation: -1.00 * g(a,r,u,t) * g(j,i,a,s) * g(q,p,j,i) / [(E_corr + e_u + e_t - e_a - e_r)] * [(E_corr + e_u + e_t + e_j + e_i - e_a - e_r - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,r,u,s) * g(j,i,a,t) * g(q,p,j,i) / [(E_corr + e_u + e_s - e_a - e_r)] * [(E_corr + e_u + e_s + e_j + e_i - e_a - e_r - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,r,t,s) * g(j,i,a,u) * g(q,p,j,i) / [(E_corr + e_t + e_s - e_a - e_r)] * [(E_corr + e_t + e_s + e_j + e_i - e_a - e_r - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 5, 3))

    # --- Skeleton ffaa2bb2 contains 18 permutations ---
    # Base Term: +2.00 * g(a,r,t,i) * g(j,i,a,s) * g(q,p,u,j) / [(E_corr + e_t + e_i - e_a - e_r)] * [(E_corr + e_t + e_i + e_u + e_j - e_a - e_r - e_q - e_p)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, occ_idx, virt_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(act_idx, act_idx, act_idx, occ_idx)]
    if 'int_66' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_i in range(len(occ_idx)):
                for i_j in range(len(occ_idx)):
                    b0 = b0_full[i_a, :, :, i_i]
                    b1 = b1_full[i_j, i_i, i_a, :]
                    d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :] + (eps_spin[occ_idx[i_i]] + (shift / 2 if shift is not None else 0.0)) - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None]
                    d1[np.abs(d1) < 1e-12] = 1e-12
                    b0 = b0 / d1
                    num = np.einsum('FH,G->FGH', b0, b1, optimize=True)
                    I1[i_a, i_i, i_j, :, :, :] += num
        _cache['int_66'] = I1
    for i_a in range(len(virt_idx)):
        for i_i in range(len(occ_idx)):
            for i_j in range(len(occ_idx)):
                I1 = _cache['int_66'][i_a, i_i, i_j, :, :, :]
                b2 = b2_full[:, :, :, i_j]
                num = np.einsum('FGH,EDI->DEFGHI', I1, b2, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, :, None] + (eps_spin[occ_idx[i_i]] + (shift / 4 if shift is not None else 0.0)) + (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[occ_idx[i_j]] + (shift / 4 if shift is not None else 0.0)) - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, :, None, None, None] - eps_spin[act_idx][None, :, None, None, None, None] - eps_spin[act_idx][:, None, None, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                I2 = num
                g_temp[3][:, :, :, :, :, :] += 2.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_66'] -= 1
    if _use_count['int_66'] == 0:
        del _cache['int_66']
    # Projecting permutation: +2.00 * g(a,r,t,i) * g(j,i,a,s) * g(q,p,u,j) / [(E_corr + e_t + e_i - e_a - e_r)] * [(E_corr + e_t + e_i + e_u + e_j - e_a - e_r - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: -2.00 * g(a,r,s,i) * g(j,i,a,t) * g(q,p,u,j) / [(E_corr + e_s + e_i - e_a - e_r)] * [(E_corr + e_s + e_i + e_u + e_j - e_a - e_r - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: -2.00 * g(a,r,u,i) * g(j,i,a,s) * g(q,p,t,j) / [(E_corr + e_u + e_i - e_a - e_r)] * [(E_corr + e_u + e_i + e_t + e_j - e_a - e_r - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 5, 4))
    # Projecting permutation: +2.00 * g(a,r,u,i) * g(j,i,a,t) * g(q,p,s,j) / [(E_corr + e_u + e_i - e_a - e_r)] * [(E_corr + e_u + e_i + e_s + e_j - e_a - e_r - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 3, 4))
    # Projecting permutation: +2.00 * g(a,r,s,i) * g(j,i,a,u) * g(q,p,t,j) / [(E_corr + e_s + e_i - e_a - e_r)] * [(E_corr + e_s + e_i + e_t + e_j - e_a - e_r - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: -2.00 * g(a,r,t,i) * g(j,i,a,u) * g(q,p,s,j) / [(E_corr + e_t + e_i - e_a - e_r)] * [(E_corr + e_t + e_i + e_s + e_j - e_a - e_r - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 4, 3))
    # Projecting permutation: -2.00 * g(a,q,t,i) * g(j,i,a,s) * g(r,p,u,j) / [(E_corr + e_t + e_i - e_a - e_q)] * [(E_corr + e_t + e_i + e_u + e_j - e_a - e_q - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 4, 5))
    # Projecting permutation: +2.00 * g(a,q,s,i) * g(j,i,a,t) * g(r,p,u,j) / [(E_corr + e_s + e_i - e_a - e_q)] * [(E_corr + e_s + e_i + e_u + e_j - e_a - e_q - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 3, 5))
    # Projecting permutation: +2.00 * g(a,q,u,i) * g(j,i,a,s) * g(r,p,t,j) / [(E_corr + e_u + e_i - e_a - e_q)] * [(E_corr + e_u + e_i + e_t + e_j - e_a - e_q - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 5, 4))
    # Projecting permutation: -2.00 * g(a,q,u,i) * g(j,i,a,t) * g(r,p,s,j) / [(E_corr + e_u + e_i - e_a - e_q)] * [(E_corr + e_u + e_i + e_s + e_j - e_a - e_q - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 5, 3, 4))
    # Projecting permutation: -2.00 * g(a,q,s,i) * g(j,i,a,u) * g(r,p,t,j) / [(E_corr + e_s + e_i - e_a - e_q)] * [(E_corr + e_s + e_i + e_t + e_j - e_a - e_q - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 5, 3))
    # Projecting permutation: +2.00 * g(a,q,t,i) * g(j,i,a,u) * g(r,p,s,j) / [(E_corr + e_t + e_i - e_a - e_q)] * [(E_corr + e_t + e_i + e_s + e_j - e_a - e_q - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 5, 4, 3))
    # Projecting permutation: +2.00 * g(a,p,t,i) * g(j,i,a,s) * g(r,q,u,j) / [(E_corr + e_t + e_i - e_a - e_p)] * [(E_corr + e_t + e_i + e_u + e_j - e_a - e_p - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 4, 5))
    # Projecting permutation: -2.00 * g(a,p,s,i) * g(j,i,a,t) * g(r,q,u,j) / [(E_corr + e_s + e_i - e_a - e_p)] * [(E_corr + e_s + e_i + e_u + e_j - e_a - e_p - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 3, 5))
    # Projecting permutation: -2.00 * g(a,p,u,i) * g(j,i,a,s) * g(r,q,t,j) / [(E_corr + e_u + e_i - e_a - e_p)] * [(E_corr + e_u + e_i + e_t + e_j - e_a - e_p - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 5, 4))
    # Projecting permutation: +2.00 * g(a,p,u,i) * g(j,i,a,t) * g(r,q,s,j) / [(E_corr + e_u + e_i - e_a - e_p)] * [(E_corr + e_u + e_i + e_s + e_j - e_a - e_p - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 5, 3, 4))
    # Projecting permutation: +2.00 * g(a,p,s,i) * g(j,i,a,u) * g(r,q,t,j) / [(E_corr + e_s + e_i - e_a - e_p)] * [(E_corr + e_s + e_i + e_t + e_j - e_a - e_p - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 5, 3))
    # Projecting permutation: -2.00 * g(a,p,t,i) * g(j,i,a,u) * g(r,q,s,j) / [(E_corr + e_t + e_i - e_a - e_p)] * [(E_corr + e_t + e_i + e_s + e_j - e_a - e_p - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 5, 4, 3))

    # --- Skeleton 6d69ce86 contains 9 permutations ---
    # Base Term: -1.00 * g(a,p,u,j) * g(j,i,t,s) * g(r,q,a,i) / [(E_corr + e_u + e_j - e_a - e_p)] * [(E_corr + e_u + e_j + e_i - e_p - e_r - e_q)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, act_idx)]
    if 'int_67' not in _cache:
        I1 = np.zeros((len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_i in range(len(occ_idx)):
                b0 = b0_full[i_a, :, :, :]
                b1 = b1_full[:, :, i_a, i_i]
                d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                b0 = b0 / d1
                num = np.einsum('DIC,FE->CDEFI', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[:, None, None, None, None] + (eps_spin[occ_idx[i_i]] + (shift / 3 if shift is not None else 0.0)) - eps_spin[act_idx][None, :, None, None, None] - eps_spin[act_idx][None, None, None, :, None] - eps_spin[act_idx][None, None, :, None, None]
                num = num / d0
                I1[i_i, :, :, :, :, :] += num
        _cache['int_67'] = I1
    for i_i in range(len(occ_idx)):
        I1 = _cache['int_67'][i_i, :, :, :, :, :]
        b2 = b2_full[:, i_i, :, :]
        num = np.einsum('CDEFI,CHG->CDEFGHI', I1, b2, optimize=True)
        I2 = np.einsum('CDEFGHI->DEFGHI', num, optimize=True)
        g_temp[3][:, :, :, :, :, :] += -1.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_67'] -= 1
    if _use_count['int_67'] == 0:
        del _cache['int_67']
    # Projecting permutation: -1.00 * g(a,p,u,j) * g(j,i,t,s) * g(r,q,a,i) / [(E_corr + e_u + e_j - e_a - e_p)] * [(E_corr + e_u + e_j + e_i - e_p - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,p,t,j) * g(j,i,u,s) * g(r,q,a,i) / [(E_corr + e_t + e_j - e_a - e_p)] * [(E_corr + e_t + e_j + e_i - e_p - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 5, 4))
    # Projecting permutation: -1.00 * g(a,p,s,j) * g(j,i,u,t) * g(r,q,a,i) / [(E_corr + e_s + e_j - e_a - e_p)] * [(E_corr + e_s + e_j + e_i - e_p - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 3, 4))
    # Projecting permutation: +1.00 * g(a,q,u,j) * g(j,i,t,s) * g(r,p,a,i) / [(E_corr + e_u + e_j - e_a - e_q)] * [(E_corr + e_u + e_j + e_i - e_q - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,q,t,j) * g(j,i,u,s) * g(r,p,a,i) / [(E_corr + e_t + e_j - e_a - e_q)] * [(E_corr + e_t + e_j + e_i - e_q - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 5, 4))
    # Projecting permutation: +1.00 * g(a,q,s,j) * g(j,i,u,t) * g(r,p,a,i) / [(E_corr + e_s + e_j - e_a - e_q)] * [(E_corr + e_s + e_j + e_i - e_q - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 5, 3, 4))
    # Projecting permutation: -1.00 * g(a,r,u,j) * g(j,i,t,s) * g(q,p,a,i) / [(E_corr + e_u + e_j - e_a - e_r)] * [(E_corr + e_u + e_j + e_i - e_r - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,r,t,j) * g(j,i,u,s) * g(q,p,a,i) / [(E_corr + e_t + e_j - e_a - e_r)] * [(E_corr + e_t + e_j + e_i - e_r - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 5, 4))
    # Projecting permutation: -1.00 * g(a,r,s,j) * g(j,i,u,t) * g(q,p,a,i) / [(E_corr + e_s + e_j - e_a - e_r)] * [(E_corr + e_s + e_j + e_i - e_r - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 5, 3, 4))

    # --- Skeleton edea33af contains 9 permutations ---
    # Base Term: -1.00 * g(a,p,j,i) * g(j,i,t,s) * g(r,q,a,u) / [(E_corr + e_j + e_i - e_a - e_p)] * [(E_corr + e_j + e_i + e_u - e_p - e_r - e_q)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, act_idx)]
    if 'int_68' not in _cache:
        I1 = np.zeros((len(occ_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_i in range(len(occ_idx)):
                b0 = b0_full[i_a, :, :, i_i]
                b1 = b1_full[:, :, i_a, :]
                d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :] + (eps_spin[occ_idx[i_i]] + (shift / 2 if shift is not None else 0.0)) - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                b0 = b0 / d1
                num = np.einsum('DC,FEI->CDEFI', b0, b1, optimize=True)
                d0 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[:, None, None, None, None] + (eps_spin[occ_idx[i_i]] + (shift / 3 if shift is not None else 0.0)) + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] - eps_spin[act_idx][None, :, None, None, None] - eps_spin[act_idx][None, None, None, :, None] - eps_spin[act_idx][None, None, :, None, None]
                num = num / d0
                I1[i_i, :, :, :, :, :] += num
        _cache['int_68'] = I1
    for i_i in range(len(occ_idx)):
        I1 = _cache['int_68'][i_i, :, :, :, :, :]
        b2 = b2_full[:, i_i, :, :]
        num = np.einsum('CDEFI,CHG->CDEFGHI', I1, b2, optimize=True)
        I2 = np.einsum('CDEFGHI->DEFGHI', num, optimize=True)
        g_temp[3][:, :, :, :, :, :] += -1.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_68'] -= 1
    if _use_count['int_68'] == 0:
        del _cache['int_68']
    # Projecting permutation: -1.00 * g(a,p,j,i) * g(j,i,t,s) * g(r,q,a,u) / [(E_corr + e_j + e_i - e_a - e_p)] * [(E_corr + e_j + e_i + e_u - e_p - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,p,j,i) * g(j,i,u,s) * g(r,q,a,t) / [(E_corr + e_j + e_i - e_a - e_p)] * [(E_corr + e_j + e_i + e_t - e_p - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 5, 4))
    # Projecting permutation: -1.00 * g(a,p,j,i) * g(j,i,u,t) * g(r,q,a,s) / [(E_corr + e_j + e_i - e_a - e_p)] * [(E_corr + e_j + e_i + e_s - e_p - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 3, 4))
    # Projecting permutation: +1.00 * g(a,q,j,i) * g(j,i,t,s) * g(r,p,a,u) / [(E_corr + e_j + e_i - e_a - e_q)] * [(E_corr + e_j + e_i + e_u - e_q - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,q,j,i) * g(j,i,u,s) * g(r,p,a,t) / [(E_corr + e_j + e_i - e_a - e_q)] * [(E_corr + e_j + e_i + e_t - e_q - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 5, 4))
    # Projecting permutation: +1.00 * g(a,q,j,i) * g(j,i,u,t) * g(r,p,a,s) / [(E_corr + e_j + e_i - e_a - e_q)] * [(E_corr + e_j + e_i + e_s - e_q - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 5, 3, 4))
    # Projecting permutation: -1.00 * g(a,r,j,i) * g(j,i,t,s) * g(q,p,a,u) / [(E_corr + e_j + e_i - e_a - e_r)] * [(E_corr + e_j + e_i + e_u - e_r - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,r,j,i) * g(j,i,u,s) * g(q,p,a,t) / [(E_corr + e_j + e_i - e_a - e_r)] * [(E_corr + e_j + e_i + e_t - e_r - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 5, 4))
    # Projecting permutation: -1.00 * g(a,r,j,i) * g(j,i,u,t) * g(q,p,a,s) / [(E_corr + e_j + e_i - e_a - e_r)] * [(E_corr + e_j + e_i + e_s - e_r - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 5, 3, 4))

    # --- Skeleton 4b4aebfb contains 9 permutations ---
    # Base Term: -0.50 * g(i,r,t,s) * g(k,j,u,i) * g(q,p,k,j) / [(E_corr + e_k + e_j - e_q - e_p)] * [(E_corr + e_u + e_i - e_q - e_p)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(occ_idx, occ_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, act_idx)]
    if 'int_69' not in _cache:
        I1 = np.zeros((len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_i in range(len(occ_idx)):
            b0 = b0_full[:, :, :, i_i]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('CBI,EDCB->BCDEI', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[:, None, None, None, None] - eps_spin[act_idx][None, None, None, :, None] - eps_spin[act_idx][None, None, :, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx[i_i]] + (shift / 2 if shift is not None else 0.0)) - eps_spin[act_idx][None, None, None, :, None] - eps_spin[act_idx][None, None, :, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_i, :, :, :] += np.einsum('BCDEI->DEI', num, optimize=True)
        _cache['int_69'] = I1
    for i_i in range(len(occ_idx)):
        I1 = _cache['int_69'][i_i, :, :, :]
        b2 = b2_full[i_i, :, :, :]
        num = np.einsum('DEI,FHG->DEFGHI', I1, b2, optimize=True)
        I2 = num
        g_temp[3][:, :, :, :, :, :] += -0.5 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_69'] -= 1
    if _use_count['int_69'] == 0:
        del _cache['int_69']
    # Projecting permutation: -0.50 * g(i,r,t,s) * g(k,j,u,i) * g(q,p,k,j) / [(E_corr + e_k + e_j - e_q - e_p)] * [(E_corr + e_u + e_i - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +0.50 * g(i,r,u,s) * g(k,j,t,i) * g(q,p,k,j) / [(E_corr + e_k + e_j - e_q - e_p)] * [(E_corr + e_t + e_i - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 5, 4))
    # Projecting permutation: -0.50 * g(i,r,u,t) * g(k,j,s,i) * g(q,p,k,j) / [(E_corr + e_k + e_j - e_q - e_p)] * [(E_corr + e_s + e_i - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 3, 4))
    # Projecting permutation: +0.50 * g(i,q,t,s) * g(k,j,u,i) * g(r,p,k,j) / [(E_corr + e_k + e_j - e_r - e_p)] * [(E_corr + e_u + e_i - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 4, 5))
    # Projecting permutation: -0.50 * g(i,q,u,s) * g(k,j,t,i) * g(r,p,k,j) / [(E_corr + e_k + e_j - e_r - e_p)] * [(E_corr + e_t + e_i - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 5, 4))
    # Projecting permutation: +0.50 * g(i,q,u,t) * g(k,j,s,i) * g(r,p,k,j) / [(E_corr + e_k + e_j - e_r - e_p)] * [(E_corr + e_s + e_i - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 5, 3, 4))
    # Projecting permutation: -0.50 * g(i,p,t,s) * g(k,j,u,i) * g(r,q,k,j) / [(E_corr + e_k + e_j - e_r - e_q)] * [(E_corr + e_u + e_i - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 4, 5))
    # Projecting permutation: +0.50 * g(i,p,u,s) * g(k,j,t,i) * g(r,q,k,j) / [(E_corr + e_k + e_j - e_r - e_q)] * [(E_corr + e_t + e_i - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 5, 4))
    # Projecting permutation: -0.50 * g(i,p,u,t) * g(k,j,s,i) * g(r,q,k,j) / [(E_corr + e_k + e_j - e_r - e_q)] * [(E_corr + e_s + e_i - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 5, 3, 4))

    # --- Skeleton 57f8c374 contains 18 permutations ---
    # Base Term: -1.00 * g(j,a,t,i) * g(i,r,a,s) * g(q,p,u,j) / [(E_corr + e_u + e_j - e_q - e_p)] * [(E_corr + e_u + e_t + e_i - e_q - e_p - e_a)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_70' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_i in range(len(occ_idx)):
                b0 = b0_full[:, i_a, :, i_i]
                b1 = b1_full[:, :, :, :]
                num = np.einsum('CH,EDIC->CDEHI', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[:, None, None, None, None] - eps_spin[act_idx][None, None, :, None, None] - eps_spin[act_idx][None, :, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] + (eps_spin[occ_idx[i_i]] + (shift / 3 if shift is not None else 0.0)) - eps_spin[act_idx][None, None, :, None, None] - eps_spin[act_idx][None, :, None, None, None] - eps_spin[virt_idx[i_a]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, i_i, :, :, :, :] += np.einsum('CDEHI->DEHI', num, optimize=True)
        _cache['int_70'] = I1
    for i_a in range(len(virt_idx)):
        for i_i in range(len(occ_idx)):
            I1 = _cache['int_70'][i_a, i_i, :, :, :, :]
            b2 = b2_full[i_i, :, i_a, :]
            num = np.einsum('DEHI,FG->DEFGHI', I1, b2, optimize=True)
            I2 = num
            g_temp[3][:, :, :, :, :, :] += -1.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_70'] -= 1
    if _use_count['int_70'] == 0:
        del _cache['int_70']
    # Projecting permutation: -1.00 * g(j,a,t,i) * g(i,r,a,s) * g(q,p,u,j) / [(E_corr + e_u + e_j - e_q - e_p)] * [(E_corr + e_u + e_t + e_i - e_q - e_p - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(j,a,s,i) * g(i,r,a,t) * g(q,p,u,j) / [(E_corr + e_u + e_j - e_q - e_p)] * [(E_corr + e_u + e_s + e_i - e_q - e_p - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: +1.00 * g(j,a,u,i) * g(i,r,a,s) * g(q,p,t,j) / [(E_corr + e_t + e_j - e_q - e_p)] * [(E_corr + e_t + e_u + e_i - e_q - e_p - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 5, 4))
    # Projecting permutation: -1.00 * g(j,a,u,i) * g(i,r,a,t) * g(q,p,s,j) / [(E_corr + e_s + e_j - e_q - e_p)] * [(E_corr + e_s + e_u + e_i - e_q - e_p - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 3, 4))
    # Projecting permutation: -1.00 * g(j,a,s,i) * g(i,r,a,u) * g(q,p,t,j) / [(E_corr + e_t + e_j - e_q - e_p)] * [(E_corr + e_t + e_s + e_i - e_q - e_p - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: +1.00 * g(j,a,t,i) * g(i,r,a,u) * g(q,p,s,j) / [(E_corr + e_s + e_j - e_q - e_p)] * [(E_corr + e_s + e_t + e_i - e_q - e_p - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 4, 3))
    # Projecting permutation: +1.00 * g(j,a,t,i) * g(i,q,a,s) * g(r,p,u,j) / [(E_corr + e_u + e_j - e_r - e_p)] * [(E_corr + e_u + e_t + e_i - e_r - e_p - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 4, 5))
    # Projecting permutation: -1.00 * g(j,a,s,i) * g(i,q,a,t) * g(r,p,u,j) / [(E_corr + e_u + e_j - e_r - e_p)] * [(E_corr + e_u + e_s + e_i - e_r - e_p - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 3, 5))
    # Projecting permutation: -1.00 * g(j,a,u,i) * g(i,q,a,s) * g(r,p,t,j) / [(E_corr + e_t + e_j - e_r - e_p)] * [(E_corr + e_t + e_u + e_i - e_r - e_p - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 5, 4))
    # Projecting permutation: +1.00 * g(j,a,u,i) * g(i,q,a,t) * g(r,p,s,j) / [(E_corr + e_s + e_j - e_r - e_p)] * [(E_corr + e_s + e_u + e_i - e_r - e_p - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 5, 3, 4))
    # Projecting permutation: +1.00 * g(j,a,s,i) * g(i,q,a,u) * g(r,p,t,j) / [(E_corr + e_t + e_j - e_r - e_p)] * [(E_corr + e_t + e_s + e_i - e_r - e_p - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 5, 3))
    # Projecting permutation: -1.00 * g(j,a,t,i) * g(i,q,a,u) * g(r,p,s,j) / [(E_corr + e_s + e_j - e_r - e_p)] * [(E_corr + e_s + e_t + e_i - e_r - e_p - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 5, 4, 3))
    # Projecting permutation: -1.00 * g(j,a,t,i) * g(i,p,a,s) * g(r,q,u,j) / [(E_corr + e_u + e_j - e_r - e_q)] * [(E_corr + e_u + e_t + e_i - e_r - e_q - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 4, 5))
    # Projecting permutation: +1.00 * g(j,a,s,i) * g(i,p,a,t) * g(r,q,u,j) / [(E_corr + e_u + e_j - e_r - e_q)] * [(E_corr + e_u + e_s + e_i - e_r - e_q - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 3, 5))
    # Projecting permutation: +1.00 * g(j,a,u,i) * g(i,p,a,s) * g(r,q,t,j) / [(E_corr + e_t + e_j - e_r - e_q)] * [(E_corr + e_t + e_u + e_i - e_r - e_q - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 5, 4))
    # Projecting permutation: -1.00 * g(j,a,u,i) * g(i,p,a,t) * g(r,q,s,j) / [(E_corr + e_s + e_j - e_r - e_q)] * [(E_corr + e_s + e_u + e_i - e_r - e_q - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 5, 3, 4))
    # Projecting permutation: -1.00 * g(j,a,s,i) * g(i,p,a,u) * g(r,q,t,j) / [(E_corr + e_t + e_j - e_r - e_q)] * [(E_corr + e_t + e_s + e_i - e_r - e_q - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 5, 3))
    # Projecting permutation: +1.00 * g(j,a,t,i) * g(i,p,a,u) * g(r,q,s,j) / [(E_corr + e_s + e_j - e_r - e_q)] * [(E_corr + e_s + e_t + e_i - e_r - e_q - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 5, 4, 3))

    # --- Skeleton 4ada4874 contains 18 permutations ---
    # Base Term: -1.00 * g(a,p,u,j) * g(i,r,t,s) * g(j,q,a,i) / [(E_corr + e_u + e_j - e_a - e_p)] * [(E_corr + e_u + e_i - e_p - e_q)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, act_idx)]
    if 'int_71' not in _cache:
        I1 = np.zeros((len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('DIC,CEB->BCDEI', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[:, None, None, None, None] - eps_spin[act_idx][None, None, :, None, None] - eps_spin[act_idx][None, None, None, :, None]
            num = num / d0
            I1 += np.einsum('BCDEI->BDEI', num, optimize=True)
        _cache['int_71'] = I1
    I1 = _cache['int_71']
    b2 = b2_full
    num = np.einsum('BDEI,BFHG->BDEFGHI', I1, b2, optimize=True)
    I2 = np.einsum('BDEFGHI->DEFGHI', num, optimize=True)
    g_temp[3][:, :, :, :, :, :] += -1.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_71'] -= 1
    if _use_count['int_71'] == 0:
        del _cache['int_71']
    # Projecting permutation: -1.00 * g(a,p,u,j) * g(i,r,t,s) * g(j,q,a,i) / [(E_corr + e_u + e_j - e_a - e_p)] * [(E_corr + e_u + e_i - e_p - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,p,t,j) * g(i,r,u,s) * g(j,q,a,i) / [(E_corr + e_t + e_j - e_a - e_p)] * [(E_corr + e_t + e_i - e_p - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 5, 4))
    # Projecting permutation: -1.00 * g(a,p,s,j) * g(i,r,u,t) * g(j,q,a,i) / [(E_corr + e_s + e_j - e_a - e_p)] * [(E_corr + e_s + e_i - e_p - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 3, 4))
    # Projecting permutation: +1.00 * g(a,q,u,j) * g(i,r,t,s) * g(j,p,a,i) / [(E_corr + e_u + e_j - e_a - e_q)] * [(E_corr + e_u + e_i - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,q,t,j) * g(i,r,u,s) * g(j,p,a,i) / [(E_corr + e_t + e_j - e_a - e_q)] * [(E_corr + e_t + e_i - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 5, 4))
    # Projecting permutation: +1.00 * g(a,q,s,j) * g(i,r,u,t) * g(j,p,a,i) / [(E_corr + e_s + e_j - e_a - e_q)] * [(E_corr + e_s + e_i - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 5, 3, 4))
    # Projecting permutation: +1.00 * g(a,p,u,j) * g(i,q,t,s) * g(j,r,a,i) / [(E_corr + e_u + e_j - e_a - e_p)] * [(E_corr + e_u + e_i - e_p - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,p,t,j) * g(i,q,u,s) * g(j,r,a,i) / [(E_corr + e_t + e_j - e_a - e_p)] * [(E_corr + e_t + e_i - e_p - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 5, 4))
    # Projecting permutation: +1.00 * g(a,p,s,j) * g(i,q,u,t) * g(j,r,a,i) / [(E_corr + e_s + e_j - e_a - e_p)] * [(E_corr + e_s + e_i - e_p - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 5, 3, 4))
    # Projecting permutation: -1.00 * g(a,r,u,j) * g(i,q,t,s) * g(j,p,a,i) / [(E_corr + e_u + e_j - e_a - e_r)] * [(E_corr + e_u + e_i - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,r,t,j) * g(i,q,u,s) * g(j,p,a,i) / [(E_corr + e_t + e_j - e_a - e_r)] * [(E_corr + e_t + e_i - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 5, 4))
    # Projecting permutation: -1.00 * g(a,r,s,j) * g(i,q,u,t) * g(j,p,a,i) / [(E_corr + e_s + e_j - e_a - e_r)] * [(E_corr + e_s + e_i - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 5, 3, 4))
    # Projecting permutation: -1.00 * g(a,q,u,j) * g(i,p,t,s) * g(j,r,a,i) / [(E_corr + e_u + e_j - e_a - e_q)] * [(E_corr + e_u + e_i - e_q - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,q,t,j) * g(i,p,u,s) * g(j,r,a,i) / [(E_corr + e_t + e_j - e_a - e_q)] * [(E_corr + e_t + e_i - e_q - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 5, 4))
    # Projecting permutation: -1.00 * g(a,q,s,j) * g(i,p,u,t) * g(j,r,a,i) / [(E_corr + e_s + e_j - e_a - e_q)] * [(E_corr + e_s + e_i - e_q - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 5, 3, 4))
    # Projecting permutation: +1.00 * g(a,r,u,j) * g(i,p,t,s) * g(j,q,a,i) / [(E_corr + e_u + e_j - e_a - e_r)] * [(E_corr + e_u + e_i - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,r,t,j) * g(i,p,u,s) * g(j,q,a,i) / [(E_corr + e_t + e_j - e_a - e_r)] * [(E_corr + e_t + e_i - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 3, 5, 4))
    # Projecting permutation: +1.00 * g(a,r,s,j) * g(i,p,u,t) * g(j,q,a,i) / [(E_corr + e_s + e_j - e_a - e_r)] * [(E_corr + e_s + e_i - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 5, 3, 4))

    # --- Skeleton 34a5181d contains 36 permutations ---
    # Base Term: +1.00 * g(a,p,u,j) * g(i,r,a,s) * g(j,q,t,i) / [(E_corr + e_u + e_j - e_a - e_p)] * [(E_corr + e_u + e_t + e_i - e_a - e_p - e_q)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_72' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_i in range(len(occ_idx)):
                b0 = b0_full[i_a, :, :, :]
                b1 = b1_full[:, :, :, i_i]
                d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                b0 = b0 / d1
                num = np.einsum('DIC,CEH->CDEHI', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] + (eps_spin[occ_idx[i_i]] + (shift / 3 if shift is not None else 0.0)) - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, :, None, None, None] - eps_spin[act_idx][None, None, :, None, None]
                num = num / d0
                I1[i_a, i_i, :, :, :, :] += np.einsum('CDEHI->DEHI', num, optimize=True)
        _cache['int_72'] = I1
    for i_a in range(len(virt_idx)):
        for i_i in range(len(occ_idx)):
            I1 = _cache['int_72'][i_a, i_i, :, :, :, :]
            b2 = b2_full[i_i, :, i_a, :]
            num = np.einsum('DEHI,FG->DEFGHI', I1, b2, optimize=True)
            I2 = num
            g_temp[3][:, :, :, :, :, :] += 1.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_72'] -= 1
    if _use_count['int_72'] == 0:
        del _cache['int_72']
    # Projecting permutation: +1.00 * g(a,p,u,j) * g(i,r,a,s) * g(j,q,t,i) / [(E_corr + e_u + e_j - e_a - e_p)] * [(E_corr + e_u + e_t + e_i - e_a - e_p - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,p,u,j) * g(i,r,a,t) * g(j,q,s,i) / [(E_corr + e_u + e_j - e_a - e_p)] * [(E_corr + e_u + e_s + e_i - e_a - e_p - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,p,t,j) * g(i,r,a,s) * g(j,q,u,i) / [(E_corr + e_t + e_j - e_a - e_p)] * [(E_corr + e_t + e_u + e_i - e_a - e_p - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 5, 4))
    # Projecting permutation: +1.00 * g(a,p,s,j) * g(i,r,a,t) * g(j,q,u,i) / [(E_corr + e_s + e_j - e_a - e_p)] * [(E_corr + e_s + e_u + e_i - e_a - e_p - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 3, 4))
    # Projecting permutation: +1.00 * g(a,p,t,j) * g(i,r,a,u) * g(j,q,s,i) / [(E_corr + e_t + e_j - e_a - e_p)] * [(E_corr + e_t + e_s + e_i - e_a - e_p - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: -1.00 * g(a,p,s,j) * g(i,r,a,u) * g(j,q,t,i) / [(E_corr + e_s + e_j - e_a - e_p)] * [(E_corr + e_s + e_t + e_i - e_a - e_p - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 4, 3))
    # Projecting permutation: -1.00 * g(a,q,u,j) * g(i,r,a,s) * g(j,p,t,i) / [(E_corr + e_u + e_j - e_a - e_q)] * [(E_corr + e_u + e_t + e_i - e_a - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,q,u,j) * g(i,r,a,t) * g(j,p,s,i) / [(E_corr + e_u + e_j - e_a - e_q)] * [(E_corr + e_u + e_s + e_i - e_a - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 3, 5))
    # Projecting permutation: +1.00 * g(a,q,t,j) * g(i,r,a,s) * g(j,p,u,i) / [(E_corr + e_t + e_j - e_a - e_q)] * [(E_corr + e_t + e_u + e_i - e_a - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 5, 4))
    # Projecting permutation: -1.00 * g(a,q,s,j) * g(i,r,a,t) * g(j,p,u,i) / [(E_corr + e_s + e_j - e_a - e_q)] * [(E_corr + e_s + e_u + e_i - e_a - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 5, 3, 4))
    # Projecting permutation: -1.00 * g(a,q,t,j) * g(i,r,a,u) * g(j,p,s,i) / [(E_corr + e_t + e_j - e_a - e_q)] * [(E_corr + e_t + e_s + e_i - e_a - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 5, 3))
    # Projecting permutation: +1.00 * g(a,q,s,j) * g(i,r,a,u) * g(j,p,t,i) / [(E_corr + e_s + e_j - e_a - e_q)] * [(E_corr + e_s + e_t + e_i - e_a - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 5, 4, 3))
    # Projecting permutation: -1.00 * g(a,p,u,j) * g(i,q,a,s) * g(j,r,t,i) / [(E_corr + e_u + e_j - e_a - e_p)] * [(E_corr + e_u + e_t + e_i - e_a - e_p - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,p,u,j) * g(i,q,a,t) * g(j,r,s,i) / [(E_corr + e_u + e_j - e_a - e_p)] * [(E_corr + e_u + e_s + e_i - e_a - e_p - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 3, 5))
    # Projecting permutation: +1.00 * g(a,p,t,j) * g(i,q,a,s) * g(j,r,u,i) / [(E_corr + e_t + e_j - e_a - e_p)] * [(E_corr + e_t + e_u + e_i - e_a - e_p - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 5, 4))
    # Projecting permutation: -1.00 * g(a,p,s,j) * g(i,q,a,t) * g(j,r,u,i) / [(E_corr + e_s + e_j - e_a - e_p)] * [(E_corr + e_s + e_u + e_i - e_a - e_p - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 5, 3, 4))
    # Projecting permutation: -1.00 * g(a,p,t,j) * g(i,q,a,u) * g(j,r,s,i) / [(E_corr + e_t + e_j - e_a - e_p)] * [(E_corr + e_t + e_s + e_i - e_a - e_p - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 5, 3))
    # Projecting permutation: +1.00 * g(a,p,s,j) * g(i,q,a,u) * g(j,r,t,i) / [(E_corr + e_s + e_j - e_a - e_p)] * [(E_corr + e_s + e_t + e_i - e_a - e_p - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 5, 4, 3))
    # Projecting permutation: +1.00 * g(a,r,u,j) * g(i,q,a,s) * g(j,p,t,i) / [(E_corr + e_u + e_j - e_a - e_r)] * [(E_corr + e_u + e_t + e_i - e_a - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,r,u,j) * g(i,q,a,t) * g(j,p,s,i) / [(E_corr + e_u + e_j - e_a - e_r)] * [(E_corr + e_u + e_s + e_i - e_a - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,r,t,j) * g(i,q,a,s) * g(j,p,u,i) / [(E_corr + e_t + e_j - e_a - e_r)] * [(E_corr + e_t + e_u + e_i - e_a - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 5, 4))
    # Projecting permutation: +1.00 * g(a,r,s,j) * g(i,q,a,t) * g(j,p,u,i) / [(E_corr + e_s + e_j - e_a - e_r)] * [(E_corr + e_s + e_u + e_i - e_a - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 5, 3, 4))
    # Projecting permutation: +1.00 * g(a,r,t,j) * g(i,q,a,u) * g(j,p,s,i) / [(E_corr + e_t + e_j - e_a - e_r)] * [(E_corr + e_t + e_s + e_i - e_a - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 5, 3))
    # Projecting permutation: -1.00 * g(a,r,s,j) * g(i,q,a,u) * g(j,p,t,i) / [(E_corr + e_s + e_j - e_a - e_r)] * [(E_corr + e_s + e_t + e_i - e_a - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 5, 4, 3))
    # Projecting permutation: +1.00 * g(a,q,u,j) * g(i,p,a,s) * g(j,r,t,i) / [(E_corr + e_u + e_j - e_a - e_q)] * [(E_corr + e_u + e_t + e_i - e_a - e_q - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,q,u,j) * g(i,p,a,t) * g(j,r,s,i) / [(E_corr + e_u + e_j - e_a - e_q)] * [(E_corr + e_u + e_s + e_i - e_a - e_q - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,q,t,j) * g(i,p,a,s) * g(j,r,u,i) / [(E_corr + e_t + e_j - e_a - e_q)] * [(E_corr + e_t + e_u + e_i - e_a - e_q - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 5, 4))
    # Projecting permutation: +1.00 * g(a,q,s,j) * g(i,p,a,t) * g(j,r,u,i) / [(E_corr + e_s + e_j - e_a - e_q)] * [(E_corr + e_s + e_u + e_i - e_a - e_q - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 5, 3, 4))
    # Projecting permutation: +1.00 * g(a,q,t,j) * g(i,p,a,u) * g(j,r,s,i) / [(E_corr + e_t + e_j - e_a - e_q)] * [(E_corr + e_t + e_s + e_i - e_a - e_q - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 5, 3))
    # Projecting permutation: -1.00 * g(a,q,s,j) * g(i,p,a,u) * g(j,r,t,i) / [(E_corr + e_s + e_j - e_a - e_q)] * [(E_corr + e_s + e_t + e_i - e_a - e_q - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 5, 4, 3))
    # Projecting permutation: -1.00 * g(a,r,u,j) * g(i,p,a,s) * g(j,q,t,i) / [(E_corr + e_u + e_j - e_a - e_r)] * [(E_corr + e_u + e_t + e_i - e_a - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,r,u,j) * g(i,p,a,t) * g(j,q,s,i) / [(E_corr + e_u + e_j - e_a - e_r)] * [(E_corr + e_u + e_s + e_i - e_a - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 4, 3, 5))
    # Projecting permutation: +1.00 * g(a,r,t,j) * g(i,p,a,s) * g(j,q,u,i) / [(E_corr + e_t + e_j - e_a - e_r)] * [(E_corr + e_t + e_u + e_i - e_a - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 3, 5, 4))
    # Projecting permutation: -1.00 * g(a,r,s,j) * g(i,p,a,t) * g(j,q,u,i) / [(E_corr + e_s + e_j - e_a - e_r)] * [(E_corr + e_s + e_u + e_i - e_a - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 5, 3, 4))
    # Projecting permutation: -1.00 * g(a,r,t,j) * g(i,p,a,u) * g(j,q,s,i) / [(E_corr + e_t + e_j - e_a - e_r)] * [(E_corr + e_t + e_s + e_i - e_a - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 4, 5, 3))
    # Projecting permutation: +1.00 * g(a,r,s,j) * g(i,p,a,u) * g(j,q,t,i) / [(E_corr + e_s + e_j - e_a - e_r)] * [(E_corr + e_s + e_t + e_i - e_a - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 5, 4, 3))

    # --- Skeleton 86ad013f contains 9 permutations ---
    # Base Term: -1.00 * g(i,a,u,t) * g(j,r,a,s) * g(q,p,j,i) / [(E_corr + e_j + e_i - e_q - e_p)] * [(E_corr + e_j + e_u + e_t - e_q - e_p - e_a)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, act_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, occ_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_73' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_i in range(len(occ_idx)):
                b0 = b0_full[i_i, i_a, :, :]
                b1 = b1_full[:, :, :, i_i]
                num = np.einsum('IH,EDC->CDEHI', b0, b1, optimize=True)
                d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[:, None, None, None, None] + (eps_spin[occ_idx[i_i]] + (shift / 2 if shift is not None else 0.0)) - eps_spin[act_idx][None, None, :, None, None] - eps_spin[act_idx][None, :, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[:, None, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] - eps_spin[act_idx][None, None, :, None, None] - eps_spin[act_idx][None, :, None, None, None] - eps_spin[virt_idx[i_a]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :, :, :] += num
        _cache['int_73'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_73'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDEHI,CFG->CDEFGHI', I1, b2, optimize=True)
        I2 = np.einsum('CDEFGHI->DEFGHI', num, optimize=True)
        g_temp[3][:, :, :, :, :, :] += -1.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_73'] -= 1
    if _use_count['int_73'] == 0:
        del _cache['int_73']
    # Projecting permutation: -1.00 * g(i,a,u,t) * g(j,r,a,s) * g(q,p,j,i) / [(E_corr + e_j + e_i - e_q - e_p)] * [(E_corr + e_j + e_u + e_t - e_q - e_p - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(i,a,u,s) * g(j,r,a,t) * g(q,p,j,i) / [(E_corr + e_j + e_i - e_q - e_p)] * [(E_corr + e_j + e_u + e_s - e_q - e_p - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: -1.00 * g(i,a,t,s) * g(j,r,a,u) * g(q,p,j,i) / [(E_corr + e_j + e_i - e_q - e_p)] * [(E_corr + e_j + e_t + e_s - e_q - e_p - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: +1.00 * g(i,a,u,t) * g(j,q,a,s) * g(r,p,j,i) / [(E_corr + e_j + e_i - e_r - e_p)] * [(E_corr + e_j + e_u + e_t - e_r - e_p - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 4, 5))
    # Projecting permutation: -1.00 * g(i,a,u,s) * g(j,q,a,t) * g(r,p,j,i) / [(E_corr + e_j + e_i - e_r - e_p)] * [(E_corr + e_j + e_u + e_s - e_r - e_p - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 3, 5))
    # Projecting permutation: +1.00 * g(i,a,t,s) * g(j,q,a,u) * g(r,p,j,i) / [(E_corr + e_j + e_i - e_r - e_p)] * [(E_corr + e_j + e_t + e_s - e_r - e_p - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 5, 3))
    # Projecting permutation: -1.00 * g(i,a,u,t) * g(j,p,a,s) * g(r,q,j,i) / [(E_corr + e_j + e_i - e_r - e_q)] * [(E_corr + e_j + e_u + e_t - e_r - e_q - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 4, 5))
    # Projecting permutation: +1.00 * g(i,a,u,s) * g(j,p,a,t) * g(r,q,j,i) / [(E_corr + e_j + e_i - e_r - e_q)] * [(E_corr + e_j + e_u + e_s - e_r - e_q - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 3, 5))
    # Projecting permutation: -1.00 * g(i,a,t,s) * g(j,p,a,u) * g(r,q,j,i) / [(E_corr + e_j + e_i - e_r - e_q)] * [(E_corr + e_j + e_t + e_s - e_r - e_q - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 5, 3))

    # --- Skeleton 4c77ec84 contains 9 permutations ---
    # Base Term: +2.00 * g(a,p,j,i) * g(i,q,a,u) * g(j,r,t,s) / [(E_corr + e_j + e_i - e_a - e_p)] * [(E_corr + e_i + e_t + e_s - e_a - e_p - e_r)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_74' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_i in range(len(occ_idx)):
                b0 = b0_full[i_a, :, :, i_i]
                b1 = b1_full[:, :, :, :]
                d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :] + (eps_spin[occ_idx[i_i]] + (shift / 2 if shift is not None else 0.0)) - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                b0 = b0 / d1
                num = np.einsum('DC,CFHG->CDFGH', b0, b1, optimize=True)
                d0 = (eps_spin[occ_idx[i_i]] + (shift / 3 if shift is not None else 0.0)) + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, :, None, None, None] - eps_spin[act_idx][None, None, :, None, None]
                num = num / d0
                I1[i_a, i_i, :, :, :, :] += np.einsum('CDFGH->DFGH', num, optimize=True)
        _cache['int_74'] = I1
    for i_a in range(len(virt_idx)):
        for i_i in range(len(occ_idx)):
            I1 = _cache['int_74'][i_a, i_i, :, :, :, :]
            b2 = b2_full[i_i, :, i_a, :]
            num = np.einsum('DFGH,EI->DEFGHI', I1, b2, optimize=True)
            I2 = num
            g_temp[3][:, :, :, :, :, :] += 2.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_74'] -= 1
    if _use_count['int_74'] == 0:
        del _cache['int_74']
    # Projecting permutation: +2.00 * g(a,p,j,i) * g(i,q,a,u) * g(j,r,t,s) / [(E_corr + e_j + e_i - e_a - e_p)] * [(E_corr + e_i + e_t + e_s - e_a - e_p - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: -2.00 * g(a,p,j,i) * g(i,q,a,t) * g(j,r,u,s) / [(E_corr + e_j + e_i - e_a - e_p)] * [(E_corr + e_i + e_u + e_s - e_a - e_p - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 5, 4))
    # Projecting permutation: +2.00 * g(a,p,j,i) * g(i,q,a,s) * g(j,r,u,t) / [(E_corr + e_j + e_i - e_a - e_p)] * [(E_corr + e_i + e_u + e_t - e_a - e_p - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 3, 4))
    # Projecting permutation: -2.00 * g(a,q,j,i) * g(i,p,a,u) * g(j,r,t,s) / [(E_corr + e_j + e_i - e_a - e_q)] * [(E_corr + e_i + e_t + e_s - e_a - e_q - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 4, 5))
    # Projecting permutation: +2.00 * g(a,q,j,i) * g(i,p,a,t) * g(j,r,u,s) / [(E_corr + e_j + e_i - e_a - e_q)] * [(E_corr + e_i + e_u + e_s - e_a - e_q - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 5, 4))
    # Projecting permutation: -2.00 * g(a,q,j,i) * g(i,p,a,s) * g(j,r,u,t) / [(E_corr + e_j + e_i - e_a - e_q)] * [(E_corr + e_i + e_u + e_t - e_a - e_q - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 5, 3, 4))
    # Projecting permutation: +2.00 * g(a,r,j,i) * g(i,p,a,u) * g(j,q,t,s) / [(E_corr + e_j + e_i - e_a - e_r)] * [(E_corr + e_i + e_t + e_s - e_a - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 4, 5))
    # Projecting permutation: -2.00 * g(a,r,j,i) * g(i,p,a,t) * g(j,q,u,s) / [(E_corr + e_j + e_i - e_a - e_r)] * [(E_corr + e_i + e_u + e_s - e_a - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 5, 4))
    # Projecting permutation: +2.00 * g(a,r,j,i) * g(i,p,a,s) * g(j,q,u,t) / [(E_corr + e_j + e_i - e_a - e_r)] * [(E_corr + e_i + e_u + e_t - e_a - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 5, 3, 4))

    # --- Skeleton 4c77ec84 contains 9 permutations ---
    # Base Term: +2.00 * g(a,p,j,i) * g(i,q,u,t) * g(j,r,a,s) / [(E_corr + e_j + e_i - e_a - e_p)] * [(E_corr + e_i + e_s - e_p - e_r)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, occ_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, act_idx)]
    if 'int_75' not in _cache:
        I1 = np.zeros((len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('DCB,CFG->BCDFG', b0, b1, optimize=True)
            d0 = (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[:, None, None, None, None] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] - eps_spin[act_idx][None, None, :, None, None] - eps_spin[act_idx][None, None, None, :, None]
            num = num / d0
            I1 += np.einsum('BCDFG->BDFG', num, optimize=True)
        _cache['int_75'] = I1
    I1 = _cache['int_75']
    b2 = b2_full
    num = np.einsum('BDFG,BEIH->BDEFGHI', I1, b2, optimize=True)
    I2 = np.einsum('BDEFGHI->DEFGHI', num, optimize=True)
    g_temp[3][:, :, :, :, :, :] += 2.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_75'] -= 1
    if _use_count['int_75'] == 0:
        del _cache['int_75']
    # Projecting permutation: +2.00 * g(a,p,j,i) * g(i,q,u,t) * g(j,r,a,s) / [(E_corr + e_j + e_i - e_a - e_p)] * [(E_corr + e_i + e_s - e_p - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: -2.00 * g(a,p,j,i) * g(i,q,u,s) * g(j,r,a,t) / [(E_corr + e_j + e_i - e_a - e_p)] * [(E_corr + e_i + e_t - e_p - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: +2.00 * g(a,p,j,i) * g(i,q,t,s) * g(j,r,a,u) / [(E_corr + e_j + e_i - e_a - e_p)] * [(E_corr + e_i + e_u - e_p - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: -2.00 * g(a,q,j,i) * g(i,p,u,t) * g(j,r,a,s) / [(E_corr + e_j + e_i - e_a - e_q)] * [(E_corr + e_i + e_s - e_q - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 4, 5))
    # Projecting permutation: +2.00 * g(a,q,j,i) * g(i,p,u,s) * g(j,r,a,t) / [(E_corr + e_j + e_i - e_a - e_q)] * [(E_corr + e_i + e_t - e_q - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 3, 5))
    # Projecting permutation: -2.00 * g(a,q,j,i) * g(i,p,t,s) * g(j,r,a,u) / [(E_corr + e_j + e_i - e_a - e_q)] * [(E_corr + e_i + e_u - e_q - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 5, 3))
    # Projecting permutation: +2.00 * g(a,r,j,i) * g(i,p,u,t) * g(j,q,a,s) / [(E_corr + e_j + e_i - e_a - e_r)] * [(E_corr + e_i + e_s - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 4, 5))
    # Projecting permutation: -2.00 * g(a,r,j,i) * g(i,p,u,s) * g(j,q,a,t) / [(E_corr + e_j + e_i - e_a - e_r)] * [(E_corr + e_i + e_t - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 3, 5))
    # Projecting permutation: +2.00 * g(a,r,j,i) * g(i,p,t,s) * g(j,q,a,u) / [(E_corr + e_j + e_i - e_a - e_r)] * [(E_corr + e_i + e_u - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 5, 3))

    # --- Skeleton ef4149f8 contains 18 permutations ---
    # Base Term: -1.00 * g(a,q,b,i) * g(b,p,u,t) * g(i,r,a,s) / [(E_corr + e_u + e_t - e_b - e_p)] * [(E_corr + e_u + e_t + e_i - e_p - e_a - e_q)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_76' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('EC,DIH->CDEHI', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None] - eps_spin[virt_idx[i_b]] - eps_spin[act_idx][None, :, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[:, None, None, None, None] - eps_spin[act_idx][None, :, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, :, None, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :, :, :] += num
        _cache['int_76'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_76'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDEHI,CFG->CDEFGHI', I1, b2, optimize=True)
        I2 = np.einsum('CDEFGHI->DEFGHI', num, optimize=True)
        g_temp[3][:, :, :, :, :, :] += -1.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_76'] -= 1
    if _use_count['int_76'] == 0:
        del _cache['int_76']
    # Projecting permutation: -1.00 * g(a,q,b,i) * g(b,p,u,t) * g(i,r,a,s) / [(E_corr + e_u + e_t - e_b - e_p)] * [(E_corr + e_u + e_t + e_i - e_p - e_a - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,q,b,i) * g(b,p,u,s) * g(i,r,a,t) / [(E_corr + e_u + e_s - e_b - e_p)] * [(E_corr + e_u + e_s + e_i - e_p - e_a - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,q,b,i) * g(b,p,t,s) * g(i,r,a,u) / [(E_corr + e_t + e_s - e_b - e_p)] * [(E_corr + e_t + e_s + e_i - e_p - e_a - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: +1.00 * g(a,p,b,i) * g(b,q,u,t) * g(i,r,a,s) / [(E_corr + e_u + e_t - e_b - e_q)] * [(E_corr + e_u + e_t + e_i - e_q - e_a - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,p,b,i) * g(b,q,u,s) * g(i,r,a,t) / [(E_corr + e_u + e_s - e_b - e_q)] * [(E_corr + e_u + e_s + e_i - e_q - e_a - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 3, 5))
    # Projecting permutation: +1.00 * g(a,p,b,i) * g(b,q,t,s) * g(i,r,a,u) / [(E_corr + e_t + e_s - e_b - e_q)] * [(E_corr + e_t + e_s + e_i - e_q - e_a - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 5, 3))
    # Projecting permutation: +1.00 * g(a,r,b,i) * g(b,p,u,t) * g(i,q,a,s) / [(E_corr + e_u + e_t - e_b - e_p)] * [(E_corr + e_u + e_t + e_i - e_p - e_a - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,r,b,i) * g(b,p,u,s) * g(i,q,a,t) / [(E_corr + e_u + e_s - e_b - e_p)] * [(E_corr + e_u + e_s + e_i - e_p - e_a - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 3, 5))
    # Projecting permutation: +1.00 * g(a,r,b,i) * g(b,p,t,s) * g(i,q,a,u) / [(E_corr + e_t + e_s - e_b - e_p)] * [(E_corr + e_t + e_s + e_i - e_p - e_a - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 5, 3))
    # Projecting permutation: -1.00 * g(a,p,b,i) * g(b,r,u,t) * g(i,q,a,s) / [(E_corr + e_u + e_t - e_b - e_r)] * [(E_corr + e_u + e_t + e_i - e_r - e_a - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,p,b,i) * g(b,r,u,s) * g(i,q,a,t) / [(E_corr + e_u + e_s - e_b - e_r)] * [(E_corr + e_u + e_s + e_i - e_r - e_a - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,p,b,i) * g(b,r,t,s) * g(i,q,a,u) / [(E_corr + e_t + e_s - e_b - e_r)] * [(E_corr + e_t + e_s + e_i - e_r - e_a - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 5, 3))
    # Projecting permutation: -1.00 * g(a,r,b,i) * g(b,q,u,t) * g(i,p,a,s) / [(E_corr + e_u + e_t - e_b - e_q)] * [(E_corr + e_u + e_t + e_i - e_q - e_a - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,r,b,i) * g(b,q,u,s) * g(i,p,a,t) / [(E_corr + e_u + e_s - e_b - e_q)] * [(E_corr + e_u + e_s + e_i - e_q - e_a - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,r,b,i) * g(b,q,t,s) * g(i,p,a,u) / [(E_corr + e_t + e_s - e_b - e_q)] * [(E_corr + e_t + e_s + e_i - e_q - e_a - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 5, 3))
    # Projecting permutation: +1.00 * g(a,q,b,i) * g(b,r,u,t) * g(i,p,a,s) / [(E_corr + e_u + e_t - e_b - e_r)] * [(E_corr + e_u + e_t + e_i - e_r - e_a - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,q,b,i) * g(b,r,u,s) * g(i,p,a,t) / [(E_corr + e_u + e_s - e_b - e_r)] * [(E_corr + e_u + e_s + e_i - e_r - e_a - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 4, 3, 5))
    # Projecting permutation: +1.00 * g(a,q,b,i) * g(b,r,t,s) * g(i,p,a,u) / [(E_corr + e_t + e_s - e_b - e_r)] * [(E_corr + e_t + e_s + e_i - e_r - e_a - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 4, 5, 3))

    # --- Skeleton e174a6df contains 18 permutations ---
    # Base Term: +2.00 * g(a,q,s,i) * g(b,p,u,t) * g(i,r,a,b) / [(E_corr + e_s + e_i - e_a - e_q)] * [(E_corr + e_s + e_i + e_u + e_t - e_a - e_q - e_b - e_p)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, act_idx)]
    if 'int_77' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                for i_i in range(len(occ_idx)):
                    b0 = b0_full[i_a, :, :, i_i]
                    b1 = b1_full[i_i, :, i_a, i_b]
                    d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :] + (eps_spin[occ_idx[i_i]] + (shift / 2 if shift is not None else 0.0)) - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][:, None]
                    d1[np.abs(d1) < 1e-12] = 1e-12
                    b0 = b0 / d1
                    num = np.einsum('EG,F->EFG', b0, b1, optimize=True)
                    I1[i_a, i_b, i_i, :, :, :] += num
        _cache['int_77'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            for i_i in range(len(occ_idx)):
                I1 = _cache['int_77'][i_a, i_b, i_i, :, :, :]
                b2 = b2_full[i_b, :, :, :]
                num = np.einsum('EFG,DIH->DEFGHI', I1, b2, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx[i_i]] + (shift / 4 if shift is not None else 0.0)) + (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, :, None, None, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[act_idx][:, None, None, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                I2 = num
                g_temp[3][:, :, :, :, :, :] += 2.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_77'] -= 1
    if _use_count['int_77'] == 0:
        del _cache['int_77']
    # Projecting permutation: +2.00 * g(a,q,s,i) * g(b,p,u,t) * g(i,r,a,b) / [(E_corr + e_s + e_i - e_a - e_q)] * [(E_corr + e_s + e_i + e_u + e_t - e_a - e_q - e_b - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: -2.00 * g(a,q,t,i) * g(b,p,u,s) * g(i,r,a,b) / [(E_corr + e_t + e_i - e_a - e_q)] * [(E_corr + e_t + e_i + e_u + e_s - e_a - e_q - e_b - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: +2.00 * g(a,q,u,i) * g(b,p,t,s) * g(i,r,a,b) / [(E_corr + e_u + e_i - e_a - e_q)] * [(E_corr + e_u + e_i + e_t + e_s - e_a - e_q - e_b - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: -2.00 * g(a,p,s,i) * g(b,q,u,t) * g(i,r,a,b) / [(E_corr + e_s + e_i - e_a - e_p)] * [(E_corr + e_s + e_i + e_u + e_t - e_a - e_p - e_b - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 4, 5))
    # Projecting permutation: +2.00 * g(a,p,t,i) * g(b,q,u,s) * g(i,r,a,b) / [(E_corr + e_t + e_i - e_a - e_p)] * [(E_corr + e_t + e_i + e_u + e_s - e_a - e_p - e_b - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 3, 5))
    # Projecting permutation: -2.00 * g(a,p,u,i) * g(b,q,t,s) * g(i,r,a,b) / [(E_corr + e_u + e_i - e_a - e_p)] * [(E_corr + e_u + e_i + e_t + e_s - e_a - e_p - e_b - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 5, 3))
    # Projecting permutation: -2.00 * g(a,r,s,i) * g(b,p,u,t) * g(i,q,a,b) / [(E_corr + e_s + e_i - e_a - e_r)] * [(E_corr + e_s + e_i + e_u + e_t - e_a - e_r - e_b - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 4, 5))
    # Projecting permutation: +2.00 * g(a,r,t,i) * g(b,p,u,s) * g(i,q,a,b) / [(E_corr + e_t + e_i - e_a - e_r)] * [(E_corr + e_t + e_i + e_u + e_s - e_a - e_r - e_b - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 3, 5))
    # Projecting permutation: -2.00 * g(a,r,u,i) * g(b,p,t,s) * g(i,q,a,b) / [(E_corr + e_u + e_i - e_a - e_r)] * [(E_corr + e_u + e_i + e_t + e_s - e_a - e_r - e_b - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 5, 3))
    # Projecting permutation: +2.00 * g(a,p,s,i) * g(b,r,u,t) * g(i,q,a,b) / [(E_corr + e_s + e_i - e_a - e_p)] * [(E_corr + e_s + e_i + e_u + e_t - e_a - e_p - e_b - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 4, 5))
    # Projecting permutation: -2.00 * g(a,p,t,i) * g(b,r,u,s) * g(i,q,a,b) / [(E_corr + e_t + e_i - e_a - e_p)] * [(E_corr + e_t + e_i + e_u + e_s - e_a - e_p - e_b - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 3, 5))
    # Projecting permutation: +2.00 * g(a,p,u,i) * g(b,r,t,s) * g(i,q,a,b) / [(E_corr + e_u + e_i - e_a - e_p)] * [(E_corr + e_u + e_i + e_t + e_s - e_a - e_p - e_b - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 5, 3))
    # Projecting permutation: +2.00 * g(a,r,s,i) * g(b,q,u,t) * g(i,p,a,b) / [(E_corr + e_s + e_i - e_a - e_r)] * [(E_corr + e_s + e_i + e_u + e_t - e_a - e_r - e_b - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 4, 5))
    # Projecting permutation: -2.00 * g(a,r,t,i) * g(b,q,u,s) * g(i,p,a,b) / [(E_corr + e_t + e_i - e_a - e_r)] * [(E_corr + e_t + e_i + e_u + e_s - e_a - e_r - e_b - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 3, 5))
    # Projecting permutation: +2.00 * g(a,r,u,i) * g(b,q,t,s) * g(i,p,a,b) / [(E_corr + e_u + e_i - e_a - e_r)] * [(E_corr + e_u + e_i + e_t + e_s - e_a - e_r - e_b - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 5, 3))
    # Projecting permutation: -2.00 * g(a,q,s,i) * g(b,r,u,t) * g(i,p,a,b) / [(E_corr + e_s + e_i - e_a - e_q)] * [(E_corr + e_s + e_i + e_u + e_t - e_a - e_q - e_b - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 3, 4, 5))
    # Projecting permutation: +2.00 * g(a,q,t,i) * g(b,r,u,s) * g(i,p,a,b) / [(E_corr + e_t + e_i - e_a - e_q)] * [(E_corr + e_t + e_i + e_u + e_s - e_a - e_q - e_b - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 4, 3, 5))
    # Projecting permutation: -2.00 * g(a,q,u,i) * g(b,r,t,s) * g(i,p,a,b) / [(E_corr + e_u + e_i - e_a - e_q)] * [(E_corr + e_u + e_i + e_t + e_s - e_a - e_q - e_b - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 4, 5, 3))

    # --- Skeleton bd5d820c contains 9 permutations ---
    # Base Term: -1.00 * g(a,b,u,t) * g(i,r,b,s) * g(q,p,a,i) / [(E_corr + e_u + e_t - e_a - e_b)] * [(E_corr + e_u + e_t + e_i - e_b - e_q - e_p)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_78' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, i_b, :, :]
                b1 = b1_full[:, :, i_a, :]
                d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[:, None] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                b0 = b0 / d1
                num = np.einsum('IH,EDC->CDEHI', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[:, None, None, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[act_idx][None, None, :, None, None] - eps_spin[act_idx][None, :, None, None, None]
                num = num / d0
                I1[i_b, :, :, :, :, :] += num
        _cache['int_78'] = I1
    for i_b in range(len(virt_idx)):
        I1 = _cache['int_78'][i_b, :, :, :, :, :]
        b2 = b2_full[:, :, i_b, :]
        num = np.einsum('CDEHI,CFG->CDEFGHI', I1, b2, optimize=True)
        I2 = np.einsum('CDEFGHI->DEFGHI', num, optimize=True)
        g_temp[3][:, :, :, :, :, :] += -1.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_78'] -= 1
    if _use_count['int_78'] == 0:
        del _cache['int_78']
    # Projecting permutation: -1.00 * g(a,b,u,t) * g(i,r,b,s) * g(q,p,a,i) / [(E_corr + e_u + e_t - e_a - e_b)] * [(E_corr + e_u + e_t + e_i - e_b - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,b,u,s) * g(i,r,b,t) * g(q,p,a,i) / [(E_corr + e_u + e_s - e_a - e_b)] * [(E_corr + e_u + e_s + e_i - e_b - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,b,t,s) * g(i,r,b,u) * g(q,p,a,i) / [(E_corr + e_t + e_s - e_a - e_b)] * [(E_corr + e_t + e_s + e_i - e_b - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: +1.00 * g(a,b,u,t) * g(i,q,b,s) * g(r,p,a,i) / [(E_corr + e_u + e_t - e_a - e_b)] * [(E_corr + e_u + e_t + e_i - e_b - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,b,u,s) * g(i,q,b,t) * g(r,p,a,i) / [(E_corr + e_u + e_s - e_a - e_b)] * [(E_corr + e_u + e_s + e_i - e_b - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 3, 5))
    # Projecting permutation: +1.00 * g(a,b,t,s) * g(i,q,b,u) * g(r,p,a,i) / [(E_corr + e_t + e_s - e_a - e_b)] * [(E_corr + e_t + e_s + e_i - e_b - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 5, 3))
    # Projecting permutation: -1.00 * g(a,b,u,t) * g(i,p,b,s) * g(r,q,a,i) / [(E_corr + e_u + e_t - e_a - e_b)] * [(E_corr + e_u + e_t + e_i - e_b - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,b,u,s) * g(i,p,b,t) * g(r,q,a,i) / [(E_corr + e_u + e_s - e_a - e_b)] * [(E_corr + e_u + e_s + e_i - e_b - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,b,t,s) * g(i,p,b,u) * g(r,q,a,i) / [(E_corr + e_t + e_s - e_a - e_b)] * [(E_corr + e_t + e_s + e_i - e_b - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 5, 3))

    # --- Skeleton 733c8b33 contains 9 permutations ---
    # Base Term: -1.00 * g(a,b,u,t) * g(i,r,a,b) * g(q,p,s,i) / [(E_corr + e_u + e_t - e_a - e_b)] * [(E_corr + e_u + e_t + e_s + e_i - e_a - e_b - e_q - e_p)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(act_idx, act_idx, act_idx, occ_idx)]
    if 'int_79' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                for i_i in range(len(occ_idx)):
                    b0 = b0_full[i_a, i_b, :, :]
                    b1 = b1_full[i_i, :, i_a, i_b]
                    d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[:, None] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                    d1[np.abs(d1) < 1e-12] = 1e-12
                    b0 = b0 / d1
                    num = np.einsum('IH,F->FHI', b0, b1, optimize=True)
                    I1[i_a, i_b, i_i, :, :, :] += num
        _cache['int_79'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            for i_i in range(len(occ_idx)):
                I1 = _cache['int_79'][i_a, i_b, i_i, :, :, :]
                b2 = b2_full[:, :, :, i_i]
                num = np.einsum('FHI,EDG->DEFGHI', I1, b2, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, None, :] + (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, None, :, None] + (eps_spin[act_idx] + (shift / 4 if shift is not None else 0.0))[None, None, None, :, None, None] + (eps_spin[occ_idx[i_i]] + (shift / 4 if shift is not None else 0.0)) - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]] - eps_spin[act_idx][None, :, None, None, None, None] - eps_spin[act_idx][:, None, None, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                I2 = num
                g_temp[3][:, :, :, :, :, :] += -1.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_79'] -= 1
    if _use_count['int_79'] == 0:
        del _cache['int_79']
    # Projecting permutation: -1.00 * g(a,b,u,t) * g(i,r,a,b) * g(q,p,s,i) / [(E_corr + e_u + e_t - e_a - e_b)] * [(E_corr + e_u + e_t + e_s + e_i - e_a - e_b - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,b,u,s) * g(i,r,a,b) * g(q,p,t,i) / [(E_corr + e_u + e_s - e_a - e_b)] * [(E_corr + e_u + e_s + e_t + e_i - e_a - e_b - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,b,t,s) * g(i,r,a,b) * g(q,p,u,i) / [(E_corr + e_t + e_s - e_a - e_b)] * [(E_corr + e_t + e_s + e_u + e_i - e_a - e_b - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: +1.00 * g(a,b,u,t) * g(i,q,a,b) * g(r,p,s,i) / [(E_corr + e_u + e_t - e_a - e_b)] * [(E_corr + e_u + e_t + e_s + e_i - e_a - e_b - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,b,u,s) * g(i,q,a,b) * g(r,p,t,i) / [(E_corr + e_u + e_s - e_a - e_b)] * [(E_corr + e_u + e_s + e_t + e_i - e_a - e_b - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 3, 5))
    # Projecting permutation: +1.00 * g(a,b,t,s) * g(i,q,a,b) * g(r,p,u,i) / [(E_corr + e_t + e_s - e_a - e_b)] * [(E_corr + e_t + e_s + e_u + e_i - e_a - e_b - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 5, 3))
    # Projecting permutation: -1.00 * g(a,b,u,t) * g(i,p,a,b) * g(r,q,s,i) / [(E_corr + e_u + e_t - e_a - e_b)] * [(E_corr + e_u + e_t + e_s + e_i - e_a - e_b - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,b,u,s) * g(i,p,a,b) * g(r,q,t,i) / [(E_corr + e_u + e_s - e_a - e_b)] * [(E_corr + e_u + e_s + e_t + e_i - e_a - e_b - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,b,t,s) * g(i,p,a,b) * g(r,q,u,i) / [(E_corr + e_t + e_s - e_a - e_b)] * [(E_corr + e_t + e_s + e_u + e_i - e_a - e_b - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 5, 3))

    # --- Skeleton 8e087a83 contains 36 permutations ---
    # Base Term: +1.00 * g(a,q,b,t) * g(b,p,u,i) * g(i,r,a,s) / [(E_corr + e_u + e_i - e_b - e_p)] * [(E_corr + e_u + e_i + e_t - e_p - e_a - e_q)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_80' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('EH,DIC->CDEHI', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[:, None, None, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[act_idx][None, :, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[:, None, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] - eps_spin[act_idx][None, :, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, None, :, None, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :, :, :] += num
        _cache['int_80'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_80'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDEHI,CFG->CDEFGHI', I1, b2, optimize=True)
        I2 = np.einsum('CDEFGHI->DEFGHI', num, optimize=True)
        g_temp[3][:, :, :, :, :, :] += 1.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_80'] -= 1
    if _use_count['int_80'] == 0:
        del _cache['int_80']
    # Projecting permutation: +1.00 * g(a,q,b,t) * g(b,p,u,i) * g(i,r,a,s) / [(E_corr + e_u + e_i - e_b - e_p)] * [(E_corr + e_u + e_i + e_t - e_p - e_a - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,q,b,s) * g(b,p,u,i) * g(i,r,a,t) / [(E_corr + e_u + e_i - e_b - e_p)] * [(E_corr + e_u + e_i + e_s - e_p - e_a - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,q,b,u) * g(b,p,t,i) * g(i,r,a,s) / [(E_corr + e_t + e_i - e_b - e_p)] * [(E_corr + e_t + e_i + e_u - e_p - e_a - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 5, 4))
    # Projecting permutation: +1.00 * g(a,q,b,u) * g(b,p,s,i) * g(i,r,a,t) / [(E_corr + e_s + e_i - e_b - e_p)] * [(E_corr + e_s + e_i + e_u - e_p - e_a - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 3, 4))
    # Projecting permutation: +1.00 * g(a,q,b,s) * g(b,p,t,i) * g(i,r,a,u) / [(E_corr + e_t + e_i - e_b - e_p)] * [(E_corr + e_t + e_i + e_s - e_p - e_a - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: -1.00 * g(a,q,b,t) * g(b,p,s,i) * g(i,r,a,u) / [(E_corr + e_s + e_i - e_b - e_p)] * [(E_corr + e_s + e_i + e_t - e_p - e_a - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 4, 3))
    # Projecting permutation: -1.00 * g(a,p,b,t) * g(b,q,u,i) * g(i,r,a,s) / [(E_corr + e_u + e_i - e_b - e_q)] * [(E_corr + e_u + e_i + e_t - e_q - e_a - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,p,b,s) * g(b,q,u,i) * g(i,r,a,t) / [(E_corr + e_u + e_i - e_b - e_q)] * [(E_corr + e_u + e_i + e_s - e_q - e_a - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 3, 5))
    # Projecting permutation: +1.00 * g(a,p,b,u) * g(b,q,t,i) * g(i,r,a,s) / [(E_corr + e_t + e_i - e_b - e_q)] * [(E_corr + e_t + e_i + e_u - e_q - e_a - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 5, 4))
    # Projecting permutation: -1.00 * g(a,p,b,u) * g(b,q,s,i) * g(i,r,a,t) / [(E_corr + e_s + e_i - e_b - e_q)] * [(E_corr + e_s + e_i + e_u - e_q - e_a - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 5, 3, 4))
    # Projecting permutation: -1.00 * g(a,p,b,s) * g(b,q,t,i) * g(i,r,a,u) / [(E_corr + e_t + e_i - e_b - e_q)] * [(E_corr + e_t + e_i + e_s - e_q - e_a - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 5, 3))
    # Projecting permutation: +1.00 * g(a,p,b,t) * g(b,q,s,i) * g(i,r,a,u) / [(E_corr + e_s + e_i - e_b - e_q)] * [(E_corr + e_s + e_i + e_t - e_q - e_a - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 5, 4, 3))
    # Projecting permutation: -1.00 * g(a,r,b,t) * g(b,p,u,i) * g(i,q,a,s) / [(E_corr + e_u + e_i - e_b - e_p)] * [(E_corr + e_u + e_i + e_t - e_p - e_a - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,r,b,s) * g(b,p,u,i) * g(i,q,a,t) / [(E_corr + e_u + e_i - e_b - e_p)] * [(E_corr + e_u + e_i + e_s - e_p - e_a - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 3, 5))
    # Projecting permutation: +1.00 * g(a,r,b,u) * g(b,p,t,i) * g(i,q,a,s) / [(E_corr + e_t + e_i - e_b - e_p)] * [(E_corr + e_t + e_i + e_u - e_p - e_a - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 5, 4))
    # Projecting permutation: -1.00 * g(a,r,b,u) * g(b,p,s,i) * g(i,q,a,t) / [(E_corr + e_s + e_i - e_b - e_p)] * [(E_corr + e_s + e_i + e_u - e_p - e_a - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 5, 3, 4))
    # Projecting permutation: -1.00 * g(a,r,b,s) * g(b,p,t,i) * g(i,q,a,u) / [(E_corr + e_t + e_i - e_b - e_p)] * [(E_corr + e_t + e_i + e_s - e_p - e_a - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 5, 3))
    # Projecting permutation: +1.00 * g(a,r,b,t) * g(b,p,s,i) * g(i,q,a,u) / [(E_corr + e_s + e_i - e_b - e_p)] * [(E_corr + e_s + e_i + e_t - e_p - e_a - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 5, 4, 3))
    # Projecting permutation: +1.00 * g(a,p,b,t) * g(b,r,u,i) * g(i,q,a,s) / [(E_corr + e_u + e_i - e_b - e_r)] * [(E_corr + e_u + e_i + e_t - e_r - e_a - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,p,b,s) * g(b,r,u,i) * g(i,q,a,t) / [(E_corr + e_u + e_i - e_b - e_r)] * [(E_corr + e_u + e_i + e_s - e_r - e_a - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,p,b,u) * g(b,r,t,i) * g(i,q,a,s) / [(E_corr + e_t + e_i - e_b - e_r)] * [(E_corr + e_t + e_i + e_u - e_r - e_a - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 5, 4))
    # Projecting permutation: +1.00 * g(a,p,b,u) * g(b,r,s,i) * g(i,q,a,t) / [(E_corr + e_s + e_i - e_b - e_r)] * [(E_corr + e_s + e_i + e_u - e_r - e_a - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 5, 3, 4))
    # Projecting permutation: +1.00 * g(a,p,b,s) * g(b,r,t,i) * g(i,q,a,u) / [(E_corr + e_t + e_i - e_b - e_r)] * [(E_corr + e_t + e_i + e_s - e_r - e_a - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 5, 3))
    # Projecting permutation: -1.00 * g(a,p,b,t) * g(b,r,s,i) * g(i,q,a,u) / [(E_corr + e_s + e_i - e_b - e_r)] * [(E_corr + e_s + e_i + e_t - e_r - e_a - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 5, 4, 3))
    # Projecting permutation: +1.00 * g(a,r,b,t) * g(b,q,u,i) * g(i,p,a,s) / [(E_corr + e_u + e_i - e_b - e_q)] * [(E_corr + e_u + e_i + e_t - e_q - e_a - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,r,b,s) * g(b,q,u,i) * g(i,p,a,t) / [(E_corr + e_u + e_i - e_b - e_q)] * [(E_corr + e_u + e_i + e_s - e_q - e_a - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,r,b,u) * g(b,q,t,i) * g(i,p,a,s) / [(E_corr + e_t + e_i - e_b - e_q)] * [(E_corr + e_t + e_i + e_u - e_q - e_a - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 5, 4))
    # Projecting permutation: +1.00 * g(a,r,b,u) * g(b,q,s,i) * g(i,p,a,t) / [(E_corr + e_s + e_i - e_b - e_q)] * [(E_corr + e_s + e_i + e_u - e_q - e_a - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 5, 3, 4))
    # Projecting permutation: +1.00 * g(a,r,b,s) * g(b,q,t,i) * g(i,p,a,u) / [(E_corr + e_t + e_i - e_b - e_q)] * [(E_corr + e_t + e_i + e_s - e_q - e_a - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 5, 3))
    # Projecting permutation: -1.00 * g(a,r,b,t) * g(b,q,s,i) * g(i,p,a,u) / [(E_corr + e_s + e_i - e_b - e_q)] * [(E_corr + e_s + e_i + e_t - e_q - e_a - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 5, 4, 3))
    # Projecting permutation: -1.00 * g(a,q,b,t) * g(b,r,u,i) * g(i,p,a,s) / [(E_corr + e_u + e_i - e_b - e_r)] * [(E_corr + e_u + e_i + e_t - e_r - e_a - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,q,b,s) * g(b,r,u,i) * g(i,p,a,t) / [(E_corr + e_u + e_i - e_b - e_r)] * [(E_corr + e_u + e_i + e_s - e_r - e_a - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 4, 3, 5))
    # Projecting permutation: +1.00 * g(a,q,b,u) * g(b,r,t,i) * g(i,p,a,s) / [(E_corr + e_t + e_i - e_b - e_r)] * [(E_corr + e_t + e_i + e_u - e_r - e_a - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 3, 5, 4))
    # Projecting permutation: -1.00 * g(a,q,b,u) * g(b,r,s,i) * g(i,p,a,t) / [(E_corr + e_s + e_i - e_b - e_r)] * [(E_corr + e_s + e_i + e_u - e_r - e_a - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 5, 3, 4))
    # Projecting permutation: -1.00 * g(a,q,b,s) * g(b,r,t,i) * g(i,p,a,u) / [(E_corr + e_t + e_i - e_b - e_r)] * [(E_corr + e_t + e_i + e_s - e_r - e_a - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 4, 5, 3))
    # Projecting permutation: +1.00 * g(a,q,b,t) * g(b,r,s,i) * g(i,p,a,u) / [(E_corr + e_s + e_i - e_b - e_r)] * [(E_corr + e_s + e_i + e_t - e_r - e_a - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 1, 0, 5, 4, 3))

    # --- Skeleton 94ccf599 contains 9 permutations ---
    # Base Term: -1.00 * g(a,b,u,i) * g(i,r,t,s) * g(q,p,a,b) / [(E_corr + e_u + e_i - e_a - e_b)] * [(E_corr + e_u + e_i - e_q - e_p)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, virt_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, act_idx, act_idx)]
    if 'int_81' not in _cache:
        I1 = np.zeros((len(occ_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[i_a, :, :, :]
            b1 = b1_full[:, :, i_a, :]
            d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx][:, None, None]
            d1[np.abs(d1) < 1e-12] = 1e-12
            b0 = b0 / d1
            num = np.einsum('BIC,EDB->BCDEI', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[act_idx][None, None, None, :, None] - eps_spin[act_idx][None, None, :, None, None]
            num = num / d0
            I1 += np.einsum('BCDEI->CDEI', num, optimize=True)
        _cache['int_81'] = I1
    I1 = _cache['int_81']
    b2 = b2_full
    num = np.einsum('CDEI,CFHG->CDEFGHI', I1, b2, optimize=True)
    I2 = np.einsum('CDEFGHI->DEFGHI', num, optimize=True)
    g_temp[3][:, :, :, :, :, :] += -1.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_81'] -= 1
    if _use_count['int_81'] == 0:
        del _cache['int_81']
    # Projecting permutation: -1.00 * g(a,b,u,i) * g(i,r,t,s) * g(q,p,a,b) / [(E_corr + e_u + e_i - e_a - e_b)] * [(E_corr + e_u + e_i - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,b,t,i) * g(i,r,u,s) * g(q,p,a,b) / [(E_corr + e_t + e_i - e_a - e_b)] * [(E_corr + e_t + e_i - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 5, 4))
    # Projecting permutation: -1.00 * g(a,b,s,i) * g(i,r,u,t) * g(q,p,a,b) / [(E_corr + e_s + e_i - e_a - e_b)] * [(E_corr + e_s + e_i - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 3, 4))
    # Projecting permutation: +1.00 * g(a,b,u,i) * g(i,q,t,s) * g(r,p,a,b) / [(E_corr + e_u + e_i - e_a - e_b)] * [(E_corr + e_u + e_i - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,b,t,i) * g(i,q,u,s) * g(r,p,a,b) / [(E_corr + e_t + e_i - e_a - e_b)] * [(E_corr + e_t + e_i - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 5, 4))
    # Projecting permutation: +1.00 * g(a,b,s,i) * g(i,q,u,t) * g(r,p,a,b) / [(E_corr + e_s + e_i - e_a - e_b)] * [(E_corr + e_s + e_i - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 5, 3, 4))
    # Projecting permutation: -1.00 * g(a,b,u,i) * g(i,p,t,s) * g(r,q,a,b) / [(E_corr + e_u + e_i - e_a - e_b)] * [(E_corr + e_u + e_i - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,b,t,i) * g(i,p,u,s) * g(r,q,a,b) / [(E_corr + e_t + e_i - e_a - e_b)] * [(E_corr + e_t + e_i - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 5, 4))
    # Projecting permutation: -1.00 * g(a,b,s,i) * g(i,p,u,t) * g(r,q,a,b) / [(E_corr + e_s + e_i - e_a - e_b)] * [(E_corr + e_s + e_i - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 5, 3, 4))

    # --- Skeleton e8fa22c4 contains 18 permutations ---
    # Base Term: +2.00 * g(a,b,u,i) * g(i,r,b,s) * g(q,p,a,t) / [(E_corr + e_u + e_i - e_a - e_b)] * [(E_corr + e_u + e_i + e_t - e_b - e_q - e_p)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, occ_idx)]
    b1_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(occ_idx, act_idx, virt_idx, act_idx)]
    if 'int_82' not in _cache:
        I1 = np.zeros((len(virt_idx), len(occ_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, i_b, :, :]
                b1 = b1_full[:, :, i_a, :]
                d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[:, None] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                b0 = b0 / d1
                num = np.einsum('IC,EDH->CDEHI', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 3 if shift is not None else 0.0))[:, None, None, None, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] - eps_spin[virt_idx[i_b]] - eps_spin[act_idx][None, None, :, None, None] - eps_spin[act_idx][None, :, None, None, None]
                num = num / d0
                I1[i_b, :, :, :, :, :] += num
        _cache['int_82'] = I1
    for i_b in range(len(virt_idx)):
        I1 = _cache['int_82'][i_b, :, :, :, :, :]
        b2 = b2_full[:, :, i_b, :]
        num = np.einsum('CDEHI,CFG->CDEFGHI', I1, b2, optimize=True)
        I2 = np.einsum('CDEFGHI->DEFGHI', num, optimize=True)
        g_temp[3][:, :, :, :, :, :] += 2.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_82'] -= 1
    if _use_count['int_82'] == 0:
        del _cache['int_82']
    # Projecting permutation: +2.00 * g(a,b,u,i) * g(i,r,b,s) * g(q,p,a,t) / [(E_corr + e_u + e_i - e_a - e_b)] * [(E_corr + e_u + e_i + e_t - e_b - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: -2.00 * g(a,b,u,i) * g(i,r,b,t) * g(q,p,a,s) / [(E_corr + e_u + e_i - e_a - e_b)] * [(E_corr + e_u + e_i + e_s - e_b - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: -2.00 * g(a,b,t,i) * g(i,r,b,s) * g(q,p,a,u) / [(E_corr + e_t + e_i - e_a - e_b)] * [(E_corr + e_t + e_i + e_u - e_b - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 5, 4))
    # Projecting permutation: +2.00 * g(a,b,s,i) * g(i,r,b,t) * g(q,p,a,u) / [(E_corr + e_s + e_i - e_a - e_b)] * [(E_corr + e_s + e_i + e_u - e_b - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 3, 4))
    # Projecting permutation: +2.00 * g(a,b,t,i) * g(i,r,b,u) * g(q,p,a,s) / [(E_corr + e_t + e_i - e_a - e_b)] * [(E_corr + e_t + e_i + e_s - e_b - e_q - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: -2.00 * g(a,b,s,i) * g(i,r,b,u) * g(q,p,a,t) / [(E_corr + e_s + e_i - e_a - e_b)] * [(E_corr + e_s + e_i + e_t - e_b - e_q - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 4, 3))
    # Projecting permutation: -2.00 * g(a,b,u,i) * g(i,q,b,s) * g(r,p,a,t) / [(E_corr + e_u + e_i - e_a - e_b)] * [(E_corr + e_u + e_i + e_t - e_b - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 4, 5))
    # Projecting permutation: +2.00 * g(a,b,u,i) * g(i,q,b,t) * g(r,p,a,s) / [(E_corr + e_u + e_i - e_a - e_b)] * [(E_corr + e_u + e_i + e_s - e_b - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 3, 5))
    # Projecting permutation: +2.00 * g(a,b,t,i) * g(i,q,b,s) * g(r,p,a,u) / [(E_corr + e_t + e_i - e_a - e_b)] * [(E_corr + e_t + e_i + e_u - e_b - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 3, 5, 4))
    # Projecting permutation: -2.00 * g(a,b,s,i) * g(i,q,b,t) * g(r,p,a,u) / [(E_corr + e_s + e_i - e_a - e_b)] * [(E_corr + e_s + e_i + e_u - e_b - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 5, 3, 4))
    # Projecting permutation: -2.00 * g(a,b,t,i) * g(i,q,b,u) * g(r,p,a,s) / [(E_corr + e_t + e_i - e_a - e_b)] * [(E_corr + e_t + e_i + e_s - e_b - e_r - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 4, 5, 3))
    # Projecting permutation: +2.00 * g(a,b,s,i) * g(i,q,b,u) * g(r,p,a,t) / [(E_corr + e_s + e_i - e_a - e_b)] * [(E_corr + e_s + e_i + e_t - e_b - e_r - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 2, 1, 5, 4, 3))
    # Projecting permutation: +2.00 * g(a,b,u,i) * g(i,p,b,s) * g(r,q,a,t) / [(E_corr + e_u + e_i - e_a - e_b)] * [(E_corr + e_u + e_i + e_t - e_b - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 4, 5))
    # Projecting permutation: -2.00 * g(a,b,u,i) * g(i,p,b,t) * g(r,q,a,s) / [(E_corr + e_u + e_i - e_a - e_b)] * [(E_corr + e_u + e_i + e_s - e_b - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 3, 5))
    # Projecting permutation: -2.00 * g(a,b,t,i) * g(i,p,b,s) * g(r,q,a,u) / [(E_corr + e_t + e_i - e_a - e_b)] * [(E_corr + e_t + e_i + e_u - e_b - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 3, 5, 4))
    # Projecting permutation: +2.00 * g(a,b,s,i) * g(i,p,b,t) * g(r,q,a,u) / [(E_corr + e_s + e_i - e_a - e_b)] * [(E_corr + e_s + e_i + e_u - e_b - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 5, 3, 4))
    # Projecting permutation: +2.00 * g(a,b,t,i) * g(i,p,b,u) * g(r,q,a,s) / [(E_corr + e_t + e_i - e_a - e_b)] * [(E_corr + e_t + e_i + e_s - e_b - e_r - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 4, 5, 3))
    # Projecting permutation: -2.00 * g(a,b,s,i) * g(i,p,b,u) * g(r,q,a,t) / [(E_corr + e_s + e_i - e_a - e_b)] * [(E_corr + e_s + e_i + e_t - e_b - e_r - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (2, 0, 1, 5, 4, 3))

    # --- Skeleton 223b1309 contains 18 permutations ---
    # Base Term: -1.00 * g(i,a,b,t) * g(b,p,u,i) * g(r,q,a,s) / [(E_corr + e_u + e_i - e_b - e_p)] * [(E_corr + e_u + e_t - e_p - e_a)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, act_idx)]
    if 'int_83' not in _cache:
        I1 = np.zeros((len(virt_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            b0 = b0_full[:, i_a, :, :]
            b1 = b1_full[:, :, :, :]
            num = np.einsum('CBH,BDIC->BCDHI', b0, b1, optimize=True)
            d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[None, :, None, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, None, :, None, None]
            d0[np.abs(d0) < 1e-12] = 1e-12
            num = num / d0
            d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None] - eps_spin[act_idx][None, None, :, None, None] - eps_spin[virt_idx[i_a]]
            d1[np.abs(d1) < 1e-12] = 1e-12
            num = num / d1
            I1[i_a, :, :, :] += np.einsum('BCDHI->DHI', num, optimize=True)
        _cache['int_83'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_83'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('DHI,FEG->DEFGHI', I1, b2, optimize=True)
        I2 = num
        g_temp[3][:, :, :, :, :, :] += -1.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_83'] -= 1
    if _use_count['int_83'] == 0:
        del _cache['int_83']
    # Projecting permutation: -1.00 * g(i,a,b,t) * g(b,p,u,i) * g(r,q,a,s) / [(E_corr + e_u + e_i - e_b - e_p)] * [(E_corr + e_u + e_t - e_p - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(i,a,b,s) * g(b,p,u,i) * g(r,q,a,t) / [(E_corr + e_u + e_i - e_b - e_p)] * [(E_corr + e_u + e_s - e_p - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: +1.00 * g(i,a,b,u) * g(b,p,t,i) * g(r,q,a,s) / [(E_corr + e_t + e_i - e_b - e_p)] * [(E_corr + e_t + e_u - e_p - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 5, 4))
    # Projecting permutation: -1.00 * g(i,a,b,u) * g(b,p,s,i) * g(r,q,a,t) / [(E_corr + e_s + e_i - e_b - e_p)] * [(E_corr + e_s + e_u - e_p - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 3, 4))
    # Projecting permutation: -1.00 * g(i,a,b,s) * g(b,p,t,i) * g(r,q,a,u) / [(E_corr + e_t + e_i - e_b - e_p)] * [(E_corr + e_t + e_s - e_p - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: +1.00 * g(i,a,b,t) * g(b,p,s,i) * g(r,q,a,u) / [(E_corr + e_s + e_i - e_b - e_p)] * [(E_corr + e_s + e_t - e_p - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 4, 3))
    # Projecting permutation: +1.00 * g(i,a,b,t) * g(b,q,u,i) * g(r,p,a,s) / [(E_corr + e_u + e_i - e_b - e_q)] * [(E_corr + e_u + e_t - e_q - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 4, 5))
    # Projecting permutation: -1.00 * g(i,a,b,s) * g(b,q,u,i) * g(r,p,a,t) / [(E_corr + e_u + e_i - e_b - e_q)] * [(E_corr + e_u + e_s - e_q - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 3, 5))
    # Projecting permutation: -1.00 * g(i,a,b,u) * g(b,q,t,i) * g(r,p,a,s) / [(E_corr + e_t + e_i - e_b - e_q)] * [(E_corr + e_t + e_u - e_q - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 5, 4))
    # Projecting permutation: +1.00 * g(i,a,b,u) * g(b,q,s,i) * g(r,p,a,t) / [(E_corr + e_s + e_i - e_b - e_q)] * [(E_corr + e_s + e_u - e_q - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 5, 3, 4))
    # Projecting permutation: +1.00 * g(i,a,b,s) * g(b,q,t,i) * g(r,p,a,u) / [(E_corr + e_t + e_i - e_b - e_q)] * [(E_corr + e_t + e_s - e_q - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 5, 3))
    # Projecting permutation: -1.00 * g(i,a,b,t) * g(b,q,s,i) * g(r,p,a,u) / [(E_corr + e_s + e_i - e_b - e_q)] * [(E_corr + e_s + e_t - e_q - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 5, 4, 3))
    # Projecting permutation: -1.00 * g(i,a,b,t) * g(b,r,u,i) * g(q,p,a,s) / [(E_corr + e_u + e_i - e_b - e_r)] * [(E_corr + e_u + e_t - e_r - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 4, 5))
    # Projecting permutation: +1.00 * g(i,a,b,s) * g(b,r,u,i) * g(q,p,a,t) / [(E_corr + e_u + e_i - e_b - e_r)] * [(E_corr + e_u + e_s - e_r - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 3, 5))
    # Projecting permutation: +1.00 * g(i,a,b,u) * g(b,r,t,i) * g(q,p,a,s) / [(E_corr + e_t + e_i - e_b - e_r)] * [(E_corr + e_t + e_u - e_r - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 5, 4))
    # Projecting permutation: -1.00 * g(i,a,b,u) * g(b,r,s,i) * g(q,p,a,t) / [(E_corr + e_s + e_i - e_b - e_r)] * [(E_corr + e_s + e_u - e_r - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 5, 3, 4))
    # Projecting permutation: -1.00 * g(i,a,b,s) * g(b,r,t,i) * g(q,p,a,u) / [(E_corr + e_t + e_i - e_b - e_r)] * [(E_corr + e_t + e_s - e_r - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 5, 3))
    # Projecting permutation: +1.00 * g(i,a,b,t) * g(b,r,s,i) * g(q,p,a,u) / [(E_corr + e_s + e_i - e_b - e_r)] * [(E_corr + e_s + e_t - e_r - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 5, 4, 3))

    # --- Skeleton 948afb76 contains 9 permutations ---
    # Base Term: -1.00 * g(i,a,t,s) * g(b,p,u,i) * g(r,q,a,b) / [(E_corr + e_u + e_i - e_b - e_p)] * [(E_corr + e_u + e_t + e_s - e_b - e_p - e_a)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(occ_idx, virt_idx, act_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, occ_idx)]
    b2_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, virt_idx)]
    if 'int_84' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[:, i_a, :, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('CHG,DIC->CDGHI', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[occ_idx] + (shift / 2 if shift is not None else 0.0))[:, None, None, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[act_idx][None, :, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None] - eps_spin[virt_idx[i_b]] - eps_spin[act_idx][None, :, None, None, None] - eps_spin[virt_idx[i_a]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, i_b, :, :, :, :] += np.einsum('CDGHI->DGHI', num, optimize=True)
        _cache['int_84'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_84'][i_a, i_b, :, :, :, :]
            b2 = b2_full[:, :, i_a, i_b]
            num = np.einsum('DGHI,FE->DEFGHI', I1, b2, optimize=True)
            I2 = num
            g_temp[3][:, :, :, :, :, :] += -1.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_84'] -= 1
    if _use_count['int_84'] == 0:
        del _cache['int_84']
    # Projecting permutation: -1.00 * g(i,a,t,s) * g(b,p,u,i) * g(r,q,a,b) / [(E_corr + e_u + e_i - e_b - e_p)] * [(E_corr + e_u + e_t + e_s - e_b - e_p - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(i,a,u,s) * g(b,p,t,i) * g(r,q,a,b) / [(E_corr + e_t + e_i - e_b - e_p)] * [(E_corr + e_t + e_u + e_s - e_b - e_p - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 5, 4))
    # Projecting permutation: -1.00 * g(i,a,u,t) * g(b,p,s,i) * g(r,q,a,b) / [(E_corr + e_s + e_i - e_b - e_p)] * [(E_corr + e_s + e_u + e_t - e_b - e_p - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 5, 3, 4))
    # Projecting permutation: +1.00 * g(i,a,t,s) * g(b,q,u,i) * g(r,p,a,b) / [(E_corr + e_u + e_i - e_b - e_q)] * [(E_corr + e_u + e_t + e_s - e_b - e_q - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 4, 5))
    # Projecting permutation: -1.00 * g(i,a,u,s) * g(b,q,t,i) * g(r,p,a,b) / [(E_corr + e_t + e_i - e_b - e_q)] * [(E_corr + e_t + e_u + e_s - e_b - e_q - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 5, 4))
    # Projecting permutation: +1.00 * g(i,a,u,t) * g(b,q,s,i) * g(r,p,a,b) / [(E_corr + e_s + e_i - e_b - e_q)] * [(E_corr + e_s + e_u + e_t - e_b - e_q - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 5, 3, 4))
    # Projecting permutation: -1.00 * g(i,a,t,s) * g(b,r,u,i) * g(q,p,a,b) / [(E_corr + e_u + e_i - e_b - e_r)] * [(E_corr + e_u + e_t + e_s - e_b - e_r - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 4, 5))
    # Projecting permutation: +1.00 * g(i,a,u,s) * g(b,r,t,i) * g(q,p,a,b) / [(E_corr + e_t + e_i - e_b - e_r)] * [(E_corr + e_t + e_u + e_s - e_b - e_r - e_a)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 5, 4))
    # Projecting permutation: -1.00 * g(i,a,u,t) * g(b,r,s,i) * g(q,p,a,b) / [(E_corr + e_s + e_i - e_b - e_r)] * [(E_corr + e_s + e_u + e_t - e_b - e_r - e_a)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 5, 3, 4))

    # --- Skeleton 0e80c6ed contains 9 permutations ---
    # Base Term: -0.50 * g(a,b,c,s) * g(c,p,u,t) * g(r,q,a,b) / [(E_corr + e_u + e_t - e_c - e_p)] * [(E_corr + e_u + e_t + e_s - e_p - e_a - e_b)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, virt_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, act_idx, act_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, virt_idx)]
    if 'int_85' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, i_b, :, :]
                b1 = b1_full[:, :, :, :]
                num = np.einsum('CG,CDIH->CDGHI', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[act_idx][None, :, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None] - eps_spin[act_idx][None, :, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[virt_idx[i_b]]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, i_b, :, :, :, :] += np.einsum('CDGHI->DGHI', num, optimize=True)
        _cache['int_85'] = I1
    for i_a in range(len(virt_idx)):
        for i_b in range(len(virt_idx)):
            I1 = _cache['int_85'][i_a, i_b, :, :, :, :]
            b2 = b2_full[:, :, i_a, i_b]
            num = np.einsum('DGHI,FE->DEFGHI', I1, b2, optimize=True)
            I2 = num
            g_temp[3][:, :, :, :, :, :] += -0.5 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_85'] -= 1
    if _use_count['int_85'] == 0:
        del _cache['int_85']
    # Projecting permutation: -0.50 * g(a,b,c,s) * g(c,p,u,t) * g(r,q,a,b) / [(E_corr + e_u + e_t - e_c - e_p)] * [(E_corr + e_u + e_t + e_s - e_p - e_a - e_b)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +0.50 * g(a,b,c,t) * g(c,p,u,s) * g(r,q,a,b) / [(E_corr + e_u + e_s - e_c - e_p)] * [(E_corr + e_u + e_s + e_t - e_p - e_a - e_b)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: -0.50 * g(a,b,c,u) * g(c,p,t,s) * g(r,q,a,b) / [(E_corr + e_t + e_s - e_c - e_p)] * [(E_corr + e_t + e_s + e_u - e_p - e_a - e_b)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: +0.50 * g(a,b,c,s) * g(c,q,u,t) * g(r,p,a,b) / [(E_corr + e_u + e_t - e_c - e_q)] * [(E_corr + e_u + e_t + e_s - e_q - e_a - e_b)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 4, 5))
    # Projecting permutation: -0.50 * g(a,b,c,t) * g(c,q,u,s) * g(r,p,a,b) / [(E_corr + e_u + e_s - e_c - e_q)] * [(E_corr + e_u + e_s + e_t - e_q - e_a - e_b)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 3, 5))
    # Projecting permutation: +0.50 * g(a,b,c,u) * g(c,q,t,s) * g(r,p,a,b) / [(E_corr + e_t + e_s - e_c - e_q)] * [(E_corr + e_t + e_s + e_u - e_q - e_a - e_b)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 5, 3))
    # Projecting permutation: -0.50 * g(a,b,c,s) * g(c,r,u,t) * g(q,p,a,b) / [(E_corr + e_u + e_t - e_c - e_r)] * [(E_corr + e_u + e_t + e_s - e_r - e_a - e_b)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 4, 5))
    # Projecting permutation: +0.50 * g(a,b,c,t) * g(c,r,u,s) * g(q,p,a,b) / [(E_corr + e_u + e_s - e_c - e_r)] * [(E_corr + e_u + e_s + e_t - e_r - e_a - e_b)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 3, 5))
    # Projecting permutation: -0.50 * g(a,b,c,u) * g(c,r,t,s) * g(q,p,a,b) / [(E_corr + e_t + e_s - e_c - e_r)] * [(E_corr + e_t + e_s + e_u - e_r - e_a - e_b)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 5, 3))

    # --- Skeleton 2edb640d contains 9 permutations ---
    # Base Term: -0.50 * g(a,p,b,c) * g(b,c,u,t) * g(r,q,a,s) / [(E_corr + e_u + e_t - e_b - e_c)] * [(E_corr + e_u + e_t - e_a - e_p)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, virt_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, act_idx)]
    if 'int_86' not in _cache:
        I1 = np.zeros((len(virt_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('DC,CIH->CDHI', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, :, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, :, None, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :] += np.einsum('CDHI->DHI', num, optimize=True)
        _cache['int_86'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_86'][i_a, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('DHI,FEG->DEFGHI', I1, b2, optimize=True)
        I2 = num
        g_temp[3][:, :, :, :, :, :] += -0.5 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_86'] -= 1
    if _use_count['int_86'] == 0:
        del _cache['int_86']
    # Projecting permutation: -0.50 * g(a,p,b,c) * g(b,c,u,t) * g(r,q,a,s) / [(E_corr + e_u + e_t - e_b - e_c)] * [(E_corr + e_u + e_t - e_a - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: +0.50 * g(a,p,b,c) * g(b,c,u,s) * g(r,q,a,t) / [(E_corr + e_u + e_s - e_b - e_c)] * [(E_corr + e_u + e_s - e_a - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: -0.50 * g(a,p,b,c) * g(b,c,t,s) * g(r,q,a,u) / [(E_corr + e_t + e_s - e_b - e_c)] * [(E_corr + e_t + e_s - e_a - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: +0.50 * g(a,q,b,c) * g(b,c,u,t) * g(r,p,a,s) / [(E_corr + e_u + e_t - e_b - e_c)] * [(E_corr + e_u + e_t - e_a - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 4, 5))
    # Projecting permutation: -0.50 * g(a,q,b,c) * g(b,c,u,s) * g(r,p,a,t) / [(E_corr + e_u + e_s - e_b - e_c)] * [(E_corr + e_u + e_s - e_a - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 3, 5))
    # Projecting permutation: +0.50 * g(a,q,b,c) * g(b,c,t,s) * g(r,p,a,u) / [(E_corr + e_t + e_s - e_b - e_c)] * [(E_corr + e_t + e_s - e_a - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 5, 3))
    # Projecting permutation: -0.50 * g(a,r,b,c) * g(b,c,u,t) * g(q,p,a,s) / [(E_corr + e_u + e_t - e_b - e_c)] * [(E_corr + e_u + e_t - e_a - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 4, 5))
    # Projecting permutation: +0.50 * g(a,r,b,c) * g(b,c,u,s) * g(q,p,a,t) / [(E_corr + e_u + e_s - e_b - e_c)] * [(E_corr + e_u + e_s - e_a - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 3, 5))
    # Projecting permutation: -0.50 * g(a,r,b,c) * g(b,c,t,s) * g(q,p,a,u) / [(E_corr + e_t + e_s - e_b - e_c)] * [(E_corr + e_t + e_s - e_a - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 5, 3))

    # --- Skeleton 07399aee contains 9 permutations ---
    # Base Term: +1.00 * g(a,p,b,s) * g(b,c,u,t) * g(r,q,a,c) / [(E_corr + e_u + e_t - e_b - e_c)] * [(E_corr + e_u + e_t + e_s - e_c - e_a - e_p)]
    g_temp = {3: np.zeros((len(act_idx),) * 6)}
    b0_full = g_anti_spin[np.ix_(virt_idx, act_idx, virt_idx, act_idx)]
    b1_full = g_anti_spin[np.ix_(virt_idx, virt_idx, act_idx, act_idx)]
    b2_full = g_anti_spin[np.ix_(act_idx, act_idx, virt_idx, virt_idx)]
    if 'int_87' not in _cache:
        I1 = np.zeros((len(virt_idx), len(virt_idx), len(act_idx), len(act_idx), len(act_idx), len(act_idx)))
        for i_a in range(len(virt_idx)):
            for i_b in range(len(virt_idx)):
                b0 = b0_full[i_a, :, i_b, :]
                b1 = b1_full[i_b, :, :, :]
                num = np.einsum('DG,CIH->CDGHI', b0, b1, optimize=True)
                d0 = (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 2 if shift is not None else 0.0))[None, None, None, :, None] - eps_spin[virt_idx[i_b]] - eps_spin[virt_idx][:, None, None, None, None]
                d0[np.abs(d0) < 1e-12] = 1e-12
                num = num / d0
                d1 = (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, None, :] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, None, :, None] + (eps_spin[act_idx] + (shift / 3 if shift is not None else 0.0))[None, None, :, None, None] - eps_spin[virt_idx][:, None, None, None, None] - eps_spin[virt_idx[i_a]] - eps_spin[act_idx][None, :, None, None, None]
                d1[np.abs(d1) < 1e-12] = 1e-12
                num = num / d1
                I1[i_a, :, :, :, :, :] += num
        _cache['int_87'] = I1
    for i_a in range(len(virt_idx)):
        I1 = _cache['int_87'][i_a, :, :, :, :, :]
        b2 = b2_full[:, :, i_a, :]
        num = np.einsum('CDGHI,FEC->CDEFGHI', I1, b2, optimize=True)
        I2 = np.einsum('CDEFGHI->DEFGHI', num, optimize=True)
        g_temp[3][:, :, :, :, :, :] += 1.0 * np.einsum('DEFGHI->DEFGHI', I2, optimize=True)
    _use_count['int_87'] -= 1
    if _use_count['int_87'] == 0:
        del _cache['int_87']
    # Projecting permutation: +1.00 * g(a,p,b,s) * g(b,c,u,t) * g(r,q,a,c) / [(E_corr + e_u + e_t - e_b - e_c)] * [(E_corr + e_u + e_t + e_s - e_c - e_a - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,p,b,t) * g(b,c,u,s) * g(r,q,a,c) / [(E_corr + e_u + e_s - e_b - e_c)] * [(E_corr + e_u + e_s + e_t - e_c - e_a - e_p)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 3, 5))
    # Projecting permutation: +1.00 * g(a,p,b,u) * g(b,c,t,s) * g(r,q,a,c) / [(E_corr + e_t + e_s - e_b - e_c)] * [(E_corr + e_t + e_s + e_u - e_c - e_a - e_p)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (0, 1, 2, 4, 5, 3))
    # Projecting permutation: -1.00 * g(a,q,b,s) * g(b,c,u,t) * g(r,p,a,c) / [(E_corr + e_u + e_t - e_b - e_c)] * [(E_corr + e_u + e_t + e_s - e_c - e_a - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 3, 4, 5))
    # Projecting permutation: +1.00 * g(a,q,b,t) * g(b,c,u,s) * g(r,p,a,c) / [(E_corr + e_u + e_s - e_b - e_c)] * [(E_corr + e_u + e_s + e_t - e_c - e_a - e_q)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 3, 5))
    # Projecting permutation: -1.00 * g(a,q,b,u) * g(b,c,t,s) * g(r,p,a,c) / [(E_corr + e_t + e_s - e_b - e_c)] * [(E_corr + e_t + e_s + e_u - e_c - e_a - e_q)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 0, 2, 4, 5, 3))
    # Projecting permutation: +1.00 * g(a,r,b,s) * g(b,c,u,t) * g(q,p,a,c) / [(E_corr + e_u + e_t - e_b - e_c)] * [(E_corr + e_u + e_t + e_s - e_c - e_a - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 3, 4, 5))
    # Projecting permutation: -1.00 * g(a,r,b,t) * g(b,c,u,s) * g(q,p,a,c) / [(E_corr + e_u + e_s - e_b - e_c)] * [(E_corr + e_u + e_s + e_t - e_c - e_a - e_r)]
    g_eff[3] += -1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 3, 5))
    # Projecting permutation: +1.00 * g(a,r,b,u) * g(b,c,t,s) * g(q,p,a,c) / [(E_corr + e_t + e_s - e_b - e_c)] * [(E_corr + e_t + e_s + e_u - e_c - e_a - e_r)]
    g_eff[3] += 1.000000000000 * np.transpose(g_temp[3], (1, 2, 0, 4, 5, 3))
    return g_eff

