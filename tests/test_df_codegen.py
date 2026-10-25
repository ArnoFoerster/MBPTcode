import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.CodeGeneration.pdaggerq_interface import raw_terms_to_einsum
from src.CodeGeneration.df_codegen import dress_integral_factor


def _exact_B(norb, naux_seed, rng):
    """Exact (naux = norb^2) RI factor: B[Q,p,r] such that
    sum_Q B[Q,p,r]*B[Q,q,s] reproduces an arbitrary PSD 'Coulomb' tensor
    V[p,q,r,s] to machine precision -- the plumbing-only exact-decomposition
    check the DF rewrite called for, independent of any
    real RI approximation error.
    """
    A = rng.standard_normal((norb * norb, norb * norb))
    M = A.T @ A  # symmetric PSD -> V[p,r,q,s] = M[(p,r),(q,s)] is a valid Coulomb-type tensor
    w, v = np.linalg.eigh(M)
    keep = w > 1e-10
    B = (v[:, keep] * np.sqrt(w[keep])).T.reshape(-1, norb, norb)
    V = np.einsum('Qpr,Qqs->pqrs', B, B)
    return B, V


def _run(raw_terms, out_indices, env):
    lines = raw_terms_to_einsum(raw_terms, 'E', out_indices)
    ns = dict(env)
    ns['np'] = np
    ns['einsum'] = np.einsum
    ns['E'] = 0.0
    dedented = "\n".join(l[4:] if l.startswith('    ') else l for l in lines)
    exec(compile(dedented, '<generated>', 'exec'), ns)
    return ns['E']


def main():
    all_ok = True
    rng = np.random.default_rng(0)

    # --- Case 1: unsuffixed (spin-orbital) bracket, both Coulomb+exchange ---
    norb, no, nv = 6, 3, 3
    o, v = slice(0, no), slice(no, norb)
    B, V = _exact_B(norb, norb * norb, rng)
    g_full = V - V.transpose(0, 1, 3, 2)  # <pq||rs> = <pq|rs> - <pq|sr>
    t2 = rng.standard_normal((nv, nv, no, no))

    direct_terms = [['0.25', '<i,j||a,b>', 't2(a,b,i,j)']]
    E_direct = _run(direct_terms, [], dict(g=g_full, t2=t2, o=o, v=v))

    df_terms = dress_integral_factor(direct_terms)
    E_df = _run(df_terms, [], dict(B=B, t2=t2, o=o, v=v))

    ok = np.isclose(E_direct, E_df, atol=1e-10, rtol=0)
    all_ok &= ok
    print(f"unsuffixed bracket DF rewrite reproduces dense contraction: {'OK' if ok else 'FAIL'} "
          f"(direct={E_direct:.12g}, df={E_df:.12g})")

    # --- Case 2: restricted spin-blocked, same-spin suffix (_aaaa), both terms ---
    B_aa, V_aa = _exact_B(norb, norb * norb, rng)
    g_aaaa = V_aa - V_aa.transpose(0, 1, 3, 2)
    t2_aaaa = rng.standard_normal((nv, nv, no, no))

    direct_terms = [['0.25', '<i,j||a,b>_aaaa', 't2_aaaa(a,b,i,j)']]
    E_direct = _run(direct_terms, [], dict(g_aaaa=g_aaaa, t2_aaaa=t2_aaaa, o=o, v=v))

    df_terms = dress_integral_factor(direct_terms)
    E_df = _run(df_terms, [], dict(B_aa=B_aa, t2_aaaa=t2_aaaa, o=o, v=v))

    ok = np.isclose(E_direct, E_df, atol=1e-10, rtol=0)
    all_ok &= ok
    print(f"_aaaa suffix DF rewrite reproduces dense contraction: {'OK' if ok else 'FAIL'} "
          f"(direct={E_direct:.12g}, df={E_df:.12g})")

    # --- Case 3: restricted spin-blocked, cross-spin suffix (_abab), Coulomb-only ---
    B_bb, V_bb = _exact_B(norb, norb * norb, rng)
    g_abab = np.einsum('Qpr,Qqs->pqrs', B_aa, B_bb)  # no exchange term
    t2_abab = rng.standard_normal((nv, nv, no, no))

    direct_terms = [['1.0', '<i,j||a,b>_abab', 't2_abab(a,b,i,j)']]
    E_direct = _run(direct_terms, [], dict(g_abab=g_abab, t2_abab=t2_abab, o=o, v=v))

    df_terms = dress_integral_factor(direct_terms)
    # exactly one term should survive (no spin-forbidden exchange piece)
    ok_count = len(df_terms) == 1
    all_ok &= ok_count
    print(f"_abab suffix DF rewrite emits exactly one (Coulomb-only) term: {'OK' if ok_count else 'FAIL'} "
          f"(got {len(df_terms)})")

    E_df = _run(df_terms, [], dict(B_aa=B_aa, B_bb=B_bb, t2_abab=t2_abab, o=o, v=v))
    ok = np.isclose(E_direct, E_df, atol=1e-10, rtol=0)
    all_ok &= ok
    print(f"_abab suffix DF rewrite reproduces dense contraction: {'OK' if ok else 'FAIL'} "
          f"(direct={E_direct:.12g}, df={E_df:.12g})")

    # --- Case 4: two brackets in one term (cross-check aux-letter freshness) ---
    t2b = rng.standard_normal((nv, nv, no, no))
    direct_terms = [['1.0', '<i,j||a,b>_aaaa', '<a,b||i,j>_aaaa']]
    df_terms = dress_integral_factor(direct_terms)
    ok = len(df_terms) == 4  # 2 brackets x {Coulomb, exchange} each
    all_ok &= ok
    print(f"two-bracket term expands to 2x2=4 DF terms: {'OK' if ok else 'FAIL'} (got {len(df_terms)})")

    E_direct = _run(direct_terms, [], dict(g_aaaa=g_aaaa, o=o, v=v))
    E_df = _run(df_terms, [], dict(B_aa=B_aa, o=o, v=v))
    ok = np.isclose(E_direct, E_df, atol=1e-10, rtol=0)
    all_ok &= ok
    print(f"two-bracket term DF rewrite reproduces dense contraction: {'OK' if ok else 'FAIL'} "
          f"(direct={E_direct:.12g}, df={E_df:.12g})")

    # --- Case 5: pass-through (no bracket factor) is untouched ---
    passthrough = [['1.0', 't2_aaaa(a,b,i,j)', 'd(i,j)']]
    ok = dress_integral_factor(passthrough) == passthrough
    all_ok &= ok
    print(f"bracket-free term passes through unchanged: {'OK' if ok else 'FAIL'}")

    print("\nALL PASSED" if all_ok else "\nFAILURES DETECTED")
    return all_ok


def test_main():
    assert main()


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
