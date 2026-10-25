"""Epstein-Nesbet diagonal shifts for MBPT doubles denominators.

Reference: Jiang & Engel, J. Chem. Phys. 125, 184108 (2006). Their Eq. (18), the
hole-hole-only "HHEN" resummation

    Ec_HHEN = 1/4 sum_ijab <ij||ab>^2 / (eps_i+eps_j-eps_a-eps_b - <ij||ij>)

generalizes (Eqs. 19-20, full Epstein-Nesbet) to the particle-particle ladder and the
four hole-particle cross terms:

    Delta^EN_ijab = <ij||ij> + <ab||ab> - <ia||ia> - <jb||jb> - <ib||ib> - <ja||ja>

with D_EN = D_bare - Delta. HHEN alone is sign-definite (bounded shift, no vanishing
denominator near degeneracy) and is what the paper recommends; the full shift is not,
and their Table I shows wrong-sign results for several ions with it. The three channels
are independent toggles here so all combinations can be swept.

Method-agnostic: consumed by the ADC solvers (U^(2) amplitude denominators) and
available to the MPn density-matrix drivers (compute_t2_1_custom_denom).
"""
import numpy as np

from src.Base.pyscf.pyscf_interface import (get_coulomb_exchange_diagonals,
                                            get_coulomb_exchange_diagonals_df)


def _diag_ladder(g, idx1, idx2):
    """<pq||pq> (spin-orbital, diagonal in both bra and ket) for p over idx1, q over
    idx2 -- the primitive every shift below is built from. Returns (len(idx1), len(idx2))."""
    return np.einsum('pqpq->pq', g[idx1, idx2, idx1, idx2], optimize=True)


def diag_ladder_df(B, idx1, idx2, W_aux=None):
    """_diag_ladder from a spin-orbital DF factor B (naux, nso, nso) instead of
    dense g -- <pq||pq> = sum_Q B[Q,p,p]B[Q,q,q] - sum_Q B[Q,p,q]B[Q,q,p]
    (rank<=2 intermediates; validated identical to the dense version).

    W_aux: (naux, naux) static inverse-dielectric metric
    (static_screened_coulomb_aux[_uhf]); with it the DIRECT term J is
    statically RPA-screened, J_W - K (BSE-kernel channel split: the
    exchange/ring term K stays bare -- see denominators.build_dressed_e_ai's
    docstring). When idx1/idx2 mix spins (e.g. a spin-orbital blockstacked B
    with cross-spin blocks zero), the cross-spin elements of both J_W and K
    vanish from B's own block structure, so no spin bookkeeping is needed."""
    d1 = np.einsum('Qpp->Qp', B[:, idx1][:, :, idx1])
    d2 = np.einsum('Qqq->Qq', B[:, idx2][:, :, idx2])
    if W_aux is None:
        J = np.einsum('Qp,Qq->pq', d1, d2, optimize=True)
    else:
        J = d1.T @ W_aux @ d2
    B12 = B[:, idx1][:, :, idx2]
    B21 = B[:, idx2][:, :, idx1]
    K = np.einsum('Qpq,Qqp->pq', B12, B21, optimize=True)
    return J - K


def epstein_nesbet_shift(g, no, layout='pphh', bare_sign=1.0, hh=True, pp=False, hp=False,
                         B=None):
    """Diagonal Epstein-Nesbet shift for a spin-orbital doubles amplitude, built from
    diagonal ladder integrals of the index pairs ALREADY present in the amplitude's own
    denominator, so there is no pairing ambiguity to resolve.

    layout is PURELY the axis order of the tensor being dressed -- 'pphh'
    (particle,particle,hole,hole) or 'hhpp'. It is INDEPENDENT of D_bare's sign
    convention: the production T2^(1) denominator uses 'pphh' AXIS order but is built as
    eps_o - eps_v, i.e. bare_sign=+1. Conflating the two silently flips the shift's
    direction for any caller whose axis order and sign convention differ.

    bare_sign: +1 if D_bare = (hole eps) - (particle eps) (Jiang & Engel's own, negative
    convention), -1 if particles-minus-holes. Sets the overall sign so that
    epstein_nesbet_denominator's `D_bare - shift` always increases |D_bare|.

    no: number of spin-orbital occupieds. Returns (nv,nv,no,no) for 'pphh' or
    (no,no,nv,nv) for 'hhpp'.

    B: optional spin-orbital DF factor (naux, nso, nso) -- when given, the
    diagonal ladder blocks are built from B (diag_ladder_df, rank<=2, no
    dense g touched) and `g` may be None. Same result to machine precision
    when B exactly factorizes g."""
    norb = len(g) if B is None else B.shape[1]
    o, v = slice(0, no), slice(no, norb)
    nv = norb - no
    ladder = _diag_ladder if B is None else (lambda _g, i1, i2: diag_ladder_df(B, i1, i2))
    if layout == 'pphh':
        p_axes, h_axes, shape = (0, 1), (2, 3), (nv, nv, no, no)
    elif layout == 'hhpp':
        p_axes, h_axes, shape = (2, 3), (0, 1), (no, no, nv, nv)
    else:
        raise ValueError(f"layout={layout!r}; expected 'pphh' or 'hhpp'")

    def _place(mat, ax_a, ax_b):
        if ax_a > ax_b:              # 'hhpp' puts the hole axis of a cross pair first
            mat, ax_a, ax_b = mat.T, ax_b, ax_a
        bshape = [1, 1, 1, 1]
        bshape[ax_a], bshape[ax_b] = mat.shape
        return mat.reshape(bshape)

    shift = np.zeros(shape)
    if hh:
        shift = shift + _place(ladder(g, o, o), *h_axes)      # <ij||ij>
    if pp:
        shift = shift + _place(ladder(g, v, v), *p_axes)      # <ab||ab>
    if hp:
        cross = ladder(g, v, o)                               # <ai||ai>
        p1, p2 = p_axes
        h1, h2 = h_axes
        shift = shift - (_place(cross, p1, h1) + _place(cross, p2, h2)
                         + _place(cross, p1, h2) + _place(cross, p2, h1))
    return bare_sign * shift


def epstein_nesbet_denominator(D_bare, g, no, layout='pphh', bare_sign=1.0,
                               hh=True, pp=False, hp=False, B=None):
    """D_bare - epstein_nesbet_shift(...). See that function for the channel flags and
    -- important -- bare_sign, which must match D_bare's OWN sign convention (independent
    of `layout`). B: optional DF factor forwarded to epstein_nesbet_shift (g may then
    be None)."""
    return D_bare - epstein_nesbet_shift(g, no, layout=layout, bare_sign=bare_sign,
                                         hh=hh, pp=pp, hp=hp, B=B)


#: How the two spin-case diagonals are combined into the ONE shift a spin-adapted (CSF)
#: solver can use. Delta_same = <pq||pq> = J-K (zero on the diagonal, no same-spin p==q
#: configuration exists); Delta_opp = <pq|pq> = J (diagonal kept -- same spatial orbital,
#: opposite spins, is a real configuration).
#:
#:   'mean'     1/2 (Delta_same + Delta_opp) = J - K/2, normalised over the spin cases
#:              that EXIST, so p==q takes Delta_opp alone.
#:   'opposite' Delta_opp = J. The true determinant diagonal of the alpha-beta
#:              configuration, which is what the restricted amplitude t2_1_abab
#:              represents. Arguably the best-motivated single choice.
#:   'sum'      Delta_same + Delta_opp = 2J - K. NOT a diagonal element of any
#:              determinant (it double-counts) -- exactly twice 'mean' off-diagonal.
EN_SPIN_WEIGHTINGS = ('mean', 'opposite', 'sum')


def epstein_nesbet_shift_restricted_spinadapted(g_same=None, B_block=None,
                                                weighting='mean'):
    """SPIN-ADAPTED Epstein-Nesbet ladder shift for a restricted (RHF, closed-shell)
    reference: ONE matrix per channel, combining the same-spin and opposite-spin
    diagonals per `weighting` (see EN_SPIN_WEIGHTINGS). The default 'mean' gives
    Delta[p,q] = J-K/2 off-diagonal and Delta[p,p] = J_pp (only the opposite-spin case
    exists at p==q, by Pauli).

    Supply either the dense spatial physicist block g_same (<pq|rs>, NOT antisymmetrized)
    or the DF factor B_block[Q,p,q]; the two agree to machine precision.

    The weighting is a genuine MODELLING CHOICE, not a derivation -- demanding a single
    denominator leaves the relative weight of the two spin cases underdetermined, and the
    damping differs enough to matter on the 23-molecule sCI set at aug-cc-pVQZ. State it
    explicitly in any write-up.

    WHY a single matrix: determinant-wise EN gives the two spin cases different
    denominators, breaking the closed-shell singlet relation t_aaaa == t_abab - t_abab^T.
    The resulting U^(2) then acquires components on QUARTET 2h1p/2p1h configurations a
    doublet-CSF basis does not contain (measured ||U_perp||: 5e-16 undressed vs 4.7e-4
    with determinant-wise hh+pp on H2O/sto-3g). A single D[i,j,a,b], symmetric under
    i<->j and a<->b separately, is exactly the condition for the dressed amplitudes to
    stay spin-pure. This is therefore a DIFFERENT METHOD from Jiang & Engel's -- EN
    partitioning in the spin-adapted basis, not the determinant basis; see
    epstein_nesbet_shift_restricted_spinresolved for the latter."""
    if (g_same is None) == (B_block is None):
        raise ValueError("epstein_nesbet_shift_restricted_spinadapted: pass "
                         "exactly one of g_same (dense) or B_block (DF)")
    if weighting not in EN_SPIN_WEIGHTINGS:
        raise ValueError(f"weighting={weighting!r}; expected one of "
                         f"{EN_SPIN_WEIGHTINGS}")
    J, K = (get_coulomb_exchange_diagonals(g_same) if g_same is not None
            else get_coulomb_exchange_diagonals_df(B_block))
    if weighting == 'opposite':
        return J.copy()                       # Delta_opp; diagonal already right
    if weighting == 'sum':
        return 2.0 * J - K                    # (J-K) + J; gives J_pp at p==q
    delta = J - 0.5 * K                       # 'mean'
    np.fill_diagonal(delta, np.diag(J))       # p==q: opposite-spin case only
    return delta


def epstein_nesbet_shift_restricted_spinresolved(g_same=None, B_block=None):
    """Spin-RESOLVED (determinant-wise, Jiang & Engel) restricted diagonal ladder shifts
    -- returns (delta_same, delta_opp), each (n,n):

        delta_same[p,q] = <pq||pq> = J_pq - K_pq,   zero on the diagonal
        delta_opp[p,q]  = <pq|pq>  = J_pq,          diagonal KEPT

    The EN shift is a diagonal Hamiltonian matrix element of ONE determinant, so it is
    spin-case dependent and does not collapse to a single number: for a same-spin pair
    the exchange integral survives and p==q is not a physical configuration at all; for
    an opposite-spin pair there is no exchange, and p==q IS legitimate (same spatial
    orbital, opposite spins) carrying J_pp = (pp|pp) != 0. Zeroing that diagonal drops a
    real term.

    Correct for a spin-BLOCKED solver; a CSF/spin-adapted solver must use
    epstein_nesbet_shift_restricted_spinadapted instead (these shifts put weight on
    quartet configurations the doublet-CSF basis cannot hold).

    Supply either the dense spatial physicist block g_same or the DF factor B_block."""
    if (g_same is None) == (B_block is None):
        raise ValueError("epstein_nesbet_shift_restricted_spinresolved: pass "
                         "exactly one of g_same (dense) or B_block (DF)")
    J, K = (get_coulomb_exchange_diagonals(g_same) if g_same is not None
            else get_coulomb_exchange_diagonals_df(B_block))
    delta_same = J - K
    np.fill_diagonal(delta_same, 0.0)      # <pp||pp> = 0 by antisymmetry
    return delta_same, J.copy()            # opposite spin: no exchange, keep diag
