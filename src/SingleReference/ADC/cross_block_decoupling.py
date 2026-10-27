"""Tarantelli-Cederbaum decoupling of the union-space cross blocks.

The union-space ISR-ADC(n) supermatrix couples the (k+1)h kp-dagger classes
(D-type, the IP classes) to the (k+1)p kh classes (C-type, the EA classes)
through the Hamiltonian blocks X = (D|H^|C); their metric partners vanish.
A unitary W = exp(S), S antisymmetric with only D x C blocks,

    S^(m)_ab = R^(m)_ab / (K_b - K_a)      on the cross positions,

with R^(m) the order-m cross content of W^T D W built from the lower orders,
removes the cross blocks order by order (Tarantelli and Cederbaum, Phys. Rev.
A 45, 2790 (1992), Sec. V). The transformed couplings B W and class blocks
W^T D W, truncated by the manifold rule, are the diagrammatic ADC(n) elements:
the same-sector blocks of the two-sector construction plus static images of
the cross blocks. At n = 4 the only image inside the rule orders is

    delta u_C = sum_D U^(1)_D X^(2)_DC / (K_C - K_D),
    delta u_D = sum_C U^(1)_C X^(2)_DC / (K_D - K_C),

the (B W)^(3) - B^(3) columns. 1/(K_C - K_D) is a 3p3h energy, so these are
the closed-form order-4 defect rows of the diagrammatic ADC(4)
(delta_u3_dressing keeps the emitted diagram form as a validation identity).
"""
import numpy as np


def series_mul(A, B, nord):
    """Cauchy product of two matrix series (lists indexed by order)."""
    return [sum(A[i] @ B[k - i] for i in range(k + 1)) for k in range(nord)]


def series_transpose(A):
    return [a.T for a in A]


def series_exp(S, nord):
    """exp(S) of a matrix series with S^(0) = S^(1) = 0, truncated at order nord-1."""
    n = S[0].shape[0]
    W = [np.eye(n)] + [np.zeros((n, n)) for _ in range(nord - 1)]
    term = [np.eye(n)] + [np.zeros((n, n)) for _ in range(nord - 1)]
    j = 1
    while True:
        term = [t / j for t in series_mul(term, S, nord)]
        if all(not np.any(t) for t in term):
            return W
        W = [W[k] + term[k] for k in range(nord)]
        j += 1


def decouple_cross_blocks(B, D, cross, nord):
    """Block-diagonalize the cross content of a class-block series.

    B: list of (norb, n) orbital-class couplings by order 0..nord-1; D: list
    of (n, n) class blocks (D[0] diagonal, the K energies); cross: boolean
    (n, n) mask of the D-type x C-type positions. Returns (B W, W^T D W, W,
    residual cross content of W^T D W through order nord-1)."""
    K = np.diag(D[0])
    denom = K[None, :] - K[:, None]
    n = D[0].shape[0]
    S = [np.zeros((n, n)) for _ in range(nord)]
    for m in range(2, nord):
        W = series_exp(S, nord)
        Dp = series_mul(series_mul(series_transpose(W), D, nord), W, nord)
        R = np.where(cross, Dp[m], 0.0)
        S[m] = np.where(cross, R / np.where(cross, denom, 1.0), 0.0)
    W = series_exp(S, nord)
    Dp = series_mul(series_mul(series_transpose(W), D, nord), W, nord)
    Bp = series_mul(B, W, nord)
    resid = max(float(np.max(np.abs(np.where(cross, Dp[m], 0.0)))) for m in range(nord))
    return Bp, Dp, W, resid


def delta_u_rows_k1(u1_ip, u1_ea, k_ip, k_ea, x_block):
    """Order-3 image of the k1 cross block on the orbital couplings.

    u1_ip/u1_ea: (n_2h1p, norb)/(n_2p1h, norb) first-order couplings U^(1);
    k_ip/k_ea: the class energies K (IP: holes minus particle, EA: particles
    minus hole); x_block: (n_2h1p, n_2p1h) X^(2). Returns (du_2h1p, du_2p1h),
    the rows to add to the two k1 couplings: the (B W)^(3) - B^(3) columns of
    decouple_cross_blocks at n = 4, in closed form (no (n, n) products)."""
    w = x_block / (k_ea[None, :] - k_ip[:, None])       # X_DC / (K_C - K_D)
    du_ea = w.T @ u1_ip                                  # sum_D U1_D X_DC / (K_C - K_D)
    du_ip = -(w @ u1_ea)                                 # sum_C U1_C X_DC / (K_D - K_C)
    return du_ip, du_ea
