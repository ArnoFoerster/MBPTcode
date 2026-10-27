"""The dense U^(n) oracle of the generated restricted ADC: u_dot_orb_isr builds
the (n_k1, norb) orbital<->k1 coupling with the (n_k1, n_k1) class overlap and
its Loewdin series formed as matrices, exact through n = 4. Production applies
U^(n) matrix-free and order-general (sigma_driver._un_pieces/_un_apply_fwd/
_un_apply_bwd); the tests check that layer against this one."""
import numpy as np

from src.SingleReference.ADC.generated_modules import generated
from src.SingleReference.ADC.isr_engine import lowdin_sqrt_inv_series
from src.SingleReference.ADC.sigma_driver import u_raw_series


def u_dot_orb_isr(rsolver, sector, p_orb, n, s2cr_module, extra_leaves=None,
                  strict=True, laplace_ntau=None):
    """ORDER-GENERAL U^(n) with the ISR/Loewdin layer taken from isr_engine
    (whose own tests use synthetic blocks; this is the wiring to real
    generated blocks). p_orb=None returns the full (n_k1, norb) U^(n)
    MATRIX. It forms the (n_k1, n_k1) class overlap Delta S and its Loewdin
    series densely: the validation oracle of the matrix-free order-general
    layer (sigma_driver._un_apply_fwd/_bwd), never a production route.

        U^(n) = sum_{p+q=n} Y^(p).T @ raw^(q)        (Loewdin, isr_engine)
                - sum_{m>=2} N^(m) raw^(n-m)          (GS normalization)
                - eps_p S^(n)                         (class-GS subtraction)

    Y^(p) are lowdin_sqrt_inv_series coefficients of the CLASS-CLASS overlap
    Delta S (s2c/s3c blocks); S^(n) is the distinct ORBITAL-class overlap.
    isr_engine's u_raw convention is (norb, ncfg), the driver's is (ncfg, norb),
    hence Y^(p).T @ raw^(q) rather than raw @ Y.

    At n=3 (Delta S^(1)=0 => Y^(1)=0, Y^(2)=-1/2 Delta S^(2)) this reduces to
    u3_generated's hand-derived form; that identity is the gate
    (tests/test_sigma_isr_order_general.py).

    laplace_ntau (int): route the raw^(m)/S^(m) rank>=3 structures through their
    Laplace-fused twins (via u_raw_series -> _assemble_raw_S), so l3/l4/t3/t4 are
    never materialized in those blocks (the auto-fusion DRIVER). The
    ISR/Loewdin layer (Delta S order<=n-1, N^(m)) is rank<=2 and unchanged. NB
    the low-rank order>=3 amplitudes the rank<=2 structures still consume
    (T1^(3)/L1^(3)/L2^(3)/L1^(4)) come from amps_order and still materialize
    their t3/t4 RSPT intermediates -- the order-4 rank<=2 Laplace amplitude
    chain is the remaining production-floor piece (see _assemble_raw_S).

    Exact through n = 4 only, and refused above: from order 5 the norm division
    stops being additive ((N^(2))^2 raw^(1), -N^(2) Y^(2) raw^(1)), the class
    metric gains the orbital projection -S_10 S_01, and the orbital block's
    static part enters as -S V_00. _un_apply_fwd carries all three."""
    if n >= 5:
        raise NotImplementedError(
            f"u_dot_orb_isr(n={n}): the additive normalization, the class metric "
            f"without its orbital projection and the dropped V_00 channel are exact "
            f"through n = 4 only; use u_dot_orb_order / _un_apply_fwd for U^({n}).")
    tag = '2h1p' if sector == 'IP' else '2p1h'
    eps, no, nv = rsolver.eps, rsolver.nocc, rsolver.nvir
    cfg = rsolver.sector_configs(sector)
    ncfg = len(cfg['K'])
    pool = {**rsolver.rargs, **rsolver.amplitude_pool(), **(extra_leaves or {})}

    raws, Ss = u_raw_series(rsolver, sector, n, strict=strict,
                            laplace_ntau=laplace_ntau, extra_leaves=extra_leaves)

    # class-class overlap series Delta S^(m) (ncfg, ncfg). Order 2 = the s2c
    # block minus the N^(2) identity piece; order 1 vanishes.
    #
    # NOTE this FORMS the (n_k1, n_k1) overlap -- the very object the
    # matrix-free design avoids (it is the ~17.5 GB aug-cc-pVTZ blocker that
    # u3_dot_orb sidesteps by applying s2c to a single vector). Acceptable here
    # because this routine exists to VALIDATE the order-general ISR on small
    # systems; the production form must push the Loewdin series through as
    # vector applications, U^(n)@p = sum_p Y^(p).T @ (raw^(q)@p), where
    # Y^(2).T@v = -1/2 (s2c matvec on v) exactly as u3_dot_orb already does.
    # The Loewdin convolution sum_{p+q=n} Y^(p).T raw^(q) reaches p = n-1 (since
    # raw^(1) exists), and Y^(p) depends on Delta S up to order p. So orders
    # 2..n-1 of Delta S must ALL be present or the series is silently truncated.
    #
    # ORDER-GENERAL: rsolver.class_overlap_series(sector, n-1)
    # supplies every order from the same adc_order_rule enumeration the raw
    # blocks use (adc5_c3_blocks_restricted for order 3, and the order-2 s2c
    # structure REUSED at higher amplitude assignments -- which is where
    # Delta S^(3) actually comes from, since pdaggerq pruned every s3c split).
    # Verified to reproduce the previous inline order-2 expression
    # (s2c - N^(2) I) BIT-EXACTLY, so n=3 is unchanged.
    if hasattr(rsolver, 'class_overlap_series') and n > 2:
        delta_s = rsolver.class_overlap_series(sector, n - 1, strict=strict,
                                               laplace_ntau=laplace_ntau)
    else:
        genc2r = generated('adc4_c2_blocks_restricted')
        n2 = rsolver.n2_norm()
        s2c = rsolver._class_block_sum(genc2r, f's2c_{tag}_', sector)
        delta_s = {}
        if s2c is not None:
            delta_s[2] = s2c - n2 * np.eye(ncfg)

    needed = set(range(2, n))
    have = set(delta_s)
    if strict and not needed <= have:
        raise RuntimeError(
            f"u_dot_orb_isr(n={n}, {sector}): Loewdin series needs Delta S orders "
            f"{sorted(needed)} but only {sorted(have)} are assembled -- the ISR "
            f"would be silently TRUNCATED. Missing {sorted(needed - have)}: build "
            f"the order-general class-class overlap assembly (s3c blocks) first.")

    y = lowdin_sqrt_inv_series(delta_s, n) if delta_s else {0: np.eye(ncfg)}

    # Loewdin convolution
    U = None
    for p in range(0, n + 1):
        q = n - p
        y_p, r_q = y.get(p), raws.get(q)
        if y_p is None or r_q is None:
            continue
        term = y_p.T @ r_q
        U = term if U is None else U + term

    # GS normalization: - sum_{m>=2} N^(m) raw^(n-m)
    for m in range(2, n):
        nm = rsolver.n_norm_order(m) if hasattr(rsolver, 'n_norm_order') else (n2 if m == 2 else None)
        if nm is None:
            raise RuntimeError(f"u_dot_orb_isr(n={n}): N^({m}) not available "
                               f"(needed for the GS-normalization term).")
        if raws.get(n - m) is not None:
            U = U - nm * raws[n - m]

    # class-GS subtraction: -eps_p * S_eff, where S_eff is the orbital-class
    # overlap in the ORTHONORMALIZED basis -- i.e. the raw Ss carried through
    # the SAME normalization (-sum N^j S^(m-j)) and Loewdin convolution
    # (sum_p Y^p.T S^q) as the raw coupling above. Using the raw Ss[n] instead
    # (the previous code) drops eps*(N^2 S^(2) + 1/2 dS^(2) @ S^(2)) at n>=4 --
    # U^(4)'s first-validated defect (against the exact order-5 ISR:
    # closes 5.4e-6/9.9e-6 -> 1e-10). Both dropped terms vanish at n<=3 because
    # S^(1)=0, so n=2 (== c2) and n=3 (== u3_generated) are byte-inert.
    S_corr = {}
    for m in range(n + 1):
        sm = Ss.get(m)
        if sm is None:
            continue
        sm = sm.copy()
        for j in range(2, m + 1):
            if Ss.get(m - j) is not None:
                sm = sm - rsolver.n_norm_order(j) * Ss[m - j]
        S_corr[m] = sm
    S_eff = None
    for p in range(n + 1):
        yp, sq = y.get(p), S_corr.get(n - p)
        if yp is None or sq is None:
            continue
        term = yp.T @ sq
        S_eff = term if S_eff is None else S_eff + term
    if S_eff is not None:
        U = U - eps[None, :] * S_eff
    return U if p_orb is None else U @ p_orb
