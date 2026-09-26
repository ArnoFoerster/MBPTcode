"""THC-GW self-energy with k-point sampling.

The k-point generalization of `GW/imaginary_time.py`, consuming the
factorization from `pbc_isdf.build_isdf_kpts` and the polarizability from
`pbc_isdf_rpa`. Structure follows Yeh & Morales, JCTC 2024, 20, 3184, eqs
20-27:

    Sigma^k(i.tau) = -(1/Nq) sum_q  G^{k-q}(i.tau) * Wt^q(i.tau)

elementwise in the interpolation index, exactly as the polarizability was a
product there. Two things carry over from the polarizability and one is new:

  * the q-sum is a CONVOLUTION over the mesh, sum_q G_{k-q} * W_q, so it is
    the plain product of FFTs -- simpler than the polarizability's, which was
    a correlation and needed an index negation. Same O(Nk ln Nk).
  * G^< and G^> are the occupied and virtual branches, and Sigma is built from
    both: cosine on their SUM, sine on their DIFFERENCE.
  * The RANGE TRAP, and it is the live one here. The tau grid must be resolved
    over the SELF-ENERGY's range, not the polarizability's. Sigma = -G Wt is a
    PRODUCT, so its decay rates are SUMS |eps_m - mu| + Omega_S and reach far
    beyond either factor's range. The molecular `DEFAULT_NTAU = 'auto'` is a
    SENTINEL, not an int, and is NOT forwarded: this module owns both halves
    of the question -- `self_energy_fit_ranges_k` for the ranges and
    `minimax_points_for_gw_k` for the point count -- because the periodic case
    needs MORE points than the molecular tolerance suggests (rS is 1.6x wider,
    measured on diamond), so a molecular ntau does not carry across.
    `qp_energy_thc` defaults to resolving it and CHECKS the accuracy it got.

TWO ROUTES, and the difference is the GRID, not the algebra:

  * `sigma_c_diag_thc` -- T = 0, minimax half-line tau. Correct for a gap and
    guarded against anything else.
  * `sigma_c_matsubara_thc` -- finite temperature on a FERMIONIC IR grid, for
    metals. The Green's functions are occupation-weighted in both.

CALL `qp_energy_thc` (bottom of this file) rather than either directly. Both
take a tau grid and a polarizability that the CALLER must have built on the
right ranges and the right frequencies, and those conventions are not
guessable -- fitting the tau grid on the screening range instead of the
self-energy range is a silent accuracy loss, not an error. The driver owns
them, and dispatches between the two routes on `beta`.
"""
import numpy as np

from src.Base.utils.time_frequency import (DEFAULT_TAU_TARGET,
                                           minimax_points_for_ranges,
                                           self_energy_fit_ranges_from_window)
from src.SingleReference.Periodic.pbc_isdf_rpa import (chemical_potential_occ,
                                                       require_integer_occupation)
from src.SingleReference.Periodic.pbc_occupations import fermi_level_from_mf


def self_energy_fit_ranges_k(mo_energy, mo_occ, mu=None):
    """((w_lo, w_hi), (sig_lo, sig_hi)) over the whole k-mesh.

    Periodic form of `GW.imaginary_time.self_energy_fit_ranges`, and the
    reason the range trap needs its own code rather than a forwarded
    sentinel: the self-energy range is the SUM of the Green's function and
    screening ranges, so it is wider than either, and it must be taken over all
    k-points rather than one.
    """
    e = np.asarray(mo_energy)
    f = np.asarray(mo_occ, dtype=float)
    mu = chemical_potential_occ(e, f) if mu is None else mu
    occ, virt = f > 1.0, f <= 1.0
    return self_energy_fit_ranges_from_window(
        e, mu,
        float(e[virt].min() - e[occ].max()),
        float(e[virt].max() - e[occ].min()))


def screened_interaction_tilde_q(Vq, Pi_q):
    """Wt^q(i.omega) = [1 - V^q Pi^q]^-1 V^q - V^q for ONE transfer.

    The bare V is a delta function at tau = 0 and belongs to the static
    exchange, which the QP solve adds separately as Sigma_x - v_xc. Leaving it
    in would double count it and is not visible in Sigma_c alone.

    PER TRANSFER, deliberately. An all-q form would be a third resident
    (nq, nfreq, M, M) tensor alongside the caller's Pi and Wt(tau), each of
    which outgrows memory on a triple- or quadruple-zeta slab. Both
    self-energy routes build it one q at a time and drop the frequency
    representation as soon as it has been transformed.
    """
    eye = np.eye(Vq.shape[0])
    return np.array([np.linalg.solve(eye - Vq @ P, Vq) - Vq for P in Pi_q])


def _pi_at(Pi, q):
    """Pi^q from either a materialized sequence or a per-q callable.

    Accepting a callable is what lets a caller avoid materializing Pi for
    every transfer at once; `qp_energy_thc` passes a closure over
    `polarizability_q_frequency_occ`.
    """
    return Pi(q) if callable(Pi) else Pi[q]


def green_functions_tau_gw(X, mo_energy, mo_occ, tau, mu):
    """(G_lesser, G_greater) at the interpolation points, each (nk, M, M).

        G^<_{nu mu}(tau) = sum_m      f_m  X^k_{m nu} X^{k*}_{m mu} e^{+(eps_m-mu) tau} / 2
        G^>_{nu mu}(tau) = -sum_m (2 - f_m) X^k_{m nu} X^{k*}_{m mu} e^{-(eps_m-mu) tau} / 2

    Written occupation-weighted so a metal is a change of grid rather than a
    change of algebra; for integer occupations the f/2 and (2-f)/2 factors are
    exactly the occupied and virtual indicator functions, reproducing
    `GW.imaginary_time.greens_function_imaginary_time`.
    """
    e = np.asarray(mo_energy)
    f = np.asarray(mo_occ, dtype=float)
    G_l, G_g = [], []
    with np.errstate(divide='ignore', invalid='ignore'):
        logf, logh = np.log(0.5 * f), np.log(0.5 * (2.0 - f))
    for k in range(len(X)):
        Xk, x = X[k], e[k] - mu
        wl = np.where(f[k] > 0, np.exp(logf[k] + x * tau), 0.0)
        wg = np.where(f[k] < 2.0, np.exp(logh[k] - x * tau), 0.0)
        G_l.append((Xk * wl) @ Xk.conj().T)
        G_g.append(-(Xk * wg) @ Xk.conj().T)
    return np.asarray(G_l), np.asarray(G_g)


def self_energy_tau_convolution(G, Wt, kmesh):
    """sum_q G^{k-q} * Wt^q for every k, elementwise in (mu, nu).

    A CONVOLUTION over the mesh, unlike the polarizability's correlation, so
    it is the plain product of forward transforms with no index negation:
    c = ifftn(fftn(G) * fftn(Wt)).
    """
    kmesh = tuple(int(n) for n in kmesh)
    nk = int(np.prod(kmesh))
    if len(G) != nk or len(Wt) != nk:
        raise ValueError(f"G has {len(G)}, Wt has {len(Wt)} k-points for kmesh "
                         f"{kmesh} (expected {nk}) -- the FFT route needs the "
                         f"full regular mesh in make_kpts order.")
    shape = kmesh + G.shape[1:]
    axes = (0, 1, 2)
    prod = np.fft.fftn(G.reshape(shape), axes=axes) * \
           np.fft.fftn(Wt.reshape(shape), axes=axes)
    return np.fft.ifftn(prod, axes=axes).reshape(nk, *G.shape[1:])


def self_energy_tau_direct(G, Wt, kminus, k):
    """sum_q G^{k-q} * Wt^q at one k, by the direct q-sum. Reference only."""
    return sum(G[kminus[q, k]] * Wt[q] for q in range(len(Wt)))


def sigma_c_diag_thc(X, V, Pi, mo_energy, mo_occ, kmesh, kn, states,
                     omega_in, omega_out, tau_points, mu=None, ranges=None,
                     kminus=None, use_fft=True):
    """Sigma_c^{n,kn}(i.omega) for a window of states, shape (nstates, nfreq).

    The periodic form of the molecular tau route
    (`GW.imaginary_time.self_energy_matrix_imaginary_time`), diagonal here:

        Wt^q(i.omega) --cos--> Wt^q(i.tau)                      fitted on rW
        Sigma^{<,>}_k(tau)   = (1/Nq) sum_q G^{k-q}(tau) * Wt^q(tau)
        Sigma(i.omega)       = -1/2 [ cos(S^> + S^<) + i sin(S^> - S^<) ]   on rS

    THE TWO RANGES ARE DIFFERENT AND BOTH ARE USED. rW is the
    screening range, which starts BELOW the smallest independent-particle
    transition because collective excitations sit under the gap. rS is the
    self-energy range, and it is much wider because Sigma = -G Wt is a PRODUCT
    whose decay rates are SUMS. Fitting either transform on the other's range
    is a silent accuracy loss, not an error.

    A window of states costs essentially what one state costs: the expensive
    object per tau is the (M, M) convolution, which does not depend on the
    state at all.

    T = 0 ONLY -- a fractional or k-ragged occupation is refused. Use
    `sigma_c_matsubara_thc` for a metal; the half-line grid there is wrong for
    the same three reasons it is wrong for the polarizability.
    """
    from src.Base.utils.time_frequency import (COSINE_TW, COSINE_WT, SINE_TW,
                                               minimax_transform_weights)

    require_integer_occupation(mo_occ, who='sigma_c_diag_thc')
    mu = chemical_potential_occ(mo_energy, mo_occ) if mu is None else mu
    rW, rS = ranges or self_energy_fit_ranges_k(mo_energy, mo_occ, mu)

    # Wt is built and transformed ONE TRANSFER AT A TIME. Materializing the
    # screened interaction twice over -- Wt(omega) for every q AND Wt(tau) for
    # every q, both alive alongside the caller's Pi -- would be three all-q
    # tensors where the algebra needs one. Per q, the omega
    # representation is consumed by the transform and dropped immediately, so
    # only Wt(tau) survives the loop.
    #
    # ONE all-q tensor is IRREDUCIBLE here, unlike in the RPA: the self-energy
    # is a CONVOLUTION over transfers, sum_q G^{k-q} Wt^q, so every tau needs
    # every q at once. A q-outer loop of the kind rpa_ecorr_thc_streaming uses
    # does not exist for this object; the choice is only WHICH representation
    # to keep, and tau is the smaller one when ntau < nfreq.
    Ctw, _ = minimax_transform_weights(COSINE_WT, tau_points, omega_in, *rW,
                                       warn=False)
    Wt_tau = np.empty((len(V), len(tau_points), V[0].shape[0], V[0].shape[0]),
                      dtype=np.complex128)
    for q in range(len(V)):
        Wq = screened_interaction_tilde_q(V[q], _pi_at(Pi, q))   # (nfreq, M, M)
        Wt_tau[q] = np.tensordot(Ctw, Wq, axes=(1, 0))
        del Wq

    states = np.atleast_1d(states)
    scalar = np.ndim(states) == 0
    ntau, nq = len(tau_points), len(V)
    sig_l = np.zeros((ntau, len(states)), dtype=np.complex128)
    sig_g = np.zeros((ntau, len(states)), dtype=np.complex128)
    Xn = np.ascontiguousarray(X[kn][:, states])                 # (M, nstates)

    for t, tau in enumerate(tau_points):
        G_l, G_g = green_functions_tau_gw(X, mo_energy, mo_occ, tau, mu)
        Wt_t = np.ascontiguousarray(Wt_tau[:, t])
        if use_fft:
            Sl = self_energy_tau_convolution(G_l, Wt_t, kmesh)[kn]
            Sg = self_energy_tau_convolution(G_g, Wt_t, kmesh)[kn]
        else:
            Sl = self_energy_tau_direct(G_l, Wt_t, kminus, kn)
            Sg = self_energy_tau_direct(G_g, Wt_t, kminus, kn)
        sig_l[t] = np.einsum('mp,mp->p', Xn.conj(), (Sl / nq) @ Xn, optimize=True)
        sig_g[t] = np.einsum('mp,mp->p', Xn.conj(), (Sg / nq) @ Xn, optimize=True)

    C, _ = minimax_transform_weights(COSINE_TW, tau_points, omega_out, *rS,
                                     warn=False)
    S, _ = minimax_transform_weights(SINE_TW, tau_points, omega_out, *rS,
                                     warn=False)
    out = -0.5 * ((sig_g + sig_l).T @ C.T + 1j * ((sig_g - sig_l).T @ S.T))
    return out[0] if scalar else out


# A frequency-route reference for Sigma_c (the counterpart of
# `pbc_isdf_rpa.polarizability_q_frequency`) was declared here and never
# implemented. It has been removed rather than left as a NotImplementedError
# that reads like a route: the imaginary-time Sigma_c already has an
# INDEPENDENT oracle in `pbc_self_energy.qp_energy_g0w0`, which reaches the
# same quasiparticle energies by a real-frequency spectral sum over GDF
# integrals (agreement <= 0.8 meV, test_pbc_isdf_gw.py rung 3). A second
# reference sharing this module's conventions would be weaker than that one.


# ---------------------------------------------------------------------------
# Metallic self-energy: fermionic IR (Matsubara) grid
# ---------------------------------------------------------------------------
#
# The T = 0 route above is exact for a gap and guarded for a metal. What a
# metal additionally needs is NOT a change of algebra -- the Green's functions
# are already occupation-weighted -- but a change of grid, and specifically a
# FERMIONIC one, since Sigma and G are fermionic while Pi and Wt are bosonic.
#
# THE OBSTACLE, and the reason this is not a two-line substitution: the two IR
# bases do NOT share tau sampling points, yet Sigma(tau) = -G(tau) Wt(tau) is a
# POINTWISE product. Resolution: take the FERMIONIC tau sampling as the common
# grid and evaluate the BOSONIC basis there via IRBasis.u_at. G is analytic in
# tau so it is simply evaluated at those points; only Wt needs transporting.
#
# ONE DIFFERENCE FROM THE MOLECULAR TEMPLATE, and it matters. There Wt(tau) is
# REAL, so the bosonic fit uses real IR coefficients (IRBasis.fit_matsubara's
# default, which stacks real and imaginary parts to dodge a rank deficiency).
# Periodically Wt^q is complex Hermitian at every q != 0, so its tau-dependence
# is genuinely complex and the fit needs COMPLEX coefficients (real=False) over
# a two-sided sampling set. Using the real path here would silently discard the
# imaginary part.
#
# No cosine/sine parity split appears here: unlike the minimax route, the
# fermionic IR basis represents a complex Sigma(i.omega_n) directly through its
# own uhat.


def self_energy_range_k(mo_energy, mo_occ, screening_max, mu=None):
    """Periodic adapter for `matsubara.self_energy_range`.

    The identity -- max|eps - mu| + screening_max, because Sigma = -G Wt is a
    PRODUCT whose poles sit at eps_m +/- omega_s -- lives there and only there.
    All this adds is deriving mu from the OCCUPATIONS rather than an integer
    split, which is what makes it usable for a metal.

    Used UNPADDED for the IR Lambda = beta * omega_max. That is deliberately
    different from `self_energy_fit_ranges_k`, whose 0.3x/3.0x are padding for
    a Remez least-squares fit: sizing a continuation and sizing a fit range are
    different questions that happen to share this core.
    """
    from src.Base.utils.matsubara import self_energy_range

    mu = chemical_potential_occ(mo_energy, mo_occ) if mu is None else mu
    return self_energy_range(mo_energy, mu, screening_max)


def green_function_tau_matsubara(X, mo_energy, mo_occ, tau, beta, mu):
    """G^k(tau) for 0 <= tau <= beta, shape (nk, M, M). ONE object, no </> split.

        G_{nu mu}(tau) = -sum_m X_{m nu} X^*_{m mu}
                          [ (1 - f_m/2) e^{-x_m tau} + (f_m/2) e^{-x_m (tau - beta)} ]

    with x = eps - mu. The second term is the antiperiodic wrap G(tau - beta) =
    -G(tau) that replaces the T = 0 occupied branch; at integer occupations the
    two terms collapse onto the virtual and occupied sums respectively, which
    is the molecular Matsubara form.

    Both terms are formed in LOGS, and both need it: for a deep occupied state
    (1 - f/2) ~ e^{-beta|x|} underflows while e^{-x tau} = e^{|x| tau}
    overflows, and for a high virtual the same happens in the other term. Each
    product is bounded on [0, beta]; neither factor is.
    """
    e = np.asarray(mo_energy)
    f = np.asarray(mo_occ, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_h, log_p = np.log(1.0 - 0.5 * f), np.log(0.5 * f)
    G = []
    for k in range(len(X)):
        Xk, x = X[k], e[k] - mu
        w = np.zeros_like(x)
        hole, part = f[k] < 2.0, f[k] > 0.0
        w[hole] = np.exp(log_h[k][hole] - x[hole] * tau)
        w[part] += np.exp(log_p[k][part] - x[part] * (tau - beta))
        G.append(-(Xk * w) @ Xk.conj().T)
    return np.asarray(G)


def sigma_c_matsubara_thc(X, V, Pi_bosonic, mo_energy, mo_occ, kmesh, beta,
                          omega_max, kn, states, mu=None, eps_ir=1e-10,
                          use_fft=True, kminus=None, return_diagnostics=False):
    """Sigma_c^{n,kn}(i.omega_n) on a FERMIONIC Matsubara grid. Metals included.

    `Pi_bosonic[q]` must already be evaluated at the BOSONIC sampling
    frequencies this routine derives -- use `bosonic_sampling(beta, omega_max)`
    to get them, so the two cannot drift apart.

    Returns (sigma, omega_fermionic) with sigma of shape (nstates, nfreq) and
    omega SIGNED.
    """
    from src.Base.utils.matsubara import IRBasis, matsubara_frequencies

    mu = chemical_potential_occ(mo_energy, mo_occ) if mu is None else mu
    bF = IRBasis(beta * omega_max, eps=eps_ir, statistics='fermion')
    bB = IRBasis(beta * omega_max, eps=eps_ir, statistics='boson')
    xF = bF.default_tau_sampling()
    tau = 0.5 * beta * (xF + 1.0)                       # the COMMON tau grid
    nB = bB.default_matsubara_sampling(positive_only=False)
    nF = bF.default_matsubara_sampling(positive_only=False)
    # SIGNED, deliberately: Sigma(-i w) = Sigma(i w)^* so the two branches are
    # different data, and the QP solve samples the branch its state sits on
    # (occupied below the gap, virtual above). Returning |w| would collapse
    # them into duplicated points and quietly fit Pade to the wrong half.
    wF = matsubara_frequencies(nF, beta, 'fermion')

    # Wt: bosonic Matsubara -> IR coefficients -> the FERMIONIC tau points.
    # real=False because Wt^q is complex Hermitian for q != 0 (see module note).
    U_bos = bB.u_at(xF).T                                # (ntau, L_B)
    nq, M = len(V), V[0].shape[0]
    Wt_tau = np.empty((nq, len(tau), M, M), dtype=np.complex128)
    resid = 0.0
    for q in range(nq):
        # ONE transfer at a time, and Pi_bosonic may be a callable so the
        # caller need not hold it for every q either. Building Wt(i.omega) for
        # all q first would put a third (nq, nfreq, M, M) tensor alongside Pi
        # and Wt(tau).
        flat = screened_interaction_tilde_q(
            V[q], _pi_at(Pi_bosonic, q)).reshape(len(nB), -1)
        # (2/beta) is the INVERSE dtau/dx Jacobian, and fit_matsubara does not
        # carry it: uhat_l integrates over the dimensionless x in [-1, 1], not
        # over tau in [0, beta]. Omitting it makes Wt(tau) -- and so Sigma --
        # too large by exactly beta/2, which is a clean constant and therefore
        # looks like a convention rather than a bug: measured ratio 199.97
        # against the T = 0 route at beta = 400. It is the same Jacobian
        # TimeFrequencyGrid.ir applies as `cosft_tw / jac`, and the mirror of
        # the (0.5 * beta) applied on the way back out below.
        cB = (2.0 / beta) * bB.fit_matsubara(nB, flat, real=False)
        back = (0.5 * beta) * bB.evaluate_matsubara(cB.T, nB).T
        resid = max(resid, float(np.abs(back - flat).max()
                                 / max(np.abs(flat).max(), 1e-300)))
        Wt_tau[q] = (U_bos @ cB).reshape(len(tau), M, M)

    states = np.atleast_1d(states)
    Xn = np.ascontiguousarray(X[kn][:, states])
    sig_tau = np.empty((len(tau), len(states)), dtype=np.complex128)
    for t, tt in enumerate(tau):
        G = green_function_tau_matsubara(X, mo_energy, mo_occ, tt, beta, mu)
        Wt_t = np.ascontiguousarray(Wt_tau[:, t])
        S = (self_energy_tau_convolution(G, Wt_t, kmesh)[kn] if use_fft else
             self_energy_tau_direct(G, Wt_t, kminus, kn))
        sig_tau[t] = np.einsum('mp,mp->p', Xn.conj(), (S / nq) @ Xn, optimize=True)
    sig_tau = -sig_tau                                   # Sigma(tau) = -G Wt

    # tau -> fermionic Matsubara. No parity split: uhat carries complex Sigma.
    cS = sig_tau.T @ np.linalg.pinv(bF.u_at(xF))
    sigma = (0.5 * beta) * (cS @ bF.uhat(nF))
    if return_diagnostics:
        return sigma, wF, {'ntau': len(tau), 'L_bos': bB.size, 'L_ferm': bF.size,
                           'wt_fit_residual': resid,
                           'cond_bos': bB.condition_number(nB, real=False)}
    return sigma, wF


def bosonic_sampling(beta, omega_max, eps_ir=1e-10):
    """The bosonic Matsubara frequencies `sigma_c_matsubara_thc` will expect.

    Exposed so Pi is built on exactly these points; deriving them twice from
    the same beta/omega_max is how the two silently drift apart.
    """
    from src.Base.utils.matsubara import IRBasis, matsubara_frequencies
    bB = IRBasis(beta * omega_max, eps=eps_ir, statistics='boson')
    nB = bB.default_matsubara_sampling(positive_only=False)
    return np.abs(matsubara_frequencies(nB, beta, 'boson'))


def minimax_points_for_gw_k(mo_energy, mo_occ, kplus, mu=None,
                            target=DEFAULT_TAU_TARGET, npoints_max=34,
                            threshold=None):
    """Smallest minimax tau grid that resolves every range this route integrates.

    The periodic counterpart of `GW.imaginary_time.minimax_points_for_gw`, and
    the same argument: ONE ntau serves three fits over three ranges and the
    WIDEST binds --

        Pi(i.tau)    -> Pi(i.omega)     [e_min, e_max]   the bare transition window
        Wt(i.omega)  -> Wt(i.tau)       rW               widened below the gap
        Sigma(i.tau) -> Sigma(i.omega)  rS               widest: Sigma = -G Wt is a
                                                         PRODUCT, so its decay
                                                         rates are SUMS

    -- so a hardcoded ntau is wrong at one end of any size series. It was
    hardcoded here (`ntau or 18`), and the periodic case is the one that can
    least afford it: it needs MORE points than the molecular tolerance
    suggests, because rS is wider (measured 1.6x on diamond). Do not carry a
    molecular ntau across.

    The transition window is taken over the OCCUPATION-WEIGHTED pair set, so
    this is defined for a metal in the same terms as everything else here.

    Returns (npoints, worst_error). `target` is not guaranteed: if nothing
    tabulated reaches it the best available is returned with the accuracy
    actually obtained, and the caller is expected to look.
    """
    from src.SingleReference.Periodic.pbc_isdf_rpa import transition_window_occ

    # No mean field in scope here, so no sigma to invert: this takes mu as
    # given, and `qp_energy_thc` passes the root-found one down.
    mu = chemical_potential_occ(mo_energy, mo_occ) if mu is None else mu
    rW, rS = self_energy_fit_ranges_k(mo_energy, mo_occ, mu)
    kw = {} if threshold is None else {'threshold': threshold}
    e_min, e_max = transition_window_occ(mo_energy, mo_occ, kplus, **kw)
    ratios = (e_max / e_min, rW[1] / rW[0], rS[1] / rS[0])

    return minimax_points_for_ranges(ratios, target=target,
                                     npoints_max=npoints_max)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def qp_energy_thc(cell, mf, npoints=None, states=None, kn=0, ntau='auto',
                  nw_screening=64, nw_sigma=16, mesh=None, coulG_fn=None,
                  beta=None, mu=None, exchange_minus_vxc=None, exxdiv=None,
                  threshold=None, solver_mode='pole_strength', ke_cutoff=None,
                  alpha=None, tau_target=DEFAULT_TAU_TARGET,
                  return_diagnostics=False):
    """G0W0 quasiparticle energies through THC, from a converged KRHF/KRKS.

    The one entry point for the whole chain -- factorization, polarizability,
    screened interaction, self-energy, continuation. It exists because the
    conventions between those steps are NOT obvious and would otherwise be
    written out by hand at every call site:

      * the tau grid is `0.5 * minimax_time_grid(ntau, *rS)`, and it is rS --
        the SELF-ENERGY range -- not rW. Fitting on rW is a silent
        accuracy loss rather than an error;
      * the screening transform is fitted on rW while the self-energy
        transform is fitted on rS, and `sigma_c_diag_thc` needs both;
      * Pi must be evaluated at exactly the frequencies the Sigma build will
        assume -- on the T = 0 route a Gauss-Legendre set seeded from rW, on
        the metallic route the BOSONIC sampling of `bosonic_sampling`;
      * the metallic continuation samples only the Matsubara branch its state
        sits on (occupied below mu, virtual above); feeding it both halves
        fits Pade across a discontinuity.

    TWO ROUTES, dispatched on `beta`, because they are different GRIDS and not
    different algebra:

      beta=None -- T = 0 minimax. Refuses fractional or k-ragged occupations.
      beta set  -- fermionic IR Matsubara. Metals. Get beta from
                   `matsubara.beta_from_mf` for a Fermi-smeared mean field.

    `states` defaults to (HOMO, LUMO) at k-point `kn`. Returns quasiparticle
    energies in HARTREE, shape (nstates,), or (energies, diagnostics) when
    `return_diagnostics`.

    NOTE the QP equation is solved for its root (solver_mode='pole_strength'
    by default, 'graphical' on request), never linearised, and must stay that
    way for a metal: a linearised solver returned Z = 0.45 and Z = -0.18 on
    bulk Al.
    """
    from src.Base.utils.grids import gauss_legendre_grid, minimax_time_grid
    from src.SingleReference.GW.qp_solve import (imaginary_axis_sample_points,
                                                 solve_qp_from_imaginary_axis)
    from src.SingleReference.Periodic.pbc_integrals import get_momentum_transfer_map
    from src.SingleReference.Periodic.pbc_isdf import build_isdf_kpts
    from src.SingleReference.Periodic.pbc_isdf_rpa import polarizability_q_frequency_occ
    from src.SingleReference.Periodic.pbc_rpa_damping import kmesh_from_kpts

    kpts = np.asarray(mf.kpts)
    mo = [np.asarray(c) for c in mf.mo_coeff]
    mo_energy = np.asarray(mf.mo_energy)
    mo_occ = np.asarray(mf.mo_occ)
    nocc = int((mo_occ[0] > 1e-8).sum())
    kmesh = kmesh_from_kpts(cell, kpts)
    kplus = get_momentum_transfer_map(cell, kpts)
    kw = {} if threshold is None else {'threshold': threshold}
    if states is None:
        states = [nocc - 1, nocc]
    states = list(np.atleast_1d(states))

    X, V, info = build_isdf_kpts(cell, mo, kpts, npoints, mesh=mesh,
                                 coulG_fn=coulG_fn, ke_cutoff=ke_cutoff,
                                 alpha=alpha)
    if mu is None:
        # A smeared mean field carries what is needed to SOLVE for mu, and
        # chemical_potential_occ's heuristic drifts ~100 meV once states are
        # genuinely partially occupied -- which propagates, since mu sets the
        # self-energy range and the range selects ntau. See
        # pbc_occupations.fermi_level for the measurements, including why
        # mf.get_fermi() is not this quantity.
        mu = fermi_level_from_mf(mf)
        if mu is None:                       # unsmeared: no sigma to invert
            mu = chemical_potential_occ(mo_energy, mo_occ)
    rW, rS = self_energy_fit_ranges_k(mo_energy, mo_occ, mu)

    if exchange_minus_vxc is None:
        # Sigma_x - v_xc is NOT assumed to vanish: on a KRHF start with
        # exxdiv=None it is 2.6e-3 Ha on diamond, which is 70 meV of QP energy.
        from src.SingleReference.Periodic.pbc_self_energy import get_exchange_minus_vxc
        xmv_all = get_exchange_minus_vxc(mf, exxdiv=exxdiv)
        xmv = [float(xmv_all[kn, n].real) for n in states]
    else:
        xmv = [float(x) for x in np.atleast_1d(exchange_minus_vxc)]
        if len(xmv) != len(states):
            raise ValueError(f"exchange_minus_vxc has {len(xmv)} entries for "
                             f"{len(states)} states.")

    diag = {'npoints': len(info['points']), 'alpha': info['alpha'],
            'mu': mu, 'rW': rW, 'rS': rS,
            'kmesh': kmesh, 'route': 'matsubara' if beta is not None else 'minimax'}

    if beta is None:
        # 'auto' resolves ntau from the three ranges this route integrates,
        # and the returned accuracy is CHECKED rather than recorded: the
        # molecular resolver's worst_error is discarded at every production
        # call site, and a grid that silently failed to reach its target is
        # exactly the kind of thing that reads as a converged answer.
        tau_err = None
        if isinstance(ntau, str) and ntau.lower() == 'auto':
            ntau, tau_err = minimax_points_for_gw_k(mo_energy, mo_occ, kplus,
                                                    mu=mu, target=tau_target,
                                                    **kw)
            if not np.isfinite(tau_err) or tau_err > tau_target:
                raise ValueError(
                    f"no tabulated minimax grid up to {ntau} points resolves "
                    f"this system's ranges to {tau_target:g} (best "
                    f"{tau_err:g}). The binding ratio is the SELF-ENERGY's "
                    f"rS = {rS[0]:.4g}..{rS[1]:.4g} (R = {rS[1] / rS[0]:.1f}), "
                    f"not the screening range -- pass ntau explicitly and "
                    f"accept the accuracy, or narrow the window.")

        omega_in = gauss_legendre_grid(nw_screening, w0=0.5 * rW[0])[0]

        def Pi(q, _w=omega_in):
            """Pi^q on demand -- see the note in the metallic branch below."""
            return polarizability_q_frequency_occ(X, mo_energy, mo_occ, kplus,
                                                  q, _w, **kw)

        tau = 0.5 * minimax_time_grid(ntau, *rS)[0]
        diag.update(ntau=ntau, nw_screening=nw_screening, tau_fit_error=tau_err)
        qp = []
        for n, x in zip(states, xmv):
            z_fit, iw = imaginary_axis_sample_points(
                gauss_legendre_grid(nw_sigma, w0=0.5 * rW[0])[0], nocc, n, mu)
            sig = sigma_c_diag_thc(X, V, Pi, mo_energy, mo_occ, kmesh, kn, [n],
                                   omega_in, iw, tau, mu=mu, ranges=(rW, rS))[0]
            qp.append(solve_qp_from_imaginary_axis(mo_energy[kn], n, x, z_fit,
                                                   sig, solver_mode=solver_mode))
    else:
        omega_max = self_energy_range_k(mo_energy, mo_occ, rW[1], mu)
        nuB = bosonic_sampling(beta, omega_max)

        # A CLOSURE, not a list. Materializing Pi for every transfer is
        # nq * nfreq * M^2, the single largest object in the route.
        # Building it per q costs one recomputation of nothing (each q is
        # visited once) and removes that tensor entirely.
        def Pi(q, _w=nuB):
            return polarizability_q_frequency_occ(X, mo_energy, mo_occ, kplus,
                                                  q, _w, **kw)

        sig, wF, sdiag = sigma_c_matsubara_thc(X, V, Pi, mo_energy, mo_occ,
                                               kmesh, beta, omega_max, kn,
                                               states, mu=mu,
                                               return_diagnostics=True)
        diag.update(beta=beta, omega_max=omega_max, **sdiag)
        qp = []
        for i, (n, x) in enumerate(zip(states, xmv)):
            # One branch only -- see the docstring. Sigma(-i w) = Sigma(i w)^*,
            # so the two halves are different data, not redundant samples.
            sign = -1.0 if n < nocc else 1.0
            keep = np.where(np.sign(wF) == sign)[0]
            qp.append(solve_qp_from_imaginary_axis(
                mo_energy[kn], n, x, mu + 1j * wF[keep], sig[i][keep],
                solver_mode=solver_mode))

    qp = np.asarray(qp, dtype=float)
    return (qp, diag) if return_diagnostics else qp
