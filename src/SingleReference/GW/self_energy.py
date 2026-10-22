from concurrent.futures import ThreadPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits
from src.SingleReference.GW.transition_amplitudes import AmplitudeGenerator
from src.SingleReference.base import get_occ_virt_indices
from src.Base.constants import (DEFAULT_BROADENING_ETA, DEFAULT_BLOCK_SIZE,
                                METHOD_REGISTRY, QSGW_BLOCK_ELEMS,
                                QSGW_SRG_QUAD_TOL,
                                UHF_PSD_CHARGE_AMPLITUDE_MIN)

# The registry's vertex modes; 'PSD3' is absent from it, unimplemented, and
# raises ValueError rather than silently falling back to plain GW.
KNOWN_VERTEX_MODES = tuple(dict.fromkeys(info['vertex_mode']
                                         for info in METHOD_REGISTRY.values()))
# Vertex modes whose self-energy carries a triplet-exciton sum.
TRIPLET_VERTEX_MODES = tuple(sorted({info['vertex_mode'] for info in METHOD_REGISTRY.values()
                                     if info['needs_triplet']}))


def _pole_sums(weights, omegas, w_grid, eps, nocc_spin, eta, calc_imag,
               block_elems=2**19):
    """sums[k, iw] = sum_{S,q} weights[k][S, q] g(w_grid[iw] - eps_q +
    sign_q Omega^(k)_S).

    g is the real or imaginary part of the broadened denominator, as in
    SelfEnergySolver._denom_grid, sign_q = +1 for q < nocc_spin and -1 otherwise.
    The exciton axis is summed in chunks of at most block_elems // (nw * norb)
    excitations, at least one, through two (nw, chunk, norb) buffers allocated
    once per call and filled in place, so a chunk costs no fresh pages. A call
    holds 2 block_elems doubles, 8 MB at the default, and a pool of concurrent
    callers that many times over.

    Parameters
    ----------
    weights : sequence of ndarray, shape (nexciton_k, norb)
    omegas : sequence of ndarray, shape (nexciton_k,)
    w_grid : ndarray, shape (nw,)
    eps : ndarray, shape (norb,)

    Returns
    -------
    sums : ndarray, shape (len(weights), nw)
    """
    norb = len(eps)
    sign = np.where(np.arange(norb) < nocc_spin, 1.0, -1.0)
    w_grid = np.asarray(w_grid, dtype=float)
    nw = len(w_grid)
    base = w_grid[:, None, None] - eps[None, None, :]            # (nw, 1, norb)
    chunk = max(1, block_elems // (nw * norb))
    out = np.zeros((len(weights), nw))
    # per-call buffers: a pool thread owns its call, so nothing here is shared
    energy = np.empty((nw, chunk, norb))
    denom = np.empty((nw, chunk, norb))
    for k, (wt, om) in enumerate(zip(weights, omegas)):
        for s0 in range(0, len(om), chunk):
            s1 = min(s0 + chunk, len(om))
            e = energy[:, :s1 - s0]                 # a view; the last chunk is short
            d = denom[:, :s1 - s0]
            np.add(base, (sign[None, :] * om[s0:s1, None])[None, :, :], out=e)
            np.multiply(e, e, out=d)
            d += eta**2
            if calc_imag:
                np.divide(-eta, d, out=d)
                d *= sign[None, None, :]
            else:
                np.divide(e, d, out=d)
            out[k] += d.reshape(nw, -1) @ wt[s0:s1].ravel()
    return out


def _srg_laplace_quadrature(mu_max, tol, nodes_per_panel=12, width=4.0):
    """(x, w): nodes in (0, 1] and weights of the Laplace quadrature

        sum_n w_n exp(-μ x_n) ≈ ∫_0^1 exp(-μ x) dx = (1 - exp(-μ)) / μ,

    to a relative error below `tol` for every μ in [0, mu_max].

    Gauss-Legendre in v = ln x, `nodes_per_panel` nodes on panels `width`
    e-folds wide from ln x_lo, x_lo = 0.01 / mu_max, up to 0, and one node at
    x_lo / 2 of weight x_lo for [0, x_lo]. The rule is checked against the
    closed form on 4000 log-spaced μ and μ = 0; a miss narrows the panels by
    3/4, at most five times, then raises ValueError. The panel count grows as
    ln(mu_max): 49 nodes at mu_max = 2.5e4, 61 at 2.5e5, for tol = 1e-7. A
    narrowed rule has one to three panels more, so the count is not monotone
    in mu_max: 73 nodes at 4e4.

    Parameters
    ----------
    mu_max : float
        Largest μ the rule must serve; values below 1 are raised to 1.
    tol : float
        Bound on max_μ |sum_n w_n exp(-μ x_n) / f(μ) - 1|.

    Returns
    -------
    x, w : ndarray, shape (nnode,)
    """
    mu_max = max(float(mu_max), 1.0)
    g, gw = np.polynomial.legendre.leggauss(nodes_per_panel)
    mu = np.concatenate([[0.0], np.logspace(-6, np.log10(mu_max), 4000)])
    exact = np.ones_like(mu)
    exact[1:] = -np.expm1(-mu[1:]) / mu[1:]
    lo = np.log(0.01 / mu_max)
    for _ in range(6):
        edges = np.linspace(lo, 0.0, int(np.ceil(-lo / width)) + 1)
        half = 0.5 * np.diff(edges)[:, None]
        v = (half * g + 0.5 * (edges[1:] + edges[:-1])[:, None]).ravel()
        x = np.concatenate([[0.5 * np.exp(lo)], np.exp(v)])
        w = np.concatenate([[np.exp(lo)], (half * gw).ravel() * np.exp(v)])
        err = np.abs(np.exp(-np.outer(mu, x)) @ w / exact - 1).max()
        if err < tol:
            return x, w
        width *= 0.75
    raise ValueError(f'the Laplace quadrature reaches a relative error of '
                     f'{err:.1e} at mu_max = {mu_max:.3g}, not tol = {tol:.1e}')


class SigmaEvaluator:
    """Sigma_pp(w) for one state, scalar or array w, from weights formed once.

    Built by SelfEnergySolver.self_energy_evaluator; reproduces
    calculate_self_energy for the same arguments to round-off. Each vertex mode
    is a fixed linear combination of pole sums with weights amp2^2, amp2*amp_L,
    amp_psd^2, amp^2 (singlet spectrum) and amp_t^2 (triplet spectrum), so the
    combination is applied to the sums instead of per frequency.

    Parameters
    ----------
    terms : list of (weight, omega) or (weight, omega, eps_k, nocc_k) tuples
        weight : ndarray, shape (nexciton, norb_k); omega : ndarray, shape
        (nexciton,). eps_k, nocc_k: the orbitals r the term sums over, when
        they are not eps, nocc_spin (the unrestricted spin-flip terms).
    combine : callable
        Maps sums, ndarray shape (n_terms, nw), to Sigma(w), shape (nw,).
    eps : ndarray, shape (norb,)
    nocc_spin : int
    eta : float
    calc_imag : bool
    """

    def __init__(self, terms, combine, eps, nocc_spin, eta, calc_imag):
        # terms: list of (weight (nexciton, norb), omega (nexciton,)); combine
        # maps the (n_terms, nw) sums to Sigma(w) of shape (nw,)
        self._weights = [t[0] for t in terms]
        self._omegas = [t[1] for t in terms]
        self._levels = [(t[2], t[3]) if len(t) == 4 else (eps, nocc_spin)
                        for t in terms]
        self._combine = combine
        self._eps = eps
        self._nocc_spin = nocc_spin
        self._eta = eta
        self._calc_imag = calc_imag
        self.n_terms = len(terms)

    def __call__(self, w):
        """Sigma_pp(w).

        Parameters
        ----------
        w : float or ndarray, shape (nw,)

        Returns
        -------
        float, or ndarray, shape (nw,)
        """
        scalar = np.isscalar(w)
        w = np.atleast_1d(w)
        if all(lv[0] is self._eps for lv in self._levels):
            sums = _pole_sums(self._weights, self._omegas, w, self._eps,
                              self._nocc_spin, self._eta, self._calc_imag)
        else:
            sums = np.concatenate([
                _pole_sums([wt], [om], w, eps_k, nocc_k, self._eta, self._calc_imag)
                for wt, om, (eps_k, nocc_k) in zip(self._weights, self._omegas,
                                                   self._levels)])
        val = self._combine(sums)
        return float(val[0]) if scalar else val

    def poles(self):
        """(residues, poles), flat: Sigma_pp(w) = sum_k r_k / (w - P_k) at
        eta = 0, with P = eps_q - sign_q Omega_S. Every vertex mode is a
        linear combination of the pole sums, so its coefficients are combine
        applied to unit sums."""
        if self._calc_imag:
            raise ValueError("poles() is the real part's pole representation")
        coef = self._combine(np.eye(self.n_terms))
        res = [c * wt for c, wt in zip(coef, self._weights)]
        pol = []
        for om, (eps_k, nocc_k) in zip(self._omegas, self._levels):
            sign = np.where(np.arange(len(eps_k)) < nocc_k, 1.0, -1.0)
            pol.append(eps_k[None, :] - sign[None, :] * om[:, None])
        return (np.concatenate([r.ravel() for r in res]),
                np.concatenate([q.ravel() for q in pol]))


class SelfEnergySolver(AmplitudeGenerator):
    """Diagonal GW and vertex-corrected self-energies (Sigma_pp): restricted/unrestricted spin, bare/screened, DF or full ERIs."""
    def __init__(self, eps, df_coeff=None, eri_chemist=None, spin_mode='restricted',
                 eta=DEFAULT_BROADENING_ETA, block_size=DEFAULT_BLOCK_SIZE):
        """eps/df_coeff/eri_chemist: single array if restricted, (alpha, beta[, ab]) tuple if unrestricted."""
        self.spin_mode = spin_mode.lower()
        self.eta = eta
        self.block_size = block_size
        
        if self.spin_mode == 'unrestricted':
            self.eps_a, self.eps_b = eps
            if df_coeff is not None:
                if len(df_coeff) == 3:
                    self.df_a, self.df_b, self.df_ab = df_coeff
                else:
                    self.df_a, self.df_b = df_coeff
                    self.df_ab = None
            else:
                self.df_a, self.df_b, self.df_ab = None, None, None
            self.eri_a, self.eri_b, self.eri_ab = eri_chemist if eri_chemist is not None else (None, None, None)
            self.norb_a = len(self.eps_a)
            self.norb_b = len(self.eps_b)
            if self.df_a is not None:
                self.naux = self.df_a.shape[0]
        else:
            self.eps = eps
            self.df_coeff = df_coeff
            self.eri_chemist = eri_chemist
            self.norb = len(self.eps)
            if self.df_coeff is not None:
                self.naux = self.df_coeff.shape[0]

    def _get_occ_virt_indices(self, eps, nocc):
        return get_occ_virt_indices(eps, nocc)

    def _denom_grid(self, w, eps, nocc_spin, eigenvalues_casida, calc_imag):
        """Vectorized (nexciton, norb) energy-denominator grid: denom[S,i] = f(w - eps[i] + sign(i)*Omega[S]), sign=+1 iff i<nocc_spin."""
        norb = len(eps)
        sign = np.where(np.arange(norb) < nocc_spin, 1.0, -1.0)
        energy = w - eps[None, :] + sign[None, :] * eigenvalues_casida[:, None]
        if calc_imag:
            return -sign[None, :] * self.eta / (energy**2 + self.eta**2)
        return energy / (energy**2 + self.eta**2)

    def calculate_self_energy(self, p_state, freq, nocc, eigenvalues_casida, chiXYa, chiXYb=None,
                              eigenvalues_casida_t=None, chiXYb_t=None, spin_channel='alpha', vertex_mode='GW',
                              calc_imag=False):
        """Diagonal self-energy Sigma_pp(omega) for a single frequency or grid; vertex_mode: 'GW', 'GWGammaInf', 'PSD1'-'PSD9'."""
        is_scalar = np.isscalar(freq)
        freq_grid = np.atleast_1d(freq)
        
        if self.spin_mode == 'unrestricted':
            eps = self.eps_a if spin_channel == 'alpha' else self.eps_b
            nocc_a, nocc_b = nocc
            nocc_spin = nocc_a if spin_channel == 'alpha' else nocc_b
            prefactor = 1.0
        else:
            eps = self.eps
            nocc_spin = nocc
            prefactor = 2.0
            
        norb = len(eps)
        nexciton = len(eigenvalues_casida)
        
        if chiXYa.ndim == 2:
            amp2 = chiXYa
        else:
            amp2 = chiXYa[:, :, p_state]
            
        if chiXYb is not None:
            if chiXYb.ndim == 2:
                amp = chiXYb
            else:
                amp = chiXYb[:, :, p_state]
        else:
            amp = None
            
        _require_vertex_inputs(vertex_mode, amp, eigenvalues_casida_t, chiXYb_t)

        sigma_grid = []
        for w in freq_grid:
            denom = self._denom_grid(w, eps, nocc_spin, eigenvalues_casida, calc_imag)

            if vertex_mode == 'GW' or amp is None:
                val = prefactor * np.sum((amp2**2) * denom)
            elif vertex_mode == 'GWGammaInf':
                amp_L = amp2 - 0.5 * amp
                val = prefactor * np.sum(amp2 * amp_L * denom)
            else:
                # PSD1/2/4/5/6/7/8/9 all build on the plain-GW sum and the
                # vertex-corrected "2*amp2-amp" sum; computed once here.
                amp_psd = 2.0 * amp2 - amp
                # Unrestricted, the spin-conserving excitons hold the singlets and
                # the M_s = 0 triplet components. The triplet modes keep them all
                # (their spin-flip sum below carries M_s = +-1); the singlet-only
                # modes drop the excitons without a charge amplitude.
                mask = 1.0
                if self.spin_mode == 'unrestricted' and vertex_mode not in TRIPLET_VERTEX_MODES:
                    mask = (np.linalg.norm(amp2, axis=1)
                            > UHF_PSD_CHARGE_AMPLITUDE_MIN)[:, None]
                    amp_psd = amp_psd * mask
                sigmaGW = prefactor * np.sum((amp2**2) * denom)
                sigmaPSDI = 0.25 * prefactor * np.sum(amp_psd * amp_psd * denom)

                if vertex_mode in ('PSD1', 'PSD2', 'PSD4'):
                    val = sigmaPSDI
                elif vertex_mode == 'PSD5':
                    val = 0.5 * (0.75 * sigmaGW + sigmaPSDI)
                elif vertex_mode in ('PSD6', 'PSD7'):
                    val = 0.5 * (sigmaGW + sigmaPSDI)
                else:  # PSD8, PSD9
                    amp_eff = amp * mask
                    sigmaTs = 0.5 * prefactor * np.sum(amp_eff * amp_eff * denom)
                    val = 0.5 * (sigmaGW + sigmaPSDI + sigmaTs)

            if eigenvalues_casida_t is not None and chiXYb_t is not None:
                if self.spin_mode == 'unrestricted':
                    # The spin-orbital eh pair: the spin-conserving sum above holds
                    # the singlets and the M_s = 0 triplets, the spin-flip
                    # excitons (r of the other spin, weight one each) hold
                    # M_s = +-1. On a closed shell this sum is (2/3) of the triplet
                    # term, so the restricted triplet weights 3/2, 3/4, 3/4, 3/2
                    # become 1, 1/2, 1/2, 3/4 here, the rest sitting in sigmaPSDI
                    # and sigmaTs.
                    eps_t = self.eps_b if spin_channel == 'alpha' else self.eps_a
                    nocc_t = nocc[1] if spin_channel == 'alpha' else nocc[0]
                    amp_t = chiXYb_t if chiXYb_t.ndim == 2 else chiXYb_t[:, :, p_state]
                    denom_t = self._denom_grid(w, eps_t, nocc_t, eigenvalues_casida_t, calc_imag)
                    sigmaSF = np.sum((amp_t**2) * denom_t)
                    if vertex_mode == 'PSD2':
                        val += sigmaSF
                    elif vertex_mode == 'PSD4':
                        val = 0.5 * val + 0.5 * sigmaSF
                    elif vertex_mode == 'PSD7':
                        val += 0.5 * sigmaSF
                    elif vertex_mode == 'PSD9':
                        val += 0.75 * sigmaSF
                else:
                    if chiXYb_t.ndim == 2:
                        amp_t = chiXYb_t
                    else:
                        amp_t = chiXYb_t[:, :, p_state]
                    denom_t = self._denom_grid(w, eps, nocc_spin, eigenvalues_casida_t, calc_imag)
                    sigmaTt = (0.5 * prefactor) * np.sum((amp_t**2) * denom_t)
                    
                    if vertex_mode == 'PSD2':
                        val += 1.5 * sigmaTt
                    elif vertex_mode == 'PSD4':
                        val = 0.5 * val + 0.75 * sigmaTt
                    elif vertex_mode == 'PSD7':
                        val += 0.75 * sigmaTt
                    elif vertex_mode == 'PSD9':
                        val += 1.5 * sigmaTt
                
            sigma_grid.append(val)
            
        if is_scalar:
            return sigma_grid[0]
        else:
            return np.array(sigma_grid)

    def self_energy_evaluator(self, p_state, nocc, eigenvalues_casida, chiXYa,
                              chiXYb=None, eigenvalues_casida_t=None,
                              chiXYb_t=None, spin_channel='alpha',
                              vertex_mode='GW', calc_imag=False):
        """SigmaEvaluator for Sigma_pp(w).

        Same arguments as calculate_self_energy; the weights are formed once
        and w may be a scalar or a grid when the returned evaluator is called.

        Parameters
        ----------
        p_state : int
        nocc : int, or (int, int) if unrestricted
        eigenvalues_casida : ndarray, shape (nexciton,)
        chiXYa : ndarray, shape (nexciton, norb) or (nexciton, norb, norb)
        chiXYb : ndarray, same shape convention as chiXYa, optional
        eigenvalues_casida_t : ndarray, shape (nexciton_t,), optional
        chiXYb_t : ndarray, same shape convention as chiXYa, optional
        spin_channel : {'alpha', 'beta'}
        vertex_mode : str
        calc_imag : bool

        Returns
        -------
        SigmaEvaluator
        """
        if self.spin_mode == 'unrestricted':
            eps = self.eps_a if spin_channel == 'alpha' else self.eps_b
            nocc_a, nocc_b = nocc
            nocc_spin = nocc_a if spin_channel == 'alpha' else nocc_b
            prefactor = 1.0
        else:
            eps = self.eps
            nocc_spin = nocc
            prefactor = 2.0
        _require_vertex_inputs(vertex_mode, chiXYb, eigenvalues_casida_t,
                               chiXYb_t)
        amp2 = chiXYa if chiXYa.ndim == 2 else chiXYa[:, :, p_state]
        amp = None
        if chiXYb is not None:
            amp = chiXYb if chiXYb.ndim == 2 else chiXYb[:, :, p_state]
        om = eigenvalues_casida

        terms = []
        if vertex_mode == 'GW':
            terms.append((amp2**2, om))
            singlet = lambda s: prefactor * s[0]
        elif vertex_mode == 'GWGammaInf':
            terms.append((amp2 * (amp2 - 0.5 * amp), om))
            singlet = lambda s: prefactor * s[0]
        else:
            amp_psd = 2.0 * amp2 - amp
            # as in calculate_self_energy: only the singlet-only modes drop the
            # unrestricted excitons without a charge amplitude
            mask = 1.0
            if self.spin_mode == 'unrestricted' and vertex_mode not in TRIPLET_VERTEX_MODES:
                mask = (np.linalg.norm(amp2, axis=1)
                        > UHF_PSD_CHARGE_AMPLITUDE_MIN)[:, None]
                amp_psd = amp_psd * mask
            terms.append((amp2**2, om))               # sigmaGW / prefactor
            # sigmaPSDI / (0.25 prefactor)
            terms.append((amp_psd * amp_psd, om))
            if vertex_mode in ('PSD8', 'PSD9'):
                amp_eff = amp * mask
                terms.append((amp_eff * amp_eff, om))  # sigmaTs / (0.5 prefactor)

            def singlet(s, mode=vertex_mode):
                sigma_gw = prefactor * s[0]
                sigma_psdi = 0.25 * prefactor * s[1]
                if mode in ('PSD1', 'PSD2', 'PSD4'):
                    return sigma_psdi
                if mode == 'PSD5':
                    return 0.5 * (0.75 * sigma_gw + sigma_psdi)
                if mode in ('PSD6', 'PSD7'):
                    return 0.5 * (sigma_gw + sigma_psdi)
                return 0.5 * (sigma_gw + sigma_psdi + 0.5 * prefactor * s[2])

        n_singlet = len(terms)
        triplet = eigenvalues_casida_t is not None and chiXYb_t is not None
        if triplet:
            if self.spin_mode == 'unrestricted':
                # spin-flip excitons: r runs over the other spin's orbitals
                eps_t = self.eps_b if spin_channel == 'alpha' else self.eps_a
                nocc_t = nocc[1] if spin_channel == 'alpha' else nocc[0]
                amp_t = chiXYb_t if chiXYb_t.ndim == 2 else chiXYb_t[:, :, p_state]
                terms.append((amp_t**2, eigenvalues_casida_t, eps_t, nocc_t))
            else:
                amp_t = chiXYb_t if chiXYb_t.ndim == 2 else chiXYb_t[:, :, p_state]
                terms.append((amp_t**2, eigenvalues_casida_t))

        unrestricted = self.spin_mode == 'unrestricted'

        def combine(s, mode=vertex_mode, unrestricted=unrestricted):
            val = singlet(s)
            if not triplet:
                return val
            if unrestricted:
                # weights of the spin-flip sum, see calculate_self_energy
                sigma_sf = s[n_singlet]
                if mode == 'PSD2':
                    return val + sigma_sf
                if mode == 'PSD4':
                    return 0.5 * val + 0.5 * sigma_sf
                if mode == 'PSD7':
                    return val + 0.5 * sigma_sf
                if mode == 'PSD9':
                    return val + 0.75 * sigma_sf
                return val
            sigma_tt = (0.5 * prefactor) * s[n_singlet]
            if mode == 'PSD2':
                return val + 1.5 * sigma_tt
            if mode == 'PSD4':
                return 0.5 * val + 0.75 * sigma_tt
            if mode == 'PSD7':
                return val + 0.75 * sigma_tt
            if mode == 'PSD9':
                return val + 1.5 * sigma_tt
            return val

        return SigmaEvaluator(terms, combine, eps, nocc_spin, self.eta, calc_imag)

    def calculate_self_energy_matrix(self, nocc, eigenvalues_casida, chiXYa, chiXYb=None, eigenvalues=None, vertex_mode='GW'):
        """Full self-energy matrix at the QP energies (eigenvalues), symmetrized."""
        if eigenvalues is None:
            eigenvalues = self.eps
            
        if self.spin_mode == 'unrestricted':
            eps = eigenvalues
            nocc_spin = nocc
            prefactor = 0.5
        else:
            eps = eigenvalues
            nocc_spin = nocc
            prefactor = 1.0
            
        nmo = len(eps)
        nexciton = len(eigenvalues_casida)
        
        sign = np.zeros(nmo)
        for i in range(nmo):
            sign[i] = 1.0 if i < nocc_spin else -1.0
            
        eps_p = eps[None, None, :]
        eps_r = eps[None, :, None]
        sign_r_omega_S = (sign[:, None] * eigenvalues_casida[None, :]).T[:, :, None]
        
        energy = eps_p - eps_r + sign_r_omega_S
        denom = energy / (energy**2 + self.eta**2)
        
        if vertex_mode == 'GW' or chiXYb is None:
            chi_T = chiXYa * denom
            tmp = np.einsum('Srq, Srp -> qp', chiXYa, chi_T)
        elif vertex_mode == 'GWGammaInf':
            chi_T = (chiXYa - 0.5 * chiXYb) * denom
            tmp = np.einsum('Srq, Srp -> qp', chiXYa, chi_T)
        elif vertex_mode in ['PSD1', 'PSD2', 'PSD4']:
            chi_T = (2.0 * chiXYa - chiXYb) * denom
            tmp = 0.5 * np.einsum('Srq, Srp -> qp', chi_T, chi_T)
        elif vertex_mode == 'PSD5':
            tmp_gw = np.einsum('Srq, Srp -> qp', chiXYa, chiXYa * denom)
            amp_psd = 2.0 * chiXYa - chiXYb
            tmp_psdi = 0.5 * np.einsum('Srq, Srp -> qp', amp_psd, amp_psd * denom)
            tmp = 0.5 * (tmp_gw + tmp_psdi)
        else:
            chi_T = chiXYa * denom
            tmp = np.einsum('Srq, Srp -> qp', chiXYa, chi_T)
            
        self_energy_matrix = prefactor * (tmp + tmp.T)
        return self_energy_matrix

    def static_self_energy_matrix(self, nocc, eigenvalues_casida, rho, eigenvalues=None,
                                  flow=None, block_elems=QSGW_BLOCK_ELEMS,
                                  quad_tol=QSGW_SRG_QUAD_TOL, n_workers=1):
        """The static Hermitian GW self-energy of qsGW, restricted: Kotani's mode A,
        or its SRG-regularized form when `flow` is given.

        With Δ_Srp = ε_p - ε_r + s_r Ω_S, s_r = +1 for r < nocc and -1 otherwise,
        and χ_Srp = sum_P ρ_PS B_Prp the amplitudes on this solver's DF factors:

            mode A (flow None):
                Σ~_pq = ½ [Σ_pq(ε_p) + Σ_pq(ε_q)]
                      = 2 sum_Sr χ_Srp χ_Srq ½ [g(Δ_Srp) + g(Δ_Srq)],
                g(Δ) = Δ / (Δ² + η²), the real part of _denom_grid's denominator;
            SRG (flow = s, in Hartree^-2):
                Σ~_pq(s) = 2 sum_Sr χ_Srp χ_Srq K(Δ_Srp, Δ_Srq),
                K(a, b) = (a + b) / (a² + b²) [1 - exp(-(a² + b²) s)].

        The 2 is the restricted spin sum of calculate_self_energy, folded into
        T + Tᵀ. On the diagonal both reduce to 2 sum_Sr χ_Srp² times a
        regularized 1/Δ_Srp; they differ where some Δ is within η, or within
        1/sqrt(2 s), of zero. s = 0 is the Hartree-Fock limit, Σ~ = 0.

        The exciton axis runs in chunks of at most block_elems // (norb * nmo)
        excitations, so no (nexciton, norb, nmo) array exists. Mode A is two
        GEMMs per chunk. K couples p and q; its Laplace form

            (1 - exp(-s λ)) / λ = ∫_0^s exp(-t λ) dt ≈ sum_n w_n exp(-t_n λ),

        λ = a² + b², separates them, K(a, b) ≈ (a + b) sum_n w_n exp(-t_n a²)
        exp(-t_n b²), so each quadrature node is one more GEMM per chunk. The
        node count grows with ln(2 s a_max²), a_max = ε_max - ε_min + Ω_max: 61
        on water/cc-pVDZ at s = 100 (_srg_laplace_quadrature). Each term of Σ~
        carries a relative error below quad_tol, so an element misses by at most
        quad_tol times 2 sum_Sr |χ_Srp χ_Srq K(Δ_Srp, Δ_Srq)|, the sum of its
        terms' magnitudes; where they cancel, that exceeds quad_tol times the
        element itself. Per node the elementwise build of U and
        V costs about as much as the GEMM once nmo nears 10³, and numpy runs it
        on one thread, so n_workers > 1 hands each worker a fixed, interleaved
        share of the chunks, its own accumulator and BLAS pinned to one thread.

        Parameters
        ----------
        nocc : int
        eigenvalues_casida : ndarray, shape (nexciton,)
            Ω_S, Hartree.
        rho : ndarray, shape (naux, nexciton), index order (P, S)
            The DF transition density ρ_PS = sum_ia B_Pia (X+Y)_ia,S, in whichever
            orbital basis the Casida problem was solved (_rho_a_df); the DF
            factors of this solver set the basis of the result.
        eigenvalues : ndarray, shape (nmo,), optional
            ε, the poles and evaluation points; default this solver's eps.
        flow : float, optional
            The SRG flow parameter s, Hartree^-2; None gives mode A with self.eta.
        block_elems : int
            Bound on the elements of each chunk buffer, summed over the workers.
        quad_tol : float
            Relative error bound of the SRG quadrature on each term.
        n_workers : int
            Threads for the SRG arm; mode A runs on BLAS's own threads.

        Returns
        -------
        ndarray, shape (nmo, nmo), index order (p, q), symmetric, Hartree

        References
        ----------
        Mode A: Kotani, van Schilfgaarde and Faleev, Phys. Rev. B 76, 165106
        (2007). SRG: Marie and Loos, J. Chem. Theory Comput. 2023,
        doi 10.1021/acs.jctc.3c00281, eq. 44 of arXiv:2303.05984.
        """
        if self.spin_mode != 'restricted':
            raise NotImplementedError(
                'static_self_energy_matrix is restricted-spin only')
        if self.df_coeff is None:
            raise ValueError('static_self_energy_matrix needs the DF factors')
        if flow is not None and not (np.isfinite(flow) and flow >= 0):
            raise ValueError(f'flow={flow!r}: the SRG flow parameter is a finite '
                             f's >= 0')
        eps = self.eps if eigenvalues is None else np.asarray(eigenvalues, float)
        om = np.asarray(eigenvalues_casida, float)
        naux, norb, nmo = self.df_coeff.shape
        if flow == 0 or len(om) == 0:
            return np.zeros((nmo, nmo))
        sign = np.where(np.arange(norb) < nocc, 1.0, -1.0)
        coeff = self.df_coeff.reshape(naux, norb * nmo)
        base = eps[None, None, :] - eps[None, :, None]              # (1, norb, nmo)
        chunk = max(1, int(block_elems) // (norb * nmo))
        tmp = np.zeros((nmo, nmo))
        if flow is None:
            for s0 in range(0, len(om), chunk):
                s1 = min(s0 + chunk, len(om))
                # χ[S, r, p] = sum_P ρ[P, S] B[P, r, p]
                chi = (rho[:, s0:s1].T @ coeff).reshape(s1 - s0, norb, nmo)
                # Δ[S, r, p] = ε_p - ε_r + s_r Ω_S
                energy = base + (sign[None, :] * om[s0:s1, None])[:, :, None]
                # mode A: T[q, p] += sum_{S, r} χ[S, r, q] χ[S, r, p] g(Δ[S, r, p])
                g = energy / (energy**2 + self.eta**2)
                g *= chi
                tmp += chi.reshape(-1, nmo).T @ g.reshape(-1, nmo)
            return tmp + tmp.T

        # every s λ of this call is at most 2 s (ε_max - ε_min + Ω_max)²
        a_max = eps.max() - eps.min() + om.max()
        x, w = _srg_laplace_quadrature(2.0 * flow * a_max**2, quad_tol)
        t_nodes, w_nodes = flow * x, flow * w
        n_workers = max(1, int(n_workers))
        chunk = max(1, int(block_elems) // (norb * nmo * n_workers))
        bounds = [(s0, min(s0 + chunk, len(om))) for s0 in range(0, len(om), chunk)]
        n_workers = min(n_workers, len(bounds))

        def accumulate(share):
            part = np.zeros((nmo, nmo))
            for s0, s1 in share:
                # χ[k, p] = sum_P ρ[P, S] B[P, r, p], k = (S, r) flattened
                chi = (rho[:, s0:s1].T @ coeff).reshape(-1, nmo)
                # Δ[k, p] = ε_p - ε_r + s_r Ω_S
                energy = (base + (sign[None, :] * om[s0:s1, None])[:, :, None]
                          ).reshape(-1, nmo)
                sq = energy**2
                v = np.empty_like(sq)
                for t, wt in zip(t_nodes, w_nodes):
                    # per node n: T[p, q] += 2 w_n sum_k U[k, p] V[k, q],
                    # U[k, p] = χ[k, p] Δ[k, p] exp(-t_n Δ[k, p]²),
                    # V[k, q] = χ[k, q] exp(-t_n Δ[k, q]²)
                    np.multiply(sq, -t, out=v)
                    np.exp(v, out=v)
                    v *= chi
                    u = v * energy
                    part += (2.0 * wt) * (u.T @ v)
            return part

        if n_workers == 1:
            tmp += accumulate(bounds)
        else:
            shares = [bounds[i::n_workers] for i in range(n_workers)]
            with threadpool_limits(limits=1, user_api='blas'):
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    for part in pool.map(accumulate, shares):
                        tmp += part
        return tmp + tmp.T

    def calculate_self_energy_diagonal_batch(self, freq, nocc, eigenvalues_casida, chiXYa,
                                              chiXYb=None, spin_channel='alpha', vertex_mode='GW',
                                              calc_imag=False):
        """Vectorized batch counterpart to calculate_self_energy: Sigma_pp(freq[p]) for every p at once (freq is length-norb).

        Restricted spin and the vertex modes without a triplet-exciton sum only.
        """
        if self.spin_mode != 'restricted':
            raise NotImplementedError('the batch self-energy is restricted-spin only')
        if vertex_mode not in KNOWN_VERTEX_MODES or vertex_mode in TRIPLET_VERTEX_MODES:
            served = [m for m in KNOWN_VERTEX_MODES if m not in TRIPLET_VERTEX_MODES]
            raise ValueError(f"the batch self-energy has no triplet-exciton sum and serves "
                             f"vertex_mode in {served}, not {vertex_mode!r}")
        _require_vertex_inputs(vertex_mode, chiXYb)
        eps = self.eps
        nocc_spin = nocc
        prefactor = 2.0
        norb = len(eps)

        sign = np.where(np.arange(norb) < nocc_spin, 1.0, -1.0)
        sign_r_omega_S = (sign[:, None] * eigenvalues_casida[None, :]).T[:, :, None]
        energy = freq[None, None, :] - eps[None, :, None] + sign_r_omega_S
        if calc_imag:
            denom = -sign[None, :, None] * self.eta / (energy**2 + self.eta**2)
        else:
            denom = energy / (energy**2 + self.eta**2)

        amp2 = chiXYa
        if vertex_mode == 'GW':
            val = prefactor * np.einsum('Srp,Srp->p', amp2**2, denom)
        elif vertex_mode == 'GWGammaInf':
            amp_L = amp2 - 0.5 * chiXYb
            val = prefactor * np.einsum('Srp,Srp->p', amp2 * amp_L, denom)
        else:
            amp_psd = 2.0 * amp2 - chiXYb
            sigmaGW = prefactor * np.einsum('Srp,Srp->p', amp2**2, denom)
            sigmaPSDI = 0.25 * prefactor * np.einsum('Srp,Srp->p', amp_psd**2, denom)
            if vertex_mode in ('PSD1', 'PSD2', 'PSD4'):
                val = sigmaPSDI
            elif vertex_mode == 'PSD5':
                val = 0.5 * (0.75 * sigmaGW + sigmaPSDI)
            elif vertex_mode in ('PSD6', 'PSD7'):
                val = 0.5 * (sigmaGW + sigmaPSDI)
            else:
                sigmaTs = 0.5 * prefactor * np.einsum('Srp,Srp->p', chiXYb**2, denom)
                val = 0.5 * (sigmaGW + sigmaPSDI + sigmaTs)
        return val

    def calculate_sigma_hx(self, mol, mf, dm, mo_coeff):
        """HF contribution beyond kinetic/external energy (Sigma_Hx = V_H + V_x) in the MO basis, for a given density matrix."""
        V_Hx = mf.get_veff(mol, dm)
        if self.spin_mode == 'unrestricted':
            mo_coeff_a, mo_coeff_b = mo_coeff
            V_Hx_a, V_Hx_b = V_Hx
            V_Hx_mo_a = mo_coeff_a.T @ V_Hx_a @ mo_coeff_a
            V_Hx_mo_b = mo_coeff_b.T @ V_Hx_b @ mo_coeff_b
            return V_Hx_mo_a, V_Hx_mo_b
        else:
            V_Hx_mo = mo_coeff.T @ V_Hx @ mo_coeff
            return V_Hx_mo

    def calculate_spectral_function(self, p_state, omega_grid, nocc, eigenvalues_casida, chiXYa, chiXYb=None, 
                                    eigenvalues_casida_t=None, chiXYb_t=None, spin_channel='alpha', vertex_mode='GW',
                                    V_xc_mo=0.0):
        """Spectral function A(omega) on a frequency grid for a target MO state."""
        if self.spin_mode == 'unrestricted':
            eps = self.eps_a if spin_channel == 'alpha' else self.eps_b
        else:
            eps = self.eps
            
        eps_p = eps[p_state]
        
        sigma_re = self.calculate_self_energy(
            p_state, omega_grid, nocc, eigenvalues_casida, chiXYa, chiXYb,
            eigenvalues_casida_t, chiXYb_t, spin_channel, vertex_mode, calc_imag=False
        )
        
        sigma_im = self.calculate_self_energy(
            p_state, omega_grid, nocc, eigenvalues_casida, chiXYa, chiXYb,
            eigenvalues_casida_t, chiXYb_t, spin_channel, vertex_mode, calc_imag=True
        )
        
        denom = (omega_grid - eps_p - sigma_re - V_xc_mo)**2 + sigma_im**2
        spectral_function = -sigma_im / (np.pi * denom)
        
        return spectral_function, sigma_re, sigma_im


def _require_vertex_inputs(vertex_mode, amplitudes, eigenvalues_t=None,
                           amplitudes_t=None):
    """Refuse a vertex mode that is unknown or lacks what its sum reads:
    without the vertex amplitudes the sum is plain GW, and without the triplet
    excitons a triplet-carrying mode drops that term."""
    if vertex_mode not in KNOWN_VERTEX_MODES:
        raise ValueError(
            f"Unknown vertex_mode '{vertex_mode}'; expected one of {KNOWN_VERTEX_MODES}."
        )
    if vertex_mode != 'GW' and amplitudes is None:
        raise ValueError(f"vertex_mode={vertex_mode!r} needs the vertex amplitudes chiXYb; "
                         f"without them the sum is plain GW")
    if vertex_mode in TRIPLET_VERTEX_MODES and (eigenvalues_t is None
                                                 or amplitudes_t is None):
        raise ValueError(f"vertex_mode={vertex_mode!r} carries a triplet-exciton term; "
                         f"hand eigenvalues_casida_t and chiXYb_t, or the sum drops it")
