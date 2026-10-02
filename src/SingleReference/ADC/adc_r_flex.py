"""Restricted one-shot FLEX self-energy on Hartree-Fock orbitals.

FLEX (Bickers, Scalapino, White, PRL 62, 961 (1989)) exchanges one channel
at a time: the self-energy is a line G times the reducible own-channel vertex
of one pair, summed over the channels, with the second-order term every
channel contains subtracted so that it is counted once. With the pair
vertices of Faddeev-ADC(3) and HF lines, one shot:

    Sigma_FLEX = Sigma_eh + Sigma_pp - 2 Sigma(2),

    Sigma_eh : the single particle-hole pair in the ORDERED three-particle
               space, i.e. the PSD2 self-energy (GW.self_energy,
               vertex_mode='PSD2') on the eh channel's phonons; it holds
               Sigma(2) twice (both orderings of the spectator),
    Sigma_pp : the single particle-particle (hole-hole) pair with its
               spectator, i.e. Faddeev-ADC(3) with both eh pairs switched
               off; it holds Sigma(2) once.

Exact through third order, like ADC(3) and Faddeev-ADC(3); from fourth order
on it lacks Faddeev's sequences of different pairs (and treats the spectator
as distinguishable in the eh channel), which is what the comparison isolates.

The eh phonons are those of adc_r_faddeev._eh_channel (route 'phonon'), so
the channel treatments 'rpa', 'first_order' and 'off' carry over; Sigma(2) is
ADC.second_order. There is no Hermitian upfolded form
(-2 Sigma(2) has negative residues), so quasiparticles come from the
diagonal QP equation.
"""
import numpy as np
import scipy.linalg as la

from src.SingleReference.ADC import adc_r_faddeev as FR
from src.SingleReference.ADC.second_order import sigma2_poles
from src.SingleReference.ADC.solve import _LanczosState, _lanczos_extend
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

LANCZOS_BLOCK = 20          # steps added per extension of a resolvent Lanczos
LANCZOS_MAX_STEPS = 2000
SIGMA_CF_TOL = 1e-10        # |change of Sigma(w)| over one block, Ha
QP_NEWTON_TOL = 1e-8
QP_NEWTON_MAX_ITER = 30


class _Resolvent:
    """u^T (w - H)^{-1} u for a symmetric H given as a matvec, for any w, from
    ONE Lanczos run started at u (Gauss quadrature of the spectral measure):
    Sigma(w) = |u|^2 sum_j S_0j^2 / (w - theta_j) with (theta, S) the Ritz
    pairs of the tridiagonal. Extended in blocks until Sigma(w) at the w asked
    for stops changing. The recurrence is solve._lanczos_extend without
    reorthogonalization or stored vectors: loss of orthogonality only
    duplicates converged Ritz values, which the quadrature tolerates."""

    def __init__(self, matvec, u):
        self.matvec = matvec
        self.norm2 = float(u @ u)
        self.state = _LanczosState(u, keep_vectors=False)
        self._ritz_cache = {}

    def _extend(self, nsteps):
        _lanczos_extend(self.matvec, self.state, nsteps, reorth=False)

    def _ritz(self, n):
        """(theta, weights) of the n-step tridiagonal, kept per n: a root scan
        evaluates Sigma at many w on the same Lanczos run."""
        if n not in self._ritz_cache:
            a = np.asarray(self.state.a[:n])
            b = np.asarray(self.state.b[:n - 1])
            try:
                th, S = la.eigh_tridiagonal(a, b)
            except la.LinAlgError:
                # MRRR (stemr) can fail on clustered Ritz values; QL (stev)
                # and a dense eigh of the tridiagonal do not
                try:
                    th, S = la.eigh_tridiagonal(a, b, lapack_driver='stev')
                except la.LinAlgError:
                    th, S = la.eigh(np.diag(a) + np.diag(b, 1) + np.diag(b, -1))
            self._ritz_cache[n] = (th, self.norm2 * S[0] ** 2)
        return self._ritz_cache[n]

    def ritz(self):
        """(theta, weights) of the run as far as it has been extended."""
        return self._ritz(len(self.state.a))

    def _eval(self, w, n):
        th, wt = self._ritz(n)
        return _pole_sum(wt, th, w)

    def __call__(self, w):
        if not self.state.a:
            self._extend(2 * LANCZOS_BLOCK)
        while True:
            n = len(self.state.a)
            s1, d1 = self._eval(w, n)
            if self.state.breakdown:
                return s1, d1
            s0, _ = self._eval(w, max(1, n - LANCZOS_BLOCK))
            if abs(s1 - s0) < SIGMA_CF_TOL:
                return s1, d1
            if n >= LANCZOS_MAX_STEPS:
                raise RuntimeError(f"_Resolvent: Sigma({w}) not settled in "
                                   f"{n} Lanczos steps (last change {abs(s1 - s0):.2e})")
            self._extend(LANCZOS_BLOCK)


def _pole_sum(residues, poles, w):
    """(sum r / (w - P), its w-derivative) of a flat pole list."""
    d = 1.0 / (w - poles)
    return float(residues @ d), -float(residues @ (d * d))


class FlexSelfEnergy:
    """Diagonal one-shot FLEX self-energy of a restricted HF reference, and
    the same diagonal QP machinery for Faddeev-ADC(3) (for a like-for-like
    comparison of the two self-energies).

    s: an ADCSolverRestricted built from the HF arrays with B_aa (DF).
    eh_triplets: the treatments of the particle-hole triplet channel to
    prepare, as in adc_r_faddeev ('rpa', 'first_order', 'off'); every
    evaluation names one. The singlet eh channel and the pp/hh ladders are
    RPA, and the pp term (its channels and Lanczos runs) is shared by all.
    tda=True puts EVERY channel (eh singlet and triplet, pp, hh) in the
    Tamm-Dancoff approximation instead: the bare-vertex ladders are kept, the
    backward amplitudes dropped (T = 0, dH = dN = 0) -- "FLEX TDA" of Marie
    and Loos, J. Chem. Phys. 163, 194115 (2025); eh_triplets is then unused.
    ladders=False skips the pp/hh ladder channels (the costly part of the
    set-up): only the eh term and Sigma(2) are then available, e.g. for PSD2
    (= the eh term on TDHF phonons) with the triplet channel at first order.
    """

    def __init__(self, s, nocc, eh_triplets=('rpa',), tda=False, ladders=True):
        if s.B_aa is None:
            raise ValueError("FlexSelfEnergy needs the DF factor B_aa")
        self.s, self.nocc, self.B = s, nocc, s.B_aa
        self.eps = np.asarray(s.eps)
        self.norb = len(self.eps)
        O = nocc
        eps, B = self.eps, self.B
        self.tda = tda
        self.se = SelfEnergySolver(eps, df_coeff=B, eta=0.0)
        # eh channels: the singlet (always RPA) and the triplet in each
        # treatment asked for, with their phonons; only those are built, so
        # an unstable TDHF triplet refuses 'rpa' alone
        self.eh = {}
        treats = ('rpa',) if tda else tuple(dict.fromkeys(eh_triplets))
        for spin, treat in [('singlet', 'rpa')] + [('triplet', t) for t in treats]:
            if tda:
                A, _ = LinearResponseSolver(eps, coeff_df=B).build_casida_matrices(
                    O, lBSE=True, triplet=(spin == 'triplet'))
                n = A.shape[0]
                omega, X = la.eigh(A)
                ch = {'T': np.zeros((n, n)), 'dH': np.zeros((n, n)),
                      'dN': np.zeros((n, n)), 'omega': omega, 'X': X,
                      'Y': np.zeros((n, n))}
            else:
                ch = FR._eh_channel(eps, O, spin, B, None, 'phonon', treat)
            self.eh[(spin, treat)] = ch
        self._ops = {}
        self._resolvents = {}
        self._poles = {}            # per-orbital (residues, poles) of the eh and Sigma(2) terms
        self._pp_channels = None
        if not ladders:
            return
        # pp channel: Faddeev-ADC(3) with both eh pairs off, pp/hh ladders RPA
        # (TDA: the bare ladders, no amplitudes)
        if tda:
            base = FR.first_order_channels(eps, O, B_aa=B)
            for key in ('pp', 'hh'):
                for spin in ('singlet', 'triplet'):
                    base[key][spin]['T'] = np.zeros_like(base[key][spin]['T'])
            base['eh']['triplet'] = FR._eh_channel(eps, O, 'triplet', B, None,
                                                   'riccati', 'off')
        else:
            pp, hh = FR.ladder_channels(eps, O, B, None)
            base = {'eh': {'triplet': FR._eh_channel(eps, O, 'triplet', B, None,
                                                     'riccati', 'off')},
                    'pp': pp, 'hh': hh}
        base['eh']['singlet'] = FR._eh_channel(eps, O, 'singlet', B, None,
                                               'riccati', 'off')
        self._pp_channels = base

    # ---- the particle-hole channel, ordered single pair ---------------------

    def sigma_eh(self, p, w, eh_triplet='rpa'):
        """(Sigma_eh_pp(w), dSigma/dw) of the ordered single eh pair."""
        return _pole_sum(*self.eh_poles(p, eh_triplet), w)

    def eh_poles(self, p, eh_triplet='rpa'):
        """(residues, poles) of the eh term's Sigma_pp: PSD2 on the singlet
        and the eh_triplet triplet phonons, cached per orbital."""
        key = ('eh', p, eh_triplet)
        if key not in self._poles:
            S = self.eh[('singlet', 'rpa')]
            T = self.eh[('triplet', eh_triplet)]
            se, O = self.se, self.nocc
            self._poles[key] = se.self_energy_evaluator(
                p, O, S['omega'], se.get_chi_a(O, S['X'], S['Y'], p_state=p),
                se.get_chi_b_vertex(O, S['X'], S['Y'], p_state=p),
                T['omega'], se.get_chi_b_vertex(O, T['X'], T['Y'], p_state=p),
                vertex_mode='PSD2').poles()
        return self._poles[key]

    def sigma2_poles(self, p):
        """(residues, poles) of Sigma(2)_pp, cached per orbital."""
        key = ('sigma2', p)
        if key not in self._poles:
            res_D, res_X, poles = sigma2_poles(self.eps, self.nocc, p, self.B)
            self._poles[key] = (res_D + res_X, poles)
        return self._poles[key]

    def _sigma2(self, p, w):
        return _pole_sum(*self.sigma2_poles(p), w)

    # ---- operator-based terms (the pp pair alone, or full Faddeev) --------

    def _operator(self, key):
        """key 'pp', or ('faddeev', eh_triplet)."""
        if self._pp_channels is None:
            raise ValueError("built with ladders=False: no pp/hh ladder channels")
        if key not in self._ops:
            if key == 'pp':
                ch = self._pp_channels
            else:
                # = FR.pair_channels(route='riccati') up to round-off (the eh
                # channels here come from the phonons), reusing the ladders
                ch = {'eh': {'singlet': self.eh[('singlet', 'rpa')],
                             'triplet': self.eh[('triplet', key[1])]},
                      'pp': self._pp_channels['pp'], 'hh': self._pp_channels['hh']}
            aop, diag, d = FR.build_operator(self.s, self.nocc, None, channels=ch)
            self._ops[key] = (aop, d['nH'])
        return self._ops[key]

    def sigma_operator(self, key, p, w):
        """(Sigma_pp(w), dSigma/dw) = u^T (w - H_sat)^{-1} u for the Faddeev
        operator with only the pp/hh pair (key='pp') or all channels
        (key=('faddeev', eh_triplet)); one Lanczos per (key, p) serves every w."""
        rkey = (key, p)
        if rkey not in self._resolvents:
            aop, nH = self._operator(key)
            norb = self.norb
            e = np.zeros(nH)
            e[p] = 1.0
            u = aop(e)[norb:]
            zero = np.zeros(norb)
            self._resolvents[rkey] = _Resolvent(
                lambda x: aop(np.concatenate([zero, x]))[norb:], u)
        return self._resolvents[rkey](w)

    def sigma_flex(self, p, w, eh_triplet='rpa'):
        """(Sigma_FLEX_pp(w), dSigma/dw)."""
        se, dse = self.sigma_eh(p, w, eh_triplet)
        sp, dsp = self.sigma_operator('pp', p, w)
        s2, ds2 = self._sigma2(p, w)
        return se + sp - 2 * s2, dse + dsp - 2 * ds2

    def sigma_poles(self, p, which='flex', eh_triplet='rpa'):
        """(residues, poles) of Sigma_pp, for scanning the QP equation for all
        its roots: exact for the eh and Sigma(2) terms, and for the operator
        term (pp pair, or the full Faddeev operator) the Ritz representation of
        its Lanczos run as far as it has been extended, i.e. converged at the
        energies evaluated so far (start one at the HF eigenvalue first)."""
        if which == 'flex':
            key = 'pp'
            re, pe = self.eh_poles(p, eh_triplet)
            r2, p2 = self.sigma2_poles(p)
        else:
            key = ('faddeev', eh_triplet)
        self.sigma_operator(key, p, self.eps[p])
        th, wt = self._resolvents[(key, p)].ritz()
        if which == 'flex':
            return np.concatenate([re, wt, -2 * r2]), np.concatenate([pe, th, p2])
        return wt, th

    def solve_qp(self, p, which='flex', static=0.0, eh_triplet='rpa',
                 tol=QP_NEWTON_TOL, max_iter=QP_NEWTON_MAX_ITER, w0=None):
        """Diagonal QP w = eps_p + static + Sigma_pp(w) by Newton, from w0
        (default eps_p + static); which='flex' or 'faddeev'. Returns
        {'e', 'Z', 'converged', 'niter'}."""
        if which == 'flex':
            sig = lambda q, x: self.sigma_flex(q, x, eh_triplet)
        else:
            sig = lambda q, x: self.sigma_operator(('faddeev', eh_triplet), q, x)
        w = self.eps[p] + static if w0 is None else w0
        for it in range(1, max_iter + 1):
            s, ds = sig(p, w)
            f = w - self.eps[p] - static - s
            step = f / (1.0 - ds)
            w -= step
            if abs(step) < tol:
                s, ds = sig(p, w)
                return {'e': w, 'Z': 1.0 / (1.0 - ds), 'converged': True,
                        'niter': it}
        return {'e': w, 'Z': 1.0 / (1.0 - ds), 'converged': False, 'niter': max_iter}
