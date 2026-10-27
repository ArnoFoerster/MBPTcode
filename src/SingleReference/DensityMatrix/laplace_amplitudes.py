"""Moller-Plesset CI-coefficient amplitudes c = N / D held as their numerator
terms and a Laplace quadrature of the denominator, never formed.

D = sum_occ eps_i - sum_vir eps_a < 0 over the amplitude's external indices,
and with the minimax grid of 1/x on x = -D in [rank*gap_min, rank*gap_max],

    1/D = -sum_tau w_tau prod_occ e^{eps_i t_tau} prod_vir e^{-eps_a t_tau},

so c = -sum_terms (numerator term) x sum_tau w_tau (one dressing row per
external index): inside einsum (cached_einsum.factorized_einsum) every term
is one contraction of the numerator's factors, the (ntau, n) dressings and the
weights over a shared quadrature letter, and the consumer, so an amplitude
whose connected part has no product form (the third-order quadruples T4^(3))
never occupies O(no^4 nv^4). The quadrature is numerical and converges with
ntau to the exact amplitude.
"""
import numpy as np

from src.Base.utils.grids.grids import minimax_time_grid
from src.SingleReference.CC.cached_einsum import FactorizedOperand


def denominator_grid(eps_occ, eps_vir, rank, ntau):
    """(tau, w) of the minimax fit of 1/x over the range of -D_rank."""
    gap_min = float(np.min(eps_vir) - np.max(eps_occ))
    gap_max = float(np.max(eps_vir) - np.min(eps_occ))
    return minimax_time_grid(ntau, rank * gap_min, rank * gap_max)


class LaplaceAmplitude(FactorizedOperand):
    """coef * (sum of `terms`) / D over rank vir then rank occ external axes
    (or a transposed view of it, e.g. the lambda twin mp_amplitudes.to_l forms).

    terms: [(coef, ((subscript, array or FactorizedOperand), ...))] whose
    external letters are `external` (vir letters then occ letters, in the
    canonical axis order) and every other letter summed -- the generated
    numerator term tables (pdaggerq_interface.raw_terms_to_term_list)."""

    def __init__(self, terms, external, eps_occ, eps_vir, tau, w, coef=1.0, axes=None):
        self.terms = terms
        self.external = tuple(external)
        rank = len(self.external) // 2
        self.eps_occ, self.eps_vir = np.asarray(eps_occ), np.asarray(eps_vir)
        self.tau, self.w = tau, w
        self.coef = coef
        self.axes = tuple(axes) if axes is not None else tuple(range(2 * rank))
        canonical = (len(self.eps_vir),) * rank + (len(self.eps_occ),) * rank
        self.shape = tuple(canonical[a] for a in self.axes)
        ext = set(self.external)
        self._dummies = [sorted({c for sub, _ in pieces for c in sub} - ext) for _, pieces in terms]
        # one more fresh letter: the quadrature point, summed inside the contraction
        self.fresh_letters = max((len(d) for d in self._dummies), default=0) + 1
        self._dress_vir = np.exp(-np.outer(tau, self.eps_vir))
        self._dress_occ = np.exp(np.outer(tau, self.eps_occ))
        self._weights = -np.asarray(w, dtype=float)

    def _view(self, axes, coef):
        return LaplaceAmplitude(self.terms, self.external, self.eps_occ, self.eps_vir,
                                self.tau, self.w, coef, axes)

    def _canonical_terms(self, letters, fresh):
        rank = len(self.external) // 2
        t = fresh[0]
        dress = (tuple((t + letters[p], self._dress_vir) for p in range(rank))
                 + tuple((t + letters[rank + p], self._dress_occ) for p in range(rank))
                 + ((t, self._weights),))
        out = []
        for (c, pieces), dummies in zip(self.terms, self._dummies):
            name = dict(zip(self.external, letters))
            name.update(zip(dummies, fresh[1:]))
            subs = [(''.join(name[ch] for ch in sub), arr) for sub, arr in pieces]
            out.append((c, tuple(subs) + dress))
        return out
