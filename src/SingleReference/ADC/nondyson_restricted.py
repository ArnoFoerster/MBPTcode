"""Non-Dyson IP/EA-ADC(n), n <= 5: restricted (closed-shell spin blocks) and
matrix-free, through isr_operator.OperatorISR. The construction is the one of
nondyson_isr (Schirmer, Trofimov and Stelter, J. Chem. Phys. 109, 4734 (1998)),
whose dense spin-orbital NonDysonISR is its reference through n = 4.

Raw blocks <E_I^dag H E_J> and <E_I^dag E_J> come from the generated apply
twins adc{N}_nd{m}_k{i}k{j}_apply_restricted_blocks
(sigma_codegen.generate_nondyson_pair_apply_restricted) on the alpha-sector
configurations of each class (the Dyson solver's kconfigs, S_z = -1/2 for IP
and +1/2 for EA), evaluated at the amplitude orders of u_order_terms /
s_order_terms with the restricted solver's amplitudes. The E0 series times the
lower-order overlaps is subtracted, so the eigenvalues are E(N -/+ 1) - E0; the
ground-state norm drops out, every class being Loewdin-normalized.

Ground state: through ADC(4) the amplitudes of Dyson ADC(5) and E^(4). ADC(5)
reads the whole order-4 ground state -- the 1h-1h block's f- and v-dressed
structures at amplitude orders (2, 3) and (1, 4): T4^(3), T2^(4), T3^(4), T1^(5)
-- and E^(5), all from the restricted chain (amps_order(5)); T4^(3) is the one
rank-4 amplitude without a product form, materialized (O(no^4 nv^4)) or, with
the solver's quadruples_ntau, Laplace-fused and never formed. The raw blocks
within ISR_DENSE_BLOCK_ELEMENTS (the 1h and 2h1p/2p1h classes) are held as
arrays, those classes' effective states precomputed.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator

from src.Base.numerics.constants import ISR_DENSE_BLOCK_ELEMENTS
from src.SingleReference.ADC.adc_order_rule import (
    kmax, u_order_terms, s_order_terms)
from src.SingleReference.ADC.batched_apply import restricted_class_matmat
from src.SingleReference.ADC.generated_modules import generated
from src.SingleReference.ADC.isr_operator import OperatorISR
from src.SingleReference.ADC.nondyson_isr import NonDysonISR
from src.SingleReference.ADC.precursor_classes import CLASS_TAGS, nondyson_apply_module_name
from src.SingleReference.ADC.sigma_driver import restricted_class_matvec
from src.SingleReference.DensityMatrix.mpn_density_driver_restricted import \
    MPnDensityDriverRestricted

# highest order of the restricted non-Dyson construction: ADC(6) would read the
# order-5 ground state, which the generators do not emit
MAX_NONDYSON_RESTRICTED_ORDER = 5


def _combo(shape, parts):
    """sum_k coef_k * map_k as one linear map (matvec, matmat and their transposes)."""
    def mv(x):
        return sum(c * A.matvec(x) for c, A in parts)

    def rmv(y):
        return sum(c * A.rmatvec(y) for c, A in parts)

    def mm(X):
        return sum(c * A.matmat(X) for c, A in parts)

    def rmm(Y):
        return sum(c * A.rmatmat(Y) for c, A in parts)
    return LinearOperator(shape, matvec=mv, rmatvec=rmv, matmat=mm, rmatmat=rmm, dtype=float)


def _identity(d):
    return LinearOperator((d, d), matvec=lambda x: x, rmatvec=lambda y: y,
                          matmat=lambda X: X, rmatmat=lambda Y: Y, dtype=float)


class NonDysonRestricted:
    """Non-Dyson ISR of one sector ('IP' or 'EA') over a restricted solver
    (BuildSuperMatrixADCnRestricted with dense g): raw block series, the ADC(n)
    matrix as a LinearOperator and its zeroth-order diagonal."""

    def __init__(self, rsolver, sector):
        if sector not in ('IP', 'EA'):
            raise ValueError(f"sector must be 'IP' or 'EA', got {sector!r}")
        if rsolver.df:
            raise ValueError("non-Dyson ADC(n) reads dense g: the restricted non-Dyson apply "
                             "modules have no DF twins; build the solver from g_aaaa/g_abab/"
                             "g_bbbb")
        self.r = rsolver
        self.sector = sector
        self._mods = {}

    # ----- classes and configurations --------------------------------------
    def configs(self, k):
        return self.r.kconfigs(self.sector, k)

    def dim(self, k):
        return len(self.configs(k)['K'])

    def classes(self, n):
        """Classes present at ADC(n): 2k <= n and a nonempty configuration space."""
        return [k for k in range(kmax(n) + 1) if self.dim(k) > 0]

    def diagonal(self, n):
        """Zeroth-order E(N -/+ 1) - E0 of every configuration, class by class."""
        sign = -1.0 if self.sector == 'IP' else 1.0     # kconfigs' K is the Dyson pole
        return np.concatenate([sign * self.configs(k)['K'] for k in self.classes(n)])

    # ----- generated structures --------------------------------------------
    def _module(self, order, k_i, k_j):
        key = (order, k_i, k_j)
        if key not in self._mods:
            self._mods[key] = generated(nondyson_apply_module_name(order, k_i, k_j)[:-3])
        return self._mods[key]

    def _resolve(self, kind, name, order, k_i, k_j):
        """(module, forward prefix, transpose prefix) of structure `name` at its
        home order (searched from `order` down to the pair's leading order, every
        module in that range must exist), or None when pdaggerq pruned it."""
        tag_i, tag_j = CLASS_TAGS[k_i][self.sector], CLASS_TAGS[k_j][self.sector]
        base = 'nd' if kind == 'u' else 'nds'
        for o in range(order, NonDysonISR.leading_order(k_i, k_j) - 1, -1):
            mod = self._module(o, k_i, k_j)
            fwd = f'apply_{base}{o}_{tag_i}_{tag_j}_{name}_'
            if any(a.startswith(fwd) for a in dir(mod)):
                return mod, fwd, f'apply_{base}t{o}_{tag_i}_{tag_j}_{name}_'
        return None

    def _amps(self, n):
        """Amplitudes through order n (the f-dressed structures at order n read
        t1^(n))."""
        return self.r.amps_order(max_order=max(n, 2))

    def _raw(self, kind, k_i, k_j, m, n):
        """The raw H^(m) ('u') or S^(m) ('s') block of (k_i, k_j) as a linear map
        (class k_j -> class k_i), or None when every structure is pruned."""
        A = self._amps(n)
        base = self.r.rargs
        no, nv = self.r.nocc, self.r.nvir
        cfg_i, cfg_j = self.configs(k_i), self.configs(k_j)
        terms = u_order_terms(m) if kind == 'u' else s_order_terms(m)
        pieces, missing = [], []
        for t in terms:
            name = t[0]
            j, a, k, b = (t[1], t[4], t[3], t[5]) if kind == 'u' else (t[1], t[3], t[2], t[4])
            res = self._resolve(kind, name, m, k_i, k_j)
            if res is None:
                continue
            pool = dict(base)
            for lead, rank, order in (('L', j, a), ('T', k, b)):
                if rank:
                    leaves = A.get(f'{lead}{rank}_{order}')
                    if leaves is None:
                        missing.append(f"{name}(l{j}^{a},t{k}^{b})")
                        pool = None
                        break
                    pool.update(leaves)
            if pool is not None:
                pieces.append((res, pool))
        if missing:
            raise RuntimeError(
                f"non-Dyson {self.sector} block ({k_i},{k_j}) order {m}: no restricted "
                f"amplitude for {sorted(set(missing))}")
        if not pieces:
            return None

        def mv(x):
            return sum(restricted_class_matvec(mod, fwd, x, cfg_i, pool, no, nv, cfg_ket=cfg_j)
                       for (mod, fwd, _), pool in pieces)

        def rmv(y):
            return sum(restricted_class_matvec(mod, tr, y, cfg_j, pool, no, nv, cfg_ket=cfg_i)
                       for (mod, _, tr), pool in pieces)

        # several columns per generated call (batched_apply): densifying a block and the
        # effective-state recursion on matrices apply it to many columns at once
        def mm(X):
            return sum(restricted_class_matmat(mod, fwd, X, cfg_i, pool, no, nv, cfg_ket=cfg_j)
                       for (mod, fwd, _), pool in pieces)

        def rmm(Y):
            return sum(restricted_class_matmat(mod, tr, Y, cfg_j, pool, no, nv, cfg_ket=cfg_i)
                       for (mod, _, tr), pool in pieces)
        return LinearOperator((self.dim(k_i), self.dim(k_j)), matvec=mv, rmatvec=rmv,
                              matmat=mm, rmatmat=rmm, dtype=float)

    # ----- ground-state energy series ---------------------------------------
    def mp_energy(self, c):
        """E^(c) of the Moller-Plesset series: E^(0) = sum of the occupied spin-
        orbital energies, E^(1) = -1/2 sum_kl <kl||kl>, E^(c >= 2) the MP(c)
        correlation energy from t2^(c-1)."""
        r, o = self.r, slice(0, self.r.nocc)
        if c == 0:
            return 2.0 * float(np.sum(r.eps[o]))
        if c == 1:
            return -float(np.einsum('klkl->', r.g_aaaa[o, o, o, o])
                          + np.einsum('klkl->', r.g_abab[o, o, o, o]))
        mpn = MPnDensityDriverRestricted(np.diag(r.eps), r.g_aaaa, r.g_abab, r.g_bbbb, r.nocc)
        t2 = self._amps(c)[f'T2_{c - 1}']
        return float(mpn.compute_E(t2['t2_aaaa'], t2['t2_abab']))

    # ----- the ADC(n) matrix ------------------------------------------------
    def raw_series(self, n, extra=None):
        """H = <E^dag (H - E0) E>, S = <E^dag E> as {(k_i, k_j): {m: map}} for
        k_i <= k_j through each block's carried order, the class dims and the
        carried orders."""
        if not 2 <= n <= MAX_NONDYSON_RESTRICTED_ORDER:
            raise ValueError(f"non-Dyson ADC({n}): the restricted construction covers "
                             f"ADC(2) to ADC({MAX_NONDYSON_RESTRICTED_ORDER})")
        cls = self.classes(n)
        dims = {k: self.dim(k) for k in cls}
        top = {}
        E = {c: self.mp_energy(c) for c in range(n + 1)}
        H, S = {}, {}
        for i, k_i in enumerate(cls):
            for k_j in cls[i:]:
                t = NonDysonISR.top_order(n, k_i, k_j, extra)
                if t < 0:
                    continue
                top[(k_i, k_j)] = t
                shape = (dims[k_i], dims[k_j])
                Ss = {0: _identity(dims[k_i])} if k_i == k_j else {}
                for m in range(1, t + 1):
                    s = self._raw('s', k_i, k_j, m, n)
                    if s is not None:
                        Ss[m] = s
                Hs = {}
                for m in range(t + 1):
                    parts = []
                    h = self._raw('u', k_i, k_j, m, n)
                    if h is not None:
                        parts.append((1.0, h))
                    for c in range(m + 1):
                        if (m - c) in Ss:
                            parts.append((-E[c], Ss[m - c]))
                    if parts:
                        Hs[m] = _combo(shape, parts)
                H[(k_i, k_j)], S[(k_i, k_j)] = Hs, Ss
        return H, S, dims, top

    def operator(self, n, extra=None, dense_max=ISR_DENSE_BLOCK_ELEMENTS):
        """(LinearOperator, zeroth-order diagonal) of non-Dyson ADC(n); `extra` =
        {(k_i, k_j): order} raises a block's carried order (ADC(2)-x: {(1, 1): 1});
        dense_max as OperatorISR's (0: every block matrix-free)."""
        H, S, dims, top = self.raw_series(n, extra)
        return (OperatorISR(H, S, dims, top, dense_max=dense_max).linear_operator(),
                self.diagonal(n))
