"""Numerical spin adaptation for the spin-orbital ADC(3) solver:
construct the sparse isometry T from the spin-orbital configuration basis
({orbital p} + 2h1p + 2p1h + 3h2p + 3p2h, as enumerated by ADCSolver's
_configs_* methods for an RHF/closed-shell reference) onto its DOUBLET
(S=1/2), single-Sz-sector subspace, i.e. the genuine CSF basis.

Why this exists: the supermatrix is spin-free, so it commutes with S^2 and
Sz; physical Dyson poles live entirely in the doublet sector reached by
adding/removing one alpha electron. The spin-orbital basis carries ~4x
redundancy per dimension for a closed-shell reference (duplicate Sz=-1/2
sector, quartet/sextet CSFs that never couple to the orbital block); T
removes it exactly:

    H_csf = T^T H_spin T   (same spectrum, each pole ONCE,
                            dims ~4x smaller, coupling matrices ~16x)

**No CSF Hamiltonian formula is derived here** -- a hand derivation of the
CSF blocks carries real fermion-sign risk. Instead T is built numerically:
exact bitstring second quantization, S^2 = S-S+ + Sz(Sz+1) applied literally
to determinants, and doublet CSFs extracted as S^2 eigenvectors within each
tiny spatial-pattern block. Every transformed block then inherits the
spin-orbital implementation's own validation, checked end-to-end against
the subspace-invariance identity ||H T - T (T^T H T)|| ~ 0 (see tests).

Convention notes (locked in by the invariance test, not by trust):
- interleaved spin orbitals, alpha=even (Sz=+1/2), beta=odd -- matching
  ADCSolverRestricted._get_spin_orbital_solver's np.repeat embedding.
- config -> Fock-state map (operator strings applied right-to-left to
  |HF>): 2h1p (i<j, a): a+_a a_j a_i; 2p1h (i, a<b): a+_a a+_b a_i;
  3h2p (i<j<k, a<b): a+_a a+_b a_k a_j a_i; 3p2h (i<j, a<b<c):
  a+_a a+_b a+_c a_j a_i.
- kept sector: states reachable from |HF> by removing (N-1 classes,
  Sz=-1/2) or adding (N+1 classes, Sz=+1/2) one ALPHA electron; the
  orbital block keeps the alpha rows (T_orb[2q, q] = 1).
"""
import numpy as np
from scipy import sparse

from src.SingleReference.ADC import base  # module import: base.py imports this file


# =====================================================================
# exact bitstring second quantization
# =====================================================================

def apply_ops(det, ops):
    """Apply a second-quantized operator string to determinant `det` (an
    int bitmask over spin orbitals; bit m set = spin orbital m occupied).

    ops is the operator string written LEFT-TO-RIGHT as on paper, each
    entry (mode, is_creation); the rightmost factor acts first. Returns
    (new_det, sign) with sign in {-1, +1}, or (None, 0) if annihilated.
    Signs are the exact Jordan-Wigner parities (-1)^{#occupied below mode}
    -- no Slater-Condon shortcuts.
    """
    sign = 1
    for mode, create in reversed(ops):
        bit = 1 << mode
        parity = bin(det & (bit - 1)).count('1') & 1
        if create:
            if det & bit:
                return None, 0
            if parity:
                sign = -sign
            det |= bit
        else:
            if not det & bit:
                return None, 0
            if parity:
                sign = -sign
            det &= ~bit
    return det, sign


def det_sz(det):
    """Total Sz of a determinant (alpha=even modes=+1/2, beta=odd=-1/2)."""
    sz = 0.0
    m = 0
    while det:
        if det & 1:
            sz += 0.5 if m % 2 == 0 else -0.5
        det >>= 1
        m += 1
    return sz


def _apply_S2(det, norb_spatial):
    """S^2 |det> = S-S+|det> + Sz(Sz+1)|det>, exactly, as a dict
    {det': coefficient}. S+ = sum_p a+_{2p} a_{2p+1} (alpha=even)."""
    out = {}
    sz = det_sz(det)
    out[det] = sz * (sz + 1.0)
    plus = {}
    for p in range(norb_spatial):
        d1, s1 = apply_ops(det, [(2 * p, True), (2 * p + 1, False)])
        if s1:
            plus[d1] = plus.get(d1, 0.0) + s1
    for d1, c1 in plus.items():
        for p in range(norb_spatial):
            d2, s2 = apply_ops(d1, [(2 * p + 1, True), (2 * p, False)])
            if s2:
                out[d2] = out.get(d2, 0.0) + c1 * s2
    return out


# =====================================================================
# config class -> Fock states
# =====================================================================

_S2_DOUBLET = 0.75
_S2_TOL = 1e-8

# per class: (target Fock Sz of the kept sector, ops builder)
def _class_ops(cls, cfg):
    if cls == '2h1p':
        i, j, a = cfg
        return [(a, True), (j, False), (i, False)]
    if cls == '2p1h':
        i, a, b = cfg
        return [(a, True), (b, True), (i, False)]
    if cls == '3h2p':
        i, j, k, a, b = cfg
        return [(a, True), (b, True), (k, False), (j, False), (i, False)]
    if cls == '3p2h':
        i, j, a, b, c = cfg
        return [(a, True), (b, True), (c, True), (j, False), (i, False)]
    raise ValueError(cls)


_CLASS_TARGET_SZ = {'2h1p': -0.5, '3h2p': -0.5, '2p1h': +0.5, '3p2h': +0.5}


def _class_configs(solver, nocc, cls):
    """The solver's config list for `cls` as an (ncfg, k) int array of spin
    orbital indices, in the solver's own ordering."""
    if cls == '2h1p':
        return np.stack(solver._configs_2h1p(nocc), axis=1)
    if cls == '2p1h':
        return np.stack(solver._configs_2p1h(nocc), axis=1)
    if cls == '3h2p':
        return np.stack(solver._configs_3h2p(nocc), axis=1)
    if cls == '3p2h':
        return np.stack(solver._configs_3p2h(nocc), axis=1)
    raise ValueError(cls)


def _class_signs_closed_form(cls, cfgs, nocc):
    """Vectorized Jordan-Wigner sign of _class_ops(cls, cfg) applied to
    |HF>, in closed form -- e.g. for 2h1p a+_a a_j a_i (right-to-left on
    the 0..nocc-1-filled determinant): a_i contributes (-1)^i, a_j then
    (-1)^(j-1) (the i-hole sits below j), a+_a lands above nocc-2 electrons.
    Derived per class the same way and validated exhaustively against the
    exact bitstring apply_ops for every config on real systems in
    tests/test_spin_adapt.py -- the gate for trusting this hand parity
    counting."""
    if cls == '2h1p':      # (-1)^(i + j - 1 + nocc)
        e = cfgs[:, 0] + cfgs[:, 1] + 1 + nocc
    elif cls == '2p1h':    # (-1)^i
        e = cfgs[:, 0]
    elif cls == '3h2p':    # (-1)^(i + j + k + 1)
        e = cfgs[:, 0] + cfgs[:, 1] + cfgs[:, 2] + 1
    elif cls == '3p2h':    # (-1)^(i + j + nocc + 1)
        e = cfgs[:, 0] + cfgs[:, 1] + nocc + 1
    else:
        raise ValueError(cls)
    return 1 - 2 * (e % 2).astype(np.int8)


_CLASS_HOLE_COLS = {'2h1p': (0, 1), '2p1h': (0,), '3h2p': (0, 1, 2), '3p2h': (0, 1)}
_CLASS_PART_COLS = {'2h1p': (2,), '2p1h': (1, 2), '3h2p': (3, 4), '3p2h': (2, 3, 4)}


def build_class_T(solver, nocc, cls, memoize=True):
    """Sparse doublet-CSF isometry for one config class.

    Returns (T, K_csf, group_sig_stats):
      T: (n_spin_cfg, n_csf) scipy CSR; columns are orthonormal doublet
         CSFs (S^2 eigenvalue 1/2(1/2+1)=0.75 within the class's kept Sz
         sector), expressed over the solver's spin-orbital config indices.
      K_csf: (n_csf,) 0th-order (orbital-energy-sum) diagonal per CSF --
         constant within each spatial pattern, so exactly diagonal in the
         CSF basis too.

    Doublet extraction is a dense eigh of the exact S^2 matrix in each
    spatial-pattern/Sz group (<=10 determinants); the S^2 matrix (and its
    doublet eigenvectors) depend only on the group's spin-pattern/sign
    signature, so the eigh is memoized by that signature -- verified
    identical to the unmemoized path in tests/test_spin_adapt.py.
    """
    norb_spatial = solver.norb // 2
    hf = (1 << nocc) - 1
    cfgs = _class_configs(solver, nocc, cls)
    ncfg = len(cfgs)
    eps = solver.eps
    target_sz = _CLASS_TARGET_SZ[cls]

    # Vectorized per-config bookkeeping (a Python loop applying bitstring
    # operators to every config was the dominant T-build cost): signs in
    # closed form (gated against apply_ops -- exhaustively in the tests AND
    # at runtime for every uncached signature below), Sz and the
    # spatial-pattern group key straight from the config index arrays,
    # grouping via lexsort.
    signs = _class_signs_closed_form(cls, cfgs, nocc)
    hole_cols = list(_CLASS_HOLE_COLS[cls])
    part_cols = list(_CLASS_PART_COLS[cls])
    s_half = 0.5 - (cfgs % 2)                       # spin of each index (alpha=even)
    sz_state = (-s_half[:, hole_cols].sum(axis=1)
                + s_half[:, part_cols].sum(axis=1))
    kept = np.where(np.abs(sz_state - target_sz) < 1e-12)[0]
    # spatial pattern columns (config index columns are sorted within the
    # hole/particle sections, so their spatial values are already sorted)
    spat_cols = [cfgs[:, c] // 2 for c in hole_cols + part_cols]
    gorder = kept[np.lexsort((kept,) + tuple(col[kept] for col in reversed(spat_cols)))]
    key_mat = np.stack([col[gorder] for col in spat_cols], axis=1)
    newgrp = np.ones(len(gorder), dtype=bool)
    if len(gorder) > 1:
        newgrp[1:] = np.any(key_mat[1:] != key_mat[:-1], axis=1)
    starts = np.where(newgrp)[0]
    ends = np.append(starts[1:], len(gorder))
    groups_list = [gorder[s:e] for s, e in zip(starts, ends)]
    groups_list.sort(key=lambda m: m[0])   # legacy order: by smallest member

    # K (orbital-energy sum) per config -- same for all members of a group
    if cls == '2h1p':
        K_cfg = eps[cfgs[:, 0]] + eps[cfgs[:, 1]] - eps[cfgs[:, 2]]
    elif cls == '2p1h':
        K_cfg = eps[cfgs[:, 1]] + eps[cfgs[:, 2]] - eps[cfgs[:, 0]]
    elif cls == '3h2p':
        K_cfg = (eps[cfgs[:, 0]] + eps[cfgs[:, 1]] + eps[cfgs[:, 2]]
                 - eps[cfgs[:, 3]] - eps[cfgs[:, 4]])
    else:  # 3p2h
        K_cfg = (eps[cfgs[:, 2]] + eps[cfgs[:, 3]] + eps[cfgs[:, 4]]
                 - eps[cfgs[:, 0]] - eps[cfgs[:, 1]])

    sig_cache = {}
    rows, cols, vals, K_csf = [], [], [], []
    ncsf = 0
    for members in groups_list:
        # signature: per member, its (spin pattern, spatial-slot identity
        # pattern, sign), in a canonical (sorted) member order. The spatial
        # pattern (which slots are the SAME spatial orbital -- e.g. a
        # doubly-annihilated hole pair vs two distinct holes) changes S^2
        # even at identical spin patterns, so it must be part of the key.
        uniq = sorted({int(x) // 2 for c in members for x in cfgs[c]})
        rank = {u: r for r, u in enumerate(uniq)}
        spins = [tuple(int(x) % 2 for x in cfgs[c]) for c in members]
        spat = [tuple(rank[int(x) // 2] for x in cfgs[c]) for c in members]
        morder = sorted(range(len(members)), key=lambda r: (spins[r], spat[r]))
        sig = tuple((spins[r], spat[r], int(signs[members[r]])) for r in morder)
        sub = [members[r] for r in morder]

        if memoize and sig in sig_cache:
            vecs = sig_cache[sig]
        else:
            # determinants are only needed here (uncached signatures --
            # a handful per class); built via the exact bitstring operators,
            # which also re-checks the closed-form signs at runtime
            dets_sub = []
            for c in sub:
                d, s_chk = apply_ops(hf, _class_ops(cls, tuple(int(x) for x in cfgs[c])))
                assert s_chk == signs[c], \
                    "closed-form Jordan-Wigner sign disagrees with apply_ops"
                dets_sub.append(d)
            index = {d: r for r, d in enumerate(dets_sub)}
            n = len(sub)
            S2 = np.zeros((n, n))
            for s_col in range(n):
                for d2, coef in _apply_S2(dets_sub[s_col], norb_spatial).items():
                    r = index.get(d2)
                    if r is not None:
                        # states are sign_c * |det_c>
                        S2[r, s_col] += signs[sub[r]] * coef * signs[sub[s_col]]
            assert np.max(np.abs(S2 - S2.T)) < 1e-10, "S^2 not symmetric"
            w, v = np.linalg.eigh(S2)
            keep = np.where(np.abs(w - _S2_DOUBLET) < _S2_TOL)[0]
            # every eigenvalue must be a valid S(S+1)
            valid = [0.75, 3.75, 8.75, 15.75]
            assert all(min(abs(x - t) for t in valid) < 1e-8 for x in w), \
                f"unexpected S^2 eigenvalue in {w}"
            vecs = v[:, keep]
            # deterministic sign fix: largest-|.| component positive
            for q in range(vecs.shape[1]):
                imax = np.argmax(np.abs(vecs[:, q]))
                if vecs[imax, q] < 0:
                    vecs[:, q] = -vecs[:, q]
            sig_cache[sig] = vecs

        for q in range(vecs.shape[1]):
            for r, c in enumerate(sub):
                if abs(vecs[r, q]) > 1e-14:
                    rows.append(c)
                    cols.append(ncsf)
                    vals.append(vecs[r, q])
            K_csf.append(K_cfg[sub[0]])
            ncsf += 1

    T = sparse.csr_matrix((vals, (rows, cols)), shape=(ncfg, ncsf))
    return T, np.asarray(K_csf)


def build_T(solver, nocc):
    """All five class isometries for a spin-orbital ADCSolver at the given
    (spin-orbital) nocc, plus the stacked SD-space and full-space versions.

    Returns a dict:
      'orb', '2h1p', '2p1h', '3h2p', '3p2h': per-class sparse T
      'K_3h2p', 'K_3p2h': CSF-basis 0th-order diagonals of the two
          downfolded sectors (exactly diagonal, see build_class_T)
      'SD': block-diag(T_orb, T_2h1p, T_2p1h) -- the (norb+n2h1p+n2p1h)
          x (norb_csf+n2h1p_csf+n2p1h_csf) isometry for the M_eff/(SD) space
      'full': block-diag of all five (for dense validation of the full
          supermatrix including the 3h2p/3p2h classes)
    """
    norb_spatial = solver.norb // 2
    T_orb = sparse.csr_matrix(
        (np.ones(norb_spatial), (2 * np.arange(norb_spatial), np.arange(norb_spatial))),
        shape=(solver.norb, norb_spatial))
    T_2h1p, K_2h1p = build_class_T(solver, nocc, '2h1p')
    T_2p1h, K_2p1h = build_class_T(solver, nocc, '2p1h')
    T_3h2p, K_3h2p = build_class_T(solver, nocc, '3h2p')
    T_3p2h, K_3p2h = build_class_T(solver, nocc, '3p2h')
    return {
        'orb': T_orb, '2h1p': T_2h1p, '2p1h': T_2p1h,
        '3h2p': T_3h2p, '3p2h': T_3p2h,
        'K_2h1p': K_2h1p, 'K_2p1h': K_2p1h,
        'K_3h2p': K_3h2p, 'K_3p2h': K_3p2h,
        'SD': sparse.block_diag([T_orb, T_2h1p, T_2p1h], format='csr'),
        'full': sparse.block_diag([T_orb, T_2h1p, T_2p1h, T_3h2p, T_3p2h],
                                   format='csr'),
    }


class _LazyT:
    """Lazy per-class view over build_T's dict interface: each class
    isometry is built on FIRST ACCESS (and cached), so an ADC(3)-only
    workflow through ADCSolverCSF (which needs only the orb/2h1p/2p1h
    pieces) never pays the much larger 3h2p/3p2h construction."""

    def __init__(self, solver, nocc):
        self._solver = solver
        self._nocc = nocc
        self._d = {}

    def __getitem__(self, key):
        d = self._d
        if key in d:
            return d[key]
        s, nocc = self._solver, self._nocc
        if key == 'orb':
            norb_sp = s.norb // 2
            d['orb'] = sparse.csr_matrix(
                (np.ones(norb_sp), (2 * np.arange(norb_sp), np.arange(norb_sp))),
                shape=(s.norb, norb_sp))
        elif key in ('2h1p', '2p1h', '3h2p', '3p2h'):
            T, K = build_class_T(s, nocc, key)
            d[key] = T
            d['K_' + key] = K
        elif key.startswith('K_'):
            self[key[2:]]
        elif key == 'SD':
            d['SD'] = sparse.block_diag(
                [self['orb'], self['2h1p'], self['2p1h']], format='csr')
        elif key == 'full':
            d['full'] = sparse.block_diag(
                [self['orb'], self['2h1p'], self['2p1h'],
                 self['3h2p'], self['3p2h']], format='csr')
        else:
            raise KeyError(key)
        return d[key]


# =====================================================================
# CSF-basis solver adapter
# =====================================================================

class ADCSolverCSF:
    """Spin-adapted (doublet-CSF-basis) front-end for a spin-orbital solver:
    exposes norb (SPATIAL), dimensions/dimensions_adc4 (CSF sizes, ~4x
    smaller per class), and build_supermatrix/solve/build_matrix_free_operator
    (ADC(3)) -- all in the CSF basis defined by build_T (every block is
    T^T (validated spin-orbital block) T, nothing is re-derived). Dense CSF
    matrices are assembled directly from ket/bra-restricted spin-orbital
    block builders; no (nH_spin)^2 intermediate is ever formed.

    nocc arguments are SPIN-orbital occupation counts (2*nocc_spatial);
    static_correction arguments are SPATIAL (norb, norb) matrices (embedded
    to spin internally); homo_index-style orbital indices are SPATIAL.
    """

    def __init__(self, spin_solver):
        assert spin_solver.norb % 2 == 0, "spin-orbital solver expected"
        self.spin = spin_solver
        self.norb = spin_solver.norb // 2
        self._cache = {}

    def _cached(self, name, nocc, compute_fn):
        key = (name, nocc)
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]

    def _T(self, nocc):
        """Lazy per-class isometry view (see _LazyT) -- classes are built on
        first access, so e.g. a plain-ADC(3) solve never constructs the
        3h2p/3p2h pieces."""
        return self._cached('T', nocc, lambda: _LazyT(self.spin, nocc))

    @staticmethod
    def _embed_sc(static_correction):
        if static_correction is None:
            return None
        return base.ADCSolverRestricted._embed_static_correction_spin(static_correction)

    # ---------------- dimensions ----------------

    def dimensions(self, nocc):
        T = self._T(nocc)
        n2h1p, n2p1h = T['2h1p'].shape[1], T['2p1h'].shape[1]
        return {'norb': self.norb, 'nocc': nocc // 2,
                'nvirt': self.norb - nocc // 2,
                'n2h1p': n2h1p, 'n2p1h': n2p1h,
                'nH': self.norb + n2h1p + n2p1h}


    def _segments(self, nocc):
        d = self.dimensions(nocc)
        off_2h1p = self.norb
        off_2p1h = off_2h1p + d['n2h1p']
        return off_2h1p, off_2p1h, d['nH']

    # ---------------- dense ADC(3) (seed/validation) ----------------

    def build_supermatrix(self, nocc, static_correction=None):
        """Dense CSF ADC(3) supermatrix (4x smaller per dimension than the
        spin-orbital one), assembled DIRECTLY in the CSF basis: F is the spatial
        diag(eps)+static_correction, the U blocks are alpha-row slices of
        the spin solver's cached (norb, n) U matrices sandwiched with T,
        K is exactly diagonal (build_class_T), and the first-order C blocks
        come from ket/bra-restricted spin block builders sandwiched in
        batches -- the full (nH_spin, nH_spin) spin-orbital matrix (10+ GB
        at aVTZ for a <1 GB CSF result) is never formed."""
        T = self._T(nocc)
        off1, off2, nH = self._segments(nocc)
        norb = self.norb
        H = np.zeros((nH, nH))

        H[:norb, :norb] = np.diag(self.spin.eps[0::2])
        if static_correction is not None:
            H[:norb, :norb] += static_correction

        # The spin-orbital solver never materializes a dense U_2h1p/U_2p1h,
        # and only the ALPHA rows are needed here anyway (RHF reference,
        # alpha=even per this module's own interleaved convention), so
        # reconstruct just those via apply_U_2h1p/apply_U_2p1h's forward
        # direction (unit z_p at each alpha orbital), norb (spatial) calls
        # total -- this is the dense/validation seed path, not the hot
        # matrix-free one.
        ing = self.spin._build_matrix_free_ingredients(nocc)
        iu, ju, au, bu = ing['iu'], ing['ju'], ing['au'], ing['bu']
        nvirt = ing['nvirt']
        alpha_orbs = np.arange(0, self.spin.norb, 2)
        zero_2h1p_full = np.zeros((nocc, nocc, nvirt))
        zero_2p1h_full = np.zeros((nocc, nvirt, nvirt))
        U2h_alpha = np.empty((len(alpha_orbs), ing['d']['n2h1p']))
        U2p_alpha = np.empty((len(alpha_orbs), ing['d']['n2p1h']))
        for row, p in enumerate(alpha_orbs):
            z_p = np.zeros(self.spin.norb)
            z_p[p] = 1.0
            dy_2h1p_full, _ = self.spin.apply_U_2h1p(nocc, z_p, zero_2h1p_full)
            dy_2p1h_full, _ = self.spin.apply_U_2p1h(nocc, z_p, zero_2p1h_full)
            U2h_alpha[row, :] = dy_2h1p_full[iu, ju, :].reshape(-1)
            U2p_alpha[row, :] = dy_2p1h_full[:, au, bu].reshape(-1)
        U2h = np.asarray((T['2h1p'].T @ U2h_alpha.T).T)   # (norb, n2h1p_csf)
        U2p = np.asarray((T['2p1h'].T @ U2p_alpha.T).T)   # (norb, n2p1h_csf)
        H[:norb, off1:off2] = U2h
        H[off1:off2, :norb] = U2h.T
        H[:norb, off2:nH] = U2p
        H[off2:nH, :norb] = U2p.T

        idx = np.arange(off1, off2)
        H[idx, idx] = T['K_2h1p']
        idx = np.arange(off2, nH)
        H[idx, idx] = T['K_2p1h']

        H[off1:off2, off1:off2] += self._C1st_csf(nocc, '2h1p')
        H[off2:nH, off2:nH] += self._C1st_csf(nocc, '2p1h')
        return H

    def solve(self, nocc, static_correction=None, threshold=5000):
        """Dense-diagonalization solve in the CSF basis; same return
        convention as ADCSolver.solve (each physical pole ONCE, not twice)."""
        H = self.build_supermatrix(nocc, static_correction=static_correction)
        eGF, Reigv = np.linalg.eigh(H)
        Z = np.sum(Reigv[:self.norb, :] ** 2, axis=0)
        order = np.argsort(eGF)
        return eGF[order], Z[order], Reigv[:, order]

    # ---------------- matrix-free ADC(3) ----------------

    def build_matrix_free_operator(self, nocc, static_correction=None):
        """CSF-basis matrix-free ADC(3) operator: lift through T_SD to the
        spin-orbital operator (cached ingredients there) and project back.
        diag uses the exact F/K CSF diagonals (K is exactly diagonal in the
        CSF basis, see build_class_T)."""
        T = self._T(nocc)
        T_SD = T['SD']
        aop_spin, _, _ = self.spin.build_matrix_free_operator(
            nocc, static_correction=self._embed_sc(static_correction))

        def aop(z):
            return T_SD.T @ aop_spin(T_SD @ z)

        F_diag = self.spin.eps[0::2].copy()
        if static_correction is not None:
            F_diag += np.diag(static_correction)
        diag = np.concatenate([F_diag, T['K_2h1p'], T['K_2p1h']])
        return aop, diag, self.dimensions(nocc)

    # ---------------- batched T-sandwich machinery ----------------

    _CSF_KET_BATCH = 2048

    def _sector_rows(self, nocc, cls):
        """Cached (rows, T_cls[rows,:].T as CSR) -- the spin-orbital config
        rows the class isometry actually touches (its Sz sector; ~half of
        the class), so spin block builders can skip the rows T annihilates
        anyway (bra_idx restriction, ~2x cheaper block evaluation)."""
        def compute():
            Tc = self._T(nocc)[cls].tocsr()
            rows = np.unique(Tc.tocoo().row)
            return rows, Tc[rows, :].T.tocsr()
        return self._cached(f'_sector_rows_{cls}', nocc, compute)

    def _ket_batches(self, nocc, key, batch):
        """Cached CSF-ket column-batch structures of T[key]: a list of
        (b0, b1, spin_cols, mix) where mix maps the involved spin-orbital
        ket columns onto the CSF kets of the batch."""
        def compute():
            T3 = self._T(nocc)[key].tocsc()
            n3 = T3.shape[1]
            out = []
            for b0 in range(0, n3, batch):
                b1 = min(b0 + batch, n3)
                sub = T3[:, b0:b1].tocoo()
                spin_cols = np.unique(sub.row)
                mix = sparse.csr_matrix(
                    (sub.data, (np.searchsorted(spin_cols, sub.row), sub.col)),
                    shape=(len(spin_cols), b1 - b0))
                out.append((b0, b1, spin_cols, mix))
            return out
        return self._cached(f'_ket_batches_{key}_{batch}', nocc, compute)

    def _sandwich(self, nocc, bra_cls, ket_key, block_fn, batch=None):
        """Dense (n_bra_csf, n_ket_csf) = T_bra^T Block_spin T_ket assembled
        in CSF-ket batches with sector-restricted bra rows -- no full-size
        spin-orbital block is ever materialized. block_fn(nocc, ket_idx,
        bra_idx=...) is one of the spin solver's block builders."""
        batch = batch or self._CSF_KET_BATCH
        bra_rows, TbT = self._sector_rows(nocc, bra_cls)
        n_bra = TbT.shape[0]
        n_ket = self._T(nocc)[ket_key].shape[1]
        out = np.empty((n_bra, n_ket))
        for b0, b1, spin_cols, mix in self._ket_batches(nocc, ket_key, batch):
            blk = block_fn(nocc, spin_cols, bra_idx=bra_rows)
            red = np.asarray(TbT @ blk)
            out[:, b0:b1] = (mix.T @ red.T).T
        return out

    def dimensions_adc4(self, nocc):
        d = self.dimensions(nocc)
        T = self._T(nocc)
        d = dict(d)
        d['n3h2p'] = T['3h2p'].shape[1]
        d['n3p2h'] = T['3p2h'].shape[1]
        d['nH_adc4'] = d['nH'] + d['n3h2p'] + d['n3p2h']
        return d

    def _C1st_csf(self, nocc, cls):
        """First-order C block in the CSF basis (T_cls^T C_spin T_cls),
        cached -- ~16x smaller than the spin-orbital block and assembled
        without ever forming it (build_supermatrix's dense CSF assembly)."""
        block_fn = (self.spin._C_2h1p_block if cls == '2h1p'
                    else self.spin._C_2p1h_block)
        return self._cached(f'_C1st_csf_{cls}', nocc,
                            lambda: self._sandwich(nocc, cls, cls, block_fn))
