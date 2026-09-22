"""Numerical spin adaptation for the polarization-propagator ADC route:
the sparse isometry T from the spin-orbital PP configuration basis
(singles (i,a) + doubles (i<j, a<b), as enumerated by ee_utils) onto its
SINGLET (S=0) or TRIPLET (S=1, Ms=0) subspace.

Same design decision as spin_adapt.py on the IP/EA side, and for the same
reason: **no CSF matrix element is derived by hand here.** T is built from
exact bitstring second quantization (reusing spin_adapt's apply_ops /
_apply_S2 / det_sz), so every spin-adapted block inherits the spin-orbital
implementation's own validation:

    H_csf = T^T H_spin T      (same poles, each once, no Ms redundancy)

Why the spin-orbital spectrum needs this at all: an RHF reference gives a
spin-free supermatrix, so it commutes with S^2 and Sz. The spin-orbital
configuration basis therefore carries every Ms component of every triplet
(3x redundancy) plus the singlet, interleaved in one spectrum. Projecting
recovers the two physical channels separately -- which is what a
comparison against a singlet-only reference such as pyscf's RHF EE-ADC
needs.

S^2 is block diagonal in (excitation manifold, spatial occupation pattern)
and preserves Ms, so it is diagonalized in tiny blocks; the Ms=0 sector is
the only one kept (it carries both S=0 and S=1).

Configuration phase convention: |Psi_I> = C_I |Phi0> with
C_I = c+_a c_i (singles) and c+_a c+_b c_i c_j, a<b, i<j (doubles) --
the paper's Eq. (2), identical to the convention ee_u_dense_full's blocks
were validated against (tests/test_ee_adc_spinorbital.py's exact
Slater-Condon gate).
"""
import numpy as np
from scipy import sparse

from src.SingleReference.ADC.eeADC import ee_utils
from src.SingleReference.ADC.spin_adapt import apply_ops, det_sz, _apply_S2

_S2_SINGLET = 0.0
_S2_TRIPLET = 2.0


def _config_states(nocc, norb):
    """[(det_bitmask, sign, manifold, spatial_pattern)] for every PP
    configuration, in the ee_utils ordering (singles then doubles)."""
    hf = (1 << nocc) - 1
    out = []
    for i, a in zip(*ee_utils.configs_singles(nocc, norb)):
        det, sign = apply_ops(hf, [(int(a), True), (int(i), False)])
        out.append((det, sign, 1))
    for i, j, a, b in zip(*ee_utils.configs_doubles(nocc, norb)):
        det, sign = apply_ops(hf, [(int(a), True), (int(b), True),
                                   (int(i), False), (int(j), False)])
        out.append((det, sign, 2))
    return out


def _spatial_pattern(det):
    """Spatial-orbital occupation numbers -- the label S^2 cannot change."""
    occ = []
    m = 0
    d = det
    while d:
        if d & 1:
            occ.append(m // 2)
        d >>= 1
        m += 1
    pat = {}
    for p in occ:
        pat[p] = pat.get(p, 0) + 1
    return tuple(sorted(pat.items()))


def csf_isometry(nocc, norb, spin='singlet', level='adc3'):
    """(n_config, n_csf) sparse isometry onto the requested spin channel.

    level only selects whether the doubles manifold is present ('adc1' is
    singles-only), matching ee_u_dense_full.build_supermatrix's dimensions."""
    target = {'singlet': _S2_SINGLET, 'triplet': _S2_TRIPLET}[spin]
    dims = ee_utils.dimensions(nocc, norb)
    states = _config_states(nocc, norb)
    n_cfg = dims['n_s'] if level == 'adc1' else dims['nH']
    states = states[:n_cfg]
    if any(s[0] is None for s in states):
        raise RuntimeError('configuration annihilated -- bad occupation layout')

    det_to_cfg = {det: (n, sign) for n, (det, sign, _) in enumerate(states)}
    blocks = {}
    for n, (det, sign, man) in enumerate(states):
        if det_sz(det) != 0.0:
            continue
        blocks.setdefault((man, _spatial_pattern(det)), []).append(n)

    norb_spatial = norb // 2
    cols, rows, vals = [], [], []
    ncol = 0
    for members in blocks.values():
        k = len(members)
        pos = {n: r for r, n in enumerate(members)}
        S2 = np.zeros((k, k))
        for n in members:
            det, sign, _ = states[n]
            for d2, c in _apply_S2(det, norb_spatial).items():
                hit = det_to_cfg.get(d2)
                if hit is None or hit[0] not in pos:
                    continue
                m, sgn_m = hit
                S2[pos[m], pos[n]] += sign * c * sgn_m
        w, v = np.linalg.eigh(0.5 * (S2 + S2.T))
        for e, vec in zip(w, v.T):
            if abs(e - target) > 1e-8:
                continue
            for n in members:
                c = vec[pos[n]]
                if abs(c) > 1e-12:
                    rows.append(n); cols.append(ncol); vals.append(c)
            ncol += 1
    return sparse.csr_matrix((vals, (rows, cols)),
                             shape=(n_cfg, max(ncol, 1))).toarray()
