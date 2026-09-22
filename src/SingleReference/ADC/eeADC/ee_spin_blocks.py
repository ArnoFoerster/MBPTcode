"""Spin-blocked spatial tensors for the spin-free EE-ADC route.

A `SB` holds one SPATIAL array per spin case: SB({'abab': X, 'aaaa': Y, ...}),
where the key gives the spin (a/b) of each tensor index in order. Missing key
= zero block. `sb_einsum` then evaluates a spin-orbital einsum by enumerating
the spin assignments of every index and contracting the corresponding spatial
blocks:

    sb_einsum('ikcd,jkcd->ij', g_oovv, t1)

is the same subscript string the spin-orbital route uses, so the ADC working
equations are transcribed ONCE (ee_equations) and evaluated either
spin-orbital (numpy.einsum on full spin-orbital arrays) or spin-free
(sb_einsum on spatial blocks). No spin-summed equation is derived by hand --
the same reasoning as spin_adapt.py's numerical isometry, applied to the
contraction layer instead of the basis.

Blocks that vanish by spin never get written, so the enumeration cost is a
few spatial einsums per term, exactly what a hand spin-summation would
produce. Storage is spatial: a four-index quantity that costs (2n)^4 in spin
orbitals costs a handful of n^4 blocks here.

Antisymmetry convention: `anti4` builds every nonzero block of a
<pq||rs>-antisymmetric quantity from the plain spatial physicist integral
V[p,q,r,s], for a closed-shell reference:

    aaaa = bbbb = V - V(pqsr)     abab = baba = V
    abba = baab = -V(pqsr)
"""
import itertools

import numpy as np


class SB:
    """Spin-blocked spatial tensor: {spin string -> ndarray}."""

    __slots__ = ('blocks', 'derived')

    def __init__(self, blocks=None, derived=None):
        self.blocks = {k: v for k, v in (blocks or {}).items() if v is not None}
        # Optional provenance: {key: (base_key, axis_permutation, sign)} when
        # a block is a signed transpose of another. A kernel that commutes
        # with such a transpose can then run once on the base and permute the
        # RESULT instead of running again -- see DFVvvvKernels.ladder, where
        # it halves the dominant cost. Every arithmetic operation below drops
        # it, so a stale map can never be used.
        self.derived = derived

    def get(self, key):
        return self.blocks.get(key)

    def keys(self):
        return self.blocks.keys()

    def items(self):
        return self.blocks.items()

    def __repr__(self):
        return f'SB({sorted(self.blocks)})'

    # ---- arithmetic (mirrors what the equations do to plain ndarrays) ----
    def _combine(self, other, sign):
        if not isinstance(other, SB):
            return NotImplemented
        out = dict(self.blocks)
        for k, v in other.blocks.items():
            out[k] = out[k] + sign * v if k in out else sign * v
        return SB(out)

    def __add__(self, other):
        return self._combine(other, +1.0)

    def __sub__(self, other):
        return self._combine(other, -1.0)

    def __neg__(self):
        return SB({k: -v for k, v in self.blocks.items()})

    def __mul__(self, c):
        return SB({k: c * v for k, v in self.blocks.items()})

    __rmul__ = __mul__

    def transpose(self, *axes):
        """Permute tensor axes AND the spin string the same way."""
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return SB({''.join(k[a] for a in axes): v.transpose(axes)
                   for k, v in self.blocks.items()})

    def scale(self, arr):
        """Multiply elementwise by a weight. A plain array is applied to every
        block (a spin-free weight such as an MP denominator); an SB is applied
        block by block (a spin-RESOLVED weight such as an Epstein-Nesbet
        denominator, whose value differs between same- and opposite-spin
        configurations)."""
        if isinstance(arr, SB):
            return SB({k: v * arr.get(k) for k, v in self.blocks.items()
                       if arr.get(k) is not None})
        return SB({k: v * arr for k, v in self.blocks.items()})

    def divide(self, arr):
        """scale() by the reciprocal, without forming 1/arr for the whole SB."""
        if isinstance(arr, SB):
            return SB({k: v / arr.get(k) for k, v in self.blocks.items()
                       if arr.get(k) is not None})
        return SB({k: v / arr for k, v in self.blocks.items()})

    def spin_flip(self):
        """The closed-shell involution alpha <-> beta."""
        tr = str.maketrans('ab', 'ba')
        return SB({k.translate(tr): v for k, v in self.blocks.items()})


def sb_einsum(subs, *ops, optimize=True):
    """einsum over spin-blocked tensors -- see the module docstring."""
    lhs, rhs = subs.split('->')
    terms = lhs.split(',')
    if len(terms) != len(ops):
        raise ValueError(f'{subs!r} expects {len(terms)} operands, got {len(ops)}')
    idx = sorted(set(''.join(terms)))
    out = {}
    for assign in itertools.product('ab', repeat=len(idx)):
        smap = dict(zip(idx, assign))
        arrays = []
        for t, op in zip(terms, ops):
            arr = op.get(''.join(smap[c] for c in t))
            if arr is None:
                break
            arrays.append(arr)
        else:
            key = ''.join(smap[c] for c in rhs)
            val = np.einsum(subs, *arrays, optimize=optimize)
            out[key] = out[key] + val if key in out else val
    return SB(out)


# ----------------------------------------------------------------------
# constructors
# ----------------------------------------------------------------------

def anti4(V, Vx=None):
    """Every nonzero spin block of a <pq||rs>-antisymmetric quantity from the
    plain spatial physicist integrals, for a closed-shell reference.

    V  = V_spatial[p, q, r, s]
    Vx = V_spatial[p, q, s, r]  (the exchange partner, already transposed back
         into the pqrs axis order). Defaults to V.transpose(0,1,3,2), which is
         only valid when the r and s axes have the same length -- for a mixed
         block such as ooov the partner lives in a DIFFERENT spatial block
         (oovo) and must be passed explicitly."""
    if Vx is None:
        Vx = V.transpose(0, 1, 3, 2)
    same = V - Vx
    neg = -Vx                      # ONE array, referenced twice: writing
                                   # `-Vx` in both slots allocated a SECOND
                                   # copy, another nv^4 (458 MB at nv = 87)
    return SB({'aaaa': same, 'bbbb': same, 'abab': V, 'baba': V,
               'abba': neg, 'baab': neg})


_G_BLOCKS = {'oooo': 'oooo', 'ooov': 'oovo', 'oovv': 'oovv',
             'ovov': 'ovvo', 'ovvv': 'ovvv', 'vvvv': 'vvvv'}


def g_blocks_sb(V, nocc_spatial, norb_spatial):
    """The six <pq||rs> occupied/virtual blocks the ADC(3) equations touch,
    spin-blocked, from the full spatial physicist tensor V[p,q,r,s]."""
    o = slice(0, nocc_spatial)
    v = slice(nocc_spatial, norb_spatial)
    sl = {'o': o, 'v': v}
    out = {}
    for fam, xfam in _G_BLOCKS.items():
        direct = V[tuple(sl[c] for c in fam)]
        # the exchange partner has the last two families swapped; transpose it
        # back so both arrays carry the same pqrs axis order
        partner = V[tuple(sl[c] for c in xfam)].transpose(0, 1, 3, 2)
        out[fam] = anti4(direct, partner)
    return out


def diag2(X):
    """A spin-diagonal two-index quantity (identity, density block, ...)."""
    return SB({'aa': X, 'bb': X})


def eye2(n):
    return diag2(np.eye(n))


# ----------------------------------------------------------------------
# spin-orbital <-> spin-blocked conversion (validation only)
# ----------------------------------------------------------------------

def _axis_index(family, spin, nocc, norb):
    """Positions of one spin within an ALREADY-SLICED spin-orbital axis: the
    occupied and virtual segments are each interleaved alpha/beta, so spatial
    orbital p sits at 2p + spin within its own segment."""
    n = nocc if family == 'o' else norb - nocc
    return np.arange(n // 2) * 2 + spin


def from_spin_orbital(T, families, nocc, norb):
    """Slice a spin-orbital tensor block (interleaved spins) into spatial spin
    blocks. `families` is a string of 'o'/'v' per axis, matching the block T
    was sliced from (ee_utils.g_blocks' convention)."""
    out = {}
    for assign in itertools.product((0, 1), repeat=len(families)):
        sl = [_axis_index(f, s, nocc, norb) for f, s in zip(families, assign)]
        blk = T[np.ix_(*sl)]
        if np.abs(blk).max() > 0:
            out[''.join('ab'[s] for s in assign)] = blk
    return SB(out)


def to_spin_orbital(sb, families, nocc, norb):
    """Inverse of from_spin_orbital (zero-filled where blocks are absent)."""
    shape = tuple(nocc if f == 'o' else norb - nocc for f in families)
    T = np.zeros(shape)
    for key, blk in sb.items():
        sl = [_axis_index(f, 'ab'.index(s), nocc, norb)
              for f, s in zip(families, key)]
        T[np.ix_(*sl)] = blk
    return T


# ----------------------------------------------------------------------
# unrestricted constructors
# ----------------------------------------------------------------------

def anti4_uhf(D_aa, X_aa, D_bb, X_bb, D_ab, X_ab, D_ba, X_ba):
    """Every nonzero spin block of a <pq||rs>-antisymmetric quantity for an
    UNRESTRICTED reference, where the alpha and beta spatial orbitals differ.

    The eight arguments are plain spatial physicist integrals:

        D_st[p,q,r,s] = <p_s q_t | r_s s_t>        X_st = the same, r <-> s

    all already in the pqrs axis order. Two blocks that `anti4` gets for free
    from a closed shell are now independent: `aaaa != bbbb` and
    `abab != baba`.

    The `ba` pair is passed in rather than transposed out of the `ab` pair.
    The identity <p_b q_a||r_b s_a> = <q_a p_b||s_a r_b> is true, but it
    permutes the ORBITAL FAMILIES along with the indices: applied to an
    `ooov` block it lands in `oovo`, which is a different array of a
    different shape. It is only self-consistent for families symmetric under
    (1,0,3,2) -- oooo, oovv, vvvv -- so relying on it silently produced
    transposed garbage for ooov, ovov and ovvv."""
    return SB({'aaaa': D_aa - X_aa, 'bbbb': D_bb - X_bb,
               'abab': D_ab, 'baba': D_ba,
               'abba': -X_ab, 'baab': -X_ba})


def _axis_index_blockstacked(family, spin, no_a, no_b, nv_a, nv_b):
    """Positions of one spin within an already-sliced BLOCK-STACKED axis.

    ee_driver.spin_orbital_arrays lays a UHF reference out as
    [occ_a, occ_b, virt_a, virt_b], so within the occupied segment alpha runs
    first and beta follows -- contiguous ranges, not the interleaving an RHF
    reference gives."""
    n_a, n_b = (no_a, no_b) if family == 'o' else (nv_a, nv_b)
    return np.arange(n_a) if spin == 0 else n_a + np.arange(n_b)


def from_spin_orbital_uhf(T, families, no_a, no_b, nv_a, nv_b, tol=0.0):
    """Slice a BLOCK-STACKED spin-orbital tensor block into spatial spin
    blocks -- the unrestricted counterpart of from_spin_orbital, and the
    oracle the unrestricted DF integral blocks are checked against."""
    out = {}
    for assign in itertools.product((0, 1), repeat=len(families)):
        sl = [_axis_index_blockstacked(f, s, no_a, no_b, nv_a, nv_b)
              for f, s in zip(families, assign)]
        blk = T[np.ix_(*sl)]
        if blk.size and np.abs(blk).max() > tol:
            out[''.join('ab'[s] for s in assign)] = blk
    return SB(out)
