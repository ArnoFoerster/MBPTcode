"""Precursor-class combinatorics and pdaggerq operator-string conventions for
the automated ADC(n) generator. Single
source of truth shared by the codegen driver (generate_adcn.py, symbolic)
and the ISR runtime (isr_engine.py, numeric) so both agree byte-for-byte
on which axis of a class's index tuple is which.

A precursor class is parametrized by k=0,1,2,... and a sector:
  IP: (nh, np) = (k+1, k) -- k=0 -> 1h, k=1 -> 2h1p, k=2 -> 3h2p, ...
  EA: (nh, np) = (k, k+1) -- k=0 -> 1p, k=1 -> 2p1h, k=2 -> 3p2h, ...

Flattening convention (verified to reproduce base.py's hand-written configs
exactly -- see tests/test_precursor_classes.py): holes are always the outer/
repeated combinatoric axis (ascending, via itertools.combinations(occ, nh)),
particles are always the inner/tiled axis (ascending, via
itertools.combinations(virt, np)) -- regardless of whether holes or
particles are the "k+1" (majority) role. This matches
_configs_2h1p/_configs_3h2p (holes outer) AND _configs_2p1h/_configs_3p2h
(holes still outer, even though particles are the majority there).
"""
import math
from itertools import combinations

import numpy as np

# Same alphabets as pdaggerq_interface.py's PDAGGERQ_SAFE_OCC/VIR (6 letters
# each) -- reused directly, not re-derived, so index-safety failures surface
# identically to generate_mpn_density.py's.
def _label_pool(base, n=64):
    """Unbounded pdaggerq index-label pool: the classic single letters first,
    then NUMBERED labels ('i7', 'i8', ... / 'a7', 'a8', ...).

    pdaggerq accepts numbered labels and preserves them ('a*(i1) a(i2)' ->
    'd(i1,i2)'), treats 'i' and 'i1' as DISTINCT, classifies by the LEADING
    letter, and handles them in pq.strings(spin_labels=...) -- all verified
    empirically. So the historic 6-letter budget was never a pdaggerq limit;
    it was this pool.

    The first len(base) entries are UNCHANGED, so every module generated to
    date regenerates byte-identically (verified: 30/30).

    Note the einsum SUBSCRIPT layer is separately bounded (24 occ / 22 vir per
    term) -- see pdaggerq_interface._subscript_map, which maps these labels to
    single characters family-preservingly.
    """
    pool = list(base)
    k = len(base) + 1
    while len(pool) < n:
        pool.append(f'{base[0]}{k}')
        k += 1
    return pool


OCC_LETTERS = _label_pool('ijklmn')
VIR_LETTERS = _label_pool('abcdef')

# D x C cross pairs of the union space: name -> (k_i of the D-type bra class,
# k_j of the C-type ket class); class tags as the generated function names
# spell them.
CROSS_PAIRS = {'k1k1': (1, 1), 'k1k2': (1, 2), 'k2k1': (2, 1), 'k2k2': (2, 2)}
CLASS_TAGS = {0: {'IP': '1h', 'EA': '1p'},
              1: {'IP': '2h1p', 'EA': '2p1h'}, 2: {'IP': '3h2p', 'EA': '3p2h'}}


def nondyson_module_name(order, k_i, k_j):
    """Module basename of the non-Dyson (single-sector, ordinary-metric) ISR
    block between classes k_i and k_j at `order`: adc{N}_nd{order}_k{k_i}k{k_j}
    with N = order + k_i + k_j, the ADC order at which that content first
    enters (orders 0 and 1 of the 1h-1h block are filed under adc2). Both
    sectors share one module; the function names carry the class tags."""
    return f'adc{max(order + k_i + k_j, 2)}_nd{order}_k{k_i}k{k_j}_blocks.py'


def cross_module_name(order, pair, restricted=False, df=False, apply=False):
    """Module basename of the D x C cross block at `order`: adc{N}_x{order}_{pair}
    with N = order + k_i + k_j the ADC order at which the block first enters
    (k1k1 at order 2 -> adc4_x2_k1k1_blocks). Suffixes follow the other
    families: _restricted[_df] for the dense spin-blocked twins,
    _apply_restricted[_df]_blocks for the sigma-vector form. Shared by the
    emitters (generate_adcn, sigma_codegen) and the runtime resolver
    (the generated-ADC runtime's cross-block resolver)."""
    k_i, k_j = CROSS_PAIRS[pair]
    stem = f'adc{order + k_i + k_j}_x{order}_{pair}'
    if apply:
        return f'{stem}_apply_restricted{"_df" if df else ""}_blocks.py'
    return f'{stem}_blocks{"_restricted" if restricted else ""}{"_df" if df else ""}.py'


def _nh_np(k, sector):
    if sector == 'IP':
        return k + 1, k
    if sector == 'EA':
        return k, k + 1
    raise ValueError(f"sector must be 'IP' or 'EA', got {sector!r}")


def class_configs(nocc, norb, k, sector='IP'):
    """Configuration index arrays for precursor class k (given sector),
    holes-outer/particles-inner flattened. Returns a tuple of 1D int arrays,
    (hole_0,...,hole_{nh-1}, particle_0,...,particle_{np-1}), each of length
    class_dimension(nocc, norb-nocc, k, sector).

    Reproduces (byte-for-byte, see tests/test_precursor_classes.py):
      class_configs(nocc, norb, 1, 'IP')  == base.py's _configs_2h1p(nocc)
      class_configs(nocc, norb, 2, 'IP')  == base.py's _configs_3h2p(nocc)
      class_configs(nocc, norb, 1, 'EA')  == base.py's _configs_2p1h(nocc)
      class_configs(nocc, norb, 2, 'EA')  == base.py's _configs_3p2h(nocc)
    """
    nh, np_ = _nh_np(k, sector)
    # Spin-orbital indices are < norb, so a narrow int dtype suffices and keeps
    # the (n_h x n_p) outer-product flattening -- the O(nv^3 nocc^2) k=2 (3p2h)
    # manifold, the downfold's peak-memory driver -- as small as possible
    # (int16 is 4x smaller than the default int64). Values compare equal to the
    # int64 reference (tests/test_precursor_classes.py uses np.array_equal).
    idt = np.int16 if norb < 32000 else np.int64
    occ = np.arange(nocc)
    virt = np.arange(nocc, norb)
    # a class with no hole (1p) or no particle (1h) has one empty combination
    holes = (np.array(list(combinations(occ, nh)), dtype=idt).reshape(-1, nh)
             if nh else np.zeros((1, 0), dtype=idt))
    parts = (np.array(list(combinations(virt, np_)), dtype=idt).reshape(-1, np_)
             if np_ else np.zeros((1, 0), dtype=idt))
    n_h, n_p = len(holes), len(parts)
    hole_cols = [np.repeat(holes[:, r], n_p) for r in range(nh)]
    part_cols = [np.tile(parts[:, r], n_h) for r in range(np_)]
    return tuple(hole_cols + part_cols)


def class_dimension(nocc, nvirt, k, sector='IP'):
    """dim of precursor class k (sector) = C(nocc,nh)*C(nvirt,np)."""
    nh, np_ = _nh_np(k, sector)
    return math.comb(nocc, nh) * math.comb(nvirt, np_)


def dimensions_adcn(nocc, norb, n, max_k=None):
    """Segment sizes for every precursor class up to order n (both sectors),
    generalizing base.py's dimensions()/dimensions_adc4(). max_k caps which
    classes are enumerated (default: n, i.e. classes up through k=n are
    considered -- callers doing real order bookkeeping should tighten this
    to whichever k-range the requested order n actually couples to, same
    "demand-driven, not blanket" caveat generate_mpn_density.py's docstring
    gives for its own order-4 ranks).
    """
    nvirt = norb - nocc
    if max_k is None:
        max_k = n
    dims = {}
    for sector in ('IP', 'EA'):
        for k in range(0, max_k + 1):
            dims[(sector, k)] = class_dimension(nocc, nvirt, k, sector)
    dims['norb'] = norb
    dims['nocc'] = nocc
    dims['nvirt'] = nvirt
    dims['nH'] = norb + sum(v for key, v in dims.items() if isinstance(key, tuple) and key[1] >= 1)
    return dims


def _check_letter_budget(nh, np_, occ_offset=0, vir_offset=0):
    """Bounds check against the LABEL POOL (now unbounded-by-construction, so
    this only fires if someone asks for more than _label_pool() pre-built).

    It is NOT a pdaggerq limit -- pdaggerq handles numbered labels fine -- so
    raising the pool size is the fix if this ever fires."""
    if occ_offset + nh > len(OCC_LETTERS) or vir_offset + np_ > len(VIR_LETTERS):
        raise ValueError(
            f"class (nh={nh}, np={np_}) at letter offsets (occ={occ_offset}, "
            f"vir={vir_offset}) exceeds the pre-built index-label pool "
            f"({len(OCC_LETTERS)} occ / {len(VIR_LETTERS)} vir labels) -- "
            "raise the `n` argument of _label_pool()")


def class_letters(k, sector='IP', occ_offset=0, vir_offset=0):
    """(hole_letters, particle_letters) for class k at the given letter
    offsets -- the offsets exist so a bra class and a ket class in the SAME
    pdaggerq call can use disjoint letters (e.g. C-block bra 2h1p = i,j/a,
    ket 2h1p = k,l/b via occ_offset=2, vir_offset=1)."""
    nh, np_ = _nh_np(k, sector)
    _check_letter_budget(nh, np_, occ_offset, vir_offset)
    return (list(OCC_LETTERS[occ_offset:occ_offset + nh]),
            list(VIR_LETTERS[vir_offset:vir_offset + np_]))


def precursor_ket_string(k, sector='IP', occ_offset=0, vir_offset=0):
    """C_K as a list of pdaggerq a*(p)/a(q) tokens (ket convention: acts on
    |Phi_0> from the right). Pattern (verified against
    generate_eom_ccsdt.py's hand-written IP_KET/EA_KET tables, which are
    this exact convention at rank R=k+1): create the np particles ascending,
    then annihilate the nh holes DESCENDING (most-recently-labeled hole
    first) -- e.g. k=1 IP (2h1p, nh=2,np=1): ['a*(a)','a(j)','a(i)'];
    k=1 EA (2p1h, nh=1,np=2): ['a*(a)','a*(b)','a(i)'].

    occ_offset/vir_offset shift which OCC_LETTERS/VIR_LETTERS are used (see
    class_letters) so bra and ket classes in one call stay disjoint.
    """
    holes, parts = class_letters(k, sector, occ_offset, vir_offset)
    return [f'a*({p})' for p in parts] + [f'a({h})' for h in reversed(holes)]


def adjoint_string(tokens):
    """Adjoint of an operator string: reversed token order, every a*<->a."""
    def flip(tok):
        return tok.replace('a*(', '\0(').replace('a(', 'a*(').replace('\0(', 'a(')

    return [flip(tok) for tok in reversed(tokens)]


def precursor_bra_string(k, sector='IP', occ_offset=0, vir_offset=0):
    """C_J^dagger as a list of pdaggerq tokens (bra convention: acts on
    <Phi_0| from the left) -- the adjoint of precursor_ket_string: reverse
    the token order and flip every a*<->a. E.g. k=1 IP:
    ['a*(i)','a*(j)','a(a)'] (matches generate_eom_ccsdt.py's IP_BRA[2]).
    """
    return adjoint_string(precursor_ket_string(k, sector, occ_offset, vir_offset))


def union_operator_string(k, sector='IP', occ_offset=0, vir_offset=0):
    """The character-(+1) operator of class k in the union space
    H_{N+1} (+) H_{N-1} (Tarantelli and Cederbaum, PRA 45, 2790 (1992)), as
    ket tokens acting on |Phi_0> from the right: EA classes are the
    precursor operators themselves (C = a+_a a+_b a_i, ...); IP classes enter
    as the adjoints of the precursor operators (D = E^dag = a+_i a+_j a_a,
    ...), creation on holes and annihilation on a particle. Every class then
    adds one electron, and the sector sign of the two-sector convention is
    carried by the composite Hamiltonian (H_P - E0) (+) -(H_M - E0) instead
    of by a reversed commutator."""
    if sector == 'EA':
        return precursor_ket_string(k, sector, occ_offset, vir_offset)
    return precursor_bra_string(k, sector, occ_offset, vir_offset)


def union_bra_string(k, sector='IP', occ_offset=0, vir_offset=0):
    """X^dag of the union-space operator X of class k (bra tokens): for an
    IP class this is the precursor operator E itself."""
    return adjoint_string(union_operator_string(k, sector, occ_offset, vir_offset))


def pair_index_convention(bra_k, bra_sector, ket_k, ket_sector):
    """Letters and output-axis order of a (bra class, ket class) element of
    arbitrary character: bra letters first (holes, then particles), then the
    ket letters drawn past the bra's letter pool, so the two classes never
    share a label. Works for same-sector pairs (where it reproduces the
    class-class generators' letter layout) and for the D x C cross blocks."""
    convb = class_index_convention(bra_k, bra_sector)
    bh, bp = class_letters(bra_k, bra_sector)
    kh, kp = class_letters(ket_k, ket_sector, occ_offset=convb['nh'],
                           vir_offset=convb['np'])
    return {
        'bra_holes': bh, 'bra_parts': bp, 'ket_holes': kh, 'ket_parts': kp,
        'bra_letters': bh + bp, 'ket_letters': kh + kp,
        'out_idx': bh + bp + kh + kp,
        'ket_occ_offset': convb['nh'], 'ket_vir_offset': convb['np'],
    }


def class_index_convention(k, sector='IP'):
    """Documents which OCC_LETTERS/VIR_LETTERS positions are holes vs
    particles for class k (sector), and in which order class_configs'
    returned tuple presents them -- the single fixed convention
    generate_adcn.py (symbolic) and the generated-ADC runtime (numeric) must both read
    off of, so they can never silently disagree."""
    nh, np_ = _nh_np(k, sector)
    return {
        'nh': nh, 'np': np_,
        'hole_letters': list(OCC_LETTERS[:nh]),
        'particle_letters': list(VIR_LETTERS[:np_]),
        'configs_order': ['hole'] * nh + ['particle'] * np_,
    }
