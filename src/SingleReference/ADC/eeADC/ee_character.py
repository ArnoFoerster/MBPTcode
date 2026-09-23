"""Excitation-character analysis for EE-ADC states, so a computed root can be
matched against a reference table row.

The QUEST database labels a state by spin, spatial irrep, its index within
that (spin, irrep) block, and a character tag -- e.g. `^1B_1 (R; n->3s)`.
Reproducing that from an ADC eigenvector needs four things, all of which this
module extracts:

  irrep            product of the hole and particle orbital irreps, weighted
                   over the ph block (Abelian groups: the pyscf irrep id is a
                   bit pattern and the product is XOR)
  singles weight   ||y_1||^2 as a percentage -- the ADC analogue of the
                   %T1 column, and the flag for a doubly excited state that
                   ADC(2)/ADC(3) cannot describe
  index in block   position among same-spin, same-irrep roots by energy
  <r^2> shift      sum over ph of |y_ia|^2 (<a|r^2|a> - <i|r^2|i>), the
                   Rydberg/valence discriminator the QUEST papers use

Degenerate pairs (E, Pi, Delta ...) come out of an Abelian subgroup as two
roots in different irreps at the same energy; `merge_degenerate` labels them
so a `^1E` or `^1Pi` reference row can be matched.

Nothing here guesses: a state whose irrep weight is not concentrated
(`irrep_purity` below the threshold) is reported as ambiguous rather than
assigned, which is the honest outcome for the strongly mixed cases the
reference papers themselves flag as hard to assign.
"""
import numpy as np

from src.Base.constants import HARTREE_TO_EV
from pyscf import symm

from src.SingleReference.ADC.eeADC import ee_r_sigma as _r



# ----------------------------------------------------------------------
# orbital data
# ----------------------------------------------------------------------

def orbital_info(mf, mol=None, frozen=0):
    """{'sym_id', 'sym_name', 'r2'} for the ACTIVE orbitals.

    RHF: spatial orbitals, which the interleaved spin-orbital layout indexes
    two at a time. UHF: assembled in the BLOCK-STACKED spin-orbital order
    [occ_a, occ_b, virt_a, virt_b] that ee_driver.spin_orbital_arrays
    produces, so the caller indexes these exactly as it indexes eps."""
    from pyscf import scf as _scf
    mol = mol if mol is not None else mf.mol
    r2_ao = mol.intor('int1e_r2')

    def one(mo):
        if mol.symmetry:
            ids = np.asarray(symm.label_orb_symm(mol, mol.irrep_id,
                                                 mol.symm_orb, mo))
            names = np.asarray(symm.label_orb_symm(mol, mol.irrep_name,
                                                   mol.symm_orb, mo))
        else:
            ids = np.zeros(mo.shape[1], dtype=int)
            names = np.array(['A'] * mo.shape[1])
        r2 = np.einsum('pi,pq,qi->i', mo, r2_ao, mo, optimize=True)
        return ids, names, r2

    if isinstance(mf, _scf.uhf.UHF):
        na, nb = mf.nelec
        occ, vir = [], []
        for k, nocc_s in ((0, na), (1, nb)):
            ids, names, r2 = one(mf.mo_coeff[k])
            occ.append((ids[frozen:nocc_s], names[frozen:nocc_s],
                        r2[frozen:nocc_s]))
            vir.append((ids[nocc_s:], names[nocc_s:], r2[nocc_s:]))
        def cat(k):
            return np.concatenate([occ[0][k], occ[1][k], vir[0][k], vir[1][k]])
        ids, names, r2 = cat(0), cat(1), cat(2)
        # Spatial symmetry of the REFERENCE determinant: the product over
        # every occupied spin orbital, frozen core included (it cancels).
        # A closed shell is totally symmetric and this is 0; an open shell
        # generally is NOT -- OH's X 2-Pi reference is B1 (or B2) in C2v --
        # and then the irrep of a STATE is the irrep of the excitation times
        # this one. Without it OH's A 2-Sigma+ comes out labelled B2 and
        # matches nothing.
        full_a, _, _ = one(mf.mo_coeff[0])
        full_b, _, _ = one(mf.mo_coeff[1])
        ref_id = 0
        for x in np.concatenate([full_a[:na], full_b[:nb]]):
            ref_id ^= int(x)
    else:
        ref_id = 0
        ids, names, r2 = one(mf.mo_coeff)
        act = slice(frozen, mf.mo_coeff.shape[1])
        ids, names, r2 = ids[act], names[act], r2[act]
    return {'sym_id': ids, 'sym_name': names, 'r2': r2,
            'ref_irrep_id': ref_id,
            'group': mol.groupname if mol.symmetry else 'C1',
            'topgroup': mol.topgroup if mol.symmetry else 'C1'}


# ----------------------------------------------------------------------
# per-state analysis
# ----------------------------------------------------------------------

def analyze(vec, no, nv, level, info, purity_tol=0.6, layout='spinfree'):
    """Character of one EE-ADC eigenvector.

    layout='spinfree'      ee_r_sigma's spatial spin-blocked vector; `no`/`nv`
                           are SPATIAL counts and info carries spatial
                           orbitals.
    layout='unrestricted'  ee_u_r_sigma_df's vector; `no`/`nv` are the PAIRS
                           (no_a, no_b) / (nv_a, nv_b), and info carries the
                           block-stacked [occ_a, occ_b, virt_a, virt_b]
                           orbitals that orbital_info builds for a UHF
                           reference.
    layout='spinorbital'   ee_u_* config layout (singles then unique
                           doubles); `no`/`nv` are SPIN-orbital counts.

    Internally each layout reduces to a list of ph CHANNELS -- a weight
    matrix plus the occupied and virtual index ranges it is indexed by -- and
    the irrep/character accumulation is then the same for all three. An
    unrestricted reference simply has two channels whose orbital sets differ,
    which is why the weights cannot be added matrix-wise the way the
    closed-shell alpha and beta blocks can."""
    vec = np.asarray(vec).ravel()
    ids, names, r2 = info['sym_id'], info['sym_name'], info['r2']
    ref_id = int(info.get('ref_irrep_id', 0))
    tot = float(vec @ vec)

    if layout == 'spinorbital':
        n_s = no * nv
        chans = [(vec[:n_s].reshape(no, nv) ** 2, slice(0, no),
                  slice(no, no + nv))]
        total_1 = chans[0][0].sum()
        d = {'singles': total_1 / tot, 'doubles': 1.0 - total_1 / tot}
    elif layout == 'unrestricted':
        from src.SingleReference.ADC.eeADC import ee_u_r_sigma_df as _u
        no_a, no_b = no
        nv_a, nv_b = nv
        dims = _u.dimensions(no_a, no_b, nv_a, nv_b, level)
        y1, _ = _u.to_blocks(vec, dims, level)
        o = {'a': slice(0, no_a), 'b': slice(no_a, no_a + no_b)}
        v = {'a': slice(no_a + no_b, no_a + no_b + nv_a),
             'b': slice(no_a + no_b + nv_a, no_a + no_b + nv_a + nv_b)}
        chans = [(y1.get(s + s) ** 2, o[s], v[s]) for s in ('a', 'b')
                 if y1.get(s + s) is not None]
        n_s = dims['n_sa'] + dims['n_sb']
        s_w = float(vec[:n_s] @ vec[:n_s])
        d = {'singles': s_w / tot, 'doubles': (tot - s_w) / tot}
        total_1 = sum(float(c[0].sum()) for c in chans)
    else:
        y1, _ = _r.to_blocks(vec, no, nv, level)
        # the two singles spin blocks carry the same spatial information for a
        # spin-pure state; use their total ph weight
        w = np.zeros((no, nv))
        for key in ('aa', 'bb'):
            blk = y1.get(key)
            if blk is not None:
                w = w + blk ** 2
        chans = [(w, slice(0, no), slice(no, no + nv))]
        total_1 = w.sum()
        d = dimensions_norm(vec, no, nv, level)

    scale = total_1 if total_1 > 0 else 1.0
    weights, dr2 = {}, 0.0
    best = (-1.0, 0, 0)
    for w, o_sl, v_sl in chans:
        wn = w / scale
        occ_id, vir_id = ids[o_sl], ids[v_sl]
        # Abelian irrep of the STATE: excitation times the reference's own
        # irrep (zero for a closed shell) -- see orbital_info.
        prod = (occ_id[:, None] ^ vir_id[None, :]) ^ ref_id
        for ir in np.unique(prod):
            weights[int(ir)] = weights.get(int(ir), 0.0) \
                + float(wn[prod == ir].sum())
        dr2 += float((wn * (r2[v_sl][None, :] - r2[o_sl][:, None])).sum())
        i, a = np.unravel_index(np.argmax(wn), wn.shape)
        if float(wn[i, a]) > best[0]:
            o0 = o_sl.start or 0
            v0 = v_sl.start or 0
            best = (float(wn[i, a]), o0 + int(i), v0 + int(a))
    ir_best = max(weights, key=weights.get)
    purity = weights[ir_best]
    dom_w, i_dom, a_dom = best

    return {
        'irrep_id': ir_best if purity >= purity_tol else None,
        'irrep': _id2name(info['group'], ir_best) if purity >= purity_tol else '?',
        'irrep_purity': purity,
        'singles_weight': 100.0 * d['singles'],
        'doubles_weight': 100.0 * d['doubles'],
        'dominant': (int(i_dom), int(a_dom)),
        'dominant_sym': (str(names[i_dom]), str(names[a_dom])),
        'dominant_weight': dom_w,
        'r2_shift': dr2,
        # ROUGH diagnostic only. A single threshold on the <r^2> shift does
        # not separate the two families cleanly -- measured on QUEST1,
        # water's Rydberg states sit at 36-56 a.u. but CO's VALENCE states
        # already reach 18-20 and its Rydberg only 30, so any cut misassigns
        # some. The QUEST papers combine <r^2> with MO inspection and
        # oscillator strengths. Split benchmark statistics on the reference
        # V/R column, not on this.
        'character': 'R' if dr2 > 25.0 else 'V',
    }


def dimensions_norm(vec, no, nv, level):
    """Fraction of the (normalized) vector in the singles and doubles blocks."""
    v = np.asarray(vec).ravel()
    d = _r.dimensions(no, nv, level)
    n_s = 2 * d['n_s']
    tot = float(v @ v)
    s = float(v[:n_s] @ v[:n_s])
    return {'singles': s / tot, 'doubles': (tot - s) / tot}


def _id2name(group, ir_id):
    try:
        return symm.irrep_id2name(group, int(ir_id))
    except Exception:
        return str(ir_id)


# ----------------------------------------------------------------------
# labelling a whole set of roots
# ----------------------------------------------------------------------

def label_states(energies_ev, vecs, no, nv, level, info, spin,
                 degeneracy_tol=1e-3, topgroup=None, layout='spinfree',
                 mult=None):
    """Attach (irrep, index-within-block, label) to a set of roots of ONE
    spin channel, in ascending energy order.

    `mult` overrides the multiplicity that `spin` would imply. It may be a
    single int, or one value per root in the order of `energies_ev` -- the
    open-shell case, where a single solve returns both 2S+1 and 2S+3 states
    (the Delta-Ms = 0 sector holds an Ms component of each) and they have to
    be counted in separate index blocks."""
    if mult is None:
        mult = {'singlet': 1, 'triplet': 3}[spin]
    mult_of = ((lambda k: int(mult)) if np.isscalar(mult)
               else (lambda k: None if mult[k] is None else int(mult[k])))
    topgroup = topgroup or info.get('topgroup', info['group'])
    out = []
    order = np.argsort(energies_ev)
    counter = {}
    for k in order:
        rec = analyze(vecs[:, k] if vecs.ndim == 2 else vecs[k], no, nv,
                      level, info, layout=layout)
        rec['energy_ev'] = float(energies_ev[k])
        rec['spin'] = spin
        rec['mult'] = mult_of(k)
        rec['sub_irrep'] = rec['irrep']
        out.append(rec)
    out = merge_degenerate(out, degeneracy_tol)
    # true-group labels, then index within (mult, true irrep)
    counter = {}
    for rec in out:
        span = sorted({rec['sub_irrep'], *rec['degenerate_with']})
        rec['irrep'] = true_irrep(topgroup, span) or rec['sub_irrep']
        if not rec['degenerate_first']:
            continue
        key = (rec['mult'], rec['irrep'])
        counter[key] = counter.get(key, 0) + 1
        rec['index_in_block'] = counter[key]
        rec['label'] = f"^{rec['mult']}{rec['irrep']}"
    for rec in out:
        rec.setdefault('index_in_block', 0)
        rec.setdefault('label', f"^{rec['mult']}{rec['irrep']}")
    return out


def merge_degenerate(records, tol=1e-3):
    """Tag roots that are degenerate but sit in different Abelian irreps --
    the components of an E / Pi / Delta level in the full point group. The
    reference tables list such a level ONCE, so only the first component
    should be matched against it."""
    for r in records:
        r['degenerate_with'] = []
        r['degenerate_first'] = True
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            if abs(a['energy_ev'] - b['energy_ev']) < tol and \
                    a['irrep'] != b['irrep']:
                a['degenerate_with'].append(b['irrep'])
                b['degenerate_with'].append(a['irrep'])
                b['degenerate_first'] = False
    return records


# ----------------------------------------------------------------------
# reference-label parsing / matching
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# non-Abelian point groups
# ----------------------------------------------------------------------
#
# pyscf works in the largest ABELIAN subgroup, so a molecule whose true group
# is C3v, C(inf)v or D(inf)h comes back labelled in Cs, C2v or D2h. The
# reference tables use the TRUE group, so a degenerate level appears there
# once (E, Pi, Delta) but here as two roots in two different subgroup irreps.
#
# The correlation below maps each true irrep onto the subgroup irrep(s) it
# splits into, so a computed root -- alone or as a degenerate cluster -- can
# be given its true label. The maps are one-to-one on the sets: a lone A1 in
# C2v can only be Sigma+, an {A1, A2} pair can only be Delta, and so on, so
# no guessing is involved. Groups absent from the table are assumed already
# Abelian (Cs, C2v, C2h, D2h, Ci, C2, C1), where the label passes through.
# NOTE pyscf keeps the LINEAR groups intact (groupname 'Coov'/'Dooh') and
# labels their irreps A1/A2/E1x/E1y/E2x/E2y, so those entries are a spelling
# map, not a subgroup descent. The genuinely non-Abelian descents here are
# C3v -> Cs and D3h -> C2v.
CORRELATION = {
    'Coov': {'Sigma+': ('A1',), 'Sigma-': ('A2',),
             'Pi': ('E1x', 'E1y'), 'Delta': ('E2x', 'E2y'),
             'Phi': ('E3x', 'E3y')},
    'Dooh': {'Sigma g+': ('A1g',), 'Sigma g-': ('A2g',),
             'Sigma u+': ('A1u',), 'Sigma u-': ('A2u',),
             'Pi g': ('E1gx', 'E1gy'), 'Pi u': ('E1ux', 'E1uy'),
             'Delta g': ('E2gx', 'E2gy'), 'Delta u': ('E2ux', 'E2uy'),
             'Phi g': ('E3gx', 'E3gy'), 'Phi u': ('E3ux', 'E3uy')},
    'C3v': {'A1': ("A'",), 'A2': ('A"',), 'E': ("A'", 'A"')},
    'D3h': {"A1'": ('A1',), "A2'": ('B1',), "E'": ('A1', 'B1'),
            'A1"': ('A2',), 'A2"': ('B2',), 'E"': ('A2', 'B2')},
}

# reference spellings of the linear-group labels, normalized to CORRELATION's
_LINEAR_ALIAS = {
    'Sigmag+': 'Sigma g+', 'Sigmag-': 'Sigma g-',
    'Sigmau+': 'Sigma u+', 'Sigmau-': 'Sigma u-',
    'Pig': 'Pi g', 'Piu': 'Pi u', 'Deltag': 'Delta g', 'Deltau': 'Delta u',
    'Sigma': 'Sigma+', 'Sigma+': 'Sigma+', 'Sigma-': 'Sigma-',
}


def true_irrep(topgroup, sub_irreps):
    """True-group label for a computed root or degenerate cluster.

    sub_irreps is the SET of Abelian subgroup irreps the cluster spans.
    Returns None when the group is not tabulated or the set matches nothing,
    which the caller reports rather than guessing around."""
    table = CORRELATION.get(topgroup)
    if table is None:
        return sub_irreps[0] if len(sub_irreps) == 1 else None
    want = tuple(sorted(sub_irreps))
    for name, parts in table.items():
        if tuple(sorted(parts)) == want:
            return name
    return None


_GREEK = {'Sigma': 'Sigma', 'Pi': 'Pi', 'Delta': 'Delta', 'Phi': 'Phi'}


def parse_reference_label(state):
    """QUEST `State` field -> (mult, normalized irrep).

    `^1B_1` -> (1, 'B1');  `^3A''` -> (3, 'A\"');  `^1E` -> (1, 'E');
    `^1\\Sigma^-` -> (1, 'Sigma-')."""
    s = str(state).strip().replace('\xa0', ' ').strip()
    s = s.replace('$', '').replace('\\mathrm', '').replace('{', '').replace('}', '')
    if not s.startswith('^'):
        return None, None
    mult = int(s[1])
    ir = s[2:].strip()
    for tag in ('[F]', '[T]', '[B]'):
        ir = ir.replace(tag, '')
    ir = ir.replace('_', '').replace('\\', '').replace(' ', '')
    ir = ir.replace('primeprime', '"').replace('prime', "'")
    ir = ir.replace("''", '"')
    ir = ir.replace('^-', '-').replace('^+', '+').replace('^', '')
    ir = _LINEAR_ALIAS.get(ir, ir)
    return mult, ir


def normalize_computed_irrep(irrep):
    """pyscf irrep name -> the reference convention (`A1`, `A"`, `B3u` ...)."""
    return str(irrep).replace("''", '"').strip()


def match(records, ref_rows, molecule):
    """Pair computed records with reference rows on (mult, irrep, index).

    Returns (matched, unmatched_ref, unmatched_calc). Reference rows whose
    irrep is a degenerate label the Abelian subgroup cannot name (E, Pi, ...)
    are matched positionally within their multiplicity instead, and flagged."""
    refs = [r for r in ref_rows if r['Molecule'] == molecule]
    seen = {}
    ref_keyed = []
    for r in refs:
        mult, ir = parse_reference_label(r['State'])
        if mult is None:
            continue
        seen[(mult, ir)] = seen.get((mult, ir), 0) + 1
        # An explicit `Index` wins over the running counter. QUEST4 states
        # come with one (its table names both endpoints of the transition,
        # so the index is given rather than implied by order), and counting
        # occurrences would be wrong there: that table is a selection, and a
        # lower state of the same symmetry left out of it would shift every
        # index above it by one.
        idx = str(r.get('Index', '')).strip()
        ref_keyed.append(((mult, ir, int(idx) if idx else seen[(mult, ir)]), r))

    calc = {}
    for rec in records:
        if not rec['degenerate_first']:
            continue
        calc[(rec['mult'], normalize_computed_irrep(rec['irrep']),
              rec['index_in_block'])] = rec

    matched, unmatched_ref = [], []
    for key, r in ref_keyed:
        rec = calc.pop(key, None)
        if rec is None:
            unmatched_ref.append(r)
        else:
            matched.append((r, rec))
    return matched, unmatched_ref, list(calc.values())
