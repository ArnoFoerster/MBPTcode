"""One-particle transition densities of ISR-ADC(2): ground state to excited
state, and excited state to excited state.

In the intermediate-state representation (ISR) a one-particle operator is the
matrix D~ of a^dag_p a_q between intermediate states and the row F of its
elements against the ground state (Schirmer and Trofimov, J. Chem. Phys. 120,
11449 (2004)). With l, r two full ADC eigenvectors (singles u1, doubles u2,
unit norm together),

    gamma^0r_pq = <Psi_0|a^dag_p a_q|r> = F^pq . r
    gamma^lr_pq = <l|a^dag_p a_q|r>     = l . (D~^pq - rho_0^pq 1) . r

ADC(2) truncates D~ - rho_0 1 consistently with its secular matrix
(singles/singles through second order, singles/doubles through first,
doubles/doubles at zeroth order) and F through second order on the singles
and first on the doubles. gamma^lr is not symmetric for l != r: it holds the
antisymmetric part an imaginary antisymmetric operator such as the spin-orbit
coupling reads, which a state's own density (ee_density, l = r) does not fix.

Spin orbitals, as ee_density (u2 the full antisymmetric doubles tensor in the
paper scaling, t2 = <ij||ab>/(e_a + e_b - e_i - e_j), t1_2 and t2_2 the
second-order singles and doubles amplitudes of ee_equations.amplitudes, p0
the MP2 ground-state density blocks; T(x)_ia = sum_jb t2_ijab x_jb):

    state to state, p1_oo = -r1 l1^T, p1_vv = l1^T r1
    oo = p1_oo                                                  oo_ss0
         - 2 sum_kab r2_ikab l2_jkab                            oo_dd
         - T(l1) T(r1)^T                                        oo_ru
         + (p1_oo p0_oo + p0_oo p1_oo) / 2                      oo_p0
         - 1/2 sum_klcd t_ikcd p1_lk t_jlcd                     oo_tt_o
         + sum_kcdb t_ikcd t_jkcb p1vv_db                       oo_tt_v
         - [sum_akc r1_ia t_jkac T(l1)_kc
            + sum_akc l1_ja t_ikac T(r1)_kc] / 2                oo_tru
    vv = p1_vv                                                  vv_ss0
         + 2 sum_ijc l2_ijac r2_ijbc                            vv_dd
         + T(r1)^T T(l1)                                        vv_ru
         - (p1_vv p0_vv + p0_vv p1_vv) / 2                      vv_p0
         - 1/2 sum_klcd t_klbc t_klad p1vv_cd                   vv_tt_v
         + sum_klcj t_klbc p1oo_jk t_jlac                       vv_tt_o
         + [sum_ikc t_ikac T(l1)_kc r1_ib
            + sum_ikc t_ikbc T(r1)_kc l1_ia] / 2                vv_tru
    ov = -2 sum_jb l1_jb r2_ijab                                ov_sd
         + 2 T(s)_ia,  s_jb = sum_kc r1_kc l2_jkbc              ov_tsd
         - p0_ov p1_vv                                          ov_p0v
         + p1_oo p0_ov                                          ov_p0o
         - sum_bklc r1_ib t_klca l2_klcb                        ov_tdv
         - sum_jkcd t_ikcd l2_jkcd r1_ja                        ov_tdo
    vo = the ov terms of gamma^rl, transposed                   vo_*

    ground to excited, u = r
    ov = u1                                                     ov_s
         - u1 p0_vv / 2                                         ov_p0v
         + p0_oo u1 / 2                                         ov_p0o
         + 1/2 sum_jbkc t_ijab t_jkbc u1_kc                     ov_ring
    vo = -T(u1)^T                                               vo_t1
         - (sum_jb t2_2_ijab u1_jb)^T                           vo_t2
    oo = -u1 t1_2^T                                             oo_s
         - sum_kab u2_ikab t_jkab                               oo_d
    vv = t1_2^T u1                                              vv_s
         + sum_ijc t_ijac u2_ijbc                               vv_d

At l = r the state-to-state terms are ee_density's (whose symmetrized terms
are the halves written out here). Both lists are the exact ISR of the
determinant space truncated at these orders, term by term (the test's
oracle); the ground row is the adcc form (Herbst, Scheurer, Fransson, Rehn
and Dreuw, WIREs Comput. Mol. Sci. 10, e1462 (2020)) and pyscf's radc_ee
transition moments with second-order amplitudes. At ADC(1) the state-to-state
density is the CIS one (ss0 terms) and the ground row is first order (ov_s,
vo_t1).

The equations are written once against the tensor backends of ee_equations
(NUMPY on spin-orbital arrays, SPIN_BLOCKED on spin blocks). The spin-blocked
evaluation reads a `Reference` (the ground state's amplitudes and MP2
density) and two states as (u1, u2) pairs:

    dense spin-free route   flat_state of a full eigenvector, the reference
                            from the full integrals (reference_from_integrals)
    folded route            fold_state of a root of ee_ri_fold, its doubles
                            rebuilt whole from the singles, the reference from
                            the operator's own DF factors (factors_of,
                            reference_from_factors: the particle ladder of
                            t2_2 from slabs of the four-virtual block)

Spin-component scaling (SCS-ADC(2), ee_ri_fold's `scaling` (c_os, c_ss)).
The scaled fold is the symmetric problem whose couplings of a single to a
doubles configuration carry sqrt(c) of the doubles' pair class; its
eigenvector's doubles are sqrt(c_os) Z and sqrt(c_ss) W. The transition
densities are the ISR terms above in that model: the reference's couplings to
the doubles carry the same sqrt(c), so t2 is sqrt(c) times MP1's, p0 and t1_2
are c weighted (the SCS-MP2 density), and t2_2 is built from the scaled t2
(ee_equations.amplitudes). At c_os = c_ss = 1 every term is ADC(2)'s.

Both hold doubles-sized arrays (o^2 v^2), so they serve molecules where that
fits. A closed shell's blocks are flip-symmetric, so every contraction forms
the alpha-first spin blocks alone (ee_spin_blocks.SB parity). A singlet and
an Ms = 0 triplet couple through the spin part (alpha - beta) of their
transition density, equal spins through the charge part (alpha + beta).
"""
from typing import NamedTuple

import numpy as np

from src.SingleReference.ADC.eeADC import ee_equations as _eq
from src.SingleReference.ADC.eeADC import ee_r_sigma as _r
from src.SingleReference.ADC.eeADC import ee_r_sigma_df as _df
from src.SingleReference.ADC.eeADC import ee_ri_density as _rd
from src.SingleReference.ADC.eeADC import ee_spin_blocks as _sb
from src.SingleReference.ADC.eeADC import ee_utils

TERMS = {
    'oo': ('oo_ss0', 'oo_dd', 'oo_ru', 'oo_p0', 'oo_tt_o', 'oo_tt_v', 'oo_tru'),
    'vv': ('vv_ss0', 'vv_dd', 'vv_ru', 'vv_p0', 'vv_tt_v', 'vv_tt_o', 'vv_tru'),
    'ov': ('ov_sd', 'ov_tsd', 'ov_p0v', 'ov_p0o', 'ov_tdv', 'ov_tdo'),
    'vo': ('vo_sd', 'vo_tsd', 'vo_p0v', 'vo_p0o', 'vo_tdv', 'vo_tdo'),
}
ALL_TERMS = TERMS['oo'] + TERMS['vv'] + TERMS['ov'] + TERMS['vo']
GROUND_TERMS = ('ov_s', 'ov_p0v', 'ov_p0o', 'ov_ring', 'vo_t1', 'vo_t2',
                'oo_s', 'oo_d', 'vv_s', 'vv_d')


def _ov_terms(ein, t2, p0, p1_oo, p1_vv, l1, l2, r1, r2):
    """The ov block of gamma^lr, (o, v)."""
    s = ein('kc,jkbc->jb', r1, l2)
    return {'sd': -2.0 * ein('jb,ijab->ia', l1, r2),
            'tsd': 2.0 * ein('ijab,jb->ia', t2, s),
            'p0v': -ein('ib,ba->ia', p0['ov'], p1_vv),
            'p0o': ein('ij,ja->ia', p1_oo, p0['ov']),
            'tdv': -ein('iklc,klca->ia', ein('ib,klcb->iklc', r1, l2), t2),
            'tdo': -ein('ij,ja->ia', ein('ikcd,jkcd->ij', t2, l2), r1)}


def transition_density_terms(be, l1, r1, l2=None, r2=None, t2=None, p0=None):
    """{term: block} of gamma^lr_pq = <l|a^dag_p a_q|r> (module docstring).

    be : ee_equations.NUMPY (spin-orbital arrays) or SPIN_BLOCKED.
    l1, r1 : (o, v) singles of the unit full eigenvectors (bra, ket).
    l2, r2 : (o, o, v, v) their doubles as the full antisymmetric tensors
             (ee_utils.PAPER_DOUBLES_SCALE); None at ADC(1).
    t2 : first-order MP amplitudes; p0 : ee_equations.density2 blocks.
    Blocks are (o, o), (v, v), (o, v) for 'ov_*' and (v, o) for 'vo_*'."""
    ein = be.ein
    p1_oo = -ein('ia,ja->ij', r1, l1)
    p1_vv = ein('ia,ib->ab', l1, r1)
    out = {'oo_ss0': p1_oo, 'vv_ss0': p1_vv}
    if l2 is None:
        return out
    tl, tr = ein('ijab,jb->ia', t2, l1), ein('ijab,jb->ia', t2, r1)
    out.update({
        'oo_dd': -2.0 * ein('ikab,jkab->ij', r2, l2),
        'oo_ru': -ein('ia,ja->ij', tl, tr),
        'oo_p0': 0.5 * (ein('ik,kj->ij', p1_oo, p0['oo'])
                        + ein('ik,kj->ij', p0['oo'], p1_oo)),
        # the t2 t2 terms in pairwise order, p1 (or the vector) first:
        # einsum's own path contracts t2 with t2 first (up to o^3 v^3)
        'oo_tt_o': -0.5 * ein('ikcd,jkcd->ij', t2,
                              ein('lk,jlcd->jkcd', p1_oo, t2)),
        'oo_tt_v': ein('ikcd,jkcd->ij', t2, ein('jkcb,db->jkcd', t2, p1_vv)),
        'oo_tru': -0.5 * (ein('ia,jkac,kc->ij', r1, t2, tl)
                          + ein('ja,ikac,kc->ij', l1, t2, tr)),
        'vv_dd': 2.0 * ein('ijac,ijbc->ab', l2, r2),
        'vv_ru': ein('ia,ib->ab', tr, tl),
        'vv_p0': -0.5 * (ein('ac,cb->ab', p1_vv, p0['vv'])
                         + ein('ac,cb->ab', p0['vv'], p1_vv)),
        'vv_tt_v': -0.5 * ein('klbd,klad->ab', ein('klbc,cd->klbd', t2, p1_vv),
                              t2),
        'vv_tt_o': ein('klbc,klac->ab', t2, ein('jk,jlac->klac', p1_oo, t2)),
        'vv_tru': 0.5 * (ein('ikac,kc,ib->ab', t2, tl, r1)
                         + ein('ikbc,kc,ia->ab', t2, tr, l1)),
    })
    ov = _ov_terms(ein, t2, p0, p1_oo, p1_vv, l1, l2, r1, r2)
    # gamma^lr_vo = (gamma^rl_ov)^T: the same terms with bra and ket swapped
    p1_oo_rl = -ein('ia,ja->ij', l1, r1)
    p1_vv_rl = ein('ia,ib->ab', r1, l1)
    vo = _ov_terms(ein, t2, p0, p1_oo_rl, p1_vv_rl, r1, r2, l1, l2)
    out.update({f'ov_{k}': v for k, v in ov.items()})
    out.update({f'vo_{k}': ein('ia->ai', v) for k, v in vo.items()})
    return out


def ground_transition_terms(be, u1, u2=None, t2=None, t2_2=None, t1_2=None,
                            p0=None):
    """{term: block} of gamma^0n_pq = <Psi_0|a^dag_p a_q|n> (module
    docstring). u1, u2 as in transition_density_terms; t2_2 and t1_2 the
    second-order doubles and singles amplitudes; u2 None (with t2 given)
    is the first-order ADC(1) row, t2 None the zeroth-order (CIS) one."""
    ein = be.ein
    out = {'ov_s': u1}
    if t2 is None:
        return out
    out['vo_t1'] = -ein('ijab,jb->ai', t2, u1)
    if u2 is None:
        return out
    out.update({
        'ov_p0v': -0.5 * ein('ib,ba->ia', u1, p0['vv']),
        'ov_p0o': 0.5 * ein('ij,ja->ia', p0['oo'], u1),
        'ov_ring': 0.5 * ein('ijab,jb->ia', t2, ein('jkbc,kc->jb', t2, u1)),
        'vo_t2': -ein('ijab,jb->ai', t2_2, u1),
        'oo_s': -ein('ia,ja->ij', u1, t1_2),
        'oo_d': -ein('ikab,jkab->ij', u2, t2),
        'vv_s': ein('ia,ib->ab', t1_2, u1),
        'vv_d': ein('ijac,ijbc->ab', t2, u2),
    })
    return out


def assemble(terms, no, nv, drop=()):
    """The (no + nv) square matrix gamma_pq of a term dict of plain (alpha or
    spin-orbital) blocks, the terms named in `drop` left out; not
    symmetrized."""
    D = np.zeros((no + nv, no + nv))
    where = {'oo': (slice(0, no), slice(0, no)),
             'vv': (slice(no, None), slice(no, None)),
             'ov': (slice(0, no), slice(no, None)),
             'vo': (slice(no, None), slice(0, no))}
    for name, blk in terms.items():
        if name not in drop:
            D[where[name[:2]]] += np.asarray(blk, float)
    return D


def spin_components(terms):
    """Spin-blocked terms (SB) -> ({term: alpha + beta}, {term: alpha - beta})
    spatial blocks: the charge and the spin (Ms = 0) transition density."""
    charge, spin = {}, {}
    for name, blk in terms.items():
        a, b = blk.get('aa'), blk.get('bb')
        a = 0.0 if a is None else a
        b = 0.0 if b is None else b
        charge[name], spin[name] = a + b, a - b
    return charge, spin


class Reference(NamedTuple):
    """The ground state's pieces the transition densities read, spin-blocked:
    t2 the first-order amplitudes, p0 the MP2 density blocks
    (ee_equations.density2), t1_2 and t2_2 the second-order singles and
    doubles amplitudes (t2_2 None when only state-to-state densities are
    wanted; all None at ADC(1) without the ground row)."""
    t2: object
    p0: object
    t1_2: object
    t2_2: object


def _reference(gb, eps, no, level, ground, vk=None, scaling=None):
    """`Reference` from spin-blocked integral blocks gb (ee_equations'
    amplitudes): t2_2, which needs the particle ladder, only for the ground
    row; scaling (c_os, c_ss) the spin-component scaled ground state
    (ee_equations.amplitudes)."""
    if level == 'adc1' and not ground:
        return Reference(None, None, None, None)
    # a closed shell's integral blocks are flip-symmetric (each beta-first
    # block is its alpha-first partner), and so is everything built from
    # them: every contraction forms the alpha-first half alone
    for blk in gb.values():
        if isinstance(blk, _sb.SB):
            blk.parity = 1.0
    d_ijab, d_ia, _ = ee_utils.orbital_denominators(np.asarray(eps, float), no)
    order = 3 if ground and level != 'adc1' else 2
    amps = _eq.amplitudes(_eq.SPIN_BLOCKED, gb, d_ijab, d_ia, order=order, vk=vk,
                          scaling=scaling)
    if level == 'adc1':
        return Reference(amps['t2_1'], None, None, None)
    return Reference(amps['t2_1'], _eq.density2(_eq.SPIN_BLOCKED, amps),
                     amps['t1_2'], amps.get('t2_2'))


def reference_from_integrals(eps, V, no, level='adc2', ground=True):
    """`Reference` of the full spatial physicist tensor V[p,q,r,s] of the
    active orbitals (the dense routes)."""
    return _reference(_sb.g_blocks_sb(V, no, len(eps)), eps, no, level, ground)


def reference_from_factors(eps, B, no, level='adc2', ground=True, scaling=None):
    """`Reference` of the DF factor B (naux, norb, norb) of the active
    orbitals: the four-virtual block is never held whole (ee_r_sigma_df's
    blocks and particle ladder, in slabs of (ac|bd)). scaling (c_os, c_ss):
    the ground state of SCS-ADC(2) (module docstring)."""
    norb = len(eps)
    return _reference(_df.g_blocks_df(B, no, norb), eps, no, level, ground,
                      vk=_df.DFVvvvKernels(B, no, norb), scaling=scaling)


def transition_density(ref, left, right, terms=False):
    """(charge, spin) spatial transition densities <l|..|r> of two states
    given as spin-blocked (u1, u2) pairs (u2 None at ADC(1)) on the
    reference `ref`: charge = alpha + beta, spin = alpha - beta (the Ms = 0
    component). Between two singlets or two triplets the spin part is zero,
    between a singlet and a triplet the charge part is. terms=True returns
    the two term dicts instead."""
    (l1, l2), (r1, r2) = left, right
    no, nv = _shape(l1)
    if l2 is None:
        out = transition_density_terms(_eq.SPIN_BLOCKED, l1, r1)
    else:
        out = transition_density_terms(_eq.SPIN_BLOCKED, l1, r1, l2, r2,
                                       ref.t2, ref.p0)
    charge, spin = spin_components(out)
    if terms:
        return charge, spin
    return assemble(charge, no, nv), assemble(spin, no, nv)


def ground_transition_density(ref, state, terms=False):
    """(charge, spin) spatial ground-to-excited transition densities
    <Psi_0|..|state> of one spin-blocked (u1, u2) state on `ref` (built with
    ground=True): a singlet has charge only, an Ms = 0 triplet spin only."""
    u1, u2 = state
    no, nv = _shape(u1)
    if u2 is None:
        out = ground_transition_terms(_eq.SPIN_BLOCKED, u1, t2=ref.t2)
    else:
        out = ground_transition_terms(_eq.SPIN_BLOCKED, u1, u2, ref.t2,
                                      ref.t2_2, ref.t1_2, ref.p0)
    charge, spin = spin_components(out)
    if terms:
        return charge, spin
    return assemble(charge, no, nv), assemble(spin, no, nv)


def _shape(u1):
    return next(iter(u1.blocks.values())).shape


def flat_state(v, no, nv, level='adc2', parity=None):
    """(u1, u2) spin-blocked of one full eigenvector of the dense spin-free
    route (ee_r_sigma's flat layout, unit norm); u2 None at ADC(1)."""
    u1, u2 = _r.to_blocks(np.asarray(v, float), no, nv, level, parity=parity)
    return u1, (None if level == 'adc1' else u2)


def fold_state(op, y, omega, t1):
    """(u1, u2) spin-blocked of one root of the folded operator `op`
    (ee_ri_fold.RIFoldADC2): the singles sqrt(t1) y and the doubles rebuilt
    from them (ee_ri_density.root_doubles), together the unit full
    eigenvector, in the paper scaling; a spin-component scaled operator's
    doubles carry sqrt(c) of their pair class (module docstring). The
    doubles are held whole: the route of molecules whose o^2 v^2 fits in
    memory (ee_ri_transition is the tiled one)."""
    no, nv = op.no, op.nv
    s = 1.0 if op.spin == 'singlet' else -1.0
    x = np.sqrt(t1 / 2.0) * np.asarray(y, float).reshape(no, nv)
    u1 = _sb.SB.from_half({'aa': x}, s)
    if op.level == 'adc1':
        return u1, None
    # (i a j b) tiles -> (i j a b); u2[i_a j_b a_b b_a] = -Z_ibja, and the
    # beta-first blocks are the flip partners, times the parity
    Z, W = _rd.root_doubles(op, y, omega, t1)
    if op.scs is not None:
        Z, W = np.sqrt(op.scs[0]) * Z, np.sqrt(op.scs[1]) * W
    return u1, _sb.SB.from_half({'aaaa': W.transpose(0, 2, 1, 3),
                                 'abab': Z.transpose(0, 2, 1, 3),
                                 'abba': -Z.transpose(0, 2, 3, 1)}, s)


def factors_of(op):
    """The DF factor B (naux, norb, norb) of the active orbitals, from the
    folded operator's own factors, every auxiliary chunk of B_vv on this
    rank (a serial operator)."""
    f = op.f
    if f.Bvv is None and sorted(op.my_chunks) != list(range(len(op.chunks))):
        raise ValueError('the full-vector transition densities read B_vv whole; '
                         'this operator holds it in auxiliary chunks over ranks')
    no, nv = f.no, f.nv
    B = np.zeros((f.naux, no + nv, no + nv))
    B[:, :no, :no] = f.BooP
    B[:, :no, no:] = f.LT
    B[:, no:, :no] = f.LT.transpose(0, 2, 1)
    for c, (p0, p1) in enumerate(op.chunks):
        B[p0:p1, no:, no:] = f.bvv(c, p0, p1).transpose(1, 0, 2)
    return B
