"""Unrelaxed one-particle difference density of an ISR-ADC(2) excited state.

The intermediate-state representation (ISR) gives an excited state's density
as Y^T D~ Y, Y the full ADC eigenvector (singles y, doubles Y) and D~ the
matrix of the density operator between intermediate states (Schirmer and
Trofimov, J. Chem. Phys. 120, 11449 (2004)). ADC(2) truncates the excitation
part D~ - rho_0 1 consistently with its secular matrix: singles/singles
through second order, singles/doubles through first, doubles/doubles at
zeroth order. The difference density to the ground state is then, in spin
orbitals (u1 = y, u2 = Y as the full antisymmetric tensor whose square sum
over ALL ijab is the doubles weight, t the first-order MP amplitudes
t_ijab = <ij||ab>/(e_a + e_b - e_i - e_j), p0 the MP2 ground-state density
blocks, sym(A) = (A + A^T)/2):

    p1_oo = -u1 u1^T     p1_vv = u1^T u1     r_ia = sum_jb t_ijab u1_jb
    p2_ov = -2 sum_jb u1_jb u2_ijab

    oo = p1_oo                                              oo_ss0
         - 2 sum_kab u2_ikab u2_jkab                        oo_dd
         - r r^T                                            oo_ru
         + sym(p1_oo p0_oo)                                 oo_p0
         - 1/2 sym(sum_klcd t_ikcd p1_lk t_jlcd)            oo_tt_o
         + sym(sum_kcdb t_ikcd t_jkcb p1vv_db)              oo_tt_v
         - sym(sum_akc u1_ia t_jkac r_kc)                   oo_tru
    vv = p1_vv                                              vv_ss0
         + 2 sum_ijc u2_ijac u2_ijbc                        vv_dd
         + r^T r                                            vv_ru
         - sym(p1_vv p0_vv)                                 vv_p0
         - 1/2 sym(sum_klcd t_klbc t_klad p1vv_cd)          vv_tt_v
         + sym(sum_klcj t_klbc p1oo_jk t_jlac)              vv_tt_o
         + sym(sum_ikc t_ikac r_kc u1_ib)                   vv_tru
    ov = p2_ov                                              ov_sd
         - sum_jb t_ijab p2ov_jb                            ov_tsd
         - p0_ov p1_vv                                      ov_p0v
         + p1_oo p0_ov                                      ov_p0o
         - sum_bklc u1_ib t_klca u2_klcb                    ov_tdv
         - sum_jkcd t_ikcd u2_jkcd u1_ja                    ov_tdo

the form of the adcc implementation (Herbst, Scheurer, Fransson, Rehn and
Dreuw, WIREs Comput. Mol. Sci. 10, e1462 (2020)), whose ADC(2) state
densities came from the adcman module of Q-Chem (Wormit, Rehn, Harbach,
Wenzel, Krauter, Epifanovsky and Dreuw, Mol. Phys. 112, 774 (2014)), and the
density the perturbative state-specific solvation of ADC uses (Mewes, You,
Wormit, Kriesche, Herbert and Dreuw, J. Phys. Chem. A 119, 5446 (2015);
Mewes, Herbert and Dreuw, Phys. Chem. Chem. Phys. 19, 1644 (2017)).

Normalization: the FULL eigenvector is a unit vector, sum u1^2 + sum u2^2 = 1
(the singles weight T1 plus the doubles); the singles are never renormalized
alone. The terms sum to a traceless matrix, so the difference density holds
no charge. The density is unrelaxed: no orbital response.

pyscf's ISR-ADC(2) excited-state density (Banerjee and Sokolov, radc_ee / uadc_ee make_rdm1)
is the TOTAL density rho_HF + T1 rho_0^(2) + Delta; this module's Delta
equals it minus rho_HF minus T1 rho_0^(2), rho_0^(2) the MP2 ground-state
correlation density, for singlets and (through uadc_ee) triplets. The two
truncations differ by the doubles weight times rho_0^(2), a fourth-order
piece: the ISR doubles/doubles block carries the reference density only.

The equations are written once against the tensor backends of ee_equations:
NUMPY on full spin-orbital arrays, SPIN_BLOCKED on spatial spin blocks (the
full eigenvectors of the dense spin-free routes). ee_ri_density evaluates the
same terms by hand-summed spin cases on density-fitted tiles, and is tested
against this module term by term.
"""
import numpy as np

from src.SingleReference.ADC.eeADC import ee_equations as _eq
from src.SingleReference.ADC.eeADC import ee_r_sigma as _r
from src.SingleReference.ADC.eeADC import ee_spin_blocks as _sb
from src.SingleReference.ADC.eeADC import ee_utils

TERMS = {
    'oo': ('oo_ss0', 'oo_dd', 'oo_ru', 'oo_p0', 'oo_tt_o', 'oo_tt_v', 'oo_tru'),
    'vv': ('vv_ss0', 'vv_dd', 'vv_ru', 'vv_p0', 'vv_tt_v', 'vv_tt_o', 'vv_tru'),
    'ov': ('ov_sd', 'ov_tsd', 'ov_p0v', 'ov_p0o', 'ov_tdv', 'ov_tdo'),
}
ALL_TERMS = TERMS['oo'] + TERMS['vv'] + TERMS['ov']


def _sym(A):
    return 0.5 * (A + A.transpose(1, 0))


def difference_density_terms(be, u1, u2=None, t2=None, p0=None):
    """{term: block} of the ISR-ADC(2) difference density (module docstring).

    be : ee_equations.NUMPY (spin-orbital arrays) or SPIN_BLOCKED (SB).
    u1 : (o, v) singles of the unit full eigenvector.
    u2 : (o, o, v, v) its doubles as the full antisymmetric tensor (the
         paper scaling of ee_utils.PAPER_DOUBLES_SCALE); None for ADC(1).
    t2 : (o, o, v, v) first-order MP amplitudes, <ij||ab>/(e_a+e_b-e_i-e_j).
    p0 : {'oo', 'vv', 'ov'} the second-order MP ground-state density
         (ee_equations.density2).
    """
    ein = be.ein
    p1_oo = -ein('ia,ja->ij', u1, u1)
    p1_vv = ein('ia,ib->ab', u1, u1)
    out = {'oo_ss0': p1_oo, 'vv_ss0': p1_vv}
    if u2 is None:
        return out
    r = ein('ijab,jb->ia', t2, u1)
    p2_ov = -2.0 * ein('jb,ijab->ia', u1, u2)
    out.update({
        'oo_dd': -2.0 * ein('ikab,jkab->ij', u2, u2),
        'oo_ru': -ein('ia,ja->ij', r, r),
        'oo_p0': _sym(ein('ik,kj->ij', p1_oo, p0['oo'])),
        'oo_tt_o': -0.5 * _sym(ein('ikcd,lk,jlcd->ij', t2, p1_oo, t2)),
        'oo_tt_v': _sym(ein('ikcd,jkcb,db->ij', t2, t2, p1_vv)),
        'oo_tru': -_sym(ein('ia,jkac,kc->ij', u1, t2, r)),
        'vv_dd': 2.0 * ein('ijac,ijbc->ab', u2, u2),
        'vv_ru': ein('ia,ib->ab', r, r),
        'vv_p0': -_sym(ein('ac,cb->ab', p1_vv, p0['vv'])),
        'vv_tt_v': -0.5 * _sym(ein('klbc,klad,cd->ab', t2, t2, p1_vv)),
        'vv_tt_o': _sym(ein('klbc,jk,jlac->ab', t2, p1_oo, t2)),
        'vv_tru': _sym(ein('ikac,kc,ib->ab', t2, r, u1)),
        'ov_sd': p2_ov,
        'ov_tsd': -ein('ijab,jb->ia', t2, p2_ov),
        'ov_p0v': -ein('ib,ba->ia', p0['ov'], p1_vv),
        'ov_p0o': ein('ij,ja->ia', p1_oo, p0['ov']),
        'ov_tdv': -ein('ib,klca,klcb->ia', u1, t2, u2),
        'ov_tdo': -ein('ikcd,jkcd,ja->ia', t2, u2, u1),
    })
    return out


def assemble(terms, no, nv, drop=()):
    """The (no + nv) square matrix of a term dict of plain (alpha or spin-
    orbital) blocks, the terms named in `drop` left out."""
    D = np.zeros((no + nv, no + nv))
    for name, blk in terms.items():
        if name in drop:
            continue
        blk = np.asarray(blk, float)
        if name.startswith('oo'):
            D[:no, :no] += blk
        elif name.startswith('vv'):
            D[no:, no:] += blk
        else:
            D[:no, no:] += blk
            D[no:, :no] += blk.T
    return D


def spin_summed(terms):
    """Spin-blocked terms (SB) -> spatial alpha + beta blocks."""
    out = {}
    for name, blk in terms.items():
        a, b = blk.get('aa'), blk.get('bb')
        out[name] = (0.0 if a is None else a) + (0.0 if b is None else b)
    return out


def spin_free_difference_density(eps, V, no, v, level='adc2', parity=None,
                                 terms=False):
    """Delta rho (norb, norb), spatial and spin-summed, of one full eigenvector
    of the dense spin-free route (ee_r_sigma's flat layout, singles then
    doubles, unit norm); the dense reference of the tiled route.

    eps : (norb,) active orbital energies; V : (norb,)*4 physicist <pq|rs> of
    the active orbitals; level 'adc1' or 'adc2'; parity +1 (singlet) / -1
    (triplet) when v is a flip eigenvector. terms=True returns the
    spin-summed term dict instead (blocks (no, no), (nv, nv), (no, nv))."""
    norb = len(eps)
    nv = norb - no
    be = _eq.SPIN_BLOCKED
    v = np.asarray(v, float)
    u1, u2 = _r.to_blocks(v, no, nv, level, parity=parity)
    if level == 'adc1':
        out = spin_summed(difference_density_terms(be, u1))
    else:
        gb = _sb.g_blocks_sb(V, no, norb)
        d_ijab, d_ia, _ = ee_utils._denominators(np.asarray(eps, float), no)
        amps = _eq.amplitudes(be, gb, d_ijab, d_ia, order=2)
        p0 = _eq.density2(be, amps)
        out = spin_summed(difference_density_terms(be, u1, u2, amps['t2_1'], p0))
    return out if terms else assemble(out, no, nv)


def spin_orbital_difference_density(eps, g, nocc, v, level='adc2',
                                    terms=False):
    """Delta rho of one full eigenvector of the spin-orbital route
    (ee_u_dense_full: singles, then unique-configuration doubles), in the
    spin-orbital basis (norb, norb); the validation form."""
    norb = len(eps)
    nv = norb - nocc
    be = _eq.NUMPY
    n_s = nocc * nv
    v = np.asarray(v, float)
    u1 = v[:n_s].reshape(nocc, nv)
    if level == 'adc1':
        out = difference_density_terms(be, u1)
    else:
        u2 = ee_utils.unfold_doubles(v[n_s:], nocc, norb) / ee_utils.PAPER_DOUBLES_SCALE
        amps = ee_utils.mp_amplitudes(eps, g, nocc, norb, order=2)
        out = difference_density_terms(be, u1, u2, amps['t2_1'],
                                       ee_utils.mp_density2(amps))
    return out if terms else assemble(out, nocc, nv)
