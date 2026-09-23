"""The EE-ADC working equations, written ONCE against an abstract tensor
backend so the spin-orbital and spin-free routes cannot drift apart.

A backend supplies three things:

    ein(subscripts, *tensors)   -- einsum
    scale(T, arr)               -- multiply T elementwise by a SPIN-FREE
                                   weight (an orbital-energy denominator)
    eye(n)                      -- identity of the backend's tensor type

`NUMPY` evaluates everything on full spin-orbital arrays; `SPIN_BLOCKED`
(ee_spin_blocks) evaluates the SAME subscript strings on spatial blocks,
enumerating spin cases. No spin-summed equation is written by hand anywhere
-- the spin-free route is the spin-orbital route with a different contraction
layer, and tests/test_ee_adc_spinfree.py checks the two agree on the sigma
vector to machine precision.

Equation labels are Leitner/Dempwolff/Dreuw, JCP 157, 184101 (2022); see
ee_utils for the truncation table and the doubles metric convention, and
ee_utils.z_intermediates for the one sign that does NOT follow the printed
appendix.
"""
import numpy as np

from src.SingleReference.ADC.eeADC import ee_spin_blocks as _sb


def _check_denominator(d, tol=1e-8):
    """Fail loudly on a vanishing amplitude denominator.

    eps_a + eps_b - eps_i - eps_j = 0 means an occupied and a virtual orbital
    are degenerate, i.e. the reference is not a valid starting point for a
    Moller-Plesset expansion. It happens for SYMMETRY-CONSTRAINED open-shell
    references with a degenerate singly-occupied shell (a 2-Pi radical such
    as OH: with symmetry enforced the two pi components stay degenerate while
    one of them is singly occupied). Without this check the NaN propagates
    silently through the amplitudes into the excitation energies.

    The usual remedy is to let the reference break symmetry (build the Mole
    with symmetry=False), or to use a reference that lifts the degeneracy."""
    import numpy as _np
    blocks = d.blocks.values() if hasattr(d, 'blocks') else [d]
    worst = min((_np.abs(_np.asarray(b)).min() for b in blocks), default=1.0)
    if worst < tol:
        raise ValueError(
            f'vanishing amplitude denominator (min |eps_a+eps_b-eps_i-eps_j| '
            f'= {worst:.2e}): an occupied and a virtual orbital are '
            'degenerate, so the MP partition is undefined. For an open-shell '
            'radical with a degenerate singly-occupied shell, build the Mole '
            'with symmetry=False so the reference can break symmetry.')


class _Backend:
    def __init__(self, ein, scale, divide, eye):
        self.ein, self.scale, self.divide, self.eye = ein, scale, divide, eye


NUMPY = _Backend(
    ein=lambda subs, *ops: np.einsum(subs, *ops, optimize=True),
    scale=lambda T, arr: T * arr,
    divide=lambda T, arr: T / arr,
    eye=np.eye,
)

SPIN_BLOCKED = _Backend(
    ein=_sb.sb_einsum,
    scale=lambda T, arr: T.scale(arr),
    divide=lambda T, arr: T.divide(arr),
    eye=lambda n: _sb.eye2(n),
)


# ======================================================================
# vvvv kernels
# ======================================================================
#
# Only three contractions in the whole ADC(3) hierarchy touch the four-virtual
# integral block. Routing them through a small kernel object lets the
# density-fitted route supply B-tensor implementations that never materialize
# <ab||cd> (ee_r_sigma_df), while the equations below stay single-sourced.

class VvvvKernels:
    """Default kernels, evaluated from an explicit <ab||cd> block."""

    def __init__(self, be, g_vvvv):
        self.be, self.g = be, g_vvvv

    def ladder(self, X):
        """sum_cd <ab||cd> X[i,j,c,d]  (the particle ladder)."""
        return self.be.ein('abcd,ijcd->ijab', self.g, X)

    def vv_density(self, rho_vv):
        """sum_cd <ac||bd> rho[c,d]  ->  (a,b)."""
        return self.be.ein('acbd,cd->ab', self.g, rho_vv)

    def z10(self, Z10):
        """sum_cd <ad||bc> Z10[i,c,j,d]  ->  (i,a,j,b)."""
        return self.be.ein('adbc,icjd->iajb', self.g, Z10)


# ======================================================================
# MP amplitudes (A3-A5), density (A21-A23), Z intermediates (A11-A20)
# ======================================================================

def amplitudes(be, gb, d_ijab, d_ia, order=3, vk=None, en_shift=None):
    """t2_1 (A3), t1_2 (A4) and, for order>=3, t2_2 (A5).

    d_ijab = eps_a + eps_b - eps_i - eps_j  and  d_ia = 2(eps_i - eps_a),
    the paper's Appendix-A preamble: doubles denominators carry the opposite
    sign to singles, and the singles denominator carries the extra factor 2.

    Both may instead be spin-RESOLVED weights (an SB on the spin-free
    backend), which is how the Epstein-Nesbet dressing enters: ee_en replaces
    these denominators and nothing else, exactly as the IP/EA side dresses
    amplitudes rather than supermatrix blocks.

    en_shift = d^EN_ijab - d^MP_ijab REMOVES A DOUBLE COUNT, and must be
    passed whenever d_ijab is EN-dressed and the caller needs t2^(2).
    Expanding the dressed first-order amplitude,

        t^(1),EN = <ij||ab>/(D + Delta) = t^(1) - t^(1) Delta/D + O(Delta^2),

    generates an O(lambda^2) amplitude correction -t^(1) Delta/D. That term is
    NUMERICALLY IDENTICAL to the diagonal part of the two ladder terms already
    inside the Moller-Plesset t2^(2) (A5): restricting -1/2 sum_kl t^ab_kl
    <ij||kl> to (k,l) = (i,j),(j,i) gives -t^(1) <ij||ij>, and the
    particle-particle ladder likewise gives -t^(1) <ab||ab>, so their sum is
    -t^(1) Delta. Verified to 1e-18 on HF/H2O/LiH.

    ADC(2) uses only t^(1), so an EN-dressed ADC(2) is clean. ADC(3) uses BOTH
    t^(1) and t^(2), so leaving t^(2) at its Moller-Plesset form counts the
    diagonal hole-hole and particle-particle ladder TWICE. Adding
    t^(1),EN * Delta back into the t2^(2) numerator removes exactly the piece
    the denominator has already resummed -- the standard requirement that
    whatever moves into H0 comes out of the perturbation."""
    vk = vk if vk is not None else VvvvKernels(be, gb.get('vvvv'))
    _check_denominator(d_ijab)
    t2_1 = be.divide(gb['oovv'], d_ijab)
    out = {'t2_1': t2_1}
    num = (be.ein('ijbc,jabc->ia', t2_1, gb['ovvv'])
           + be.ein('jkab,jkib->ia', t2_1, gb['ooov']))
    out['t1_2'] = be.divide(num, d_ia)
    if order < 3:
        return out
    x = be.ein('ikac,kbjc->ijab', t2_1, gb['ovov'])
    num2 = (x - x.transpose(1, 0, 2, 3) - x.transpose(0, 1, 3, 2)
            + x.transpose(1, 0, 3, 2))
    num2 = num2 - 0.5 * be.ein('klab,ijkl->ijab', t2_1, gb['oooo'])
    # the particle ladder of t2_1 IS Z^(5) (A15); compute it once and let
    # z_intermediates reuse it -- under DF it is the expensive kernel
    out['Z5'] = vk.ladder(t2_1)
    num2 = num2 - 0.5 * out['Z5']
    if en_shift is not None:
        num2 = num2 + be.scale(t2_1, en_shift)      # see the docstring
    out['t2_2'] = be.divide(num2, d_ijab)
    return out


def density2(be, amps):
    t1 = amps['t2_1']
    return {'oo': -0.5 * be.ein('ikab,jkab->ij', t1, t1),
            'ov': amps['t1_2'],
            'vv': 0.5 * be.ein('ijac,ijbc->ab', t1, t1)}


def z_intermediates(be, gb, amps, vk=None):
    """Z^(1..7,10) and the compounds Z^(A), Z^(B).

    The Z^(2) term of Z^(A) enters with the sign OPPOSITE to the printed
    A19 -- see ee_utils.z_intermediates for the numerical evidence."""
    vk = vk if vk is not None else VvvvKernels(be, gb.get('vvvv'))
    t1 = amps['t2_1']
    z = {}
    z['Z1'] = be.ein('kabc,ijbc->ijka', gb['ovvv'], t1)
    z['Z2'] = be.ein('kljb,ilab->ijka', gb['ooov'], t1)
    z['Z3'] = be.ein('ijkl,klab->ijab', gb['oooo'], t1)
    z['Z4'] = be.ein('kbic,jkac->ijab', gb['ovov'], t1)
    z['Z5'] = amps['Z5'] if 'Z5' in amps else vk.ladder(t1)
    z['Z6'] = be.ein('jkia,jkbc->iabc', gb['ooov'], t1)
    z['Z7'] = be.ein('jcad,ijbd->iabc', gb['ovvv'], t1)
    z['Z10'] = be.ein('ikac,jkbc->iajb', t1, t1)
    z['ZA'] = 0.5 * z['Z1'] - (z['Z2'] - z['Z2'].transpose(1, 0, 2, 3))
    z['ZB'] = -0.5 * z['Z6'] + (z['Z7'] - z['Z7'].transpose(0, 1, 3, 2))
    return z


# ======================================================================
# ph/ph block  M_ia,jb   (A31-A34)
# ======================================================================

def m_ss(be, gb, order, amps, zint, rho, d_ph, nocc, nvirt, vk=None):
    """M_ia,jb through `order`. d_ph[i,a,j,b] carries eps_a - eps_i on the
    (i,a)=(j,b) diagonal; it is applied to the identity pair so the zeroth
    order needs no index gymnastics in either backend."""
    vk = vk if vk is not None else VvvvKernels(be, gb.get('vvvv'))
    eye_o, eye_v = be.eye(nocc), be.eye(nvirt)
    M = be.scale(be.ein('ij,ab->iajb', eye_o, eye_v), d_ph)          # A31
    M = M - be.ein('jaib->iajb', gb['ovov'])                          # A32
    if order < 2:
        return M

    t1 = amps['t2_1']
    x_oo = 0.25 * be.ein('ikcd,jkcd->ij', gb['oovv'], t1)             # A33
    x_vv = 0.25 * be.ein('klac,klbc->ab', gb['oovv'], t1)
    x_ov = be.ein('ikac,jkbc->iajb', gb['oovv'], t1)
    M = M + _dab(be, x_oo + x_oo.transpose(1, 0), eye_v)
    M = M + _dij(be, x_vv + x_vv.transpose(1, 0), eye_o)
    M = M - 0.5 * (x_ov + x_ov.transpose(2, 3, 0, 1))
    if order < 3:
        return M

    t2, Z3, Z4, Z5, Z10 = (amps['t2_2'], zint['Z3'], zint['Z4'],
                           zint['Z5'], zint['Z10'])
    r_oo, r_ov, r_vv = rho['oo'], rho['ov'], rho['vv']

    A = (0.25 * be.ein('ikcd,jkcd->ij', gb['oovv'], t2)               # A34
         - be.ein('ikcd,jkcd->ij', t1, 0.5 * Z4 + 0.125 * Z3)
         - be.ein('ikjc,kc->ij', gb['ooov'], r_ov)
         - 0.5 * be.ein('ikjl,kl->ij', gb['oooo'], r_oo)
         - 0.5 * be.ein('icjd,cd->ij', gb['ovov'], r_vv))
    B = (0.25 * be.ein('klac,klbc->ab', gb['oovv'], t2)
         + 0.5 * be.ein('klac,klcb->ab', t1, Z4)
         - 0.125 * be.ein('klac,klbc->ab', t1, Z5)
         - be.ein('kbac,kc->ab', gb['ovvv'], r_ov)
         + 0.5 * be.ein('kalb,kl->ab', gb['ovov'], r_oo)
         + 0.5 * vk.vv_density(r_vv))
    t_vvvv = be.ein('klad,klbc->adbc', t1, t1)
    t_oooo = be.ein('ilcd,jkcd->iljk', t1, t1)
    W = 0.5 * Z3 + 0.5 * Z5 + Z4 + Z4.transpose(1, 0, 3, 2)
    C = (0.5 * be.ein('jcid,adbc->iajb', gb['ovov'], t_vvvv)
         + 0.5 * be.ein('lakb,iljk->iajb', gb['ovov'], t_oooo)
         + be.ein('ikac,jkbc->iajb', t1, W)
         - be.ein('ikac,jkbc->iajb', gb['oovv'], t2)
         - 2.0 * be.ein('jakc,ickb->iajb', gb['ovov'], Z10)
         - be.ein('iljk,kalb->iajb', gb['oooo'], Z10)
         - vk.z10(Z10)
         + 2.0 * be.ein('ikja,kb->iajb', gb['ooov'], r_ov)
         + 2.0 * be.ein('ibac,jc->iajb', gb['ovvv'], r_ov)
         - be.ein('jakb,ik->iajb', gb['ovov'], r_oo)
         + be.ein('jaic,bc->iajb', gb['ovov'], r_vv))
    M = M + _dab(be, A + A.transpose(1, 0), eye_v)
    M = M + _dij(be, B + B.transpose(1, 0), eye_o)
    return M + 0.5 * (C + C.transpose(2, 3, 0, 1))


def _dab(be, X_oo, eye_v):
    return be.ein('ij,ab->iajb', X_oo, eye_v)


def _dij(be, X_vv, eye_o):
    return be.ein('ij,ab->iajb', eye_o, X_vv)


# ======================================================================
# sigma pieces
# ======================================================================

def sigma_s_from_d(be, gb, amps, zint, Y, order):
    """W^a_i(D) -- A41 (first order) and A42 (second)."""
    w = (be.ein('jabc,ijbc->ia', gb['ovvv'], Y)
         + be.ein('jkib,jkab->ia', gb['ooov'], Y))
    if order < 2:
        return w
    t1, ZA, ZB = amps['t2_1'], zint['ZA'], zint['ZB']
    x_vv = be.ein('jkbd,jkbc->dc', t1, Y)
    x_oo = be.ein('klbc,jlbc->kj', t1, Y)
    return w + (-be.ein('jkib,jkab->ia', ZA, Y)
                + be.ein('jabc,ijbc->ia', ZB, Y)
                + be.ein('icad,dc->ia', gb['ovvv'], x_vv)
                + be.ein('ikja,kj->ia', gb['ooov'], x_oo))


def sigma_d_from_s(be, gb, amps, zint, y1, order):
    """W^ab_ij(S) -- A54 (first order) and A55 (second)."""
    W = (-0.5 * _p_ab(be.ein('ijka,kb->ijab', gb['ooov'], y1))
         - 0.5 * _p_ij(be.ein('icab,jc->ijab', gb['ovvv'], y1)))
    if order < 2:
        return W
    t1, ZA, ZB = amps['t2_1'], zint['ZA'], zint['ZB']
    u_vv = be.ein('kacd,kd->ac', gb['ovvv'], y1)
    w_oo = be.ein('klic,lc->ki', gb['ooov'], y1)
    W = W + 0.5 * _p_ab(be.ein('ijka,kb->ijab', ZA, y1)
                        + be.ein('ijbc,ac->ijab', t1, u_vv))
    return W + 0.5 * _p_ij(be.ein('jcab,ic->ijab', ZB, y1)
                           + be.ein('jkab,ki->ijab', t1, w_oo))


def sigma_d_from_d(be, gb, Y, d_ijab, order, vk=None):
    """W^ab_ij(D) -- A61 (Fock diagonal) and A63 (first order)."""
    vk = vk if vk is not None else VvvvKernels(be, gb.get('vvvv'))
    W = be.scale(Y, d_ijab)
    if order < 1:
        return W
    W = W + 0.5 * (vk.ladder(Y) + be.ein('ijkl,klab->ijab', gb['oooo'], Y))
    return W - _p_ij(_p_ab(be.ein('kaic,jkbc->ijab', gb['ovov'], Y)))


def _p_ab(X):
    return X - X.transpose(0, 1, 3, 2)


def _p_ij(X):
    return X - X.transpose(1, 0, 2, 3)
