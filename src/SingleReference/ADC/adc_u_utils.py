"""Spin-orbital helpers shared by the adc_u_* route modules: MP1
denominators, the EN-dressed T2^(1) builder + its u1 shift (the
production open-shell EN route), and the matrix-free ingredient set."""
import numpy as np

from src.SingleReference.CC.cached_einsum import einsum as _cached_einsum
from src.SingleReference.DensityMatrix.generated_mpn import (
    mpn_density_pieces as mpn_gen)
from src.SingleReference.DensityMatrix.mpn_density_driver import (
    MPnDensityDriver, _denom, g_vvoo_df, t1_2_numerator_df)
from src.SingleReference.EpsteinNesbet.shifts import epstein_nesbet_denominator


def u2_denominators(eps, nocc):
    """(D_hhpp, D_pphh): the two bare MP1 doubles denominators the U^(2)
    blocks divide by."""
    eps_o, eps_v = eps[:nocc], eps[nocc:]
    d_hhpp = (eps_o[:, None, None, None] + eps_o[None, :, None, None]
              - eps_v[None, None, :, None] - eps_v[None, None, None, :])
    d_pphh = (eps_v[:, None, None, None] + eps_v[None, :, None, None]
              - eps_o[None, None, :, None] - eps_o[None, None, None, :])
    return d_hhpp, d_pphh


def u1_dressing_shift(s, nocc, t2_ijcd):
    """(nocc,nvirt) w array restoring exact u2_denom_dress parity for a
    dressed t2 fed to apply_U_2h1p/apply_U_2p1h (identically ~0 for bare
    t2): the exact residual of the telescoping (l1f - eps_p*S2) identity,
    one t1_2_numerator contraction on (t2_bare - t2_fed) per solve."""
    norb = s.norb
    o, v = slice(0, nocc), slice(nocc, norb)
    t2_fed_native = t2_ijcd.transpose(2, 3, 0, 1)          # -> (nv,nv,no,no)
    if s.g is not None:
        t2_bare_native = MPnDensityDriver(s.eps, s.g, nocc).compute_t2_1()
        num_delta = mpn_gen.t1_2_numerator(
            g=s.g, kd=np.eye(norb), o=o, v=v,
            nv=norb - nocc, no=nocc, t2_1=(t2_bare_native - t2_fed_native))
    else:
        t2_bare_native = g_vvoo_df(s.B_spin, nocc) / _denom(s.eps[o], s.eps[v], 2)
        num_delta = t1_2_numerator_df(s.B_spin, nocc,
                                      t2_bare_native - t2_fed_native)
    return num_delta.T                                      # w[j,a]


def dressed_t2_amplitudes(s, nocc, u2_denom_dress):
    """The (nocc,nocc,nvirt,nvirt) EN-dressed T2^(1) amplitude for the
    matrix-free dressing hook -- determinant-wise EN channels
    {'hh'/'pp'/'hp'}; 'spin_adapted': True is refused (CSF concept).
    g-free via B_spin when s.g is None."""
    if u2_denom_dress.get('spin_adapted', False):
        raise NotImplementedError(
            "u2_denom_dress={'spin_adapted': True} is a restricted/CSF concept "
            "with no meaning in this determinant-basis spin-orbital solver -- "
            "use ADCSolverRestricted.")
    o, v = slice(0, nocc), slice(nocc, s.norb)
    channels = {k: val for k, val in u2_denom_dress.items()
                if k in ('hh', 'pp', 'hp')}
    D_bare = _denom(s.eps[o], s.eps[v], 2)
    if s.g is not None:
        D_dressed = epstein_nesbet_denominator(D_bare, s.g, nocc, layout='pphh',
                                               bare_sign=1.0, **channels)
        t2n = MPnDensityDriver(s.eps, s.g, nocc).compute_t2_1_custom_denom(D_dressed)
    else:
        D_dressed = epstein_nesbet_denominator(D_bare, None, nocc, layout='pphh',
                                               bare_sign=1.0, B=s.B_spin, **channels)
        t2n = g_vvoo_df(s.B_spin, nocc) / D_dressed
    return t2n.transpose(2, 3, 0, 1)


def u1_shift_terms_2h1p(norb, nocc, z_p, Vfull, u1_shift):
    """(dy_shift, dy_p_shift) delta-structure terms for a dressed t2;
    (0.0, 0.0) when u1_shift is None."""
    if u1_shift is None:
        return 0.0, 0.0
    occ = slice(0, nocc)
    zp_o = z_p[occ]
    dy_shift = (zp_o[:, None, None] * u1_shift[None, :, :]
                - zp_o[None, :, None] * u1_shift[:, None, :])
    dy_p_shift = np.zeros(norb)
    # full-tensor adjoint of the delta structure is 2*sum w*Vfull
    # (antisymmetry collapses the two delta terms); the packed-config
    # adjoint carries the same 0.5 double-counting prefactor as
    # T0_adj etc. below, leaving a net factor 1 -- factor 2 here
    # breaks H's symmetry by O(w) (caught by an explicit symmetry
    # check; eigvalsh alone masks it via its lower-triangle read).
    dy_p_shift[occ] = _cached_einsum('ja,pja->p', u1_shift, Vfull, optimize=True)
    return dy_shift, dy_p_shift


def u1_shift_terms_2p1h(norb, nocc, z_p, Vfull, u1_shift):
    """2p1h mirror of u1_shift_terms_2h1p (internal -w sign)."""
    if u1_shift is None:
        return 0.0, 0.0
    virt = slice(nocc, norb)
    w2 = -u1_shift
    zp_v = z_p[virt]
    dy_shift = (zp_v[None, :, None] * w2[:, None, :]
                - zp_v[None, None, :] * w2[:, :, None])
    dy_p_shift = np.zeros(norb)
    # net factor 1, not 2 -- same packed-config 0.5 prefactor
    # argument as apply_U_2h1p's shift adjoint (see comment there).
    dy_p_shift[virt] = _cached_einsum('ib,iab->a', w2, Vfull, optimize=True)
    return dy_shift, dy_p_shift


def build_matrix_free_ingredients(s, nocc):
    """Uncached ingredient set (dims/config indices/K diagonals + the
    B-factor slices in DF mode, dense g slices otherwise); rank<=3 only
    in DF mode. Cache through s._build_matrix_free_ingredients."""
    eps = s.eps
    norb = s.norb
    occ, virt = slice(0, nocc), slice(nocc, norb)
    nvirt = norb - nocc

    d = s.dimensions(nocc)
    i_2h, j_2h, a_2h = s._configs_2h1p(nocc)
    i_2p, a_2p, b_2p = s._configs_2p1h(nocc)
    iu, ju = np.triu_indices(nocc, k=1)
    au, bu = np.triu_indices(nvirt, k=1)
    npair_o, npair_v = len(iu), len(au)
    off_2h1p = norb
    off_2p1h = off_2h1p + d['n2h1p']

    K_2h1p = eps[i_2h] + eps[j_2h] - eps[a_2h]
    K_2p1h = eps[a_2p] + eps[b_2p] - eps[i_2p]

    B = s.B_spin
    ing = {
        'd': d, 'nvirt': nvirt, 'npair_o': npair_o, 'npair_v': npair_v,
        'iu': iu, 'ju': ju, 'au': au, 'bu': bu,
        'off_2h1p': off_2h1p, 'off_2p1h': off_2p1h,
        'K_2h1p': K_2h1p, 'K_2p1h': K_2p1h,
    }
    if B is not None:
        ing.update({
            'g_oooo': None, 'g_cvov': None, 'g_vvvv': None, 'g_ovov': None,
            'B_v': B[:, virt][:, :, virt],        # (naux,V,V)
            'B_v_full': B[:, virt, :],             # (naux,V,norb)
            'B_o_full': B[:, occ, :],               # (naux,O,norb)
            'B_ov': B[:, occ][:, :, virt],          # (naux,O,V)
            'B_oo': B[:, occ][:, :, occ],           # (naux,O,O)
            'B_vo': B[:, virt][:, :, occ],          # (naux,V,O)
        })
    else:
        g = s.g
        ing.update({
            'g_oooo': g[occ, occ, occ, occ],
            'g_cvov': g[virt, occ, virt, occ],
            'g_vvvv': g[virt, virt, virt, virt],
            'g_ovov': g[occ, virt, occ, virt],
            'B_v': None, 'B_v_full': None, 'B_o_full': None,
            'B_ov': None, 'B_oo': None, 'B_vo': None,
        })
    return ing
    return ing
