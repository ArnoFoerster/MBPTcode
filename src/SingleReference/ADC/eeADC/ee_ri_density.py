"""The ISR-ADC(2) difference density of a folded root on density-fitted tiles.

ee_density writes the unrelaxed ISR-ADC(2) difference density once, in spin
orbitals, for the dense routes. Here the same terms (the names of
ee_density.TERMS) are evaluated for one spin channel of a closed shell from
the tiled operator of ee_ri_fold: the doubles of the root are rebuilt one
occupied tile at a time from its singles, never held whole, and so are the
first-order amplitudes.

Spin cases. The state is a flip eigenvector of parity s (+1 singlet, -1
Ms = 0 triplet), so its density is the same for both spins and the spatial
difference density is twice the alpha block. With the alpha singles x
(x = sqrt(T1) y / sqrt 2, y the unit channel vector, T1 the singles weight),
the spin-orbital doubles are, in the (i a j b) layout of a tile,

    Z_iajb = u2[i_a j_b a_a b_b]      W_iajb = u2[i_a j_a a_a b_a]
    singlet  Z = sqrt(T1) w (H + H^T) / (2 sqrt 2),   W = Z - Z~
    triplet  Z = sqrt(T1) w (H - H^T) / (2 sqrt 2),
             W = sqrt(T1) w (S - S~) / (2 sqrt 2),    S = H + H^T

with H, w = 1/(omega - D) and X~_iajb = X_ibja as in ee_ri_fold, so that
sum_all u2^2 = T1 |Y|^2 is the doubles weight of the full eigenvector; the
first-order amplitudes are tau_iajb = (ia|jb)/(e_a + e_b - e_i - e_j)
(opposite spins) and ts = tau - tau~ (same spin). Under the pair swap
(i a) <-> (j b) tau, ts and W are symmetric and Z has parity s, which moves
every contraction onto the tile's own first index. The spin sums are

    oo_dd   -2 sum_kab [W_iakb W_jakb + 2 Z_iakb Z_jakb]
    vv_dd    2 sum_ijc [W_iajc W_ibjc + 2 Z_iajc Z_ibjc]
    ov_sd   -2 sum_jb x_jb (W_iajb + s Z_iajb)
    r        T_s x,   (T_s v)_ia = sum_jb (ts + s tau)_iajb v_jb
    p0_oo   -1/2 sum_kab [ts_iakb ts_jakb + 2 tau_iakb tau_jakb]
    p0_vv    1/2 sum_ijc [ts_iajc ts_ibjc + 2 tau_iajc tau_ibjc]
    p0_ov    sum_P [sum_j B_jiP G_jaP - sum_c G_icP B_cPa] / (e_i - e_a),
             G_icP = sum_jb (ts + tau)_icjb B_jbP
    oo_tt_o -1/2 sym sum_klcd [ts_ickd ts_jcld + 2 tau_ickd tau_jcld] p1_lk
    oo_tt_v  sym sum_kcdb [ts_ickd ts_jckb + tau_ickd tau_jckb
                           + tau_idkc tau_jbkc] p1vv_db
    vv_tt_v -1/2 sym sum_klcd [ts_kblc ts_kald + 2 tau_kblc tau_kald] p1vv_cd
    vv_tt_o  sym sum_klcj [ts_kblc ts_jalc + tau_kblc tau_jalc
                           + tau_lbkc tau_lajc] p1_jk
    TU_oo    sum_kab [ts_iakb W_jakb + 2 tau_iakb Z_jakb]   (ov_tdo = -TU_oo x)
    TU_vv    sum_ijc [ts_iajc W_ibjc + 2 tau_iajc Z_ibjc]   (ov_tdv = -x TU_vv^T)
    ov_tsd  -T_+ p2_ov   (p2_ov is spin-symmetric whatever the state)

and the rest (oo_ss0, vv_ss0, oo_ru, vv_ru, oo_p0, vv_p0, oo_tru, vv_tru,
ov_p0v, ov_p0o) are products of these no-by-nv and square matrices.

These are the transition density <k|a^dag_p a_q|k> of ee_ri_transition at the
pair (k, k), s_l s_r = 1, and the density is evaluated by its passes: every
term but one contracts the tile's first occupied index away or keeps it, so
it is a sum over occupied tiles; oo_tt_o couples two occupied indices of
DIFFERENT amplitudes through p1_lk, which no occupied tile holds together,
and is a sum over tiles of one VIRTUAL index d instead. Each pass is about
one folded product's work, its tiles dealt over the operator's ranks and
summed in tile order (ee_ri_tiles.Pass), so the density is the same bits at
every rank count. A density is Hermitian, so its tt terms are symmetrized
here, which a transition density's are not.
"""
import numpy as np

from src.Base.numerics.constants import EE_RI_TILE_GB
from src.SingleReference.ADC.eeADC.ee_density import assemble
from src.SingleReference.ADC.eeADC.ee_ri_transition import transition_densities
from src.SingleReference.ADC.eeADC.ee_utils import symmetrize

#: the density's terms in the order they are summed into its matrix
ORDER = ('oo_ss0', 'vv_ss0', 'oo_ru', 'vv_ru', 'oo_p0', 'vv_p0', 'oo_tt_v',
         'vv_tt_v', 'vv_tt_o', 'ov_p0v', 'ov_p0o', 'oo_dd', 'vv_dd', 'ov_sd',
         'ov_tdv', 'ov_tdo', 'oo_tru', 'vv_tru', 'ov_tsd', 'oo_tt_o')
#: the terms symmetrized here (module docstring)
SYMMETRIZED = ('oo_tt_o', 'oo_tt_v', 'vv_tt_v', 'vv_tt_o')


def difference_density(op, y, omega, t1, terms=False, tile_gb=EE_RI_TILE_GB):
    """Delta rho (no + nv square, active spatial orbitals, both spins) of one
    root of the tiled operator `op` (ee_ri_fold.RIFoldADC2): y its unit
    singles in the channel basis, omega its frequency (Hartree), t1 its
    singles weight. terms=True returns {term: spatial block} (ee_density's
    names) instead. A spin-scaled operator has no unrelaxed (ISR) difference
    density and is refused. A collective over op's ranks: every rank calls it
    with the same arguments and gets the same bits."""
    no, nv = op.no, op.nv
    if op.level == 'adc1':
        x = (np.sqrt(t1 / 2.0) * np.asarray(y, float).ravel()).reshape(no, nv)
        both = {'oo_ss0': -2.0 * x @ x.T, 'vv_ss0': 2.0 * x.T @ x}
        return both if terms else assemble(both, no, nv)
    if op.scs is not None:
        raise ValueError('a spin-scaled ADC(2) operator has no unrelaxed (ISR) '
                         'difference density; take the relaxed one')
    (charge, _), = transition_densities({0: (op, y, omega, t1)}, pairs=[(0, 0)],
                                        terms=True, tile_gb=tile_gb).values()
    both = {name: symmetrize(charge[name]) if name in SYMMETRIZED else charge[name]
            for name in ORDER}
    return both if terms else assemble(both, no, nv)
