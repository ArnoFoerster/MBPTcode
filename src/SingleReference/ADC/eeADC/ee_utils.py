"""Shared spin-orbital ingredients for the polarization-propagator (EE) ADC
route modules -- configuration spaces, MP amplitudes, the Z contraction
intermediates and the second-order ground-state density.

Working equations follow Leitner, Dempwolff, Dreuw, J. Chem. Phys. 157,
184101 (2022) (references/adc/leitner-dempwolff-dreuw-2022-adc4-polarization-propagator.pdf), whose ADC(4) appendix is
resolved order by order and therefore contains ADC(1)/(2)/(2)-x/(3) by
truncation with Schirmer's block-order rule order(I,J) = n - mu_I - mu_J
(mu = 0 singles, 1 doubles; see adc_order_rule.blocks_at, which is
particle-number agnostic and reproduces the paper's Fig. 1 exactly):

    n | ph/ph | ph/2p2h | 2p2h/2p2h        equations (paper labels)
    1 |   1   |    -    |    -             A31 A32
    2 |   2   |    1    |    0             + A33 | A41/A54 | A61
    2x|   2   |    1    |    1             + A63
    3 |   3   |    2    |    1             + A34 | A42/A55 | A63

Index convention throughout: spin orbitals, occupied first, alpha/beta
interleaved (get_antisymmetrized_spin_eri's layout). `g[p,q,r,s] = <pq||rs>`
(physicist, antisymmetrized). Doubles amplitudes are stored as the FULL
antisymmetric tensor t[i,j,a,b] = t^ab_ij; the paper's sums run over full
index ranges and every formula here is transcribed in that convention. The
ADC vector space, by contrast, uses the UNIQUE configurations i<j, a<b (an
orthonormal ISR basis) -- the two are bridged in the route modules.

Amplitude sign convention (paper, Appendix A preamble): doubles denominators
are (eps_a + eps_b - eps_i - eps_j); singles/triples/quadruples carry the
opposite sign, and the singles denominator additionally carries a factor 2.
"""
import numpy as np

from src.SingleReference.CC.cached_einsum import einsum as _einsum


# ----------------------------------------------------------------------
# configuration spaces
# ----------------------------------------------------------------------

def dimensions(nocc, norb):
    """Segment sizes of the EE-ADC vector space (singles then doubles).

    Singles: (i, a), i outer / a inner -- nocc*nvirt.
    Doubles: (i<j, a<b), hole pair outer / particle pair inner, matching
    adc_u_dense_full's 2h1p/2p1h flattening idiom."""
    nvirt = norb - nocc
    n_s = nocc * nvirt
    n_d = (nocc * (nocc - 1) // 2) * (nvirt * (nvirt - 1) // 2)
    return {'norb': norb, 'nocc': nocc, 'nvirt': nvirt,
            'n_s': n_s, 'n_d': n_d, 'nH': n_s + n_d}


def configs_singles(nocc, norb):
    """(i, a) index arrays for the singles space, i outer / a inner."""
    occ = np.arange(nocc)
    virt = np.arange(nocc, norb)
    return (np.repeat(occ, len(virt)), np.tile(virt, nocc))


def configs_doubles(nocc, norb):
    """(i, j, a, b) index arrays for the doubles space with i<j, a<b,
    hole pair outer / particle pair inner."""
    occ = np.arange(nocc)
    virt = np.arange(nocc, norb)
    iu, ju = np.triu_indices(nocc, k=1)
    au, bu = np.triu_indices(len(virt), k=1)
    npo, npv = len(iu), len(au)
    return (np.repeat(occ[iu], npv), np.repeat(occ[ju], npv),
            np.tile(virt[au], npo), np.tile(virt[bu], npo))


def spin_labels(mf, nocc, norb):
    """+1/-1 (alpha/beta) per spin orbital, in the layout the route uses.

    RHF: interleaved, so alpha is the even index. UHF: block-stacked
    [occ_a, occ_b, virt_a, virt_b] (see ee_driver.spin_orbital_arrays)."""
    from pyscf import scf as _scf
    if not isinstance(mf, _scf.uhf.UHF):
        return np.where(np.arange(norb) % 2 == 0, 1, -1)
    na, nb = mf.nelec
    nmo_a = mf.mo_coeff[0].shape[1]
    sz = np.empty(norb, dtype=int)
    sz[:na] = 1                      # occupied alpha
    sz[na:nocc] = -1                 # occupied beta
    sz[nocc:nocc + (nmo_a - na)] = 1  # virtual alpha
    sz[nocc + (nmo_a - na):] = -1     # virtual beta
    return sz


def ms_sector_mask(sz, nocc, norb, delta_ms=0):
    """Boolean mask over the configuration basis keeping one Delta-Ms sector.

    Sz commutes with H, so the supermatrix is block diagonal in Delta Ms and
    the sector is simply a SELECTION of configurations -- no new equations.
    Delta Ms = 0 is the spin-conserving sector, the one that carries physical
    excitations; the others hold the other Ms components of the same
    multiplets (for a closed shell, the Ms = +-1 partners of every triplet;
    for an open shell, also the near-zero root that is just the other Ms
    component of the REFERENCE, which is why an unrestricted spin-orbital run
    shows spurious ~0 eV 'excitations' if the sector is not imposed)."""
    i_s, a_s = configs_singles(nocc, norb)
    keep_s = (sz[a_s] - sz[i_s]) == 2 * delta_ms
    i_d, j_d, a_d, b_d = configs_doubles(nocc, norb)
    keep_d = (sz[a_d] + sz[b_d] - sz[i_d] - sz[j_d]) == 2 * delta_ms
    return np.concatenate([keep_s, keep_d])


def pair_indices(nocc, norb):
    """The (iu, ju) / (au, bu) triangular index pairs behind configs_doubles,
    for route modules that need to fold a full (i,j,a,b) tensor down to the
    unique-configuration axis."""
    iu, ju = np.triu_indices(nocc, k=1)
    au, bu = np.triu_indices(norb - nocc, k=1)
    return iu, ju, au, bu


def fold_doubles(T, nocc, norb):
    """Full antisymmetric (nocc,nocc,nvirt,nvirt) tensor -> (n_d,) vector on
    the unique i<j, a<b configurations (hole pair outer, particle pair
    inner)."""
    iu, ju, au, bu = pair_indices(nocc, norb)
    return T[iu[:, None], ju[:, None], au[None, :], bu[None, :]].reshape(-1)


PAPER_DOUBLES_SCALE = 2.0
"""Bridge between this repo's ADC vector space and the paper's amplitude vector.

The ADC eigenvalue problem is posed in the ORTHONORMAL basis of unique
doubly-excited ISR configurations (i<j, a<b). The paper instead stores the
doubles part as the full antisymmetric tensor Y^ab_ij and normalizes it with
UNRESTRICTED sums, Y'Y = 1 over all ijab (its Eq. 13) -- so each of the four
tensor entries belonging to one configuration is HALF the orthonormal
coefficient. The isometry between the two is

    U(y)     = unfold_doubles(y) / 2        (orthonormal -> paper)
    U^dag(T) = 2 * fold_doubles(T)          (paper -> orthonormal)

Consequences, each verified numerically:
  * doubles/doubles: the two factors cancel, so the paper's D-row equations
    restricted to unique configurations ARE the orthonormal block;
  * doubles/singles: multiply by 2 (matches exact Slater-Condon at 1e-15);
  * singles/doubles: divide by 2, which then equals the doubles/singles
    transpose -- Hermiticity is the check that pins the whole convention.

Getting this wrong is silent: the supermatrix stays symmetric and only the
coupling strength is off, which looks like a plausible-but-wrong ADC(2).
"""


def unfold_doubles(v, nocc, norb):
    """(n_d,) unique-configuration vector -> full antisymmetric
    (nocc,nocc,nvirt,nvirt) tensor (the inverse embedding of fold_doubles)."""
    nvirt = norb - nocc
    iu, ju, au, bu = pair_indices(nocc, norb)
    T = np.zeros((nocc, nocc, nvirt, nvirt), dtype=v.dtype)
    blk = v.reshape(len(iu), len(au))
    T[iu[:, None], ju[:, None], au[None, :], bu[None, :]] = blk
    T[ju[:, None], iu[:, None], au[None, :], bu[None, :]] = -blk
    T[iu[:, None], ju[:, None], bu[None, :], au[None, :]] = -blk
    T[ju[:, None], iu[:, None], bu[None, :], au[None, :]] = blk
    return T


# ----------------------------------------------------------------------
# integral blocks
# ----------------------------------------------------------------------

def g_blocks(g, nocc, norb):
    """The six <pq||rs> occupied/virtual blocks the ADC(3) equations touch."""
    o = slice(0, nocc)
    v = slice(nocc, norb)
    return {
        'oooo': g[o, o, o, o], 'ooov': g[o, o, o, v], 'oovv': g[o, o, v, v],
        'ovov': g[o, v, o, v], 'ovvv': g[o, v, v, v], 'vvvv': g[v, v, v, v],
    }


# ----------------------------------------------------------------------
# MP amplitudes (paper A3-A5) and the second-order density (A21-A23)
# ----------------------------------------------------------------------

def _denominators(eps, nocc):
    """(d_ijab, d_ia, d_ph): the spin-free weight arrays every route shares.

    Paper Appendix-A preamble: the doubles denominator is
    eps_a + eps_b - eps_i - eps_j while singles carry the opposite sign AND an
    extra factor 2. d_ph holds eps_a - eps_i on the ph diagonal."""
    eo, ev = eps[:nocc], eps[nocc:]
    d_ijab = (ev[None, None, :, None] + ev[None, None, None, :]
              - eo[:, None, None, None] - eo[None, :, None, None])
    d_ia = 2.0 * (eo[:, None] - ev[None, :])
    d_ph = np.broadcast_to((ev[None, :] - eo[:, None])[:, :, None, None],
                           (nocc, len(ev), nocc, len(ev)))
    return d_ijab, d_ia, d_ph


def mp_amplitudes(eps, g, nocc, norb, order=3, denominators=None):
    """{'t2_1', 't1_2', 't2_2'} in the spin-orbital basis (A3-A5).

    `denominators` optionally overrides (d_ijab, d_ia) -- the hook the
    Epstein-Nesbet dressing uses (ee_en.en_denominators)."""
    from src.SingleReference.ADC.eeADC import ee_equations as _eq
    gb = g_blocks(g, nocc, norb)
    d_ijab, d_ia, _ = _denominators(eps, nocc)
    en_shift = None
    if denominators is not None:
        d_en, d_ia = denominators
        # what the dressing resummed, removed again from the t2^(2) numerator
        # -- see ee_equations.amplitudes
        en_shift = d_en - d_ijab
        d_ijab = d_en
    out = _eq.amplitudes(_eq.NUMPY, gb, d_ijab, d_ia, order=order,
                         en_shift=en_shift)
    out['d_ijab'] = d_ijab
    return out


def mp_density2(amps):
    """Second-order ground-state density blocks (A21-A23)."""
    from src.SingleReference.ADC.eeADC import ee_equations as _eq
    return _eq.density2(_eq.NUMPY, amps)


def z_intermediates(g, amps, nocc, norb, order=3):
    """Z^(1..7,10) and the compounds Z^(A), Z^(B) (A11-A20).

    ADC(2) needs none of these. NOTE the sign on the Z^(2) term of Z^(A):
    the paper prints

        Z^(A)_ijka = 1/2 Z^(1)_ijka + P^-_ij Z^(2)_ijka        (A19)

    with P^-_pq = 1 - P_pq (A1), i.e. +(Z2[i,j,k,a] - Z2[j,i,k,a]). That form
    is NOT third-order exact: ADC(3) then carries a residual O(lambda^3) error
    of ~9e-3 (HF) / 1.2e-2 (LiH) against the exact determinant-space
    lambda-expansion, and its energies miss pyscf's EE-ADC(3). Flipping this
    one sign takes the lambda^3 defect to the numerical floor (~1.5e-5) and
    reproduces pyscf exactly; every other sign combination of
    (Z^(A), Z^(B)) is worse by two to three orders of magnitude. The likely
    printed slip is an i<->j transposition in A12 rather than the sign in A19
    -- the two are equivalent here. Z^(B) (A20) stands as printed.
    Gated by tests/test_ee_adc_spinorbital.py::test_third_order_lambda_exact.
    """
    if order < 3:
        return {}
    from src.SingleReference.ADC.eeADC import ee_equations as _eq
    return _eq.z_intermediates(_eq.NUMPY, g_blocks(g, nocc, norb), amps)
