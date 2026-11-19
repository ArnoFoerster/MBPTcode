"""The direct RPA as the low-level method of the downfolding.

`run` solves the RPA of the whole system once, before any active space is
known, and keeps what every later active space needs: on the casida route
the excitation energies Omega, the amplitudes X+Y, the full correlation
energy E_RPA,corr = 1/2 (sum Omega - Tr A), and the integrals the Casida
matrices were built from. For a restricted closed-shell reference the
response is the singlet one (A = delta + 2K, B = 2K, K_ia,jb = (ia|jb));
triplet modes contribute nothing to the direct RPA.

An unrestricted reference (UHF/UKS) is two spin channels, each with its own
orbitals, orbital energies and occupied count, and one RPA problem over the
particle-hole pairs of both spins, stacked: the alpha pairs in C order over
(i, a), then the beta ones, with A = delta + K, B = K and
K_(ia s),(jb s') = (i_s a_s | j_s' b_s'). Every formula below is written
once with the spin weight g, the number of electrons an occupied orbital of
one channel holds: g = 2 for the one channel of a restricted reference,
g = 1 for each of two. A + B = delta + 2g K, the static response is
-2g [(A+B)^-1], and W is one spin-independent operator.

`screening_correction` gives the static (omega = 0) screening correction of an
active space on the casida route: the active pairs deleted from A+B and one
Cholesky solve, or for 'mrpa' (casida only) the interaction that keeps the
full RPA's zeroth moment eta0 = (X+Y)(X+Y)^T on the active ph pairs
(Scott and Booth's Eq. 5, from the kept X+Y). `terms` assembles an active
space's DownfoldingTerms from it: vtilde, the correlation-energy pair
E_c[v] - E_c^A[v_eff] (both by the plasmon formula, on the same v) and the
static defect |W(0) - W^A[v_eff](0)|, zero for crpa.

On the imagfrequency route (density fitted, N^4) `run` keeps the existing
imaginary-axis energy, its minimax frequency grid and the static
polarizability P(0) of the whole system in the normalized fitting basis.
The screening correction is P - P^A and one Dyson step, with P^A the active
ph pairs' part of P(0) (`active_polarizability_static`), projected on the
factors of every active pair (`active_pair_factors`): the casida route's
deletion written another way, equal to rounding on the same three-index
object. E_c^A[v_eff] is the same quadrature on the same grid with the bare
response of the active ph pairs, so the E_c pair is one formula on both
sides; the dense-Casida E_c^A is kept as the 'ec_active_casida' diagnostic.

The space-time route (separable ISDF factors (X, D), imaginary time, N^3,
serial) is the same on its own machinery: `run` keeps the factors, the static
tau grid with P(0) on it, and the energy of `rpa_correlation_energy_space_time`
with its time-frequency grid. P^A(0) is built as P(0) was, with the active
columns of X on the same grid, so with every orbital active P - P^A is zero
to the bit; E_c^A takes the active ph pairs' exp(-delta tau) through the
energy's own tau -> omega transform.

`embedding_self_energy` gives the static environment self-energy
Sigma_bar^E on the casida route: the residues of the full W
between the active and the environment orbitals (`screened_residues`) and
the mean-field energies, reduced by Marie and Loos's SRG (Eq. 44, 'srg') or
by mode A at eta = 0 (Eq. 12, 'mode_a', which warns near the poles,
`pole_window`). `terms(..., sigma=...)` hands it to the general layer in
both spin channels with the Hartree-only double counting (dc='hartree'), and
records the 'sigma_poles' diagnostic.

On imagfrequency there are no residues: S^E(mu + i w) comes from the
Lorentzian fit of W on the minimax W grid with the internal orbital
restricted to the environment (`sigma_env_imag_axis`, W cached on the first
request), is continued by one Pade fit per element, and each element is
evaluated at its own point (`continued_self_energy`): Eq. 42, Re S^E_tu(z_tu),
for 'srg' (s = inf; srg_s enters the pole check alone), the real axis for
'mode_a'. Both warn near the poles and point to casida; Omega_min of the
window is the lowest root of the same Casida problem (`lowest_excitation`).
The 'sigma_continuation' diagnostic is the spread of two Pade fits on the
even and the odd nodes. Space-time samples S^E the same way, on its own
objects: W on the same minimax grid from chi0 of the ISDF factors, and the
active-environment fitting factors B_Q,tr = sum_P D_PQ X_Pt X_Pr. Sampling
the space-time GW self-energy through its imaginary-time quadrature instead
left about 1e-7 of noise on the nodes, which the continuation amplified to
1e-4 and more for orbitals far from the gap; the fit's samples are an exact
rational function and continue cleanly, at N^3 times the active orbitals.
"""
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
from pyscf import scf
from scipy.linalg import cho_factor, cho_solve

from src.Base.factorization.sliced_factors import unpack_factors
from src.Base.numerical_flags import collecting, flag
from src.Base.numerics.constants import (BSE_DAVIDSON_NROOTS, BYTES_PER_GB,
                                         DOWNFOLDING_DENSE_OMEGA_MAX_PAIRS,
                                         DOWNFOLDING_OMEGA_DAVIDSON_GUESS,
                                         DOWNFOLDING_SIGMA_CONTINUATION_WARN,
                                         DOWNFOLDING_SIGMA_TILE_GB,
                                         DOWNFOLDING_SRG_DAMPING_WARN,
                                         DOWNFOLDING_SRG_S,
                                         GW_DENSITY_W_FIT_TIKHONOV)
from src.Base.numerics.numerics import numerics_or_default, rpa_quadrature
from src.Base.orbital_spaces import particle_hole_window
from src.Base.pyscf.pyscf_interface import (get_density_fitting_coefficients,
                                            get_orbital_energies,
                                            get_two_electron_integrals_chemist)
from src.Base.utils.grids.analytic_continuation import (pade_eval_elementwise,
                                                        pade_fit_elementwise,
                                                        pade_nodes)
from src.Base.utils.grids.grids import gap_scaled_w0, gauss_legendre_grid
from src.Base.utils.grids.time_frequency import (TimeFrequencyGrid,
                                                 minimax_points_for_accuracy)
from src.SingleReference.common.active_space import closed_shell_nocc
from src.SingleReference.GW.imaginary_axis import (
    fitted_self_energy_matrix_imaginary_axis, screening_fit_grid)
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.davidson import solve_casida_davidson
from src.SingleReference.LinearResponse.imaginary_frequency import \
    frequency_factor
from src.SingleReference.LinearResponse.linear_response import \
    LinearResponseSolver
from src.SingleReference.LinearResponse.rpa_energy import (
    imaginary_axis_energy_grid, rpa_correlation_energy_imaginary_axis,
    solve_polarizability_imaginary_axis)
from src.SingleReference.LinearResponse.space_time import (
    chi0_aux, chi0_imaginary_frequency, dielectric_log_determinant,
    dyson_screening, layout_tile_gb, rpa_correlation_energy_space_time,
    static_screening_grid)

from ..partition import active_pair_mask, channels_of, ph_element_mask
from ..terms import DownfoldingTerms, choice

METHOD = 'rpa'
MODES = ('casida', 'imagfrequency', 'space-time')
SCREENINGS = ('bare', 'crpa', 'crpa_ph', 'mrpa')
SIGMAS = (None, 'srg', 'mode_a')
DIAGNOSTICS = ('static_defect', 'ec_active_casida', 'sigma_poles',
               'sigma_continuation')
SPINS = ('alpha', 'beta')
# The spin channels (first pair, second pair) of the two-body blocks: the one
# block of a restricted reference, or (aa, ab, bb).
_BLOCKS = {1: ((0, 0),), 2: ((0, 0), (0, 1), (1, 1))}


@dataclass(eq=False)
class RPALowLevel:
    """The RPA low-level result: the mean field, the full RPA, and a cache.

    For an unrestricted reference (`unrestricted`) eps, nocc and factors are
    pairs (alpha, beta), eri is the three blocks (aa, ab, bb) with
    ab[p,q,r,s] = (p_a q_a | r_b s_b), and omega and xpy are those of the
    one RPA problem on the stacked pairs of both spins, the alpha pairs in
    C order over (i, a), then the beta ones, where
    xpy diag(1/omega) xpy^T = (A+B)^-1 = (delta + 2K)^-1. What follows
    describes the one channel of a restricted reference.

    eps, nocc: the orbital energies and the occupied count of `mf`.
    e_c_full: E_RPA,corr of the whole system: on casida 1/2 (sum omega -
        Tr A), on imagfrequency the minimax quadrature of
        `rpa_correlation_energy_imaginary_axis`, on space-time
        `rpa_correlation_energy_space_time`.
    omega: (n_pair,) the RPA excitation energies, ascending; None on
        imagfrequency and space-time, which do not diagonalize.
    xpy: (n_pair, n_pair) X+Y, one excitation per column; the rows are the
        n_pair = nocc*nvirt (i, a) pairs in C order, i over all occupied and
        a over all virtual orbitals, the order of `active_pair_mask`.
        Normalized as the Casida solver leaves it (X^T X - Y^T Y = 1), which
        makes, with factor 1,
            xpy diag(1/omega) xpy^T = (A+B)^-1 = (delta + 4K)^-1,
        so the static density response is chi(0)_ia,jb = -4 [(A+B)^-1]_ia,jb
        = -4 sum_n xpy_ia,n xpy_jb,n / omega_n (tests/test_downfolding_two_body.py
        pins the factor). None on imagfrequency and space-time.
    factors: the three-index object B[Q, p, q] of the MOs, (naux, nmo, nmo),
        with (pq|rs) = sum_Q B_Q,pq B_Q,rs, when df; else None, and None on
        space-time, which never builds it whole (`active_pair_factors`).
    eri: the exact chemist MO integrals (pq|rs) when not df; else None.
    cache: what a later `terms` call builds the first time it needs it. On
        imagfrequency `run` puts there 'grid', the (points, weights) of the
        energy's minimax frequency quadrature, and 'p0', P(0) = 1 - W(0)^-1
        = chi0(0) = -4 B_ov diag(1/delta) B_ov^T (naux, naux) in the
        normalized fitting basis (the convention of
        `solve_polarizability_imaginary_axis`). The first Sigma^E request
        adds 'omega_min' (`lowest_excitation`) and 'w_grid', the (points, W)
        of `_screening_grid`. On space-time `run` puts there 'factors',
        'static_grid', 'p0' and 'energy_grid' (`_space_time`); the first
        Sigma^E request adds 'omega_min' and 'w_grid', the same (points, W)
        on the ISDF factors. 'flags' holds, per entry added on a request,
        the named flags raised while building it, which every later active
        space that uses the entry gets too (`_cached`).
    numerical_flags: the named flags raised during the run, filled by
        `run_low_level`.
    """
    mf: object
    mode: str
    df: bool
    numerics: object
    eps: np.ndarray
    nocc: int
    e_c_full: float
    omega: Optional[np.ndarray]
    xpy: Optional[np.ndarray]
    factors: Optional[np.ndarray]
    eri: Optional[np.ndarray]
    cache: dict = field(default_factory=dict)
    method: str = METHOD
    numerical_flags: list = field(default_factory=list)

    @property
    def unrestricted(self):
        """True for a UHF/UKS reference: two spin channels."""
        return isinstance(self.nocc, tuple)


def _refuse_options(opts, which):
    if opts:
        raise TypeError(f"method={METHOD!r} {which} does not take "
                        + ', '.join(sorted(opts)) + '.')


@dataclass(frozen=True)
class _Channel:
    """One spin channel of an active-space request: the orbital energies,
    the occupied count and the three-index factors (None without) of that
    spin, its Partition, and its name, 'alpha' or 'beta' (None for the one
    channel of a restricted reference)."""
    eps: np.ndarray
    nocc: int
    factors: Optional[np.ndarray]
    part: object
    spin: Optional[str]


def _channels(low, part):
    """The spin channels of `low` with the partitions of `part`: one, for a
    restricted result and one Partition; two, for an unrestricted result and
    the pair (alpha, beta). Anything else is refused."""
    parts = channels_of(part)
    if not low.unrestricted:
        if len(parts) != 1:
            raise ValueError(
                f'a restricted low-level result takes one Partition; got '
                f'{len(parts)}. A pair (alpha, beta) goes with an '
                f'unrestricted (UHF/UKS) one.')
        return (_Channel(low.eps, low.nocc, low.factors, parts[0], None),)
    if len(parts) != 2:
        raise ValueError(
            'an unrestricted (UHF/UKS) low-level result takes a pair of '
            'Partitions, (alpha, beta), one per spin; got one.')
    factors = (None, None) if low.factors is None else low.factors
    return tuple(_Channel(low.eps[s], low.nocc[s], factors[s], parts[s],
                          SPINS[s]) for s in (0, 1))


def _spin_weight(chans):
    """g, the electrons an occupied orbital of one channel holds: 2.0 for
    the one channel of a restricted reference, 1.0 for each of two."""
    return 2.0 if len(chans) == 1 else 1.0


def _join(arrays, axis=0):
    """The arrays of the spin channels one after the other along `axis`; the
    array of one channel as it is, not a copy."""
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=axis)


def _square(rows):
    """The matrix of the blocks rows[s][t] over the spin channels; the one
    block of one channel as it is."""
    return rows[0][0] if len(rows) == 1 else np.block(rows)


def _per_channel(values):
    """What a public function returns for a quantity of each spin channel:
    the one value of a restricted reference, the pair (alpha, beta) of an
    unrestricted one."""
    return values[0] if len(values) == 1 else tuple(values)


def _orbital(chan):
    """How a message names an orbital of the channel: 'orbital', or
    'alpha orbital' / 'beta orbital' for an unrestricted reference."""
    return 'orbital' if chan.spin is None else f'{chan.spin} orbital'


def _window_text(chans, windows):
    """'window (lo, hi)' or, for two channels, 'windows alpha (lo, hi) and
    beta (lo, hi)', for the messages."""
    if len(chans) == 1:
        return 'window ({:.6f}, {:.6f})'.format(*windows[0])
    return 'windows ' + ' and '.join(
        f'{c.spin} ({lo:.6f}, {hi:.6f})' for c, (lo, hi) in zip(chans, windows))


def _per_block(blocks):
    """What a public function returns for the two-body blocks: the one array
    of a restricted reference, the tuple (aa, ab, bb) of an unrestricted
    one."""
    return blocks[0] if len(blocks) == 1 else tuple(blocks)


def _two_body_blocks(v, chans):
    """The two-body blocks a public function is handed, as a tuple: the one
    (n_act,)*4 array of a restricted reference, or the three (aa, ab, bb)
    of an unrestricted one. Refused when they do not fit the channels."""
    blocks = (v,) if isinstance(v, np.ndarray) else tuple(v)
    if len(blocks) != len(_BLOCKS[len(chans)]):
        raise ValueError(
            'an unrestricted reference takes the three blocks (aa, ab, bb) '
            'of the active interaction, a restricted one a single array; '
            f'got {len(blocks)} for {len(chans)} spin channel(s).')
    return blocks


def _spin_block(blocks, s, t):
    """The block of `blocks` ((v,) or (aa, ab, bb)) with the first pair in
    spin channel s and the second in channel t; the beta-alpha one is the
    alpha-beta one with its pairs exchanged."""
    if len(blocks) == 1:
        return blocks[0]
    aa, ab, bb = blocks
    return ((aa, ab), (ab.transpose(2, 3, 0, 1), bb))[s][t]


def _delta(chan, mask=None):
    """eps_a - eps_i of the (i, a) pairs of one channel in C order, those of
    the bool mask when given."""
    delta = (chan.eps[chan.nocc:][None, :]
             - chan.eps[:chan.nocc][:, None]).ravel()
    return delta if mask is None else delta[mask]


def _ov_factors(chan):
    """B_Q,ia (naux, n_pair) of one channel, the (i, a) pairs in C order."""
    b = chan.factors
    return b[:, :chan.nocc, chan.nocc:].reshape(b.shape[0], -1)


def _act_factors(chan):
    """B_Q,tu (naux, n_act^2) of the active orbitals of one channel."""
    b, act = chan.factors, chan.part.act
    return b[:, act][:, :, act].reshape(b.shape[0], len(act) ** 2)


def _eri_block(low, s, t):
    """The exact chemist (p q|r s) of low.eri with p, q in spin channel s
    and r, s in channel t."""
    if not low.unrestricted:
        return low.eri
    aa, ab, bb = low.eri
    return ((aa, ab), (ab.transpose(2, 3, 0, 1), bb))[s][t]


def _residual_integrals(low, chans, res):
    """(K, V) on the stacked pairs of `res`, one bool mask over the (i, a)
    pairs per channel: K_ia,jb = (ia|jb) between them, and V, one array per
    channel t, V[t][ia, tu] = (ia|tu) with t, u the active orbitals of that
    channel. From the integrals of the full RPA, low.factors or low.eri."""
    n = chans[0].part.n_act
    if low.factors is not None:
        b_res = _join([_ov_factors(c)[:, r] for c, r in zip(chans, res)],
                      axis=1)
        return b_res.T @ b_res, [b_res.T @ _act_factors(c) for c in chans]
    k_rows, v_cols = [], [[] for _ in chans]
    for s, (chan_s, res_s) in enumerate(zip(chans, res)):
        k_rows.append([])
        for t, (chan_t, res_t) in enumerate(zip(chans, res)):
            eri = _eri_block(low, s, t)
            ov = eri[:chan_s.nocc, chan_s.nocc:].reshape(
                res_s.size, *eri.shape[2:])[res_s]
            k_rows[s].append(ov[:, :chan_t.nocc, chan_t.nocc:].reshape(
                len(ov), -1)[:, res_t])
            act = chan_t.part.act
            v_cols[t].append(ov[:, act][:, :, act].reshape(len(ov), n * n))
    return _square(k_rows), [_join(col) for col in v_cols]


def run(mf, mode='casida', df=True, numerics=None, **opts):
    """The RPALowLevel of `mf`: the full direct RPA of `mode`, solved once.

    mode: 'casida' | 'imagfrequency' | 'space-time'.
    df: True builds everything from the mean field's density fit (for a
        mean field without one, the helper's exact eigendecomposition of the
        MO integrals, eigenvalues below 1e-12 dropped); on space-time from
        the separable (ISDF) factors instead. False, casida only: the exact
        4-centre integrals; imagfrequency is density fitted and space-time
        ISDF fitted by construction, and both refuse it.
    numerics: the run's Numerics (None: the defaults), kept on the result
        and read by the eigensolve (casida), the frequency count of the
        energy (imagfrequency, `rpa_quadrature`), or on space-time the ISDF
        grid (numerics.isdf, which has no default: None is refused there)
        and the tau and frequency counts of `_space_time`.

    On casida E_RPA,corr comes from the same solve as Omega and X+Y; on
    imagfrequency from `rpa_correlation_energy_imaginary_axis` on the
    minimax grid kept in low.cache['grid'], with P(0) in low.cache['p0'];
    on space-time from `rpa_correlation_energy_space_time` (`_space_time`).
    """
    _refuse_options(opts, 'run')
    mode = choice(mode, MODES, 'mode')
    df = bool(df)
    if mode != 'casida' and not df:
        how = ('density fitted' if mode == 'imagfrequency'
               else 'ISDF fitted (separable factors)')
        raise ValueError(
            f"method={METHOD!r} mode={mode!r} is {how}; df=False (the exact "
            f"4-centre screening) is mode='casida' only.")
    if isinstance(mf, scf.uhf.UHF):
        if mode != 'casida':
            raise NotImplementedError(
                f"method={METHOD!r} mode={mode!r} takes a restricted "
                f"reference only; for an unrestricted one (UHF/UKS) use "
                f"mode='casida'.")
        eps = tuple(np.asarray(e) for e in get_orbital_energies(mf))
        nocc = tuple(closed_shell_nocc(occ) for occ in mf.mo_occ)
    else:
        eps = np.asarray(get_orbital_energies(mf))
        nocc = closed_shell_nocc(mf.mo_occ)
    if mode == 'space-time':
        e_c, cache = _space_time(mf, eps, nocc, numerics)
        return RPALowLevel(
            mf=mf, mode=mode, df=df, numerics=numerics, eps=eps, nocc=nocc,
            e_c_full=e_c, omega=None, xpy=None, factors=None, eri=None,
            cache=cache)
    mol = mf.mol
    if df:
        factors = get_density_fitting_coefficients(mol, mf, 'spatial')
        eri = None
    else:
        factors = None
        eri = get_two_electron_integrals_chemist(mol, mf, 'spatial')
    if mode == 'imagfrequency':
        e_c, cache = _imaginary_axis(eps, nocc, factors, numerics)
        return RPALowLevel(
            mf=mf, mode=mode, df=df, numerics=numerics, eps=eps, nocc=nocc,
            e_c_full=e_c, omega=None, xpy=None, factors=factors, eri=None,
            cache=cache)
    e_c, omega, xpy = _plasmon(eps, nocc, factors, eri, numerics)
    return RPALowLevel(
        mf=mf, mode=mode, df=df, numerics=numerics, eps=eps, nocc=nocc,
        e_c_full=e_c, omega=omega, xpy=xpy, factors=factors, eri=eri)


def _imaginary_axis(eps, nocc, factors, numerics):
    """(E_c, cache) of the restricted direct RPA on the imaginary axis:
    E_c by `rpa_correlation_energy_imaginary_axis` (minimax grid), and the
    cache with 'grid', the points and weights that energy integrates on
    (`imaginary_axis_energy_grid`, the function it calls), and 'p0',
    P(0) = 1 - W(0)^-1 in the normalized fitting basis. The active E_c^A is
    integrated on this grid so that the E_c pair is one quadrature."""
    lr = LinearResponseSolver(eps, coeff_df=factors, spin_mode='restricted')
    e_c = rpa_correlation_energy_imaginary_axis(lr, nocc, numerics)
    grid = imaginary_axis_energy_grid(eps, nocc, numerics)
    p0 = solve_polarizability_imaginary_axis(lr, nocc, [0.0])[0]
    return float(e_c), {'grid': grid, 'p0': p0}


def _space_time(mf, eps, nocc, numerics):
    """(E_c, cache) of the restricted direct RPA in imaginary time, serial,
    on the separable factors (X, D) of `separable_factors` on
    numerics.isdf's grid (refused, ValueError, when it names none):

    'factors': (X, D), X (M, nmo) the MO collocation, D (M, naux), so the
        fitted v is (pq|rs) = sum_Q B_Q,pq B_Q,rs with B_Q,pq = sum_P D_PQ
        X_Pp X_Pq.
    'static_grid': the one-frequency (omega = 0) minimax grid of a static
        screening (`static_screening_grid` at quadrature.ntau_w, as the BSE's
        W), and 'p0' P(0) = chi0(0) on it (`chi0_imaginary_frequency`), the
        chi0 `static_screening_matrix` inverts: 1 - W(0)^-1, built directly.
    'energy_grid': the TimeFrequencyGrid of E_c, built as the RPA
        ground-state chain builds its own: minimax tau (quadrature.ntau_rpa,
        else ntau_w, else the minimax count of the window at tau_target)
        transformed onto a Gauss-Legendre frequency axis
        (quadrature.nfreq_rpa, else RPA_ENERGY_NFREQ; `rpa_quadrature`)
        scaled to the gap. The chain takes its counts from the chain before
        it where there is one, and always the default tau target, so the
        two do not share one function.

    E_c = `rpa_correlation_energy_space_time` on 'energy_grid'. P^A(0) and
    E_c^A are built on these same grids and transforms, so the active part
    cancels to rounding with every orbital active."""
    num = numerics_or_default(numerics)
    if not num.isdf.grid_named():
        raise ValueError(
            f"method={METHOD!r} mode='space-time' builds separable (ISDF) "
            f"factors on the grid numerics.isdf names, and this run's names "
            f"none: pass numerics with numerics.isdf.level (e.g. 'G1') or "
            f"counts.")
    x, d = unpack_factors(separable_factors(mf, mf.mol, numerics=numerics,
                                            comm=None))[:2]
    q = num.quadrature
    static_grid = static_screening_grid(eps, nocc, ntau=q.ntau_w,
                                        tau_target=q.tau_target)
    tile_gb = layout_tile_gb(numerics)
    p0 = chi0_imaginary_frequency(x, d, eps, nocc, static_grid,
                                  tile_memory_gb=tile_gb)[0]
    ntau, nfreq = rpa_quadrature(num)
    e_min, e_max, _ = particle_hole_window(eps, nocc)
    if ntau is None:
        ntau = minimax_points_for_accuracy(e_min, e_max, q.tau_target)[0]
    nu, wt = gauss_legendre_grid(nfreq, gap_scaled_w0(eps, nocc))
    grid = TimeFrequencyGrid.minimax_split(ntau, e_min, e_max, nu, wt,
                                           with_sine=False,
                                           with_inverse=False)
    e_c = rpa_correlation_energy_space_time(x, d, eps, nocc, grid,
                                            tile_gb=tile_gb, comm=None)
    return float(e_c), {'factors': (x, d), 'static_grid': static_grid,
                        'p0': p0, 'energy_grid': grid}


def _plasmon(eps, nocc, factors, eri, numerics):
    """(E_c, omega, X+Y) of the direct RPA on the integrals `factors`
    (three-index) or `eri` (chemist), by the plasmon formula
    E_c = 1/2 (sum omega - Tr A): the restricted singlet problem for one
    channel, and for two (eps, nocc and factors pairs, eri the blocks
    (aa, ab, bb)) the problem on the stacked pairs of both spins. A and B
    are freed by the solver, so Tr A is taken before it. The full RPA and
    the active one go through here, so the E_c pair is the same formula on
    the same solver."""
    if isinstance(nocc, tuple):
        # the solver takes the three blocks as (aa, bb, ab)
        blocks = None if eri is None else (eri[0], eri[2], eri[1])
        lr = LinearResponseSolver(eps, coeff_df=factors, eri_chemist=blocks,
                                  spin_mode='unrestricted')
    else:
        lr = LinearResponseSolver(eps, coeff_df=factors, eri_chemist=eri,
                                  spin_mode='restricted')
    A, B = lr.build_casida_matrices(nocc, lBSE=False)
    trace_a = np.trace(A)
    solver = CasidaSolver(A, B)
    del A, B
    omega, xpy, y = solver.solve(numerics=numerics)
    xpy += y
    del y
    return float(0.5 * (np.sum(omega) - trace_a)), omega, xpy


def screening_correction(low, part, screening):
    """The static screening correction vt = W^r(0) - v on the active orbitals.

    Returns (n_act,)*4, chemist order (tu|vw), on part.act (occupied first).
    For an unrestricted result `part` is the pair of Partitions (alpha, beta)
    and the three blocks (aa, ab, bb) are returned, ab[t,u,v,w] =
    (t_a u_a | v_b w_b), on the casida route:
        vt^{ss'}_tu,vw = -2 sum_{r,r'} (t_s u_s|r) [(delta + 2K)_rr^-1]_rr' (r'|v_s' w_s'),
    r, r' over the stacked pairs of both spins that are not active, one
    operator W^r - v between each spin's own active orbitals. 'crpa_ph'
    keeps the elements with a particle-hole pair on both sides, each side
    in its own spin's partition. What follows is the restricted case, the
    same formula with the spin weight 2 on one channel.
    screening: 'bare' gives zeros; 'crpa' the restricted cRPA correction
        vt_tu,vw = -4 sum_{ia,jb in r} (tu|ia) [(delta + 4K)_rr^-1]_ia,jb (jb|vw),
    r the (i, a) pairs that are not both active: the
    response of the active pairs is deleted from A+B = delta + 4K, which is
    the paper's zeroing of A+B and A-B on the active block with its Omega = 0
    modes dropped, and the static response chi^r(0) = -4 [(A+B)_rr]^-1 of
    the spin-summed restricted singlet is one Cholesky solve on the deleted
    matrix. 'crpa_ph' keeps that correction on the particle-hole elements
    alone (`ph_element_mask`). 'mrpa', casida only, is Scott and Booth's
    moment-conserving interaction (PRL 132, 076401, Eq. 5) on the active ph
    pairs,
        (ia|jb)_eff = 1/4 [eta0_S^-1 delta_A eta0_S^-1 - delta_A],
        vt_ia,jb = (ia|jb)_eff - (ia|jb),
    eta0_S = xpy_A xpy_A^T the zeroth moment of the full RPA's singlet
    modes on the active pairs (xpy_A the rows of low.xpy of
    `active_pair_mask`, factor 1) and delta_A their eps_a - eps_i. In spin
    orbitals eta0 = (X+Y)(X+Y)^T over the same-spin pairs of both spins is
    1/2 (S + T) on the same-spin and 1/2 (S - T) on the opposite-spin
    blocks, S = xpy xpy^T of the singlet modes (xpy/sqrt(2) on each spin)
    and T = 1 of the triplet ones (A + B = A - B = delta), so Eq. 5,
    v = 1/2 [eta0_A^-1 delta_A eta0_A^-1 - delta_A], is the same in both
    blocks, 1/4 [S_A^-1 delta_A S_A^-1 - delta_A], and zero in the triplet
    channel. With every orbital active S^-1 delta S^-1 = delta + 4K and
    vt is zero to rounding. vt is set on (ia|jb), (ai|jb), (ia|bj) and
    (ai|bj) and is zero on every other element; zero without an active ph
    pair.

    (A+B)_rr and (tu|ia) are built here from the same integrals as the full
    RPA, low.factors (K = B_ov^T B_ov) or low.eri. With every orbital active
    no pair is left to screen and the correction is zero, without a solve;
    with no active occupied or no active virtual orbital nothing is deleted
    and it is the fully screened W(0) - v on those orbitals.

    On imagfrequency the same correction is P - P^A and one Dyson step,
    vt_tu,vw = sum_PQ B_P,tu [W^r - 1]_PQ B_Q,vw with W^r =
    [1 - (P(0) - P^A(0))]^-1 in the normalized fitting basis, P(0) from
    low.cache['p0'], P^A(0) `active_polarizability_static` and B the
    factors of every active pair (`active_pair_factors`). By the
    push-through identity this is the casida formula on the same factors,
    and P - P^A, not W - W^A: screening is not additive. On space-time it is
    the same Dyson step on the ISDF factors, P(0) and P^A(0) both on the
    static tau grid (low.cache['static_grid']).
    """
    screening = choice(screening, SCREENINGS, 'screening')
    if screening == 'mrpa':
        if low.mode != 'casida':
            raise NotImplementedError(
                f"method={METHOD!r} screening='mrpa' is built on the "
                f"Casida eigenvectors only: use mode='casida' (this low-level"
                f" result is mode={low.mode!r}).")
        if low.unrestricted:
            raise NotImplementedError(
                f"method={METHOD!r} screening='mrpa' takes a restricted "
                f"reference only.")
        return _mrpa_correction(low, part)
    chans = _channels(low, part)
    spins = _BLOCKS[len(chans)]
    n = chans[0].part.n_act
    res = [~active_pair_mask(c.part) for c in chans]
    if screening == 'bare' or n == 0 or not any(r.any() for r in res):
        return _per_block([np.zeros((n,) * 4) for _ in spins])
    if low.mode != 'casida':
        b_act = active_pair_factors(low, part).reshape(-1, n * n)
        w_r = dyson_screening(
            low.cache['p0'] - active_polarizability_static(low, part),
            correlation=True)
        vt = (b_act.T @ w_r @ b_act).reshape((n,) * 4)
        if screening == 'crpa_ph':
            vt *= ph_element_mask(part)
        return vt
    g = _spin_weight(chans)
    k_rr, v_rt = _residual_integrals(low, chans, res)
    apb = (2.0 * g) * k_rr
    apb[np.diag_indices_from(apb)] += _join(
        [_delta(c, r) for c, r in zip(chans, res)])
    factor = cho_factor(apb)
    solved = [cho_solve(factor, v) for v in v_rt]
    vt = [((-2.0 * g) * v_rt[s].T @ solved[t]).reshape((n,) * 4)
          for s, t in spins]
    if screening == 'crpa_ph':
        for (s, t), block in zip(spins, vt):
            block *= ph_element_mask(chans[s].part, chans[t].part)
    return _per_block(vt)


def _mrpa_correction(low, part):
    """The 'mrpa' vt of `screening_correction` (casida). (ia|jb) is the v
    of the full RPA (low.factors or low.eri), the v eta0 was solved with, so
    that vt vanishes with every orbital active; it is computed there too,
    not short-cut. eta0_S is a principal block of a positive definite
    matrix, so one Cholesky factor serves both solves."""
    n, k = part.n_act, part.n_act_occ
    vt = np.zeros((n,) * 4)
    if k == 0 or k == n:
        return vt
    xpy_a = low.xpy[active_pair_mask(part)]
    chans = _channels(low, part)
    delta, k_ph, _ = _active_ph(chans, _active_v(low, chans))
    eta0 = cho_factor(xpy_a @ xpy_a.T)
    k_eff = cho_solve(eta0, cho_solve(eta0, np.diag(delta)).T)
    k_eff = 0.25 * (0.5 * (k_eff + k_eff.T) - np.diag(delta))
    ph = (k_eff - k_ph).reshape(k, n - k, k, n - k)
    vt[:k, k:, :k, k:] = ph
    vt[k:, :k, :k, k:] = ph.transpose(1, 0, 2, 3)
    vt[:k, k:, k:, :k] = ph.transpose(0, 1, 3, 2)
    vt[k:, :k, k:, :k] = ph.transpose(1, 0, 3, 2)
    return vt


def active_pair_factors(low, part):
    """B[Q, t, u] (naux, n_act, n_act): the fitting factors of every active
    pair, occupied-occupied, occupied-virtual and virtual-virtual, on
    part.act (occupied first), so (tu|vw) = sum_Q B_Q,tu B_Q,vw is the
    full RPA's fitted v on the active orbitals. vtilde needs all of them,
    not only the ov block. On space-time they are built from
    the active columns of the collocation X alone, B_Q,tu = sum_P D_PQ X_Pt
    X_Pu (low.cache['factors']), the factors chi0 is built from."""
    return _pair_factors(low, part.act)


def _pair_factors(low, act):
    """`active_pair_factors` on the orbitals `act`."""
    if low.mode == 'space-time':
        x, d = low.cache['factors']
        n, x_act = len(act), x[:, act]
        pairs = (x_act[:, :, None] * x_act[:, None, :]).reshape(len(x), n * n)
        return (d.T @ pairs).reshape(d.shape[1], n, n)
    return low.factors[:, act][:, :, act]


def active_polarizability_static(low, part):
    """P^A(0) (naux, naux): the static bare response of the active ph pairs
    alone (i active occupied, a active virtual) in the normalized fitting
    basis, chi0(0) of the full P(0) restricted to those columns of B_ov,
    2 sum_ia B_ia f_ia(0) B_ia^T = -4 sum_ia B_ia B_ia^T / delta_ia, with
    the same frequency factor and spin factor as the full one, so P - P^A
    keeps every other pair exactly. Zero without an active ph pair.

    On space-time it is the full P(0)'s own build on the stored static grid
    (`chi0_imaginary_frequency`: `polarizability_projected_tau` at every
    tau point, transformed by the grid's cosine weights) with X restricted
    to the active columns and the full system's mid-gap mu, i.e. the
    transform of -2 sum_ia B_ia B_ia^T exp(-delta_ia tau) over the active ph
    pairs: with every orbital active it is P(0) itself, to the bit."""
    if low.mode == 'space-time':
        x, d = low.cache['factors']
        k = part.n_act_occ
        if k == 0 or k == part.n_act:
            return np.zeros((d.shape[1],) * 2)
        act = part.act
        mu = particle_hole_window(low.eps, low.nocc)[2]
        return chi0_imaginary_frequency(
            x[:, act], d, low.eps[act], k, low.cache['static_grid'], mu=mu,
            tile_memory_gb=layout_tile_gb(low.numerics))[0]
    nocc = low.nocc
    b = low.factors
    mask = active_pair_mask(part)
    c_a = b[:, :nocc, nocc:].reshape(b.shape[0], -1)[:, mask]
    eps = low.eps
    delta = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()[mask]
    return chi0_aux([(c_a, frequency_factor(delta, 0.0, True))], 2.0)


def _active_v(low, chans):
    """The chemist (pq|rs) of the full RPA's v on the active orbitals of the
    channels, one (n_act,)*4 block per spin pair, (v,) or (aa, ab, bb): from
    low.factors or low.eri, and on space-time the ISDF v of
    `active_pair_factors`.

    Not the exact v of H_eff (`two_body.active_eri`): E_c^A, W^A and the
    static defect take the integrals the full RPA was solved with,
    so that the double counting cancels against E_c and W."""
    n = chans[0].part.n_act
    if low.mode == 'space-time':
        b_act = _pair_factors(low, chans[0].part.act).reshape(-1, n * n)
        return ((b_act.T @ b_act).reshape((n,) * 4),)
    spins = _BLOCKS[len(chans)]
    if low.factors is not None:
        b_act = [_act_factors(c) for c in chans]
        return tuple((b_act[s].T @ b_act[t]).reshape((n,) * 4)
                     for s, t in spins)
    blocks = []
    for s, t in spins:
        first, second = chans[s].part.act, chans[t].part.act
        blocks.append(_eri_block(low, s, t)[np.ix_(first, first)]
                      [:, :, second][:, :, :, second])
    return tuple(blocks)


def _active_ov(low, chans):
    """The (pq|ia) of the full RPA's v, one (n*n, n_pair) array per channel:
    p, q over the active orbitals of that channel and (i, a) over every
    particle-hole pair in C order, of both spins stacked for two channels;
    from low.factors or low.eri (casida). Its cost, n^2 naux n_pair with the
    factors, is that of the whole pair space, so it is built only where it
    is read: the casida W(0) of `static_defect`."""
    n = chans[0].part.n_act
    if low.factors is not None:
        b_ov = _join([_ov_factors(c) for c in chans], axis=1)
        return [_act_factors(c).T @ b_ov for c in chans]
    out = []
    for s, chan_s in enumerate(chans):
        act, cols = chan_s.part.act, []
        for t, chan_t in enumerate(chans):
            eri = _eri_block(low, s, t)
            # spelt out: n may be 0
            n_pair = chan_t.nocc * (eri.shape[2] - chan_t.nocc)
            cols.append(eri[np.ix_(act, act)][:, :, :chan_t.nocc,
                                              chan_t.nocc:].reshape(n * n,
                                                                    n_pair))
        out.append(_join(cols, axis=1))
    return out


def _active_ph(chans, v_eff):
    """(delta, K, v_ph) of the active ph pairs (k active occupied times the
    active virtual, local indices, C order; those of both spins stacked for
    two channels): delta the orbital energy differences, K_ia,jb =
    v_eff[i,a,j,b], and v_ph, one (n*n, n_ph) array per channel, the columns
    v_eff[:, :, i, a] with the first pair in that channel. v_eff: the blocks
    (v,) or (aa, ab, bb)."""
    n = chans[0].part.n_act
    occ = [c.part.n_act_occ for c in chans]
    delta = []
    for chan, k in zip(chans, occ):
        eps = chan.eps[chan.part.act]
        delta.append((eps[k:][None, :] - eps[:k][:, None]).ravel())
    k_rows, v_ph = [], []
    for s, k_s in enumerate(occ):
        k_rows.append([])
        cols = []
        for t, k_t in enumerate(occ):
            block = _spin_block(v_eff, s, t)
            cols.append(block[:, :, :k_t, k_t:].reshape(n * n,
                                                        delta[t].size))
            k_rows[s].append(block[:k_s, k_s:, :k_t, k_t:].reshape(
                delta[s].size, delta[t].size))
        v_ph.append(_join(cols, axis=1))
    return _join(delta), _square(k_rows), v_ph


def active_correlation_energy(low, part, v_eff):
    """E_c^A[v_eff], the direct RPA correlation energy of the active space
    alone: the restricted singlet dRPA on the n_A_o n_A_v active ph pairs
    with K_ia,jb = v_eff[i,a,j,b] (local indices, occupied first) and the
    orbital energies of the reference, by the plasmon formula of the full
    RPA. v_eff is (n_act,)*4 chemist; for the E_c pair to cancel when every
    orbital is active it has to be built on the same v as low.e_c_full
    (low.factors or low.eri). Zero when there is no active ph pair.

    On imagfrequency it is the full energy's quadrature on the full
    energy's grid (low.cache['grid'], not one of the active window):

        E_c^A = 1/(2 pi) sum_k w_k Tr[ln(1 - Pi^A(i w_k) U) + Pi^A(i w_k) U],

    in the active ph space, U = K^A the ph block of v_eff and Pi^A the
    diagonal bare response 2 f(i w) = -4 delta/(delta^2 + w^2), the spin
    and frequency factor of the full chi0
    (`solve_polarizability_imaginary_axis`). With every orbital active and
    v_eff the fitted v, U = B^T B and
    Tr ln(1 - Pi B^T B) = ln det(1 - B Pi B^T): the full integrand at every
    point, so the pair cancels to rounding. The dense-Casida value on the
    same v_eff is the 'ec_active_casida' diagnostic.

    On space-time the same quadrature on the energy's grid
    (low.cache['energy_grid']), with Pi^A(i w_k) = -2 sum_tau
    cosft_wt[k, tau] exp(-delta tau): exp(-delta tau) pushed through the
    energy's own tau -> omega transform, as the full chi0 was
    (`rpa_correlation_energy_space_time`), not the exact Lorentzian, so the
    pair again cancels to rounding with every orbital active.

    For an unrestricted result `part` is the pair (alpha, beta) and v_eff
    the three blocks (aa, ab, bb): the direct RPA on the active ph pairs of
    both spins, stacked, with K^A_(ia s),(jb s') = v_eff^{ss'}[i,a,j,b]. It
    is zero only when neither channel has an active ph pair; with one in a
    single channel the problem holds that spin's pairs."""
    chans = _channels(low, part)
    v_eff = _two_body_blocks(v_eff, chans)
    if not any(0 < c.part.n_act_occ < c.part.n_act for c in chans):
        return 0.0
    if low.mode != 'casida':
        return _active_ec_quadrature(low, chans, v_eff)
    return _active_ec_casida(low, chans, v_eff)


def _active_ec_casida(low, chans, v_eff):
    """The dense dRPA E_c^A[v_eff] of the active ph pairs by the plasmon
    formula (there is at least one active ph pair); v_eff the blocks (v,)
    or (aa, ab, bb)."""
    eps = [c.eps[c.part.act] for c in chans]
    occ = [c.part.n_act_occ for c in chans]
    if len(chans) == 1:
        return _plasmon(eps[0], occ[0], None, v_eff[0], low.numerics)[0]
    return _plasmon(tuple(eps), tuple(occ), None, v_eff, low.numerics)[0]


def _active_ec_quadrature(low, chans, v_eff):
    """E_c^A[v_eff] on the imaginary axis, the formula of
    `active_correlation_energy`, with `dielectric_log_determinant` (its
    warning and flag for a non-positive det(1 - Pi^A U))."""
    delta, k_ph, _ = _active_ph(chans, v_eff)
    eye = np.eye(delta.size)
    if low.mode == 'space-time':
        grid = low.cache['energy_grid']
        weights = grid.omega_weights
        pi = -2.0 * grid.cosft_wt @ np.exp(-np.outer(grid.tau_points, delta))
    else:
        points, weights = low.cache['grid']
        pi = [2.0 * frequency_factor(delta, w, True) for w in points]
    e_c = 0.0
    for k, (pi_k, wt) in enumerate(zip(pi, weights)):
        pu = pi_k[:, None] * k_ph
        e_c += wt * (dielectric_log_determinant(eye - pu, k) + np.trace(pu))
    return float(e_c / (2.0 * np.pi))


def static_defect(low, part, v_eff):
    """max |W(0) - W^A[v_eff](0)| over the active (pq|rs): how far the
    active-space RPA of v_eff falls short of the full static screening.

    W(0) - v = -4 V xpy diag(1/omega) xpy^T V^T, V = (pq|ia) on the v of the
    full RPA; W^A[v_eff](0) = v_eff - 4 v_eff[:, ph] (delta_A + 4 K^A)^-1
    v_eff[ph, :] on the active ph pairs, K^A the ph block of v_eff. v_eff has
    to be built on that same v: for crpa, W^A[W^r](0) = W(0)
    exactly and the defect is zero to rounding. delta_A + 4 K^A is positive
    definite for every built screening (for crpa it is a Schur complement of
    delta + 4K, for mrpa eta0_S^-1 delta_A eta0_S^-1), so one Cholesky
    solve gives the active response.
    Zero for an empty active space.

    On imagfrequency W(0) is the cache's: B_act^T [1 - P(0)]^-1 B_act with
    B_act `active_pair_factors`, against the same W^A on v_eff
    of the fitted v + vtilde, so the crpa defect is again an identity. On
    space-time likewise, with the static grid's response of each active ph
    pair in W^A: -4/delta becomes pi(0) = -2 sum_tau cosft_wt[0, tau]
    exp(-delta tau), the one P^A(0) carries (`active_polarizability_static`),
    i.e. delta_A -> -4/pi(0) above, without which the crpa defect would be
    the transform's error and not zero.

    For an unrestricted result `part` is the pair (alpha, beta) and v_eff
    the three blocks (aa, ab, bb). Both sides are then the two-spin ones,
    W(0) - v = -2 V xpy diag(1/omega) xpy^T V^T on the stacked pairs and
    W^A[v_eff](0) = v_eff - 2 v_eff[:, ph] (delta_A + 2 K^A)^-1 v_eff[ph, :]
    on the active ph pairs of both spins, and the defect is the largest
    deviation over the three blocks.
    """
    chans = _channels(low, part)
    v_eff = _two_body_blocks(v_eff, chans)
    spins = _BLOCKS[len(chans)]
    n = chans[0].part.n_act
    if n == 0:
        return 0.0
    g = _spin_weight(chans)
    if low.mode != 'casida':
        b_act = active_pair_factors(low, part).reshape(-1, n * n)
        w_full = [(b_act.T @ dyson_screening(low.cache['p0'])
                   @ b_act).reshape((n,) * 4)]
    else:
        r = [ov @ low.xpy for ov in _active_ov(low, chans)]
        w_full = [v - (2.0 * g) * ((r[s] / low.omega) @ r[t].T).reshape(
                      (n,) * 4)
                  for v, (s, t) in zip(_active_v(low, chans), spins)]
    w_act = [v.copy() for v in v_eff]
    delta, k_ph, v_ph = _active_ph(chans, v_eff)
    if delta.size:
        if low.mode == 'space-time':
            grid = low.cache['static_grid']
            delta = 2.0 / (np.exp(-np.outer(delta, grid.tau_points))
                           @ grid.cosft_wt[0])
        apb = (2.0 * g) * k_ph
        apb[np.diag_indices_from(apb)] += delta
        factor = cho_factor(apb)
        for block, (s, t) in zip(w_act, spins):
            block -= ((2.0 * g) * v_ph[s]
                      @ cho_solve(factor, v_ph[t].T)).reshape((n,) * 4)
    return max(float(np.abs(full - act).max())
               for full, act in zip(w_full, w_act))


def screened_residues(low, part, orbitals=None):
    """w (nexc, n_act, n_r): the residues of the full W between the active
    orbitals (part.act, occupied first) and `orbitals` (default part.env,
    the inactive then the environment virtual orbitals), one slice per RPA
    excitation:

        w[n, t, r] = sqrt(2) sum_ia (t r|i a) xpy_ia,n.

    The sqrt(2) makes w the spin-orbital residue for either spin of t and r
    (a restricted singlet mode puts xpy/sqrt(2) on both spins of each pair),
    so sum_r w_tr w_ur over spatial r is the spin-orbital sum over the r of
    t's spin and the self-energy formulas need no further spin factor; W - v
    = sum_n w w^T 2 Omega_n/(omega^2 - Omega_n^2), which at omega = 0 is the
    -4/Omega_n of `static_defect` on the residues without sqrt(2). With r
    over every orbital, `srg_self_energy` of these w is the active block of
    the qsGW SRG kernel (`SelfEnergySolver.static_self_energy_matrix`,
    2 sum chi chi K with chi = w/sqrt(2)); tests/test_downfolding_sigma.py
    pins the factor there. (tr|ia) is built from the integrals of the full
    RPA, low.factors or low.eri. Casida only: the other routes keep no
    eigenvectors, and refuse (NotImplementedError).

    For an unrestricted result `part` is the pair (alpha, beta), `orbitals`
    a pair of index arrays when given, and the pair (w_alpha, w_beta) is
    returned, each between that spin's active orbitals and its `orbitals`:

        w^s[n, t, r] = sum_{ia s'} (t_s r_s|i_s' a_s') xpy_{ia s', n},

    the sum over the stacked pairs of both spins and no further factor: the
    modes are those of the two-spin problem, normalized over both spins.
    The sqrt(2) above is sqrt(g) of the one restricted channel.
    """
    if low.mode != 'casida':
        raise NotImplementedError(
            f"screened_residues needs the Casida eigenvectors X+Y, which "
            f"mode={low.mode!r} does not keep: use mode='casida'.")
    chans = _channels(low, part)
    if orbitals is None:
        orbs = [c.part.env for c in chans]
    elif len(chans) == 1:
        orbs = [np.asarray(orbitals, dtype=int)]
    else:
        orbs = [np.asarray(o, dtype=int) for o in orbitals]
    return _per_channel(_residues(low, chans, orbs))


def _residues(low, chans, orbs):
    """`screened_residues` per channel: one (nexc, n_act, n_r) array for
    each, between its active orbitals and its index array of `orbs`."""
    scale, nexc = np.sqrt(_spin_weight(chans)), len(low.omega)
    out = []
    if low.factors is not None:
        rho = _join([_ov_factors(c) for c in chans], axis=1) @ low.xpy
        for chan, r in zip(chans, orbs):
            b, act = chan.factors, chan.part.act
            n, m = len(act), len(r)
            w = rho.T @ b[:, act][:, :, r].reshape(b.shape[0], n * m)
            out.append(scale * w.reshape(nexc, n, m))
        return out
    for s, (chan_s, r) in enumerate(zip(chans, orbs)):
        act = chan_s.part.act
        n, m = len(act), len(r)
        cols = []
        for t, chan_t in enumerate(chans):
            eri = _eri_block(low, s, t)
            n_pair = chan_t.nocc * (eri.shape[2] - chan_t.nocc)
            cols.append(eri[np.ix_(act, r)][:, :, :chan_t.nocc,
                                            chan_t.nocc:].reshape(n * m,
                                                                  n_pair))
        w = low.xpy.T @ _join(cols, axis=1).T
        out.append(scale * w.reshape(nexc, n, m))
    return out


def _denominators(eps_t, eps_r, occupied_r, omega):
    """D[n, t, r] = eps_t - eps_r - sgn(eps_r - eps_F) Omega_n: + Omega_n for
    an occupied r, - Omega_n for an empty one."""
    sign = np.where(np.asarray(occupied_r, dtype=bool), 1.0, -1.0)
    return (np.asarray(eps_t)[None, :, None] - np.asarray(eps_r)[None, None, :]
            + sign[None, None, :] * np.asarray(omega)[:, None, None])


# The (nexc, n_act, n_act, n_r) arrays `_srg_pair` holds at once in one
# block: a + b, lam, the s = inf kernel, -s lam and its expm1 (the damped
# kernel overwrites one of them), counted for its byte budget.
_SRG_BLOCK_ARRAYS = 5


def _excitation_blocks(nexc, n_act, n_r, tile_gb):
    """Slices of the nexc excitations, consecutive, each with as many as
    keep the _SRG_BLOCK_ARRAYS (n, n_act, n_act, n_r) temporaries of one
    block within tile_gb GB, at least one."""
    per = _SRG_BLOCK_ARRAYS * 8 * max(n_act, 1) ** 2 * max(n_r, 1)
    size = max(1, int(tile_gb * BYTES_PER_GB // per))
    return [slice(k, min(k + size, nexc)) for k in range(0, nexc, size)]


def _sigma_tile_gb(numerics):
    """The byte budget, GB, of one block of `srg_self_energy`:
    numerics.layout.tile_gb, DOWNFOLDING_SIGMA_TILE_GB where it is None."""
    gb = numerics_or_default(numerics).layout.tile_gb
    return DOWNFOLDING_SIGMA_TILE_GB if gb is None else float(gb)


def _srg_pair(w, eps_t, eps_r, occupied_r, omega, s, tile_gb=None,
              infinite=False):
    """(Sigma(s), Sigma(inf) or None) of `srg_self_energy`, both from one
    loop over blocks of excitations (`_excitation_blocks` at tile_gb, None:
    DOWNFOLDING_SIGMA_TILE_GB): the four-index kernel of a block is built,
    contracted and dropped, so the working set is tile_gb and not
    (nexc, n_act, n_act, n_r). infinite: also Sigma at s = inf from the
    same block's kernel before its damping, the damping diagnostic's
    reference, the very numbers a separate s = None call gives."""
    tile_gb = DOWNFOLDING_SIGMA_TILE_GB if tile_gb is None else tile_gb
    d = _denominators(eps_t, eps_r, occupied_r, omega)
    n_act = d.shape[1]
    out = np.zeros((n_act, n_act))
    out_inf = np.zeros((n_act, n_act)) if infinite and s is not None else None
    for blk in _excitation_blocks(len(d), n_act, d.shape[2], tile_gb):
        a, b = d[blk, :, None, :], d[blk, None, :, :]
        lam = a * a + b * b
        kern = np.divide(a + b, lam, out=np.zeros_like(lam), where=lam > 0)
        wb = w[blk]
        if out_inf is not None:
            out_inf += np.einsum('ntr,nur,ntur->tu', wb, wb, kern)
        if s is not None:
            kern *= -np.expm1(-s * lam)
        out += np.einsum('ntr,nur,ntur->tu', wb, wb, kern)
    out = 0.5 * (out + out.T)
    if out_inf is not None:
        return out, 0.5 * (out_inf + out_inf.T)
    return out, (out if infinite else None)


def srg_self_energy(w, eps_t, eps_r, occupied_r, omega, s, tile_gb=None):
    """The SRG static self-energy (n_act, n_act), Marie and Loos Eq. 44:

        Sigma_tu = sum_rn (D_tr + D_ur)/(D_tr^2 + D_ur^2) w_tr w_ur
                   [1 - exp(-(D_tr^2 + D_ur^2) s)],

    D_tr = eps_t - eps_r - sgn(eps_r - eps_F) Omega_n (`_denominators`), w
    the residues of `screened_residues` (nexc, n_act, n_r), occupied_r a bool
    array over the internal orbitals r. s = None is s = inf, Eq. 42, the
    bracket 1. Every term is finite for finite s (the kernel is bounded by
    (1 + sqrt 2)/(2 max(|a|, |b|)) and vanishes with a^2 + b^2), so this
    needs no pole check; it is symmetric in t, u by construction. A term
    exactly at a pole (D_tr = D_ur = 0, 0/0 as written) is 0, its limit at
    finite s; Eq. 42 has no value there and gets the same 0.

    The sum over n runs over blocks of excitations whose four-index kernel
    fits in tile_gb GB (None: DOWNFOLDING_SIGMA_TILE_GB; `_srg_pair`):
    whole, that kernel is nexc n_act^2 n_r doubles per array, about 1 GB on
    benzene/cc-pVTZ (10,10), where w is 0.1 GB. The blocks change only the
    order of the sum.
    """
    return _srg_pair(w, eps_t, eps_r, occupied_r, omega, s, tile_gb)[0]


def mode_a_self_energy(w, eps_t, eps_r, occupied_r, omega):
    """Mode A (n_act, n_act), Marie and Loos Eq. 12 at eta = 0:

        Sigma_tu = 1/2 sum_rn [1/D_tr + 1/D_ur] w_tr w_ur
                 = 1/2 [S_tu(eps_t) + S_tu(eps_u)],

    S the pole sum of iG(W - v) over the internal orbitals r, arguments as
    in `srg_self_energy`. Undamped: a D_tr near zero makes it grow without
    bound, of which `embedding_self_energy` warns; with a D_tr exactly zero
    the element is not finite (inf or nan, without a numpy warning), which
    `embedding_self_energy` refuses. Its temporaries are the size of w (one
    denominator per t, r), so unlike Eq. 44 it needs no blocks.
    """
    d = _denominators(eps_t, eps_r, occupied_r, omega)
    with np.errstate(divide='ignore', invalid='ignore'):
        half = np.einsum('ntr,nur,ntr->tu', w, w, 1.0 / d)
        return 0.5 * (half + half.T)


def lowest_excitation(low, solver=None):
    """Omega_min, the lowest excitation energy of the full RPA (None without
    a particle-hole pair), the one number the pole-free window needs.

    casida: the lowest of low.omega. imagfrequency and space-time, which
    diagonalize nothing: the lowest root of the same restricted singlet
    Casida problem on the route's own v (low.factors, or the ISDF factors
    (X, D) of low.cache['factors']), A = delta + 2K, B = 2K; by the dense
    solve up to DOWNFOLDING_DENSE_OMEGA_MAX_PAIRS pairs and above that by
    the Davidson for one root (the DF or the ISDF block action, the run's
    other Davidson settings), which never builds the matrix: above that
    size a dense solve for this one number would cost more than the route's
    own RPA. Cached in low.cache['omega_min'] on the first request.
    solver: 'dense' or 'davidson' forces that solver and bypasses the cache
    (for tests).
    """
    if low.mode == 'casida':
        return float(low.omega.min()) if low.omega.size else None

    def solved():
        nocc = low.nocc
        n_pair = nocc * (len(low.eps) - nocc)
        if not n_pair:
            return None
        rule = ('dense' if n_pair <= DOWNFOLDING_DENSE_OMEGA_MAX_PAIRS
                else 'davidson')
        how = choice(solver or rule, ('dense', 'davidson'), 'solver')
        return _lowest_dense(low) if how == 'dense' else _lowest_davidson(low)

    return solved() if solver is not None else _cached(low, 'omega_min',
                                                       solved)


def _cached(low, key, build):
    """low.cache[key], built by `build()` on the first request.

    One low-level result serves any number of active spaces, and only the
    first request builds the value. The named flags raised while building it
    (src.Base.numerical_flags) are therefore kept beside it, in
    low.cache['flags'][key], and handed to the collecting block of every
    later request too: each Hamiltonian that uses the value has them in its
    record. The warning itself is issued once, by the build."""
    if key not in low.cache:
        raised = []
        with collecting(raised):
            low.cache[key] = build()
        low.cache.setdefault('flags', {})[key] = raised
        return low.cache[key]
    with collecting([]) as again:
        again.extend(low.cache.get('flags', {}).get(key, ()))
    return low.cache[key]


def _lowest_dense(low):
    """The lowest root of the dense restricted singlet Casida problem on the
    occupied-virtual factors of the route: K = B_ov^T B_ov, B_ov from
    low.factors or, on space-time, sum_P D_PQ X_Pi X_Pa."""
    nocc, eps = low.nocc, low.eps
    if low.mode == 'space-time':
        x, d = low.cache['factors']
        pairs = x[:, :nocc, None] * x[:, None, nocc:]
        b_ov = d.T @ pairs.reshape(len(x), -1)
    else:
        b = low.factors
        b_ov = b[:, :nocc, nocc:].reshape(b.shape[0], -1)
    k = b_ov.T @ b_ov
    a = 2.0 * k
    a[np.diag_indices_from(a)] += (eps[nocc:][None, :]
                                   - eps[:nocc][:, None]).ravel()
    return float(np.min(CasidaSolver(a, 2.0 * k).solve(
        numerics=low.numerics)[0]))


def _lowest_davidson(low):
    """The lowest root of the same problem by the matrix-free Davidson for
    one root (`solve_casida_davidson`, the run's other Davidson settings):
    the DF block action on low.factors, the ISDF one on (X, D).

    The one root starts from as many unit vectors as the run's own Davidson
    would seed for its nroots (guess_factor each), never fewer than
    DOWNFOLDING_OMEGA_DAVIDSON_GUESS: a root from a guess that missed the
    irrep of the lowest one lies above Omega_min and widens the pole-free
    window in the unsafe direction.

    An unconverged root does not stop the build: Omega_min decides only
    whether the near-pole warning is raised, no number of H_eff. Its last
    estimate is kept, with the Davidson's own named flag
    ('davidson_not_converged' or 'davidson_unconfirmed') and a
    SigmaNearPoleWarning, the flag 'sigma_pole_window_unconverged': the
    estimate can lie above Omega_min, so the window may be too wide and an
    orbital near a pole may go without its warning."""
    num = numerics_or_default(low.numerics)
    bse = num.bse
    guesses = max(bse.guess_factor * (bse.nroots or BSE_DAVIDSON_NROOTS),
                  DOWNFOLDING_OMEGA_DAVIDSON_GUESS)
    num = replace(num, bse=replace(bse, nroots=1, guess_factor=guesses))
    raised = []
    with collecting(raised):
        if low.mode == 'space-time':
            lr = LinearResponseSolver(low.eps, spin_mode='restricted')
            omega = solve_casida_davidson(lr, low.nocc, numerics=num,
                                          isdf_factors=low.cache['factors'],
                                          refuse_unconverged=False)[0]
        else:
            lr = LinearResponseSolver(low.eps, coeff_df=low.factors,
                                      spin_mode='restricted')
            omega = solve_casida_davidson(lr, low.nocc, numerics=num,
                                          refuse_unconverged=False)[0]
    om = float(np.min(omega))
    if raised:
        names = ', '.join(sorted({e['flag'] for e in raised}))
        flag('sigma_pole_window_unconverged',
             f"Omega_min = {om:.6f} Ha, the lowest RPA excitation that bounds "
             f"the pole-free window of the environment self-energy, comes "
             f"from a Davidson that did not finish ({names}): it can lie "
             f"above the true Omega_min, so the pole-free window may be too "
             f"wide and an active orbital near a pole may go without its "
             f"SigmaNearPoleWarning. Sigma_bar^E and H_eff do not depend on "
             f"it. More cycles (numerics.bse.max_cycle) settle it.",
             SigmaNearPoleWarning, stacklevel=2)
    return om


def pole_window(low, part):
    """(lo, hi), the pole-free window of the environment self-energy:
    (max eps_I - Omega_min, min eps_Ev + Omega_min), Omega_min from
    `lowest_excitation`. Every pole of S^E lies at eps_i - Omega_n <= lo
    (inactive i) or eps_a + Omega_n >= hi (environment virtual a), so between
    them every D_tr is non-zero. A side without orbitals (or no excitation at
    all) is -inf or +inf; with neither side Omega_min is not computed.

    For an unrestricted result `part` is the pair (alpha, beta) and the pair
    of windows is returned, each with its own spin's orbital energies and
    the one Omega_min of the two-spin problem. That Omega_min includes the
    spin-density modes, which carry no weight in W for a closed shell, so
    the windows are conservative: narrower than needed, never wider."""
    return _per_channel([_window(low, c) for c in _channels(low, part)])


def _window(low, chan):
    """`pole_window` of one spin channel."""
    part = chan.part
    if not len(part.inactive) and not len(part.env_virt):
        return -np.inf, np.inf
    om = lowest_excitation(low)
    if om is None:
        return -np.inf, np.inf
    eps = chan.eps
    lo = eps[part.inactive].max() - om if len(part.inactive) else -np.inf
    hi = eps[part.env_virt].min() + om if len(part.env_virt) else np.inf
    return float(lo), float(hi)


class SigmaNearPoleWarning(RuntimeWarning):
    """An active orbital energy lies within the margin of an edge of the
    pole-free window of the environment self-energy, or outside the window:
    Sigma_bar^E is computed all the same, undamped ('mode_a' on casida) or
    continued from the imaginary axis (imagfrequency, space-time), and is
    unreliable on that orbital."""


def _warn_near_poles(low, chans, settings, sigma, stacklevel):
    """Warn (SigmaNearPoleWarning, the named flag 'sigma_near_pole') of an
    active orbital whose energy is within the margin of an edge of
    `pole_window`, or outside it, in any spin channel (one warning, naming
    the spin of each orbital); Sigma_bar is computed all the same. The
    margin: on casida pole_margin/sqrt(2 s), s = DOWNFOLDING_SRG_S when
    srg_s is None, for sigma='mode_a' alone, pointing to 'srg'; on
    imagfrequency and space-time the larger of that and continuation_margin
    (the Pade continuation loses accuracy towards the poles), for both
    forms, pointing to casida. stacklevel: as for `flag`, counted from this
    function."""
    s = DOWNFOLDING_SRG_S if settings.srg_s is None else settings.srg_s
    margin = settings.pole_margin / np.sqrt(2.0 * s)
    why = (f"pole_margin {settings.pole_margin:g} / sqrt(2 s), s = {s:g}")
    if low.mode != 'casida' and settings.continuation_margin > margin:
        margin = settings.continuation_margin
        why = (f"continuation_margin {margin:g} Ha, the distance beyond "
               f"which the Pade continuation from the imaginary axis is "
               f"accurate")
    named, windows = [], []
    for chan in chans:
        lo, hi = _window(low, chan)
        windows.append((lo, hi))
        act = chan.part.act
        eps = chan.eps[act]
        dist = np.minimum(eps - lo, hi - eps)
        bad = dist <= margin
        named += [f'{_orbital(chan)} {p} (eps = {e:.6f}, distance {x:.6f})'
                  for p, e, x in zip(act[bad], eps[bad], dist[bad])]
    if named:
        named = ', '.join(named)
        if low.mode == 'casida':
            what = ("'mode_a' is undamped and grows without bound towards a "
                    "pole, so its elements on these orbitals are dominated by "
                    "that pole; sigma='srg' is finite everywhere.")
        else:
            what = (f"On mode={low.mode!r} it is continued from the imaginary "
                    f"axis without damping (s = inf) and loses accuracy "
                    f"towards the poles: its elements on these orbitals have "
                    f"been measured off by 1e-4 Ha at 0.16 Ha from the edge "
                    f"and by 7e-4 Ha at 0.09 Ha (water), an error the "
                    f"'sigma_continuation' check does not show. Where that "
                    f"matters use mode='casida' with sigma='srg', finite "
                    f"everywhere.")
        flag(
            'sigma_near_pole',
            f"sigma={sigma!r}: active orbitals near a pole of the "
            f"environment self-energy, {named}, within the margin "
            f"{margin:.6f} Ha ({why}) of the pole-free "
            f"{_window_text(chans, windows)} or outside it (a negative "
            f"distance). Sigma_bar^E is computed all the same. " + what,
            SigmaNearPoleWarning, stacklevel=stacklevel)


def _require_finite(sigma_bar, chan, sigma):
    """Refuse (FloatingPointError) a Sigma_bar of one spin channel with an
    element that is not finite, naming the orbital pairs (and their spin):
    an active orbital energy on a pole. Near a pole Sigma_bar is kept with a
    warning (`_warn_near_poles`); this one cannot go into H_eff."""
    bad = np.argwhere(~np.isfinite(sigma_bar))
    if len(bad):
        act = chan.part.act
        named = ', '.join(f'({act[t]}, {act[u]})' for t, u in bad[:8])
        more = f' and {len(bad) - 8} more' if len(bad) > 8 else ''
        raise FloatingPointError(
            f"sigma={sigma!r}: Sigma_bar^E is not finite at the elements "
            f"{named}{more} ({_orbital(chan)} pairs): an active orbital energy "
            f"coincides with a pole of the environment self-energy, and "
            f"such a Sigma_bar cannot go into H_eff. Use sigma='srg' on "
            f"mode='casida', which is finite everywhere, or move the orbital "
            f"out of the active space.")


def embedding_self_energy(low, part, sigma):
    """Sigma_bar^E (n_act, n_act), the static environment self-energy on
    part.act (occupied first): iG^E(W - v) with r over part.env, the full
    RPA's W and the mean-field energies.

    casida: the residues of W (`screened_residues`) and the RPA's Omega.
    sigma 'srg' is Eq. 44 at s = numerics.downfolding.srg_s (None: Eq. 42),
        finite everywhere; 'mode_a', Eq. 12 at eta = 0, which warns
        (SigmaNearPoleWarning, naming the orbitals) when an active energy
        is within pole_margin/sqrt(2 s) of the pole-free window's edge or
        outside it (`pole_window`), and is computed all the same.
    imagfrequency: S^E(mu + i w) on the Pade nodes (`sigma_env_imag_axis`),
        continued once and evaluated per element (`continued_self_energy`):
        'srg' is Eq. 42, Re S^E_tu(z_tu), at s = inf whatever srg_s is
        (srg_s enters the pole check alone), 'mode_a' 1/2 Re[S^E_tu(eps_t)
        + S^E_tu(eps_u)]. Both warn near the poles as 'mode_a' does on
        casida, within the larger of pole_margin/sqrt(2 s) and
        numerics.downfolding.continuation_margin, pointing to mode='casida';
        the continuation is inaccurate there (constants.py,
        DOWNFOLDING_CONTINUATION_MARGIN).
    space-time: the same, with S^E(mu + i w) sampled as on imagfrequency
        on the route's own ISDF factors (`sigma_env_imag_axis`).
    Zero when the environment is empty. A Sigma_bar that is not finite (an
    active energy on a pole) is refused, FloatingPointError.

    For an unrestricted result `part` is the pair (alpha, beta) and the pair
    (Sigma_bar^alpha, Sigma_bar^beta) is returned: the same formulas per
    spin, with the residues between that spin's active and environment
    orbitals, that spin's orbital energies, and the Omega_n of the two-spin
    RPA, which both spins share. One sigma and one srg_s serve both.
    """
    return _per_channel(_embedding(low, part, sigma)[0])


def _embedding(low, part, sigma, infinite=False, stacklevel=3):
    """`_embedding_unchecked` after the pole warning and with the result
    required finite: (Sigma_bar, kept), each a list over the spin channels
    of `part`. stacklevel: where the pole warning points, counted from this
    function as for warnings.warn (3: the caller of this function's
    caller)."""
    sigma = choice(sigma, SIGMAS[1:], 'sigma')
    settings = numerics_or_default(low.numerics).downfolding
    chans = _channels(low, part)
    if sigma == 'mode_a' or low.mode != 'casida':
        _warn_near_poles(low, chans, settings, sigma, stacklevel + 1)
    sigma_bar, kept = _embedding_unchecked(low, part, sigma, settings,
                                           infinite)
    for chan, s_bar in zip(chans, sigma_bar):
        _require_finite(s_bar, chan, sigma)
    return sigma_bar, kept


def _embedding_unchecked(low, part, sigma, settings, infinite):
    """(Sigma_bar, kept) of `embedding_self_energy`, each a list over the
    spin channels; kept is what the diagnostics read without building it
    again: on casida 'args' = (w, eps_t, eps_r, occupied_r, omega) of the
    environment, 'srg_s', the s applied
    by 'srg', and, with infinite and sigma 'srg', 'srg_inf', Sigma_bar at
    s = inf from the same block loop (`_srg_pair`; the damping diagnostic's
    reference); on imagfrequency and space-time 'z' and 'samples', the Pade
    nodes and S^E there, and 'fit', the (Z, A) of `pade_fit_elementwise`
    (None without an environment)."""
    if low.mode != 'casida':
        s_bar, kept = _embedding_imag_axis(low, part, sigma)
        return [s_bar], [kept]
    chans = _channels(low, part)
    residues = screened_residues(low, part)
    residues = [residues] if len(chans) == 1 else list(residues)
    sigma_bar, kepts = [], []
    for chan, w in zip(chans, residues):
        env = chan.part.env
        args = (w, chan.eps[chan.part.act], chan.eps[env], env < chan.nocc,
                low.omega)
        kept = {'args': args, 'srg_s': settings.srg_s}
        if sigma == 'srg':
            s_bar, kept['srg_inf'] = _srg_pair(
                *args, settings.srg_s, _sigma_tile_gb(low.numerics), infinite)
        else:
            s_bar = mode_a_self_energy(*args)
        sigma_bar.append(s_bar)
        kepts.append(kept)
    return sigma_bar, kepts


def _screening_grid(low):
    """(points, W): the full RPA's screened interaction W(i w) =
    [1 - chi0(i w)]^-1 in the normalized fitting basis, on the minimax grid of
    numerics.imagfreq.nfreq points over the particle-hole window
    (`screening_fit_grid`, shared with the GW density's
    `self_energy_matrix_imaginary_axis`), built on the first request and
    kept in low.cache['w_grid'].

    imagfrequency: chi0 from the three-index factors, exact at each point.
    space-time: chi0 of the ISDF factors (X, D) on those same points, through
    a minimax tau axis with as many points as the static grid's
    (`chi0_imaginary_frequency`, the mid-gap mu), then the same Dyson step."""
    return _cached(low, 'w_grid', lambda: _build_screening_grid(low))


def _build_screening_grid(low):
    """The (points, W) of `_screening_grid`, computed."""
    e_min, e_max, mu = particle_hole_window(low.eps, low.nocc)
    points, weights, _ = screening_fit_grid(low.eps, low.nocc, low.numerics)
    if low.mode == 'space-time':
        x, d = low.cache['factors']
        ntau = len(low.cache['static_grid'].tau_points)
        grid = TimeFrequencyGrid.minimax_split(ntau, e_min, e_max, points,
                                               weights, with_sine=False,
                                               with_inverse=False)
        w = chi0_imaginary_frequency(
            x, d, low.eps, low.nocc, grid, mu=mu,
            tile_memory_gb=layout_tile_gb(low.numerics))
        for k in range(len(w)):
            w[k] = dyson_screening(w[k])
        return points, w
    lr = LinearResponseSolver(low.eps, coeff_df=low.factors,
                              spin_mode='restricted')
    return points, lr.solve_rpa_screening(points, low.nocc, is_imaginary=True)


def sigma_env_imag_axis(low, part, query_freqs):
    """S^E_tu(mu + i w) (nquery, n_act, n_act) complex, at w = query_freqs:
    the correlation self-energy iG(W - v) of the full RPA's W on part.act
    (occupied first) with the internal orbital restricted to part.env, mu
    the whole system's mid-gap. imagfrequency and space-time
    (NotImplementedError on casida, which has the residues instead).

    Both routes: the GW density's `fitted_self_energy_matrix_imaginary_axis`
    on the fitting factors B[Q, t, r] (t active, r environment) and the
    environment's eps_r - mu: the Lorentzian fit of W - 1 on
    `_screening_grid` with the poles of `screening_fit_grid` and the GW
    regularization (GW_DENSITY_W_FIT_TIKHONOV),
    whose one-pole convolution with G_r is exact; the full system's mu and
    fit range, so only the internal sum changes. The samples are an exact
    rational function of the frequency, which is what lets the Pade
    continuation reach orbitals far from the gap.

    imagfrequency slices B from the three-index factors. space-time builds
    it from the ISDF factors, B_Q,tr = sum_P D_PQ X_Pt X_Pr, for the active
    and environment columns alone (N^3 times the active orbitals), and uses
    the W of its own chi0.
    """
    if low.mode == 'casida':
        raise NotImplementedError(
            f"sigma_env_imag_axis samples the screened interaction of the "
            f"numerical routes on the imaginary axis; this low-level result "
            f"is mode='casida' (its self-energy is `embedding_self_energy`, "
            f"from the residues of W): use mode='imagfrequency' or "
            f"mode='space-time'.")
    n, env = part.n_act, part.env
    query_freqs = np.atleast_1d(np.asarray(query_freqs, dtype=float))
    if n == 0 or len(env) == 0:
        return np.zeros((len(query_freqs), n, n), dtype=complex)
    mu = particle_hole_window(low.eps, low.nocc)[2]
    points, w_grid = _screening_grid(low)
    poles = screening_fit_grid(low.eps, low.nocc, low.numerics)[2]
    if low.mode == 'space-time':
        x, d = low.cache['factors']
        x_act, x_env = x[:, part.act], x[:, env]
        pairs = (x_act[:, :, None] * x_env[:, None, :]).reshape(len(x), -1)
        b = (d.T @ pairs).reshape(d.shape[1], n, len(env))
    else:
        b = low.factors[:, part.act][:, :, env]
    return fitted_self_energy_matrix_imaginary_axis(
        b, low.eps[env] - mu, points, w_grid, query_freqs, poles,
        GW_DENSITY_W_FIT_TIKHONOV)


def _evaluate_continued(fit, eps_act, sigma):
    """The static reduction of a `pade_fit_elementwise` fit (Z, A) of S^E:
    'srg', Re S_tu(z_tu) with z_tu = (eps_t + eps_u)/2 + i |eps_u - eps_t|/2
    (Eq. 42; S(z*) = S(z)*, so the half plane of the fit's nodes serves both
    t, u and u, t); 'mode_a', 1/2 Re[S_tu(eps_t) + S_tu(eps_u)]."""
    e = np.asarray(eps_act, dtype=float)
    n = len(e)
    if sigma == 'srg':
        z = 0.5 * (e[:, None] + e[None, :]) + 0.5j * np.abs(e[None, :]
                                                            - e[:, None])
        at = pade_eval_elementwise(z.ravel(), *fit).reshape((n,) * 4)
        return np.einsum('tutu->tu', at).real
    half = np.einsum('ttu->tu', pade_eval_elementwise(e, *fit)).real
    return 0.5 * (half + half.T)


def continued_self_energy(z, samples, eps_act, sigma):
    """Sigma_bar (n_act, n_act) from S^E sampled at the complex frequencies z
    (samples (nz, n_act, n_act), symmetric): one Thiele-Pade fit per element
    (`pade_fit_elementwise`, greedy order, the upper triangle), then each
    element at its own point, Re S_tu(z_tu) for 'srg' (Eq. 42, s = inf) or
    1/2 Re[S_tu(eps_t) + S_tu(eps_u)] for 'mode_a', eps_act the active
    orbitals' energies."""
    if len(eps_act) == 0:
        return np.zeros((0, 0))
    fit = pade_fit_elementwise(z, samples, symmetric=True)
    return _evaluate_continued(fit, eps_act, sigma)


def _embedding_imag_axis(low, part, sigma):
    """`_embedding` on imagfrequency and space-time: S^E on the Pade nodes
    of the run's numerics (`pade_nodes`, on the line mu + i w), fitted
    once."""
    nodes = pade_nodes(numerics_or_default(low.numerics).quadrature)
    z = particle_hole_window(low.eps, low.nocc)[2] + 1j * nodes
    samples = sigma_env_imag_axis(low, part, nodes)
    kept = {'z': z, 'samples': samples, 'fit': None}
    n = part.n_act
    if n == 0 or len(part.env) == 0:
        return np.zeros((n, n)), kept
    kept['fit'] = pade_fit_elementwise(z, samples, symmetric=True)
    return _evaluate_continued(kept['fit'], low.eps[part.act], sigma), kept


class SigmaDampingWarning(RuntimeWarning):
    """The SRG damping of Sigma_bar^E is not negligible: an active orbital
    energy lies within a few 1/sqrt(2 s) of a pole of the environment
    self-energy, so 'srg' there depends on srg_s."""


class SigmaContinuationWarning(RuntimeWarning):
    """Two Pade fits of S^E on disjoint halves of the nodes disagree by more
    than DOWNFOLDING_SIGMA_CONTINUATION_WARN at an element's evaluation
    point: the continuation of Sigma_bar^E is not to be trusted there."""


def _pole_weights(w, eps_t, eps_r, occupied_r, omega):
    """Z_t = 1/(1 - S'_tt(eps_t)) of each active orbital t, S the pole sum
    of `mode_a_self_energy`, S_tt(x) = sum_rn w_tr^2 / (x - eps_r - sgn
    Omega_n), so S'_tt(eps_t) = -sum_rn w_tr^2 / D_tr^2 and Z_t = 1/(1 +
    sum_rn w_tr^2 / D_tr^2), in (0, 1]: 1 far from every pole, towards 0
    near one. A term exactly at a pole (D_tr = 0) makes Z_t 0."""
    d = _denominators(eps_t, eps_r, occupied_r, omega)
    w2, d2 = w * w, d * d
    ratio = np.divide(w2, d2, out=np.where(w2 > 0, np.inf, 0.0),
                      where=d2 > 0)
    return 1.0 / (1.0 + ratio.sum(axis=(0, 2)))


# The step of the central difference that gives S'_tt(eps_t) from the
# continued fraction, Ha: the fraction is smooth on the real axis inside the
# pole-free window (the poles are at least the pole margin away), so the
# O(h^2) error is far below the continuation's own.
_SLOPE_STEP = 1e-5


def _continued_pole_weights(fit, eps_act):
    """Z_t = 1/(1 - S'_tt(eps_t)) from a Pade fit of S^E, the slope by a
    central difference of the continued fraction at eps_t. 1 for every t
    without a fit (no environment)."""
    e = np.asarray(eps_act, dtype=float)
    if fit is None:
        return np.ones(len(e))
    h = _SLOPE_STEP
    step = (pade_eval_elementwise(e + h, *fit)
            - pade_eval_elementwise(e - h, *fit)).real / (2.0 * h)
    return 1.0 / (1.0 - np.einsum('ttt->t', step))


def _sigma_poles(low, chans, sigma, sigma_bar, kept):
    """The 'sigma_poles' record, and a SigmaDampingWarning (the
    named flag 'sigma_srg_damping') naming the active orbitals whose row of
    Sigma_bar the SRG damping moves by more than DOWNFOLDING_SRG_DAMPING_WARN.
    sigma_bar and kept are the lists over the spin channels of `_embedding`.
    For two channels window, pole_distance and z are pairs (alpha, beta);
    srg_s is the one s of both and srg_damping the largest over both.

    window: `pole_window`. pole_distance: per active orbital, the distance
        of eps_t to the nearer edge of the window, negative outside it.
    z: per active orbital, Z_t = 1/(1 - S'_tt(eps_t)): of the pole sum on
        casida (`_pole_weights`), of the continued fraction on imagfrequency
        and space-time.
    srg_s: the s applied, None for s = inf (always on imagfrequency and
        space-time, which apply no damping) and for 'mode_a' (undamped).
    srg_damping: max |Sigma_bar_srg(s) - Sigma_bar_srg(inf)|; 0.0 when no
        finite s is applied.
    """
    windows, distances, weights, named = [], [], [], []
    applied, damping = None, 0.0
    for chan, s_bar, kept_s in zip(chans, sigma_bar, kept):
        lo, hi = _window(low, chan)
        act = chan.part.act
        eps = chan.eps[act]
        dist = np.minimum(eps - lo, hi - eps)
        windows.append((lo, hi))
        distances.append(dist)
        if low.mode != 'casida':
            weights.append(_continued_pole_weights(kept_s['fit'], eps))
            continue
        weights.append(_pole_weights(*kept_s['args']))
        applied = kept_s['srg_s'] if sigma == 'srg' else None
        if applied is not None and s_bar.size:
            moved = np.abs(s_bar - kept_s['srg_inf'])
            damping = max(damping, float(moved.max()))
            rows = moved.max(axis=1) > DOWNFOLDING_SRG_DAMPING_WARN
            named += [f'{_orbital(chan)} {p} (eps = {e:.6f}, distance to the '
                      f'window {x:.6f})'
                      for p, e, x in zip(act[rows], eps[rows], dist[rows])]
    if named:
        named = ', '.join(named)
        flag(
            'sigma_srg_damping',
            f"sigma='srg': the SRG damping (s = {applied:g}) changes "
            f"Sigma_bar^E by up to {damping:.2e} Ha on the active {named}"
            f", within a few 1/sqrt(2 s) = {1 / np.sqrt(2 * applied):.4f}"
            f" Ha of a pole of the environment self-energy (pole-free "
            f"{_window_text(chans, windows)}); Sigma_bar depends on srg_s "
            f"there.", SigmaDampingWarning, stacklevel=4)
    return {'window': _per_channel(windows),
            'pole_distance': _per_channel(distances),
            'z': _per_channel(weights), 'srg_s': applied,
            'srg_damping': damping}


def continuation_spread(z, samples, eps_act, sigma):
    """(n_act, n_act): per element, |Sigma_bar from the even-indexed z -
    Sigma_bar from the odd-indexed z|, each by `continued_self_energy` at
    the element's own evaluation point. Two fits on disjoint halves of the
    nodes: an indicator of the continuation's error, not a bound on it. Far
    from the poles it can exceed the error of the full fit; near the edge of
    the pole-free window it falls short of it by ten to twenty times, which
    is why the margin, not this spread, guards the edge."""
    z, samples = np.asarray(z), np.asarray(samples)
    even, odd = (continued_self_energy(z[k::2], samples[k::2], eps_act,
                                       sigma) for k in (0, 1))
    return np.abs(even - odd)


def _sigma_continuation(low, part, sigma, kept):
    """The 'sigma_continuation' record: `continuation_spread` on
    the nodes Sigma_bar was continued from, and a SigmaContinuationWarning
    (the named flag 'sigma_continuation_spread') naming the elements (as
    orbital pairs) above DOWNFOLDING_SIGMA_CONTINUATION_WARN. Zero without
    an environment."""
    n = part.n_act
    if kept['fit'] is None:
        return np.zeros((n, n))
    spread = continuation_spread(kept['z'], kept['samples'],
                                 low.eps[part.act], sigma)
    bad = np.argwhere(np.triu(spread > DOWNFOLDING_SIGMA_CONTINUATION_WARN))
    if len(bad):
        act = part.act
        named = ', '.join(f'({act[t]}, {act[u]}) {spread[t, u]:.2e}'
                          for t, u in bad)
        flag(
            'sigma_continuation_spread',
            f"sigma={sigma!r} on mode={low.mode!r}: two Pade fits of the "
            f"environment self-energy on the even and the odd nodes differ "
            f"by more than {DOWNFOLDING_SIGMA_CONTINUATION_WARN:g} Ha at the "
            f"elements {named} (orbital pairs, Ha); the continued Sigma_bar^E "
            f"there is uncertain. More nodes (numerics.quadrature.npade) or "
            f"mode='casida' avoid the continuation.",
            SigmaContinuationWarning, stacklevel=4)
    return spread


def _sigma_name(sigma):
    """None, or the SIGMAS entry the string `sigma` names, in any case and
    with '-' or '_' (`choice`, as screening and mode); refused (ValueError)
    otherwise. None is kept apart, not passed through `choice`, which would
    read it as 'none' and so accept the string 'None'."""
    if sigma is None:
        return None
    if isinstance(sigma, str):
        try:
            return choice(sigma, SIGMAS[1:], 'sigma')
        except ValueError:
            pass
    raise ValueError(f"Unknown sigma={sigma!r}; choose one of "
                     + ', '.join(repr(s) for s in SIGMAS) + '.')


def _diagnostic_names(diagnostics):
    """The DIAGNOSTICS entries `diagnostics` names (a name, an iterable of
    names, or None for none), each refused unless it is one."""
    if diagnostics is None:
        return set()
    if isinstance(diagnostics, str):
        diagnostics = (diagnostics,)
    return {choice(d, DIAGNOSTICS, 'diagnostics') for d in diagnostics}


def terms(low, part, screening='crpa', sigma=None, diagnostics=DIAGNOSTICS,
          **opts):
    """The DownfoldingTerms of the RPA on the active space `part`.

    `part` is one Partition for a restricted low-level result, and the terms
    are those of one spin channel: vtilde = (v,), sigma_bar = (s,). For an
    unrestricted one it is the pair (alpha, beta), and vtilde = (aa, ab, bb),
    sigma_bar = (alpha, beta); the correlation energies and the static
    defect are those of both spins together, and in the diagnostics what
    belongs to one spin (the pole-free window, the pole distances, Z) is a
    pair.

    screening: 'bare' | 'crpa' | 'crpa_ph' | 'mrpa' (casida only, refused
        on imagfrequency and space-time); vtilde is its static correction
        (`screening_correction`), the same in every spin block (restricted).
    sigma: None, the paper's scheme: full double counting, no Sigma_bar.
        'srg' or 'mode_a': the static environment self-energy
        (`embedding_self_energy`), one block for the restricted reference,
        with dc='hartree',
        the Hartree-only double counting that goes with it; E_0 then gains
        the exchange energy term and -Tr[Sigma_bar gamma^A,0].
        Allowed with every screening; the third term of the
        embedding self-energy, iG^A(W - W^A[v_eff]), is dropped, which is
        exact only for 'crpa' in the static limit. On imagfrequency and
        space-time 'srg' is s = inf (Eq. 42). Near the poles Sigma_bar is
        computed with a SigmaNearPoleWarning ('mode_a' on casida; both
        forms on the other routes, pointing to casida), and refused only
        when it is not finite (`embedding_self_energy`). The
        names in any case and with '-' or '_' ('SRG', 'mode-a'), as
        screening; anything but None and those is refused.
    diagnostics: the DIAGNOSTICS to record in energies.diagnostics (None:
        none, as ()); a name that does not apply here (on casida
        'ec_active_casida' and 'sigma_continuation', and 'sigma_poles' and
        'sigma_continuation' without sigma) is skipped and absent, an
        unknown name refused. One that applies is recorded even when it is
        exactly zero (an empty space, no environment, no ph pair: 0.0).
        'sigma_poles' (`_sigma_poles`): the window, pole distances, Z and
        the SRG damping. 'sigma_continuation' (imagfrequency, space-time,
        `_sigma_continuation`): per element the spread of two Pade fits on
        the even and the odd nodes. Their two warnings do not depend on
        this keyword: with sigma set, a SigmaDampingWarning (casida, the
        SRG damping not negligible) and a SigmaContinuationWarning (the
        spread above DOWNFOLDING_SIGMA_CONTINUATION_WARN) are always
        checked and raised, since Sigma_bar goes into H_eff either way;
        the keyword only decides whether the numbers are recorded.
        'ec_active_casida' (imagfrequency, space-time): {'casida': the
        dense-Casida E_c^A on the same v_eff, 'difference': the route's
        quadrature value minus it}.

    E_c[v] is the full RPA's; E_c^A[v_eff] is the dRPA of the active ph
    pairs with v_eff = v + vtilde, v the same integrals as the full RPA
    (low.factors or low.eri), not the exact v of H_eff. 'bare'
    keeps E_c[v] - E_c^A[v], the mechanical embedding.
    """
    _refuse_options(opts, 'terms')
    names = _diagnostic_names(diagnostics)
    screening = choice(screening, SCREENINGS, 'screening')
    sigma = _sigma_name(sigma)
    chans = _channels(low, part)
    vt = _two_body_blocks(screening_correction(low, part, screening), chans)
    v_eff = tuple(v + t for v, t in zip(_active_v(low, chans), vt))
    found = {}
    if 'static_defect' in names:
        found['static_defect'] = static_defect(low, part, v_eff)
    e_c_active = active_correlation_energy(low, part, v_eff)
    if 'ec_active_casida' in names and low.mode != 'casida':
        pairs = any(0 < c.part.n_act_occ < c.part.n_act for c in chans)
        dense = _active_ec_casida(low, chans, v_eff) if pairs else 0.0
        found['ec_active_casida'] = {'casida': dense,
                                     'difference': e_c_active - dense}
    sigma_bar, dc = None, 'full'
    if sigma is not None:
        s_bars, kepts = _embedding(low, part, sigma, infinite=True,
                                   stacklevel=4)
        sigma_bar, dc = tuple(s_bars), 'hartree'
        # the two checks of Sigma_bar always run and warn; `diagnostics`
        # only decides whether their numbers are kept
        checks = {'sigma_poles': _sigma_poles(low, chans, sigma, s_bars,
                                              kepts)}
        if low.mode != 'casida':
            checks['sigma_continuation'] = _sigma_continuation(
                low, part, sigma, kepts[0])
        found.update({name: value for name, value in checks.items()
                      if name in names})
    return DownfoldingTerms(
        vtilde=vt, sigma_bar=sigma_bar, e_c_full=low.e_c_full,
        e_c_active=e_c_active, dc=dc, diagnostics=found)
