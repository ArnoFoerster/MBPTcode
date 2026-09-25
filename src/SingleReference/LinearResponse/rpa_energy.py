"""The dRPA correlation energy, and the ground-state total energy it completes.

    E_0 = E_ref + (E_x^HF[rho] - E_xc[rho]) + E_c^dRPA

is the plasmon (Klein) ground state of Toelle, Kitsaras and Loos
(arXiv:2507.02160) Eq. (15), and every quasiparticle, BSE or embedded surface
sits on it: a total energy is only comparable with another when the two share
this functional, which is what `src.Base.declaration.GroundState` declares and
`ground_state_energy` here is the ONE assembly of. The middle term is the
exact-exchange double counting that turns E_KS into E_HF at the same density
-- identically zero on a Hartree-Fock reference, 10.8 eV on B3LYP/water -- so
every starting point reaches the same E_0.

The exchange decomposition of the reference (`xc_hybrid_coeff`, `rsh_split`,
`exchange_channels`) and the double-counting energy live here, with the energy
they belong to, rather than beside the nuclear derivatives that differentiate
them: a forward energy that can only be reached through a gradient module is a
forward energy the gradient module owns.
"""
from dataclasses import dataclass

import numpy as np
from pyscf import scf as pyscf_scf

from src.Base.declaration import GroundState
from src.Base.isdf_jk import ISDFJK
from src.Base.utils.grids import gauss_legendre_grid, gap_scaled_w0, minimax_frequency_grid
from src.Base.utils.mpi_grid import lockstep
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.GW.imaginary_axis import solve_screening_imaginary_axis


@dataclass(frozen=True)
class GroundStateEnergy:
    """E_0 in Hartree with the additive terms its declaration names.

    `terms` is name -> Hartree in the order of `GroundState.terms()` and sums
    to `total`. An E_0 reported as one number cannot be checked against another
    route's: the terms are what say WHICH functional the number belongs to.
    """
    total: float
    terms: dict
    declaration: GroundState


def rpa_correlation_energy_casida(lr_solver, nocc):
    """dRPA/ring-CCD correlation energy from the RPA (Casida) spectrum: E_c = 1/2(sum_s Omega_s - Tr[A]) (Furche's trace formula).

    Works with either DF or full-ERI LinearResponseSolver (dispatched inside build_casida_matrices).
    """
    A, B = lr_solver.build_casida_matrices(nocc, lBSE=False)
    trace_a = np.trace(A)
    solver = CasidaSolver(A, B)
    del A, B
    omega, _, _ = solver.solve()
    return 0.5 * (np.sum(omega) - trace_a)


def solve_polarizability_imaginary_axis(lr_solver, nocc, freq_points):
    """Z(i*omega_k) = P^(0)(i*omega_k).v in the whitened DF/RI-V basis (eq 27, Spadetto et al., JCTC 2023, 19, 1499).

    Restricted spin, DF only. Recovered as Z = I - W^-1 from the already-computed
    screened interaction W(iw), rather than rebuilding chi0 from scratch.
    """
    W_grid = solve_screening_imaginary_axis(lr_solver, nocc, freq_points)
    naux = W_grid.shape[-1]
    eye = np.eye(naux)
    return np.array([eye - np.linalg.inv(W) for W in W_grid])


def rpa_correlation_energy_imaginary_axis(lr_solver, nocc, nfreq=20, grid='minimax', w0=None):
    """dRPA correlation energy via imaginary-frequency-axis integration (eqs 26-28, Spadetto et al., JCTC 2023, 19, 1499).

    E_c = (1/2pi) int dw Tr{log(1-Z(iw)) + Z(iw)} = sum_k w_k Tr{log(1-Z(i*wk)) + Z(i*wk)},
    with Tr[log(1-Z)] via slogdet. Restricted spin, DF only. Should agree with
    rpa_correlation_energy_casida to grid precision.

    grid: 'minimax' (default) or 'gauss_legendre' (uses w0, default gap_scaled_w0).
    """
    eps = lr_solver.eps
    occ, virt = get_occ_virt_indices(eps, nocc)

    if grid == 'minimax':
        e_min = eps[virt].min() - eps[occ].max()
        e_max = eps[virt].max() - eps[occ].min()
        freq_points, freq_weights = minimax_frequency_grid(nfreq, e_min, e_max)
    elif grid == 'gauss_legendre':
        if w0 is None:
            w0 = gap_scaled_w0(eps, nocc)
        freq_points, freq_weights = gauss_legendre_grid(nfreq, w0=w0)
    else:
        raise ValueError(f"Unknown grid '{grid}'; choose 'minimax' or 'gauss_legendre'.")

    Z_grid = solve_polarizability_imaginary_axis(lr_solver, nocc, freq_points)
    naux = Z_grid.shape[-1]
    eye = np.eye(naux)

    e_c = 0.0
    for k, Z in enumerate(Z_grid):
        _, logdet = np.linalg.slogdet(eye - Z)
        e_c += freq_weights[k] * (logdet + np.trace(Z))
    return e_c / (2.0 * np.pi)


def xc_hybrid_coeff(mf):
    """(is_ks, a_x): the exact-exchange fraction the reference's Fock carries.

    ONE scalar, so this describes a GLOBAL hybrid and nothing else. On a
    range-separated hybrid it returns alpha, the long-range fraction, and says
    nothing about the erf-screened term that carries beta -- see `rsh_split`,
    and `require_no_range_separation` for the refusal that keeps that silence
    from reaching a gradient.
    """
    if not hasattr(mf, 'xc'):
        return False, 1.0
    return True, float(mf._numint.hybrid_coeff(mf.xc, spin=mf.mol.spin))


def rsh_split(mf):
    """(omega, alpha, beta) of the reference's exchange; omega = 0 if global.

    pyscf's convention is K = alpha K_full + beta K_lr(omega), so the
    short-range fraction is alpha + beta: CAM-B3LYP is
    (0.33, 0.65, -0.46) and wB97X (0.30, 1.00, -0.842).
    """
    if not hasattr(mf, 'xc'):
        return 0.0, 1.0, 0.0
    omega, alpha, beta = mf._numint.rsh_coeff(mf.xc)
    return float(omega), float(alpha), float(beta)


def exchange_channels(mf):
    """[(omega, weight)] with the reference's exact exchange = sum w K(omega).

    pyscf builds K_eff = hyb K_full + (alpha - hyb) K_lr(omega) and its
    `rsh_coeff` reports (omega, alpha, beta) with hyb = alpha + beta, so the
    channels are (0, alpha + beta) and (omega, -beta). Checked against
    `get_veff`'s own operator to machine precision on B3LYP, PBE0, CAM-B3LYP,
    wB97X and LRC-wPBEh.

    `xc_hybrid_coeff` IS NOT THIS and must not be used for a skeleton on a
    range-separated hybrid: it returns alpha, which for a global hybrid happens
    to be the full-range weight and for an RSH is the LONG-range one. On
    LRC-wPBEh it gives 1.0 where the full-range weight is 0.2, so a skeleton
    built on it is wrong in the term it keeps as well as in the one it drops.
    """
    omega, alpha, beta = rsh_split(mf)
    if omega == 0.0 or beta == 0.0:
        return [(0.0, alpha)]
    return [(0.0, alpha + beta), (float(omega), -beta)]


def exx_double_counting(mf, mol=None):
    """E_x^exact[D] - E_xc[rho]: what turns E_KS into E_HF at the SAME density.

    The Klein/plasmon ground state of Toelle Eq. (15) is E_0 = E_HF + E_c^dRPA,
    with E_HF the HARTREE-FOCK energy. A Kohn-Sham mean field does not supply
    it: E_KS carries E_xc, whose correlation the dRPA term would then count a
    second time and whose exchange is not the exact one. Adding this returns
    the exact exchange and removes the double counting, so ANY starting point
    lands on the same E_0.

        E_HF[rho] = E_KS[rho] + (E_x^exact[rho] - E_xc[rho])

    A hybrid needs care: pyscf's `nr_rks` energy excludes the exact exchange
    its E_xc contains, so that term is added back -- ONE PER CHANNEL, since a
    range-separated hybrid carries an erf-attenuated term as well as a
    full-range one and adding back only alpha K_full would leave the larger
    half out. Identically zero on Hartree-Fock.
    """
    mol = mf.mol if mol is None else mol
    is_ks, _ = xc_hybrid_coeff(mf)
    dm = mf.make_rdm1()

    def e_x_at(omega):
        with mol.with_range_coulomb(omega):
            k = mf.get_k(mol, dm, hermi=1, omega=omega) if omega else \
                mf.get_k(mol, dm, hermi=1)
        return -0.25 * float(np.einsum('ij,ji->', dm, k))

    e_x = e_x_at(0.0)
    if not is_ks:
        return 0.0
    e_xc = float(mf._numint.nr_rks(mol, mf.grids, mf.xc, dm)[1])
    e_xc += sum(w * e_x_at(o) for o, w in exchange_channels(mf))
    return e_x - e_xc


def reference_energy(mf, mol=None):
    """E_0^HF of the plasmon formula: the Hartree-Fock energy AT this density.

    For a Hartree-Fock mean field this is just `mf.e_tot`. FOR A KOHN-SHAM ONE
    IT IS NOT, and the difference is not a detail:

        E_KS - E_HF[rho] = E_xc[rho] - E_x^exact[rho]

    so `mf.e_tot + E_c^dRPA` on a KS reference adds the RPA correlation on top
    of the correlation already inside E_xc -- a double count -- and carries an
    approximate exchange where the Klein functional wants the exact one.
    Toelle/Kitsaras/Loos 2025 Eq. (15) writes E_0 = E_0^HF + (1/2) sum_beta
    Omega_beta^RPA - (1/2) Tr(A^RPA) and says E_0^HF is the HF ground-state
    energy; RPA@KS therefore means E_HF[rho_KS] + E_c^RPA, which is what this
    returns.

    E_HF IS REASSEMBLED, not corrected: a Hartree-Fock mirror at this density
    on the mean field's own interaction. A density-fitted mean field gets a
    fresh fit on the same auxiliary basis; an ISDF one gets ITS OWN `ISDFJK`
    -- the interpolation points, collocation and fit matrix the SCF converged
    with, J from the same integral-direct DF-J -- which is the Hartree-Fock
    partner `exx_double_counting_skeleton` differentiates. A fresh density fit
    there is a different functional: 5.5e-4 Ha off on a C1 water/cc-pVDZ at
    148 points per atom, PBE0 and LRC-wPBEh alike, and the dRPA force missed a
    difference of it by 9.46e-4 Ha/Bohr against 1.5e-8 now. `exx_double_counting`
    is the same quantity taken from the operators instead, E_x^exact - E_xc on
    the mean field's own K; the two agree to 1e-14 on either route.

    DO NOT TAKE THIS ENERGY WITHOUT ITS GRADIENT. A chain that adopts this and
    leaves `mf.Gradients()` alone is WORSE than one that adopts neither: with
    both on `mf.e_tot` the force is at least the derivative of the energy, so
    an optimizer finds a stationary point of something, while half-adopted it
    is stationary for NEITHER -- and that converges silently, looking like a
    relaxed geometry. Either refuse the Kohn-Sham gradient outright or land the
    energy, the orbital response `exx_double_counting_Y` and the skeleton
    `exx_double_counting_skeleton` together, the last with `grid_response=True`
    on the mean field's own Kohn-Sham gradient, which is differenced against
    it. All three vanish identically on Hartree-Fock, which is why that path
    stays bitwise unchanged.

    The mirror's energy is rank 0's on every rank: each rank builds it with
    pyscf's threaded J/K, whose OpenMP GEMM adds its partial sums in
    thread-arrival order, so the ranks' own mirrors differ in their last bits.
    """
    mol = mf.mol if mol is None else mol
    if not xc_hybrid_coeff(mf)[0]:
        return float(mf.e_tot)
    dm = mf.make_rdm1()
    hf = pyscf_scf.RHF(mol)
    with_df = getattr(mf, 'with_df', None)
    if with_df is not None:
        hf = hf.density_fit(auxbasis=with_df.auxbasis)
    if isinstance(with_df, ISDFJK):
        # the SCF's own factors: a new fit would carry its own interpolation
        hf.with_df = with_df
    e_hf = lockstep(float(hf.energy_tot(dm=dm)))
    if not hasattr(mf, 'with_solvent'):
        return e_hf
    # The reaction field is the same functional of the same density in both
    # members, so it survives the exchange substitution untouched and is taken
    # as what the continuum added to THIS mean field. Omitting it would leave a
    # surface with no solvation energy under a force that has one, which is
    # stationary for neither.
    gas = mf.undo_solvent()
    return e_hf + lockstep(float(mf.e_tot) - float(gas.energy_tot(dm=dm)))


def declared_ground_state(mf, kind='rpa'):
    """The `GroundState` a mean field carries: its functional, 'hf' for Hartree-Fock."""
    return GroundState(kind, getattr(mf, 'xc', None) or 'hf')


def ground_state_energy(ground_state, mf, mol=None, e_corr=None,
                        environment=None):
    """E_0 and its terms, as the declaration says. THE ONE ASSEMBLY OF E_0.

    kind='dft'  E_0 = E_KS[xc], the mean field's own energy. There is no
                correlation term to add, so an `e_corr` is refused rather than
                dropped: a caller holding one means a different functional.
    kind='rpa'  E_0 = E_HF[rho] + E_c^dRPA (Toelle Eq. 15), reported as
                E_ref + (E_x^HF - E_xc) + E_c^dRPA. The middle term is taken as
                E_HF[rho] - E_ref, the double counting AS THE TOTAL CARRIES IT,
                so the three terms sum to `total` exactly; it is the quantity
                `exx_double_counting` builds from the operators, and is
                identically zero on a Hartree-Fock reference.

    e_corr: E_c^dRPA in Hartree, the caller's -- built on whatever interaction
        the caller's route screens with, which is what makes this assembly and
        not the correlation energy the shared thing.
    environment: the environment the caller's declaration names. It reaches E_0
        through the mean field, whose `e_tot` already carries the reaction
        field, and through `e_corr`, built on the dressed interaction; nothing
        here applies it, so the only thing to do with it is refuse one that is
        not the mean field's, which would be a number and a declaration that
        describe different surfaces.
    """
    attached = getattr(mf, 'with_screening', None)
    if environment is not None and environment is not attached:
        raise ValueError(
            f'{ground_state.label()} was declared in {environment!r}, but the '
            f'mean field carries {attached!r}: E_0 reaches the environment '
            f'through the mean field and through E_c, so the number would be '
            f'one surface and the declaration another.')
    if ground_state.kind == 'dft':
        if e_corr is not None:
            raise ValueError(
                f"{ground_state.label()} is the mean field's own energy and has "
                f'no correlation term, so e_corr={e_corr!r} belongs to a '
                f"different functional: declare GroundState('rpa', "
                f'{ground_state.xc!r}) to carry it.')
        total = float(mf.e_tot)
        return GroundStateEnergy(total, {'E_ref': total}, ground_state)
    if e_corr is None:
        raise ValueError(
            f'{ground_state.label()} is E_HF + E_c^dRPA and e_corr is None: '
            f'dropping E_c moves the surface rather than its zero, since E_c '
            f'is a functional of the geometry (6.3 eV on water/cc-pVDZ). '
            f"Declare GroundState('dft', {ground_state.xc!r}) for the mean "
            f"field's own surface.")
    e_ref = float(mf.e_tot)
    e_hf = reference_energy(mf, mol)
    e_c = float(e_corr)
    return GroundStateEnergy(e_hf + e_c,
                             {'E_ref': e_ref, 'E_x^HF - E_xc': e_hf - e_ref,
                              'E_c^dRPA': e_c},
                             ground_state)
