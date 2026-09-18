import numpy as np

from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.GW.imaginary_axis import solve_screening_imaginary_axis
from src.Base.utils.grids import gauss_legendre_grid, gap_scaled_w0, minimax_frequency_grid


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
