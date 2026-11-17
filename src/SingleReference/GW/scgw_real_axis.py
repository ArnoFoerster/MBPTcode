"""Real-axis part of scGW: from a Pade fit of the converged Sigma(i.omega_n)
to G(omega + i.eta), the spectral function A(omega) and its quasiparticle
peaks.

`finish_qp_from_pade` is what `scgw.solve_qp_energy_scgw` hands the fit to.
Sigma(omega) is evaluated from the fit a chunk of frequencies at a time
(`trace_Greal_from_pade`, `diag_Greal_from_pade`), so no
(nfreq, nmo, nmo) array is stored. `pack_pade` / `unpack_pade` store the fit
compactly in the scGW .h5 file.
"""
import resource

import numpy as np
from pyscf.data.elements import chemcore

from src.Base.utils.grids.analytic_continuation import pade_eval_elementwise
from src.SingleReference.GW.spectral_peaks import analyze_peaks, analyze_orbital_peaks

PADE_EVAL_CHUNK = 16        # real frequencies per Sigma(w) / G(w) evaluation in trace_Greal_from_pade


def trace_Greal_from_pade(freq_points, Z, A, F, eta, chunk=PADE_EVAL_CHUNK):
    """Tr[G(z)] at z = omega + i.eta, with Sigma(z) evaluated from its Pade fit.

    G(z) = [z I - F - Sigma(z)]^-1, with Sigma taken at the same
    z = omega + i.eta. This is not double broadening: it is G at one complex
    point, so every pole of G becomes a Lorentzian of half-width eta and height
    Z / (pi * eta). Taking Sigma at omega + i0 instead would give widths Z * eta
    and the same height 1 / (pi * eta) for every peak, whatever its weight.

    Sigma(z) comes from the fit (Z, A) of pade_fit_elementwise, chunk
    frequencies at a time, so
    neither the (nfreq, nmo, nmo) Sigma nor G is ever stored: memory is the
    fit plus a few (chunk, nmo, nmo) arrays, whatever the number of frequencies.
    """
    eye = np.eye(F.shape[0])
    tr_G = np.empty(len(freq_points), dtype=complex)
    for start in range(0, len(freq_points), chunk):
        z = np.asarray(freq_points[start:start + chunk]) + 1j*eta
        sigma = pade_eval_elementwise(z, Z, A)
        tr_G[start:start + chunk] = np.trace(
            np.linalg.inv(z[:, None, None]*eye - F - sigma), axis1=1, axis2=2)
    return tr_G

def diag_Greal_from_pade(freq_points, Z, A, F, eta, U=None, chunk=PADE_EVAL_CHUNK):
    """Diagonal of G(z) at z = omega + i.eta, (nfreq, nmo), in the basis of
    the (real, orthonormal) columns of U; U=None is the MO basis F is in.

    As trace_Greal_from_pade, Sigma(z) comes from the Pade fit, chunk
    frequencies at a time; the sum over the last axis is Tr[G]. With U,
    G U = M^-1 U is one solve, the cost of the inverse itself, and
    (U^T G U)_ii = sum_j U_ji (G U)_ji, so U^T G U is never formed.
    """
    eye = np.eye(F.shape[0])
    diag_G = np.empty((len(freq_points), F.shape[0]), dtype=complex)
    for start in range(0, len(freq_points), chunk):
        z = np.asarray(freq_points[start:start + chunk]) + 1j*eta
        M = z[:, None, None]*eye - F - pade_eval_elementwise(z, Z, A)
        if U is None:
            diag_G[start:start + chunk] = np.diagonal(np.linalg.inv(M), axis1=1, axis2=2)
        else:
            GU = np.linalg.solve(M, np.broadcast_to(U, M.shape))
            diag_G[start:start + chunk] = (U * GU).sum(axis=-2)
    return diag_G


def pack_pade(Z, A, omega_points):
    """A symmetric `pade_fit_elementwise` fit (Z, A) in compact form for saving.

    Every node in Z is one of the fit points mu + i*omega_points, so it is
    stored as an index into omega_points (smallest integer type that fits)
    instead of a complex number, and only the upper triangle is kept.
    Elements below the fit's floor have all-zero coefficients, so their
    nodes do not matter; they get index 0.

    Returns {'node_index': (n_nodes, npair), 'coeff': (n_nodes, npair)}, with
    npair = nmo (nmo + 1) / 2 in np.triu_indices order; `unpack_pade` inverts it.
    """
    iu = np.triu_indices(Z.shape[-1])
    Z_u, A_u = Z[:, iu[0], iu[1]], A[:, iu[0], iu[1]]

    sorter = np.argsort(omega_points)
    pos = np.searchsorted(omega_points[sorter], Z_u.imag)
    index = sorter[np.clip(pos, 0, len(omega_points) - 1)]
    fitted = (A_u != 0).any(axis=0)
    index[:, ~fitted] = 0
    if not np.array_equal(omega_points[index][:, fitted], Z_u.imag[:, fitted]):
        raise ValueError('Pade nodes are not on omega_points')

    return {'node_index': index.astype(np.min_scalar_type(len(omega_points) - 1)),
            'coeff': A_u}

def unpack_pade(pade, omega_points, mu):
    """Inverse of `pack_pade`: the full (Z, A), each (n_nodes, nmo, nmo),
    ready for `pade_eval_elementwise` or `trace_Greal_from_pade`."""
    index, A_u = pade['node_index'], pade['coeff']
    npair = A_u.shape[1]
    nmo = int(round((np.sqrt(8 * npair + 1) - 1) / 2))
    iu = np.triu_indices(nmo)

    Z = np.zeros((len(A_u), nmo, nmo), dtype=complex)
    A = np.zeros((len(A_u), nmo, nmo), dtype=complex)
    Z[:, iu[0], iu[1]] = mu + 1j * np.asarray(omega_points)[index]
    A[:, iu[0], iu[1]] = A_u
    Z[:, iu[1], iu[0]] = Z[:, iu[0], iu[1]]
    A[:, iu[1], iu[0]] = A[:, iu[0], iu[1]]
    return Z, A


def finish_qp_from_pade(mf, mol, pade_Z, pade_A, F, mu, E_gm, save_file=None, **kwargs):
    """Real-axis part of `solve_qp_energy_scgw`, from a Pade fit of Sigma.

    Builds A(omega) = -Tr[Im G(omega + i.eta)] / pi with Sigma from the fit
    (pade_Z, pade_A) of `pade_fit_elementwise` (or `unpack_pade`) and takes
    the quasiparticle energies from its peaks. Shared by the run that has
    just fitted Sigma and the reconstruction from a saved fit.

    kwargs: `eta` the real-axis broadening; `A_window` and `A_range` the
        margin around and the range of the real frequency axis;
        `min_qp_weight` the smallest |weight| of a kept peak (default 0.7,
        see `analyze_peaks`); `return_peak_info` (default False) to also
        return the peaks' weights and dip flags;
        `analysis`: 'sum' (default) searches the peaks of A itself;
        'separate' builds A_i = -Im G_ii / pi in the eigenbasis of F and
        searches each degenerate group of A_i on its own
        (`analyze_orbital_peaks`), so every weight is a Z per orbital and
        each quasiparticle energy belongs to an orbital (about 10% more on
        the inversions, the Pade evaluation is unchanged);
        `degen_tol` (default 1e-4 Ha) the eps_F spacing below which
        orbitals are grouped, for 'separate'.
    save_file: if given, (w_real, A, mu) is written there as CSV (A the
        total, also for 'separate').

    Returns (qpes, A, w_real, E_gm, mu), as `solve_qp_energy_scgw`, plus a
    dict with return_peak_info. With 'sum', qpes are the kept peaks of A
    and the dict is {'weights', 'negative'}, aligned with qpes. With
    'separate', qpes has one entry per orbital of F (eps_F order; nan when
    the orbital has no kept peak in the window, e.g. high virtuals) and the
    dict is that of `analyze_orbital_peaks` plus 'A_diag'.
    """
    # The frequency window runs from the lowest valence level to the highest
    # level of F, with 0.5 Ha margin. eta is the broadening; the frequency
    # step is finer, so each peak (half-width ~eta) is well sampled.
    eta = kwargs.get('eta', 1e-2)
    w_step = eta / 10

    analysis = kwargs.get('analysis', 'sum').lower()
    if analysis not in ('sum', 'separate'):
        raise ValueError(f"analysis must be 'sum' or 'separate', got {analysis!r}")

    eps_F, U_F = np.linalg.eigh(F)

    # the spectral function's window and range (kwargs A_window, A_range)
    A_window = kwargs.get('A_window') or 0.5
    eps_low = eps_F[max(chemcore(mol) - getattr(mf, "frozen_core", 0), 0)]
    A_range = kwargs.get('A_range') or [eps_low - A_window, 2 * mu - eps_low + A_window]

    w_real = np.arange(A_range[0], A_range[1], w_step)
    print(f'\nSigma(w) on {len(w_real)} real frequencies in [{w_real[0]}, {w_real[-1]}] Ha, '
          f'eta = {eta}: Pade fit {pade_Z.shape}')

    min_qp_weight = kwargs.get('min_qp_weight', 0.7)

    # Sigma(w) is evaluated a chunk of frequencies at a time inside
    # trace_Greal_from_pade / diag_Greal_from_pade, so
    # (len(w_real), nmo, nmo) is never stored.
    if analysis == 'sum':
        tr_G_w = trace_Greal_from_pade(w_real, pade_Z, pade_A, F, eta)
        A = -1/np.pi * tr_G_w.imag
    else:
        A_diag = -1/np.pi * diag_Greal_from_pade(w_real, pade_Z, pade_A, F, eta, U=U_F).imag
        A = A_diag.sum(axis=1)

    if save_file is not None:
        np.savetxt(save_file, (w_real, A, np.full_like(w_real, mu)), delimiter=',')

    if analysis == 'sum':
        # Peaks and dips of A, filtered on their integrated weight (analyze_peaks).
        qpes, weights, negative = analyze_peaks(w_real, A, eta, w_step,
                                                min_qp_weight=min_qp_weight)
        peak_info = {'weights': weights, 'negative': negative}
    else:
        qpes, peak_info = analyze_orbital_peaks(w_real, A_diag, eps_F, mu, eta, w_step,
                                                min_qp_weight=min_qp_weight,
                                                degen_tol=kwargs.get('degen_tol', 1e-4))
        weights, negative = peak_info['weights'], peak_info['negative']
        peak_info['A_diag'] = A_diag

    # A dip means Im G has the wrong sign there (the continuation breaks
    # causality), so a HOMO or LUMO read off a dip is flagged. HOMO / LUMO
    # are the highest / lowest qpe below / above mu (qpes need not be sorted
    # and can hold nan with 'separate').
    for label, side in (('HOMO', qpes < mu), ('LUMO', qpes > mu)):
        if side.any():
            idx = np.flatnonzero(side)
            i = idx[np.argmax(qpes[idx])] if label == 'HOMO' else idx[np.argmin(qpes[idx])]
            if negative[i]:
                print(f'WARNING: the {label} peak (w = {qpes[i]:.6f} Ha, weight = '
                      f'{weights[i]:.3f}) is a dip in A: the Pade continuation is '
                      f'not causal there; try another eta.')

    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # ru_maxrss is in KiB on Linux
    print(f'\nPeak memory usage: {peak_gib} GiB\n')

    if kwargs.get('return_peak_info', False):
        return qpes, A, w_real, E_gm, mu, peak_info
    return qpes, A, w_real, E_gm, mu
