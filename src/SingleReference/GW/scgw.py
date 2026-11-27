"""Self-consistent GW (scGW) on intermediate-representation (IR) grids.

Everything is done in the MO basis of the starting mean-field calculation.
Each iteration of the cycle is:

    G(i.omega) -> G(tau) -> chi(tau) -> chi(i.omega) -> W(i.omega) -> W(tau)
               -> Sigma(tau) = G(tau) W(tau) -> Sigma(i.omega)
    F = h + 2J[gamma] - K[gamma]
    G(i.omega) = [(i.omega + mu) I - F - Sigma(i.omega)]^-1

with mu adjusted so that Tr[gamma] equals the number of occupied orbitals.

Two IR grids are used, both at the same beta and omega_max: a fermion grid
for G and Sigma, and a boson grid for chi and W. Once converged, Sigma is
continued to the real axis with Pade and the quasiparticle energies are read
off the peaks of the spectral function. The total energy is the
Galitskii-Migdal energy of the converged G and Sigma
(`galitskii_migdal_energy`).

`solve_qp_energy_scgw` is the entry point. This module holds the driver (IR
grids, the self-consistency cycle with its DIIS packing, saving and loading
the .h5); the pieces it calls live in
    scgw_sigma.py      G -> Sigma (ISDF and RI)
    scgw_dyson.py      F, Dyson equation, density matrix, mu search, GM energy
    scgw_real_axis.py  G(omega) from the Pade fit of Sigma, A(omega)
    spectral_peaks.py  quasiparticle peaks of A(omega)
"""
import numpy as np

import h5py
from pyscf import lib
from src.Solvers.diis import DIIS
from src.Base.numerics.constants import SURFACE_GRID_ACCURACY
from src.Base.pyscf.pyscf_interface import get_orbital_energies
from src.Base.utils.grids.time_frequency import (TimeFrequencyGrid)
from src.Base.orbital_spaces import get_occ_virt_indices
from src.SingleReference.GW.space_time import separable_factors
from src.Base.utils.grids.analytic_continuation import (pade_fit_elementwise,
                                                        pade_eval_elementwise)
from src.SingleReference.GW.scgw_dyson import (density_matrix_ir, galitskii_migdal_energy,
                                               build_new_F, build_G_from_F, mu_cycle)
# calc_new_sigma is imported by name and looked up in this module's namespace
# by the cycle, so tests can replace it with scgw.calc_new_sigma = ...
from src.SingleReference.GW.scgw_sigma import build_B_ri, calc_new_sigma
# The real-axis part; trace/diag_Greal_from_pade and the peak analysis are
# also imported here so that they can still be imported from scgw.
from src.SingleReference.GW.scgw_real_axis import (finish_qp_from_pade, pack_pade, unpack_pade,
                                                   trace_Greal_from_pade, diag_Greal_from_pade)
from src.SingleReference.GW.spectral_peaks import analyze_peaks, analyze_orbital_peaks

# --- Convergence parameters ---
# Outer scGW cycle
RESIDUAL_CONVERGENCE = 1e-7 # max |Sigma_out - Sigma_in|, |F_out - F_in| at which the cycle stops
LINEAR_MIXING = 1.0         # fraction of the new (Sigma, F) kept in linear-mixing steps
DIIS_SIZE = 8               # DIIS subspace; 0 mixes linearly throughout
DIIS_START = 3              # linear-mixing iterations before DIIS takes over
MAX_SCF_ITER = 150          # cap on scGW iterations

def pack_sigma_F(sigma, F, sigma_weight=1.0):
    """(Sigma, F) as one real vector, for DIIS.

    Sigma is stored as its complex memory read as float64 (real and
    imaginary parts interleaved), so the plain dot product of two such
    vectors is Re<e1|e2>, the DIIS inner product, and `unpack_sigma_F` can
    return views instead of copies.
    sigma_weight scales the Sigma block: it has nfreq times more entries
    than F and would otherwise dominate the DIIS B-matrix. It is applied to
    the error vectors only; the iterates are packed unweighted, so DIIS
    combines them without unpacking the weight again.

    sigma: (nfreq, nmo, nmo) complex. F: (nmo, nmo) real.
    """
    n2 = 2 * sigma.size
    x = np.empty(n2 + F.size)
    x[:n2] = np.ascontiguousarray(sigma).view(np.float64).ravel()
    x[n2:] = F.ravel()
    if sigma_weight != 1.0:
        x[:n2] *= sigma_weight
    return x

def unpack_sigma_F(x, sigma_shape):
    """Inverse of `pack_sigma_F` (for sigma_weight = 1). Sigma and F are
    views into x, not copies: the packed DIIS vector and the Dyson input
    share memory, and x stays alive as long as either is used."""
    n2 = 2 * int(np.prod(sigma_shape))
    sigma = x[:n2].view(np.complex128).reshape(sigma_shape)
    F = x[n2:].reshape(sigma_shape[1:])
    return sigma, F

def pack_G(G):
    """G as one real vector for DIIS, laid out as in `pack_sigma_F`; a view
    of G's memory (no copy) when G is contiguous."""
    return np.ascontiguousarray(G).view(np.float64).ravel()

def unpack_G(x, G_shape):
    """Inverse of `pack_G`; a view of x."""
    return x.view(np.complex128).reshape(G_shape)

def _solve_qp_energy_scgw(mf, mol, nocc, skip_scf=False, method='ISDF', grid_kind='IR', save_file=None,
                          save_sigma=False, save_W0=False, **kwargs):
    """Self-consistent GW on IR grids, up to (not including) the real axis.

    Starts from the mean-field G0 and iterates Sigma and F to
    self-consistency (DIIS on (Sigma, F), convergence on their residual).
    The total energy is the Galitskii-Migdal energy of the converged G and
    Sigma.

    mf, mol: converged pyscf mean-field object and its molecule.
    nocc: number of doubly occupied orbitals.
    skip_scf: stop after the first Dyson step (see Returns).
    grid_kind: only 'IR' is implemented.
    save_file: if given, the converged F_mo, mu, density (dm_ao, gamma_mo),
        mo_coeff, the Matsubara frequencies omega_points, E_gm, E_c, e_hf
        and method are written to save_file + '.h5' (pyscf chkfile, key
        'scgw', with mol), together with what rebuilds the three-index
        factor in the gauge of W0_aux: nocc, mo_energy (of mf), frozen_core,
        auxbasis (when given) and, for ISDF, the factors isdf_D and
        isdf_X_ao.
    save_sigma: also write the converged Sigma (sigma_mo, on omega_points)
        to the h5. Off by default: it is (n_omega, nmo, nmo) complex and
        can be very large for large systems. Ignored without save_file.
    save_W0: also write the static screened interaction W(i omega = 0) of
        the converged G (W0_aux, (naux, naux) in the auxiliary basis of the
        ISDF factor D or the RI tensor B, for BSE). Every Sigma build returns
        its W0, a naux^2 copy, and only the latest is kept, so the one
        written is that of the last Sigma build, from the converged G.
        Ignored without save_file.
    method: 'ISDF' (separable factors) or 'RI' (plain RI three-index
        tensor, needs `auxbasis`); see `calc_new_sigma`.
    kwargs: `factors` (precomputed output of `separable_factors`) or
        `auxbasis` and `grid_accuracy` (ISDF grid level, default
        SURFACE_GRID_ACCURACY; used to build them); `auxbasis` is also the
        RI auxiliary basis; `beta` and `omega_max` override
        the IR grid parameters that are otherwise derived from mf's
        orbital energies (e.g. to run different starts on the same grid);
        `diis_size`, `diis_start` and `mixing` override DIIS_SIZE,
        DIIS_START and LINEAR_MIXING; `mix` picks what is mixed, 'sigma_F'
        (default) or 'G' (see the comment at the cycle).

    Returns (sigma, F, mu, omega_points, E_gm): the converged Sigma(i w_n)
    and F, of which the converged G is the Dyson solution, the chemical
    potential, the fermion Matsubara frequencies Sigma is on, and the
    Galitskii-Migdal total energy (`galitskii_migdal_energy`).
    With skip_scf=True, returns
    (G, F, G0, mu, h_mo, grid_f, grid_b, X_mo, D) instead.
    """

    if grid_kind.upper() != 'IR':
        raise NotImplementedError('The only method now implemented is the IR method')

    # --- Starting point: mean-field orbitals, mu in the middle of the gap ---
    eps = get_orbital_energies(mf, representation='spatial')

    occ, virt = get_occ_virt_indices(eps, nocc)
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])
    F_bare = np.diag(eps)

    # --- IR grids: fermion for G/Sigma, boson for chi/W ---
    # omega_max must cover G, W and Sigma = G*W at once, so it is the sum of
    # the largest orbital energy (from mu) and the largest transition, padded.
    dG = np.abs(eps - mu)
    e_min = eps[virt].min() - eps[occ].max()
    e_max = eps[virt].max() - eps[occ].min()
    omega_max = kwargs.get('omega_max') or 3.0 * (dG.max() + e_max)

    # beta must be large enough that the gap is effectively at T = 0, but
    # not larger: Lambda = beta * omega_max also sets how well conditioned
    # the omega -> tau transform is. Measured on Ne/cc-pVDZ, with the default
    # Matsubara sampling: 100/dG.min() diverged within ~5 iterations,
    # 25/dG.min() was stable.
    beta = kwargs.get('beta') or 25.0 / dG.min()

    # The default Matsubara sampling only reaches 1.1 - 1.4 x omega_max,
    # which leaves the transform ill-conditioned near tau = 0 and beta, where
    # the density is read off. n_matsubara_factor is chosen so the sampling
    # reaches about `reach` x omega_max, from
    #     omega_top / omega_max ~ 1 + factor * size * 2 pi / Lambda.
    # size_guess overestimates the IR size, so the true reach is a bit lower.
    # Measured on Ne/cc-pVDZ at Lambda = 1.7e4: reach 1.12 diverges, 5.4 is
    # stable.
    reach = 6.0
    size_guess = 100
    n_matsubara_factor = int(np.ceil(
        (reach - 1.0) * beta * omega_max / (2 * np.pi * size_guess)))

    grid_f = TimeFrequencyGrid.ir(beta, omega_max, statistics='fermion',
                                    n_matsubara_factor=n_matsubara_factor)
    
    grid_b = TimeFrequencyGrid.ir(beta, omega_max, statistics='boson',
                                    n_matsubara_factor=n_matsubara_factor)

    print(f'IR grids: Lambda={beta * omega_max:.4g}, '
            f'fermion size={grid_f.meta["basis"].size}, '
            f'boson size={grid_b.meta["basis"].size}, '
            f'n_matsubara_factor={n_matsubara_factor}, '
            f'highest sampled omega = '
            f'{grid_f.omega_points.max() / omega_max:.2f} x omega_max')
    # Grid quality check. duality_error is used rather than roundtrip_error
    # because the latter's test function is bosonic and so not valid for
    # the fermion grid.
    print(f'\nfermion duality_error (even/odd) = '
            f'{grid_f.duality_error("even"):.2e} / {grid_f.duality_error("odd"):.2e}')
    print(f'boson   duality_error (even/odd) = '
            f'{grid_b.duality_error("even"):.2e} / {grid_b.duality_error("odd"):.2e}\n')

    method = method.upper()
    if method == 'ISDF':
        X_mo, D, X_ao, _ = (kwargs.get('factors') if kwargs.get('factors') is not None
                                    else separable_factors(mf, mol, auxbasis=kwargs.get('auxbasis'),
                                                        grid_accuracy=kwargs.get('grid_accuracy',
                                                                                    SURFACE_GRID_ACCURACY)))
        B = None
    elif method == 'RI':
        if kwargs.get('auxbasis') is None:
            raise ValueError("If method == 'RI' auxbasis must be specified")
        B = build_B_ri(mol, mf.mo_coeff, kwargs.get('auxbasis'))

        X_mo = None
        D = X_ao = None
    else:
        raise ValueError(f"Unknown method {method!r}, expected 'ISDF' or 'RI'")

    # --- First iteration: G0 -> gamma, Sigma -> F -> G ---
    freq_points = grid_f.omega_points
    nmo = len(eps)
    eye = np.eye(nmo)
    G0 = np.array([np.linalg.inv((1j * w + mu) * eye - F_bare)
                    for w in freq_points])

    print(f'G0 (IR, bare): shape={G0.shape}, max|G0|={np.max(np.abs(G0)):.4e}')

    gamma = density_matrix_ir(G0, grid_f)
    print(f'Trace gamma_G0: {np.trace(gamma)}, expected nocc: {nocc}\n')

    sigma_new = calc_new_sigma(G0, grid_f, grid_b, X_mo=X_mo, D=D, B=B, method=method)

    h_mo = mf.mo_coeff.T @ mf.get_hcore(mol) @ mf.mo_coeff
    e_nuc = mol.energy_nuc()

    F = build_new_F(gamma, mf, h_mo)
    G = build_G_from_F(mu, grid_f.omega_points, sigma_new, F)

    if skip_scf:
        return G, F, G0, mu, h_mo, grid_f, grid_b, X_mo, D
    del G0

    # --- Self-consistency cycle ---
    # mix='sigma_F': the iterate is the Dyson input x_in = (Sigma_in, F_in).
    #   1. x_out = (Sigma[G], F[gamma[G]]) from the current G
    #   2. residual r = x_out - x_in; stop when it is below tolerance
    #   3. new x_in by linear mixing (first diis_start iterations, or when
    #      the DIIS B-matrix is singular) or DIIS
    #   4. mu search with x_in fixed -> G, gamma
    # The mu search comes last, so every G is the Dyson solution of x_in with
    # Tr[gamma] = nocc, and its GM energy is meaningful at every iteration.
    #
    # mix='G': the iterate is G_in (gamma_in = gamma[G_in] is linear in G,
    # so mixing G mixes gamma as well).
    #   1. Sigma[G_in], F[gamma_in], mu search -> G_out
    #   2. residual r = G_out - G_in; stop when it is below tolerance
    #   3. new G_in by linear mixing or DIIS
    # Every G_out has Tr[gamma] = nocc and the mixing coefficients sum to 1,
    # so G_in keeps the right particle number, but it is not the Dyson
    # solution of any Sigma; the GM energy is that of G_out.
    #
    # mu itself is not mixed in either: the search fixes it from Sigma and F.
    mix = kwargs.get('mix', 'sigma_F')
    if mix not in ('sigma_F', 'G'):
        raise ValueError(f"mix={mix!r}: choose 'sigma_F' or 'G'")
    alpha = kwargs.get('mixing', LINEAR_MIXING)
    diis_size = kwargs.get('diis_size', DIIS_SIZE)
    diis_start = kwargs.get('diis_start', DIIS_START)
    diis = DIIS(diis_size, start_iter=0) if diis_size > 0 else None

    # Weight of the Sigma block in the DIIS error, see pack_sigma_F.
    sigma_weight = 1.0 / np.sqrt(len(freq_points))
    sigma_shape = sigma_new.shape
    n_sigma = 2 * sigma_new.size    # length of the Sigma block of a packed vector

    # With mix='sigma_F', x_in is the only copy of the iterate: sigma_in and
    # F_in are views into it (`unpack_sigma_F`), so the Dyson input and the
    # DIIS vector are not stored twice.
    if mix == 'sigma_F':
        x_in = pack_sigma_F(sigma_new, F)
        sigma_in, F_in = unpack_sigma_F(x_in, sigma_shape)
    else:
        x_in = None
        sigma_in, F_in = sigma_new, F
    del sigma_new, F, G, gamma
    # print(f'Sigma_in: {sigma_in.nbytes / 2**30:.3f} GiB')
    mu, G, gamma = mu_cycle(mu, F_in, nocc, sigma_mo=sigma_in,
                            grid_f=grid_f, grid_kind='IR')

    print(f'\n----SCF (mixing {mix})----\n')

    # W0 = W(i omega = 0) of the latest Sigma build, overwritten each
    # iteration; at the break it is that of the converged G.
    want_W0 = save_W0 and save_file is not None

    def sigma_and_W0(G):
        if want_W0:
            return calc_new_sigma(G, grid_f, grid_b, X_mo=X_mo, D=D, B=B,
                                  method=method, return_W0=True)
        return calc_new_sigma(G, grid_f, grid_b, X_mo=X_mo, D=D, B=B, method=method), None

    residual = np.inf
    for it in range(MAX_SCF_ITER):
        mu_old = mu
        if mix == 'sigma_F':
            # 1. Outputs from the current G
            sigma_out, W0 = sigma_and_W0(G)
            F_out = build_new_F(gamma, mf, h_mo)
            x_out = pack_sigma_F(sigma_out, F_out)
            del sigma_out, F_out

            # 2. Residual, x_out - x_in, read off the packed vectors and
            # then weighted in place into the DIIS error.
            error = x_out - x_in
            res_sigma = np.max(np.abs(error[:n_sigma].view(np.complex128)))
            res_F = np.max(np.abs(error[n_sigma:]))
            residual = max(res_sigma, res_F)
            print(f'Iteration {it}: residual Sigma: {res_sigma:.3e}, F: {res_F:.3e}')
            if residual <= RESIDUAL_CONVERGENCE:
                del x_out, error
                break
            error[:n_sigma] *= sigma_weight
        else:
            # 1. G_out from the current G_in
            sigma_in, W0 = sigma_and_W0(G)
            F_in = build_new_F(gamma, mf, h_mo)
            mu, G_out, gamma_out = mu_cycle(mu, F_in, nocc, sigma_mo=sigma_in,
                                            grid_f=grid_f, grid_kind='IR')

            # 2. Residual. x_in and x_out are views of G and G_out, so only
            # the error is a new array.
            x_in = pack_G(G)
            x_out = pack_G(G_out)
            error = x_out - x_in
            res_G = np.max(np.abs(error.view(np.complex128)))
            res_gamma = np.max(np.abs(gamma_out - gamma))
            residual = res_G
            print(f'Iteration {it}: residual G: {res_G:.3e}, gamma: {res_gamma:.3e}')
            if residual <= RESIDUAL_CONVERGENCE:
                G, gamma = G_out, gamma_out
                x_in = None
                del G_out, gamma_out, x_out, error
                break

        # 3. Mix
        x_new = None
        if diis is not None and it >= diis_start:
            try:
                x_new = diis.compute_new_vec(x_out, error)
                step = 'DIIS'
            except np.linalg.LinAlgError:
                # Near convergence the error vectors can become linearly
                # dependent; take a linear step instead.
                pass
        if x_new is None:
            # x_in + alpha * (x_out - x_in), with one new array instead of three
            x_new = x_out - x_in
            x_new *= alpha
            x_new += x_in
            step = f'linear (alpha = {alpha})'
        del x_out, error

        if mix == 'sigma_F':
            x_in = x_new
            sigma_in, F_in = unpack_sigma_F(x_in, sigma_shape)
            del x_new, G, gamma

            # 4. Dyson + particle number
            mu, G, gamma = mu_cycle(mu, F_in, nocc, sigma_mo=sigma_in,
                                    grid_f=grid_f, grid_kind='IR')
            G_dyson = G
        else:
            # x_in is a view of the old G; drop it so that G can be freed.
            x_in = None
            G = unpack_G(x_new, sigma_shape)
            gamma = density_matrix_ir(G, grid_f)
            G_dyson = G_out
            del x_new, G_out, gamma_out

        # G_dyson is the Dyson solution for sigma_in, so this is the GM
        # energy of the current iterate; it converges with the cycle.
        E_gm, E_c = galitskii_migdal_energy(G_dyson, sigma_in, mf, h_mo, grid_f, e_nuc)
        print(f'Step: {step}\nmu: {mu} (delta mu: {abs(mu - mu_old):.3e})\n'
              f'E_GM: {E_gm} (E_c: {E_c})\n\n')
        del G_dyson
    else:
        raise RuntimeError(f'scGW: not converged in {MAX_SCF_ITER} iterations '
                           f'(residual = {residual:.2e})')

    print('\n----END SCF----\n')

    # G is the Dyson solution of (sigma_in, F_in); these are what is returned
    # and continued to the real axis, so G, F and Sigma stay consistent.
    sigma_new, F = sigma_in, F_in

    E_gm, E_c = galitskii_migdal_energy(G, sigma_new, mf, h_mo, grid_f, e_nuc)
    print(f'Galitskii-Migdal energy: E = {E_gm} Ha, E_c = {E_c} Ha\n')

    if save_file is not None:
        # Read back with lib.chkfile.load(save_file + '.h5', 'scgw') and
        # load_mol. gamma is per spin in the MO basis of mf, so the total AO
        # density is 2 C gamma C^T. With save_sigma, sigma_mo, F_mo, mu and
        # omega_points are enough to redo the real-axis continuation without
        # the cycle; sigma_mo is (n_omega, nmo, nmo) complex, so it is off by
        # default.
        data = {
            'dm_ao': 2.0 * mf.mo_coeff @ gamma @ mf.mo_coeff.T,
            'gamma_mo': gamma, 'mo_coeff': mf.mo_coeff, 'F_mo': F,
            'omega_points': grid_f.omega_points,
            'mu': mu, 'E_gm': E_gm, 'E_c': E_c, 'e_hf': mf.e_tot,
            'method': method,
            # What a later step (e.g. BSE on W0_aux) needs to rebuild the
            # three-index factor in the same auxiliary gauge, and the mean
            # field it started from. The orbital basis is in the saved mol.
            'nocc': nocc, 'mo_energy': eps,
            'frozen_core': int(getattr(mf, 'frozen_core', 0))}
        if kwargs.get('auxbasis') is not None:
            # a basis name; a dict basis is stored by its repr
            aux = kwargs['auxbasis']
            data['auxbasis'] = aux if isinstance(aux, str) else repr(aux)
        if method == 'ISDF':
            # The ISDF gauge is the fit itself (grid and auxbasis), so the
            # factors are stored rather than the keywords that made them:
            # B[P,p,q] = sum_k (X_ao C)[k,p] (X_ao C)[k,q] D[k,P].
            data['isdf_D'] = D
            if X_ao is not None:
                data['isdf_X_ao'] = X_ao
        if save_sigma:
            data['sigma_mo'] = sigma_new
        if want_W0:
            data['W0_aux'] = W0
        lib.chkfile.save(save_file + '.h5', 'scgw', data)
        lib.chkfile.save_mol(mol, save_file + '.h5')

    # Everything else from the cycle (G, the factors, DIIS, grid_b) is freed
    # on return, before the real-axis arrays are allocated.
    return sigma_new, F, mu, grid_f.omega_points, E_gm

def solve_qp_energy_scgw(mf, mol, nocc, ref_file=None, method='ISDF', grid_kind='IR', skip_scf=False, save_file=None,
                         save_pade=False, **kwargs):
    """Self-consistent GW quasiparticle energies on IR grids.

    Runs the scGW cycle (`_solve_qp_energy_scgw`), then continues the
    converged Sigma to the real axis with Pade and takes the quasiparticle
    energies from the peaks of the spectral function
    A(omega) = -Tr[Im G(omega)] / pi (`finish_qp_from_pade`).

    Arguments are those of `_solve_qp_energy_scgw`, plus:
    ref_file: an .h5 file written by an earlier run (its save_file + '.h5').
        If given, the cycle is skipped and the real-axis part is redone from
        the saved 'scgw/sigma_mo' (refitted), or else from 'scgw/pade'
        (the saved fit, used as is). mf and mol must be those of that run.
    save_file: if given, (w_real, A, mu) is also written there as CSV.
    save_sigma, save_W0: passed on to `_solve_qp_energy_scgw` through kwargs.
    save_pade: also write the Pade fit of Sigma to save_file + '.h5' under
        'scgw/pade' (see `pack_pade`), enough to redo A(omega), e.g. at a
        smaller eta, without Sigma. Smaller than sigma_mo, but fixes the
        fit (max_order, node choice). Ignored without save_file.
    kwargs: `eta` the real-axis broadening; `A_window` and `A_range` the
        margin around and the range of the real frequency axis;
        `min_qp_weight`, `return_peak_info`, `analysis` ('sum' or
        'separate') and `degen_tol` (see `finish_qp_from_pade`).

    Returns (qpes, A, w_real, E_gm, mu): the peak positions, the spectral
    function, the real frequency axis it is on, the Galitskii-Migdal
    total energy of the converged G and Sigma (`galitskii_migdal_energy`)
    and the converged chemical potential. With analysis='separate', qpes
    has one entry per orbital of F (nan where none). With
    return_peak_info=True, a sixth element gives each peak's integrated
    weight and whether it is a dip in A (see `finish_qp_from_pade`).
    With skip_scf=True (and no ref_file), returns
    (G, F, G0, mu, h_mo, grid_f, grid_b, X_mo, D) instead.
    """
    if ref_file is None:
        out = _solve_qp_energy_scgw(mf, mol, nocc, skip_scf=skip_scf, method=method,
                                    grid_kind=grid_kind, save_file=save_file, **kwargs)
        if skip_scf:
            return out
        sigma, F, mu, omega_points, E_gm = out

    else:
        if not ref_file.endswith('.h5'):
            raise ValueError(f"ref_file must be an .h5 file, got '{ref_file}'")

        with h5py.File(ref_file, 'r') as f:
            if 'scgw' not in f:
                raise ValueError(f"No 'scgw' group in {ref_file}")
            keys = f['scgw'].keys()
            has_sigma, has_pade = 'sigma_mo' in keys, 'pade' in keys

        if not (has_sigma or has_pade):
            raise ValueError('Ref. file was given to skip calculation, but no '
                             f'sigma or pade weights found in {ref_file}')

        F            = lib.chkfile.load(ref_file, 'scgw/F_mo')
        mu           = lib.chkfile.load(ref_file, 'scgw/mu')
        omega_points = lib.chkfile.load(ref_file, 'scgw/omega_points')
        E_gm         = lib.chkfile.load(ref_file, 'scgw/E_gm')

        if not has_sigma:
            # Saved fit, used as is: no Sigma(i*w_n) to refit or check against.
            pade_Z, pade_A = unpack_pade(lib.chkfile.load(ref_file, 'scgw/pade'),
                                         omega_points, mu)
            return finish_qp_from_pade(mf, mol, pade_Z, pade_A, F, mu, E_gm,
                                        save_file=save_file, **kwargs)

        sigma = lib.chkfile.load(ref_file, 'scgw/sigma_mo')

    # --- Real axis: Pade-continue Sigma, build G(omega), find peaks of A ---
    pade_max_order = 24
    z_fit = mu + 1j * omega_points

    # Fit once; finish_qp_from_pade then evaluates Sigma(w) from the fit.
    pade_Z, pade_A = pade_fit_elementwise(z_fit, sigma, greedy=True,
                                          symmetric=True, max_order=pade_max_order)

    print(f'Pade check, max |Sigma_pade(i*w_n) - Sigma(i*w_n)|: '
        f'{np.max(np.abs(pade_eval_elementwise(z_fit, pade_Z, pade_A) - sigma))}\n')
    del sigma

    if save_file is not None and save_pade:
        # Next to the cycle's results; unpack_pade(d['pade'], d['omega_points'],
        # d['mu']) with d = lib.chkfile.load(save_file + '.h5', 'scgw') gives
        # back (Z, A) for trace_Greal_from_pade, e.g. to redo A at another eta.
        lib.chkfile.save(save_file + '.h5', 'scgw/pade',
                         pack_pade(pade_Z, pade_A, omega_points))

    return finish_qp_from_pade(mf, mol, pade_Z, pade_A, F, mu, E_gm,
                                save_file=save_file, **kwargs)
