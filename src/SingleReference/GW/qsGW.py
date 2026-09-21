"""
qsGW: the orbitals and the eigenvalues reinjected until the static Hermitian
self-energy stops moving.

evGW moves the eigenvalues and keeps the mean field's orbitals. Here a static
Hermitian self-energy Sigma~ replaces v_xc altogether: the Fock-like matrix
h + J[D] + K[D] + Sigma~ is diagonalized, its eigenvectors define the next
density, W and Sigma, and the cycle repeats to a fixed point that no longer
remembers the starting functional:

    eps, mo_coeff, info = qsgw_eigenvalues(mf, mol, screening='updated')
    info['cycles'], info['converged'], info['w_aux'], info['df_coeff']

Sigma~ is Marie and Loos's SRG-regularized form (J. Chem. Theory Comput. 2023,
doi 10.1021/acs.jctc.3c00281). Its diagonal is that of Kotani, van Schilfgaarde
and Faleev's mode A, 1/2 [Sigma_pq(eps_p) + Sigma_pq(eps_q)] (Phys. Rev. B 76,
165106, 2007), with every term whose energy denominator lies within about
1/sqrt(2 s) of zero damped; off the diagonal the two differ at every s
(SelfEnergySolver.static_self_energy_matrix). Plain mode A on the pole sum puts
high virtuals on poles of Sigma, and its loop does not settle; `flow=None`
still selects it, with eta as the broadening. qsGW0 keeps the mean field's RPA
Casida solution, `screening='fixed'`: W stays, the amplitudes and the poles
follow the rotated orbitals. Restricted spin, Casida route only.

The DF factors are the mean field's, rotated into the current orbitals,
B_P,pq = sum_mn U_mp B_P,mn U_nq with U = C0^T S C, and never rebuilt: without
with_df they are an eigendecomposition of the MO-basis ERI, whose auxiliary
index would follow the orbitals and leave qsGW0's transition density, W and
the returned factors in three different auxiliary bases.
"""
import warnings

import numpy as np
import scipy.linalg
from pyscf import scf
from pyscf.scf import diis as scf_diis

from src.Base.constants import (DEFAULT_BROADENING_ETA, EVGW_MAX_CYCLE, EVGW_TOL,
                                HARTREE_TO_EV, QSGW_BLOCK_ELEMS, QSGW_DIIS_SIZE,
                                QSGW_DM_TOL, QSGW_MIXING_LAMBDA, QSGW_SRG_FLOW,
                                get_method_info)
from src.Base.environment import environment_of
from src.Base.pyscf_interface import get_density_fitting_coefficients
from src.SingleReference.GW import qp_energy  # module import: it imports this file
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

#: What the iterate screens: the Casida problem rebuilt from it (qsGW), or the
#: mean field's kept and only the amplitudes and poles moved (qsGW0).
SCREENINGS = ('updated', 'fixed')
#: How the AO Hamiltonian is mixed between cycles: PySCF's CDIIS, or linear
#: mixing, Kaplan et al.'s H <- lambda H_new + (1 - lambda) S C diag(eps) C^T S.
MIXINGS = ('diis', 'linear')


def _rpa_spectrum(eps, coeff, nocc, eta, tda):
    """(omega, X, Y) of the RPA Casida problem on `eps` with the DF factors
    `coeff`, and the solver it was built with."""
    lr = LinearResponseSolver(eps, coeff_df=coeff, spin_mode='restricted', eta=eta)
    spectrum = qp_energy._casida_spectrum(lr, nocc, 'RPA', None, tda,
                                          {'GW': get_method_info('GW')}, ['GW'],
                                          False, True)
    return spectrum, lr


def _rotated_factors(coeff0, c0, ovlp, mo_coeff):
    """B_P,pq of the orbitals `mo_coeff`, shape (naux, nmo, nmo), from the mean
    field's B_P,mn of `c0` in the same auxiliary basis; exact when both span
    the full AO space, since then C = C0 U with U = C0^T S C."""
    rot = c0.T @ ovlp @ mo_coeff
    # B_P,pq = sum_mn U_mp B_P,mn U_nq
    return np.einsum('Pmn,mp,nq->Ppq', coeff0, rot, rot, optimize=True)


def qsgw_eigenvalues(mf, mol=None, screening='updated', mixing='diis',
                     converge_on=None, max_cycle=EVGW_MAX_CYCLE, tol=EVGW_TOL,
                     dm_tol=QSGW_DM_TOL, diis_size=QSGW_DIIS_SIZE,
                     mixing_lambda=QSGW_MIXING_LAMBDA, flow=QSGW_SRG_FLOW,
                     block_elems=QSGW_BLOCK_ELEMS, keep_spectrum=False,
                     verbose=False, df=True,
                     eta=DEFAULT_BROADENING_ETA, tda=False, n_workers=None):
    """(eps, mo_coeff, info): the quasiparticle-self-consistent GW spectrum and
    orbitals, in Hartree and the AO basis.

    screening: 'updated' (qsGW) solves the RPA Casida problem on the rotated
        orbitals every cycle; 'fixed' (qsGW0) keeps the mean field's and
        re-expands its transition density in the rotated orbitals.
    mixing: 'diis', PySCF's CDIIS on the AO Hamiltonian with the commutator
        FDS - SDF as error; 'linear', eq. 21 of Kaplan et al. (J. Chem. Theory
        Comput. 12, 2528 (2016)), H <- lambda H_new + (1 - lambda) H_eps with
        H_eps = S C diag(eps) C^T S, the last iterate's eigenvalues in its own
        orbitals, the mean field's at the first cycle.
    mixing_lambda: that lambda, the new Hamiltonian's weight, 0 < lambda <= 1;
        1 is no mixing.
    converge_on: orbitals whose eigenvalue movement decides convergence, HOMO
        and LUMO by default.
    tol:    max |delta eps| over `converge_on`, in Hartree.
    dm_tol: ||D' - D||_F / nmo, PySCF's criterion. Both must hold.
    flow: the SRG flow parameter s of the static self-energy, in Hartree^-2;
        None gives plain mode A broadened by `eta`
        (SelfEnergySolver.static_self_energy_matrix).
    keep_spectrum: also return the last cycle's Casida solution and transition
        density in `info`, for a fixed-point check; large, off by default.
    df, eta, tda: the Casida route's, as `calc_qp_energy` takes them.
    n_workers: threads for the SRG static self-energy; None takes
        OMP_NUM_THREADS, else the CPUs in the affinity mask, as the Casida
        route's QP scan does.

    `info['df_coeff']`, shape (naux, nmo, nmo), are the DF factors in the
    returned orbitals and `info['w_aux']`, shape (naux, naux), the static RPA
    W in the same auxiliary basis: built from those factors and `eps` for
    qsGW, the mean field's for qsGW0. A BSE on top takes both. Beside them:
    'cycles', 'converged'; 'history' and 'dm_history', the two criteria per
    cycle; 'screening', 'mixing', 'flow', 'converge_on'; 'eps_mean_field' and
    'mo_coeff_mean_field', the start; 'sigma_static', the last cycle's Sigma~
    in the MO basis that cycle started from, not the returned one; 'spectrum'
    and 'rho', None unless `keep_spectrum`.

    Restricted spin only; an attached environment (solvent) is refused, since
    its reaction field enters the mean field's eigenvalues and not this
    Hamiltonian.
    """
    mol = mf.mol if mol is None else mol
    if screening not in SCREENINGS:
        raise ValueError(f'screening={screening!r}: choose one of {SCREENINGS}')
    if mixing not in MIXINGS:
        raise ValueError(f'mixing={mixing!r}: choose one of {MIXINGS}')
    if not 0.0 < mixing_lambda <= 1.0:
        # at 0 the Hamiltonian never moves, and the change the criterion reads,
        # lambda times the fixed-point residual, vanishes at the start
        raise ValueError(f'mixing_lambda={mixing_lambda!r}: linear mixing takes '
                         f'that fraction of the new Hamiltonian and needs '
                         f'0 < mixing_lambda <= 1')
    if flow is not None and not (np.isfinite(flow) and flow >= 0):
        raise ValueError(f'flow={flow!r}: the SRG flow parameter is a finite s >= 0')
    if int(max_cycle) < 1:
        raise ValueError(f'max_cycle={max_cycle!r}: the loop needs one cycle or more')
    if mixing == 'diis' and int(diis_size) < 1:
        raise ValueError(f"diis_size={diis_size!r}: mixing='diis' needs a subspace "
                         f"of one Hamiltonian or more; mixing='linear' needs none")
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError('qsgw_eigenvalues is restricted-spin only')
    if getattr(environment_of(mf), 'screens', True):
        raise NotImplementedError('qsgw_eigenvalues runs in the gas phase only')
    if not df:
        raise NotImplementedError('qsgw_eigenvalues builds Sigma~ from DF factors')
    label = 'qsGW0' if screening == 'fixed' else 'qsGW'
    nocc = mol.nelectron // 2
    eps0 = np.asarray(mf.mo_energy, float)
    c0 = np.asarray(mf.mo_coeff, float)
    mo_occ = np.asarray(mf.mo_occ)
    nmo = len(eps0)
    # an ROHF/ROKS object is not a UHF one, so the spin test above lets it by;
    # eigh sorts the orbitals, so the occupied ones must be the lowest nocc
    off = int(np.count_nonzero(mo_occ != np.where(np.arange(nmo) < nocc, 2.0, 0.0)))
    if mol.spin != 0 or off:
        raise NotImplementedError(
            f'qsgw_eigenvalues needs a closed-shell aufbau reference, the lowest '
            f'{nocc} orbitals doubly occupied and the rest empty; this one has '
            f'spin {mol.spin} and {off} occupations off that pattern')
    # the loop diagonalizes h + J + K + Sigma~ in the full AO space
    if c0.shape[1] != c0.shape[0]:
        raise NotImplementedError(
            f'qsgw_eigenvalues diagonalizes in the full AO space; this mean field '
            f'keeps {c0.shape[1]} of {c0.shape[0]} orbitals (linear dependencies '
            f'removed), which the loop would not reproduce')
    if converge_on is None:
        converge_on = [nocc - 1, nocc]
    tested = np.unique(np.atleast_1d(converge_on).astype(int))
    if tested.size == 0 or tested[0] < 0 or tested[-1] >= nmo:
        raise ValueError(f'converge_on={tested.tolist()}: name one orbital or more, '
                         f'each an index 0 to {nmo - 1} of the spectrum')

    workers = qp_energy._resolve_workers(n_workers, nocc * (nmo - nocc))
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    mf_hf = scf.RHF(mol)
    coeff0 = get_density_fitting_coefficients(mol, mf, representation='spatial')
    if screening == 'fixed':
        # the mean field's Casida problem, solved once; its transition density
        # lives in coeff0's auxiliary basis, which the rotated factors keep
        spectrum, lr0 = _rpa_spectrum(eps0, coeff0, nocc, eta, tda)
        omega, X, Y = spectrum['singlet']
        rho = SelfEnergySolver(eps0, df_coeff=coeff0, spin_mode='restricted',
                               eta=eta)._rho_a_df(nocc, X, Y)
        w_aux = lr0.static_screening_aux(nocc)
        # the loop needs omega and rho only; X and Y are nexciton x nocc nvirt
        del lr0, X, Y
        spectrum = spectrum if keep_spectrum else None

    eps, mo_coeff = eps0.copy(), c0.copy()
    dm = mf_hf.make_rdm1(mo_coeff, mo_occ)
    accel = scf_diis.CDIIS() if mixing == 'diis' else None
    if accel is not None:
        accel.space = int(diis_size)
    history, dm_history = [], []
    converged = False
    sigma = None
    for cycle in range(int(max_cycle)):
        coeff = _rotated_factors(coeff0, c0, ovlp, mo_coeff)
        se = SelfEnergySolver(eps, df_coeff=coeff, spin_mode='restricted', eta=eta)
        if screening == 'updated':
            # X and Y of the last cycle go before this solve allocates its own;
            # the loop needs omega and rho only
            spectrum = None
            spectrum, _ = _rpa_spectrum(eps, coeff, nocc, eta, tda)
            omega, X, Y = spectrum['singlet']
            rho = se._rho_a_df(nocc, X, Y)
            del X, Y
            spectrum = spectrum if keep_spectrum else None
        # Sigma~ in the current MO basis, then to the AO basis: C^-1 = C^T S
        sigma = se.static_self_energy_matrix(nocc, omega, rho, eigenvalues=eps,
                                             flow=flow, block_elems=block_elems,
                                             n_workers=workers)
        cs = ovlp @ mo_coeff
        ham = hcore + mf_hf.get_veff(mol, dm) + cs @ sigma @ cs.T
        if accel is not None:
            ham = accel.update(ovlp, dm, ham)
        else:
            # H_eps,mn = sum_p (SC)_mp eps_p (SC)_np, the iterate (eps, C) as an
            # operator: the last mixed Hamiltonian, and the mean field's Fock
            # matrix at the first cycle
            ham = mixing_lambda * ham + (1.0 - mixing_lambda) * (cs * eps) @ cs.T
        eps_new, mo_coeff = scipy.linalg.eigh(ham, ovlp)
        dm_new = mf_hf.make_rdm1(mo_coeff, mo_occ)
        delta = float(np.abs((eps_new - eps)[tested]).max())
        d_dm = float(np.linalg.norm(dm_new - dm) / nmo)
        history.append(delta)
        dm_history.append(d_dm)
        eps, dm = eps_new, dm_new
        if verbose:
            gap = (eps[nocc] - eps[nocc - 1]) * HARTREE_TO_EV
            print(f'  {label} cycle {cycle + 1:2d}  max|d eps| '
                  f'{delta * HARTREE_TO_EV:9.6f} eV   |dD| {d_dm:8.2e}'
                  f'   gap {gap:8.4f} eV')
        if delta < tol and d_dm < dm_tol:
            converged = True
            break

    if not converged:
        warnings.warn(
            f'{label} did not converge in {max_cycle} cycles: max |delta eps| is '
            f'{history[-1] * HARTREE_TO_EV:.4f} eV against {tol * HARTREE_TO_EV:.4f} '
            f'eV, |dD| {dm_history[-1]:.1e} against {dm_tol:.1e}. The spectrum '
            f'returned is the last iterate, not a fixed point; raise max_cycle or '
            f'switch the mixing.', RuntimeWarning, stacklevel=2)

    # the factors of the returned orbitals, and for qsGW the W they screen with
    coeff = _rotated_factors(coeff0, c0, ovlp, mo_coeff)
    if screening == 'updated':
        w_aux = LinearResponseSolver(eps, coeff_df=coeff, spin_mode='restricted',
                                     eta=eta).static_screening_aux(nocc)
    info = {'cycles': len(history), 'converged': converged, 'history': history,
            'dm_history': dm_history, 'screening': screening, 'mixing': mixing,
            'flow': flow, 'converge_on': tested, 'eps_mean_field': eps0,
            'mo_coeff_mean_field': c0, 'df_coeff': coeff, 'w_aux': w_aux,
            'sigma_static': sigma, 'spectrum': spectrum,
            'rho': rho if keep_spectrum else None}
    return eps, mo_coeff, info
