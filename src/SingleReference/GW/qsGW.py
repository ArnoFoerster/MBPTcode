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
doi 10.1021/acs.jctc.3c00281): Kotani, van Schilfgaarde and Faleev's mode A,
1/2 [Sigma_pq(eps_p) + Sigma_pq(eps_q)] (Phys. Rev. B 76, 165106, 2007), with
every term whose energy denominator lies within about 1/sqrt(2 s) of zero
damped. Plain mode A on the pole sum puts high virtuals on poles of Sigma, and
its loop does not settle; `flow=None` still selects it, with eta as the
broadening. qsGW0 keeps the mean field's RPA Casida solution,
`screening='fixed'`: W stays, the amplitudes and the poles follow the rotated
orbitals. Restricted spin, Casida route only.
"""
import warnings

import numpy as np
import scipy.linalg
from pyscf import scf
from pyscf.scf import diis as scf_diis

from src.Base.constants import (DEFAULT_BROADENING_ETA, EVGW_MAX_CYCLE, EVGW_TOL,
                                HARTREE_TO_EV, QSGW_BLOCK_ELEMS, QSGW_DAMPING,
                                QSGW_DIIS_SIZE, QSGW_DM_TOL, QSGW_SRG_FLOW,
                                get_method_info)
from src.Base.environment import environment_of
from src.Base.pyscf_interface import get_density_fitting_coefficients
# module imports: both files import this one through qp_energy
from src.SingleReference.GW import evGW, qp_energy
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

#: What the iterate screens: the Casida problem rebuilt from it (qsGW), or the
#: mean field's kept and only the amplitudes and poles moved (qsGW0).
SCREENINGS = ('updated', 'fixed')
#: How the AO Hamiltonian is mixed between cycles: PySCF's CDIIS, or linear
#: mixing H <- (1 - d) H_new + d H_old with d = damping.
MIXINGS = ('diis', 'linear')


def _rpa_spectrum(eps, coeff, nocc, eta, tda):
    """(omega, X, Y) of the RPA Casida problem on `eps` with the DF factors
    `coeff`, and the solver it was built with."""
    lr = LinearResponseSolver(eps, coeff_df=coeff, spin_mode='restricted', eta=eta)
    spectrum = qp_energy._casida_spectrum(lr, nocc, 'RPA', None, tda,
                                          {'GW': get_method_info('GW')}, ['GW'],
                                          False, True)
    return spectrum, lr


def qsgw_eigenvalues(mf, mol=None, screening='updated', mixing='diis',
                     converge_on=None, max_cycle=EVGW_MAX_CYCLE, tol=EVGW_TOL,
                     dm_tol=QSGW_DM_TOL, diis_size=QSGW_DIIS_SIZE,
                     damping=QSGW_DAMPING, flow=QSGW_SRG_FLOW,
                     block_elems=QSGW_BLOCK_ELEMS, keep_spectrum=False,
                     verbose=False, df=True,
                     eta=DEFAULT_BROADENING_ETA, tda=False):
    """(eps, mo_coeff, info): the quasiparticle-self-consistent GW spectrum and
    orbitals, in Hartree and the AO basis.

    screening: 'updated' (qsGW) solves the RPA Casida problem on the rotated
        orbitals every cycle; 'fixed' (qsGW0) keeps the mean field's and
        re-expands its transition density in the rotated orbitals.
    mixing: 'diis', PySCF's CDIIS on the AO Hamiltonian with the commutator
        FDS - SDF as error; 'linear', H <- (1 - damping) H_new + damping H_old.
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

    `info['df_coeff']` are the DF factors in the converged orbitals and
    `info['w_aux']` the static RPA W: built from those factors and `eps` for
    qsGW, the mean field's for qsGW0. A BSE on top takes both.

    Restricted spin only; an attached environment (solvent) is refused, since
    its reaction field enters the mean field's eigenvalues and not this
    Hamiltonian.
    """
    mol = mf.mol if mol is None else mol
    if screening not in SCREENINGS:
        raise ValueError(f'screening={screening!r}: choose one of {SCREENINGS}')
    if mixing not in MIXINGS:
        raise ValueError(f'mixing={mixing!r}: choose one of {MIXINGS}')
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
    if converge_on is None:
        converge_on = [nocc - 1, nocc]
    tested = np.intersect1d(np.atleast_1d(converge_on).astype(int), np.arange(nmo))
    if tested.size == 0:
        raise ValueError('converge_on and the spectrum do not overlap, so nothing '
                         'would decide convergence')

    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    mf_hf = scf.RHF(mol)
    if screening == 'fixed':
        # the mean field's Casida problem, solved once; its transition density
        # is an auxiliary-space object that every rotated basis can read
        coeff0 = get_density_fitting_coefficients(mol, mf, representation='spatial')
        spectrum, lr0 = _rpa_spectrum(eps0, coeff0, nocc, eta, tda)
        omega, X, Y = spectrum['singlet']
        rho = SelfEnergySolver(eps0, df_coeff=coeff0, spin_mode='restricted',
                               eta=eta)._rho_a_df(nocc, X, Y)
        w_aux = lr0.static_screening_aux(nocc)

    eps, mo_coeff = eps0.copy(), c0.copy()
    dm = mf_hf.make_rdm1(mo_coeff, mo_occ)
    accel = scf_diis.CDIIS() if mixing == 'diis' else None
    if accel is not None:
        accel.space = int(diis_size)
    ham_old = None
    history, dm_history = [], []
    converged = False
    sigma = None
    for cycle in range(int(max_cycle)):
        # the DF factors of the current orbitals; get_density_fitting_coefficients
        # reads the view's mo_coeff
        view = evGW.rotated_mean_field(mf, eps, mo_coeff)
        coeff = get_density_fitting_coefficients(mol, view, representation='spatial')
        se = SelfEnergySolver(eps, df_coeff=coeff, spin_mode='restricted', eta=eta)
        if screening == 'updated':
            spectrum, _ = _rpa_spectrum(eps, coeff, nocc, eta, tda)
            omega, X, Y = spectrum['singlet']
            rho = se._rho_a_df(nocc, X, Y)
        # Sigma~ in the current MO basis, then to the AO basis: C^-1 = C^T S
        sigma = se.static_self_energy_matrix(nocc, omega, rho, eigenvalues=eps,
                                             flow=flow, block_elems=block_elems)
        cs = ovlp @ mo_coeff
        ham = hcore + mf_hf.get_veff(mol, dm) + cs @ sigma @ cs.T
        if accel is not None:
            ham = accel.update(ovlp, dm, ham)
        elif ham_old is not None:
            ham = (1.0 - damping) * ham + damping * ham_old
        ham_old = ham
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
    view = evGW.rotated_mean_field(mf, eps, mo_coeff)
    coeff = get_density_fitting_coefficients(mol, view, representation='spatial')
    if screening == 'updated':
        w_aux = LinearResponseSolver(eps, coeff_df=coeff, spin_mode='restricted',
                                     eta=eta).static_screening_aux(nocc)
    info = {'cycles': len(history), 'converged': converged, 'history': history,
            'dm_history': dm_history, 'screening': screening, 'mixing': mixing,
            'flow': flow, 'converge_on': tested, 'eps_mean_field': eps0,
            'mo_coeff_mean_field': c0, 'df_coeff': coeff, 'w_aux': w_aux,
            'sigma_static': sigma,
            'spectrum': spectrum if keep_spectrum else None,
            'rho': rho if keep_spectrum else None}
    return eps, mo_coeff, info
