"""
evGW: the quasiparticle energies reinjected into G and P0 until convergence.

G0W0 evaluates the self-energy once, on the mean field's own eigenvalues. Here
the whole spectrum is updated, fed back into the Green's function and the
polarizability, and the cycle repeated to a fixed point:

    eps_qp, info = evgw_eigenvalues(mf, mol, mode='space-time')
    info['cycles'], info['converged'], info['residual_untested']

The equation stays ANCHORED on the mean field. w = eps_p^MF + <Sigma_x - v_xc>
+ Re Sigma_c(w) with Sigma_c screened by the current iterate
"""
import warnings

import numpy as np

from src.Base.constants import (EVGW_DAMPING, EVGW_DIIS_SIZE, EVGW_DIIS_START,
                                EVGW_MAX_CYCLE, EVGW_TOL, HARTREE_TO_EV)
from src.Base.pyscf_interface import get_orbital_energies
from src.SingleReference.CC.diis import DIIS
from src.SingleReference.GW.qp_energy import calc_qp_energy

#: The routes the dispatcher knows, in its own naming.
MODES = ('casida', 'imagfrequency', 'space-time')


def shifted_mean_field(mf, eps):
    """
    `mf` carrying `eps` in place of its own eigenvalues.

    A shallow copy through PySCF's own `copy`, so every integral, fit, cavity
    and the direct-SCF optimizer the reference already holds is shared once
    per cycle. `copy.copy` would go through the pickle hooks of the mean
    field, which drop that optimizer, and a J/K build on a molecule too large
    for the in-core branch then fails on the copy.
    """
    out = mf.copy()
    out.mo_energy = np.asarray(eps, float)
    return out


def quasiparticle_spectrum(mf, mol, mode, eps_anchor, spin_channel='alpha',
                           **route_kw):
    """
    Every orbital's G0W0 energy of one spin channel in Hartree, screened
    with the spectrum `mf` carries and anchored on `eps_anchor`.
    """
    n = np.asarray(get_orbital_energies(mf, representation='spatial')).shape[-1]
    out = calc_qp_energy(mf, selfenergy='GW', polarizability='RPA', mode=mode,
                         state=list(range(n)), eps_anchor=eps_anchor,
                         spin_channel=spin_channel, **route_kw)
    if isinstance(out, dict):
        out = [out[p]['GW'] for p in range(n)]
    return np.asarray(out, float) / HARTREE_TO_EV


def evgw_eigenvalues(mf, mol=None, mode='space-time', converge_on=None,
                     max_cycle=EVGW_MAX_CYCLE, tol=EVGW_TOL,
                     diis_size=EVGW_DIIS_SIZE, diis_start=EVGW_DIIS_START,
                     damping=EVGW_DAMPING, verbose=False, **route_kw):
    """(eps_qp, info): the eigenvalue-self-consistent GW spectrum, in Hartree.

    `eps_qp` is the full array -- every orbital is updated and every orbital
    screens.

    mode: 'casida', 'imagfrequency' or 'space-time', and `route_kw` goes to
        that route through `calc_qp_energy`.
    converge_on: orbitals whose movement decides convergence. Default is the
        HOMO and the LUMO.
    tol:    convergence on max |delta eps| over `converge_on`, in Hartree.
    diis_size: DIIS subspace; 0 falls back to linear mixing through `damping`.
    damping: linear mixing, eps <- (1 - d) eps_new + d eps_old, used only when
        DIIS is off.

    An unrestricted reference is driven channel by channel: the spectrum is
    (2, nmo), the anchor and the convergence test are per channel, DIIS runs
    on both at once. The Casida route serves it; the imaginary-axis routes
    refuse an unrestricted reference themselves.
    """
    mol = mf.mol if mol is None else mol
    if mode not in MODES:
        raise ValueError(f'mode={mode!r}: choose one of {MODES}')
    eps0 = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    unrestricted = eps0.ndim == 2
    channels = ('alpha', 'beta') if unrestricted else ('alpha',)
    nmo = eps0.shape[-1]
    noccs = tuple(mf.nelec) if unrestricted else (mol.nelectron // 2,)
    states = np.arange(nmo)
    # The convergence test on the flattened (channel, orbital) vector; for a
    # restricted reference that is the orbital index itself.
    if converge_on is None:
        converge_on = [s * nmo + p for s, n in enumerate(noccs) for p in (n - 1, n)]
    flat = np.arange(len(channels) * nmo)
    tested = np.intersect1d(np.atleast_1d(converge_on).astype(int), flat)
    if tested.size == 0:
        raise ValueError('converge_on and states do not overlap, so nothing '
                         'would decide convergence')
    untested = np.setdiff1d(flat, tested)

    eps = eps0.copy()
    history, untested_history = [], []
    converged = False
    accel = DIIS(int(diis_size), start_iter=int(diis_start)) if diis_size else None
    for cycle in range(int(max_cycle)):
        view = shifted_mean_field(mf, eps)
        eps_new = np.array([quasiparticle_spectrum(view, mol, mode, eps0,
                                                   spin_channel=ch, **route_kw)
                            for ch in channels]).reshape(eps0.shape)
        residual = (eps_new - eps).ravel()
        delta = float(np.abs(residual[tested]).max())
        rest = (float(np.abs(residual[untested]).max())
                if untested.size else 0.0)
        history.append(delta)
        untested_history.append(rest)
        if verbose:
            top = np.atleast_2d(eps_new)[0]
            gap = (top[noccs[0]] - top[noccs[0] - 1]) * HARTREE_TO_EV
            print(f'  evGW cycle {cycle + 1:2d}  max|d eps| '
                  f'{delta * HARTREE_TO_EV:9.6f} eV   gap {gap:8.4f} eV'
                  f'   (untested {rest * HARTREE_TO_EV:8.5f} eV)')
        if delta < tol:
            eps = eps_new
            converged = True
            break
        # DIIS extrapolates the whole spectrum against the whole residual: the
        # map being accelerated is the one on all of it, whatever subset is
        # tested for convergence.
        if accel is not None:
            eps = accel.compute_new_vec(eps_new.ravel(), residual).reshape(eps0.shape)
        else:
            eps = (1.0 - damping) * eps_new + damping * eps

    if not converged:
        warnings.warn(
            f'evGW did not converge in {max_cycle} cycles: max |delta eps| is '
            f'{history[-1] * HARTREE_TO_EV:.4f} eV against a tolerance of '
            f'{tol * HARTREE_TO_EV:.4f} eV. The spectrum returned is the last '
            f'iterate, not a fixed point; raise max_cycle or damping.',
            RuntimeWarning, stacklevel=2)

    return eps, {'cycles': len(history), 'converged': converged,
                 'history': history, 'states': states, 'mode': mode,
                 'converge_on': tested,
                 'residual_untested': untested_history[-1],
                 'eps_mean_field': eps0, 'shift': eps - eps0}
