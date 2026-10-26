"""The ADC(2) spectrum of one geometry, for the property layer: the ground
state's energy, the lowest singlets and Ms = 0 triplets of the fold, and the
ISR-ADC(2) one-particle transition densities between them
(SingleReference.ADC.eeADC.ee_transition_density).

    spec = adc2_spectrum(mp2, n_states)        # gas phase
    charge, spin = spec.transition_density(('ground', 0), ('singlet', 0))
    charge, spin = spec.transition_density(('singlet', 0), ('triplet', 1))

E_0 is the MP2 state model's energy in the field (h, e) (E_HF + E_MP2, the
ground state under every ADC(2) excitation, as `ADC2State`'s), and each root
is solved as the ADC(2) force solves it (`ee_gradient.solve_root`: the fold,
Newton-polished to EE_FORCE_ROOT_TOL), with the singlets' LR term when a
transition environment is given. A transition density is the (nmo, nmo)
matrix gamma_pq = <bra|a^dag_p a_q|ket> in the reference's molecular
orbitals, split into its charge (alpha + beta) and spin (alpha - beta, the Ms
= 0 triplet's) parts, zero on the frozen core; the bra is the ground state or
a singlet, the ket an excited state. The rebuilt doubles are held whole
(`ee_transition_density.fold_state`), so this serves molecules whose
o^2 v^2 fits in memory; the factors are the fold's own, read whole on one
rank, and the evaluation is serial: under ranks every rank computes it and
keeps rank 0's roots (lockstep).

An unconverged root is warned about and flagged (`flags`, the fold's, the
reference's and the roots' own: the policy refuse inputs, warn numerics); a
root the solve did not return is flagged 'adc_root_missing' and asking for
its density is refused.
"""
import numpy as np

from src.Base.numerics.constants import EE_FORCE_ROOT_TOL
from src.Base.utils.parallel.mpi_grid import lockstep
from src.SingleReference.ADC.eeADC import ee_transition_density as etd
from src.SingleReference.ADC.eeADC.ee_follow import flag_unconverged, root_missing
from src.SingleReference.ADC.eeADC.ee_ri_fold import bare_factors
from src.gradients.adc.ee_gradient import solve_root

SPINS = ('singlet', 'triplet')


class ADC2Spectrum:
    """The roots of both spins at one geometry and their transition densities
    (module docstring). e0 (Hartree), omega {spin: (n,) Hartree, lowest
    first}, t1 {spin: (n,) singles weights}, flags, mol, mf, frozen."""

    def __init__(self, mol, mf, frozen, e0, roots, flags):
        self.mol, self.mf, self.frozen, self.e0 = mol, mf, int(frozen or 0), float(e0)
        self._roots = roots
        self.omega = {s: np.asarray(roots[s][1]['omega'], float) for s in SPINS}
        self.t1 = {s: np.asarray(roots[s][1]['t1'], float) for s in SPINS}
        self.flags = flags
        self._states, self._ref = {}, None

    def energy(self, spin, k):
        """E_0 + omega of root k of `spin` (Hartree)."""
        return self.e0 + float(self.omega[spin][k])

    def _state(self, spin, k):
        if not 0 <= k < self.omega[spin].size:
            raise ValueError(f'ADC(2) {spin} root {k} was not returned by the solve '
                             f"(flagged 'adc_root_missing')")
        if (spin, k) not in self._states:
            op, out = self._roots[spin]
            self._states[spin, k] = etd.fold_state(op, np.asarray(out['y'])[:, k],
                                                   float(out['omega'][k]),
                                                   float(out['t1'][k]))
        return self._states[spin, k]

    def _reference(self):
        """The ground state's amplitudes on the fold's factors, with the
        second-order doubles of the ground row (serving the state-to-state
        densities too), built once."""
        if self._ref is None:
            op = self._roots['singlet'][0]
            f = op.f
            self._ref = etd.reference_from_factors(
                np.concatenate([f.eo, f.ev]), etd.factors_of(op), f.no, ground=True)
        return self._ref

    def _full(self, gamma):
        """An active-space (no + nv) square matrix in the full MO basis, zero
        on the frozen core."""
        nmo = np.asarray(self.mf.mo_coeff).shape[1]
        out = np.zeros((nmo, nmo))
        out[self.frozen:, self.frozen:] = gamma
        return out

    def transition_density(self, bra, ket):
        """(charge, spin) (nmo, nmo) of <bra|a^dag_p a_q|ket>: bra ('ground',
        0) or ('singlet', k), ket (spin, k) of an excited state."""
        if ket[0] not in SPINS:
            raise ValueError(f'the ket is an excited state (spin, k), not {ket!r}')
        right = self._state(*ket)
        if bra[0] == 'ground':
            charge, spin = etd.ground_transition_density(self._reference(), right)
        elif bra[0] == 'singlet':
            charge, spin = etd.transition_density(self._reference(),
                                                  self._state(*bra), right)
        else:
            raise ValueError(f"the bra is ('ground', 0) or ('singlet', k), not {bra!r}")
        return self._full(charge), self._full(spin)


def adc2_spectrum(mp2, n_states, h=None, e=0.0, lr_environment=None):
    """`ADC2Spectrum` of the lowest n_states singlets and triplets on the MP2
    state model `mp2` (`gradients.state_models.MP2State`) in the field (h,
    e) (h None: the gas phase), the singlets with the LR term of
    `lr_environment` (a transition environment, or None). The roots are
    solved to EE_FORCE_ROOT_TOL, the tolerance the state model's force
    solves them to, so the spectrum and the force see the same roots."""
    mf0 = mp2.mf
    if h is None:
        h = np.zeros_like(np.asarray(mf0.get_hcore(), float))
    r = mp2(h, e)
    mf, mol = mp2.mf, mp2.mol
    flags = list(r.info.get('flags', []))
    roots = {}
    # one set of bare factors for both spins
    factors = bare_factors(mol, mf, frozen=mp2.frozen, auxbasis=mp2.auxbasis)
    for spin in SPINS:
        # serial on every rank (no communicator), rank 0's roots kept
        op, _, out = solve_root(mol, mf, spin, n_states, mp2.frozen, mp2.auxbasis,
                                env=lr_environment, conv_tol=EE_FORCE_ROOT_TOL,
                                factors=factors)
        omega, y, t1, conv = lockstep((np.asarray(out['omega'], float),
                                       np.asarray(out['y'], float),
                                       np.asarray(out['t1'], float),
                                       np.atleast_1d(np.asarray(out['converged'], bool))))
        flags += list(out.get('flags', []))
        flag_unconverged(conv, f'ADC(2) {spin} spectrum', flags)
        root_missing(n_states - 1, omega.size, f'ADC(2) {spin} spectrum', flags)
        roots[spin] = (op, dict(omega=omega, y=y, t1=t1, converged=conv))
    return ADC2Spectrum(mol, mf, mp2.frozen, r.energy, roots, flags)
