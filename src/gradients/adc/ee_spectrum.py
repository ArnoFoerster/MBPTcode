"""The ADC(2) or SCS-ADC(2) spectrum of one geometry, for the property layer:
the ground state's energy, the lowest singlets and Ms = 0 triplets of the
fold, and the ISR-ADC(2) one-particle transition densities between them
(SingleReference.ADC.eeADC.ee_ri_transition).

    spec = adc2_spectrum(mp2, n_states)        # gas phase
    charge, spin = spec.transition_density(('ground', 0), ('singlet', 0))
    out = spec.transition_densities([(('singlet', 0), ('triplet', 1)),
                                     (('triplet', 0), ('triplet', 1))])

E_0 is the MP2 state model's energy in the field (h, e) (E_HF + E_MP2, the
ground state under every ADC(2) excitation, as `ADC2State`'s; SCS-MP2 and
SCS-ADC(2) when the model carries a spin-component scaling), and each root
is solved as `ADC2State` solves it (`ee_gradient.solve_root`: the fold's
block solver to the model's Numerics), with the singlets' LR term when a
transition environment is given. A transition density is the (nmo, nmo)
matrix gamma_pq = <bra|a^dag_p a_q|ket> in the reference's molecular
orbitals, split into its charge (alpha + beta) and spin (alpha - beta, the Ms
= 0 triplet's) parts, zero on the frozen core; the bra is the ground state,
a singlet, or a triplet with a triplet ket. The doubles are rebuilt one
occupied tile at a time and never held whole; the fold and the densities run
over the ranks of the current communicator (every rank calls with the same
arguments and gets the same bits), the memory per rank a few tiles.

An unconverged root is warned about and flagged (`flags`, the fold's, the
reference's and the roots' own: the policy refuse inputs, warn numerics); a
root the solve did not return is flagged 'adc_root_missing' and asking for
its density is refused.
"""
import numpy as np

from src.Base.numerics.constants import EE_RI_TILE_GB
from src.Base.utils.parallel.mpi_grid import current_comm, lockstep
from src.SingleReference.ADC.eeADC import ee_ri_transition as ert
from src.SingleReference.ADC.eeADC.ee_follow import flag_unconverged, root_missing
from src.SingleReference.ADC.eeADC.ee_ri_fold import SPINS, bare_factors
from src.gradients.adc.ee_gradient import solve_root


class ADC2Spectrum:
    """The roots of both spins at one geometry and their transition densities
    (module docstring). e0 (Hartree), omega {spin: (n,) Hartree, lowest
    first}, t1 {spin: (n,) singles weights}, flags, mol, mf, frozen, scaling
    ((c_os, c_ss) or None), method ('adc2' or 'scs-adc2') and numerics (the
    MP2 model's one Numerics, whose layout.tile_gb, None: EE_RI_TILE_GB, sizes
    the density pass's RI tiles)."""

    def __init__(self, mol, mf, frozen, e0, roots, flags, numerics, scaling=None):
        self.mol, self.mf, self.frozen, self.e0 = mol, mf, int(frozen or 0), float(e0)
        self._roots = roots
        self.omega = {s: np.asarray(roots[s][1]['omega'], float) for s in SPINS}
        self.t1 = {s: np.asarray(roots[s][1]['t1'], float) for s in SPINS}
        self.flags = flags
        self.scaling = scaling
        self.method = 'adc2' if scaling is None else 'scs-adc2'
        self.numerics = numerics
        tile = numerics.layout.tile_gb
        self.tile_gb = EE_RI_TILE_GB if tile is None else tile
        self._done = {}

    def energy(self, spin, k):
        """E_0 + omega of root k of `spin` (Hartree)."""
        return self.e0 + float(self.omega[spin][k])

    def _root(self, key):
        spin, k = key
        if spin not in SPINS:
            raise ValueError(f'an excited state is (spin, k), not {key!r}')
        if not 0 <= k < self.omega[spin].size:
            raise ValueError(f'ADC(2) {spin} root {k} was not returned by the solve '
                             f"(flagged 'adc_root_missing')")
        op, out = self._roots[spin]
        return (op, np.asarray(out['y'])[:, k], float(out['omega'][k]),
                float(out['t1'][k]))

    def _full(self, gamma):
        """An active-space (no + nv) square matrix in the full MO basis, zero
        on the frozen core."""
        nmo = np.asarray(self.mf.mo_coeff).shape[1]
        out = np.zeros((nmo, nmo))
        out[self.frozen:, self.frozen:] = gamma
        return out

    def transition_densities(self, pairs):
        """{(bra, ket): (charge, spin)} (nmo, nmo) of <bra|a^dag_p a_q|ket>
        for every pair: bra ('ground', 0), ('singlet', k), or ('triplet', k)
        with a triplet ket; ket (spin, k) of an excited state. One pass over
        the tiles serves every pair not computed before; a collective."""
        excited, grounds, keys = [], [], set()
        for bra, ket in pairs:
            if ket[0] not in SPINS:
                raise ValueError(f'the ket is an excited state (spin, k), not {ket!r}')
            if bra[0] == 'triplet' and ket[0] != 'triplet':
                raise ValueError(f'a triplet bra takes a triplet ket, not {ket!r}')
            if bra[0] not in ('ground', 'singlet', 'triplet'):
                raise ValueError(f"the bra is ('ground', 0), ('singlet', k) or "
                                 f"('triplet', k), not {bra!r}")
            self._root(ket)
            if bra[0] != 'ground':
                self._root(bra)
            if (bra, ket) in self._done:
                continue
            if bra[0] == 'ground':
                grounds.append(ket)
                keys.add(ket)
            else:
                excited.append((bra, ket))
                keys.update((bra, ket))
        if excited or grounds:
            got = ert.transition_densities({k: self._root(k) for k in sorted(keys)},
                                           excited, grounds, tile_gb=self.tile_gb)
            for key, (charge, spin) in got.items():
                self._done[key] = (self._full(charge), self._full(spin))
        return {(bra, ket): self._done[bra, ket] for bra, ket in pairs}

    def transition_density(self, bra, ket):
        """(charge, spin) (nmo, nmo) of <bra|a^dag_p a_q|ket>
        (`transition_densities` of the one pair)."""
        return self.transition_densities([(bra, ket)])[bra, ket]


def adc2_spectrum(mp2, n_states, h=None, e=0.0, lr_environment=None):
    """`ADC2Spectrum` of the lowest n_states singlets and triplets on the MP2
    state model `mp2` (`gradients.state_models.MP2State`, its spin_scaling
    giving SCS-ADC(2)) in the field (h, e) (h None: the gas phase), the
    singlets with the LR term of `lr_environment` (a transition environment,
    or None). The roots are solved as `ADC2State` solves them: the block
    solver to the model's Numerics (adc.conv_tol, adc.residual_tol,
    adc.dense_limit); the fold and the transition densities run over the
    current communicator's ranks on the model's RI tiles (`MP2State.tile_gb`),
    both spins on one set of factors."""
    mf0 = mp2.mf
    if h is None:
        h = np.zeros_like(np.asarray(mf0.get_hcore(), float))
    r = mp2(h, e)
    mf, mol = mp2.mf, mp2.mol
    flags = list(r.info.get('flags', []))
    roots = {}
    comm = current_comm()
    # one set of bare factors for both spins
    factors = bare_factors(mol, mf, frozen=mp2.frozen, auxbasis=mp2.auxbasis, comm=comm)
    adc = mp2.numerics.adc
    for spin in SPINS:
        op, _, out = solve_root(mol, mf, spin, n_states, mp2.frozen, mp2.auxbasis,
                                env=lr_environment, conv_tol=adc.conv_tol,
                                tile_gb=mp2.tile_gb, comm=comm, factors=factors,
                                scaling=mp2.scaling, solver='block',
                                residual_tol=adc.residual_tol,
                                dense_limit=adc.dense_limit, numerics=mp2.numerics)
        omega, y, t1, conv = lockstep((np.asarray(out['omega'], float),
                                       np.asarray(out['y'], float),
                                       np.asarray(out['t1'], float),
                                       np.atleast_1d(np.asarray(out['converged'], bool))))
        flags += list(out.get('flags', []))
        flag_unconverged(conv, f'ADC(2) {spin} spectrum', flags)
        root_missing(n_states - 1, omega.size, f'ADC(2) {spin} spectrum', flags)
        roots[spin] = (op, dict(omega=omega, y=y, t1=t1, converged=conv))
    return ADC2Spectrum(mol, mf, mp2.frozen, r.energy, roots, flags, mp2.numerics, mp2.scaling)
