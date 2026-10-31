"""The density response of a mean field equilibrated in its own reaction
field, contracted with a fixed operator: (natm, 3) of

    Tr[h dD_0/dR]

with D_0 the SCF density of the ground state's self-consistent loop
(`src.environments.solvation_loop.equilibrate` of a mean-field state model
in a `ReactionField`) and h an AO one-electron operator held fixed. At the
loop's convergence that state is the SCF of the continuum at eps_static
(its charges are Q_s V(D_0)), so its density moves with the atoms through a
coupled-perturbed problem whose kernel holds the continuum's response at
eps_static, and through the derivative of the continuum's own operator at
fixed density. Both are pyscf's for a PCM-wrapped mean field
(`pyscf_interface.response_kernel` sets its equilibrium response,
`mean_field_response.fock_partial_skeleton` adds the reaction field's
skeleton), so the loop's mean field is rewrapped as one: the same orbitals,
the vacuum core Hamiltonian, and the medium's continuum at eps_static
(`ContinuumField.scf_continuum`).

With C the orbitals, U_ai = -(dF_ai - eps_i dS_ai) / (eps_a - eps_i) and
U_ij = -dS_ij / 2 (dF, dS in the fixed MO basis, dF with the density's
response),

    Tr[h dD_0/dR] = 4 sum_ai h_ai U_ai - 2 sum_ij h_ij dS_ij,

which `orbital_coupling.z_vector_contraction` evaluates with ONE Z-vector.
A non-equilibrium surface's force (`solvation_loop.force_neq`) reads it: the
slow charges of S0's equilibrium follow the ground state's density.
Closed-shell references only.
"""
import numpy as np

from src.gradients.orbital_coupling import z_vector_contraction


def scf_in_continuum(mf, h0, continuum):
    """A copy of the converged mean field `mf` (its orbitals, energies and
    occupations; its core Hamiltonian replaced by the vacuum one h0) wrapped
    in pyscf's continuum `continuum`: the SCF of the loop's equilibrium as
    pyscf's PCM mean field, without a new SCF (module docstring); None: the
    gas phase's."""
    base = mf.copy()
    base.get_hcore = lambda *a, **k: h0
    if continuum is None:
        return base
    wrapped = base.PCM()
    wrapped.with_solvent = continuum
    continuum._intermediates.pop('dm', None)
    return wrapped


def loop_density_response(mf, h0, continuum, h_ao, numerics=None):
    """(natm, 3) of Tr[h_ao dD_0/dR] for the ground state's loop (module
    docstring): mf its converged mean field in the loop's charges, h0 its
    vacuum core Hamiltonian, continuum the medium's pyscf continuum at
    eps_static; numerics the run's `Numerics` (the Z-vector's solve)."""
    if getattr(mf, 'mo_occ', None) is None or np.asarray(mf.mo_occ).ndim != 1:
        raise ValueError('the loop density response is built for a closed-shell '
                         'reference')
    m = scf_in_continuum(mf, h0, continuum)
    C = np.asarray(m.mo_coeff, float)
    eps = np.asarray(m.mo_energy, float)
    nocc = int(np.count_nonzero(np.asarray(m.mo_occ) > 0))
    norb = len(eps)
    h = C.T @ np.asarray(h_ao, float) @ C
    h = 0.5 * (h + h.T)
    gap = eps[nocc:, None] - eps[None, :nocc]
    w_f = np.zeros((norb, norb))
    w_s = np.zeros((norb, norb))
    # a sum over the symmetric derivatives' two triangles: half on each
    w_f[nocc:, :nocc] = -2.0 * h[nocc:, :nocc] / gap
    w_s[nocc:, :nocc] = 2.0 * h[nocc:, :nocc] * eps[None, :nocc] / gap
    w_f = w_f + w_f.T
    w_s = w_s + w_s.T
    w_s[:nocc, :nocc] = -2.0 * h[:nocc, :nocc]
    return z_vector_contraction(m.mol, m, nocc, w_f, w_s, numerics)
