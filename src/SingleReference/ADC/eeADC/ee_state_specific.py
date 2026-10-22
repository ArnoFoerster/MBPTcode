"""State-specific (SS) continuum solvation of an excited state from its
difference density, the companion of the linear-response (LR) term.

A polarizable continuum reaches a solvated excitation in three places:

    ground state  the SCF relaxed in PCM(eps_static): the ADC operator is that
                  of H_vac + V.q0 with the ground-state charges q0 frozen
                  (PTE; Lunkenheimer and Koehn, J. Chem. Theory Comput. 9,
                  977 (2013))
    LR            the reaction field of the transition density, 2 (ia|vt|jb)
                  in the singlet singles block at eps_inf (ee_ri_fold); zero
                  for triplets, whose transition density carries no charge
    SS (here)     the reaction field of the state's density difference
                  Delta rho, added to its eigenvalue

the partition of the solvated BSE (Duchemin, Guido, Jacquemin and Blase,
Chem. Sci. 9, 4430 (2018), Eqs. (18) and (20)) and of corrected LR
(Caricato et al., J. Chem. Phys. 124, 124520 (2006)), so that ADC(2) and
GW-BSE solvent shifts are read the same way.

With Q the surface response (q = Q V, Q negative semi-definite), omega
already holds Delta V . q0; letting the medium answer Delta rho adds
(Cammi and Mennucci, J. Chem. Phys. 110, 9877 (1999); Mewes, You, Wormit,
Kriesche, Herbert and Dreuw, J. Phys. Chem. A 119, 5446 (2015))

    E_SS = (1/2) Delta V^T Q Delta V ,   Delta V = V[Delta rho]  (<= 0)

    vertical     Q = Q(eps_inf): only the solvent electrons follow
    equilibrium  Q = Q(eps_static): the solvent reorients (at the
                 ground-state geometry)

E_SS is quadratic in Delta rho and holds no ground-state field, and the LR
term (at eps_inf in both modes: split equilibrium, Guido et al., J. Chem.
Theory Comput. 17, 5155 (2021)) answers the transition density, so nothing
is counted twice.

Delta rho is either the unrelaxed ISR-ADC(2) difference density (ee_density,
ee_ri_density; the default) or the orbital-relaxed one, d omega / d h, with
the ground state's surface charges frozen in the orbital response
(ee_relaxed_density; ee_driver's ss_density='relaxed'). Relaxation shrinks
the dipole change of a charge-transfer-like state and with it E_SS. Left
out: the MP2 correlation density of the reference
(PTD). Delta V and Q use the LR term's smeared surface points
(SolventScreening.response_matrix).
"""
import numpy as np

MODES = ('off', 'vertical', 'equilibrium')


def check_mode(mode):
    """mode itself, if it is one of MODES."""
    if mode not in MODES:
        raise ValueError(f'state_specific={mode!r}; expected one of {MODES}')
    return mode


def shift_modes(env, mode):
    """The response modes to evaluate for the requested mode, 'vertical' first."""
    check_mode(mode)
    modes = ('vertical',) if getattr(env, 'eps_static', None) is None else \
        ('vertical', 'equilibrium')
    if mode not in modes:
        raise ValueError(f'state_specific={mode!r} needs the continuum\'s '
                         'eps_static')
    return modes


def delta_dipoles(mol, mo_act, C, delta_density, delta_density_relaxed):
    """{'unrelaxed', 'relaxed'}: (nroots, 3) dipole change of every root, from
    the difference densities in the active orbitals mo_act and in all MOs C."""
    r = mol.intor('int1e_r')
    return {
        'unrelaxed': np.array([-np.einsum('xmn,mn->x', r, mo_act @ D @ mo_act.T)
                               for D in delta_density]),
        'relaxed': np.array([-np.einsum('xmn,mn->x', r, C @ D @ C.T)
                             for D in delta_density_relaxed])}


def surface_potential(env, mol, dm_ao):
    """Delta V_k = sum_mn dm_mn v_k[phi_m phi_n], shape (ngrids,): the
    potential of a (spin-summed) AO density on the cavity points, grid block
    by grid block."""
    dm_ao = np.asarray(dm_ao, float)
    out = np.empty(env.ngrids)
    for k0, k1 in env.grid_blocks(mol.nao, live=1):
        v = env.ao_grid_potential(mol, grids=slice(k0, k1))
        out[k0:k1] = np.einsum('mnk,mn->k', v, dm_ao, optimize=True)
    return out


def responses(env, modes):
    """{mode: Q} of the modes asked for: the optical response for vertical,
    the static partner's for equilibrium."""
    out = {}
    for mode in modes:
        if mode == 'vertical':
            out[mode] = env.response_matrix()
        elif mode == 'equilibrium':
            if getattr(env, 'eps_static', None) is None:
                raise ValueError('the equilibrium state-specific term needs '
                                 "the continuum's eps_static")
            out[mode] = env.static_partner().response_matrix()
        else:
            raise ValueError(f'mode={mode!r}: vertical or equilibrium')
    return out


def state_specific_energies(env, mol, mo_act, densities,
                            modes=('vertical', 'equilibrium')):
    """{mode: (nroots,) Hartree}: (1/2) Delta V^T Q Delta V of every
    difference density, densities a list of (nact, nact) spatial MO arrays of
    the active orbitals mo_act (nao, nact), frozen core dropped (it is
    uncorrelated, so its rows of Delta rho are zero)."""
    mo_act = np.asarray(mo_act, float)
    Q = responses(env, modes)
    V = [surface_potential(env, mol, mo_act @ np.asarray(D, float) @ mo_act.T)
         for D in densities]
    return {m: np.array([0.5 * float(v @ (Q[m] @ v)) for v in V])
            for m in modes}
