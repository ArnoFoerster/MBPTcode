"""The two-step downfolding driver: run a low-level method on the mean field
once, then build the effective Hamiltonian of any number of active spaces
from it.

    low = run_low_level(mf, method='hf')
    g_eff, space, energies = build_effective_hamiltonian(low, active_space)
    e = solve('pyscf_fci', g_eff, space['n_orbs'], space['n_elec_active'],
              spin_free=space['spin_free'])

The driver knows no method: it reaches them through `LOW_LEVEL` and reads
only the `DownfoldingTerms` a method hands back. The reference is restricted,
so the general layer gets ONE spin channel (the orbitals, the `Partition`,
one vtilde) and computes everything once: one exact J/K build of the inactive
core, shared by the core Fock operator and the environment energy, and one
integral transformation of the active block.

The three parts of H_eff come from the general layer's three files, E_0 from
energy.py, h_eff from one_body.py and v_eff from two_body.py; `pack` here
writes them into the g_eff dictionary the active-space solvers read
(src/Solvers/active_space.py).
"""
import numpy as np

from src.Base.pyscf.pyscf_interface import (get_orbital_energies,
                                            require_gas_phase_ground_state)
from src.Base.utils.parallel.mpi_grid import grid_comm
from src.SingleReference.common.active_space import closed_shell_nocc
from src.Solvers.exact_diagonalization import spatial_to_spin_orbitals

from .energy import assemble_energies, environment_hf_energy
from .methods import LOW_LEVEL
from .one_body import (DC_MODES, core_fock, double_counting,
                       effective_one_body, inactive_jk)
from .partition import partition
from .terms import choice
from .two_body import effective_interaction

WHO = 'Downfolding'
RESTRICTED_TOL = 1e-12


def run_low_level(mf, method='rpa', **method_opts):
    """The low-level result of `method` on the mean field `mf` (attributes
    `method` and `mf`, plus whatever the method keeps for its `terms`).

    The result does not depend on an active space, so one run serves every
    active space built from it. Only a restricted closed-shell, gas-phase,
    T = 0 mean field is taken, and only serially: on more than one MPI rank
    every rank would repeat the whole calculation. method_opts go to the
    method, which refuses the ones it does not know.
    """
    _, _, size = grid_comm()
    if size > 1:
        raise NotImplementedError(
            f'{WHO} runs serially only; this process is one of {size} MPI '
            f'ranks. Run it on a single rank.')
    require_gas_phase_ground_state(mf, WHO)
    if getattr(mf, 'with_solvent', None) is not None:
        # a pyscf solvent model is not an attached environment, which is all
        # require_gas_phase looks for
        raise NotImplementedError(
            f'{WHO} is gas phase only; mf carries a solvent model '
            f'({type(mf.with_solvent).__name__}).')
    method = choice(method, tuple(LOW_LEVEL), 'method')
    return LOW_LEVEL[method].run(mf, **method_opts)


def build_effective_hamiltonian(low, active_space, spin_free=True,
                                **method_opts):
    """(g_eff, space, energies): the effective Hamiltonian of `active_space`
    downfolded from the low-level result `low`.

    g_eff and space follow QDPT's build_effective_hamiltonian, so the result
    goes straight to src.Solvers.active_space.solve: g_eff = {0: E_0, 1:
    h_eff, 2: physicist v_eff}, spatial when spin_free, else over the 2 n_act
    interleaved spin orbitals; space holds n_orbs, n_elec_active, act_idx,
    act_occ_idx, act_vir_idx (canonical spatial indices), spin_free and
    active_space. energies is the `Energies` record of every E_0 term.

    h_eff = f - t^DC (+ Sigma_bar) with f the Fock operator of the inactive
    core, v_eff = v + vtilde with v the exact active integrals; the method
    supplies vtilde, Sigma_bar, the correlation-energy pair and the
    double-counting mode. method_opts go to the method's `terms`.
    """
    mf = low.mf
    mol = mf.mol
    eps = get_orbital_energies(mf)
    nocc = closed_shell_nocc(mf.mo_occ)
    if (active_space.norb, active_space.nocc) != (len(eps), nocc):
        raise ValueError(
            f'the active space is set up for {active_space.nocc} occupied of '
            f'{active_space.norb} orbitals; the mean field has {nocc} '
            f'occupied of {len(eps)}.')
    part = partition(active_space, eps)
    method = choice(low.method, tuple(LOW_LEVEL), 'method')
    terms = LOW_LEVEL[method].terms(low, part, **method_opts)
    dc = choice(terms.dc, DC_MODES, 'dc')
    if terms.sigma_bar is not None and dc == 'full':
        raise ValueError(
            f"method={method!r} returned a static self-energy with dc='full'; "
            f"Sigma_bar carries the exchange, so it comes with dc='hartree', "
            f"which also sets the double counting and E_0 to match.")
    channels = [len(terms.vtilde)]
    if terms.sigma_bar is not None:
        channels.append(len(terms.sigma_bar))
    if channels != [1] * len(channels):
        raise NotImplementedError(
            f"method={method!r} returned {len(terms.vtilde)} vtilde blocks"
            + (f" and {len(terms.sigma_bar)} Sigma_bar blocks"
               if terms.sigma_bar is not None else "")
            + "; this driver runs a restricted reference, which is one spin "
              "channel: vtilde = (v,), sigma_bar = (s,). The unrestricted "
              "form, (aa, ab, bb) and (a, b), is deferred.")

    c = mf.mo_coeff
    mo, parts = (c,), (part,)
    n_act_occ = (part.n_act_occ,)
    hcore = mf.get_hcore()
    jk = inactive_jk(mol, mo, parts)
    f = core_fock(mol, hcore, mo, parts, jk=jk)
    hartree, exchange = double_counting(terms.vtilde, n_act_occ)
    h_eff = effective_one_body(f, hartree, exchange, dc, terms.sigma_bar)
    v_eff = effective_interaction(mol, (c[:, part.act],), terms.vtilde)
    environment = environment_hf_energy(mol, hcore, mo, parts, jk=jk)
    energies = assemble_energies(mf.energy_nuc(), *environment, terms,
                                 n_act_occ)
    g_eff = pack(h_eff, v_eff, energies.e0, spin_free)

    space = {
        'n_orbs': part.n_act if spin_free else 2 * part.n_act,
        'n_elec_active': active_space.nelec_active,
        'act_idx': active_space.act,
        'act_occ_idx': active_space.occ_active,
        'act_vir_idx': active_space.virt_active,
        'spin_free': spin_free,
        'active_space': active_space,
    }
    return g_eff, space, energies


def pack(h_eff, v_eff, e0, spin_free):
    """The g_eff dictionary {0: e0, 1: one-body, 2: two-body} of the solvers.

    h_eff and v_eff hold the one-body term and the chemist active interaction
    (bare plus vtilde) per spin channel: (h,) and (v,) for the one channel of
    a restricted reference, or (h_a, h_b) and (aa, ab, bb). Spin-free,
    g_eff[1] is the spatial one-body and g_eff[2] the spatial physicist
    <pq|rs> = (pr|qs); otherwise both go through spatial_to_spin_orbitals and
    key 0 is added here.

    Only a restricted Hamiltonian is packed: one channel as it is, and two
    channels only where they agree, h_a with h_b and the three blocks of
    v_eff, to 1e-12 (relative to max(1, max element)). The arrays of g_eff
    are copies, never views of the inputs; an empty active space packs
    zero-size arrays. A spin-resolved g_eff for the unrestricted case, solved
    by fci.direct_uhf, is deferred.
    """
    if len(h_eff) == 1:
        (h_a,), (aa,) = h_eff, v_eff
    else:
        h_a, h_b = h_eff
        aa, ab, bb = v_eff
        if not (_same(h_a, h_b) and _same(aa, ab) and _same(aa, bb)):
            raise NotImplementedError(
                "pack takes a restricted Hamiltonian only (one spin channel, "
                "or equal alpha and beta channels); the unrestricted "
                "extension, a spin-resolved g_eff solved with fci.direct_uhf, "
                "is deferred.")
    spatial = {1: np.array(h_a), 2: np.array(aa.transpose(0, 2, 1, 3))}
    if spin_free:
        return {0: e0, **spatial}
    g = spatial_to_spin_orbitals(spatial, h_a.shape[0])
    g[0] = e0
    return g


def _same(x, y):
    if x.shape != y.shape:
        return False
    if x.size == 0:
        return True
    scale = max(1.0, float(np.max(np.abs(x))), float(np.max(np.abs(y))))
    return np.max(np.abs(x - y)) <= RESTRICTED_TOL * scale
