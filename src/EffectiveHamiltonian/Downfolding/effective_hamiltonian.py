"""The two-step downfolding driver: run a low-level method on the mean field
once, then build the effective Hamiltonian of any number of active spaces
from it.

    low = run_low_level(mf, method='hf')
    h = build_effective_hamiltonian(low, active_space)
    e = h.solve('pyscf_fci')          # h.g_eff, h.space, h.energies, ...

The driver knows no method: it reaches them through `LOW_LEVEL` and reads
only the `DownfoldingTerms` a method hands back. The mean-field object alone
decides the number of spin channels. A restricted reference (RHF/RKS) is ONE
channel (the orbitals, the `ActiveSpace`, one vtilde), and everything is
computed once: one exact J/K build of the inactive core, shared by the core
Fock operator and the environment energy, and one integral transformation of
the active block. An unrestricted one (UHF/UKS) is two, alpha and beta, each
with its own orbitals, occupied count and active space, also when the two
coincide:

    low = run_low_level(uhf, method='hf')
    h = build_effective_hamiltonian(low, (space_alpha, space_beta))
    e = h.solve('pyscf_fci')

The corrections are computed once each, the frozen core (core.py) and the
double counting (double_counting.py), and the three parts of H_eff only add
them up: E_0 in zero_body.py, h_eff in one_body.py, v_eff in two_body.py.
`pack` here writes them into the g_eff dictionary the active-space solvers
read (src/EffectiveHamiltonian/solvers/solve.py).
"""
from dataclasses import dataclass

import numpy as np
from pyscf import scf

from src.Base.numerical_flags import collecting, gathered
from src.Base.orbital_spaces import closed_shell_nocc, unrestricted_nocc
from src.Base.pyscf.pyscf_interface import (get_antisymmetrized_integrals,
                                            get_orbital_energies,
                                            require_gas_phase_ground_state,
                                            spin_orbital_chemist,
                                            spin_orbital_one_body,
                                            unrestricted_reference_diagnostics)
from src.Base.utils.parallel.mpi_grid import grid_comm
from src.Base.utils.choices import require_choice
from src.EffectiveHamiltonian.solvers.determinants import (
    spatial_to_spin_orbitals)
from src.EffectiveHamiltonian.solvers.solve import solve

from .core import core_terms
from .double_counting import double_counting
from .methods import LOW_LEVEL
from .one_body import effective_one_body
from .spin import SPINS
from .two_body import effective_interaction
from .zero_body import Energies, assemble_energies

WHO = 'Downfolding'


def _spin_channels(mf):
    """(mo, eps, nocc), each a tuple over the spin channels of `mf`: one
    channel for a restricted reference, (alpha, beta) for UHF/UKS. nocc is
    the occupied count of each channel, refused unless its occupied orbitals
    are its lowest ones: every active space takes [0, nocc) as the occupied
    block. An unrestricted reference is also refused unless every
    occupation is 0 or 1 (`unrestricted_nocc`)."""
    if not isinstance(mf, scf.uhf.UHF):
        return ((mf.mo_coeff,), (get_orbital_energies(mf),),
                (closed_shell_nocc(mf.mo_occ),))
    return (tuple(mf.mo_coeff), tuple(get_orbital_energies(mf)),
            unrestricted_nocc(mf.mo_occ))


def _active_spaces(active_space, eps, nocc):
    """The `ActiveSpace` of each spin channel as a tuple: one for a
    restricted reference, (alpha, beta) for an unrestricted one, whose active
    space is a pair with the same number of active orbitals. Each space must
    be set up for the orbital and occupied counts of its own channel, and
    must not cut a degenerate set of its channel's orbitals
    (`ActiveSpace.require_no_degenerate_cut`)."""
    paired = isinstance(active_space, (tuple, list))
    if len(eps) == 1:
        if paired:
            raise ValueError(
                'a restricted reference takes one ActiveSpace; a pair '
                '(space_alpha, space_beta) goes with a UHF/UKS reference.')
        spaces, names = (active_space,), (None,)
    else:
        if not paired or len(active_space) != 2:
            raise ValueError(
                'an unrestricted (UHF/UKS) reference takes a pair of active '
                'spaces, (space_alpha, space_beta): one ActiveSpace per spin, '
                'each built from the orbital and occupied counts of that '
                'spin.')
        spaces, names = tuple(active_space), SPINS
        if spaces[0].n_act != spaces[1].n_act:
            raise ValueError(
                f'the alpha and beta active spaces hold {spaces[0].n_act} and '
                f'{spaces[1].n_act} orbitals; both spins take the same number '
                f'of active orbitals.')
    for space, e, n, spin in zip(spaces, eps, nocc, names):
        if (space.norb, space.nocc) != (len(e), n):
            which, has = (('the active space', f'{n} occupied') if spin is None
                          else (f'the {spin} active space',
                                f'{n} {spin} occupied'))
            raise ValueError(
                f'{which} is set up for {space.nocc} occupied of '
                f'{space.norb} orbitals; the mean field has {has} of '
                f'{len(e)}.')
        space.require_no_degenerate_cut(e, spin)
    return spaces


@dataclass(frozen=True, eq=False)
class DownfoldedHamiltonian:
    """One effective Hamiltonian, as `build_effective_hamiltonian` returns it.

    g_eff: {0: E_0, 1: h_eff, 2: v_eff}, the input of the active-space
        solvers. space: the active space in QDPT's form (n_orbs,
        n_elec_active, the index arrays, spin_free, active_space, and for an
        unrestricted reference spin_overlap). energies: the `Energies`, every
        term of E_0. diagnostics: the low-level method's checks, keyed by
        name. numerical_flags: the named flags behind this Hamiltonian
        (src.Base.numerical_flags), each {'flag': name, 'message': text};
        empty for a clean result.
    """
    g_eff: dict
    space: dict
    energies: Energies
    diagnostics: dict
    numerical_flags: list

    def solve(self, solver, **kwargs):
        """The lowest energies of this Hamiltonian by `solver`
        (src.EffectiveHamiltonian.solvers.solve.solve), E_0 included: the
        record's g_eff, orbital and electron counts and spin_free are filled
        in, and for an unrestricted reference its spin_overlap, so <S^2>
        can be reported (out=...). Other kwargs go to `solve`."""
        if 'spin_overlap' in self.space:
            kwargs.setdefault('spin_overlap', self.space['spin_overlap'])
        return solve(solver, self.g_eff, self.space['n_orbs'],
                     self.space['n_elec_active'],
                     spin_free=self.space['spin_free'], **kwargs)


def run_low_level(mf, method='rpa', **method_opts):
    """The low-level result of `method` on the mean field `mf` (attributes
    `method` and `mf`, plus whatever the method keeps for its `downfold`).

    The result does not depend on an active space, so one run serves every
    active space built from it. A gas-phase, T = 0 mean field is taken,
    restricted closed-shell (RHF/RKS) or unrestricted (UHF/UKS) with the
    lowest orbitals of each spin occupied by one electron each (no
    fractional occupations), and only serially: on more than
    one MPI rank every rank would repeat the whole calculation. An open
    shell run restricted (ROHF/ROKS) is refused: its orbitals are not
    canonical for the alpha and beta Fock operators. method_opts go to the
    method, which refuses the ones it does not know.

    Every named flag raised during the run (src.Base.numerical_flags) is
    collected into `low.numerical_flags`, a list of {'flag': name, 'message':
    text}, empty for a clean run. An unrestricted reference that is
    spin-contaminated or internally unstable is taken, with a warning and
    the flags 'uhf_spin_contaminated' / 'uhf_reference_unstable' there
    (`unrestricted_reference_diagnostics`, whose numbers are kept on the
    mean field as `mf.reference_diagnostics`).
    """
    _, _, size = grid_comm()
    if size > 1:
        raise NotImplementedError(
            f'{WHO} runs serially only; this process is one of {size} MPI '
            f'ranks. Run it on a single rank.')
    require_gas_phase_ground_state(mf, WHO, unrestricted=True)
    if getattr(mf, 'with_solvent', None) is not None:
        # a pyscf solvent model is not an attached environment, which is all
        # require_gas_phase looks for
        raise NotImplementedError(
            f'{WHO} is gas phase only; mf carries a solvent model '
            f'({type(mf.with_solvent).__name__}).')
    if isinstance(mf, scf.uhf.UHF):
        # the check above does not look at the occupations of an
        # unrestricted reference: refuse, before the method runs, a spin
        # whose occupations are fractional or whose occupied orbitals are
        # not its lowest
        _spin_channels(mf)
    method = require_choice(method, tuple(LOW_LEVEL), 'method')
    flags = []
    with collecting(flags):
        unrestricted_reference_diagnostics(mf, WHO,
                                           what='effective Hamiltonians')
        low = LOW_LEVEL[method].run(mf, **method_opts)
    low.numerical_flags = gathered(flags)
    return low


def build_effective_hamiltonian(low, active_space, spin_free=True,
                                **method_opts):
    """The `DownfoldedHamiltonian` of `active_space`, downfolded from the
    low-level result `low`: g_eff, space, energies, diagnostics and
    numerical_flags. `h.solve(solver)` solves it.

    g_eff and space follow QDPT's build_effective_hamiltonian, so g_eff goes
    straight to src.EffectiveHamiltonian.solvers.solve.solve:
    g_eff = {0: E_0, 1: h_eff, 2: physicist v_eff}, spatial when spin_free,
    else over the 2 n_act interleaved spin orbitals; space holds n_orbs,
    n_elec_active, act_idx, act_occ_idx, act_vir_idx (canonical spatial
    indices), spin_free and active_space. energies is the `Energies` record
    of every E_0 term; diagnostics are the method's, keyed by name.

    For an unrestricted reference (UHF/UKS) `active_space` is a pair
    (space_alpha, space_beta), one ActiveSpace per spin with independent
    orbital indices and the same number of active orbitals. Then g_eff[1] =
    (h_alpha, h_beta) and g_eff[2] = (aa, ab, bb) when spin_free, each
    two-body block physicist like the restricted one, the opposite-spin
    block <p_alpha q_beta | r_alpha s_beta>; else the same 2 n_act
    interleaved spin orbitals as for a restricted reference. In space,
    n_elec_active = (N_alpha, N_beta), the index entries and active_space
    are pairs (alpha, beta), and spin_overlap holds the blocks of
    C_alpha^T S C_beta between the inactive ('core') and the active orbitals
    of the two spins, 'core_core', 'core_active' (alpha core, beta active),
    'active_core' and 'active_active', which `solve` takes to report <S^2>.
    Every term of energies is the total over both spins.

    h_eff = f - t^DC (+ Sigma_bar) with f the Fock operator of the inactive
    core, v_eff = v + vtilde with v the exact active integrals; the method
    supplies vtilde, Sigma_bar, the correlation-energy pair and the
    double-counting mode. method_opts go to the method's `downfold`.

    `numerical_flags` lists every named flag behind this
    Hamiltonian (src.Base.numerical_flags): those of the low-level run
    (`low.numerical_flags`) and those raised while building it, each
    distinct {'flag': name, 'message': text} once; empty for a clean result.
    """
    mf = low.mf
    mol = mf.mol
    mo, eps, nocc = _spin_channels(mf)
    spaces = _active_spaces(active_space, eps, nocc)
    unrestricted = len(spaces) == 2
    method = require_choice(low.method, tuple(LOW_LEVEL), 'method')
    flags = []
    with collecting(flags):
        # `downfold` is called from this function itself, not from a helper:
        # the method's warnings count the frames up to the caller of this one
        terms = LOW_LEVEL[method].downfold(
            low, spaces if unrestricted else spaces[0], **method_opts)
        _check_terms(terms, method, unrestricted)

        core = core_terms(mol, mf.get_hcore(), mo, spaces)
        dc = double_counting(terms, tuple(s.n_act_occ for s in spaces))
        h_eff = effective_one_body(core.fock, dc.hartree, dc.exchange,
                                   dc.mode, terms.sigma_bar)
        v_eff = effective_interaction(
            mol, tuple(c[:, s.act] for c, s in zip(mo, spaces)), terms.vtilde)
        energies = assemble_energies(mf.energy_nuc(), core, dc, terms)
        g_eff = pack(h_eff, v_eff, energies.e0, spin_free)
    inherited = list(getattr(low, 'numerical_flags', ()))
    return DownfoldedHamiltonian(
        g_eff=g_eff, space=_space(mf, mo, spaces, spin_free),
        energies=energies, diagnostics=dict(terms.diagnostics),
        numerical_flags=gathered(inherited + flags))


def _check_terms(terms, method, unrestricted):
    """Refuse a `DownfoldingTerms` whose spin blocks do not fit the reference:
    vtilde = (v,) and sigma_bar = (s,) for a restricted one, (aa, ab, bb)
    and (a, b) for an unrestricted one (spin.py)."""
    blocks = [len(terms.vtilde)]
    if terms.sigma_bar is not None:
        blocks.append(len(terms.sigma_bar))
    if blocks != ([3, 2] if unrestricted else [1, 1])[:len(blocks)]:
        raise ValueError(
            f"method={method!r} returned {len(terms.vtilde)} vtilde blocks"
            + (f" and {len(terms.sigma_bar)} Sigma_bar blocks"
               if terms.sigma_bar is not None else "")
            + ("; an unrestricted reference is two spin channels: "
               "vtilde = (aa, ab, bb), sigma_bar = (a, b)."
               if unrestricted else
               "; a restricted reference is one spin channel: "
               "vtilde = (v,), sigma_bar = (s,)."))


def _space(mf, mo, spaces, spin_free):
    """The `space` record of `build_effective_hamiltonian`: the active space
    in the form QDPT's builder gives it, each index entry a pair (alpha, beta)
    for an unrestricted reference, which also gets spin_overlap."""
    restricted = len(spaces) == 1

    def per_channel(view):
        return view(spaces[0]) if restricted else tuple(map(view, spaces))

    n_act = spaces[0].n_act
    space = {
        'n_orbs': n_act if spin_free else 2 * n_act,
        'n_elec_active': (spaces[0].nelec_active if restricted
                          else tuple(s.n_act_occ for s in spaces)),
        'act_idx': per_channel(lambda s: s.act),
        'act_occ_idx': per_channel(lambda s: s.occ_active),
        'act_vir_idx': per_channel(lambda s: s.virt_active),
        'spin_free': spin_free,
        'active_space': spaces[0] if restricted else spaces,
    }
    if not restricted:
        space['spin_overlap'] = _spin_overlap(mf, mo, spaces)
    return space


def _spin_overlap(mf, mo, spaces):
    """The blocks of O = C_alpha^T S C_beta between the inactive ('core') and
    the active orbitals of the two spins, rows alpha and columns beta, the
    active orbitals occupied first as in H_eff: what <S^2> of an active-space
    state with its core determinant needs, since alpha orbital p and beta
    orbital p of an unrestricted reference are different functions."""
    (c_a, c_b), (space_a, space_b) = mo, spaces
    overlap = c_a.T @ mf.get_ovlp() @ c_b
    rows = {'core': space_a.env_occ, 'active': space_a.act}
    cols = {'core': space_b.env_occ, 'active': space_b.act}
    return {f'{left}_{right}': overlap[np.ix_(i, j)]
            for left, i in rows.items() for right, j in cols.items()}


def pack(h_eff, v_eff, e0, spin_free):
    """The g_eff dictionary {0: e0, 1: one-body, 2: two-body} of the solvers.

    h_eff and v_eff hold the one-body term and the chemist active interaction
    (bare plus vtilde) per spin channel, (h,) and (v,) or (h_a, h_b) and
    (aa, ab, bb) (spin.py). The number of channels alone decides the form;
    two channels are never collapsed to one, also when they are equal.

    One channel, spin-free: g_eff[1] is the spatial one-body and g_eff[2] the
    spatial physicist <pq|rs> = (pr|qs); otherwise both go through
    spatial_to_spin_orbitals and key 0 is added here.

    Two channels, spin-free: the per-spin blocks, g_eff[1] = (h_a, h_b) and
    g_eff[2] = (aa, ab, bb), each two-body block physicist like the
    one-channel one, block[p,q,r,s] = <pq|rs> = (pr|qs), so the
    opposite-spin block is <p_a q_b | r_a s_b>. Otherwise the same
    interleaved spin orbitals as for one channel (alpha at 2p, beta at
    2p + 1) with the antisymmetrized physicist two-body term.

    The arrays of g_eff are copies, never views of the inputs; an empty
    active space packs zero-size arrays.
    """
    if len(h_eff) == 1:
        (h_a,), (aa,) = h_eff, v_eff
        spatial = {1: np.array(h_a), 2: np.array(aa.transpose(0, 2, 1, 3))}
        if spin_free:
            return {0: e0, **spatial}
        g = spatial_to_spin_orbitals(spatial, h_a.shape[0])
        g[0] = e0
        return g
    if spin_free:
        return {0: e0, 1: tuple(np.array(h) for h in h_eff),
                2: tuple(np.array(v.transpose(0, 2, 1, 3)) for v in v_eff)}
    physicist = spin_orbital_chemist(*v_eff).transpose(0, 2, 1, 3)
    return {0: e0, 1: spin_orbital_one_body(*h_eff),
            2: get_antisymmetrized_integrals(physicist)}
