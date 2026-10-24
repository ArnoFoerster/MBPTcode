"""GW-BSE states as state models of the self-consistent solvation loop.

`src.environments.solvation_loop` iterates a state model against a medium
(`src.environments.reaction_field`) through one call,

    result = state(h, e)

h an AO one-electron operator added to the core Hamiltonian (the induced
charges the loop holds), e a constant; result.energy the state's energy in
H_vac + h + e and result.dm_eff its orbital-relaxed density dE/dh (AO,
spin-summed). This module is that call for GW-BSE:

    GWBSEState(mol, make_scf)                     the reference (HF or KS):
        E = E_ref, D_eff = D_ref
    GWBSEState(mol, make_scf, 'singlet', state=k, lr_environment=...)
        E = E_ref + Omega_k, D_eff = D_ref + drho_rel(Omega_k)

The reference is `state_models.HFState` (the mean field in the field, its
refusal of loose orbital gradients, its rank lockstep before any decision);
each call hands the chain a copy of it, because the chain holds what it
computed at a geometry by the identity of the mean field
(`state_following.ExcitedStateChain.forward`), and a mean field
re-converged in place would be taken for the last one.

drho_rel = d Omega / d h is the chain's Lagrangian density
(`ExcitedStateChain.excitation_density(relaxed=True, frozen_field=True)`):
the orbitals respond with the gas-like kernel of the mean field in the
charges held, which is the loop's partition (the charges are a fixed
external field inside one call; the loop makes them self-consistent).

The BSE reads the medium only through its linear-response term, the fast
answer to the transition density: `lr_environment` is the medium's
`transition_environment()` (an optical continuum; the loop refuses another,
`solvation_loop.require_field_lr`), wrapped as
`split_equilibrium.LinearResponseOnly` (exchange at eps_inf, Eq. (18) and the
direct W on the bare interaction); at eps_inf = 1 it answers nothing and the
BSE is the gas phase's. The state-specific part is the loop's, so this is
scheme B (classical state-specific, `split_equilibrium.ClassicalStateSpecific`)
made self-consistent; its first loop cycle in the ground state's equilibrium
charges is scheme B's relaxed first-order term.

`state.force(h, e)` -> (F, info): the nuclear force in the fixed charges, the
external operator's own derivative left to the medium
(`reaction_field.ContinuumField.force` at info['dm_eff']): the reference's
SCF gradient plus the chain's dOmega/dR, which already holds the derivative of
its exchange dressing (its cavity moves with the atoms), so info['lr'] is
empty.

One surface. The chain is built once, on the first call's mean field, and
freezes its conventions there (`FrozenConventions`); each later call hands it
the mean field converged in the new charges at the same nuclei, and the root
is the chain's (by index, or followed with `bse.track`). `at(mol, make_scf)`
is the same surface at other nuclei (the reference's chain), handed the
loop field's `transition_environment()` there, which the loop checks as it
does any model's and the model refuses unless it is the chain's continuum
(model, eps_inf, cavity settings) around the new nuclei; a chain built
afresh at each geometry freezes its own conventions and is another surface
(3.8e-5 Ha/Bohr off the force of water's T1 in toluene), so a walk takes
`at` and `surface_key` (the chain) tells a walker which surface a model is
on (`gradients.solvated_surface.SolvatedSurface` refuses a second one).
`commit(mol)` is a walk's accepted step: the chain's conventions and its
followed state move there (`ExcitedStateChain.commit`).

Refreeze. `refrozen(mol, lr_environment)` is this state on the surface a
chain built afresh at `mol` freezes, as a vertical energy's model does: a
model with no chain, whose first call builds one on the mean field it is
handed at `mol` (`gradients.solvated_surface.SolvatedSurface.refreeze`
hands it the zero-charge one, as the loop of a vertical does). The physics,
the kernel, the root and the Numerics are this model's; a chain following its
state (`bse.track`) hands the followed state over, its index at `mol`
(`followed_root`) and its committed eigenvector, which the new chain's first
evaluation matches by overlap (as `ExcitedStateChain.refreeze` does), so the
rebuilt surface cannot start on the other state of a near-degenerate pair.
The reference (spin None) freezes nothing and is its own refreeze.

Checkpoint (`solvation_loop.equilibrate(checkpoint=...)`): `checkpoint()`
holds the reference's warm start and, for a model that built its chain, the
operator of its first call, from which `restore` rebuilds the same chain (the
same SCF from no guess, the same first evaluation) before it takes the warm
start back; a model moved with `at` holds no chain of its own and is
restored only on the chain it was given. A chain following its state
(`bse.track`) is rebuilt following the first call's root, which at one
geometry is the root the loop converged on. `identity()` keys the
checkpoint: the reference, spin, root, kernel and the chain's Numerics.

Under ranks every rank runs the loop: the mean field (HFState), the energy,
the density and the force are rank 0's on every rank (`lockstep`), so the
charges the loop derives from them, and every decision it takes, agree on all
ranks.
"""
import numpy as np

from src.Base.numerics.numerics import numerics_of
from src.environments.split_equilibrium import LinearResponseOnly
from src.Base.utils.parallel.mpi_grid import lockstep
from src.gradients.df_correlation_force import hf_force
from src.gradients.state_models import HFState, StateResult
from src.gradients.gwbse.excited_state import ExcitedStateChain, dipole_change
from src.gradients.surface import followed_root


class GWBSEState:
    """The reference (spin None) or one BSE root of a G0W0-BSE surface in an
    external field (module docstring).

    make_scf(mol) -> a configured, unconverged pyscf mean field (RHF or RKS,
    density fitting, conv_tol_grad at most STATE_MODEL_SCF_GRAD_TOL) in the
    gas phase; its core Hamiltonian is replaced by h_vac + h on each call and
    it is warm-started from the last. spin 'singlet' | 'triplet' | None;
    state: zero-based root; numerics: the one `Numerics` of the calculation
    (or a mapping for `numerics_of`), required: the chain's settings and the
    loop's (its solvent section, `solvation_loop.loop_numerics`); bse_tda the
    kernel.
    """

    def __init__(self, mol, make_scf, spin=None, state=0, bse_tda=False,
                 lr_environment=None, numerics=None, chain=None):
        if numerics is None:
            raise ValueError('GWBSEState needs its Numerics (the ISDF grid, the BSE '
                             'solver, the loop tolerances): there are no defaults '
                             'to fall back on')
        self.mol, self.spin, self.state, self.bse_tda = mol, spin, state, bse_tda
        self.make_scf = make_scf
        self.reference = HFState(mol, make_scf)
        # what the loop checks against its field's transition_environment()
        self.lr_environment = lr_environment
        env = None if lr_environment is None else LinearResponseOnly(lr_environment)
        # a continuum at eps_inf = 1 answers with exact zeros: the gas phase
        self.environment = None if env is None or env.vanishes else env
        self.numerics = (numerics_of(numerics) if hasattr(numerics, 'items')
                         else numerics)
        self.chain = chain
        self._owns_chain = chain is None
        # a followed state handed over by `refrozen`, for the chain built first
        self._handover = None
        self._first = None
        self.mf = None
        # (h, e, result) of the last call, read back for the same operator
        self._last = None

    def _scf(self, h, e):
        r = self.reference(h, e)
        if self._first is None:
            self._first = (np.array(h, float), float(e))
        # a fresh identity per call (module docstring)
        self.mf = self.reference.mf.copy()
        return r

    @staticmethod
    def _no_factory(mol):
        raise RuntimeError('a GW-BSE state model is evaluated in the mean '
                           'fields its loop hands it, at one geometry')

    @property
    def surface_key(self):
        """The object that fixes this model's surface: its chain (None for
        the reference, which has none, and before the first call)."""
        return self.chain

    def at(self, mol, make_scf, lr_environment=None):
        """This state on the same surface at the nuclei of `mol`: a fresh
        model whose chain is this one's, so the conventions stay those frozen
        at the first geometry (`FrozenConventions`), as on a walk.
        lr_environment: the loop field's `transition_environment()` at `mol`;
        refused unless it is the chain's continuum there (module docstring)."""
        if self.spin is not None and self.chain is None:
            raise ValueError('the surface is fixed at the first evaluation; '
                             'call this model once before moving it')
        moved = GWBSEState(mol, make_scf, self.spin, self.state, self.bse_tda,
                           lr_environment=lr_environment, numerics=self.numerics,
                           chain=self.chain)
        self._require_same_continuum(moved)
        return moved

    def _require_same_continuum(self, other):
        """Refuse `other` unless its environment is the continuum this
        surface's chain answers in (module docstring)."""
        if _settings(other.environment) != _settings(self.environment):
            raise ValueError('the field\'s transition_environment() is not the '
                             'continuum this surface\'s chain answers in (model, '
                             'eps_inf or cavity differ, or one of them is absent): '
                             'build the field as at the first geometry')

    def refrozen(self, mol, lr_environment=None):
        """This state on the surface a chain built afresh at the nuclei of
        `mol` freezes (module docstring); lr_environment: the loop field's
        `transition_environment()` at `mol`, as for `at`."""
        if self.spin is None:
            return self
        if self.chain is None:
            raise ValueError('the surface is fixed at the first evaluation; '
                             'call this model once before refreezing it')
        chain = self.chain
        following = chain.following()
        fresh = GWBSEState(mol, self.make_scf, self.spin,
                           followed_root(chain, mol) if following is not None
                           else self.state,
                           self.bse_tda, lr_environment=lr_environment,
                           numerics=self.numerics)
        self._require_same_continuum(fresh)
        if following is not None:
            fresh._handover = following
        return fresh

    def commit(self, mol):
        """A walk accepted `mol` (module docstring); the reference follows
        nothing."""
        if self.chain is not None:
            self.chain.commit(mol)

    def _chain(self, mf):
        if self.chain is None:
            self.chain = ExcitedStateChain(
                self.mol, self._no_factory, spin=self.spin, state=self.state,
                bse_tda=self.bse_tda, environment=self.environment, mf=mf,
                numerics=self.numerics)
            if self._handover is not None:
                self.chain.hand_over(self._handover)
        return self.chain

    def __call__(self, h, e):
        h = np.asarray(h, float)
        last = self._last
        if (last is not None and float(e) == last[1]
                and np.array_equal(h, last[0])):
            # the same operator at the same nuclei: the stored answer, not a
            # second SCF, forward and relaxed density (a gas-phase loop's every
            # cycle, the force after a converged loop)
            r = last[2]
            return StateResult(r.energy, np.array(r.dm_eff), dict(r.info))
        r = self._evaluate(h, e)
        self._last = (h.copy(), float(e), StateResult(
            r.energy, np.array(r.dm_eff), dict(r.info)))
        return r

    def _evaluate(self, h, e):
        r = self._scf(h, e)
        if self.spin is None:
            return StateResult(r.energy, r.dm_eff, {'e_ref': r.energy,
                                                    'flags': r.info['flags']})
        mf = self.mf
        chain = self._chain(mf)
        omega = float(lockstep(np.array([chain.excitation(self.mol, mf)]))[0])
        drho = lockstep(chain.excitation_density(self.mol, mf, relaxed=True,
                                                 frozen_field=True))
        dip = dipole_change(self.mol, drho)
        return StateResult(r.energy + omega, r.dm_eff + drho, {
            'e_ref': r.energy, 'omega': omega, 'delta_density': drho,
            'delta_dipole': dip, 'flags': r.info['flags']})

    def force(self, h, e):
        """(F, info) in the fixed charges (module docstring): the reference's
        own gradient class on the mean field this call handed the chain
        (`HFState.force` would converge it once more), plus dOmega/dR."""
        r = self(h, e)
        F = np.asarray(hf_force(self.mf), float)
        if self.spin is not None:
            F = F + self.chain.excitation_gradient(self.mol, self.mf)[0]
        return lockstep(F), dict(dm_eff=r.dm_eff, lr=[], flags=r.info['flags'])

    def identity(self):
        """What this model computes (module docstring)."""
        return dict(self.reference.identity(), model='GW-BSE', spin=self.spin,
                    root=self.state, bse_tda=bool(self.bse_tda),
                    lr=self.environment is not None,
                    numerics=self.numerics.identity())

    def checkpoint(self):
        """The arrays `restore` needs (module docstring)."""
        out = self.reference.checkpoint()
        if self.spin is not None and self._owns_chain and self._first is not None:
            out['first_h'], out['first_e'] = self._first[0], np.array(self._first[1])
        return out

    def restore(self, saved):
        """Take back what `checkpoint` held: the same chain, then the warm
        start."""
        if self.spin is not None and self.chain is None:
            if 'first_h' not in saved:
                raise ValueError('the checkpoint holds no chain of its own (a model '
                                 'moved with `at`): restore it on a model built with '
                                 'the same chain')
            self(saved['first_h'], float(saved['first_e']))
        self.reference.restore(saved)


def _settings(environment):
    """What fixes a continuum's linear-response answer: its model, eps_inf
    and cavity settings (None for none)."""
    if environment is None:
        return None
    optical = environment.optical
    cavity = tuple(sorted((k, np.asarray(v).tobytes() if isinstance(v, np.ndarray)
                           else v) for k, v in optical.cavity_settings.items()))
    return optical.method, float(optical.eps_optical), cavity
