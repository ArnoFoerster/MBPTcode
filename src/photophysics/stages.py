"""The stages of a TADF run on the production BSE@GW chain, each a function
of a `Run` that returns its record, and `run_tadf`, which runs them in order.

    ground_minimum   s0    the ground-state minimum R0 on the declared mean
                           field (in PCM at eps_static in a solvent)
    ground_modes     hess  the S0 Hessian at R0 and its normal modes:
                           pyscf's analytic Hessian of a density-fitted twin,
                           or central differences of the declared mean
                           field's force (`hessian_numerics`)
    vertical         vert  ONE shared evaluation at R0
                           (`StateManifold.evaluate`): both spectra, the
                           oscillator strengths, the spin-orbit table,
                           d(S0,S1), d(S1,S2), d(T1,T2), the second-order
                           spin-vibronic derivative of <S1|H_SO|T1>, and with
                           a zero nudge the S1 force, the S1 walk's first
                           point
    s1_minimum       s1    the S1 walk from R0 (or from the far start), then
                           one shared evaluation at its minimum
    t1_minimum       t1    the T1 walk from the S1 minimum, started on the S1
                           walk's final Hessian, then one evaluation there
    uisc.vib         vib   the UISC model's promoting-mode derivatives at
                           the S1 and T1 minima (`uisc`)
    rates            rate  the rate table off the records: the declared
                           rate model (`rate_models`, `uisc`)

Each stage is skipped when its record is in the store under this
declaration's stamp. The physics of each policy is written on its field in
`declaration.Policies`; how the stages carry it:

    nudge        every excited walk starts displaced: the frozen surface is
                 invariant to rigid motion, so an unnudged walk keeps the
                 point group of its start; refreeze passes are not nudged
    far start    the S1 walk starts on a chain built at the start
                 (`Run.started_at`), its own reference (R0's conventions
                 refrozen by orbital identity do not carry that far); its
                 record is
                 's1_start_chain', from which a later T1 rebuild starts
    refreeze     the conventions frozen at R0 are rebuilt at each minimum
                 and the walk goes again while the drift exceeds
                 REFREEZE_TOL_MEV (at most REFREEZE_MAX_PASSES); a drift
                 still above it is warned about and flagged
                 ('refreeze_not_converged')
    adaptive     the set selected at R0 is in vert; each refreeze selects
                 again out of R0's candidates, grow-only
    step refusal a walk step is shortened only for a StepFailed; root
                 following matches against the last accepted geometry

WHAT A RELAUNCH REUSES, inside the s1 and t1 stages, coarsest first:

    <stage>_walked     the converged walk and the surface it ended on; the
                       relaunch rebuilds that surface from its record (one
                       SCF at its reference) and goes on at the saddle check
    <stage>_progress   the walk after its last accepted step, in either pass;
                       the relaunch resumes there on the surface rebuilt from
                       its record
    hess_<stage>/      each displaced force of the saddle check's Hessian
    <stage>_escape_progress   the saddle check's escape walk, as above

A WALK THAT DID NOT CONVERGE IS WALKED AGAIN. Its stage record (or its
'<stage>_walked') is flagged 'walk_not_converged' and the run carried on from
where it stopped; a relaunch that asks for the stage again sets that stage
and every stage built on its minimum aside (`walk_again`: s0 takes every
stage, s1 takes t1, vib and rate, t1 takes vib and rate) into
unconverged/<stage>-<time>/, and the stage walks again from its start, as the
solvated driver does (tools/solvation, `_drop_unconverged`). A far start's
chain ('s1_start_chain') is the far start's and stays.

Under ranks (`with distributed(comm):` around `run_tadf`) every rank runs
the stages in lockstep and rank 0 writes; no stage takes a communicator.

NUMERICAL FLAGS (refuse inputs, warn numerics). A
numerical event does not stop the run: it is warned about and named in the
stage record's `numerical_flags` ({flag: what it says}, empty when clean):
'walk_not_converged' and 'refreeze_not_converged' of a walk (`relaxation_flags`),
'saddle_check' where the saddle check did not end on a minimum,
'scf_not_converged' (the count of SCFs the stage built that stopped short)
and 'isdf_df_check_failed' (vert, s1, t1). The rate table gathers them under
`numerical_flags` by stage, beside its own ('fc_integrand_not_decayed'), so
every rate, adiabatic energy and dE_ST built on a flagged step says so.
"""
import functools
import time
import warnings
from dataclasses import asdict, replace
from typing import Callable, NamedTuple

import numpy as np
from pyscf import gto

from src.Base.constants import (HARTREE_TO_CM, HARTREE_TO_EV, HARTREE_TO_MEV,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import Excitation
from src.Base.environment import equilibrium_environment
from src.Base.geometry import elements_of, molecule_at
from src.Base.pyscf_interface import occupied_count
from src.Base.scf_convergence import WarmStart
from src.Base.separable_ri import (frame_turning_record, resolve_isdf_grid,
                                   shell_tuple)
from src.Base.utils.mpi_grid import (broadcast, current_comm, distributed,
                                     is_root, mpi_map, n_ranks)
from src.SingleReference.GW.qp_selection import AdaptivePartition
from src.SingleReference.LinearResponse.davidson import oscillator_strengths
from src.gradients.derivative_coupling import (analytic_coupling,
                                               analytic_ground_coupling)
from src.gradients.state_manifold import StateManifold
from src.photophysics import chain, df_check, rate_models, uisc
from src.photophysics.minimum import (GW_BSE, GroundModes,
                                      from_gw_bse_record)
from src.photophysics.records import (StageRefused, WalkNotRelaxed,
                                      following, label, log,
                                      refreeze_summary, relaxation_flags,
                                      walk_fields)
from src.properties.excitations import (grid_record, qp_bookkeeping,
                                        saddle_check, surface_of)
from src.properties.hessian import (REFERENCE_FRAMES, numerical_hessian,
                                    require_hessian_counts,
                                    translation_residual)
from src.properties.optimize import (MeanFieldSurface, relax,
                                     saddle_test, surface_from_record,
                                     surface_record)
from src.properties.spin_orbit import spin_orbit_couplings
from src.properties.spin_vibronic import (merge_scf_flags,
                                          spin_vibronic_coupling)
from src.properties.surface import (driven_chain, followed_root,
                                    lockstep_geometry)
from src.properties.surfaces import reference_mean_field
from src.properties.vibronic import normal_modes

#: The stages in the order they run ('vib' is the UISC model's alone).
STAGE_ORDER = ('s0', 'hess', 'vert', 's1', 't1', 'vib', 'rate')
#: The stages whose geometry the ISDF-vs-DF acceptance check reads.
CHECKED_STAGES = ('vert', 's1', 't1')


# ----------------------------------------------------------------- stamps
class Stamps:
    """What each stage's record is a function of, for one declaration at
    the input molecule: the declared fields the stage reads, each grid by its
    name AND content (`chain.grid_content`), and the span the run is solved
    in (the run's `span`). A record inside a stage (a
    walk's progress, its walked state, the far start's chain) carries its
    stage's stamp."""

    def __init__(self, declaration, mol, span):
        d, n = declaration, declaration.numerics
        mean_field = {'name': d.mean_field_name,
                      'grid': (None if n.scf.route == 'df' else
                               chain.grid_content(n.scf.grid_level, mol))}
        ground = {'system': (d.system.name, d.system.atom, d.system.basis),
                  'reference': d.reference, 'environment': d.environment,
                  'mean_field': mean_field,
                  'scf': n.to_record()['scf'],
                  'span': span}
        hess = dict(ground, hessian=d.policies.hessian)
        if d.policies.hessian == 'numerical':
            h = d.hessian_numerics
            hess.update(hessian_numerics=h.identity(),
                        hessian_grid=(None if h.scf.route == 'df' else
                                      chain.grid_content(h.scf.grid_level,
                                                         mol)),
                        # the displaced mean fields fit on R0's pair layout
                        # and place their points on R0's frames
                        fd_hessian={'pair_layout': 'reference',
                                    'frames': REFERENCE_FRAMES})
        vert = dict(ground, numerics=n.identity(),
                    factors=chain.grid_content(n.isdf.level, mol),
                    kernel=d.kernel, ground=d.ground,
                    qp_states=asdict(d.qp_states),
                    states=(d.singlet, d.triplet, d.n_states))
        p = d.policies
        walk = dict(vert, solvation=d.solvation, nudge_bohr=p.nudge_bohr,
                    refreeze=p.refreeze, saddle_check=p.saddle_check,
                    far_start=(None if p.far_start is None
                               else p.far_start.sha256()),
                    engine=p.engine, scf_warm_start=p.scf_warm_start,
                    scf_finish=p.scf_finish)
        t1 = dict(walk)
        vib = dict(walk, hess=hess, rate_model=asdict(d.rate_model))
        rate = dict(vib, temperature=d.temperature,
                    lambda_env_ev=d.lambda_env_ev, df_check=p.df_check,
                    manifold_gaps=d.manifold_gaps)
        self.stamps = {'s0': ground, 'hess': hess, 'vert': vert, 's1': walk,
                       't1': t1, 'vib': vib, 'rate': rate}

    def __call__(self, name):
        """The stamp of record `name`: a stage's, or the stage a record
        inside one belongs to ('s1_progress', 's1_escape_walked',
        'isdf_df_check_vert')."""
        stage = name.replace('isdf_df_check_', '').split('_')[0]
        return self.stamps[stage]


# -------------------------------------------------------------- the run
class Run:
    """One run of `declaration` on `store`: the factories, the stamps, and
    the evaluations a later stage takes over from an earlier one in the same
    launch (the frozen chain, a first point). `max_memory` (MB) is a
    resource of the launch, not part of the declaration.

    The span, and with it the stamps and the factory, is chosen at first
    use (`span`): a run that stops before its first SCF (a dry run) probes
    nothing."""

    def __init__(self, declaration, store, max_memory):
        d = declaration
        self.declaration, self.store = d, store
        self.max_memory = int(max_memory)
        self.mol_in = gto.M(atom=d.system.atom, basis=d.system.basis,
                            max_memory=self.max_memory, verbose=0)
        self.env = chain.continuum(d, self.mol_in)
        #: The `SCFHistory` of every mean field the factories built.
        self.histories = []
        #: The walks' last converged density, every optimization SCF's guess.
        self.warm_start = WarmStart() if d.policies.scf_warm_start else None
        self.spec = chain.chain_spec(d, self.env)
        # the relaxed states' declaration: in a continuum at equilibrium
        self.relaxed_env = (equilibrium_environment(self.env)
                            if self.env is not None
                            and d.policies.relaxed_solvation == 'equilibrium'
                            else self.env)
        self.relaxed_spec = (self.spec if self.relaxed_env is self.env
                             else chain.chain_spec(d, self.relaxed_env))
        self.frozen = None   # the relaxed chain whose conventions R0 froze
        # the adaptive partition once selected at R0 (or read back from
        # vert): every later chain of this run takes it as it is
        self.partition = None
        self.first = {}      # {stage: (manifold, evaluation[, mol])}
        self.mol0 = None
        self.s_key = ('singlet', d.singlet - 1)
        self.t_key = ('triplet', d.triplet - 1)
        self.states = (tuple(('singlet', i) for i in range(d.n_states))
                       + tuple(('triplet', j) for j in range(d.n_states)))
        self.couplings = (following(*self.s_key), following(*self.t_key))

    # ---------------------------------------------------- span and stamps
    @functools.cached_property
    def span(self):
        """The span every SCF of the run is solved in (`chain.span_choice`),
        counted at the input geometry."""
        return chain.span_choice(self.declaration, self.mol_in)

    @functools.cached_property
    def stamp(self):
        """The run's `Stamps`, in its span."""
        return Stamps(self.declaration, self.mol_in, self.span)

    @functools.cached_property
    def factory(self):
        """The run's mean-field factory: its numerics, in its span,
        warm-started where the declaration says."""
        return self.mean_field_factory(self.declaration.numerics,
                                       warm_start=self.warm_start)

    # ------------------------------------------------------------ helpers
    def mean_field_factory(self, numerics, warm_start=None, span=None):
        """`chain.mean_field_factory` of this run on `numerics` (the run's
        own, or one derived from it), with its finish, in `span` (n_kept,
        cut), by default the run's own."""
        d = self.declaration
        if span is None:
            span = (self.span['n_kept'], self.span['cut'])
        return chain.mean_field_factory(
            d.reference, numerics, self.max_memory, self.histories,
            finish=d.policies.scf_finish, warm_start=warm_start, span=span,
            surface_route=d.numerics.scf.route)

    def write(self, name, record):
        self.store.write(name, record, self.stamp(name))

    def flag(self, stage, **flags):
        """Merge `flags` into the `numerical_flags` of the stored record of
        `stage`."""
        held = dict(self.read(stage).get('numerical_flags') or {})
        held.update(flags)
        self.store.update(stage, self.stamp(stage), numerical_flags=held)

    def unconverged_scfs(self, since=0):
        """How many of the mean fields built from `histories[since]` on
        stopped short of convergence (their last SCF run)."""
        return sum(1 for h in self.histories[since:]
                   if getattr(h, 'not_converged', False)
                   or (h.runs and h.runs[-1].get('converged') is False))

    def read(self, name):
        return self.store.read(name, self.stamp(name))

    def reusable(self, name):
        return self.store.reusable(name, self.stamp(name))

    def molecule_at(self, coords):
        """The run's molecule at `coords` (Bohr), one geometry on every
        rank."""
        return lockstep_geometry(molecule_at(self.mol_in, coords))

    def refuse_early(self):
        """Every refusal that would otherwise arrive hours into the run,
        before the first SCF: a mean-field grid no table validates at the
        basis, a factor grid no table validates (`grid_record`), and a
        finite-difference Hessian grid no table validates or that sits below
        ISDF_HESSIAN_MIN_GRID. Returns the factors' grid record."""
        d, mol = self.declaration, self.mol_in
        if d.numerics.scf.route == 'isdf':
            resolve_isdf_grid(d.numerics.scf.grid_level, str(mol.basis),
                              elements_of(mol),
                              auxbasis=chain.factor_auxbasis(mol))
        if d.policies.hessian == 'numerical' and \
                d.hessian_numerics.scf.route == 'isdf':
            counts, _ = resolve_isdf_grid(
                d.hessian_numerics.scf.grid_level, str(mol.basis),
                elements_of(mol), auxbasis=chain.factor_auxbasis(mol))
            shells = shell_tuple(counts)
            require_hessian_counts(str(mol.basis),
                                   {el: shells for el in
                                    elements_of(mol)})
        return grid_record(self.spec, mol)

    def manifold_states(self, *keys):
        """The reported roots of both spins, plus `keys` and the roots above
        them."""
        out = list(self.states)
        for spin, root in keys:
            for k in following(spin, root):
                if k not in out:
                    out.append(k)
        return tuple(out)

    def reference(self, gradients=(), couplings=(), relaxed=True):
        """The chain at R0 and one evaluation there, which freezes the
        conventions every later evaluation reads: the relaxed states'
        declaration, or with relaxed=False the vertical one (they are one
        declaration in the gas phase)."""
        spec = self.relaxed_spec if relaxed else self.spec
        mol0 = self.mol0
        mf0 = reference_mean_field(mol0, self.factory, spec.environment)
        excited = self.excited_at(spec, mol0, mf0, self.selected_partition())
        man = StateManifold(excited, states=self.manifold_states(
            self.s_key, self.t_key))
        ev = man.evaluate(mol0, mf0, gradients=gradients, couplings=couplings)
        self.partition = man.driven.resolved.qp_partition
        if spec is self.relaxed_spec:
            self.frozen = man.driven
        return man, ev

    def excited_at(self, spec, mol, mf, partition=None):
        """The excited surface of `spec` with its reference at `mol` on the
        mean field `mf` there, on `partition` (R0's adaptive partition,
        `selected_partition`), or selecting its own where None. A partition
        selected on the vertical declaration whose candidates the relaxed
        one does not share (another spectrum in another reaction field) is
        not carried, and that declaration selects its own."""
        excitation = Excitation('singlet', root=self.declaration.singlet,
                                kernel=self.declaration.kernel)
        try:
            return surface_of(with_partition(spec, partition), excitation,
                              mol, self.factory, mf=mf)
        except ValueError as exc:
            if partition is None or 'belongs to its candidates' not in str(exc):
                raise
            log(f'the adaptive partition does not carry to this declaration '
                f'({exc}); it selects its own')
            return surface_of(spec, excitation, mol, self.factory, mf=mf)

    def started_at(self, mol):
        """The manifold of the relaxed chain built AT the far start `mol`,
        which is its own reference: the relaxed declaration on its own mean
        field there, the window, the adaptive selection and every convention
        resolved at `mol` by one evaluation there, nothing taken from R0's
        chain. The chain's record is kept in 's1_start_chain', from which a
        later launch's T1 rebuild starts."""
        spec = self.relaxed_spec
        mf = reference_mean_field(mol, self.factory, spec.environment)
        man = self.manifold(self.s_key, self.t_key, chain=driven_chain(
            self.excited_at(spec, mol, mf)))
        man.evaluate(mol, mf)
        self.write('s1_start_chain', {
            'start_bohr': np.array(mol.atom_coords(), float),
            'surface': surface_record(man.driven)})
        return man

    def start_chain(self, mol):
        """The chain the far start built at `mol` (`started_at`), rebuilt
        from 's1_start_chain' (one SCF at the start, nothing derived again),
        or built there again where the record is missing or of another
        start."""
        if self.store.has('s1_start_chain'):
            held = self.read('s1_start_chain')
            if np.array_equal(held['start_bohr'], mol.atom_coords()):
                log(f'the S1 start\'s chain is rebuilt from '
                    f'{self.store.path("s1_start_chain")}')
                return self.restored(held['surface'])
        log('WARNING s1_start_chain is missing or of another start; the S1 '
            'start\'s chain is built there again')
        return self.started_at(mol).driven

    def selected_partition(self):
        """The adaptive partition this run selected at R0, in this launch or
        an earlier one (the vert record), or None."""
        if self.partition is None and self.store.has('vert'):
            stored = self.read('vert').get('qp_partition')
            if stored is not None:
                self.partition = AdaptivePartition.from_record(stored)
        return self.partition

    def posteriori(self, man, ev, key):
        """The adaptive set's check at a walk's minimum for the walked `key`,
        or None for another set."""
        chain_ = man.driven
        if chain_.resolved.qp_partition is None:
            return None
        return chain_.posteriori_check(ev.mol, ev.mf, ev.spectrum, (key,))

    def manifold(self, *keys, chain=None):
        """A fresh manifold over the reported roots and `keys`, its own
        root-following history, on `chain`'s conventions: by default the
        frozen relaxed chain's, R0's."""
        if chain is None:
            if self.frozen is None:
                self.reference()
            chain = self.frozen
        return StateManifold(chain, states=self.manifold_states(*keys))

    def rebuilt_at(self, mol, qp_floor=None):
        """R0's relaxed chain with its conventions rebuilt at `mol`, the
        adaptive partition selected again there out of R0's candidates,
        grow-only from R0's explicit set and `qp_floor` (the explicit set of
        the surface the S1 walk ended on). R0 is not evaluated for it."""
        chain_ = self.frozen
        if chain_ is None:
            spec = self.relaxed_spec
            mf0 = reference_mean_field(self.mol0, self.factory,
                                       spec.environment)
            chain_ = driven_chain(self.excited_at(
                spec, self.mol0, mf0, self.selected_partition()))
        if qp_floor is None:
            return chain_.refreeze(mol)
        return chain_.refreeze(mol, qp_floor=qp_floor)

    def rebuilt_for_t1(self, mol, qp_floor):
        """The chain a T1 walk in a later launch than the S1 walk starts on:
        the chain the S1 walk started from rebuilt at the S1 minimum `mol`,
        grow-only from `qp_floor` -- R0's (`rebuilt_at`), or with a far start
        the chain built at the start (`start_chain`)."""
        if self.declaration.policies.far_start is None:
            return self.rebuilt_at(mol, qp_floor)
        return self.start_chain(self.start_molecule()).refreeze(
            mol, qp_floor=qp_floor)

    def start_molecule(self):
        """The far start's geometry, in the atom order of the input."""
        far = self.declaration.policies.far_start
        start = gto.M(atom=far.atom, basis=self.declaration.system.basis,
                      verbose=0)
        if start.natm != self.mol_in.natm:
            raise StageRefused(f'the far start has {start.natm} atoms, the '
                               f'system {self.mol_in.natm}')
        return self.molecule_at(start.atom_coords())

    def flag_unrelaxed(self, stage, info):
        """Warn about a walk whose last pass did not converge or whose
        refreeze drift stayed above REFREEZE_TOL_MEV, and name it in
        `info['numerical_flags']` (`relaxation_flags`), which the stage
        record carries; the stage goes on where the walk stopped
        (refuse inputs, warn numerics)."""
        flags = relaxation_flags(stage, info)
        info['numerical_flags'] = dict(info.get('numerical_flags') or {},
                                       **flags)
        return flags

    def walk(self, surface, mol, stage=None, hess_init=None, resume=None):
        """(minimum, info, the root the walk followed there, the surface the
        last walk ran on), refrozen at the minimum (`refreeze_pass`); a
        refused refreeze comes back as its record, `refreeze` 'refused',
        which `relaxation_flags` flags.

        Every accepted step is stored as '<stage>_progress', and `resume`,
        such a state read back, is where the walk goes on from; geomeTRIC's
        files go to geometric_<stage>/. A walk with no `stage` keeps no
        checkpoint."""
        p = self.declaration.policies
        given = {} if hess_init is None else {'hess_init': hess_init}
        if stage is not None and p.engine != 'cartesian':
            workdir = self.store.workdir(f'geometric_{stage}')
            # one geomeTRIC per rank: the others keep their files apart
            given['workdir'] = (str(workdir) if self.store.root_rank
                                else None)
        start = np.array(mol.atom_coords(), float)

        def checkpoint(state):
            self.write(f'{stage}_progress', {'start_bohr': start,
                                             'state': state})
        # the view carries the evaluation's own root-following entries
        before = len(getattr(surface, 'follow_log', None) or [])
        mol_opt, info = relax(surface, mol, engine=p.engine,
                              verbose=self.store.root_rank,
                              refreeze=int(p.refreeze),
                              nudge=p.nudge_bohr,
                              checkpoint=None if stage is None
                              else checkpoint, resume=resume, **given)
        final = info.pop('final_surface', surface)
        # one root-following entry per forward pass the first walk ran: one
        # fewer than its evaluations when the first point was handed over
        log_ = getattr(surface, 'follow_log', None) or []
        info['first_point_replayed'] = (len(log_) - before
                                        == len(info.get('history') or []) - 1)
        # the root is the last walk's, followed on its own surface at the
        # minimum it returned, not at a step it rejected after
        root = followed_root(final, mol_opt)
        return lockstep_geometry(mol_opt), info, root, final

    def progress(self, stage, start):
        """The walk state '<stage>_progress' holds for a walk from `start` on
        this declaration, or None. Another stamp refuses the run; a walk
        from another start is walked again and its progress overwritten."""
        name = f'{stage}_progress'
        if not self.store.has(name):
            return None
        held = self.read(name)
        if not np.array_equal(held['start_bohr'], start.atom_coords()):
            log(f'WARNING {name} is a walk from another start; walking again '
                f'from this one')
            return None
        return held['state']

    def restored(self, record):
        """The relaxed states' surface rebuilt from its record
        (`surface_from_record`): one SCF at its reference geometry, every
        frozen convention read back."""
        return surface_from_record(record, self.mol_in, self.factory,
                                   environment=self.relaxed_spec.environment)

    def resumable(self, stage, start):
        """(surface, state) of a walk '<stage>_progress' resumes from
        `start`, the surface rebuilt from the state's record; None."""
        state = self.progress(stage, start)
        if state is None or state['surface'] is None:
            return None
        log(f'{stage}: resuming the walk in pass {state["pass"]} after '
            f'{len(state["history"])} evaluations '
            f'({self.store.path(stage + "_progress")})')
        return self.restored(state['surface']), state

    def write_walked(self, stage, mol, info, root, final):
        """'<stage>_walked': the converged walk, its record, the root it
        followed and the surface it ended on, before anything else is done
        at its minimum, so a relaunch resumes there (`walked`)."""
        self.write(f'{stage}_walked', {
            'coords_bohr': np.array(mol.atom_coords(), float),
            'info': {k: v for k, v in info.items() if k != 'final_surface'},
            'root': int(root), 'surface': surface_record(final)})

    def walked(self, stage):
        """(minimum, info, root, surface) of '<stage>_walked', the surface
        rebuilt from its record; None where there is none."""
        name = f'{stage}_walked'
        if not self.store.has(name):
            return None
        held = self.read(name)
        if held['surface'] is None:
            return None
        log(f'{stage}: the walk is stored ({self.store.path(name)}); resuming '
            f'at its minimum')
        return (self.molecule_at(held['coords_bohr']), held['info'],
                held['root'], self.restored(held['surface']))

    def saddle_checked(self, stage, mol, info, root, final, t0):
        """(minimum, info, root, surface, record) after the saddle check:
        the state's Hessian at the walk's minimum on the surface it ended
        on, and a saddle left along its negative mode and walked again
        (`escape_saddle`). A check that does not end on a minimum (a second
        saddle, an escape walk that did not converge) is warned about and
        flagged ('saddle_check' in the escape walk's `numerical_flags`), and
        the stage goes on where the escape walk stopped; a walk that did not
        relax is not checked (its 6N forces would test a point with a
        residual force). Without the policy the walk's own, and no record."""
        if not self.declaration.policies.saddle_check:
            return mol, info, root, final, None
        name = stage.upper()
        if (info.get('numerical_flags') or {}).get('walk_not_converged'):
            log(f'{stage}: no saddle check, the walk did not relax')
            return mol, info, root, final, {
                'status': 'not run: the walk did not relax'}
        log(f'{stage}: {name} Hessian at its minimum, {6 * mol.natm} forces')
        walked_ = {}

        def walk(surface, start):
            escape = f'{stage}_escape'
            held = self.resumable(escape, start)
            if held is not None:
                surface = held[0]
            moved, again, followed, last = self.walk(
                surface, start, escape,
                resume=None if held is None else held[1])
            walked_.update(root=followed, final=last)
            log(f'{stage}: {name} Hessian at the new minimum')
            return moved, dict(again, final_surface=last)

        moved, again, record = saddle_check(
            final, mol, walk, verbose=self.store.root_rank,
            cache_dir=self.store.workdir(f'hess_{stage}'))
        first = record['first']
        log(f'{stage}: {name} lowest internal curvature '
            f'{first["lowest_curvature"]:.3e} Ha/Bohr^2 (saddle below '
            f'{first["tolerance"]:.1e}): {record["status"]}', t0)
        if again is None:
            return mol, info, root, final, record
        again.pop('final_surface', None)
        record['saddle_walk'] = walk_fields(info)
        log(f'{stage}: left the saddle along {record["direction"]:+d} x its '
            f'mode, {record["step_bohr"]} Bohr; the new minimum lies '
            f'{record.get("energy_drop", float("nan")) * HARTREE_TO_MEV:.2f}'
            f' meV below it')
        flags = relaxation_flags(f'{stage} escape', again)
        if record['status'] != 'escaped':
            # Refuse inputs, warn numerics
            flags['saddle_check'] = record['status']
            warnings.warn(f'{name}: {record["status"]} (saddle check); the '
                          f'stage carries on where the escape walk stopped '
                          f'and its record is flagged', WalkNotRelaxed,
                          stacklevel=2)
        again['numerical_flags'] = flags
        return moved, again, walked_['root'], walked_['final'], record

    def analytic_df_hessian(self):
        """pyscf's analytic Hessian of the density-fitted twin at R0, on rank
        0, broadcast. The twin's numerics are the run's, derived: J and K
        fitted, its orbital gradient converged for a derivative."""
        d = self.declaration
        twin = d.numerics.derive(
            'the analytic Hessian of a density-fitted twin at R0',
            scf={'route': 'df', 'grid_level': None,
                 'conv_tol_grad': SCF_DIFFERENTIABLE_GRAD_TOL})

        def build():
            factory = self.mean_field_factory(twin)
            mf = chain.solvated(self.env, factory)(self.mol0)
            return np.asarray(mf.Hessian().kernel())
        comm, root = current_comm(), is_root()
        with distributed(None):
            hess = build() if root else None
        return broadcast(hess, comm)

    def accept(self, stage):
        """With the DF acceptance check, the check of `stage`'s geometry
        (`df_check.check_stage`), run once on rank 0, its output kept as
        'isdf_df_check_<stage>' and its verdict written into the stage's
        record as `isdf_df_check`. A record that already holds one is not
        checked again."""
        if not self.declaration.policies.df_check or \
                not self.store.has(stage):
            return
        record = self.read(stage)
        if record.get('isdf_df_check') is not None:
            return
        t0 = time.time()
        comm = current_comm()
        if is_root():
            with distributed(None):
                try:
                    result = df_check.check_stage(self, stage)
                    check = dict(result, output=f'isdf_df_check_{stage}')
                except Exception as exc:   # the verdict, not the run, fails
                    result = None
                    check = {'verdict': 'fail', 'output': None,
                             'reason': f'{type(exc).__name__}: {exc}'}
        else:
            result = check = None
        result, check = broadcast((result, check), comm)
        if result is not None:
            self.write(f'isdf_df_check_{stage}', result)
        self.store.update(stage, self.stamp(stage), isdf_df_check=check)
        log(f'{stage}: ISDF-vs-DF acceptance {check["verdict"]}'
            + (f' ({check["reason"]})' if check.get('reason') else ''), t0)


def with_partition(spec, partition):
    """`spec` whose chain takes the frozen `partition` instead of
    selecting."""
    if partition is None:
        return spec
    return replace(spec, numerics=dict(spec.numerics, qp_partition=partition))


# ------------------------------------------------------------ properties
def geometry_properties(ev, s_key, t_key, n_states):
    """Everything a rate needs at one geometry, off ONE evaluation: the
    energies of every reported root, the oscillator strengths and transition
    dipoles, the spin-orbit table, d(S0, S), d(S, S+1), d(T, T+1) and the
    second-order derivative of <S|H_SO|T> through S+1, T+1 and the ground
    state, with the couplings' `scf_flags` (empty when clean)."""
    nocc = occupied_count(ev.mf)
    singlet, triplet = ev.spectrum['singlet'], ev.spectrum['triplet']
    f_osc, dipole = oscillator_strengths(ev.mf, ev.mol, nocc, *singlet)
    soc = spin_orbit_couplings(ev.mf, ev.mol, nocc, singlet, triplet,
                               nroots=n_states, two_electron='somf')
    s_next, t_next = following(*s_key)[1], following(*t_key)[1]
    d01, diags = analytic_ground_coupling(None, s_key, evaluation=ev)
    # the couplings' mean-field flags (above scf_force_tol, warned about and
    # carried on), copied into the record
    scf = merge_scf_flags([], diags.get('scf_flags'))
    nac = {}
    for m, k in ((s_key, s_next), (t_key, t_next)):
        nac[f'{label(m)}-{label(k)}'], diags = analytic_coupling(
            None, m, k, evaluation=ev)
        merge_scf_flags(scf, diags.get('scf_flags'))
    sv = spin_vibronic_coupling(ev, soc['h_mo'], s_key, t_key,
                                singlet_paths=(s_next,),
                                triplet_paths=(t_next,))
    merge_scf_flags(scf, sv.get('scf_flags'))
    keys = list(ev.energy)
    return {'coords_bohr': ev.mol.atom_coords(), 'e0': ev.e0,
            'energy': {label(k): float(ev.energy[k]) for k in keys},
            'omega': {label(k): float(ev.omega[k]) for k in keys},
            'root': {label(k): int(ev.root[k]) for k in keys},
            'emitter': label(s_key), 'reservoir': label(t_key),
            'f_osc': f_osc[:n_states], 'dipole': dipole[:n_states],
            'socme': soc['socme'], 'd_ground': d01, 'nac': nac,
            'soc_v0': sv['v0'], 'soc_v0_vector': sv['v0_vector'],
            'soc_dv_cart': sv['dv_cart'],
            'soc_dv_paths': sv['paths'], 'scf_flags': scf}


def report_properties(where, props):
    """The block a log reader wants at one geometry."""
    log(f'{where}: E_0 {props["e0"]:.9f} Ha')
    for name in sorted(props['omega']):
        log(f'    {name}  Omega '
            f'{props["omega"][name] * HARTREE_TO_EV:8.4f} eV  '
            f'E {props["energy"][name]:.9f} Ha  root {props["root"][name]}')
    log('    f ' + ' '.join(f'{x:.4f}' for x in props['f_osc']))
    soc = np.abs(np.asarray(props['socme'])) * HARTREE_TO_CM
    for i, row in enumerate(soc):
        log(f'    |<{"S0" if i == 0 else f"S{i}"}|H_SO|T_j>| / cm^-1 '
            + ' '.join(f'{x:9.4f}' for x in row))
    log(f'    |V({props["emitter"]},{props["reservoir"]})| '
        f'{props["soc_v0"] * HARTREE_TO_CM:.4f} cm^-1, max |dV/dR| '
        f'{np.abs(props["soc_dv_cart"]).max() * HARTREE_TO_CM:.4f} '
        f'cm^-1/Bohr; max |d_0S| {np.abs(props["d_ground"]).max():.3e}; '
        + ', '.join(f'max |d {k}| {np.abs(v).max():.3e}'
                    for k, v in props['nac'].items()))


def surface_fields(manifold, mol, span):
    """The quasiparticle treatment of the surface a walk ended on: its
    bookkeeping and, for an adaptive set, the partition its refreeze
    selected; the run's `span` and the factors' residuals there."""
    partition = manifold.driven.resolved.qp_partition
    return {'qp_bookkeeping': qp_bookkeeping(manifold.chain, mol),
            'qp_partition': (None if partition is None
                             else partition.as_record()),
            **chain.span_fields(manifold, mol, span),
            # atoms whose covariant ISDF frame turns fast at this anchor
            'frame_turning': frame_turning_record(mol)}


# ---------------------------------------------------------------- stages
def ground_minimum(run):
    """s0: the ground-state minimum R0 on the declared mean field; sets
    `run.mol0`. Returns the record."""
    d = run.declaration
    if run.reusable('s0'):
        record = run.read('s0')
        run.mol0 = run.molecule_at(record['coords_bohr'])
        log(f's0 stored: {run.store.path("s0")}')
        return record
    t0 = time.time()
    surface = MeanFieldSurface(run.mol_in, run.factory, environment=run.env)
    log(f's0: relaxing the ground state, {run.mol_in.natm} atoms, nao '
        f'{run.mol_in.nao_nr()}, {d.reference}/{run.mol_in.basis} '
        f'{d.mean_field_name}, {d.environment}')
    mol, info = relax(surface, run.mol_in, engine=d.policies.engine,
                      verbose=run.store.root_rank)
    # a walk that did not converge is flagged; R0 is where it stopped
    run.flag_unrelaxed('s0', info)
    run.mol0 = lockstep_geometry(mol)
    record = {'coords_bohr': run.mol0.atom_coords(),
              'e0_walk': info.get('energy'), 'seconds': time.time() - t0,
              **walk_fields(info)}
    run.write('s0', record)
    run.store.write_xyz('s0.xyz', run.mol0,
                        f'S0 minimum {d.reference}/{run.mol0.basis} '
                        f'{d.environment}')
    log(f's0: E {info.get("energy"):.9f} Ha, {info.get("cycles")} cycles, '
        f'residual {info.get("opt_grad_max"):.2e}', t0)
    return record


def ground_modes(run):
    """hess: the S0 Hessian at R0 and its normal modes. A saddle point at R0
    is kept as 'hess_failed' and refused: no rate is defined about one."""
    d = run.declaration
    if run.reusable('hess'):
        log('hess stored')
        return run.read('hess')
    t0 = time.time()
    route = d.policies.hessian
    info = {}
    if route == 'numerical':
        fd = run.mean_field_factory(d.hessian_numerics, warm_start=WarmStart())
        # mpi_map is the builtin map outside a distribution
        map_fn = functools.partial(mpi_map, comm=current_comm())
        log(f'hess: {6 * run.mol0.natm} displaced forces on '
            f'{d.hessian_numerics.scf.grid_level}, {n_ranks()} at a time')
        hess = numerical_hessian(run.mol0, chain.solvated(run.env, fd),
                                 map_fn=map_fn, verbose=run.store.root_rank,
                                 info=info)
    else:
        log('hess: the analytic Hessian of a density-fitted twin at R0')
        hess = run.analytic_df_hessian()
    omega, modes, masses, _ = normal_modes(None, run.mol0, hess=hess)
    n_imag = int((omega < 0).sum())
    # the saddle test the excited minima get, free here; R0 is not left
    saddle = saddle_test(hess, run.mol0.atom_coords())
    record = {'hess': np.asarray(hess), 'omega': omega, 'modes': modes,
              'masses': masses, 'route': route,
              'translation': translation_residual(hess), 'info': info,
              'saddle': saddle}
    run.write('hess_failed' if n_imag else 'hess', record)
    cm = np.sort(omega) * HARTREE_TO_CM
    log(f'hess: {route}, {len(omega)} modes, {n_imag} imaginary, lowest '
        f'{cm[:3].round(1).tolist()} cm^-1, lowest internal curvature '
        f'{saddle["lowest_curvature"]:.2e} Ha/Bohr^2, translation residual '
        f'{record["translation"]:.2e}', t0)
    if n_imag:
        raise StageRefused(
            f'{n_imag} imaginary modes at R0 (lowest internal curvature '
            f'{saddle["lowest_curvature"]:.2e} Ha/Bohr^2, mode in '
            f'hess_failed): a saddle point, and no rate is defined about one')
    return record


def vertical(run, walks_next=False):
    """vert: ONE shared evaluation at R0, the vertical spectrum and every
    coupling. With `walks_next` (the S1 walk follows in this launch from R0
    on the vertical declaration) and a zero nudge, the S1 force rides along
    as the walk's first point."""
    d = run.declaration
    shared = run.relaxed_spec is run.spec
    walks = (shared and walks_next and d.policies.far_start is None
             and not run.reusable('s1'))
    want_force = walks and d.handover
    if run.reusable('vert') and not want_force:
        run.accept('vert')
        log('vert stored')
        return run.read('vert')
    t0 = time.time()
    man, ev = run.reference(gradients=(run.s_key,) if want_force else (),
                            couplings=run.couplings, relaxed=shared)
    if walks:
        run.first['s1'] = (man, ev)
    if run.store.has('vert'):
        log('vert stored; R0 evaluated again for the S1 walk\'s first point',
            t0)
        return run.read('vert')
    props = geometry_properties(ev, run.s_key, run.t_key, d.n_states)
    record = {'props': props, 'seconds': time.time() - t0,
              'qp_states': man.chain.realization.qp_states,
              'qp_bookkeeping': qp_bookkeeping(man.chain, run.mol0),
              **chain.span_fields(man, run.mol0, run.span),
              'frame_turning': frame_turning_record(run.mol0),
              'qp_partition': (None if run.partition is None
                               else run.partition.as_record()),
              'factor_layout': chain.factor_layout(man),
              'method': GW_BSE, 'kernel': d.kernel}
    run.write('vert', record)
    report_properties('vert (R0)', props)
    log('vert done', t0)
    run.accept('vert')
    return run.read('vert')


def stored_stage(run, stage):
    """The record of `stage` where the store holds it under this
    declaration, after its acceptance check (`Run.accept`); else None."""
    if not run.reusable(stage):
        return None
    run.accept(stage)
    log(f'{stage} stored')
    return run.read(stage)


def resume_point(run, stage, start, progress=True):
    """(walked, held) of `stage`: its stored converged walk (`Run.walked`)
    or, failing that and with `progress`, the walk state to resume from
    `start` (`Run.resumable`). Either one drops the evaluation an earlier
    stage handed over: the resumed walk does not start from it."""
    walked = run.walked(stage)
    held = None
    if walked is None and progress:
        held = run.resumable(stage, start)
    if walked is not None or held is not None:
        run.first.pop(stage, None)
    return walked, held


def walked_minimum(run, stage, surface, start, resume, t0, hess_init=None):
    """(minimum, info, root, surface) of the walk of `stage` on `surface`
    from `start`, flagged where it did not relax (`Run.flag_unrelaxed`),
    and kept as '<stage>_walked'."""
    mol, info, root, final = run.walk(surface, start, stage,
                                      hess_init=hess_init, resume=resume)
    run.flag_unrelaxed(stage, info)
    run.write_walked(stage, mol, info, root, final)
    return mol, info, root, final


def finish_minimum(run, stage, walked, t0, extra):
    """The record of `stage` ('s1', 't1') at the walk's minimum: the
    saddle check, ONE shared evaluation on the surface the last walk ran
    on, every property there, the record written and accepted.
    `extra(info)` adds the stage's own fields."""
    d = run.declaration
    mol, info, root, final, saddle = run.saddle_checked(stage, *walked, t0)
    singlet = stage == 's1'
    key = ('singlet' if singlet else 'triplet', root)
    s_key, t_key = (key, run.t_key) if singlet else (run.s_key, key)
    man = run.manifold(s_key, t_key, chain=final)
    # at the S1 minimum the T1 force starts the T1 walk
    ev = man.evaluate(mol, gradients=((run.t_key,) if singlet and d.handover
                                      else ()),
                      couplings=(following(*s_key), following(*t_key)))
    if singlet:
        run.first['t1'] = (man, ev, mol)
    props = geometry_properties(ev, s_key, t_key, d.n_states)
    record = {'props': props, 'root_followed': root,
              'solvation': d.solvation,
              'adaptive_posteriori': run.posteriori(man, ev, key),
              'factor_layout': chain.factor_layout(man),
              'e_walk': info.get('energy'), 'seconds': time.time() - t0,
              'saddle_check': saddle, **extra(info),
              'method': GW_BSE, 'kernel': d.kernel,
              **surface_fields(man, mol, run.span), **walk_fields(info)}
    run.write(stage, record)
    run.store.write_xyz(f'{stage}.xyz', mol, f'{label(key)} minimum')
    report_properties(f'{stage} minimum ({label(key)})', props)
    log(f'{stage}: {info.get("cycles")} cycles, residual '
        f'{info.get("opt_grad_max"):.2e}'
        + (f', walk E {info.get("energy")} vs evaluation '
           f'{props["energy"][label(key)]}' if singlet else '')
        + f'; {refreeze_summary(record)}', t0)
    if record['numerical_flags']:
        log(f'{stage}: FLAGGED {record["numerical_flags"]}')
    run.accept(stage)
    return run.read(stage)


def s1_minimum(run):
    """s1: the S1 walk from R0 (or from the far start, on a chain built
    there) and one shared evaluation at its minimum."""
    d = run.declaration
    far = d.policies.far_start
    stored = stored_stage(run, 's1')
    if stored is not None:
        return stored
    t0 = time.time()
    start, where = run.mol0, 'R0'
    if far is not None:
        start = run.start_molecule()
        where = f'the far start (sha256 {far.sha256()[:12]})'
    walked, held = resume_point(run, 's1', start)
    if walked is None:
        if held is not None:
            surface, resume = held
        elif far is not None:
            # the start is its own reference: a chain built there
            resume = None
            run.first.pop('s1', None)
            surface = run.started_at(start).surface(run.s_key)
        else:
            resume = None
            if 's1' not in run.first:
                run.first['s1'] = run.reference(
                    gradients=(run.s_key,) if d.handover else ())
            man, ev = run.first.pop('s1')
            surface = man.surface(run.s_key, first_point=(
                ev if d.handover else None))
        log(f's1: relaxing {label(run.s_key)} from {where} on '
            f'{surface.label()}'
            + ('' if d.handover
               else f', start nudged by {d.policies.nudge_bohr} Bohr'))
        walked = walked_minimum(run, 's1', surface, start, resume, t0)

    def extra(info):
        return {'hessian': info.get('hessian'),
                'far_start': (None if far is None else
                              {'atom': far.atom, 'sha256': far.sha256()}),
                # which reference the walked chain had; a later launch's T1
                # rebuild starts from the same one
                'chain_reference': ({'built_at': 'R0'} if far is None else
                                    {'built_at': 'start',
                                     'sha256': far.sha256(),
                                     'record': 's1_start_chain'})}
    return finish_minimum(run, 's1', walked, t0, extra)


def t1_minimum(run):
    """t1: the T1 walk from the S1 minimum, started on the S1 walk's final
    Hessian, and one shared evaluation at its minimum."""
    d = run.declaration
    stored = stored_stage(run, 't1')
    if stored is not None:
        return stored
    if not run.reusable('s1'):
        raise StageRefused('t1 walks from the S1 minimum: run s1 first')
    t0 = time.time()
    s1 = run.read('s1')
    mol_s1 = run.molecule_at(s1['props']['coords_bohr'])
    walked, held = resume_point(run, 't1', mol_s1,
                                progress='t1' not in run.first)
    if walked is None:
        if held is not None:
            surface, resume = held
        else:
            resume = None
            if 't1' not in run.first:
                s_key = ('singlet', int(s1['root_followed']))
                # grow-only: the S1 minimum's explicit set is carried
                floor = (s1.get('qp_partition') or {}).get('explicit')
                man1 = run.manifold(s_key, run.t_key,
                                    chain=run.rebuilt_for_t1(mol_s1, floor))
                ev1 = man1.evaluate(mol_s1, gradients=(
                    (run.t_key,) if d.handover else ()))
                run.first['t1'] = (man1, ev1, mol_s1)
            man1, ev1, mol_s1 = run.first.pop('t1')
            surface = man1.surface(run.t_key,
                                   first_point=ev1 if d.handover else None)
        hess_init = s1.get('hessian')
        log(f't1: relaxing {label(run.t_key)} from the S1 minimum'
            + ('' if hess_init is None else ' on the S1 walk\'s Hessian')
            + ('' if d.handover
               else f', start nudged by {d.policies.nudge_bohr} Bohr'))
        walked = walked_minimum(run, 't1', surface, mol_s1, resume, t0,
                                hess_init=hess_init)
    return finish_minimum(run, 't1', walked, t0, lambda info: {})


# ------------------------------------------------------------ rate models
class AcceptanceCheckFailed(RuntimeWarning):
    """An ISDF-vs-DF acceptance check that failed at a stage whose record
    the rate table reads: the table is computed and flagged."""


class RateRules(NamedTuple):
    """What a rate model adds to the shared stages: its own stages (run
    before 'rate', whose records its table reads too), its table off the
    records and its log block."""
    own_stages: tuple
    table: Callable
    printer: Callable


def tadf_table(run, minima, records):
    """The TADF table (`rate_models.tadf_rate_table`)."""
    d = run.declaration
    return rate_models.tadf_rate_table(*minima, run.mol_in, d.temperature,
                                       d.lambda_env_ev, d.manifold_gaps)


def uisc_table(run, minima, records):
    """The UISC table on the vib record (`uisc.uisc_rate_table`)."""
    d = run.declaration
    return uisc.uisc_rate_table(*minima, records['vib'], run.mol_in,
                                d.temperature, d.lambda_env_ev, d.rate_model,
                                d.manifold_gaps)


def print_tadf_table(system, environment, table):
    """The TADF table's log block (`rate_models.print_rate_table`)."""
    rate_models.print_rate_table(f'{system}, {environment}', table)


#: The rules of each rate model, by its name.
RATE_RULES = {'tadf': RateRules((), tadf_table, print_tadf_table),
              'uisc': RateRules(('vib',), uisc_table, uisc.print_uisc_table)}
#: The stages one rate model or another adds.
MODEL_STAGES = tuple(s for r in RATE_RULES.values() for s in r.own_stages)


def rates(run):
    """rate: the declared rate model's table off the stored records
    (`RATE_RULES`). Refused where a record it reads is missing. A DF
    acceptance check that failed at a checked stage is warned about and
    flagged ('isdf_df_check_failed'); the table's `numerical_flags` holds
    every stage record's flags by stage, beside the table's own."""
    d = run.declaration
    rules = RATE_RULES[d.rate_model.name]
    need = ('hess', 'vert', 's1', 't1') + rules.own_stages
    missing = [n for n in need if not run.reusable(n)]
    if missing:
        raise StageRefused(f'rate: needs {", ".join(missing)}')
    records = {n: run.read(n) for n in need}
    checks = {n: (records[n].get('isdf_df_check') or {}).get('verdict')
              for n in CHECKED_STAGES}
    failed = sorted(k for k, v in checks.items()
                    if v is not None and v != 'pass')
    if failed:
        # Refuse inputs, warn numerics
        warnings.warn(f'the ISDF-vs-DF acceptance check failed at '
                      f'{", ".join(failed)} ({checks}); see '
                      f'isdf_df_check_<stage> in {run.store.dir}. The rates '
                      f'are computed and flagged', AcceptanceCheckFailed,
                      stacklevel=2)
    minima = (GroundModes.from_record(records['hess']),
              from_gw_bse_record(records['vert'], 'R0'),
              from_gw_bse_record(records['s1'], 'S1'),
              from_gw_bse_record(records['t1'], 'T1'))
    table = rules.table(run, minima, records)
    table['isdf_df_check'] = checks
    held = dict(records, s0=run.read('s0')) if run.reusable('s0') else records
    stage_flags = {n: dict(r.get('numerical_flags') or {})
                   for n, r in held.items()}
    for n in failed:
        stage_flags[n]['isdf_df_check_failed'] = checks[n]
    flags = {n: f for n, f in stage_flags.items() if f}
    if table.get('numerical_flags'):
        flags['rate'] = table['numerical_flags']
    table['numerical_flags'] = flags
    run.write('rate', table)
    rules.printer(d.system.name, d.environment, table)
    return table


def vib(run):
    """vib: the UISC model's promoting-mode derivatives (`uisc.vib`);
    refused under a rate model that has no such stage."""
    if 'vib' not in RATE_RULES[run.declaration.rate_model.name].own_stages:
        raise StageRefused('vib is a stage of the UISC rate model; this run '
                           'declares the TADF table')
    return uisc.vib(run)


#: Each stage's function.
STAGES = {'s0': ground_minimum, 'hess': ground_modes, 'vert': vertical,
          's1': s1_minimum, 't1': t1_minimum, 'vib': vib, 'rate': rates}


def stages_of(model):
    """The stages of a run under rate model `model`, in order: the shared
    ones and the model's own (`RATE_RULES`)."""
    own = RATE_RULES[model.name].own_stages
    return tuple(s for s in STAGE_ORDER if s not in MODEL_STAGES or s in own)


def describe_run(run, stages, grid):
    """The run's header line: system, chain, ranks, SCF, policies, stages
    and store; then what the store holds."""
    d, store = run.declaration, run.store
    log(f'{d.system.name}: {run.mol_in.natm} atoms, nao '
        f'{run.mol_in.nao_nr()}, {d.reference}/{run.mol_in.basis}, '
        f'{d.mean_field_name}, factors {d.numerics.isdf.level} '
        f'({grid["grid_points_per_atom"]} points per atom), {d.environment}, '
        f'kernel {d.kernel}, {n_ranks()} '
        f'rank(s), SCF {d.numerics.scf.conv_tol:.0e}/'
        f'{d.numerics.scf.conv_tol_grad:.0e}'
        f'{" warm-started" if d.policies.scf_warm_start else ""}, Hessian '
        f'{d.policies.hessian}, relaxed states {d.solvation}, stages '
        f'{",".join(stages)}, store {store.dir}')
    log('stored: ' + (', '.join(store.names()) or 'nothing'))


def flag_scfs(run, name, stage):
    """The record of `stage(run)`, with 'scf_not_converged' in its
    `numerical_flags` where SCFs it built stopped short (each warned about
    by `chain.mean_field_factory`)."""
    since = len(run.histories)
    record = stage(run)
    short = run.unconverged_scfs(since)
    if short and name != 'rate' and run.store.has(name):
        run.flag(name, scf_not_converged=short)
        record = run.read(name)
    return record


#: The stages built on the minimum of each walked stage: what a relaunch
#: walks again with it (`walk_again`).
BUILT_ON = {'s0': STAGE_ORDER, 's1': ('s1', 't1', 'vib', 'rate'),
            't1': ('t1', 'vib', 'rate')}


def unconverged_walk(run, stage):
    """The status of the stored walk of `stage` that did not converge, its
    stage record's or its '<stage>_walked''s 'walk_not_converged' flag
    (`relaxation_flags`), read under this declaration's stamp (another
    refuses, `Store.read`); None where none is stored or it converged."""
    for name in (stage, f'{stage}_walked'):
        if not run.store.has(name):
            continue
        record = run.read(name)
        if name != stage:
            record = record.get('info') or {}
        status = (record.get('numerical_flags') or {}).get(
            'walk_not_converged')
        if status is not None:
            return status
    return None


def stage_files(store, stage):
    """The names in the store's directory that belong to `stage`: its
    record, every record inside it (walked, progress, the escape walk's),
    its acceptance check, its xyz, its saddle check's force cache and
    geomeTRIC's files; not a far start's chain."""
    out = []
    for name in store.names():
        stem = name.split('.')[0]
        if stem == f'{stage}_start_chain':
            continue
        if (stem == stage or stem.startswith(f'{stage}_')
                or stem in (f'isdf_df_check_{stage}', f'hess_{stage}')
                or stem.startswith(f'geometric_{stage}')):
            out.append(name)
    return out


def walk_again(run, stages):
    """Set aside each walked stage of `stages` whose stored walk did not
    converge, with every stage built on its minimum (`BUILT_ON`), so this
    launch walks it again from its start (module docstring). Returns
    {stage: the names set aside}."""
    out = {}
    for stage in BUILT_ON:
        if stage not in stages:
            continue
        status = unconverged_walk(run, stage)
        if status is None:
            continue
        names = [n for s in BUILT_ON[stage] for n in stage_files(run.store, s)]
        under = (f'unconverged/{stage}-'
                 f'{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}')
        out[stage] = run.store.set_aside(names, under)
        log(f'{stage}: the stored walk did not converge ({status}); walking '
            f'it again, {", ".join(out[stage])} set aside in {under}')
    return out


def run_tadf(declaration, store, *, max_memory, stages=None, dry_run=False):
    """Every asked stage of `declaration` (all of its rate model's,
    `stages_of`, when None) in order on `store`, inside the caller's
    distribution; returns {stage: record}. The s0 record is read whenever a
    later stage needs R0. `max_memory` (MB) is a resource of the launch.
    `dry_run` builds the run, makes every early refusal and returns before
    the first SCF: a job's capability check."""
    if stages is None:
        stages = stages_of(declaration.rate_model)
    unknown = set(stages) - set(STAGE_ORDER)
    if unknown:
        raise ValueError(f'stages {sorted(unknown)} not in {STAGE_ORDER}')
    run = Run(declaration, store, max_memory)
    describe_run(run, stages, run.refuse_early())
    if dry_run:
        return {}
    out = {}
    walk_again(run, stages)
    later = [s for s in stages if s != 's0']
    if 's0' in stages or (later and store.has('s0')):
        out['s0'] = flag_scfs(run, 's0', ground_minimum)
    elif later:
        raise StageRefused('every later stage needs R0: run s0 first')
    for name in STAGE_ORDER[1:]:
        if name not in stages:
            continue
        log(f'===== {name} =====')
        if name == 'vert':
            out[name] = flag_scfs(run, name, functools.partial(
                vertical, walks_next='s1' in stages))
        else:
            out[name] = flag_scfs(run, name, STAGES[name])
    log(f'{declaration.system.name}: done; SCFs built {len(run.histories)}')
    return out
