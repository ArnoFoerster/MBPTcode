"""A geometry optimization under the distribution context is ONE walk taken
on every rank, and it is the serial walk.

WHAT BREAKS IF THIS FAILS. Every evaluation the optimizer asks for is a
collective: the space-time sweeps reduce over imaginary time and frequency, the
SCF at each displaced geometry reduces over the auxiliary index and the
quadrature grid. A reduction adds the ranks' partials, so the partials have to
be partials of ONE calculation -- of one geometry. Every rank runs the
trust-region loop, and ranks that each stepped on their own numbers would agree
on one machine, where two threads of one process compute bitwise the same
thing, and diverge across nodes, where the mean-field force a surface adds on
its own is not bit-identical: one rank then accepts a step its neighbour
rejected, and the next reduction adds terms of two different molecules.
`surface.evaluate` closes that at the boundary -- the geometry is locked to
rank 0's on the way in, the energy, force and diagnostics on the way out -- so
every rank takes every step from rank 0's numbers, and the walk's result is
locked once more at the end. These gates hold both halves: every decision is
taken from rank 0's numbers on every rank, the record is one record, and what
comes back is the serial answer.

WHY 1e-6 BOHR IS THE GATE, and where it comes from rather than being tuned.
The distributed surface reproduces the serial gradient to the ISDF gradient
reproducibility floor, 1e-8 Ha/Bohr (tests/test_simulated_ranks.py,
tests/test_surface_comm.py), and the distributed SCF adds nothing the
Lagrangian amplifies past it -- 2.95e-09 Ha/Bohr at two ranks
(tests/test_chain_distributed_scf.py). A minimum moves by dx = H^-1 dg under a
force error dg, and the softest INTERNAL curvature of this molecule is its
bend, 0.196 Ha/Bohr^2 (the analytic Hessian at cc-pVDZ Hartree-Fock, rigid-body
modes dropped), so 1e-8 Ha/Bohr propagates to 5e-8 Bohr. 1e-6 Bohr is twenty
times that and three orders BELOW `GEOM_OPT_CONV['step_max']` = 1.8e-3 Bohr,
the radius inside which either walk is converged at all: a rank that took a
different step moves the geometry by at least a step, which is a thousand
times the gate. MEASURED here, distributed against the serial walk:

    mean-field ground state   size=2  1.46e-13 Bohr   size=3  5.20e-13 Bohr
    the same, through geomeTRIC       size=2  4.08e-13 Bohr
    BSE@GW, one step          size=2  1.98e-09 Bohr
    adiabatic, one step each  size=2  1.00e-09 Bohr (state), 0.0 (ground)

with the cycle count equal to the serial one in every case and the geometry,
the energy and the whole record (every history entry) EQUAL between the ranks
of one run.

1e-6 BOHR IS THE FLOOR OF AN ANCHORED BAR. Every step of either walk runs SCFs
and forces a threaded pyscf does not repeat bit for bit, so the serial walk is
taken twice and the distributed geometry is gated at `COMPOSED_GRAD_K` times
what the repeat moved it, 1e-6 Bohr at the least (`serial_walk`); here the
repeat moves nothing. The cycle counts stay exact: the ranks walk rank 0's
walk (`rank_zero_walk`), whose forces sit within the anchored force bar of
the serial ones, which moves a step by far less than the radius a walk
converges inside, so a different count is a different walk.

THE GATE CAN FAIL, shown two ways rather than argued. The probe puts a defect
into ONE rank's force inside the evaluation: on rank 1 it must change nothing,
because the boundary's lockstep replaces that rank's numbers with rank 0's
before any decision is taken from them, and on rank 0 it must move the answer
far past the gate. Measured, with 1e-3 Ha/Bohr added to one Cartesian
component -- five orders above the 1e-8 floor the gate is derived from:

    on rank 0   1.42e-03 Bohr from the serial minimum, 1400x the gate
    on rank 1   1.46e-13 Bohr, bitwise the unperturbed distributed answer

With the boundary's result lockstep removed the rank-1 defect is no longer
harmless: rank 1 walks on its own force, leaves the walk at a different point,
and the next collective refuses on every rank (`lockstep` meeting a dict on
rank 1 where rank 0 holds a tuple).

A RESCALING IS NOT THE PROBE, and that is a property of the optimizer rather
than of the gate: the zero of alpha g is the zero of g, and a BFGS Hessian
built from alpha dg scales with it, so a quasi-Newton walk is very nearly
invariant under a constant force error. Measured, rank 0's force scaled by
1 + 1e-3: the minimum moved 3.0e-07 Bohr, INSIDE the 1e-6 gate, and the cycle
count did not change at all. The defect a per-rank divergence produces is not
a rescaling -- it is one rank stepping somewhere else -- so the control adds a
force instead of multiplying one.

ONE MOLE PER RANK, for the reason tests/test_simulated_ranks.py gives at
length: the nuclear gradient evaluates the nuclear-attraction derivative inside
`mol.with_rinv_at_nucleus`, an in-place write to the shared `mol._env`, and two
rank THREADS sharing a Mole overwrite each other's rinv origin.

THE GEOMETRIC WALK RUNS ON EVERY RANK TOO, and its files go under an absolute
prefix: the working directory belongs to the process, which the rank threads
share, so a walk that changed into its own directory would move the others'.
Its final geometry is geomeTRIC's own arithmetic rather than an evaluated
point, so it is the one walk whose geometry lock in `rank_zero_walk` is not
already implied by `surface.evaluate`; rank 1's is moved by 1e-12 relative
here, and every rank must still return rank 0's. SHOWN TO FAIL with the
geometry lockstep removed from `rank_zero_walk`: rank 1 returns its own.

Every check ASSERTS: pytest discards a returned verdict and passes on False.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import COMPOSED_GRAD_K, GEOM_OPT_CONV
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.utils.mpi_grid import current_comm, run_simulated
from src.properties.excitations import (SurfaceSpec,
                                        calc_adiabatic_excitation)
from src.properties.optimize import optimize
from src.properties.surfaces import potential_energy_surface

BASIS = 'cc-pvdz'
#: `systems.py`'s water, the fastest end-to-end case in the suite.
H2O = 'O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692'
SIZES = [2, 3]
#: The least of how far a distributed walk may end from the serial one, in
#: Bohr; the module docstring derives it from the 1e-8 Ha/Bohr gradient floor.
GEOMETRY_FLOOR = 1e-6
#: What the negative control adds to one Cartesian component of one rank's
#: force, in Ha/Bohr. Five orders above the 1e-8 reproducibility floor the
#: geometry gate is derived from, and the smallest kind of defect that can
#: move a minimum at all: a rescaling cannot (see the module docstring).
FORCE_OFFSET = 1e-3
#: The first trust radius of a ground-state walk, the one `relax_ground_state`
#: uses: the excited-state default of 0.1 Bohr is for a fragile surface.
GROUND_TRUST = 0.3
#: The optimizer MODULE; `src.properties` re-exports its `optimize` function
#: under the same name, which shadows the submodule as a package attribute.
OPTIMIZER = importlib.import_module('src.properties.optimize')


def built(mol):
    """A density-fitted mean field BUILT AND NOT RUN: the surface converges it.

    The factory a distributed run hands its surface, so that the SCF at every
    geometry is divided over the ranks (`converged_factory`); what is gated
    here is the path a cluster run takes. Tolerances for gradient work, since
    the Lagrangian assumes the occupied-virtual Fock block vanishes.
    """
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    return mf


def converged(mol):
    """The same mean field, converged by the factory itself -- the serial arm."""
    mf = built(mol)
    mf.kernel()
    return mf


def own_water():
    """This rank's OWN Mole; see the module docstring on `mol._env`."""
    return gto.M(atom=H2O, basis=BASIS, verbose=0)


class OffsetForce:
    """`surface` with ONE rank's force wrong: the probe that says the gate can fail.

    It wraps the surface rather than patching a chain, so the defect enters
    exactly where a real per-rank divergence would -- in what the collective
    evaluation hands back to the optimizer ON THAT RANK, after every reduction
    has already agreed. One Cartesian component is enough and a component
    rather than the whole vector is deliberate: a uniform force is a
    translation and the optimizer projects those out.
    """

    def __init__(self, surface, rank, offset=FORCE_OFFSET):
        self.surface, self.rank, self.offset = surface, rank, offset
        self.mol0 = surface.mol0

    def total_gradient(self, mol=None, mf=None):
        grad, energy, diags = self.surface.total_gradient(mol, mf)
        if current_comm().Get_rank() == self.rank:
            grad = np.asarray(grad).copy()
            grad[0, 2] += self.offset
        return grad, energy, diags

    def total_energy(self, mol=None, mf=None):
        return self.surface.total_energy(mol, mf)

    def mean_field(self, mol=None, mf=None):
        return self.surface.mean_field(mol, mf)

    def refreeze(self, mol):
        return OffsetForce(self.surface.refreeze(mol), self.rank, self.offset)

    def label(self):
        return f'{self.surface.label()} [rank {self.rank} force + {self.offset}]'


def mean_field_surface(factory):
    """E_KS as a surface, on this rank's own water. Its one stage is the SCF."""
    mol = own_water()
    return potential_energy_surface(
        mol, factory, ground_state=GroundState('dft', 'hf')), mol


def bse_surface(factory):
    """E_HF + E_c^dRPA + Omega_S1 on the cubic ISDF/space-time route."""
    mol = own_water()
    return potential_energy_surface(
        mol, factory, ground_state=GroundState('rpa', 'hf'),
        excitation=Excitation('singlet')), mol


def walked(surface, mol, **kw):
    """What one walk is compared on: the geometry, the record's decisions,
    and the whole record."""
    mol_opt, info = optimize(surface, mol, verbose=False, **kw)
    return {'coords': mol_opt.atom_coords(), 'cycles': info['cycles'],
            'converged': info['converged'], 'energy': info['energy'],
            'rejected': info['rejected'], 'status': info['status'],
            'refreeze_shift': info['refreeze_shift'], 'info': info}


def ground_walk(factory, **kw):
    """A full relaxation of the mean-field ground state, on every rank of the
    context it runs in."""
    surface, mol = mean_field_surface(factory)
    return walked(surface, mol, trust=GROUND_TRUST, **kw)


def serial_walk(run, keys=('coords',)):
    """`run()` twice outside the context: the serial reference, and the bar a
    distributed walk's geometries `keys` are gated at -- `COMPOSED_GRAD_K`
    times what the repeat moved them, `GEOMETRY_FLOOR` at the least."""
    reference, again = run(), run()
    spread = max(np.abs(again[k] - reference[k]).max() for k in keys)
    return reference, max(GEOMETRY_FLOOR, COMPOSED_GRAD_K * spread)


def assert_same_walk(reference, results, bar):
    """Every rank took the serial walk -- the same decisions, the geometry
    within `bar` -- and the ranks of one run hold one record: the geometry
    bitwise, and every entry of the history."""
    for rank in results:
        # exact: the ranks walk rank 0's walk (`rank_zero_walk`)
        assert rank['cycles'] == reference['cycles']
        assert rank['converged'] == reference['converged']
        assert rank['status'] == reference['status']
        assert rank['rejected'] == reference['rejected']
        moved = np.abs(rank['coords'] - reference['coords']).max()
        assert moved < bar
    for rank in results[1:]:
        assert np.array_equal(rank['coords'], results[0]['coords'])
        assert rank['energy'] == results[0]['energy']
        assert rank['info'] == results[0]['info']


@pytest.mark.parametrize('size', SIZES)
def test_the_mean_field_relaxation_is_the_serial_relaxation(size):
    """The cheapest surface that distributes: its only stage is the SCF.

    Nothing here is post-SCF, so what the ranks divide is the mean field at
    every geometry the optimizer visits -- and the walk is still one walk.
    """
    reference, bar = serial_walk(lambda: ground_walk(converged))
    results = run_simulated(lambda comm: ground_walk(built), size)

    assert reference['converged'] and reference['cycles'] > 1
    assert_same_walk(reference, results, bar)


@pytest.mark.parametrize('size', SIZES)
def test_the_refreeze_loop_runs_over_the_ranks(size):
    """The outer loop runs on every rank too.

    `surface.refreeze` on a chain rebuilds the grid on every rank and locks it
    to rank 0's, so every rank must refreeze at the same point of the same
    walk. The mean-field surface refreezes to itself, so the measured drift is
    zero and what is gated is that the second walk happened at all, on every
    rank, and ended where the serial one did.
    """
    reference, bar = serial_walk(lambda: ground_walk(converged, refreeze=1))
    results = run_simulated(lambda comm: ground_walk(built, refreeze=1), size)

    assert reference['refreeze_shift'] == pytest.approx(
        0.0, abs=GEOM_OPT_CONV['step_max'])
    for rank in results:
        assert rank['refreeze_shift'] is not None
    assert_same_walk(reference, results, bar)


def test_one_bse_at_gw_step_is_the_serial_step():
    """One optimizer step on the full cubic surface, which is two chains' sweeps.

    `max_cycle=1` is one accepted step and two gradients -- the driving force
    at R0 and the force where it lands -- which is all a BSE@GW walk needs to
    show that the geometry the ranks stepped to is rank 0's.
    """
    reference, bar = serial_walk(
        lambda: walked(*bse_surface(converged), max_cycle=1))
    results = run_simulated(
        lambda comm: walked(*bse_surface(built), max_cycle=1), 2)

    assert reference['cycles'] == 1
    assert_same_walk(reference, results, bar)


def test_the_adiabatic_entry_point_relaxes_both_surfaces_over_the_ranks():
    """The driver's own path: two relaxations from one declaration, on every
    rank of the context.

    `calc_adiabatic_excitation` relaxes the state and the ground state and
    differences the two minima, and both halves run on every rank from the one
    spec. Declared `dft`, which is the production row: the state is
    `ExcitedStateChain` and the ground state is the mean field's own energy,
    whose only stage is the SCF.

    `max_cycle=1` is one step on each surface, which is what makes this
    affordable; the walks themselves are gated above.
    """
    def one_run(factory):
        mol = own_water()
        spec = SurfaceSpec(GroundState('dft', 'hf'),
                           qp_states=QPStates('frontier'))
        record = calc_adiabatic_excitation(
            spec, Excitation('singlet'), mol, factory, refreeze=0,
            engine='cartesian', max_cycle=1, verbose=False)
        return {'adiabatic_eV': record['adiabatic_eV'],
                'cycles': record['cycles'],
                'ground_cycles': record['ground_cycles'],
                'excited': record['mol_excited_minimum'].atom_coords(),
                'ground': record['mol_ground_minimum'].atom_coords()}

    reference, bar = serial_walk(lambda: one_run(converged),
                                 keys=('excited', 'ground'))
    results = run_simulated(lambda comm: one_run(built), 2)

    for rank in results:
        assert rank['cycles'] == reference['cycles'] == 1
        assert rank['ground_cycles'] == reference['ground_cycles']
        for name in ('excited', 'ground'):
            assert np.abs(rank[name] - reference[name]).max() < bar
    for rank in results[1:]:
        assert rank['adiabatic_eV'] == results[0]['adiabatic_eV']
        assert np.array_equal(rank['excited'], results[0]['excited'])
        assert np.array_equal(rank['ground'], results[0]['ground'])


def test_a_refusal_on_the_driver_reaches_every_rank():
    """What rank 0 raises, every rank raises -- it is not a deadlock.

    The two spellings of the force threshold given together and disagreeing is
    refused inside the walk, and every rank runs the walk, so every rank meets
    the refusal at the same point with nobody left waiting in a collective --
    on a cluster a rank that raised alone would leave the others burning the
    job's wall clock with nothing in the log.
    """
    def one_rank(comm):
        surface, mol = mean_field_surface(built)
        with pytest.raises(ValueError, match='force threshold'):
            optimize(surface, mol, verbose=False,
                     conv={'grad_max': 1e-4, 'opt_grad_max': 1e-5})
        return 'refused'

    assert run_simulated(one_rank, 2) == ['refused', 'refused']


def offset_walk(rank):
    """A ground-state walk with `rank`'s own force wrong by FORCE_OFFSET."""
    surface, mol = mean_field_surface(built)
    return walked(OffsetForce(surface, rank=rank), mol, trust=GROUND_TRUST)


def test_a_worker_ranks_own_force_decides_nothing():
    """Rank 1's force wrong by 1e-3 Ha/Bohr: the answer must not move at all.

    The boundary's property stated as a measurement. `surface.evaluate` hands
    every rank rank 0's force, so a defect in what the surface returns ON
    RANK 1 never reaches rank 1's optimizer -- and the two ranks must come
    back holding one geometry and one record.
    """
    clean = run_simulated(lambda comm: ground_walk(built), 2)
    results = run_simulated(lambda comm: offset_walk(rank=1), 2)

    for rank, reference in zip(results, clean):
        assert rank['cycles'] == reference['cycles']
        assert np.array_equal(rank['coords'], reference['coords'])
        assert rank['energy'] == reference['energy']
        assert rank['info'] == reference['info']
    assert np.array_equal(results[0]['coords'], results[1]['coords'])


def test_the_same_defect_on_rank_zero_moves_the_answer():
    """...and on the rank that DOES decide it walks past the gate.

    Without this the check above says only that something was ignored. Rank 0's
    force is what every step is taken from, so a defect in it ends the walk
    somewhere else, and the 1e-6 Bohr gate the tests above pass is crossed by
    three orders -- which is what makes those a check rather than a tautology.
    """
    reference = ground_walk(converged)
    results = run_simulated(lambda comm: offset_walk(rank=0), 2)

    moved = [np.abs(rank['coords'] - reference['coords']).max()
             for rank in results]
    assert min(moved) > 100 * GEOMETRY_FLOOR
    # Still ONE answer: every rank returns the geometry rank 0 walked to,
    # wrong though it now is, rather than each returning its own.
    assert np.array_equal(results[0]['coords'], results[1]['coords'])


def geometric_walk(factory):
    """A full geomeTRIC relaxation of the mean-field ground state, on every
    rank of the context it runs in, each in a fresh directory of its own."""
    surface, mol = mean_field_surface(factory)
    mol_opt, info = OPTIMIZER.optimize_geometric(surface, mol, verbose=False)
    return {'coords': mol_opt.atom_coords(), 'cycles': info['cycles'],
            'converged': info['converged'], 'energy': info['energy'],
            'rejected': 0, 'status': info['status'],
            'refreeze_shift': info['refreeze_shift'], 'info': info}


def test_the_geometric_walk_is_the_serial_walk(monkeypatch):
    """geomeTRIC on every rank, its last geometry drifted on rank 1: one
    record, the serial one, and rank 0's geometry on every rank."""
    pytest.importorskip('geometric')
    real = OPTIMIZER.geometric_engine

    def drifted_engine():
        Engine, GeoMolecule, run_optimizer = real()

        def run(**kw):
            out = run_optimizer(**kw)
            comm = current_comm()
            if comm is not None and comm.Get_rank() == 1:
                out.xyzs[-1] = np.asarray(out.xyzs[-1]) * (1.0 + 1e-12)
            return out
        return Engine, GeoMolecule, run

    reference, bar = serial_walk(lambda: geometric_walk(converged))
    monkeypatch.setattr(OPTIMIZER, 'geometric_engine', drifted_engine)
    results = run_simulated(lambda comm: geometric_walk(built), 2)

    assert reference['converged'] and reference['cycles'] > 1
    assert_same_walk(reference, results, bar)


def test_a_rank_whose_step_drifted_returns_rank_zeros_record(monkeypatch):
    """Rank 1's RFO step one ulp off in every component: one record anyway.

    The optimizer's own arithmetic is serial code on every rank, and across
    nodes an `eigh` of the augmented RFO matrix need not be bit-identical.
    The evaluations are rank 0's regardless (`surface.evaluate` locks the
    geometry), and `rank_zero_walk` locks the geometry and the record the
    walk returns, so rank 1's own `step_max` entries never reach its caller.
    SHOWN TO FAIL with the record's lockstep removed from `rank_zero_walk`:
    rank 1's history then carries its own step lengths.
    """
    real = OPTIMIZER._rfo_step

    def drifted(hess, grad, trust, proj):
        step = real(hess, grad, trust, proj)
        comm = current_comm()
        if comm is not None and comm.Get_rank() == 1:
            step = np.nextafter(step, np.inf)
        return step

    monkeypatch.setattr(OPTIMIZER, '_rfo_step', drifted)
    reference, bar = serial_walk(lambda: ground_walk(converged))
    results = run_simulated(lambda comm: ground_walk(built), 2)
    assert_same_walk(reference, results, bar)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
