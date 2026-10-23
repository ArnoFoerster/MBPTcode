"""The solvated mean field converged over the ranks.

`SolventScreening.mean_field` wraps the factory's mean field in PCM(eps_static)
and re-converges it through `distributed_mean_field`: every rank runs pyscf's
loop against the distributed J/K and quadrature the factory's SCF left on the
mean field (DF rows, or the ISDF-K tiles of a `DistributedISDFJK`), and the
reaction field -- pyscf's `with_solvent.kernel`, which `get_veff` only TAGS on
the J/K and `get_fock` adds -- is rank 0's on every rank inside
`distributed_fock`.

Gates, water/cc-pVDZ PBE0 in toluene (eps_static 2.38):
  * every rank holds the same bits (energy, density, orbitals, spectrum), and
    at one rank the serial ones;
  * against serial: the energy within RANK_SPLIT_ENERGY_TOL, and the density
    and orbital energies within 2 |g| / gap of each other -- each SCF's
    density sits within |g| / gap of the stationary point it converges to;
    the J/K reductions over ranks are re-associated, so the two loops stop at
    different points inside that ball;
  * a rank whose reaction field differs from rank 0's ends on rank 0's mean
    field: the lock moves it;
  * the Newton finish and the warm start run on the solvated mean field over
    ranks;
  * the production chain's S1 force in PCM at 2 and 3 ranks is the serial
    chain's on the same mean field to
    RANK_SPLIT_FORCE_TOL.
"""
import warnings

import numpy as np
import pytest
from pyscf import dft, gto
from pyscf import solvent as pyscf_solvent

from tests.pins import assert_same
from src.Base.numerics.constants import (RANK_SPLIT_ENERGY_TOL, RANK_SPLIT_FORCE_TOL,
                                         SCF_DIFFERENTIABLE_CONV_TOL,
                                         SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.factorization.distributed_df import distributed_mean_field
from src.gradients.isdf_mean_field import isdf_jk_with_force
from src.Base.numerics.numerics import numerics_of
from src.Base.pyscf.scf_convergence import WarmStart, newton_finished, warm_started
from src.Base.factorization.separable_ri import resolve_isdf_grid
from src.environments.solvent_screening import SolventScreening
from src.Base.utils.parallel.mpi_grid import current_comm, distributed
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients.reference_mean_field import converged_factory
from src.gradients.gwbse.surface_factory import potential_energy_surface

WATER = 'O 0.0 0.0 0.1173; H 0.03 0.7572 -0.4692; H -0.02 -0.7472 -0.4492'
COUNTS, N_START = resolve_isdf_grid('G1', 'cc-pvdz', ['H', 'O'],
                                    auxbasis='cc-pvdz-ri')


def water(shift=0.0):
    """Water, asymmetric, cc-pVDZ; the oxygen moved by `shift` Angstrom."""
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    if not shift:
        return mol
    coords = mol.atom_coords(unit='Angstrom')
    coords[0, 1] += shift
    return gto.M(atom=[(mol.atom_symbol(i), coords[i])
                       for i in range(mol.natm)], basis='cc-pvdz', verbose=0)


def built(mol, route):
    """A PBE0 mean field on `route` ('df' or 'isdf'), built and not run, at
    the differentiable tolerances; serially the factory converges it."""
    mf = dft.RKS(mol, xc='pbe0')
    mf = (mf.density_fit(auxbasis='cc-pvdz-ri') if route == 'df'
          else isdf_jk_with_force(mf, auxbasis='cc-pvdz-ri', counts=COUNTS,
                       n_start=N_START))
    mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    return mf


def solvated(mol, factory):
    """The factory's mean field in PCM(toluene), converged in the current
    region's ranks."""
    warnings.simplefilter('ignore')
    env = SolventScreening(mol, solvent='toluene')
    return env.mean_field(mol, converged_factory(factory))


def arrays(mf):
    """What a mean field hands downstream."""
    return dict(e=mf.e_tot, dm=mf.make_rdm1(), mo_energy=mf.mo_energy,
                mo_coeff=mf.mo_coeff, converged=mf.converged)


def same_bits(a, b):
    assert_same(a['e'], b['e'])
    for key, unit in (('dm', '1'), ('mo_energy', 'Ha'), ('mo_coeff', '1')):
        assert_same(a[key], b[key], unit=unit, label=key)


def converged_bound(mf):
    """2 |g| / gap: how far two SCFs, each at |g| < conv_tol_grad, may sit
    apart in density and orbital energies."""
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    gap = mf.mo_energy[nocc] - mf.mo_energy[nocc - 1]
    return 2 * mf.conv_tol_grad / gap


def same_mean_field(ranked, serial, bound):
    assert abs(ranked['e'] - serial['e']) < RANK_SPLIT_ENERGY_TOL
    assert np.abs(ranked['dm'] - serial['dm']).max() < bound
    assert np.abs(ranked['mo_energy'] - serial['mo_energy']).max() < bound


@pytest.mark.parametrize('route', ['df', 'isdf'])
@pytest.mark.parametrize('size', [1, 2, 3])
def test_solvated_mean_field_over_ranks(route, size):
    """The PCM mean field at 1, 2 and 3 simulated ranks: the ranks agree bit
    for bit, one rank is serial bit for bit, and more are the serial mean
    field to the convergence both ran to."""
    with distributed(None):
        reference = solvated(water(), lambda mol: built(mol, route).run())
    assert hasattr(reference, 'with_solvent') and reference.converged
    serial = arrays(reference)

    def one_rank(comm):
        mf = solvated(water(), lambda mol: built(mol, route))
        assert hasattr(mf, 'with_solvent')
        return arrays(mf)

    out = run_simulated(one_rank, size)
    for rank in out:
        assert rank['converged']
        same_bits(rank, out[0])
    if size == 1:
        same_bits(out[0], serial)
    same_mean_field(out[0], serial, converged_bound(reference))


@pytest.mark.parametrize('size', [2, 3])
def test_reaction_field_is_rank_zeros(size):
    """A rank whose reaction field is off rank 0's by 1e-10 in every element
    ends on rank 0's mean field bit for bit: inside the distributed Fock build
    every rank adds rank 0's (e_solvent, v_solvent)."""
    def one_rank(comm):
        warnings.simplefilter('ignore')
        mf = pyscf_solvent.PCM(built(water(), 'isdf'))
        mf.with_solvent.eps = 2.38
        own = mf.with_solvent.kernel
        if comm.Get_rank() > 0:
            def perturbed(dm):
                e, v = own(dm)
                return e + 1e-10, v + 1e-10
            mf.with_solvent.kernel = perturbed
        distributed_mean_field(mf)
        # the rank's own field, outside the block, at the converged density
        moved = (float(np.abs(mf.with_solvent.kernel(mf.make_rdm1())[1]
                              - own(mf.make_rdm1())[1]).max())
                 if comm.Get_rank() > 0 else 0.0)
        return arrays(mf), moved, 'kernel' in vars(mf.with_solvent)

    out = run_simulated(one_rank, size)
    for rank, (result, moved, patched) in enumerate(out):
        same_bits(result, out[0][0])
        if rank > 0:
            # the perturbation is real, and the lock is gone after the block
            assert moved == pytest.approx(1e-10, rel=1e-3)
            assert patched
        else:
            assert not patched


@pytest.mark.parametrize('size', [2, 3])
def test_newton_finish_and_warm_start_in_solvent(size):
    """--scf-finish newton and --scf-warm-start on the solvated mean field
    over ranks: two geometries of a walk, each finished by Newton steps to
    conv_tol_grad, the second started from the first's density; every rank
    takes the same steps and products, and both mean fields are the serial
    ones to the convergence both ran to."""
    def factory(holder):
        return lambda mol: warm_started(newton_finished(built(mol, 'isdf')),
                                        holder)

    def walk():
        holder = WarmStart()
        out = []
        for shift in (0.0, 0.05):
            mf = solvated(water(shift), factory(holder))
            out.append((arrays(mf), dict(mf._newton_finish),
                        converged_bound(mf)))
        return out, holder.used

    def serial_factory(holder):
        build = factory(holder)
        return lambda mol: build(mol).run()

    with distributed(None):
        holder = WarmStart()
        serial = [arrays(solvated(water(shift), serial_factory(holder)))
                  for shift in (0.0, 0.05)]
    out = run_simulated(lambda comm: walk(), size)
    for steps, used in out:
        assert used >= 1
        for (result, newton, _), (result0, newton0, _) in zip(steps,
                                                             out[0][0]):
            assert result['converged'] and newton['converged']
            same_bits(result, result0)
            assert newton['steps'] == newton0['steps']
            assert newton['hessian_products'] == newton0['hessian_products']
    for (result, _, bound), reference in zip(out[0][0], serial):
        same_mean_field(result, reference, bound)


@pytest.mark.parametrize('size', [2, 3])
def test_solvated_production_force_over_ranks(size):
    """The production chain (ISDF-K G1, space-time, SOP, frontier set,
    Davidson, grid adjoint) in PCM(toluene), S1 force at 2 and 3 simulated
    ranks: the ranks agree to a few ulp and the force is the serial chain's on
    the same mean field to RANK_SPLIT_FORCE_TOL. The factory leaves the SCF
    unrun under a communicator, as the production driver's does.

    The same mean field, not the serial SCF: the chain freezes its SOP poles
    from its reference mean field by vector fitting, which is not continuous
    in it, so two SCFs converged to the same tolerance freeze two pole sets
    and two consistent surfaces (2.4e-9 Ha/Bohr apart at one thread)."""
    def factory(mol):
        mf = built(mol, 'isdf')
        if current_comm() is None:
            mf.kernel()
        return mf

    def force(mf=None):
        warnings.simplefilter('ignore')
        mol = water()
        chain = potential_energy_surface(
            mol, factory, ground_state=GroundState('dft', 'pbe0'),
            excitation=Excitation('singlet', root=2, kernel='bse'),
            environment=SolventScreening(mol, solvent='toluene'),
            chi0='space-time', residues='sop', solver='davidson',
            factorization='isdf', qp_states=QPStates('frontier'),
            mf=mf, numerics=numerics_of({'grid_accuracy': 'G1',
                                         'bse_adjoint': 'grid', 'nroots': 5}))
        assert hasattr(chain.mf0, 'with_solvent')
        return np.array(chain.total_gradient()[0]), chain.mf0

    out = run_simulated(lambda comm: force(), size)
    for rank in out:
        # each rank assembles its own force: rank 0's to a few ulp
        assert_same(rank[0], out[0][0], quantity='gradient', label='rank force')
    with distributed(None):
        serial = force(out[0][1])[0]
    assert np.abs(out[0][0] - serial).max() < RANK_SPLIT_FORCE_TOL
