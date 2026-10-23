"""The solvated mean field and the solvated production force under real ranks.

    python tests/test_solvated_scf_mpi.py                 # one rank
    OMP_NUM_THREADS=1 mpirun -n 2 python tests/test_solvated_scf_mpi.py
    OMP_NUM_THREADS=1 mpirun -n 3 python tests/test_solvated_scf_mpi.py

Script, not a pytest module: it initializes MPI, which the sandboxed test
runner cannot, and it exits with a status. `main(comm)` takes the
communicator, so `run_simulated(main, n)` runs the same checks over
thread-ranks; tests/test_solvated_scf_ranks.py gates the same properties in
pytest over simulated ranks. Every rank's verdict is gathered: a check that
fails on rank 2 fails the run.

Checks, water/cc-pVDZ PBE0 in PCM(toluene), the factory leaving its SCF unrun
under the communicator as the production driver's does:
  * the solvated mean field on DF and on ISDF-K G1: one energy, density,
    orbitals and spectrum on every rank; against the serial one the energy
    within RANK_SPLIT_ENERGY_TOL and density and orbital energies within
    2 conv_tol_grad / gap (each SCF within |g| / gap of its stationary point)
  * the same with the Newton finish: every rank takes the same steps and
    Hessian products
  * the production chain (ISDF-K G1, space-time, SOP, frontier set, Davidson,
    grid adjoint) S1 force in PCM: the same bits on every rank, and the
    serial chain's force on the SAME mean field to RANK_SPLIT_FORCE_TOL. Not
    the serial chain on its own SCF: the chain freezes its SOP poles from its
    reference mean field by vector fitting, which is not continuous in it --
    the 2-rank SCF, 4e-12 Ha from the serial one in the orbital energies,
    froze poles 1.2e-2 Ha apart and an S1 force 2.4e-9 Ha/Bohr apart at one
    thread per rank, each a consistent surface. That distance is printed.
  * the same force on the row-distributed fit and sliced factors
    (sliced=True, fit='rows'): the same bits on every rank, the one-rank row
    chain's on the same mean field to RANK_SPLIT_FORCE_TOL, and the serial
    replicated chain's to FIT_REALIZATION_FORCE_TOL.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

import numpy as np
from pyscf import dft, gto

from src.Base.numerics.constants import (FIT_REALIZATION_FORCE_TOL,
                                         RANK_SPLIT_ENERGY_TOL, RANK_SPLIT_FORCE_TOL,
                                         SCF_DIFFERENTIABLE_CONV_TOL,
                                         SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import Excitation, GroundState, QPStates
from src.gradients.isdf_mean_field import isdf_jk_with_force
from src.Base.numerics.numerics import numerics_of
from src.Base.pyscf.scf_convergence import newton_finished
from src.Base.factorization.separable_ri import resolve_isdf_grid
from src.environments.solvent_screening import SolventScreening
from src.Base.utils.parallel.mpi_grid import current_comm, distributed, grid_comm
from src.gradients.reference_mean_field import converged_factory
from src.gradients.gwbse.surface_factory import potential_energy_surface
from tests.test_distributed_fit_mpi import Gate, bitwise, serial
from tests.pins import RECORDING_BLAS, matches

WATER = 'O 0.0 0.0 0.1173; H 0.03 0.7572 -0.4692; H -0.02 -0.7472 -0.4492'
COUNTS, N_START = resolve_isdf_grid('G1', 'cc-pvdz', ['H', 'O'],
                                    auxbasis='cc-pvdz-ri')


def factory(route, newton=False):
    """A PBE0 factory on `route`, converging its own SCF only outside a
    communicator."""
    def build(mol):
        mf = dft.RKS(mol, xc='pbe0')
        mf = (mf.density_fit(auxbasis='cc-pvdz-ri') if route == 'df'
              else isdf_jk_with_force(mf, auxbasis='cc-pvdz-ri', counts=COUNTS,
                           n_start=N_START))
        mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
        mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
        mf.max_cycle = 200
        if newton:
            mf = newton_finished(mf)
        if current_comm() is None:
            mf.kernel()
        return mf
    return build


def solvated(route, newton=False):
    """The PCM(toluene) mean field of water in the current region."""
    warnings.simplefilter('ignore')
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    env = SolventScreening(mol, solvent='toluene')
    return env.mean_field(mol, converged_factory(factory(route, newton)))


def force(mf=None, rows=False):
    """(S1 force, mean field, frozen SOP poles) of the production chain in
    PCM(toluene), on `mf` where given; rows: on the row-distributed fit and
    sliced factors."""
    warnings.simplefilter('ignore')
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    chain = potential_energy_surface(
        mol, factory('isdf'), ground_state=GroundState('dft', 'pbe0'),
        excitation=Excitation('singlet', root=2, kernel='bse'),
        environment=SolventScreening(mol, solvent='toluene'),
        chi0='space-time', residues='sop', solver='davidson',
        factorization='isdf', qp_states=QPStates('frontier'),
        mf=mf, numerics=numerics_of(dict(
            grid_accuracy='G1', bse_adjoint='grid', nroots=5,
            **(dict(sliced=True, fit='rows') if rows else {}))))
    g = np.array(chain.total_gradient()[0])
    poles = np.concatenate([chain.resolved.sop_poles[k]
                            for k in sorted(chain.resolved.sop_poles)])
    return g, chain.mf0, poles


def scf_check(gate, route, newton):
    """The solvated mean field on `route` against the serial one."""
    label = f"{route}{', Newton finish' if newton else ''}"
    gate.section(f'solvated mean field, {label}')
    ref = serial(solvated, route, newton)
    mf = solvated(route, newton)
    gate.check(hasattr(mf, 'with_solvent') and mf.converged,
               'a converged PCM mean field')
    mine = (np.array([mf.e_tot]), mf.make_rdm1(), mf.mo_coeff, mf.mo_energy)
    rank0 = (gate.everyone(mine)[0] if gate.comm is not None else mine)
    gate.check(bitwise(mine, rank0), "rank 0's energy, density, orbitals "
               'and spectrum on this rank')
    nocc = int(np.count_nonzero(ref.mo_occ > 0))
    bound = 2 * ref.conv_tol_grad / (ref.mo_energy[nocc]
                                     - ref.mo_energy[nocc - 1])
    de = abs(mf.e_tot - ref.e_tot)
    ddm = float(np.abs(mf.make_rdm1() - ref.make_rdm1()).max())
    deps = float(np.abs(mf.mo_energy - ref.mo_energy).max())
    gate.check(de < RANK_SPLIT_ENERGY_TOL, 'energy at the serial one',
               f'{de:.2e} Ha')
    gate.check(ddm < bound and deps < bound,
               'density and spectrum at the serial ones',
               f'{ddm:.2e}, {deps:.2e} against {bound:.1e}')
    if newton:
        steps = (mf._newton_finish['steps'],
                 mf._newton_finish['hessian_products'])
        gate.check(mf._newton_finish['converged']
                   and len(set(gate.everyone(steps))) == 1,
                   'every rank took the same Newton steps and products',
                   f'{steps}')


def main(comm):
    """Every check on this rank of `comm` (None serially); 0 when every rank
    passed."""
    gate = Gate(comm)
    with distributed(comm):
        for route in ('df', 'isdf'):
            scf_check(gate, route, newton=False)
        scf_check(gate, 'isdf', newton=True)
        gate.section('production S1 force in PCM(toluene)')
        mine, mf, poles = force()
        everyone = gate.everyone(mine)
        # each rank assembles its own force: rank 0's to a few ulp
        gate.check(all(matches(g, everyone[0], RECORDING_BLAS,
                               quantity='gradient') for g in everyone),
                   'the same force on every rank')
        # the rank split proper: the serial chain on this run's mean field
        same, _, same_poles = serial(force, mf)
        diff = float(np.abs(mine - same).max())
        gate.check(diff < RANK_SPLIT_FORCE_TOL,
                   "the serial chain's force on the same mean field",
                   f'{diff:.2e} Ha/Bohr')
        own, _, own_poles = serial(force)
        gate.info(f'against the serial chain on its own SCF '
                  f'{np.abs(mine - own).max():.2e} Ha/Bohr; its frozen SOP '
                  f'poles {np.abs(poles - own_poles).max():.1e} Ha apart '
                  f'(on the same mean field '
                  f'{np.abs(poles - same_poles).max():.1e})')
        gate.section('the same force on the row fit and sliced factors')
        rows, _, _ = force(mf, rows=True)
        everyone = gate.everyone(rows)
        # each rank assembles its own force: rank 0's to a few ulp
        gate.check(all(matches(g, everyone[0], RECORDING_BLAS,
                               quantity='gradient') for g in everyone),
                   'the same force on every rank')
        one, _, _ = serial(force, mf, rows=True)
        diff = float(np.abs(rows - one).max())
        gate.check(diff < RANK_SPLIT_FORCE_TOL,
                   "the one-rank row chain's force on the same mean field",
                   f'{diff:.2e} Ha/Bohr')
        diff = float(np.abs(rows - same).max())
        gate.check(diff < FIT_REALIZATION_FORCE_TOL,
                   "the serial replicated chain's force on the same mean "
                   'field', f'{diff:.2e} Ha/Bohr')
    return gate.finish()


if __name__ == '__main__':
    sys.exit(main(grid_comm()[0]))
