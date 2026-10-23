"""Solvated chains on the row-distributed fit and sliced factors.

In a dressing environment every factor a chain reads is a gauge of ONE fit,
D_g = M^T (V + vtilde_g)^1/2: the dressed one (the BSE kernel), the bare one
(the self-energy and Eq. (18)) and a split continuum's static one. On the row
fit each gauge is formed by rows from the one fit's M^T tiles, and the fit
adjoint (`separable_ri.fit_rows_adjoints`) carries every gauge in one pass:
D_bar_g R_g summed into MT_bar, each root's adjoint M D_bar_g on rank 0, and
the cavity's own derivative (`aux_kernel_adjoint`) from there.

Gated on water/cc-pVDZ in PCM(toluene):

  (a) THE ADJOINT ON FIXED ADJOINTS: `fit_rows_adjoint` with a dressed gauge
      (root_bar and kernel_bar on it) beside the bare one, the same to a few
      ulp at 1, 2 and 3 simulated ranks, and against `dfactor_adjoint_gauges` formed
      whole over every product pair within FIT_REALIZATION_REL_TOL: two
      realizations of one fit on random seeds.

The production chain (PBE0 on ISDF-K G1, space-time GW, SOP residues,
frontier set, Davidson, grid adjoint), S1, non-equilibrium, equilibrium and
split equilibrium (its static gauge a third factor on the one fit):

  (b) ONE RANK, the row fit against the replicated whole chain on the SAME
      mean field: the BSE roots within FIT_REALIZATION_OBSERVABLE_REL_TOL
      relative, the total energy within FIT_REALIZATION_ENERGY_TOL and the
      total force within FIT_REALIZATION_FORCE_TOL. The two realizations
      hand the pole fit wc 1e-12 apart (relative); with coalesced poles
      merged (SOP_POLE_MERGE_TOL) every fit lands on the same pole set and
      the roots agree to ~1e-13;
  (c) 2 AND 3 SIMULATED RANKS, every rank's own SCF: the ranks agree to a
      few ulp, and the force and energy are the one-rank row chain's on rank 0's
      mean field to RANK_SPLIT_FORCE_TOL and RANK_SPLIT_ENERGY_TOL;
      with (e) THE MEMORY CENSUS at every factor build and force on every
      rank (tests/test_chain_row_fit.py's scan): no reachable array holds
      the whole grid by a factor's or the fit's width (X_mo, X_ao, D of any
      gauge, M, F D^T, the Gram matrix), every rows object is the row fit's
      with its ledger at this rank's tiles, and the nuclear assembly, traced
      line by line, forms no grid-by-width array beyond the adjoints handed
      in, its ledger at this rank's tiles;
  (d) the row-fit force against a central difference of its own energy
      (ISDF-K G3, step 1e-3 Bohr, within 1e-6 Ha/Bohr).

What breaks the census of (e): `gauge_factor` also keeping the bare gauge's D
whole on its rows (gathered) fails it at 2 ranks on every scan after the
forward, naming (888, 84), the whole (npts, naux).
"""
import types
import warnings

import numpy as np
import pytest
from pyscf import dft, gto

from src.Base.numerics.constants import (FIT_REALIZATION_ENERGY_TOL,
                                         FIT_REALIZATION_FORCE_TOL,
                                         FIT_REALIZATION_OBSERVABLE_REL_TOL,
                                         FIT_REALIZATION_REL_TOL,
                                         RANK_SPLIT_ENERGY_TOL, RANK_SPLIT_FORCE_TOL)
from src.Base.declaration import Excitation, GroundState, QPStates
from src.gradients.isdf_mean_field import isdf_jk_with_force
from src.Base.numerics.numerics import Numerics, numerics_of
from src.Base.factorization.separable_ri import (GaugeSeeds, fit_rows_adjoint,
                                                 resolve_isdf_grid)
from src.Base.factorization.sliced_factors import GridTileRows
from src.environments.solvent_screening import SolventScreening
from src.environments.split_equilibrium import SplitEquilibrium
from src.Base.utils.parallel.mpi_grid import current_comm, distributed
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients.rpa.factor_chain import FrozenFactorization
from src.gradients.isdf_derivatives import (GaugeAdjoint,
                                            dfactor_adjoint_gauges,
                                            point_chain, product_pairs)
from src.properties.excitations import SurfaceSpec, surface_of
from src.gradients.gwbse.surface_factory import potential_energy_surface
from tests.test_chain_row_fit import (TILE, adjoint_ledger_faults,
                                      adjoint_widths, ledger_faults, molecule,
                                      relative, traced, whole_fit_arrays)
from tests.test_chain_sliced_factors import H2O
from tests.test_solvated_scf_ranks import built, water
from tests.isdf_test_grid import g1
from tests.pins import assert_same

SIZES = [1, 2, 3]
SOLVENT = 'toluene'
#: (d): the finite difference on the ISDF-K G3 grid (a smooth surface for the
#: difference), step 1e-3 Bohr, two force components, bar 1e-6 Ha/Bohr
FD_STEP = 1e-3
TOL = 1e-6
COMPONENTS = ((0, 2), (1, 1))
MODES = {'non-equilibrium': False, 'equilibrium': True, 'split': 'split'}
ROWS = dict(sliced=True, fit='rows', fit_block=TILE)


def environment(mol, equilibrium=False):
    """PCM(toluene) around `mol`: the response at eps_inf, in equilibrium at
    eps_static, or split (`SplitEquilibrium`: the density terms at
    eps_static, the exchange at eps_inf)."""
    warnings.simplefilter('ignore')
    if equilibrium == 'split':
        return SplitEquilibrium(SolventScreening(mol, solvent=SOLVENT))
    return SolventScreening(mol, solvent=SOLVENT, equilibrium=equilibrium)


def factory(mol):
    """`built`'s ISDF-K PBE0, left unrun under a communicator."""
    mf = built(mol, 'isdf')
    if current_comm() is None:
        mf.kernel()
    return mf


def production(mol, equilibrium, layout, mf=None):
    """The production chain's S1 in PCM(toluene), whole or on `ROWS`."""
    warnings.simplefilter('ignore')
    return potential_energy_surface(
        mol, factory, ground_state=GroundState('dft', 'pbe0'),
        excitation=Excitation('singlet', root=2, kernel='bse'),
        environment=environment(mol, equilibrium), chi0='space-time',
        residues='sop', solver='davidson', factorization='isdf',
        qp_states=QPStates('frontier'), mf=mf,
        numerics=numerics_of(dict(grid_accuracy='G1', bse_adjoint='grid',
                                  nroots=5,
                                  **(ROWS if layout == 'rows' else {}))))


def observed(chain):
    """(BSE roots, total force, total energy) at the reference."""
    roots = np.array(chain.spectrum())
    force, energy, _ = chain.total_gradient()
    return roots, np.array(force), float(energy)


def census(chain, log, rank, size):
    """Scan `chain` right after every factor build, and trace every force's
    nuclear assembly, logging (whole arrays, ledger faults) each time."""
    shim = types.SimpleNamespace(ground=chain)
    build, assemble = chain.factors_at, chain.nuclear_gradient
    branches, ledgers = chain.row_fit_branches, []

    def scan():
        log.append((whole_fit_arrays(shim), ledger_faults(shim, rank, size)))

    def factors_at(mol, mf):
        out = build(mol, mf)
        scan()
        return out

    def own_ledger(*args, **kwargs):
        out = branches(*args, **kwargs)
        ledgers.append(out[2])
        return out

    def nuclear_gradient(mol, mf, auxmol, crd, x_mo, eps_bar, x_bar, d_bar,
                         **kwargs):
        # every gauge the forward built is held at this point
        scan()
        inputs = [a.rows if isinstance(a, GridTileRows) else a
                  for a in (x_bar, d_bar)]
        found = set()
        ledgers.clear()
        out = traced(lambda: assemble(mol, mf, auxmol, crd, x_mo, eps_bar,
                                      x_bar, d_bar, **kwargs),
                     found, *adjoint_widths(chain), inputs)
        log.append((sorted(found), adjoint_ledger_faults(
            ledgers[0] if ledgers else None, chain, rank, size)))
        return out

    chain.factors_at = factors_at
    chain.row_fit_branches = own_ledger
    chain.nuclear_gradient = nuclear_gradient
    return scan


# ------------------------------------------------------------------- (a)
def test_dressed_gauges_on_fixed_adjoints():
    """A dressed gauge with root and kernel adjoints beside the bare one:
    the same to a few ulp at 1/2/3 ranks, and the whole adjoint within
    FIT_REALIZATION_REL_TOL."""
    mol = molecule(H2O)
    with distributed(None):
        fac = FrozenFactorization(mol, Numerics(isdf={'counts': g1(mol)}))
    aux, crd = fac.auxmol(mol), fac.coords(mol)
    npts, naux = len(crd), aux.nao_nr()
    rng = np.random.default_rng(5)
    d_bar, d_bare = (rng.normal(size=(npts, naux)) for _ in range(2))
    root_bar, kernel_bar = (rng.normal(size=(naux, naux)) for _ in range(2))
    chain = dict(pts_local=fac.pts_local, atom_of_point=fac.owner,
                 with_frames=False)

    def rank(comm):
        env = environment(mol)
        adj = fit_rows_adjoint(mol, aux, crd, d_bare, fac.layout, block=TILE,
                               gauges=[GaugeSeeds(d_bar, env,
                                                  root_bar=root_bar,
                                                  kernel_bar=kernel_bar)])
        return (adj.fit_centre + point_chain(mol, adj.fit_points, **chain),)

    runs = {size: run_simulated(rank, size) for size in SIZES}
    one = runs[1][0]
    for size, out in runs.items():
        for r, got in enumerate(out):
            assert len(got) == len(one)
            for a, b in zip(got, one):
                assert_same(a, b, quantity='gradient',
                            label=f'rank {r} of {size} != one rank')
    with distributed(None):
        env = environment(mol)
        whole = dfactor_adjoint_gauges(
            mol, aux, crd, [GaugeAdjoint(d_bar, env, root_bar=root_bar,
                                         kernel_bar=kernel_bar),
                            GaugeAdjoint(d_bare, None)],
            fac.layout, fac.pts_local, fac.owner, with_frames=False,
            gram_layout=product_pairs(mol))
        gas = dfactor_adjoint_gauges(
            mol, aux, crd, [GaugeAdjoint(d_bar, None),
                            GaugeAdjoint(d_bare, None)],
            fac.layout, fac.pts_local, fac.owner, with_frames=False,
            gram_layout=product_pairs(mol))
    dist = relative(one[0], whole)
    print(f'\nrows vs whole: {dist:.2e} relative; the dressing moves the '
          f'branch by {relative(gas, whole):.2e}')
    assert dist <= FIT_REALIZATION_REL_TOL
    # the gate can fail: the cavity's share is far above the tolerance
    assert relative(gas, whole) > 1e3 * FIT_REALIZATION_REL_TOL


# ------------------------------------------------------------------- (b)
@pytest.mark.parametrize('mode', list(MODES))
def test_row_fit_is_the_replicated_chain_at_one_rank(mode):
    """Roots, energy and force of the solvated production chain on the row
    fit against the replicated whole chain on the same mean field, within
    the FIT_REALIZATION_* tolerances."""
    with distributed(None):
        whole_chain = production(water(), MODES[mode], 'whole')
        whole = observed(whole_chain)
        rows = observed(production(water(), MODES[mode], 'rows',
                                   mf=whole_chain.mf0))
    roots = relative(rows[0], whole[0])
    force = float(np.abs(rows[1] - whole[1]).max())
    energy = abs(rows[2] - whole[2])
    print(f'\n{mode}: roots {roots:.2e} relative, force {force:.2e} Ha/Bohr, '
          f'energy {energy:.2e} Ha')
    assert roots <= FIT_REALIZATION_OBSERVABLE_REL_TOL
    assert force <= FIT_REALIZATION_FORCE_TOL
    assert energy <= FIT_REALIZATION_ENERGY_TOL


# ------------------------------------------------------------ (c) and (e)
@pytest.mark.parametrize('mode', list(MODES))
@pytest.mark.parametrize('size', [2, 3])
def test_row_fit_over_ranks_with_the_census(size, mode):
    """Every rank's force and energy the same to a few ulp, the one-rank row chain's
    on rank 0's mean field to RANK_SPLIT_*, and the census clean at every
    factor build and force."""
    def rank(comm):
        chain = production(water(), MODES[mode], 'rows')
        log = []
        scan = census(chain, log, comm.Get_rank(), size)
        force, energy, _ = chain.total_gradient()
        scan()
        return np.array(force), float(energy), log, chain.mf0

    out = run_simulated(rank, size)
    for r, (force, energy, log, _) in enumerate(out):
        # each rank converges its own SCF and chain: rank 0's to a few ulp
        assert_same(force, out[0][0], quantity='gradient', label=(r, 'force'))
        assert_same(energy, out[0][1], label=(r, 'energy'))
        assert len(log) >= 2, log
        assert all(entry == ([], []) for entry in log), (r, log)
    with distributed(None):
        force, energy, _ = production(water(), MODES[mode], 'rows',
                                      mf=out[0][3]).total_gradient()
    dforce = float(np.abs(out[0][0] - np.array(force)).max())
    print(f'\n{size} ranks, {mode}: force {dforce:.2e} Ha/Bohr, energy '
          f'{abs(out[0][1] - energy):.2e} Ha')
    assert dforce < RANK_SPLIT_FORCE_TOL
    assert abs(out[0][1] - energy) < RANK_SPLIT_ENERGY_TOL


# ------------------------------------------------------------------- (d)
def isdf_pbe0(mol):
    """PBE0 on ISDF-K at the G3 grid, converged."""
    counts, n_start = resolve_isdf_grid('G3', 'cc-pvdz', ['H', 'O'],
                                        auxbasis='cc-pvdz-ri')
    mf = isdf_jk_with_force(dft.RKS(mol, xc='pbe0'), auxbasis='cc-pvdz-ri',
                 counts=counts, n_start=n_start)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-11, 200
    mf.kernel()
    return mf


def pcm_molecule(coords=None):
    """Water/cc-pVDZ, at `coords` (Bohr) when given."""
    mol = gto.M(atom=H2O, basis='cc-pvdz', verbose=0)
    if coords is not None:
        mol.set_geom_(coords, unit='Bohr')
        mol.build(False, False)
    return mol


def central(energy, mol0, ia, k):
    """The central difference of `energy` along atom ia's axis k."""
    x0 = mol0.atom_coords()
    out = []
    for s in (+1, -1):
        x = x0.copy()
        x[ia, k] += s * FD_STEP
        out.append(energy(pcm_molecule(x)))
    return (out[0] - out[1]) / (2 * FD_STEP)


def test_row_fit_force_is_the_energy_derivative():
    """The production chain's S1 total force in PCM(toluene) on the row fit
    against a central difference of its own energy."""
    mol0 = pcm_molecule()
    spec = SurfaceSpec(GroundState('dft', 'pbe0'),
                       environment=environment(mol0), chi0='space-time',
                       residues='sop', solver='davidson',
                       factorization='isdf',
                       qp_states=QPStates(kind='frontier'),
                       numerics=numerics_of({'grid_accuracy': 'G3', 'bse_adjoint': 'grid',
                                 'nroots': 4, **ROWS}))
    with distributed(None):
        chain = surface_of(spec, Excitation('singlet', root=1), mol0,
                           isdf_pbe0)
        force, _, _ = chain.total_gradient()
        for ia, k in COMPONENTS:
            fd = central(chain.total_energy, mol0, ia, k)
            print(f'\n({ia}, {k}): force {force[ia, k]:.10f}, '
                  f'difference {fd:.10f}')
            assert abs(force[ia, k] - fd) < TOL, (ia, k, force[ia, k], fd)
