"""The energy-only GW and BSE routes in a continuum on the row-distributed fit
and sliced factors.

Duchemin, Guido, Jacquemin and Blase, Chem. Sci. 9, 4430 (2018) Eq. (18)
contracts the orbital densities A = D^T X_mo^2 twice with the static
screening. On `SlicedFactors` A is a sum over the grid of row products in
fixed tiles (`reaction_field.density_projection`), so no rank gathers X_mo,
and the bare-gauge factor D T the self-energy screens with is the solve's own
gather of D carried over in place.

Gated on water/cc-pVDZ Hartree-Fock in PCM(toluene), the quasiparticle window
HOMO-1..LUMO+1, the Eq. (18) shift of every orbital and the three lowest
ISDF BSE roots (`solve_bse_isdf`):

  (a) ONE RANK, the row fit against the replicated fit on the same mean
      field: the shift, the quasiparticle energies and the roots within
      FIT_REALIZATION_OBSERVABLE_REL_TOL relative;
  (b) 2 AND 3 SIMULATED RANKS on the row fit's slices: every rank rank 0's
      roots (lockstepped) and shift (to a few ulp), and the shift, the quasiparticle energies and the roots within
      RANK_SPLIT_ENERGY_TOL of the one-rank row fit's, with the densities in
      the production tile (one tile on this grid) and cut into 7 tiles;
  (c) THE MEMORY CENSUS: the continuum's solve makes the gas phase's gathers
      on the same slices, by name and by collective, plus the one gather of
      X_o and X_v the window's static W(0) sweep makes (the Eq. (18) shift
      reads it; the gas-phase window builds none), X_mo never among them,
      the densities are formed from this rank's rows, and the bare gauge is
      the gathered D's own buffer;
  (d) the gas phase never reaches the densities.

Measured: (a) the shift 3.3e-12, the roots 2.9e-13 relative, the window
bitwise; (b) the shift 8.9e-16 Ha cut into tiles (bitwise in one), the roots
1.2e-13 Ha, the window bitwise.

What breaks (c): densities formed from X_mo and D gathered whole, at 2 and 3
ranks, naming two X_mo and two more D gathers.

The quasiparticle root search resolves 6.4e-9 Ha on this window (a D scaled
by 1 + 1e-10 leaves it bitwise), which is why the shift itself is compared.
"""
import threading
import warnings
from collections import Counter, defaultdict

import numpy as np
import pytest
from pyscf import df, gto, scf

from src.Base import sliced_factors
from src.Base.constants import (FIT_REALIZATION_OBSERVABLE_REL_TOL,
                                RANK_SPLIT_ENERGY_TOL)
from src.Base.environment import attach_environment
from src.Base.numerics import Numerics
from src.Base.sliced_factors import SlicedFactors
from src.Base.solvent_screening import SolventScreening
from src.Base.utils.mpi_grid import current_comm, distributed, run_simulated
from src.SingleReference.GW import reaction_field, space_time
from src.SingleReference.GW.qp_solve import static_exchange_mean_field_matrix
from src.SingleReference.GW.space_time import (separable_factors,
                                               solve_qp_energy_space_time)
from src.SingleReference.LinearResponse.davidson import solve_bse_isdf
from tests.isdf_test_grid import g1
from tests.pins import assert_same

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
BASIS, AUXBASIS = 'cc-pvdz', 'cc-pvdz-ri'
SOLVENT = 'toluene'
TILE = 64
NROOTS = 3
SIZES = [2, 3]
#: <p|Sigma_x - v_xc|q> of each mean field, by id, filled by `water`.
EXCHANGE = {}


def rhf(mol):
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = 1e-11
    mf.kernel()
    return mf


@pytest.fixture(scope='module')
def water():
    """(mol, gas mean field, solvated mean field, nocc): the solvated one
    converged at eps_static and responding at eps_inf of toluene."""
    warnings.simplefilter('ignore')
    mol = gto.M(atom=WATER, basis=BASIS, verbose=0)
    env = SolventScreening(mol, solvent=SOLVENT)
    solvated = attach_environment(env.mean_field(mol, rhf), env)
    gas = rhf(mol)
    # built here, serially: pyscf's PCM solver is one object the simulated
    # ranks' threads would otherwise enter at once
    for mf in (gas, solvated):
        EXCHANGE[id(mf)] = static_exchange_mean_field_matrix(mf, mol)
    return mol, gas, solvated, mol.nelectron // 2


@pytest.fixture
def spies(monkeypatch):
    """{rank: [...]} of the Eq. (18) shifts, the densities' inputs, the bare
    gauges and the row gathers each rank's solves made."""
    log, lock = defaultdict(lambda: defaultdict(list)), threading.Lock()
    shift, densities = (space_time.separable_quasiparticle_shift,
                        reaction_field.orbital_densities)
    bare, gather = space_time._bare_gauge, sliced_factors.allgather_rows

    def rank():
        comm = current_comm()
        return 0 if comm is None else comm.Get_rank()

    def record(key, value):
        with lock:
            log[rank()][key].append(value)

    def spy_shift(x_mo, d_mo, w, transform, nocc):
        out = shift(x_mo, d_mo, w, transform, nocc)
        record('shift', out)
        return out

    def spy_densities(x_mo, d_mo):
        record('densities', tuple(
            (type(f).__name__, f.rows, f.X_mo.shape, f.D.shape)
            if isinstance(f, SlicedFactors) else (type(f).__name__, f.shape)
            for f in (x_mo, d_mo)))
        return densities(x_mo, d_mo)

    def spy_bare(x_mo, d, transform):
        out = bare(x_mo, d, transform)
        record('bare_in_place', out is d)
        return out

    def spy_gather(a, comm):
        record('allgather_rows', a.shape)
        return gather(a, comm)

    monkeypatch.setattr(space_time, 'separable_quasiparticle_shift', spy_shift)
    monkeypatch.setattr(reaction_field, 'orbital_densities', spy_densities)
    monkeypatch.setattr(space_time, '_bare_gauge', spy_bare)
    monkeypatch.setattr(sliced_factors, 'allgather_rows', spy_gather)
    return log


def observed(mf, mol, nocc, factors):
    """(quasiparticle window, BSE roots) on `factors`."""
    window = np.arange(nocc - 2, nocc + 2)
    exchange = EXCHANGE[id(mf)]
    qp = solve_qp_energy_space_time(mf, mol, nocc, window, factors=factors,
                                    sigma_x_matrix=exchange)
    om = solve_bse_isdf(mf, mol, nocc, numerics=Numerics(bse={'nroots': NROOTS}), factors=factors,
                        progress=False, sigma_x_matrix=exchange)[0]
    return qp, np.asarray(om)


def row_fit(mf, mol, sliced=False):
    return separable_factors(mf, mol, auxbasis=AUXBASIS, fit='rows',
                             fit_block=TILE, sliced=sliced, counts=g1(mol))


def relative(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max()
                 / np.abs(np.asarray(b)).max())


def test_one_rank_row_fit_is_the_replicated_fit(water, spies):
    mol, _, mf, nocc = water
    with distributed(None):
        rep = observed(mf, mol, nocc,
                       separable_factors(mf, mol, auxbasis=AUXBASIS, counts=g1(mol)))
        rows = observed(mf, mol, nocc, row_fit(mf, mol))
    # the window's solve and the BSE diagonal's, on each fit
    shifts = spies[0]['shift']
    assert len(shifts) == 4 and np.abs(shifts[0]).max() > 1e-3
    worst = {'shift': relative(shifts[2], shifts[0]),
             'qp': relative(rows[0], rep[0]), 'bse': relative(rows[1], rep[1])}
    print(f'\nrow fit against replicated, relative: {worst}')
    assert max(worst.values()) <= FIT_REALIZATION_OBSERVABLE_REL_TOL, worst


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('tiles', ['production', 'cut'])
def test_slices_are_the_one_rank_row_fit(water, spies, monkeypatch, size,
                                         tiles):
    mol, _, mf, nocc = water
    if tiles == 'cut':
        # the 444-point grid in 7 tiles: the densities summed across tiles
        monkeypatch.setattr(reaction_field, 'FIT_CHOLESKY_BLOCK', TILE)
        monkeypatch.setattr(space_time, 'FIT_CHOLESKY_BLOCK', TILE)
    with distributed(None):
        ref = observed(mf, mol, nocc, row_fit(mf, mol))
    ref_shifts = list(spies.pop(0)['shift'])

    def rank(comm):
        return observed(mf, mol, nocc, row_fit(mf, mol, sliced=True))

    res = run_simulated(rank, size)
    for r, out in enumerate(res):
        # the window's and the BSE's roots end in a lockstep of rank 0's
        assert all(np.array_equal(a, b) for a, b in zip(out, res[0])), r
        # each rank forms its own Eq. (18) shift: rank 0's to a few ulp
        assert len(spies[r]['shift']) == len(spies[0]['shift']), r
        for a, b in zip(spies[r]['shift'], spies[0]['shift']):
            assert_same(a, b, label=(r, 'shift'))
    worst = {'shift': max(np.abs(a - b).max() for a, b in
                          zip(spies[0]['shift'], ref_shifts)),
             'qp': np.abs(res[0][0] - ref[0]).max(),
             'bse': np.abs(res[0][1] - ref[1]).max()}
    print(f'\n{size} ranks against one, Ha: {worst}')
    assert max(worst.values()) <= RANK_SPLIT_ENERGY_TOL, worst


@pytest.mark.parametrize('size', SIZES)
def test_the_continuum_gathers_what_the_gas_phase_does(water, spies, size):
    mol, gas, mf, nocc = water
    naux = df.addons.make_auxmol(mol, auxbasis=AUXBASIS).nao_nr()

    def rank(comm):
        out = {}
        for tag, m in (('gas', gas), ('solvated', mf)):
            part = row_fit(m, mol, sliced=True)
            before = len(spies[comm.Get_rank()]['allgather_rows'])
            observed(m, mol, nocc, part)
            out[tag] = (dict(part.gathers), part.rows,
                        spies[comm.Get_rank()]['allgather_rows'][before:])
        return out

    res = run_simulated(rank, size)
    npts, nmo = res[0]['gas'][2][0][0], mf.mo_coeff.shape[1]
    static_sweep = Counter({(npts, nocc): 1, (npts, nmo - nocc): 1})
    for r, out in enumerate(res):
        assert out['solvated'][0] == dict(Counter(out['gas'][0])
                                          + Counter({'X_o': 1, 'X_v': 1})), \
            (r, out)
        assert (Counter(out['solvated'][2])
                == Counter(out['gas'][2]) + static_sweep), r
        assert 'X_mo' not in out['solvated'][0]
        r0, r1 = out['solvated'][1]
        rows = ('SlicedFactors', (r0, r1), (r1 - r0, mf.mo_coeff.shape[1]),
                (r1 - r0, naux))
        # the window's solve and the BSE diagonal's, each on this rank's rows
        assert spies[r]['densities'] == [(rows, rows)] * 2, spies[r]
        assert spies[r]['bare_in_place'] == [True, True]


def test_the_gas_phase_reaches_no_densities(water, spies):
    mol, gas, _, nocc = water

    def rank(comm):
        return observed(gas, mol, nocc, row_fit(gas, mol, sliced=True))

    run_simulated(rank, 2)
    assert not any(spies[r]['densities'] or spies[r]['shift']
                   for r in range(2))
