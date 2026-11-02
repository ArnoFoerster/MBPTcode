"""The tiled MP2 force keeps to its factors' communicator and its declared
tile budget (`gradients.df_correlation_force_tiled`).

  * the fitted mean-field addend runs over the factors' ranks, the
    communicator its Z-vector and correlation force run over, even when the
    caller's region is another: it ran over the current region, so a force
    called with comm= from inside a serial region took that addend whole on
    every rank;
  * an MP2State on the tiled route tiles its loop density and its force alike,
    on the Numerics layout's tile_gb: the density took mp2_pieces' default.

Water, cc-pVDZ, frozen core, DF-HF (tests/adc_common.py), over 2 simulated
ranks.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.mpi_grid import current_comm, distributed
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.gradients import df_correlation_force_tiled as tcf
from src.gradients import df_factors
from src.gradients.state_models import HFState, MP2State
from tests.adc_common import ATOMS, AUX, NF, make_scf_loop, molecule
from tests.test_correlated_mean_field_force import TILED, water

#: GB: one occupied orbital per tile
TILE_GB = 1e-9


def test_the_mean_field_addend_runs_over_the_factors_ranks(monkeypatch):
    mol, mf = water()
    seen, own = [], tcf.mean_field_skeleton_force

    def recorded(m):
        comm = current_comm()
        seen.append(1 if comm is None else comm.Get_size())
        return own(m)
    monkeypatch.setattr(tcf, 'mean_field_skeleton_force', recorded)

    def run(comm):
        with distributed(None):
            return tcf.mp2_force_tiled(mf, mol, df_factors.mp2_factors, NF, AUX,
                                       comm=comm, **TILED, numerics=Numerics())[0]
    out = run_simulated(run, 2)
    assert seen == [2, 2], seen
    np.testing.assert_array_equal(out[0], out[1])


def test_the_loop_density_and_the_force_tile_alike(monkeypatch):
    tiles, own = [], tcf.mp2_pieces

    def recorded(factors, tile_gb=None, scaling=None):
        tiles.append(tile_gb)
        return own(factors, tile_gb, scaling)
    monkeypatch.setattr(tcf, 'mp2_pieces', recorded)
    mol = molecule(ATOMS)
    model = MP2State(HFState(mol, make_scf_loop), NF, AUX, force_route='tiled',
                     numerics=Numerics(layout={'tile_gb': TILE_GB}))
    h = np.zeros((mol.nao, mol.nao))
    model(h, 0.0)
    model.force(h, 0.0)
    assert len(tiles) == 2 and set(tiles) == {TILE_GB}, tiles
