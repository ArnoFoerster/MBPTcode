"""No rank holds a points-by-points array of the AO-product fit, its adjoint or
the ISDF-K mean field built on it, on the row fit (`fit='rows'`).

The census traces every line of the package's functions under two simulated
ranks and records each array a local variable holds (directly, or as a value
of a dict, list or tuple) with two axes of the point count: water C1/cc-pVDZ,
PBE0, 150 points pivoted from the G1 parent, tiles of 40 points (the
distributed SCF's tile set to the same), through

  * the distributed ISDF-K SCF on the AO-product row fit
    (`DistributedISDFJK`),
  * its mean-field force, whose interpolated exchange skeleton runs on the
    row fit's tiles (`isdf_exchange_skeleton`),
  * the excited-state chain's S1 force on the row fit, sliced, with the
    exchange skeletons' seeds pooled into the fit adjoint's one call.

Gated: no such array in the fit, the fit adjoint, the exchange skeleton or
the ISDF-K handles (`CENSUS_FILES`). The census does not cover the GW and BSE
kernels: the space-time polarizability forms the Green's functions in the
point basis per tau (`polarizability_projected_tau`), a separate matter.
The census sees the replicated fit's (nk, nk) Gram matrix, so it can fail.

The walk route is censused whole, every module of the package (and so is
scheme B's: the singlet state model of the self-consistent loop in toluene,
its relaxed density and force on the row fit): water
cc-pVDZ, LRC-wPBEh on a density-fitted mean field (cc-pVDZ-JKFIT), the
AO-product factors on 700 points by rows, sliced, sum-over-poles residues on
the admitted set, TDA Davidson and the grid adjoint, the S1 force, in the gas
phase and in a toluene continuum, over two simulated ranks. The (M, M)
intermediates of the kernels go in (rows, M) tiles of at most
ISDF_SQUARE_TILE_ROWS rows (`space_time.square_tile_rows`), set to 256 here so
that 700 points exceed it, as production counts exceed 2048.

Beside it: on the AO-product ISDF-K mean field the exchange skeleton on the
row tiles is the replicated one's (`fit='replicated'`) to 1e-8 of its scale
(measured 1.1e-13).
"""
import os
import sys
import threading
import warnings

import numpy as np
import pytest
from pyscf import df as pyscf_df
from pyscf import dft, gto

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.Base.factorization.distributed_isdf_jk as distributed_isdf_jk
import src.SingleReference.LinearResponse.space_time as space_time
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.factorization.ao_product_fit import ao_product_fit
from src.Base.factorization.distributed_df import (distributed_mean_field,
                                                   release_distributed)
from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.environments.reaction_field import ContinuumField
from src.environments.solvation_loop import equilibrate
from src.environments.solvent_screening import SolventScreening
from src.gradients.gwbse.state_manifold import StateManifold
from src.photophysics.chain import distributed_here
from src.properties.excitations import SurfaceSpec, surface_of
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.state_model import GWBSEState
from src.gradients.isdf_derivatives import isdf_exchange_skeleton
from src.gradients.isdf_mean_field import (isdf_jk_with_force,
                                           isdf_mean_field_gradient)

H2O_C1 = 'O 0 0.05 0.117; H 0.1 0.757 -0.468; H 0 -0.757 -0.468'
NK = 150
TILE = 40
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
#: The modules of the fit, its adjoint and the ISDF-K mean field.
CENSUS_FILES = tuple(os.path.join(SRC, f) for f in (
    'Base/factorization/ao_product_fit.py',
    'Base/factorization/separable_ri.py',
    'Base/factorization/sliced_factors.py',
    'Base/factorization/isdf_jk.py',
    'Base/factorization/distributed_isdf_jk.py',
    'gradients/ao_product_fit_adjoint.py',
    'gradients/isdf_fit_adjoint.py',
    'gradients/isdf_derivatives.py',
    'gradients/isdf_mean_field.py',
    'gradients/skeleton_tiles.py',
    'gradients/rpa/factor_chain.py'))
NUMERICS = Numerics(isdf={'level': 'G1', 'test_set': 'ao_products',
                          'pivot_points': NK, 'fit': 'rows'},
                    scf={'grid_level': 'G1'},
                    layout={'sliced': True, 'fit_block': TILE})


class Census:
    """Every (function, variable, shape) of an array with two axes of `nk`
    a traced function of `files` (files, or directories whose every module
    is traced) held in a local at a line or a return."""

    def __init__(self, nk, files):
        self.nk, self.files = nk, files
        self.found = set()

    def _arrays(self, value):
        if isinstance(value, np.ndarray):
            yield value
        elif isinstance(value, dict):
            yield from (v for v in value.values() if isinstance(v, np.ndarray))
        elif isinstance(value, (list, tuple)):
            yield from (v for v in value if isinstance(v, np.ndarray))

    def _local(self, frame, event, arg):
        if event in ('line', 'return'):
            for name, value in frame.f_locals.items():
                for a in self._arrays(value):
                    if sum(n == self.nk for n in a.shape) >= 2:
                        self.found.add((os.path.basename(
                            frame.f_code.co_filename), frame.f_code.co_name,
                            name, a.shape))
        return self._local

    def __call__(self, frame, event, arg):
        name = frame.f_code.co_filename
        if event == 'call' and (name in self.files or any(
                name.startswith(f + os.sep) for f in self.files)):
            return self._local
        return None

    def __enter__(self):
        threading.settrace(self)
        sys.settrace(self)
        return self

    def __exit__(self, *exc):
        sys.settrace(None)
        threading.settrace(None)


def mean_field(mol):
    mf = isdf_jk_with_force(dft.RKS(mol, xc='pbe0'), auxbasis='cc-pvdz-ri',
                            numerics=NUMERICS)
    mf.grids.level = 3
    mf.conv_tol, mf.conv_tol_grad = 1e-11, 1e-8
    return mf


def test_the_census_sees_a_replicated_fit():
    mol = gto.M(atom=H2O_C1, basis='cc-pvdz', verbose=0)
    mf = mean_field(mol)
    mf.kernel()
    with Census(NK, CENSUS_FILES) as census:
        # moved points: a fit the mean field's build has not made already
        ao_product_fit(mol, pyscf_df.addons.make_auxmol(mol, 'cc-pvdz-ri'),
                       mf.with_df.coords + 1e-3)
    assert ('ao_product_fit.py', 'balanced_gram', 'G', (NK, NK)) \
        in census.found


def test_no_rank_holds_a_points_by_points_array(monkeypatch):
    warnings.simplefilter('ignore')
    monkeypatch.setattr(distributed_isdf_jk, 'FIT_CHOLESKY_BLOCK', TILE)

    def one(comm):
        # one Mole per rank: the nuclear gradient writes into mol._env
        mol = gto.M(atom=H2O_C1, basis='cc-pvdz', verbose=0)
        mf = mean_field(mol)
        distributed_mean_field(mf)
        g_mf = isdf_mean_field_gradient(mf)
        chain = ExcitedStateChain(mol, mean_field, spin='singlet',
                                  numerics=NUMERICS)
        g = chain.total_gradient(mol)[0]
        release_distributed(mf)
        return g_mf, g
    with Census(NK, CENSUS_FILES) as census:
        out = run_simulated(one, 2)
    assert not census.found, sorted(census.found)
    for g_mf, g in out[1:]:
        assert np.array_equal(g_mf, out[0][0]) and np.array_equal(g, out[0][1])


WALK_NK = 700
WALK_NUMERICS = Numerics(isdf={'level': 'G1', 'test_set': 'ao_products',
                               'fit': 'rows', 'pivot_points': WALK_NK},
                         layout={'sliced': True},
                         scf={'conv_tol': 1e-10, 'conv_tol_grad': 1e-9},
                         bse={'nroots': 3, 'adjoint': 'grid'})


def walk_mean_field(mol):
    mf = dft.RKS(mol, xc='lrc-wpbeh').density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol, mf.conv_tol_grad = 1e-10, 1e-9
    if distributed_here():
        return mf
    mf.kernel()
    return mf


@pytest.mark.parametrize('solvent', [None, 'toluene'])
def test_the_walk_route_holds_no_points_by_points_array(monkeypatch, solvent):
    warnings.simplefilter('ignore')
    monkeypatch.setattr(space_time, 'ISDF_SQUARE_TILE_ROWS', 256)
    key = (('singlet', 0),)

    def one(comm):
        mol = gto.M(atom=H2O_C1, basis='cc-pvdz', verbose=0)
        env = (None if solvent is None
               else SolventScreening(mol, solvent=solvent))
        spec = SurfaceSpec(GroundState('dft', 'lrc-wpbeh'), environment=env,
                           chi0='space-time', residues='sop',
                           solver='davidson', factorization='isdf',
                           qp_states=QPStates('admitted'),
                           numerics=WALK_NUMERICS)
        surface = surface_of(spec, Excitation('singlet', kernel='bse-tda'),
                             mol, walk_mean_field)
        ev = StateManifold(surface, states=key).evaluate(mol, gradients=key)
        assert surface.factorization.M == WALK_NK
        return np.asarray(ev.gradient[key[0]])
    with Census(WALK_NK, (SRC,)) as census:
        out = run_simulated(one, 2)
    assert not census.found, sorted(census.found)
    assert np.array_equal(out[0], out[1])
    assert np.abs(out[0].sum(axis=0)).max() < 1e-8


def scheme_b_scf(mol):
    """The scheme-B state models' mean field, configured, unconverged (each
    model converges it in its charges)."""
    mf = dft.RKS(mol, xc='lrc-wpbeh').density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol, mf.conv_tol_grad = 1e-10, 1e-9
    return mf


def test_the_scheme_b_route_holds_no_points_by_points_array(monkeypatch):
    """Scheme B on the row fit, censused whole: the singlet state model
    (its linear-response gauge on the optical continuum beside the bare
    one) in toluene in S0's equilibrium charges, its relaxed density and its
    force, over two simulated ranks."""
    warnings.simplefilter('ignore')
    monkeypatch.setattr(space_time, 'ISDF_SQUARE_TILE_ROWS', 256)

    def one(comm):
        mol = gto.M(atom=H2O_C1, basis='cc-pvdz', verbose=0)
        field = ContinuumField.from_environment(
            SolventScreening(mol, solvent='toluene'))
        q = equilibrate(GWBSEState(mol, scheme_b_scf, numerics=WALK_NUMERICS),
                        field).q
        state = GWBSEState(mol, scheme_b_scf, 'singlet', bse_tda=True,
                           lr_environment=field.transition_environment(),
                           numerics=WALK_NUMERICS)
        h, e = field.operator(q)
        state(h, e)
        assert state.chain.factorization.M == WALK_NK
        return np.asarray(state.force(h, e)[0])
    with Census(WALK_NK, (SRC,)) as census:
        out = run_simulated(one, 2)
    assert not census.found, sorted(census.found)
    assert np.array_equal(out[0], out[1])


def test_the_exchange_skeleton_on_the_rows_is_the_replicated_one():
    warnings.simplefilter('ignore')
    mol = gto.M(atom=H2O_C1, basis='cc-pvdz', verbose=0)
    mf = mean_field(mol)
    mf.kernel()
    rows = isdf_exchange_skeleton(mf, fit='rows')
    whole = isdf_exchange_skeleton(mf, fit='replicated')
    assert np.abs(rows - whole).max() < 1e-8 * np.abs(whole).max(), \
        np.abs(rows - whole).max() / np.abs(whole).max()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
