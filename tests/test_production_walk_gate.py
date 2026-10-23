"""The production configuration, walked: the one gate that runs it whole.

Every piece of the excited-state layer is gated alone, and the combination
that runs in production was walked by nothing: formaldehyde/cc-pVDZ on
LRC-wPBEh with an ISDF-K mean field, the GW and BSE factors on the G3 grid,
sum-over-poles residues, a Davidson Casida solve, the admitted quasiparticle
set with its calibrated scissor, the state followed by overlap, one refreeze
at the minimum, in the gas phase and in toluene (non-equilibrium, and the
split equilibrium a relaxed state stands in), on the Cartesian engine. It is
built the way the photophysics stages build it (`stages.Run`: its factory and
its declaration), with the SCF converged for differentiation.

Each walk must converge; the refreeze drift at the minimum must stay within
REFREEZE_TOL_MEV or be flagged ('refreeze_not_converged'), so a drift within it is
converged; the followed state must still overlap its R0 anchor by more than
ROOT_FOLLOW_WEIGHT_MIN; and the total gradient AT THE MINIMUM, on the
surface the walk ended on, must match a five-point finite difference of its
energy along two internal directions within ISDF_GRADIENT_FLOOR at
NUCLEAR_FD_STEP or at half of it. A defect in
the force is independent of the step and fails both; the stencil's
truncation falls as h^4 and the energies' noise rises as 1/h, and each
alone fails one. Every other finite-difference gate is centred where the
chain froze its conventions; this one is not.

THE GATE'S SURFACE IS SMOOTHER THAN PRODUCTION'S, on purpose:
  * every mean field it builds -- the walks, the refreeze, the finite
    difference -- integrates the exchange-correlation potential on pyscf's
    grid level GATE_XC_GRID_LEVEL = 5 instead of production's
    XC_GRID_LEVEL = 4 (`--xc-grid`). On level 3, production's until
    2026-10-22,
    pyscf's LRC-wPBEh puts steps of up to 1.2e-7 Ha into the Kohn-Sham
    orbital energies between geometries 2.5e-4 Bohr apart (none in E_KS),
    which G0W0 carries into the excitation energy, -8e-9 Ha at 1.4e-3 Bohr
    from the toluene minimum: a five-point miss of 6.6e-7 Ha/Bohr at
    h = 1e-3. Level 5 has none there (two cluster runs);
  * the Davidson converges to GATE_BSE_CONV_TOL = 1e-10. At the default
    1e-8 two evaluations of one geometry differ by up to 1.2e-9 Ha in
    toluene, 1e-8 to 4e-8 Ha/Bohr at h = 5e-4, which failed the gate on one
    run in two; at 1e-10 they differ by 3e-13 Ha and every miss at h = 5e-4
    was below 2.5e-9 in three repeats.
The step and the bar are NUCLEAR_FD_STEP and ISDF_GRADIENT_FLOOR as before.
Production is on XC grid level 4 (smooth there
too, at +1% per SCF and +6% per force) and the default Davidson tolerance.

The negative control: a NotImplementedError raised by the quasiparticle
solve at the first trial step leaves the walk instead of collapsing its
trust radius.

SLOW: about four minutes per environment for the walks and a few more for
the finite differences on two threads of this workstation, so it runs only
with MBPT_SLOW_GATES=1; the negative control runs always (under a minute).
"""
import os
import warnings

import numpy as np
import pytest


from src.Base.numerics.constants import (ISDF_GRADIENT_FLOOR, NUCLEAR_FD_STEP,
                                         REFREEZE_TOL_MEV, ROOT_FOLLOW_WEIGHT_MIN)
from src.Base.geometry.geometry import molecule_at
from src.properties.optimize import (RefreezeNotConverged, relax,
                                     translation_rotation_basis)
from src.gradients.surface import driven_chain
from tests.photophysics_cases import production
from tests.pins import assert_same

SLOW = pytest.mark.skipif(
    not os.environ.get('MBPT_SLOW_GATES'),
    reason='the production walk gate takes several minutes per environment; '
           'MBPT_SLOW_GATES=1 runs it')

#: (environment, relaxed solvation) of the three walks.
SETTINGS = {'gas': ('gas', 'nonequilibrium'),
            'toluene-nonequilibrium': ('toluene', 'nonequilibrium'),
            'toluene-split': ('toluene', 'equilibrium')}
#: Bohr: the two five-point steps (see the module docstring).
FD_STEPS = (NUCLEAR_FD_STEP, 0.5 * NUCLEAR_FD_STEP)
#: pyscf's exchange-correlation grid level of every mean field the gate
#: builds (module docstring; production's is XC_GRID_LEVEL).
GATE_XC_GRID_LEVEL = 5
#: The gate's Davidson residual (module docstring; production's is
#: BSE_DAVIDSON_CONV_TOL).
GATE_BSE_CONV_TOL = 1e-10
#: Internal directions the finite difference is taken along.
DIRECTIONS = 2
DIRECTION_SEED = 11


@pytest.fixture
def gate_surface():
    """`production` on the gate's numerics: every Kohn-Sham mean field on XC
    grid GATE_XC_GRID_LEVEL, the Davidson at GATE_BSE_CONV_TOL (module
    docstring)."""
    def build(store_root, environment, solvation):
        return production(store_root, environment, solvation,
                          '--bse-conv-tol', repr(GATE_BSE_CONV_TOL),
                          '--xc-grid', str(GATE_XC_GRID_LEVEL))
    return build


def five_point(surface, mol, u, h):
    """dE/ds along the unit Cartesian direction u at `mol`, five points."""
    x = mol.atom_coords()
    e = {k: surface.total_energy(molecule_at(mol, x + k * h * u))
         for k in (-2, -1, 1, 2)}
    return (e[-2] - 8 * e[-1] + 8 * e[1] - e[2]) / (12 * h)


def internal_directions(mol, n, seed):
    """`n` random unit directions with rigid motion projected out."""
    coords = mol.atom_coords()
    tr = translation_rotation_basis(coords)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = rng.standard_normal(coords.size)
        v -= tr.T @ (tr @ v)
        out.append((v / np.linalg.norm(v)).reshape(coords.shape))
    return out


@SLOW
@pytest.mark.parametrize('setting', list(SETTINGS))
def test_the_production_walk(setting, tmp_path, gate_surface):
    run, surface = gate_surface(tmp_path, *SETTINGS[setting])
    chain0 = driven_chain(surface)
    assert chain0.bse_conv_tol == GATE_BSE_CONV_TOL
    assert chain0.mf0.grids.level == GATE_XC_GRID_LEVEL
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RefreezeNotConverged)
        mol, info = relax(surface, run.mol_in, engine='cartesian',
                          refreeze=1, verbose=False)
    refused = info['refreeze'] == 'not converged'
    final = info['final_surface']
    assert info['first_walk_converged'] and info['converged'], info['status']
    drift = info['refreeze_drift_meV']
    print(f'{setting}: cycles {info["first_walk_cycles"]} + {info["cycles"]}, '
          f'refreeze drift {drift:+.4f} meV, refused {refused}')
    assert refused or abs(drift) <= REFREEZE_TOL_MEV, (
        f'drift {drift:+.4f} meV above the tolerance and not refused')
    chain = driven_chain(final)
    assert chain.follow_log[-1]['anchor'] > ROOT_FOLLOW_WEIGHT_MIN
    grad, energy, _ = final.total_gradient(mol)
    # the gate evaluates the final surface again: the walk's energy to a few ulp
    assert_same(energy, info['energy'],
                label='the walk and the gate read two numbers')
    # The finite difference reads energies off a cold Davidson. A walk's
    # Davidson starts from the roots it committed last (`davidson_warm_start`),
    # and where a seeded solve stops inside BSE_DAVIDSON_CONV_TOL depends on
    # its seed: in toluene-split up to ~1e-10 Ha (test_chain_record), and the
    # five-point stencil at h = 5e-4 read warm missed by 1.4e-7 and 2.0e-7
    # Ha/Bohr against 1.9e-8 cold (two cluster runs, at the default
    # Davidson tolerance). The force is the same either way.
    chain.davidson_warm_start = None
    for u in internal_directions(mol, DIRECTIONS, DIRECTION_SEED):
        analytic = float(np.sum(grad * u))
        misses = [abs(analytic - five_point(final, mol, u, h))
                  for h in FD_STEPS]
        print(f'{setting}: g.u {analytic:+.6e}, five-point misses '
              f'{misses[0]:.2e}, {misses[1]:.2e} Ha/Bohr at h = {FD_STEPS}')
        assert min(misses) < ISDF_GRADIENT_FLOOR, (
            f'{setting}: analytic {analytic:.6e} Ha/Bohr misses the finite '
            f'difference at the minimum by {misses} at h = {FD_STEPS}')


def test_a_defect_in_the_quasiparticle_solve_leaves_the_production_walk(
        tmp_path, gate_surface):
    run, surface = gate_surface(tmp_path, 'gas', 'nonequilibrium')
    chain = driven_chain(surface)
    solve, first = chain.quasiparticles.qp_set_solve, []

    def at_the_start_only(x_mo, d_sigma, eps, *args, **kw):
        first.append(np.array(eps))
        if not np.array_equal(eps, first[0]):
            raise NotImplementedError('a member with no pole model here')
        return solve(x_mo, d_sigma, eps, *args, **kw)

    chain.quasiparticles.qp_set_solve = at_the_start_only
    with pytest.raises(NotImplementedError, match='no pole model'):
        relax(surface, run.mol_in, engine='cartesian', max_cycle=3,
              verbose=False)
