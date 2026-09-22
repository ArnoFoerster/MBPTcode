"""One declaration, one set of orbital indices, and the two legacy windows it replaces.

`SingleReference/GW/qp_states.py` is where a `QPStates` declaration becomes the
orbitals a surface solves explicitly. Two incompatible windows used to live
under the name `qp_window` -- an integer half-width around the Fermi level
widened over degenerate blocks (`ExcitedStateChain._qp_set`) and the valence
occupied plus ten virtuals of the dense route (`dense_surfaces.qp_window`) --
so the gates here are `array_equal` against those two functions, not
tolerances: a surface whose quasiparticle set moved is a different surface and
its energies are not differenceable against the old ones.

The reference for the two moved windows is a VERBATIM copy of the code that was
replaced (`_old_qp_set`, `_old_qp_window`), because `dense_surfaces.qp_window`
is now an alias of the production function and comparing against it alone would
gate nothing.

Every gate below was shown to fail once, by breaking in the source what the
gate watches and running the 28 cases of this file against it; each run was
restored from a backup copy and the restore checked byte for byte:

  * the degeneracy widening dropped from production `frontier_qp_states` (both
    while loops removed): 5 fail, 23 pass. On methane/cc-pVDZ, whose t2
    orbitals are degenerate to machine precision, half_width 1 goes from
    [2 3 4 5] to [4 5] and half_width 2 from [2 3 4 5 6 7 8] to [3 4 5 6]: half
    of a degenerate triple then carries a quasiparticle energy and half the
    mean field, which differentiates across an arbitrary rotation. Water, whose
    frontier is non-degenerate, does not move at all -- which is why the
    methane case is in the file.
  * the valence window shifted by one, `range(ncore + 1, nocc)` in production
    `valence_qp_states`: 4 fail, 24 pass. The window loses orbital 1 on both
    molecules -- the 2s of oxygen and of carbon, the inner valence, which is
    exactly where a frozen shift has the most leverage on a frontier
    quasiparticle (3.6-10 meV per eV, against 0.05 meV per 20 eV for the core).
  * `admitted_by_pole_condition` dividing the worst swept pole by `limit`
    squared: 3 fail, 25 pass. On water/cc-pVDZ, where E_g = 0.6787 Ha, the O 1s
    reach reads 43.543 instead of 29.552 and the set the condition admits at
    limit = E_g shrinks from [2 3 4 5 6 7 8] to [2 3 4 5 6], so it stops
    agreeing with `compressible` in both the verdict and the number. The
    threshold='gap' route cannot see it, because that one calls `compressible`
    itself -- which is the point of gating the two against each other.
  * `qp_space_time` wrapping the moved `frozen_scissor` in a forwarding
    function instead of re-exporting it: 1 fail, 27 pass. The behaviour is
    identical and only the identity gate sees it -- which is the failure mode a
    re-export actually has, a name that resolves to something merely resembling
    the moved object.
  * the import gate cannot be broken from inside production: importing
    anything of `src.gradients` there is circular through
    `gradients/__init__.py` -> `qp_space_time` and dies at import time. Its
    sensitivity is a positive control in the same test instead -- the same
    subprocess probe run on `src.gradients.qp_space_time`, which must show
    `src.gradients` in sys.modules.
"""
import subprocess
import sys

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.data.elements import chemcore

from src.Base.constants import QP_WINDOW_Z_MIN
from src.Base.declaration import QPStates
from src.SingleReference.GW import qp_states as prod
from src.SingleReference.GW.sum_over_poles import compressible
from src.gradients import dense_surfaces, qp_space_time
from src.gradients.excited_state import ExcitedStateChain

#: The chain's own degeneracy tolerance; `resolve_qp_states` has no default.
DEGENERACY_TOL = 1e-4

MOLS = {
    'water': 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
    'methane': ('C 0 0 0; H 0.6276 0.6276 0.6276; H -0.6276 -0.6276 0.6276; '
                'H -0.6276 0.6276 -0.6276; H 0.6276 -0.6276 -0.6276'),
}

#: A core state, two inner-valence ones, then the frontier -- the spectrum
#: tests/test_scissor_tier.py calibrates on.
EPS_TIER = np.array([-11.0, -1.35, -1.20, -0.62, -0.35, 0.18, 0.44])
NOCC_TIER = 5


def _old_qp_set(eps, nocc, window, tol):
    """`ExcitedStateChain._qp_set` verbatim, as it stood before the move."""
    if str(window).lower() == 'all':
        return np.arange(len(eps))
    w = int(window)
    lo, hi = max(nocc - w, 0), min(nocc + w, len(eps))
    while lo > 0 and eps[lo] - eps[lo - 1] < tol:
        lo -= 1
    while hi < len(eps) and eps[hi] - eps[hi - 1] < tol:
        hi += 1
    return np.arange(lo, hi)


def _old_qp_window(mol, norb, nocc):
    """`dense_surfaces.qp_window` verbatim, as it stood before the move."""
    ncore = chemcore(mol)
    n_vir = min(norb - nocc, nocc + 10)
    return list(range(ncore, nocc)) + list(range(nocc, nocc + n_vir))


@pytest.fixture(scope='module')
def spectra():
    """(mol, eps, nocc) for water and methane at cc-pVDZ Hartree-Fock.

    Methane carries a degenerate t2 triple in both the occupied and the virtual
    space, which is what makes the widening visible; water has neither.
    """
    out = {}
    for name, atom in MOLS.items():
        mol = gto.M(atom=atom, basis='cc-pVDZ', verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        assert mf.converged
        out[name] = (mol, np.asarray(mf.mo_energy, float),
                     mol.nelectron // 2)
    return out


# ------------------------------------------------------------------- frontier

@pytest.mark.parametrize('name', sorted(MOLS))
@pytest.mark.parametrize('half_width', (1, 2, 3))
def test_frontier_is_the_chains_window(spectra, name, half_width):
    """The production function and the chain's staticmethod, index for index."""
    _, eps, nocc = spectra[name]
    got = prod.frontier_qp_states(eps, nocc, half_width, DEGENERACY_TOL)
    assert np.array_equal(got, ExcitedStateChain._qp_set(eps, nocc, half_width,
                                                        DEGENERACY_TOL))
    assert np.array_equal(got, _old_qp_set(eps, nocc, half_width,
                                           DEGENERACY_TOL))


def test_frontier_widens_over_a_degenerate_block(spectra):
    """Methane's t2 orbitals are degenerate to machine precision, so a
    half-width that would cut one off has to take the whole triple."""
    _, eps, nocc = spectra['methane']
    assert eps[4] - eps[2] == pytest.approx(0.0, abs=1e-12), 'occupied t2'
    assert eps[8] - eps[6] == pytest.approx(0.0, abs=1e-12), 'virtual t2'
    assert np.array_equal(prod.frontier_qp_states(eps, nocc, 1, DEGENERACY_TOL),
                          [2, 3, 4, 5]), 'widened down over the occupied triple'
    assert np.array_equal(prod.frontier_qp_states(eps, nocc, 2, DEGENERACY_TOL),
                          [2, 3, 4, 5, 6, 7, 8]), 'and up over the virtual one'


def test_frontier_takes_all_when_asked(spectra):
    _, eps, nocc = spectra['water']
    assert np.array_equal(prod.frontier_qp_states(eps, nocc, 'all',
                                                  DEGENERACY_TOL),
                          np.arange(len(eps)))
    assert np.array_equal(ExcitedStateChain._qp_set(eps, nocc, 'all',
                                                    DEGENERACY_TOL),
                          np.arange(len(eps)))


def test_resolve_frontier_matches_the_chain(spectra):
    _, eps, nocc = spectra['methane']
    spec = QPStates(kind='frontier', half_width=2)
    got = prod.resolve_qp_states(spec, eps, nocc, degeneracy_tol=DEGENERACY_TOL)
    assert np.array_equal(got.explicit,
                          ExcitedStateChain._qp_set(eps, nocc, 2, DEGENERACY_TOL))
    assert got.reach == {}, 'reach is the admitted criterion and nothing else'


# -------------------------------------------------------------------- valence

@pytest.mark.parametrize('name', sorted(MOLS))
def test_valence_is_the_dense_routes_window(spectra, name):
    mol, eps, nocc = spectra[name]
    got = prod.valence_qp_states(mol, mol.nao, nocc)
    assert got == _old_qp_window(mol, mol.nao, nocc)
    assert got == dense_surfaces.qp_window(mol, mol.nao, nocc)
    spec = QPStates(kind='valence')
    resolved = prod.resolve_qp_states(spec, eps, nocc, mol=mol,
                                      degeneracy_tol=DEGENERACY_TOL)
    assert list(resolved.explicit) == got


def test_the_dense_route_calls_production(spectra):
    """The alias is the production object, not a second implementation."""
    assert dense_surfaces.qp_window is prod.valence_qp_states


def test_the_z_filter_drops_a_satellite_and_keeps_the_rest(spectra):
    """A root below QP_WINDOW_Z_MIN carries less than half the spectral weight
    of the state, so it is not the quasiparticle the window asked for."""
    mol, _, nocc = spectra['water']
    window = _old_qp_window(mol, mol.nao, nocc)
    z = np.full(mol.nao, 0.9)
    z[window[3]] = 0.3
    assert z[window[3]] < QP_WINDOW_Z_MIN
    kept = prod.valence_qp_states(mol, mol.nao, nocc, filter_z=True, z=z)
    assert kept == [p for p in window if p != window[3]]


def test_the_z_filter_refuses_without_the_pole_strengths(spectra):
    """Paired with the working call: the refusal is the missing argument, not
    the filter."""
    mol, _, nocc = spectra['water']
    z = np.full(mol.nao, 0.9)
    assert prod.valence_qp_states(mol, mol.nao, nocc, filter_z=True, z=z) \
        == _old_qp_window(mol, mol.nao, nocc)
    with pytest.raises(ValueError, match='filter_z'):
        prod.valence_qp_states(mol, mol.nao, nocc, filter_z=True, z=None)


def test_valence_needs_the_molecule(spectra):
    _, eps, nocc = spectra['water']
    with pytest.raises(ValueError, match='mol='):
        prod.resolve_qp_states(QPStates(kind='valence'), eps, nocc,
                               degeneracy_tol=DEGENERACY_TOL)


# ------------------------------------------------------------------- admitted

def test_admitted_is_exactly_what_compressible_admits(spectra):
    """The admitted set is Eq. (27) read off the mean-field spectrum, with
    omega_p = eps_p because the root is not known before the solve."""
    _, eps, nocc = spectra['water']
    resolved = prod.resolve_qp_states(QPStates(kind='admitted'), eps, nocc,
                                      degeneracy_tol=DEGENERACY_TOL)
    want = [p for p in range(len(eps)) if compressible(float(eps[p]), eps, nocc)[0]]
    assert list(resolved.explicit) == want == [2, 3, 4, 5, 6, 7, 8]
    assert resolved.label == 'admitted(gap): 7 of 24 orbitals explicit'


def test_admitted_holds_the_frontier_and_excludes_the_oxygen_core(spectra):
    """The O 1s sweeps a pole 29.6 gaps away: the pole model cannot carry it,
    and the reach says so for every orbital, not only the excluded ones."""
    _, eps, nocc = spectra['water']
    resolved = prod.resolve_qp_states(QPStates(kind='admitted'), eps, nocc,
                                      degeneracy_tol=DEGENERACY_TOL)
    frontier = ExcitedStateChain._qp_set(eps, nocc, 2, DEGENERACY_TOL)
    assert set(frontier) <= set(resolved.explicit)
    assert 0 in resolved.outside
    assert resolved.reach[0] > 1.0
    assert resolved.reach[0] == pytest.approx(29.55, abs=0.01)
    assert len(resolved.reach) == len(eps), 'recorded for every orbital'
    assert all(resolved.reach[p] < 1.0 for p in resolved.explicit)


def test_the_pole_condition_reproduces_compressible_on_the_gap(spectra):
    """`compressible` is this test with limit fixed to E_g, so at that limit
    the two must agree bitwise -- in the verdict AND in the reach."""
    for name in MOLS:
        _, eps, nocc = spectra[name]
        limit = prod.particle_hole_gap(eps, nocc)
        for p in range(len(eps)):
            got = prod.admitted_by_pole_condition(eps, nocc, p, limit)
            want = compressible(float(eps[p]), eps, nocc)
            assert got[0] is want[0], (name, p)
            assert got[1] == want[1], (name, p)


def test_omega1_refuses_without_the_lowest_drpa_root(spectra):
    """Paired with the working case: at omega1 = E_g the condition IS
    `compressible`, and the gap is only the lower bound on it."""
    _, eps, nocc = spectra['water']
    spec = QPStates(kind='admitted', threshold='omega1')
    with pytest.raises(ValueError, match='omega1'):
        prod.resolve_qp_states(spec, eps, nocc, degeneracy_tol=DEGENERACY_TOL)
    at_gap = prod.resolve_qp_states(spec, eps, nocc, omega1=prod.particle_hole_gap(eps, nocc),
                                    degeneracy_tol=DEGENERACY_TOL)
    on_gap = prod.resolve_qp_states(QPStates(kind='admitted'), eps, nocc,
                                    degeneracy_tol=DEGENERACY_TOL)
    assert at_gap.explicit == on_gap.explicit
    assert at_gap.reach == on_gap.reach


def test_the_gap_is_the_conservative_threshold(spectra):
    """Om_1 >= E_g, so the condition as written admits at least what the gap
    does: testing against the gap can only leave a state out, never let one in
    that the true Om_1 would exclude."""
    _, eps, nocc = spectra['water']
    gap = prod.particle_hole_gap(eps, nocc)
    loose = prod.resolve_qp_states(QPStates(kind='admitted', threshold='omega1'),
                                   eps, nocc, omega1=2.0 * gap,
                                   degeneracy_tol=DEGENERACY_TOL)
    on_gap = prod.resolve_qp_states(QPStates(kind='admitted'), eps, nocc,
                                    degeneracy_tol=DEGENERACY_TOL)
    assert set(on_gap.explicit) < set(loose.explicit)
    assert loose.reach[0] == pytest.approx(0.5 * on_gap.reach[0])


# ------------------------------------------------------- the resolved record

def test_all_is_every_orbital(spectra):
    _, eps, nocc = spectra['water']
    resolved = prod.resolve_qp_states(QPStates(kind='all'), eps, nocc,
                                      degeneracy_tol=DEGENERACY_TOL)
    assert list(resolved.explicit) == list(range(len(eps)))
    assert resolved.outside == ()


@pytest.mark.parametrize('spec', (QPStates(kind='admitted'),
                                  QPStates(kind='frontier'),
                                  QPStates(kind='all')))
def test_explicit_and_outside_partition_the_orbitals(spectra, spec):
    _, eps, nocc = spectra['water']
    resolved = prod.resolve_qp_states(spec, eps, nocc,
                                      degeneracy_tol=DEGENERACY_TOL)
    assert not set(resolved.explicit) & set(resolved.outside)
    assert sorted(resolved.explicit + resolved.outside) == list(range(len(eps)))
    assert list(resolved.explicit) == sorted(resolved.explicit)


def test_the_resolved_record_is_hashable(spectra):
    """A frozen set is a key: two surfaces that resolved the same declaration
    at the same spectrum must land in the same bucket."""
    _, eps, nocc = spectra['water']
    a = prod.resolve_qp_states(QPStates(kind='admitted'), eps, nocc,
                               degeneracy_tol=DEGENERACY_TOL)
    b = prod.resolve_qp_states(QPStates(kind='admitted'), eps, nocc,
                               degeneracy_tol=DEGENERACY_TOL)
    assert hash(a) == hash(b) and a == b
    assert len({a, b}) == 1


# ------------------------------------------------------------- the moved pair

def test_the_scissor_pair_is_one_object_on_both_paths():
    """`qp_space_time` re-exports the moved names rather than keeping a copy:
    two spellings of the tier assignment is how a displaced geometry stops
    matching the reference one."""
    for name in ('calibrate_scissor', 'scissor_route', 'frozen_scissor'):
        assert getattr(qp_space_time, name) is getattr(prod, name), name


def test_the_scissor_pair_answers_the_same_on_both_paths():
    roots = {0: -11.6, 1: -1.50}
    want = prod.calibrate_scissor(EPS_TIER, NOCC_TIER, roots, excluded=[0, 1, 2])
    got = qp_space_time.calibrate_scissor(EPS_TIER, NOCC_TIER, roots,
                                          excluded=[0, 1, 2])
    assert got == want
    assert want[0] == pytest.approx(-0.6) and want[2] == pytest.approx(-0.15)
    start = float(EPS_TIER[0] + 0.01)
    assert (qp_space_time.scissor_route(want, 0, EPS_TIER, NOCC_TIER, start)
            == prod.scissor_route(want, 0, EPS_TIER, NOCC_TIER, start)
            == ('scissor', want[0]))


# ------------------------------------------------------------------ the layer

def _imports(module):
    """Whether a fresh interpreter importing `module` pulls in src.gradients."""
    code = (f'import sys; import {module}; '
            "print(any(m == 'src.gradients' or m.startswith('src.gradients.') "
            'for m in sys.modules))')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, cwd='.')
    assert out.returncode == 0, out.stderr
    return out.stdout.strip() == 'True'


def test_production_does_not_reach_into_the_gradient_package():
    """Production is the layer a gradient route depends on, not the other way
    round: an import back into `src.gradients` would make the two one module
    and put the surface's frozen conventions inside the physics.

    The positive control is the same probe on the gradient module, which must
    see what this one denies -- otherwise the probe proves nothing.
    """
    assert not _imports('src.SingleReference.GW.qp_states')
    assert _imports('src.gradients.qp_space_time'), 'the probe can see it'
