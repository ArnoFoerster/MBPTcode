"""The refactor baseline still describes this tree.

`tools/refactor_gates/record_baseline.py` recorded what the GW/BSE surface code
computes -- quasiparticle energies by eight routes, BSE roots by four drivers,
the three pieces of E_0, every surface's energy, gradient and diagnostics, and
the production drivers' own JSON -- on one frozen tree, as JSON text. Every
later phase of the refactor gates its moves against that file BITWISE: Python
floats round-trip exactly through `repr`, so a matching record means the moved
code computes the same numbers and not merely close ones.

WHAT A FAILURE HERE MEANS.

  * the schema tests fail  -- the record is not usable as a gate. A missing
    section, a missing commit or tree, or a setting left as 'auto' means the
    record cannot say WHICH integral was computed, so a later disagreement
    could always be blamed on a grid nobody wrote down.
  * a live test fails     -- the working tree no longer computes what the
    baseline recorded. That is either the refactor having changed a number it
    was supposed to preserve, or the baseline having been taken on a different
    tree. Neither is a rounding question: find which moved before continuing.

The live checks run the two CHEAPEST items in the record, seconds apiece: the
mean-field surface's energy and gradient on water at both recorded geometries,
and the three E_0 terms on water/Hartree-Fock. The rest of the record --
eight-route quasiparticle audits, the BSE drivers, every surface gradient, the
campaign drivers -- is minutes to hours and runs only under
REFACTOR_BASELINE_FULL=1, by re-running the recorder and comparing its output
against the stored record with `tools/refactor_gates/compare_baselines.py`.

`tests/baseline_3f09ac0.json` is wicks' own recorded baseline, copied here
verbatim: `tools/refactor_gates/` (the recorder and `compare_baselines.py`)
is out of scope for this port, so REFACTOR_BASELINE_FULL=1 has nothing to
re-run against and that one test stays skipped.
"""
import glob
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.gradients.isdf_derivatives import exx_double_counting
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.properties.optimize import MeanFieldSurface

REPO = pathlib.Path(__file__).resolve().parent.parent
GATES = REPO / 'tools' / 'refactor_gates'
#: Where the copied baseline lives: tests/, since tools/refactor_gates/ was not
#: ported (record_baseline.py and compare_baselines.py are out of scope here).
BASELINE_DIR = pathlib.Path(__file__).resolve().parent

#: Set to 1 to re-run the whole record rather than the two cheap items.
FULL_ENV = 'REFACTOR_BASELINE_FULL'

#: Sections a record must carry to be a gate at all.
REQUIRED_SECTIONS = ('qp_routes', 'bse', 'e0_terms', 'surfaces',
                     'campaign_records', 'timing')

#: Settings that must be INTEGERS wherever they appear. A grid size held as a
#: string cannot be compared, and 'auto' cannot be rebuilt at all.
INTEGER_SETTINGS = ('ntau', 'ntau_gw', 'ntau_w', 'ntau_rpa', 'nfreq',
                    'nfreq_cd', 'nfreq_cd_after', 'nfreq_rpa', 'npade',
                    'n_start', 'nroots', 'naux', 'isdf_points', 'qp_window',
                    'n_ov', 'dense_max_nov', 'nocc', 'n_poles')

#: Where a driver records 'auto' as the PROVENANCE of a value, the resolved
#: value sits beside it under this key. The literal is then the faithful record
#: -- it says the sizer ran -- and the number the gate compares is the sibling.
AUTO_RESOLVED = {'ntau_source': 'ntau',
                 'solver': 'solver_used',
                 'solver_requested': 'solver_used',
                 'residue_route': 'residue_route_taken'}


def leaves(obj, path=''):
    """Every (path, value) leaf of a record, dicts and lists alike."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from leaves(value, f'{path}/{key}')
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from leaves(value, f'{path}[{i}]')
    else:
        yield path, obj


def dicts(obj):
    """Every dict inside a record, the record itself included."""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from dicts(value)


def own_blocks(record):
    """The record minus the driver output it embeds verbatim.

    The route audit's `ntau_source` and the state-pair driver's `ntau_w` and
    `solver` are what those scripts write today and are recorded as they are;
    the settings this test holds to integers are the recorder's own.
    """
    out = json.loads(json.dumps(record))
    for section in ('qp_routes', 'campaign_records'):
        for entry in out.get(section, {}).values():
            if isinstance(entry, dict):
                entry.pop('records', None)
    return out


def unresolved(record):
    """Every 'auto' left in the recorder's own blocks that names no resolved value.

    A setting recorded as 'auto' makes the record unable to say which grid,
    which quadrature or which solver ran, and two phases gating against it
    would be comparing different integrals. An 'auto' is acceptable only where
    it is provenance -- the resolved integer is its sibling, or the string
    itself names it, as in 'auto (minimax, 24 points)'.
    """
    out = []
    for node in dicts(own_blocks(record)):
        for key, value in node.items():
            if not isinstance(value, str):
                continue
            if not value.strip().lower().startswith('auto'):
                continue
            if any(ch.isdigit() for ch in value):
                continue
            companion = AUTO_RESOLVED.get(key)
            sibling = node.get(companion) if companion else None
            if sibling is None or str(sibling).lower().startswith('auto'):
                out.append((key, value, companion, sibling))
    return out


def newest_baseline():
    """The newest `baseline_<sha7>.json`, repeats excluded."""
    paths = [p for p in glob.glob(str(BASELINE_DIR / 'baseline_*.json'))
             if not p.endswith('_repeat.json')]
    if not paths:
        pytest.skip(f'no baseline_*.json in {BASELINE_DIR}')
    return pathlib.Path(max(paths, key=os.path.getmtime))


@pytest.fixture(scope='module')
def baseline():
    return json.loads(newest_baseline().read_text())


@pytest.fixture(scope='module')
def water(baseline):
    """The molecule the record was made on, from the record's own geometry."""
    spec = baseline['geometries']['water']
    return gto.M(atom=spec['atom'], basis=baseline['basis'], verbose=0)


def scf_factory(mol, baseline):
    """The record's own mean-field construction, to its own tolerances."""
    settings = baseline['scf']
    mf = scf.RHF(mol).density_fit(auxbasis=settings['auxbasis'])
    mf.conv_tol = settings['conv_tol']
    mf.conv_tol_grad = settings['conv_tol_grad']
    mf.max_cycle = settings['max_cycle']
    mf.kernel()
    assert mf.converged
    return mf


def at_coords(mol, coords):
    """`mol` moved onto the coordinates the record stored, in Bohr."""
    out = mol.copy()
    out.set_geom_(np.asarray(coords, float), unit='Bohr')
    return out


# ------------------------------------------------------------------- schema
def test_the_record_names_the_tree_it_describes(baseline):
    """Without a commit and a tree the record gates nothing: a number whose
    code cannot be checked out again is not a reference."""
    assert baseline['kind'] == 'gw_bse_refactor_baseline'
    assert len(baseline['sha']) == 40
    assert len(baseline['tree']) == 40
    assert baseline['sha7'] == baseline['sha'][:7]
    assert baseline['tree'] == baseline['expect_tree']


def test_every_section_is_present(baseline):
    """A hole is allowed; an ABSENT section is not, because nothing downstream
    would notice that it was never measured."""
    for section in REQUIRED_SECTIONS:
        assert section in baseline, section
        assert baseline[section], section
    assert baseline[  # a section that was not run says so in its own status
        'qp_routes'].get('status') != 'not requested on this run'


def test_no_setting_is_left_unresolved(baseline):
    """Every grid, quadrature and solver in the record is an integer or an
    explicit name."""
    assert unresolved(baseline) == []
    for path, value in leaves(own_blocks(baseline)):
        key = path.rsplit('/', 1)[-1].split('[')[0]
        if key not in INTEGER_SETTINGS or value is None:
            continue
        # A grid size may be recorded as the sizer's own phrase -- 'auto
        # (minimax, 18 points)' -- which names the integer it resolved to and
        # is therefore explicit; `unresolved` above is what rejects a bare one.
        if isinstance(value, str) and any(ch.isdigit() for ch in value):
            continue
        assert isinstance(value, int) and not isinstance(value, bool), \
            f'{path} = {value!r} is not an integer'


def test_the_isdf_grids_came_from_the_table(baseline):
    """A re-optimized grid is a DIFFERENT grid from any tabulated row, so a
    record built on one cannot be reproduced from the repository alone."""
    optimized = []
    for node in dicts(baseline):
        for element, entry in (node.get('radii_source') or {}).items():
            if entry.get('source') != 'shipped table':
                optimized.append((element, entry.get('source')))
    assert optimized == []


# --------------------------------------------------------- the cheap re-checks
def test_mean_field_surface_is_bitwise_what_was_recorded(baseline, water):
    """The whole record stands on this mean field: every chain in it reads
    these orbitals, so a move here moves everything at once."""
    recorded = baseline['surfaces']['water']['MeanFieldSurface']
    assert recorded['status'] == 'ok'
    surface = MeanFieldSurface(water, lambda mol: scf_factory(mol, baseline))
    for label, entry in recorded['geometries'].items():
        mol = at_coords(water, entry['atom_coords_bohr'])
        _, mf = surface.mean_field(mol)
        assert float(mf.e_tot) == entry['e_scf'], label
        assert float(surface.total_energy(mol, mf)) == entry['total_energy'], \
            label
        grad, energy, _ = surface.total_gradient(mol, mf)
        assert float(energy) == entry['gradient_energy'], label
        assert np.asarray(grad, float).tolist() == entry['gradient'], label


def test_the_three_e0_terms_are_bitwise_what_was_recorded(baseline, water):
    """E_0 = E_HF[rho] + E_c^dRPA, and the exact-exchange term that turns a
    Kohn-Sham energy into the first one. Every total energy on a dRPA or BSE
    surface is built from these three."""
    recorded = baseline['e0_terms']['water_HF']
    chain = RPAGroundStateChain(water, lambda mol: scf_factory(mol, baseline))
    mf = chain.mf0
    assert float(mf.e_tot) == recorded['e_scf']
    assert float(chain.reference_energy(water, mf)) \
        == recorded['reference_energy']
    assert float(exx_double_counting(mf, water)) \
        == recorded['exx_double_counting']
    assert float(chain.correlation_energy(water, mf)) == recorded['e_corr_dRPA']


# ------------------------------------------------------------ the whole record
@pytest.mark.skipif(os.environ.get(FULL_ENV) != '1',
                    reason=f'the expensive sections re-run only under '
                           f'{FULL_ENV}=1')
def test_the_whole_record_reproduces():
    """Re-run the recorder and compare its output against the stored record,
    bitwise, section by section."""
    path = newest_baseline()
    stored = json.loads(path.read_text())
    scratch = pathlib.Path(os.environ.get('TMPDIR', '/tmp')) / 'refactor_gate'
    scratch.mkdir(parents=True, exist_ok=True)
    fresh = scratch / 'baseline_live.json'
    made = subprocess.run(
        [sys.executable, str(GATES / 'record_baseline.py'),
         '--repo', str(REPO), '--expect-tree', stored['expect_tree'],
         '--out', str(fresh), '--scratch', str(scratch)],
        capture_output=True, text=True, cwd=str(REPO))
    assert made.returncode == 0, made.stderr[-4000:]
    check = subprocess.run(
        [sys.executable, str(GATES / 'compare_baselines.py'),
         '--first', str(path), '--second', str(fresh),
         '--out', str(scratch / 'merged.json')],
        capture_output=True, text=True, cwd=str(REPO))
    assert check.returncode == 0, check.stdout[-8000:]
