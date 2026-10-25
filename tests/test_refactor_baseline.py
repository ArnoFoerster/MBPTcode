"""The recorded baseline still describes this tree.

The baseline (tests/baseline_forward_routes.json) records what the GW/BSE surface
code computes (quasiparticle energies by eight routes, BSE roots by four
drivers, the three pieces of E_0, every surface's energy, gradient and
diagnostics, and the production drivers' own JSON) on one tree, as JSON text.
Python floats round-trip exactly through `repr`, so a matching record means
the code computes the same numbers, bit for bit.

A failing schema test means the record is not usable as a gate: a missing
section, or a setting left as 'auto', leaves it unable to say
which integral was computed. A failing live test means the working tree
computes something other than what the baseline recorded, which is not a
rounding question.

The live checks run the two cheapest items in the record: the mean-field
surface's energy and gradient on water at both recorded geometries, and the
three E_0 terms on water/Hartree-Fock.
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics.numerics import Numerics
from src.SingleReference.LinearResponse.rpa_energy import exx_double_counting
from src.gradients.rpa.ground_state import RPAGroundStateChain
from src.gradients.mean_field_surface import MeanFieldSurface
from tests.isdf_test_grid import g1
from tests.pins import assert_pin, recorded_blas

REPO = pathlib.Path(__file__).resolve().parent.parent
#: Where the recorded baseline lives.
BASELINE_DIR = pathlib.Path(__file__).resolve().parent

#: The E_0 terms as this tree's ISDF fit computes them.
PINS = json.loads((REPO / 'tests' / 'one_fit_pins.json').read_text())[
    'e0_terms']
PINS_BLAS = recorded_blas(REPO / 'tests' / 'one_fit_pins.json')
#: Hartree the dRPA correlation energy may sit from the record: the record
#: was taken on an earlier ISDF grid, whose E_c^dRPA sits 4.0e-5 above the
#: current one's.
E0_FIT_MOVE_HA = 1e-4

#: Sections a record must carry to be a gate at all.
REQUIRED_SECTIONS = ('qp_routes', 'bse', 'e0_terms', 'surfaces',
                     'driver_records', 'timing')

#: Settings that must be integers wherever they appear: a grid size held as
#: a string cannot be compared, and 'auto' cannot be rebuilt.
INTEGER_SETTINGS = ('ntau', 'ntau_gw', 'ntau_w', 'ntau_rpa', 'nfreq',
                    'nfreq_cd', 'nfreq_cd_after', 'nfreq_rpa', 'npade',
                    'n_start', 'nroots', 'naux', 'isdf_points', 'qp_window',
                    'n_ov', 'dense_max_nov', 'nocc', 'n_poles')

#: Where a driver records 'auto' as the provenance of a value, the resolved
#: value sits beside it under this key, and the gate compares that sibling.
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

    The QP route driver's `ntau_source` and the state-pair driver's `ntau_w` and
    `solver` are recorded as those scripts write them; the settings this test
    holds to integers are the recorder's own.
    """
    out = json.loads(json.dumps(record))
    for section in ('qp_routes', 'driver_records'):
        for entry in out.get(section, {}).values():
            if isinstance(entry, dict):
                entry.pop('records', None)
    return out


def unresolved(record):
    """Every 'auto' left in the recorder's own blocks that names no resolved value.

    A setting recorded as 'auto' leaves the record unable to say which grid,
    quadrature or solver ran. An 'auto' is acceptable only as provenance: the
    resolved value is its sibling, or the string itself names it, as in
    'auto (minimax, 24 points)'.
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
    """The recorded baseline, `baseline_forward_routes.json`."""
    return BASELINE_DIR / 'baseline_forward_routes.json'


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
def test_the_record_names_what_it_was_made_on(baseline):
    """The record says which code and which machine it was made on: the
    versions and the BLAS that produced its bits."""
    assert baseline['kind'] == 'gw_bse_refactor_baseline'
    assert baseline['recorded_on']
    assert {'python', 'numpy', 'scipy', 'pyscf'} <= set(baseline['versions'])
    assert {'numpy', 'pyscf', 'machine'} <= set(baseline['blas'])


def test_every_section_is_present(baseline):
    """A hole is allowed; an absent section is not, because nothing
    downstream would notice that it was never measured."""
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
        # a grid size may be recorded as the sizer's own phrase, 'auto
        # (minimax, 18 points)', which names its integer; `unresolved` above
        # rejects a bare one
        if isinstance(value, str) and any(ch.isdigit() for ch in value):
            continue
        assert isinstance(value, int) and not isinstance(value, bool), \
            f'{path} = {value!r} is not an integer'


def test_the_isdf_grids_came_from_the_table(baseline):
    """A re-optimized grid differs from every tabulated row, so a record
    built on one cannot be reproduced from the repository alone."""
    optimized = []
    for node in dicts(baseline):
        for element, entry in (node.get('radii_source') or {}).items():
            if entry.get('source') != 'shipped table':
                optimized.append((element, entry.get('source')))
    assert optimized == []


# --------------------------------------------------------- the cheap re-checks
def test_mean_field_surface_is_bitwise_what_was_recorded(baseline, water):
    """The mean field every chain in the record reads, bitwise the
    record's energy and gradient at both geometries."""
    recorded = baseline['surfaces']['water']['MeanFieldSurface']
    assert recorded['status'] == 'ok'
    surface = MeanFieldSurface(water, lambda mol: scf_factory(mol, baseline))
    blas = baseline['blas']
    for label, entry in recorded['geometries'].items():
        mol = at_coords(water, entry['atom_coords_bohr'])
        _, mf = surface.mean_field(mol)
        assert_pin(mf.e_tot, entry['e_scf'], blas, quantity='energy',
                   label=label)
        assert_pin(surface.total_energy(mol, mf), entry['total_energy'], blas,
                   quantity='energy', label=label)
        grad, energy, _ = surface.total_gradient(mol, mf)
        assert_pin(energy, entry['gradient_energy'], blas, quantity='energy',
                   label=label)
        assert_pin(grad, entry['gradient'], blas, quantity='gradient',
                   label=label)


def test_the_three_e0_terms_are_bitwise_what_was_recorded(baseline, water):
    """E_0 = E_HF[rho] + E_c^dRPA, and the exact-exchange term that turns a
    Kohn-Sham energy into the first one. Every total energy on a dRPA or BSE
    surface is built from these three. E_c^dRPA reads the chain's ISDF
    factors: `==` its pin, the record within `E0_FIT_MOVE_HA`."""
    recorded = baseline['e0_terms']['water_HF']
    chain = RPAGroundStateChain(water, lambda mol: scf_factory(mol, baseline), numerics=Numerics(isdf={'counts': g1(water)}))
    mf = chain.mf0
    blas = baseline['blas']
    assert_pin(mf.e_tot, recorded['e_scf'], blas, quantity='energy')
    assert_pin(chain.reference_energy(water, mf),
               recorded['reference_energy'], blas, quantity='energy')
    assert_pin(exx_double_counting(mf, water),
               recorded['exx_double_counting'], blas, quantity='energy')
    e_corr = float(chain.correlation_energy(water, mf))
    assert_pin(e_corr, PINS['water_HF']['e_corr_dRPA'], PINS_BLAS,
               quantity='energy')
    assert abs(e_corr - recorded['e_corr_dRPA']) <= E0_FIT_MOVE_HA
