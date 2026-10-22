"""Gates for the frozen scissor: what the orbitals outside the quasiparticle
set carry.

`ExcitedStateChain` solves the quasiparticle equation for the orbitals in its
set and puts those roots on the BSE diagonal. Every other orbital carries the
frozen shift `calibrate_scissor` reads off the explicit roots at the reference
geometry -- explicit inside the window, the scissor outside it. The bare
mean-field eigenvalue outside the set (`outside='mean-field'`) and the
mean-field diagonal (`at_mean_field`) are refused.

WHAT THE SCISSOR IS. Water/cc-pVDZ, RHF/cc-pVDZ-ri, the default window
(qp_set 3-6, so orbitals 0-2 and 7-23 outside). `calibrate_scissor` gives every
outside orbital the correction of the explicit orbital nearest it in orbital
energy, which on this set is two tiers:

    occupied outside (0, 1, 2)     +0.9798 eV
    virtual outside (7 ... 23)     -0.3169 eV

THE SHIFT IS A CONSTANT, so d eps^QP_p/dR = d eps_p/dR outside the set and the
adjoint chain needs no term of its own. The finite-difference gate below is
what says so, at the step and the tolerance tests/test_excited_state.py uses:
h = 1e-4, four-point, worst deviation relative to the largest force < 1e-6.

THE PINS are the chain's surface with `outside='scissor'`, computed once and
pinned bitwise (`PINS`); the recorded baseline holds the mean-field-outside
surface and is read for its geometries and SCF energies only.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import json
import pathlib

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics import Numerics
from src.Base.constants import HARTREE_TO_EV
from src.Base.declaration import Excitation, GroundState
from src.SingleReference.GW.qp_states import calibrate_scissor, frozen_scissor
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.rpa_bse_surface import RPABSESurface
from src.properties.surfaces import potential_energy_surface
from tests.pins import assert_pin, assert_same, recorded_blas
from tests.isdf_test_grid import g1_numerics, g1_numerics_kw

#: The directory holding the recorded baseline and the pins this file reads.
GATES = pathlib.Path(__file__).resolve().parent

#: The two tiers the default window calibrates to on water, in eV. Gated so
#: the numbers in the docstring are measured rather than remembered.
WATER_TIERS_EV = (0.979797900918945, -0.3168834834722673)

#: The finite-difference step and tolerance tests/test_excited_state.py gates
#: its own gradients at: a frozen scissor only adds a constant to the chain.
FD_STEP = 1e-4
FD_TOL = 1e-6


def newest_baseline():
    """The newest `baseline_<sha7>.json`, repeats excluded."""
    paths = [p for p in glob.glob(str(GATES / 'baseline_*.json'))
             if not p.endswith('_repeat.json')]
    if not paths:
        pytest.skip(f'no baseline_*.json in {GATES}; run record_baseline.py')
    return pathlib.Path(max(paths, key=os.path.getmtime))


RECORD = json.loads(newest_baseline().read_text())
RECORD_BLAS = recorded_blas(newest_baseline())
#: The record's energies and gradients as this tree computes them.
PINS = json.loads(
    (GATES / 'one_fit_pins.json').read_text())['outside_scissor']
PINS_BLAS = recorded_blas(GATES / 'one_fit_pins.json')


def scf_factory(mol):
    """The baseline record's own mean field, to its own tolerances, read
    off the record."""
    settings = RECORD['scf']
    mf = scf.RHF(mol).density_fit(auxbasis=settings['auxbasis'])
    mf.conv_tol = settings['conv_tol']
    mf.conv_tol_grad = settings['conv_tol_grad']
    mf.max_cycle = settings['max_cycle']
    mf.kernel()
    assert mf.converged
    return mf


def molecule(name):
    """One of the record's own molecules, at its own geometry and basis."""
    spec = RECORD['geometries'][name]
    return gto.M(atom=spec['atom'], basis=RECORD['basis'], verbose=0)


def at_coords(mol, coords):
    """`mol` moved onto the coordinates the record stored, in Bohr."""
    out = mol.copy()
    out.set_geom_(np.asarray(coords, float), unit='Bohr')
    return out


def outside_of(chain, norb):
    """The orbitals with no quasiparticle equation of their own."""
    inside = {int(p) for p in chain.resolved.qp_set}
    return [p for p in range(norb) if p not in inside]


@pytest.fixture(scope='module')
def water():
    """The record's water, with the mean field every chain below reads."""
    mol = molecule('water')
    return mol, scf_factory(mol)


# ------------------------------------------------- (a) the scissor, bitwise
@pytest.mark.parametrize('name,spin', [('water', 'singlet'),
                                       ('water', 'triplet'),
                                       ('formaldehyde', 'singlet'),
                                       ('formaldehyde', 'triplet')])
def test_the_chain_is_its_scissor_surface_bitwise(name, spin):
    """The chain built with no outside keyword is the scissor surface:
    energy and gradient bitwise the pins at both of
    the record's geometries (within the bar of each quantity class on another BLAS), the reference geometry evaluated first (the
    conventions are frozen there and the displaced geometry spends them)."""
    recorded = RECORD['surfaces'][name][f'ExcitedStateChain[{spin}]']
    assert recorded['status'] == 'ok'
    pins = PINS[f'{name}-{spin}']
    mol = molecule(name)
    chain = ExcitedStateChain(mol, scf_factory, spin=spin, numerics=g1_numerics(mol))
    for label, entry in recorded['geometries'].items():
        pin = pins[label]
        at = mol if label == 'R0' else at_coords(mol, entry['atom_coords_bohr'])
        assert at.atom_coords().tolist() == entry['atom_coords_bohr'], label
        _, mf = chain.mean_field(at)
        assert_pin(mf.e_tot, entry['e_scf'], RECORD_BLAS, quantity='energy',
                   label=label)
        assert_pin(chain.total_energy(at, mf), pin['total_energy'], PINS_BLAS,
                   quantity='energy', label=label)
        grad, energy, _ = chain.total_gradient(at, mf)
        assert_pin(energy, pin['gradient_energy'], PINS_BLAS,
                   quantity='energy', label=label)
        assert_pin(grad, pin['gradient'], PINS_BLAS, quantity='gradient',
                   label=label)
    assert chain.resolved.outside_shift


# -------------------------------------------- (b) nothing outside, nothing to do
def test_with_every_orbital_explicit_the_scissor_is_the_mean_field(water):
    """`qp_window='all'` leaves no orbital outside: no shift is calibrated
    and the BSE diagonal is the explicit roots alone. A scissor that moved a
    number here would be shifting orbitals that carry their own root."""
    mol, mf = water
    sciss = ExcitedStateChain(
        mol, scf_factory, mf=mf, numerics=g1_numerics(mol, Numerics(qp={'window': 'all'})))
    _, pieces = sciss._forward(mol, mf)
    assert sciss.resolved.outside_shift == {}
    # a second forward solves the roots again: to a few ulp, not bitwise
    assert_same(sciss._forward(mol, mf)[1][7], pieces[7], quantity='energy',
                label='eps_qp')


# ------------------------------------------------- (c) the shift, and what it moves
@pytest.mark.parametrize('spin', ['singlet', 'triplet'])
def test_the_outside_orbitals_carry_the_calibrated_shift(water, spin):
    """eps^QP_p = eps_p + shift(p) outside the set, the explicit roots untouched.

    `shift` is `calibrate_scissor` on the chain's OWN roots, called here
    directly: the chain may not invent a calibration of its own. The two
    tiers are in this module's docstring.
    """
    mol, mf = water
    sciss = ExcitedStateChain(mol, scf_factory, mf=mf, spin=spin, numerics=g1_numerics(mol))
    _, pieces = sciss._forward(mol, mf)
    eps, eps_qp = pieces[6], pieces[7]
    qp_set = [int(p) for p in sciss.resolved.qp_set]
    roots = {p: float(eps_qp[p]) for p in qp_set}
    outside = outside_of(sciss, len(eps))
    expect = calibrate_scissor(eps, sciss.nocc, roots, outside)
    assert sciss.resolved.outside_shift == expect
    for p in outside:
        assert eps_qp[p] == eps[p] + frozen_scissor(expect, p), p
    tiers = sorted({v for v in expect.values()}, reverse=True)
    assert np.allclose([t * HARTREE_TO_EV for t in tiers], WATER_TIERS_EV,
                       atol=1e-6)


# ------------------------------------------------------- (d) the force follows it
def test_the_scissor_gradient_follows_the_energy(water):
    """dE_ex/dR with the frozen scissor, against a finite difference of E_ex.

    A frozen shift is a constant, so the adjoint chain carries no term for it.
    The finite difference differences a surface whose outside orbitals carry
    the reference geometry's shift at every displaced geometry.
    """
    mol, mf = water
    chain = ExcitedStateChain(mol, scf_factory, mf=mf, numerics=g1_numerics(mol))
    grad, _, diags = chain.total_gradient(mol, mf)
    assert diags['outside'] == 'scissor'
    outside = outside_of(chain, len(mf.mo_energy))
    assert len(diags['outside_shift_ev']) == len(outside)
    worst = 0.0
    for ia in range(mol.natm):
        for x in range(3):
            v = []
            for k in (-2, -1, 1, 2):
                step = np.zeros((mol.natm, 3))
                step[ia, x] = k * FD_STEP
                m = mol.copy()
                m.set_geom_(mol.atom_coords() + step, unit='Bohr')
                m.build(False, False)
                v.append(chain.energy(m)[0])
            fd = (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * FD_STEP)
            worst = max(worst, abs(fd - grad[ia, x]))
    assert worst / np.abs(grad).max() < FD_TOL


# --------------------------------------------------------- (e) refreeze rebuilds it
def test_refreeze_recalibrates_the_shift_at_the_new_geometry(water):
    """A refrozen chain calibrates its own scissor and records the drift:
    the new shifts and the ones they replaced are both readable, so the
    optimizer can measure how the conventions it walked on moved.
    """
    mol, mf = water
    recorded = RECORD['surfaces']['water']['ExcitedStateChain[singlet]']
    chain = ExcitedStateChain(mol, scf_factory, mf=mf, numerics=g1_numerics(mol))
    chain.excitation(mol, mf)
    before = dict(chain.resolved.outside_shift)
    moved_to = at_coords(mol, recorded['geometries']['R0+0.05x_atom0']
                         ['atom_coords_bohr'])
    fresh = chain.refreeze(moved_to)
    assert fresh.resolved.outside_shift is None
    fresh.excitation()
    assert set(fresh.resolved.outside_shift) == set(before)
    assert all(fresh.resolved.outside_shift[p] != before[p] for p in before)
    record = fresh.conventions.outside_record()
    assert record['outside_shift_before_ev'] == {
        p: v * HARTREE_TO_EV for p, v in sorted(before.items())}
    assert record['outside_shift_ev'] == {
        p: v * HARTREE_TO_EV for p, v in sorted(fresh.resolved.outside_shift.items())}
    assert record['outside_shift_moved_ev'] > 0.0


def test_the_composed_surface_forwards_and_refreezes_the_treatment(water):
    """`RPABSESurface`'s excited half carries the scissor, and so does the
    excited half of its refrozen surface, calibrated at its own geometry.

    The dRPA ground state has no BSE diagonal, so the scissor is the excited
    half's alone -- but it is the SURFACE that an optimizer refreezes.
    """
    mol, mf = water
    recorded = RECORD['surfaces']['water']['ExcitedStateChain[singlet]']
    surface = RPABSESurface(mol, scf_factory, mf=mf, numerics=g1_numerics(mol))
    surface.total_energy(mol, mf)
    assert surface.excited.resolved.outside_shift
    moved_to = at_coords(mol, recorded['geometries']['R0+0.05x_atom0']
                         ['atom_coords_bohr'])
    refrozen = surface.refreeze(moved_to)
    refrozen.total_energy(moved_to)
    assert refrozen.excited.resolved.outside_shift
    assert refrozen.excited.resolved.outside_shift != surface.excited.resolved.outside_shift


# ------------------------------------------------------------ (f) the entry point
def test_the_entry_point_asks_for_the_scissor_and_records_it(water):
    """`potential_energy_surface` declares the outside treatment and gets it.

    Against the chain built with the same arguments by hand: the recorded
    request is what the chain ran. Each converges its own SCF, so to a few
    ulp (`assert_same` of tests/pins.py), not bitwise.
    """
    mol, _ = water
    surface = potential_energy_surface(mol, scf_factory,
                                       ground_state=GroundState('dft', 'hf'),
                                       excitation=Excitation('singlet'),
                                       chi0='space-time')
    assert surface.realization.outside_treatment == 'scissor'
    direct = ExcitedStateChain(
        mol, scf_factory, spin='singlet', state=0,
        numerics=g1_numerics(mol, Numerics(isdf={'counts': surface.counts, 'n_start': surface.n_start},
                                           qp={'window': surface.realization.qp_explicit, 'scissor': 'calibrate', 'residue_route': 'explicit'},
                                           bse={'solver': 'dense'})))
    assert_same(surface.total_energy(), direct.total_energy(),
                quantity='energy', label='total energy')


# ------------------------------------------------------------------ (g) refusals
@pytest.mark.parametrize('retired', [{'outside': 'mean-field'},
                                     {'outside': 'scissor'},
                                     {'at_mean_field': True}])
def test_the_retired_outside_treatments_are_refused(water, retired):
    """The mean-field outside and the mean-field diagonal are refused: a
    caller naming either keyword is refused, not silently given the
    scissor."""
    mol, mf = water
    with pytest.raises(TypeError, match=next(iter(retired))):
        ExcitedStateChain(mol, scf_factory, mf=mf, **g1_numerics_kw(mol, retired))
