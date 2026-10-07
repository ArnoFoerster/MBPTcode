"""`calc_qp_energy(continuation=...)`: the contour and the pole model in production.

`mode` is the chi0 realization and `continuation` is how that chi0 reaches the
real axis. The reference is the `qp_routes` section of
`tests/qp_energy_continuation_baseline.json`, a record of the route audit on
water, formaldehyde and thioformaldehyde (cc-pVDZ, density-fitted RHF on
cc-pvdz-ri, conv_tol 1e-12, conv_tol_grad 1e-11) through the gradient chain's
`qp_gradient_space_time(want_grad=False)`. Geometries, shell counts and the SCF
energy are read from the record, so the gate compares against the run that
produced it.

`ExcitedStateChain` fits M with `separable_ri.fit_M_streaming` on its frozen
pair layout, which at the reference geometry is the geometry's own screen, so
its X_mo and D are `space_time.separable_factors`' bit for bit (gated below).
The gates hand production the chain's factors through the `factors` hook so
the fit is formed once per molecule.

The record was written with a fit whose Gram matrix runs over the screened
pairs alone, a different estimator wherever the screen drops pairs. The routes
are therefore pinned bitwise to a recording through the one fit
(tests/one_fit_pins.json), and the record is held within `RECORD_MOVE_EV` /
`RECORD_MOVE_Z` of it: water keeps all 576 pairs, formaldehyde drops 6 of 1444,
thioformaldehyde 114 of 1764.

  cd    energy and Z bitwise on all nine rows.
  lap   energy and Z bitwise on all nine rows. Every record row carries
        `residue_route_taken` 'laplace', so all nine gate the cosh transform
        itself. Against `cd` the six frontier rows, which sweep no residue,
        are bitwise; the three homo-1 rows that sweep one differ by at most
        6.9e-14 eV in the root and 3.2e-15 in Z, the quadrature's difference.
        The bare Laplace fit at the frequencies the converged roots ask for
        is at most 2.5e-11, against the 1e-08 of `LAPLACE_SCREENING_TOL`.
  sop   energy and Z bitwise against the pins. Against the record the root
        may be one ulp out: the audit chain starts the pole model's Newton at
        the contour root, production at eps_p pushed off its own pole; both
        reach the same fixed point, and the last accepted step (|step| <
        QP_CD_NEWTON_TOL) rounds differently from the two starts. Seeding
        production from the contour root would cost a full O(N^4) contour
        solve per state, the expense the pole model exists to avoid.
        `SOP_SEED_TOL_EV` covers the difference.
  pade  bitwise against the pinned `space-time` row, which is
        `calc_qp_energy(mode='space-time')` itself, and not against the
        `st-pade` row: that is the gradient chain's Pade on the contour grid
        (ntau 24, e_min half the gap), a different quadrature, and the test
        asserts the two differ.
"""
import json
import pathlib

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import (CD_NFREQ, HARTREE_TO_EV,
                                LAPLACE_SCREENING_TOL, QP_POLE_OFFSET)
from src.SingleReference.GW.contour_deformation import (cd_frequency_grid,
                                                        solve_qp_energy_contour)
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.space_time import separable_factors
from src.gradients.excited_state import ExcitedStateChain

BASELINE = json.loads((pathlib.Path(__file__).resolve().parent
                       / 'qp_energy_continuation_baseline.json').read_text())
#: The record's rows re-recorded through the one fit.
PINS = json.loads((pathlib.Path(__file__).resolve().parent
                   / 'one_fit_pins.json').read_text())['qp_routes']

MOLECULES = ('water', 'formaldehyde', 'thioformaldehyde')

#: How far the one fit moves each molecule's roots (eV) and pole strengths
#: from the record, with headroom: the realization alone on water, which keeps
#: every pair, and growing with the pairs the screen drops (6 on
#: formaldehyde, 114 on thioformaldehyde).
RECORD_MOVE_EV = {'water': 1e-10, 'formaldehyde': 1e-7,
                  'thioformaldehyde': 5e-5}
RECORD_MOVE_Z = {'water': 1e-11, 'formaldehyde': 5e-9,
                 'thioformaldehyde': 5e-6}
#: Added to `RECORD_MOVE_EV` for the space-time row alone, in eV: the record
#: was written on a W frequency grid that does not span the range its
#: transform is fitted over, which moves that row by -4.967e-4 (water),
#: -2.574e-4 (formaldehyde) and +4.53e-6 (thioformaldehyde).
W_GRID_MOVE_EV = {'water': 5e-4, 'formaldehyde': 2.6e-4,
                  'thioformaldehyde': 5e-6}

#: Largest quasiparticle energy difference the sop gate accepts, in eV: the
#: route differs from the record only in the Newton seed, whose last-step
#: rounding is worth one ulp of the root.
SOP_SEED_TOL_EV = 1e-12
#: How far the record's sop rows sit from the pins beyond the one fit's move:
#: the frequency pass factorizes 1 - chi0(i.nu) by Cholesky where the record
#: took an LU, and the pole fit amplifies that rounding to 8.0e-10 eV (water
#: homo-1) and 2.1e-11 in Z.
SOP_FACTORIZATION_MOVE_EV = 2e-9
SOP_FACTORIZATION_MOVE_Z = 1e-10

#: What the two residue backends of one contour may differ by, in eV and in Z.
#: They solve the same equation; only the rows that sweep a residue differ.
LAPLACE_CD_TOL_EV = 1e-12
LAPLACE_CD_TOL_Z = 1e-13

#: The anchor step of the Z cross-check, in Hartree, and the agreement a
#: central difference of a Newton root reaches against the closed-form slope.
Z_ANCHOR_STEP = 1e-4
Z_ANCHOR_TOL = 1e-6

_CASES = {}


def case(name):
    """(mol, mf, chain factors, chain, records) for one baseline molecule.

    Built once per molecule and shared, since every gate here reads the same
    mean field: two SCF solutions of one molecule differ by more than the
    continuations being compared.
    """
    if name in _CASES:
        return _CASES[name]
    records = BASELINE[name]['records']
    rec = records[0]
    mol = gto.M(atom=rec['atom'], basis=rec['basis'], verbose=0,
                max_memory=8000)

    def scf_factory(m):
        mf = scf.RHF(m).density_fit(auxbasis=rec['auxbasis'])
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-11
        mf.max_cycle = 200
        mf.kernel()
        assert mf.converged
        return mf

    mf = scf_factory(mol)
    assert mf.e_tot == rec['e_scf'], (
        f'{name}: this mean field is {mf.e_tot!r} and the record was written '
        f'on {rec["e_scf"]!r}; every number below inherits the orbitals, so '
        f'nothing bitwise can hold across that.')
    assert rec['cd_grid']['nfreq_cd'] == CD_NFREQ, (
        'the recorded contour grid grew past its starting size; the gate below '
        'builds it at CD_NFREQ')
    chain = ExcitedStateChain(mol, scf_factory, mf=mf, basis=rec['basis'],
                              auxbasis=rec['auxbasis'],
                              counts=rec['isdf']['counts'],
                              n_start=rec['isdf']['n_start'],
                              ntau_gw=rec['cd_grid']['ntau_gw'],
                              nfreq_cd=CD_NFREQ, residue_route='explicit')
    _CASES[name] = (mol, mf, chain.factors_at(mol, mf)[:2], chain, records)
    return _CASES[name]


def contour_run(name, continuation):
    """The whole baseline orbital set of one molecule, one call, with Z."""
    mol, mf, factors, chain, records = case(name)
    states = [rec['orbital'] for rec in records]
    energies, z = calc_qp_energy(mf, state=states, mode='space-time',
                                 continuation=continuation, return_z=True,
                                 factors=factors,
                                 ntau=records[0]['cd_grid']['ntau_gw'])
    return records, energies, z


def assert_pinned(name, rec, route, energy, pole_strength=None, slack=0.0,
                  z_slack=0.0):
    """`energy` (and Z) bitwise the one-fit pin of `route` for `rec`, and
    the record within the one fit's move of it (plus `slack` eV and
    `z_slack` in Z)."""
    where = (name, rec['orbital_label'], route)
    pin = PINS[name][rec['orbital_label']][route]
    row = rec['routes'][route]
    assert row['status'] == 'ok'
    assert energy == pin['energy_eV'], (where, energy - pin['energy_eV'])
    moved = abs(energy - row['energy_eV'])
    assert moved <= RECORD_MOVE_EV[name] + slack, (where, moved)
    if pole_strength is not None:
        assert pole_strength == pin['z'], (where, pole_strength - pin['z'])
        moved = abs(pole_strength - row['z'])
        assert moved <= RECORD_MOVE_Z[name] + z_slack, (where, moved)


@pytest.mark.parametrize('name', MOLECULES)
def test_the_chain_factors_are_productions(name):
    """X_mo and D of the chain are `separable_factors`' at the same grid,
    bitwise: both fit with `fit_M_streaming` on the same points and pairs."""
    mol, mf, factors, chain, records = case(name)
    rec = records[0]
    x_mo, d, _, _ = separable_factors(mf, mol, auxbasis=rec['auxbasis'],
                                      counts=rec['isdf']['counts'],
                                      n_start=rec['isdf']['n_start'])
    assert np.array_equal(factors[0], x_mo)
    assert np.array_equal(factors[1], d)


@pytest.mark.parametrize('name', MOLECULES)
def test_contour_grid_reproduces_the_chain(name):
    """`cd_frequency_grid` is the chain's `_build_cd_grid`, bitwise.

    Production may not depend on `src.gradients`, so the helper is a copy,
    pinned to the original on every molecule: the quadrature, its weights, the
    imaginary-time axis and the cosine transform between them.
    """
    mol, mf, factors, chain, records = case(name)
    eps = np.asarray(mf.mo_energy, float)
    nu, weights, grid, w0 = cd_frequency_grid(
        eps, mol.nelectron // 2, ntau=records[0]['cd_grid']['ntau_gw'],
        nfreq_cd=CD_NFREQ)
    assert np.array_equal(nu, chain.nu)
    assert np.array_equal(weights, chain.wt)
    assert w0 == chain.w0_cd
    assert np.array_equal(grid.tau_points, chain.gw_grid.tau_points)
    assert np.array_equal(grid.tau_weights, chain.gw_grid.tau_weights)
    assert np.array_equal(grid.cosft_wt, chain.gw_grid.cosft_wt)
    assert w0 == records[0]['cd_grid']['e_min_gw']


@pytest.mark.parametrize('name', MOLECULES)
def test_cd_reproduces_the_baseline_bitwise(name):
    """The contour route, energy and pole strength, `==` against the pins,
    the record within the one fit's move."""
    records, energies, z = contour_run(name, 'cd')
    for rec, energy, pole_strength in zip(records, energies, z):
        assert_pinned(name, rec, 'cd', energy, pole_strength)


@pytest.mark.parametrize('name', MOLECULES)
def test_cd_residue_count_and_guard_match_the_record(name):
    """The residue set and the guard band, which decide which branch was solved.

    Two routes can agree on an energy and have solved different equations; the
    residue count says the contour swept the same poles, and the guard band is
    the Newton path the record was written on.
    """
    mol, mf, factors, chain, records = case(name)
    _, _, diagnostics = solve_qp_energy_contour(
        mf, mol, mol.nelectron // 2, [rec['orbital'] for rec in records],
        continuation='cd', factors=factors,
        ntau=records[0]['cd_grid']['ntau_gw'])
    assert diagnostics['nfreq_cd'] == records[0]['cd_grid']['nfreq_cd']
    for rec, got in zip(records, diagnostics['states']):
        row = rec['routes']['cd']['params']
        assert got['residues'] == row['residues'], rec['orbital_label']
        assert got['pole_offset'] == row['pole_offset'], rec['orbital_label']
        assert got['sop_admits'] == row['sop_eq27_admits']


@pytest.mark.parametrize('name', MOLECULES)
def test_laplace_reproduces_the_baseline_bitwise(name):
    """The cubic residue backend, energy and pole strength, `==` the pins,
    the record within the one fit's move.

    Every record row carries `residue_route_taken` 'laplace', so all nine are
    gated as solved by the cosh transform and none as a fallback.
    """
    records, energies, z = contour_run(name, 'laplace')
    for rec, energy, pole_strength in zip(records, energies, z):
        assert rec['routes']['laplace']['params']['residue_route_taken'] == \
            'laplace'
        assert_pinned(name, rec, 'laplace', energy, pole_strength)


@pytest.mark.parametrize('name', MOLECULES)
def test_laplace_and_cd_agree_below_the_gap(name):
    """Two backends for one equation: the same root and Z below the gap.

    The contour is identical up to where each residue reads W, the explicit
    O(N^4) chi0(w') or the O(N^3) cosh transform of proj(tau), so the two
    differ only by quadrature, within `LAPLACE_CD_TOL_EV` / `LAPLACE_CD_TOL_Z`;
    rows that sweep no residue are bitwise.
    """
    records, laplace, z_laplace = contour_run(name, 'laplace')
    _, cd, z_cd = contour_run(name, 'cd')
    for rec, e_l, e_c, zl, zc in zip(records, laplace, cd, z_laplace, z_cd):
        where = (name, rec['orbital_label'])
        assert abs(e_l - e_c) <= LAPLACE_CD_TOL_EV, (where, e_l - e_c)
        assert abs(zl - zc) <= LAPLACE_CD_TOL_Z, (where, zl - zc)
        if rec['routes']['laplace']['params']['residues'] == 0:
            assert e_l == e_c and zl == zc, where


@pytest.mark.parametrize('name', MOLECULES)
def test_laplace_diagnostics_name_the_route_and_its_representation_error(name):
    """The diagnostics name the backend that ran and how well it was fit.

    The residue count says the contour swept the same poles,
    `residue_route_taken` says which backend answered them, and
    `representation_error` is the bare Laplace quadrature's residual at the
    frequencies the converged root asked for (below `LAPLACE_SCREENING_TOL`),
    None where no residue was swept.
    """
    mol, mf, factors, chain, records = case(name)
    _, _, diagnostics = solve_qp_energy_contour(
        mf, mol, mol.nelectron // 2, [rec['orbital'] for rec in records],
        continuation='laplace', factors=factors,
        ntau=records[0]['cd_grid']['ntau_gw'])
    assert diagnostics['residue_route_taken'] == 'laplace'
    for rec, got in zip(records, diagnostics['states']):
        row = rec['routes']['laplace']['params']
        assert got['residues'] == row['residues'], rec['orbital_label']
        assert got['pole_offset'] == row['pole_offset'], rec['orbital_label']
        error = got['representation_error']
        if row['residues'] == 0:
            assert error is None, rec['orbital_label']
        else:
            assert 0.0 < error < LAPLACE_SCREENING_TOL, (rec['orbital_label'],
                                                         error)


def test_laplace_refuses_a_residue_the_tau_grid_cannot_carry():
    """Above the gap the cosh transform does not exist, and the refusal says so.

    chi0(w') = int 2 cosh(w' tau) proj(tau) dtau holds only while the grid's
    bare 1/y quadrature carries every pair energy d -/+ w', so a state whose
    contour sweeps a residue past that is refused rather than silently served
    from the explicit backend. Water's oxygen 1s sweeps a residue 19 Ha out and
    its 2a1, 0.64 Ha out, is past the grid's e_min; continuation='cd' answers
    both.
    """
    mol, mf, factors, chain, records = case('water')
    isdf = dict(factors=factors, ntau=records[0]['cd_grid']['ntau_gw'])
    for state, freq in ((0, 19.2141), (1, 0.6365)):
        with pytest.raises(ValueError) as refusal:
            calc_qp_energy(mf, state=state, mode='space-time',
                           continuation='laplace', **isdf)
        message = str(refusal.value)
        assert f'orbital {state}' in message
        assert f'real frequency {freq:.4f} Ha' in message
        assert 'not carried by the tau grid' in message
        # ... and the explicit backend serves the same state on the same grid
        energy, pole_strength = calc_qp_energy(
            mf, state=state, mode='space-time', continuation='cd',
            return_z=True, **isdf)
        assert np.isfinite(energy) and 0.0 < pole_strength <= 1.0


@pytest.mark.parametrize('name', MOLECULES)
def test_sop_reproduces_the_baseline(name):
    """The pole model, energy and Z bitwise against the pins; the record
    within the one fit's move, one ulp of the root beyond it (the Newton
    seed, see the module docstring) and the factorization's move.
    """
    records, energies, z = contour_run(name, 'sop')
    for rec, energy, pole_strength in zip(records, energies, z):
        assert_pinned(name, rec, 'sop', energy, pole_strength,
                      slack=SOP_SEED_TOL_EV + SOP_FACTORIZATION_MOVE_EV,
                      z_slack=SOP_FACTORIZATION_MOVE_Z)


def test_sop_refuses_a_state_eq27_excludes():
    """Eq. (27) excludes the oxygen 1s of water, and the refusal names it.

    A core state sweeps poles many particle-hole gaps away, where the
    individual poles of W matter rather than their envelope and no number of
    them converges. The contour serves the same state.
    """
    mol, mf, factors, chain, records = case('water')
    with pytest.raises(ValueError) as refusal:
        calc_qp_energy(mf, state=0, mode='space-time', continuation='sop',
                       factors=factors, ntau=records[0]['cd_grid']['ntau_gw'])
    message = str(refusal.value)
    assert 'orbital 0' in message
    assert 'gaps away' in message and 'Eq. (27)' in message
    reach = float(message.split('sweeps a pole')[1].split()[0])
    assert reach > 1.0
    # and the state the pole model refuses is one the contour answers
    energy, pole_strength = calc_qp_energy(
        mf, state=0, mode='space-time', continuation='cd', return_z=True,
        factors=factors, ntau=records[0]['cd_grid']['ntau_gw'])
    assert energy < -500.0 and 0.0 < pole_strength <= 1.0


@pytest.mark.parametrize('name', MOLECULES)
def test_pade_is_todays_space_time_route(name):
    """The default continuation and 'pade' named are the space-time route.

    Bitwise against `calc_qp_energy(mode='space-time')` with no continuation
    named and against the pinned `space-time` row, the record's within the
    one fit's move plus `W_GRID_MOVE_EV`. The record's `st-pade` row is a
    different quadrature (the gradient chain's Pade on the contour grid) and
    is asserted to differ. At 18 points the W fit and the Sigma fits are both
    near 1e-3, and the Pade continuation turns 1e-6 Ha changes in them into
    sub-meV moves of the root.
    """
    mol, mf, factors, chain, records = case(name)
    rec = records[0]
    row = rec['routes']['space-time']
    shared = dict(state=rec['orbital'], mode='space-time', factors=factors,
                  auxbasis=rec['auxbasis'], ntau=row['params']['ntau'],
                  nfreq='auto', npade=row['params']['npade'])
    default = calc_qp_energy(mf, **shared)
    named = calc_qp_energy(mf, continuation='pade', **shared)
    assert default == named
    assert_pinned(name, rec, 'space-time', default,
                  slack=W_GRID_MOVE_EV[name])
    assert default != rec['routes']['st-pade']['energy_eV']


def test_eps_anchor_carries_the_quasiparticle_slope():
    """dw/d(eps_p) by the anchor is the Z the contour Newton returns.

    `eps_anchor` shifts the eps_p the equation is anchored on and leaves G, W
    and the pole guard on the mean field, so by the implicit function theorem
    the derivative of the root with respect to it is Z. A central difference of
    the whole route is checked against the closed-form slope it reports.
    """
    mol, mf, factors, chain, records = case('water')
    rec = records[0]
    eps = np.asarray(mf.mo_energy, float)
    p, kw = rec['orbital'], dict(state=rec['orbital'], mode='space-time',
                                 continuation='cd', factors=factors,
                                 ntau=rec['cd_grid']['ntau_gw'])
    _, z = calc_qp_energy(mf, return_z=True, **kw)
    moved = []
    for step in (Z_ANCHOR_STEP, -Z_ANCHOR_STEP):
        anchor = eps.copy()
        anchor[p] += step
        moved.append(calc_qp_energy(mf, eps_anchor=anchor, **kw))
    z_anchor = ((moved[0] - moved[1]) / HARTREE_TO_EV) / (2.0 * Z_ANCHOR_STEP)
    assert abs(z_anchor - z) < Z_ANCHOR_TOL, (z_anchor, z)


def test_validity_table_refuses_and_admits_each_pair():
    """Every (mode, continuation) refusal, each paired with the case it allows.

    A refusal not contrasted with the call it lets through would also pass for
    a mode that refuses everything.
    """
    mol, mf, factors, chain, records = case('water')
    rec = records[0]
    isdf = dict(factors=factors, ntau=rec['cd_grid']['ntau_gw'])

    # casida runs 'spectral' and nothing else, and has none of the keywords the
    # imaginary-axis realizations read.
    assert np.isfinite(calc_qp_energy(mf, state=rec['orbital'],
                                      continuation='spectral'))
    for bad, expect in ((dict(continuation='cd'), 'accepts'),
                        (dict(continuation='sop'), 'accepts')):
        with pytest.raises(ValueError, match=expect):
            calc_qp_energy(mf, state=rec['orbital'], **bad)
    for keyword in ('nfreq_cd', 'w0_cd', 'pole_offset', 'n_poles',
                    'sop_stride', 'ntau', 'counts'):
        with pytest.raises(TypeError, match=f"'spectral'.*{keyword}"):
            calc_qp_energy(mf, state=rec['orbital'], **{keyword: 1})

    # the imaginary-frequency realization continues by Pade and refuses the
    # two continuations that read proj(tau), which it never builds
    assert np.isfinite(calc_qp_energy(mf, state=rec['orbital'],
                                      mode='imagfrequency', nfreq=20,
                                      grid='minimax', greedy=True))
    with pytest.raises(NotImplementedError, match='solve_rpa_screening_df'):
        calc_qp_energy(mf, state=rec['orbital'], mode='imagfrequency',
                       continuation='cd')
    for name in ('laplace', 'sop'):
        with pytest.raises(ValueError, match='proj'):
            calc_qp_energy(mf, state=rec['orbital'], mode='imagfrequency',
                           continuation=name)

    # the space-time realization runs all four, the frontier state below the
    # gap being the one every residue backend can serve
    assert np.isfinite(calc_qp_energy(mf, state=rec['orbital'],
                                      mode='space-time',
                                      continuation='laplace', **isdf))
    with pytest.raises(ValueError, match='choose one of'):
        calc_qp_energy(mf, state=rec['orbital'], mode='space-time',
                       continuation='thiele', **isdf)

    # ... and each continuation reads its own keywords and no other's
    assert np.isfinite(calc_qp_energy(mf, state=rec['orbital'],
                                      mode='space-time', continuation='cd',
                                      nfreq_cd=CD_NFREQ, w0_cd=None,
                                      pole_offset=QP_POLE_OFFSET, **isdf))
    assert np.isfinite(calc_qp_energy(mf, state=rec['orbital'],
                                      mode='space-time', continuation='sop',
                                      n_poles=12, sop_stride=8, **isdf))
    for keyword, continuation in (('n_poles', 'cd'), ('sop_stride', 'cd'),
                                  ('npade', 'cd'), ('n_poles', 'laplace'),
                                  ('npade', 'laplace'), ('nfreq', 'sop'),
                                  ('greedy', 'sop')):
        with pytest.raises(TypeError, match=f"'{continuation}'.*{keyword}"):
            calc_qp_energy(mf, state=rec['orbital'], mode='space-time',
                           continuation=continuation, **dict(isdf,
                                                             **{keyword: 1}))
    for keyword in ('nfreq_cd', 'w0_cd', 'pole_offset', 'n_poles',
                    'sop_stride'):
        with pytest.raises(TypeError, match=f"'pade'.*{keyword}"):
            calc_qp_energy(mf, state=rec['orbital'], mode='space-time',
                           continuation='pade', **dict(isdf, **{keyword: 1}))

    # the contour's Newton has no root selector for qp_solver to set
    with pytest.raises(ValueError, match='qp_solver'):
        calc_qp_energy(mf, state=rec['orbital'], mode='space-time',
                       continuation='cd', qp_solver='graphical', **isdf)


def test_return_z_follows_the_continuation():
    """Z comes back from the two routes whose Newton computes one, and only those.

    A Z differenced from a Thiele continuation is noise -- forward mode through
    the recursion divides by inverse differences approaching zero -- and the
    Casida route's root finder returns the root alone.
    """
    mol, mf, factors, chain, records = case('water')
    rec = records[0]
    isdf = dict(factors=factors, ntau=rec['cd_grid']['ntau_gw'])
    for continuation in ('cd', 'laplace', 'sop'):
        energy, z = calc_qp_energy(mf, state=rec['orbital'], mode='space-time',
                                   continuation=continuation, return_z=True,
                                   **isdf)
        pin = PINS['water'][rec['orbital_label']][continuation]['energy_eV']
        assert abs(energy - pin) <= SOP_SEED_TOL_EV
        assert 0.0 < z <= 1.0
    with pytest.raises(ValueError, match='not differentiable'):
        calc_qp_energy(mf, state=rec['orbital'], mode='space-time',
                       return_z=True, **isdf)
    with pytest.raises(ValueError, match='not differentiable'):
        calc_qp_energy(mf, state=rec['orbital'], return_z=True)
    # a list of states brings back a list of each
    energies, zs = calc_qp_energy(mf, state=[rec['orbital']], mode='space-time',
                                  continuation='cd', return_z=True, **isdf)
    assert len(energies) == len(zs) == 1
