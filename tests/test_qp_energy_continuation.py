"""`calc_qp_energy(continuation=...)`: the contour and the pole model in production.

`mode` is the chi0 realization and `continuation` is how that chi0 reaches the
real axis. The reference for the three continuation-free ones is
`tests/qp_energy_continuation_baseline.json`, the `qp_routes` section of
a recorded baseline of what the route audit returned on water, formaldehyde and
thioformaldehyde -- cc-pVDZ, density-fitted RHF on cc-pvdz-ri, conv_tol 1e-12,
conv_tol_grad 1e-11 -- through the gradient chain's
`qp_gradient_space_time(want_grad=False)`. The geometries, the shell counts and
the SCF energy are read out of the record itself, so the gate compares against
the run that produced it and not against a second definition of it.

THE FACTORS COME FROM THE CHAIN. `ExcitedStateChain` fits M with
`separable_ri.fit_M_stable` and `space_time.separable_factors` with
`build_separable_ri`; the two X_ao agree bitwise and the two D do not (Z differs
by 1.1e-08 relative on water). The audit tool handed every ISDF route the
chain's factors, so this gate hands production the same ones through the
`factors` hook -- a bitwise gate on the continuation cannot also carry a
different factorization.

  cd    energy AND Z bitwise on all nine rows.
  lap   energy AND Z bitwise on all nine rows. No row of the record refused or
        fell back -- every one carries `residue_route_taken` 'laplace' -- so all
        nine gate the cosh transform itself. Against `cd` the six frontier rows
        are bitwise, sweeping no residue at all, and the three homo-1 rows that
        do sweep one differ by -6.9e-14 eV (water), -3.6e-15 (formaldehyde) and
        -2.1e-14 (thioformaldehyde) in the root, and -3.2e-15, -1.6e-15 and 0 in
        Z: two backends for one equation, and the difference is the
        quadrature's. The bare Laplace fit at the frequencies the converged
        roots asked for is 1.5e-11, 1.5e-11 and 2.5e-11, against the 1e-08 of
        `LAPLACE_SCREENING_TOL`.
  sop   Z bitwise on all nine rows; the energy bitwise on five and one ulp of
        the root out on four. The cause is the Newton SEED and nothing else:
        the audit tool's chain freezes the contour root of its grid-sizing pass
        and starts the pole model's Newton there, while production starts every
        state at eps_p pushed off its own pole. Both iterate the same
        closed-form model to the same fixed point, and the last accepted step
        -- taken as soon as |step| < QP_CD_NEWTON_TOL, with a residual then of
        order its square -- rounds differently from the two starts. Seeding
        production from the contour root reproduces all nine bitwise and costs
        a full O(N^4) contour solve per state, which is the whole expense the
        pole model exists to avoid. Gated at `SOP_SEED_TOL_EV` instead, 250
        times the largest difference seen.
  pade  bitwise against the baseline's `space-time` row, which is
        `calc_qp_energy(mode='space-time')` itself. NOT against its `st-pade`
        row: that one is the gradient chain's own Pade on the CONTOUR grid
        (ntau 24, e_min half the gap), a different quadrature, and the test
        asserts the two differ.

Every gate was shown once to fail, by breaking in the source what it watches
and running this file against a backup copy, restored and `cmp`-verified. The
seven below ran against the 19 tests the file then held:

  * the residue term of Sigma dropped (`if res:` -> `if False:` in `sigma_cd`):
    3 failed, 16 passed. The cd energy gate goes on the three rows that sweep a
    residue -- water homo-1 by 0.143 eV, formaldehyde homo-1 by 0.058 eV,
    thioformaldehyde homo-1 by 1.513 eV -- and holds on the six frontier rows,
    which sweep none. The residue-COUNT gate still passes: the set is unchanged
    and only its value was thrown away, which is why the count is not a
    substitute for the energy.
  * the guard band doubled (`QP_POLE_OFFSET` -> `2 * QP_POLE_OFFSET` in
    `newton_seeds`): 5 failed, 14 passed. The Newton starts 1e-03 Ha further
    from eps_p, converges to the same root and rounds its last step
    differently, so formaldehyde and thioformaldehyde homo-1 lose bitwise
    equality (by 3.6e-15 and 1.1e-16 eV) while water's three rows round back
    onto the same bits -- and the guard diagnostic disagrees with the record on
    all three molecules, which is the gate that catches it everywhere.
  * the pole fit's stride shifted by one (`stride=sop_stride + 1` in
    `_contour_roots`): 4 failed, 15 passed. The auxiliary poles are placed from
    other columns of wc and the sop energies move by 8.3e-07, 7.1e-08 and
    3.4e-07 eV, 1e5 to 1e6 times `SOP_SEED_TOL_EV`.
  * the tau axis built on [gap, e_max] instead of reaching below the gap
    (`gap - shift` -> `gap` in `cd_frequency_grid`): 10 failed, 9 passed. It is
    the grid the chain's docstring warns about, and it moves the transform, the
    screening and therefore both contour routes.
  * the space-time default continuation switched from 'pade' to 'cd' in
    `MODE_CONTINUATIONS`: 4 failed, 15 passed. `calc_qp_energy(mode=
    'space-time')` then silently returns the contour number, which is what the
    default gate exists to catch.
  * 'n_poles' dropped from `SOP_KEYWORDS`: 1 failed, 18 passed. The keyword is
    then accepted for continuation='cd', which reads no pole model and would
    ignore it.
  * the Eq. (27) admission never refusing (`if not admits:` -> `if False:`): 1
    failed, 18 passed. The pole model then returns a number for a core state,
    which is the one failure mode `compressible` exists for.

and the three below against the 29 it holds now:

  * the cosh weights' normalization dropped (`w = 2.0 * grid.tau_weights` ->
    `w = grid.tau_weights` in `real_screening.real_frequency_weights`): 6
    failed, 23 passed. chi0(w') is then half of itself and the three rows that
    sweep a residue move by -0.064 eV (water homo-1), -0.027 (formaldehyde
    homo-1) and -0.554 (thioformaldehyde homo-1) against both the record and
    `cd` -- 1e11 times the 1e-12 eV the two backends are held to. The six
    frontier rows hold, which is why the gate runs on all nine and not on the
    three alone.
  * the validity check never firing (`if not err < self.tol:` -> `if False:` in
    `LaplaceRealScreening._weights`): 1 failed, 28 passed. Every gated row is
    below the gap, so only the refusal test sees it -- and what it sees is not a
    wrong number but a Newton walking off on a screening the grid does not
    represent and giving up after 100 steps.
  * 'laplace' silently served from the explicit backend (`if continuation ==
    'cd'` -> `in ('cd', 'laplace')` and the `LaplaceRealScreening` build skipped
    in `solve_qp_energy_contour`): 10 failed, 19 passed. Every laplace gate
    goes. The record's laplace rows are NOT cd's on the three that sweep a
    residue, the diagnostics cannot report a representation error a backend
    without one has, and the refusal never fires -- while the laplace-against-cd
    gate would pass by construction, which is why it is not the only one.
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
from src.gradients.excited_state import ExcitedStateChain

BASELINE = json.loads((pathlib.Path(__file__).resolve().parent
                       / 'qp_energy_continuation_baseline.json').read_text())

MOLECULES = ('water', 'formaldehyde', 'thioformaldehyde')

#: Largest quasiparticle energy difference the sop gate accepts, in eV. The
#: route is bitwise but for the Newton seed, whose last-step rounding is worth
#: one ulp of the root -- 3.6e-15 eV at worst on these nine rows.
SOP_SEED_TOL_EV = 1e-12

#: What the two residue backends of one contour may differ by, in eV and in Z.
#: They solve the same equation and the six frontier rows are bitwise; the
#: three that sweep a residue are 6.9e-14 eV and 3.3e-15 apart at worst.
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


@pytest.mark.parametrize('name', MOLECULES)
def test_contour_grid_reproduces_the_chain(name):
    """`cd_frequency_grid` IS the chain's `_build_cd_grid`, bitwise.

    The production helper had to be copied rather than imported -- the chain
    lives in `src.gradients`, which production may not depend on -- so the copy
    is pinned to the original on every molecule of the record: the quadrature,
    its weights, the imaginary-time axis it is the frequency side of, and the
    cosine transform between them.
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
    """The contour route, energy and pole strength, `==` against the record."""
    records, energies, z = contour_run(name, 'cd')
    for rec, energy, pole_strength in zip(records, energies, z):
        row = rec['routes']['cd']
        assert row['status'] == 'ok'
        assert energy == row['energy_eV'], (name, rec['orbital_label'],
                                            energy - row['energy_eV'])
        assert pole_strength == row['z'], (name, rec['orbital_label'],
                                           pole_strength - row['z'])


@pytest.mark.parametrize('name', MOLECULES)
def test_cd_residue_count_and_guard_match_the_record(name):
    """The residue SET and the guard band, which decide which branch was solved.

    Two routes can agree on an energy and have solved different equations; the
    residue count is what says the contour swept the same poles, and the guard
    band is the Newton path the record was written on.
    """
    mol, mf, factors, chain, records = case(name)
    _, _, diagnostics = solve_qp_energy_contour(
        mf, mol, mol.nelectron // 2, [rec['orbital'] for rec in records],
        continuation='cd', factors=factors,
        ntau=records[0]['cd_grid']['ntau_gw'])
    assert diagnostics['nfreq_cd'] == records[0]['cd_grid']['nfreq_cd']
    assert diagnostics['cd_grid_resolved']
    for rec, got in zip(records, diagnostics['states']):
        row = rec['routes']['cd']['params']
        assert got['residues'] == row['residues'], rec['orbital_label']
        assert got['pole_offset'] == row['pole_offset'], rec['orbital_label']
        assert got['sop_admits'] == row['sop_eq27_admits']


@pytest.mark.parametrize('name', MOLECULES)
def test_laplace_reproduces_the_baseline_bitwise(name):
    """The cubic residue backend, energy and pole strength, `==` the record.

    No row of the record refused or fell back -- every one carries
    `residue_route_taken` 'laplace' and no warning -- so all nine are gated as
    solved by the cosh transform and none as a fallback.
    """
    records, energies, z = contour_run(name, 'laplace')
    for rec, energy, pole_strength in zip(records, energies, z):
        row = rec['routes']['laplace']
        assert row['status'] == 'ok'
        assert row['params']['residue_route_taken'] == 'laplace'
        assert energy == row['energy_eV'], (name, rec['orbital_label'],
                                            energy - row['energy_eV'])
        assert pole_strength == row['z'], (name, rec['orbital_label'],
                                           pole_strength - row['z'])


@pytest.mark.parametrize('name', MOLECULES)
def test_laplace_and_cd_agree_below_the_gap(name):
    """Two backends for one equation: the same root and the same Z.

    The contour is identical up to where each residue reads W -- the explicit
    O(N^4) chi0(w') or the O(N^3) cosh transform of proj(tau) -- so below the
    particle-hole gap the two are one number, and the difference is the
    quadrature's and not the route's. The six frontier rows sweep no residue at
    all and are bitwise; the three that do sweep one differ by at most
    6.9e-14 eV in the root and 3.3e-15 in Z.
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
    """What the driver decided: the backend that ran, and how well it was fit.

    Two routes can agree on an energy and have solved different equations. The
    residue count says the contour swept the same poles, `residue_route_taken`
    says which backend answered them, and `representation_error` is the bare
    Laplace quadrature's residual at the frequencies the converged root
    actually asked for -- the measurable the route's validity rests on, and
    None exactly where no residue was swept.
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
    bare 1/y quadrature still carries every pair energy d -/+ w', so a state
    whose contour sweeps a residue past that is refused rather than served from
    the explicit backend under the name asked for -- a silent fallback would
    return a different functional. No row of the record was refused, so the gate
    is shown where it must fire: water's oxygen 1s sweeps a residue 19 Ha out,
    and its 2a1, 0.64 Ha out, is already past the grid's e_min. Both are states
    continuation='cd' answers.
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
    """The pole model: Z bitwise, the energy inside one ulp of the root.

    The module docstring carries the reason the energy is not bitwise on every
    row and what seeding it bitwise would cost.
    """
    records, energies, z = contour_run(name, 'sop')
    for rec, energy, pole_strength in zip(records, energies, z):
        row = rec['routes']['sop']
        assert row['status'] == 'ok'
        assert abs(energy - row['energy_eV']) <= SOP_SEED_TOL_EV, (
            name, rec['orbital_label'], energy - row['energy_eV'])
        assert pole_strength == row['z'], (name, rec['orbital_label'],
                                           pole_strength - row['z'])


def test_sop_refuses_a_state_eq27_excludes():
    """Eq. (27) excludes the oxygen 1s of water, and the refusal names it.

    No row of the record was refused -- all nine are frontier or first inner
    valence -- so the refusal is shown where it must fire: a core state sweeps
    poles many particle-hole gaps away, the individual poles of W matter there
    rather than their envelope, and no number of them converges. The contour
    serves the same state.
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
    """The default continuation, and 'pade' named, ARE today's space-time route.

    Bitwise against `calc_qp_energy(mode='space-time')` with no continuation
    named and against the record's `space-time` row. The record's `st-pade` row
    is a different quadrature -- the gradient chain's Pade on the CONTOUR grid
    -- and is asserted to differ, so that the two are never read as one number.
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
    assert default == row['energy_eV'], (name, default - row['energy_eV'])
    assert default != rec['routes']['st-pade']['energy_eV']


def test_eps_anchor_carries_the_quasiparticle_slope():
    """dw/d(eps_p) by the anchor IS the Z the contour Newton returns.

    `eps_anchor` shifts the eps_p the equation is anchored on and leaves G, W
    and the pole guard on the mean field, so by the implicit function theorem
    the derivative of the root with respect to it is exactly Z. That makes the
    keyword's one use a check on the other: a central difference of the whole
    route against the closed-form slope it reports.
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

    A refusal that is never contrasted with the call it is meant to let through
    passes for a mode that refuses everything.
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
        assert abs(energy - rec['routes'][continuation]['energy_eV']) <= \
            SOP_SEED_TOL_EV
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
