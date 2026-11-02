"""The stage pipeline's solvent scheme and external S0 Hessian, without a
GW-BSE evaluation.

  1. one scheme per run: 'gas' in the gas phase and only there; a continuum
     names 'scheme-a' or a scheme-B one (the command line
     refuses none of them and an unnamed one); the 'noneq' scheme declares
     its relaxed states in non-equilibrium and 'eq' in equilibrium; scheme B refuses each choice the loop does not run
     (SCHEME_B_REFUSES: among them a quasiparticle set selected per
     geometry), and the loop tolerances belong to scheme B alone.
  2. the stamps: a scheme-B stage carries the scheme and the loop's
     tolerances, the other schemes' stamps hold no such key (their stores
     resume as written), and a record of the other scheme in the same store
     is refused.
  3. the external Hessian: an ORCA .hess at R0 gives the hess record's modes
     (those of the file's matrix), its sha256 is in the hess stamp; a file at
     another geometry, or changed after it was declared, is refused, and the
     route and the file are declared together.
  4. `free_energies`: each root at its spin's anchor plus its splitting in
     the property field, omega against e0, the field's numbers kept.
  5. the walk arms: the bare-exchange schemes build singlets without the
     linear-response term; a non-equilibrium scheme's excited surfaces run
     in S0's slow charges; the Tamm-Dancoff T1 minimum is declared by a
     scheme-B full-BSE run alone and stamped on what reads t1; a walk stopped
     (or raising) at the triplet instability goes to the single point there;
     the loop tied to the gradient by name, stamped, and its surfaces handed
     the walk's convergence bar.

Run: python -m pytest tests/test_photophysics_scheme_b.py
"""
from dataclasses import replace

import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics.constants import (EXTERNAL_HESSIAN_GEOMETRY_TOL_BOHR,
                                         GEOM_OPT_CONV, LOOP_FORCE_FRACTION,
                                         PHOTOPHYSICS_LOOP_TOL_ENERGY,
                                         PHOTOPHYSICS_LOOP_TOL_Q)
from src.Base.declaration import QPStates
from src.photophysics import stages, walks
from src.photophysics.declaration import (DeclarationRefused,
                                          ExternalHessian, UiscModel)
from src.photophysics.records import StageRefused
from src.photophysics.store import StoreRefused
from src.properties.duschinsky import CartesianHessian
from src.properties.vibronic import normal_modes
from tests.photophysics_cases import declaration, run_at_r0, run_in
from tests.test_duschinsky import write_orca_hessian

TOLUENE = ('--environment', 'toluene', '--relaxed-solvation', 'equilibrium')
SCHEME_B = TOLUENE + ('--solvent-scheme', 'scheme-b-eq', '--scf', 'df',
                      '--qp-states', 'frontier')
#: Formaldehyde bent out of plane: six pair springs hold all six internal
#: modes (`spring_hessian`).
BENT = ('--atom', 'C 0 0 -0.5296; O 0 0 0.6742; H 0 0.9429 -1.1123; '
        'H 0.3 -0.9429 -1.1123')


# ------------------------------------------------------------- the scheme
def test_the_scheme_matches_the_environment():
    assert declaration().solvent_scheme == 'gas'
    assert declaration(*SCHEME_B).scheme_b
    with pytest.raises(SystemExit, match='takes none'):
        declaration('--solvent-scheme', 'scheme-a')
    with pytest.raises(SystemExit, match='needs --solvent-scheme'):
        declaration(*TOLUENE)
    d = declaration(*TOLUENE, '--solvent-scheme', 'scheme-a')
    with pytest.raises(DeclarationRefused, match='gas'):
        replace(d, solvent_scheme='gas')
    with pytest.raises(DeclarationRefused, match='gas'):
        replace(declaration(), solvent_scheme='scheme-a')
    with pytest.raises(DeclarationRefused, match='solvent_scheme'):
        replace(d, solvent_scheme='pcm')


@pytest.mark.parametrize('change,why', [
    (dict(ground='rpa'), 'ground'),
    (dict(policies={'df_check': True}), 'df_check'),
    (dict(policies={'scf_finish': 'newton'}), 'scf_finish'),
    (dict(policies={'relaxed_solvation': 'nonequilibrium'}),
     'relaxed_solvation'),
    (dict(rate_model=UiscModel(1, 0.1, 2, 2, (0.1, 0.1))), 'rate_model'),
    (dict(qp_states=QPStates('admitted')), 'qp_states'),
    (dict(qp_states=QPStates('adaptive', targets=(('singlet', 0),
                                                 ('triplet', 0)))),
     'qp_states'),
    (dict(qp_states=QPStates('valence', filter_z=True)), 'qp_states'),
])
def test_scheme_b_refuses_what_the_loop_does_not_run(change, why):
    d = declaration(*SCHEME_B)
    policies = change.pop('policies', None)
    if policies is not None:
        change['policies'] = replace(d.policies, **policies)
    numerics = change.pop('numerics', None)
    if numerics is not None:
        change['numerics'] = d.numerics.derive('a choice scheme B refuses',
                                               **numerics)
    with pytest.raises(DeclarationRefused, match=f'scheme-b-eq: .*{why}'):
        replace(d, **change)


@pytest.mark.parametrize('extra,why', [
    (('--scf', 'isdf'), 'route'),
    (('--span', '1e-6'), 'span'),
    (('--scf-grad-tol', '1e-8'), 'conv_tol_grad')])
def test_scheme_b_refuses_numerics_the_loop_does_not_run(extra, why):
    with pytest.raises(DeclarationRefused, match=f'scheme-b-eq: .*{why}'):
        declaration(*SCHEME_B, *extra)


def test_four_named_schemes():
    """Each scheme by its name, with the relaxed states its loop declares;
    the old unnamed default and a mismatched relaxed solvation are refused."""
    noneq = declaration(*TOLUENE[:2], '--solvent-scheme', 'scheme-b-noneq',
                        '--relaxed-solvation', 'nonequilibrium', '--scf', 'df',
                        '--qp-states', 'frontier')
    assert noneq.scheme_b and noneq.solvation == 'nonequilibrium'
    with pytest.raises(DeclarationRefused, match='scheme-b-noneq: .*relaxed'):
        replace(noneq, policies=replace(noneq.policies,
                                        relaxed_solvation='equilibrium'))
    with pytest.raises(SystemExit):
        declaration(*TOLUENE, '--solvent-scheme', 'duchemin')
    a = declaration(*TOLUENE, '--solvent-scheme', 'scheme-a')
    assert not a.scheme_b and a.solvent_scheme == 'scheme-a'


def test_every_record_names_its_scheme(tmp_path):
    for d in (declaration(), declaration(*SCHEME_B)):
        run = run_in(tmp_path / d.solvent_scheme, d)
        run.write('s0', {'coords_bohr': run.mol_in.atom_coords()})
        assert run.read('s0')['solvent_scheme'] == d.solvent_scheme
    eq = run_in(tmp_path / 'b', declaration(*SCHEME_B))
    eq.write('s0', {'coords_bohr': eq.mol_in.atom_coords()})
    noneq = run_in(tmp_path / 'b', declaration(
        *TOLUENE[:2], '--solvent-scheme', 'scheme-b-noneq',
        '--relaxed-solvation', 'nonequilibrium', '--scf', 'df',
        '--qp-states', 'frontier'))
    with pytest.raises(StoreRefused, match='solvent_scheme'):
        noneq.read('s0')


def test_scheme_b_fits_the_factors_by_rows():
    """Every scheme fits the factors by rows, scheme B too."""
    assert declaration(*SCHEME_B).numerics.isdf.fit == 'rows'
    assert declaration().numerics.isdf.fit == 'rows'


def test_the_loop_tolerances_belong_to_scheme_b():
    loop = declaration(*SCHEME_B, '--scf-conv-tol', '1e-12').numerics.solvent
    assert (loop.loop_tol_q, loop.loop_tol_energy) == (
        PHOTOPHYSICS_LOOP_TOL_Q, PHOTOPHYSICS_LOOP_TOL_ENERGY)
    # G settles no further than the SCF's energy: the default follows it,
    # and a tighter loop is refused
    loose = declaration(*SCHEME_B, '--scf-conv-tol', '1e-10').numerics
    assert loose.solvent.loop_tol_energy == 1e-10
    with pytest.raises(DeclarationRefused, match='loop_tol_energy'):
        declaration(*SCHEME_B, '--scf-conv-tol', '1e-10',
                    '--loop-tol-energy', '1e-11')
    given = declaration(*SCHEME_B, '--scf-conv-tol', '1e-12',
                        '--loop-tol-energy', '1e-12').numerics
    assert given.solvent.loop_tol_energy == 1e-12
    # the other schemes run no loop: no solvent section, the same numerics
    scheme_a = declaration(*TOLUENE, '--solvent-scheme', 'scheme-a')
    assert scheme_a.numerics.identity() == declaration().numerics.identity()
    with pytest.raises(SystemExit, match='scheme-b'):
        declaration(*TOLUENE, '--solvent-scheme', 'scheme-a',
                    '--loop-tol-q', '1e-9')


# ------------------------------------------------- the quasiparticle Hamiltonian
def test_scheme_b_solves_the_quasiparticle_hamiltonian_by_default():
    """Scheme B's loop moves the charges under which a near-degenerate
    explicit pair (DMABN's LUMO and LUMO+1) mixes, so the pair has to be one
    frozen group: scheme B declares `qp.hamiltonian` on unless the command
    line turns it off; the other schemes keep it off. The choice is in the
    numerics identity the vertical stage's stamp carries."""
    on = declaration(*SCHEME_B)
    assert on.numerics.qp.hamiltonian
    off = declaration(*SCHEME_B, '--no-qp-hamiltonian')
    assert not off.numerics.qp.hamiltonian
    assert not declaration(*TOLUENE, '--solvent-scheme', 'scheme-a'
                           ).numerics.qp.hamiltonian
    assert declaration(*TOLUENE, '--solvent-scheme', 'scheme-a',
                       '--qp-hamiltonian').numerics.qp.hamiltonian
    assert on.numerics.identity() != off.numerics.identity()


# ------------------------------------------------------------- the stamps
def test_one_scheme_per_store(tmp_path):
    gas = run_in(tmp_path, declaration())
    scheme_a = run_in(tmp_path, declaration(*TOLUENE, '--solvent-scheme',
                                            'scheme-a'))
    scheme_b = run_in(tmp_path, declaration(*SCHEME_B, '--scf-conv-tol',
                                            '1e-12'))
    for run in (gas, scheme_a):
        assert 'solvent_scheme' not in run.stamp('s0')
    stamped = scheme_b.stamp('t1')['solvent_scheme']
    assert stamped['scheme'] == 'scheme-b-eq'
    assert stamped['loop']['loop_tol_energy'] == PHOTOPHYSICS_LOOP_TOL_ENERGY
    # the two continuum schemes share the store's directory and refuse
    # each other's records
    assert scheme_a.store.dir == scheme_b.store.dir
    scheme_a.write('s0', {'coords_bohr': scheme_a.mol_in.atom_coords()})
    with pytest.raises(StoreRefused, match='solvent_scheme'):
        scheme_b.read('s0')


# ------------------------------------------------------ external Hessian
def spring_hessian(mol, k=0.3):
    """A Hessian with real, positive internal modes: a spring of constant k
    on every pair of atoms at its rest length."""
    x = mol.atom_coords()
    h = np.zeros((3 * mol.natm, 3 * mol.natm))
    for i in range(mol.natm):
        for j in range(i + 1, mol.natm):
            u = (x[i] - x[j]) / np.linalg.norm(x[i] - x[j])
            block = k * np.outer(u, u)
            for a, b, sign in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
                h[3 * a:3 * a + 3, 3 * b:3 * b + 3] += sign * block
    return h


def external(tmp_path, mol, h, shift=0.0, name='s0.hess'):
    """The ExternalHessian of `h` written at `mol`, its first atom moved
    `shift` Bohr along x."""
    at = mol.copy()
    coords = mol.atom_coords().copy()
    coords[0, 0] += shift
    at.set_geom_(coords, unit='Bohr')
    path = tmp_path / name
    write_orca_hessian(path, CartesianHessian(at, h))
    return ExternalHessian.of(path)


def external_run(tmp_path, shift=0.0):
    mol = declaration_mol()
    ext = external(tmp_path, mol, spring_hessian(mol), shift)
    d = declaration(*BENT, '--hessian', 'external', '--hessian-file', ext.path)
    return run_at_r0(tmp_path / 'store', d), ext


def declaration_mol():
    """The tests' molecule at its input geometry, R0 of `run_at_r0`."""
    d = declaration(*BENT)
    return gto.M(atom=d.system.atom, basis=d.system.basis, verbose=0)


def test_the_external_hessian_gives_the_modes(tmp_path):
    run, ext = external_run(tmp_path)
    record = stages.ground_modes(run)
    assert record['route'] == 'external'
    assert record['info']['sha256'] == ext.sha256
    assert record['info']['max_offset_bohr'] < 1e-6
    omega = normal_modes(None, run.mol0, hess=spring_hessian(run.mol0))[0]
    assert np.abs(record['omega'] - omega).max() < 1e-9
    assert run.stamp('hess')['external_hessian'] == ext.sha256
    assert 'external_hessian' not in run_in(tmp_path / 'other',
                                            declaration()).stamp('hess')


def test_an_external_hessian_off_r0_is_refused(tmp_path):
    run, _ = external_run(tmp_path, shift=10 * EXTERNAL_HESSIAN_GEOMETRY_TOL_BOHR)
    with pytest.raises(StageRefused, match='another geometry'):
        stages.ground_modes(run)


def test_a_changed_external_hessian_is_refused(tmp_path):
    run, ext = external_run(tmp_path)
    with open(ext.path, 'a') as fh:
        fh.write('\n')
    with pytest.raises(StageRefused, match='not the file declared'):
        stages.ground_modes(run)


def test_the_route_and_the_file_come_together(tmp_path):
    mol = declaration_mol()
    ext = external(tmp_path, mol, spring_hessian(mol))
    with pytest.raises(SystemExit, match='hessian-file'):
        declaration('--hessian', 'external')
    with pytest.raises(SystemExit, match='hessian-file'):
        declaration('--hessian-file', ext.path)
    d = declaration()
    with pytest.raises(DeclarationRefused, match='external_hessian'):
        replace(d, policies=replace(d.policies, external_hessian=ext))


# --------------------------------------------------------- free energies
def test_free_energies_anchor_each_spin():
    props = {'e0': -1.0, 'energy': {}, 'omega': {
        'S1': 0.20, 'S2': 0.25, 'T1': 0.15, 'T2': 0.22}}
    out = stages.free_energies(props, {'S1': -0.7, 'T1': -0.8}, -1.1)
    assert out['energy']['S2'] == pytest.approx(-0.65)
    assert out['energy']['T2'] == pytest.approx(-0.73)
    assert out['omega']['S1'] == pytest.approx(0.4)
    assert out['field_omega'] == props['omega'] and out['field_e0'] == -1.0
    # one anchor: every root from it
    one = stages.free_energies(props, {'S1': -0.7}, -0.9)
    assert one['energy']['T1'] == pytest.approx(-0.75)
    assert one['omega']['S1'] == pytest.approx(0.2)



# -------------------------------------------------------- the walk arms
NONEQ = TOLUENE[:2] + ('--relaxed-solvation', 'nonequilibrium', '--scf', 'df',
                       '--qp-states', 'frontier')


def test_the_bare_and_non_equilibrium_arms_by_name():
    """scheme-b-*-bare: no linear response on the singlet (the loop's
    medium alone); scheme-b-noneq(-bare): the excited surfaces are G_neq(X |
    S0), their loop run in S0's slow charges (`SolvatedSurface(ground=)`),
    S0's own G_eq."""
    eq, bare = declaration(*SCHEME_B), declaration(
        *SCHEME_B, '--solvent-scheme', 'scheme-b-eq-bare')
    noneq = declaration(*NONEQ, '--solvent-scheme', 'scheme-b-noneq-bare')
    assert eq.scheme_b_lr and not bare.scheme_b_lr and not noneq.scheme_b_lr
    assert noneq.scheme_b_noneq and not bare.scheme_b_noneq
    assert noneq.solvation == 'nonequilibrium'


@pytest.mark.parametrize('scheme, lr', [('scheme-b-eq', True),
                                        ('scheme-b-eq-bare', False)])
def test_a_bare_singlet_carries_no_linear_response(tmp_path, scheme, lr):
    run = run_in(tmp_path, declaration(*SCHEME_B, '--solvent-scheme', scheme))
    make_field, make_state, _ = run.loop
    field = make_field(run.mol_in)
    singlet = make_state('singlet')(run.mol_in, field)
    triplet = make_state('triplet')(run.mol_in, field)
    assert (singlet.lr_environment is field.transition_environment()) == lr
    assert (singlet.lr_environment is None) == (not lr)
    assert triplet.lr_environment is None


@pytest.mark.parametrize('scheme', ['scheme-b-eq', 'scheme-b-noneq'])
def test_a_non_equilibrium_scheme_walks_in_s0s_slow_charges(tmp_path, scheme):
    options = SCHEME_B if scheme == 'scheme-b-eq' else NONEQ + (
        '--solvent-scheme', scheme)
    run = run_in(tmp_path, declaration(*options))
    noneq = scheme == 'scheme-b-noneq'
    for kind in ('singlet', 'triplet'):
        surface = run.loop_surface(kind, run.mol_in, 'test')
        assert surface.noneq == noneq and surface.make_ground is not None
    assert not run.loop_surface('S0', run.mol_in, 'test').noneq


def test_the_tda_minimum_belongs_to_a_full_bse_scheme_b_run(tmp_path):
    xyz = tmp_path / 't1.xyz'
    xyz.write_text('4\nformaldehyde\nC 0 0 -0.5296\nO 0 0 0.6742\n'
                   'H 0 0.9429 -1.1123\nH 0 -0.9429 -1.1123\n')
    full = declaration(*SCHEME_B, '--tda-t1-minimum', str(xyz))
    assert full.kernel == 'bse' and full.policies.tda_t1_minimum is not None
    for options in ((*SCHEME_B, '--tda'), (*TOLUENE, '--solvent-scheme',
                                           'scheme-a'), ()):
        with pytest.raises(DeclarationRefused, match='tda_t1_minimum'):
            declaration(*options, '--tda-t1-minimum', str(xyz))
    run, plain = run_in(tmp_path / 'a', full), run_in(tmp_path / 'b',
                                                      declaration(*SCHEME_B))
    sha = full.policies.tda_t1_minimum.sha256()
    for stage in ('t1', 'vib', 'rate'):
        assert run.stamp(stage)['tda_t1_minimum'] == sha
    assert 'tda_t1_minimum' not in run.stamp('s1')
    assert 'tda_t1_minimum' not in plain.stamp('t1')


def test_the_walk_stopped_at_the_instability():
    collapsed = ('trust radius collapsed at 1.0e-03 Bohr; the state stopped '
                 'evaluating ({}: min eig(A - B) -2.0e-04)')
    for name, stopped in (('CasidaInstability', True), ('UnstableReference', True),
                          ('CasidaBreakdown', True), ('QPRootNotConverged', False)):
        info = {'converged': False, 'status': collapsed.format(name)}
        assert stages.stopped_at_instability(info) == stopped, name
    assert stages.stopped_at_instability({'converged': False, 'status':
                                          'CasidaInstability: at the start'})
    assert not stages.stopped_at_instability({'converged': True, 'status': 'ok'})


def test_a_t1_walk_that_raises_at_the_instability_falls_back(tmp_path,
                                                              monkeypatch):
    """A refrozen pass that stops at the instability raises out of the walk:
    the stage takes the single point all the same; without a declared TDA
    minimum it raises."""
    xyz = tmp_path / 't1.xyz'
    xyz.write_text('4\nformaldehyde\nC 0 0 -0.5296\nO 0 0 0.6742\n'
                   'H 0 0.9429 -1.1123\nH 0 -0.9429 -1.1123\n')
    taken = []
    monkeypatch.setattr(stages, 'stored_stage', lambda run, stage: None)
    monkeypatch.setattr(stages, 'resume_point', lambda *a, **k: (None, None))
    monkeypatch.setattr(stages, 'tda_single_point',
                        lambda run, walked, reason, t0: taken.append(reason) or {})

    def unstable(*a, **k):
        raise stages.CasidaInstability('min eig(A - B) -1.0e-03')
    monkeypatch.setattr(stages, 'walked_minimum', unstable)
    for options, falls in (((*SCHEME_B, '--tda-t1-minimum', str(xyz)), True),
                           (SCHEME_B, False)):
        run = run_in(tmp_path / str(falls), declaration(*options))
        monkeypatch.setattr(run, 'reusable', lambda stage: True)
        monkeypatch.setattr(run, 'read', lambda stage: {
            'props': {'coords_bohr': run.mol_in.atom_coords()}})
        monkeypatch.setattr(run, 'loop_surface', lambda *a: None)
        if falls:
            stages.t1_minimum(run)
            assert taken == ['CasidaInstability: min eig(A - B) -1.0e-03']
        else:
            with pytest.raises(stages.CasidaInstability):
                stages.t1_minimum(run)


def test_the_loop_tied_to_the_gradient_by_name(tmp_path):
    tied = declaration(*SCHEME_B, '--loop-tolerance', 'gradient-tied')
    assert tied.numerics.solvent.loop_force_fraction == LOOP_FORCE_FRACTION
    fixed = declaration(*SCHEME_B)
    assert fixed.numerics.solvent.loop_force_fraction is None
    assert declaration(*SCHEME_B, '--loop-tolerance', 'fixed').numerics.identity() \
        == fixed.numerics.identity()
    stamped = run_in(tmp_path, tied).stamp('s0')['solvent_scheme']['loop']
    assert stamped['loop_force_fraction'] == LOOP_FORCE_FRACTION
    with pytest.raises(SystemExit, match='scheme-b'):
        declaration(*TOLUENE, '--solvent-scheme', 'scheme-a',
                    '--loop-tolerance', 'gradient-tied')
    surface = run_in(tmp_path / 'b', tied).loop_surface('singlet', None, 'test')
    assert surface.final_grad_max == GEOM_OPT_CONV['opt_grad_max']


def test_the_saddle_check_takes_its_forces_on_untied_loops(tmp_path, monkeypatch):
    """A tied walk's saddle check differences 6N displaced forces, each above
    the walk bar: the stage hands the check the surface's untied twin."""
    tied = declaration(*SCHEME_B, '--loop-tolerance', 'gradient-tied',
                       '--saddle-check', '1')
    run = run_in(tmp_path, tied)
    surface = run.loop_surface('singlet', run.mol_in, 'test')
    seen = {}

    def saddle_check(final, mol, walk, verbose=False, cache_dir=None):
        seen['final'] = final
        raise RuntimeError('stop')
    monkeypatch.setattr(walks, 'saddle_check', saddle_check)
    with pytest.raises(RuntimeError, match='stop'):
        run.saddle_checked('s1', run.mol_in, {'converged': True}, None, surface, 0.0)
    final = seen['final']
    assert final is not surface and final._untied and not surface._untied
    assert final.loop_log is surface.loop_log and final.flags is surface.flags
