"""The stage pipeline's solvent scheme and external S0 Hessian, without a
GW-BSE evaluation.

  1. one scheme per run: 'gas' in the gas phase and only there; a continuum
     names 'scheme-a', 'scheme-b-eq' or 'scheme-b-noneq' (the command line
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

Run: python -m pytest tests/test_photophysics_scheme_b.py
"""
from dataclasses import replace

import numpy as np
import pytest
from pyscf import gto

from src.Base.numerics.constants import (EXTERNAL_HESSIAN_GEOMETRY_TOL_BOHR,
                                         PHOTOPHYSICS_LOOP_TOL_ENERGY,
                                         PHOTOPHYSICS_LOOP_TOL_Q)
from src.Base.declaration import QPStates
from src.photophysics import stages
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
