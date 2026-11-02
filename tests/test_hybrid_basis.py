"""A per-element or per-atom orbital basis on the photophysics route
(`basis_map`): one name for every atom and another for some elements or
atoms, e.g. aug-cc-pVDZ on the heavy atoms and cc-pVDZ on hydrogen, or
aug-cc-pVDZ on an emitter's atoms and cc-pVDZ on its hosts.

  * `--basis-for EL=NAME` and `atoms=I-J:NAME` build the canonical map: the
    plain name where all atoms carry one set, every element named where each
    element carries one, else the atoms of a second set labelled 'El@k' and
    every symbol and label named; an element or atom the molecule lacks, an
    atom named twice, an atom labelled already, a malformed item and a partial
    map are refused;
  * each element's (each label's) auxiliary sets follow its own basis: -ri
    for the factors, -jkfit for a density-fitted mean field;
  * an ISDF accuracy level holds on a map where every named basis gives it
    the same counts, and is refused where they differ; the Hessian grid floor
    is read per element;
  * the grid stamp and the run's stamps carry the map, so a store written
    under one map refuses a run under another, per-element or per-atom (the
    solvated driver's record too);
  * the solvent's auxiliary-partner check is made key by key;
  * a map naming the same basis on every element is the plain name: the S1
    energy (to CROSS_PATH_RTOL) and force (to the gradient class bar) on
    formaldehyde/cc-pVDZ, ISDF-K and DF mean fields; a per-atom map (a
    labelled hydrogen) that gives every atom the set of the per-element map
    is that map on water, to the same bars;
  * water with aug-cc-pVDZ on O and cc-pVDZ on H: the S1 force of the
    production chain (G3 AO-product factors) is the derivative of its energy
    along an internal direction within ISDF_GRADIENT_FLOOR, on the DF mean
    field (per-element -jkfit) and on the ISDF-K mean field (per-element
    grid rows); and with aug-cc-pVDZ on O and one H, cc-pVDZ on the other
    (a per-atom map), on the ISDF-K mean field (the tiered outside scissor:
    the curve overshoots its knots for this split and is refused).
"""
import argparse
import json
from dataclasses import replace

import numpy as np
import pytest
from pyscf import df as pyscf_df
from pyscf import gto

from src.Base.basis.basis_map import (atom_basis_names, atom_elements,
                                      molecule_basis, parse_basis_for)
from src.Base.basis.ri_fallback import paired_auxbasis, paired_jk_auxbasis
from src.Base.factorization.separable_ri import (grid_identity, resolve_isdf_grid,
                                                 validated_levels)
from src.Base.geometry.geometry import elements_of, molecule_at
from src.environments.solvent_screening import SolventScreening
from src.gradients.gwbse.surface_factory import reference_mean_field
from src.photophysics import chain, solvated
from src.photophysics.declaration import DeclarationRefused, System
from src.photophysics.stages import ground_minimum
from src.photophysics.store import StoreRefused
from src.properties.hessian import require_hessian_counts
from tests.photophysics_cases import FORMALDEHYDE, declaration, run_in
from tests.pins import (CROSS_PATH_RTOL, PIN_CROSS_BLAS_GRADIENT_BAR,
                         isdf_gradient_bar)

WATER = 'O 0 0.05 0.117; H 0.1 0.757 -0.468; H 0 -0.757 -0.468'
HYBRID = {'H': 'cc-pvdz', 'O': 'aug-cc-pvdz'}
#: The per-atom map: aug-cc-pVDZ on O and the first H, cc-pVDZ on the second.
PER_ATOM = ('O 0 0.05 0.117; H 0.1 0.757 -0.468; H@1 0 -0.757 -0.468',
            {'H': 'aug-cc-pvdz', 'H@1': 'cc-pvdz', 'O': 'aug-cc-pvdz'})
#: Bohr, the five-point step.
FD_STEP = 1e-3


# ------------------------------------------------------------------ the map
def test_the_map_is_canonical():
    assert parse_basis_for(['h=cc-pVDZ', 'O = aug-cc-pvdz']) == (
        {'H': 'cc-pVDZ', 'O': 'aug-cc-pvdz'}, {})
    assert parse_basis_for(['atoms=1-2,4:aug-cc-pvdz']) == (
        {}, {(0, 1, 3): 'aug-cc-pvdz'})
    for bad in (['H'], ['H='], ['Xx=cc-pvdz'], ['H1=cc-pvdz'],
                ['H=cc-pvdz', 'h=aug-cc-pvdz'], ['atoms=1-2'], ['atoms=0:x'],
                ['atoms=2-1:x'], ['atoms=1-2:x', 'atoms=2:y']):
        with pytest.raises(ValueError):
            parse_basis_for(bad)
    assert atom_elements(WATER) == ['H', 'O']
    assert molecule_basis(WATER, 'aug-cc-pVDZ', {'H': 'cc-pvdz'}) == (WATER, HYBRID)
    assert molecule_basis(WATER, 'cc-pvdz', {'H': 'CC-PVDZ'}) == (WATER, 'cc-pvdz')
    # atoms that make up whole elements give the per-element map
    assert molecule_basis(WATER, 'cc-pvdz', {}, {(0,): 'aug-cc-pvdz'}) == (
        WATER, HYBRID)
    # one H apart: labelled, over the element's map
    assert molecule_basis(WATER, 'cc-pvdz', {'O': 'aug-cc-pvdz'},
                          {(1,): 'aug-cc-pvdz'}) == PER_ATOM
    assert atom_basis_names(*PER_ATOM) == ['aug-cc-pvdz', 'aug-cc-pvdz', 'cc-pvdz']
    for args, match in (((WATER, 'cc-pvdz', {'N': 'cc-pvdz'}), 'does not have'),
                        ((WATER, 'cc-pvdz', {}, {(3,): 'x'}), 'has 3 atoms'),
                        (('O 0 0 0; H1 0 1 0; H 0 -1 0', 'cc-pvdz', {},
                          {(1,): 'aug-cc-pvdz'}), 'labelled already')):
        with pytest.raises(ValueError, match=match):
            molecule_basis(*args)


def test_each_element_pairs_its_own_auxiliary_sets():
    mol = gto.M(atom=WATER, basis=HYBRID, verbose=0)
    ri = {'H': 'cc-pvdz-ri', 'O': 'aug-cc-pvdz-ri'}
    assert paired_auxbasis(HYBRID) == chain.factor_auxbasis(mol) == ri
    assert chain.scf_auxbasis(mol, 'isdf') == ri
    assert paired_jk_auxbasis(HYBRID) == chain.scf_auxbasis(mol, 'df') == {
        'H': 'cc-pvdz-jkfit', 'O': 'aug-cc-pvdz-jkfit'}
    plain = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    assert chain.scf_auxbasis(plain, 'df') == 'cc-pvdz-jkfit'
    labelled = gto.M(atom=PER_ATOM[0], basis=PER_ATOM[1], verbose=0)
    assert labelled.nao_nr() == 23 + 9 + 5
    assert elements_of(labelled) == ['H', 'H@1', 'O']
    assert chain.factor_auxbasis(labelled) == {
        'H': 'aug-cc-pvdz-ri', 'H@1': 'cc-pvdz-ri', 'O': 'aug-cc-pvdz-ri'}
    assert chain.scf_auxbasis(labelled, 'df')['H@1'] == 'cc-pvdz-jkfit'
    aux = pyscf_df.addons.make_auxmol(labelled, chain.factor_auxbasis(labelled))
    def nao(atom, basis):
        return gto.M(atom=atom, basis=basis, spin=None, verbose=0).nao_nr()
    assert aux.nao_nr() == (nao('O 0 0 0; H 0 0 1', 'aug-cc-pvdz-ri')
                            + nao('H 0 0 0', 'cc-pvdz-ri'))
    # the labelled H takes cc-pVDZ's grid row, the other aug-cc-pVDZ's
    ident = grid_identity('G3', PER_ATOM[1], elements_of(labelled))
    assert set(ident['counts']) == {'H', 'H@1', 'O'}
    assert ident['radii_sha256'] != grid_identity(
        'G3', 'aug-cc-pvdz', ['H', 'O'])['radii_sha256']


def test_a_level_on_a_map():
    assert validated_levels(HYBRID) == validated_levels('aug-cc-pvdz')
    counts, _ = resolve_isdf_grid('G3', HYBRID, ['H', 'O'])
    assert counts == resolve_isdf_grid('G3', 'aug-cc-pvdz', ['H', 'O'])[0]
    mixed = {'H': 'cc-pvdz', 'O': 'aug-cc-pvtz'}      # G3 counts differ
    assert 'G3' not in validated_levels(mixed)
    with pytest.raises(ValueError, match='grid does not exist'):
        resolve_isdf_grid('G3', mixed, ['H', 'O'])
    shells = tuple(counts[s] for s in ('A1', 'A2', 'A3', 'B1'))
    require_hessian_counts(HYBRID, {'H': shells, 'O': shells}, 'G3')
    with pytest.raises(NotImplementedError):
        require_hessian_counts(HYBRID, {'H': (8, 5, 3, 1), 'O': shells}, 'G3')
    # the grid's identity is per element: the H row is cc-pVDZ's
    mixed_id = grid_identity('G3', HYBRID, ['H', 'O'])
    assert mixed_id['radii_sha256'] != grid_identity(
        'G3', 'aug-cc-pvdz', ['H', 'O'])['radii_sha256']


def test_the_command_line_builds_the_map():
    d = declaration('--atom', WATER, '--system', 'water', '--basis',
                    'aug-cc-pvdz', '--basis-for', 'H=cc-pvdz', '--scf', 'df')
    assert d.system.basis == HYBRID
    same = declaration('--basis', 'cc-pvdz', '--basis-for', 'H=cc-pvdz')
    assert same.system.basis == 'cc-pvdz'
    with pytest.raises(SystemExit):
        declaration('--basis-for', 'N=cc-pvdz')
    with pytest.raises(DeclarationRefused, match='per-element'):
        System('water', WATER, {'H': 'cc-pvdz'})
    per_atom = declaration('--atom', WATER, '--system', 'water', '--basis',
                           'cc-pvdz', '--basis-for', 'O=aug-cc-pvdz',
                           '--basis-for', 'atoms=2:aug-cc-pvdz', '--scf', 'df')
    assert (per_atom.system.atom, per_atom.system.basis) == PER_ATOM
    with pytest.raises(DeclarationRefused, match='per-element'):
        System('water', PER_ATOM[0], HYBRID)


def test_the_solvent_partner_is_checked_per_element():
    mol = gto.M(atom=WATER, basis=HYBRID, verbose=0)
    screening = SolventScreening(mol, solvent='toluene')
    wrong = gto.M(atom=WATER, basis='aug-cc-pvdz-ri', verbose=0)
    with pytest.raises(ValueError, match='on H'):
        screening.require_fit(mol, wrong, np.zeros((mol.nao_nr(), 2)), 1)


# ------------------------------------------------------------------ resume
def test_a_store_refuses_another_map(tmp_path):
    hybrid = declaration('--atom', WATER, '--system', 'water', '--basis',
                         'aug-cc-pvdz', '--basis-for', 'H=cc-pvdz', '--scf', 'df')
    run_a = run_in(tmp_path, hybrid)
    run_a.write('s0', {'coords_bohr': run_a.mol_in.atom_coords()})
    ground_minimum(run_a)                      # its own map reads it back
    other = replace(hybrid, system=replace(hybrid.system, basis='aug-cc-pvdz'))
    with pytest.raises(StoreRefused, match='system'):
        ground_minimum(run_in(tmp_path, other))
    # per atom: the same sets on other atoms is another map
    atom, basis = PER_ATOM
    one = replace(hybrid, system=replace(hybrid.system, atom=atom, basis=basis))
    run_b = run_in(tmp_path / 'atoms', one)
    run_b.write('s0', {'coords_bohr': run_b.mol_in.atom_coords()})
    ground_minimum(run_b)
    swapped = atom.replace('H 0.1', 'H@1 0.1').replace('H@1 0 -', 'H 0 -')
    two = replace(one, system=replace(one.system, atom=swapped))
    with pytest.raises(StoreRefused, match='system'):
        ground_minimum(run_in(tmp_path / 'atoms', two))


def test_the_solvated_record_refuses_another_map(tmp_path):
    xyz = tmp_path / 'water.xyz'
    xyz.write_text('3\n\n' + '\n'.join(a.strip() for a in WATER.split(';')) + '\n')
    out = tmp_path / 'record.json'

    def args(*basis_for, jk=None):
        ns = argparse.Namespace(
            xyz=str(xyz), method='adc2', basis='aug-cc-pvdz',
            basis_for=list(basis_for) or None, auxbasis=None, jk_auxbasis=jk,
            solvent=None, frozen=1, nroots=3, follow='index', saddle_check=True,
            tol_q=1e-4, tol_energy=1e-7, xc=None, isdf_grid=None,
            bse_kernel=None, bse_solver=None, refreeze=None, out=str(out))
        solvated._resolve_basis(ns)
        return ns
    hybrid = args('H=cc-pvdz')
    assert hybrid.basis == HYBRID
    assert hybrid.jk_auxbasis == paired_jk_auxbasis(HYBRID)
    assert hybrid.auxbasis == paired_auxbasis(HYBRID)
    assert args().jk_auxbasis == 'cc-pvdz-jkfit'
    with pytest.raises(SystemExit, match='jk-auxbasis'):
        args('H=cc-pvdz', jk='cc-pvdz-jkfit')
    out.write_text(json.dumps(solvated._settings(hybrid)))
    solvated._start_record(hybrid)             # its own map resumes
    with pytest.raises(SystemExit, match='basis'):
        solvated._start_record(args('H=aug-cc-pvdz'))
    # per atom: the record keys each atom's set
    first = args('atoms=2:cc-pvdz')
    assert first.atom.split(';')[2].split()[0] == 'H@1'
    assert solvated._settings(first)['basis'] == [
        'aug-cc-pvdz', 'cc-pvdz', 'aug-cc-pvdz']
    out.write_text(json.dumps(solvated._settings(first)))
    solvated._start_record(first)
    with pytest.raises(SystemExit, match='basis'):
        solvated._start_record(args('atoms=3:cc-pvdz'))


# ----------------------------------------------------------------- surfaces
def s1_chain(store_root, decl):
    """The production S1 chain of `decl` at its input geometry
    (`photophysics_cases.production` on a declaration handed in)."""
    run = run_in(store_root, replace(decl, qp_states=type(decl.qp_states)(
        kind='admitted')))
    mf0 = reference_mean_field(run.mol_in, run.factory,
                               run.relaxed_spec.environment)
    return run, run.excited_at(run.relaxed_spec, run.mol_in, mf0)


def assert_same_s1(a, b):
    """Two S1 (energy, force) pairs of one molecule given its basis two ways:
    two SCFs whose last bits differ with the BLAS, so the energy to
    CROSS_PATH_RTOL and the force to the gradient class bar."""
    assert abs(a[0] - b[0]) <= CROSS_PATH_RTOL * abs(a[0]), (a[0], b[0])
    apart = np.abs(a[1] - b[1]).max()
    assert apart <= PIN_CROSS_BLAS_GRADIENT_BAR, apart


@pytest.mark.parametrize('route', [('--scf', 'isdf'), ('--scf', 'df')])
def test_a_uniform_map_is_the_plain_name(route, tmp_path):
    plain = declaration(*route)
    assert plain.system.basis == 'cc-pvdz'
    uniform = replace(plain, system=replace(plain.system, basis={
        el: 'cc-pvdz' for el in atom_elements(FORMALDEHYDE)}))
    out = []
    for name, decl in (('plain', plain), ('map', uniform)):
        run, ch = s1_chain(tmp_path / name, decl)
        assert run.mol_in.basis == decl.system.basis
        g, e, _ = ch.total_gradient(run.mol_in)
        out.append((e, np.asarray(g)))
    assert_same_s1(*out)


@pytest.mark.parametrize('route', [('--scf', 'isdf', '--scf-grid', 'G3'),
                                   ('--scf', 'df')])
def test_a_per_atom_map_is_its_per_element_map(route, tmp_path):
    """Water with its second H labelled and given the H set of the
    per-element map: every atom carries the same set, the labelled one under
    its label (basis, auxiliary sets, grid row), and the S1 energy and force
    are the per-element map's (`assert_same_s1`)."""
    element = declaration('--atom', WATER, '--system', 'water', '--basis',
                          'aug-cc-pvdz', '--basis-for', 'H=cc-pvdz', *route)
    labelled = WATER.replace('H 0 -', 'H@1 0 -')
    atom = replace(element, system=replace(element.system, atom=labelled,
                                           basis=dict(HYBRID, **{'H@1': 'cc-pvdz'})))
    out = []
    for name, decl in (('element', element), ('atom', atom)):
        run, ch = s1_chain(tmp_path / name, decl)
        g, e, _ = ch.total_gradient(run.mol_in)
        out.append((e, np.asarray(g)))
    assert_same_s1(*out)


def internal_direction(mol, seed):
    """A unit Cartesian direction with no rigid translation or rotation."""
    x = mol.atom_coords()
    x = x - x.mean(0)
    rigid = [np.tile(np.eye(3)[k], (len(x), 1)) for k in range(3)]
    rigid += [np.cross(np.eye(3)[k], x) for k in range(3)]
    basis = np.linalg.qr(np.array([r.ravel() for r in rigid]).T)[0]
    v = np.random.default_rng(seed).normal(size=x.size)
    v -= basis @ (basis.T @ v)
    return (v / np.linalg.norm(v)).reshape(x.shape)


@pytest.mark.parametrize('route,basis_for,nao', [
    (('--scf', 'df'), ('H=cc-pvdz',), 33),
    (('--scf', 'isdf', '--scf-grid', 'G3'), ('H=cc-pvdz',), 33),
    # this split's outside curve overshoots its knots and is refused there
    (('--scf', 'isdf', '--scf-grid', 'G3', '--outside-scissor', 'tiers'),
     ('atoms=3:cc-pvdz',), 37)])
def test_the_s1_force_on_a_hybrid_basis(route, basis_for, nao, tmp_path):
    decl = declaration('--atom', WATER, '--system', 'water', '--basis',
                       'aug-cc-pvdz', *(a for b in basis_for
                                        for a in ('--basis-for', b)), *route)
    run, ch = s1_chain(tmp_path, decl)
    mol = run.mol_in
    assert mol.nao_nr() == nao
    g, _, _ = ch.total_gradient(mol)
    u = internal_direction(mol, 5)
    x = mol.atom_coords()
    e = {k: ch.total_energy(molecule_at(mol, x + k * FD_STEP * u))
         for k in (-2, -1, 1, 2)}
    fd = (e[-2] - 8 * e[-1] + 8 * e[1] - e[2]) / (12 * FD_STEP)
    err = abs(fd - float((np.asarray(g) * u).sum()))
    print(f'{route}: dE/ds analytic {float((np.asarray(g) * u).sum()):.10f}, '
          f'five-point {fd:.10f}, error {err:.2e} Ha/Bohr')
    assert err < isdf_gradient_bar(), err
