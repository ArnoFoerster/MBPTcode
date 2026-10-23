"""The property layer spells each small thing once.

"A molecule at new coordinates" was written in seven places, the occupied
count in seven, git provenance in four with `except Exception -> 'unknown'`,
and `spin_vibronic.evaluated_coupling` repeated `analytic_coupling`'s
evaluation branch. Now: `geometry.molecule_at`, `pyscf_interface.occupied_count`,
`record_io.git_commit` (which warns when git cannot answer) and
`derivative_coupling.evaluated_coupling`. The photophysics package (the
stages, the chain, the rate models, the vertical arms, the acceptance check,
the UISC model) is held to the same rule.

No SCF: the coupling is checked on a stand-in evaluation with the
configuration term stubbed.
"""
import ast
import pathlib
import subprocess
import types

import numpy as np
import pytest
from pyscf import gto

from src.Base.utils import record_io
from src.Base.utils.record_io import code_id, git_commit, same_code
from src.Base.geometry.geometry import displaced, molecule_at
from src.Base.pyscf.pyscf_interface import occupied_count
from src.gradients.gwbse import derivative_coupling as dc
from src.properties import hessian, spin_vibronic
from tests.photophysics_cases import calls, occupied_counts_by_hand

REPO = pathlib.Path(__file__).resolve().parents[1]
#: The modules this cleanup owns; none may move a molecule or count the
#: occupied orbitals by hand.
OWNED = ('src/photophysics/rates.py', 'src/properties/hessian.py',
         'src/properties/nonadiabatic.py', 'src/properties/spin_orbit.py',
         'src/properties/spin_vibronic.py', 'src/gradients/gwbse/surface_factory.py',
         'src/gradients/gwbse/derivative_coupling.py',
         'src/photophysics/stages.py', 'src/photophysics/chain.py',
         'src/photophysics/rate_models.py', 'src/photophysics/minimum.py',
         'src/photophysics/vertical_arms.py', 'src/photophysics/df_check.py',
         'src/photophysics/uisc.py')
WATER = 'O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59'


@pytest.mark.parametrize('rel', OWNED)
def test_no_hand_spelling_left(rel):
    if not (REPO / rel).is_file():
        pytest.fail(f'{rel} is missing')
    tree = ast.parse((REPO / rel).read_text())
    assert calls(tree, 'set_geom_') == [], rel
    assert occupied_counts_by_hand(tree) == [], rel


def test_the_displaced_molecules_are_molecule_at():
    """One displaced molecule (`geometry.displaced`): no
    module of this layer spells its own any more."""
    mol = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    x = mol.atom_coords()
    want = x.copy()
    want[1, 2] += 1e-3
    delta = np.zeros_like(x)
    delta[1, 2] = 1e-3
    for moved in (displaced(mol, hessian.component_step(mol, 1, 2, 1e-3)),
                  displaced(mol, delta, x),
                  molecule_at(mol, want)):
        assert np.array_equal(moved.atom_coords(), want)
        assert moved.basis == mol.basis and moved.nao == mol.nao
    own = [f'{p.relative_to(REPO)}:{n.name}' for top in ('src/gradients',
           'src/properties', 'src/photophysics')
           for p in sorted((REPO / top).rglob('*.py'))
           for n in ast.walk(ast.parse(p.read_text()))
           if isinstance(n, ast.FunctionDef)
           and n.name in ('displaced', 'displaced_along', 'step_ladder')]
    assert not own, own


def test_one_occupied_count():
    mf = types.SimpleNamespace(mo_occ=np.array([2.0, 2.0, 0.0, 0.0]))
    assert occupied_count(mf) == 2
    assert dc.occupied_count is occupied_count
    assert spin_vibronic.occupied_count is occupied_count


class Evaluation:
    """The pieces of a `StateEvaluation` the coupling reads."""

    def __init__(self):
        rng = np.random.default_rng(3)
        x, y = rng.standard_normal((2, 6, 3))
        self.target = {('singlet', 0): ('singlet', 0),
                       ('singlet', 1): ('singlet', 1)}
        self.root = {('singlet', 0): 0, ('singlet', 1): 1}
        self.spectrum = {'singlet': (np.array([0.20, 0.27, 0.31]), x, y)}
        num = rng.standard_normal((3, 3))
        self.interstate = {(('singlet', 0), ('singlet', 1)):
                           (num, {'gap': float(0.27 - 0.20)})}
        self.mol = self.mf = None

    def require_scf_quality(self, where):
        return []


def test_one_evaluated_coupling(monkeypatch):
    """Both entry points read one function; either ordering of the stored pair
    serves, antisymmetric; the gap is the evaluation's own."""
    monkeypatch.setattr(dc, 'configuration_coupling',
                        lambda mol, mf, nocc, goo, gvv: np.full((3, 3), 0.5))
    monkeypatch.setattr(dc, 'occupied_count', lambda mf: 2)
    ev = Evaluation()
    s0, s1 = ('singlet', 0), ('singlet', 1)
    d01, diags = dc.analytic_coupling(None, s0, s1, evaluation=ev)
    num = ev.interstate[(s0, s1)][0]
    assert np.array_equal(d01, num / (0.27 - 0.20) + 0.5)
    assert diags['gap'] == 0.27 - 0.20
    assert spin_vibronic.evaluated_coupling is dc.evaluated_coupling
    d10, diags = dc.evaluated_coupling(ev, s1, s0)
    assert np.array_equal(d10, num / (0.20 - 0.27) + 0.5)
    assert diags['gap'] == 0.20 - 0.27
    with pytest.raises(ValueError, match='interstate'):
        dc.evaluated_coupling(ev, s0, ('singlet', 2))


def test_git_commit_names_the_tree_and_warns_without_one(tmp_path):
    assert record_io.CHECKOUT == REPO
    with pytest.warns(RuntimeWarning, match='no commit'):
        assert git_commit(tmp_path) == 'unknown'
    head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short',
                           'HEAD'], capture_output=True, text=True)
    if head.returncode:
        pytest.skip('a git archive: no commit to compare against')
    assert git_commit(REPO).split('+')[0] == head.stdout.strip()


def test_one_code_id_sees_an_edited_tree(tmp_path):
    """`record_io.code_id` is `git_commit` plus a digest of the edit, so a
    record (the solvation driver's resume key) made on an edited tree says so
    and two edits of one commit are two codes; the old full-commit stamp of a
    clean tree still resumes. No other module asks git for a commit."""
    def git(*args):
        return subprocess.run(['git', '-C', str(tmp_path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git('init', '-q')
    (tmp_path / 'a.txt').write_text('one\n')
    git('add', 'a.txt')
    git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'a')
    clean = code_id(tmp_path)
    assert clean == git('rev-parse', '--short', 'HEAD')
    assert same_code(git('rev-parse', 'HEAD'), clean)
    (tmp_path / 'a.txt').write_text('two\n')
    edit_1 = code_id(tmp_path)
    (tmp_path / 'a.txt').write_text('three\n')
    edit_2 = code_id(tmp_path)
    assert edit_1.startswith(clean + '+dirty:') and edit_1 != edit_2
    assert not same_code(git('rev-parse', 'HEAD'), edit_1)
    assert not same_code(edit_1, edit_2)
    asking = [path.relative_to(REPO).as_posix() for top in ('src', 'tools')
              for path in (REPO / top).rglob('*.py')
              if "'rev-parse'" in path.read_text()]
    asking.remove('src/Base/utils/record_io.py')
    assert not asking, asking
