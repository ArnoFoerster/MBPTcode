"""A walk across orbitals that swap or mix at the edge of a frozen
quasiparticle-Hamiltonian group (`gradients.conventions`,
`properties.optimize`).

The surface is a harmonic spring in the interatomic distances of water whose
evaluation resolves real `FrozenConventions` (group (5, 6) of an explicit set,
orbitals 4 and 7-9 outside it) at every geometry. Between the keys' geometry
and any other the orbitals rotate by a fixed fraction between orbital 6 (in
the group) and 7 (outside it), at every step length, as at a sharp crossing:

  - a SWAP (0.97 of 6 now at index 7) is matched to the new indices and
    relabelled: the walk converges with no refusal, no refreeze and no
    rejected step, and its commits record the moved slots;
  - a MIX (0.3, below CONVENTION_SPAN_MIN in the group's span) is refused
    with orbital 7 named as the partner; the walk refreezes the conventions
    at the last accepted geometry with 7 absorbed into the group, once,
    keeps its trust radius (Cartesian walker and geomeTRIC alike), and
    converges with no rejected step;
  - on a GW-BSE chain (formaldehyde/cc-pVDZ, Hartree-Fock, the quasiparticle
    Hamiltonian on: group 12-13) absorbed orbitals join the explicit set on
    the refrozen chain with every orbital between them (11 and 18 bring
    14-17: a group is a contiguous window), and its group where the pole
    model admits them (11); those it does not (14-18, solved by explicit
    residues) are dropped from the group and recorded; every later refreeze
    carries what was absorbed; a scheme-B state model built afresh takes it
    in.
"""
import types
import warnings

import numpy as np
import pytest
from pyscf import gto, scf

import src.gradients.gwbse.conventions as conventions
from src.Base.numerics.constants import CONVENTION_SPAN_MIN
from src.Base.numerics.numerics import Numerics
from src.gradients.gwbse.conventions import ConventionsLost
from src.gradients.gwbse.state_model import GWBSEState
from src.properties.optimize import optimize, optimize_geometric
from tests.test_frozen_conventions import frozen, rotation
from tests.test_geometric_step_failed_retry import DISPLACEMENT, distances

NORB = 10
#: the group, the explicit set and the outside orbital the group meets
GROUP, EXPLICIT, PARTNER = (5, 6), (1, 2, 3, 5, 6), 7
#: the weight orbital 6 moves to orbital 7 between any two geometries
SWAP, MIX = 0.97, 0.3
#: Bohr and Ha: the walks agree with the minimum within these
SAME_DISTANCE, SAME_ENERGY = 2e-3, 1e-6


class CrossingSpring:
    """E = sum_ij (r_ij - r*_ij)^2 / 2 over the distances `target`, evaluated
    only where its frozen conventions match: the orbitals at a geometry other
    than the keys' are the keys' rotated by `mixing` between orbitals 6 and
    7. `holder['surface']` is the surface body_overlap answers for."""

    def __init__(self, mol, target, mixing, holder, groups=(GROUP,),
                 explicit=EXPLICIT, refrozen=0):
        self.mol0, self.target, self.mixing = mol, target, mixing
        self.holder, self.refrozen = holder, refrozen
        self.refused = 0
        self.conv = frozen(groups=groups, explicit=explicit, norb=NORB)
        self.conv.key_mol = mol
        holder['surface'] = self

    def identity(self):
        return None

    @staticmethod
    def _mf():
        return types.SimpleNamespace(mo_coeff=np.eye(NORB))

    def overlap(self, key_mol, key_mo, mol, mo):
        moved = not np.array_equal(np.asarray(key_mol.atom_coords()),
                                   np.asarray(mol.atom_coords()))
        r = rotation(NORB, 6, 7, self.mixing if moved else 0.0)
        return key_mo.T @ r @ mo

    def commit(self, mol):
        self.conv.commit(mol, self._mf())

    def total_gradient(self, mol=None, mf=None):
        mol = self.mol0 if mol is None else mol
        try:
            self.conv.at(mol, self._mf())
        except ConventionsLost:
            self.refused += 1
            raise
        coords = mol.atom_coords()
        r = distances(coords)
        i, j = np.triu_indices(len(coords), 1)
        pull = ((r - self.target) / r)[:, None] * (coords[i] - coords[j])
        grad = np.zeros_like(coords)
        np.add.at(grad, i, pull)
        np.add.at(grad, j, -pull)
        return grad, 0.5 * float(((r - self.target) ** 2).sum()), {}

    def total_energy(self, mol=None, mf=None):
        return self.total_gradient(mol)[1]

    def refreeze(self, mol, mf=None):
        """Conventions frozen afresh at `mol`, what was absorbed one group
        with the group, its orbitals explicit."""
        absorbed = self.conv.absorbed_at_keys()
        group = tuple(sorted(set(GROUP) | {p for g in absorbed for p in g}))
        out = CrossingSpring(mol, self.target, self.mixing, self.holder,
                             groups=(group,),
                             explicit=tuple(sorted(set(EXPLICIT) | set(group))),
                             refrozen=self.refrozen + 1)
        out.conv.absorbed = absorbed
        return out


@pytest.fixture
def water(monkeypatch):
    """(start, minimum geometry, holder); body_overlap answers the rotation
    of the surface in the holder."""
    minimum = gto.M(atom='O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11',
                    unit='Bohr', basis='sto-3g', verbose=0)
    start = minimum.copy()
    start.set_geom_(minimum.atom_coords() + DISPLACEMENT, unit='Bohr')
    start.build(False, False)
    holder = {}
    monkeypatch.setattr(conventions, 'body_overlap',
                        lambda *a: holder['surface'].overlap(*a))
    return start, minimum.atom_coords(), holder


def spring(water, mixing):
    start, minimum, holder = water
    return CrossingSpring(start, distances(minimum), mixing, holder)


def at_minimum(mol, info, minimum):
    assert np.abs(distances(mol.atom_coords()) - distances(minimum)).max() \
        < SAME_DISTANCE
    assert abs(info['energy']) < SAME_ENERGY


# ------------------------------------------------------------ the matcher
def test_a_swap_is_relabelled_and_a_mix_names_its_partner(monkeypatch):
    c = frozen(groups=(GROUP,), explicit=EXPLICIT, norb=NORB)
    c.qp_seeds.update({6: -1.0})
    mol = types.SimpleNamespace(atom_coords=lambda: np.ones((1, 3)))
    mf = types.SimpleNamespace(mo_coeff=np.eye(NORB))
    monkeypatch.setattr(conventions, 'body_overlap',
                        lambda *a: rotation(NORB, 6, 7, SWAP))
    view = c.at(mol, mf)
    assert view.index(6) == 7 and view.qp_seeds[7] == -1.0
    c.commit(mol, mf)
    assert c.moves[-1]['moved'][6] == 7 and c.key_index[6] == 7
    assert c.absorbed == ()

    c = frozen(groups=(GROUP,), explicit=EXPLICIT, norb=NORB)
    monkeypatch.setattr(conventions, 'body_overlap',
                        lambda *a: rotation(NORB, 6, 7, MIX))
    assert MIX < 1 - CONVENTION_SPAN_MIN + 0.1 and 1 - MIX < CONVENTION_SPAN_MIN
    with pytest.raises(ConventionsLost) as exc:
        c.at(mol, mf)
    # the outside unit is lost too, but only the group takes orbitals in
    assert exc.value.mixing == ((GROUP, (PARTNER,)),)
    assert exc.value.conventions is c
    assert c.absorb(exc.value.mixing)
    assert c.absorbed == ((5, 6, 7),)
    assert not c.absorb(exc.value.mixing)


def test_a_mix_without_quasiparticle_groups_names_no_partner(monkeypatch):
    c = frozen(explicit=(1, 2, 3, 5, 6))
    monkeypatch.setattr(conventions, 'body_overlap',
                        lambda *a: rotation(NORB, 6, 7, MIX))
    with pytest.raises(ConventionsLost) as exc:
        c.at(types.SimpleNamespace(atom_coords=lambda: np.ones((1, 3))),
             types.SimpleNamespace(mo_coeff=np.eye(NORB)))
    assert exc.value.mixing == ()


# --------------------------------------------------------------- the walks
@pytest.mark.parametrize('walker', ['cartesian', 'geometric'])
def test_a_swap_walks_without_a_refreeze(water, walker, tmp_path):
    if walker == 'geometric':
        pytest.importorskip('geometric')
    _, free = run(walker, spring(water, 0.0), water[0], tmp_path / 'free')
    s = spring(water, SWAP)
    mol, info = run(walker, s, water[0], tmp_path)
    assert info['converged'], info['status']
    assert s.refused == 0 and info['convention_refreezes'] == []
    assert info['rejected'] == free['rejected']
    assert info['final_surface'] is s and s.refrozen == 0
    assert s.conv.moves, 'the walk never relabelled'
    at_minimum(mol, info, water[1])


@pytest.mark.parametrize('walker', ['cartesian', 'geometric'])
def test_a_mix_refreezes_once_and_keeps_the_trust_radius(water, walker,
                                                         tmp_path):
    if walker == 'geometric':
        pytest.importorskip('geometric')
    _, free = run(walker, spring(water, 0.0), water[0], tmp_path / 'free')
    s = spring(water, MIX)
    mol, info = run(walker, s, water[0], tmp_path)
    assert info['converged'], info['status']
    # the refusal is no rejected step: the walk rejects what the walk
    # without a crossing rejects
    assert s.refused == 1 and info['final_surface'].refused == 0
    assert info['rejected'] == free['rejected']
    events = info['convention_refreezes']
    assert len(events) == 1
    assert events[0]['absorbed'] == [[5, 6, 7]]
    assert events[0]['status'].startswith('ConventionsLost: ')
    assert info['final_surface'].refrozen == 1
    assert set(int(p) for p in info['final_surface'].conv.qp_set) >= {5, 6, 7}
    if walker == 'cartesian':
        # the refused cycle and the retry plan on one radius
        k = events[0]['cycle']
        assert info['history'][k + 1]['trust'] == info['history'][k]['trust']
        assert events[0]['trust'] == info['history'][k]['trust']
    else:
        assert info['retries'] == []
        assert events[0]['trust'] > 0
    at_minimum(mol, info, water[1])


def run(walker, surface, start, tmp_path):
    warnings.simplefilter('ignore')
    if walker == 'cartesian':
        return optimize(surface, start, verbose=False, nudge=0)
    workdir = tmp_path / 'walk'
    workdir.mkdir(parents=True)
    return optimize_geometric(surface, start, verbose=False, nudge=0,
                              workdir=str(workdir))


# ------------------------------------------------------ a GW-BSE chain
def test_absorbed_orbitals_join_the_set_and_the_group():
    from tests.test_qp_hamiltonian import WINDOW, chain
    ch = chain(qp_hamiltonian=True)
    ch.excitation()
    assert ch.conventions.group_slots() == ((12, 13),)
    # 11: explicit, on the pole model; 18: outside the set, which the
    # refrozen chain solves explicitly, off the pole model
    inside, outside = 11, max(WINDOW) + 1
    assert outside not in set(int(p) for p in ch.conventions.qp_set)
    assert ch.conventions.absorb((((12, 13), (inside, outside)),))
    rebuilt = ch.refreeze(ch.mol0)
    # the window from 11 to 18, every orbital between them
    window = tuple(range(inside, outside + 1))
    assert rebuilt.conventions.absorbed == (window,)
    rebuilt.excitation()
    assert set(window) <= set(int(p) for p in rebuilt.conventions.qp_set)
    assert rebuilt.conventions.group_slots() == ((11, 12, 13),)
    record = rebuilt.conventions.qp_subspace_record
    assert [(d['orbital'], d['reason']) for d in record['dropped']] == [
        (p, 'off the pole model') for p in range(14, outside + 1)]
    assert record['cut'] == []
    again = rebuilt.refreeze(ch.mol0)
    assert again.conventions.absorbed == (window,)


def test_a_scheme_b_state_built_afresh_takes_the_absorbed_orbitals():
    took = []

    class Built:
        def absorb_groups(self, groups):
            took.append(groups)

    conv = types.SimpleNamespace(absorbed_at_keys=lambda: ((5, 6, 7),))
    old = types.SimpleNamespace(conventions=conv, following=lambda: None)
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', unit='Bohr', basis='sto-3g',
                verbose=0)
    def make_scf(m):
        mf = scf.RHF(m).density_fit()
        mf.conv_tol_grad = 1e-12
        return mf

    state = GWBSEState(mol, make_scf, spin='singlet', numerics=Numerics(),
                       chain=old, build_chain=lambda *a: Built())
    fresh = state.refrozen(mol)
    fresh._chain(None)
    assert took == [((5, 6, 7),)]
