"""A window edge closer than QP_HAMILTONIAN_WINDOW to the first orbital
outside the window is a hole of the next probe inward, on both sides of the
gap, so it and the outside orbitals carry one frozen shift and rotate freely
in one unit of the frozen conventions.

  - the selection layer (synthetic spectra): such an edge is never mandatory
    nor selected, whatever the budget, a carried floor or a broken
    continuation ask; it borrows the next probe inward, which is the edge
    then; a far edge and the frontier are untouched;
  - the chain (formaldehyde/cc-pVDZ, HF, G1, a window ending 0.11 eV below
    the next orbital): the edge is a hole of the orbital below it, carries
    the outside's shift bit for bit, and is matched in the outside's unit;
  - DMABN at the gas S1 minimum of the twisted-start walk (window 22..57, KS
    orbitals 57/58 0.30 eV apart): the explicit set the selection takes is
    matched 0.12 Bohr along the T1 walk's start
    nudge, where the set with 57 a lone explicit orbital is refused
    (`ConventionsLost`).
"""
import warnings

import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.numerics.numerics import Numerics
from src.Base.numerics.constants import HARTREE_TO_EV, QP_HAMILTONIAN_WINDOW
from src.Base.declaration import QPStates
from src.Base.factorization.separable_ri import resolve_isdf_grid
from src.SingleReference.GW.qp_selection import (calibrated_selection,
                                                 mandatory_members,
                                                 select_explicit)
from src.SingleReference.GW.qp_states import (calibrate_scissor,
                                              calibrate_scissor_tiers)
from src.gradients.gwbse.conventions import ConventionsLost, FrozenConventions
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.state_manifold import StateManifold
from src.properties.optimize import nudge_vector
from tests.isdf_test_grid import g1_numerics

DEG = 1e-4
TARGETS = (('singlet', 0), ('triplet', 0))
#: Ha: well inside and well outside QP_HAMILTONIAN_WINDOW
NEAR = 0.3 * QP_HAMILTONIAN_WINDOW
FAR = 3.0 * QP_HAMILTONIAN_WINDOW
#: Bohr: the largest atomic displacement of the DMABN step
DMABN_STEP = 0.12


def spectrum(top=NEAR, bottom=NEAR):
    """Fourteen orbitals, six occupied, the window 1..12: orbital 0 lies
    `bottom` below the window's bottom edge 1, orbital 13 `top` above its
    top edge 12."""
    eps = np.array([0.0, 0.0, -0.95, -0.80, -0.62, -0.40,
                    0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00, 0.0])
    eps[0] = eps[2] - 0.20 - bottom
    eps[1] = eps[2] - 0.20
    eps[13] = eps[12] + top
    return eps, 6, tuple(range(1, 13))


def shifts(cands):
    """A smooth AC shift per candidate, Ha."""
    return {p: (0.8 - 0.1 * i) / HARTREE_TO_EV for i, p in enumerate(cands)}


def unit_weights(eps, cands):
    """One target that weighs every candidate."""
    n = np.zeros(len(eps))
    n[list(cands)] = 0.1
    return {TARGETS[0]: n}


def test_an_edge_near_the_outside_is_not_mandatory():
    eps, nocc, cands = spectrum()
    why = mandatory_members(eps, nocc, cands, DEG)
    assert 1 not in why and 12 not in why
    assert why[2] == 'edge' and why[11] == 'edge'
    assert why[5] == 'frontier' and why[6] == 'frontier'
    # a state the continuation cannot judge is not solved at the edge either
    assert 12 not in mandatory_members(eps, nocc, cands, DEG,
                                       unjudgeable={12: 'Pade failed'})


def test_a_far_edge_stays_mandatory():
    eps, nocc, cands = spectrum(top=FAR, bottom=FAR)
    why = mandatory_members(eps, nocc, cands, DEG)
    assert why[1] == 'edge' and why[12] == 'edge'


@pytest.mark.parametrize('tol', [0.0, 1e9])
def test_the_edge_is_a_hole_of_the_next_probe_inward(tol):
    """Whatever the budget -- the whole window at tol 0 -- and whatever the
    start holds (a carried floor), the near edges are holes of their inner
    neighbours, and only they are on the far-edge spectrum's other side."""
    eps, nocc, cands = spectrum()
    start = set(mandatory_members(eps, nocc, cands, DEG)) | {1, 12}
    sel = select_explicit(eps, nocc, cands, shifts(cands),
                          unit_weights(eps, cands), tol, start=start,
                          degeneracy_tol=DEG, hole_max_ev=None)
    assert 1 not in sel.explicit and 12 not in sel.explicit
    assert sel.tier_of[1] == (2,) and sel.tier_of[12] == (11,)
    if tol == 0.0:
        assert sel.explicit == tuple(range(2, 12))
    # only one side near: the other edge is explicit
    eps, nocc, cands = spectrum(bottom=FAR)
    sel = select_explicit(eps, nocc, cands, shifts(cands),
                          unit_weights(eps, cands), tol,
                          start=mandatory_members(eps, nocc, cands, DEG),
                          degeneracy_tol=DEG, hole_max_ev=None)
    assert 1 in sel.explicit and sel.tier_of[12] == (11,)


def test_a_broken_continuation_solves_the_window_but_its_edge_holes():
    eps, nocc, cands = spectrum()
    a = shifts(cands)
    explicit = {p: a[p] + 1.0 for p in (2, 5, 6, 11)}
    sel, _ = calibrated_selection(eps, nocc, cands, a, explicit,
                                  unit_weights(eps, cands), 1.0,
                                  degeneracy_tol=DEG, gate_ev=0.1)
    assert sel.fell_back == 'calibration'
    assert sel.explicit == tuple(range(2, 12))
    assert sel.tier_of == {1: (2,), 12: (11,)}


def test_the_frontier_is_never_a_hole():
    """A window whose virtual side is the LUMO alone keeps it, near or not."""
    eps, nocc, _ = spectrum()
    cands = tuple(range(1, 7))
    eps = eps.copy()
    eps[7] = eps[6] + NEAR
    why = mandatory_members(eps, nocc, cands, DEG)
    assert 6 in why
    sel = select_explicit(eps, nocc, cands, shifts(cands),
                          unit_weights(eps, cands), 0.0, start=why,
                          degeneracy_tol=DEG, hole_max_ev=None)
    assert 6 in sel.explicit


# --------------------------------------------------------------- the chain
def hf_factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    return mf


#: formaldehyde, HF/cc-pVDZ: orbital 12 lies 0.11 eV below 13
FORMALDEHYDE = 'C 0 0 0; O 0 0 1.205; H 0 0.9429 -0.5876; H 0 -0.9429 -0.5876'
EDGE, INNER = 12, 11


def test_the_chain_holds_the_edge_with_the_outside():
    warnings.simplefilter('ignore')
    mol = gto.M(atom=FORMALDEHYDE, basis='cc-pvdz', verbose=0)
    mf = hf_factory(mol)
    eps = np.asarray(mf.mo_energy, float)
    assert eps[EDGE + 1] - eps[EDGE] < QP_HAMILTONIAN_WINDOW
    assert eps[EDGE] - eps[INNER] > QP_HAMILTONIAN_WINDOW
    counts, n_start = resolve_isdf_grid('G1', 'cc-pvdz', ['C', 'H', 'O'],
                                        auxbasis='cc-pvdz-ri')
    chain = ExcitedStateChain(
        mol, hf_factory, spin='singlet', state=0, mf=mf,
        numerics=g1_numerics(mol, Numerics(isdf={'counts': counts, 'n_start': n_start},
                                           qp={'window': list(range(3, EDGE + 1)), 'residue_route': 'sop', 'scissor': 'calibrate', 'select': QPStates('adaptive', targets=TARGETS)},
                                           bse={'solver': 'dense'})))
    StateManifold(chain, states=TARGETS).evaluate(mol, mf)
    part = chain.resolved.qp_partition
    assert EDGE not in part.explicit and INNER in part.explicit
    assert part.tier_of[EDGE] == (INNER,)
    assert chain.selection_record()['window_edge_holes'] == {EDGE: INNER}
    shift = chain.conventions.outside_shift
    assert shift[EDGE] == shift[EDGE + 1]
    unit = [u for u in chain.conventions.units() if EDGE in u.slots]
    assert unit[0].kind == 'shared' and EDGE + 1 in unit[0].slots


# ------------------------------------------------------- the DMABN case
DMABN_SYMBOLS = ['C', 'N', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'N',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']
#: Bohr: DMABN gas-phase S1 minimum of the twisted-start walk
DMABN_S1 = np.array([
    [-4.432036468811, -0.785274037196, -2.048883493684],
    [-2.903578915357, 0.080209090618, 0.052231661882],
    [-4.318331830686, 1.029875651950, 2.196111281800],
    [-0.319081392980, 0.012641786704, 0.013522251150],
    [0.910881237435, -1.569361942683, -1.781191567604],
    [3.605474410135, -1.583470949503, -1.844992035371],
    [4.946488141168, -0.115698541177, -0.063433094732],
    [3.732475988712, 1.413366136208, 1.757000342349],
    [1.039552534600, 1.529917933788, 1.770419019072],
    [7.647882553172, -0.186733920930, -0.100967492682],
    [9.844538680969, -0.245328624728, -0.130938942376],
    [-5.309857395340, -2.623472094342, -1.610959983397],
    [-5.942012306653, 0.604076018904, -2.391685621780],
    [-3.256838804830, -0.935958997889, -3.746503343971],
    [-5.896746661247, -0.268964571637, 2.582260984576],
    [-5.098366561784, 2.917511170228, 1.784133998598],
    [-3.087878112111, 1.108652368846, 3.859055009699],
    [-0.136457753335, -3.086644632221, -2.708143566994],
    [4.630979354889, -2.792767707945, -3.157466733961],
    [4.852838682044, 2.569831250421, 3.039027759313],
    [0.094665353214, 3.095352040768, 2.726788518131]])
DMABN_NOCC = 39
DMABN_WINDOW = tuple(range(22, 58))
#: the quasiparticle-Hamiltonian groups s1.json records
DMABN_GROUPS = (tuple(range(23, 38)), (39, 40), tuple(range(41, 57)))


def dmabn(coords):
    mol = gto.M(atom=list(zip(DMABN_SYMBOLS, coords)), unit='Bohr',
                basis='cc-pvdz', verbose=0)
    mf = dft.RKS(mol, xc='lrc-wpbeh').density_fit()
    mf.conv_tol = 1e-9
    mf.kernel()
    return mol, mf


def frozen(mol, mf, explicit, tier_of):
    """The conventions of a surface with `explicit` solved and `tier_of`
    holes, keyed at (mol, mf): every outside orbital borrows its nearest
    explicit one, as `_freeze_outside_shift` lends it, on stand-in roots."""
    eps = np.asarray(mf.mo_energy, float)
    roots = {p: eps[p] + 0.01 * (p - DMABN_NOCC) for p in explicit}
    outside = [p for p in range(len(eps))
               if p not in explicit and p not in tier_of]
    shift = calibrate_scissor(eps, DMABN_NOCC, roots, outside)
    shift.update(calibrate_scissor_tiers(eps, roots, tier_of))
    c = FrozenConventions(key_mol=mol, key_mo=np.asarray(mf.mo_coeff).copy(),
                          nocc=DMABN_NOCC)
    c.qp_set = np.array(sorted(explicit), int)
    c.qp_subspaces = (tuple(c.qp_set), tuple(
        tuple(p for p in g if p in explicit) for g in DMABN_GROUPS))
    c.outside_shift = shift
    c.anchored = True
    return c


@pytest.fixture(scope='module')
def dmabn_step():
    warnings.simplefilter('ignore')
    mol, mf = dmabn(DMABN_S1)
    moved = dmabn(DMABN_S1 + nudge_vector(DMABN_S1, DMABN_STEP))
    return mol, mf, moved


def test_dmabn_a_lone_explicit_edge_is_lost_on_the_step(dmabn_step):
    """The step is long enough to matter: with 57 a lone explicit orbital,
    as the far-start rebuild had it, the match is refused."""
    mol, mf, (mol2, mf2) = dmabn_step
    with pytest.raises(ConventionsLost, match=r'explicit \[57\]'):
        frozen(mol, mf, set(DMABN_WINDOW), {}).at(mol2, mf2)


def test_dmabn_the_selected_set_survives_the_step(dmabn_step):
    mol, mf, (mol2, mf2) = dmabn_step
    eps = np.asarray(mf.mo_energy, float)
    assert eps[58] - eps[57] < QP_HAMILTONIAN_WINDOW
    start = mandatory_members(eps, DMABN_NOCC, DMABN_WINDOW, DEG)
    a = {p: 0.0 for p in DMABN_WINDOW}
    weights = {TARGETS[0]: np.zeros(len(eps))}
    sel = select_explicit(eps, DMABN_NOCC, DMABN_WINDOW, a, weights, 0.0,
                          start=start, degeneracy_tol=DEG, hole_max_ev=None)
    assert 57 not in sel.explicit and sel.tier_of[57] == (56,)
    c = frozen(mol, mf, set(sel.explicit), sel.tier_of)
    assert c.outside_shift[57] == c.outside_shift[58]
    view = c.at(mol2, mf2)
    assert view is not None
