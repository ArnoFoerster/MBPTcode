"""`ExcitedStateChain` says what it computes, and the entry point CHECKS it.

Three rules meet here.

DECLARATION. The chain carries `physics_ground_state` (E_KS, the mean field's
own functional -- this chain adds Omega to `mf.e_tot` and no correlation term),
`physics_excitation` (the spin, the kernel `bse_tda` selects and the followed
root) and the `SurfacePhysics` they assemble with its environment. Because the
class declares it, `potential_energy_surface` does not STAMP a label over
whatever was constructed: it compares the declaration it built from the user's
`GroundState`/`Excitation` against the class's own and refuses the pair naming
both. A stamp can only ever agree with itself.

ONE SOLVER RULE. `solver_used` resolves 'auto' through production's
`bse.solver_choice` instead of re-spelling the memory rule (a `dense_max_nov`
cap on it is refused). The Tamm-Dancoff form above the
rule goes matrix-free like the full kernel (`solve_casida_davidson(tda=True)`).
`calc_qp_energy(mode='casida')` reads the same rule and refuses for a stronger
reason -- its Sigma_c is a Lehmann sum over EVERY Casida root, so no Davidson
spectrum can feed it at any size.

NO IMPORT ORDER. `src.properties` and `src.gradients` import cleanly in either
order in a fresh interpreter: `driven_chain` and the root-attribute vocabulary
live in `src.gradients.surface`, below both, so `nonadiabatic` reaches into
`src.gradients.gwbse.state_manifold` nowhere.

What breaks each gate:

  * `physics_ground_state` returning `declared_ground_state(self.mf0, 'rpa')`
    instead of 'dft', so the chain declares E_HF + E_c^dRPA under a surface
    that computes `mf.e_tot + Omega`: test_the_chain_declares_the_mean_field_
    it_adds_omega_to and test_the_dispatcher_checks_what_the_class_declares.
    Both catch it, from the class's side and the entry point's.
  * `solver_used` flipping the rule ('davidson' where `solver_choice` says
    'dense'): test_solver_used_is_the_one_rule at n_ov = 1, 11999 and 12000.
  * `declare_physics` skipping the comparison, which is a stamp in all but
    name: test_the_dispatcher_refuses_a_declaration_the_class_does_not_carry.
  * `SurfaceSpec` carrying a `comm: object = None` field:
    test_a_spec_takes_no_communicator. A spec is a declaration, and a
    communicator travelling in one is a second way to the ranks that the
    context would have to agree with.
  * `nonadiabatic` importing `driven_chain` from `src.gradients.gwbse.state_manifold`
    at module scope, which is a cycle: only the properties-first order of
    test_either_import_order_works_in_a_fresh_interpreter fails, the
    gradients-first one resolves. That asymmetry is the content of the gate.
"""
import os
import subprocess
import sys
from dataclasses import fields

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto, scf

from tests.pins import assert_same
from src.Base.numerics.numerics import Numerics
from src.Base.numerics.constants import (BSE_DENSE_MAX_NOV, SCF_DIFFERENTIABLE_GRAD_TOL,
                                         SCF_TIGHT_CONV_TOL)
from src.Base.declaration import (ChargedExcitation, Excitation, GroundState,
                                  SurfacePhysics)
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.LinearResponse.casida import solver_choice
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.properties.excitations import SurfaceSpec, surface_of
from src.gradients.gwbse.surface_factory import declare_physics, potential_energy_surface
from tests.isdf_test_grid import g1, g1_numerics
from tests.reference_scf import rhf_tight

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'

#: Pair counts either side of the memory rule's boundary, which
#: BSE_DENSE_MAX_NOV writes as a pair count.
PAIR_COUNTS = (1, BSE_DENSE_MAX_NOV - 1, BSE_DENSE_MAX_NOV,
               BSE_DENSE_MAX_NOV + 1, 10 * BSE_DENSE_MAX_NOV)


def water():
    return gto.M(atom=H2O, basis=BASIS, verbose=0)


def chain_scf(mol):
    """A density-fitted mean field, the one the cubic chains are built on."""
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol = SCF_TIGHT_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.kernel()
    return mf


def pbe0(mol):
    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
    mf.kernel()
    return mf


@pytest.fixture(scope='module')
def mol():
    return water()


@pytest.fixture(scope='module')
def ks(mol):
    return pbe0(mol)


@pytest.fixture(scope='module')
def hf(mol):
    return rhf_tight(mol)


# ------------------------------------------------------------- declaration
def test_the_chain_declares_the_mean_field_it_adds_omega_to(mol, ks):
    """E_0 is E_KS, because `energy` returns `mf.e_tot + Omega`.

    Declaring the dRPA ground state here instead would put E_c^dRPA -- 6.3 eV
    on water/cc-pVDZ -- into the label of a number that does not carry it,
    which is the 0.9 eV adiabatic disagreement the declaration exists to stop.
    `RPABSESurface` is the surface that DOES carry it, and composes this chain
    with `RPAGroundStateChain` to do so.
    """
    chain = ExcitedStateChain(mol, pbe0, mf=ks, numerics=g1_numerics(mol))
    assert chain.physics_ground_state == GroundState('dft', 'pbe0')
    assert chain.physics.ground_state.kind == 'dft'
    e_tot, e_0, omega = chain.energy(mol)
    assert e_0 == ks.e_tot and e_tot == e_0 + omega


def test_a_hartree_fock_reference_declares_itself(mol, hf):
    """`GroundState('dft', 'hf')` is E_HF: the kind is which FUNCTIONAL E_0 is,
    and a Hartree-Fock mean field's own energy is one of them."""
    chain = ExcitedStateChain(mol, rhf_tight, mf=hf, numerics=g1_numerics(mol))
    assert chain.physics_ground_state == GroundState('dft', 'hf')


@pytest.mark.parametrize('spin', ['singlet', 'triplet'])
@pytest.mark.parametrize('tda', [False, True])
def test_the_excitation_is_the_chains_own_settings(mol, ks, spin, tda):
    """Spin, kernel and root come off the constructor arguments.

    `state` is a zero-based index into the energy-ordered manifold and
    `Excitation.root` is one-based; an off-by-one here would label root n as
    root n+1 in every record built from the surface.
    """
    chain = ExcitedStateChain(mol, pbe0, mf=ks, spin=spin, state=2,
                              bse_tda=tda, numerics=g1_numerics(mol))
    assert chain.physics_excitation == Excitation(
        spin, root=3, kernel='bse-tda' if tda else 'bse')
    assert chain.physics == SurfacePhysics(GroundState('dft', 'pbe0'),
                                           chain.physics_excitation)


# -------------------------------------------------------- dispatcher check
def test_the_dispatcher_checks_what_the_class_declares(mol, ks):
    """The positive case: the entry point's declaration and the chain's agree.

    And `physics` is a CLASS-level property, so the agreement is a check and
    not a stamp that agreed with itself.
    """
    surface = potential_energy_surface(
        mol, pbe0, ground_state=GroundState('dft', 'pbe0'),
        excitation=Excitation('triplet', root=2, kernel='bse-tda'))
    assert isinstance(type(surface).physics, property)
    assert surface.physics == SurfacePhysics(
        GroundState('dft', 'pbe0'),
        Excitation('triplet', root=2, kernel='bse-tda'))
    assert surface.bse_tda is True and surface.state == 1
    assert surface.spin == 'triplet'


def test_the_dispatcher_refuses_a_declaration_the_class_does_not_carry(mol, ks):
    """The negative case: `bse-tda` asked for, a full-kernel chain built.

    The two labels are the SAME string -- `SurfacePhysics.label` names E_0 and
    'Omega', not the kernel -- so the refusal has to name the states, or a
    Tamm-Dancoff number would come back under the full kernel's name with
    nothing in the message able to tell them apart.
    """
    chain = ExcitedStateChain(mol, pbe0, mf=ks, spin='singlet', bse_tda=False, numerics=g1_numerics(mol))
    asked = SurfacePhysics(GroundState('dft', 'pbe0'),
                           Excitation('singlet', kernel='bse-tda'))
    with pytest.raises(ValueError) as refusal:
        declare_physics(chain, asked)
    message = str(refusal.value)
    assert 'ExcitedStateChain' in message
    assert "kernel='bse-tda'" in message and "kernel='bse'" in message
    # and the chain keeps its own declaration, unstamped
    assert chain.physics.excitation.kernel == 'bse'


def test_a_class_without_a_declaration_still_gets_the_stamp():
    """The rule is check-where-declared, not check-everywhere: an object with
    no `physics` of its own keeps the label, which is what lets
    `compare_surfaces` and `describe` work for it at all."""
    class Bare:
        pass

    surface, asked = Bare(), SurfacePhysics(GroundState('dft', 'hf'), None)
    declare_physics(surface, asked)
    assert surface.physics is asked


# -------------------------------------------------------------- one solver
@pytest.mark.parametrize('n_ov', PAIR_COUNTS)
def test_solver_used_is_the_one_rule(mol, hf, n_ov):
    """'auto' on the chain IS `solver_choice`, on both sides of the boundary.

    Two spellings of one rule is how a chain and the entry point that
    dispatched it came to disagree about which solver was going to run.
    """
    chain = ExcitedStateChain(
        mol, rhf_tight, mf=hf, numerics=g1_numerics(mol, Numerics(bse={'solver': 'auto'})))
    assert chain.solver_used(n_ov) == solver_choice(n_ov)


def test_the_pair_count_cap_is_retired(mol, hf):
    """`dense_max_nov` duplicated the memory rule's own boundary
    (BSE_DENSE_MAX_GB is 2 * BSE_DENSE_MAX_NOV**2 * 8 bytes) and is refused;
    `solver='davidson'` is how a caller forces the matrix-free route."""
    with pytest.raises(TypeError, match='dense_max_nov'):
        ExcitedStateChain(
            mol, rhf_tight, mf=hf, dense_max_nov=1,
            numerics=g1_numerics(mol, Numerics(bse={'solver': 'auto'})))


def test_an_explicit_solver_is_not_resolved(mol, hf):
    """A named solver runs whatever the rule would have picked: the rule is a
    default, not a veto, and a 'dense' asked for at size is the caller paying
    the memory deliberately."""
    chain = ExcitedStateChain(
        mol, rhf_tight, mf=hf, numerics=g1_numerics(mol, Numerics(bse={'solver': 'dense'})))
    assert chain.solver_used(10 * BSE_DENSE_MAX_NOV) == 'dense'


def test_tamm_dancoff_above_the_rule_goes_matrix_free(mol, hf):
    """TDA is not exempt from the memory rule: above it the Tamm-Dancoff
    problem is solved matrix-free on the A half of the block action."""
    chain = ExcitedStateChain(
        mol, rhf_tight, mf=hf, bse_tda=True,
        numerics=g1_numerics(mol, Numerics(bse={'solver': 'auto'})))
    assert chain.solver_used(BSE_DENSE_MAX_NOV) == 'dense'
    assert chain.solver_used(BSE_DENSE_MAX_NOV + 1) == 'davidson'


def test_the_casida_quasiparticle_route_has_no_matrix_free_solve(mol, hf):
    """`calc_qp_energy(mode='casida')` reads `solver`, and refuses two of three.

    Its Sigma_c is a Lehmann sum over EVERY neutral excitation, and a Davidson
    returns the lowest `nroots` of them, so the refusal is not about memory at
    all -- it is that the route's self-energy needs a spectrum the matrix-free
    solver does not produce. 'auto' is the same rule the BSE uses and resolves
    to 'dense' at this size, which is what the default computes.
    """
    homo = mol.nelectron // 2 - 1
    with pytest.raises(NotImplementedError, match='Lehmann'):
        calc_qp_energy(hf, state=homo, mode='casida', solver='davidson')
    with pytest.raises(TypeError, match='does not read'):
        calc_qp_energy(hf, state=homo, mode='space-time', solver='dense', counts=g1(hf))
    auto = calc_qp_energy(hf, state=homo, mode='casida')
    # two solves of one root: the same to a few ulp, not bitwise
    assert_same(auto, calc_qp_energy(hf, state=homo, mode='casida',
                                     solver='dense'),
                unit='eV', quantity='energy', label="'auto'")


# ------------------------------------------------------------ import order
@pytest.mark.parametrize('order', [('src.properties', 'src.gradients'),
                                   ('src.gradients', 'src.properties')])
def test_either_import_order_works_in_a_fresh_interpreter(order):
    """The cycle is gone, so neither package has to be imported first.

    `nonadiabatic` does not reach `src.gradients.gwbse.state_manifold` from inside
    `spectrum_solve`; `driven_chain` and the root-attribute vocabulary live
    in `src.gradients.surface`, below both. A subprocess, because an import
    order can only be tested on an interpreter that has imported neither.
    """
    first, second = order
    out = subprocess.run(
        [sys.executable, '-c',
         f'import {first}; import {second}; '
         f'from src.gradients.surface import driven_chain; print("ok")'],
        cwd=REPO, capture_output=True, text=True, env=dict(os.environ))
    assert out.returncode == 0, out.stderr[-2000:]
    assert 'ok' in out.stdout


def test_a_charged_state_declares_the_orbital_it_removes(mol, hf):
    """The composed quasiparticle surface's declaration is a
    `ChargedExcitation`, and it has to survive the round trip through the
    entry point's own `state = orbital - (nocc - 1)` arithmetic."""
    homo = mol.nelectron // 2 - 1
    surface = potential_energy_surface(
        mol, rhf_tight, ground_state=GroundState('rpa', 'hf'),
        excitation=ChargedExcitation(homo, -1))
    assert surface.physics.excitation == ChargedExcitation(homo, -1)
    assert np.sign(surface.sign) == -1


# ------------------------------------------------------- spec under ranks
def ground_energy(spec, factory):
    """E_0 through `surface_of`, on this rank's OWN Mole.

    A Mole per rank because two rank THREADS of one process sharing one
    corrupt each other's `with_rinv_at_nucleus` origin; the spec is the same
    declaration either way, which is the point of the comparison.
    """
    return surface_of(spec, None, water(), factory).total_energy()


def test_a_spec_builds_one_surface_on_every_rank():
    """Two simulated ranks read the SAME BITS as one: the context reaches the
    kernels of the surface a spec builds, and nothing in the spec says how
    many ranks there are."""
    spec = SurfaceSpec(GroundState('rpa', 'hf'))
    serial = ground_energy(spec, chain_scf)
    for energy in run_simulated(lambda comm: ground_energy(spec, chain_scf),
                                2):
        # each rank solves its own SCF: a few ulp on the recording BLAS,
        # within the energy class bar on another
        assert_same(energy, serial, label='rank energy against serial')


def test_a_spec_takes_no_communicator():
    """How many ranks added the same terms up is not part of a declaration."""
    assert 'comm' not in {f.name for f in fields(SurfaceSpec)}
    with pytest.raises(TypeError, match='comm'):
        SurfaceSpec(GroundState('rpa', 'hf'), comm=object())
    assert 'comm' not in SurfaceSpec(GroundState('rpa', 'hf')).as_kwargs()


def test_a_dense_row_runs_whole_on_every_rank():
    """The dense quasi-boson route holds the whole (pq|rs) in one process and
    reaches no distributed kernel, so under the context every rank computes
    the serial surface on its own."""
    spec = SurfaceSpec(GroundState('rpa', 'hf'), chi0='dense-qb',
                       factorization='four-index')
    serial = ground_energy(spec, rhf_tight)
    for energy in run_simulated(lambda comm: ground_energy(spec, rhf_tight), 2):
        # two SCF solves: a few ulp on the recording BLAS, class bar elsewhere
        assert_same(energy, serial, label='rank energy against serial')
