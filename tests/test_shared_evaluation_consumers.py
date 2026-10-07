"""The property layer reads one shared evaluation instead of evaluating again.

`spin_orbit.chain_manifolds(evaluation=)` returns the two spectra a
`StateManifold.evaluate` solved; the gate is that they are, bit for bit, the
spectra `chain_manifolds` computes itself on a fresh chain of the same mean
field and factorization. `derivative_coupling.analytic_coupling` takes its
numerator and its vectors off ONE forward pass, bit for bit the coupling it
computed with two, and off an evaluation's `interstate` the same bits again;
`analytic_ground_coupling` off an evaluation is its own number on a fresh
chain. Water/cc-pVDZ: the Hartree-Fock Davidson route with the grid adjoint,
and the production surface (ISDF-K LRC-wPBEh, sum-over-poles residues,
frontier set with the outside scissor, Davidson, grid adjoint).

Beside them: `vibronic.relax_state` records the total and Omega at the
minimum off ONE forward pass, the record bit for bit the two-pass one;
`track='overlap'` reaches the chain through `potential_energy_surface` and,
where no root crosses, gives the index-following forces bit for bit; and the
finite-difference Hessian's grid floor is refused on the shell counts alone.
"""
import os
import sys

import numpy as np
import pytest
from pyscf import dft, gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import (ISDF_HESSIAN_MIN_GRID,
                                SCF_DIFFERENTIABLE_CONV_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.isdf_jk import isdf_jk
from src.Base.separable_ri import resolve_isdf_grid
from src.gradients.derivative_coupling import (analytic_coupling,
                                               analytic_ground_coupling,
                                               configuration_coupling,
                                               state_to_state_density)
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.state_manifold import StateManifold
from src.properties.excitations import SurfaceSpec, surface_of
from src.properties.hessian import require_hessian_counts
from src.properties.optimize import relax
from src.properties.spin_orbit import chain_manifolds
from src.properties.surface import surface_mean_field
from src.properties.vibronic import relax_state

BASIS, AUX = 'cc-pvdz', 'cc-pvdz-ri'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
GRID_ACCURACY = 'G1'
MAX_MEMORY = 4000
S0, S1, T0, T1 = ('singlet', 0), ('singlet', 1), ('triplet', 0), ('triplet', 1)

PRODUCTION = SurfaceSpec(GroundState('dft', 'lrc-wpbeh'), environment=None,
                         chi0='space-time', residues='sop', solver='davidson',
                         factorization='isdf',
                         qp_states=QPStates(kind='frontier'),
                         numerics={'grid_accuracy': GRID_ACCURACY,
                                   'bse_adjoint': 'grid', 'nroots': 4})


def molecule(atom=H2O):
    return gto.M(atom=atom, basis=BASIS, verbose=0, max_memory=MAX_MEMORY)


def converged(mf):
    mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.kernel()
    return mf


def rhf(mol):
    return converged(scf.RHF(mol).density_fit(auxbasis=AUX))


def isdf_lrc(mol):
    base = dft.RKS(mol, xc='lrc-wpbeh')
    base.max_memory = MAX_MEMORY
    elements = sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})
    counts, n_start = resolve_isdf_grid(GRID_ACCURACY, BASIS, elements,
                                        auxbasis=AUX)
    return converged(isdf_jk(base, auxbasis=AUX, counts=counts,
                             n_start=n_start))


def one_mean_field_per_geometry(factory):
    """`factory` converging once per geometry."""
    made = {}

    def build(mol):
        key = np.asarray(mol.atom_coords()).tobytes()
        if key not in made:
            made[key] = factory(mol)
        return made[key]
    return build


def prototype(case, mol):
    """An unevaluated chain of `case`, its mean field memoized by geometry."""
    if case == 'production':
        factory = one_mean_field_per_geometry(isdf_lrc)
        return surface_of(PRODUCTION, Excitation('singlet', root=1), mol,
                          factory, mf=factory(mol))
    factory = one_mean_field_per_geometry(rhf)
    return ExcitedStateChain(mol, factory, mf=factory(mol), solver='davidson',
                             bse_adjoint='grid', nroots=4)


def fresh(proto, target):
    """A chain of `proto`'s settings, mean field and factorization at `target`."""
    chain = proto.refreeze(proto.mol0, factorization=proto.factorization)
    chain.spin, chain.state = target
    return chain


def same_bits(a, b):
    return all(np.array_equal(np.asarray(x), np.asarray(y))
               for x, y in zip(a, b))


@pytest.mark.parametrize('case', ['hf-davidson-grid', 'production'])
def test_chain_manifolds_reads_the_evaluation(case):
    proto = prototype(case, molecule())
    alone = chain_manifolds(fresh(proto, S0))
    man = StateManifold(fresh(proto, S0), states=(S0, S1, T0, T1))
    ev = man.evaluate()
    shared = chain_manifolds(None, evaluation=ev)
    for spin, a, b in zip(('singlet', 'triplet'), alone, shared):
        assert same_bits(a, b), spin


def coupling_before(chain, m, n):
    """`analytic_coupling` from two forward passes: the numerator's own
    forward, then a second for the vectors."""
    mol, mf = chain.mean_field()
    num, diags = chain.interstate_gradient(m, n, mol=mol, mf=mf)
    _, pieces = chain._forward(mol, mf)
    xn, yn = pieces[10], pieces[11]
    goo, gvv = state_to_state_density(chain.nocc, xn[:, m], yn[:, m],
                                      xn[:, n], yn[:, n])
    return (num / diags['gap']
            + configuration_coupling(mol, mf, chain.nocc, goo, gvv))


@pytest.mark.parametrize('case', ['hf-davidson-grid', 'production'])
def test_analytic_coupling_one_forward_and_off_the_evaluation(case):
    proto = prototype(case, molecule())
    before = {spin: coupling_before(fresh(proto, (spin, 0)), 0, 1)
              for spin in ('singlet', 'triplet')}
    for spin in ('singlet', 'triplet'):
        now, _ = analytic_coupling(fresh(proto, (spin, 0)), 0, 1)
        assert np.array_equal(now, before[spin]), spin
    man = StateManifold(fresh(proto, S0), states=(S0, S1, T0, T1))
    ev = man.evaluate(couplings=((S0, S1), (T0, T1)))
    for spin, pair in (('singlet', (S0, S1)), ('triplet', (T0, T1))):
        shared, _ = analytic_coupling(None, *pair, evaluation=ev)
        assert np.array_equal(shared, before[spin]), spin
    with pytest.raises(ValueError, match='interstate'):
        analytic_coupling(None, S1, S0, evaluation=ev)


@pytest.mark.parametrize('case', ['hf-davidson-grid', 'production'])
def test_analytic_ground_coupling_off_the_evaluation(case):
    proto = prototype(case, molecule())
    alone, info = analytic_ground_coupling(fresh(proto, S0), 0)
    man = StateManifold(fresh(proto, S0), states=(S0, T0))
    ev = man.evaluate()
    shared, info_ev = analytic_ground_coupling(None, S0, evaluation=ev)
    assert np.array_equal(shared, alone)
    assert info_ev == info
    with pytest.raises(ValueError, match='singlet'):
        analytic_ground_coupling(None, T0, evaluation=ev)


def counted_forwards(chain):
    """`chain` counting its shared forward passes."""
    seen = {'n': 0}
    shared = chain._shared_forward

    def counted(mol, mf):
        seen['n'] += 1
        return shared(mol, mf)

    chain._shared_forward = counted
    return seen


def relaxed_before(surface, mol, **kw):
    """`relax_state`'s record with the total and Omega each off its own
    forward pass at the minimum."""
    mol_opt, info = relax(surface, mol, **kw)
    mf = surface_mean_field(surface, mol_opt)
    return {'mol': mol_opt, 'e_total': surface.total_energy(mol_opt, mf),
            'e_scf': float(mf.e_tot),
            'omega': surface.excitation(mol_opt, mf)}


@pytest.mark.parametrize('spin', ['singlet', 'triplet'])
def test_relax_state_takes_one_forward_at_the_minimum(spin):
    proto = prototype('hf-davidson-grid', molecule())
    walk = dict(engine='cartesian', max_cycle=2, verbose=False)
    before = relaxed_before(fresh(proto, (spin, 0)), None, **walk)
    chain = fresh(proto, (spin, 0))
    record = relax_state(chain, None, **walk)
    assert np.array_equal(record['mol'].atom_coords(),
                          before['mol'].atom_coords())
    for name in ('e_total', 'e_scf', 'omega'):
        assert record[name] == before[name], name
    # the walk's own evaluations, then one at the minimum
    chain = fresh(proto, (spin, 0))
    walked = counted_forwards(chain)
    relax(chain, None, **walk)
    chain = fresh(proto, (spin, 0))
    recorded = counted_forwards(chain)
    relax_state(chain, None, **walk)
    assert recorded['n'] == walked['n'] + 1


def test_track_reaches_the_production_surface():
    """`track='overlap'` through `potential_energy_surface`: the chain follows
    by overlap and records it, and where no root crosses, the forces at R0 and
    at a displaced geometry are bit for bit the index-following surface's."""
    mol = molecule()
    factory = one_mean_field_per_geometry(isdf_lrc)
    tracked_spec = SurfaceSpec(
        PRODUCTION.ground_state, chi0='space-time', residues='sop',
        solver='davidson', factorization='isdf',
        qp_states=QPStates(kind='frontier'),
        numerics=dict(PRODUCTION.numerics, track='overlap'))
    surfaces = [surface_of(spec, Excitation('singlet', root=1), mol, factory,
                           mf=factory(mol))
                for spec in (PRODUCTION, tracked_spec)]
    by_index, by_overlap = surfaces
    assert by_overlap.track == 'overlap' and by_index.track is None
    assert by_overlap.numerics['track'] == 'overlap'
    assert 'track' not in by_index.numerics
    moved = molecule('O 0 0 0.127; H 0 0.767 -0.468; H 0 -0.757 -0.478')
    for m in (mol, moved):
        g_a, e_a, _ = by_index.total_gradient(m, factory(m))
        g_b, e_b, _ = by_overlap.total_gradient(m, factory(m))
        assert np.array_equal(g_a, g_b) and e_a == e_b
    assert [step['index'] for step in by_overlap.follow_log] == [0, 0]
    assert by_overlap.follow_log[1]['weight'] > 0.9


def test_track_is_refused_where_no_state_is_followed():
    mol = molecule()
    spec = SurfaceSpec(PRODUCTION.ground_state, chi0='space-time',
                       residues='sop', solver='davidson', factorization='isdf',
                       qp_states=QPStates(kind='frontier'),
                       numerics=dict(PRODUCTION.numerics, track='index'))
    with pytest.raises(ValueError, match='track'):
        surface_of(spec, Excitation('singlet', root=1), mol, isdf_lrc)


def test_hessian_grid_floor_is_checked_on_counts_before_any_scf():
    """`require_hessian_counts` refuses a grid below ISDF_HESSIAN_MIN_GRID
    from its shell counts alone, and passes the floor itself."""
    floor = resolve_isdf_grid(ISDF_HESSIAN_MIN_GRID, BASIS, ['C', 'H', 'O'],
                              auxbasis=AUX)[0]
    below = resolve_isdf_grid(GRID_ACCURACY, BASIS, ['C', 'H', 'O'],
                              auxbasis=AUX)[0]
    shells = ('A1', 'A2', 'A3', 'B1')
    require_hessian_counts(BASIS, {el: tuple(floor[s] for s in shells)
                                   for el in 'CHO'})
    with pytest.raises(NotImplementedError, match=ISDF_HESSIAN_MIN_GRID):
        require_hessian_counts(BASIS, {el: tuple(below[s] for s in shells)
                                       for el in 'CHO'})
    with pytest.raises(NotImplementedError, match='no ISDF grid'):
        require_hessian_counts('sto-3g', {'H': (1, 1, 1, 1)})


class Harmonic:
    """E = |R - R*|^2 / 2 about a point 0.05 Bohr from the start, recording
    every geometry it is evaluated at."""

    def __init__(self, mol):
        self.mol0 = mol
        self.minimum = mol.atom_coords() + 0.05
        self.seen = []

    def total_gradient(self, mol=None, mf=None):
        x = (self.mol0 if mol is None else mol).atom_coords()
        self.seen.append(np.array(x))
        d = x - self.minimum
        return d, 0.5 * float((d ** 2).sum()), {}

    def total_energy(self, mol=None, mf=None):
        return self.total_gradient(mol)[1]

    def refreeze(self, mol):
        return self


def test_geometric_walk_evaluates_its_start_at_its_own_bits():
    """geomeTRIC hands the start back through its own Bohr radius (~1e-10
    Bohr off); the walk's first evaluation is the start bit for bit, which a
    surface's first point needs to answer it."""
    mol = molecule()
    surface = Harmonic(mol)
    relax(surface, mol, engine='geometric', verbose=False)
    assert np.array_equal(surface.seen[0], mol.atom_coords())
    assert len(surface.seen) > 1


def test_chain_manifolds_refuses_a_missing_spin():
    proto = prototype('hf-davidson-grid', molecule())
    ev = StateManifold(fresh(proto, S0), states=(S0,)).evaluate()
    with pytest.raises(ValueError, match='triplet'):
        chain_manifolds(None, evaluation=ev)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
