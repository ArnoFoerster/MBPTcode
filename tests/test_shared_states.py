"""One shared evaluation for several states at one geometry
(`StateManifold.evaluate`, `ExcitedStateChain._shared_forward`/`spin_view`).

G1, bitwise. On ONE mean field and ONE `FrozenFactorization`, chains of each
target (spin, root) evaluated alone, and one manifold over all of them: every
total force, total energy, root index, Casida spectrum (roots and vectors) and
the ground-state force `array_equal`, with the manifold's targets in both
orders (singlet first, triplet first). Cases, water/cc-pVDZ:
  * 'hf-dense', 'pbe0-dense': the dense Casida route, the explicit adjoint,
    Hartree-Fock and a hybrid (the Sigma_x - v_xc chain); S0, S1 and T0, the
    S0-S1 interstate numerator beside them, and a second, displaced geometry
    on the conventions the first froze;
  * 'hf-davidson-grid': the Davidson route with the grid adjoint, S0 beside
    T0 and T1, so one Davidson per spin reads to the highest root asked of it;
  * 'pcm': in a water continuum (`SolventScreening`, the dressed and bare
    gauges, the ground state relaxed at eps_static);
  * 'production': ISDF-K LRC-wPBEh, sum-over-poles residues, the row fit on
    sliced factors, the grid adjoint, whose mean field's exchange skeleton
    rides the first target's fit adjoint;
  * the composed E_HF + E_c^dRPA + Omega surface (`RPABSESurface`), its dRPA
    ground-state force once.
Serially, and 'hf-dense', 'hf-davidson-grid' and 'production' at 2 and 3
simulated ranks, where every rank also holds rank 0's bits.

The quasiparticle tape is held from the forward to the last reverse pass:
proj(tau) is swept ONCE per evaluation, counted at
`qp_space_time.polarizability_projected_rows`. A pinned forward that
released the tape in the first reverse pass would sweep it again for every
further root, and the root-only check fails on it.

A first point: the surface `surface(t, first_point=ev)` answers its first
force at that geometry from the evaluation, without a forward pass, and a
different spin, root or geometry is evaluated. `calc_vertical_states`: each
state's record is `calc_vertical_excitation`'s for that state alone on the
same mean field.
"""
import os
import sys
import warnings

import numpy as np
import pytest
from pyscf import dft, gto, lib, scf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))

from src.Base.constants import (SCF_DIFFERENTIABLE_CONV_TOL,  # noqa: E402
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import Excitation, GroundState, QPStates  # noqa: E402
from src.Base.isdf_jk import isdf_jk  # noqa: E402
from src.Base.separable_ri import resolve_isdf_grid  # noqa: E402
from src.Base.solvent_screening import SolventScreening  # noqa: E402
from src.Base.utils.mpi_grid import current_comm, distributed, run_simulated  # noqa: E402
from src.gradients import qp_space_time  # noqa: E402
from src.gradients.excited_state import ExcitedStateChain  # noqa: E402
from src.gradients.factor_chain import converged_factory  # noqa: E402
from src.gradients.rpa_bse_surface import RPABSESurface  # noqa: E402
from src.gradients.state_manifold import StateManifold  # noqa: E402
from src.properties.excitations import (SurfaceSpec, calc_vertical_excitation,  # noqa: E402
                                        calc_vertical_states, surface_of)

BASIS, AUX = 'cc-pvdz', 'cc-pvdz-ri'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: One hydrogen 0.03 A along y: the second geometry of the dense cases.
H2O_DISPLACED = 'O 0 0 0.117; H 0 0.787 -0.468; H 0 -0.757 -0.468'
#: The ISDF-K mean field's grid level and memory (tests/test_one_fit_adjoint.py's).
GRID_ACCURACY = 'G1'
MAX_MEMORY = 4000
S0, S1 = ('singlet', 0), ('singlet', 1)
T0, T1 = ('triplet', 0), ('triplet', 1)
#: (targets, interstate pairs, second geometry) per case.
TARGETS = {'hf-dense': ((S0, S1, T0), ((S0, S1),), True),
           'pbe0-dense': ((S0, T0), (), True),
           'hf-davidson-grid': ((S0, T0, T1), (), False),
           'pcm': ((S0, T0), (), False),
           'production': ((S0, T0), (), False)}
RANK_CASES = ('hf-dense', 'hf-davidson-grid', 'production')


# ------------------------------------------------------------------ helpers
def molecule(atom=H2O):
    return gto.M(atom=atom, basis=BASIS, verbose=0, max_memory=MAX_MEMORY)


def converged(mf):
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    return mf


def rhf(mol):
    return converged(scf.RHF(mol).density_fit(auxbasis=AUX))


def pbe0(mol):
    return converged(dft.RKS(mol, xc='pbe0').density_fit(auxbasis=AUX))


def in_water(mol):
    """The Hartree-Fock ground state relaxed in the continuum at eps_static,
    which the chain's environment then takes as it is."""
    return SolventScreening(mol, solvent='water').mean_field(mol, rhf)


def isdf_lrc(mol):
    """The ISDF-K LRC-wPBEh mean field, converged serially and left unrun
    over ranks for the distributed SCF."""
    base = dft.RKS(mol, xc='lrc-wpbeh')
    base.max_memory = MAX_MEMORY
    elements = sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})
    counts, n_start = resolve_isdf_grid(GRID_ACCURACY, BASIS, elements,
                                        auxbasis=AUX)
    mf = isdf_jk(base, auxbasis=AUX, counts=counts, n_start=n_start)
    mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    comm = current_comm()
    if comm is None or comm.Get_size() == 1:
        mf.kernel()
    return mf


def one_mean_field_per_geometry(factory):
    """`factory` converging once per geometry, every chain there handed that
    one mean field."""
    made = {}

    def build(mol):
        key = np.asarray(mol.atom_coords()).tobytes()
        if key not in made:
            made[key] = factory(mol)
        return made[key]
    return build


PRODUCTION = SurfaceSpec(GroundState('dft', 'lrc-wpbeh'), environment=None,
                         chi0='space-time', residues='sop', solver='davidson',
                         factorization='isdf',
                         qp_states=QPStates(kind='frontier'),
                         numerics={'grid_accuracy': GRID_ACCURACY,
                                   'sliced': True, 'fit': 'rows',
                                   'bse_adjoint': 'grid'})


def prototype(case, mol):
    """An unevaluated chain of `case` whose mean field and factorization
    every chain of the case shares."""
    if case == 'production':
        factory = one_mean_field_per_geometry(isdf_lrc)
        return surface_of(PRODUCTION, Excitation('singlet', root=1), mol,
                          factory, mf=converged_factory(factory)(mol))
    factory = one_mean_field_per_geometry(
        {'hf-dense': rhf, 'pbe0-dense': pbe0, 'hf-davidson-grid': rhf,
         'pcm': in_water}[case])
    kw = {'hf-davidson-grid': dict(solver='davidson', bse_adjoint='grid'),
          'pcm': dict(environment=SolventScreening(mol, solvent='water'))
          }.get(case, dict(solver='dense'))
    return ExcitedStateChain(mol, factory, mf=factory(mol), **kw)


def fresh(proto, target):
    """A chain of `proto`'s settings, mean field and factorization, nothing
    evaluated on it, at `target`."""
    chain = proto.refreeze(proto.mol0, factorization=proto.factorization)
    chain.spin, chain.state = target
    return chain


def watched(chain):
    """`chain` keeping every spectrum its Casida step returns and every
    mean-field force it computes, in the order made."""
    seen = {'spectra': [], 'g0': []}
    casida, g0 = chain._casida_forward, chain.mean_field_gradient

    def kept_casida(shared):
        om, pieces = casida(shared)
        seen['spectra'].append((om, pieces[10], pieces[11]))
        return om, pieces

    def kept_g0(mf):
        out = g0(mf)
        seen['g0'].append(out)
        return out

    chain._casida_forward, chain.mean_field_gradient = kept_casida, kept_g0
    return seen


def bitwise(a, b):
    """Every array of `a` holds the bits of the same array of `b`."""
    return len(a) == len(b) and all(
        np.asarray(x).shape == np.asarray(y).shape
        and np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


def evaluated_alone(proto, targets, couplings, displaced):
    """{target: (force, energy, root, spectrum, g0)} of each target's own
    chain, at R0 and, if asked, at the displaced geometry; and the
    interstate numerators of fresh chains."""
    out = {}
    for t in targets:
        chain = fresh(proto, t)
        seen = watched(chain)
        g, e, d = chain.total_gradient()
        out[t] = [(g, e, d['root'], seen['spectra'][-1], seen['g0'][-1])]
        if displaced:
            g, e, d = chain.total_gradient(molecule(H2O_DISPLACED))
            out[t].append((g, e, d['root'], seen['spectra'][-1],
                           seen['g0'][-1]))
    for m, n in couplings:
        out[(m, n)] = fresh(proto, m).interstate_gradient(m[1], n[1])[0]
    return out


def evaluated_together(proto, targets, couplings, displaced):
    """The same numbers off one manifold, one evaluation per geometry."""
    man = StateManifold(fresh(proto, targets[0]), states=targets)
    out = {t: [] for t in targets}
    mols = [None] + ([molecule(H2O_DISPLACED)] if displaced else [])
    for i, mol in enumerate(mols):
        ev = man.evaluate(mol, gradients=targets,
                          couplings=couplings if i == 0 else ())
        for t in targets:
            out[t].append((ev.gradient[t], ev.energy[t], ev.info[t]['root'],
                           ev.spectrum[t[0]], ev.g0))
        if i == 0:
            for c in couplings:
                out[c] = ev.interstate[c][0]
    return out


def g1_mismatches(alone, together, targets, couplings):
    """The (target, geometry, quantity) whose bits differ."""
    names = ('force', 'energy', 'root', 'spectrum', 'g0')
    bad = []
    for t in targets:
        for k, (a, b) in enumerate(zip(alone[t], together[t])):
            for name, x, y in zip(names, a, b):
                same = (bitwise(x, y) if name == 'spectrum'
                        else bitwise((x,), (y,)))
                if not same:
                    bad.append((t, k, name))
    bad += [(c, 0, 'interstate') for c in couplings
            if not bitwise((alone[c],), (together[c],))]
    return bad


@pytest.fixture
def pyscf_one_thread():
    """pyscf's OpenMP GEMM on one thread inside the rank threads."""
    threads = lib.num_threads()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lib.num_threads(1)
    yield
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lib.num_threads(threads)


def g1(case):
    """Both orders of `case` against its chains alone: the mismatches, and
    rank 0's comparison material."""
    targets, couplings, displaced = TARGETS[case]
    proto = prototype(case, molecule())
    alone = evaluated_alone(proto, targets, couplings, displaced)
    bad, kept = [], []
    for order in (targets, targets[::-1]):
        together = evaluated_together(proto, order, couplings, displaced)
        bad += [(order[0],) + m for m in
                g1_mismatches(alone, together, targets, couplings)]
        kept.append(together)
    return bad, alone, kept


# ------------------------------------------------------------------ G1
@pytest.mark.parametrize('case', list(TARGETS))
def test_one_evaluation_is_each_state_alone(case):
    with distributed(None):
        bad, alone, _ = g1(case)
    assert not bad, f'{case}: (order, target, geometry, quantity) {bad}'
    targets = TARGETS[case][0]
    # the states are different states: a shared pass that returned one
    # state for all would pass the comparison above only if alone did too
    forces = [alone[t][0][0] for t in targets]
    assert all(np.abs(forces[0] - f).max() > 1e-4 for f in forces[1:])


@pytest.mark.parametrize('size', [2, 3])
@pytest.mark.parametrize('case', RANK_CASES)
def test_one_evaluation_over_ranks(case, size, pyscf_one_thread):
    def rank(comm):
        return g1(case)

    res = run_simulated(rank, size)
    targets, couplings, _ = TARGETS[case]
    for r, (bad, alone, kept) in enumerate(res):
        assert not bad, f'rank {r} of {size}, {case}: {bad}'
        ref = res[0][1]
        for t in targets:
            for got, want in zip(alone[t], ref[t]):
                assert bitwise(got[:3], want[:3]), f'rank {r}: {t} != rank 0'
        for c in couplings:
            assert bitwise((alone[c],), (ref[c],)), f'rank {r}: {c}'


def test_the_composed_surface_takes_its_ground_state_once():
    """E_HF + E_c^dRPA + Omega: the dRPA ground-state force is computed once
    and every target's total is its own surface's, bitwise."""
    with distributed(None):
        mol = molecule()
        factory = one_mean_field_per_geometry(rhf)
        mf = factory(mol)
        first = RPABSESurface(mol, factory, mf=mf)
        alone = {}
        for spin in ('singlet', 'triplet'):
            surface = RPABSESurface(mol, factory, spin=spin, mf=mf,
                                    factorization=first.ground.factorization)
            alone[spin] = surface.total_gradient()
        man = StateManifold(RPABSESurface(
            mol, factory, mf=mf, factorization=first.ground.factorization),
            states=(S0, T0))
        calls, ground = [], man.chain.ground.total_gradient

        def counted(*args):
            calls.append(1)
            return ground(*args)

        man.chain.ground.total_gradient = counted
        ev = man.evaluate(gradients=(T0, S0))
    assert len(calls) == 1
    for t in (S0, T0):
        g, e, _ = alone[t[0]]
        assert bitwise((g, e), (ev.gradient[t], ev.energy[t])), t


# ------------------------------------------------------- one proj(tau) sweep
@pytest.fixture
def sweeps(monkeypatch):
    """The proj(tau) sweeps of the quasiparticle set solve, counted."""
    calls, sweep = [], qp_space_time.polarizability_projected_rows

    def counted(*args, **kwargs):
        calls.append(1)
        return sweep(*args, **kwargs)

    monkeypatch.setattr(qp_space_time, 'polarizability_projected_rows',
                        counted)
    return calls


def test_proj_tau_is_swept_once_for_several_roots(sweeps):
    """Two roots' forces off one manifold: one sweep, the forward's. The
    pinned forward released the tape in the first reverse pass, and this
    counted 2."""
    with distributed(None):
        proto = prototype('hf-dense', molecule())
        StateManifold(fresh(proto, S0), states=(0, 1)).gradients()
    assert len(sweeps) == 1, f'{len(sweeps)} proj(tau) sweeps for two roots'


def test_proj_tau_is_swept_once_per_evaluation(sweeps):
    """Two spins, four forces and a coupling: one sweep per geometry; one
    chain alone sweeps once per force, its forward's."""
    with distributed(None):
        proto = prototype('hf-dense', molecule())
        man = StateManifold(fresh(proto, S0), states=(S0, S1, T0, T1))
        man.evaluate(gradients=(S0, S1, T0, T1), couplings=((T0, T1),))
        assert len(sweeps) == 1
        man.evaluate(molecule(H2O_DISPLACED), gradients=(S1, T0))
        assert len(sweeps) == 2
        fresh(proto, T0).total_gradient()
        assert len(sweeps) == 3


# ------------------------------------------------------------ first point
def test_a_first_point_is_answered_once_without_a_forward(monkeypatch):
    with distributed(None):
        proto = prototype('hf-dense', molecule())
        man = StateManifold(fresh(proto, S0), states=(S0, T0))
        ev = man.evaluate(gradients=(S0, T0))
        forwards, shared = [], ExcitedStateChain._shared_forward

        def counted(self, *args):
            forwards.append(1)
            return shared(self, *args)

        monkeypatch.setattr(ExcitedStateChain, '_shared_forward', counted)
        walk = man.surface(T0, first_point=ev)
        assert (walk.spin, walk.state) == T0
        first = walk.total_gradient(molecule())
        assert forwards == [] and first[0] is ev.gradient[T0]
        again = walk.total_gradient(molecule())
        assert forwards == [1]
        assert np.abs(again[0] - first[0]).max() < 1e-9
        # another state's point, or another geometry, is not replayed
        other = man.surface(S0)
        other.first_point = ev.first_point(T0)
        other.total_gradient()
        assert forwards == [1, 1]
        moved = man.surface(T0, first_point=ev)
        moved.total_gradient(molecule(H2O_DISPLACED))
        assert forwards == [1, 1, 1]
        with pytest.raises(ValueError, match='no force'):
            man.evaluate(states=(S0,)).first_point(S0)


def test_what_one_evaluation_cannot_serve_is_refused():
    with distributed(None):
        proto = prototype('hf-dense', molecule())
        chain = fresh(proto, S0)
        with pytest.raises(RuntimeError, match='evaluate the chain once'):
            chain.spin_view('triplet')
        man = StateManifold(chain, states=(S0, T0))
        with pytest.raises(ValueError, match='ONE Casida solve'):
            man.evaluate(couplings=((S0, T0),))
        with pytest.raises(ValueError, match='not in this manifold'):
            man.evaluate(gradients=(T1,))
        with pytest.raises(ValueError, match='spin'):
            StateManifold(chain, states=(('quintet', 0),))


# ------------------------------------------------------ the vertical records
#: The fields of a vertical record that hold a number of the state rather
#: than a clock, a commit or an object.
RECORD_FIELDS = ('omega_eV', 'e0_hartree', 'en_hartree', 'gradient',
                 'driving_force_max', 'total_translation_residual',
                 'translation_residual', 'qp_z', 'residue_route_taken',
                 'qp_bookkeeping', 'held_grid')


def test_vertical_states_are_each_states_own_record():
    states = (Excitation('singlet', root=1), Excitation('triplet', root=1))
    with distributed(None):
        mol = molecule()
        factory = one_mean_field_per_geometry(isdf_lrc)
        together = calc_vertical_states(PRODUCTION, states, mol, factory)
        for x in states:
            alone = calc_vertical_excitation(PRODUCTION, x, mol, factory)
            got = together[x]
            for key in RECORD_FIELDS:
                assert got[key] == alone[key], (x, key)
            assert got['physics']['label'] == alone['physics']['label']
            assert repr(got['realization']) == repr(alone['realization'])
    with pytest.raises(ValueError, match='more than spin and root'):
        calc_vertical_states(PRODUCTION, (states[0], Excitation(
            'triplet', kernel='bse-tda')), mol, factory)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
