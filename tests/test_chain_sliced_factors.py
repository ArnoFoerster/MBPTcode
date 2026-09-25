"""The gradient chains on sliced ISDF factors: each rank its grid rows between
and inside the stages, every whole-array read one gather per sweep or solve,
and every number the whole layout's at the same rank count -- bit for bit
where pyscf repeats its bits.

A factorization built with `sliced=True` hands its chains, over more than one
rank, ONE `SlicedFactors` per geometry: each rank's `contiguous_block` of the
grid rows of X_mo, D and X_ao, cut from the whole products after they are
formed exactly as the whole layout forms them. Gated over 2, 3 and 8
simulated ranks (`run_simulated`, threads of this process with the real
collectives) on water/cc-pVDZ Hartree-Fock at 148 points per atom, each rank
on its own Mole (the nuclear gradient writes `mol._env` in place), against
the whole layout on the same rank count:

  * serially the flag is inert: the whole arrays come back and every force is
    the unsliced one;
  * the composed state-pair surface (`RPABSESurface`, dRPA ground state plus
    the BSE@GW singlet through the Davidson on rows): force, energy, the BSE
    roots and vectors, the dRPA force and a one-cycle optimizer walk the
    whole layout's, and every rank's equal to rank 0's. Both layouts are
    handed ONE mean field at every geometry (`one_mean_field_per_geometry`),
    so the layout and each chain's own pyscf K builds are all that differ,
    and the sliced numbers are gated within `COMPOSED_GRAD_K` times what a
    repeat of the whole layout on the same mean fields moves them: bitwise
    here, where pyscf repeats its bits and the repeat moves nothing. A threaded
    pyscf, whose OpenMP GEMM adds its K partials in thread-arrival order,
    moves them run to run; tests/test_mpi_routes.py compares the layouts
    there on the spread of several repeats, and
    tests/test_force_serial_shaped.py shows the layout bitwise on a
    shape-sensitive BLAS and every rank rank 0's under an emulated race;
  * THE MEMORY ASSERTION, between steps: at every evaluation the walk asks
    for, the arrays reachable from the surface hold no factor-shaped array
    with the whole grid on an axis, and every rows object it holds is its
    rank's block of X_mo, D and X_ao and the grid points -- read off the
    objects. The same scan on the whole layout finds the fit it caches, so
    the scan can fail;
  * THE GATHER COUNT: one evaluation gathers X_mo, D, X_o and X_v a fixed
    number of times, one per sweep or solve, and the same number when the
    tau grids grow or the Davidson iterates longer;
  * the excited chain alone (dense and Davidson Casida steps), its
    quasiparticle force and its interstate element, and the charged surface,
    the whole layout's on slices;
  * every layout comparison on ONE mean field per geometry
    (`one_mean_field_per_geometry`) and within `COMPOSED_GRAD_K` times what
    a repeat of the whole layout moves the number (`within_repeat`): bitwise
    here, where pyscf repeats its bits;
  * what slices cannot serve is refused by name: a reaction field before any
    SCF, a chain whose kernels read the factors whole, a layout that
    contradicts a shared factorization, Eq. (18)'s densities, and a dense
    row handed the layout keyword.

SHOWN TO FAIL, each once, then restored and byte-compared (`cmp`):
  * `sliced_factors_at` also keeping the whole D on the factorization on
    rank 1 (`self.factorization._whole_D = d` where `comm.Get_rank() == 1`):
    `test_state_pair_chain_on_slices` failed at 2, 3 and 8 ranks on the
    between-steps scan, rank 1 holding D (444, 84) whole at each of the three
    evaluations of the walk, every force still bitwise;
  * `polarizability_backward` re-gathering D at every tau point:
    `test_gathers_are_one_per_sweep` failed at 2 and 3 ranks, D 8 -> 22
    gathers per composed force (one more per point of the static screening's
    14-point grid), while `test_state_pair_chain_on_slices` stayed green: a
    gather moves bytes, not bits.

HEXAMER ESTIMATE (the whole per-stage table is in
`src/gradients/factor_chain.py`'s module docstring), per rank over 8 ranks,
factor arrays only, from M 117762, nmo = nao 10980, nocc 972, naux 28236
(14721 rows per rank), GB whole -> sliced:

  between steps                      43.3 -> 5.9 per live geometry
  chi0/W adjoint sweep               127.6 -> 79.8 (adjoints 36.9 whole)
  quasiparticle self-energy solve    90.6 -> 53.2
  BSE Davidson, per trial vector     90.6 -> 8.0
"""
import os
import sys
import types
import weakref

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import COMPOSED_GRAD_K
from src.Base.declaration import Excitation, GroundState
from src.Base.sliced_factors import SlicedFactors
from src.Base.solvent_screening import SolventScreening
from src.Base.utils.mpi_grid import (contiguous_block, distributed,
                                     run_simulated)
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.factor_chain import FactorChain, FrozenFactorization
from src.gradients.reaction_field_adjoint import static_screening
from src.gradients.rpa_bse_surface import RPABSESurface, RPAQPSurface
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.properties.optimize import optimize
from src.properties.surface import evaluate
from src.properties.surfaces import potential_energy_surface

SIZES = [2, 3, 8]
BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: One hydrogen 0.03 A along y: a geometry the chain fits and differentiates
#: on its own, away from the reference's cached rows.
H2O_DISPLACED = 'O 0 0 0.117; H 0 0.787 -0.468; H 0 -0.757 -0.468'
#: What one composed force at a displaced geometry gathers, by name: the dRPA
#: sweep X_mo + D and its assembly X_mo; the static W D + X_o + X_v, the
#: quasiparticle solve X_mo + D, the Davidson D + X_o, the BSE cache X_mo + D,
#: its adjoint X_mo + D, the quasiparticle adjoint X_mo + D, the chi0 adjoint
#: D + X_o + X_v and the excited assembly X_mo.
COMPOSED_GATHERS = {'X_mo': 7, 'D': 8, 'X_o': 3, 'X_v': 2}


def chain_scf(mol):
    """A mean field converged for gradient work (conv_tol_grad 1e-11)."""
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    return mf


def one_mean_field_per_geometry():
    """`chain_scf` converging once per geometry and handing that mean field to
    every surface that asks there, so two layouts share one."""
    converged = {}

    def factory(mol):
        key = np.asarray(mol.atom_coords()).tobytes()
        if key not in converged:
            converged[key] = chain_scf(mol)
        return converged[key]
    return factory


def own_water(atom=H2O):
    return gto.M(atom=atom, basis=BASIS, verbose=0)


def state_pair(sliced, scf_factory=chain_scf, **kw):
    """The composed singlet surface on this rank's OWN Mole, the Casida step
    matrix-free so the block action runs on rows."""
    mol = own_water()
    return RPABSESurface(mol, scf_factory, spin='singlet', mf=scf_factory(mol),
                         solver='davidson', sliced=sliced, **kw)


def reachable_arrays(root):
    """Every ndarray reachable from `root` through instance attributes,
    containers and weak dictionaries. Communicators are not entered: a
    simulated world holds every rank's buffers."""
    seen, found, stack = set(), [], [root]
    skip = (types.ModuleType, type, types.FunctionType, types.MethodType,
            types.BuiltinFunctionType, str, bytes, int, float, complex)
    while stack:
        obj = stack.pop()
        if id(obj) in seen or isinstance(obj, skip):
            continue
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            found.append(obj)
        elif 'Comm' in type(obj).__name__:
            continue
        elif isinstance(obj, (weakref.WeakKeyDictionary,
                              weakref.WeakValueDictionary)):
            stack.extend(obj.keys())
            stack.extend(obj.values())
        elif isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        elif isinstance(getattr(obj, '__dict__', None), dict):
            stack.extend(vars(obj).values())
    return found


def reachable_rows(root):
    """Every `SlicedFactors` reachable from `root`, once each."""
    seen, found, stack = set(), [], [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen or isinstance(obj, (types.ModuleType, type, str)):
            continue
        seen.add(id(obj))
        if isinstance(obj, SlicedFactors):
            found.append(obj)
        elif 'Comm' in type(obj).__name__ or isinstance(obj, np.ndarray):
            continue
        elif isinstance(obj, (weakref.WeakKeyDictionary, dict)):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        elif isinstance(getattr(obj, '__dict__', None), dict):
            stack.extend(vars(obj).values())
    return found


def whole_factor_arrays(root, npts, widths):
    """Shapes of the reachable float arrays that carry the whole grid on one
    axis and a factor's width on the other: a whole X_mo, X_ao, D or fit."""
    return sorted({a.shape for a in reachable_arrays(root)
                   if a.ndim == 2 and a.dtype.kind == 'f'
                   and ((a.shape[0] == npts and a.shape[1] in widths)
                        or (a.shape[1] == npts and a.shape[0] in widths))})


def rows_held(rows, rank, size):
    """(what `rows` holds, this rank's block of the grid): bytes by name."""
    r0, r1 = contiguous_block(rows.npts, rank, size)
    n = r1 - r0
    return rows.held_bytes(), {'X_mo': n * rows.nmo * 8,
                               'D': n * rows.naux * 8,
                               'X_ao': n * rows.nao * 8,
                               'coords': rows.npts * 3 * 8}


def watch_between_steps(surface, log):
    """Scan `surface` at the start of every evaluation, which is what the
    previous step left behind, and log (whole arrays, rows held, block)."""
    evaluate_now = surface.total_gradient
    chain = surface.ground
    npts = chain.M
    widths = {chain.mf0.mo_coeff.shape[1], chain.mol0.nao, chain.naux}

    def scanned(mol=None, mf=None):
        log.append(whole_factor_arrays(surface, npts, widths))
        return evaluate_now(mol, mf)

    surface.total_gradient = scanned


def bitwise(a, b):
    """Every array of `a` holds the bits of the same array of `b`."""
    return len(a) == len(b) and all(
        np.asarray(x).shape == np.asarray(y).shape
        and np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


def spectrum_and_vectors(surface):
    """(Omega, X, Y) of the excited half at its reference geometry."""
    ex = surface.excited
    om, pieces = ex._forward(ex.mol0, ex.mf0)
    return om, pieces[10], pieces[11]


def within_repeat(got, ref, repeat):
    """Every array of `got` within `COMPOSED_GRAD_K` times what `repeat`, the
    same calculation as `ref` run again, moved it: bitwise where the repeat
    is."""
    return len(got) == len(ref) == len(repeat) and all(
        np.asarray(x).shape == np.asarray(y).shape
        and np.abs(np.asarray(x) - np.asarray(y)).max()
        <= COMPOSED_GRAD_K * np.abs(np.asarray(z) - np.asarray(y)).max()
        for x, y, z in zip(got, ref, repeat))


def test_serially_the_flag_is_inert():
    with distributed(None):
        scf_factory = one_mean_field_per_geometry()
        whole, sliced, repeat = (state_pair(flag, scf_factory)
                                 for flag in (None, True, None))
        x_mo, d, *_ = sliced.ground.factors_at(sliced.ground.mol0,
                                               sliced.ground.mf0)
        assert isinstance(x_mo, np.ndarray) and isinstance(d, np.ndarray)
        gw, ew, _ = whole.total_gradient()
        gs, es, ds = sliced.total_gradient()
        gr, er, _ = repeat.total_gradient()
    assert within_repeat((gs, es), (gw, ew), (gr, er))
    assert 'factor_gathers' not in ds


@pytest.mark.parametrize('size', SIZES)
def test_state_pair_chain_on_slices(size):
    """Force, energy, roots, vectors, dRPA force and a one-cycle walk the
    whole layout's on one mean field per geometry, within what repeating the
    whole layout moves them; rows alone between steps."""

    def rank(comm):
        out = {}
        scf_factory = one_mean_field_per_geometry()
        for tag, sliced in (('whole', None), ('sliced', True),
                            ('repeat', None)):
            surface = state_pair(sliced, scf_factory)
            # at the reference first: its factors live as long as mf0 does,
            # so every scan below has the reference's to read
            spectrum = spectrum_and_vectors(surface)
            grad, e, diags = evaluate(surface, own_water(H2O_DISPLACED))
            ground = surface.ground.total_gradient(own_water(H2O_DISPLACED))
            log = []
            watch_between_steps(surface, log)
            mol_opt, info = optimize(surface, max_cycle=1, verbose=False)
            log.append(whole_factor_arrays(
                surface, surface.ground.M,
                {surface.ground.mf0.mo_coeff.shape[1],
                 surface.ground.mol0.nao, surface.ground.naux}))
            rows = reachable_rows(surface)
            out[tag] = dict(
                force=(grad, e, diags['omega'], diags['root']),
                ground=(ground[0], ground[1]),
                spectrum=spectrum,
                walk=(mol_opt.atom_coords(), info['energy'],
                      np.array([h['e'] for h in info['history']]),
                      np.array([h['grad_max'] for h in info['history']])),
                scans=log,
                held=[rows_held(r, comm.Get_rank(), size) for r in rows],
                factorization_rows=[type(r).__name__ for r in rows])
        return out

    res = run_simulated(rank, size)
    for r, out in enumerate(res):
        for key in ('force', 'ground', 'spectrum', 'walk'):
            assert within_repeat(out['sliced'][key], out['whole'][key],
                                 out['repeat'][key]), (
                f'rank {r} of {size}: {key} sliced != whole')
            assert bitwise(out['sliced'][key], res[0]['sliced'][key]), (
                f'rank {r} of {size}: {key} != rank 0')
        # Between steps: no whole factor anywhere on the sliced surface.
        scans = out['sliced']['scans']
        assert len(scans) >= 3 and all(s == [] for s in scans), (
            f'rank {r} of {size} holds whole factors between steps: {scans}')
        # ... which the scan does see on the whole layout, whose fit it caches.
        assert all(s for s in out['whole']['scans']), out['whole']['scans']
        held = out['sliced']['held']
        assert held, f'rank {r}: no rows reachable from the sliced surface'
        for got, block in held:
            assert got == block, (
                f'rank {r} of {size} holds {got}, its block {block}')
        assert out['whole']['held'] == []


@pytest.mark.parametrize('size', [2, 3])
def test_gathers_are_one_per_sweep(size):
    """One gather per sweep or solve: the same count whatever the tau grids
    or the Davidson's iterations, and the count the stages add up to."""

    def gathers(surface):
        """The gathers of the one rows object both halves read at a fresh
        geometry, as the excited half's assembly, the last, reports them."""
        return surface.total_gradient(own_water(H2O_DISPLACED))[2][
            'factor_gathers']

    def with_ground_tau(ntau):
        mol = own_water()
        mf = chain_scf(mol)
        ground = RPAGroundStateChain(mol, chain_scf, mf=mf, ntau=ntau,
                                     sliced=True)
        return RPABSESurface(mol, chain_scf, spin='singlet', mf=mf,
                             solver='davidson', ground=ground)

    def rank(comm):
        return (gathers(state_pair(True)),
                gathers(state_pair(True, ntau_gw=28, ntau_w=12)),
                gathers(state_pair(True, bse_conv_tol=1e-8)),
                gathers(with_ground_tau(16)))

    for r, (base, more_tau, tight, ground_tau) in enumerate(
            run_simulated(rank, size)):
        assert base == COMPOSED_GATHERS, f'rank {r}: {base}'
        assert more_tau == base and tight == base and ground_tau == base, (
            f'rank {r}: {base} / {more_tau} / {tight} / {ground_tau}')


@pytest.mark.parametrize('size', [2, 3])
@pytest.mark.parametrize('solver', ['dense', 'davidson'])
def test_excited_chain_on_slices(size, solver):
    """The excited chain alone: its force, quasiparticle force and
    interstate element the whole layout's on slices, every rank rank 0's."""

    def rank(comm):
        out = {}
        scf_factory = one_mean_field_per_geometry()
        for tag, sliced in (('whole', None), ('sliced', True),
                            ('repeat', None)):
            mol = own_water()
            chain = ExcitedStateChain(mol, scf_factory, mf=scf_factory(mol),
                                      solver=solver, sliced=sliced)
            here = own_water(H2O_DISPLACED)
            g_ex, d_ex = chain.excitation_gradient(here)
            g_qp, d_qp = chain.quasiparticle_gradient(0, here)
            g_st, _ = chain.interstate_gradient(0, 1)
            out[tag] = (g_ex, d_ex['omega'], g_qp, d_qp['qp_energy'],
                        d_qp['qp_z'], g_st, chain.quasiparticle(1))
        return out

    res = run_simulated(rank, size)
    for r, out in enumerate(res):
        assert within_repeat(out['sliced'], out['whole'], out['repeat']), (
            f'rank {r}: sliced != whole')
        assert bitwise(out['sliced'], res[0]['sliced']), f'rank {r} != rank 0'


def test_charged_surface_on_slices():
    """E^(N-1) = E_0^dRPA - eps^QP_HOMO: the quasiparticle force on slices."""

    def rank(comm):
        out = []
        scf_factory = one_mean_field_per_geometry()
        for sliced in (None, True, None):
            mol = own_water()
            surface = RPAQPSurface(mol, scf_factory, state=0,
                                   mf=scf_factory(mol), sliced=sliced)
            g, e, _ = surface.total_gradient(own_water(H2O_DISPLACED))
            out.append((g, e))
        return out

    res = run_simulated(rank, 2)
    for r, (whole, sliced, repeat) in enumerate(res):
        assert within_repeat(sliced, whole, repeat), f'rank {r}'
        assert bitwise(sliced, res[0][1]), f'rank {r} != rank 0'


def test_the_dispatcher_carries_the_layout():
    """`potential_energy_surface` reaches the factorization with `sliced`; a
    dense row, which has no factors to lay out, refuses the keyword."""
    ground = GroundState('rpa', 'hf')

    def rank(comm):
        mol = own_water()
        surface = potential_energy_surface(mol, chain_scf, ground_state=ground,
                                           excitation=Excitation('singlet'),
                                           sliced=True)
        assert surface.ground.factorization is surface.excited.factorization
        assert surface.ground.factorization.sliced
        g, e, diags = evaluate(surface, own_water(H2O_DISPLACED))
        return diags['factor_gathers']

    for gathers in run_simulated(rank, 2):
        assert set(gathers) == {'X_mo', 'D', 'X_o', 'X_v'}
    with pytest.raises(TypeError, match='sliced'):
        potential_energy_surface(own_water(), chain_scf, ground_state=ground,
                                 excitation=Excitation('singlet'),
                                 chi0='dense-qb', factorization='four-index',
                                 sliced=True)


def test_what_slices_cannot_serve_is_refused():
    mol = own_water()
    solvated = SolventScreening(mol, eps=1.78)
    built = []

    def counted(m):
        built.append(m)
        return chain_scf(m)

    with pytest.raises(ValueError, match='reaction field'):
        ExcitedStateChain(mol, counted, environment=solvated, sliced=True)
    assert built == [], 'the reaction field was refused only after an SCF'
    with pytest.raises(ValueError, match='sliced=False'):
        FactorChain(mol, counted, sliced=True)
    with pytest.raises(ValueError, match='layout|sliced'):
        ExcitedStateChain(mol, counted, sliced=False,
                          factorization=FrozenFactorization(mol, sliced=True))
    assert built == []
    # the dRPA chain has no bare gauge to need, and takes a continuum on slices
    RPAGroundStateChain(mol, chain_scf, environment=solvated, sliced=True)

    def rank(comm):
        m = own_water()
        chain = ExcitedStateChain(m, chain_scf, mf=chain_scf(m), sliced=True)
        rows = chain.factors_at(m, chain.mf0)[0]
        with pytest.raises(ValueError, match='densities'):
            static_screening(rows, rows, chain.mf0.mo_energy, chain.nocc,
                             chain.w_grid)
        with pytest.raises(TypeError):
            tuple(rows)
        return True

    assert all(run_simulated(rank, 2))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
