"""Every way the cubic chain assembles an orbital's quasiparticle energy.

An explicitly solved orbital q carries its root of

    w_q = eps_q + xc_q + Sigma_c,qq(w_q),   xc_q = <q|Sigma_x - v_xc|q> + Sigma^env_qq,

Sigma^env the static reaction field of Duchemin, Jacquemin and Blase, J. Chem.
Phys. 144, 164106 (2016), Eq. (18), zero in the gas phase. Every other way an
orbital gets an energy freezes a piece of such a root at the reference
geometry R0 and spends it at every geometry R:

    inside scissor   eps_p(R) + xc_p(R) + [w_p - eps_p - xc_p](R0)
    outside scissor  eps_p(R) + [w_q - eps_q - Sigma^env_qq](R0) + Sigma^env_pp(R),
    and demoted      q the explicit orbital nearest p in eps at R0

so a frozen shift carries the GW correction alone: Sigma_x - v_xc once (the
orbital's own inside the set, the probe's outside it), Sigma^env once (always
the orbital's own), Sigma_c once (frozen). The matrix covers formaldehyde and
water / cc-pVDZ, Hartree-Fock, PBE0 and LRC-wPBEh references, the gas phase
and PCM(toluene), at R0 and at a displaced geometry, and checks:

- (a) at R0 an inside-scissor orbital sits on the root its calibration
  solved, to 1e-12 Ha, and on the root the same set solves with no tier
  (`scissor=None`), to the Newton tolerance;
- (b) the static term each explicit orbital is solved with is the
  independently built <p|Sigma_x - v_xc|p> plus its own Sigma^env_pp, and
  every frozen shift reproduces its formula above to 1e-12 Ha;
  explicit orbitals keep their R0 route at R; one quasiparticle solved alone
  (`quasiparticle`) is the set's root; and in the gas phase the explicit
  Kohn-Sham roots agree with the dense quasi-boson route's, which assembles
  its static shift on its own, to within the two realizations' difference;
- (c) on Hartree-Fock in the gas phase the inside scissor is w_p - eps_p
  up to round-off;
- (d) on a Kohn-Sham reference, in the gas phase and in toluene, with inside-
  and outside-scissor orbitals present, the S1 force is a central difference
  of its own energy to 1e-6 Ha/Bohr.

The mechanisms: the pole model ('sop'), the contour deformation with
Laplace or explicit residues, the inside scissor (scissor='calibrate'), the
outside scissor of a window, a root rejected at R0 (demoted to the outside
scissor; forced here by handing the settle step a pole strength outside
(0, 1]), and a state whose pole-model verdict was frozen at R0 although its
own root lies past the Eq. (27) limit.

Slow (every column builds five chains; the force cells take 16 energies
each). Run: python tests/test_qp_energy_assembly.py   (or pytest)
"""
import os
import sys
import warnings

import numpy as np
import pytest
from pyscf import dft, gto, scf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))

from src.Base.constants import (HARTREE_TO_EV, QP_CD_NEWTON_TOL,  # noqa: E402
                                SCF_DIFFERENTIABLE_CONV_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.solvent_screening import SolventScreening  # noqa: E402
from src.SingleReference.GW.sum_over_poles import compressible  # noqa: E402
from src.gradients import excited_state  # noqa: E402
from src.gradients.dense_surfaces import QuasiparticleSurface  # noqa: E402
from src.gradients.excited_state import ExcitedStateChain  # noqa: E402
from src.properties.surfaces import reference_mean_field  # noqa: E402

BASIS = 'cc-pvdz'
MOLECULES = {
    'formaldehyde': 'C 0 0 -0.5296; O 0 0 0.6763; H 0 0.9339 -1.1088; '
                    'H 0 -0.9339 -1.1088',
    'water': 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'}
#: The window set per molecule: the core and the highest virtuals in it sit
#: outside the pole model's reach, so it holds inside-scissor states, and
#: the rest of the spectrum is outside it.
WINDOWS = {'formaldehyde': tuple(range(12)), 'water': tuple(range(9))}
REFERENCES = ('hf', 'pbe0', 'lrc-wpbeh')
ENVIRONMENTS = ('gas', 'toluene')
COLUMNS = [(m, r, e) for m in MOLECULES for r in REFERENCES
           for e in ENVIRONMENTS]
#: The R0 -> R displacement, Bohr, per atom (no symmetry kept).
DISPLACEMENT = 0.01 * np.array([[0.3, -0.2, 0.1], [0.0, 0.4, -0.3],
                                [0.2, 0.1, 0.0], [-0.1, 0.0, 0.2]])
#: A frozen shift is a sum of the same floats: exact up to rounding.
EXACT_HA = 1e-12
#: Sigma_x - v_xc on Hartree-Fock: -K/2 - (J - K/2 - J), round-off.
ROUND_OFF_HA = 1e-13
#: The static term against its independent build: two contractions of the
#: same J, K and v_xc.
STATIC_HA = 1e-9
#: Cubic (explicit residues) against dense (exact ERIs) frontier roots: the
#: realizations differ by about a meV on formaldehyde PBE0, a static term
#: counted twice or not at all by eV.
DENSE_EV = 0.05
#: The contour-deformation chain's set, offsets from the LUMO: the states one
#: in from the frontier, whose residues at the start lie above the gap where
#: the Laplace backend carries them.
CD_STATES = (-3, -2, 1, 2)
#: The orbital whose R0 root is rejected in the demotion column: LUMO+2.
DEMOTE_ABOVE_HOMO = 3
#: The finite-difference gate: 4-point stencil, absolute.
FD_STEP = 4e-4
FD_TOL = 1e-6
FD_COMPONENTS = ((0, 2), (1, 2), (2, 1), (2, 2))


def factory_for(reference):
    def factory(mol):
        mf = scf.RHF(mol) if reference == 'hf' else dft.RKS(mol, xc=reference)
        mf = mf.density_fit(auxbasis=BASIS + '-jkfit')
        mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
        mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
        mf.max_cycle = 200
        mf.kernel()
        assert mf.converged
        return mf
    return factory


def molecule(name):
    return gto.M(atom=MOLECULES[name], basis=BASIS, verbose=0)


def displaced(mol):
    out = mol.copy()
    out.set_geom_(mol.atom_coords() + DISPLACEMENT[:mol.natm], unit='Bohr')
    out.build(False, False)
    return out


def environment(mol, name):
    return None if name == 'gas' else SolventScreening(mol, solvent=name)


def sigma_x_minus_vxc(mf):
    """<p|Sigma_x - v_xc|p> over every orbital: -K/2 - (v_eff - J), built
    from the mean field's own J, K and v_eff."""
    mol, dm = mf.mol, mf.make_rdm1()
    o = -0.5 * mf.get_k(mol, dm) - (mf.get_veff(mol, dm) - mf.get_j(mol, dm))
    c = np.asarray(mf.mo_coeff)
    return np.einsum('mp,mn,np->p', c, o, c)


class RecordRoutes:
    """`qp_set_gradient` that keeps the routes of its last call: the chain
    reports them in `route_out` and keeps none of its own."""

    def __init__(self):
        self.real, self.routes = excited_state.qp_set_gradient, {}

    def __call__(self, *args, **kw):
        out = self.real(*args, **kw)
        route_out = kw.get('route_out')
        if route_out and 'routes' in route_out:
            self.routes = {int(p): r for p, r in route_out['routes'].items()}
        return out


class DemoteOne:
    """`qp_set_gradient` whose settle step reads a pole strength outside
    (0, 1] for one orbital, as a root on a negative-weight branch has: the
    orbital is demoted at R0 exactly as such a root would be."""

    def __init__(self, orbital, inner):
        self.orbital, self.inner = int(orbital), inner

    def __call__(self, *args, **kw):
        out = self.inner(*args, **kw)
        states = [int(p) for p in np.atleast_1d(args[7])]
        route_out = kw.get('route_out')
        if self.orbital in states and route_out and 'z' in route_out:
            route_out['z'] = np.array(route_out['z'], float)
            route_out['z'][states.index(self.orbital)] = -0.01
        return out


def snapshot(chain, mol, mf, recorder):
    """What one forward at `mol` assembled, orbital by orbital."""
    pieces = chain.kernel_pieces(mol, mf)
    eps, eps_qp, shift = pieces[6], pieces[7], pieces[14]
    states = [int(p) for p in chain.qp_set]
    norb = len(eps)
    env = np.zeros(norb) if shift is None else np.asarray(shift, float)
    xc = chain._xc_correction(mf, chain.qp_set, shift)
    return {'eps': np.asarray(eps, float),
            'eps_qp': np.asarray(eps_qp, float),
            'xc': dict(zip(states, np.atleast_1d(xc))),
            'env': env, 'sxv': sigma_x_minus_vxc(mf),
            'routes': dict(recorder.routes)}


def build(kind, mol, factory, mf, env, window):
    """One chain of the matrix, by the mechanism it exercises."""
    common = dict(mf=mf, environment=env, solver='dense')
    if kind in ('window', 'plain', 'demoted'):
        return ExcitedStateChain(
            mol, factory, qp_window=list(window), residue_route='sop',
            scissor=None if kind == 'plain' else 'calibrate',
            outside='scissor', **common)
    if kind == 'cd':
        # 'auto': the Laplace backend where the tau grid carries a state's
        # residues, the explicit one where not, none where it has none
        nocc = mol.nelectron // 2
        window = [nocc + k for k in CD_STATES]
        return ExcitedStateChain(mol, factory, qp_window=window,
                                 residue_route='auto', outside='scissor',
                                 **common)
    return ExcitedStateChain(mol, factory, qp_window=2,
                             residue_route='explicit', outside='scissor',
                             **common)


CHAINS = ('window', 'plain', 'demoted', 'cd', 'explicit')


def build_column(name, reference, env_name):
    """{chain kind: (chain, snapshot at R0, snapshot at R)} of one column."""
    warnings.simplefilter('ignore')
    factory = factory_for(reference)
    mol = molecule(name)
    mol1 = displaced(mol)
    out = {'name': name, 'reference': reference, 'env': env_name}
    env = environment(mol, env_name)
    mf = reference_mean_field(mol, factory, env)
    for kind in CHAINS:
        recorder = RecordRoutes()
        with pytest.MonkeyPatch.context() as mp:
            solve = recorder
            if kind == 'demoted':
                solve = DemoteOne(mol.nelectron // 2 - 1 + DEMOTE_ABOVE_HOMO,
                                  recorder)
            mp.setattr(excited_state, 'qp_set_gradient', solve)
            chain = build(kind, mol, factory, mf, env, WINDOWS[name])
            chain.excitation()
            at_r0 = snapshot(chain, chain.mol0, chain.mf0, recorder)
            at_r = snapshot(chain, mol1, chain.mean_field(mol1)[1], recorder)
        out[kind] = (chain, at_r0, at_r)
    return out


@pytest.fixture(scope='module', params=COLUMNS,
                ids=['-'.join(c) for c in COLUMNS])
def column(request):
    return build_column(*request.param)


def mechanism(chain, s0, p):
    """How orbital p's energy is assembled on `chain`."""
    if p in set(int(q) for q in chain.qp_set):
        route = s0['routes'][p]
        if route == 'scissor':
            return 'inside'
        if route == 'sop':
            past = not compressible(chain.qp_seeds[p], s0['eps'],
                                    chain.nocc)[0]
            return 'route-frozen' if past else 'sop'
        return 'explicit' if route == 'explicit' else 'cd'
    if p in chain.qp_demoted:
        return 'demoted'
    return 'outside'


def expected(chain, s0, s, p, kind):
    """eps^QP_p at the geometry of `s` from its mechanism's formula; an
    inside-scissor state's root at R0 is the one its calibration solved."""
    if kind == 'inside':
        return (s['eps'][p] + s['xc'][p] - s0['eps'][p] - s0['xc'][p]
                + chain.qp_seeds[p])
    explicit = sorted(int(q) for q in chain.qp_set)
    q = min(explicit, key=lambda r: abs(s0['eps'][r] - s0['eps'][p]))
    lent = s0['eps_qp'][q] - s0['eps'][q] - s0['env'][q]
    return s['eps'][p] + lent + s['env'][p]


def assembly_errors(col):
    """{(chain, mechanism): [worst |error| in Ha, orbitals checked]} of the
    frozen-shift formulas, at R0 and at R, and of the static term each
    explicit orbital is solved with."""
    out = {}
    for kind in CHAINS:
        chain, s0, s1 = col[kind]
        for p in range(len(s0['eps'])):
            mech = mechanism(chain, s0, p)
            worst = 0.0
            if p in s0['xc']:
                for s in (s0, s1):
                    worst = max(worst, abs(s['xc'][p] - s['sxv'][p]
                                           - s['env'][p]))
            if mech in ('inside', 'outside', 'demoted'):
                for s in (s0, s1):
                    worst = max(worst, abs(s['eps_qp'][p]
                                           - expected(chain, s0, s, p, mech)))
            cell = out.setdefault((kind, mech), [0.0, 0])
            cell[0] = max(cell[0], worst)
            cell[1] += 1
    return out


def test_every_mechanism_assembles_its_formula(column):
    """(a) and (b): every orbital of every chain, at R0 and at R."""
    errors = assembly_errors(column)
    seen = {mech for _, mech in errors}
    assert {'inside', 'outside', 'demoted', 'sop', 'explicit'} <= seen, seen
    bad = {}
    for (kind, mech), (worst, n) in errors.items():
        tol = (EXACT_HA if mech in ('inside', 'outside', 'demoted')
               else STATIC_HA)
        if worst > tol:
            bad[(kind, mech)] = (worst * HARTREE_TO_EV, n)
    assert not bad, f'eV off the formula, orbitals checked: {bad}'


def test_an_inside_scissor_state_sits_on_the_untiered_root(column):
    """(a): the window's inside-scissor states at R0 against the same set
    solved with no tier, where they stay on the quadrature."""
    (chain, s0, _), plain = column['window'], column['plain'][1]
    inside = [p for p, r in s0['routes'].items() if r == 'scissor']
    assert inside
    off = max(abs(s0['eps_qp'][p] - plain['eps_qp'][p]) for p in inside)
    assert off < QP_CD_NEWTON_TOL, off * HARTREE_TO_EV


def test_explicit_states_keep_their_reference_route(column):
    for kind in CHAINS:
        _, s0, s1 = column[kind]
        assert s1['routes'] == s0['routes'], kind


def test_one_quasiparticle_alone_is_the_set_root(column):
    chain, s0, _ = column['window']
    for offset in (0, 1):
        p = chain.nocc - 1 + offset
        assert s0['routes'][p] in ('sop', 'scissor')
        off = chain.quasiparticle(offset) - s0['eps_qp'][p]
        assert abs(off) < QP_CD_NEWTON_TOL, (offset, off * HARTREE_TO_EV)


def test_hartree_fock_gas_stores_what_it_always_stored(column):
    """(c): on Hartree-Fock in the gas phase the static term is round-off,
    so the inside shift is w_p - eps_p to the last bit or two."""
    if column['reference'] != 'hf' or column['env'] != 'gas':
        pytest.skip('Hartree-Fock in the gas phase only')
    chain, s0, _ = column['window']
    assert chain.scissor_map
    for p, shift in chain.scissor_map.items():
        before = chain.qp_seeds[p] - s0['eps'][p]
        assert abs(s0['xc'][p]) < ROUND_OFF_HA, (p, s0['xc'][p])
        # the round-off of eps_p + xc_p and of the two differences
        ulps = 2 * np.spacing(abs(s0['eps'][p])) + np.spacing(abs(before))
        assert abs(shift - before) <= abs(s0['xc'][p]) + ulps, p


def test_kohn_sham_roots_agree_with_the_dense_route(column):
    """The dense quasi-boson route assembles its static shift on its own; a
    term counted twice or not at all on either side is eV, not meV."""
    if column['reference'] == 'hf' or column['env'] != 'gas':
        pytest.skip('a Kohn-Sham reference in the gas phase only')
    chain, s0, _ = column['explicit']
    mol, factory = chain.mol0, factory_for(column['reference'])
    # charge_change -1 removes the HOMO electron, +1 adds one to the LUMO
    for change, p in ((-1, chain.nocc - 1), (+1, chain.nocc)):
        surface = QuasiparticleSurface(mol, factory, charge_change=change)
        info = surface.total_gradient(mol, surface.scf_factory(mol))[2]
        off = info['qp_energy_eV'] - s0['eps_qp'][p] * HARTREE_TO_EV
        assert abs(off) < DENSE_EV, (p, off)


@pytest.mark.parametrize('reference,env_name', [
    ('pbe0', 'gas'), ('pbe0', 'toluene'),
    ('lrc-wpbeh', 'gas'), ('lrc-wpbeh', 'toluene')])
def test_the_force_is_the_derivative_of_its_energy(reference, env_name):
    """(d): the window chain, with inside- and outside-scissor orbitals."""
    warnings.simplefilter('ignore')
    factory = factory_for(reference)
    mol = molecule('formaldehyde')
    env = environment(mol, env_name)
    chain = build('window', mol, factory, reference_mean_field(
        mol, factory, env), env, WINDOWS['formaldehyde'])
    analytic, _ = chain.excitation_gradient()
    assert chain.scissor_map
    assert chain.outside_shift
    for ia, x in FD_COMPONENTS:
        e = []
        for k in (-2, -1, 1, 2):
            m = mol.copy()
            d = np.zeros((mol.natm, 3))
            d[ia, x] = k * FD_STEP
            m.set_geom_(mol.atom_coords() + d, unit='Bohr')
            m.build(False, False)
            e.append(chain.excitation(m))
        fd = (e[0] - 8 * e[1] + 8 * e[2] - e[3]) / (12 * FD_STEP)
        assert abs(analytic[ia, x] - fd) < FD_TOL, (ia, x, analytic[ia, x],
                                                     fd)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
