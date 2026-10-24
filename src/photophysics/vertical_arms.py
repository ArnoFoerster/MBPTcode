"""Vertical S_n and T_n of one geometry on the production BSE@GW chain, with the
quasiparticle energies taken one of three ways (`vertical_arm`), energies only.

    from src.photophysics.vertical_arms import vertical_arm
    record = vertical_arm(declaration, 'ac-all', max_memory=4000)

The geometry is the declaration's (`declaration.system.atom`, Angstrom), and
the chain is the declaration's: `chain.chain_spec` and
`chain.mean_field_factory` build it here as they do for the stages of a run, so
the arms and the walk of the same declaration share one surface. The arms
differ from the walk in two ways that are not a different chain: the mean field
is converged for an energy (`energy_numerics`, derived from the declaration's
numerics) where a walk converges it for a force, and the arm names its own
quasiparticle declaration (`arm_declaration`).

One reference SCF, one forward pass (factors, the static W, the shift of
Duchemin, Guido, Jacquemin and Blase, Chem. Sci. 9,
4430 (2018), Eqs. (18a,b): the static COHSEX difference of their J. Chem. Phys.
144, 164106 (2016), Eq. (19), taken as a self-element), one Davidson per spin.

THE ARMS differ in the quasiparticle energies on the BSE diagonal alone:

    sop-admitted  the sum-over-poles residues on every orbital the pole model
                  admits, the frozen scissor outside: what a gradient run
                  solves.
    sop-adaptive  the same, on the explicit set the adaptive partition selects
                  out of the admitted window for the followed singlet and
                  triplet (one continuation at the geometry; the declaration's
                  own budget where it declares an adaptive set).
    ac-all        the admitted chain's forward pass is run as in sop-admitted
                  and its spectrum recorded beside; then every orbital is
                  solved by the analytic continuation of the space-time
                  self-energy (`ac_quasiparticle_shifts`) on the SAME bare
                  factors and the SAME static term, no scissor anywhere, and
                  the same Davidson is solved again on those energies with the
                  same W. The two spectra of this arm differ by the residue
                  treatment and the scissor alone.

An orbital whose continuation fails (non-finite root or pole strength) keeps
the admitted chain's value and is listed in `ac_failed`; a frontier orbital
among them fails the run.

The record carries Omega of every reported root of both spins, Delta E_ST, the
quasiparticle HOMO, LUMO and gap, the explicit and total orbital counts and the
stage walls. A factorization that misses an orbital density is the verdict:
the record then carries `refusal` and no spectrum.
"""
import time
from dataclasses import replace

import numpy as np
from pyscf import gto

from src.Base.numerics.constants import (HARTREE_TO_EV, SCF_ENERGY_CONV_TOL,
                                         SCF_ENERGY_GRAD_TOL, VERTICAL_FRONTIER_LISTED,
                                         VERTICAL_WORST_LISTED)
from src.Base.declaration import Excitation, QPStates
from src.Base.pyscf.pyscf_interface import occupied_count
from src.Base.factorization.separable_ri import UnrepresentedOrbitals
from src.gradients.gwbse.state_manifold import StateEvaluation, StateManifold
from src.photophysics import chain
from src.photophysics.records import log
from src.gradients.gwbse.diagnostics import qp_bookkeeping
from src.properties.excitations import grid_record, surface_of
from src.gradients.gwbse.surface_factory import reference_mean_field

#: The quasiparticle arms.
ARMS = ('sop-admitted', 'sop-adaptive', 'ac-all')


# ------------------------------------------------------- the arm's chain
def arm_declaration(declaration, arm):
    """`declaration` with the arm's quasiparticle declaration and nothing else
    changed: the adaptive set for the followed singlet and triplet (the
    declaration's own where it is adaptive), else the admitted window (the
    ac-all arm's chain included)."""
    if arm not in ARMS:
        raise ValueError(f'arm {arm!r} not in {ARMS}')
    kind = declaration.qp_states.kind
    if arm == 'sop-adaptive':
        if kind == 'adaptive':
            return declaration
        qp = QPStates('adaptive', tol_meV=None,
                      targets=(('singlet', declaration.singlet - 1),
                               ('triplet', declaration.triplet - 1)))
    else:
        if kind == 'admitted':
            return declaration
        qp = QPStates(kind='admitted')
    return replace(declaration, qp_states=qp)


def energy_numerics(numerics):
    """`numerics` with the mean field converged for an energy (SCF_ENERGY_*)
    where a walk converges it for a force; `numerics` itself where it already
    is."""
    if (numerics.scf.conv_tol, numerics.scf.conv_tol_grad) == (
            SCF_ENERGY_CONV_TOL, SCF_ENERGY_GRAD_TOL):
        return numerics
    return numerics.derive(
        'vertical energies only: the mean field is converged for an energy, '
        'not for the orbital response of a force',
        scf={'conv_tol': SCF_ENERGY_CONV_TOL,
             'conv_tol_grad': SCF_ENERGY_GRAD_TOL})


def arm_mean_field_factory(declaration, max_memory, histories, span):
    """`chain.mean_field_factory` of `declaration` for an energy: its
    reference on its route, in `span` (n_kept, cut; `chain.span_choice`)."""
    return chain.mean_field_factory(
        declaration.reference, energy_numerics(declaration.numerics),
        max_memory, histories, finish=declaration.policies.scf_finish,
        warm_start=None, span=span,
        surface_route=declaration.numerics.scf.route)


def reported_states(declaration):
    """The reported roots of both spins as (spin, root) keys."""
    n = declaration.n_states
    return (tuple(('singlet', i) for i in range(n))
            + tuple(('triplet', j) for j in range(n)))


# ----------------------------------------------------------- the record
def spectrum_block(omegas, n_states):
    """{'S1': eV, ...} of the reported roots, and Delta E_ST = S1 - T1."""
    block = {}
    for spin in ('singlet', 'triplet'):
        om = np.asarray(omegas[spin], float)
        for i in range(n_states):
            block[f'{spin[0].upper()}{i + 1}'] = float(om[i] * HARTREE_TO_EV)
    block['dEST'] = block['S1'] - block['T1']
    return block


def frontier_block(eps_qp, eps, nocc):
    """The quasiparticle HOMO, LUMO and gap, and the orbitals about them, eV."""
    lo = max(0, nocc - VERTICAL_FRONTIER_LISTED)
    hi = min(len(eps), nocc + VERTICAL_FRONTIER_LISTED)
    return {'homo_eV': float(eps_qp[nocc - 1] * HARTREE_TO_EV),
            'lumo_eV': float(eps_qp[nocc] * HARTREE_TO_EV),
            'gap_eV': float((eps_qp[nocc] - eps_qp[nocc - 1]) * HARTREE_TO_EV),
            'ks_gap_eV': float((eps[nocc] - eps[nocc - 1]) * HARTREE_TO_EV),
            'frontier_eV': {int(p): float(eps_qp[p] * HARTREE_TO_EV)
                            for p in range(lo, hi)}}


def bookkeeping_summary(man, mol):
    """The explicit set and its partition, without the pole positions."""
    book = qp_bookkeeping(man.chain, mol) or {}
    keep = ('n_orbitals', 'explicit', 'n_explicit', 'n_outside', 'n_scissor',
            'n_inside_scissor', 'inside_scissor_eV', 'scissor_tiers', 'route',
            'z_outside', 'adaptive', 'z', 'nfreq_cd', 'outside_scissor',
            'scissor_curve', 'blocks')
    out = {k: book[k] for k in keep if k in book}
    sop = book.get('sop') or {}
    out['sop_n_poles_asked'] = sop.get('n_poles_asked')
    return out


def qp_report(shared, nocc):
    """Log the quasiparticle spectrum the Davidson is about to read: a virtual
    below an occupied level makes (A - B) indefinite before any kernel
    enters."""
    eps = np.asarray(shared.eps, float)
    eps_qp = np.asarray(shared.eps_qp, float)
    occ, vir = eps_qp[:nocc], eps_qp[nocc:]
    shift = (eps_qp - eps) * HARTREE_TO_EV
    worst = np.argsort(-np.abs(shift))[:VERTICAL_WORST_LISTED]
    log(f'QP spectrum: max occupied {occ.max() * HARTREE_TO_EV:.4f} eV '
        f'(orbital {int(np.argmax(occ))}), min virtual '
        f'{vir.min() * HARTREE_TO_EV:.4f} eV (orbital '
        f'{nocc + int(np.argmin(vir))}), '
        f'{int(np.count_nonzero(vir < occ.max()))} virtual(s) below the '
        f'highest occupied, {int(np.count_nonzero(occ > vir.min()))} occupied '
        f'above the lowest virtual')
    log('QP largest shifts eps_qp - eps (eV): '
        + ', '.join(f'{int(p)}: {shift[p]:+.3f}' for p in worst))


def refusal_of(exc):
    """The record of a refusal (`UnrepresentedOrbitals`), which is the
    verdict: its message and the worst orbital densities."""
    worst = [int(p) for p in exc.orbitals[:VERTICAL_WORST_LISTED]]
    return dict(message=str(exc), orbitals=worst,
                residuals=[float(exc.residuals[p]) for p in worst],
                worst_residual=(float(np.max(exc.residuals))
                                if len(exc.residuals) else None),
                n_orbitals=int(len(exc.residuals)))


# ------------------------------------------------------------ the passes
def sop_forward(man, mol, mf, states):
    """(the shared forward, {spin: omega}) of the chain's own explicit set:
    `StateManifold.evaluate`'s forward and Casida steps, energies only, the
    adaptive set checked on its eigenvectors as there. The forward is returned
    unreleased for the ac-all arm to solve again on."""
    shared = man.forward(mol, mf)
    qp_report(shared, occupied_count(mf))
    shared, solved = man.spectra(mol, mf, shared, states,
                                 StateEvaluation(mol, mf))
    return shared, {spin: s.omega for spin, s in solved.items()}


def ac_all(man, shared, mol, mf, nocc):
    """(eps^QP of every orbital by the analytic continuation, {spin: omega},
    the record's diagnostics): the continuation on the forward's bare factors
    and static term, then each spin's Davidson again on the same W."""
    driven = man.driven
    eps = np.asarray(shared.eps, float)
    states = np.arange(len(eps))
    # Sigma screens BARE, on the chain's own static term
    a, z, _ = man.ac_shifts(mol, mf, shared, states)
    eps_ac = np.array(shared.eps_qp, float)
    failed = []
    for p in states:
        if np.isfinite(a[int(p)]) and np.isfinite(z[int(p)]):
            eps_ac[p] = eps[p] + a[int(p)]
        else:
            failed.append(int(p))
    near = set(range(max(0, nocc - VERTICAL_FRONTIER_LISTED),
                     min(len(eps), nocc + VERTICAL_FRONTIER_LISTED)))
    if near & set(failed):
        raise RuntimeError(f'the continuation failed on frontier orbitals '
                           f'{sorted(near & set(failed))}')
    omegas = {spin: np.sort(om)
              for spin, om in man.spectra_on(shared, eps_ac).items()}
    explicit = np.asarray(driven.resolved.qp_set, int)
    # the residue treatment alone: SOP's explicit roots against AC's
    resid = (np.asarray(shared.eps_qp)[explicit]
             - eps_ac[explicit]) * HARTREE_TO_EV
    z_all = np.array([z[int(p)] for p in states], float)
    # the scissor alone: what the SOP pass put outside the set against AC
    outside = {'occupied': [], 'virtual': []}
    for p in states:
        if int(p) not in set(explicit.tolist()) and int(p) not in failed:
            outside['occupied' if p < nocc else 'virtual'].append(int(p))
    scissor_err = {}
    for side, idx in outside.items():
        if idx:
            err = (np.asarray(shared.eps_qp)[idx] - eps_ac[idx]) * HARTREE_TO_EV
            scissor_err[side] = {'n': len(idx),
                                 'maxabs_eV': float(np.max(np.abs(err))),
                                 'meanabs_eV': float(np.mean(np.abs(err))),
                                 'mean_eV': float(np.mean(err))}
    diag = {'ac_failed': failed,
            'sop_minus_ac_outside_eV': scissor_err,
            'ac_z_min': float(np.nanmin(z_all)),
            'ac_z_max': float(np.nanmax(z_all)),
            'sop_minus_ac_explicit_maxabs_eV': float(np.max(np.abs(resid))),
            'sop_minus_ac_explicit_meanabs_eV': float(np.mean(np.abs(resid))),
            'sop_minus_ac_by_orbital_eV': {int(p): float(r) for p, r in
                                           zip(explicit, resid)}}
    return eps_ac, omegas, diag


# ------------------------------------------------------------ the entry
def vertical_arm(declaration, arm, max_memory):
    """The record of one arm ('sop-admitted', 'sop-adaptive', 'ac-all') at the
    declaration's geometry (module docstring). Under ranks every rank runs it
    in lockstep inside the caller's `distributed(comm)`; `max_memory` (MB) is
    a resource of the launch."""
    d = arm_declaration(declaration, arm)
    t_all = time.time()
    mol = gto.M(atom=d.system.atom, basis=d.system.basis,
                max_memory=int(max_memory), verbose=0)
    env = chain.continuum(d, mol)
    spec = chain.chain_spec(d, env)
    states = reported_states(d)
    log(f'{d.system.name}: {mol.natm} atoms, nao {mol.nao_nr()}, '
        f'{d.reference}/{mol.basis}, {d.mean_field_name} mean field, factors '
        f'{d.numerics.isdf.level}, environment {d.environment}, arm {arm}')
    rec = dict(kind='vertical_arm', system=d.system.name, natm=mol.natm,
               nao=mol.nao_nr(), basis=str(mol.basis), reference=d.reference,
               environment=d.environment, arm=arm,
               scf=d.numerics.scf.route, scf_grid=d.numerics.scf.grid_level,
               kernel=d.kernel, nroots=d.numerics.bse.nroots,
               n_states=d.n_states, qp_states=d.qp_states.kind,
               outside_scissor=d.numerics.qp.outside_scissor,
               qp_hamiltonian=d.numerics.qp.hamiltonian,
               bse_conv_tol=d.numerics.bse.conv_tol,
               auxbasis=chain.factor_auxbasis(mol),
               scf_auxbasis=chain.scf_auxbasis(mol, d.numerics.scf.route),
               **grid_record(spec, mol))
    walls = {}
    histories = []
    span = rec['span_choice'] = chain.span_choice(d, mol)
    t0 = time.time()
    factory = arm_mean_field_factory(d, max_memory, histories,
                                     (span['n_kept'], span['cut']))
    mf = reference_mean_field(mol, factory, env)
    walls['scf'] = time.time() - t0
    if getattr(mf, 'span', None) is not None:
        rec['span'] = mf.span
    nocc = occupied_count(mf)
    e_ks_orb = np.asarray(mf.mo_energy, float) * HARTREE_TO_EV
    rec['ks_homo_eV'], rec['ks_lumo_eV'] = (float(e_ks_orb[nocc - 1]),
                                            float(e_ks_orb[nocc]))
    log(f'reference SCF: E {mf.e_tot:.10f} Ha, nocc {nocc}, '
        f'nmo {len(mf.mo_energy)}', t0)

    t0 = time.time()
    excited = surface_of(spec, Excitation('singlet', root=1, kernel=d.kernel),
                         mol, factory, mf=mf)
    man = StateManifold(excited, states=states)
    mol, mf = man.mean_field(mol, mf)
    walls['chain'] = time.time() - t0
    t0 = time.time()
    try:
        shared, omegas = sop_forward(man, mol, mf, states)
    except UnrepresentedOrbitals as exc:
        rec['refusal'] = refusal_of(exc)
        log(f'REFUSED: {exc}', t0)
        walls['total'] = time.time() - t_all
        rec['walls_s'] = walls
        return rec
    walls['sop_forward_and_bse'] = time.time() - t0
    eps = np.asarray(shared.eps, float)
    sop = dict(spectrum_block(omegas, d.n_states),
               **frontier_block(np.asarray(shared.eps_qp, float), eps, nocc))
    log(f'SOP ({d.qp_states.kind}): S1 {sop["S1"]:.4f} T1 {sop["T1"]:.4f} '
        f'dEST {sop["dEST"]:.4f} QP gap {sop["gap_eV"]:.4f} eV', t0)
    rec['e_ks'] = float(mf.e_tot)
    rec['qp_bookkeeping'] = bookkeeping_summary(man, mol)
    rec['n_explicit'] = int(len(man.driven.resolved.qp_set))
    rec['n_orbitals'] = int(len(eps))
    if arm == 'ac-all':
        rec['sop_admitted_same_forward'] = sop
        t0 = time.time()
        eps_ac, omegas_ac, diag = ac_all(man, shared, mol, mf, nocc)
        walls['ac_all_and_bse'] = time.time() - t0
        rec['spectrum'] = dict(spectrum_block(omegas_ac, d.n_states),
                               **frontier_block(eps_ac, eps, nocc))
        rec['ac'] = diag
        rec['n_explicit'] = int(len(eps) - len(diag['ac_failed']))
        rec['ac_minus_sop_same_forward_eV'] = {
            k: rec['spectrum'][k] - sop[k]
            for k in ('S1', 'T1', 'dEST', 'gap_eV', 'homo_eV', 'lumo_eV')}
        s = rec['spectrum']
        log(f'AC-all: S1 {s["S1"]:.4f} T1 {s["T1"]:.4f} dEST {s["dEST"]:.4f} '
            f'QP gap {s["gap_eV"]:.4f} eV; SOP-AC on the explicit set max '
            f'{diag["sop_minus_ac_explicit_maxabs_eV"]:.4f} eV, '
            f'{len(diag["ac_failed"])} continuation(s) failed', t0)
    else:
        rec['spectrum'] = sop
    shared.release()
    walls['total'] = time.time() - t_all
    rec['walls_s'] = walls
    rec['scf_cycles'] = [len(h.summary()) for h in histories]
    # an SCF that stopped short is warned about by the factory and counted
    rec['scf_not_converged'] = sum(
        1 for h in histories if getattr(h, 'not_converged', False))
    return rec
