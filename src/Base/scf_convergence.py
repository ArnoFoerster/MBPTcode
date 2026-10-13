"""How a mean field was converged, cycle by cycle.

pyscf stops its loop when |dE| < conv_tol AND |g| < conv_tol_grad, with
|g| = ||2 F_vo||_F the orbital gradient. What a force needs of the mean field
is the second (the Lagrangian assumes F_vo = 0); what an SCF costs is the
number of cycles both take, and a rank count only shortens a cycle. So the
record of an SCF is its cycles: the energy change, |g|, the density change
and the seconds of each, read off pyscf's own `callback`, which sees every
cycle's locals and changes nothing it computes.

The callback runs on every rank of a distributed SCF (each runs pyscf's loop
against the reduced J/K), so every rank keeps a history; a record is rank
0's.

WARM STARTS. Along a walk over geometries (an optimization, a finite
difference) each SCF starts from pyscf's atomic guess unless told otherwise.
`warm_started` makes the guess the last converged density of the walk
(`WarmStart`) wherever that density belongs to the same atoms, basis and a
geometry within SCF_WARM_START_MAX_SHIFT -- pyscf's own scanners reuse the
density the same way. The guess is where the loop starts, not what it
converges to: the converged mean field is the same one to its tolerances, and
the cycles it saves are the measurement. Under ranks every rank takes the
guess its own copy holds; the first Fock build locksteps the density to rank
0's, as for any guess.

SECOND-ORDER FINISH. DIIS converges linearly, and the last decades of |g|
a force asks for (1e-7 to 1e-11) cost it as many cycles as the first seven.
`newton_finished` stops the DIIS loop at SCF_NEWTON_FINISH_FROM and takes
exact Newton steps on the orbital Hessian from there (`newton_finish`), each
a conjugate-gradient solve whose products are response Fock builds of the
same J/K and quadrature the loop used -- under ranks the distributed
handles, with every decision taken on lockstepped numbers. Newton converges
quadratically, so the finish lands where DIIS would have, to the tolerance.
"""
import time

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.scf import hf
from pyscf.soscf import newton_ah

from src.Base.constants import (SCF_HISTORY_GRAD_MARKS,
                                SCF_NEWTON_DIIS_CONV_TOL,
                                SCF_NEWTON_FINISH_FROM, SCF_NEWTON_MAX_HOPS,
                                SCF_NEWTON_MAX_STEPS, SCF_NEWTON_PRECOND_FLOOR,
                                SCF_NEWTON_RESIDUAL_FRACTION,
                                SCF_WARM_START_MAX_SHIFT)
from src.Base.utils.mpi_grid import lockstep


class SCFHistory:
    """Every cycle of the pyscf SCF this is the `callback` of.

    Each cycle is (cycle, E, dE, |g|, |ddm|, seconds): the energy, its change
    over the cycle, pyscf's orbital-gradient norm, the norm of the density
    change and the wall seconds since the previous cycle -- the first counts
    from the kernel's start, its guess included. A kernel run again on the
    same mean field (a second SCF, a finish) opens a new run.

    chained: a callback already on the mean field, still called after this.
    """

    def __init__(self, chained=None):
        self.chained = chained
        self.runs = []
        self._kernel_start = None

    def attach(self, mf):
        """Install on `mf` as its callback, keeping any it had; returns `mf`."""
        if mf.callback is not None and mf.callback is not self:
            self.chained = mf.callback
        mf.callback = self
        return mf

    def __call__(self, envs):
        now = time.perf_counter()
        start = envs['cput0'][1]
        if self._kernel_start != start:
            # a new kernel call: its start is pyscf's own clock at entry
            self._kernel_start = start
            self.runs.append({'phase': 'diis', 'cycles': []})
        run = self.runs[-1]
        last = run['cycles'][-1][-1] if run['cycles'] else None
        seconds = now - (start if last is None else last)
        e_tot = float(envs['e_tot'])
        run['cycles'].append([int(envs['cycle']) + 1, e_tot,
                              e_tot - float(envs['last_hf_e']),
                              float(envs['norm_gorb']),
                              float(envs['norm_ddm']), seconds, now])
        run['converged'] = bool(envs.get('scf_conv', False))
        if self.chained is not None:
            self.chained(envs)

    def add_run(self, phase, cycles, converged):
        """Append a run another driver made: `cycles` of
        (cycle, E, dE, |g|, |ddm|, seconds)."""
        self.runs.append({'phase': phase, 'converged': bool(converged),
                          'cycles': [list(c) + [None] for c in cycles]})

    def summary(self, marks=SCF_HISTORY_GRAD_MARKS):
        """The record of every run: cycles, seconds, the cycle each gradient
        mark was first passed, the last |g| and dE, and the history itself."""
        out = []
        for run in self.runs:
            cycles = run['cycles']
            seconds = [c[5] for c in cycles]
            grads = [c[3] for c in cycles]
            first = {}
            for mark in marks:
                below = [c[0] for c in cycles if c[3] < mark]
                first[f'{mark:.0e}'] = below[0] if below else None
            out.append({
                'phase': run['phase'],
                'converged': run.get('converged'),
                'cycles': len(cycles),
                'seconds': float(np.sum(seconds)) if seconds else 0.0,
                # the first cycle carries the guess; the rest are the loop's
                'seconds_per_cycle': (float(np.mean(seconds[1:]))
                                      if len(seconds) > 1 else None),
                'first_cycle_below': first,
                'final_grad_norm': grads[-1] if grads else None,
                'min_grad_norm': float(min(grads)) if grads else None,
                'final_delta_e': cycles[-1][2] if cycles else None,
                'history': [[c[0], c[2], c[3], c[4], c[5]] for c in cycles]})
        return out


class WarmStart:
    """The last converged density of a walk, and the geometry it belongs to.

    One holder is shared by every mean field of the walk: each converged SCF
    leaves its density here (`keep`) and the next one starts from it (`guess`)
    when the atoms and the basis are the same and no atom moved further than
    `max_shift` Bohr -- a rotated copy of the molecule (a symmetry-oriented
    frame), or another molecule, starts from pyscf's own guess instead.
    `used` and `declined` count the two outcomes.
    """

    def __init__(self, max_shift=SCF_WARM_START_MAX_SHIFT):
        self.max_shift = float(max_shift)
        self.dm = self.mo_coeff = self.mo_occ = None
        self.charges = self.coords = None
        self.used = self.declined = 0

    def keep(self, mf):
        """Hold `mf`'s converged density as the walk's next guess."""
        if not mf.converged:
            return
        self.mo_coeff = np.array(mf.mo_coeff)
        self.mo_occ = np.array(mf.mo_occ)
        self.dm = np.asarray(mf.make_rdm1(self.mo_coeff, self.mo_occ))
        self.charges = mf.mol.atom_charges().copy()
        self.coords = mf.mol.atom_coords().copy()

    def guess(self, mol):
        """The held density if it belongs to `mol`'s atoms and basis near its
        geometry, else None."""
        nao = mol.nao_nr()
        fits = (self.dm is not None and self.dm.shape[-2:] == (nao, nao)
                and np.array_equal(self.charges, mol.atom_charges())
                and float(np.abs(mol.atom_coords() - self.coords).max())
                <= self.max_shift)
        if not fits:
            self.declined += 1
            return None
        self.used += 1
        # tagged with the orbitals it is C n C^T of, which J/K builds read
        return lib.tag_array(self.dm.copy(), mo_coeff=self.mo_coeff,
                             mo_occ=self.mo_occ)


class WarmStarted:
    """pyscf's SCF with its initial guess taken from a `WarmStart` where the
    holder has one for this molecule, and its converged density left there."""

    def get_init_guess(self, mol=None, key='minao', **kwargs):
        dm = self._warm_start.guess(self.mol if mol is None else mol)
        if dm is None:
            return super().get_init_guess(mol, key, **kwargs)
        return dm

    def scf(self, dm0=None, **kwargs):
        super().scf(dm0, **kwargs)
        self._warm_start.keep(self)
        return self.e_tot


class NewtonFinished:
    """pyscf's SCF whose DIIS loop stops at an orbital gradient of
    `_finish_from` (with `_finish_conv_tol` for the energy) and is finished by
    `newton_finish` to the mean field's own conv_tol_grad."""

    def scf(self, dm0=None, **kwargs):
        grad_tol, conv_tol = self.conv_tol_grad, self.conv_tol
        self.conv_tol_grad = max(self._finish_from, grad_tol)
        self.conv_tol = max(self._finish_conv_tol, conv_tol)
        try:
            super().scf(dm0, **kwargs)
        finally:
            self.conv_tol_grad, self.conv_tol = grad_tol, conv_tol
        if self.converged:
            newton_finish(self, grad_tol)
        return self.e_tot


def warm_started(mf, holder):
    """`mf`, in place, starting its SCF from `holder` (a `WarmStart`) and
    leaving its converged density there."""
    mf._warm_start = holder
    return lib.set_class(mf, (WarmStarted, mf.__class__))


def newton_finished(mf, finish_from=SCF_NEWTON_FINISH_FROM,
                    diis_conv_tol=SCF_NEWTON_DIIS_CONV_TOL):
    """`mf`, in place, converged by DIIS to `finish_from` and then by exact
    Newton steps (`newton_finish`) to its own conv_tol_grad."""
    mf._finish_from, mf._finish_conv_tol = float(finish_from), float(diis_conv_tol)
    return lib.set_class(mf, (NewtonFinished, mf.__class__))


def newton_finish(mf, grad_tol, max_steps=SCF_NEWTON_MAX_STEPS,
                  max_hops=SCF_NEWTON_MAX_HOPS,
                  fraction=SCF_NEWTON_RESIDUAL_FRACTION):
    """Newton steps on the closed-shell orbital Hessian from `mf`'s orbitals
    until |g| = ||2 F_vo||_F < grad_tol; `mf` is updated in place.

    Each step solves H x = -g (`newton_ah.gen_g_hop_rhf`: the exact Hessian,
    its products one response Fock build each, the xc kernel and every
    exchange range included) by preconditioned conjugate gradients to a
    residual of `fraction` x grad_tol, and rotates C -> C exp(x - x^T). The
    final orbitals are made canonical within the occupied and the virtual
    blocks of the last Fock matrix, which leaves the density -- and so the
    energy and the gradient just measured -- unchanged.

    Every decision is taken on lockstepped numbers (the gradient norm, the
    residual norms, the step), so every rank makes the same Fock builds in
    the same order: under ranks the builds are the mean field's distributed
    handles, entered collectively. Records its steps on `mf._newton_finish`
    and, where `mf.callback` is an `SCFHistory`, as a run of phase 'newton'.
    """
    mol = mf.mol
    mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ
    if not np.all(np.isin(np.asarray(mo_occ), (0, 2))):
        raise NotImplementedError(
            'newton_finish takes a closed-shell mean field (occupations 0 '
            'and 2); converge an open-shell one with its own DIIS loop')
    h1e, s1e = mf.get_hcore(mol), mf.get_ovlp(mol)
    e_last, dm_last, hops = float(mf.e_tot), None, 0
    cycles = []
    clock = time.perf_counter()
    for step in range(max_steps + 1):
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        e_tot = float(mf.energy_tot(dm, h1e, vhf))
        g = mf.get_grad(mo_coeff, mo_occ, fock)
        gnorm = float(lockstep(float(np.linalg.norm(g))))
        ddm = 0.0 if dm_last is None else float(np.linalg.norm(dm - dm_last))
        now = time.perf_counter()
        cycles.append((step, e_tot, e_tot - e_last, gnorm, ddm, now - clock))
        clock, e_last, dm_last = now, e_tot, dm
        if gnorm < grad_tol or step == max_steps:
            break
        _, h_op, h_diag = newton_ah.gen_g_hop_rhf(mf, mo_coeff, mo_occ,
                                                  fock_ao=fock, h1e=h1e)
        x, used = conjugate_gradient(h_op, h_diag, -g, fraction * grad_tol,
                                     max_hops)
        hops += used
        rotation = scipy.linalg.expm(hf.unpack_uniq_var(x, mo_occ))
        mo_coeff = lockstep(np.dot(mo_coeff, rotation), check=True)
    mo_energy, mo_coeff = canonical_blocks(fock, mo_coeff, mo_occ)
    mf.mo_coeff, mf.mo_energy, mf.e_tot = mo_coeff, mo_energy, e_tot
    mf.converged = gnorm < grad_tol
    mf._newton_finish = {'steps': len(cycles) - 1, 'hessian_products': hops,
                         'converged': bool(mf.converged),
                         'seconds': float(sum(c[5] for c in cycles))}
    if isinstance(mf.callback, SCFHistory):
        mf.callback.add_run('newton', cycles, mf.converged)
    return mf._newton_finish


def conjugate_gradient(h_op, h_diag, rhs, target, max_hops):
    """(x, products) with |H x - rhs| < target, by conjugate gradients
    preconditioned with the diagonal `h_diag`, from x = 0; the residual norm
    each iteration tests is rank 0's, and so is the x returned."""
    precond = 1.0 / np.maximum(np.abs(h_diag), SCF_NEWTON_PRECOND_FLOOR)
    x = np.zeros_like(rhs)
    r = rhs.copy()
    z = precond * r
    p = z.copy()
    rz = float(r @ z)
    hops = 0
    while hops < max_hops:
        hp = h_op(p)
        hops += 1
        alpha = rz / float(p @ hp)
        x += alpha * p
        r -= alpha * hp
        if float(lockstep(float(np.linalg.norm(r)))) < target:
            break
        z = precond * r
        rz, rz_last = float(r @ z), rz
        p = z + (rz / rz_last) * p
    return lockstep(x, check=True), hops


def canonical_blocks(fock, mo_coeff, mo_occ):
    """(orbital energies, orbitals) diagonalizing `fock` within the occupied
    and within the virtual orbitals of `mo_coeff`, never mixing the two."""
    mo_coeff = np.array(mo_coeff)
    mo_energy = np.empty(mo_coeff.shape[1])
    occupied = np.asarray(mo_occ) > 0
    for block in (occupied, ~occupied):
        c = mo_coeff[:, block]
        e, u = scipy.linalg.eigh(c.T @ fock @ c)
        mo_coeff[:, block] = c @ u
        mo_energy[block] = e
    return lockstep((mo_energy, mo_coeff), check=True)


def scf_record(mf):
    """What one converged mean field says about its SCF: the thresholds it
    ran to, its flag, pyscf's cycle count, every run of its `SCFHistory`
    where it carries one, and the distributed SCF's stage clock where it ran
    over ranks (`mf._distributed_timings`)."""
    if mf is None:
        return None
    history = mf.callback if isinstance(mf.callback, SCFHistory) else None
    timings = getattr(mf, '_distributed_timings', None)
    return {'conv_tol': float(mf.conv_tol),
            'conv_tol_grad': (None if mf.conv_tol_grad is None
                              else float(mf.conv_tol_grad)),
            'max_cycle': int(mf.max_cycle),
            'converged': bool(mf.converged),
            'pyscf_cycles': int(getattr(mf, 'cycles', 0) or 0),
            'runs': None if history is None else history.summary(),
            'distributed_timings': None if timings is None else dict(timings)}
