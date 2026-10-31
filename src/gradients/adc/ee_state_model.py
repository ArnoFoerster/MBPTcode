"""ADC(2) excited states as state models (src.gradients.state_models):
the energy E_MP2-total + omega of one folded root in an external field and its
orbital-relaxed density, for src.environments.solvation_loop. On an MP2 model
with spin_scaling (MP2State(..., spin_scaling='scs')) the state is SCS-ADC(2)
on SCS-MP2 with the same factors: energy, density and force alike; its
triplet is the Ms = 0 component (spin-component scaling is not spin-rotation
invariant), Turbomole's spin-adapted triplet.

    state = ADC2State(MP2State(HFState(mol, make_scf), frozen, aux, numerics=numerics),
                      'singlet', root=0, nroots=3,
                      lr_environment=field.transition_environment())
    G = solvation_loop.equilibrate(state, field).energy

D_eff = D_HF + D_MP2,rel + Delta rho_rel(root) is the derivative of
E_HF + E_MP2 + omega with respect to the one-electron Hamiltonian
(ee_relaxed_density; the orbitals respond with the gas-like kernel of the
mean field in the fixed field). lr_environment: the medium's fast answer to
the transition density, read by the singlet singles block (ee_ri_fold's LR
term; None for none). follow='overlap' (default) follows the root between
calls by the overlap of its singles amplitudes with the previous call's
(y_follow, an ee_follow.FollowedRoot: AO transition amplitudes, so the pick
does not depend on the orbitals' gauge; a lost or mixed root is taken at its
best overlap, warned about and flagged, ee_follow.pick_followed): the diabatic
state, for walks along one
character. follow='index' always takes root number `root` in energy order:
the adiabatic state (S1 = the lowest singlet), a single-valued function of
the geometry. Optimizations to the adiabatic minima take 'index': with
'overlap' the followed vector is carried across rejected optimizer steps, so
after a crossing the same geometry can return a different root. A solve with
an unconverged root, in the call or in the force, is warned about and flagged
(the policy: refuse inputs, warn numerics): the call's
info['flags'] holds the fold's flags (ee_fold.FoldResult.flags), the pick's
and the reference's ({'flag': name, 'message': text} each), and the force's
info['flags'] the same for its own solve; a root index the solve did not
return has no value: the call returns NaN energy and density, the force a NaN
force, flagged 'adc_root_missing' (ee_follow.root_missing; the loop stops on
it and a walk rejects the step). A force at the last call's field, on the
orbitals that call converged, polishes the call's root on the call's operator
to the force's tolerance (ee_gradient.polish_root: no SCF, no factors, no
other root); anywhere else, or when the polish does not hold, it converges
its SCF, re-solves the roots and picks as the call does; a root that is not
the last call's is warned about and flagged
(ee_follow.FollowingWarning): under 'index'
'force_root_order_differs', one whose AO transition amplitude overlaps the
call's root by less than ROOT_FOLLOW_WEIGHT_MIN (the two solves ordered the
roots differently), and under either `follow`, at the call's own field,
'force_omega_mismatch', one whose omega differs from the call's by more than
EE_FORCE_OMEGA_MATCH times conv_tol (another state's force: a force built on
it says so). The force takes the dense reference or
the tiled route as the MP2 model's force_route decides (state_models
.tiled_force_route). Over MPI ranks the fold and the relaxed densities are
shared out (solve_ee_adc under the current communicator) and every call and
force returns rank 0's bits on every rank (state_models docstring).
"""
import numpy as np
from pyscf import gto

from src.Base.numerics.constants import EE_FORCE_OMEGA_MATCH, ROOT_FOLLOW_WEIGHT_MIN
from src.Base.numerical_flags import collecting as cphf_flags_into
from src.Base.utils.parallel.mpi_grid import broadcast, current_comm, distributed, lockstep
from src.SingleReference.ADC.eeADC.ee_driver import relaxed_densities
from src.SingleReference.ADC.eeADC.solve import solve_ee_adc
from src.SingleReference.ADC.eeADC.ee_follow import (FollowedRoot, flag,
                                                     flag_unconverged, pick_followed,
                                                     root_missing,
                                                     transition_amplitudes)
from src.gradients.adc.ee_gradient import adc2_force, adc2_force_tiled
from src.gradients.state_models import StateResult, tiled_force_route


class ADC2State:
    """One folded ADC(2) root of one spin channel (module docstring)."""

    def __init__(self, mp2, spin, root=0, nroots=3, lr_environment=None,
                 level='adc2', follow='overlap'):
        if follow not in ('overlap', 'index'):
            raise ValueError(f"follow must be 'overlap' or 'index', not {follow!r}")
        if not 0 <= int(root) < int(nroots):
            raise ValueError(f'root {root} of nroots={nroots}: the solve never '
                             'returns it')
        self.mp2, self.mol = mp2, mp2.mol
        self.follow = follow
        self.spin, self.root, self.nroots = spin, root, nroots
        self.lr_environment = lr_environment
        # the solve's tolerance and tiles: the model's one Numerics
        self.conv_tol, self.tile_gb = mp2.numerics.adc.conv_tol, mp2.tile_gb
        self.level = level
        self.y_follow = None
        # (h, e, omega, FollowedRoot) of the last call's root: the force's root
        # is checked against it
        self.called = None
        # (h, e, orbitals, op, k, y, omega) of the last call's root, which a
        # force at the same field and orbitals polishes on op instead of
        # solving every root again; dropped at the next call
        self.solved = None
        # (mol, T (nroots, nao, nao)): the last call's roots as AO transition
        # amplitudes, the next call's block start (carried between geometries
        # by a walk, as y_follow is)
        self.fold_start = None

    @property
    def numerics(self):
        """The model's one Numerics, its MP2 model's (a loop reads it)."""
        return self.mp2.numerics

    def _start(self, mf):
        """The last call's roots as singles vectors (no * nv, nroots) of mf's
        orbitals at this geometry, C_o^T S12^T T S12 C_v, or None."""
        if self.fold_start is None:
            return None
        mol0, T = self.fold_start
        S12 = gto.intor_cross('int1e_ovlp', mol0, self.mol)
        C = np.asarray(mf.mo_coeff, float)
        nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))
        Co, Cv = S12 @ C[:, self.mp2.frozen or 0:nocc], S12 @ C[:, nocc:]
        return np.einsum('mi,kmn,na->iak', Co, T, Cv,
                         optimize=True).reshape(-1, len(T))

    def __call__(self, h, e):
        self.solved = None
        r = self.mp2(h, e)
        mf = self.mp2.mf
        flags = list(r.info.get('flags', []))
        with cphf_flags_into(flags):
            _, _, res = solve_ee_adc(mf, self.mol, level=self.level,
                                     spin=self.spin, nroots=self.nroots, fold=True,
                                     frozen=self.mp2.frozen,
                                     auxbasis=self.mp2.auxbasis,
                                     environment=self.lr_environment,
                                     conv_tol=self.conv_tol, return_fold=True,
                                     residual_tol=self.mp2.numerics.adc.residual_tol,
                                     dense_limit=self.mp2.numerics.adc.dense_limit,
                                     tile_gb=self.tile_gb,
                                     numerics=self.mp2.numerics,
                                     spin_scaling=self.mp2.scaling,
                                     start=self._start(mf))
        omega_all, y, t1_all, conv = lockstep((
            np.array(res.omega, float), np.array(res.y), np.array(res.t1, float),
            np.array(np.atleast_1d(res.converged), bool)))
        # the reference's, the relaxed densities' CPHF and the fold's, rank 0's
        flags = broadcast(flags + list(res.flags), current_comm())
        flag_unconverged(conv, f'ADC(2) {self.spin}', flags)
        T = transition_amplitudes(mf, self.mp2.frozen or 0, y)
        self.fold_start = (self.mol, T)
        if self.y_follow is None or self.follow == 'index':
            k = self.root
            if root_missing(k, omega_all.size, f'ADC(2) {self.spin}', flags):
                # no number for the root: NaN, flagged; the followed root and
                # the force's check stay as they were
                self.called = None
                nan = np.full_like(np.asarray(r.dm_eff, float), np.nan)
                return StateResult(np.nan, nan, dict(
                    r.info, omega=np.nan, index=None, omega_all=omega_all,
                    t1=np.nan, converged=False, flags=flags,
                    delta_dipole=np.full(3, np.nan)))
        else:
            k = pick_followed(self.y_follow.weights(self.mol, T),
                              f'ADC(2) {self.spin} (overlap following)', flags)
        self.y_follow = FollowedRoot(self.mol, T[k])
        conv = bool(conv.all())
        # the relaxed density of the root kept, and of no other
        dflags = []
        with cphf_flags_into(dflags):
            d_k, = relaxed_densities(res, self.mol, mf, self.mp2.frozen,
                                     self.tile_gb, self.mp2.auxbasis, roots=[k],
                                     numerics=self.mp2.numerics)
        d_k = lockstep(np.asarray(d_k))
        flags += broadcast(dflags, current_comm())
        C = np.asarray(mf.mo_coeff)
        d_rel = C @ d_k @ C.T
        omega = float(omega_all[k])
        self.called = (np.array(h, float), float(e), omega, self.y_follow.copy())
        self.solved = (self.called[0], float(e), np.array(mf.mo_coeff, float),
                       res.operator, k, np.array(y[:, k]), omega)
        r_dip = self.mol.intor('int1e_r')
        return StateResult(r.energy + omega, r.dm_eff + d_rel, dict(
            r.info, omega=omega, index=k, omega_all=omega_all,
            t1=float(t1_all[k]), converged=conv, flags=flags,
            delta_dipole=-np.einsum('xmn,mn->x', r_dip, d_rel)))

    def checkpoint(self):
        """The reference's warm start, the followed root's amplitude and the
        roots' fold start (state_models docstring)."""
        out = self.mp2.checkpoint()
        if self.y_follow is not None:
            out['y_follow'] = self.y_follow.T
        if self.fold_start is not None:
            out['fold_start'] = self.fold_start[1]
        return out

    def identity(self):
        """What this model computes (state_models docstring)."""
        return dict(self.mp2.identity(), model='ADC(2)', level=self.level,
                    spin=self.spin, root=self.root, nroots=self.nroots,
                    follow=self.follow, lr=self.lr_environment is not None)

    def restore(self, saved):
        self.mp2.restore(saved)
        if saved.get('y_follow') is not None:
            # already normalized: FollowedRoot() would normalize it again
            y = FollowedRoot.__new__(FollowedRoot)
            y.mol, y.T = self.mol, np.array(saved['y_follow'], float)
            self.y_follow = y
        if saved.get('fold_start') is not None:
            self.fold_start = (self.mol, np.array(saved['fold_start'], float))

    def force(self, h, e):
        """(F, info) of the root in the fixed field (state_models docstring),
        the dense reference (ee_gradient.adc2_force) or the tiled route
        (adc2_force_tiled), the root picked as in __call__ (`follow`), and
        refused when it is not the last call's root (module docstring), and
        refused for a level other than ADC(2): the analytic force is the ADC(2)
        one, which would not be the derivative of another level's energy."""
        if self.level != 'adc2':
            raise ValueError(f"ADC2State(level={self.level!r}).force: the analytic force "
                             "is written for level='adc2' only")
        solved = self._solved_here(h, e)
        # the call's mean field is this field's: its SCF is not run again
        ref_flags = (list(self.mp2.hf.flags) if solved is not None
                     else self.mp2.hf(h, e).info['flags'])
        args = (self.mp2.mf, self.mol, self.spin, self.root, self.nroots,
                self.mp2.frozen, self.mp2.auxbasis)
        y_ref = (None if self.follow == 'index' or self.y_follow is None
                 else self.y_follow.in_orbitals(self.mp2.mf, self.mp2.frozen or 0))
        cphf = []
        with cphf_flags_into(cphf):
            if tiled_force_route(self.mp2.force_route, self.mol):
                F, info = adc2_force_tiled(*args, env=self.lr_environment,
                                           y_ref=y_ref, tile_gb=self.mp2.tile_gb,
                                           max_memory=self.mp2.max_memory,
                                           numerics=self.mp2.numerics,
                                           solved=solved,
                                           spin_scaling=self.mp2.scaling)
            else:
                # the dense reference is serial: replicated on every rank, so
                # it reuses only a serial call's operator
                if solved is not None and solved[0].size > 1:
                    solved = None
                    ref_flags = self.mp2.hf(h, e).info['flags']
                with distributed(None):
                    F, info = adc2_force(*args, env=self.lr_environment,
                                         y_ref=y_ref, numerics=self.mp2.numerics,
                                         solved=solved,
                                         spin_scaling=self.mp2.scaling)
        F, dm, lr = lockstep((F, info['dm_eff'], list(info.get('lr', ()))))
        flags = ref_flags + cphf + list(info.get('flags', []))
        if info.get('index') is not None:
            self._same_root(h, e, info, flags)
        return F, dict(info, dm_eff=dm, lr=lr,
                       flags=broadcast(flags, current_comm()))

    def _solved_here(self, h, e):
        """(op, k, y, omega) of the last call's root when the force is asked at
        that call's field and its orbitals are still the mean field's, else
        None (the force then converges its SCF and solves the roots afresh)."""
        if self.solved is None:
            return None
        h0, e0, mo, op, k, y, omega = self.solved
        if not (np.array_equal(np.asarray(h, float), h0) and float(e) == e0
                and np.array_equal(np.asarray(self.mp2.mf.mo_coeff), mo)):
            return None
        return op, k, y, omega

    def _same_root(self, h, e, info, flags):
        """Flag a force whose root is not the last call's (module docstring)."""
        label = f'ADC(2) {self.spin} force'
        if self.called is None:
            return
        h0, e0, omega0, root0 = self.called
        if self.follow == 'index':
            # under 'index' the force's Newton solve could order or miss roots
            # unlike the call's block solve
            T = transition_amplitudes(self.mp2.mf, self.mp2.frozen or 0,
                                      np.asarray(info['y'])[:, None])
            w = float(root0.weights(self.mol, T)[0])
            if w < ROOT_FOLLOW_WEIGHT_MIN:
                flag(flags, 'force_root_order_differs',
                     f"{label}: root {self.root} of the force's solve overlaps the "
                     f"call's root {self.root} by {w:.2f} (< {ROOT_FOLLOW_WEIGHT_MIN}): "
                     'the two solves order the roots differently; the force is '
                     "its own solve's root")
        if np.array_equal(np.asarray(h, float), h0) and float(e) == e0:
            d = abs(float(info['omega']) - omega0)
            if d > EE_FORCE_OMEGA_MATCH * self.conv_tol:
                flag(flags, 'force_omega_mismatch',
                     f"{label}: omega {info['omega']:.10f} Ha against the call's "
                     f'{omega0:.10f} Ha at the same field ({d:.1e} Ha apart): '
                     "possibly another state's force")
