"""The electronic-structure side of state-specific solvation: state models.

A state model answers one question, with no knowledge of what produced the
field:

    result = state(h, e)

h an AO one-electron operator added to the core Hamiltonian, e a constant;
result.energy the state's energy in H_vac + h + e and result.dm_eff its
orbital-relaxed density dE/dh (AO, spin-summed), so that the medium's charges
equilibrate with the density the energy actually responds through. A model
warm-starts each call from its previous one. src.environments.solvation_loop iterates
state models against any src.environments.reaction_field medium; a new method joins
by implementing this call (src.gradients.adc.ee_state_model for
ADC(2)).

    HFState    the mean field in the field (the SCF re-converged): Hartree-Fock,
               or a Kohn-Sham reference (its force is its own gradient class)
    MP2State   + DF-MP2 and its relaxed correlation density (Hartree-Fock
               references only: its CPHF answers with the HF kernel)

Both refuse a mean field whose orbital gradient is not converged to
STATE_MODEL_SCF_GRAD_TOL (conv_tol_grad unset or looser): the relaxed density
and the Hellmann-Feynman force assume stationary orbitals. An SCF in the
field that stops short of its tolerances is a numerical event (the policy:
refuse inputs, warn numerics): it is warned about and flagged in
the call's info['flags'] as 'scf_not_converged', and the loop carries on.
Each call's info['flags'] is a list of {'flag': name, 'message': text}, the
model's numerical events at that call (src.environments.solvation_loop copies the last
cycle's into its result).

A model whose loop can be resumed (src.environments.solvation_loop, checkpoint)
answers state.checkpoint() -> its warm start as arrays, state.restore(saved)
takes it back, and state.identity() says what it computes (model, basis,
reference, method settings, root), the key a checkpoint is written for.

A model that can be optimized also answers state.force(h, e) -> (F, info):
its nuclear force in the fixed field with the external operator's own
derivative left out (the medium adds it, src.environments.reaction_field.force), and
info['dm_eff'], info['lr'] = [(weight, transition density)] for its LR terms.
Forces take the dense reference (src.gradients.df_correlation_force) or,
above EE_FORCE_DENSE_MAX_NAO basis functions or when asked (force_route), the
tiled route over the current MPI ranks (df_correlation_force_tiled).

Over MPI ranks (src.Base.utils.parallel.mpi_grid.current_comm) every rank calls the
model with the same field; every call and force returns rank 0's bits on every
rank (lockstep), the mean field included (lockstep_mean_field), so the loop
above takes one decision on all ranks. The replicated pieces (the SCF, pyscf's
DF-MP2) run on every rank; serially the locksteps are no-ops.
"""
import warnings

import numpy as np
from pyscf import lib
from pyscf.dft.rks import KohnShamDFT
from pyscf.mp import dfmp2_native

from src.Base.numerics.constants import (EE_FORCE_DENSE_MAX_NAO, EE_RI_TILE_GB,
                                         STATE_MODEL_SCF_GRAD_TOL)
from src.Base.utils.record_io import record_hash
from src.Base.utils.parallel.mpi_grid import (broadcast, current_comm, distributed, lockstep,
                                              lockstep_mean_field)
from src.gradients import df_correlation_force as dcf
from src.gradients import df_correlation_force_tiled as tcf
from src.gradients.df_factors import mp2_factors
from src.gradients.multipliers import rotation_gradient, solve_orbital_multipliers


FORCE_ROUTES = ('auto', 'dense', 'tiled')


class StateResult:
    """energy (Hartree), dm_eff (AO, spin-summed), info (model-specific)."""

    def __init__(self, energy, dm_eff, info=None):
        self.energy, self.dm_eff, self.info = float(energy), dm_eff, info or {}


class HFState:
    """Closed-shell mean field in an external field. make_scf(mol) -> a
    configured, unconverged pyscf RHF or RKS (density fitting, tolerances,
    conv_tol_grad at most STATE_MODEL_SCF_GRAD_TOL) in the gas phase; its core
    Hamiltonian is replaced by h_vac + h on each call."""

    def __init__(self, mol, make_scf):
        self.mol = mol
        self.mf = make_scf(mol)
        grad = getattr(self.mf, 'conv_tol_grad', None)
        if grad is None or grad > STATE_MODEL_SCF_GRAD_TOL:
            raise ValueError(f'the state model\'s SCF has conv_tol_grad {grad}: set it '
                             f'to {STATE_MODEL_SCF_GRAD_TOL:.0e} or tighter (the relaxed '
                             'density and the force assume stationary orbitals)')
        self._h0 = self.mf.get_hcore()
        self._dm = None

    def __call__(self, h, e):
        h1 = self._h0 + h
        self.mf.get_hcore = lambda *a, **k: h1
        self.mf.kernel(dm0=self._dm)
        lockstep_mean_field(self.mf)
        # rank 0's mean field, converged flag included, on every rank
        flags = []
        if not self.mf.converged:
            text = 'the SCF in the external field did not converge; kept where it stopped'
            warnings.warn(text, RuntimeWarning, stacklevel=2)
            flags.append({'flag': 'scf_not_converged', 'message': text})
        self._dm = self.mf.make_rdm1()
        return StateResult(self.mf.e_tot + e, self._dm,
                           {'e_hf': self.mf.e_tot + e, 'flags': flags})

    def force(self, h, e):
        r = self(h, e)
        return lockstep(dcf.hf_force(self.mf)), dict(dm_eff=r.dm_eff, lr=[],
                                                     flags=r.info['flags'])

    def checkpoint(self):
        """The SCF warm start (module docstring) with the orbitals it is
        tagged with: pyscf builds the first exchange of a tagged density from
        its occupied orbitals, of an untagged one from the density, and the
        two differ in the last bits."""
        if self._dm is None:
            return {'dm': None}
        return {'dm': np.asarray(self._dm), 'mo_coeff': self._dm.mo_coeff,
                'mo_occ': self._dm.mo_occ}

    def identity(self):
        """The reference this model converges (module docstring)."""
        mf, mol = self.mf, self.mol
        df = getattr(mf, 'with_df', None)
        return dict(model='mean field', scf=type(mf).__name__,
                    xc=getattr(mf, 'xc', None) if isinstance(mf, KohnShamDFT) else None,
                    basis=record_hash(mol._basis), charge=int(mol.charge),
                    spin=int(mol.spin),
                    jk_auxbasis=None if df is None else str(getattr(df, 'auxbasis', None)))

    def restore(self, saved):
        dm = saved.get('dm')
        self._dm = None if dm is None else lib.tag_array(
            np.array(dm, float), mo_coeff=np.array(saved['mo_coeff'], float),
            mo_occ=np.array(saved['mo_occ'], float))


class MP2State:
    """MP2 (density-fitted, `auxbasis`, `frozen` core orbitals) on an HFState.
    force_route, tile_gb, max_memory (MB, default mol.max_memory): the force's
    route (tiled_force_route) and the tiled route's tiles and memory bound;
    factors: the tiled route's factor provider
    (`df_correlation_force_tiled.mp2_force_tiled`), by default the fit's own
    RI factors (`df_factors.mp2_factors`); numerics: the caller's `Numerics`
    (the Z-vector's tolerances)."""

    def __init__(self, hf, frozen=0, auxbasis=None, force_route='auto',
                 tile_gb=EE_RI_TILE_GB, max_memory=None, numerics=None, factors=None):
        if isinstance(hf.mf, KohnShamDFT):
            raise ValueError('MP2State needs a Hartree-Fock reference: its relaxed '
                             'density and force solve the CPHF with the HF kernel, '
                             f'not {type(hf.mf).__name__}\'s')
        self.hf, self.frozen, self.auxbasis = hf, frozen, auxbasis
        self.mol = hf.mol
        tiled_force_route(force_route, self.mol)
        self.force_route, self.tile_gb, self.max_memory = force_route, tile_gb, max_memory
        self.numerics = numerics
        self.factors = mp2_factors if factors is None else factors

    @property
    def mf(self):
        return self.hf.mf

    def __call__(self, h, e):
        r = self.hf(h, e)
        flags = list(r.info['flags'])
        e2, D2 = mp2_relaxed(self.hf.mf, self.frozen, self.auxbasis, self.numerics, flags)
        e2, D2 = lockstep((e2, D2))
        flags = broadcast(flags, current_comm())
        return StateResult(r.energy + e2, r.dm_eff + D2,
                           dict(r.info, e_mp2=e2, flags=flags))

    def checkpoint(self):
        return self.hf.checkpoint()

    def identity(self):
        return dict(self.hf.identity(), model='MP2', frozen=self.frozen,
                    auxbasis=str(self.auxbasis))

    def restore(self, saved):
        self.hf.restore(saved)

    def force(self, h, e):
        flags = list(self.hf(h, e).info['flags'])
        mf, mol = self.hf.mf, self.mol
        if tiled_force_route(self.force_route, mol):
            F, (_, gam, _, _) = tcf.mp2_force_tiled(mf, mol, self.factors, self.frozen,
                                                    self.auxbasis,
                                                    tile_gb=self.tile_gb,
                                                    max_memory=self.max_memory,
                                                    numerics=self.numerics, flags=flags)
        else:
            with distributed(None):   # the dense reference: serial, on every rank
                F, (_, gam, _, _) = dcf.mp2_force(mf, mol, self.frozen, self.auxbasis,
                                                  numerics=self.numerics, flags=flags)
        C = np.asarray(mf.mo_coeff)
        F, info = lockstep((F, dict(dm_eff=mf.make_rdm1() + C @ gam @ C.T, lr=[])))
        return F, dict(info, flags=broadcast(flags, current_comm()))


def mp2_relaxed(mf, frozen, auxbasis, numerics=None, flags=None):
    """(e_mp2, D) of DF-MP2 on mf: D the orbital-relaxed correlation density
    (AO, spin-summed, the SCF part removed), d E_MP2 / d h, the orbitals
    responding through mf's own kernel. pyscf's DF-MP2 builds the density;
    its CPHF is the one Z-vector entry of every gradient (`multipliers`, its
    tolerances from `numerics`, a short solve flagged into `flags`), where
    pyscf's own krylov stops near a residual of 3e-6."""
    mp = dfmp2_native.DFRMP2(mf, frozen=frozen or None, auxbasis=auxbasis)
    mp.verbose = 0
    e_mp2 = mp.kernel()

    def cphf(mf, Lvo, _cycles, _tol, _logger):
        # pyscf's call, its own cycles, tolerance and logger unread:
        # (e_a - e_i) z + R[z] = -Lvo is antisym(Y_E + Y[Lambda]) = 0
        # on the virtual-occupied pairs with Y_E[a, i] = Lvo and z = 2 Lambda
        C, eps = np.asarray(mf.mo_coeff), np.asarray(mf.mo_energy, float)
        nmo, nocc = C.shape[1], int(np.count_nonzero(mf.mo_occ > 0))
        kernel, fock = dcf.scf_kernel(mf), np.diag(eps)
        Y_E = np.zeros((nmo, nmo))
        Y_E[nocc:, :nocc] = Lvo
        free = np.zeros((nmo, nmo), bool)
        free[:nocc, nocc:] = True
        Lam, _ = solve_orbital_multipliers(
            lambda L: rotation_gradient(C, fock, L, nocc, kernel), Y_E, eps,
            numerics=numerics, free=free, flags=flags)
        return 2.0 * Lam[nocc:, :nocc]
    with lib.temporary_env(dfmp2_native, solve_cphf_rhf=cphf):
        rdm1 = mp.make_rdm1(relaxed=True)
    nocc = mp.nocc
    rdm1[:nocc, :nocc] -= 2.0 * np.eye(nocc)
    C = mf.mo_coeff
    mp.delete()
    return float(e_mp2), C @ rdm1 @ C.T


def tiled_force_route(route, mol):
    """True when a correlated state model's force takes the tiled route:
    route 'tiled', or 'auto' above EE_FORCE_DENSE_MAX_NAO basis functions."""
    if route not in FORCE_ROUTES:
        raise ValueError(f'force_route={route!r}; expected one of {FORCE_ROUTES}')
    return route == 'tiled' or (route == 'auto' and mol.nao > EE_FORCE_DENSE_MAX_NAO)
