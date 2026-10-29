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
               or a Kohn-Sham reference (its force `isdf_mean_field.
               mean_field_skeleton_force`, the quadrature's motion included)
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

A mean-field model also answers state.loop_response(h, field): the
response of its density, equilibrated in the medium, contracted with a fixed
operator (src.gradients.scf_loop_response), which the force of a state in
its slow charges reads (src.environments.solvation_loop.force_neq).

A model that can be optimized also answers state.force(h, e) -> (F, info):
its nuclear force in the fixed field with the external operator's own
derivative left out (the medium adds it, src.environments.reaction_field.force), and
info['dm_eff'], info['lr'] = [(weight, transition density)] for its LR terms.
Forces take the dense reference (src.gradients.df_correlation_force) or,
above EE_FORCE_DENSE_MAX_NAO basis functions or when asked (force_route), the
tiled route over the current MPI ranks (df_correlation_force_tiled); an MP2
call's energy and relaxed density come from the same engine and route
(mp2_density / mp2_density_tiled), which also serve a mean field that dropped
linearly dependent AO combinations (nmo < nao).

Over MPI ranks (src.Base.utils.parallel.mpi_grid.current_comm) every rank calls the
model with the same field; every call and force returns rank 0's bits on every
rank (lockstep), the mean field included (lockstep_mean_field), so the loop
above takes one decision on all ranks. The replicated pieces (the SCF and the
dense reference) run on every rank; serially the locksteps are no-ops.
"""
import warnings

import numpy as np
from pyscf import lib
from pyscf.dft.rks import KohnShamDFT

from src.Base.numerics.constants import (EE_FORCE_DENSE_MAX_NAO, EE_RI_TILE_GB,
                                         STATE_MODEL_SCF_GRAD_TOL)
from src.Base.utils.record_io import record_hash
from src.Base.utils.parallel.mpi_grid import (broadcast, current_comm, distributed, lockstep,
                                              lockstep_mean_field)
from src.gradients import df_correlation_force as dcf
from src.gradients import df_correlation_force_tiled as tcf
from src.gradients.df_factors import mp2_factors
from src.gradients.isdf_mean_field import mean_field_skeleton_force
from src.SingleReference.common.df_amplitudes import spin_scaling_factors
from src.gradients.scf_loop_response import loop_density_response


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
        # the last call's flags
        self.flags = []

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
        self.flags = flags
        self._dm = self.mf.make_rdm1()
        return StateResult(self.mf.e_tot + e, self._dm,
                           {'e_hf': self.mf.e_tot + e, 'flags': flags})

    def force(self, h, e):
        r = self(h, e)
        return lockstep(mean_field_skeleton_force(self.mf)), dict(dm_eff=r.dm_eff, lr=[],
                                                     flags=r.info['flags'])

    def loop_response(self, h, field):
        """(natm, 3) Tr[h dD/dR] of this mean field equilibrated in `field`
        (its last call the loop's converged cycle): the density's response
        with the continuum's at eps_static (`scf_loop_response`)."""
        return lockstep(loop_density_response(self.mf, self._h0,
                                              field.scf_continuum(), h))

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
    numerics: the one `Numerics` of the calculation, required (the Z-vector's
    tolerances, scf; the RI tiles, layout.tile_gb, None: EE_RI_TILE_GB; and an
    ADC2State's solve on this model, adc); force_route, max_memory (MB, default
    mol.max_memory): the force's route (tiled_force_route) and the tiled
    route's memory bound; factors: the tiled route's factor provider
    (`df_correlation_force_tiled.mp2_force_tiled`), by default the fit's own
    RI factors (`df_factors.mp2_factors`); spin_scaling: None (MP2), 'scs', 'sos'
    (ADC2_SPIN_SCALING) or (c_os, c_ss): SCS-MP2, the ground state of an
    ADC2State on this model, which is then SCS-ADC(2) with the same factors."""

    def __init__(self, hf, frozen=0, auxbasis=None, force_route='auto',
                 max_memory=None, numerics=None, factors=None, spin_scaling=None):
        if numerics is None:
            raise ValueError('MP2State needs its Numerics (the Z-vector\'s tolerances, '
                             'the RI tiles, the ADC solve of a state on it): there are '
                             'no defaults to fall back on')
        if isinstance(hf.mf, KohnShamDFT):
            raise ValueError('MP2State needs a Hartree-Fock reference: its relaxed '
                             'density and force solve the CPHF with the HF kernel, '
                             f'not {type(hf.mf).__name__}\'s')
        self.hf, self.frozen, self.auxbasis = hf, frozen, auxbasis
        self.mol = hf.mol
        tiled_force_route(force_route, self.mol)
        tile_gb = numerics.layout.tile_gb
        self.tile_gb = EE_RI_TILE_GB if tile_gb is None else tile_gb
        self.force_route, self.max_memory = force_route, max_memory
        self.numerics = numerics
        self.factors = mp2_factors if factors is None else factors
        self.spin_scaling = spin_scaling
        self.scaling = spin_scaling_factors(spin_scaling)

    @property
    def mf(self):
        return self.hf.mf

    def __call__(self, h, e):
        r = self.hf(h, e)
        flags = list(r.info['flags'])
        e2, D2 = lockstep(self._relaxed(flags))
        flags = broadcast(flags, current_comm())
        return StateResult(r.energy + e2, r.dm_eff + D2,
                           dict(r.info, e_mp2=e2, flags=flags))

    def _relaxed(self, flags):
        """(e_mp2, D): D the orbital-relaxed correlation density (AO,
        spin-summed, the SCF part removed), d E_MP2 / d h, from the force's
        own engine (no derivative integrals): the tiled route over the ranks or
        the dense reference, as force_route decides; a short Z-vector solve
        flagged into flags."""
        mf, mol = self.hf.mf, self.mol
        if tiled_force_route(self.force_route, mol):
            comm, _ = tcf.prepare(mf, mol, self.frozen, self.auxbasis, False,
                                  self.tile_gb, max_memory=self.max_memory)
            provider = self.factors(mol, mf, self.frozen, self.auxbasis, self.tile_gb,
                                    comm, None)
            e2, gam = tcf.mp2_density_tiled(mf, mol, provider, self.frozen,
                                            self.auxbasis, self.numerics, flags,
                                            self.scaling)
        else:
            with distributed(None):   # the dense reference: serial, on every rank
                e2, gam = dcf.mp2_density(mf, mol, self.frozen, self.auxbasis,
                                          self.numerics, flags, self.scaling)
        C = np.asarray(mf.mo_coeff)
        return float(e2), C @ gam @ C.T

    def checkpoint(self):
        return self.hf.checkpoint()

    def identity(self):
        out = dict(self.hf.identity(), model='MP2', frozen=self.frozen,
                   auxbasis=str(self.auxbasis))
        if self.scaling is not None:
            out['spin_scaling'] = list(self.scaling)
        return out

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
                                                    numerics=self.numerics, flags=flags,
                                                    scaling=self.scaling)
        else:
            with distributed(None):   # the dense reference: serial, on every rank
                F, (_, gam, _, _) = dcf.mp2_force(mf, mol, self.frozen, self.auxbasis,
                                                  numerics=self.numerics, flags=flags,
                                                  scaling=self.scaling)
        C = np.asarray(mf.mo_coeff)
        F, info = lockstep((F, dict(dm_eff=mf.make_rdm1() + C @ gam @ C.T, lr=[])))
        return F, dict(info, flags=broadcast(flags, current_comm()))


def tiled_force_route(route, mol):
    """True when a correlated state model's force takes the tiled route:
    route 'tiled', or 'auto' above EE_FORCE_DENSE_MAX_NAO basis functions."""
    if route not in FORCE_ROUTES:
        raise ValueError(f'force_route={route!r}; expected one of {FORCE_ROUTES}')
    return route == 'tiled' or (route == 'auto' and mol.nao > EE_FORCE_DENSE_MAX_NAO)
