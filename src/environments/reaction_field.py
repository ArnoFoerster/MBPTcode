"""The environment side of state-specific solvation: a polarizable medium seen
through its induced charges, with no knowledge of the electronic structure.

The self-consistent loops of `src.environments.solvation_loop` talk to the medium only
through this contract (`ReactionField`); the electronic-structure side only
through the state models of `src.gradients.state_models`. A medium is a
set of sites carrying induced charges q that answer the solute's potential V on
those sites linearly,

    q = Q V,        Q symmetric negative semi-definite,

in two parts on two time scales: the fast response (the medium's electrons,
optical) and the static one (fast plus the slow reorientation). The solute sees
the charges as a one-electron operator plus a constant (their interaction with
the nuclei). Nothing else is asked of a medium, so a continuum (here), a
polarizable embedding (induced dipoles on sites) or any other linear medium
plugs into the same loops.

The contract:

    nsites                         number of response sites
    potential(dm)                  V_tot on the sites: nuclei minus the
                                   electrons of the AO density dm (spin-summed)
    electronic_potential(dm)       the electrons' part alone (density differences)
    charges(V, response)           Q V, response 'static' or 'fast'
    operator(q)                    (h, e): the AO one-electron operator of the
                                   charges q and their interaction with the nuclei
    response_matrix(response)      the dense Q (gates, first-order terms)
    response_solve(q, response)    Q^-1 q (the free energy's quadratic term),
                                   Q factorized once per field and response
    transition_environment()       the object a method's linear-response term
                                   (the fast answer to a transition density)
                                   reads, or None when the method has none
    identity()                     what the medium is (model, eps, cavity sites),
                                   the key a loop's checkpoint is written for
    force(dm)                      d/dR [(1/2) V(dm)^T Q_s V(dm)] at a fixed AO
                                   density (sites, nuclei and basis moving)
    transition_force(dm)           d/dR [(1/2) V_e(dm)^T Q_f V_e(dm)] of the
                                   electrons of dm alone (a transition density)

force and transition_force take a symmetric AO density (a non-symmetric one is
refused, src.environments.pcm_derivatives.potential_integral_gradient) and are refused
on a medium whose sites do not move with the atoms (a continuum around an
enclosing shell: `differentiable` False).

By the stationarity of the free energy in the charges, the force of an
equilibrated state is the method's force in the fixed charges (the external
operator's own derivative left out) plus force(D_eff) plus, for a method with
an LR term, its weighted transition_force (src.environments.solvation_loop.force).
"""
from typing import Protocol

import numpy as np
import scipy.linalg
from pyscf.solvent import pcm as pyscf_pcm
from pyscf.solvent.grad import pcm as pcm_grad

from src.environments.pcm_derivatives import (bound_pyscf_pcm_blocks,
                                     potential_integral_gradient, solvation_gradient)
from src.environments.pcm_factorization import factorize_once
from src.Base.utils.record_io import record_hash
from src.environments.solvent_screening import SolventScreening, symmetrized_response

RESPONSES = ('static', 'fast')
CAVITY_KEYS = ('lebedev_order', 'vdw_scale', 'r_probe', 'radii_table')


def _cavity_of(env):
    """The cavity settings of a SolventScreening."""
    cavity = env.cavity_settings
    return {k: cavity[k] for k in CAVITY_KEYS}


class ReactionField(Protocol):
    """See the module docstring."""

    nsites: int

    def potential(self, dm): ...

    def electronic_potential(self, dm): ...

    def charges(self, V, response='static'): ...

    def operator(self, q): ...

    def response_matrix(self, response='static'): ...

    def response_solve(self, q, response='static'): ...

    def transition_environment(self): ...

    def identity(self): ...

    def force(self, dm, response='static'): ...

    def transition_force(self, dm, response='fast'): ...


class ContinuumField:
    """A polarizable continuum (pyscf's PCM: IEF-PCM, C-PCM, COSMO, SS(V)PE)
    on one cavity, answering at eps_static ('static') and eps_inf ('fast').

    Built from a `SolventScreening`, whose cavity settings it copies, so the
    surface is the one its linear-response term and its ground-state SCF use.
    Charges follow pyscf's symmetrized response, Q = (K^-1 R + R^T K^-T) / 2."""

    def __init__(self, mol, eps_static, eps_inf, method='IEF-PCM',
                 cavity=None, new_pcm=None, transition=None, differentiable=True):
        if eps_static < eps_inf:
            raise ValueError(f'eps_static {eps_static} below eps_inf {eps_inf}')
        self.mol = mol
        self.eps = {'static': float(eps_static), 'fast': float(eps_inf)}
        self.method = method
        self._cavity = dict(cavity or {})
        self._new_pcm = new_pcm or pyscf_pcm.PCM
        self._transition = transition
        self.differentiable = bool(differentiable)
        self._pcm = {r: self._build(self.eps[r]) for r in RESPONSES}
        self._Q = {}
        self._Q_factor = {}
        self.nsites = self._pcm['static'].surface['grid_coords'].shape[0]

    @classmethod
    def from_environment(cls, env, mol=None):
        """The continuum of a SolventScreening (its eps_static, its optical
        eps, its cavity); its linear-response term reads `env` itself."""
        mol = env.mol if mol is None else mol
        if env.eps_static is None:
            raise ValueError('the continuum has no eps_static: an equilibrium '
                             'reaction field needs one')
        return cls(mol, env.eps_static, env.eps_optical, env.method, _cavity_of(env),
                   env.new_pcm, transition=_optical_only(env, mol),
                   differentiable=env.differentiable)

    def _build(self, eps):
        p = self._new_pcm(self.mol)
        p.eps, p.method, p.verbose = eps, self.method, 0
        for k, v in self._cavity.items():
            setattr(p, k, v)
        bound_pyscf_pcm_blocks(p)
        p.build()
        factorize_once(p)
        return p

    def electronic_potential(self, dm):
        return -self._pcm['static']._get_v(np.asarray(dm)[None])[0]

    def potential(self, dm):
        return self._pcm['static'].v_grids_n + self.electronic_potential(dm)

    def response_matrix(self, response='static'):
        if response not in self._Q:
            self._Q[response] = symmetrized_response(self._pcm[response])
        return self._Q[response]

    def response_solve(self, q, response='static'):
        """Q^-1 q by a Cholesky factor of -Q (Q is negative definite for eps
        > 1), factorized once per response. At eps = 1, Q = 0: the only
        charges are zero and so is their quadratic term."""
        q = np.asarray(q, float)
        if response not in self._Q_factor:
            Q = self.response_matrix(response)
            self._Q_factor[response] = (None if not np.any(Q)
                                        else scipy.linalg.cho_factor(-Q))
        if self._Q_factor[response] is None:
            if np.any(q):
                raise ValueError(f'nonzero charges in a {response} response that is '
                                 'zero (eps = 1)')
            return np.zeros_like(q)
        return -scipy.linalg.cho_solve(self._Q_factor[response], q)

    def charges(self, V, response='static'):
        return self.response_matrix(response) @ np.asarray(V)

    def operator(self, q):
        q = np.asarray(q, float)
        p = self._pcm['static']
        return p._get_vmat(q[None])[0], float(q @ p.v_grids_n)

    def transition_environment(self):
        return self._transition

    def identity(self):
        """What this medium is, for a checkpoint's key: the model, both
        dielectric constants, the cavity settings and a digest of its sites
        and their areas."""
        surf = self._pcm['static'].surface
        return dict(medium='continuum', method=self.method,
                    eps_static=self.eps['static'], eps_inf=self.eps['fast'],
                    cavity={k: (None if v is None else str(v))
                            for k, v in sorted(self._cavity.items())},
                    sites=record_hash((np.asarray(surf['grid_coords'], float),
                                       np.asarray(surf['area'], float))))

    def _require_differentiable(self):
        if not self.differentiable:
            raise ValueError('this continuum is not differentiable in the nuclear '
                             'coordinates (an enclosing shell\'s spheres do not move '
                             'with the atoms): no force on it')

    def force(self, dm, response='static'):
        """d/dR [(1/2) q . V_tot] at fixed dm, q = Q V_tot (pyscf's PCM force:
        nuclei, the integrals of the moving basis, the cavity's own)."""
        self._require_differentiable()
        p = self._pcm[response]
        p._intermediates.pop('dm', None)
        return solvation_gradient(p, np.asarray(dm))

    def transition_force(self, dm, response='fast'):
        """d/dR [(1/2) q . V_e] at fixed dm with the nuclear potential left
        out: the medium's part of an LR term 2 t^T Q t (weight 4)."""
        self._require_differentiable()
        p = self._pcm[response]
        dm = np.asarray(dm)
        v_n = p.v_grids_n
        try:
            p.v_grids_n = np.zeros_like(v_n)
            p._intermediates.pop('dm', None)
            return potential_integral_gradient(p, dm) + pcm_grad.grad_solver(p, dm)
        finally:
            p.v_grids_n = v_n
            p._intermediates.pop('dm', None)


class NoMedium:
    """The gas phase as a medium with no sites: every loop of
    src.environments.solvation_loop then converges in its second cycle to the bare
    state, so gas-phase and solvated runs share one code path."""

    nsites = 0
    differentiable = True

    def __init__(self, mol):
        self.mol = mol

    def potential(self, dm):
        return np.zeros(0)

    def electronic_potential(self, dm):
        return np.zeros(0)

    def charges(self, V, response='static'):
        return np.zeros(0)

    def operator(self, q):
        return np.zeros((self.mol.nao, self.mol.nao)), 0.0

    def response_matrix(self, response='static'):
        return np.zeros((0, 0))

    def response_solve(self, q, response='static'):
        return np.zeros(0)

    def transition_environment(self):
        return None

    def identity(self):
        return dict(medium='none')

    def force(self, dm, response='static'):
        return np.zeros((self.mol.natm, 3))

    def transition_force(self, dm, response='fast'):
        return np.zeros((self.mol.natm, 3))


def _optical_only(env, mol):
    """The SolventScreening of env's optical response alone (no eps_static):
    what a method's LR term reads when the reaction field of its states is
    handled by the loop and not by a solvated SCF."""
    return SolventScreening(mol, eps=env.eps_optical, allow_static_eps=True,
                            method=env.method, cavity_atoms=env.cavity_atoms,
                            cavity_unit='Bohr', **_cavity_of(env))
