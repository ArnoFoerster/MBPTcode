"""GW-BSE behind the fragment-diabatic gradient: `BSEFragmentMethod`, the
`FragmentMethod` of an ExcitedStateChain, and its `SinglesResponse`.

The Lagrangian of an element -- diabat responses, rotation density,
localization response -- is method-neutral and lives in
`src.gradients.fragment_diabatic`; the partition in `src.fragments.partition`;
the operator in `src.gradients.gwbse.fragment_bse`. What is GW-BSE's own is
here:

THE REVERSE PASS (term 1 of the Lagrangian). Every first derivative is a
contraction of dK with a few canonical vector pairs (s, t), fed to the
excited-state reverse chain as an interstate seed: `bse_backward` and
`interstate_backward` contract dA with x_s x_t^T + y_s y_t^T and dB with
x_s y_t^T + y_s x_t^T, which is s^T dK t, and they only read columns of the
amplitude arrays, so the diabat vectors take the place of roots, and
`fold_to_nuclei` carries the quasiparticle, screening and integral adjoints
unchanged. With the chain's `bse_adjoint='grid'` that pass runs over the ISDF
grid without any (naux, n_occ, n_vir) block.

THE QUASIPARTICLE HAMILTONIAN. Where the chain solves one
(`qp.hamiltonian`), K carries its off-diagonal one-body blocks
(`fragment_bse`), and every pair adds its adjoint on them
(`QPHamiltonian.one_body_adjoint`) to the reverse pass, which takes it
through each group's fixed point as the chain's own roots do.

A near-degenerate pair solved diagonally is refused
(`refuse_diagonal_near_degeneracies`): two orbitals closer than
`qp.degeneracy_tol` whose own corrections differ (an accidental
near-degeneracy, which the chain keeps on separate corrections) give an A_eff
that depends on how the two rotate into each other, so A_eff(R) steps where
they mix. Only a group of the quasiparticle Hamiltonian that holds both is
invariant to that.

THE GAUGE (term 3). The chain keeps the Kohn-Sham Fock matrix diagonal, the
canonical condition, so its reverse pass moves the canonical orbitals with the
canonical derivative overlap: `frame_force` is
`orbital_coupling.configuration_coupling`, with its one Z-vector. This holds
inside a group of the quasiparticle Hamiltonian too. H is invariant to how a
group's orbitals rotate into each other, but the screening and the
self-energy's internal lines read the members' Kohn-Sham energies on the
diagonal, so a rotation inside a group still costs a canonical rotation:
what it carries is proportional to the members' gap, and so is its weight
1/(eps_p - eps_q)'s numerator.

GATED (tests/test_fragment_diabatic.py, offset ethylene dimer, cc-pVDZ), in
both kernels: a root's gradient through the partition equals the
supermolecular root gradient (Tamm-Dancoff or full) to 1e-8 relative, with the
orbital-rotation terms vanishing for it as they must; each element's gradient
along a random direction equals the relocalized finite difference, with the
canonical-gauge and localization terms several percent of the total. The
iterative path reproduces the dense one to 1e-8 relative on the test system.
"""

import numpy as np

from src.Base.numerics.numerics import surface_numerics
from src.Base.utils.parallel.mpi_grid import lockstep
from src.fragments.partition import split_xy
from src.gradients.gwbse.bse_isdf import bse_backward, bse_cache, interstate_backward
from src.gradients.gwbse.fragment_bse import BSEOperator
from src.gradients.gwbse.isdf_bse_adjoint import (
    isdf_bse_backward, isdf_interstate_backward)
from src.gradients.orbital_coupling import configuration_coupling


class BSEFragmentMethod:
    """The `FragmentMethod` (`src.gradients.fragment_diabatic`) of an
    ExcitedStateChain: its mean field, its BSE kernel (Tamm-Dancoff or full,
    `bse_tda`) and its reverse chain."""

    def __init__(self, chain):
        self.chain = chain
        self.mol0 = chain.mol0
        self.numerics = surface_numerics(chain)

    def require_differentiable_environment(self):
        self.chain.require_differentiable_environment()

    def mean_field(self, mol=None, mf=None):
        return self.chain.mean_field(mol, mf)

    def singles(self, mol, mf, route='auto', omega0=None):
        """(BSEOperator, BSEResponse) at one geometry; route 'auto' | 'dense'
        | 'isdf'. The kernel does not depend on Omega, so `omega0` is the
        partition's alone."""
        op, shared = BSEOperator.from_chain(self.chain, mol, mf, route=route)
        refuse_diagonal_near_degeneracies(self.chain)
        return op, BSEResponse(self.chain, op, shared, mol, mf)

    def ground_energy(self, mf):
        return mf.e_tot

    def ground_gradient(self, mf):
        return self.chain.mean_field_gradient(mf)

    def refreeze(self, mol, mf=None):
        return BSEFragmentMethod(self.chain.refreeze(mol, mf=mf))


def refuse_diagonal_near_degeneracies(chain):
    """Raise ValueError if the chain's frozen set holds a near-degenerate
    pair with corrections of its own that no frozen group of the
    quasiparticle Hamiltonian holds whole (module docstring). The pairs are
    the chain's own record (`qp_block_record`'s rejected blocks), decided at
    the reference geometry; true degeneracies share one correction and pass."""
    frozen = getattr(chain, 'conventions', chain)
    record = getattr(frozen, 'qp_block_record', None) or {}
    subspaces = getattr(frozen, 'qp_subspaces', None)
    groups = [set(g) for g in subspaces[1]] if subspaces else []
    loose = [b for b in record.get('rejected', [])
             if not any(set(b['members']) <= g for g in groups)]
    if loose:
        raise ValueError(
            'fragment-diabatic GW-BSE: near-degenerate orbitals solved on '
            'separate quasiparticle corrections, '
            f'{[b["members"] for b in loose]} (orbital-energy spreads (Ha) '
            f'{[b["energy_spread_Ha"] for b in loose]}, correction spreads '
            f'(Ha) {[b["correction_spread_Ha"] for b in loose]}). A_eff then '
            'depends on how they rotate into each other. Solve them in one '
            "group of the quasiparticle Hamiltonian: numerics 'qp_hamiltonian'"
            ": True, with every member on the pole model (residue_route "
            "'sop').")


class BSEResponse:
    """The `SinglesResponse` of one BSE operator (module docstring)."""

    def __init__(self, chain, operator, shared, mol, mf):
        self.chain, self.operator, self.shared = chain, operator, shared
        self.mol, self.mf = mol, mf
        self._cache = {}
        # the chain records its own numerical events; none of this pass's
        self.flags = {}

    def _one_body_adjoint(self, pairs):
        """The pairs' adjoint on the quasiparticle Hamiltonian's off-diagonal
        elements, rank 0's on every rank, or None without one."""
        qp_h, nocc = self.shared.qp_h, self.operator.nocc
        if qp_h is None:
            return None
        nv, n_ov = self.operator.nvir, self.operator.n_ov
        terms = []
        for v, _, wgt in pairs:
            xs, ys = split_xy(v, n_ov)
            terms.append((xs[:, 0].reshape(nocc, nv), ys[:, 0].reshape(nocc, nv),
                          xs[:, 1].reshape(nocc, nv), ys[:, 1].reshape(nocc, nv),
                          wgt))
        return lockstep(np.ascontiguousarray(qp_h.one_body_adjoint(nocc, terms)),
                        check=True)

    def pair_force(self, pairs):
        """The reverse chain for every pair, in the chain's `bse_adjoint`
        realization: 'explicit' through the three-index blocks, 'grid' over
        the ISDF grid, divided over the ranks, with no (naux, n_occ, n_vir)
        block."""
        chain = self.chain
        sh = self.shared
        x_mo, d, eps_qp, w_aux = sh.x_mo, sh.d, sh.eps_qp, sh.w_aux
        grid = getattr(chain, 'bse_adjoint', 'explicit') == 'grid'
        kw = dict(spin=chain.spin, bse_tda=self.operator.tda)
        cache = self._cache
        if not grid and not cache:
            cache.update(bse_cache(x_mo, d, eps_qp, chain.nocc, **kw))
        total = None
        for v, same, wgt in pairs:
            xs, ys = split_xy(v, self.operator.n_ov)
            if grid and same:
                part = isdf_bse_backward(0, x_mo, d, eps_qp, w_aux, chain.nocc,
                                         xs, ys, omega_bar=wgt, **kw)
            elif grid:
                # one pass: the grid element is symmetric in its two vectors
                part = isdf_interstate_backward(0, 1, x_mo, d, eps_qp, w_aux,
                                                chain.nocc, xs, ys,
                                                omega_bar=wgt, **kw)
            elif same:
                part = bse_backward(0, x_mo, d, eps_qp, w_aux, chain.nocc,
                                    cache, xs, ys, omega_bar=wgt)
            else:
                part = interstate_backward(0, 1, x_mo, d, eps_qp, w_aux,
                                           chain.nocc, cache, xs, ys,
                                           omega_bar=wgt)
            total = part if total is None else tuple(
                x + y for x, y in zip(total, part))
        return chain.reverse.fold_to_nuclei(
            self.shared, *total, h_bar=self._one_body_adjoint(pairs))[0]

    def frame_force(self, dc_o, dc_v):
        """The canonical gauge: - sum D (.) A^can, `configuration_coupling`."""
        return configuration_coupling(self.mol, self.mf, self.operator.nocc,
                                      dc_o.T, -dc_v,
                                      surface_numerics(self.chain))
