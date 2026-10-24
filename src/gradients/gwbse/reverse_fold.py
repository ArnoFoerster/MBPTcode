"""The reverse fold of the excited-state chain: Casida-level adjoints to nuclei.

Everything below the Casida step is linear in its seed, so the quasiparticle,
screening, reaction-field and Kohn-Sham stages carry dOmega_n or an interstate
element alike. `ExcitedStateChain` inherits these methods.
"""
from dataclasses import dataclass

import numpy as np

from src.Base.factorization.sliced_factors import GridTileRows
from src.Base.numerics.constants import FIT_CHOLESKY_BLOCK
from src.Base.utils.parallel.mpi_grid import current_comm
from src.gradients.gwbse.bse_isdf import screening_backward
from src.gradients.gwbse.reaction_field_adjoint import (
    added_reaction_field_adjoint, offdiagonal_reaction_field_backward,
    reaction_field_backward)
from src.gradients.isdf_derivatives import GaugeAdjoint
from src.gradients.rpa.space_time_adjoint import chi0_backward, chi0_backward_rows


@dataclass(eq=False)
class ReverseState:
    """The adjoints one reverse pass accumulates stage by stage: the seeds of
    X_mo, D and eps, the bare-factor and screening-gauge adjoints, and what
    the stages hand each other (set solve output, static weights, split)."""

    xc: object
    block: object
    x_bar: object
    d_bar: object
    d_sigma: object
    d_bar_bare: object
    eps_bar: np.ndarray
    mask: np.ndarray
    lam: object
    qp_h: object
    seed_set: np.ndarray
    pairs: tuple
    h_bar: object = None
    qp_out: dict = None
    split: object = None
    d_w: object = None
    d_bar_w: object = None
    t_bar: object = None
    chi0_bar: object = None
    w_corr: np.ndarray = None


class ReversePass:
    """The reverse pass from the Casida adjoints to the nuclear gradient."""

    def __init__(self, chain):
        self.chain = chain

    def adjoint_rows(self, x_mo):
        """The tile edge the factors' adjoints are held in over the ranks,
        or None serially, where they are whole.

        Over more than one rank every adjoint of X_mo and D is `GridTileRows`
        in the row fit's tiles (`fit_block`, `FIT_CHOLESKY_BLOCK` on the
        replicated fit), owned t % size: the sweeps are tile-major and the
        fit adjoint reads the same tiles, so no rank holds a whole pair. The
        factors' own layout, sliced or whole, reaches the same calls, so the
        two layouts keep one force.
        """
        comm = current_comm()
        if comm is None or comm.Get_size() < 2:
            return None
        return FIT_CHOLESKY_BLOCK if self.chain.fit_block is None else self.chain.fit_block

    @staticmethod
    def as_rows(a, block):
        """`a` as `GridTileRows` in tiles of `block`: a whole array cut to
        this rank's tiles, verbatim, and tiles handed back as they are."""
        if isinstance(a, GridTileRows):
            return a
        return GridTileRows.from_whole(np.asarray(a), block, current_comm())

    def fold_to_nuclei(self, shared, eqp_bar, x_bar, d_bar, w_bar,
                       release_tape=True, h_bar=None):
        """(natm, 3) from the four Casida-level adjoints, and diagnostics.

        Everything below the Casida step is LINEAR in the seed, so the same
        sequence -- quasiparticle, screening, reaction field, Kohn-Sham
        correction, integrals -- carries dOmega_n and the interstate element
        <m| dH |n> without knowing which it is holding.

        x_bar and d_bar are CONSUMED: every stage's adjoint is added into them
        in place, in the order the sums always ran, and dropped, so the
        reverse pass holds one pair of the factors' shape beside the stage
        at work however many stages feed it.

        Over ranks the pair is `GridTileRows` (`adjoint_rows`): the
        seeds are cut to this rank's tiles, and the quasiparticle and chi0
        sweeps return their adjoints in the same tiles, tile-major.

        release_tape: drop the forward's quasiparticle tape after reading it.
        False where more reverse passes follow off the same forward
        (`StateManifold.evaluate`), which releases it once at the end; each
        of them then reads proj(tau) and the slices instead of sweeping them.
        `shared` is the forward the seeds are of (`SharedForward`); the
        static correction it read is reused.

        h_bar: the adjoint on the quasiparticle Hamiltonian's off-diagonal
        elements (`one_body_seed`), None on the diagonal route.
        """
        st = self._fold_seeds(shared, eqp_bar, x_bar, d_bar, h_bar)
        self._fold_quasiparticle(shared, st, eqp_bar, release_tape)
        self._fold_screening(shared, st, w_bar)
        self._fold_reaction_field(shared, st, eqp_bar)
        return self._fold_integrals(shared, st)

    def _fold_seeds(self, shared, eqp_bar, x_bar, d_bar, h_bar):
        """The reverse pass's state: the seeds cut to tiles, the shared
        correction and the quasiparticle Hamiltonian's backward."""
        x_mo, d, eps, d_bare = shared.x_mo, shared.d, shared.eps, shared.d_bare
        # the conventions in the indices the forward read them in
        if shared.resolved is not None:
            self.chain.resolved_view = shared.resolved
        xc = shared.xc_correction
        block = self.adjoint_rows(x_mo)
        if block is not None:
            x_bar, d_bar = self.as_rows(x_bar, block), self.as_rows(d_bar,
                                                                      block)
        d_sigma = d if d_bare is None else d_bare
        d_bar_bare = (None if d_bare is None
                      else self.chain.zero_adjoint(d_bare, block))
        if d_bar_bare is not None and block is not None:
            d_bar_bare = self.as_rows(d_bar_bare, block)
        eps_bar = np.zeros_like(eps)
        mask = self.chain.quasiparticles.outside_window(len(eps))
        lam = None
        qp_h = shared.qp_h
        # a near-degenerate block's shared correction: the mean seed on
        # its roots, the remainder straight onto its eigenvalues
        seed_set, eps_direct = self.chain.quasiparticles.block_seeds(eqp_bar)
        eps_bar += eps_direct
        pairs = ()
        if qp_h is not None:
            # a group's members are H's diagonal, not their own roots
            grouped, eps_h, pairs, lam = self.chain.quasiparticles.qp_hamiltonian_backward(
                qp_h, h_bar, eqp_bar, eps)
            seed_set[grouped] = 0.0
            eps_bar += eps_h
        return ReverseState(
            xc=xc, block=block, x_bar=x_bar, d_bar=d_bar, d_sigma=d_sigma,
            d_bar_bare=d_bar_bare, eps_bar=eps_bar, mask=mask, lam=lam,
            qp_h=qp_h, seed_set=seed_set, pairs=pairs, h_bar=h_bar)

    def _fold_quasiparticle(self, shared, st, eqp_bar, release_tape):
        """The set's proj(tau) sweep and the seed outside it, added into the
        adjoints."""
        mf, x_mo, eps, mu = shared.mf, shared.x_mo, shared.eps, shared.mu
        d_bare, shift, qp_tape = shared.d_bare, shared.shift, shared.qp_tape
        # the whole set folds through ONE proj(tau) sweep
        with self.chain.phase('t_qp_backward'):
            if st.xc is None:
                st.xc = self.chain.xc_correction(mf, self.chain.resolved.qp_set, shift)
            (_, e_qp, x_qp, d_qp), st.qp_out = self.chain.quasiparticles.qp_set_solve(
                x_mo, st.d_sigma, eps, mu, st.xc, st.seed_set,
                tape=qp_tape, rows_block=st.block, pairs=st.pairs)
        # read once: its proj(tau) rows and slices are not held through
        # the rest of the reverse pass unless another reverse follows
        held = (qp_tape,) if release_tape else ()
        for tape in held + (st.qp_out.pop('tape', None),):
            if tape is not None:
                tape.release()
        st.eps_bar += e_qp
        # Outside the set eps^QP_p is eps_p, or eps_p plus a FROZEN
        # tier scissor: a constant either way, so the seed passes straight
        # to the mean-field eigenvalue and the shift adds no term of its own.
        st.eps_bar[st.mask] += eqp_bar[st.mask]
        # a curve's Delta(eps_p) moves with eps_p: its slope rides too
        for idx, _, slope in self.chain.quasiparticles.curve_at(eps):
            st.eps_bar[idx] += eqp_bar[idx] * slope
        st.x_bar += x_qp
        # Sigma screened with the BARE factor, so its adjoint lands there
        if d_bare is None:
            st.d_bar += d_qp
        else:
            st.d_bar_bare += d_qp
        del x_qp, d_qp

    def _fold_screening(self, shared, st, w_bar):
        """The static screening's adjoint through chi0, and a split
        continuum's T W_s T^T.

        In a continuum the kernel's W is the dressed screening Eq. (18)
        reads as well, on the same factors and grid: its chi0 adjoint is
        handed to the reaction field (`st.chi0_bar`) and swept there with
        Eq. (18)'s own, one sweep for both."""
        x_mo, d, eps, w_aux = shared.x_mo, shared.d, shared.eps, shared.w_aux
        shift, split = shared.shift, shared.split
        d_w, d_bar_w, t_bar = d, st.d_bar, None
        if split is not None:
            # W' = T W_s T^T: W_s's adjoint, and T's for the two roots
            d_w = split.d
            d_bar_w = self.chain.zero_adjoint(d_w, st.block)
            if st.block is not None:
                d_bar_w = self.as_rows(d_bar_w, st.block)
            t_bar = (w_bar + w_bar.T) @ split.t @ split.w
            w_aux, w_bar = split.w, split.t.T @ w_bar @ split.t
        with self.chain.phase('t_screening_backward'):
            chi0_bar = screening_backward(w_aux, w_bar)
            if shift is not None:
                st.chi0_bar = chi0_bar
            elif st.block is None:
                e2, x2, d2 = chi0_backward(chi0_bar[None, :, :], x_mo, d_w,
                                           eps, self.chain.nocc, self.chain.w_grid)
            else:
                e2, x2, d2 = chi0_backward_rows(chi0_bar[None, :, :], x_mo,
                                                d_w, eps, self.chain.nocc,
                                                self.chain.w_grid, block=st.block)
        if shift is None:
            st.eps_bar = st.eps_bar + e2
            st.x_bar += x2
            d_bar_w += d2
            del x2, d2
        st.split, st.d_w, st.d_bar_w, st.t_bar = split, d_w, d_bar_w, t_bar

    def _fold_reaction_field(self, shared, st, eqp_bar):
        """The reaction field's adjoint and the static weights each
        orbital's correction rides at."""
        x_mo, eps, d_bare = shared.x_mo, shared.eps, shared.d_bare
        shift, screening = shared.shift, shared.screening
        d_w, qp_h, lam = st.d_w, st.qp_h, st.lam
        w_corr = np.zeros(len(eps))
        # Z PER STATE, as on the single-quasiparticle path: each orbital in
        # the set has its own pole strength and its own correction, so one
        # shared factor would be as wrong as none.
        w_corr[self.chain.resolved.qp_set] = st.qp_out['z'] * st.seed_set
        if lam is not None:
            # inside a group the static term is e0's, at Lambda's weight
            for g in qp_h.groups:
                w_corr[list(g.orbitals)] = np.diag(lam)[list(g.orbitals)]
        # Sigma^env is added after every G0W0 solve, whole
        # (`add_reaction_field`): no Newton equation renormalizes it, so it
        # rides at the bare adjoint everywhere -- the set's own seed (a
        # degenerate block's mean), H's on a group, and eqp_bar outside the
        # window, where `env_static_outside` put it on the BSE diagonal.
        w_env = np.zeros(len(eps))
        w_env[self.chain.resolved.qp_set] = st.seed_set
        for g in (qp_h.groups if qp_h is not None else ()):
            w_env[list(g.orbitals)] = eqp_bar[list(g.orbitals)]
        w_env[st.mask] = eqp_bar[st.mask]
        if shift is not None:
            with self.chain.phase('t_reaction_field_backward'):
                st.eps_bar = added_reaction_field_adjoint(
                    st.eps_bar, st.x_bar, st.d_bar_w, st.d_bar_bare,
                    reaction_field_backward(
                        w_env, x_mo, d_w, d_bare, eps, self.chain.nocc,
                        grid=self.chain.w_grid, screening=screening,
                        block=st.block, dressed_chi0_bar=st.chi0_bar))
            if lam is not None and st.h_bar is not None:
                # the quasiparticle Hamiltonian's <p|Sigma^env|q>, added to
                # the solved H, at the seed H_bar itself (zero without one)
                with self.chain.phase('t_reaction_field_backward'):
                    st.eps_bar = added_reaction_field_adjoint(
                        st.eps_bar, st.x_bar, st.d_bar_w, st.d_bar_bare,
                        offdiagonal_reaction_field_backward(
                            [2.0 * st.h_bar[p, q] for _, _, p, q in qp_h.pairs],
                            [(p, q) for _, _, p, q in qp_h.pairs], x_mo, d_w,
                            d_bare, eps, self.chain.nocc, grid=self.chain.w_grid,
                            screening=screening, block=st.block))
        st.w_corr = w_corr

    def _fold_integrals(self, shared, st):
        """The integrals' and the fit's adjoint into the nuclei, with the
        gauge adjoints of a split continuum."""
        mol, mf, auxmol, crd = shared.mol, shared.mf, shared.auxmol, shared.crd
        x_mo = shared.x_mo
        split, lam, t_bar = st.split, st.lam, st.t_bar
        root_bar, gauges = None, ()
        if split is None:
            d_bar = st.d_bar_w
        else:
            d_bar = st.d_bar
            # T = R_inf^-1 R_s: each root's adjoint joins its own gauge
            root_bar = -np.linalg.solve(split.root_inf.T, t_bar @ split.t.T)
            gauges = (GaugeAdjoint(st.d_bar_w, split.environment,
                                   root_bar=np.linalg.solve(split.root_inf.T,
                                                            t_bar)),)
        # the Sigma_x - v_xc skeleton's fit adjoint rides the assembly's
        with self.chain.one_fit_adjoint(mf):
            # the Hamiltonian's off-diagonal static term rides at its own
            # weight beside the Z-renormalized diagonal
            static = st.w_corr
            if lam is not None and self.chain.is_ks:
                static = np.diag(st.w_corr) + lam - np.diag(np.diag(lam))
            y_xc, g_xc = self.chain.xc_chain(mf, static)
            grad, diags = self.chain.nuclear_gradient(mol, mf, auxmol, crd, x_mo,
                                                st.eps_bar, st.x_bar, d_bar,
                                                y_extra=y_xc, g_extra=g_xc,
                                                d_bar_bare=st.d_bar_bare,
                                                root_bar=root_bar,
                                                gauges=gauges)
        return grad, dict(diags, **self.chain.conventions.outside_record())

    def root_gradient(self, shared, solution, root, release_tape=True):
        """(dOmega_root/dR, diagnostics) off one forward and its Casida
        solution."""
        with self.chain.phase('t_bse_backward'):
            seeds = self.chain.follower.casida_seeds(shared, solution, root)
            h_bar = self.chain.quasiparticles.one_body_seed(shared, solution, root)
        grad, diags = self.fold_to_nuclei(shared, *seeds,
                                          release_tape=release_tape,
                                          h_bar=h_bar)
        om = solution.omega
        return grad, dict(diags, omega=float(om[root]), root=int(root))

    def interstate_gradient_of(self, shared, solution, m, n,
                               release_tape=True):
        """(d<m|H|n>/dR, diagnostics) off one forward and its Casida
        solution."""
        with self.chain.phase('t_bse_backward'):
            seeds = self.chain.follower.casida_seeds(shared, solution, n, m)
            h_bar = self.chain.quasiparticles.one_body_seed(shared, solution, n, m)
        grad, diags = self.fold_to_nuclei(shared, *seeds,
                                          release_tape=release_tape,
                                          h_bar=h_bar)
        om = solution.omega
        return grad, dict(diags, omega_m=float(om[m]), omega_n=float(om[n]),
                          gap=float(om[n] - om[m]))
