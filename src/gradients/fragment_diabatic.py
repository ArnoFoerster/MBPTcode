"""Analytic nuclear gradients of a fragment-diabatic matrix A_eff(Omega_0), for
any method behind the one `FragmentMethod` protocol: the Lagrangian of an
element, shared; the method's own response behind `SinglesResponse`.

The diabats, the effective matrix and the resolvent vectors are
`src.fragments.partition`; the finite-difference reference this module
reproduces is `tests/fragment_diabatic_fd.py`. Omega_0 is held fixed. The
implementations: GW-BSE (`src.gradients.gwbse.fragment_diabatic`,
Tamm-Dancoff or full) and ADC(2) (`src.gradients.adc.fragment_adc2`, singlets
or triplets).

ONE PENCIL. Everything is written for the symmetric pencil K v = Omega S v of
the partition -- S = 1 for a Hermitian singles operator (Tamm-Dancoff BSE,
ADC(2) folded at Omega_0); v = (X, Y), K = [[A, B], [B, A]],
S = diag(1, -1) for the full BSE -- so one code serves every method.

WHAT MOVES. An element E_ab = [A_eff(Omega_0)]_ab depends on the geometry
through the canonical singles matrix K, through the fragment-localized
orbitals that K is rotated into (K_loc = T^T K T, the same T on the X and the
Y half), and through the diabats, which are eigenvectors of blocks of K_loc.
Four terms, none optional:

1. AMPLITUDES (the method's reverse pass). At fixed diabats and fixed local
   orbitals, dE_ab = u_a^T dK_loc u_b with u = p + y (`src.fragments.partition`).
   Every first derivative is a contraction of dK_loc with a few vector pairs
   (s, t), each mapped to the canonical basis by T and handed to the method
   (`SinglesResponse.pair_force`), which returns sum w s^T dK t with the
   orbitals moving in ITS gauge (below).

2. DIABAT RESPONSE. A diabat p_c is the lowest eigenvector of its block B_c of
   the pencil, so it turns as B_c does: dp_c = -(B_c - E_c S)^+ dB_c p_c,
   normalized in the metric. Moving P (and with it Q, the complement
   orthogonal in the metric) is the same as transforming K by a transformation
   that preserves S; carried through the Schur complement, sum_ab W_ab dE_ab
   gains sum_c g_c^T dp_c with

       g_c = K u V e_c - S y V E e_c ,   V = W + W^T ,

   E the matrix differentiated (y = 0 and E = K_PP undressed). Rotations of
   diabats among themselves -- two states of one block -- are included
   exactly. The block solve turns g_c into z_c, and the term is
   p_c^T dK_loc z_c: ONE linear solve in the block per diabat and another
   vector pair for the method. K is K(Omega_0) for an energy-dependent
   operator, held at Omega_0 like everything else.

3. GAUGE. K_loc is K rotated by T, and T = U_o (x) U_v moves: with
   G = U^T dU the generator of the canonical-to-local rotation, every pair
   (s, t) adds sum G (.) D, D built from s, t, K s and K t. G is the local
   orbitals' derivative overlap less that of the frame the method's reverse
   pass moved the canonical orbitals in: the canonical gauge for GW-BSE,
   whose chain keeps the quasiparticle block diagonal
   (`orbital_coupling.configuration_coupling`), the symmetric gauge for
   ADC(2), whose correlation force takes the Fock derivative in a
   non-canonical basis (`orbital_coupling.symmetric_gauge_coupling`). The
   method supplies this term (`SinglesResponse.frame_force`).

4. LOCALIZATION. The local orbitals' own derivative follows from the
   stationarity of the fragment Pipek-Mezey functional. Linearizing it gives
   H theta = -s'(rest), with H its rotation Hessian; the Lagrangian z solves
   H^T z = D once, and what theta would have contributed becomes the pairing
   of z with the non-rotational part of the orbital motion: the
   occupied-virtual rotation (the coupled-perturbed response, one more
   Z-vector through the ground-state-coupling machinery), the moving basis
   functions, and the Loewdin populations' S^1/2. All of these are closed
   form and the mean field's alone. The derivative overlap
   <phi_p | d phi_q> is exactly antisymmetric for orthonormal orbitals, so no
   separate metric term appears; the overlap derivative enters only through
   the moving basis and through S^1/2.

Only Loewdin populations are differentiated here (`scheme='lowdin'`); the IAO
populations are a finite-difference cross-check (`fragments.localization`).

THE PROTOCOLS. A `FragmentMethod` is what an element's gradient reads of a
method, and all it reads:

    mol0                                the reference geometry
    require_differentiable_environment()  refuses an environment it cannot
                                        differentiate
    mean_field(mol=None, mf=None)       (mol, mf) at a geometry
    singles(mol, mf, route, omega0)     (SinglesOperator, SinglesResponse) at
                                        that geometry, route 'auto' | 'dense'
                                        | a matrix-free route of the method
    ground_energy(mf), ground_gradient(mf)   the ground state E_0 and its
                                        nuclear gradient (a diabat surface)
    refreeze(mol, mf=None)              the method refrozen at another
                                        reference

and its `SinglesResponse`:

    pair_force(pairs)                   (natm, 3) sum w s^T dK t at fixed
                                        Omega_0, pairs [(v, same, w)], v the
                                        (dim, 2) canonical [s, t], same True
                                        when s = t
    frame_force(dc_o, dc_v)             (natm, 3) -sum D (.) A of the frame
                                        pair_force moves the orbitals in, D the
                                        within-block rotation density in the
                                        canonical basis (term 3)
    flags                               {name: detail} of the numerical events
                                        of its last pair_force (the policy:
                                        refuse inputs, warn numerics)

SIZE. Nothing is dense beyond `partition.dense_limit` rows (FRAGMENT_DENSE_MAX
with the matrix at hand, FRAGMENT_MATRIX_FREE_DENSE_MAX matrix-free, where
forming a block costs one whole-system action per row): the diabats come from
a Davidson (`partition`), the resolvent from conjugate gradients, the diabat
response from a projected MINRES in its block, and the localization response
from MINRES over the rotations BETWEEN fragments, where the diabatic
quantities live; inside one fragment the functional is nearly flat, those
rotations decouple, and the full residual is checked. Each element costs one
reverse pass of the method plus a few operator actions.

OVER MPI RANKS the module is a replicated driver: every rank runs it, and the
operator action and the method's reverse pass are divided over the ranks. A
serial step whose threaded arithmetic need not agree to the bit takes rank
0's result: the localization, its response, the block Davidsons and the
diabat responses run on rank 0 alone, the other ranks serving its operator
actions (`krylov.root_driven_solve`); the guard's Newton and the
conjugate-gradient resolvent run on every rank on rank 0's slope, step and
stopping mask (`lockstep`). Every rank thus enters the same collectives
equally often and fails or succeeds together
(tests/test_fragment_diabatic_ranks.py).

WHAT IS NOT HERE. A diabat block with several states whose energies come close
(the eigenvector response divides by their gap and is refused below
`DIABAT_GAP_MIN`); a derivative with respect to Omega_0, which is held fixed
by construction; any environment beyond what the method itself
differentiates.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, minres

from src.Base.numerics.constants import (DIABAT_GAP_MIN, DIABAT_RESPONSE_PRECONDITIONER_FLOOR,
                                         FRAGMENT_DECOUPLING_RESIDUAL_FACTOR,
                                         FRAGMENT_LOCALIZATION_FULL_MAX_ITER,
                                         FRAGMENT_LOCALIZATION_MINRES_MAX_ITER,
                                         FRAGMENT_RESPONSE_ACCEPT_FACTOR,
                                         FRAGMENT_RESPONSE_MINRES_MAX_ITER, FRAGMENT_ROOT_MATCH_TOL,
                                         FRAGMENT_SOLVE_TOL)
from src.fragments.localization import (FragmentOrbitals,
                                        fragment_ao_indices)
from src.fragments.partition import (FragmentCanonical, FragmentPartition,
                                     dense_limit, flag_numerics, to_canonical,
                                     to_local)
from src.Base.utils.linear_algebra.krylov import root_driven_solve
from src.Base.utils.parallel.mpi_grid import lockstep
from src.Base.utils.timing import stage
from src.gradients.surface import PotentialEnergySurface
from src.gradients.orbital_coupling import ov_coupling, sigma_contraction



# ---------------------------------------------------------------- helpers

def _ket_derivative(mol, weight):
    """(natm, 3) sum_{mu nu} W_{mu nu} <mu | d nu / dR> at fixed coefficients:
    `sigma_contraction` in the AO basis itself."""
    return sigma_contraction(mol, np.eye(mol.nao), weight, False)


def _overlap_derivative(mol, w_s):
    """(natm, 3) sum W_{mu nu} dS_{mu nu}: dS = <d mu|nu> + <mu|d nu>."""
    return _ket_derivative(mol, w_s + w_s.T)


def _rotation_density(s, t, k_s, k_t, nocc, nvir):
    """(D_o, D_v): d(s^T K_loc t) = sum G_o (.) D_o + G_v (.) D_v under a
    rotation x -> G_o X + X G_v^T of the local amplitudes -- of both halves
    of a full-BSE vector, which rotate alike."""
    d_o, d_v = 0.0, 0.0
    for h in range(len(s) // (nocc * nvir)):
        sl = slice(h * nocc * nvir, (h + 1) * nocc * nvir)
        S, T = s[sl].reshape(nocc, nvir), t[sl].reshape(nocc, nvir)
        Ks, Kt = k_s[sl].reshape(nocc, nvir), k_t[sl].reshape(nocc, nvir)
        d_o = d_o + Kt @ S.T + Ks @ T.T
        d_v = d_v + Kt.T @ S + Ks.T @ T
    return d_o, d_v


# ------------------------------------------------- localization response

class _PMResponse:
    """The fragment-PM stationarity s(C; L) = R - R^T, R_ij = sum_K q_Ki M_K,ij,
    M_K = C^T L_K C, and the pieces of its linearization this module needs."""

    def __init__(self, mol, c_loc, fragments_ao):
        s = mol.intor_symmetric('int1e_ovlp')
        w, v = np.linalg.eigh(s)
        self.sw, self.sv = np.sqrt(w), v
        self.x = (v * self.sw) @ v.T                     # S^1/2
        self.s_inv = (v / w) @ v.T
        self.c = c_loc
        self.rows = fragments_ao
        self.l = []
        for r in fragments_ao:
            pi = np.zeros(len(s))
            pi[r] = 1.0
            self.l.append((self.x * pi[None, :]) @ self.x)
        self.m = [self.c.T @ lk @ self.c for lk in self.l]
        self.q = [np.diag(mk).copy() for mk in self.m]
        # the last solve's relative residual and verdict (`solve`)
        self.residual, self.converged = None, True

    def _b(self, z_anti):
        zz = z_anti - z_anti.T
        return [np.diag((zz * mk).sum(axis=1)) + zz * qk[:, None]
                for mk, qk in zip(self.m, self.q)]

    def c_bar(self, z_anti):
        """Ĉ with <z, s'(dC, 0)> = Tr[Ĉ^T dC]."""
        return sum(lk @ self.c @ (bk + bk.T)
                   for lk, bk in zip(self.l, self._b(z_anti)))

    def l_bar(self, z_anti):
        """L̄_K with <z, s'(0, dL)> = sum_K L̄_K (.) dL_K."""
        return [self.c @ bk @ self.c.T for bk in self._b(z_anti)]

    def hessian_t(self, z_anti):
        """H^T z = antisym(C^T Ĉ(z)), on antisymmetric z."""
        m = self.c.T @ self.c_bar(z_anti)
        return 0.5 * (m - m.T)

    def solve(self, d_anti, labels):
        """Antisymmetric z with H^T z = D (D antisymmetric, zero inside each
        fragment), by MINRES over the rotations BETWEEN fragments.

        H is the Hessian of the localization functional, symmetric, and nearly
        singular inside each fragment, where the functional is almost flat.
        The diabatic quantities live between fragments (D has no
        intra-fragment block), and inside-fragment rotations decouple from
        the between-fragment ones to the solver's precision, so the reduced
        system is solved and the FULL residual checked; if the decoupling
        fails, MINRES runs on the full system, which a singular but
        consistent symmetric system admits. Tolerance: FRAGMENT_SOLVE_TOL.
        """
        tol = FRAGMENT_SOLVE_TOL
        n = d_anti.shape[0]
        iu = np.triu_indices(n, 1)
        if len(iu[0]) == 0:
            return np.zeros((n, n))
        between = np.flatnonzero(labels[iu[0]] != labels[iu[1]])

        def unpack(v):
            z = np.zeros((n, n))
            z[iu] = v
            return z - z.T

        def full_mv(v):
            return self.hessian_t(unpack(v))[iu]
        rhs = d_anti[iu]
        scale = max(float(np.abs(rhs).max()), 1e-300)
        if between.size:
            def red_mv(v):
                full = np.zeros(len(iu[0]))
                full[between] = v
                return full_mv(full)[between]
            op = LinearOperator((between.size,) * 2, matvec=red_mv, dtype=float)
            sol_b, _ = minres(op, rhs[between], rtol=tol,
                              maxiter=FRAGMENT_LOCALIZATION_MINRES_MAX_ITER)
            sol = np.zeros(len(iu[0]))
            sol[between] = sol_b
            resid = float(np.abs(full_mv(sol) - rhs).max())
            self.residual = resid / scale
            if resid <= FRAGMENT_DECOUPLING_RESIDUAL_FACTOR * tol * scale:
                return unpack(sol)
        op = LinearOperator((len(iu[0]),) * 2, matvec=full_mv, dtype=float)
        sol, info = minres(op, rhs, rtol=tol,
                           maxiter=FRAGMENT_LOCALIZATION_FULL_MAX_ITER)
        resid = float(np.abs(full_mv(sol) - rhs).max())
        self.residual = resid / scale
        # a solve that stops short keeps its last iterate (`converged`)
        self.converged = resid <= FRAGMENT_RESPONSE_ACCEPT_FACTOR * tol * scale
        return unpack(sol)

    def s_bar_from_l_bar(self, l_bars):
        """dS-weight of sum_K L̄_K (.) dL_K, through L_K = X Pi_K X, X = S^1/2."""
        xbar = np.zeros_like(self.x)
        for lb, r in zip(l_bars, self.rows):
            lb = 0.5 * (lb + lb.T)
            pi = np.zeros(len(self.x))
            pi[r] = 1.0
            xbar += (lb @ self.x) * pi[None, :] + (pi[:, None] * self.x) @ lb
        # Sylvester X dX + dX X = dS: in S's eigenbasis dX'_ij = dS'_ij / (x_i + x_j)
        xb = self.sv.T @ (0.5 * (xbar + xbar.T)) @ self.sv
        sb = xb / (self.sw[:, None] + self.sw[None, :])
        return self.sv @ sb @ self.sv.T


def _localization_term(mol, mf, orbitals, d_occ, d_vir, flags=None):
    """(natm, 3) sum D_anti (.) theta, theta the local orbitals' own rotation;
    a localization response that stops short keeps its last iterate and is
    flagged 'localization_response_not_converged' in `flags`."""
    nocc = orbitals.nocc
    c = np.asarray(mf.mo_coeff, float)
    c_o, c_v = c[:, :nocc], c[:, nocc:]
    rows = fragment_ao_indices(mol, orbitals.fragments)
    w_s_total = np.zeros((mol.nao, mol.nao))
    w_1_total = np.zeros((mol.nao, mol.nao))
    v_ai = np.zeros((c_v.shape[1], nocc))
    v_ia = np.zeros((nocc, c_v.shape[1]))
    for tag, d, c_loc, u, labels in (
            ('occ', d_occ, orbitals.c_occ, orbitals.u_occ, orbitals.occ_labels),
            ('vir', d_vir, orbitals.c_vir, orbitals.u_vir, orbitals.vir_labels)):
        d_anti = 0.5 * (d - d.T)
        # zero inside each fragment by the subspace invariance; what an
        # iterative diabat leaves there is solver noise, not signal
        d_anti = np.where(labels[:, None] == labels[None, :], 0.0, d_anti)
        if not np.any(d_anti):
            continue
        pm = _PMResponse(mol, c_loc, rows)
        # MINRES on rank 0 alone, with its stop (`root_driven_solve`); its
        # solution and verdict are rank 0's on every rank
        z, converged, residual = root_driven_solve(
            lambda act: (pm.solve(d_anti, labels), pm.converged, pm.residual))
        if not converged:
            flag_numerics(flags, 'localization_response_not_converged',
                          f'the {tag} localization response did not converge '
                          f'(relative residual {residual:.2e}); its last '
                          f'iterate is kept', space=tag, residual=residual)
        cbar = pm.c_bar(z)
        # (a) occupied-virtual mixing of this space
        if tag == 'occ':
            v_ai += c_v.T @ cbar @ u.T
        else:
            v_ia += c_o.T @ cbar @ u.T
        # (b) moving basis: the coefficients' own derivative is
        # C_all (A - C_all^T S^(1) C_loc), with A = <phi|d phi> exactly
        # antisymmetric (the orbitals stay orthonormal), so A's occupied block
        # is the rotation theta and carries no metric part of its own
        # -S^-1 S^(1) C_loc
        w_1_total += -pm.s_inv @ cbar @ c_loc.T
        # (c) Loewdin populations
        w_s_total += pm.s_bar_from_l_bar(pm.l_bar(z))
    # sum V_ai A_ai + V'_ia A_ia, and A_ai = -A_ia: the orbitals are
    # orthonormal at every geometry, so <phi_a|dphi_i> + <phi_i|dphi_a> = 0
    out = ov_coupling(mol, mf, nocc, v_ia - v_ai.T)
    out = out + _overlap_derivative(mol, w_s_total) + _ket_derivative(mol, w_1_total)
    return -out




# ---------------------------------------------------------- the gradient

class DiabaticGradient:
    """Analytic gradients of A_eff(Omega_0) elements at one geometry.

    `method` is a `FragmentMethod` (module docstring): GW-BSE's
    `BSEFragmentMethod(chain)`, ADC(2)'s `ADC2FragmentMethod`. fragments,
    sites, ct and omega0 as for `partition.FragmentPartition.build`.
    `localization='response'` (the default) includes term 4; 'frozen' leaves
    it out, which is exact only for quantities that do not depend on the
    localization (a root).

    `dressed=False` differentiates the bare diabatic matrix A_PP instead of
    A_eff(Omega_0): the same machinery with u = p. `reference_orbitals`, a
    FragmentOrbitals at a nearby geometry, starts the localization there so a
    walk stays on one local maximum. `checkpoint`, a
    `fragments.checkpoint.PartitionCheckpoint`, restores the partition's long
    solves from files of the same problem.
    """

    def __init__(self, method, fragments, sites, ct=None, omega0=None,
                 mol=None, mf=None, route='auto', localization='response',
                 dressed=True, reference_orbitals=None, progress=None,
                 checkpoint=None):
        if localization not in ('response', 'frozen'):
            raise ValueError(f"localization must be 'response' or 'frozen', "
                             f"got {localization!r}")
        method.require_differentiable_environment()
        self.method, self.localization = method, localization
        self.dressed = bool(dressed)
        # wall seconds per stage, summed over calls
        self.timings = {}
        # the numerical flags of its response solves (`flag_numerics`)
        self.flags = {}
        # `progress`, a callable taking one string, hears each stage begin
        self.progress = progress
        with self._timed('mean_field'):
            self.mol, self.mf = method.mean_field(mol, mf)
        with self._timed('kernel'):
            self.operator, self.response = method.singles(
                self.mol, self.mf, route=route, omega0=omega0)
        with self._timed('localization'):
            self.orbitals = FragmentOrbitals.from_mf(
                self.mf, fragments, reference=reference_orbitals)
        with self._timed('partition'):
            self.partition = FragmentPartition.build(
                self.operator, self.orbitals, sites, ct, omega0=omega0,
                progress=progress, checkpoint=checkpoint)

    def _timed(self, key):
        """Report `key` to `progress` and time the block into `timings`."""
        if getattr(self, 'progress', None) is not None:
            self.progress(key)
        return stage(self.timings, key)

    # -- local-basis K action
    def _a(self, x):
        o = self.orbitals
        return to_local(o, self.operator.apply(to_canonical(o, x)))

    def _eigvec_response(self, c, g):
        """z_c with p_c^T dK z_c = g^T dp_c, the diabat-response term.

        With L = B_c - E_c S on diabat c's block, null direction p_c, and the
        projector Pi_c = 1 - p_c (S p_c)^T off p_c along the metric:
        z = -Pi_c L^+ Pi_c^T g. Solved densely below `dense_limit` rows,
        by MINRES above (L is symmetric; for the lowest state of a block it is
        positive semidefinite with p_c as its only null direction, and higher
        states of a block are indefinite, which MINRES takes as well).
        """
        part = self.partition
        if part.gaps[c] < DIABAT_GAP_MIN:
            raise ValueError(f'diabat {part.labels[c]} is within '
                             f'{part.gaps[c]:.2e} Ha of another state of its '
                             f'block; its response is not defined')
        p = part.p_local[:, c]
        rows = part.block_rows[c]
        pc = p[rows]
        sig = part.metric[rows]
        spc = sig * pc
        e_c = float(part.a_pp[c, c])
        gb = g[rows] - spc * (pc @ g[rows])             # Pi_c^T g
        dim = rows.size
        b = np.zeros(len(p))

        def project(x):                                 # Pi_c
            return x - np.outer(pc, spc @ x)

        def project_t(x):                               # Pi_c^T
            return x - np.outer(spc, pc @ x)

        def block(x):
            full = np.zeros((len(p), x.shape[1]))
            full[rows] = x
            return self._a(full)[rows]
        if dim <= dense_limit(self.operator):
            m = block(np.eye(dim))
            m = 0.5 * (m + m.T) - e_c * np.diag(sig)
            lhs = project_t(project_t(m.T).T) + np.outer(pc, pc)
            b[rows] = project(-np.linalg.solve(0.5 * (lhs + lhs.T),
                                               gb)[:, None])[:, 0]
            return b

        # Preconditioner: the exact quasiparticle diagonal of the
        # fragment-canonical basis (`FragmentCanonical`), rotated back; the
        # block maps onto itself, and R D^-1 R^T stays positive definite, as
        # MINRES needs.
        work = FragmentCanonical(self.orbitals, self.operator.eps)
        d_ov = work.diagonal()
        diag = np.tile(d_ov, len(p) // len(d_ov))[rows] - e_c * sig
        prec = 1.0 / np.maximum(np.abs(diag), DIABAT_RESPONSE_PRECONDITIONER_FLOOR)

        def rotate(x, to):
            full = np.zeros((len(p), x.shape[1]))
            full[rows] = x
            return to(full)[rows]

        def apply_prec(x):
            x = project_t(np.reshape(x, (dim, 1)))
            x = rotate(rotate(x, work.from_local) * prec[:, None],
                       work.to_local)
            return project(x)[:, 0]

        def solve(act):
            # rank 0 alone iterates (`root_driven_solve`): MINRES stops on
            # its own arithmetic, and the block action is collective
            def mv(x):
                x = project(np.reshape(x, (dim, 1)))
                return project_t(act(x) - e_c * sig[:, None] * x)[:, 0]
            op = LinearOperator((dim, dim), matvec=mv, dtype=float)
            pre = LinearOperator((dim, dim), dtype=float, matvec=apply_prec)
            sol, info = minres(op, -gb, M=pre, rtol=FRAGMENT_SOLVE_TOL,
                               maxiter=FRAGMENT_RESPONSE_MINRES_MAX_ITER)
            resid = np.abs(mv(sol) + gb).max()
            short = resid > (FRAGMENT_RESPONSE_ACCEPT_FACTOR * FRAGMENT_SOLVE_TOL
                             * max(np.abs(gb).max(), 1e-300))
            return sol, (float(resid) if short else None)
        # a response that stops short keeps its last iterate; the solution and
        # the verdict are rank 0's on every rank
        sol, residual = root_driven_solve(solve, block, dim)
        if residual is not None:
            flag_numerics(self.flags, 'diabat_response_not_converged',
                          f'the diabat response for {part.labels[c]} did not '
                          f'converge (residual {residual:.2e}); its last '
                          f'iterate is kept', diabat=part.labels[c],
                          residual=residual)
        b[rows] = project(sol[:, None])[:, 0]
        return lockstep(b, check=True)

    def _pairs(self, w):
        """[(s, t, weight)] with sum_ab W_ab dE_ab = sum weight * s^T dK_loc t.

        The amplitude part sum_ab W_ab u_a^T dK u_b is n_p pairs (u_a, U W_a);
        each diabat adds one more, (p_c, z_c), from its eigenvector response
        with the weight g_c the module docstring derives.
        """
        part = self.partition
        w = np.asarray(w, float)
        u = part.u_local if self.dressed else part.p_local
        sy = part.metric[:, None] * (part.y_local if self.dressed
                                     else np.zeros_like(u))
        ku = self._a(u)
        v = w + w.T
        g = ku @ v - sy @ (v @ self.matrix())          # column c is g_c
        pairs = []
        for a in range(u.shape[1]):
            if np.any(w[a]):
                pairs.append((u[:, a], u @ w[a], 1.0))
        for c in range(u.shape[1]):
            if not np.any(g[:, c]):
                continue
            z = self._eigvec_response(c, g[:, c])
            if np.any(z):
                pairs.append((part.p_local[:, c], z, 1.0))
        return pairs

    def _chain_term(self, pairs):
        """Term 1 and the diabat responses' amplitude part: the method's
        reverse pass (`SinglesResponse.pair_force`) for every pair, each
        handed over in the canonical basis."""
        o = self.orbitals
        return self.response.pair_force(
            [(to_canonical(o, np.column_stack([s, t])), np.allclose(s, t), wgt)
             for s, t, wgt in pairs])

    def _rotation_terms(self, pairs):
        o = self.orbitals
        nocc = o.nocc
        d_o = np.zeros((nocc, nocc))
        d_v = np.zeros((o.nvir, o.nvir))
        # one block action for every vector of every pair
        with self._timed('rotation_actions'):
            ks = self._a(np.column_stack([v for s, t, _ in pairs
                                          for v in (s, t)]))
        for k, (s, t, wgt) in enumerate(pairs):
            do, dv = _rotation_density(s, t, ks[:, 2 * k], ks[:, 2 * k + 1],
                                       nocc, o.nvir)
            d_o += wgt * do
            d_v += wgt * dv
        d_o, d_v = lockstep((np.ascontiguousarray(0.5 * (d_o - d_o.T)),
                             np.ascontiguousarray(0.5 * (d_v - d_v.T))),
                            check=True)
        # the method's frame: - sum (U D U^T) (.) A of the gauge its reverse
        # pass moved the canonical orbitals in (`SinglesResponse.frame_force`)
        dc_o = o.u_occ @ d_o @ o.u_occ.T
        dc_v = o.u_vir @ d_v @ o.u_vir.T
        with self._timed('canonical'):
            canon = self.response.frame_force(dc_o, dc_v)
        if self.localization == 'frozen':
            return canon, np.zeros_like(canon)
        with self._timed('localization_response'):
            loc = _localization_term(self.mol, self.mf, o, d_o, d_v, self.flags)
        return canon, loc

    def gradient(self, w):
        """(sum_ab W_ab dE_ab/dR (natm, 3), diagnostics) in Hartree/Bohr."""
        with self._timed('pairs'):
            pairs = self._pairs(w)
        with self._timed('chain'):
            chain = self._chain_term(pairs)
        canon, loc = self._rotation_terms(pairs)
        # The three terms are each rank's own threaded arithmetic after the
        # last lockstep, so their bits can differ between ranks; every rank
        # returns rank 0's.
        total, chain, canon, loc = lockstep(
            tuple(np.ascontiguousarray(a, dtype=float)
                  for a in (chain + canon + loc, chain, canon, loc)),
            check=True)
        # the partition's solver flags (`FragmentPartition.guard_flags`), the
        # method's reverse pass's (`SinglesResponse.flags`) and this
        # gradient's own response flags
        return total, dict(chain=chain, canonical=canon, localization=loc,
                           flags={**self.partition.guard_flags,
                                  **self.response.flags, **self.flags})

    def matrix(self):
        """The matrix being differentiated: A_eff(Omega_0), or A_PP undressed."""
        part = self.partition
        return part.a_eff if self.dressed else part.a_pp

    def index(self, label):
        """Diabat index from its label ('site 0.0', 'ct 1->0.0') or an int."""
        if isinstance(label, (int, np.integer)):
            return int(label)
        return self.partition.labels.index(label)

    def element(self, a, b):
        """(dE_ab/dR (natm, 3), diagnostics) in Hartree/Bohr."""
        n = len(self.partition.labels)
        w = np.zeros((n, n))
        w[a, b] += 0.5
        w[b, a] += 0.5
        grad, diags = self.gradient(w)
        return grad, dict(diags, value=float(self.matrix()[a, b]),
                          labels=(self.partition.labels[a],
                                  self.partition.labels[b]))

    def root_gradient(self, guess):
        """(dOmega/dR, diagnostics) of the root of A_eff(Omega) c = Omega c
        nearest `guess`, which must equal Omega_0.

        dOmega = Z c^T dA_eff(Omega) c at the root, so the partition has to be
        built AT the root (omega0 = Omega). With a complete Q this is the
        supermolecular root's gradient -- the identity the tests check,
        and one the localization terms cannot change, because a root does not
        depend on how the orbitals are localized.
        """
        part = self.partition
        evals, evecs = np.linalg.eigh(part.a_eff)
        k = int(np.argmin(np.abs(evals - guess)))
        if abs(evals[k] - part.omega0) > FRAGMENT_ROOT_MATCH_TOL:
            raise ValueError(f'the partition is built at Omega_0 = '
                             f'{part.omega0:.10f}, not at the root '
                             f'{evals[k]:.10f}; rebuild it there')
        c = evecs[:, k]
        z = 1.0 / (1.0 - float(c @ part.dsigma @ c))
        grad, diags = self.gradient(z * np.outer(c, c))
        return grad, dict(diags, omega=float(evals[k]), z=z)


# ------------------------------------------------- surfaces and models

class DiabaticSurface(PotentialEnergySurface):
    """One diabat's total energy over the nuclei: E_0(R) + E_cc(R).

    E_0 is the method's ground state (`FragmentMethod.ground_energy`: the mean
    field for GW-BSE, HF + MP2 for ADC(2)) and E_cc the diabat's diagonal element
    of A_eff(Omega_0) (or of A_PP, `dressed=False`), with Omega_0 frozen at the
    reference geometry. It satisfies the `PotentialEnergySurface` protocol, so
    `optimize.relax` relaxes a diabat -- in particular a charge-transfer diabat
    near the crossing with a local excitation, where the adiabatic root is a
    mixture of both and its surface is neither's.

    Along a walk the localization starts from the previous geometry's local
    orbitals, so the diabat keeps its identity.
    """

    #: The ingredients this surface provides (`surface.INGREDIENT_DOCS`).
    INGREDIENTS = frozenset({'mean_field'})

    def __init__(self, method, fragments, sites, state, ct=None, omega0=None,
                 dressed=True, route='auto'):
        self.method, self.fragments, self.sites, self.ct = (method, fragments,
                                                            sites, ct)
        self.state, self.dressed, self.route = state, dressed, route
        self.mol0 = method.mol0
        self._last = None
        # None until the reference is built: the reference partition then
        # chooses Omega_0, frozen for every later geometry
        self.omega0 = omega0
        ref = self._gradient_object(None, None, omega0)
        self.omega0 = ref.partition.omega0

    def _gradient_object(self, mol, mf, omega0=None):
        g = DiabaticGradient(
            self.method, self.fragments, self.sites, self.ct,
            omega0=self.omega0 if omega0 is None else omega0, mol=mol, mf=mf,
            route=self.route, dressed=self.dressed,
            reference_orbitals=None if self._last is None else self._last)
        self._last = g.orbitals
        return g

    def mean_field(self, mol=None, mf=None):
        return self.method.mean_field(mol, mf)

    def total_energy(self, mol=None, mf=None):
        g = self._gradient_object(mol, mf)
        c = g.index(self.state)
        return float(self.method.ground_energy(g.mf) + g.matrix()[c, c])

    def total_gradient(self, mol=None, mf=None):
        g = self._gradient_object(mol, mf)
        c = g.index(self.state)
        grad, diags = g.element(c, c)
        grad = grad + self.method.ground_gradient(g.mf)
        e = float(self.method.ground_energy(g.mf) + g.matrix()[c, c])
        return grad, e, dict(diags, omega=float(g.matrix()[c, c]),
                             omega0=self.omega0)

    def refreeze(self, mol, mf=None):
        """The diabat on its method refrozen at `mol` (on `mf` where given)."""
        return DiabaticSurface(self.method.refreeze(mol, mf=mf), self.fragments,
                               self.sites, self.state, self.ct,
                               omega0=self.omega0, dressed=self.dressed,
                               route=self.route)

    def identity(self):
        """None: this surface freezes nothing that its forces could be told
        apart by (`surface_identity`)."""
        return None

    def label(self):
        kind = 'A_eff' if self.dressed else 'A_PP'
        return f'diabat {self.state} [{kind}, Omega_0 = {self.omega0:.6f} Ha]'
