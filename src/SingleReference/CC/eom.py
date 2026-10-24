"""EOM-CCSDT (EE/IP/EA) via generated sigma vectors, diagonalized
with pyscf's non-Hermitian Davidson (pyscf.lib.davidson_nosym1) -- the same
architecture pyscf's own eom_rccsd/eom_gccsd modules use (matvec + diagonal
preconditioner + davidson_nosym1), just built from CCSDT-level generated
equations instead of hand-derived CCSD ones. The generator,
src/CodeGeneration/generate_eom_ccsdt.py, documents the r/l amplitude storage
conventions and the bra/ket projector operator strings.

Hbar = e^{-T} H e^{T} still carries the Hamiltonian's absolute energy scale
(e.g. sigma1's leading `f(j,j)*r1(i)` term) regardless of whether the rank-0
component is in the manifold, so raw Davidson eigenvalues sit near E_cc, not
near zero -- `EOMCC.kernel` subtracts E_cc to return the physical excitation
energy / IP / EA.

r2/r3 (and l2/l3) are stored as redundant antisymmetric arrays (e.g. IP's
r2(a,i,j) = -r2(a,j,i)), the same convention T2/T3/L2/L3 already use
elsewhere in this package. Unlike the fixed-point T/Lambda iterations (whose
update formula is manifestly antisymmetric given an antisymmetric input),
Davidson's raw unit-vector guesses and its diagonal preconditioner have no
reason to respect that constraint on their own -- so every matvec here
explicitly projects both its input and output onto the antisymmetric
subspace (`_antisymmetrize`). Skipping this silently corrupts the lowest
eigenvalues: an early version of this module found a spuriously low,
highly-degenerate IP eigenvalue (0.081 Ha, 6-fold "degenerate") instead of
the correct 0.268 Ha, traced by dense-diagonalizing the raw (unprojected)
sigma matrix and comparing block-by-block against the determinant_space
oracle -- the antisymmetry-violating subspace was leaking into the low end
of the spectrum.

`determinant_space.py` (exact, exponential-scaling determinant enumeration)
is kept as a private, tests-only oracle to validate this generated/iterative
solver on small systems -- it is not part of the public API.
"""
import warnings
from itertools import permutations

import numpy as np
from pyscf import lib, scf

from . import amplitudes
from . import solver as _lambda_solver
from .integrals import build_spinorbital_integrals_from_mf, energy_denominators
from .generated_eom import ip_sigma, ea_sigma, ee_sigma
from .generated_eom import ee_density

_SIGMA_MODULE = {'ip': ip_sigma, 'ea': ea_sigma, 'ee': ee_sigma}
_RANKS = (1, 2, 3)


def _antisym_pair(x, axes):
    a, b = axes
    return 0.5 * (x - x.swapaxes(a, b))


def _antisym_triple(x, axes):
    def sign(p):
        p, s = list(p), 1
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if p[i] > p[j]:
                    s = -s
        return s
    out = np.zeros_like(x)
    base = list(range(x.ndim))
    for perm in permutations(range(3)):
        ax_map = base.copy()
        for k in range(3):
            ax_map[axes[k]] = axes[perm[k]]
        out = out + sign(perm) * x.transpose(ax_map)
    return out / 6.0


# which axis pairs/triples of each r{2,3}/l{2,3} block must be antisymmetric,
#
_ANTISYM_SPEC = {
    ('ip', 'r2'): [('pair', (1, 2))],
    ('ip', 'r3'): [('pair', (0, 1)), ('triple', (2, 3, 4))],
    ('ip', 'l2'): [('pair', (0, 1))],
    ('ip', 'l3'): [('triple', (0, 1, 2)), ('pair', (3, 4))],
    ('ea', 'r2'): [('pair', (0, 1))],
    ('ea', 'r3'): [('triple', (0, 1, 2)), ('pair', (3, 4))],
    ('ea', 'l2'): [('pair', (1, 2))],
    ('ea', 'l3'): [('pair', (0, 1)), ('triple', (2, 3, 4))],
    ('ee', 'r2'): [('pair', (0, 1)), ('pair', (2, 3))],
    ('ee', 'r3'): [('triple', (0, 1, 2)), ('triple', (3, 4, 5))],
    ('ee', 'l2'): [('pair', (0, 1)), ('pair', (2, 3))],
    ('ee', 'l3'): [('triple', (0, 1, 2)), ('triple', (3, 4, 5))],
}


def _antisymmetrize(sector, key, x):
    for kind, axes in _ANTISYM_SPEC.get((sector, key), ()):
        x = _antisym_pair(x, axes) if kind == 'pair' else _antisym_triple(x, axes)
    return x


def _biorthogonalize_degenerate(omega, r_vectors, l_vectors, tol=1e-6):
    """Rotate (independently Davidson-solved) right/left eigenvectors within
    each near-degenerate omega cluster so that <L_m|R_n> = delta_mn.

    Solving the right and left eigenproblems separately gives, within any
    degenerate subspace, an ARBITRARY (and generally uncorrelated) choice of
    basis on each side -- pairing states by sorted-omega order then silently
    breaks (near-zero or wildly cross-mixed overlaps): confirmed the hard way
    on He/def2-SVP (whose p-orbital excitations are exactly 3-fold
    degenerate), which produced an exact NaN (division by a zero overlap) in
    the naive per-state normalization.

    Fix: within each cluster of size k, form the k x k overlap matrix
    S[m,n] = <L_m|R_n> and SVD it, S = U @ diag(sigma) @ Vt. Rotating
    L' = L @ U, R' = R @ Vt.T gives L'^T R' = U^T S V = diag(sigma) exactly
    (U, V orthogonal) -- a mutually consistent biorthogonal pair per cluster,
    which a plain per-state scalar renormalization can never produce since it
    only ever rescales, never rotates, each vector individually."""
    omega = np.asarray(omega)
    order = np.argsort(omega)
    clusters, start = [], 0
    for i in range(1, len(order) + 1):
        if i == len(order) or omega[order[i]] - omega[order[start]] > tol:
            clusters.append(order[start:i])
            start = i

    r_new = [None] * len(omega)
    l_new = [None] * len(omega)
    for idx in clusters:
        R = np.stack([r_vectors[i] for i in idx], axis=1)   # (dim, k)
        L = np.stack([l_vectors[i] for i in idx], axis=1)
        S = L.T @ R
        U, sigma, Vt = np.linalg.svd(S)
        if np.min(np.abs(sigma)) < 1e-10:
            raise RuntimeError(f"degenerate EE cluster at omega={omega[idx[0]]:.6f} "
                               f"has a near-singular L/R overlap (sigma={sigma}) -- "
                               f"left/right Davidson solves did not resolve a "
                               f"consistent basis for this subspace")
        Rp = R @ Vt.T / sigma[np.newaxis, :]
        Lp = L @ U
        for k, i in enumerate(idx):
            r_new[i] = Rp[:, k]
            l_new[i] = Lp[:, k]
    return r_new, l_new


def _shapes(sector, no, nv):
    if sector == 'ip':
        return {'r1': (no,), 'r2': (nv, no, no), 'r3': (nv, nv, no, no, no),
                'l1': (no,), 'l2': (no, no, nv), 'l3': (no, no, no, nv, nv)}
    if sector == 'ea':
        return {'r1': (nv,), 'r2': (nv, nv, no), 'r3': (nv, nv, nv, no, no),
                'l1': (nv,), 'l2': (no, nv, nv), 'l3': (no, no, nv, nv, nv)}
    if sector == 'ee':
        return {'r1': (nv, no), 'r2': (nv, nv, no, no), 'r3': (nv, nv, nv, no, no, no),
                'l1': (no, nv), 'l2': (no, no, nv, nv), 'l3': (no, no, no, nv, nv, nv)}
    raise ValueError(f"unknown sector '{sector}'")


def _diagonal(sector, side, no, nv, eps):
    """Approximate (orbital-energy-sum) diagonal of the manifold-projected
    Hbar, for the Davidson preconditioner -- standard "bare" EOM-CC diagonal,
    exact enough to precondition, not meant to be the true Hbar diagonal."""
    eo, ev = eps[:no], eps[no:]
    n = np.newaxis
    if sector == 'ip':
        d1 = -eo
        d2 = ev[:, n, n] - eo[n, :, n] - eo[n, n, :]
        d3 = (ev[:, n, n, n, n] + ev[n, :, n, n, n]
              - eo[n, n, :, n, n] - eo[n, n, n, :, n] - eo[n, n, n, n, :])
        if side == 'l':
            d2 = d2.transpose(1, 2, 0)          # (occ,occ,vir)
            d3 = d3.transpose(2, 3, 4, 0, 1)     # (occ,occ,occ,vir,vir)
    elif sector == 'ea':
        d1 = ev
        d2 = ev[:, n, n] + ev[n, :, n] - eo[n, n, :]
        d3 = (ev[:, n, n, n, n] + ev[n, :, n, n, n] + ev[n, n, :, n, n]
              - eo[n, n, n, :, n] - eo[n, n, n, n, :])
        if side == 'l':
            d2 = d2.transpose(2, 0, 1)           # (occ,vir,vir)
            d3 = d3.transpose(3, 4, 0, 1, 2)     # (occ,occ,vir,vir,vir)
    elif sector == 'ee':
        d1 = ev[:, n] - eo[n, :]
        d2 = ev[:, n, n, n] + ev[n, :, n, n] - eo[n, n, :, n] - eo[n, n, n, :]
        d3 = (ev[:, n, n, n, n, n] + ev[n, :, n, n, n, n] + ev[n, n, :, n, n, n]
              - eo[n, n, n, :, n, n] - eo[n, n, n, n, :, n] - eo[n, n, n, n, n, :])
        if side == 'l':
            d1 = d1.transpose(1, 0)              # (occ,vir)
            d2 = d2.transpose(2, 3, 0, 1)         # (occ,occ,vir,vir)
            d3 = d3.transpose(3, 4, 5, 0, 1, 2)   # (occ,occ,occ,vir,vir,vir)
    else:
        raise ValueError(sector)
    return {1: d1, 2: d2, 3: d3}


class EOMCC:
    """Ground-state CCSDT amplitudes + per-sector Davidson EOM-CCSDT.

    mf : converged closed-shell spin-restricted pyscf mean-field.
    level : 'ccsdt' only (singles+doubles+triples manifold). EOM-CCSD is
        not implemented here: use pyscf (pyscf.cc.eom_rccsd / eom_gccsd).
    t_amps : optional (t1, t2, t3) to skip the amplitude solve.
    t_diis_size : DIIS subspace for the ground-state T solve (0 disables it and
        restores plain fixed-point iteration). DIIS holds 2*t_diis_size flat
        copies of (t1, t2, t3), which at CCSDT is 2*t_diis_size copies of t3 --
        lower it (or zero it) when the solve is memory-bound rather than
        iteration-bound; see amplitudes.kernel.
    """

    def __init__(self, mf, level='ccsdt', t_amps=None, t_stopping_eps=1e-10,
                 max_iter=200, verbose=False, t_diis_size=6):
        if level != 'ccsdt':
            raise ValueError(f"EOMCC supports level='ccsdt' only, not {level!r}; "
                             f"EOM-CCSD is pyscf's (pyscf.cc.eom_rccsd / eom_gccsd)")
        if isinstance(mf, scf.uhf.UHF):
            raise ValueError("EOMCC needs a closed-shell RHF mean field: its transition "
                             "densities and polarizability assume interleaved spin orbitals")
        self.level = level
        self.ranks = _RANKS
        self.ints = build_spinorbital_integrals_from_mf(mf)
        ints = self.ints
        self.nocc, self.nvir = ints['nocc'], ints['nvir']
        self.norb = self.nocc + self.nvir
        o, v = slice(None, self.nocc), slice(self.nocc, None)

        # convergence flags (numerical events are warned about and flagged);
        # True when the amplitudes were handed in
        self.t_converged = True
        self.lambda_converged = None
        if t_amps is not None:
            self.t1, self.t2, self.t3 = t_amps
        else:
            t_out = {}
            e_ai, e_abij, e_abcijk = energy_denominators(ints['fock'], self.nocc, self.nvir)
            t1 = np.zeros((self.nvir, self.nocc))
            t2 = np.zeros((self.nvir, self.nvir, self.nocc, self.nocc))
            t3 = np.zeros((self.nvir,) * 3 + (self.nocc,) * 3)
            t1, t2, t3 = amplitudes.kernel(
                t1, t2, t3, ints['fock'], ints['g'], o, v,
                e_ai, e_abij, e_abcijk, ints['hf_energy'],
                max_iter=max_iter, stopping_eps=t_stopping_eps,
                diis_size=t_diis_size, out=t_out)
            self.t1, self.t2, self.t3 = t1, t2, t3
            self.t_converged = t_out['converged']

        self.e_cc = amplitudes.cc_energy(self.t1, self.t2, ints['fock'], ints['g'], o, v)
        self.e_tot = self.e_cc + ints['nuclear_repulsion']
        self._lambda = None

    def lambda_amplitudes(self, l_stopping_eps=1e-9, max_iter=200, verbose=False):
        """Ground-state Lambda multipliers (lam1, lam2, lam3), solved once
        and cached -- needed for transition_densities()'s rho_n (which
        sandwiches the ground-state bra <(1+Lambda)|, not an EOM eigenvector).

        `lambda_converged` records whether the Lambda-CCSDT solve converged
        (it warns when it did not)."""
        if self._lambda is None:
            ints = self.ints
            o, v = slice(None, self.nocc), slice(self.nocc, None)
            e_ai, e_abij, e_abcijk = energy_denominators(ints['fock'], self.nocc, self.nvir)
            l_out = {}
            self._lambda = _lambda_solver.solve_lambda_ccsdt(
                self.t1, self.t2, self.t3, ints['fock'], ints['g'], o, v,
                e_ai, e_abij, e_abcijk, max_iter=max_iter,
                stopping_eps=l_stopping_eps, verbose=verbose, out=l_out)
            self.lambda_converged = l_out['converged']
        return self._lambda

    def _pack(self, d, side, no, nv):
        return np.concatenate([d[f'{side}{r}'].ravel() for r in self.ranks])

    def _unpack(self, vec, side, no, nv):
        shapes = _shapes(self._sector, no, nv)
        out, pos = {}, 0
        for r in self.ranks:
            key = f'{side}{r}'
            shp = shapes[key]
            size = int(np.prod(shp))
            out[key] = _antisymmetrize(self._sector, key, vec[pos:pos + size].reshape(shp).copy())
            pos += size
        for r in (1, 2, 3):
            key = f'{side}{r}'
            if key not in out:
                out[key] = np.zeros(shapes[key])
        return out

    def gen_matvec(self, sector, left=False):
        """Return (matvec, diag) for one sector/side, matching pyscf's own
        eom.gen_matvec(imds, left=left) convention (see eom_rccsd.kernel)."""
        self._sector = sector
        no, nv = self.nocc, self.nvir
        o, v = slice(None, no), slice(no, None)
        f, g, t1, t2, t3 = self.ints['fock'], self.ints['g'], self.t1, self.t2, self.t3
        mod = _SIGMA_MODULE[sector]
        side = 'l' if left else 'r'
        suffix = '_left' if left else ''
        sigma_fns = [getattr(mod, f'{sector}_sigma{r}{suffix}') for r in self.ranks]

        def matvec(vec):
            amps = self._unpack(vec, side, no, nv)
            args = (amps[f'{side}1'], amps[f'{side}2'], amps[f'{side}3'], t1, t2, t3, f, g, o, v)
            out = {}
            for r, fn in zip(self.ranks, sigma_fns):
                key = f'{side}{r}'
                out[key] = _antisymmetrize(sector, key, fn(*args))
            return self._pack(out, side, no, nv)

        eps = np.diagonal(f)
        dblocks = _diagonal(sector, side, no, nv, eps)
        diag = np.concatenate([dblocks[r].ravel() for r in self.ranks])
        return matvec, diag

    def kernel(self, sector='ee', nroots=4, left=False, guess=None,
              tol=1e-9, max_cycle=100, max_space=30, verbose=0):
        """Diagonalize sector's manifold-projected Hbar for the lowest
        `nroots` real eigenvalues via non-Hermitian Davidson.

        Hbar = e^{-T} H e^{T} still carries the Hamiltonian's absolute
        energy scale (e.g. sigma1's leading `f(j,j)*r1(i)` term) regardless
        of whether the rank-0/reference component is in the manifold, so raw
        eigenvalues sit near E_cc (~ -few to tens of Hartree), not near zero.
        Subtracting E_cc gives the physical quantity: excitation energy
        ('ee'), IP ('ip', E(N-1)-E(N) > 0 for a bound system), EA ('ea')."""
        no, nv = self.nocc, self.nvir
        matvec, diag = self.gen_matvec(sector, left=left)
        size = diag.size
        nroots = min(nroots, size)

        if guess is None:
            # raw unit vectors at arbitrary flat positions are generally NOT
            # antisymmetric (e.g. a unit vector at r2(a,i,j) with i>j isn't
            # the antisymmetric combination r2(a,i,j)=-r2(a,j,i)) -- project
            # each guess through the same _unpack/_pack round trip matvec
            # uses, so Davidson's trial space starts (and, since matvec is
            # itself antisymmetry-preserving, stays) inside the physical
            # subspace. Guard against an accidental zero vector (e.g. a
            # unit vector at r2(a,i,i), whose antisymmetric projection
            # vanishes identically) by scanning to the next-lowest diagonal
            # index instead.
            side = 'l' if left else 'r'
            order = np.argsort(diag)
            guess, oi = [], 0
            while len(guess) < nroots and oi < len(order):
                v = np.zeros(size)
                v[order[oi]] = 1.0
                oi += 1
                amps = self._unpack(v, side, no, nv)
                v = self._pack(amps, side, no, nv)
                if np.linalg.norm(v) > 1e-10:
                    guess.append(v / np.linalg.norm(v))

        def precond(r, e0, x0):
            return r / (e0 - diag + 1e-12)

        def aop(xs):
            return [matvec(x) for x in xs]

        conv, es, vs = lib.davidson_nosym1(aop, guess, precond, nroots=nroots,
                                           tol=tol, max_cycle=max_cycle,
                                           max_space=max_space, verbose=verbose)
        es = np.atleast_1d(np.asarray(es))
        if nroots == 1 and not isinstance(vs, list):
            vs = [vs]
            conv = np.atleast_1d(conv)

        max_imag = float(np.max(np.abs(es.imag))) if np.iscomplexobj(es) else 0.0
        if max_imag > EOM_IMAG_TOL:
            # a numerical event: warned about, not refused; the
            # result says so (`EOMResult.complex_eigenvalues`)
            warnings.warn(f'EOM-{sector} eigenvalues have |Im| up to '
                          f'{max_imag:.2e}; their real parts are returned',
                          ComplexEOMEigenvalues, stacklevel=2)

        order = np.argsort(es.real)
        omega = es.real[order] - self.e_cc
        vectors = [np.asarray(vs[i]).real for i in order]
        conv = np.asarray(conv)[order]
        if not np.all(conv):
            # davidson_nosym1's tol is an eigenvalue change, not a residual norm
            warnings.warn(f'EOM-{sector} Davidson left {int(np.sum(~conv.astype(bool)))} of '
                          f'{len(conv)} roots unconverged (tol {tol:.1e}, max_cycle '
                          f'{max_cycle}); EOMResult.conv says which',
                          UnconvergedEOMRoots, stacklevel=2)
        return EOMResult(self, sector, left, omega, vectors, conv,
                         max_imag=max_imag)


class ComplexEOMEigenvalues(RuntimeWarning):
    """A non-Hermitian EOM solve that returned eigenvalues with an
    imaginary part above EOM_IMAG_TOL: the real parts are kept and the result
    is flagged (`EOMResult.complex_eigenvalues`)."""


class UnconvergedEOMRoots(RuntimeWarning):
    """The EOM Davidson stopped with some roots unconverged: they are
    returned and flagged (`EOMResult.conv` per root, `EOMResult.converged`)."""


#: |Im omega| above which an EOM eigenvalue is reported as complex.
EOM_IMAG_TOL = 1e-6


class EOMResult:
    """The roots of one EOM solve. `max_imag` is the largest |Im| of the
    eigenvalues the solver returned and `complex_eigenvalues` whether it
    exceeds EOM_IMAG_TOL: the omega kept are the real parts. `conv` is the
    Davidson's per-root convergence flag and `converged` whether all roots
    converged."""

    def __init__(self, eomcc, sector, left, omega, vectors, conv,
                 max_imag=0.0):
        self.eomcc = eomcc
        self.sector = sector
        self.left = left
        self.omega = omega
        self.vectors = vectors
        self.conv = conv
        self.converged = bool(np.all(conv))
        self.max_imag = float(max_imag)
        self.complex_eigenvalues = self.max_imag > EOM_IMAG_TOL

    def amplitudes(self, n):
        """Unpack the n-th eigenvector into its r{1,2,3} (or l{1,2,3})
        amplitude blocks,."""
        side = 'l' if self.left else 'r'
        no, nv = self.eomcc.nocc, self.eomcc.nvir
        self.eomcc._sector = self.sector
        return self.eomcc._unpack(self.vectors[n], side, no, nv)

    def transition_densities(self, left_result=None):
        """EOM-EE transition densities for every state in this (right)
        result: rho[n](pq) = <(1+Lambda_gs)|e^-T p^dag q e^T|R_n> and
        rho_star[n](pq) = <L_n|e^-T p^dag q e^T|1>, validated to machine
        precision against the determinant_space oracle (full CCSDT, see
        tests/test_eom_ccsdt.py).

        Requires the matching LEFT eigenvectors for the SAME states; if
        `left_result` isn't supplied, solves for it (same nroots). Right/left
        vectors are biorthogonalized per near-degenerate omega cluster (see
        `_biorthogonalize_degenerate`) before use -- required whenever the
        excitation spectrum has genuine degeneracies (e.g. spin multiplets),
        since the independent left/right Davidson solves otherwise return
        uncorrelated bases within each degenerate subspace.

        Returns (rho, rho_star), each shape (nstates, norb, norb).
        """
        if self.sector != 'ee':
            raise ValueError("transition_densities is only defined for the 'ee' sector")
        if self.left:
            raise ValueError("call transition_densities on the RIGHT EOMResult "
                             "(kernel('ee', left=False)), not the left one")

        eomcc = self.eomcc
        no, nv = eomcc.nocc, eomcc.nvir
        norb = eomcc.norb
        o, v = slice(None, no), slice(no, None)
        t1, t2, t3 = eomcc.t1, eomcc.t2, eomcc.t3
        lam1, lam2, lam3 = eomcc.lambda_amplitudes()
        kd = np.eye(norb)

        nstates = len(self.omega)
        if left_result is None:
            # Request a few extra roots beyond nstates: non-Hermitian Davidson
            # needs "elbow room" past the exact size of a degenerate cluster
            # to reliably resolve every member of it -- confirmed directly on
            # He/def2-SVP's 3-fold-degenerate p-shell excitation, where
            # nroots=3 silently returned only 2 of the 3 degenerate left
            # eigenvectors (plus an unrelated extra root at a different
            # eigenvalue) while nroots=6 resolved the full triplet cleanly.
            left_result = eomcc.kernel('ee', nroots=nstates + 4, left=True)
        if len(left_result.omega) < nstates or np.max(np.abs(
                left_result.omega[:nstates] - self.omega[:nstates])) > 1e-5:
            raise RuntimeError("left/right EE eigenvalues do not match -- "
                               "cannot pair L_n/R_n states for transition densities")

        r_vecs, l_vecs = _biorthogonalize_degenerate(
            self.omega, self.vectors[:nstates], left_result.vectors[:nstates])

        rho = np.zeros((nstates, norb, norb))
        rho_star = np.zeros((nstates, norb, norb))
        for n in range(nstates):
            self.eomcc._sector = 'ee'
            r_amps = eomcc._unpack(r_vecs[n], 'r', no, nv)
            l_amps = eomcc._unpack(l_vecs[n], 'l', no, nv)
            r1n, r2n, r3n = r_amps['r1'], r_amps['r2'], r_amps['r3']
            l1n, l2n, l3n = l_amps['l1'], l_amps['l2'], l_amps['l3']

            rho[n, o, o] = ee_density.rho_oo(lam1, lam2, lam3, r1n, r2n, r3n, t1, t2, t3, kd, o, v)
            rho[n, v, v] = ee_density.rho_vv(lam1, lam2, lam3, r1n, r2n, r3n, t1, t2, t3, kd, o, v)
            rho[n, o, v] = ee_density.rho_ov(lam1, lam2, lam3, r1n, r2n, r3n, t1, t2, t3, kd, o, v)
            rho[n, v, o] = ee_density.rho_vo(lam1, lam2, lam3, r1n, r2n, r3n, t1, t2, t3, kd, o, v)

            rho_star[n, o, o] = ee_density.rho_star_oo(l1n, l2n, l3n, t1, t2, t3, kd, o, v)
            rho_star[n, v, v] = ee_density.rho_star_vv(l1n, l2n, l3n, t1, t2, t3, kd, o, v)
            rho_star[n, o, v] = ee_density.rho_star_ov(l1n, l2n, l3n, t1, t2, t3, kd, o, v)
            rho_star[n, v, o] = ee_density.rho_star_vo(l1n, l2n, l3n, t1, t2, t3, kd, o, v)

        return rho, rho_star

    def polarizability(self, dip_so, omega_grid=None):
        """Dynamic dipole polarizability tensor from the Lehmann sum:
        alpha_xy(w) = sum_n [ mu_x(0n) mu_y(n0)/(Omega_n - w)
                              + mu_y(0n) mu_x(n0)/(Omega_n + w) ].
        dip_so: (3, norb, norb) dipole integrals in the spin-orbital MO basis.
        Returns alpha, shape (len(omega_grid), 3, 3) (omega_grid defaults to
        [0.0], i.e. the static polarizability)."""
        rho, rho_star = self.transition_densities()
        mu_0n = np.einsum('xpq,npq->nx', dip_so, rho)
        mu_n0 = np.einsum('xpq,npq->nx', dip_so, rho_star)
        if omega_grid is None:
            omega_grid = np.array([0.0])
        omega_grid = np.asarray(omega_grid)
        alpha = np.zeros((len(omega_grid), 3, 3))
        for k, wv in enumerate(omega_grid):
            d1 = 1.0 / (self.omega - wv)
            d2 = 1.0 / (self.omega + wv)
            alpha[k] = (np.einsum('nx,ny,n->xy', mu_0n, mu_n0, d1)
                        + np.einsum('ny,nx,n->xy', mu_0n, mu_n0, d2))
        return alpha

