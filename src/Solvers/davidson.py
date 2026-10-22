"""Symmetric Davidson eigensolver: the single iterative core behind both ADC
routes -- charged (IP/EA, root-following) and neutral (ee, lowest-k).

The iteration itself is pyscf's lib.davidson1. What lives here is the
problem-agnostic setup the two routes were duplicating: the Jacobi-Davidson
preconditioner, the overlap `pick` that root-follows a reference vector, the
lowest-diagonal seeding, and one tolerance contract.

Tolerance contract: `tol_residual` is a RESIDUAL NORM everywhere, and it is
the binding criterion. pyscf's own eigenvalue tolerance (`tol_eig` here, its
`tol`) defaults to the same number, which for a symmetric operator is the
looser of the two -- the Ritz-value error goes as |r|^2/gap, so a root that
meets the residual stopped moving long before. Pass `tol_eig` to decouple
them. This is deliberately NOT pyscf's own convention, where `tol` is the
eigenvalue tolerance and the residual tolerance is its SQUARE ROOT, so
conv_tol=1e-6 there silently means |r| <= 1e-3.

Not the solver for the other two eigenproblems in this tree: the Casida form
is non-Hermitian and paired (LinearResponse/davidson.py, pyscf real_eig) and
EOM-CC is non-Hermitian and biorthogonal (CC/eom.py, pyscf davidson_nosym1).
Neither reduces to this one.
"""
import warnings

import numpy as np
from pyscf import lib as pyscf_lib
from pyscf.lib.linalg_helper import LinearDependenceError

from src.Base.constants import DAVIDSON_COMPLETION_SEED, DAVIDSON_FOLLOW_OVERLAP_TIE


def jacobi_davidson_precond(diag, floor=1e-8):
    """Diagonal (K) preconditioner with the Jacobi-Davidson projection:

        t = K^-1 r - alpha K^-1 u,   alpha = (u.K^-1 r) / (u.K^-1 u)

    i.e. the plain diagonal correction with its component along the Ritz
    vector u removed. That component is the one Davidson cannot use -- u is
    already in the subspace -- and leaving it in is the classic stagnation
    mode on operators whose off-diagonal is small compared to the spread of
    the diagonal, which is exactly the shape of an ADC 2h1p/2p1h block.

    floor clamps |diag - e| so a configuration sitting on the Ritz value does
    not produce an infinite correction.

    One pyscf quirk to know: davidson1 calls precond(r_k, e[0], u_k) -- the
    residual and Ritz vector of root k, but the LOWEST Ritz value for every
    root. The projection is therefore always exact (it uses u_k) while the
    shift is only exact for the lowest root. Preconditioning cannot move a
    converged eigenpair, so this costs iterations on the upper roots, never
    accuracy.
    """
    diag = np.asarray(diag, float)

    def precond(dx, e, x0):
        d = diag - e
        d = np.where(np.abs(d) < floor, floor, d)
        Kr = np.asarray(dx) / d
        u = np.asarray(x0)
        Ku = u / d
        uKu = float(u @ Ku)
        if abs(uKu) > 1e-300:
            Kr = Kr - ((u @ Kr) / uKu) * Ku
        return Kr

    return precond


def overlap_pick(ref):
    """A davidson1 `pick` that root-FOLLOWS: at every step take the nroots
    Ritz vectors with the largest |overlap| onto the fixed reference `ref`,
    rather than the algebraically lowest.

    This is how the charged-ADC route targets a specific quasiparticle (the
    Koopmans unit vector on the HOMO row, or a downfolded/satellite seed)
    instead of whatever happens to be lowest. The selected roots are returned
    sorted ascending in energy, which is the order davidson1 then reports.

    An (n, m) `ref` follows the span of its columns: a Ritz vector's overlap
    is the norm of its projection onto that span (a warm start that carries
    the previous root beside the original seed). Overlaps within
    DAVIDSON_FOLLOW_OVERLAP_TIE of each other are a tie, taken lowest energy
    first: asked for more roots than the reference spans, the rest have
    overlap ~0 (a degenerate spin partner, an uncoupled sector), and an
    unbroken tie would re-pick them at random from one step to the next.
    """
    ref = np.asarray(ref, float)
    span = np.linalg.qr(ref)[0] if ref.ndim == 2 else ref[:, None]

    def pick(w, v, nroots, envs):
        xs = envs['xs']
        ref_coeff = np.array([span.T @ x for x in xs])          # (nx, m)
        overlap = np.linalg.norm(ref_coeff.T @ v, axis=0)
        key = np.round(overlap / DAVIDSON_FOLLOW_OVERLAP_TIE)
        idx = np.lexsort((w, -key))[:nroots]
        order = idx[np.argsort(w[idx])]
        return w[order], v[:, order], order

    return pick


def diagonal_seeds(diag, count):
    """`count` unit vectors on the lowest entries of `diag` -- the zeroth-order
    guess for the lowest roots. Seeding wider than nroots is deliberate: a
    bare-nroots seed can collapse a degenerate pair, because the members of a
    degenerate set are generally distinct configurations and one seed can only
    open one of them."""
    diag = np.asarray(diag, float)
    count = max(1, min(int(count), diag.size))
    cols = np.argsort(diag)[:count]
    seeds = np.zeros((count, diag.size))
    seeds[np.arange(count), cols] = 1.0
    return [seeds[i] for i in range(count)]


def _as_guess_list(x0, n):
    """Normalized column list from an (n,) or (n, m) guess, dropping null
    columns. Returns None if nothing usable survives."""
    x0 = np.asarray(x0, float)
    x0 = x0.reshape(n, -1) if x0.ndim > 1 else x0.reshape(n, 1)
    out = []
    for j in range(x0.shape[1]):
        nrm = np.linalg.norm(x0[:, j])
        if nrm > 1e-12:
            out.append(x0[:, j] / nrm)
    return out or None


def _default_subspace(nroots):
    """Trial vectors to hold by default.

    Generous on purpose. davidson1 restarts by collapsing onto the nroots Ritz
    vectors alone, so a subspace only a little wider than the seed restarts
    almost immediately and then again and again, re-climbing from nroots every
    time. That is not a slow convergence, it is a stalled one: an ee-ADC(2)
    channel that stays unconverged at 6*nroots+20 converges at 10*nroots+30 in
    a small fraction of the matrix-vector products.

    Memory-limited callers pass max_subspace explicitly; that is what the knob
    is for, and the cost of turning it down is exactly the thrashing above.
    """
    return max(10 * nroots + 30, 60)


def _lindep_for(tol_residual, floor=1e-22, ceiling=1e-14):
    """davidson1's linear-dependence threshold, scaled to the residual asked
    for.

    This is not tuning. davidson1 reuses `lindep` for a second, unrelated
    test: a root that has NOT converged but whose residual satisfies
    |r|^2 <= lindep has its correction vector dropped as linearly dependent,
    and when every root is dropped the solver breaks out reporting whatever
    it has. lindep therefore puts a hard floor of sqrt(lindep) on the
    residual any root can ever reach -- 1e-7 at pyscf's default 1e-14. Ask
    for |r| <= 1e-8 with that default and the solve stalls at ~1e-7 and tells
    you it did not converge, forever, however many cycles it is given.

    So the threshold has to follow the tolerance: two orders below |r|^2,
    never looser than pyscf's own default, and floored well above double
    precision round-off (a vector orthogonalized to a squared norm of 1e-22
    still has ~6 honest digits left, and davidson1 projects out the existing
    subspace only once per correction rather than twice).
    """
    return float(max(floor, min(ceiling, 1e-2 * tol_residual ** 2)))


def solve_symmetric(matvec, diag, nroots=1, x0=None, pick=None,
                    tol_residual=1e-6, tol_eig=None, max_cycle=200,
                    max_subspace=None, precond=None, lindep=None, verbose=0,
                    warn_unconverged=True, label='Davidson', max_memory=None,
                    complete_lowest=True):
    """The `nroots` lowest eigenpairs of a real symmetric matrix-free
    operator -- or, with `pick`, the `nroots` it selects.

    matvec: callable (n,) -> (n,), or anything with a .matvec. ONE vector at a
        time; davidson1 is handed a list wrapper. An object with a `.matmat`,
        (n, m) -> (n, m), is handed each batch of trial vectors at once.
    diag: (n,) the operator's (approximate) diagonal. Used for the
        preconditioner AND, when x0 is None, for the seeds. It must be the
        diagonal of the operator `matvec` ACTUALLY applies -- if the caller
        shifts or projects inside matvec, the shifted diagonal is the one that
        belongs here.
    nroots: how many eigenpairs, clamped to n.
    x0: optional (n,) or (n, m) guess, used verbatim -- the caller owns its
        width. None seeds 2*nroots+4 lowest-diagonal unit vectors.
    pick: optional davidson1 `pick`; see overlap_pick. None takes the lowest.
    tol_residual / tol_eig: see the module docstring. The residual binds.
    max_subspace: cap on the number of trial vectors held. Storage peaks at
        roughly TWICE this, because davidson1 keeps the operator's image (ax)
        alongside the subspace (xs) -- at a doubles vector of no^2 nv^2 that
        factor is the difference between fitting in memory and not. The
        DEFAULT is sized for convergence, not for memory; this argument is the
        memory answer, and setting it low is paid for in matrix-vector
        products (see _default_subspace).
    max_memory: MB davidson1 may hold its subspace in. Above it the subspace
        goes to HDF5 files under PYSCF_TMPDIR, and the disk traffic can cost
        more than the matrix-vector products. It bounds the subspace only, so
        a caller holding other arrays passes what they leave free. None keeps
        davidson1's own default, pyscf's 4000 MB.
    max_cycle: davidson1 restarts by collapsing the subspace onto the nroots
        Ritz vectors alone, so it re-climbs out of a restart slowly and wants
        a generous budget.
    precond: override the default Jacobi-Davidson preconditioner.
    lindep: davidson1's linear-dependence threshold, on SQUARED norms. It is
        scaled with tol_residual by default and you almost certainly want
        that -- see _lindep_for.

    complete_lowest: without `pick`, a converged set is certified to be the
        LOWEST nroots, not only eigenpairs: a copy of a degenerate level the
        subspace never reached (floating point does not leave an invariant
        subspace the start missed) sits below the top root unseen. So the
        lowest root of the operator on the complement of X (X lifted above
        the top root) is solved, from the lowest-diagonal unit vectors and
        seeded random directions off X, and merged while it lies below the
        top root, at most nroots + 1 rounds (_complete_lowest). If that solve
        does not converge or no round certifies the set, every conv is set
        False. False skips it (the complement solve itself).

    Returns (e, X, conv): e (nroots,) ascending, X (n, nroots) columns
    matching e, conv (nroots,) bool. conv is a real flag -- check it.
    """
    block = getattr(matvec, 'matmat', None)
    matvec = matvec.matvec if hasattr(matvec, 'matvec') else matvec
    diag = np.asarray(diag, float).ravel()
    n = diag.size
    nroots = max(1, min(int(nroots), n))
    if tol_eig is None:
        tol_eig = tol_residual
    if max_subspace is None:
        max_subspace = _default_subspace(nroots)
    max_subspace = int(min(max_subspace, n))
    if n > max_subspace < 3 * nroots + 4:
        warnings.warn(
            f'{label}: max_subspace={max_subspace} leaves almost no room above '
            f'the {nroots} Ritz vectors a restart collapses to, so the solve '
            'will restart continuously and may never converge. Raise it, or '
            'ask for fewer roots.', RuntimeWarning, stacklevel=3)
    # davidson1 inflates its own max_space by 4*(nroots-1) and holds nroots
    # more Ritz vectors on top of it, so the trial vectors actually held
    # number max_space + 4*(nroots-1) + nroots. Invert that, so max_subspace
    # means what it says.
    max_space = max(nroots, max_subspace - nroots - 4 * (nroots - 1))
    if lindep is None:
        lindep = _lindep_for(tol_residual)

    guess = _as_guess_list(x0, n) if x0 is not None else None
    if guess is None:
        # wider than nroots for degenerate clusters, but never so wide that
        # the seed alone crowds the subspace into restarting immediately
        guess = diagonal_seeds(diag, min(n, 2 * nroots + 4,
                                         max(nroots, max_subspace // 2)))
    if precond is None:
        precond = jacobi_davidson_precond(diag)

    def aop(xs):
        if block is not None and len(xs) > 1:
            # one product for the whole batch: an operator whose cost is
            # dominated by vector-independent work shares it
            Y = block(np.column_stack([np.asarray(x).ravel() for x in xs]))
            return [Y[:, k] for k in range(Y.shape[1])]
        return [matvec(x) for x in xs]

    try:
        conv, e, c = pyscf_lib.davidson1(
            aop, guess, precond, nroots=nroots, pick=pick,
            tol=tol_eig, tol_residual=tol_residual, lindep=lindep,
            max_cycle=max_cycle, max_space=max_space, verbose=verbose,
            **({} if max_memory is None else {'max_memory': max_memory}))
    except LinearDependenceError as exc:
        # The trial subspace has spanned everything reachable, which at these
        # dimensions means the operator is small enough that iterating on it
        # was never the right call.
        raise RuntimeError(
            f'{label}: the trial subspace went linearly dependent before '
            f'{nroots} roots converged on a dimension-{n} operator. Below a '
            'few thousand configurations, Davidson exhausts the space long '
            'before it converges -- build the supermatrix and diagonalize it '
            'densely instead (matrix_free=False, or the route driver\'s '
            'dense_limit).') from exc

    e = np.atleast_1d(np.asarray(e, float))
    c = np.asarray(c)
    if c.ndim == 1:
        c = c[None, :]
    conv = np.atleast_1d(np.asarray(conv, bool))
    # davidson1 can return fewer roots than asked for (it says so on its own
    # logger and then hands back what it has); keep conv aligned with e.
    if conv.size < e.size:
        conv = np.concatenate([conv, np.zeros(e.size - conv.size, bool)])
    conv = conv[:e.size]

    order = np.argsort(e)
    e, X, conv = e[order], c[order].T, conv[order]

    if complete_lowest and pick is None and conv.all() and e.size < n:
        e, X, conv = _complete_lowest(
            matvec, block, diag, e, X, warn_unconverged, dict(
                tol_residual=tol_residual, tol_eig=tol_eig, max_cycle=max_cycle,
                max_subspace=max_subspace, precond=precond, lindep=lindep,
                verbose=verbose, label=label + ' (complement)',
                max_memory=max_memory))
        if not conv.all():
            return e, X, conv               # _complete_lowest said why

    if warn_unconverged and not conv.all():
        stuck = np.flatnonzero(~conv)
        warnings.warn(
            f'{label} left {stuck.size} of {e.size} roots unconverged after '
            f'{max_cycle} cycles at |r| <= {tol_residual:g} (roots '
            f'{stuck.tolist()}); their energies are whatever the last '
            'subspace happened to give. Raise max_cycle or max_subspace, or '
            'loosen the tolerance.', RuntimeWarning, stacklevel=3)
    return e, X, conv


class _Deflated:
    """A on the complement of the orthonormal columns of X, with X lifted to
    `lift`: P A P + lift X X^T, P = 1 - X X^T. A batch goes through `block`
    when the operator has one."""

    def __init__(self, matvec, block, X, lift):
        self.X, self.lift, self._mv, self._block = X, lift, matvec, block
        if block is not None:
            self.matmat = self._matmat

    def _matmat(self, V):
        X = self.X
        C = X.T @ V
        AV = self._block(V - X @ C)
        return AV - X @ (X.T @ AV) + self.lift * (X @ C)

    def matvec(self, v):
        X = self.X
        v = np.asarray(v).ravel()
        c = X.T @ v
        Av = np.asarray(self._mv(v - X @ c)).ravel()
        return Av - X @ (X.T @ Av) + self.lift * (X @ c)


def _off(V, X):
    """The columns of V projected off X (twice) and normalized; null ones dropped."""
    V = V - X @ (X.T @ V)
    V = V - X @ (X.T @ V)
    nrm = np.linalg.norm(V, axis=0)
    return V[:, nrm > 1e-8] / nrm[nrm > 1e-8]


def _complete_lowest(matvec, block, diag, e, X, warn, opts):
    """(e, X, conv) of solve_symmetric's lowest-k set, completed: the lowest
    root of A on the complement of X (X lifted above the top root) replaces the
    top root when it lies below it by more than tol_residual (so an exactly
    degenerate copy at the top is not swapped back and forth); repeated until
    a complement solve certifies the set, at most k + 1 rounds. One root per
    round: the decision needs only the lowest, and a many-root complement
    solve stalls at tight tolerance when its roots span a wide range. conv all
    False if a complement solve does not converge or no round certifies."""
    n, k = X.shape
    tol = opts['tol_residual']
    rng = np.random.default_rng(DAVIDSON_COMPLETION_SEED)
    seeds = np.zeros((n, min(n, 2 * k + 4)))
    seeds[np.argsort(diag)[:seeds.shape[1]], np.arange(seeds.shape[1])] = 1.0
    why = None
    for _ in range(k + 1):
        if n - k < 1:
            return e, X, np.ones(k, bool)
        lift = e[-1] + abs(e[-1]) + (e[-1] - e[0]) + 1.0
        start = _off(np.column_stack([seeds, rng.standard_normal((n, k))]), X)
        mu, Y, cconv = solve_symmetric(
            _Deflated(matvec, block, X, lift), diag, nroots=1, x0=start,
            warn_unconverged=False, complete_lowest=False, **opts)
        if not cconv.all():
            why = 'the solve on their complement did not converge'
            break
        if mu[0] >= e[-1] - tol:
            return e, X, np.ones(k, bool)
        y = np.asarray(Y)[:, 0]
        y = y - X @ (X.T @ y)
        y = y / np.linalg.norm(y)
        w_all = np.append(e, mu[0])
        X_all = np.column_stack([X, y])
        order = np.argsort(w_all, kind='stable')[:k]
        e, X = w_all[order], X_all[:, order]
    else:
        why = f'{k + 1} rounds of merging did not end in a certificate'
    if warn:
        warnings.warn(
            f"{opts['label']}: the {k} converged roots are not certified to be "
            f'the lowest ({why}); all reported unconverged.',
            RuntimeWarning, stacklevel=3)
    return e, X, np.zeros(k, bool)
