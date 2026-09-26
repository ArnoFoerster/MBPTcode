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
    Ritz vectors with the largest |overlap| onto the fixed reference vector
    `ref`, rather than the algebraically lowest.

    This is how the charged-ADC route targets a specific quasiparticle (the
    Koopmans unit vector on the HOMO row, or a downfolded/satellite seed)
    instead of whatever happens to be lowest. The selected roots are returned
    sorted ascending in energy, which is the order davidson1 then reports.
    """
    ref = np.asarray(ref, float)

    def pick(w, v, nroots, envs):
        xs = envs['xs']
        ref_coeff = np.array([np.dot(ref, x) for x in xs])
        overlap = np.abs(ref_coeff @ v)
        idx = np.argsort(-overlap)[:nroots]
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
                    warn_unconverged=True, label='Davidson'):
    """The `nroots` lowest eigenpairs of a real symmetric matrix-free
    operator -- or, with `pick`, the `nroots` it selects.

    matvec: callable (n,) -> (n,), or anything with a .matvec. ONE vector at a
        time; davidson1 is handed a list wrapper.
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
    max_cycle: davidson1 restarts by collapsing the subspace onto the nroots
        Ritz vectors alone, so it re-climbs out of a restart slowly and wants
        a generous budget.
    precond: override the default Jacobi-Davidson preconditioner.
    lindep: davidson1's linear-dependence threshold, on SQUARED norms. It is
        scaled with tol_residual by default and you almost certainly want
        that -- see _lindep_for.

    Returns (e, X, conv): e (nroots,) ascending, X (n, nroots) columns
    matching e, conv (nroots,) bool. conv is a real flag -- check it.
    """
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
        return [matvec(x) for x in xs]

    try:
        conv, e, c = pyscf_lib.davidson1(
            aop, guess, precond, nroots=nroots, pick=pick,
            tol=tol_eig, tol_residual=tol_residual, lindep=lindep,
            max_cycle=max_cycle, max_space=max_space, verbose=verbose)
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

    if warn_unconverged and not conv.all():
        stuck = np.flatnonzero(~conv)
        warnings.warn(
            f'{label} left {stuck.size} of {e.size} roots unconverged after '
            f'{max_cycle} cycles at |r| <= {tol_residual:g} (roots '
            f'{stuck.tolist()}); their energies are whatever the last '
            'subspace happened to give. Raise max_cycle or max_subspace, or '
            'loosen the tolerance.', RuntimeWarning, stacklevel=3)
    return e, X, conv
