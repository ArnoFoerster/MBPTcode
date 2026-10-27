"""The nuclear derivative of the reaction field's quasiparticle shift.

Duchemin, Guido, Jacquemin and Blase, Chem. Sci. 9, 4430 (2018) Eq. (18) puts
the continuum on the quasiparticle diagonal as the self-polarization of the
orbital carrying the added charge,

    Delta eps_p = -(1/2) <ii|Delta W|ii>   occupied
    Delta eps_p = +(1/2) <aa|Delta W|aa>   virtual

with Delta W = W[v + vtilde] - W[v] at omega = 0, so a solvated excited-state
force needs its derivative.

TWO FACTORIZATIONS, NOT ONE CHI0 AND A CONGRUENCE. Reaching the bare screening
from the dressed one through T = Vt^(-1/2) V^(1/2) saves a chi0 build on a
forward pass, but differentiating that would need the derivative of a
pseudo-inverse of a matrix square root. Building Delta W from the two
separable factors instead leaves each an ORDINARY D -- the dressed one
carrying the cavity, the bare one not -- whose nuclear derivative the D branch
already carries, environment argument and all, so `dfactor_adjoint_gauges`
differentiates both in one pass. The two forward passes agree to the ISDF
reproducibility floor and share everything below the metric root.

The screening is static, so chi0 is needed at one frequency and the tau ->
omega transform runs in that direction only.
"""
import numpy as np

from src.Base.factorization.sliced_factors import GridTileRows, SlicedFactors, whole_factor
from src.Base.utils.parallel.mpi_grid import agreement, current_comm, lockstep
from src.SingleReference.GW.reaction_field import (
    density_projection, pair_densities, projected_quasiparticle_shift)
from src.SingleReference.LinearResponse.space_time import (
    static_screening_grid, static_screening_matrix)
from src.gradients.rpa.space_time_adjoint import chi0_backward, chi0_backward_rows


def static_screening(x_mo, d, eps, nocc, grid=None, mu=None, comm=None,
                     densities=True):
    """(A, W): the orbital densities on the auxiliary index, A[Q,p] = sum_k
    D[k,Q] X[k,p]^2, and [1 - chi0(0)]^-1 in that factor's own gauge.

    W is `LinearResponse.space_time.static_screening_matrix` -- the same
    routine, on the same axis, that gives the BSE kernel its static screening,
    so a solvated chain builds W once and hands the pair to both. Only A and
    the chemical potential `mu` are this side's: Eq. (18) contracts W twice
    against A. `densities=False` leaves A None, for a caller that screens a
    kernel and has no Eq. (18) to build.

    comm: the ranks all call this in lockstep; the tau partition and the single
    (naux, naux) all-reduce are `static_screening_matrix`'s, handed the comm
    explicitly. None is `current_comm()`. A is a local contraction of the
    factors and carries no partition at all. eps is a `lockstep` of rank 0's
    at entry, since the default grid's point count is read off it; the
    factors are identical by construction, and an audited run compares their
    digests and those of (A, W).

    `SlicedFactors`: D is gathered whole once and the sweep gathers the two
    branches of X_mo, so W is the whole factors' bit for bit. A is a sum over
    the grid of row products, formed from the rows (`density_projection`).
    """
    comm = current_comm() if comm is None else comm
    multi = comm is not None and comm.Get_size() > 1
    sliced = isinstance(x_mo, SlicedFactors)
    a = (density_projection(x_mo, d) if sliced and densities else None)
    d = whole_factor(d, 'D')
    if multi:
        eps = lockstep(np.asarray(eps, float), comm)
        agreement((x_mo.coords if sliced else x_mo, d), comm, audit_only=True,
                  label='static_screening inputs')
    grid = static_screening_grid(eps, nocc) if grid is None else grid
    if densities and not sliced:
        a = d.T @ (x_mo ** 2)
    out = (a, static_screening_matrix(x_mo, d, eps, nocc, grid, mu=mu,
                                      comm=comm))
    if multi:
        agreement(out, comm, audit_only=True, label='static_screening outputs')
    return out


def reaction_field_shift(x_mo, d_dressed, d_bare, eps, nocc, grid=None,
                         mu=None, screening=None):
    """Eq. (18) for every orbital, in Hartree, from the TWO factorizations.

    Production's `projected_quasiparticle_shift` is the contraction; this side
    supplies the bare partner by screening the bare factor rather than by
    congruence, for the reason in the module docstring, and the two agree to
    the ISDF reproducibility floor.

    screening: ((A, W) dressed, (A, W) bare) already built by the caller
               (`static_screening`) on this same grid.
    """
    grid = static_screening_grid(eps, nocc) if grid is None else grid
    if screening is None:
        screening = (static_screening(x_mo, d_dressed, eps, nocc, grid, mu),
                     static_screening(x_mo, d_bare, eps, nocc, grid, mu))
    (a_d, w_d), (a_b, w_b) = screening
    return projected_quasiparticle_shift(a_d, w_d, a_b, w_b, nocc)


def reaction_field_backward(weights, x_mo, d_dressed, d_bare, eps, nocc,
                            grid=None, mu=None, screening=None, block=None,
                            dressed_chi0_bar=None):
    """(eps_bar, x_bar, d_dressed_bar, d_bare_bar) of sum_p w_p Delta eps_p.

    The two D adjoints are separate because they land on DIFFERENT metrics:
    `dfactor_adjoint_gauges` takes the dressed one with the environment and the
    bare one without, and it is the environment argument that carries the
    cavity's own motion into the force.

    screening: ((A, W) dressed, (A, W) bare) already built by the caller
               (`static_screening`) on this same grid.
    block: on `SlicedFactors`, the tile edge of the grid tiles the three
           adjoints are returned in (`GridTileRows`), the chi0 sweep's.
    dressed_chi0_bar: (naux, naux), another adjoint on the dressed chi0(0)
           (the BSE kernel's static W, screened on the same factors and
           grid), added to this one's before the sweep: chi0's adjoint is
           linear in its seed, so one sweep carries both.
    """
    grid = static_screening_grid(eps, nocc) if grid is None else grid
    if screening is None:
        screening = (static_screening(x_mo, d_dressed, eps, nocc, grid, mu),
                     static_screening(x_mo, d_bare, eps, nocc, grid, mu))
    # the occupancy sign and the one half ride the weights
    u = 0.5 * np.asarray(weights, float).copy()
    u[:nocc] *= -1.0
    if isinstance(x_mo, SlicedFactors):
        return _backward_rows(u, x_mo, (d_dressed, d_bare), screening, eps,
                              nocc, grid, mu, block, dressed_chi0_bar)

    rho = x_mo ** 2
    eps_bar = np.zeros(np.asarray(eps, float).shape)
    x_bar = np.zeros(x_mo.shape)
    rho_bar = np.zeros(rho.shape)
    d_bars = []
    seeds = (dressed_chi0_bar, None)
    for (a, w), d, sign, seed in zip(screening, (d_dressed, d_bare),
                                     (1.0, -1.0), seeds):
        au = a * u
        a_bar = (2.0 * sign) * (w @ au)
        # W = [1 - chi0]^-1, so dW = W dchi0 W and the adjoint is sandwiched
        chi_bar = (sign * (w @ (au @ a.T) @ w))[None]
        if seed is not None:
            chi_bar[0] += seed
        e_c, x_c, d_c = chi0_backward(chi_bar, x_mo, d, eps, nocc, grid, mu=mu)
        eps_bar += e_c
        x_bar += x_c
        rho_bar += d @ a_bar
        d_bars.append(d_c + rho @ a_bar.T)
    x_bar += 2.0 * x_mo * rho_bar
    return eps_bar, x_bar, d_bars[0], d_bars[1]


def added_reaction_field_adjoint(eps_bar, x_bar, d_bar, d_bar_bare, terms):
    """eps_bar + the eps adjoint of `terms`, a `reaction_field_backward`
    (or off-diagonal) result (eps_bar, X_bar, D_bar dressed, D_bar bare), with
    its other three added into x_bar, d_bar and d_bar_bare in place: the one
    accumulation every reverse pass through Eq. (18) ends in."""
    e_rf, x_rf, dd_rf, db_rf = terms
    x_bar += x_rf
    d_bar += dd_rf
    d_bar_bare += db_rf
    return eps_bar + e_rf


def offdiagonal_reaction_field_backward(weights, pairs, x_mo, d_dressed,
                                        d_bare, eps, nocc, grid=None, mu=None,
                                        screening=None, block=None):
    """(eps_bar, x_bar, d_dressed_bar, d_bare_bar) of sum_k w_k Sigma^env_pk qk
    (`offdiagonal_reaction_field`), the same four adjoints
    `reaction_field_backward` returns, whole or in grid tiles of `block`.

    Each element is bilinear, t (a_p + a_q)^T W a_pq, so the orbital
    densities take t W a_pq, the pair density t W (a_p + a_q), and chi0 the
    symmetric part of t W (a_p + a_q) a_pq^T W.
    """
    grid = static_screening_grid(eps, nocc) if grid is None else grid
    if screening is None:
        screening = (static_screening(x_mo, d_dressed, eps, nocc, grid, mu),
                     static_screening(x_mo, d_bare, eps, nocc, grid, mu))
    p = np.array([int(a) for a, _ in pairs])
    q = np.array([int(b) for _, b in pairs])
    t = 0.5 * np.asarray(weights, float) * np.where(p < nocc, -1.0, 1.0)
    sliced = isinstance(x_mo, SlicedFactors)
    if sliced and block is None:
        raise ValueError('sliced factors return the adjoints in grid tiles; '
                         'pass their tile edge (block)')
    x = x_mo.X_mo if sliced else x_mo
    eps_bar = np.zeros(np.asarray(eps, float).shape)
    if sliced:
        comm = x_mo.require(current_comm()).comm
        x_bar = GridTileRows.zeros(x_mo.npts, x.shape[1], block, comm)
    else:
        x_bar = np.zeros(x.shape)
    rho_bar = np.zeros(x.shape)
    d_bars = []
    for (a, w), d, sign in zip(screening, (d_dressed, d_bare), (1.0, -1.0)):
        pd = pair_densities(x_mo, d, pairs)
        v = a[:, p] + a[:, q]
        ts = sign * t
        wv, wpd = w @ (v * ts), w @ pd
        a_bar = np.zeros(a.shape)
        np.add.at(a_bar.T, p, (wpd * ts).T)
        np.add.at(a_bar.T, q, (wpd * ts).T)
        pd_bar = wv
        half = wv @ wpd.T
        chi_bar = (0.5 * (half + half.T))[None]
        dd = d.D if sliced else d
        pair_bar = dd @ pd_bar
        rho_bar += dd @ a_bar
        rows = (x ** 2) @ a_bar.T + (x[:, p] * x[:, q]) @ pd_bar.T
        xb = np.zeros(x.shape)
        np.add.at(xb.T, p, (x[:, q] * pair_bar).T)
        np.add.at(xb.T, q, (x[:, p] * pair_bar).T)
        if sliced:
            e_c, x_c, d_c = chi0_backward_rows(chi_bar, x_mo, d, eps, nocc,
                                               grid, mu=mu, block=block)
            d_c += GridTileRows.from_blocks(rows, x_mo.npts, block, comm)
            x_bar += GridTileRows.from_blocks(xb, x_mo.npts, block, comm)
        else:
            e_c, x_c, d_c = chi0_backward(chi_bar, x_mo, d, eps, nocc, grid,
                                          mu=mu)
            d_c = d_c + rows
            x_bar += xb
        eps_bar += e_c
        x_bar += x_c
        d_bars.append(d_c)
    tail = 2.0 * x * rho_bar
    if sliced:
        x_bar += GridTileRows.from_blocks(tail, x_mo.npts, block, comm)
    else:
        x_bar += tail
    return eps_bar, x_bar, d_bars[0], d_bars[1]


def _backward_rows(u, x_mo, factors, screening, eps, nocc, grid, mu, block,
                   dressed_chi0_bar=None):
    """`reaction_field_backward` on `SlicedFactors`: (eps_bar, X_bar, D_bar
    dressed, D_bar bare), the three adjoints `GridTileRows` in tiles of
    `block`. A = D^T rho is linear in both, so its adjoints are row-local --
    D_bar += rho a_bar^T, rho_bar = D a_bar, X_bar += 2 X rho_bar on this
    rank's rows -- and moved into the tiles verbatim; the screening's adjoint
    is the chi0 sweep's own in the same tiles (`chi0_backward_rows`)."""
    if block is None:
        raise ValueError('sliced factors return the adjoints in grid tiles; '
                         'pass their tile edge (block)')
    comm = x_mo.require(current_comm()).comm
    npts = x_mo.npts
    rho = x_mo.X_mo ** 2
    eps_bar = np.zeros(np.asarray(eps, float).shape)
    x_bar = GridTileRows.zeros(npts, rho.shape[1], block, comm)
    rho_bar = np.zeros(rho.shape)
    d_bars = []
    for (a, w), d, sign, seed in zip(screening, factors, (1.0, -1.0),
                                     (dressed_chi0_bar, None)):
        au = a * u
        a_bar = (2.0 * sign) * (w @ au)
        # W = [1 - chi0]^-1, so dW = W dchi0 W and the adjoint is sandwiched
        chi_bar = (sign * (w @ (au @ a.T) @ w))[None]
        if seed is not None:
            chi_bar[0] += seed
        e_c, x_c, d_c = chi0_backward_rows(chi_bar, x_mo, d, eps, nocc, grid,
                                           mu=mu, block=block)
        eps_bar += e_c
        x_bar += x_c
        rho_bar += d.D @ a_bar
        d_c += GridTileRows.from_blocks(rho @ a_bar.T, npts, block, comm)
        d_bars.append(d_c)
    x_bar += GridTileRows.from_blocks(2.0 * x_mo.X_mo * rho_bar, npts, block,
                                      comm)
    return eps_bar, x_bar, d_bars[0], d_bars[1]
