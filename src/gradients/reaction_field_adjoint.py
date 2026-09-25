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

from src.Base.sliced_factors import SlicedFactors
from src.Base.utils.mpi_grid import agreement, current_comm, lockstep
from src.SingleReference.GW.imaginary_time import DEFAULT_TAU_TARGET
from src.SingleReference.GW.reaction_field import projected_quasiparticle_shift
from src.SingleReference.LinearResponse.davidson import (
    static_screening_grid, static_screening_matrix)
from src.Base.sliced_factors import whole_factor
from src.gradients.space_time_adjoint import chi0_backward


def static_grid(eps, nocc, ntau=None, tau_target=DEFAULT_TAU_TARGET):
    """The one-point (omega = 0) minimax grid Delta W is built on.

    `LinearResponse.davidson.static_screening_grid` itself, under this
    package's name for it, so a chain that builds the reaction field and the
    BSE kernel screens them on ONE axis and the two cannot drift apart.
    """
    return static_screening_grid(eps, nocc, ntau=ntau, tau_target=tau_target)


def static_screening(x_mo, d, eps, nocc, grid=None, mu=None, comm=None,
                     densities=True):
    """(A, W): the orbital densities on the auxiliary index, A[Q,p] = sum_k
    D[k,Q] X[k,p]^2, and [1 - chi0(0)]^-1 in that factor's own gauge.

    W is `LinearResponse.davidson.static_screening_matrix` -- the same routine,
    on the same axis, that gives the BSE kernel its static screening, so a
    solvated chain builds W once and hands the pair to both. Only A and the
    chemical potential `mu` are this side's: Eq. (18) contracts W twice
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
    branches of X_mo, so W is the whole factors' bit for bit. A contracts the
    whole X_mo beside them and is refused on slices, which serve W alone.
    """
    comm = current_comm() if comm is None else comm
    multi = comm is not None and comm.Get_size() > 1
    sliced = isinstance(x_mo, SlicedFactors)
    if sliced and densities:
        raise ValueError(
            'static_screening on sliced factors serves W alone: the orbital '
            'densities of Eq. (18), A = D^T X_mo^2, read X_mo whole beside '
            'the chi0 sweep; pass densities=False, or build the factors '
            'unsliced for a reaction field')
    d = whole_factor(d, 'D')
    if multi:
        eps = lockstep(np.asarray(eps, float), comm)
        agreement((x_mo.coords if sliced else x_mo, d), comm, audit_only=True,
                  label='static_screening inputs')
    grid = static_grid(eps, nocc) if grid is None else grid
    out = (d.T @ (x_mo ** 2) if densities else None,
           static_screening_matrix(x_mo, d, eps, nocc, grid, mu=mu, comm=comm))
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
    grid = static_grid(eps, nocc) if grid is None else grid
    if screening is None:
        screening = (static_screening(x_mo, d_dressed, eps, nocc, grid, mu),
                     static_screening(x_mo, d_bare, eps, nocc, grid, mu))
    (a_d, w_d), (a_b, w_b) = screening
    return projected_quasiparticle_shift(a_d, w_d, a_b, w_b, nocc)


def reaction_field_backward(weights, x_mo, d_dressed, d_bare, eps, nocc,
                            grid=None, mu=None, screening=None):
    """(eps_bar, x_bar, d_dressed_bar, d_bare_bar) of sum_p w_p Delta eps_p.

    The two D adjoints are separate because they land on DIFFERENT metrics:
    `dfactor_adjoint_gauges` takes the dressed one with the environment and the
    bare one without, and it is the environment argument that carries the
    cavity's own motion into the force.

    screening: ((A, W) dressed, (A, W) bare) already built by the caller
               (`static_screening`) on this same grid.
    """
    grid = static_grid(eps, nocc) if grid is None else grid
    if screening is None:
        screening = (static_screening(x_mo, d_dressed, eps, nocc, grid, mu),
                     static_screening(x_mo, d_bare, eps, nocc, grid, mu))
    # the occupancy sign and the one half ride the weights
    u = 0.5 * np.asarray(weights, float).copy()
    u[:nocc] *= -1.0

    rho = x_mo ** 2
    eps_bar = np.zeros(np.asarray(eps, float).shape)
    x_bar = np.zeros(x_mo.shape)
    rho_bar = np.zeros(rho.shape)
    d_bars = []
    for (a, w), d, sign in zip(screening, (d_dressed, d_bare), (1.0, -1.0)):
        au = a * u
        a_bar = (2.0 * sign) * (w @ au)
        # W = [1 - chi0]^-1, so dW = W dchi0 W and the adjoint is sandwiched
        chi_bar = (sign * (w @ (au @ a.T) @ w))[None]
        e_c, x_c, d_c = chi0_backward(chi_bar, x_mo, d, eps, nocc, grid, mu=mu)
        eps_bar += e_c
        x_bar += x_c
        rho_bar += d @ a_bar
        d_bars.append(d_c + rho @ a_bar.T)
    x_bar += 2.0 * x_mo * rho_bar
    return eps_bar, x_bar, d_bars[0], d_bars[1]
