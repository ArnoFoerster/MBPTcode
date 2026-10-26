"""The nuclear adjoint of the AO-product Coulomb-metric ISDF fit
(`ao_product_fit`) on frozen points.

The forward fit, in the notation of `ao_product_fit`:

    A  = X X^T,  s = diag(A),  d = 1/s
    G  = (A o A) o (d d^T) + reg I
    E  = sum_mn (P|mn) X_gm X_gn,   F = V^-1 E
    B  = (F diag(d)) G^-1,          M = B diag(d)
    D_g = M^T R_g,  R_g = (V + vtilde_g)^1/2 for every gauge g

read backwards from adjoints D_bar_g on the factors:

    M_bar = sum_g R_g D_bar_g^T,  R_bar_g = M D_bar_g  (-> V, vtilde_g)
    B_bar = M_bar d,              H_bar = B_bar G^-1     (H = F d)
    G_bar = -B^T H_bar            low rank, the two solves made once
    F_bar = H_bar d,              E_bar = V^-1 F_bar,  V_bar -= E_bar F^T
    d_bar = sum_P (M_bar o B + H_bar o F) + ((G_bar + G_bar^T) o A o A) d
    X_bar = 2 ((G_bar + G_bar^T) o A o d d^T) X - 2 (d_bar d^2) o X
            + 2 sum_nP E_bar[P, g] (P|mn) X_gn
    (mn|P)_bar = sum_g E_bar[P, g] X_gm X_gn

Nothing of the test set's nao^2 width is formed: the Gram terms are (nk, nk)
products, formed by every rank, and E and its adjoint are carried by one pass
over the three-centre integrals and their derivatives, blocked over the
shells of the first AO, each rank on its own share of the blocks
(`ao_product_fit.own_runs`). (mn|P)_bar is symmetric in (m, n), as (mn|P) is, so the derivative
of the second AO's centre equals the first's and only int3c2e_ip1 and
int3c2e_ip2 of the block are needed. X_bar closes on the AO centres and,
through the frozen points carried with their atoms, on the point chain
(`isdf_derivatives.point_chain`).

The forward's arithmetic is repeated here rather than read from the fit:
the force is the derivative of the energy the fit gives, to the rounding of
the balanced Gram matrix.
"""
import numpy as np
import scipy.linalg

from src.Base.factorization.ao_product_fit import (ao_product_projection,
                                                   balanced_gram,
                                                   half_transform_operand,
                                                   half_transformed,
                                                   own_runs, point_tiles,
                                                   shell_runs,
                                                   three_centre_block)
from src.Base.factorization.separable_ri import ao_values, aux_metric_sqrt
from src.Base.utils.parallel.mpi_grid import (current_comm, lockstep, ordered_sum,
                                              reduce_sum)
from src.environments.environment import dresses_interaction
from src.Base.pyscf.pyscf_interface import auxmol_of
from src.gradients.isdf_derivatives import (GaugeAdjoint, PointChain,
                                            basis_centre_forces,
                                            isdf_exchange_adjoints,
                                            mean_field_point_chain,
                                            point_layout, sqrtm_adjoint,
                                            two_centre_adjoint)


def ao_product_fit_adjoint_gauges(mol, auxmol, coords, gauges, pts_local,
                                  atom_of_point, frames=None,
                                  with_frames=False, body=None,
                                  numerics=None):
    """(natm, 3) gradient of sum over `gauges` of sum_{gP} D_bar[g,P] D[g,P],
    D = M^T (V + vtilde)^1/2 with M the AO-product fit on `coords`.

    gauges: `GaugeAdjoint`s, or (D_bar, environment) pairs, as in
    `isdf_derivatives.dfactor_adjoint_gauges`: one fit, every gauge's metric
    root. The points are r_g = p_g F_i + R_i over the clouds `pts_local` and
    their owners `atom_of_point`; frozen frames turn with the body frame
    `body` (with_frames False), the chains' convention.
    """
    def seeds(X, M, V):
        M_bar = np.zeros_like(M)
        V_bar = np.zeros_like(V)
        env_grad = np.zeros((mol.natm, 3))
        for gauge in gauges:
            gauge = (gauge if isinstance(gauge, GaugeAdjoint)
                     else GaugeAdjoint(*gauge))
            environment = gauge.environment
            dressed = dresses_interaction(environment, auxmol)
            if not dressed and gauge.kernel_bar is not None:
                raise ValueError(
                    f'{environment!r} dresses nothing, so this gauge has no '
                    'vtilde for a kernel adjoint to land on; it belongs to '
                    'the gauge that carries the environment')
            V_gauge = V + environment.aux_kernel(auxmol) if dressed else V
            root = aux_metric_sqrt(auxmol, environment, V=V)
            M_bar += root @ gauge.d_bar.T
            root_bar = M @ gauge.d_bar
            if gauge.root_bar is not None:
                root_bar = root_bar + gauge.root_bar
            Y = sqrtm_adjoint(V_gauge, root_bar)
            V_bar += Y
            if dressed:
                kernel_bar = (Y if gauge.kernel_bar is None
                              else Y + gauge.kernel_bar)
                env_grad += environment.aux_kernel_adjoint(auxmol, kernel_bar)
        return M_bar, V_bar, None, env_grad

    chain = PointChain(pts_local, atom_of_point, frames, with_frames,
                       body=body)
    return ao_product_fit_adjoint(mol, auxmol, coords, chain, seeds,
                                  numerics=numerics)


def ao_product_fit_adjoint(mol, auxmol, coords, chain, seeds, numerics=None):
    """(natm, 3) gradient of a target that reads the AO-product fit on
    `coords` through its outputs (X, M, V): the fit's reverse pass, its
    integrals and collocation, and the points closed by `chain` (a
    `PointChain`, r_g = p_g F_i + R_i).

    seeds(X, M, V) -> (M_bar, V_bar, X_bar, grad): the target's adjoints on
    M, on the bare metric V (beside the fit's own V^-1) and on the
    collocation X (None for none), and whatever of its gradient it closed
    itself ((natm, 3) or 0) -- the factor gauges'
    (`ao_product_fit_adjoint_gauges`), an interpolated exchange's
    (`exchange_skeleton`). The forward is formed here once and handed to it.
    """
    # the ranks of the enclosing distribution, if any (`distributed`)
    comm = current_comm()
    X = ao_values(mol, coords)
    G, s = balanced_gram(X, numerics)
    d = 1.0 / s
    V = auxmol.intor('int2c2e', aosym='s1')
    E = ao_product_projection(mol, auxmol, X, numerics, comm)
    F = np.linalg.solve(V, E)
    del E
    cho = scipy.linalg.cho_factor(G, lower=True, overwrite_a=True)
    del G
    B = scipy.linalg.cho_solve(cho, (F * d[None, :]).T).T
    M = B * d[None, :]

    M_bar, V_bar, X_seed, extra = seeds(X, M, V)
    V_bar = np.array(V_bar, dtype=float)
    del M

    # --- the fit's solve: B = H G^-1 with H = F d
    B_bar = M_bar * d[None, :]
    d_bar = np.einsum('Pg,Pg->g', M_bar, B)
    del M_bar
    H_bar = scipy.linalg.cho_solve(cho, B_bar.T).T
    del cho, B_bar
    S = -(B.T @ H_bar)                 # G_bar, low rank
    S += S.T                           # G_bar + G_bar^T
    del B
    F_bar = H_bar * d[None, :]
    d_bar += np.einsum('Pg,Pg->g', H_bar, F)
    del H_bar

    # --- the balanced Gram matrix G = (A o A) o d d^T + reg I
    A = X @ X.T
    S *= A                             # (G_bar + G_bar^T) o A
    d_bar += (S * A) @ d
    del A
    S *= d[:, None]
    S *= d[None, :]
    s_bar = -d_bar * d * d             # d = 1/s, s_g = A_gg = |X_g|^2
    X_bar = 2.0 * (S @ X) + 2.0 * s_bar[:, None] * X
    del S
    if X_seed is not None:
        X_bar += X_seed

    # --- F = V^-1 E: the metric and the projection's adjoint
    E_bar = np.linalg.solve(V, F_bar)
    V_bar -= E_bar @ F.T
    del F, F_bar

    # --- E = sum_mn (P|mn) X_gm X_gn: the collocation and the integrals
    grad = _projection_adjoint(mol, auxmol, X, E_bar, X_bar, numerics)
    del E_bar
    grad += two_centre_adjoint(auxmol, V_bar)
    grad += extra
    centre, P = basis_centre_forces(mol, coords, X_bar)
    grad += centre
    grad += chain(mol, P)
    return grad


def _projection_adjoint(mol, auxmol, X, E_bar, X_bar, numerics):
    """(natm, 3) of the three-centre integrals' centres under
    (mn|P)_bar = sum_g E_bar[P, g] X_gm X_gn, and the collocation's share
    2 sum_nP E_bar[P, g] (P|mn) X_gn added into X_bar in place.

    Each rank evaluates its own share of the shell runs (`own_runs`) and
    their derivative integrals, over every point tile. A run's X_bar columns
    and AO-centre terms are its own AOs', disjoint from every other run's, so
    they join exactly; the auxiliary-centre terms are added in run order
    (`ordered_sum`): the force's integral branch is the same bits at every
    rank count.

    pyscf's int3c2e_ip1 and int3c2e_ip2 are +(grad .), so each nuclear
    derivative carries a minus; the second AO's centre gives what the
    first's does, (mn|P)_bar and (mn|P) both being symmetric in (m, n).
    """
    comm = current_comm()
    nao, naux, nk = mol.nao_nr(), auxmol.nao_nr(), X.shape[0]
    Et = np.ascontiguousarray(E_bar.T)                     # (nk, naux)
    x_bar = np.zeros_like(X_bar)
    t_ao = np.zeros((3, nao))
    aux_terms = []
    runs = shell_runs(mol, auxmol, numerics, comp=3)
    mine = own_runs(runs, comm)
    first = runs.index(mine[0]) if mine else 0
    for k, run in enumerate(mine, start=first):
        a0, a1 = run[2], run[3]
        nb = a1 - a0
        operand = half_transform_operand(three_centre_block(mol, auxmol, run))
        # the derivative blocks AO-major, (nb, 3, nao naux) and
        # (nb, 3, nao, naux), so each AO's slab is contiguous
        d1 = np.ascontiguousarray(three_centre_block(
            mol, auxmol, run, 'int3c2e_ip1', 3).transpose(1, 0, 2, 3)
        ).reshape(nb, 3, -1)
        d2 = np.ascontiguousarray(three_centre_block(
            mol, auxmol, run, 'int3c2e_ip2', 3).transpose(1, 0, 2, 3))
        t_aux = np.zeros((3, naux))
        for t0, t1 in point_tiles(nk):
            T = half_transformed(operand, X, t0, t1, nb)
            x_bar[t0:t1, a0:a1] += 2.0 * np.einsum('mPg,gP->gm', T, Et[t0:t1])
            del T
            Xt, Ett = X[t0:t1], Et[t0:t1]
            for m in range(a0, a1):
                # (mn|P)_bar of this tile for the AO m, (nao, naux)
                G3 = (Xt * Xt[:, m:m + 1]).T @ Ett
                t_ao[:, m] -= 2.0 * (d1[m - a0] @ G3.reshape(-1))
                t_aux -= np.einsum('xnP,nP->xP', d2[m - a0], G3)
        aux_terms.append((k, t_aux))
        del operand, d1, d2
    # disjoint columns: a sum of one term and zeros, exact in any order
    reduce_sum(x_bar, comm)
    reduce_sum(t_ao, comm)
    X_bar += x_bar
    aux_total = ordered_sum(aux_terms, comm, shape=(3, naux))
    grad = np.zeros((mol.natm, 3))
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        grad[ia] += t_ao[:, p0:p1].sum(axis=1)
    for ia, (_, _, q0, q1) in enumerate(auxmol.aoslice_by_atom()):
        grad[ia] += aux_total[:, q0:q1].sum(axis=1)
    return grad


def exchange_skeleton(mf, dm, dm_other, prefactor, channels):
    """(natm, 3) of d/dR E_K^ISDF at fixed density, for an ISDF-K mean field
    on the AO-product fit: `isdf_derivatives.isdf_exchange_skeleton`'s
    terms, with the fit's reverse pass `ao_product_fit_adjoint`. Rank 0's on
    every rank.

    Per exchange channel (omega, weight), E = -(w/4) sum Z W1 W2 with
    Z = M^T V_w M: the collocation and Z take their adjoints
    (`isdf_exchange_adjoints`), Z's reaches M on both sides and V_w between;
    the bare metric joins the fit's own V, an attenuated one is
    differentiated apart. The points are the mean field's own: its pivoted
    selection, else its grid's clouds, closed on its frames
    (`mean_field_point_chain`).
    """
    with_df = mf.with_df
    if not any(weight != 0.0 for _, weight in channels):
        return np.zeros((mf.mol.natm, 3))
    mol = with_df.mol
    auxmol = auxmol_of(mf)
    coords = with_df.coords
    if coords is None:
        coords = lockstep(with_df.place_grid()[0])
    if with_df.selection is not None:
        pts_local, owner = with_df.selection
    else:
        pts_local, owner = point_layout(mol, with_df.grid_radii,
                                        with_df.grid_origins)
    chain = mean_field_point_chain(with_df, mol, pts_local, owner)

    def seeds(X, M, V):
        X_bar = np.zeros_like(X)
        M_bar = np.zeros_like(M)
        V_bar = np.zeros_like(V)
        extra = np.zeros((mol.natm, 3))
        for omega, weight in channels:
            if weight == 0.0:
                continue
            if omega == 0.0:
                V_w = V
            else:
                with mol.with_range_coulomb(omega), \
                        auxmol.with_range_coulomb(omega):
                    V_w = auxmol.intor('int2c2e', aosym='s1')
            xb, Z_bar = isdf_exchange_adjoints(X, M.T @ (V_w @ M), dm,
                                               prefactor * weight, dm_other)
            X_bar += xb
            M_bar += 2.0 * (V_w @ M) @ Z_bar
            if omega == 0.0:
                V_bar += M @ Z_bar @ M.T
            else:
                extra += two_centre_adjoint(auxmol, M @ Z_bar @ M.T,
                                            omega=omega)
        return M_bar, V_bar, X_bar, extra

    return lockstep(ao_product_fit_adjoint(mol, auxmol, coords, chain, seeds,
                                           numerics=with_df.numerics))
