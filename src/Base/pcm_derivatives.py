"""Nuclear derivatives of the PCM reaction field for two INDEPENDENT vectors.

pyscf differentiates the solvation energy, `0.5 v^T K^-1 R v`, in which the
same grid potential sits on both sides: `grad/pcm.py:grad_solver` takes no
vector argument at all and reads `v` and `q` out of the PCM object. Every
adjoint of a screened quantity needs the bilinear form instead --

    d/dR [ v_left^T K^-1 R v_right ]        v_left != v_right

-- because the left vector is an adjoint and the right one a density. The two
are the same object only for the mean-field energy.

The generalization is mechanical once the roles are named: in pyscf's terms
`vK_1 = K^-T v` is the LEFT role and `q = K^-1 R v` the RIGHT one, and every
term is already written as `vK_1 ... q`. So the bilinear derivative is that same
expression with the two built from different vectors and the 0.5 that belongs to
the energy removed. `solver_bilinear_gradient(pcm, v, v)` reproduces
`2 * grad_solver(dm)` exactly, which is the gate.

The geometry enters K and R through S, D and A only; `charge_exp`, `norm_vec`,
`weights` and the radii have exactly zero nuclear derivative, and
`grid_coords[k] = atom_coords[owner(k)] + R_vdw[k] * norm_vec[k]` exactly, so a
cavity point moves rigidly with its own atom.

SIZE: `get_dD_dS` materializes (ngrids, ngrids, 3) arrays, so this route holds
three times what the PCM's own K costs -- fine to a few thousand cavity points,
and NOT the production path at a 60-atom emitter, where those are terabytes.
`pyscf.solvent.hessian.pcm`'s `get_dS_dot_q` family contracts the same
derivatives against a vector without ever forming them, and is what a large
system needs; the algebra below is unchanged by that substitution.

TRAP: the retained point set depends on the geometry through `w*switch > 1e-16`,
so `ngrids` can change between displaced geometries. The energy stays continuous
because a dropped point carries no weight, but a finite-difference check must
freeze the point set at the reference geometry -- see `frozen_surface`.
"""
import numpy as np
from pyscf import lib
from pyscf.solvent.grad.pcm import (get_dD_dS, get_dF_dA, grad_nuc, grad_qv,
                                    grad_solver)

from src.Base.constants import PCM_CROSS_TERM_STEP

PI = np.pi

CPCM = ('C-PCM', 'CPCM', 'COSMO')
IEFPCM = ('IEF-PCM', 'IEFPCM')
SSVPE = ('SS(V)PE',)


def by_atom(per_point, gridslice):
    """Sum a (ngrids, 3) point contribution onto the atom each point rides.

    Public because every cavity-derivative consumer needs it: a surface point
    moves rigidly with the atom that owns it, so scattering a per-point
    quantity onto atoms is the last step of each of them.
    """
    return np.asarray([per_point[p0:p1].sum(axis=0) for p0, p1 in gridslice])


def solver_bilinear_gradient(pcmobj, v_left, v_right):
    """(natm, 3) of d/dR [v_left^T K^-1 R v_right], the vectors held fixed.

    The cavity's response to the nuclei, with no assumption that the two sides
    come from one density. `grad_solver`'s own quantity is the v_left = v_right
    case at half this value.
    """
    method = pcmobj.method.upper()
    if not pcmobj._intermediates:
        pcmobj.build()
    inter = pcmobj._intermediates
    gridslice = pcmobj.surface['gslice_by_atom']
    A, D, S, K = inter['A'], inter['D'], inter['S'], inter['K']
    R = inter['R']

    # A batch of vector PAIRS is summed over, so the ngrids^2 derivative
    # intermediates below are built once rather than once per pair -- the
    # auxiliary adjoint needs naux of them.
    v_left = np.atleast_2d(np.asarray(v_left, float))
    v_right = np.atleast_2d(np.asarray(v_right, float))
    if v_left.shape != v_right.shape:
        raise ValueError(f'left and right batches disagree: {v_left.shape} '
                         f'vs {v_right.shape}')
    # The two roles. Everything below is bilinear in exactly these.
    vK_1 = np.linalg.solve(K.T, v_left.T).T
    q = np.linalg.solve(K, R.dot(v_right.T)).T

    dF, dA = get_dF_dA(pcmobj.surface)
    with_D = method in IEFPCM + SSVPE
    dD, dS, dSii = get_dD_dS(pcmobj.surface, dF, with_D=with_D, with_S=True)
    dS = dS.transpose([2, 0, 1])
    dSii = dSii.transpose([2, 0, 1])
    if with_D:
        dD = dD.transpose([2, 0, 1])
        dA_t = dA.transpose([2, 0, 1])

    def split(left, right, dM):
        """d/dR of sum_v left_v^T M right_v; M[i,j] rides i and j oppositely."""
        out = np.einsum('vi,xij,vj->ix', left, dM, right, optimize=True)
        out -= np.einsum('vi,xij,vj->jx', left, dM, right, optimize=True)
        return by_atom(out, gridslice)

    def diagonal(weight, dMii):
        """The self-interaction diagonal, which rides its own atom directly."""
        return np.einsum('vi,xin->nx', weight, dMii, optimize=True)

    de = np.zeros((pcmobj.mol.natm, 3))
    if method in CPCM:
        # K = S, R = -f I: dR vanishes and only dK survives.
        de_dS = split(vK_1, q, dS) + diagonal(vK_1 * q, dSii)
        de -= de_dS                        # dR = 0, so de = -(vK_1^T dK q)
    elif method in IEFPCM:
        f_eps = (pcmobj.eps - 1.0) / (pcmobj.eps + 1.0)
        fac = f_eps / (2.0 * PI)
        DA = D * A

        # dR = fac (dD A + D dA)
        Av = A * v_right
        de_dR = split(vK_1, Av, dD)
        de_dR += diagonal((vK_1 @ D) * v_right, dA_t)
        de_dR *= fac

        # dK = dS - fac (dD A S + D dA S + D A dS)
        de_dS0 = split(vK_1, q, dS) + diagonal(vK_1 * q, dSii)
        vK_1_DA = vK_1 @ DA
        de_dS1 = split(vK_1_DA, q, dS) + diagonal(vK_1_DA * q, dSii)
        Sq = q @ S.T
        de_dD = split(vK_1, A * Sq, dD)
        de_dA = diagonal((vK_1 @ D) * Sq, dA_t)
        de_dK = de_dS0 - fac * (de_dD + de_dA + de_dS1)
        de += de_dR - de_dK
    elif method in SSVPE:
        f_eps = (pcmobj.eps - 1.0) / (pcmobj.eps + 1.0)
        fac_R = f_eps / (2.0 * PI)
        fac_K = f_eps / (4.0 * PI)
        DA = D * A

        Av = A * v_right
        de_dR = split(vK_1, Av, dD)
        de_dR += diagonal((vK_1 @ D) * v_right, dA_t)
        de_dR *= fac_R

        # K = S - fac_K (D A S + (D A S)^T), so every term appears twice, once
        # with the transpose acting on the LEFT role -- which is exactly where a
        # bilinear form stops agreeing with the quadratic one.
        de_dS0 = split(vK_1, q, dS) + diagonal(vK_1 * q, dSii)
        vK_1_DA = vK_1 @ DA
        de_dS1 = split(vK_1_DA, q, dS) + diagonal(vK_1_DA * q, dSii)
        ADT_q = (q @ D) * A
        de_dS1_T = split(vK_1, ADT_q, dS) + diagonal(vK_1 * ADT_q, dSii)

        Sq = q @ S.T
        de_dD = split(vK_1, A * Sq, dD)
        vK_1_S = vK_1 @ S
        de_dD_T = split(vK_1_S * A, q, -dD.transpose(0, 2, 1))

        de_dA = diagonal((vK_1 @ D) * Sq, dA_t)
        de_dA_T = diagonal(vK_1_S * (q @ D), dA_t)

        de_dK = de_dS0 - fac_K * (de_dD + de_dA + de_dS1
                                  + de_dD_T + de_dA_T + de_dS1_T)
        de += de_dR - de_dK
    else:
        raise NotImplementedError(
            f'PCM method {pcmobj.method!r} has no bilinear nuclear derivative '
            f'here; C-PCM, IEF-PCM and SS(V)PE do')
    return de


def frozen_surface(pcmobj):
    """The retained point count, for asserting a stencil did not change it.

    `pcm.py` keeps a cavity point only while `weight * switch > 1e-16`, so a
    displaced geometry can carry a different number of points. The energy is
    continuous across that -- a dropped point contributes nothing -- but a
    finite-difference gate that compares per-point arrays is not, and a changed
    count is the signal to widen the step or move the reference.
    """
    return pcmobj.surface['grid_coords'].shape[0]


def solvation_gradient(pcmobj, dm):
    """(natm, 3) of d/dR of the solvation energy of a FIXED density matrix.

    pyscf's three pieces summed: the derivative of the potential integrals at
    fixed surface charges, the nuclear term, and the cavity solver's own
    response. Each re-solves q = K^-1 R v for whatever density it is handed, so
    this is the energy of `dm` in its OWN reaction field, not of `dm` in the
    mean field's.
    """
    return (np.asarray(grad_qv(pcmobj, dm)) + np.asarray(grad_nuc(pcmobj, dm))
            + np.asarray(grad_solver(pcmobj, dm)))


def reaction_field_fock_skeleton(pcmobj, dm, gamma_ao,
                                 step=PCM_CROSS_TERM_STEP):
    """(natm, 3) of d/dR Tr[gamma_ao V_PCM[dm]], both densities held fixed.

    What a CORRELATED relaxed density owes the ground-state reaction field. A
    mean field's own force answers the SCF density alone; a dRPA or BSE
    gradient also carries gamma, and V_PCM sits in the Fock, so the skeleton of
    Tr[gamma F] has a reaction-field entry exactly as it has a Coulomb one.
    Omitting it leaves a force that is stationary for neither functional --
    4e-04 Ha/Bohr on water/cc-pVDZ in water.

    NO TRUNCATION ERROR. The solvation energy is EXACTLY quadratic in the
    density it is built from, E = (1/2)(v[dm] + v[N]) Q (v[dm] + v[N]) with v
    linear in dm and Q the cavity response, so

        d/dR B[gamma, dm + N] = [G(dm + t.gamma) - G(dm - t.gamma)] / 2t

    holds at ANY t, the quadratic and gamma-independent parts cancelling
    identically. `step` is a conditioning choice, not an accuracy one; measured
    step-independent to 3e-15 from t = 1e-1 to 1e-3.

    COST is two solver gradients, so a correlated force pays pyscf's PCM
    gradient three times over: `grad_solver` materializes the (ngrids, ngrids,
    3) derivatives this module's header sizes, and the two calls here cannot be
    folded into one without losing exactness -- a forward difference leaves
    (t/2) B[gamma, gamma] behind.

    The charge cache is restored to `dm` on the way out, since pyscf's pieces
    leave the PCM object holding the charges of whatever they were last given.
    """
    g = np.asarray(gamma_ao, float)
    g = 0.5 * (g + g.T)
    grad = (solvation_gradient(pcmobj, dm + step * g)
            - solvation_gradient(pcmobj, dm - step * g)) / (2.0 * step)
    pcmobj._get_vind(dm)
    return grad
