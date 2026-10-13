"""Nuclear derivatives of the PCM reaction field for two independent vectors.

pyscf differentiates the solvation energy `0.5 v^T K^-1 R v`, with the same
grid potential on both sides (`grad/pcm.py:grad_solver` reads `v` and `q` out
of the PCM object). An adjoint of a screened quantity needs the bilinear form

    d/dR [ v_left^T K^-1 R v_right ]        v_left != v_right

since the left vector is an adjoint and the right one a density. In pyscf's
terms `vK_1 = K^-T v` is the left role and `q = K^-1 R v` the right one, and
every term is written as `vK_1 ... q`, so the bilinear derivative is the same
expression with the two built from different vectors and without the energy's
0.5. `solver_bilinear_gradient(pcm, v, v)` reproduces `2 * grad_solver(dm)`.

The geometry enters K and R through S, D and A only; `charge_exp`, `norm_vec`,
`weights` and the radii have zero nuclear derivative, and
`grid_coords[k] = atom_coords[owner(k)] + R_vdw[k] * norm_vec[k]`, so a cavity
point moves rigidly with its own atom.

`get_dD_dS` materializes (ngrids, ngrids, 3) arrays, three times the PCM's own
K: fine to a few thousand cavity points, too large beyond that.
`pyscf.solvent.hessian.pcm`'s `get_dS_dot_q` family contracts the same
derivatives against a vector without forming them; the algebra below is
unchanged by that substitution.

The retained point set depends on the geometry through `w*switch > 1e-16`, so
`ngrids` can change between displaced geometries. The energy stays continuous
(a dropped point carries no weight), but a finite-difference check must freeze
the point set at the reference geometry; see `frozen_surface`.
"""
import numpy as np
from pyscf import df as pyscf_df
from pyscf import gto, lib
from pyscf.solvent.grad.pcm import (get_dD_dS, get_dF_dA, grad_nuc,
                                    grad_solver)

from src.Base.constants import ISDF_TILE_GB, PCM_CROSS_TERM_STEP

PI = np.pi

#: The largest (comp, nao, nao, nk) block of three-centre integrals libcint
#: can write: its driver strides component n of the block by
#: n * nao * nao * nk held in a C int, which wraps negative past 2^31 - 1 and
#: puts every component after the first below the array. pyscf blocks its PCM
#: cavity integrals by `max_memory` alone, so with a large memory share a
#: molecule of about a hundred atoms in a double-zeta basis asks for a block
#: past the limit and crashes.
LIBCINT_BLOCK_LIMIT = 2 ** 31 - 1

#: The largest `max_memory` (MB) at which every pyscf PCM routine's own blocks
#: stay under LIBCINT_BLOCK_LIMIT: they take int(max_memory * 0.9e6 / 8 /
#: nao^2 / c) cavity points for a comp-c integral (c = 1, 3 or 9), so
#: nao^2 * nk * comp <= max_memory * 0.9e6 / 8 for all of them.
PYSCF_PCM_MAX_MEMORY_MB = LIBCINT_BLOCK_LIMIT * 8 / 0.9e6

CPCM = ('C-PCM', 'CPCM', 'COSMO')
IEFPCM = ('IEF-PCM', 'IEFPCM')
SSVPE = ('SS(V)PE',)


def by_atom(per_point, gridslice):
    """Sum a (ngrids, 3) per-point contribution onto the atom each point
    moves rigidly with; the last step of every cavity derivative."""
    return np.asarray([per_point[p0:p1].sum(axis=0) for p0, p1 in gridslice])


def cavity_blocks(nao, ngrids, comp=3, budget_gb=ISDF_TILE_GB):
    """[(k0, k1)] slices of the cavity grid, one (comp, nao, nao, k1 - k0)
    three-centre integral block each.

    Every block holds nao^2 nk comp under LIBCINT_BLOCK_LIMIT, the bound that
    keeps libcint's component stride inside a C int, and its array under
    `budget_gb`. The flop count does not depend on the blocking.
    """
    per_point = int(nao) * int(nao) * max(int(comp), 1)
    by_index = LIBCINT_BLOCK_LIMIT // per_point
    if by_index < 1:
        raise ValueError(
            f'one cavity point already needs {per_point} integrals, past the '
            f'{LIBCINT_BLOCK_LIMIT} libcint can index in one block')
    by_memory = int(budget_gb * 1024 ** 3 / (8.0 * per_point))
    nk = max(1, min(by_index, by_memory))
    return [(k0, min(k0 + nk, ngrids)) for k0 in range(0, ngrids, nk)]


def bound_pyscf_pcm_blocks(pcmobj):
    """Cap `pcmobj.max_memory` at PYSCF_PCM_MAX_MEMORY_MB and return it.

    For the PCM routines pyscf runs itself -- the reaction-field gradient of a
    pyscf `Gradients()` object, the PCM Hessian -- whose cavity blocks this
    module does not plan. Their block sizes come from `max_memory` only; below
    the cap no block passes LIBCINT_BLOCK_LIMIT, unless pyscf's floor of 400
    points does it alone (nao above ~1300).
    """
    pcmobj.max_memory = min(float(pcmobj.max_memory), PYSCF_PCM_MAX_MEMORY_MB)
    return pcmobj


def potential_integral_gradient(pcmobj, dm, q_sym=None, budget_gb=ISDF_TILE_GB):
    """(natm, 3) of the surface charges' interaction with the density `dm`,
    differentiated through the integrals at fixed charges: pyscf's
    `grad/pcm.py:grad_qv`, term for term, on `cavity_blocks` (pyscf's own
    blocking by `max_memory` passes LIBCINT_BLOCK_LIMIT at about 90 atoms in
    cc-pVDZ with a large memory budget).
    """
    if not pcmobj._intermediates:
        pcmobj.build()
    dm = np.asarray(dm, float)
    dm_cache = pcmobj._intermediates.get('dm', None)
    if dm_cache is None or np.linalg.norm(dm_cache - dm) >= 1e-10:
        pcmobj._get_vind(dm)
    if q_sym is None:
        q_sym = pcmobj._intermediates['q_sym']
    mol = pcmobj.mol
    nao = mol.nao
    grid_coords = pcmobj.surface['grid_coords']
    exponents = pcmobj.surface['charge_exp']
    ngrids = q_sym.shape[0]
    blocks = cavity_blocks(nao, ngrids, comp=3, budget_gb=budget_gb)

    int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env,
                                         int3c2e_ip1)
    dvj = np.zeros((3, nao))
    for p0, p1 in blocks:
        fakemol = gto.fakemol_for_charges(grid_coords[p0:p1],
                                          expnt=exponents[p0:p1] ** 2)
        v_nj = pyscf_df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1,
                                      aosym='s1', cintopt=cintopt)
        dvj += np.einsum('xijk,ij,k->xi', v_nj, dm, q_sym[p0:p1])
        del v_nj

    int3c2e_ip2 = mol._add_suffix('int3c2e_ip2')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env,
                                         int3c2e_ip2)
    dq = np.empty((3, ngrids))
    for p0, p1 in blocks:
        fakemol = gto.fakemol_for_charges(grid_coords[p0:p1],
                                          expnt=exponents[p0:p1] ** 2)
        q_nj = pyscf_df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip2,
                                      aosym='s1', cintopt=cintopt)
        dq[:, p0:p1] = np.einsum('xijk,ij,k->xk', q_nj, dm, q_sym[p0:p1])
        del q_nj

    gridslice = pcmobj.surface['gslice_by_atom']
    aoslice = mol.aoslice_by_atom()
    dq = np.asarray([np.sum(dq[:, p0:p1], axis=1) for p0, p1 in gridslice])
    dvj = 2.0 * np.asarray([np.sum(dvj[:, p0:p1], axis=1)
                            for p0, p1 in aoslice[:, 2:]])
    return dq + dvj


def solver_bilinear_gradient(pcmobj, v_left, v_right):
    """(natm, 3) of d/dR [v_left^T K^-1 R v_right], the vectors held fixed.

    `grad_solver`'s own quantity is the v_left = v_right case at half this
    value.
    """
    method = pcmobj.method.upper()
    if not pcmobj._intermediates:
        pcmobj.build()
    inter = pcmobj._intermediates
    gridslice = pcmobj.surface['gslice_by_atom']
    A, D, S, K = inter['A'], inter['D'], inter['S'], inter['K']
    R = inter['R']

    # A batch of vector pairs is summed over, so the ngrids^2 derivative
    # intermediates are built once rather than once per pair (the auxiliary
    # adjoint needs naux pairs).
    v_left = np.atleast_2d(np.asarray(v_left, float))
    v_right = np.atleast_2d(np.asarray(v_right, float))
    if v_left.shape != v_right.shape:
        raise ValueError(f'left and right batches disagree: {v_left.shape} '
                         f'vs {v_right.shape}')
    # The two roles; everything below is bilinear in these.
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
        # with the transpose acting on the left role; this is where the
        # bilinear form departs from the quadratic one.
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
    continuous across that, but per-point arrays are not; a changed count
    means the step must be widened or the reference moved.
    """
    return pcmobj.surface['grid_coords'].shape[0]


def solvation_gradient(pcmobj, dm):
    """(natm, 3) of d/dR of the solvation energy of a fixed density matrix.

    pyscf's three pieces summed: potential integrals at fixed surface charges,
    the nuclear term, and the cavity solver's response. Each re-solves
    q = K^-1 R v for the density it is handed, so this is the energy of `dm` in
    its own reaction field, not in the mean field's.
    """
    return (potential_integral_gradient(pcmobj, dm)
            + np.asarray(grad_nuc(pcmobj, dm))
            + np.asarray(grad_solver(pcmobj, dm)))


def reaction_field_fock_skeleton(pcmobj, dm, gamma_ao,
                                 step=PCM_CROSS_TERM_STEP):
    """(natm, 3) of d/dR Tr[gamma_ao V_PCM[dm]], both densities held fixed.

    The reaction-field term a correlated relaxed density gamma (dRPA, BSE)
    owes: V_PCM sits in the Fock, so the skeleton of Tr[gamma F] has a
    reaction-field entry as it has a Coulomb one. Omitting it leaves a force
    stationary for neither functional (4e-04 Ha/Bohr on water/cc-pVDZ in
    water).

    The solvation energy E = (1/2)(v[dm] + v[N]) Q (v[dm] + v[N]) is quadratic
    in the density (v linear in dm, Q the cavity response), so

        d/dR B[gamma, dm + N] = [G(dm + t.gamma) - G(dm - t.gamma)] / 2t

    holds at any t; `step` only sets the conditioning (step-independent to
    3e-15 from t = 1e-1 to 1e-3). A forward difference would leave
    (t/2) B[gamma, gamma] behind, so this costs two solver gradients.

    The PCM charge cache is restored to `dm` on return, since pyscf's pieces
    leave it holding the charges of the last density they were given.
    """
    g = np.asarray(gamma_ao, float)
    g = 0.5 * (g + g.T)
    grad = (solvation_gradient(pcmobj, dm + step * g)
            - solvation_gradient(pcmobj, dm - step * g)) / (2.0 * step)
    pcmobj._get_vind(dm)
    return grad
