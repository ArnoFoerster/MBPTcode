"""Dielectric environment around a SLAB: v -> v + vtilde for periodic GW-BSE.

The periodic counterpart of src/Base/solvent_screening.py. Same one-line
physics -- fold the environment's reducible polarizability into the Coulomb
kernel seen inside the cavity, then do an ordinary gas-phase calculation
(Duchemin, Jacquemin and Blase, J. Chem. Phys. 144, 164106 (2016)) -- but the
cavity is a slab: periodic in-plane, with the environment filling the two
half-spaces |z| > z0.

Why a slab and not bulk
-----------------------
A 3D crystal has no outside, so there is no cavity and nothing to embed. The
meaningful periodic cases are a monolayer in a solvent, a molecule or 2D
material on a dielectric substrate, and 1D wires -- all of which are slabs in
the sense used here: the environment occupies the vacuum region.

The kernel
----------
For a cavity -z0 < z < z0 between semi-infinite dielectrics eps_top (above)
and eps_bot (below), solving Poisson's equation in the in-plane Fourier
representation gives the reaction part of the interaction in closed form. With
k = |q_par + G_par|, s = exp(-2 k z0), and the interface reflection
coefficients b_i = (eps_i - 1)/(eps_i + 1),

    vtilde(k; z, z') = -(2pi/k) (s/D) [ b_top e^{k(z+z')} + b_bot e^{-k(z+z')}
                                        - 2 b_top b_bot s cosh(k(z-z')) ]
    D = 1 - b_top b_bot s^2 .

Limits, all checked in tests/test_pbc_solvent_screening.py:
  * b_bot = 0 collapses to the single image charge at the mirror point
    2 z0 - z, as it must;
  * z = z' = z0 = 0 gives v + vtilde = (2pi/k) 2/(eps_top + eps_bot), the
    textbook effective interaction of a 2D sheet between two dielectrics;
  * eps = 1 gives exactly zero.

Rank two, exactly
-----------------
Every term above factorizes: e^{k(z+z')} = e^{kz} e^{kz'} and the cosh splits
into e^{kz}e^{-kz'} + e^{-kz}e^{kz'}. So in the two-function basis

    u_+(z) = e^{k(z-z0)},   u_-(z) = e^{-k(z+z0)},   |z| < z0

(rescaled by e^{-k z0} so nothing overflows -- the raw exponentials are always
multiplied by s, and s e^{2 k z0} = 1), the kernel is EXACTLY a 2x2 form

    vtilde(k; z, z') = sum_{mn} M_{mn}(k) u_m(z) u_n(z') ,
    M(k) = -(2pi/k)/D [[b_top, -b_top b_bot s], [-b_top b_bot s, b_bot]] .

That is the same V Q V^T shape the molecular kernel has, so it drops into the
periodic RI-V metric as a rank-2 update of J(q) and B(q):

    J~(q) = J(q) + dJ(q),   B~(q) = B(q) + dB(q),   L~ = J~^{-1/2} B~

with, writing Y_{P,m}(G_par) = sum_{G_z} chi_P(q+G)^* w(G) ubar_m(k, G_z),

    dJ_PQ = A sum_{G_par} sum_{mn} M_mn(k) Y_{P,m} Y_{Q,n}^*

and dB the same with the MO pair density in place of chi_Q. One kws per side
is the correct measure: for a translationally invariant kernel it collapses
back to the code's own `chi_P^* v(G) kws chi_Q` (see the derivation note in
`_planar_grid`). Cost is the same order as the existing J build.

Because the whole periodic pipeline reads only PBCDFIntegrals.L, replacing L
by L~ gives screened RPA, screened W^Q, screened BSE A/B blocks, screened
Sigma_c and screened vertex corrections at once -- exactly the property the
molecular version has through the two integral chokepoints.

What this does NOT do
---------------------
  * No ground state. pyscf has no periodic PCM, so the static-eps half of the
    paper's non-equilibrium split has no counterpart here; the orbitals are
    gas-phase (or whatever the mean field was). Only the optical response of
    the environment to the added electron or hole is modelled.
  * Charge leaking out of the cavity. u_+/u_- are truncated at |z| = z0, i.e.
    density outside the cavity is dropped from the reaction field rather than
    being treated as sitting inside the dielectric (which is what the paper's
    Appendix modified IEF-PCM does for the molecular case).
    `leaked_density_fraction` reports how much that is; keep it small.
  * The small-q head. The single in-plane channel with q_par + G_par = 0 is
    DROPPED by default (`head_value=0.0`), the usual periodic G = 0
    convention. It cannot be regularized the way a 3D Coulomb head is: 2pi/k
    is the MIXED-representation kernel, so a value there applies to the whole
    G_par = 0 LINE of the plane-wave grid, not to the single G = 0 point that
    the code's own coulG regularizes. `head_value` is the hook, and
    pbc_smallq supplies what belongs in it -- see that module for the 2D
    forms, the mini-BZ measure and the measurements below.

    What limits it is that vtilde is NEGATIVE, so too large a head drives the
    whitened metric J + dJ indefinite. That ceiling is GEOMETRIC: on this
    module's own test slab it is 6.07, and it does not move when the z-mesh
    goes 72 -> 144 -> 216 points, along with the head channel's entire
    contribution (|d(dJ)/d(head)| = 14.7716 throughout) -- the G-space
    quadrature weight already carries the 1/nGz. It tracks the reaction field
    instead: 13.94 for water alone, 6.07 once a conductor sits below.
    (The number of G_z points is not the mechanism.)

    Measured against that ceiling on a 2x2x1 mesh: the bare mini-BZ average
    <2pi/q> = 53.59 is 8.8x too large and the constant approximation
    2pi/q_min = 15.12 is 2.5x too large, both giving an indefinite metric,
    while the METAL-SCREENED head <2pi/(q+kappa)> = 2.92 at the free-electron
    kappa = 2 sits at 0.48x and is admissible. The head becomes usable exactly
    when metallic screening is included, which is also when it stops being
    optional: switching it on doubles the reaction-field self-energy on that
    slab, +0.309 -> +0.637 eV on the occupied levels. For a metallic slab the
    required form is the sqrt(q) plasmon regime rather than anything
    Drude-like. See tests/test_pbc_smallq.py.
"""
import numpy as np
from pyscf import lib
from pyscf.data import radii as pyscf_radii
from pyscf.pbc import tools
from pyscf.pbc.df import ft_ao

from src.Base.solvent_screening import resolve_optical_eps, solvent_dielectrics
from src.SingleReference.Periodic.pbc_rpa import (make_auxcell, _bz_index,
                                                  coulomb_metric_inv_sqrt)
from src.SingleReference.Periodic.pbc_integrals import (
    PBCDFIntegrals, get_momentum_transfer_map)
from src.SingleReference.Periodic.pbc_rpa_damping import assert_damping_fits

einsum = lib.einsum

#: |k| below which the 2pi/k head is replaced by its Brillouin-zone-cell average.
_K_HEAD_TOL = 1e-8


class SlabDielectricEnvironment:
    """Semi-infinite dielectrics above and below a slab cavity |z| < z0.

    eps / solvent: the OPTICAL dielectric constant of the environment (n^2),
    same rule and the same guard as the molecular SolventScreening -- an added
    electron or hole is too fast for anything but the environment's electrons.
    Give `eps_top`/`eps_bot` instead for an asymmetric setup (a 2D material on
    a substrate with solvent above, say); either may be 1.0 for vacuum.

    z_center / z_half_width define the cavity. Both default from the geometry:
    the centre is the mean atomic z, and the half width is
    max_i (|z_i - z_c| + vdw_scale * R_vdw(Z_i)) -- the same van der Waals
    cavity rule pyscf's PCM uses, projected onto z.
    """

    def __init__(self, cell, eps=None, solvent=None, eps_top=None, eps_bot=None,
                 z_center=None, z_half_width=None, vdw_scale=1.2,
                 allow_static_eps=False):
        if eps_top is None or eps_bot is None:
            eps_sym, eps_static = resolve_optical_eps(eps, solvent, allow_static_eps)
            eps_top = eps_sym if eps_top is None else eps_top
            eps_bot = eps_sym if eps_bot is None else eps_bot
        elif eps is not None or solvent is not None:
            raise ValueError("give either eps/solvent (symmetric) or both "
                             "eps_top and eps_bot, not both forms")
        else:
            eps_static = None
        for name, value in (('eps_top', eps_top), ('eps_bot', eps_bot)):
            if value < 1.0:
                raise ValueError(f"{name} = {value} < 1 is not a dielectric constant")
        # Two facing near-perfect conductors make D = 1 - b_top b_bot s^2 vanish
        # as k -> 0. The physical vtilde stays finite there (v + vtilde -> 0,
        # complete screening), but the individual rank-2 entries diverge like
        # 1/k^2 and cancel only afterwards, so the factorization loses all
        # precision. One metallic side -- the electrode-plus-electrolyte case --
        # keeps D >= 1 - b > 0 and is fine.
        b_top, b_bot = _reflection(eps_top), _reflection(eps_bot)
        if b_top * b_bot > 1 - 1e-6:
            raise ValueError(
                f"eps_top = {eps_top} and eps_bot = {eps_bot} are both "
                f"essentially perfect conductors. The rank-2 factorization is "
                f"numerically unusable there (its two terms diverge as 1/k^2 "
                f"and cancel). Model one side as the metal and the other as "
                f"the electrolyte, which is the physical electrode geometry.")

        self.cell = cell
        self.eps_top = eps_top
        self.eps_bot = eps_bot
        self.eps_static = eps_static
        self.solvent = solvent
        self.vdw_scale = vdw_scale

        coords = cell.atom_coords()          # bohr
        self.z_center = float(np.mean(coords[:, 2])) if z_center is None else float(z_center)
        if z_half_width is None:
            charges = cell.atom_charges()
            rvdw = np.array([pyscf_radii.VDW[int(z)] for z in charges])
            self.z_half_width = float(np.max(np.abs(coords[:, 2] - self.z_center)
                                             + vdw_scale * rvdw))
        else:
            self.z_half_width = float(z_half_width)

        lz = float(np.linalg.norm(cell.a[2]))
        if 2 * self.z_half_width >= lz:
            raise ValueError(
                f"cavity half width {self.z_half_width:.2f} bohr does not fit "
                f"in the {lz:.2f} bohr cell: the dielectric would have zero "
                f"thickness (or wrap onto the slab's own image). Add vacuum, "
                f"or set z_half_width explicitly.")

    # ---- reflection coefficients and the rank-2 kernel --------------------

    @property
    def beta(self):
        """(b_top, b_bot), the interface reflection coefficients seen from
        inside the cavity. Zero for a vacuum side, exactly 1 for a perfect
        conductor (eps = inf) -- the image-charge model of a metal electrode,
        which is the substrate side of a Li/graphite-plus-electrolyte cell."""
        return (_reflection(self.eps_top), _reflection(self.eps_bot))

    def rank2_matrix(self, kpar, head_value):
        """M(k), shape (nk, 2, 2) real symmetric, in the RESCALED u_+/u_- basis.

        head_value replaces 2pi/k wherever k is (numerically) zero.
        """
        kpar = np.asarray(kpar, dtype=float)
        b_top, b_bot = self.beta
        small = kpar < _K_HEAD_TOL
        safe_k = np.where(small, 1.0, kpar)
        two_pi_over_k = np.where(small, head_value, 2.0 * np.pi / safe_k)
        s = np.exp(-2.0 * kpar * self.z_half_width)
        denom = 1.0 - b_top * b_bot * s ** 2

        M = np.empty(kpar.shape + (2, 2))
        pref = -two_pi_over_k / denom
        M[..., 0, 0] = pref * b_top
        M[..., 1, 1] = pref * b_bot
        M[..., 0, 1] = M[..., 1, 0] = pref * (-b_top * b_bot * s)
        return M

    def z_factors(self, kpar, gz):
        """(ubar_+, ubar_-), each (nk, nGz) complex: the 1D Fourier transforms

            ubar_m(k, G_z) = int_{-z0}^{z0} u_m(z) e^{-i G_z z} dz

        of the rescaled, cavity-truncated exponentials, taken about z_center.
        Bounded by 2 z0 in magnitude for every k -- the e^{-k z0} rescaling is
        what keeps this finite where the raw image series would overflow.
        """
        kpar = np.asarray(kpar, dtype=float)[:, None]
        gz = np.asarray(gz, dtype=float)[None, :]
        z0 = self.z_half_width
        phase = np.exp(-1j * gz * self.z_center)      # shift the cavity centre
        s = np.exp(-2.0 * kpar * z0)

        # int e^{k(z-z0)} e^{-i Gz z} dz = [e^{-i Gz z0} - s e^{i Gz z0}]/(k - i Gz),
        # with the removable singularity at k = Gz = 0 (value 2 z0) patched.
        out = []
        for sign in (+1, -1):
            denom = kpar - 1j * sign * gz
            degenerate = np.abs(denom) < _K_HEAD_TOL
            num = np.exp(-1j * sign * gz * z0) - s * np.exp(1j * sign * gz * z0)
            out.append(np.where(degenerate, 2.0 * z0 + 0j,
                                num / np.where(degenerate, 1.0, denom)) * phase)
        return tuple(out)

    def leaked_density_fraction(self, mf):
        """Fraction of the electron density lying outside the cavity, where the
        truncated u_+/u_- drop it from the reaction field. Report it; a few
        percent is normal, a large value means z_half_width is too small."""
        cell = self.cell
        grids_coords, weights = _real_space_grid(cell)
        rho = _density_on_grid(cell, mf, grids_coords)
        outside = np.abs(grids_coords[:, 2] - self.z_center) > self.z_half_width
        total = float(np.dot(rho, weights))
        return float(np.dot(rho[outside], weights[outside]) / total)

    def __repr__(self):
        label = f"solvent={self.solvent!r}, " if self.solvent else ""
        return (f"SlabDielectricEnvironment({label}eps_top={self.eps_top:.4f}, "
                f"eps_bot={self.eps_bot:.4f} (optical), "
                f"cavity z = {self.z_center:.2f} +/- {self.z_half_width:.2f} bohr)")


def _reflection(eps):
    """(eps - 1)/(eps + 1), with the perfect-conductor limit eps = inf -> 1."""
    return 1.0 if np.isinf(eps) else (eps - 1.0) / (eps + 1.0)


def reaction_kernel_mixed(env, kpar, z, zp, head_value=None):
    """vtilde(k; z, z') in closed form, (nk, nz, nz'), for validation and for
    anyone who wants the kernel itself rather than its matrix elements.

    z, z' are absolute coordinates (the cavity centre is subtracted here).
    """
    kpar = np.asarray(kpar, dtype=float)[:, None, None]
    zeta = np.asarray(z, dtype=float)[None, :, None] - env.z_center
    zetap = np.asarray(zp, dtype=float)[None, None, :] - env.z_center
    b_top, b_bot = env.beta
    z0 = env.z_half_width
    small = kpar < _K_HEAD_TOL
    safe = np.where(small, 1.0, kpar)
    two_pi_over_k = (2.0 * np.pi / safe if head_value is None
                     else np.where(small, head_value, 2.0 * np.pi / safe))
    s = np.exp(-2.0 * kpar * z0)
    denom = 1.0 - b_top * b_bot * s ** 2
    return -(two_pi_over_k) * (s / denom) * (
        b_top * np.exp(kpar * (zeta + zetap))
        + b_bot * np.exp(-kpar * (zeta + zetap))
        - 2.0 * b_top * b_bot * s * np.cosh(kpar * (zeta - zetap)))


def reaction_field_form(env, chi_a, chi_b, qtrue, gpar, gz, wz, area, head):
    """A sum_{Gpar} sum_mn M_mn(k) Y^a_m Y^b_n * -- the reaction-field bilinear
    form <a| vtilde |b> between two sets of G-space coefficients.

    chi_a, chi_b: (nG, n) plane-wave coefficients chi(q+G) in the SAME grid
    order as gpar/gz (in-plane index slowest). Returns (n_a, n_b).
    """
    n_par, n_z = len(gpar), len(gz)
    kpar = np.linalg.norm(qtrue[None, :2] + gpar[:, :2], axis=1)
    M = env.rank2_matrix(kpar, head)
    u_plus, u_minus = env.z_factors(kpar, gz)
    wu = np.stack([u_plus, u_minus], axis=1) * wz[None, None, :]   # (nGpar,2,nGz)

    a3 = chi_a.reshape(n_par, n_z, -1)
    b3 = chi_b.reshape(n_par, n_z, -1)
    Ya = einsum('gza,gmz->gma', a3.conj(), wu)
    Yb = einsum('gzb,gmz->gmb', b3, wu.conj())
    return area * einsum('gmn,gma,gnb->ab', M, Ya, Yb)


def _real_space_grid(cell):
    from pyscf.pbc.dft import gen_grid
    grids = gen_grid.UniformGrids(cell)
    grids.build()
    return grids.coords, grids.weights


def _density_on_grid(cell, mf, coords):
    from pyscf.pbc.dft import numint
    dm = np.asarray(mf.make_rdm1())
    if dm.ndim == 4:                       # (spin, nkpts, nao, nao)
        dm = dm.sum(axis=0)
    ao_kpts = numint.eval_ao_kpts(cell, coords, kpts=np.asarray(mf.kpts))
    rho = np.zeros(len(coords))
    for k, ao in enumerate(ao_kpts):
        rho += np.einsum('gp,pq,gq->g', ao.conj(), dm[k], ao).real
    return rho / len(ao_kpts)


def _planar_grid(cell, mesh):
    """Split the plane-wave grid into (in-plane, z) axes.

    Returns (gpar, gz, weights_z, shape) with gpar (nGpar, 3) the in-plane part
    of each G, gz (nGz,) the z components, and weights_z (nGz,) the G-space
    quadrature weight (pyscf's kws, which for a slab depends on G_z only).

    Measure note. The periodic builders evaluate a translationally invariant
    kernel as `sum_G chi_P(G)^* v(G) kws(G) chi_Q(G)`, i.e. kws is the Parseval
    measure of this discretization (kws = w_quad(G_z)/(2 pi A) for
    dimension=2 / inf_vacuum). A two-point kernel therefore carries one kws per
    side, and that convention collapses back to the diagonal one exactly:
    a translationally invariant kernel has double transform
    A delta_{GparGpar'} delta_{GzGz'} 2 pi v(G)/w_quad, and
    kws^2 * A * 2 pi / w_quad = kws.
    """
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    shape = tuple(len(g) for g in Gvbase)
    if int(np.prod(shape)) != len(Gv):
        raise RuntimeError(f"plane-wave grid {shape} does not match nG={len(Gv)}")
    G3 = Gv.reshape(shape + (3,))
    w3 = np.broadcast_to(np.asarray(kws, dtype=float).reshape(-1), (len(Gv),)).reshape(shape)

    # The slab construction needs the LAST grid axis to be the vacuum (z)
    # direction and the other two to be strictly in-plane, so that the kernel
    # -- diagonal in G_par, a matrix in G_z -- separates. pyscf orders the
    # non-periodic direction last for dimension < 3, but a tilted cell would
    # break it silently, so check rather than assume.
    if np.abs(G3[:, :, :, :2] - G3[:, :, :1, :2]).max() > 1e-10:
        raise ValueError("the in-plane part of G varies along the third grid "
                         "axis: the cell's third lattice vector is not "
                         "perpendicular to the other two. Use an orthogonal "
                         "slab cell with the vacuum along a3.")
    if np.abs(G3[:, :, :, 2] - G3[:1, :1, :, 2]).max() > 1e-10:
        raise ValueError("G_z varies with the in-plane grid indices; the slab "
                         "kernel needs a separable (G_par, G_z) grid.")
    if np.abs(w3 - w3[:1, :1, :]).max() > 1e-14 * max(w3.max(), 1.0):
        raise ValueError("the G-space quadrature weight is not a function of "
                         "G_z alone; the slab kernel assumes it is.")

    gpar = G3[:, :, 0, :].reshape(-1, 3).copy()
    gpar[:, 2] = 0.0
    return gpar, G3[0, 0, :, 2].copy(), w3[0, 0, :].copy(), shape


def build_dfintegrals_screened(mf, env, coulG_fn=None, auxbasis='weigend',
                              mesh=None, head_value=0.0, check_support=True):
    """PBCDFIntegrals whose interaction is v + vtilde: the periodic twin of
    attach_solvent_screening.

    Structurally identical to build_dfintegrals_coulG (same L^q convention,
    same J^{-1/2} whitening), with the environment's rank-2 reaction field
    added to BOTH the auxiliary metric J(q) and the three-center B(q), so the
    RI-V fit is consistent in the screened kernel. Pass the same `coulG_fn`
    you would use in the gas phase -- a slab needs a non-negative kernel, so
    `cell.low_dim_ft_type='inf_vacuum'` or a damped kernel
    (pbc_rpa_damping.make_coulG_damped), see coulomb_metric_inv_sqrt.

    head_value replaces 2pi/k in the single in-plane channel with
    q_par + G_par = 0. The default 0.0 drops that channel; see the module
    docstring for why no naive regularization works there.
    """
    cell = mf.cell
    if mesh is not None:
        cell = cell.copy()
        cell.mesh = mesh
        cell.build(False, False)
    if coulG_fn is None:
        coulG_fn = lambda c, q, Gv: tools.get_coulG(c, k=q, mesh=c.mesh, Gv=Gv)
    if env.cell.nao_nr() != cell.nao_nr():
        raise ValueError("the environment was built for a different cell")

    kpts = np.asarray(mf.kpts)
    # A damped kernel's real-space support must still fit the cell along
    # every unsampled direction at THIS k-mesh -- r0 grows with the grid while
    # the vacuum does not, so a slab that was fine on a coarse mesh can silently
    # stop being fine on a finer one. No-op for any non-damped kernel.
    if check_support:
        assert_damping_fits(cell, kpts, coulG_fn)
    nkpts = len(kpts)
    mo = np.asarray(mf.mo_coeff)
    nmo = mo.shape[-1]
    aux = make_auxcell(cell, auxbasis)
    naux = aux.nao_nr()
    kscaled = cell.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]
    if np.abs(kscaled[:, 2]).max() > 1e-8:
        raise ValueError("the slab kernel needs a single k-point along the "
                         "vacuum direction (k_z = 0 for every k-point).")
    b = cell.reciprocal_vectors()
    Gv, _, kws = cell.get_Gv_weights(cell.mesh)
    kws = np.asarray(kws)
    nG = len(Gv)

    gpar, gz, wz, shape = _planar_grid(cell, cell.mesh)
    area = cell.vol / np.linalg.norm(cell.a[2])
    kconserv_pair = get_momentum_transfer_map(cell, kpts)

    L = []
    for iq in range(nkpts):
        qs = kscaled[iq]
        J = np.zeros((naux, naux), dtype=np.complex128)
        Bf = np.zeros((naux, nkpts, nmo, nmo), dtype=np.complex128)
        for ki in range(nkpts):
            kj, G0 = _bz_index(kscaled, kscaled[ki] + qs)
            qtrue = (kscaled[kj] + G0 - kscaled[ki]).dot(b)
            vG = coulG_fn(cell, qtrue, Gv) * kws
            auxG = ft_ao.ft_ao(aux, Gv, kpt=qtrue)
            pqG = ft_ao.ft_aopair(cell, Gv, aosym='s1',
                                  kpti_kptj=(kpts[ki], kscaled[kj].dot(b)),
                                  q=qtrue).reshape(nG, nmo, nmo)
            rho = einsum('gpq,pP,qR->gPR', pqG, mo[ki].conj(), mo[kj])
            wA = auxG.conj() * vG[:, None]
            J += einsum('gP,gQ->PQ', wA, auxG)
            Bf[:, ki] += einsum('gP,gpr->Ppr', wA, rho)

            # --- the reaction field, rank 2 per in-plane momentum ----------
            J += reaction_field_form(env, auxG, auxG, qtrue, gpar, gz, wz,
                                     area, head_value)
            Bf[:, ki] += reaction_field_form(
                env, auxG, rho.reshape(nG, nmo * nmo), qtrue, gpar, gz, wz,
                area, head_value).reshape(naux, nmo, nmo)

        J /= nkpts
        try:
            Jm12 = coulomb_metric_inv_sqrt(J)
        except ValueError as err:
            # v + vtilde is a positive kernel, so an indefinite metric here has
            # exactly two sources, and which one it is depends on whether the
            # head channel was switched on.
            if head_value:
                # The head is negative and unbounded from below in its effect
                # on J + dJ; too large a value over-screens the single
                # q_par + G_par = 0 channel. The ceiling is geometric -- it
                # tracks the reaction field's strength, not the mesh -- so the
                # fix is a SCREENED head, not a finer grid. See pbc_smallq.
                raise ValueError(
                    f"the screened auxiliary metric at momentum transfer "
                    f"q={iq} is not positive definite, and head_value="
                    f"{head_value:.4g} is the likely cause: the reaction field "
                    f"is negative, so an over-large head over-screens the "
                    f"single q_par + G_par = 0 channel. A BARE head "
                    f"(<2pi/q> over the mini-BZ) exceeds the admissible "
                    f"ceiling on every mesh measured; a metal-screened one, "
                    f"pbc_smallq.slab_head_value(cell, kmesh, kappa), does "
                    f"not. Refining the mesh does not help -- the ceiling is "
                    f"geometric. Pass head_value=0.0 to drop the channel "
                    f"instead.\nUnderlying: {err}") from err
            # Otherwise it is the cavity truncation showing: either the cavity
            # does not enclose the density (u_+/u_- drop what leaks out, and
            # the reaction field then no longer matches the direct term it has
            # to dominate), or the G_z grid is too coarse to resolve the
            # truncation at |z - z_center| = z_half_width. Both are fixable
            # inputs, and both are silent without this check.
            raise ValueError(
                f"the screened auxiliary metric at momentum transfer q={iq} is "
                f"not positive definite. For the slab kernel that means the "
                f"cavity (z = {env.z_center:.2f} +/- {env.z_half_width:.2f} "
                f"bohr) is too tight for the density, or the vacuum/mesh along "
                f"a3 (L = {np.linalg.norm(cell.a[2]):.1f} bohr, "
                f"{cell.mesh[2]} points) does not resolve its edge. Check "
                f"env.leaked_density_fraction(mf) -- keep it well under a "
                f"percent -- then widen z_half_width or refine mesh[2].\n"
                f"Underlying: {err}") from err
        Lhalf = einsum('PQ,Qkpr->Pkpr', Jm12, Bf)
        L.append(np.ascontiguousarray(Lhalf.transpose(1, 0, 2, 3)))

    return PBCDFIntegrals(L, kconserv_pair, mf.mo_energy, mf.mo_occ, kpts,
                          cell=cell, mf=mf)


def solvent_cohsex_kpts(dfints_bare, dfints_screened):
    """Static COHSEX reaction-field operator per k-point, (nkpts, nmo, nmo).

    The periodic form of SolventScreening.cohsex_correction, and needed for the
    same reason: the mean field underneath carries BARE exchange, so the
    first-order reaction field -- which is where essentially all of the
    polarization energy lives -- is not generated by the v -> v + vtilde
    substitution alone.

        Sigma^solv_{k,pq} = 1/2 (1/nk) sum_k' ( sum_a^virt - sum_i^occ )
                            (p_k n_k' | vtilde | n_k' q_k)

    with the vtilde-only four-index object taken as the difference of the two
    factorizations, (..|v+vtilde|..) - (..|v|..) = L~^T L~ - L^T L, so no new
    integral machinery is needed. Pass the SAME auxbasis/mesh/coulG_fn to both
    builders or the difference is contaminated by the RI error.

    Add the diagonal to `exchange_minus_vxc` in qp_energy_g0w0.

    The result is symmetrized explicitly. vtilde is a real symmetric kernel, so
    Sigma^solv is Hermitian by construction; what symmetrizing removes is the
    plane-wave mesh error of the G-space RI-V build underneath, which breaks
    L^{k2,k1}_{qp} = conj(L^{k1,k2}_{pq}) at the same level in the bare and the
    screened factors alike (measured on a slab: 4.8e-4 relative at mesh
    [13,13,72], 5.0e-6 at [19,19,108], 2.4e-7 at [25,25,144] -- clean
    convergence, and 8e-14 for the non-G-space PBCDFIntegrals.from_scf route).
    Symmetrizing is not a substitute for converging that mesh.
    """
    if dfints_bare.nkpts != dfints_screened.nkpts:
        raise ValueError("the two integral sets have different k-meshes")
    nkpts, nmo = dfints_bare.nkpts, dfints_bare.nmo
    nocc = dfints_bare.nocc
    sigma = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
    for kn in range(nkpts):
        for kp in range(nkpts):
            for dfints, sign in ((dfints_screened, 1.0), (dfints_bare, -1.0)):
                L1 = dfints.Lblock(kn, kp)          # (naux, nmo, nmo)
                L2 = dfints.Lblock(kp, kn)
                occ = slice(0, nocc[kp])
                virt = slice(nocc[kp], nmo)
                term = (einsum('Ppa,Paq->pq', L1[:, :, virt], L2[:, virt, :])
                        - einsum('Ppi,Piq->pq', L1[:, :, occ], L2[:, occ, :]))
                sigma[kn] += sign * 0.5 * term / nkpts
    return 0.5 * (sigma + sigma.conj().transpose(0, 2, 1))
