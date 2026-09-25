"""Induced dipoles on classical polarizable sites: the response a QM/MMPol
environment screens with.

A site k carries an isotropic polarizability alpha_k and responds to the field
at its position with a dipole. The dipoles see each other, so they are coupled:

    mu_k = alpha_k ( E_k + sum_{k' != k} T_kk' mu_k' )
    (alpha^-1 - T) mu = E,        B = (alpha^-1 - T)^-1

B is the classical response matrix, (3N, 3N), symmetric, and the whole of what
the environment contributes to a screened interaction:

    vtilde(r, r') = - sum_kk' grad_k v(r) B_kk' grad_k' v(r')

which is the polarizable-embedding analogue of the PCM reaction field, with a
FIELD (dipole) response where the continuum has a CHARGE (surface) one. The
same rank structure follows, so the same cubic adjoint does. The sign is the
reaction field's: the dipole a charge induces attracts that charge, so
vtilde(r, r) = -E^T B E < 0 and v + vtilde is a REDUCED interaction, negative
semidefinite exactly as the continuum's v chi v is.

    T_kk' = lambda5 * 3 r r^T / r^5 - lambda3 * I / r^3,   r = R_k' - R_k

THE DAMPING IS NOT OPTIONAL. Undamped, two sites closer than
(4 alpha_k alpha_k')^(1/6) make (alpha^-1 - T) singular and the induced dipoles
diverge -- the polarization catastrophe (Thole, Chem. Phys. 59, 341 (1981)).
The exponential damping smears each dipole over a width set by its own
polarizability,

    u = r / (alpha_k alpha_k')^(1/6),   lambda3 = 1 - exp(-a u^3),
    lambda5 = 1 - (1 + a u^3) exp(-a u^3)

`a` is the Thole factor; 2.5874 is the value cppe and the standard polarizable
force fields use.

WHAT THE DAMPING DOES AND DOES NOT DO. It keeps T BOUNDED: on two sites of
alpha = 5 at 0.5 Bohr the undamped T_zz is 16.0 and the damped one -0.45. It
does NOT make (alpha^-1 - T) positive definite at every separation -- that pair
stays indefinite below about 2.2 Bohr either way. Real sites sit far outside
that, so the response is stable where it is used, but a site list that puts two
polarizable centres inside ~1 Angstrom of each other is describing something
the model cannot represent, damped or not.

THE SITE-TO-QM DISTANCE IS THE SECOND CATASTROPHE, and no damping touches it:
the field integrals between a site and the QM charge are bare. The folding of W
onto the QM region is exact only because the two subsystems' orbitals do not
overlap, so a site the QM density reaches sits outside the model altogether. It
is refused against MIN_SITE_TO_QM_DISTANCE rather than left to appear
downstream as an indefinite v + vtilde.

WHAT THE SITES ARE WORTH IS SET BY WHAT THEY ARE CALIBRATED TO. A site set
standing in for a molecule has to carry that molecule's polarizability, at the
level of response the route it screens for is built from: the direct RPA for a
GW calculation, not the CPHF value a finite field gives. The two differ by a
third to a half in a small basis, and the difference is the whole error of a
QM/MM partition -- `rpa_polarizability` against `finite_field_polarizability`.
"""
import numpy as np
from pyscf import ao2mo, gto, scf
from scipy.optimize import brentq

from src.Base.constants import (BOHR_TO_ANGSTROM, MIN_SITE_TO_QM_DISTANCE,
                                POLARIZABILITY_FIELD, POLARIZABILITY_SCF_TOL)
from src.Base.separable_ri import auxmol_key
from src.Base.utils.threads import blas_single_threaded

#: Thole exponential-damping factor, dimensionless.
THOLE_FACTOR = 2.5874
#: Below this separation in Bohr two sites are treated as coincident, which no
#: damping can rescue -- a site list with a genuine duplicate is an input error.
MIN_SITE_SEPARATION = 1e-6


def dipole_interaction_matrix(coords, alphas, thole=THOLE_FACTOR):
    """T, the (3N, 3N) damped dipole-dipole coupling; zero on the diagonal blocks.

    thole=None removes the damping, which is correct only for sites far enough
    apart that it does nothing and is otherwise the catastrophe above.

    Built as one (n, n, 3, 3) broadcast rather than a double loop: the loop is
    fine for the handful of sites a test uses and hopeless for the hundreds a
    real site list carries. The n^2 working set is the ceiling -- 72 MB at a
    thousand sites, 1.8 GB at five thousand -- so a genuinely large environment
    wants blocking, which this does not yet do.
    """
    coords = np.asarray(coords, float).reshape(-1, 3)
    alphas = np.asarray(alphas, float).reshape(-1)
    n = len(coords)
    if len(alphas) != n:
        raise ValueError(f'{n} sites but {len(alphas)} polarizabilities')
    if np.any(alphas <= 0.0):
        raise ValueError('a polarizability must be positive')

    r = coords[None, :, :] - coords[:, None, :]            # r[k, k'] = R_k' - R_k
    d = np.linalg.norm(r, axis=2)
    off = ~np.eye(n, dtype=bool)
    if n > 1 and d[off].min() < MIN_SITE_SEPARATION:
        k, kp = np.unravel_index(np.where(off, d, np.inf).argmin(), d.shape)
        raise ValueError(f'sites {k} and {kp} are {d[k, kp]:.2e} Bohr apart; '
                         'no damping makes that finite')
    # The self term is masked at the END rather than by putting inf on the
    # diagonal: inf kills 1/d^3 as intended but sends lambda5 through
    # (1 + inf) * exp(-inf) = inf * 0 = NaN, which then poisons the inverse.
    d = np.where(off, d, 1.0)

    l3 = l5 = 1.0
    if thole is not None:
        au3 = thole * (d / np.outer(alphas, alphas) ** (1.0 / 6.0)) ** 3
        e = np.exp(-au3)
        l3, l5 = (1.0 - e)[..., None, None], (1.0 - (1.0 + au3) * e)[..., None, None]
    d3, d5 = (d ** 3)[..., None, None], (d ** 5)[..., None, None]
    T = (l5 * 3.0 * r[:, :, :, None] * r[:, :, None, :] / d5
         - l3 * np.eye(3) / d3)                            # (n, n, 3, 3)
    T *= off[:, :, None, None]
    return np.transpose(T, (0, 2, 1, 3)).reshape(3 * n, 3 * n)


def closest_pair(coords):
    """(k, k', distance) of the two closest sites, or (None, None, inf) below two."""
    coords = np.asarray(coords, float).reshape(-1, 3)
    if len(coords) < 2:
        return None, None, np.inf
    d = np.linalg.norm(coords[None, :, :] - coords[:, None, :], axis=2)
    np.fill_diagonal(d, np.inf)
    k, kp = np.unravel_index(d.argmin(), d.shape)
    return int(k), int(kp), float(d[k, kp])


def response_matrix(coords, alphas, thole=THOLE_FACTOR):
    """B = (alpha^-1 - T)^-1, the (3N, 3N) classical response.

    Symmetric because T is, and positive definite only where the sites are far
    enough apart for the coupling to stay under alpha^-1, which every physical
    site list satisfies and which the damping alone does not guarantee. Past
    that point the induced dipoles reinforce one another without bound -- the
    polarization catastrophe -- and the inverse of the indefinite matrix is
    not a response: it answers some fields with a dipole ANTIPARALLEL to them
    and lowers no energy, so it is refused rather than returned.
    """
    coords = np.asarray(coords, float).reshape(-1, 3)
    alphas = np.asarray(alphas, float).reshape(-1)
    T = dipole_interaction_matrix(coords, alphas, thole=thole)
    inv_alpha = np.repeat(1.0 / alphas, 3)
    M = np.diag(inv_alpha) - T
    M = 0.5 * (M + M.T)
    try:
        np.linalg.cholesky(M)                    # the definiteness test itself
    except np.linalg.LinAlgError:
        k, kp, d = closest_pair(coords)
        raise ValueError(
            f'alpha^-1 - T is indefinite (smallest eigenvalue '
            f'{np.linalg.eigvalsh(M).min():.3e}), so its inverse is not a '
            f'response: the induced dipoles run away. Closest pair: sites '
            f'{k} and {kp} at {d:.3f} Bohr with alphas {alphas[k]:.3f} and '
            f'{alphas[kp]:.3f} Bohr^3, against the undamped catastrophe at '
            f'{(4.0 * alphas[k] * alphas[kp]) ** (1.0 / 6.0):.3f} Bohr'
        ) from None
    return np.linalg.inv(M)


def check_site_clearance(coords, centres, minimum=MIN_SITE_TO_QM_DISTANCE):
    """Refuse a site within `minimum` Bohr of a QM centre, naming the pair.

    The folding of W onto the QM region is exact only because the two
    subsystems' orbitals do not overlap (Li, D'Avino, Duchemin, Beljonne and
    Blase, Phys. Rev. B 97, 035108 (2018), Sec. II C). Nothing damps the field
    integral between a site and the QM charge the way Thole damping bounds the
    site-site coupling, so a site the QM density reaches polarizes against a
    field the model has no physics for: the polarization catastrophe of a
    QM/MMPol site list, seen downstream as an indefinite v + vtilde.
    """
    coords = np.asarray(coords, float).reshape(-1, 3)
    centres = np.asarray(centres, float).reshape(-1, 3)
    if len(coords) == 0 or len(centres) == 0:
        return
    d = np.linalg.norm(coords[:, None, :] - centres[None, :, :], axis=2)
    if d.min() >= minimum:
        return
    k, ia = np.unravel_index(d.argmin(), d.shape)
    raise ValueError(
        f'site {k} is {d[k, ia]:.3f} Bohr from QM centre {ia} at '
        f'{np.array2string(centres[ia], precision=3)} Bohr, inside the '
        f'{minimum:.3f} Bohr the non-overlap of the QM and MM orbitals needs; '
        f'the site-to-QM field integral is undamped, so the model has no '
        f'physics there')


def induced_dipoles(coords, alphas, field, thole=THOLE_FACTOR):
    """mu = B E for a field given as (N, 3) or flat, returned as (N, 3)."""
    B = response_matrix(coords, alphas, thole=thole)
    mu = B @ np.asarray(field, float).reshape(-1)
    return mu.reshape(-1, 3)


def polarization_energy(coords, alphas, field, thole=THOLE_FACTOR):
    """-1/2 E^T B E, the energy of polarizing the sites in a fixed field.

    Negative for every field wherever B is positive definite, which is the
    statement that polarizing an environment lowers the energy. Inside the
    separation where B loses definiteness the sign is not guaranteed, and
    neither is the model.
    """
    e = np.asarray(field, float).reshape(-1)
    return float(-0.5 * e @ response_matrix(coords, alphas, thole=thole) @ e)


def site_field(auxmol, coords):
    """F[P, k, x]: the field at site k along x from auxiliary function chi_P.

    The field is minus the gradient of the potential with respect to the SITE,
    E(R_k) = -grad_{R_k} (chi_P | 1/|r - R_k|), which is what pyscf's
    `int2c2e_ip1` returns when the charges are its first argument -- verified
    against a finite difference of the potential to 4e-11.
    """
    fake = gto.fakemol_for_charges(np.asarray(coords, float).reshape(-1, 3))
    raw = gto.mole.intor_cross('int2c2e_ip1', fake, auxmol)   # (3, nsite, naux)
    return np.transpose(raw, (2, 1, 0))                       # (naux, nsite, 3)


def site_field_aux_gradient(auxmol, coords):
    """dF[P, k, x] / dR[aux centre, y], as (3, naux, nsite, 3) with y first.

    The mixed second derivative of the same two-centre integral. `int2c2e_ipip1`
    differentiates its FIRST argument twice, so with the auxiliary molecule
    there one index is the auxiliary centre and the other is the field
    direction -- verified against a finite difference of `site_field` under a
    nuclear displacement to 1e-11.
    """
    fake = gto.fakemol_for_charges(np.asarray(coords, float).reshape(-1, 3))
    raw = gto.mole.intor_cross('int2c2e_ipip1', auxmol, fake)  # (9, naux, nsite)
    n_aux, n_site = raw.shape[1], raw.shape[2]
    return np.transpose(raw.reshape(3, 3, n_aux, n_site), (0, 2, 3, 1))


def collective_polarizability(coords, alphas, thole=THOLE_FACTOR):
    """The site set's (3, 3) collective polarizability tensor, in Bohr^3.

    Column y is the total dipole the COUPLED sites answer a unit uniform field
    along y with. It is not diag(sum(alpha)): the dipole-dipole coupling raises
    the isotropic average above the sum at second order in alpha/r^3, and every
    bit of the anisotropy comes from the coupling, since isotropic sites summed
    without it can only be isotropic.
    """
    alphas = np.asarray(alphas, float).reshape(-1)
    B = response_matrix(coords, alphas, thole=thole)
    tensor = np.zeros((3, 3))
    for y in range(3):
        field = np.zeros((len(alphas), 3))
        field[:, y] = 1.0
        tensor[:, y] = (B @ field.reshape(-1)).reshape(-1, 3).sum(axis=0)
    return tensor


def isotropic_polarizability(coords, alphas, thole=THOLE_FACTOR):
    """(1/3) Tr of the collective polarizability tensor, in Bohr^3."""
    return float(np.trace(collective_polarizability(coords, alphas,
                                                    thole=thole)) / 3.0)


def calibrated_site_alphas(coords, target, thole=THOLE_FACTOR, weights=None):
    """Isotropic site polarizabilities on `coords`, one common scale, whose
    coupled isotropic response is `target` Bohr^3.

    A classical model of a molecule is only consistent with the QM molecule it
    replaces if it carries that molecule's polarizability (Li, D'Avino,
    Duchemin, Beljonne and Blase, J. Phys. Chem. Lett. 7, 2814 (2016): "both
    models ... take as input the molecular polarizability tensor computed at
    the desired level of accuracy"). Dividing the target among the sites
    overshoots, because the sites see each other; the scale is found by a root
    solve on the coupled response instead. `weights` sets the relative sizes,
    equal by default.
    """
    coords = np.asarray(coords, float).reshape(-1, 3)
    w = np.ones(len(coords)) if weights is None else np.asarray(weights, float)
    if target <= 0.0:
        raise ValueError(f'a polarizability target of {target} is not positive')

    def residual(scale):
        return isotropic_polarizability(coords, scale * w, thole=thole) - target

    # the coupling can only add to the sum, so sum(alpha) = target brackets
    # the root from above; halving walks the lower end down to an uncoupled set
    hi = target / w.sum()
    while residual(hi) < 0.0:
        hi *= 1.5
    lo = 0.5 * hi
    while residual(lo) > 0.0:
        lo *= 0.5
    return brentq(residual, lo, hi) * w


def finite_field_polarizability(mol, field=POLARIZABILITY_FIELD,
                                conv_tol=POLARIZABILITY_SCF_TOL):
    """The (3, 3) HF dipole polarizability of `mol` in Bohr^3, by finite field.

        alpha_xy = d mu_x / d F_y

    with the uniform field entering the electronic Hamiltonian as +F.r (an
    electron carries charge -1) and the dipole taken as a central difference,
    which cancels the nuclear term and every odd order in F.

    A relaxed HF density answers the field, so this is the CPHF (equivalently
    TDHF at zero frequency) response, exchange kernel included -- NOT the
    response a GW calculation screens with. See `rpa_polarizability`.
    """
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        r = mol.intor('int1e_r', comp=3)
    h0 = scf.RHF(mol).get_hcore()
    alpha = np.zeros((3, 3))
    for y in range(3):
        dipoles = []
        for sign in (1.0, -1.0):
            mf = scf.RHF(mol)
            mf.get_hcore = lambda *args, s=sign, c=y, **kwargs: h0 + s * field * r[c]
            mf.conv_tol = conv_tol
            with blas_single_threaded():
                mf.kernel()
            if not mf.converged:
                raise RuntimeError('the finite-field SCF did not converge; no '
                                   'polarizability follows from it')
            dipoles.append(np.asarray(mf.dip_moment(mol, mf.make_rdm1(),
                                                    unit='AU', verbose=0)))
        alpha[:, y] = (dipoles[0] - dipoles[1]) / (2.0 * field)
    return alpha


def rpa_polarizability(mol, conv_tol=POLARIZABILITY_SCF_TOL):
    """The (3, 3) static direct-RPA dipole polarizability of `mol`, in Bohr^3.

        alpha = 4 d^T (A + B)^-1 d,
        (A + B)_{ia,jb} = delta_ij delta_ab (eps_a - eps_i) + 4 (ia|jb)

    the zero-frequency limit of the Hartree-only (exchange-free) response: the
    same chi0 -> chi the screened interaction W of a GW calculation is built
    from, and therefore the response a classical site set must carry if it is
    to stand in for this molecule inside one. The exchange kernel of the CPHF
    response is not small -- it raises alpha by 39 % for Ar/cc-pVDZ and 61 %
    for water/cc-pVDZ -- so the two levels are not interchangeable.
    """
    mf = scf.RHF(mol)
    mf.conv_tol = conv_tol
    with blas_single_threaded():
        mf.kernel()
    if not mf.converged:
        raise RuntimeError('the SCF did not converge; no response follows '
                           'from its orbitals')
    nocc = mol.nelectron // 2
    occ, virt = mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]
    de = (mf.mo_energy[nocc:][None, :] - mf.mo_energy[:nocc][:, None]).reshape(-1)
    ov = ao2mo.general(mol, [occ, virt, occ, virt],
                       compact=False).reshape(len(de), len(de))
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        d = np.einsum('mp,xmn,nq->xpq', occ, mol.intor('int1e_r', comp=3),
                      virt).reshape(3, -1)
    return 4.0 * d @ np.linalg.solve(np.diag(de) + 4.0 * ov, d.T)


def sites_for_molecule(mol, target=None, thole=THOLE_FACTOR, heavy_only=True):
    """(coords in Bohr, alphas) of a site set carrying `target` Bohr^3 of
    isotropic polarizability on one site per heavy atom.

    A hydrogen carries no site of its own by default; its response is folded
    into the heavy atom it hangs off, as an induced-dipole model of a molecular
    crystal does. This is the calibration that makes a molecule's QM and MM
    descriptions interchangeable, so that moving it across the QM/MM partition
    moves its polarizability from chi0_11 to chi*_22 and nowhere else.

    `target` therefore has to be the molecule's response AT THE LEVEL THE ROUTE
    SCREENS AT, which for every GW route here is the direct RPA -- the default.
    A site calibrated on the CPHF value instead over-screens by the ratio of
    the two, and the partition stops being invariant.
    """
    coords = mol.atom_coords()
    if heavy_only:
        coords = coords[mol.atom_charges() > 1]
    if len(coords) == 0:
        raise ValueError('no heavy atom to carry a site; pass heavy_only=False')
    if target is None:
        target = float(np.trace(rpa_polarizability(mol)) / 3.0)
    return coords, calibrated_site_alphas(coords, target, thole=thole)


class PolarizableSites:
    """Classical polarizable sites screening the interaction: QM/MMPol.

    The sibling of `SolventScreening` with a discrete environment in place of a
    continuum. Where the continuum answers the QM density with surface charges
    and a response `sym(K^-1 R)`, the sites answer it with induced dipoles and
    a response `B = (alpha^-1 - T)^-1`, and the screened interaction has the
    same shape with a FIELD where the continuum has a POTENTIAL:

        vtilde_PQ = - sum_{kx,k'x'} F[P,kx] B[kx,k'x'] F[Q,k'x']

    THE SITES DO NOT MOVE WITH THE MOLECULE. They are the environment, fixed in
    space, so `for_geometry` is the identity and B carries no nuclear
    derivative at all -- the whole geometry dependence is in the field
    integrals. That is what makes this adjoint so much smaller than the
    continuum's, whose cavity rides the atoms.

    Like the continuum, this is a POST-SCF screening: `mean_field` hands the
    factory's own mean field back, so a ground state that is itself polarized
    is the factory's business, not this object's.
    """

    differentiable = True
    screens = True

    def __init__(self, coords, alphas, thole=THOLE_FACTOR, unit='Angstrom',
                 mol=None):
        scale = 1.0 / BOHR_TO_ANGSTROM if str(unit).lower() != 'bohr' else 1.0
        self.coords = np.asarray(coords, float).reshape(-1, 3) * scale
        self.alphas = np.asarray(alphas, float).reshape(-1)
        if len(self.alphas) != len(self.coords):
            raise ValueError(f'{len(self.coords)} sites but {len(self.alphas)} '
                             'polarizabilities')
        if mol is not None:
            check_site_clearance(self.coords, mol.atom_coords())
        self.thole = thole
        # B is a property of the environment alone, so it is built once
        self.B = response_matrix(self.coords, self.alphas, thole=thole)
        self._aux_cache = {}

    @classmethod
    def from_response_matrix(cls, coords, B, thole=None, unit='Bohr', mol=None):
        """The same environment built from an already-coupled response B.

        `coords` and B are ALL the site-side algebra below reads: `aux_kernel`
        and its adjoint fold `-F B F^T` through the field integrals and never
        look at how B was assembled. This is the hook a caller with its own
        coupled response -- an anisotropic, exclusion-aware B from `cppe`
        (`src/Base/cppe_interface.py`), for instance -- uses in place of the
        isotropic `dipole_interaction_matrix` this constructor builds B with.
        `thole` is kept only for `__repr__`; it plays no further role once B
        already exists.
        """
        self = cls.__new__(cls)
        scale = 1.0 / BOHR_TO_ANGSTROM if str(unit).lower() != 'bohr' else 1.0
        self.coords = np.asarray(coords, float).reshape(-1, 3) * scale
        n = 3 * len(self.coords)
        B = np.asarray(B, float)
        if B.shape != (n, n):
            raise ValueError(f'{len(self.coords)} sites need a ({n}, {n}) '
                             f'response matrix, got {B.shape}')
        if mol is not None:
            check_site_clearance(self.coords, mol.atom_coords())
        self.alphas = None
        self.thole = thole
        self.B = B
        self._aux_cache = {}
        return self

    @property
    def nsites(self):
        return len(self.coords)

    def for_geometry(self, mol):
        """The same sites: an environment does not ride the atoms it surrounds."""
        return self

    def mean_field(self, mol, scf_factory):
        """The factory's own mean field; this screening is post-SCF."""
        return scf_factory(mol)

    def dynamic_factor(self, omega):
        """1 at every frequency: the sites carry a STATIC polarizability.

        `alpha` is the w -> 0 amplitude of a site's response and this model
        gives it no spectrum, so the dressed interaction is v + vtilde at every
        frequency -- the adiabatic limit. A site list calibrated against a
        molecular polarizability with its own absorption would need g(iw) here
        before it could carry a correlation energy.
        """
        return np.ones_like(np.asarray(omega, float))

    def whitened_transform(self, mol, mf):
        """Refused: the sites reach a route through `aux_kernel` alone.

        Not None, which would hand a density-fitted route the BARE interaction
        and read as a converged gas-phase answer. The field integrals F are
        built against the AUXILIARY centres, so the separable factorization is
        the representation this screening has; use the ISDF or space-time route.
        """
        raise NotImplementedError(
            'PolarizableSites screens through the auxiliary metric '
            '(`aux_kernel`), not through pyscf\'s density-fitted factor: run '
            'the ISDF or space-time route, or use SolventScreening for a '
            'continuum, which has both forms')

    def kernel_mo(self, mol, mo_bra, mo_ket=None):
        """Refused: no four-index form, for the reason `whitened_transform` gives."""
        raise NotImplementedError(
            'PolarizableSites has no four-index vtilde; it screens through the '
            'auxiliary metric (`aux_kernel`)')

    def kernel_ao(self, mol):
        """Refused: no four-index form, for the reason `whitened_transform` gives."""
        raise NotImplementedError(
            'PolarizableSites has no four-index vtilde; it screens through the '
            'auxiliary metric (`aux_kernel`)')

    def aux_kernel(self, auxmol):
        """vtilde_PQ = -F B F^T between auxiliary functions, (naux, naux).

        Negative semidefinite: an induced dipole lowers the interaction it
        mediates. With the opposite sign the sites anti-screen and open the
        gap, and nothing downstream sees it, because `aux_metric_sqrt` watches
        v + vtilde for LOST positivity only.

        Keyed on the basis CONTENT, never on `id(auxmol)`: a displaced geometry
        builds a fresh object whose id python is free to reuse once the
        previous one is collected, and the cache would then hand back the
        reference geometry's kernel for a moved one -- intermittently, and
        looking perfectly reasonable.

        The auxiliary centres ARE the QM atoms, so this is where a site list
        meets the density it has to stay out of: the clearance is checked here
        for every geometry a route screens at, not only against the molecule a
        caller happened to pass at construction.
        """
        check_site_clearance(self.coords, auxmol.atom_coords())
        key = auxmol_key(auxmol)
        if key not in self._aux_cache:
            F = site_field(auxmol, self.coords).reshape(auxmol.nao_nr(), -1)
            self._aux_cache[key] = -(F @ self.B @ F.T)
        return self._aux_cache[key]

    def static_self_energy(self, mf, mol=None):
        """No COHSEX term of its own.

        On the GW routes the sites' self-polarization enters as the Eq. (18)
        self-element of Delta W = W[v + vtilde] - W[v] built from `aux_kernel`
        (Duchemin, Guido, Jacquemin, Blase, Chem. Sci. 9, 4430 (2018)), the
        discrete-environment form of Li, D'Avino, Duchemin, Beljonne, Blase,
        Phys. Rev. B 97, 035108 (2018) Eqs. (14)-(15); the routes that still
        take a COHSEX static term get none from this object. A polarized
        ground state belongs in the mean field, which `mean_field` leaves to
        the factory.
        """
        return None

    def aux_kernel_adjoint(self, auxmol, v_bar):
        """(natm, 3) of d/dR Tr[v_bar^T vtilde_aux], the adjoint held fixed.

            d/dR Tr[-Y F B F^T] = -2 sum_{P,kx} L[P,kx] dF[P,kx]/dR,  L = Y F B

        with only the auxiliary centre moving. Only the symmetric part of
        `v_bar` survives, vtilde being symmetric, and an auxiliary function
        contributes to the atom that carries it.
        """
        n_aux = auxmol.nao_nr()
        Y = 0.5 * (np.asarray(v_bar, float) + np.asarray(v_bar, float).T)
        F = site_field(auxmol, self.coords).reshape(n_aux, -1)
        L = (Y @ F) @ self.B                                   # (naux, 3 nsite)
        dF = site_field_aux_gradient(auxmol, self.coords).reshape(3, n_aux, -1)
        per_aux = -2.0 * np.einsum('Pk,yPk->Py', L, dF, optimize=True)
        de = np.zeros((auxmol.natm, 3))
        for ia, (_, _, p0, p1) in enumerate(auxmol.aoslice_by_atom()):
            de[ia] = per_aux[p0:p1].sum(axis=0)
        return de

    def static_self_energy_adjoint(self, mf, weights):
        """Zero, there being no static term to differentiate."""
        return 0.0, np.zeros((mf.mol.natm, 3))

    def __repr__(self):
        if self.alphas is None:
            return f'PolarizableSites({self.nsites} sites, external B, ' \
                   f'thole={self.thole})'
        return (f'PolarizableSites({self.nsites} sites, '
                f'alpha {self.alphas.min():.2f}-{self.alphas.max():.2f} Bohr^3, '
                f'thole={self.thole})')
