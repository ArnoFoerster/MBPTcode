"""Permanent charges and polarizable sites as ONE environment.

Li, D'Avino, Duchemin, Beljonne and Blase, Phys. Rev. B 97, 035108 (2018)
split a classical environment into two channels that must not be confused.
The permanent multipoles change the electrostatic potential the QM system sits
in and belong to the ground state (Sec. II A: an electrostatic effect "well
described in a ground-state DFT calculation [that] should not be confused with
the dynamic reaction of the system to the ionization"), while only the
multipoles INDUCED by the added or removed charge reach the reaction field
(Sec. II D: "fixed charges in the MM part ... do not contribute to the
reaction field matrix"). A composite carries both channels with neither
leaking into the other's: the charges reach the mean field and contribute
nothing to `aux_kernel`, the sites screen and leave the ground state alone.

WHY ONE OBJECT RATHER THAN TWO ATTACHMENTS. The quasiparticle shift of
Duchemin, Guido, Jacquemin and Blase, Chem. Sci. 9, 4430 (2018) Eq. (18) is a
SELF-ELEMENT of the TOTAL Delta W = W[v + vtilde] - W[v]. Delta W is not
additive in vtilde -- it runs through a Dyson inversion -- so two environments
attached in turn, each forming its own shift from its own kernel, would give a
sum that is not the shift of the total dressed interaction. The members'
kernels are therefore summed here, before any route sees them, and exactly one
shift is formed downstream. With fixed charges contributing None the sum is
the sites' own kernel and the identity is automatic, which is what makes the
charges' effect on the shift purely indirect: they change the orbitals the
self-element is evaluated on and nothing else.

A CONTINUUM AND SITES (`ContinuumWithSites`) are two RESPONDING members, and
there summing kernels is wrong outright: each polarizes the other, so the
coupled response is one linear problem in the surface charges and the induced
dipoles together, solved once into one kernel. The continuum's cavity then
encloses the explicit shell the sites stand in.
"""
import numpy as np

from src.Base.polarizable_sites import check_site_clearance, site_field
from src.Base.separable_ri import auxmol_key


class CompositeEnvironment:
    """Fixed point charges plus polarizable sites behind the `Environment` contract.

    `charges` carries the permanent multipoles into the mean field (pyscf's
    `qmmm.mm_charge`), `sites` carries the induced dipoles into the screened
    interaction. Both are fixed in space, so `for_geometry` is the identity and
    the whole geometry dependence of the composite is the one the members
    already have.
    """

    def __init__(self, charges, sites):
        self.charges = charges
        self.sites = sites
        self.members = (charges, sites)
        # a chain asks before paying for a reverse pass: the composite can
        # return a force only if every member can
        self.differentiable = all(bool(m.differentiable) for m in self.members)
        # ... and it dresses the interaction if ANY member does, which is the
        # basis-free half of `dresses_interaction`
        self.screens = any(getattr(m, 'screens', True) for m in self.members)

    def for_geometry(self, mol):
        """The same environment: neither fixed charges nor fixed sites ride the atoms."""
        return self

    def mean_field(self, mol, scf_factory):
        """The converged mean field of `mol` in the permanent field of the charges.

        Chained rather than picked: the sites see a factory that already
        carries the charges, so a ground state polarized by the sites -- which
        this screening does not build -- would compose with the charges instead
        of replacing them.
        """
        return self.sites.mean_field(
            mol, lambda m: self.charges.mean_field(m, scf_factory))

    def aux_kernel(self, auxmol):
        """The members' vtilde_PQ summed, or None when nothing responds.

        ONE kernel, so the route forms ONE Eq. (18) shift from the total
        dressed interaction. A fixed charge does not respond and adds nothing.
        """
        kernels = [k for k in (m.aux_kernel(auxmol) for m in self.members)
                   if k is not None]
        return None if not kernels else sum(kernels)

    def dynamic_factor(self, omega):
        """The responding member's g(iw); 1 when none responds.

        NOT a product or a sum: g weights `aux_kernel`, and the composite's
        kernel is the members' SUM, so one scalar describes it only while one
        member responds. Two that responded at different frequencies would need
        the weighting inside the sum, which this contract cannot express.
        """
        responding = [m for m in self.members if getattr(m, 'screens', True)]
        if not responding:
            return np.ones_like(np.asarray(omega, float))
        if len(responding) > 1:
            raise NotImplementedError(
                'two responding members need their own g(iw) inside the kernel '
                'sum, which one scalar factor cannot carry')
        return responding[0].dynamic_factor(omega)

    def whitened_transform(self, mol, mf):
        """The screening member's T, or None when no member responds.

        NOT a sum: a congruence B -> T B does not add, so a composite of two
        responding members has no single one and refuses rather than picking.
        At most one member of a composite screens (a second responding member
        is refused), so this is the delegation it looks like.
        """
        screening = [m for m in self.members if getattr(m, 'screens', True)]
        if not screening:
            return None
        if len(screening) > 1:
            raise NotImplementedError(
                'two responding members have no single whitened transform '
                'between them -- a congruence does not add; run the separable '
                'route, whose `aux_kernel` does')
        return screening[0].whitened_transform(mol, mf)

    def kernel_mo(self, mol, mo_bra, mo_ket=None):
        """The members' four-index vtilde summed, as their aux kernels are."""
        terms = [k for k in (m.kernel_mo(mol, mo_bra, mo_ket) for m in self.members)
                 if k is not None]
        return None if not terms else sum(terms)

    def kernel_ao(self, mol):
        """The members' four-index vtilde summed, as their aux kernels are."""
        terms = [k for k in (m.kernel_ao(mol) for m in self.members)
                 if k is not None]
        return None if not terms else sum(terms)

    def static_self_energy(self, mf, mol=None):
        """The members' static one-body terms summed, or None when there is none."""
        terms = [t for t in (m.static_self_energy(mf, mol) for m in self.members)
                 if t is not None]
        return None if not terms else sum(terms)

    def aux_kernel_adjoint(self, auxmol, v_bar):
        """(natm, 3): the members' adjoints summed, as their kernels are."""
        return sum(m.aux_kernel_adjoint(auxmol, v_bar) for m in self.members)

    def static_self_energy_adjoint(self, mf, weights):
        """The orbital-rotation gradient and the (natm, 3) skeleton, summed."""
        extra = 0.0
        skeleton = np.zeros((mf.mol.natm, 3))
        for member in self.members:
            member_extra, member_skeleton = member.static_self_energy_adjoint(
                mf, weights)
            extra = extra + member_extra
            skeleton = skeleton + member_skeleton
        return extra, skeleton

    def __repr__(self):
        return f'CompositeEnvironment({self.charges!r}, {self.sites!r})'


def dipole_surface_potential(continuum, coords):
    """G[k, jx]: the potential on surface charge k of a unit dipole at site j
    along x, (ngrids, 3 nsite), averaged over that charge's Gaussian.

    Minus the field that surface charge puts at the site, so it reuses the
    sites' own field integral (`site_field`) on the continuum's Gaussian
    surface charges, erf smearing and all: one sign convention for every
    field in the coupled response below.
    """
    fakemol = continuum._fakemol()
    return -site_field(fakemol, coords).reshape(continuum.ngrids, -1)


class ContinuumWithSites:
    """A PCM continuum and polarizable sites that polarize each other.

    An explicit first shell of polarizable sites inside a continuum: the
    cavity encloses the solute AND the shell (`SolventScreening(...,
    cavity_atoms=shell)`), so the continuum begins beyond the explicit
    molecules, and the two respond to a source together. With U the source's
    potential on the surface, E its field at the sites, Q the continuum's
    response (q = Q v), B the sites' (mu = B E) and G the potential of a site
    dipole on the surface (`dipole_surface_potential`), the induced charges and
    dipoles solve

        q  = Q (U + G mu),       mu = B (E - G^T q),

    and a probe density meets U^T q - E^T mu. Eliminating q,

        vtilde = U Q U^T - E' M E'^T,
        E'     = E - U Q G,               the field at the sites of the source
                                          AND of the surface charges it induces,
        M      = (1 + B G^T Q G)^-1 B,    the sites' response dressed by their
                                          own image in the continuum,

    ONE kernel, so every route forms ONE Eq. (18) shift from the total dressed
    interaction. B = 0 leaves the continuum's kernel and Q = 0 (eps = 1) the
    sites', exactly. M is positive definite only while each site's image
    interaction stays below its restoring term, which a site well inside the
    cavity satisfies and one near the surface need not; that case, and a site
    outside every sphere of the cavity, is refused.

    The ground state is the continuum's (PCM at eps_static on the enclosing
    cavity); the sites stay post-SCF, as `PolarizableSites` is. No force: the
    shell's spheres do not ride the atoms and the coupled kernel has no adjoint
    here. The solvated dRPA correlation energy is refused unless both members
    are adiabatic, since the coupled kernel's frequency dependence is no
    longer one scalar factor.
    """

    screens = True
    differentiable = False

    def __init__(self, continuum, sites):
        if getattr(continuum, 'cavity_atoms', None) is None:
            raise ValueError(
                'the continuum of a ContinuumWithSites must enclose the '
                'explicit shell: build it with SolventScreening(..., '
                'cavity_atoms=shell), so the continuum begins beyond the '
                'polarizable sites rather than through them')
        centres, radii = continuum.cavity_spheres()
        outside = [j for j, r in enumerate(np.asarray(sites.coords))
                   if not np.any(np.linalg.norm(centres - r, axis=1) < radii)]
        if outside:
            raise ValueError(
                f'polarizable sites {outside} lie outside every sphere of the '
                f'cavity; the continuum would sit where they are. Add the '
                f'shell atoms they belong to to cavity_atoms.')
        self.continuum, self.sites = continuum, sites
        self.members = (continuum, sites)
        self._coupled = None
        self._aux_cache = {}

    def for_geometry(self, mol):
        """The same shell and sites around a new solute geometry: the cavity is
        rebuilt around the solute's atoms, the sites stay where they are."""
        continuum = self.continuum.for_geometry(mol)
        if continuum is self.continuum:
            return self
        return ContinuumWithSites(continuum, self.sites)

    def mean_field(self, mol, scf_factory):
        """The continuum's ground state on the enclosing cavity; the sites are
        post-SCF."""
        return self.continuum.mean_field(mol, scf_factory)

    def coupling(self):
        """(G, M): the dipole potentials on the surface and the sites' response
        dressed by the continuum, refused where M is not positive definite."""
        if self._coupled is None:
            G = dipole_surface_potential(self.continuum, self.sites.coords)
            Q = self.continuum.response_matrix()
            B = np.asarray(self.sites.B, float)
            M = np.linalg.solve(np.eye(len(B)) + B @ (G.T @ Q @ G), B)
            M = 0.5 * (M + M.T)
            w = np.linalg.eigvalsh(M)
            if w.size and w.min() < -1e-10 * max(np.abs(w).max(), 1e-300):
                raise ValueError(
                    f'the sites\' response dressed by the continuum is not '
                    f'positive definite (lowest eigenvalue {w.min():.3e}): a '
                    f'site sits so close to the cavity surface that its image '
                    f'outweighs its polarizability. Enlarge the cavity around '
                    f'the shell or move the site inward.')
            self._coupled = (G, M)
        return self._coupled

    def aux_kernel(self, auxmol):
        """vtilde_PQ of the coupled response between auxiliary functions."""
        check_site_clearance(self.sites.coords, auxmol.atom_coords())
        key = auxmol_key(auxmol)
        if key not in self._aux_cache:
            U = self.continuum.aux_grid_potential(auxmol)
            Q = self.continuum.response_matrix()
            E = site_field(auxmol, self.sites.coords).reshape(auxmol.nao_nr(),
                                                               -1)
            G, M = self.coupling()
            screened = E - (U @ Q) @ G
            self._aux_cache[key] = (self.continuum.aux_kernel(auxmol)
                                    - screened @ M @ screened.T)
        return self._aux_cache[key]

    def dynamic_factor(self, omega):
        """1 when both members are adiabatic; refused otherwise (see class)."""
        omega = np.asarray(omega, float)
        for member in self.members:
            if not np.allclose(member.dynamic_factor(omega), 1.0):
                raise NotImplementedError(
                    f'{member!r} responds with a frequency-dependent g(iw), '
                    f'and the coupled kernel is not one scalar times a static '
                    f'one; give the continuum an explicit eps with no '
                    f'omega_p, or leave the dRPA energy out')
        return np.ones_like(omega)

    def whitened_transform(self, mol, mf):
        """Refused: the sites reach a route through `aux_kernel` alone."""
        raise NotImplementedError(
            'ContinuumWithSites screens through the auxiliary metric '
            '(`aux_kernel`), not through a density-fitted factor: run a '
            'route that builds W from the separable factors')

    def kernel_mo(self, mol, mo_bra, mo_ket=None):
        """Refused, for the reason `whitened_transform` gives."""
        raise NotImplementedError(
            'ContinuumWithSites has no four-index vtilde; it screens through '
            'the auxiliary metric (`aux_kernel`)')

    def kernel_ao(self, mol):
        """Refused, for the reason `whitened_transform` gives."""
        raise NotImplementedError(
            'ContinuumWithSites has no four-index vtilde; it screens through '
            'the auxiliary metric (`aux_kernel`)')

    def static_self_energy(self, mf, mol=None):
        """Refused: a route that never forms W (ADC) would take the
        continuum's COHSEX term alone, without the sites or their coupling."""
        raise NotImplementedError(
            'ContinuumWithSites enters the GW routes as the Eq. (18) shift of '
            'its coupled kernel; a route without W has no static term for it')

    def aux_kernel_adjoint(self, auxmol, v_bar):
        raise NotImplementedError('ContinuumWithSites has no nuclear derivative')

    def static_self_energy_adjoint(self, mf, weights):
        raise NotImplementedError('ContinuumWithSites has no nuclear derivative')

    def __repr__(self):
        return f'ContinuumWithSites({self.continuum!r}, {self.sites!r})'
