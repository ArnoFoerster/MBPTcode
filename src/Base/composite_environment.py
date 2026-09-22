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
"""
import numpy as np


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
        Today at most one member screens, so this is the delegation it looks
        like.
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
