"""A `PolarizableSites` environment sourced from a real polarizable-embedding
potential file, through cppe.

`PolarizableSites` builds its coupled response B = (alpha^-1 - T)^-1 from a
hand-rolled, isotropic-alpha, uniformly-Thole-damped
`dipole_interaction_matrix` -- fine for a handful of sites built by hand, but
it has no anisotropic polarizability tensors and no 1-2/1-3 exclusion lists,
both of which every real force-field potential file (PyFraME, DALTON's PE
library) carries per atom. cppe (github.com/maxscheurer/cppe; Olsen, Kongsted,
J. Chem. Phys. 141, 194109 (2015) for the model, https://doi.org/10.1021/
acs.jctc.9b00016 for the library) is the reference implementation of exactly
that coupled response, built from the same anisotropic tensors and exclusions
the potential file specifies.

THE INTERFACE IS NARROW ON PURPOSE. This module hands cppe only the
POLARIZABLE-SITE half of a potential file -- positions, anisotropic
polarizability tensors, exclusion lists -- and takes back one matrix, B. It
does not touch cppe's permanent-multipole or SCF-embedding machinery: every
`Environment` in this repository treats a classical environment as POST-SCF
screening (`PolarizableSites.mean_field` hands the factory's own mean field
back unchanged), so coupling permanent charges into the Fock matrix is
`PointCharges`' and `SolventScreening`'s job, not this one's -- mixing the two
models inside one site set would double count the permanent field. Once cppe
has built B, the result is handed to `PolarizableSites.from_response_matrix`
and folds through the exact same `aux_kernel` / `aux_kernel_adjoint` algebra
as a hand-built, isotropic `PolarizableSites`: both are one (3N, 3N) response
matrix and a set of fixed site coordinates, and that algebra never looks past
B to ask where it came from.

cppe's own Thole damping is NOT bit-identical to `dipole_interaction_matrix`
at the same THOLE_FACTOR -- checked directly against two isotropic alpha = 5
Bohr^3 sites 6 Bohr apart, undamped they agree to machine precision, damped
they differ by several tenths of a percent in B. Both are the same physical
model (Thole, Chem. Phys. 59, 341 (1981)) at a different implementation
convention, not a bug in either; a site list read through this module carries
cppe's damping throughout, one built by hand through `PolarizableSites`
carries wicks's own, and the two are not meant to be mixed.
"""
import numpy as np

from src.Base.polarizable_sites import PolarizableSites, THOLE_FACTOR


def _cppe():
    """The cppe module, imported lazily: an optional dependency only this
    interface needs, not the polarizable-sites physics itself."""
    try:
        import cppe
    except ImportError as exc:
        raise ImportError(
            'polarizable_sites_from_potfile needs a loadable cppe '
            '(pip install cppe)') from exc
    return cppe


def polarizable_sites_from_potfile(potfile, mol=None, thole=THOLE_FACTOR,
                                    damp_induced=True):
    """A `PolarizableSites` environment built from a CPPE/PyFraME potential file.

    Reads every polarizable site in `potfile` (Bohr positions -- cppe converts
    a file given in Angstrom itself -- anisotropic polarizability tensors and
    1-2/1-3 exclusion lists) with cppe's own parser, solves the coupled
    induced-dipole response with cppe's `BMatrix` (its exclusions and damping,
    not `dipole_interaction_matrix`'s), and returns the object
    `PolarizableSites` builds by hand: same `aux_kernel`, same adjoint, same
    every other `Environment` entry, none of which read anything but
    `self.coords` and `self.B`.

    `mol`, given, is checked for `MIN_SITE_TO_QM_DISTANCE` clearance exactly as
    the hand-built constructor does -- a potential file built for a QM region
    that has since moved or grown can otherwise place a site inside the
    density the folding of W onto the QM region assumes it never reaches.

    A non-polarizable site in the file (a bare multipole with no
    `@POLARIZABILITIES` entry) contributes no coupling and is dropped by
    cppe's own `get_polarizable_sites` before B is built; carrying its
    permanent field into the mean field, if that matters here, is
    `PointCharges`' job on the same coordinates, done separately.
    """
    cppe = _cppe()
    sites = cppe.get_polarizable_sites(cppe.PotfileReader(str(potfile)).read())
    if not sites:
        raise ValueError(f'{potfile}: no polarizable site in the potential file')
    coords = np.array([site.position for site in sites])
    options = {'damp_induced': damp_induced, 'damping_factor_induced': thole}
    B = cppe.BMatrix(sites, options).direct_inverse()
    return PolarizableSites.from_response_matrix(coords, B, thole=thole,
                                                  unit='Bohr', mol=mol)
