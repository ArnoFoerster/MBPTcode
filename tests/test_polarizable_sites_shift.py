"""The static half of the MMPol route: the Eq. (18) self-polarization shift a
site list gives a quasiparticle level, against classical electrostatics.

An orbital compact against the site distance is a unit point charge at its
centroid as seen from the site, so its self-polarization energy is the Born-like
-1/2 alpha E^2 = -1/2 alpha / R^4 of one charge and the dipole it induces, and
Duchemin, Guido, Jacquemin and Blase, Chem. Sci. 9, 4430 (2018) Eq. (18) turns
that into a level shift: occupied levels rise, virtual levels fall, the gap
closes. It is the polarization energy P_n of Li, D'Avino, Duchemin, Beljonne and
Blase, Phys. Rev. B 97, 035108 (2018) Eq. (15) for a discrete environment, and
the sign the whole embedding rests on.

The limit is approached from BELOW, as 1/R: Delta W is the reaction field
screened by the molecule's own electrons, so the charge the site sees is the
orbital's plus the molecule's zero-monopole response to it, whose dipole
corrects the field by O(1/R^3) against the 1/R^2 of the charge. On water the
shift is 6 % under the Born value at 15 Bohr and 3 % under it at 30.

THE MANY-SITE LIMIT IS THE CONTINUUM, and it is the strongest cross-check in
the module: a ball of sites at the Clausius-Mossotti density for eps IS a
dielectric of that permittivity, so its Eq. (18) shift must be the one
`SolventScreening` gives for the same cavity -- two implementations of one
physics, one discrete and one a surface-charge discretization, meeting through
a single route. PRB footnote 3 makes exactly this comparison for pentacene
(CR 1.00/1.13 eV against PCM 0.96/1.31). The discrete environment is finite,
so it reaches the continuum along the R^-1 (equivalently N^-1/3) law of PRB
Fig. 1, whose exact form here is the shell result 1 - a/b.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import BOHR_TO_ANGSTROM, HARTREE_TO_EV
from src.Base.environment import attached_environment
from src.Base.polarizable_sites import PolarizableSites
from src.Base.solvent_screening import SolventScreening
from src.SingleReference.GW.reaction_field import environment_quasiparticle_shift

#: Bohr^3, one site on the C2 axis of water at the distances below (Bohr).
ALPHA = 10.0
DISTANCES = (15.0, 20.0, 30.0)
#: Angstrom; the HOMO is the oxygen lone pair, whose centroid sits on O to
#: 0.05 Bohr.
Z_OXYGEN = 0.117
#: water's optical dielectric constant, n^2
EPS = 1.78
#: Bohr: the cavity both environments carve, and the outer radii the discrete
#: one is truncated at. The cavity clears MIN_SITE_TO_QM_DISTANCE from every
#: nucleus, which a solvation-shell radius would not.
CAVITY = 9.0
OUTER = (15.0, 18.0, 21.0)
#: Bohr, the two lattice constants the discretization error is measured on.
SPACINGS = (3.0, 2.5)


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=f'O 0 0 {Z_OXYGEN}; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                basis='cc-pvdz', verbose=0)
    return mol, scf.RHF(mol).run(conv_tol=1e-10)


def shift_at(mf, distance):
    env = PolarizableSites(np.array([[0.0, 0.0, distance]]), np.array([ALPHA]),
                           unit='Bohr')
    with attached_environment(mf, env):
        return environment_quasiparticle_shift(mf)


def born_limit(distance):
    """1/2 alpha / R^4 for a unit charge on the oxygen, in Hartree."""
    return 0.5 * ALPHA / (distance - Z_OXYGEN / BOHR_TO_ANGSTROM) ** 4


def cm_shell(centre, inner, outer, spacing, eps=EPS):
    """A cubic lattice of sites clipped to inner <= |r - centre| <= outer, at
    the Clausius-Mossotti density for `eps`.

        (eps - 1)/(eps + 2) = (4 pi / 3) n alpha,   n = 1 / spacing^3

    so the shell is a dielectric of permittivity eps with a spherical void.
    The site polarizability follows from the LATTICE SPACING, not from a count
    divided by a volume: the carved cavity would corrupt that ratio, and the
    density the Clausius-Mossotti relation refers to is the lattice's own.
    """
    m = int(np.ceil(outer / spacing)) + 1
    g = (np.arange(-m, m + 1) + 0.5) * spacing
    pts = np.array(np.meshgrid(g, g, g, indexing='ij')).reshape(3, -1).T + centre
    d = np.linalg.norm(pts - centre, axis=1)
    pts = pts[(d >= inner) & (d <= outer)]
    cm = (eps - 1.0) / (eps + 2.0)
    return pts, np.full(len(pts), 3.0 * spacing ** 3 * cm / (4.0 * np.pi))


def spherical_cavity(mol, radius, eps=EPS):
    """`SolventScreening` on ONE sphere of `radius` Bohr centred on the oxygen.

    pyscf's PCM takes `radii_table` indexed by nuclear charge and in Bohr, and
    uses it INSTEAD of vdw_scale * Bondi + r_probe, so it is the finished
    radius. Giving hydrogen a radius small enough to sit inside the oxygen
    sphere switches its own surface points off entirely -- the union of spheres
    collapses to the one sphere the discrete environment carves its cavity as.
    """
    table = np.full(int(mol.atom_charges().max()) + 1, 0.1)
    table[8] = radius
    return SolventScreening(mol, eps=eps, radii_table=table)


@pytest.fixture(scope='module')
def continuum_limit(water):
    """The Eq. (18) shifts of the continuum and of the discrete shells, in eV.

    One mean field for all of them: the two environments are compared as
    reaction fields, not as ground states, so neither relaxes the SCF.
    """
    mol, mf = water
    centre = mol.atom_coords()[0]
    with attached_environment(mf, spherical_cavity(mol, CAVITY)):
        pcm = environment_quasiparticle_shift(mf) * HARTREE_TO_EV
    sites = {}
    for spacing in SPACINGS:
        for outer in OUTER:
            coords, alphas = cm_shell(centre, CAVITY, outer, spacing)
            env = PolarizableSites(coords, alphas, unit='Bohr', mol=mol)
            with attached_environment(mf, env):
                sites[spacing, outer] = (
                    environment_quasiparticle_shift(mf) * HARTREE_TO_EV,
                    len(coords))
    return pcm, sites


def test_the_shift_closes_the_gap(water):
    """Occupied up, virtual down: a hole or an added electron is stabilised by
    the dipole it induces, so both charged excitations get cheaper."""
    mol, mf = water
    nocc = mol.nelectron // 2
    shift = shift_at(mf, DISTANCES[0])
    assert shift[nocc - 1] > 0.0
    assert shift[nocc] < 0.0
    assert (shift[:nocc] > 0.0).all()


def test_the_homo_shift_is_the_born_self_polarization_of_one_charge(water):
    """Far from the molecule the compact lone pair is a point charge, so the
    self-element of Delta W is -alpha E^2 with E = 1/R^2: the magnitude fixes
    the 1/2 of Eq. (18) and the scale of the kernel at once. The molecule's
    screening of its own reaction field keeps the shift UNDER the Born value,
    by a margin that shrinks as 1/R."""
    mol, mf = water
    nocc = mol.nelectron // 2
    ratios = [shift_at(mf, d)[nocc - 1] / born_limit(d) for d in DISTANCES]
    for distance, ratio in zip(DISTANCES, ratios):
        assert 0.9 < ratio < 1.0, (distance, ratio)
    assert ratios[0] < ratios[1] < ratios[2]
    # a 1/R deficit halves between 15 and 30 Bohr; an R^-2 one would quarter
    deficits = [1.0 - r for r in ratios]
    assert 1.5 < deficits[0] / deficits[2] < 3.0


def test_the_shift_falls_off_as_the_fourth_power(water):
    """The field of a charge goes as R^-2 and the polarization energy as its
    square; the ratio between the two distances is what a wrong power of R
    in the field integrals would move first."""
    mol, mf = water
    nocc = mol.nelectron // 2
    near = shift_at(mf, DISTANCES[0])[nocc - 1]
    far = shift_at(mf, DISTANCES[2])[nocc - 1]
    expected = born_limit(DISTANCES[0]) / born_limit(DISTANCES[2])
    # the 1/R screening deficit is worth 4 % of the ratio, a wrong power 100 %
    assert abs(near / far - expected) / expected < 0.06


# --------------------------------------------------------- the many-site limit
def test_the_continuum_cavity_is_the_classical_born_energy(water, continuum_limit):
    """The reference the discrete environment is measured against, checked on
    its own terms first: a spherical cavity of radius a in a continuum of eps
    shifts a compact orbital by the Born self-polarization

        1/2 (1 - 1/eps) / a

    which at a = 9 Bohr and eps = 1.78 is 662.4 meV. The PCM discretization
    gives +663.4 for the HOMO and -648.4 for the LUMO -- 0.15 % and 2.1 %, the
    latter the more diffuse orbital seeing the cavity wall."""
    mol, _ = water
    nocc = mol.nelectron // 2
    pcm, _ = continuum_limit
    born = 0.5 * (1.0 - 1.0 / EPS) / CAVITY * HARTREE_TO_EV
    assert abs(pcm[nocc - 1] / born - 1.0) < 0.01
    assert abs(abs(pcm[nocc]) / born - 1.0) < 0.05
    assert pcm[nocc - 1] > 0.0 > pcm[nocc]


@pytest.mark.parametrize('outer', OUTER)
def test_a_clausius_mossotti_shell_reproduces_the_continuum(water, continuum_limit,
                                                            outer):
    """A ball of Clausius-Mossotti sites with a spherical void IS a dielectric
    shell, whose exact reaction potential at the centre is the full-cavity one
    reduced by the missing outside:

        P(a, b) = P(a) (1 - a/b)

    so the discrete and continuum implementations must agree on that number
    through one route. At the converged lattice constant 2.5 Bohr they do, to
    1.0 %: 0.4006 / 0.4982 / 0.5657 of the continuum shift for b = 15 / 18 / 21
    against 0.4000 / 0.5000 / 0.5714. The LUMO follows the HOMO.

    THE LATTICE CONSTANT IS THE ERROR, not the model: at 3.0 Bohr the same
    shells give 0.3659 / 0.4552 / 0.5277, 7.6-9.0 % low, because the jagged
    lattice boundary of the void sits further out than the sphere it
    approximates and the reaction field at the centre is dominated by that
    innermost surface. Refining the lattice moves the answer onto the continuum
    and nothing else changes, which is what says the disagreement is
    discretization and not physics.
    """
    mol, _ = water
    nocc = mol.nelectron // 2
    pcm, sites = continuum_limit
    exact = 1.0 - CAVITY / outer
    fine = sites[SPACINGS[1], outer][0]
    coarse = sites[SPACINGS[0], outer][0]
    assert abs(fine[nocc - 1] / pcm[nocc - 1] - exact) < 0.02
    assert abs(fine[nocc] / pcm[nocc] - exact) < 0.02
    assert coarse[nocc - 1] / pcm[nocc - 1] < fine[nocc - 1] / pcm[nocc - 1]
    assert 0.05 < 1.0 - coarse[nocc - 1] / (exact * pcm[nocc - 1]) < 0.12


def test_the_polarization_energy_converges_as_the_inverse_cube_root(water,
                                                                    continuum_limit):
    """PRB Fig. 1: |Delta^COHSEX| is linear in N^-1/3 over MM cluster radii of
    25-40 Angstrom and extrapolates to the bulk polarization energy. Here N^-1/3
    is 1/b up to the sites the cavity removed, so the count is taken over the
    whole ball; the three shells are linear in it to 0.095 meV of 375 and
    extrapolate to 98.7 % of the continuum value.

    This is the law that says whether a truncated site list is converged, and
    at b = 21 Bohr the answer is still 43 % short of the continuum -- the
    reaction field's R^-1 tail is why the papers extrapolate rather than
    enlarge."""
    mol, _ = water
    nocc = mol.nelectron // 2
    pcm, sites = continuum_limit
    spacing = SPACINGS[1]
    counts = np.array([sites[spacing, b][1]
                       + 4.0 / 3.0 * np.pi * CAVITY ** 3 / spacing ** 3
                       for b in OUTER])
    x = counts ** (-1.0 / 3.0)
    y = np.array([sites[spacing, b][0][nocc - 1] for b in OUTER])
    slope, intercept = np.polyfit(x, y, 1)
    assert np.abs(np.polyval([slope, intercept], x) - y).max() < 0.004 * y.max()
    assert slope < 0.0
    assert 0.9 < intercept / pcm[nocc - 1] < 1.0
    assert y.max() / pcm[nocc - 1] < 0.6
