"""`PolarizableSites` as an `Environment`: the screened kernel and its adjoint.

The continuum's sibling, with induced dipoles where the continuum has surface
charges. Gated on the same three things the continuum's gates cover -- the
kernel's own limits, the adjoint against finite differences, and the protocol
the chain calls through -- plus the one property that distinguishes it: the
sites do NOT ride the molecule, so B carries no nuclear derivative and the
whole geometry dependence sits in the field integrals.

The field integral conventions underneath were pinned numerically before
anything was built on them: `site_field` against a finite difference of the
potential with respect to a site (4e-11), and `site_field_aux_gradient`
against a finite difference of `site_field` under a nuclear displacement
(1e-11). Those are the two signs a reverse pass cannot recover from.

TWO CATASTROPHES, TWO GUARDS. Thole damping bounds the coupling BETWEEN sites
and the definiteness of alpha^-1 - T says where even that fails; nothing damps
the field integral between a site and the QM charge, so the site-to-QM
clearance is a separate refusal at the model's own non-overlap assumption. The
dressed metric's positivity is the backstop behind both and stays reachable:
an admissible distance with an inadmissible polarizability still trips it.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import df, gto

from src.Base.constants import BOHR_TO_ANGSTROM, MIN_SITE_TO_QM_DISTANCE
from src.Base.environment import Environment
from src.Base.polarizable_sites import (PolarizableSites, response_matrix,
                                        site_field)
from src.Base.separable_ri import aux_metric_sqrt
from src.Base.solvent_screening import SolventScreening

#: Bohr, on the auxiliary centres. Central, so the residual is O(h^2).
H = 1e-4
ATOM = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: Bohr, 7.2-10.0 from the nearest nucleus: outside the QM density, which is
#: where the model is defined. Closer in is CLOSE_SITES below, the refused case.
SITES = np.array([[0.0, 0.0, 9.0], [5.0, 2.0, -6.0], [-5.0, -6.0, 2.0]])
CLOSE_SITES = 0.5 * SITES
ALPHAS = np.array([5.0, 3.0, 8.0])


def molecules(atom=ATOM):
    mol = gto.M(atom=atom, basis='sto-3g', verbose=0)
    return mol, df.addons.make_auxmol(mol, auxbasis='cc-pvdz-ri')


@pytest.fixture(scope='module')
def env():
    return PolarizableSites(SITES, ALPHAS, unit='Bohr')


def test_it_satisfies_the_environment_protocol(env):
    """A chain calls through these six and asks `differentiable` before paying
    for a reverse pass; a missing one is a runtime failure deep in a gradient."""
    assert isinstance(env, Environment)
    for name in ('for_geometry', 'mean_field', 'aux_kernel',
                 'static_self_energy', 'aux_kernel_adjoint',
                 'static_self_energy_adjoint'):
        assert callable(getattr(env, name)), name
    assert env.differentiable is True


def test_the_sites_do_not_ride_the_molecule(env):
    """`for_geometry` is the identity: an environment surrounds the atoms, it
    does not move with them. This is what makes B geometry-independent."""
    other, _ = molecules('O 0 0 0.5; H 0 0.8 -0.4; H 0 -0.8 -0.4')
    assert env.for_geometry(other) is env


def test_the_kernel_is_symmetric(env):
    _, auxmol = molecules()
    v = env.aux_kernel(auxmol)
    assert v.shape == (auxmol.nao_nr(),) * 2
    assert np.abs(v - v.T).max() < 1e-12


def test_one_distant_site_is_its_own_polarizability():
    """A single site far from everything responds with alpha alone, so the
    kernel collapses to -alpha F F^T -- the limit that fixes the SCALE, which a
    symmetry or decay test cannot."""
    mol, auxmol = molecules()
    env = PolarizableSites(np.array([[0.0, 0.0, 30.0]]), np.array([4.0]),
                           unit='Bohr')
    F = site_field(auxmol, env.coords).reshape(auxmol.nao_nr(), 3)
    assert np.abs(env.aux_kernel(auxmol) + 4.0 * (F @ F.T)).max() < 1e-10


def test_the_kernel_is_the_reaction_field_of_two_classical_charges():
    """Two tight s-Gaussians are two point charges q_P at r_P, and one site of
    polarizability alpha at R returns the potential of the dipole they induce:

        vtilde_PQ = -alpha E_P(R) . E_Q(R),   E_P(R) = q_P (R - r_P) / |R - r_P|^3

    The sign is the physics: the dipole a charge induces attracts it, so two
    charges on the same side of the site couple NEGATIVELY. Built from the
    classical formula rather than from `site_field`, so the field integral's
    own sign convention is tested here instead of assumed."""
    alpha = 5.0
    charges = gto.M(atom='H 0 0 0; H 0 0 2', unit='Bohr', verbose=0,
                    basis={'H': [[0, [1000.0, 1.0]]]})
    site = np.array([[0.0, 0.0, 10.0]])
    env = PolarizableSites(site, np.array([alpha]), unit='Bohr')
    centres = charges.atom_coords()
    # q_P from the potential at a far probe: a spherical Gaussian charge is a
    # point charge outside itself, so potential * distance is q exactly
    probe = gto.fakemol_for_charges(np.array([[0.0, 0.0, 50.0]]))
    q = (gto.mole.intor_cross('int2c2e', probe, charges)[0]
         * np.linalg.norm(probe.atom_coords()[0] - centres, axis=1))
    r = site[0] - centres
    field = q[:, None] * r / np.linalg.norm(r, axis=1)[:, None] ** 3
    expected = -alpha * field @ field.T
    assert expected[0, 1] < 0.0
    got = env.aux_kernel(charges)
    assert np.abs(got - expected).max() < 1e-8 * np.abs(expected).max()


def test_the_kernel_is_a_reduced_interaction(env):
    """Negative semidefinite, as the continuum's is: v + vtilde < v. The
    positivity guard on the dressed metric fires on the OTHER sign, so this
    is the only place a flipped kernel shows."""
    mol, auxmol = molecules()
    assert np.linalg.eigvalsh(env.aux_kernel(auxmol)).max() < 1e-12
    pcm = SolventScreening(mol, eps=1.78)
    assert np.linalg.eigvalsh(pcm.aux_kernel(auxmol)).max() < 1e-12


def test_a_site_the_density_reaches_is_refused(env):
    """The non-overlap of the QM and MM orbitals is what makes the folding of W
    exact, and the site-to-QM field integral carries no damping to soften a
    violation: a site inside the QM density polarizes against a field the model
    has no physics for. It is refused at the distance rather than left to
    surface as an indefinite v + vtilde, against a molecule at construction and
    against the auxiliary centres -- which ARE the QM atoms -- when a route
    screens. Both messages name the offending site and nucleus."""
    mol, auxmol = molecules()
    with pytest.raises(ValueError, match=r'site 1 is 3\.30\d Bohr from QM centre 1'):
        PolarizableSites(CLOSE_SITES, ALPHAS, unit='Bohr', mol=mol)
    late = PolarizableSites(CLOSE_SITES, ALPHAS, unit='Bohr')
    with pytest.raises(ValueError, match='non-overlap'):
        late.aux_kernel(auxmol)
    with pytest.raises(ValueError, match='non-overlap'):
        aux_metric_sqrt(auxmol, late)
    # the admissible fixture keeps every site outside vdW contact
    d = np.linalg.norm(SITES[:, None, :] - mol.atom_coords()[None, :, :], axis=2)
    assert d.min() > MIN_SITE_TO_QM_DISTANCE
    assert np.isfinite(aux_metric_sqrt(auxmol, env)).all()


def test_an_over_polarizable_site_still_trips_the_metric_backstop():
    """The clearance guard bounds the DISTANCE, not the response, so the
    dressed metric's positivity check stays the backstop: a site at an
    admissible 7 Bohr with a polarizability no atom has over-screens the bare
    interaction and v + vtilde loses positivity. alpha = 200 Bohr^3 still
    screens (smallest eigenvalue 4.95e-4 against the bare 5.44e-4); 1000
    does not."""
    _, auxmol = molecules()
    site = np.array([[0.0, 0.0, 7.0]])
    assert np.isfinite(aux_metric_sqrt(
        auxmol, PolarizableSites(site, np.array([200.0]), unit='Bohr'))).all()
    runaway = PolarizableSites(site, np.array([1000.0]), unit='Bohr')
    with pytest.raises(RuntimeError, match='indefinite'):
        aux_metric_sqrt(auxmol, runaway)


def test_a_runaway_site_pair_is_not_a_response():
    """(alpha^-1 - T) loses positive definiteness for two alpha = 5 sites below
    about 2.2 Bohr, damped or not, and its inverse is then not a response: it
    answers some fields with a dipole antiparallel to them. Returning that
    inverse is worse than refusing it, because B enters every kernel silently.
    The message names the closest pair, which is where the catastrophe is."""
    for r in (1.0, 2.0):
        with pytest.raises(ValueError, match='sites 0 and 1'):
            response_matrix(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]]),
                            np.array([5.0, 5.0]))
    B = response_matrix(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]]),
                        np.array([5.0, 5.0]))
    assert np.linalg.eigvalsh(B).min() > 0.0


def test_the_kernel_falls_off_as_the_sites_recede():
    """The field goes as r^-2, so the kernel goes as r^-4: ten times further is
    ten thousand times weaker."""
    _, auxmol = molecules()
    near = PolarizableSites(np.array([[0., 0., 10.]]), np.array([4.0]), unit='Bohr')
    far = PolarizableSites(np.array([[0., 0., 100.]]), np.array([4.0]), unit='Bohr')
    ratio = (np.abs(near.aux_kernel(auxmol)).max()
             / np.abs(far.aux_kernel(auxmol)).max())
    assert 5e3 < ratio < 2e4


def test_aux_kernel_adjoint_against_finite_differences(env):
    """The adjoint carries a fixed v_bar to the nuclei; the finite difference
    moves the atoms and rebuilds the auxiliary basis with them, which is the
    only thing that moves -- the sites and B stay put."""
    mol, auxmol = molecules()
    rng = np.random.default_rng(5)
    v_bar = rng.normal(size=(auxmol.nao_nr(),) * 2)
    v_bar = 0.5 * (v_bar + v_bar.T)
    analytic = env.aux_kernel_adjoint(auxmol, v_bar)

    def value(dz):
        z = 0.117 + dz * BOHR_TO_ANGSTROM
        _, aux = molecules(f'O 0 0 {z}; H 0 0.757 -0.468; H 0 -0.757 -0.468')
        return float((v_bar * env.aux_kernel(aux)).sum())

    num = (value(H) - value(-H)) / (2.0 * H)
    assert abs(analytic[0, 2] - num) < 1e-6 * max(1.0, abs(num))


def test_the_adjoint_does_carry_a_net_force(env):
    """And it MUST. Translation invariance would zero the atomic force sum only
    if the sites moved with the molecule, and they do not -- they are the
    environment. Sliding the atoms past fixed sites really does change the
    energy, so a `sum(axis=0) == 0` assertion here, correct for every gas-phase
    gradient in this repository, would be wrong."""
    _, auxmol = molecules()
    rng = np.random.default_rng(6)
    v_bar = rng.normal(size=(auxmol.nao_nr(),) * 2)
    de = env.aux_kernel_adjoint(auxmol, 0.5 * (v_bar + v_bar.T))
    assert np.isfinite(de).all()
    assert np.abs(de).max() > 0.0


def test_the_static_term_is_absent_and_says_so(env):
    """None rather than zeros: a caller adds it unconditionally, and a zero
    array of the wrong shape would be silently added to the wrong thing."""
    mol, _ = molecules()
    assert env.static_self_energy(None, mol) is None


def test_mismatched_site_and_polarizability_counts_are_refused():
    with pytest.raises(ValueError, match='polarizabilities'):
        PolarizableSites(SITES, np.array([1.0, 2.0]), unit='Bohr')


def test_the_kernel_cache_distinguishes_two_geometries(env):
    """Keyed on basis CONTENT, not on `id(auxmol)`.

    A displaced geometry builds a fresh auxmol whose id python may reuse once
    the previous one is collected, so an id-keyed cache returns the reference
    kernel for a moved molecule -- intermittently, and looking reasonable. The
    churn below is what makes that reuse likely rather than hypothetical.
    """
    first = env.aux_kernel(molecules()[1]).copy()
    for _ in range(50):
        molecules('O 0 0 0.5; H 0 0.757 -0.468; H 0 -0.757 -0.468')
    moved = env.aux_kernel(
        molecules('O 0 0 0.5; H 0 0.757 -0.468; H 0 -0.757 -0.468')[1])
    assert np.abs(first - moved).max() > 1e-6
