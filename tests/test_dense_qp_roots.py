"""Every orbital of the dense quasi-boson reference carries its quasiparticle.

The all-orbital dense route is the reference the sum-over-poles route is
measured against, so every orbital is solved explicitly and none falls back to
its mean-field level. Every root of the diagonal quasiparticle equation
w = eps_p + Sigma_pp(w) is an eigenvalue of the arrowhead supermatrix, its pole
strength Z the eigenvector's weight on the bare orbital, and the weights sum
to one. The quasiparticle is the root of largest Z, and these tests hold the
declared route to it:

- on water/cc-pVDZ every orbital's root is the largest-weight eigenvalue of the
  dense supermatrix;
- on formaldehyde/cc-pVDZ the virtuals above 38 eV, where a Newton from eps_p
  stops on a satellite with Z down to 1e-3, carry the largest-Z root of an
  independent enumeration of every root;
- the all-orbital force is the derivative of the all-orbital energy, the
  central difference converging onto it at second order;
- the formaldehyde S1 walk on the all-orbital surface converges.

The Newton root from eps_p is a satellite on 7 water orbitals (orbital 17 at
65.95 eV, Z = 0.011, against its quasiparticle at 62.99 eV, Z = 0.292) and on
15 of formaldehyde's 16 virtuals above 38 eV (orbital 22 at 38.49 eV,
Z = 0.012, against 36.22 eV, Z = 0.307); on those roots the formaldehyde S1
walk stops with its trust radius collapsed at 1.1e-3 Bohr. The force check
fails at 5.6e-3 Ha/Bohr on a force without the boson-amplitude response.
"""
import os
import sys

import numpy as np
import pytest
from pyscf import gto

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import QB_QP_ROOT_TOL
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.eri_blocks import mo_eri
from src.SingleReference.GW.quasi_boson import QPqb, largest_weight_root
from src.SingleReference.LinearResponse.quasi_boson_bse import BSEqb
from src.gradients.dense_surfaces import tight_rhf
from src.properties.optimize import optimize
from src.properties.surfaces import potential_energy_surface
from tests.test_dense_surfaces import FD_TOL

#: A root found by `largest_weight_root` sits within QB_QP_ROOT_TOL of the root
#: of the equation it solved (its last Newton step), and that equation within
#: QB_QP_ROOT_TOL of the full one (the couplings it dropped, by Weyl).
ROOT_TOL = 2.0 * QB_QP_ROOT_TOL

#: Central-difference steps (Bohr) of the force check. An all-orbital surface
#: carries continuum orbitals whose largest-weight root changes branch within
#: 1e-3 Bohr (water orbital 22 between 5e-4 and 1e-3), so the steps sit below
#: that, where the difference still falls fourfold per halving.
FD_STEPS = (2.5e-4, 1.25e-4)


def water():
    return gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                 basis='cc-pvdz', verbose=0)


def formaldehyde():
    return gto.M(atom='C 0 0 -0.5296; O 0 0 0.6742; H 0 0.9429 -1.1123; '
                      'H 0 -0.9429 -1.1123', basis='cc-pvdz', verbose=0)


def bare_quasiparticles(mol):
    """The quasiparticle solver on `mol`'s reference mean field."""
    mf = tight_rhf(mol)
    return QPqb(mf.mo_energy, mo_eri(mf, mol), mol.nelectron // 2)


def all_orbital_surface(mol, spin='singlet'):
    """The reference as declared: dense, every orbital explicit."""
    return potential_energy_surface(
        mol, tight_rhf, ground_state=GroundState('rpa', 'hf'),
        excitation=Excitation(spin), chi0='dense-qb',
        factorization='four-index', qp_states=QPStates('all'))


def every_root(qp, p):
    """(w, Z) of every root of orbital p's equation, by bisection on each
    interval between neighbouring poles: an enumeration that shares nothing
    with `largest_weight_root` but the secular function."""
    nh, ph, npp, pp = qp._pole_arrays(p)
    num = np.concatenate([nh.ravel(), npp.ravel()])
    pol = np.concatenate([ph.ravel(), pp.ravel()])
    # couplings forbidden by symmetry (|W| ~ 1e-17) hold roots no bisection
    # can separate from their pole; they carry no weight
    keep = num > num.max() * np.finfo(float).eps
    num, pol = num[keep], pol[keep]
    order = np.argsort(pol)
    num, pol = num[order], pol[order]
    a = qp.eps[p] + qp.delta[p]
    reach = 10.0 * (1.0 + np.sqrt(num.sum()) + abs(a))
    lo = np.concatenate([[pol[0] - reach], pol])
    hi = np.concatenate([pol, [pol[-1] + reach]])
    # a midpoint can land on a pole: f is then -inf or +inf, its sign still right
    with np.errstate(divide='ignore', invalid='ignore'):
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            f = mid - a - (num[None, :] / (mid[:, None] - pol[None, :])).sum(1)
            lo, hi = np.where(f > 0, lo, mid), np.where(f > 0, mid, hi)
        w = 0.5 * (lo + hi)
        z = 1.0 / (1.0 + (num[None, :]
                          / (w[:, None] - pol[None, :]) ** 2).sum(1))
    return w, z


def test_every_water_orbital_carries_the_supermatrix_quasiparticle():
    """The diagonal holds, orbital by orbital, the eigenvalue of the arrowhead
    supermatrix with the largest weight on the bare orbital."""
    surface = all_orbital_surface(water())
    qp = bare_quasiparticles(surface.mol0)
    assert surface.qp_set == list(range(surface.mol0.nao))
    for p in surface.qp_set:
        w, z = qp.solve_diag_dense(p)
        assert abs(surface.seeds[p] - w) < ROOT_TOL, (p, surface.seeds[p], w)
        assert surface.qp_z[p] == pytest.approx(z, abs=ROOT_TOL), p


def test_no_formaldehyde_virtual_carries_a_satellite():
    """The virtuals above 38 eV, where the nearest root is a satellite, carry
    the root of largest Z among every root of their equation."""
    surface = all_orbital_surface(formaldehyde())
    qp = bare_quasiparticles(surface.mol0)
    for p in range(22, surface.mol0.nao):
        w, z = every_root(qp, p)
        best = np.argmax(z)
        assert abs(surface.seeds[p] - w[best]) < ROOT_TOL, (
            p, surface.seeds[p], w[best], z[best])
        assert surface.qp_z[p] == pytest.approx(z[best], abs=ROOT_TOL), p
    assert surface.qp_z[22] > 0.3, surface.qp_z[22]


def test_the_all_orbital_force_is_the_derivative_of_its_energy():
    """The finer central difference of the energy and the Richardson
    extrapolation of the two meet the force to FD_TOL."""
    surface = all_orbital_surface(water())
    force, _, _ = surface.total_gradient()
    x0 = surface.mol0.atom_coords()
    differences = []
    for h in FD_STEPS:
        g = np.zeros_like(x0)
        for atom in range(x0.shape[0]):
            for axis in range(3):
                e = []
                for sign in (1.0, -1.0):
                    mol = surface.mol0.copy()
                    x = x0.copy()
                    x[atom, axis] += sign * h
                    mol.set_geom_(x, unit='Bohr')
                    e.append(surface.total_energy(mol))
                g[atom, axis] = (e[0] - e[1]) / (2.0 * h)
        differences.append(g)
    fine = np.max(np.abs(differences[1] - force))
    extrapolated = (4.0 * differences[1] - differences[0]) / 3.0
    assert fine < FD_TOL, fine
    assert np.max(np.abs(extrapolated - force)) < FD_TOL


def test_the_formaldehyde_s1_walk_converges():
    """The walk on the largest-weight roots reaches its minimum."""
    surface = all_orbital_surface(formaldehyde())
    _, info = optimize(surface, max_cycle=100, verbose=False)
    assert info['converged'], info['status']


def test_the_rule_is_refused_with_a_seed_and_by_an_unknown_name():
    """A seed would be ignored under 'weight', so it is refused."""
    mol = water()
    mf = tight_rhf(mol)
    eri = mo_eri(mf, mol)
    with pytest.raises(ValueError, match='follows no seed'):
        BSEqb(mf, eri, 5, qp_orbs=[4], seeds={4: -0.5}, qp_root='weight')
    with pytest.raises(ValueError, match='qp_root'):
        BSEqb(mf, eri, 5, qp_orbs=[4], qp_root='nearest')


@pytest.mark.parametrize('seed', range(4))
def test_the_largest_weight_root_is_the_arrowhead_eigenvalue(seed):
    """Random poles, crowded and sparse, with weights down to the round-off
    floor: the root is the arrowhead matrix's eigenvalue of largest weight."""
    rng = np.random.default_rng(seed)
    n = 300
    pol = np.concatenate([rng.uniform(-3.0, -0.5, n // 2),
                          rng.uniform(0.5, 4.0, n - n // 2)])
    num = rng.uniform(0.0, 1e-3, n) * (rng.uniform(size=n) > 0.3)
    num[rng.integers(0, n, 20)] = 1e-34
    a = rng.uniform(-1.0, 1.0)
    w, z = largest_weight_root(num, pol, a)
    h = np.diag(np.concatenate([[a], pol]))
    h[0, 1:] = h[1:, 0] = np.sqrt(num)
    vals, vecs = np.linalg.eigh(h)
    best = np.argmax(vecs[0] ** 2)
    assert abs(w - vals[best]) < ROOT_TOL
    assert z == pytest.approx(vecs[0, best] ** 2, abs=ROOT_TOL)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
