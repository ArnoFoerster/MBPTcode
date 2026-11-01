"""Gates for the phase tracking of `herzberg_teller.PhaseTrackedDipole`: the
finite-difference dmu/dq of a BSE@GW transition dipole.

An eigensolver returns each excited state with an arbitrary sign, and a
near-degenerate pair as an arbitrary rotation of itself; a central difference
of dipoles of two phases is not a derivative. The tracking aligns every
displaced solve to the reference one by the transported overlap of their
eigenvectors in the RPA metric, a group of near-degenerate roots as a
subspace (orthogonal Procrustes).

References: a model manifold with a known rotation inside a degenerate pair
(no electronic structure), and BSE@G0W0@PBE0/cc-pVDZ of formaldehyde, whose
S1 (n -> pi*, A2) is dipole-forbidden at the C2v minimum and borrows
intensity only along the b1 and b2 modes. The real gate holds the derivative
against random sign flips of every displaced eigenvector (bitwise) and
against the step (the h and h/2 stencils, to 2 % of the largest slope); with
the tracking removed the flips destroy it.
"""
import os
import sys

import numpy as np
import pytest
from pyscf import dft, gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.numerics.numerics import Numerics
from src.SingleReference.BSE.solve import solve_bse
from src.properties import herzberg_teller
from src.properties.duschinsky import CartesianHessian, harmonic_state
from src.properties.herzberg_teller import (ExcitedRoots, PhaseTrackedDipole,
                                            aligned_dipoles,
                                            degenerate_group,
                                            dipole_derivative)

#: Formaldehyde, the DF-PBE0/cc-pVDZ minimum (geomeTRIC, |g| < 3e-4), Bohr,
#: symmetrized to C2v in the yz plane.
CH2O = [('C', (0.0, 0.0, -1.115158)), ('O', (0.0, 0.0, 1.152957)),
        ('H', (0.0, 1.785261, -2.247859)), ('H', (0.0, -1.785261, -2.247859))]
NROOTS = 6
#: Relative bar on |dmu/dq(h) - dmu/dq(h/2)| for the forbidden S1, against
#: the largest slope: the stencils differ by 0.5 % (h^2 and solve noise).
STEP_TOL = 2e-2


# ------------------------------------------------- the model manifold
def model_roots(mol, mo, x, dipole, omega):
    return ExcitedRoots(mol, mo, mol.nelectron // 2, np.asarray(omega, float),
                        x, None, dipole)


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    nocc = mol.nelectron // 2
    nov = nocc * (mol.nao - nocc)
    rng = np.random.default_rng(3)
    x = np.linalg.qr(rng.normal(size=(nov, 4)))[0]      # <X|X> = 1, Y = 0
    dip = rng.normal(size=(4, 3))
    return mol, mf.mo_coeff, x, dip


def turned(x, dip, angle, signs, order):
    """The displaced solve of the model: roots 1 and 2 rotated by `angle`
    into each other, every root signed and the roots permuted. The dipole
    is linear in the vector, so it turns with it."""
    q = np.eye(x.shape[1])
    c, s = np.cos(angle), np.sin(angle)
    q[1:3, 1:3] = [[c, -s], [s, c]]
    q = q * np.asarray(signs, float)[None, :]
    q = q[:, order]
    return x @ q, q.T @ dip


def test_a_rotated_degenerate_pair_is_matched_as_a_subspace(water):
    """Roots 1 and 2 degenerate, the displaced solve returns them rotated by
    37 degrees, signed and permuted: the subspace match returns the reference
    dipoles exactly; a per-root sign does not."""
    mol, mo, x, dip = water
    omega = [0.30, 0.40, 0.40 + 1e-5, 0.55]
    ref = model_roots(mol, mo, x, dip, omega)
    xd, dd = turned(x, dip, np.radians(37.0), [1, -1, 1, -1], [3, 2, 0, 1])
    disp = model_roots(mol, mo, xd, dd, np.asarray(omega)[[3, 2, 0, 1]])
    assert degenerate_group(omega, 1) == (1, 2)
    got = aligned_dipoles(ref, disp, (1, 2))
    assert np.abs(got - dip[1:3]).max() < 1e-12
    for root in (0, 3):
        assert np.abs(aligned_dipoles(ref, disp, (root,))[0]
                      - dip[root]).max() < 1e-12
    # one root alone: the best sign of a 37-degree rotation is not the root
    alone = aligned_dipoles(ref, disp, (1,))[0]
    assert np.abs(alone - dip[1]).max() > 0.1 * np.abs(dip[1]).max()


def test_a_displaced_solve_without_the_roots_is_refused(water):
    """Displaced vectors orthogonal to the followed group: no phase can be
    assigned, and the dipole is refused rather than guessed."""
    mol, mo, x, dip = water
    ref = model_roots(mol, mo, x, dip, [0.3, 0.4, 0.5, 0.6])
    nov = x.shape[0]
    rng = np.random.default_rng(4)
    rest = rng.normal(size=(nov, 4))
    rest -= x[:, :2] @ (x[:, :2].T @ rest)
    rest = np.linalg.qr(rest)[0]
    disp = model_roots(mol, mo, rest, dip, [0.3, 0.4, 0.5, 0.6])
    with pytest.raises(ValueError, match='no phase can be assigned'):
        aligned_dipoles(ref, disp, (0, 1))


# --------------------------------------------- BSE@G0W0 of formaldehyde
@pytest.fixture(scope='module')
def formaldehyde():
    mol = gto.M(atom=CH2O, unit='Bohr', basis='cc-pvdz', verbose=0)
    cache = {}

    def mean_field(m):
        mf = dft.RKS(m).density_fit(auxbasis='cc-pvdz-jkfit')
        mf.xc, mf.conv_tol = 'pbe0', 1e-11
        return mf.run()

    def solve(m):
        key = m.atom_coords().tobytes()
        if key not in cache:
            mf = mean_field(m)
            omega, x, y, info = solve_bse(
                mf, m, numerics=Numerics(bse={'nroots': NROOTS}),
                solver='dense', integrals='df')
            cache[key] = ExcitedRoots.from_solve(mf, omega, x, y, info)
        return cache[key]

    state = harmonic_state(CartesianHessian(
        mol, mean_field(mol).Hessian().kernel(), 'pyscf analytic DF-PBE0'))
    return mol, state, solve


def flipping(solve, seed, reference):
    """`solve` with every root of every displaced solve signed at random,
    the dipoles with them; the solve at `reference` keeps its phase, which
    is the phase the derivative is expressed in."""
    rng = np.random.default_rng(seed)

    def flipped(m):
        r = solve(m)
        if np.array_equal(m.atom_coords(), reference.atom_coords()):
            return r
        s = rng.choice([-1.0, 1.0], size=len(r.omega))
        return ExcitedRoots(r.mol, r.mo_coeff, r.nocc, r.omega, r.X * s,
                            r.Y * s, r.dipole * s[:, None])
    return flipped


def test_bse_dipole_derivative_is_phase_and_step_stable(formaldehyde):
    """S1 (n -> pi*): forbidden at the minimum (mu0 ~ 0), dmu/dq nonzero
    along the b1 wag and the b2 modes only. The tracked derivative does not
    see random sign flips of the displaced eigenvectors (bitwise), agrees
    between steps h and h/2, and the untracked one under the same flips is a
    different number."""
    mol, state, solve = formaldehyde
    root = 0
    ref = solve(mol)
    assert np.linalg.norm(ref.dipole[root]) < 1e-6
    clean = dipole_derivative(PhaseTrackedDipole(solve, mol, root), state,
                              'ground')
    scale = np.abs(clean.dmu_dq).max()
    assert scale > 1e-3 and clean.suspect_modes == ()
    for seed in (1, 2):
        flipped = dipole_derivative(
            PhaseTrackedDipole(flipping(solve, seed, mol), mol, root), state,
            'ground')
        assert np.array_equal(flipped.dmu_dq, clean.dmu_dq)
    half = dipole_derivative(PhaseTrackedDipole(solve, mol, root), state,
                             'ground', step=0.5 * herzberg_teller.HT_FD_STEP)
    assert np.abs(half.dmu_dq - clean.dmu_dq).max() < STEP_TOL * scale
    # the same flips without the tracking: the slope is another number
    flips = flipping(solve, 1, mol)
    with pytest.warns(herzberg_teller.TransitionDipolePhase):
        raw = dipole_derivative(lambda m: flips(m).dipole[root], state,
                                'ground')
    assert np.abs(raw.dmu_dq - clean.dmu_dq).max() > 0.5 * scale


def test_an_allowed_bse_dipole_is_phase_stable(formaldehyde):
    """S2 (allowed): the tracked derivative under random flips is the
    unflipped one, bitwise, and no mode is reported suspect."""
    mol, state, solve = formaldehyde
    clean = dipole_derivative(PhaseTrackedDipole(solve, mol, 1), state,
                              'ground')
    flipped = dipole_derivative(PhaseTrackedDipole(flipping(solve, 5, mol), mol, 1),
                                state, 'ground')
    assert np.array_equal(flipped.dmu_dq, clean.dmu_dq)
    assert clean.suspect_modes == ()
