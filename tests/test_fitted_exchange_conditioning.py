"""The exchange half of the fitted Fock skeleton
(`skeleton_tiles.fitted_fock_skeleton`) on an ill-conditioned fitting metric:
ethylene/aug-cc-pVDZ with aug-cc-pVDZ-JKFIT, whose metric's smallest
eigenvalue is 1.5e-5 (cc-pVDZ-JKFIT on the same molecule: 1e-3).

The skeleton at fixed AO densities is the derivative of
-(1/2) tr[g K[D]], K the fitted exchange, with the basis and the fit moving
with the atoms. A Richardson central difference of that energy is the
reference, every coordinate. Its two-centre term Vinv T Vinv formed by
matrix products loses digits as the square of the metric's condition number:
2.7e-8 Ha/Bohr from the difference here (about 1e-6 on a 21-atom
aug-cc-pVDZ molecule); taken as <g Kt^P, Kt^Q> it is 1.4e-10 from it.

The long-range channel of a range-separated hybrid (omega 0.2) fits on the
attenuated metric, singular here (Cholesky fails), through the eigenvalues
above pyscf's LINEAR_DEP_THR. With that pseudo-inverse formed and multiplied,
a last-bit change of the density moved the skeleton by 2.5e-8 to 1.2e-7
Ha/Bohr; applied as its eigenvector and eigenvalue factors, by about 1e-14.
"""
import pathlib
import sys

import numpy as np
from pyscf import df, gto, scf

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.gradients.skeleton_tiles import (FittedFockSkeleton,  # noqa: E402
                                          fitted_fock_skeleton)

GEOM = ('C 0.02 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; '
        'H 0 -0.9589 1.2321; H 0.03 0.9289 -1.2321; H 0 -0.9289 -1.2021')
BASIS = 'aug-cc-pvdz'
AUX = 'aug-cc-pvdz-jkfit'
#: Central-difference step (Bohr); Richardson of h and 2h.
STEP = 1e-3
#: What the skeleton may miss the difference by (Ha/Bohr), forces up to 0.7.
FD_BAR = 2e-9
#: Long-range operator of the range-separated channel (LRC-wPBEh's).
OMEGA = 0.2
#: Relative size of the last-bit change of the occupied factor.
LAST_BIT = 1e-16
#: How far that change may move the long-range skeleton (Ha/Bohr), forces
#: up to 3.9: a linear response of the change, not a fresh rounding draw.
LAST_BIT_BAR = 1e-12


def densities():
    """(mol, g, D, Cd): a symmetric partial g (random MO matrix) and the SCF
    density D = Cd Cd^T, AO basis."""
    mol = gto.M(atom=GEOM, basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    mf.conv_tol = 1e-10
    mf.kernel()
    C, n, nocc = mf.mo_coeff, mf.mo_coeff.shape[1], mol.nelectron // 2
    r = np.random.default_rng(7).standard_normal((n, n))
    g = C @ (0.5 * (r + r.T) / n) @ C.T
    Cd = np.sqrt(2.0) * C[:, :nocc]
    return mol, g, Cd @ Cd.T, Cd


def exchange_energy(mol, X, g, D):
    """-(1/2) tr[g K[D]] at coordinates X (Bohr), g and D held."""
    m = mol.copy()
    m.set_geom_(X, unit='Bohr')
    _, vk = scf.RHF(m).density_fit(auxbasis=AUX).get_jk(m, D, hermi=1, with_j=False)
    return -0.5 * float(np.einsum('ij,ij', g, vk))


def test_exchange_skeleton_is_the_derivative_on_an_ill_conditioned_metric():
    mol, g, D, Cd = densities()
    aux = df.addons.make_auxmol(mol, AUX)
    lam = float(np.linalg.eigvalsh(aux.intor('int2c2e'))[0])
    got = fitted_fock_skeleton(mol, aux, g, D, occ=Cd, coulomb=False,
                               exchange=1.0, comm=None)
    X0 = mol.atom_coords()
    fd = np.zeros_like(X0)
    for ia in range(mol.natm):
        for k in range(3):
            E = {}
            for s in (-2, -1, 1, 2):
                X = X0.copy()
                X[ia, k] += s * STEP
                E[s] = exchange_energy(mol, X, g, D)
            d1 = (E[1] - E[-1]) / (2 * STEP)
            d2 = (E[2] - E[-2]) / (4 * STEP)
            fd[ia, k] = (4.0 * d1 - d2) / 3.0
    d = float(np.abs(got - fd).max())
    print(f'metric smallest eigenvalue {lam:.1e}; |skeleton - difference| '
          f'{d:.2e} of {np.abs(fd).max():.2e}')
    assert d <= FD_BAR, d


def test_long_range_skeleton_is_stable_to_a_last_bit_change():
    mol, g, D, Cd = densities()
    aux = df.addons.make_auxmol(mol, AUX)

    def skeleton(Cd):
        sk = FittedFockSkeleton(mol, aux, g, Cd @ Cd.T, occ=Cd, coulomb=True,
                                exchange=1.0, omega=OMEGA)
        sk.prepare(range(len(sk.tiles)), None)
        assert sk.metric == 'eigen'
        return sum(sk.addend(t) for t in range(len(sk.tiles)))

    ref = skeleton(Cd)
    worst = 0.0
    for seed in range(4):
        r = np.random.default_rng(seed).standard_normal(Cd.shape)
        worst = max(worst, float(np.abs(skeleton(Cd * (1.0 + LAST_BIT * r))
                                        - ref).max()))
    print(f'long-range skeleton moved by {worst:.2e} of '
          f'{np.abs(ref).max():.2e} under a last-bit change')
    assert worst <= LAST_BIT_BAR, worst
