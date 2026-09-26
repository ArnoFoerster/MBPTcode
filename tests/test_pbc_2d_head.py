"""The 2D head restored, as an alternative to damping on a slab.

pyscf's `dimension=2` kernel puts `coulG[G=0] = -pi L_z^2 / 2`, the only
negative entry in the whole kernel, and that single value is what makes the
RI-V metric indefinite -- which is the entire reason the AUTO spherical damped
kernel was adopted for slabs.

THAT VALUE IS A DISCARDED DIVERGENCE, NOT THE KERNEL. The Sundararaman-Arias
form has two different G -> 0 limits (+pi L^2/2 along G_z, and
2 pi L/q - pi L^2/2 along G_par); pyscf takes the G_par branch and keeps only
its finite part. The dropped `L * (2 pi / q)` is the 2D head that
`head_average_2d` already integrates over the mini-BZ in closed form, so it can
be put back.

WHY IT MATTERS, and it is a cost argument rather than an accuracy one: the
damped kernel's real-space support must fit inside half the cell height, so
refining the in-plane k-mesh FORCES a taller vacuum -- measured with this
repo's own `nyquist_params`/`damping_support`, MoS2 needs 24.6 A of cell height
at 12x12 and 61.6 A at 30x30. The z FFT mesh grows with L_z at fixed cutoff, so
every grid-touching step (the ISDF fit, `coulomb_matrix_q`) then goes as N_k^3
in-plane instead of N_k^2. Restoring the head removes the reason for damping
and with it that coupling.

THE TWO ROUTES WANT OPPOSITE THINGS, which is the part worth remembering:
damping needs the vacuum TALL relative to the mesh; the restored head needs the
mesh FINE relative to the vacuum (R < 8/L_z). Production wants fine meshes, so
the trade is favourable -- but the head route genuinely fails on a coarse mesh
over a tall cell, and it raises there rather than returning an indefinite
metric.

The numbers below were measured before the head was implemented.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import lib
from pyscf.pbc import gto, tools
from pyscf.pbc.df import ft_ao

from src.SingleReference.Periodic.pbc_rpa import (coulomb_metric_inv_sqrt,
                                                  make_auxcell)
from src.SingleReference.Periodic.pbc_rpa_damping import (damping_support,
                                                          nyquist_params)
from src.SingleReference.Periodic.pbc_smallq import (head_average_2d,
                                                     make_coulG_2d_head,
                                                     minibz_disc_radius,
                                                     sa_head_normalization)


def _slab(a_ang=4.0, Lz=16.0, mesh=(13, 13, 48)):
    cell = gto.Cell()
    cell.atom = 'H 0 0 -0.37; H 0 0 0.37'
    cell.a = np.diag([a_ang, a_ang, Lz])
    cell.basis, cell.pseudo = 'gth-dzvp', 'gth-pade'
    cell.dimension, cell.verbose = 2, 0
    cell.mesh = list(mesh)
    cell.build()
    return cell


def _metric(cell, coulG):
    """The RI-V Coulomb metric J this kernel produces."""
    Gv = cell.get_Gv(cell.mesh)
    _, _, kws = cell.get_Gv_weights(cell.mesh)
    aux = make_auxcell(cell)
    auxG = ft_ao.ft_ao(aux, Gv, kpt=np.zeros(3))
    vG = np.asarray(coulG) * np.asarray(kws)
    return lib.einsum('gP,gQ->PQ', auxG.conj() * vG[:, None], auxG)


# --- the premise ------------------------------------------------------------

def test_pyscf_2d_head_is_exactly_minus_pi_L2_over_2():
    """And it is the ONLY negative entry, which is what makes this fixable."""
    for Lz in (16.0, 24.0, 30.0):
        cell = _slab(Lz=Lz)
        L, finite = sa_head_normalization(cell)
        Gv = cell.get_Gv(cell.mesh)
        cg = tools.get_coulG(cell, k=np.zeros(3), mesh=cell.mesh, Gv=Gv)
        i0 = int(np.argmin(np.einsum('gx,gx->g', Gv, Gv)))
        assert cg[i0] == pytest.approx(finite, rel=1e-10), (Lz, cg[i0], finite)
        assert int((cg < 0).sum()) == 1, (Lz, int((cg < 0).sum()))


def test_the_normalization_carries_exactly_one_factor_of_L():
    """THE trap: the 3D-convention kernel takes L * (2 pi / q), not 2 pi / q.

    Derived rather than guessed, because a plausible wrong normalization here
    gives a plausible wrong head. The Sundararaman-Arias form along G_par is
    4 pi/q^2 [1 - exp(-q L/2)] = 2 pi L/q - pi L^2/2 + O(q), so subtracting the
    finite part and multiplying by q/(2 pi L) must tend to 1.
    """
    L = 45.35
    for q in (1e-3, 1e-4, 1e-5):
        v_sa = 4 * np.pi / q ** 2 * (1 - np.exp(-q * L / 2))
        ratio = (v_sa + np.pi * L ** 2 / 2) * q / (2 * np.pi * L)
        assert abs(ratio - 1.0) < 2e-4 * (q / 1e-4 + 1), (q, ratio)
    # and the tightest point is essentially exact
    q = 1e-5
    v_sa = 4 * np.pi / q ** 2 * (1 - np.exp(-q * L / 2))
    assert abs((v_sa + np.pi * L ** 2 / 2) * q / (2 * np.pi * L) - 1) < 1e-7


# --- the fix ----------------------------------------------------------------

def test_restored_head_makes_the_metric_psd_without_damping():
    """THE GATE: no damping, no assert_damping_fits, and the metric is accepted."""
    cell = _slab(Lz=16.0)
    kmesh = [4, 4, 1]
    Gv = cell.get_Gv(cell.mesh)

    bare = tools.get_coulG(cell, k=np.zeros(3), mesh=cell.mesh, Gv=Gv)
    with pytest.raises(ValueError, match='not positive definite'):
        coulomb_metric_inv_sqrt(_metric(cell, bare))

    cg = make_coulG_2d_head(cell, kmesh)
    v = cg(cell, np.zeros(3), Gv)
    assert v.min() >= 0.0, v.min()
    J = _metric(cell, v)
    coulomb_metric_inv_sqrt(J)                     # must not raise
    assert np.linalg.eigvalsh(J).min() > 0


def test_restored_head_equals_L_times_the_minibz_average():
    """The substituted value is exactly L <2pi/q> + the finite part."""
    cell = _slab(Lz=16.0)
    kmesh = [4, 4, 1]
    L, finite = sa_head_normalization(cell)
    cg = make_coulG_2d_head(cell, kmesh, shape='disc')
    R = minibz_disc_radius(cell, kmesh)
    assert cg.head_2d[1] == pytest.approx(L * head_average_2d(R) + finite)


def test_a_mini_bz_too_large_for_the_vacuum_is_refused():
    """The route's OWN failure mode, and it must be loud.

    Restoring the head is not unconditionally positive: it needs
    R < 8 / L_z. Measured, a 24 A cell at 2x2 gives v(G=0) = -814.
    """
    cell = _slab(Lz=24.0)
    L, _ = sa_head_normalization(cell)
    R = minibz_disc_radius(cell, [2, 2, 1])
    assert R > 8.0 / L                             # the premise
    with pytest.raises(ValueError, match='mini-BZ is too large'):
        make_coulG_2d_head(cell, [2, 2, 1])


def test_the_head_is_ill_conditioned_near_the_crossover():
    """The head is a DIFFERENCE OF TWO LARGE NUMBERS, so quote it with its cell.

    `L<2pi/q>` and `pi L^2/2` are both O(1e3) here while their difference is
    O(1e2), so a small change in the cell geometry swings the head by a large
    RELATIVE amount. Two calculations of "graphene at 24 A, 4x4" gave -449
    against -242: one used the true hexagonal cell, the other
    a square placeholder at the same lattice parameter. Neither was wrong --
    R scales as 1/sqrt(A_cell), the areas differ by 15%, so R differs by 7.5%
    and the head by a factor of 1.9.

    The consequence for production is not a bug but a margin: satisfying
    R < 8/L_z only just is not enough, because the value you get there is
    sensitive to details that do not otherwise matter.
    """
    cell = _slab(Lz=24.0)
    L, finite = sa_head_normalization(cell)

    amp, sens = [], []
    for nk in (4, 6, 8):                       # coarse -> fine, i.e. toward safety
        R = minibz_disc_radius(cell, [nk, nk, 1])
        head = L * head_average_2d(R)
        total = head + finite
        amp.append(abs(head) / abs(total))
        # a 7.5% change in R -- exactly the hexagonal-vs-square cell area
        # difference that produced the two calculations' 2x disagreement
        sens.append(abs(L * head_average_2d(R * 1.075) + finite - total) / abs(total))

    # Cancellation, and it is WORST nearest the crossover: measured
    # amplification 2.98 / 1.80 / 1.50 at 4x4 / 6x6 / 8x8.
    assert amp == sorted(amp, reverse=True), amp
    assert amp[0] > 2.5, amp
    # and there a 7.5% geometric change is a 21% change in the head
    assert sens == sorted(sens, reverse=True), sens
    assert sens[0] > 0.15, sens


def test_the_two_routes_want_opposite_things():
    """Damping needs vacuum >= f(mesh); the head needs mesh fine enough.

    This is the cost argument in one assertion: the damped kernel's minimum
    cell height GROWS with the in-plane mesh, while the head route's condition
    R < 8/L_z gets EASIER as the mesh is refined.
    """
    cell = _slab(Lz=24.0)
    L, _ = sa_head_normalization(cell)
    need_vacuum, head_ok = [], []
    for nk in (4, 6, 12, 18):
        r0, beta, _ = nyquist_params(cell, [nk, nk, 1])
        need_vacuum.append(2 * damping_support(r0, beta))
        head_ok.append(minibz_disc_radius(cell, [nk, nk, 1]) < 8.0 / L)
    # damping: strictly more vacuum demanded as the mesh is refined
    assert need_vacuum == sorted(need_vacuum) and need_vacuum[0] < need_vacuum[-1]
    # head: once admissible, refining never breaks it
    assert head_ok == sorted(head_ok), head_ok
    assert head_ok[-1]


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-s']))
