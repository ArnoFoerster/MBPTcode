"""The two-body term of the downfolded Hamiltonian and the packing of g_eff.

`active_eri` gives the bare interaction on the active orbitals from exact
4-centre integrals; `pack` turns the assembled one- and two-body terms into
the g_eff dictionary the active-space solvers read (src/Solvers/active_space.py).
"""
import numpy as np
from pyscf import ao2mo

from src.Solvers.exact_diagonalization import spatial_to_spin_orbitals

RESTRICTED_TOL = 1e-12


def active_eri(mol, mo_act):
    """(aa, ab, bb), each (n_act,)*4: the exact chemist integrals (tu|vw) of the
    active orbitals, with ab[t,u,v,w] = (t_a u_a | v_b w_b).

    mo_act = (C_Aa, C_Ab) are the active orbital coefficients of the two spin
    channels. The integrals are the exact 4-centre ones of `mol`, never from a
    density-fitted object, because the bare v must not carry a fitting error.
    An empty active space gives zero-size blocks without a transformation.
    """
    ca, cb = mo_act
    return (_block(mol, ca, ca), _block(mol, ca, cb), _block(mol, cb, cb))


def _block(mol, c1, c2):
    n1, n2 = c1.shape[1], c2.shape[1]
    if n1 == 0 or n2 == 0:
        return np.zeros((n1, n1, n2, n2))
    eri = ao2mo.general(mol, (c1, c1, c2, c2), compact=False)
    return eri.reshape(n1, n1, n2, n2)


def _same(x, y):
    if x.shape != y.shape:
        return False
    if x.size == 0:
        return True
    scale = max(1.0, float(np.max(np.abs(x))), float(np.max(np.abs(y))))
    return np.max(np.abs(x - y)) <= RESTRICTED_TOL * scale


def pack(h_eff, v_eff, e0, spin_free):
    """The g_eff dictionary {0: e0, 1: one-body, 2: two-body} of the solvers.

    h_eff = (h_a, h_b), v_eff = (aa, ab, bb) are the chemist active
    interaction (bare plus vtilde). Spin-free, g_eff[1] is the spatial
    one-body and g_eff[2] the spatial physicist <pq|rs> = (pr|qs); otherwise
    both go through spatial_to_spin_orbitals and key 0 is added here.

    Only a restricted Hamiltonian is packed: the two spin channels of h_eff
    and the three blocks of v_eff must agree to 1e-12 (relative to max(1, max
    element)). The arrays of g_eff are copies, never views of the inputs; an
    empty active space packs zero-size arrays. A spin-resolved g_eff for the
    unrestricted case, solved by fci.direct_uhf, is deferred.
    """
    h_a, h_b = h_eff
    aa, ab, bb = v_eff
    if not (_same(h_a, h_b) and _same(aa, ab) and _same(aa, bb)):
        raise NotImplementedError(
            "pack takes a restricted Hamiltonian only (equal alpha and beta "
            "channels); the unrestricted extension, a spin-resolved g_eff "
            "solved with fci.direct_uhf, is deferred.")
    spatial = {1: np.array(h_a), 2: np.array(aa.transpose(0, 2, 1, 3))}
    if spin_free:
        return {0: e0, **spatial}
    g = spatial_to_spin_orbitals(spatial, h_a.shape[0])
    g[0] = e0
    return g
