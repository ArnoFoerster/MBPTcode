"""One factorization of the PCM matrix per geometry, not two solves per cycle.

pyscf's polarizable continuum solves K q = R v and K^T x = v afresh on every
SCF iteration, each a dense O(n^3) in the number of surface points. K is built
from the cavity and the dielectric constant alone, so it does not change while
the density iterates: the factorization belongs outside the loop and only the
triangular solves belong inside.

WHAT THIS IS AND IS NOT WORTH. The triangular solves are 80 times cheaper than
the dense ones at anthracene's size and more above it, but measured against a
whole iteration they are a small part of it: on anthracene at Lebedev order 29,
a cycle spends 28 s in the exchange-correlation potential, 21 s building the
surface potential of the density and putting the charges back on the basis,
and 1.4 s in these two solves. So this is a few percent, taken because it is
free and exact, and not a cure.

The cost that dominates the continuum is the SURFACE POTENTIAL, which is
n_ao^2 x n_surf, and it shares its cubic growth with everything else. Both it
and these solves fall with the number of surface points, which is the knob that
actually moves: on formaldehyde the solvation energy changes by 0.4 meV between
Lebedev order 11 and order 29 while the potential costs four times more at 29.
"""
import numpy as np
import scipy.linalg

#: Attribute the factorization is cached under, on the PCM object itself.
_CACHE = '_wicks_pcm_lu'


def _factorization(obj, K):
    """(lu, luT) for this K, rebuilt whenever `build` replaces the matrix."""
    cached = getattr(obj, _CACHE, None)
    if cached is not None and cached[0] is K:
        return cached[1], cached[2]
    lu, lu_t = scipy.linalg.lu_factor(K), scipy.linalg.lu_factor(K.T)
    setattr(obj, _CACHE, (K, lu, lu_t))
    return lu, lu_t


def _get_vind(self, dms):
    """pyscf's own reaction-field step with the two dense solves factorized.

    The body follows `pyscf.solvent.pcm.PCM._get_vind` exactly; only the two
    `numpy.linalg.solve` calls become triangular solves against a cached LU.
    A gate asserts this against the unpatched routine, so a change upstream
    shows up as a disagreement rather than as silence.
    """
    if not self._intermediates:
        self.build()

    nao = dms.shape[-1]
    dms = dms.reshape(-1, nao, nao)
    if dms.shape[0] == 2:
        dms = (dms[0] + dms[1]).reshape(-1, nao, nao)

    K = self._intermediates['K']
    R = self._intermediates['R']
    lu, lu_t = _factorization(self, K)
    v_grids_e = self._get_v(dms)
    v_grids = self.v_grids_n - v_grids_e

    b = np.dot(R, v_grids.T)
    q = scipy.linalg.lu_solve(lu, b).T

    vK_1 = scipy.linalg.lu_solve(lu_t, v_grids.T)
    qt = np.dot(R.T, vK_1).T
    q_sym = (q + qt) / 2.0

    vmat = self._get_vmat(q_sym)
    epcm = 0.5 * np.dot(q_sym[0], v_grids[0])

    self._intermediates['q'] = q[0]
    self._intermediates['q_sym'] = q_sym[0]
    self._intermediates['v_grids'] = v_grids[0]
    self._intermediates['dm'] = dms
    return epcm, vmat[0]


def factorize_once(with_solvent):
    """Give one PCM object the cached factorization, and return it.

    Idempotent, and safe on an object that is rebuilt: the cache is keyed on
    the identity of K, so a `build()` that replaces the matrix invalidates it.
    """
    if with_solvent is None or getattr(with_solvent, '_wicks_factorized', False):
        return with_solvent
    with_solvent._get_vind = _get_vind.__get__(with_solvent)
    with_solvent._wicks_factorized = True
    return with_solvent
