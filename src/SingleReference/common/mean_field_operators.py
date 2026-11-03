"""Mean-field operators the quasiparticle methods and their gradients share.

`xc_potential` is the exchange-correlation part of a mean field's own Fock
build, and `mean_field_static_operator` is Sigma_x - v_xc at the mean field's
own density, built once per mean field and kept beside it.
"""
import hashlib
import threading
import weakref

import numpy as np
from pyscf import gto as _pyscf_gto

from src.Base.factorization.distributed_df import distributed_fock
from src.Base.utils.parallel.mpi_grid import current_comm, reduce_max
from src.Base.utils.parallel.threads import blas_single_threaded


def xc_potential(mf, mol, dm):
    """(v_xc, J): v_xc = v_eff - J[D], exact exchange included, J of the
    total density -- the operator the eigenvalues came from, less the Hartree
    term every spin sees alike; restricted (nao, nao), unrestricted
    (2, nao, nao) for dm (2, nao, nao)."""
    total = dm[0] + dm[1] if np.ndim(dm) == 3 else dm
    v_eff = mf.get_veff(mol, dm)
    j = mf.get_j(mol, total)
    return v_eff - j, j


#: each restricted mean field's Sigma_x - v_xc at its own density, with the
#: content it was built for (`mean_field_static_operator`)
_STATIC_OPERATORS = weakref.WeakKeyDictionary()
_STATIC_OPERATORS_LOCK = threading.Lock()


def mean_field_static_operator(mf, mol=None):
    """Sigma_x - v_xc = -K/2 - (v_eff - J) in the AO basis at a restricted
    mean field's own density, from its own K, v_eff and J; read-only.

    THE ONE OPERATOR the quasiparticle static term (`_local_static_exchange`
    at exchange='mf'), the Kohn-Sham static correction's orbital response and
    the EXX double counting's (`gradients.isdf_derivatives`) all read: built
    once per mean field and kept beside it (a weak reference, so it goes with
    the mean field) for as long as its density, molecule, functional and
    fit are the ones it was built from; any change rebuilds it.

    Over ranks the build may be a collective (`distributed_fock`), so a
    held operator is used only where every rank holds one.
    """
    mol = mf.mol if mol is None else mol
    dm = mf.make_rdm1()        # tagged with its orbitals, as pyscf builds K
    key = (hashlib.blake2b(np.ascontiguousarray(dm).data,
                           digest_size=32).digest(),
           mol._atm.tobytes(), mol._bas.tobytes(),
           mol._env[_pyscf_gto.mole.PTR_ENV_START:].tobytes(),
           getattr(mf, 'xc', None), id(getattr(mf, 'with_df', None)))
    with _STATIC_OPERATORS_LOCK:
        held = _STATIC_OPERATORS.get(mf)
    hit = held is not None and held[0] == key
    comm = current_comm()
    if comm is not None and comm.Get_size() > 1:
        hit = reduce_max(np.array([0.0 if hit else 1.0]), comm)[0] == 0.0
    if hit:
        return held[1]
    # K and the exchange-correlation potential are pyscf's own OpenMP -- the
    # fit's contraction and `nr_rks` on the DFT grid -- so BLAS is held at one
    # thread across them and the two pools stop spinning against each other.
    with distributed_fock(mf, build=False), blas_single_threaded():
        K = mf.get_k(mol, dm)
        op = -0.5 * K - xc_potential(mf, mol, dm)[0]
    op.setflags(write=False)
    with _STATIC_OPERATORS_LOCK:
        _STATIC_OPERATORS[mf] = (key, op)
    return op
