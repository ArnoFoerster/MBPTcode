"""The two-electron term of the spin-orbit mean-field operator, built once over
the ranks.

The spin-orbit mean-field reduction of Neese (J. Chem. Phys. 122, 034107
(2005), Eq. 19) needs J - (3/2) K over `int2e_p1vxp1` at the ground-state
density: a pass over nao^4 x 3 integrals (or sum_A nao_A^4 in the atomic
mean-field approximation), the one step of `properties.spin_orbit` whose cost
grows with the fourth power of the basis. Under a communicator rank 0 runs it
from its own density and `lockstep` sends the (3, nao, nao) result to every
rank; the other ranks run no integral pass, and every rank holds rank 0's
operator to the bit whatever density it was handed. Outside a communicator it
is the plain serial build, the same bits.

The resolution of the identity ('ri-somf') is the cheaper route to the same
operator. The density side of every integral is fitted in the Coulomb metric of
a named auxiliary basis,

    (ij|kl)^SO ~ sum_PQ (ij|P)^SO [V^-1]_PQ (Q|kl),

with (ij|P)^SO the three-centre `int3c2e_pvxp1`, which is the RI-SOMF of
Neese's paper with its exchange kept whole (no one-centre restriction). With
dm = sum_m w_m u_m u_m^T,

    J^eta_ij = sum_P (ij|P)^SO c_P,          c = V^-1 (P|kl) dm_kl
    K^eta    = K1 - K1^T,   K1_il = sum_mP [sum_j (ij|P)^SO u_jm w_m] D_Pml,
    D_Pml    = sum_Q [V^-1]_PQ sum_k (Q|kl) u_km ,

so its cost is nao^2 naux three-centre integrals and nao^2 naux nocc of GEMM
in place of the nao^4 pass, and its error falls with the auxiliary basis.
It divides over the ranks by tiles of the auxiliary shells: a tile's
spin-orbit integrals, its J and K rows and its fitted D rows are one rank's.
D is evaluated by tiles of AO columns (each needs the whole metric) and one
exchange per round of column tiles turns its columns into the auxiliary rows
the tile owners hold. The tiles' terms add in tile order (`ordered_chain_sum`),
the tiles are cut from the basis and SOMF_RI_TILE_GB alone, and the factors
every tile reads (the metric's Cholesky factor and the density's eigen-factor)
are rank 0's: the result is the same bits at every rank count.

This is a realization kernel: `comm` defaults to `current_comm()`, and the
property code that calls it never sees a communicator.
"""
import copy

import numpy as np
import scipy.linalg
from pyscf import df, lib
from pyscf.scf import _vhf

from src.Base.numerics.constants import (SOC_TWO_ELECTRON_TERMS,
                                         SOMF_RI_TILE_GB)
from src.Base.utils.parallel.mpi_grid import (broadcast, contiguous_block,
                                              current_comm, exchange_blocks,
                                              lockstep, ordered_chain_sum,
                                              ordered_sum)

#: The two-electron reductions `mean_field_term` builds.
TWO_ELECTRON_TERMS = SOC_TWO_ELECTRON_TERMS

#: Eigenvalues of the density below this are dropped from its factor: they
#: carry no electrons, and a factor column of zero weight is GEMM work alone.
DENSITY_FACTOR_TOL = 1e-12


def _soc_jk(mol, dm):
    """(vj, vk) of the two-electron SOC operator at density `dm`, (3, nao, nao).

    K here is the sum of BOTH exchange contractions, (pi|iq) + (iq|pi), which is
    what the SOMF factor of 3/2 multiplies.
    """
    vj, vk, vk1 = _vhf.direct_mapdm(mol._add_suffix('int2e_p1vxp1'),
                                    'a4ij', ('lk->s2ij', 'jk->s1il', 'li->s1kj'),
                                    dm, 3, mol._atm, mol._bas, mol._env)
    for i in range(3):
        lib.hermi_triu(vj[i], hermi=2, inplace=True)
    return vj, vk + vk1


def _soc_jk_amfi(mol, dm):
    """(vj, vk) in the atomic-mean-field approximation: one-centre blocks only.

    The two-electron SOC operator is short-ranged and dominated by the nucleus
    it is centred on, so the multi-centre blocks are discarded. Cost falls from
    nao^4 to sum_A nao_A^4, which is what makes a 60-90 atom emitter reachable.
    """
    nao = mol.nao_nr()
    vj = np.zeros((3, nao, nao))
    vk = np.zeros((3, nao, nao))
    atom = copy.copy(mol)
    for b0, b1, p0, p1 in mol.aoslice_by_atom():
        atom._bas = mol._bas[b0:b1]
        vj1, vk1 = _soc_jk(atom, np.asarray(dm)[p0:p1, p0:p1])
        vj[:, p0:p1, p0:p1] = vj1
        vk[:, p0:p1, p0:p1] = vk1
    return vj, vk


def _shell_tiles(ao_loc, width):
    """[(s0, s1, p0, p1)]: consecutive shells grouped into tiles of at most
    `width` functions (one shell at least), cut from the basis alone."""
    tiles, s0 = [], 0
    nshl = len(ao_loc) - 1
    while s0 < nshl:
        s1 = s0 + 1
        while s1 < nshl and ao_loc[s1 + 1] - ao_loc[s0] <= width:
            s1 += 1
        tiles.append((s0, s1, int(ao_loc[s0]), int(ao_loc[s1])))
        s0 = s1
    return tiles


def _ri_factors(mol, auxmol, dm, comm):
    """(low, uw, u): rank 0's Cholesky factor of the auxiliary metric and
    eigen-factor of the density, dm = uw u^T, on every rank."""
    rank0 = comm is None or comm.Get_rank() == 0
    if rank0:
        low = scipy.linalg.cholesky(auxmol.intor('int2c2e'), lower=True)
        w, u = np.linalg.eigh(np.asarray(dm, float))
        keep = abs(w) > DENSITY_FACTOR_TOL
        w, u = w[keep], np.ascontiguousarray(u[:, keep])
        nf = len(w)
    nf = broadcast(nf if rank0 else None, comm)
    if not rank0:
        naux, nao = auxmol.nao_nr(), mol.nao_nr()
        low, w, u = np.zeros((naux, naux)), np.zeros(nf), np.zeros((nao, nf))
    if comm is not None:
        low, w, u = lockstep((low, w, u), comm)
    return low, np.ascontiguousarray(u * w), u


def _soc_jk_ri(mol, dm, auxbasis, comm, numerics=None):
    """(3, nao, nao) J - (3/2) K of the two-electron SOC operator by the
    resolution of the identity in `auxbasis` (module docstring), on every
    rank; collective over `comm`, serial where it is None. Its tiles hold
    `numerics.layout.tile_gb` GB each (None: SOMF_RI_TILE_GB)."""
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    size = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    low, uw, u = _ri_factors(mol, auxmol, dm, comm)
    nf = u.shape[1]
    dm = np.array(dm, float, order='C')
    if comm is not None:
        dm = lockstep(dm, comm)
    budget = None if numerics is None else numerics.layout.tile_gb
    words = (SOMF_RI_TILE_GB if budget is None else budget) * 1e9 / 8
    ao_loc, aux_loc = mol.ao_loc_nr(), auxmol.ao_loc_nr()
    cols = _shell_tiles(ao_loc, max(1, int(words / (nao * naux))))
    rows = _shell_tiles(aux_loc, max(1, int(words / (nf * nao))))
    widest = max(p1 - p0 for _, _, p0, p1 in rows)
    rblock = _shell_tiles(ao_loc, max(1, int(words / (3 * nao * widest))))
    # aux tiles in contiguous runs per rank, so the chain sum below crosses
    # ranks size - 1 times; column tiles round robin, one a rank per round
    t0, t1 = contiguous_block(len(rows), rank, size)
    mine = list(range(t0, t1))
    held = {t: np.zeros((nf, rows[t][3] - rows[t][2], nao)) for t in mine}
    take = [[p for t in range(*contiguous_block(len(rows), s, size))
             for p in range(rows[t][2], rows[t][3])] for s in range(size)]
    rhs = []
    for r0 in range(0, len(cols), size):
        c = r0 + rank
        if c < len(cols):
            s0, s1, l0, l1 = cols[c]
            plain = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1',
                                     shls_slice=(0, mol.nbas, s0, s1, 0,
                                                 auxmol.nbas))
            rhs.append((c, np.einsum('klQ,kl->Q', plain, dm[:, l0:l1])))
            fit = np.einsum('klQ,km->Qml', plain, u, optimize=True)
            del plain
            fit = scipy.linalg.cho_solve((low, True), fit.reshape(naux, -1))
            fit = fit.reshape(naux, nf, l1 - l0)
            send = [np.ascontiguousarray(fit[take[s]]) for s in range(size)]
        else:
            send = [np.zeros((len(take[s]), nf, 0)) for s in range(size)]
        width = [cols[r0 + s][3] - cols[r0 + s][2]
                 if r0 + s < len(cols) else 0 for s in range(size)]
        got = exchange_blocks(send, [(len(take[rank]), nf, width[s])
                                     for s in range(size)], comm)
        for s, piece in enumerate(got):
            if width[s]:
                l0, l1 = cols[r0 + s][2:]
                at = 0
                for t in mine:
                    n = rows[t][3] - rows[t][2]
                    held[t][:, :, l0:l1] = piece[at:at + n].transpose(1, 0, 2)
                    at += n
    coef = scipy.linalg.cho_solve(
        (low, True), ordered_sum(rhs, comm, shape=(naux,)))
    terms = {}
    for t in mine:
        a0, a1, p0, p1 = rows[t]
        vj = np.zeros((3, nao, nao))
        vk = np.zeros((3, nao, nao))
        fit = held.pop(t).reshape(nf * (p1 - p0), nao)
        for s0, s1, i0, i1 in rblock:
            soc = df.incore.aux_e2(mol, auxmol, 'int3c2e_pvxp1', aosym='s1',
                                   comp=3, shls_slice=(s0, s1, 0, mol.nbas,
                                                       a0, a1))
            vj[:, i0:i1] = soc @ coef[p0:p1]
            half = np.einsum('xijP,jm->ximP', soc, uw, optimize=True)
            del soc
            vk[:, i0:i1] = (half.reshape(3 * (i1 - i0), -1) @ fit
                            ).reshape(3, i1 - i0, nao)
        terms[t] = vj - 1.5 * (vk - vk.transpose(0, 2, 1))
    owners = [s for s in range(size)
              for _ in range(*contiguous_block(len(rows), s, size))]
    return ordered_chain_sum(terms, owners, comm, np.zeros((3, nao, nao)))


def jkfit_auxbasis(mol):
    """{atom label: name}: the JK-fitting basis of each atom's orbital basis,
    pySCF's predefined set (aug-cc-pVDZ-jkfit for aug-cc-pVDZ), what
    'ri-somf' fits in where no basis is named. An orbital basis without one
    is refused, not fitted in pySCF's even-tempered fallback: the fit's error
    is shown for the predefined sets."""
    out = df.addons.make_auxbasis(mol, xc='HF')
    labels = {a[0] for a in mol._atom}
    missing = sorted(k for k in labels if not isinstance(out.get(k), str))
    if missing:
        raise ValueError(
            f"two_electron='ri-somf': the orbital basis of {missing} has no "
            "predefined JK-fitting basis; name one in soc_auxbasis "
            "(Numerics.properties) or declare soc_two_electron='somf'")
    return {k: out[k] for k in sorted(labels)}


def mean_field_term(mol, dm, two_electron='somf', comm=None, auxbasis=None):
    """(3, nao, nao) J - (3/2) K of the two-electron spin-orbit operator at the
    AO density `dm`, without the alpha^2/4 prefactor.

    two_electron: 'somf' (exact two-electron integrals), 'ri-somf' (the same
        operator with the density side fitted in `auxbasis`) or 'amfi'
        (one-centre blocks only).
    auxbasis: the fitting basis of 'ri-somf', by name; None fits in the
        JK-fitting basis of the orbital basis (`jkfit_auxbasis`). Refused
        with any other term.
    comm: defaults to `current_comm()`. With more than one rank, 'somf' and
        'amfi' are built by rank 0 from ITS `dm` and every rank returns rank
        0's array; 'ri-somf' divides its tiles over the ranks, on rank 0's
        `dm`, and returns the same bits at every rank count. The `dm` of the
        other ranks is not read. Collective.
    """
    if two_electron not in TWO_ELECTRON_TERMS:
        raise ValueError(f"two_electron={two_electron!r}: one of "
                         f"{TWO_ELECTRON_TERMS}")
    if auxbasis is not None and two_electron != 'ri-somf':
        raise ValueError(
            f"two_electron={two_electron!r} with auxbasis={auxbasis!r}: the "
            "fitting basis is named for 'ri-somf' alone")
    if two_electron == 'ri-somf' and auxbasis is None:
        auxbasis = jkfit_auxbasis(mol)
    comm = current_comm() if comm is None else comm
    if two_electron == 'ri-somf':
        return _soc_jk_ri(mol, dm, auxbasis, comm)
    if comm is not None and comm.Get_size() > 1 and comm.Get_rank() != 0:
        nao = mol.nao_nr()
        return lockstep(np.zeros((3, nao, nao)), comm)
    jk = _soc_jk_amfi if two_electron == 'amfi' else _soc_jk
    vj, vk = jk(mol, dm)
    return lockstep(np.ascontiguousarray(vj - 1.5 * vk), comm)
