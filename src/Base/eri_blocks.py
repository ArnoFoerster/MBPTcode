"""The MO integral blocks the quasi-boson GW/BSE energy actually needs.

The dense quasi-boson route was written against a full (pq|rs), which is nao^4
and becomes the binding constraint long before the physics does: formaldehyde in
aug-cc-pVTZ is 2.9 GB, acrolein 46 GB. But the energy path only ever touches
three blocks --

    ovov   (ia|jb)     the RPA/BSE particle-hole kernel
    oovv   (ij|ab)     the BSE direct term
    pqov   (pq|ia)     the bare boson couplings V^I_pq

-- and every other slice it takes is one of these under the eight-fold
symmetry. `pqov` dominates at norb^2 * nocc * nvirt, which is smaller than
nao^4 by nao^2 / (nocc * nvirt): a factor of 18 for formaldehyde and 19 for
acrolein, taking that 46 GB to 2.5 GB.

`from_dense` slices a full array and reproduces the dense route's numbers
bitwise; `from_mol` builds only the blocks and never forms nao^4 at all. The
two are not bitwise each other: they are different `ao2mo.general` calls, and
on water/cc-pVDZ the (ij|ab) block differs by 4.6e-15 while H4/sto-3g agrees
exactly, so a gate taken on the smallest molecule would not see it.

GRADIENTS ARE NOT AVAILABLE from blocks: the correlation gradient contracts a
four-index density against the full tensor. `require_dense` is the refusal, and
the callers use it rather than silently returning something else.

`mo_eri` builds that full tensor from the BARE interaction and consults no
environment, which is what tells it apart from
`pyscf_interface.get_two_electron_integrals_chemist`: the two agree bitwise in
the gas phase and differ by the reaction field v -> v + vtilde wherever an
environment is attached. A route reporting E_c^dRPA[v] uses this one and
refuses a solvated mean field rather than quietly dressing the interaction.

The density-fitted interaction, (pq|rs) = sum_PQ J_pq,P [V^-1]_PQ J_rs,Q, is
here as well: a route that differentiates a FITTED energy has to evaluate that
same fitted energy, so the forward assembly and the three-index gradient
skeleton (`gradients.df_assembly`) read one pair of integral builders.
"""
import numpy as np
from pyscf import ao2mo
from pyscf import df as pyscf_df


class MOEriBlocks:
    """(ia|jb), (ij|ab) and (pq|ia) in the MO basis, and nothing else.

    Indexing convention throughout: `ovov[i, a, j, b] = (ia|jb)`,
    `oovv[i, j, a, b] = (ij|ab)`, `pqov[p, q, i, a] = (pq|ia)`, with a and b
    counted from the first virtual, so `pqov`'s trailing pair flattens to the
    same row-major I = i * nvirt + a the rest of the route uses.
    """

    def __init__(self, ovov, oovv, pqov, nocc, dense=None):
        self.ovov = ovov
        self.oovv = oovv
        self.pqov = pqov
        self.nocc = nocc
        self.norb = pqov.shape[0]
        self.nvirt = self.norb - nocc
        self.n_ov = nocc * self.nvirt
        # The full tensor when there is one -- the gradient path needs it, and
        # keeping it here is what lets one object serve both routes.
        self.dense = dense
        expected = {'ovov': (nocc, self.nvirt, nocc, self.nvirt),
                    'oovv': (nocc, nocc, self.nvirt, self.nvirt),
                    'pqov': (self.norb, self.norb, nocc, self.nvirt)}
        for name, shape in expected.items():
            got = getattr(self, name).shape
            if got != shape:
                raise ValueError(f'{name} has shape {got}, expected {shape} '
                                 f'for norb={self.norb}, nocc={nocc}')

    @classmethod
    def from_dense(cls, eri_mo, nocc):
        """Slice a full (pq|rs). Bitwise what the route used to index itself."""
        eri_mo = np.asarray(eri_mo)
        norb = eri_mo.shape[0]
        occ, virt = slice(0, nocc), slice(nocc, norb)
        return cls(eri_mo[occ, virt, occ, virt], eri_mo[occ, occ, virt, virt],
                   eri_mo[:, :, occ, virt], nocc, dense=eri_mo)

    @classmethod
    def from_mol(cls, mol, mo_coeff, nocc, auxmol=None):
        """Build the three blocks directly, never forming nao^4.

        auxmol: fit the integrals in that auxiliary basis instead of using the
        exact four-centre ones -- the same RI the cubic route fits in, so the
        two stay comparable.
        """
        C = np.asarray(mo_coeff)
        Co, Cv = C[:, :nocc], C[:, nocc:]
        if auxmol is None:
            def block(mos):
                shape = tuple(m.shape[1] for m in mos)
                return ao2mo.general(mol, mos, compact=False).reshape(shape)
            return cls(block((Co, Cv, Co, Cv)), block((Co, Co, Cv, Cv)),
                       block((C, C, Co, Cv)), nocc)

        # RI: (pq|rs) = sum_P J_mo[p,q,P] K_mo[r,s,P] with K = V^-1 J, exactly
        # as df_eri_mo forms it -- only contracted per block.
        J, V = df_integrals(mol, auxmol)
        J_mo = np.einsum('mp,nq,mnP->pqP', C, C, J, optimize=True)
        K_mo = np.linalg.solve(V, J_mo.reshape(-1, V.shape[0]).T).T.reshape(
            J_mo.shape)
        occ, virt = slice(0, nocc), slice(nocc, C.shape[1])
        return cls(np.einsum('iaP,jbP->iajb', J_mo[occ, virt], K_mo[occ, virt],
                             optimize=True),
                   np.einsum('ijP,abP->ijab', J_mo[occ, occ], K_mo[virt, virt],
                             optimize=True),
                   np.einsum('pqP,iaP->pqia', J_mo, K_mo[occ, virt],
                             optimize=True),
                   nocc)

    @property
    def vovo(self):
        """(aj|bi) indexed [a, j, b, i]: `ovov` under both intra-pair swaps."""
        return self.ovov.transpose(1, 0, 3, 2)

    def require_dense(self, what):
        """The full tensor, or a refusal naming what asked for it."""
        if self.dense is None:
            raise NotImplementedError(
                f'{what} needs the full (pq|rs), which this integral set does '
                f'not hold: it was built as blocks precisely to avoid the '
                f'nao^4 array. Energies are fine; for gradients build the '
                f'integrals densely.')
        return self.dense


def as_blocks(eri_mo, nocc):
    """Accept either a full array or an already-built block set."""
    if isinstance(eri_mo, MOEriBlocks):
        if eri_mo.nocc != nocc:
            raise ValueError(f'blocks were built for nocc={eri_mo.nocc}, '
                             f'asked for {nocc}')
        return eri_mo
    return MOEriBlocks.from_dense(eri_mo, nocc)


def mo_eri(mf, mol):
    """Full (pq|rs) in the MO basis: nao^4, which is what bounds this route.

    The BARE interaction: no environment is consulted, so a route that builds
    E_c^dRPA from this reports the correlation energy of v alone.
    """
    n = mol.nao
    return ao2mo.general(mol, (mf.mo_coeff,) * 4,
                         compact=False).reshape((n,) * 4)


def df_integrals(mol, auxmol):
    """(J, V): three-centre (nao, nao, naux) and two-centre (naux, naux)."""
    J = pyscf_df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1')
    return J.reshape(mol.nao, mol.nao, auxmol.nao), auxmol.intor('int2c2e')


def df_eri_mo(mol, auxmol, C):
    """The RI four-index ERI in the MO basis -- the energy a fitted route differentiates.

    Differentiating a fitted energy with exact derivative integrals is not a
    gradient of anything; the ERIs have to be the fitted ones for the
    factorized assembly to be exact rather than approximate.
    """
    J, V = df_integrals(mol, auxmol)
    J_mo = np.einsum('mp,nq,mnP->pqP', C, C, J, optimize=True)
    K_mo = np.linalg.solve(V, J_mo.reshape(-1, V.shape[0]).T).T.reshape(J_mo.shape)
    return np.einsum('pqP,rsP->pqrs', J_mo, K_mo, optimize=True)
