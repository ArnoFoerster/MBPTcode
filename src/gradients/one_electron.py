"""The one-electron and nuclear terms every nuclear force shares.

The mean-field force (`isdf_mean_field`) and every correlated Lagrangian
(`grad_engine`, `isdf_derivatives`, `derivative_coupling`) contract a density
and an energy-weighted density with dh/dR and dS/dR, external potentials
included; both halves read one gradient object.
"""
import numpy as np
from pyscf.grad import rhf as rhf_grad
from pyscf.qmmm import itrf as qmmm_itrf


def one_electron_gradient(mf):
    """pyscf's gradient object for the one-electron and nuclear terms of `mf`.

    Built from the class, not from `mf.Gradients()`: `attach_isdf_gradient`
    binds that to the object it was called on, so on a mean field wrapped
    afterwards it answers for the unwrapped one. Point charges from
    `qmmm.mm_charge` live in the wrapped gradient's `get_hcore` (their field
    on the electrons) and `grad_nuc` (their pull on the nuclei). Every
    one-electron skeleton reads it: the mean-field force here and the
    correlated Lagrangian (`one_electron_skeleton`).
    """
    grad = rhf_grad.Gradients(mf)
    if isinstance(mf, qmmm_itrf.QMMM):
        grad = qmmm_itrf.qmmm_grad_for_scf(grad)
    return grad


def one_electron_skeleton(mol, mf, gamma_ao_list, M_ao_list):
    """sum_ab gamma_ab dh_ab/dR_A + sum_ab M_ab dS_ab/dR_A for every target.

    dh carries every external potential of `mf`'s Hamiltonian, point charges
    included, through the gradient object its mean-field force reads
    (`one_electron_gradient`): `mf.Gradients()` of an ISDF
    mean field wrapped after its gradient was attached is the gas phase's.
    """
    hgen = one_electron_gradient(mf).hcore_generator(mol)
    s1 = -mol.intor('int1e_ipovlp', comp=3)
    aoslices = mol.aoslice_by_atom()
    out = np.zeros((len(gamma_ao_list), mol.natm, 3))
    for iatm in range(mol.natm):
        p0, p1 = aoslices[iatm, 2:]
        dh_all = hgen(iatm)
        for xyz in range(3):
            dS = np.zeros((mol.nao, mol.nao))
            dS[p0:p1, :] += s1[xyz, p0:p1, :]
            dS[:, p0:p1] += s1[xyz, p0:p1, :].T
            for k, (g_ao, M_ao) in enumerate(zip(gamma_ao_list, M_ao_list)):
                out[k, iatm, xyz] = (np.einsum('ab,ab->', g_ao, dh_all[xyz])
                                     + np.einsum('ab,ab->', M_ao, dS))
    return out
