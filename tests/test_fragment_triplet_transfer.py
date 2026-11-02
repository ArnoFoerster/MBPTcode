"""The triplet channel of the fragment partition and the triplet-triplet
(Dexter) coupling between two site diabats, against exact references.

The offset stacked ethylene dimer of tests/test_fragment_diabatic.py (4 A
apart, slipped), cc-pVDZ, RHF, BSE@G0W0 on the ISDF/space-time chain, TRIPLET
(the kernel without its exchange term), Tamm-Dancoff and full; one site
state per monomer. The two monomers are images under a translation, so the
two site diabats are degenerate and the coupling is half the splitting of the
two lowest triplets. The ADC(2) part repeats the coupling on the folded
ADC(2) triplet operator (the same partition and the same coupling code).

WHAT EACH GATE IS FOR:

- the triplet partition is exact: Sigma and dSigma/dOmega equal the dense
  Feshbach oracle, the roots of A_eff(Omega) c = Omega c with complete Q are
  the dimer's triplet BSE roots of the same chain (its own solve), and Z is
  the exact P weight; in both kernels. The full-BSE triplet is where a
  triplet instability would show: the pole guard on Q and the positive
  roots say it is absent here.
- the matrix-free ISDF triplet action is the dense matrix, and the iterative
  path gives the dense partition.
- the ROOT gradient through the triplet partition equals the supermolecular
  triplet root gradient, with the orbital-rotation terms vanishing for it.
- each triplet ELEMENT's gradient, both charge-transfer diabats explicit,
  equals the relocalized finite difference.
- FOLDED OR EXPLICIT: the coupling with the charge-transfer diabats explicit
  and folded at E = Omega_0 is the coupling of the partition that leaves
  them in Q, to the last bits (the quotient property of Schur complements);
  without the folded block's own A_ff the identity fails by the
  superexchange, which is most of the coupling here.
- the folded coupling's gradient, one weighted reverse pass, equals the
  finite difference of the coupling folded at every displaced geometry.
- the same coupling code on ADC(2) triplets: the fold identity.
- refusals: a singlet partition, a charge-transfer diabat as donor, a
  transfer energy at or above a folded diabat.
- TRIPLET-TRIPLET ANNIHILATION through the same partitions: the model
  (`fragments.tta_coupling`) takes a singlet and a triplet partition of either
  method on one set of local orbitals; its direct element is the
  configuration elements' (tests/test_tta_coupling.py checks those against
  full CI), its mediated part the fold through the singlet A_eff's
  charge-transfer block; it refuses the full BSE, partitions on different
  orbitals and an energy above a charge-transfer level.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto

from src.Base.geometry.geometry import displaced
from src.Base.numerics.numerics import numerics_of
from src.fragments import partition as fragment_partition
from src.fragments.localization import FragmentOrbitals
from src.fragments.partition import FragmentPartition, to_local
from src.fragments.transfer import dexter_coupling, fold_diabats, site_coupling
from src.fragments.tta_coupling import tta_coupling, triplet_pair_elements
from src.gradients.adc.fragment_adc2 import ADC2FragmentMethod, ADC2Operator
from src.gradients.fragment_diabatic import DiabaticGradient
from src.gradients.gwbse.bse_isdf import bse_solve
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.fragment_bse import BSEOperator
from src.gradients.gwbse.fragment_diabatic import BSEFragmentMethod
from tests.feshbach_oracle import FeshbachOracle
from tests.fragment_diabatic_fd import diabatic_matrix
from tests.isdf_test_grid import g1
from tests.test_fragment_diabatic import (AUX, BASIS, BOTH_CT, DIMER,
                                         FRAGMENTS, SITES, factory,
                                         force_iterative, random_direction)

#: Omega_0 below the lowest triplet of both kernels. Dimer triplets (Ha):
#: Tamm-Dancoff 0.169936, 0.171539, 0.303710; full 0.159373, 0.160875,
#: 0.303185. Q's lowest eigenvalue with one site state per monomer is 0.3037
#: (Tamm-Dancoff) and 0.3032 (full), so either root can be Omega_0 itself.
OMEGA0 = 0.15
#: The same for ADC(2) triplets (roots 0.167330, 0.168960; Q 0.320450).
OMEGA0_ADC2 = 0.16
SITE_D, SITE_A = 'site 0.0', 'site 1.0'


def triplet_chain(tda):
    mol = gto.M(atom=DIMER, basis=BASIS, verbose=0)
    return ExcitedStateChain(mol, factory, spin='triplet', bse_tda=tda,
                             numerics=numerics_of({'auxbasis': AUX,
                                                   'counts': g1(mol)}))


class Kernel:
    """One kernel's triplet chain, dense operator and partitions."""

    def __init__(self, tda):
        self.tda = tda
        self.chain = triplet_chain(tda)
        self.op = BSEOperator.from_chain(self.chain, route='dense')[0]
        self.orbitals = FragmentOrbitals.from_mf(self.chain.mf0, FRAGMENTS)
        self.sites = FragmentPartition.build(self.op, self.orbitals, SITES,
                                             omega0=OMEGA0)
        self.with_ct = FragmentPartition.build(self.op, self.orbitals, SITES,
                                               BOTH_CT, omega0=OMEGA0)

    def roots(self):
        """(Omega, the local-basis stacked eigenvectors) of the dense dimer."""
        n = self.op.n_ov
        if self.tda:
            w, x = np.linalg.eigh(self.op.dense)
            return w, to_local(self.orbitals, x)
        w, x, y = bse_solve(self.op.dense[:n, :n], self.op.dense[:n, n:])
        return w, to_local(self.orbitals, np.vstack([x, y]))


KERNELS = {}


@pytest.fixture(scope='module', params=[True, False], ids=['tda', 'full'])
def kernel(request):
    if request.param not in KERNELS:
        KERNELS[request.param] = Kernel(request.param)
    return KERNELS[request.param]


def local_dense(kernel):
    t = kernel.orbitals.transition_rotation()
    if kernel.tda:
        return t.T @ kernel.op.dense @ t
    zero = np.zeros_like(t)
    tt = np.block([[t, zero], [zero, t]])
    return tt.T @ kernel.op.dense @ tt


def test_triplet_partition_is_the_exact_feshbach_elimination(kernel):
    op = kernel.op
    assert op.spin == 'triplet'
    for part in (kernel.sites, kernel.with_ct):
        oracle = FeshbachOracle(local_dense(kernel), part.p_local, op.metric)
        assert np.abs(oracle.sigma(part.omega0) - part.sigma).max() < 1e-12
        assert np.abs(oracle.dsigma(part.omega0) - part.dsigma).max() < 1e-12
    w, v = kernel.roots()
    # the chain's own triplet solve of the dimer, root 0
    _, info = kernel.chain.excitation_gradient()
    assert abs(info['omega'] - w[0]) < 1e-10
    part = kernel.sites
    oracle = FeshbachOracle(local_dense(kernel), part.p_local, op.metric)
    found = []
    for guess in np.linalg.eigvalsh(part.a_eff):
        om, _, z = oracle.root(guess)
        k = int(np.argmin(np.abs(w - om)))
        assert abs(om - w[k]) < 1e-10
        assert abs(z - np.sum((part.p_local.T @ (op.metric * v[:, k])) ** 2)) \
            < 1e-10
        found.append(k)
    # the two site diabats are the two lowest triplets of the dimer
    assert sorted(found) == [0, 1]
    assert part.q_lowest > w[1] + 0.1


def test_triplet_isdf_action_and_iterative_path_match_dense(kernel,
                                                            monkeypatch):
    isdf = BSEOperator.from_chain(kernel.chain, route='isdf')[0]
    x = np.random.default_rng(5).normal(size=(kernel.op.dim, 3))
    assert np.abs(isdf.apply(x) - kernel.op.dense @ x).max() < 1e-10
    force_iterative(monkeypatch)
    iterative = FragmentPartition.build(isdf, kernel.orbitals, SITES, BOTH_CT,
                                        omega0=OMEGA0)
    ref = kernel.with_ct.a_eff
    assert np.abs(np.diag(iterative.a_eff) - np.diag(ref)).max() < 1e-8
    assert np.abs(np.abs(iterative.a_eff) - np.abs(ref)).max() < 1e-8


def test_triplet_root_gradient_is_the_supermolecular_root_gradient(kernel):
    ref, info = kernel.chain.excitation_gradient()
    omega = info['omega']
    grad = DiabaticGradient(BSEFragmentMethod(kernel.chain), FRAGMENTS, SITES,
                            omega0=omega)
    g, d = grad.root_gradient(omega)
    assert np.abs(g - ref).max() < 1e-8 * max(1.0, np.abs(ref).max())
    assert np.abs(d['canonical']).max() < 1e-10
    assert np.abs(d['localization']).max() < 1e-10


def central_partitions(chain, direction, h, reference):
    method = BSEFragmentMethod(chain)
    return [diabatic_matrix(method, FRAGMENTS, SITES, BOTH_CT,
                            mol=displaced(chain.mol0, s * h * direction),
                            reference=reference) for s in (1.0, -1.0)]


def test_triplet_elements_and_folded_coupling_match_finite_differences(
        kernel):
    chain = kernel.chain
    direction = random_direction(chain.mol0.natm, 7)
    grad = DiabaticGradient(BSEFragmentMethod(chain), FRAGMENTS, SITES,
                            BOTH_CT, omega0=OMEGA0)
    ref = diabatic_matrix(BSEFragmentMethod(chain), FRAGMENTS, SITES,
                          BOTH_CT, omega0=OMEGA0)
    fd_a, fd_v = [], []
    for h in (1e-3, 5e-4):
        plus, minus = central_partitions(chain, direction, h, ref)
        fd_a.append((plus.a_eff - minus.a_eff) / (2 * h))
        fd_v.append((dexter_coupling(plus, SITE_D, SITE_A).coupling
                     - dexter_coupling(minus, SITE_D, SITE_A).coupling)
                    / (2 * h))
    fd_a = (4.0 * fd_a[1] - fd_a[0]) / 3.0
    fd_v = (4.0 * fd_v[1] - fd_v[0]) / 3.0
    scale = np.abs(fd_a).max()
    n = len(grad.partition.labels)
    for a in range(n):
        for b in range(a, n):
            an = float((grad.element(a, b)[0] * direction).sum())
            assert abs(an - fd_a[a, b]) < 2e-5 * scale, (a, b)
    v = dexter_coupling(grad.partition, SITE_D, SITE_A)
    an = float((grad.gradient(v.weights)[0] * direction).sum())
    print(f'dV/dlambda analytic {an:.10e} FD {fd_v:.10e} (Ha/Bohr)')
    assert abs(an - fd_v) < 1e-5 * abs(fd_v)
    # the fold's own derivative is not the element's: the weights matter
    an_element = float((grad.element(0, 1)[0] * direction).sum())
    assert abs(an_element - fd_v) > 0.3 * abs(fd_v)


def test_folded_and_explicit_charge_transfer_give_one_coupling(kernel):
    explicit = dexter_coupling(kernel.with_ct, SITE_D, SITE_A)
    in_q = dexter_coupling(kernel.sites, SITE_D, SITE_A)
    assert explicit.folded == ('ct 0->1.0', 'ct 1->0.0') and in_q.folded == ()
    assert abs(explicit.coupling - in_q.coupling) < 1e-12
    assert np.abs(np.array(explicit.site_energies)
                  - np.diag(kernel.sites.a_eff)).max() < 1e-12
    # the superexchange through the explicit diabats is most of it here
    assert abs(explicit.superexchange) > 0.5 * abs(explicit.coupling)
    # without the folded block's own A_ff the identity fails by about the
    # superexchange
    a = kernel.with_ct.a_eff.copy()
    a[2:, 2:] = 0.0
    wrong = fold_diabats(a, (0, 1), OMEGA0, margin=-1.0)[0][0, 1]
    assert abs(wrong - in_q.coupling) > 0.3 * abs(in_q.coupling)
    # a symmetric dimer: the coupling is half the splitting of the two-state
    # matrix built self-consistently at each of the two lowest roots
    w = kernel.roots()[0]
    for k in (0, 1):
        part = FragmentPartition.build(kernel.op, kernel.orbitals, SITES,
                                       BOTH_CT, omega0=w[k])
        c = dexter_coupling(part, SITE_D, SITE_A)
        h = np.array([[c.site_energies[0], c.coupling],
                      [c.coupling, c.site_energies[1]]])
        assert np.abs(np.linalg.eigvalsh(h) - w[k]).min() < 1e-10


def test_the_coupling_refuses_what_it_cannot_take(kernel):
    with pytest.raises(ValueError, match='not a site diabat'):
        dexter_coupling(kernel.with_ct, SITE_D, 'ct 0->1.0')
    with pytest.raises(ValueError, match='real intermediate'):
        dexter_coupling(kernel.with_ct, SITE_D, SITE_A,
                        energy=float(kernel.with_ct.a_eff[2, 2]))
    with pytest.raises(ValueError, match='same diabat'):
        dexter_coupling(kernel.with_ct, SITE_D, SITE_D)
    singlet = ExcitedStateChain(kernel.chain.mol0, factory, bse_tda=True,
                                numerics=numerics_of(
                                    {'auxbasis': AUX,
                                     'counts': g1(kernel.chain.mol0)}))
    op = BSEOperator.from_chain(singlet, route='dense')[0]
    part = FragmentPartition.build(op, kernel.orbitals, SITES, omega0=0.30)
    with pytest.raises(ValueError, match="'singlet' channel"):
        dexter_coupling(part, SITE_D, SITE_A)
    # the same partition's singlet coupling is a site coupling
    assert site_coupling(part, SITE_D, SITE_A).spin == 'singlet'


def test_adc2_triplets_through_the_same_coupling():
    mol = gto.M(atom=DIMER, basis=BASIS, verbose=0)
    method = ADC2FragmentMethod(mol, factory, 'triplet', AUX)
    mol, mf = method.mean_field()
    orbitals = FragmentOrbitals.from_mf(mf, FRAGMENTS)
    op = ADC2Operator.from_mean_field(mol, mf, 'triplet', AUX, OMEGA0_ADC2,
                                      route='dense')
    assert op.spin == 'triplet'
    sites = FragmentPartition.build(op, orbitals, SITES)
    with_ct = FragmentPartition.build(op, orbitals, SITES, BOTH_CT)
    explicit = dexter_coupling(with_ct, SITE_D, SITE_A)
    in_q = dexter_coupling(sites, SITE_D, SITE_A)
    assert abs(explicit.coupling - in_q.coupling) < 1e-12
    assert explicit.spin == 'triplet' and explicit.omega0 == OMEGA0_ADC2


#: The singlet partition of the annihilation test: Omega_0 below Q's lowest
#: eigenvalue with one site state per monomer and both CT diabats explicit.
OMEGA0_SINGLET = {'bse': 0.30, 'adc2': 0.29}
JK = 'cc-pvdz-jkfit'


def tta_partitions(name):
    """(singlet, triplet partition, mean field) of one method on one set of
    local orbitals."""
    mol = gto.M(atom=DIMER, basis=BASIS, verbose=0)
    if name == 'bse':
        trip = triplet_chain(True)
        op_t = BSEOperator.from_chain(trip, route='dense')[0]
        # the singlet on the same frozen objects (after the first forward)
        op_s = BSEOperator.from_chain(trip.spin_view('singlet'),
                                      route='dense')[0]
        mf = trip.mf0
        ops = [op_s, op_t]
    else:
        method = ADC2FragmentMethod(mol, factory, 'triplet', AUX)
        mol, mf = method.mean_field()
        ops = [ADC2Operator.from_mean_field(mol, mf, s, AUX, om, route='dense')
               for s, om in (('singlet', OMEGA0_SINGLET['adc2']),
                             ('triplet', OMEGA0_ADC2))]
    orbitals = FragmentOrbitals.from_mf(mf, FRAGMENTS)
    singlet = FragmentPartition.build(ops[0], orbitals, SITES, BOTH_CT,
                                      omega0=OMEGA0_SINGLET[name])
    triplet = FragmentPartition.build(ops[1], orbitals, SITES,
                                      omega0=OMEGA0 if name == 'bse'
                                      else OMEGA0_ADC2)
    return singlet, triplet, mf


@pytest.mark.parametrize('name', ['bse', 'adc2'])
def test_triplet_triplet_annihilation_takes_either_method(name):
    singlet, triplet, mf = tta_partitions(name)
    out = tta_coupling(singlet, triplet, mf, JK)
    o = singlet.orbitals
    amp = lambda part, lab: part.p_local[:, part.labels.index(lab)].reshape(
        o.nocc, o.nvir)
    labels = ['site 0.0', 'site 1.0', 'ct 0->1.0', 'ct 1->0.0']
    el = triplet_pair_elements(o.mol, mf.get_hcore(), o.c_occ, o.c_vir, JK,
                               amp(triplet, 'site 0.0'),
                               amp(triplet, 'site 1.0'),
                               [amp(singlet, lab) for lab in labels])
    a = singlet.a_eff
    for k, lab in enumerate(labels[:2]):
        c = out[lab]
        assert abs(c.v_direct - el['tt'][k]) < 1e-14
        assert np.abs(c.v_tt_ct - el['tt'][2:]).max() < 1e-14
        assert abs(c.e_tt - np.trace(triplet.a_eff)) < 1e-14
        assert abs(c.energy - 0.5 * (a[k, k] + c.e_tt)) < 1e-14
        g = np.linalg.inv(c.energy * np.eye(2) - a[2:, 2:])
        assert abs(c.mediated - c.v_tt_ct @ g @ a[2:, k]) < 1e-14
        print(f'{name} {lab}: E_S1 {a[k, k]:.6f} E_TT {c.e_tt:.6f} E_CT '
              f'{np.round(c.e_ct, 6)} V_direct {c.v_direct:.3e} V_TT,CT '
              f'{np.round(c.v_tt_ct, 7)} V_CT,S {np.round(c.v_ct_s, 6)} '
              f'V_eff {c.v_eff:.3e} Ha')
    with pytest.raises(ValueError, match='real intermediate'):
        tta_coupling(singlet, triplet, mf, JK, energy=float(a[2, 2]))
    with pytest.raises(ValueError, match='singlet partition and a triplet'):
        tta_coupling(triplet, singlet, mf, JK)


def test_annihilation_refuses_full_bse_and_foreign_orbitals():
    kernel = KERNELS.get(False) or Kernel(False)
    singlet, triplet, mf = tta_partitions('bse')
    full = FragmentPartition.build(kernel.op, triplet.orbitals, SITES,
                                   omega0=OMEGA0)
    with pytest.raises(ValueError, match='Tamm-Dancoff'):
        tta_coupling(singlet, full, mf, JK)
    other = FragmentOrbitals.from_mf(mf, FRAGMENTS)
    other.u_occ = -other.u_occ
    moved = FragmentPartition.build(triplet.operator, other, SITES,
                                    omega0=OMEGA0)
    with pytest.raises(ValueError, match='share their local orbitals'):
        tta_coupling(singlet, moved, mf, JK)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
