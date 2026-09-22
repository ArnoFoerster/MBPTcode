"""Moving a molecule across the QM/MM partition must not move the gap.

The central claim of Li, D'Avino, Duchemin, Beljonne and Blase, J. Phys. Chem.
Lett. 7, 2814 (2016), Fig. 2(d): with 1, 3 and 5 pentacenes in the QM region
the gas-phase G0W0 gap falls 4.90 -> 4.65 -> 4.58 eV while the classical
polarization energy falls by the same 0.32 eV, and the embedded gap is 2.77 /
2.78 / 2.77. "A QM/MM scheme that fully accounts for both QM and MM
polarization effects permits one to vary the size of the QM subsystem without
any loss in the polarization energy." The mechanism is the exact folding of W:
a molecule moved from MM to QM moves its polarizability out of the classical
chi*_22 that builds v_reac and into the quantum chi0_11 that builds the gap.

THE PARTNER HAS TO BE A SPECTATOR for that statement to be about the partition
rather than about the partner. Argon at 10 Bohr from water's oxygen is one: it
carries no permanent multipole, so no crystal field reaches the QM molecule,
and its levels are nowhere near water's, so the frontier orbitals stay water's
own (Mulliken weight on the partner 4e-09 and 4e-06). A second water is NOT:
the inversion-symmetric dimer's frontier orbitals are 50/50 combinations of
the two monomers' by symmetry, and a hole spread over both polarizes them at a
quarter of the energy each. That counter-case is gated below, because it is
what says which property of the partition the compensation depends on.

THE CALIBRATION LEVEL IS THE WHOLE RESULT. The site carries the partner's
DIRECT-RPA polarizability, the response W is built from; with the CPHF value a
finite field gives, 1.3923 times larger here, the classical side over-screens
by exactly that factor and the compensation fails at 39 % of the swing.

BASIS-SET SUPERPOSITION SETS A LONGER RANGE THAN THE CLEARANCE GUARD. The two
partitions do not share a basis -- the QM one has the partner's functions and
the MM one does not -- so the comparison only means something where that
difference is small. A ghost partner measures it: 1.3 % of the swing at 10
Bohr, but 98 % of it at 7 Bohr, where the site-to-QM clearance the model
itself needs is still satisfied.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import (BOHR_TO_ANGSTROM, HARTREE_TO_EV,
                                MIN_SITE_TO_QM_DISTANCE)
from src.Base.environment import attached_environment
from src.Base.polarizable_sites import (PolarizableSites,
                                        calibrated_site_alphas,
                                        finite_field_polarizability,
                                        sites_for_molecule)
from src.SingleReference.GW.qp_energy import calc_qp_energy

BASIS = 'cc-pvdz'
Z_OXYGEN = 0.117
WATER = f'O 0 0 {Z_OXYGEN}; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: Bohr from the oxygen to the partner atom. FAR is where the two partitions
#: share a basis to 1.3 % of the effect; NEAR is admissible to the clearance
#: guard and useless all the same, which is the point of measuring both.
NEAR, FAR = 7.0, 10.0
#: Bohr between the oxygens of the counter-case dimer.
DIMER_SEPARATION = 9.0
#: the papers' compensation is 0.01 eV on a 0.32 eV swing; anything under this
#: fraction of the swing is a gap that did not notice the partition moving
COMPENSATION = 0.2


def rhf(mol):
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    return mf


def gw_levels(mf, nocc, environment=None, extra=0):
    """Quasiparticle energies in eV, from HOMO - extra to LUMO + extra."""
    states = list(range(nocc - 1 - extra, nocc + 1 + extra))
    with attached_environment(mf, environment):
        out = calc_qp_energy(mf, mode='casida', state=states)
    return np.array([out[p]['GW'] for p in states])


def mean_field_gap(mf, nocc):
    return (mf.mo_energy[nocc] - mf.mo_energy[nocc - 1]) * HARTREE_TO_EV


def partner_weight(mol, mf, atoms, orbitals):
    """Mulliken population of each orbital on the partner's atoms."""
    idx = np.array([i for i, l in enumerate(mol.ao_labels())
                    if int(l.split()[0]) in atoms], dtype=int)
    s = mol.intor('int1e_ovlp')
    return np.array([float(mf.mo_coeff[idx, p] @ (s[idx] @ mf.mo_coeff[:, p]))
                     for p in orbitals])


@pytest.fixture(scope='module')
def solute():
    """Water on its own: the gas-phase reference both partitions are measured
    against, and the QM molecule of the MM partition."""
    mol = gto.M(atom=WATER, basis=BASIS, verbose=0)
    mf = rhf(mol)
    nocc = mol.nelectron // 2
    levels = gw_levels(mf, nocc)
    return mol, mf, nocc, levels[1] - levels[0], mean_field_gap(mf, nocc)


@pytest.fixture(scope='module')
def argon(solute):
    """Both partitions of water plus argon, at the two separations, in eV.

    Per separation: the QM partition's gap change, the MM partition's with the
    site calibrated at each of the two response levels, the basis-only change
    from a ghost partner, the mean-field gap change, and how much of the
    frontier orbitals sits on the partner.
    """
    mol_a, mf_a, nocc, gap, hf_gap = solute
    out = {}
    for r in (NEAR, FAR):
        z = Z_OXYGEN + r * BOHR_TO_ANGSTROM
        partner = gto.M(atom=f'Ar 0 0 {z}', basis=BASIS, verbose=0)
        together = gto.M(atom=f'{WATER}; Ar 0 0 {z}', basis=BASIS, verbose=0)
        ghost = gto.M(atom=f'{WATER}; ghost-Ar 0 0 {z}', basis=BASIS, verbose=0)
        mf_d, mf_g = rhf(together), rhf(ghost)
        nd, ng = together.nelectron // 2, ghost.nelectron // 2

        coords, alphas = sites_for_molecule(partner)
        cphf = float(np.trace(finite_field_polarizability(partner)) / 3.0)
        sites = PolarizableSites(coords, alphas, unit='Bohr', mol=mol_a)
        over = PolarizableSites(coords, calibrated_site_alphas(coords, cphf),
                                unit='Bohr', mol=mol_a)
        out[r] = {
            'alpha': float(alphas[0]), 'alpha_cphf': cphf,
            'clearance': float(np.linalg.norm(
                coords[:, None, :] - mol_a.atom_coords()[None, :, :],
                axis=2).min()),
            'qm': np.diff(gw_levels(mf_d, nd))[0] - gap,
            'mm': np.diff(gw_levels(mf_a, nocc, environment=sites))[0] - gap,
            'mm_cphf': np.diff(gw_levels(mf_a, nocc, environment=over))[0] - gap,
            'basis': np.diff(gw_levels(mf_g, ng))[0] - gap,
            'mean_field': mean_field_gap(mf_d, nd) - hf_gap,
            'weight': partner_weight(together, mf_d, {3}, (nd - 1, nd)),
        }
    return out


@pytest.fixture(scope='module')
def symmetric_dimer():
    """The counter-case: two waters related by inversion through the midpoint.

    The gap of the QM dimer in both conventions -- edge to edge, and the band
    centre to centre the papers use for a QM region of several molecules --
    against one monomer screened by the other as calibrated sites.
    """
    half = 0.5 * DIMER_SEPARATION * BOHR_TO_ANGSTROM
    first = [f'O {half} 0 {Z_OXYGEN}', f'H {half} 0.757 -0.468',
             f'H {half} -0.757 -0.468']
    second = [f'O {-half} 0 {-Z_OXYGEN}', f'H {-half} -0.757 0.468',
              f'H {-half} 0.757 0.468']
    mol_a = gto.M(atom='; '.join(first), basis=BASIS, verbose=0)
    mol_b = gto.M(atom='; '.join(second), basis=BASIS, verbose=0)
    mol_d = gto.M(atom='; '.join(first + second), basis=BASIS, verbose=0)
    mf_a, mf_d = rhf(mol_a), rhf(mol_d)
    na, nd = mol_a.nelectron // 2, mol_d.nelectron // 2
    coords, alphas = sites_for_molecule(mol_b)
    sites = PolarizableSites(coords, alphas, unit='Bohr', mol=mol_a)
    monomer = gw_levels(mf_a, na)
    gap = monomer[1] - monomer[0]
    dimer = gw_levels(mf_d, nd, extra=1)
    return {
        'p': np.diff(gw_levels(mf_a, na, environment=sites))[0] - gap,
        'edge': (dimer[2] - dimer[1]) - gap,
        'c2c': 0.5 * (dimer[2] + dimer[3]) - 0.5 * (dimer[0] + dimer[1]) - gap,
        'mean_field': mean_field_gap(mf_d, nd) - mean_field_gap(mf_a, na),
        'weight': partner_weight(mol_d, mf_d, {3, 4, 5},
                                 (nd - 2, nd - 1, nd, nd + 1)),
    }


def test_the_gap_does_not_notice_the_partition_moving(argon):
    """Argon at 10 Bohr, as one calibrated site or as part of the QM region.
    Water's quasiparticle gap closes by 5.327 meV when argon joins the QM
    region and by 5.334 meV when argon is a classical site instead: the swing
    is compensated to 0.007 meV, 0.12 % of it, where the papers' own figure is
    0.01 eV on 0.32, i.e. 3 %. The gap closes either way -- both a hole and an
    added electron are stabilised by the polarization they induce."""
    far = argon[FAR]
    assert far['qm'] < 0.0 and far['mm'] < 0.0
    assert abs(far['qm'] - far['mm']) < COMPENSATION * abs(far['qm'])
    assert abs(far['qm'] - far['mm']) * 1e3 < 0.05
    assert abs(far['qm']) * 1e3 == pytest.approx(5.327, abs=0.05)


def test_the_frontier_orbitals_stay_on_the_solute(argon):
    """What makes argon the partition's spectator rather than half of the
    system: its occupied levels lie 10 eV below water's HOMO and its virtuals
    far above water's LUMO, so the frontier orbitals carry 4e-09 and 4e-06 of a
    Mulliken population on it. A gap built from orbitals shared with the
    partner would be a different quantity in the two partitions and the
    comparison would mean nothing."""
    for r in (NEAR, FAR):
        assert argon[r]['weight'].max() < 1e-2
    assert argon[FAR]['weight'].max() < 1e-5


def test_the_mean_field_gap_does_not_compensate(argon):
    """The papers' own negative control: across 1/3/5 QM pentacenes the
    Kohn-Sham gap is flat at 2.54 eV while the G0W0 gap moves 0.32, so "the use
    of Kohn-Sham energies does not allow one to capture the polarization
    effects at the origin of the large gap closing". Here the mean-field gap
    moves +0.221 meV where the polarization energy is -5.334, and with the
    wrong sign: adding P to it would overshoot by more than the whole
    compensation the quasiparticle gap achieves."""
    far = argon[FAR]
    assert abs(far['mean_field']) < 0.1 * abs(far['mm'])
    assert abs(far['mean_field'] - far['mm']) > 0.8 * abs(far['mm'])


def test_the_site_must_carry_the_response_the_screening_is_built_from(argon):
    """W is built from the Hartree-only chi, so a site standing in for the
    partner inside a GW calculation has to carry the partner's direct-RPA
    polarizability: 2.8006 Bohr^3 for argon here. A finite field gives the CPHF
    response instead, 3.8993, and the classical polarization energy comes out
    larger by 1.3923 -- the ratio of the two polarizabilities to four figures,
    since the Eq. (18) shift is linear in vtilde and vtilde is linear in alpha.
    The residual is then 39 % of the swing where the calibrated site gives
    0.1 %, so the level of response is not a refinement of this test but the
    difference between passing and failing it."""
    far = argon[FAR]
    assert far['mm_cphf'] / far['mm'] == pytest.approx(
        far['alpha_cphf'] / far['alpha'], rel=1e-3)
    assert abs(far['qm'] - far['mm_cphf']) > COMPENSATION * abs(far['qm'])


def test_the_two_partitions_have_to_share_a_basis(argon):
    """The QM partition carries the partner's basis functions and the MM
    partition does not, so part of any gap change is variational freedom rather
    than polarization. A ghost partner -- its basis, no electrons and no
    nucleus -- measures that part: 0.071 meV at 10 Bohr, 1.3 % of the swing, but
    26.428 meV at 7 Bohr, 98 % of it. At 7 Bohr the CPHF-calibrated site agrees
    with the QM partition to 0.8 %, which is two errors of opposite sign and
    not a compensation at all.

    The clearance guard does not protect against this. 7 Bohr is outside
    MIN_SITE_TO_QM_DISTANCE, where the induced-dipole model is defined; the
    basis overlap that ruins the COMPARISON reaches further out than the
    orbital overlap that ruins the MODEL."""
    near, far = argon[NEAR], argon[FAR]
    assert near['clearance'] == pytest.approx(NEAR, abs=1e-6)
    assert near['clearance'] > MIN_SITE_TO_QM_DISTANCE
    assert abs(far['basis']) < 0.05 * abs(far['qm'])
    assert abs(near['basis']) > 0.9 * abs(near['qm'])
    assert abs(near['qm'] - near['mm_cphf']) < 0.02 * abs(near['qm'])


def test_a_symmetric_dimer_does_not_carry_the_statement(symmetric_dimer):
    """Two waters related by inversion put exactly half of every frontier
    orbital on each monomer, which breaks the comparison twice over.

    The edge-to-edge gap closes by 155.861 meV, of which the mean field
    supplies 156.607: that is the g/u splitting of two nearly degenerate
    monomer levels, level repulsion and not polarization at all. Taking the
    band centres instead, as the papers do for several QM molecules, removes it
    and leaves +10.383 meV -- the gap OPENS, because the non-uniform
    electrostatic field of the partner's dipole reaches the diffuse virtual
    more than the lone pair. Against either, the classical sites give -11.741
    meV and nothing compensates.

    A hole spread over two monomers polarizes each by a quarter of what a
    localized hole would, so half the polarization energy is gone before any of
    this. The statement needs the charge to sit where the papers put it."""
    d = symmetric_dimer
    assert np.abs(d['weight'] - 0.5).max() < 1e-3
    assert abs(d['edge'] - d['mean_field']) < 0.02 * abs(d['edge'])
    assert d['c2c'] > 0.0 > d['p']
    for swing in (d['edge'], d['c2c']):
        assert abs(swing - d['p']) > COMPENSATION * abs(swing)
