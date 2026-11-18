"""The paper's SI (tests/data/downfolding_paper_si.json) as reference data.

The SI rows come from AMS BAND with spherical Gaussian basis sets; here the
same molecules run on pyscf (spherical, exact 4-centre integrals, df=False),
the casida RPA and pyscf FCI in the active space. Diatomics lie along z at
the CSV's r (the grid has 0.5, 0.75, 1.5, 3.0 and 1.1, 1.6, 2.2 exactly).
Active spaces: H2 (1,1) and N2 (3,3) by ActiveSpace.from_counts; the N2
window is the 2p space (1pi_u, 3sigma_g; 1pi_g, 3sigma_u) at all three
distances and the degenerate-pair check refuses none of them. (0,0) puts
every orbital in the environment, (-1,-1) makes every orbital active.

What the SI columns are, determined from the rows:
- Etot is the solver's lowest root (g_eff[0] = E_0 included) and, in every
  SI row, Etot = Enuc + Ecore + Eh + Ex + Ec + Eact to the 8 printed digits
  (checked here on every row of the JSON). For (0,0) no solver runs and
  Etot = E_0.
- Ecore = e_hf_core, Tr[h D] of the inactive core (0 when no orbital is
  inactive, as in every H2 (1,1) row).
- Eh, Ex: the Hartree and exchange halves of e_hf_ee + e_hf_ee_vtilde, i.e.
  1/2 Tr[J D] and -1/4 Tr[K D] of the inactive core plus the Hartree
  2 sum_kl vt_kk,ll and exchange -sum_kl vt_kl,lk of vtilde on the active
  occupied orbitals. Evidence: the H2 cRPA (1,1) rows have no core but
  Eh = -0.19178962, Ex = +0.09589481 at r = 0.5, which are 2 vt_00,00 and
  -vt_00,00 of our cRPA vtilde to 1.9e-6; BARE and cRPA_ph rows have the
  vtilde part zero (cRPA_ph keeps no (kk|ll) element); the N2 (3,3) cRPA
  rows carry both parts.
- Ec = e_c_full - e_c_active = E_c[v] - E_c^A[v_eff]: E_c[v] in (0,0)
  rows (1.2e-6 for H2 HF r=0.5), 0 in (-1,-1) rows, E_c[v] - E_c^A[v] in
  BARE (1,1) rows (1.4e-6), E_c[v] - E_c^A[W^r] in cRPA rows; the SI's
  cRPA and cRPAph rows carry the same Ec, as ours do (E_c^A reads only the
  particle-hole block, which cRPA_ph keeps).
- Eact = root - E_0, the active-space energy.
- trV = sum_pq (pq|pq) of the bare active v, trW the same of v_eff = v +
  vtilde, the trace over the compound pair index; trV == trW in every BARE
  row of the JSON.

Measured deviations (ours - SI), against the targets of the plan:
- H2/cc-pVTZ, HF and PBE, (0,0): Ecore, Eh, Ex up to 1.6e-5, Ec 2.9e-6,
  Etot 7.9e-6 (target 5e-5).
- H2 (1,1) bare/crpa/crpa_ph and (-1,-1): Etot up to 1.0e-5, Eact 1.0e-5,
  Ec 1.5e-6, Eh/Ex 5.7e-6, Enuc 6.8e-8 (target 5e-5).
- N2/cc-pVDZ HF, (3,3) and (0,0): Ecore, Eh, Ex up to 6.8e-4 (target
  1e-3), Ec 9.5e-6, Etot(screening) - Etot(BARE) 5.2e-6 (target 1e-4),
  Eact 6.3e-5, Etot 5.2e-4.

The traces (ruling R14). The primary check is trW - trV, the screening
content, on every row: measured 5.4e-6 (H2) and 2.0e-5 (N2), tolerance
5e-5. The absolute traces see the bare integrals of the two codes' basis
representations, which differ where the cc-pVTZ overlap of H2 is nearly
singular: the smallest eigenvalue of S is 3.3e-4, 1.7e-3, 0.064, 0.093 at
r = 0.5, 0.75, 1.5, 3.0, and the full-space (-1,-1) trace misses by 10.5,
0.58, 0.064, 0.049 (HF and PBE alike), the (1,1) trace at r = 0.5 by
5.5e-4 (HF) and 1.5e-4 (PBE), while the energies of those rows agree to
1e-5. So the absolute H2 traces are tested at 5e-5 on the (1,1) rows with
r >= 0.75 (measured 1.7e-5) and skipped at r = 0.5 and on (-1,-1). The N2
absolute traces miss the plan's 1e-4 (measured 1.28e-4 trV and 1.41e-4 trW
at r = 1.1, 1.0e-4 / 1.08e-4 at 1.6, 5.1e-5 at 2.2), in line with the N2
HF terms' 6.8e-4, and are tested at 2e-4.

The mRPA rows and the two benzene entries wait for Task 19 (mRPA).
"""
import functools
import json
import pathlib

import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.MultiReference.Downfolding import (build_effective_hamiltonian,
                                            run_low_level)
from src.SingleReference.common.active_space import ActiveSpace
from src.Solvers.active_space import solve
from tests.downfolding_oracle import reference_energy, water

DATA = json.loads((pathlib.Path(__file__).parent / 'data'
                   / 'downfolding_paper_si.json').read_text())
CASES = DATA['cases']
ENERGIES = ('Enuc', 'Ecore', 'Eh', 'Ex', 'Ec', 'Eact', 'Etot')
MRPA_SKIP = pytest.mark.skip(reason='mRPA comes in Task 19')
H2_TRACE_SKIP = pytest.mark.skip(
    reason='absolute trace of the bare H2/cc-pVTZ integrals: near-singular '
           'overlap (smallest S eigenvalue 3.3e-4 at r = 0.5, 1.7e-3 at '
           '0.75) makes the AMS and pyscf virtuals differ; measured misses '
           '5.5e-4 (HF) / 1.5e-4 (PBE) for (1,1) at r = 0.5 and 10.5, 0.58, '
           '0.064, 0.049 for (-1,-1) at r = 0.5, 0.75, 1.5, 3.0 (ruling R14; '
           'trW - trV is tested on these rows)')


def _cases(molecule, spaces, skip_if=None):
    """pytest params of the JSON cases of `molecule` with (n_act_occ,
    n_act_virt) in `spaces`, the mRPA ones skipped, and those for which
    skip_if returns a mark marked with it."""
    out = []
    for c in CASES:
        if c['molecule'] != molecule or (c['n_act_occ'],
                                         c['n_act_virt']) not in spaces:
            continue
        marks = MRPA_SKIP if c['screening'] == 'mrpa' else ()
        if not marks and skip_if is not None and skip_if(c):
            marks = skip_if(c)
        cid = (f"{c['xc']}-r{c['r']}-{c['screening']}-"
               f"{c['n_act_occ']}_{c['n_act_virt']}")
        out.append(pytest.param(c, id=cid, marks=marks))
    return out


def _bare_of(case):
    """The SI's BARE row of the same molecule, xc, r and active space."""
    keys = ('molecule', 'xc', 'r', 'n_act_occ', 'n_act_virt')
    return next(c for c in CASES if c['screening'] == 'bare'
                and all(c[k] == case[k] for k in keys))


@functools.lru_cache(maxsize=None)
def _low(molecule, basis, xc, r):
    """The converged mean field of the diatomic along z at r Angstrom
    (spherical basis) and its casida RPA on exact integrals."""
    atom = molecule[0]
    mol = gto.M(atom=f'{atom} 0 0 0; {atom} 0 0 {r}', basis=basis, unit='A',
                cart=False, verbose=0)
    mf = scf.RHF(mol) if xc == 'HF' else dft.RKS(mol, xc=xc)
    mf.conv_tol = 1e-11
    mf.kernel()
    return run_low_level(mf, method='rpa', mode='casida', df=False)


def _ours(case):
    """Our value of every SI column for `case`, and the Energies record."""
    low = _low(case['molecule'], case['basis'], case['xc'], case['r'])
    mf = low.mf
    nocc, nmo = low.nocc, len(mf.mo_energy)
    n_occ, n_virt = case['n_act_occ'], case['n_act_virt']
    if (n_occ, n_virt) == (-1, -1):
        n_occ, n_virt = nocc, nmo - nocc
    aspace = ActiveSpace.from_counts(mf.mo_energy, nocc, n_occ, n_virt)
    g_bare, _, _ = build_effective_hamiltonian(low, aspace, screening='bare')
    g, space, en = build_effective_hamiltonian(low, aspace,
                                               screening=case['screening'])
    n = space['n_orbs']
    if n:
        e_tot = solve('pyscf_fci', g, n, space['n_elec_active'],
                      spin_free=True)[0]
    else:
        e_tot = en.e0
    v = g_bare[2].transpose(0, 2, 1, 3)
    v_eff = g[2].transpose(0, 2, 1, 3)
    vt = (v_eff - v)[:n_occ, :n_occ, :n_occ, :n_occ]
    c_core = mf.mo_coeff[:, aspace.env_occ]
    dm = 2 * c_core @ c_core.T
    vj, vk = mf.get_jk(mf.mol, dm)
    e_h = 0.5 * np.einsum('ij,ji->', vj, dm) + 2 * np.einsum('kkll->', vt)
    e_x = -0.25 * np.einsum('ij,ji->', vk, dm) - np.einsum('kllk->', vt)
    ours = {'trV': np.einsum('pqpq->', v), 'trW': np.einsum('pqpq->', v_eff),
            'Enuc': en.e_nuc, 'Ecore': en.e_hf_core, 'Eh': e_h, 'Ex': e_x,
            'Ec': en.e_c_full - en.e_c_active, 'Eact': e_tot - en.e0,
            'Etot': e_tot}
    ours['dTr'] = ours['trW'] - ours['trV']
    return ours, en


def _misses(ours, case, tols):
    """{column: (ours - SI)} for the columns whose deviation exceeds tols;
    the column 'dTr' is trW - trV."""
    ref = dict(case, dTr=case['trW'] - case['trV'])
    dev = {k: ours[k] - ref[k] for k in tols}
    return {k: d for k, d in dev.items() if abs(d) > tols[k]}


def test_si_columns_add_up():
    """Every SI row: Etot = Enuc + Ecore + Eh + Ex + Ec + Eact to the printed
    8 digits (5e-8 rounding per column), and trV == trW in BARE rows: the
    column reading the module docstring builds on."""
    for c in CASES:
        parts = sum(c[k] for k in ('Enuc', 'Ecore', 'Eh', 'Ex', 'Ec', 'Eact'))
        assert abs(parts - c['Etot']) < 4e-8, c
        if c['screening'] == 'bare':
            assert c['trV'] == c['trW'], c


@pytest.mark.parametrize('case', _cases('H2', {(0, 0)}))
def test_h2_all_environment_row(case):
    """H2/cc-pVTZ, HF and PBE orbitals, every orbital in the environment:
    the HF terms, E_RPA,corr and E_0. Measured: Ecore/Eh/Ex at most 1.6e-5,
    Ec 2.9e-6, Etot 7.9e-6 (tolerance 5e-5)."""
    ours, en = _ours(case)
    assert abs(ours['Eh'] + ours['Ex'] - en.e_hf_ee) < 1e-10
    assert not _misses(ours, case, dict.fromkeys(ENERGIES + ('dTr',), 5e-5))


@pytest.mark.parametrize('case', _cases('H2', {(1, 1), (-1, -1)}))
def test_h2_downfolded_rows(case):
    """H2/cc-pVTZ (1,1) for bare, crpa, crpa_ph and every orbital active:
    every energy column and trW - trV. Measured: energies at most 1.0e-5
    (Etot, Eact), trW - trV 5.4e-6 (tolerance 5e-5)."""
    ours, en = _ours(case)
    assert abs(ours['Eh'] + ours['Ex']
               - (en.e_hf_ee + en.e_hf_ee_vtilde)) < 1e-10
    assert not _misses(ours, case, dict.fromkeys(ENERGIES + ('dTr',), 5e-5))


def _h2_trace_skip(case):
    """The skip mark of the absolute H2 traces (ruling R14), or None."""
    if case['n_act_occ'] == -1 or case['r'] < 0.75:
        return H2_TRACE_SKIP
    return None


@pytest.mark.parametrize('case', _cases('H2', {(1, 1), (-1, -1)},
                                        skip_if=_h2_trace_skip))
def test_h2_absolute_traces(case):
    """trV and trW of the H2 (1,1) rows with r >= 0.75: measured at most
    1.7e-5 (tolerance 5e-5). Skipped where the overlap is near-singular
    (module docstring)."""
    ours, _ = _ours(case)
    assert not _misses(ours, case, dict.fromkeys(('trV', 'trW'), 5e-5))


@pytest.mark.parametrize('case', _cases('N2', {(3, 3), (0, 0)}))
def test_n2_downfolded_rows(case):
    """N2/cc-pVDZ HF (3,3) for bare, crpa, crpa_ph, and every orbital in the
    environment. The absolute HF terms differ by up to 6.8e-4 from AMS
    (tolerance 1e-3); Ec (9.5e-6) and Etot(screening) - Etot(BARE)
    (5.2e-6) to 1e-4; Eact (6.3e-5) to 1e-4; Etot (5.2e-4) to 1e-3;
    trW - trV (2.0e-5) to 5e-5 and the absolute traces (1.41e-4, ruling
    R14) to 2e-4."""
    ours, en = _ours(case)
    assert abs(ours['Eh'] + ours['Ex']
               - (en.e_hf_ee + en.e_hf_ee_vtilde)) < 1e-10
    tols = {'Enuc': 1e-6, 'Ecore': 1e-3, 'Eh': 1e-3, 'Ex': 1e-3, 'Ec': 1e-4,
            'Eact': 1e-4, 'Etot': 1e-3, 'dTr': 5e-5}
    if case['n_act_occ'] > 0:
        tols.update(trV=2e-4, trW=2e-4)
    assert not _misses(ours, case, tols)
    bare = _bare_of(case)
    ours_bare = _ours(bare)[0]
    rel = ((ours['Etot'] - ours_bare['Etot'])
           - (case['Etot'] - bare['Etot']))
    assert abs(rel) < 1e-4


def test_pbe0_orbitals_no_xc_leak():
    """Water/6-31G (4,4) on PBE0 orbitals: E_nuc + e_hf_core + e_hf_ee plus
    the reference determinant's active part is the HF energy expression on
    those orbitals, no exchange-correlation functional in it (spec §3.1).
    Measured 0 (tolerance 1e-9)."""
    mf = dft.RKS(water().mol, xc='pbe0')
    mf.conv_tol = 1e-11
    mf.verbose = 0
    mf.run()
    low = run_low_level(mf, method='rpa', mode='casida', df=False)
    aspace = ActiveSpace.from_counts(mf.mo_energy, 5, 2, 2)
    g, _, en = build_effective_hamiltonian(low, aspace, screening='bare')
    active = reference_energy(g, 2) - g[0]
    e_hf = scf.RHF(mf.mol).energy_tot(dm=mf.make_rdm1())
    assert abs(en.e_nuc + en.e_hf_core + en.e_hf_ee + active - e_hf) < 1e-9


@pytest.mark.skip(reason='mRPA and the benzene geometry come in Task 19')
def test_benzene_66_frobenius():
    """sqrt(sum C^2) of the (6,6) screening tables (see the JSON's
    multiplicity note)."""


@pytest.mark.skip(reason='mRPA and the benzene geometry come in Task 19')
def test_benzene_1010_omega_r():
    """The environment's neutral excitations Omega^R of benzene (10,10)."""
