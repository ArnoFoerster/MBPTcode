"""The adaptive selection's analytic continuation in a continuum, sliced over
ranks: it screens with the bare gauge, as serially.

The continuation (`ac_quasiparticle_shifts`) takes the chain's static term,
<p|Sigma_x - v_xc|p> + Sigma^env_pp (Eq. (18) of Duchemin, Jacquemin and
Blase, J. Chem. Phys. 144, 164106 (2016)), so its Sigma_c must screen with
W_gas. Sliced, X_mo's own D is the dressed gauge the BSE kernel reads; the
bare gauge's rows are a `SlicedFactors` of their own (`bare_factor`).

Water/cc-pVDZ, DF Hartree-Fock, PCM(toluene), G1 factors sliced over 2 and 3
simulated ranks, SOP residues, the adaptive set for S1 and T1: the
selection's AC shifts and the a posteriori check's are the serial ones to
round-off, and the partition and S1 are the serial ones.

What breaks the gate: both sites handing the continuation X_mo (whose own D
is the dressed gauge) move, under ranks, the occupied shifts by -0.009 to
+0.020 eV and the virtual ones by +0.079 to +0.126 eV, the check's by up to
0.126 eV; the partition holds.
"""
import warnings

import numpy as np
import pytest
from pyscf import gto, lib

from src.Base.numerics.numerics import Numerics
from src.Base.numerics.constants import RANK_SPLIT_ENERGY_TOL
from src.Base.declaration import QPStates
from src.environments.solvent_screening import SolventScreening
from src.Base.utils.parallel.mpi_grid import distributed
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.GW.qp_states import resolve_qp_states
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.gwbse.diagnostics import adaptive_check
from tests.isdf_test_grid import g1_numerics
from tests.reference_scf import rhf_df_jkfit

H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
TARGETS = (('singlet', 0), ('triplet', 0))
#: eV; a Pade continuation over re-associated sums. Measured 3e-15 eV; the
#: double count it guards against is 0.13 eV.
AC_SPLIT_TOL_EV = 1e-8


def selected(comm=None):
    """(selection AC shifts, check AC shifts, partition record, S1) on this
    rank's own Mole, cavity and mean field."""
    warnings.simplefilter('ignore')
    mol = gto.M(atom=H2O, basis='cc-pvdz', verbose=0)
    mf = rhf_df_jkfit(mol)
    window = list(resolve_qp_states(
        QPStates('adaptive'), np.asarray(mf.mo_energy, float),
        mol.nelectron // 2, degeneracy_tol=1e-4).explicit)
    chain = ExcitedStateChain(
        mol, rhf_df_jkfit, mf=mf,
        environment=SolventScreening(mol, solvent='toluene'),
        numerics=g1_numerics(mol, Numerics(qp={'window': window, 'residue_route': 'sop', 'scissor': 'calibrate', 'select': QPStates('adaptive', targets=TARGETS)},
                                           bse={'solver': 'dense'},
                                           layout={'sliced': True})))
    omega = chain.excitation(mol, mf)
    record = chain.selection_record()
    check = adaptive_check(chain, mol)
    return (record['ac']['shift_eV'], check['ac_shift_eV'],
            record['partition'], float(omega))


@pytest.fixture(scope='module')
def serial():
    threads = lib.num_threads()
    lib.num_threads(1)
    try:
        with distributed(None):
            return selected()
    finally:
        lib.num_threads(threads)


@pytest.mark.parametrize('size', [2, 3])
def test_sliced_continuation_screens_bare(serial, size):
    threads = lib.num_threads()
    lib.num_threads(1)
    try:
        out = run_simulated(selected, size)
    finally:
        lib.num_threads(threads)
    ac0, post0, part0, om0 = serial
    assert part0['tier_of'], 'the gate needs holes'
    for rank, (ac, post, part, om) in enumerate(out):
        d_ac = max(abs(ac[p] - ac0[p]) for p in ac0)
        d_post = max(abs(post[p] - post0[p]) for p in post0)
        print(f'\n{size} ranks, rank {rank}: selection AC {d_ac:.2e} eV, '
              f'check AC {d_post:.2e} eV, S1 {abs(om - om0):.2e} Ha')
        assert set(ac) == set(ac0) and d_ac < AC_SPLIT_TOL_EV, rank
        assert set(post) == set(post0) and d_post < AC_SPLIT_TOL_EV, rank
        assert part == part0, rank
        assert abs(om - om0) < RANK_SPLIT_ENERGY_TOL, rank
