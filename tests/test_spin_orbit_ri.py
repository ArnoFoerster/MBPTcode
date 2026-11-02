"""The spin-orbit mean field by the resolution of the identity ('ri-somf').

Gated, on formaldehyde (aug-cc-pVDZ, B3LYP) with four singlet and four
triplet Tamm-Dancoff roots:
  * every |<S_I|H_SO|T_J>| of the table (ground state included) is the exact
    SOMF one to RI_SOC_BAR_CM with aug-cc-pVDZ-jkfit, and the operator's
    error falls when the fitting basis grows to aug-cc-pVTZ-jkfit;
  * cut into many tiles it is the one-tile operator to summation order;
  * over 1, 2 and 3 thread-ranks, each rank handed its own density, every
    rank returns the serial operator of rank 0's density, bit for bit, many
    tiles to a rank;
  * unnamed, the fitting basis is the orbital basis's JK-fitting set; one
    is named for 'ri-somf' alone, and an orbital basis without a predefined
    set is refused (pc-1), naming soc_auxbasis;
  * the photophysics pass runs the declared term, 'ri-somf' by default, and
    its record names it; a Numerics record without the properties section
    reads the exact SOMF, and the section is not hashed.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto, tdscf

from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.simulated_comm import run_simulated
from src.SingleReference.LinearResponse.linear_response import from_pyscf
from src.Base.pyscf.spin_orbit_mean_field import (_soc_jk, _soc_jk_ri,
                                                 jkfit_auxbasis,
                                                 mean_field_term)
from src.properties.spin_orbit import (SOC_PREFACTOR, socme_table,
                                       x2c_one_electron)

CH2O = 'O 0 0 0; C 0 0 1.205; H 0 0.94 1.79; H 0 -0.94 1.79'
HARTREE_TO_CM = 219474.6313705

#: The largest |SOCME| error, cm^-1, allowed of aug-cc-pVDZ-jkfit (measured
#: 5.6e-4 against a largest element of 57 cm^-1).
RI_SOC_BAR_CM = 5e-3

#: Tiles of 2e-4 GB, which cut this molecule into several tiles per rank.
SMALL_TILES = Numerics().derive('several tiles a rank on a small molecule',
                                layout={'tile_gb': 2e-4})


@pytest.fixture(scope='module')
def ch2o():
    mol = gto.M(atom=CH2O, basis='aug-cc-pvdz', verbose=0)
    mf = dft.RKS(mol, xc='b3lyp').density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    roots = {}
    for spin in ('singlet', 'triplet'):
        td = tdscf.TDA(mf)
        td.singlet = spin == 'singlet'
        td.nstates = 4
        td.kernel()
        roots[spin] = from_pyscf(td)
    vj, vk = _soc_jk(mol, mf.make_rdm1())
    return mol, mf, roots, vj - 1.5 * vk


def _table(mol, mf, roots, term):
    h = x2c_one_electron(mol) + term * SOC_PREFACTOR
    h_mo = np.einsum('xpq,pi,qj->xij', h, mf.mo_coeff, mf.mo_coeff)
    (_, xs, ys), (_, xt, yt) = roots['singlet'], roots['triplet']
    return socme_table(h_mo, mol.nelectron // 2, xs, ys, xt, yt) * HARTREE_TO_CM


def test_the_fitted_table_is_the_exact_one_within_its_bar(ch2o):
    mol, mf, roots, exact = ch2o
    dm = mf.make_rdm1()
    ref = _table(mol, mf, roots, exact)
    ri = mean_field_term(mol, dm, 'ri-somf', auxbasis='aug-cc-pvdz-jkfit')
    err = abs(_table(mol, mf, roots, ri) - ref).max()
    assert ref.max() > 10.0
    assert err < RI_SOC_BAR_CM, err
    larger = mean_field_term(mol, dm, 'ri-somf', auxbasis='aug-cc-pvtz-jkfit')
    assert abs(larger - exact).max() < abs(ri - exact).max()


def test_many_tiles_are_one_tile_to_summation_order(ch2o):
    mol, mf, _, _ = ch2o
    dm = mf.make_rdm1()
    one = _soc_jk_ri(mol, dm, 'aug-cc-pvdz-jkfit', None)
    many = _soc_jk_ri(mol, dm, 'aug-cc-pvdz-jkfit', None, SMALL_TILES)
    assert abs(many - one).max() < 1e-11


@pytest.mark.parametrize('size', [1, 2, 3])
def test_every_rank_count_gives_the_serial_bits(ch2o, size):
    mol, mf, _, _ = ch2o
    dm = mf.make_rdm1()
    serial = _soc_jk_ri(mol, dm, 'aug-cc-pvdz-jkfit', None, SMALL_TILES)

    def one_rank(comm):
        mine = dm + 1e-9 * comm.Get_rank()
        return _soc_jk_ri(mol, mine, 'aug-cc-pvdz-jkfit', comm, SMALL_TILES)
    for r, h in enumerate(run_simulated(one_rank, size)):
        assert np.array_equal(h, serial), f'rank {r} of {size}'


def test_the_fitting_basis_belongs_to_ri_somf_alone(ch2o):
    """Unnamed, it is the orbital basis's JK-fitting set; named, it belongs to
    'ri-somf' alone; an orbital basis without a predefined set is refused,
    naming the setting to declare."""
    mol, mf, _, _ = ch2o
    dm = mf.make_rdm1()
    assert set(jkfit_auxbasis(mol).values()) == {'aug-cc-pvdz-jkfit'}
    assert np.array_equal(mean_field_term(mol, dm, 'ri-somf'),
                          mean_field_term(mol, dm, 'ri-somf',
                                          auxbasis='aug-cc-pvdz-jkfit'))
    with pytest.raises(ValueError, match='fitting basis'):
        mean_field_term(mol, dm, 'somf', auxbasis='aug-cc-pvdz-jkfit')
    with pytest.raises(ValueError, match='fitting basis'):
        Numerics(properties={'soc_two_electron': 'somf',
                             'soc_auxbasis': 'aug-cc-pvdz-jkfit'})
    bare = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='pc-1',
                 verbose=0)
    with pytest.raises(ValueError, match='soc_auxbasis'):
        mean_field_term(bare, np.eye(bare.nao_nr()), 'ri-somf')


def test_the_properties_pass_runs_the_declared_term(ch2o, monkeypatch):
    """`geometry_properties` hands the run's properties section (term and
    fitting basis) to the spin-orbit pass, 'ri-somf' in the JK-fitting set at
    its default, and the record names it; the property routines are
    stubbed."""
    from types import SimpleNamespace
    from src.photophysics import stages
    mol, mf, roots, _ = ch2o
    seen = []
    monkeypatch.setattr(stages, 'oscillator_strengths',
                        lambda *a: (np.zeros(1), np.zeros((1, 3))))
    monkeypatch.setattr(stages, 'spin_orbit_couplings',
                        lambda *a, **k: seen.append(k) or {
                            'socme': np.zeros((2, 1)), 'h_mo': None})
    monkeypatch.setattr(stages, 'condon_element',
                        lambda *a: (0.0, np.zeros(3)))
    ev = SimpleNamespace(mol=mol, mf=mf, e0=0.0, energy={}, omega={},
                         root={}, spectrum=roots)
    keys = (('singlet', 0), ('triplet', 0))
    declared = Numerics().derive('the fitted operator', properties={
        'soc_two_electron': 'ri-somf', 'soc_auxbasis': 'aug-cc-pvdz-jkfit'})
    exact = Numerics().derive('the exact operator', properties={
        'soc_two_electron': 'somf'})
    records = [stages.geometry_properties(ev, *keys, 1, derivatives=False,
                                          numerics=n)
               for n in (declared, Numerics(), exact)]
    terms = [('ri-somf', 'aug-cc-pvdz-jkfit'), ('ri-somf', None),
             ('somf', None)]
    assert [k['numerics'] for k in seen] == [declared, Numerics(), exact]
    assert [(r['soc_term']['two_electron'], r['soc_term']['auxbasis'])
            for r in records] == terms
    assert {r['soc_term']['one_electron'] for r in records} == {'x2c'}


def test_a_record_without_the_section_reads_the_exact_somf():
    """A record written before the properties section existed was computed
    with the exact SOMF and reads so, while the default is 'ri-somf'; the
    section is recorded and not hashed, so every identity (and every store
    stamped with one) is the one it was."""
    base = Numerics()
    assert (base.properties.soc_two_electron,
            base.properties.soc_auxbasis) == ('ri-somf', None)
    record = base.to_record()
    assert Numerics.from_record(record).properties == base.properties
    record.pop('properties')
    old = Numerics.from_record(record)
    assert old.properties.soc_two_electron == 'somf'
    assert old.identity() == base.identity()
    fitted = base.derive('a named fitting basis', properties={
        'soc_auxbasis': 'aug-cc-pvdz-jkfit'})
    assert fitted.identity() == base.identity()
