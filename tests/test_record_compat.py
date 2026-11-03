"""One older record of each kind is read by the rules of `record_compat`, as
the current code would have written the same run.

  1. a Numerics record without the later sections, without scf.xc_grid and
     with the retired scf fields at the values they always held reads as the
     current record at pyscf's default XC grid and the exact SOMF; a retired
     field at another value is refused;
  2. a photophysics stamp whose scf sections name the retired fields reads as
     the stamp without them, hashed again;
  3. a stage stamp field written only away from its absence's meaning;
  4. a frozen factorization record without test_set and pivot_points reads as
     the Duchemin-Blase fit on the grid's own points;
  5. a solvated record without the later settings reads at the values they
     were made with (an ADC(2) solvent record ran the LR arm);
  6. a code stamp of the full-commit form names its short commit.
"""
import pytest
from pyscf import gto

from src.Base.numerics.constants import ISDF_DUCHEMIN_BLASE, XC_GRID_LEVEL_UNRECORDED
from src.Base.numerics.numerics import SECTIONS, Numerics
from src.Base.utils import record_compat as rc
from src.Base.utils.record_io import record_hash
from src.gradients.rpa.factor_chain import FrozenFactorization
from src.photophysics.store import stamp_as_read
from tests.isdf_test_grid import g1
from tests.molecules import WATER



def test_an_older_numerics_record():
    now = Numerics().to_record()
    old = {k: (dict(v) if isinstance(v, dict) else v) for k, v in now.items()
           if k not in rc.NUMERICS_LATER_SECTIONS}
    old['scf'].pop('xc_grid')
    for section, name, held in rc.NUMERICS_RETIRED:
        old[section][name] = held
    read = Numerics.from_record(old)
    assert read.scf.xc_grid == XC_GRID_LEVEL_UNRECORDED
    want = Numerics(scf={'xc_grid': XC_GRID_LEVEL_UNRECORDED},
                    **rc.NUMERICS_UNRECORDED_SECTIONS)
    assert read.values() == want.values()
    assert read.identity() == want.identity()
    section, name, held = rc.NUMERICS_RETIRED[0]
    old[section][name] = 2 * held
    with pytest.raises(ValueError, match=f'{section}.{name}, a retired'):
        Numerics.from_record(old)
    assert set(rc.NUMERICS_LATER_SECTIONS) <= set(SECTIONS)


def test_an_older_photophysics_stamp():
    scf = Numerics().to_record()['scf']
    stamp = {'s0': {'scf': scf}, 'hess': {'ground': {'scf': dict(scf)}}}
    old = {'s0': {'scf': dict(scf, cphf_tol=1e-9)},
           'hess': {'ground': {'scf': dict(scf, cphf_max_cycle=100)}}}
    read = stamp_as_read(dict(old, identity=record_hash(old)))
    assert read == dict(stamp, identity=record_hash(stamp))
    held = dict(stamp, identity=record_hash(stamp))
    assert stamp_as_read(held) is held


def test_a_stage_stamp_field_is_written_only_away_from_its_absence():
    for name, absent in rc.STAGE_STAMP_ABSENT.items():
        assert rc.stamp_unless({name: 'x'}, name, absent) == {}
        assert rc.stamp_unless({}, name, 'declared') == {name: 'declared'}


def test_an_older_factorization_record():
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    fz = FrozenFactorization(mol, Numerics(isdf={'test_set': ISDF_DUCHEMIN_BLASE,
                                                 'counts': g1(mol)}))
    record = fz.to_record()
    old = {k: v for k, v in record.items() if k not in rc.FACTORIZATION_ABSENT}
    read = FrozenFactorization.from_record(old)
    assert (read.test_set, read.pivot_points) == (ISDF_DUCHEMIN_BLASE, None)
    assert record_hash(read.to_record()) == record_hash(record)


def test_an_older_solvated_record():
    for key, value in rc.SOLVATED_ABSENT.items():
        assert rc.solvated_absent({'state_method': 'gwbse'}, key) == value
    assert rc.solvated_absent({'state_method': 'adc2', 'solvent': 'toluene'},
                              'solvent_arm') == 'ss+lr'
    assert rc.solvated_absent({}, 'no_such_setting') == '<absent>'


def test_an_older_code_stamp():
    full = '0123456789abcdef0123456789abcdef01234567'
    assert rc.same_code(full, full[:8])
    assert not rc.same_code(full, full[:8] + '+dirty')
    assert not rc.same_code(full, 'unknown')
    assert not rc.same_code(full[:8], full[:8] + ':abcdef012345')
