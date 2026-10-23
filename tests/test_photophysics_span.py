"""The span of a photophysics run is declared, never chosen by refusal, and
the records name the cut that ran.

On PXZ-TRZ
aug-cc-pVDZ in toluene the 1e-3 span moved T1 by -29 and dEST by +30 meV
against DF in the full basis. Every basis is tried in the full basis; a
molecule whose orbital densities the factors miss is refused there.

  * `--span` has no default: 'full' or a cut between 0 and 1; 'refusal' and
    a cut outside (0, 1) are refused, and the declaration has no span policy;
  * the full basis, augmented or not, is not probed: no SCF before the first
    stage, and every SCF of the run is in the full basis;
  * a declared cut is counted at the input geometry, every SCF of the run is
    solved in it, and the stamps carry its cut and count;
  * a vertical arm records the orbital refusal of the full basis;
  * formaldehyde aug-cc-pVDZ runs a vertical arm in the full basis on the
    chain itself (DF mean field, G3 factors) and is not refused.
"""
from dataclasses import fields, replace

import numpy as np
import pytest

from src.Base.factorization.isdf_jk import canonical_span_size
from src.Base.factorization.separable_ri import UnrepresentedOrbitals
from src.photophysics import chain, vertical_arms
from src.photophysics.cli import parser
from src.photophysics.declaration import DeclarationRefused, Policies
from tests.photophysics_cases import FORMALDEHYDE, argv, declaration, run_in

BENZENE = ('C 0.0000 1.3970 0.0000; C 1.2098 0.6985 0.0000; '
           'C 1.2098 -0.6985 0.0000; C 0.0000 -1.3970 0.0000; '
           'C -1.2098 -0.6985 0.0000; C -1.2098 0.6985 0.0000; '
           'H 0.0000 2.4810 0.0000; H 2.1486 1.2405 0.0000; '
           'H 2.1486 -1.2405 0.0000; H 0.0000 -2.4810 0.0000; '
           'H -2.1486 -1.2405 0.0000; H -2.1486 1.2405 0.0000')


def augmented(*extra):
    """The tests' declaration on benzene aug-cc-pVDZ (13 of 192 overlap
    eigenvalues below 1e-4), on a DF mean field (no ISDF-K grid is
    validated at GH for the basis)."""
    return declaration('--atom', BENZENE, '--system', 'benzene', '--basis',
                       'aug-cc-pvdz', '--scf', 'df', *extra)


def test_the_span_is_declared():
    full = argv()
    k = full.index('--span')
    with pytest.raises(SystemExit):
        parser().parse_args(full[:k] + full[k + 2:])
    for word in ('refusal', '0', '1', '-1e-4', 'none'):
        with pytest.raises(SystemExit):
            parser().parse_args(argv('--span', word))
    assert 'span' not in {f.name for f in fields(Policies)}
    d = declaration()
    assert d.numerics.span.cut is None
    with pytest.raises(DeclarationRefused, match='numerics.span.cut'):
        replace(d, numerics=d.numerics.derive('a cut above the overlap',
                                              span={'cut': 2.0}))
    assert declaration('--span', '1e-04').numerics.span.cut == 1e-4
    assert declaration('--span', '3e-4').numerics.span.cut == 3e-4


def test_the_full_basis_is_not_probed(tmp_path, monkeypatch):
    """cc-pVDZ and aug-cc-pVDZ alike: the full basis, no SCF before the
    first stage, and the run's factory builds every SCF in it."""
    run = run_in(tmp_path, declaration())
    assert run.span == {'cut': None, 'n_kept': None,
                        'nao': run.mol_in.nao_nr()}
    assert run.histories == [], 'no SCF before the first stage'
    assert run.stamp('s0')['span'] == run.span
    aug = run_in(tmp_path / 'aug', augmented('--span', 'full'))
    assert aug.span == {'cut': None, 'n_kept': None, 'nao': 192}
    assert aug.histories == []
    calls = []
    monkeypatch.setattr(chain, 'mean_field_factory',
                        lambda *a, **k: calls.append(k['span']))
    aug.mean_field_factory(aug.declaration.numerics)
    assert calls == [(None, None)], 'every SCF of the run in the full basis'


def test_a_declared_cut_is_counted_and_stamped(tmp_path, monkeypatch):
    run = run_in(tmp_path, augmented('--span', '1e-04'))
    kept = canonical_span_size(run.mol_in, 1e-4)
    assert kept == 179
    assert run.span == {'cut': 1e-4, 'n_kept': kept, 'nao': 192}
    assert run.histories == []
    assert run.stamp('vert')['span'] == run.span
    calls = []
    monkeypatch.setattr(chain, 'mean_field_factory',
                        lambda *a, **k: calls.append(k['span']))
    run.mean_field_factory(run.declaration.numerics)
    assert calls == [(kept, 1e-4)], 'every SCF of the run is in its span'


def test_a_vertical_arm_records_the_orbital_refusal(monkeypatch):
    """Where the factors miss an orbital density in the full basis the arm
    stops with the refusal in its record, the span it ran in beside it."""
    def refused(*_):
        raise UnrepresentedOrbitals('missed', [5], np.array([0.4] * 6))
    monkeypatch.setattr(vertical_arms, 'sop_forward', refused)
    rec = vertical_arms.vertical_arm(declaration(qp_states='admitted'),
                                     'sop-admitted', max_memory=4000)
    assert rec['refusal']['orbitals'] == [5]
    assert rec['span_choice']['cut'] is None
    assert 'spectrum' not in rec


def test_an_augmented_basis_runs_in_the_full_basis():
    """Formaldehyde aug-cc-pVDZ (one overlap eigenvalue below 1e-3, none
    cut): the DF mean field and the G3 factors in the full basis represent
    every orbital density, and the vertical arm runs to its spectrum."""
    d = declaration('--atom', FORMALDEHYDE, '--basis', 'aug-cc-pvdz',
                    '--scf', 'df', '--span', 'full', qp_states='admitted')
    rec = vertical_arms.vertical_arm(d, 'sop-admitted', max_memory=4000)
    assert 'refusal' not in rec, rec.get('refusal')
    assert rec['span_choice'] == {'cut': None, 'n_kept': None, 'nao': 64}
    assert 'span' not in rec, 'no span-confined mean field'
    assert 0 < rec['spectrum']['T1'] < rec['spectrum']['S1']
