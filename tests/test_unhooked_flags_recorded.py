"""Flags the library hands out are copied by their callers into what those
return or keep:

  * a charged surface's orbital tracking: an ambiguous match
    (`track_orbital(out=)`) is in its `follow_log` entry;
  * `calc_qp_energy(out=)` receives the unrestricted reference's
    diagnostics, which `quasiparticle_order` returns;
  * a quasi-boson root followed off its branch (`bracketed_root(out=)`) is
    warned about and kept in `QPqb.branch_log`;
  * the spin-vibronic coupling returns its couplings' `scf_flags`.

Stand-ins where the event cannot be made cheaply; one OH/STO-3G UHF.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import ao2mo, gto, scf

from src.gradients.gwbse import rpa_bse_surface
from src.properties import characters, spin_vibronic
from src.SingleReference.GW import quasi_boson
from src.SingleReference.GW.qp_energy import calc_qp_energy

AMBIGUOUS = {'candidates': [4, 5], 'margin': 0.01}


def test_an_ambiguous_orbital_match_is_in_the_follow_log(monkeypatch):
    def tracking(fingerprint, mf, out=None):
        out['ambiguous'] = dict(AMBIGUOUS)
        return 4, {4: 0.80, 5: 0.79}

    monkeypatch.setattr(rpa_bse_surface, 'track_orbital', tracking)
    surface = rpa_bse_surface._FollowsOrbital.__new__(
        rpa_bse_surface._FollowsOrbital)
    surface.nocc, surface.state, surface.track = 5, 0, True
    surface._fingerprint, surface.mol0, surface.follow_log = {}, object(), []
    assert surface.offset_at(object(), None) == 0
    entry, = surface.follow_log
    assert entry['ambiguous'] == AMBIGUOUS and entry['orbital'] == 4


def test_the_qp_energy_hands_out_the_reference_diagnostics():
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='sto-3g', spin=1,
                verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    seen = {}
    calc_qp_energy(mf, df=False, state='homo', out=seen)
    diagnostics = seen['reference_diagnostics']
    assert diagnostics is mf.reference_diagnostics
    assert {'s2', 's2_exact', 'stable', 'spin_contaminated'} <= set(diagnostics)


def test_quasiparticle_order_returns_the_diagnostics(monkeypatch):
    held = {'s2': 0.76, 's2_exact': 0.75, 'stable': True,
            'spin_contaminated': False}

    def energies(mf, states, spin_channel):
        return {p: -10.0 - p for p in states}, held

    mol = gto.M(atom='He 0 0 0', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run()
    out = characters.quasiparticle_order(mf, 0, energies)
    assert out['reference_diagnostics'] is held


def test_a_quasi_boson_root_off_its_branch_is_logged(monkeypatch):
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run()
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), mf.mo_coeff.shape[1])
    qp = quasi_boson.QPqb(mf.mo_energy, eri, 1)

    def flipped(num, pol, a, w0, labels=None, out=None):
        out.update(labels=None, largest=(0.1, 0.9), followed=True,
                   flipped=True)
        return -0.55, 0.3

    monkeypatch.setattr(quasi_boson, 'bracketed_root', flipped)
    with pytest.warns(RuntimeWarning, match='branch is kept and logged'):
        w, z = qp.solve_diag(0, w0=-0.6)
    assert (w, z) == (-0.55, 0.3)
    entry, = qp.branch_log
    assert entry['orbital'] == 0 and entry['largest'] == [0.1, 0.9]


def test_the_spin_vibronic_coupling_returns_the_couplings_scf_flags(
        monkeypatch):
    flag = {'where': 'evaluated_coupling', 'residual': 3e-6, 'tol': 1e-6}
    natm = 2
    monkeypatch.setattr(spin_vibronic, 'occupied_count', lambda mf: 1)
    monkeypatch.setattr(spin_vibronic, 'check_normalization',
                        lambda *a: None)
    monkeypatch.setattr(spin_vibronic, 'soc_vectors',
                        lambda *a: np.ones(3))
    s1, s2, t1, t2 = (('singlet', 0), ('singlet', 1), ('triplet', 0),
                      ('triplet', 1))
    x = np.eye(2)
    ev = SimpleNamespace(
        mf=None, target={k: (k[0], k[1]) for k in (s1, s2, t1, t2)},
        root={k: k[1] for k in (s1, s2, t1, t2)},
        spectrum={'singlet': (np.array([0.1, 0.2]), x, 0 * x),
                  'triplet': (np.array([0.05, 0.15]), x, 0 * x)},
        coupling=lambda m, n: (np.zeros((natm, 3)),
                               {'scf_flags': [dict(flag)]}))
    out = spin_vibronic.spin_vibronic_coupling(ev, None, s1, t1, (s2,), (t2,))
    assert out['scf_flags'] == [flag]
