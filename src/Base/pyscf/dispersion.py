"""The empirical dispersion correction of a mean field, and who may carry one.

A hybrid has no long-range correlation, so a semi-classical -C6/R^6 sum is
added to it; pyscf spells the choice in the functional itself (`xc='pbe0-d4'`)
and applies it to both the energy and the nuclear gradient. The term is a
function of the NUCLEAR COORDINATES ALONE -- it does not touch the orbitals,
the density, or any excitation energy -- so it moves a potential energy surface
without moving a spectrum.

WHO MAY CARRY ONE. A dispersion-corrected functional is the right ground state
for a molecule whose own fragments are in van der Waals contact, which is every
twisted donor-acceptor emitter. It is the WRONG partner for a direct-RPA
ground state, which already contains dispersion in its correlation energy: the
two would be added twice, and the parameters of a D3 or D4 fit are fitted to
what a particular density functional LACKS, which is not what dRPA lacks.
"""
import numpy as np

#: The corrections pyscf can apply, as they are spelled in an `xc` string.
#: Written out rather than read from pyscf, because the two routines that
#: answer a question ABOUT A NAME -- which functional carries a correction,
#: and whether a second one is being added to it -- must work where pyscf is
#: older than its own dispersion support. A cluster environment that cannot
#: evaluate the term can still be told not to ask for it twice.
DISPERSION_VERSIONS = ('d3bj', 'd3zero', 'd3bjm', 'd3zerom', 'd3op', 'd4')

_MISSING = ('the dispersion backend is not installed: `pip install '
            'pyscf-dispersion` provides the D3 and D4 libraries that '
            '{version!r} needs')


#: Names that already carry dispersion or a nonlocal correlation term of their
#: own. A separate keyword on top of one of these is a double count, and unlike
#: the dRPA case nothing downstream would notice.
_SELF_CONTAINED = ('-d3', '-d4', '-d2', '-v', '-3c')


def functional_carries_dispersion(name):
    """Whether a functional's own name says it already includes dispersion.

    True for the range-separated hybrids fitted with it (wB97X-D3, wB97X-D4),
    for the ones carrying a nonlocal VV10 term (wB97X-V, wB97M-V, B97M-V), and
    for the composite methods that bundle it (r2SCAN-3c, B97-3c, PBEh-3c).
    """
    key = str(name).strip().lower()
    return any(key.endswith(tail) or f'{tail}-' in key for tail in _SELF_CONTAINED)


def refuse_double_dispersion(xc, dispersion):
    """Raise if a separate correction is asked for on top of a functional that
    already has one."""
    if not dispersion or str(dispersion).lower() == 'none':
        return
    if functional_carries_dispersion(xc):
        raise ValueError(
            f'{xc!r} already includes dispersion by construction, so adding '
            f'{dispersion!r} on top of it counts the same physics twice. Ask '
            f'for the plain functional with a correction, or the corrected '
            f'functional with none.')


def _pyscf_dispersion():
    """pyscf's dispersion module, or a clear failure. Imported on use, so a
    pyscf without it still serves the name checks above."""
    try:
        from pyscf.scf import dispersion
    except ImportError as exc:
        raise ImportError(
            'this pyscf has no `pyscf.scf.dispersion`, so it cannot evaluate '
            'a dispersion correction. The name checks in this module work '
            'without it; evaluating needs a newer pyscf and '
            '`pip install pyscf-dispersion`.') from exc
    return dispersion


def dispersion_version(mf):
    """The correction `mf` carries, as in 'd4', or None if it carries none."""
    xc = getattr(mf, 'xc', None)
    if xc is None:
        return None
    return _pyscf_dispersion().parse_disp(xc)[1] or None


def dispersion_energy(mf):
    """The dispersion energy of `mf` in Hartree, or 0 if it carries none.

    pyscf already adds this inside `mf.e_tot`; it is exposed separately so a
    surface can report what the term contributes, and so a test can assert
    that an excitation energy does not move when it is switched on.
    """
    version = dispersion_version(mf)
    if version is None:
        return 0.0
    try:
        return float(_pyscf_dispersion().get_dispersion(mf, disp=version))
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_MISSING.format(version=version)) from exc


def dispersion_gradient(mf):
    """d E_disp / dR in Hartree/Bohr, (natm, 3), or zeros if none is carried.

    pyscf adds this inside its own `Gradients().kernel()` for a functional
    whose name carries the correction.
    """
    version = dispersion_version(mf)
    natm = mf.mol.natm
    if version is None:
        return np.zeros((natm, 3))
    try:
        from pyscf.grad import dispersion as grad_dispersion
        grad = grad_dispersion.get_dispersion(mf.Gradients(), disp=version)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_MISSING.format(version=version)) from exc
    return np.asarray(grad, float).reshape(natm, 3)


def refuse_dispersion_under_rpa(mf, what):
    """Raise if `mf` carries a dispersion correction, for a caller that adds
    E_c^dRPA to it.

    Direct RPA already contains dispersion, so the empirical term would be the
    second copy. It is worse than a double count in the gradient: the
    Kohn-Sham-to-Hartree-Fock skeleton is a DIFFERENCE of two pyscf gradients
    at one density, and only the Kohn-Sham one carries the dispersion force, so
    the cancellation those two rely on tears and a dispersion force survives
    with the wrong sign.
    """
    version = dispersion_version(mf)
    if version is None:
        return
    raise ValueError(
        f'{what} adds E_c^dRPA to a mean field whose functional carries '
        f'{version!r}. Direct RPA already contains dispersion, so the '
        f'empirical term would be counted twice, and in the gradient it does '
        f'not merely double: the Kohn-Sham-to-Hartree-Fock skeleton '
        f'differences two pyscf gradients at one density and only one of them '
        f'carries the dispersion force. Use the plain functional here, or put '
        f'the excitation on a dispersion-corrected mean field instead of on '
        f'the dRPA ground state.')
