"""Epstein-Nesbet denominator partitioning (Jiang & Engel, JCP 125, 184108 (2006)).

Method-agnostic: the shifts dress an MBPT doubles denominator, so they are shared by the
ADC solvers and the MPn density-matrix drivers rather than owned by either.
"""
from src.SingleReference.EpsteinNesbet.shifts import (
    EN_SPIN_WEIGHTINGS,
    epstein_nesbet_shift,
    epstein_nesbet_denominator,
    epstein_nesbet_shift_restricted_spinadapted,
    epstein_nesbet_shift_restricted_spinresolved,
)
from src.SingleReference.EpsteinNesbet.denominators import (
    EpsteinNesbetDenominators,
    restricted_channel_shifts,
    build_dressed_e_ai,
    build_dressed_e_abij,
    build_dressed_denoms_uhf,
    coulomb_exchange_diagonals_uhf_df,
    dressed_doubles_block,
)

__all__ = [
    'EN_SPIN_WEIGHTINGS',
    'epstein_nesbet_shift',
    'epstein_nesbet_denominator',
    'epstein_nesbet_shift_restricted_spinadapted',
    'epstein_nesbet_shift_restricted_spinresolved',
    'EpsteinNesbetDenominators',
    'restricted_channel_shifts',
    'build_dressed_e_ai',
    'build_dressed_e_abij',
    'build_dressed_denoms_uhf',
    'coulomb_exchange_diagonals_uhf_df',
    'dressed_doubles_block',
]


EN_DRESS_KEYS = frozenset({'hh', 'pp', 'hp', 'singles', 'spin_adapted', 'shift'})
EN_CHANNELS = ('hh', 'pp', 'hp')


def validate_en_dress(en_dress, keys=EN_DRESS_KEYS):
    """Normalize + validate an Epstein-Nesbet dressing spec: True -> the
    hole-hole + particle-particle default {'hh': True, 'pp': True}, the same
    in every ADC (IP/EA and EE); dicts are checked against the full key
    vocabulary and the EN_SPIN_WEIGHTINGS shift tokens so a typo can never
    silently no-op, and a dict that switches no channel on is refused (pass
    None for Moller-Plesset denominators). keys: the subset of EN_DRESS_KEYS
    the calling method implements (EE-ADC dresses the doubles determinant-wise
    only); any other key is refused by name. Returns the normalized dict (or
    None)."""
    if en_dress is None or en_dress is False:
        return None
    if en_dress is True:
        return {'hh': True, 'pp': True}
    if not isinstance(en_dress, dict):
        raise TypeError(f"en_dress must be True or a dict, got {type(en_dress)}")
    unknown = set(en_dress) - set(keys)
    if unknown:
        raise ValueError(f"en_dress: unknown key(s) {sorted(unknown)}; "
                         f"valid keys are {sorted(keys)}")
    shift = en_dress.get('shift', 'mean')
    if shift not in EN_SPIN_WEIGHTINGS:
        raise ValueError(f"en_dress['shift']={shift!r}; expected one of "
                         f"{EN_SPIN_WEIGHTINGS}")
    singles = en_dress.get('singles', True)
    if singles not in (True, False, 'screened'):
        raise ValueError(f"en_dress['singles']={singles!r}; expected True, False "
                         f"or 'screened' (static-RPA-screened e-h direct term, "
                         f"see build_dressed_e_ai)")
    on = [en_dress.get(c, False) for c in EN_CHANNELS]
    if 'singles' in keys:
        on.append(singles)
    if not any(on):
        raise ValueError(f"en_dress={en_dress!r} switches no channel on; pass "
                         "None for Moller-Plesset denominators")
    return dict(en_dress)
