"""Excitation-energy (polarization-propagator) ADC. See
src/SingleReference/ADC/__init__.py for the module map; the entry is
solve.solve_ee_adc, exported here with its result type EEResult.

The export is resolved at first use: importing one ee_* module (the IP/EA
Faddeev routes read ee_utils and ee_r_sigma_df) must not pull in the drivers
behind solve."""

__all__ = ['solve_ee_adc', 'EEResult']


def __getattr__(name):
    if name in __all__:
        from src.SingleReference.ADC.eeADC import solve
        return getattr(solve, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
