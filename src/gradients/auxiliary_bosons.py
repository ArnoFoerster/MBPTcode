"""Re-export of the auxiliary-boson (AB-G0W0) expansion, which lives in production.

The route is forward, ground-state physics with no gradient content, so it sits
in `src.SingleReference.GW.auxiliary_bosons`. This module keeps the gradient-side
import path pointing at it.
"""
from src.Base.constants import AB_RCOND
from src.SingleReference.GW.auxiliary_bosons import (ab_basis, ab_bosons,
                                                     ab_couplings,
                                                     ab_from_factors,
                                                     exact_bosons,
                                                     exact_from_factors)

__all__ = ['AB_RCOND', 'ab_basis', 'ab_bosons', 'ab_couplings',
           'ab_from_factors', 'exact_bosons', 'exact_from_factors']
