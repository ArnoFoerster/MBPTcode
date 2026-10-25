"""Skip marks for tests of the generated ADC runtime.

Generated modules are never committed (src.SingleReference.ADC.generated_modules):
a machine writes them with GenerateRuntime, which needs pdaggerq. A test that
reads them is skipped where they are absent (CI), never failed.

    pytestmark = requires_runtime(4)      # every ADC(2)-(4) runtime module

The MP amplitude numerators of order 4 and above are DensityMatrix modules that
GenerateRuntime writes with the next ADC order (orders=(..., 5) for order 4):

    @requires_mp_numerators(4)
"""
import importlib.util

import pytest

from src.SingleReference.ADC.generated_modules import RUNTIME_MODULES, is_generated


def missing_runtime(order):
    """Names of the runtime modules of ADC order <= `order` not generated here."""
    return [m for m, (n, _) in RUNTIME_MODULES.items() if n <= order and not is_generated(m)]


def requires_runtime(order):
    """skipif mark: some ADC(<=order) runtime module is not generated here."""
    missing = missing_runtime(order)
    return pytest.mark.skipif(
        bool(missing),
        reason=(f"{len(missing)} generated ADC(<={order}) runtime modules missing (e.g. "
                f"{missing[0] if missing else ''}); write them with "
                f"src.CodeGeneration.generate_adc.GenerateRuntime(orders=(2, ..., {order}))"))


def requires_mp_numerators(order):
    """skipif mark: the MP amplitude numerators of `order` are not generated here."""
    name = f'src.SingleReference.DensityMatrix.generated_mpn.mpn_amp_numerators_o{order}'
    return pytest.mark.skipif(
        importlib.util.find_spec(name) is None,
        reason=(f"MP order-{order} amplitude numerators ({name}) not generated; write "
                f"them with src.CodeGeneration.generate_adc.GenerateRuntime(orders=(2, "
                f"..., {order + 1}))"))
