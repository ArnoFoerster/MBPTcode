"""Code generators: derive equations with pdaggerq (or a Goldstone-diagram
engine) and emit NumPy/einsum modules.

pdaggerq is an optional dependency (see src/Base/pdaggerq_interface.py). The
only calculations that import this package are the order-5 diagram
evaluators (src/SingleReference/ADC/delta5_*), which enumerate Goldstone
skeletons at run time and need no pdaggerq. Generated ADC modules go to
src/SingleReference/ADC/generatedADC/ (GenerateRuntime in generate_adc), which
is not version controlled;
the MPn-density, CC and QDPT generators reproduce modules that are.
"""
