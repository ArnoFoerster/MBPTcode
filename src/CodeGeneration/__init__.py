"""Code generators: derive equations with pdaggerq (or a Goldstone-diagram
engine) and emit NumPy/einsum modules.

pdaggerq is an optional dependency (see src/CodeGeneration/pdaggerq_interface.py); no
calculation imports this package. Generated ADC modules go to
src/SingleReference/ADC/generatedADC/ (GenerateRuntime in generate_adc), which
is not version controlled; the MPn-density, CC and QDPT generators reproduce
modules that are (tests/test_generated_committed.py). codegen_common holds what
the generators share: the module writer and header, the index letters, the
spin-blocked leaf lists and the density blocks.
"""
