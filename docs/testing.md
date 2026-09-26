# Tests

Two styles coexist. The ADC/CC/density-matrix/GW-core suite are standalone
scripts that print their own verdict and exit non-zero on failure:

```bash
python tests/test_adc3.py
```

The gradients/properties/environment suite (`src/gradients`, `src/properties`,
`src/Base/{declaration,dispersion,polarizable_sites,composite_environment,
cppe_interface,pcm_factorization,pcm_derivatives}.py` and their
`SingleReference` dependents) is ordinary pytest:

```bash
pytest tests/test_excited_state.py
```

`tests/README.md` maps which tests cover what.
