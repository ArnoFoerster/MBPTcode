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

## Continuous integration

`.github/workflows/tests.yml` runs the suite on GitHub in tiers, listed in
`tools/ci/`. `fast.txt` (files that take under half a minute) runs on every
push and pull request, in four shards, on the versions in `tools/ci/pinned.txt`.
`slow.txt` runs nightly on the newest releases of the dependencies, in six
shards. `heavy.txt` (hour-long runs, real MPI) does not run on GitHub; run those
on the cluster. `triage.txt` lists files that failed when the tiers were drawn
up and have not been looked at yet.

Run one tier on a single machine the same way CI does:

```bash
python tools/ci/run_tier.py fast --shard 1/4
```
