# Tests

The tests run under pytest, which the package itself does not need:

```bash
pip install pytest
```

Every test file runs as a script, which prints a verdict per check and exits
non-zero if any check failed, and under pytest:

```bash
python tests/test_adc3.py
pytest tests/test_excited_state.py
```

`tests/test_imports.py` is the cheap one to run first: it imports every module
under `src/` in about a second. `tests/README.md` maps which tests cover what.

Run the tests with two threads, as the CI does:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
```

## What the tests hold the code to

- **Layers.** `tests/test_layering.py` reads every module of `src/` and
  `tools/` and fails on an import that points up the layers of
  [Source layout](layout.md), on an import between two method subpackages of
  the gradient layer that must stay apart, and on a private name (one starting
  with `_`) read from another module.
- **Pinned numbers.** Two computations of one number on the same machine may
  differ in the last bits, because a sum taken in another order rounds
  differently. A pin therefore holds a recorded number to 8 units in the last
  place (`PIN_SAME_ULPS` in `tests/pins.py`) on the BLAS library it was
  recorded with, and to a looser bar per kind of quantity (energy, gradient,
  vector) on any other. Exact equality is kept for a value and a stored copy
  of it, such as a checkpoint or a broadcast.

## Continuous integration

`.github/workflows/tests.yml` runs the suite on GitHub in tiers, listed in
`tools/ci/`. `fast.txt` (files that take under about a minute) runs on every
push and pull request, in four shards, on the versions in `tools/ci/pinned.txt`.
`slow.txt` runs nightly on the newest releases of the dependencies, in six
shards. Also nightly, one job builds pdaggerq, writes the generated ADC(2)-(4)
runtime with `GenerateRuntime(orders=(2, 3, 4))` (both cached until a generator
source changes), and four shards run the fast and slow files that read it
(`run_tier.py fast slow --generated`); the rest of the suite skips those tests
where the runtime is absent. The ADC(5) runtime takes hours to write, so its
tests run on a cluster. `heavy.txt` (hour-long runs, real MPI) does not run on
GitHub; run those on a cluster. `triage.txt` lists files that fail for a known
reason, written beside each. Every `tests/test_*.py` is in exactly one list.

Run one tier on a single machine the same way CI does:

```bash
python tools/ci/run_tier.py fast --shard 1/4
python tools/ci/run_tier.py fast slow --generated   # after GenerateRuntime
```

## On a cluster

Intel's MKL may pick a different code path from one run to the next, which
moves results in about the twelfth digit. For runs that should repeat to the
bit, as a test comparing two versions of the code needs, fix the path and
keep one OpenMP thread:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=2 MKL_CBWR=COMPATIBLE
```
