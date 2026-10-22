"""The TADF rates of one emitter: the photophysics package's command line.

    python tools/photophysics/run_tadf.py --system formaldehyde \
        --atom 'C 0 0 -0.5296; O 0 0 0.6742; H 0 0.9429 -1.1123; H 0 -0.9429 -1.1123' \
        --basis cc-pvdz --environment gas --temperature 300 --lambda-env 0.05 \
        --nudge 0.02 --no-far-start --refreeze 1 --saddle-check 0 --df-check 0 \
        --hessian analytic-df --relaxed-solvation equilibrium --span full \
        --manifold-gaps stacked \
        --out tadf

It is `python -m src.photophysics`, from any directory: the repository root
is put on the path relative to this file. Every option is the package's
(`src/photophysics/cli.py`, `--help`); a relaunch with the same options
reuses what the store holds.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.photophysics.launch import set_threads, threads_in  # noqa: E402

set_threads(threads_in(sys.argv[1:]))

# imported after the threads are set: numpy sizes its pools at import
from src.photophysics.cli import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
