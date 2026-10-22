# TADF rates of one emitter

The staged TADF run is library code, `src/photophysics`; this folder holds the
launchers around it.

    run_tadf.py       the command line (`python -m src.photophysics`), from
                      any directory
    run_tadf.slurm    one emitter on one node of a SLURM cluster, eight MPI
                      ranks of sixteen threads

A run is a `Declaration`: the molecule, the functional, the environment, the
one `Numerics` object, and every procedural policy. The policies (start nudge,
far start, refreeze, saddle check, ISDF-vs-DF acceptance check, Hessian route,
relaxed solvation, the span cut, temperature, lambda_env, the gap rule of the rate
table's state manifold) have no default; the command line
refuses a run until each is given.

The smallest run, formaldehyde in the gas phase on a single machine:

    python tools/photophysics/run_tadf.py --system formaldehyde \
        --atom 'C 0 0 -0.5296; O 0 0 0.6742; H 0 0.9429 -1.1123; H 0 -0.9429 -1.1123' \
        --basis cc-pvdz --environment gas --temperature 300 --lambda-env 0.05 \
        --nudge 0.02 --no-far-start --refreeze 1 --saddle-check 0 --df-check 0 \
        --hessian analytic-df --relaxed-solvation equilibrium --span full \
        --manifold-gaps stacked \
        --out tadf

The records land in `tadf/formaldehyde/gas/`: `s0`, `hess`, `vert`, `s1`,
`t1` and `rate` (JSON beside an .npz of its arrays), each stamped with what it
is a function of and refused under another declaration. A killed run resumes
from them, inside a walk from its last accepted step.

The UISC protocol of Samanta et al. is the same run with `--rate-model uisc`,
whose stages add `vib` before `rate`.

On a SLURM cluster, `run_tadf.slurm` runs one emitter on one node; give its
log an absolute path on the sbatch line:

    sbatch -o /path/to/campaign/logs/tadf_%j.out \
        --export=ALL,REPO=$PWD,XYZ=/path/to/emitter.xyz,OUT=/path/to/store \
        tools/photophysics/run_tadf.slurm
