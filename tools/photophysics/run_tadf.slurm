#!/bin/bash
# One emitter on one node: 8 MPI ranks x 16 threads, every stage in one
# launch; a resubmission resumes from the store. Adjust the partition, the
# environment activation and the paths to the cluster. The log goes where the
# sbatch line puts it:
#     sbatch -o /path/to/campaign/logs/tadf_%j.out \
#         --export=ALL,REPO=...,XYZ=...,OUT=... run_tadf.slurm
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH --exclusive
#SBATCH --time 1-00:00:00
#SBATCH -J tadf
set -euo pipefail
REPO=${REPO:?the repository root}
XYZ=${XYZ:?the emitter geometry, xyz in Angstrom}
OUT=${OUT:?the store root}
# MPICH and Intel MPI need the PMI2 plugin, or srun starts eight serial copies
srun --mpi=pmi2 python "$REPO/tools/photophysics/run_tadf.py" \
    --xyz "$XYZ" --basis cc-pvdz --environment toluene \
    --temperature 300 --lambda-env 0.05 --nudge 0.02 --no-far-start \
    --refreeze 1 --saddle-check 1 --df-check 1 --hessian analytic-df \
    --relaxed-solvation equilibrium --span full --manifold-gaps stacked \
    --distribute --threads 16 --out "$OUT"
