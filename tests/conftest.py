import os

# scipy and pyscf's C libraries link Apple Accelerate on macOS, whose reductions
# follow VECLIB_MAXIMUM_THREADS; the bitwise refactor baseline
# (tests/baseline_3f09ac0.json) was recorded with it at 2 (its `threads`), and
# left unset it runs on every core and moves the last bits (1e-17..1e-10).
# Pinned before any test module imports a BLAS user.
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')

# test_mpi_grid_distribution.py initializes MPI and calls sys.exit at import
# time (it is meant for `mpirun -n 3 python tests/test_mpi_grid_distribution.py`);
# collected in-process it terminates the whole pytest session.
# test_mpi_routes.py is an mpirun script too, whose verdict is its exit status.
collect_ignore = ['test_mpi_grid_distribution.py',
                  'test_mpi_routes.py']
