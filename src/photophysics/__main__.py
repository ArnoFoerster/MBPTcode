"""`python -m src.photophysics ...`: the command line (`cli`), its threads
set before numpy loads (`launch`); `python -m src.photophysics solvated ...`
is the solvated absorption/emission/adiabatic run (`solvated`), on the
environment's threads."""
import sys

if sys.argv[1:2] == ['solvated']:
    from src.photophysics.solvated import main as solvated
    sys.exit(solvated(sys.argv[2:]))

from src.photophysics.launch import set_threads, threads_in  # noqa: E402

set_threads(threads_in(sys.argv[1:]))

# imported after the threads are set: numpy sizes its pools at import
from src.photophysics.cli import main  # noqa: E402

sys.exit(main())
