"""`python -m src.photophysics ...`: the command line (`cli`), its threads
set before numpy loads (`launch`)."""
import sys

from src.photophysics.launch import set_threads, threads_in

set_threads(threads_in(sys.argv[1:]))

# imported after the threads are set: numpy sizes its pools at import
from src.photophysics.cli import main  # noqa: E402

sys.exit(main())
