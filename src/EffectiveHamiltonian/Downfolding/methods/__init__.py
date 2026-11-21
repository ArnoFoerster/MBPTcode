"""One module per low-level method of the downfolding, registered here under
the method's name. Each module has `run(mf, **opts)`, which returns the
low-level result (attributes `method` and `mf`), and `terms(low, part,
**opts)`, which returns its `DownfoldingTerms`; the driver reaches them only
through `LOW_LEVEL`."""
from . import hf, rpa

LOW_LEVEL = {'hf': hf, 'rpa': rpa}
