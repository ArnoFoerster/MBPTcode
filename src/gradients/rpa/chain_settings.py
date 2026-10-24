"""The settings of the cubic BSE@GW chain
(`gradients.gwbse.excited_state.ExcitedStateChain`) and of the composed surfaces
over it, read off the one `Numerics` the chain holds.

The physics -- spin, root, kernel, environment, mean field -- are the
chain's own arguments; how the surface is computed is the `Numerics` its
driver built (`gradients.gwbse.surface_factory.excited_kwargs`), held as `chain.settings`
and carried verbatim by `refreeze` and by a checkpoint restore: the grids a
chain resolved (`CarriedGrids`) travel beside it, so the object stays exactly
what was asked and its identity is one across a refreeze. It is validated
where it is built, before any mean field exists.
"""
from dataclasses import dataclass, replace
from typing import Optional

from src.Base.numerics.numerics import rpa_quadrature


@dataclass(frozen=True)
class CarriedGrids:
    """The grids a refrozen or restored chain starts from, as resolved by the
    chain before it: a quadrature found too coarse is never stepped back from,
    and a quadrature that changes size between two geometries is two
    functionals whose energies do not compare.

    ntau_w: the static W's tau count, taken exactly; nfreq_cd: the
    contour-deformation size, a floor the new chain can grow above;
    window: the explicit quasiparticle set in the new geometry's orbital
    indices, None where the window is a half-width or 'all' taken again;
    ntau_rpa, nfreq_rpa: the dRPA energy's tau and frequency counts.
    """
    ntau_w: Optional[int] = None
    nfreq_cd: Optional[int] = None
    window: Optional[object] = None
    ntau_rpa: Optional[int] = None
    nfreq_rpa: Optional[int] = None


def rpa_counts(numerics, carried):
    """(ntau, nfreq) of a dRPA energy: the carried counts where a chain
    before this one resolved them, else `rpa_quadrature` of `numerics`."""
    ntau, nfreq = rpa_quadrature(numerics)
    carried = CarriedGrids() if carried is None else carried
    return (ntau if carried.ntau_rpa is None else carried.ntau_rpa,
            nfreq if carried.nfreq_rpa is None else carried.nfreq_rpa)


def held_numerics(chain):
    """The `Numerics` the chain holds; a chain without one is refused."""
    out = chain.__dict__.get('settings')
    if out is None:
        raise TypeError(f'{type(chain).__name__} holds no Numerics '
                        f'(`settings`): build it with one')
    return out


def with_setting(numerics, field, value):
    """`numerics` with `field` ('section.name') set to `value`, validated as
    the section validates: what a chain's setting is written through."""
    section, key = field.split('.')
    return replace(numerics, **{section: replace(getattr(numerics, section),
                                                 **{key: value})})


def chain_setting(field):
    """A chain attribute that IS one choice of its `Numerics`, at `field`
    ('section.name'): read off `chain.settings`, and written by replacing it
    (`with_setting`), so what the kernels read and what `refreeze`,
    `to_record` and the force identity read cannot be two values."""
    section, key = field.split('.')

    def read(chain):
        return getattr(getattr(held_numerics(chain), section), key)

    def write(chain, value):
        chain.__dict__['settings'] = with_setting(held_numerics(chain), field,
                                                  value)

    return property(read, write, doc=f'`Numerics.{field}`')
