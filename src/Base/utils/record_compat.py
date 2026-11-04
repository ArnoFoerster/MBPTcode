"""How a record written by an earlier form of the code is read: every rule in
one place.

Records and stamps on disk outlive the code that wrote them, and a run resumes
on them. Each rule below names a field an older record lacks, or holds at a
value it always held, and what that means; the readers (`Numerics`, the
photophysics store and stages, the frozen factorization, the solvated driver's
record, the code stamp) take the rule from here and write no rule of their own.
No rule changes what a record means: an older record reads as the current code
would have written the same run.

    Numerics (`Numerics.from_record`, `identity`)
      NUMERICS_LATER_SECTIONS   sections a record may lack: read at their
                                defaults (or NUMERICS_UNRECORDED_SECTIONS),
                                and an identity at the defaults leaves them
                                out
      NUMERICS_ABSENT           fields a record may lack, and their value
      NUMERICS_RETIRED          fields that are no declaration of their own,
                                at the one value every object held
                                (`without_retired`; another value refuses)
      IDENTITY_AT_DEFAULT_OMITTED, IDENTITY_OMITTED_AT
                                fields an identity leaves out at their
                                default, or at one named value
    photophysics stamps (`store.stamp_as_read`, `stages.Stamps`)
      retired_dropped           every scf section of a stamp read through
                                `without_retired`
      stamp_unless              a field stamped only away from the value its
                                absence means (STAGE_STAMP_ABSENT)
    the frozen factorization (`factor_chain.FrozenFactorization.from_record`)
      FACTORIZATION_ABSENT
    the solvated driver's record (`solvated._start_record`)
      SOLVATED_ABSENT, solvated_absent
    the code stamp
      same_code                 a full commit of the old form
"""
from src.Base.numerics.constants import ISDF_DUCHEMIN_BLASE, XC_GRID_LEVEL_UNRECORDED

# ------------------------------------------------------------------ Numerics
#: Sections added after records and identities were first written: a record
#: without one reads its defaults, and one at its defaults leaves the identity
#: what it was before the section existed.
NUMERICS_LATER_SECTIONS = ('evgw', 'qsgw', 'imagfreq', 'adc', 'properties')

#: Later sections whose default is not what a record written without them was
#: computed with: such a record reads these values, the rest at the defaults.
#: The spin-orbit pass was the exact SOMF before the section existed.
NUMERICS_UNRECORDED_SECTIONS = {'properties': {'soc_two_electron': 'somf'}}

#: (section, field) -> the value of a record that names none: the scf
#: section's xc_grid is pyscf's default, which every mean field was built on
#: until the field existed.
NUMERICS_ABSENT = {('scf', 'xc_grid'): XC_GRID_LEVEL_UNRECORDED}

#: Fields that are no declaration of their own, with the one value every
#: object holds and the identity is hashed at; a record naming one at that
#: value is read without it, at any other value refused (the orbital-response
#: solve's tolerances are the scf section's multiplier_* fields).
NUMERICS_RETIRED = (('scf', 'cphf_tol', 1e-9), ('scf', 'cphf_max_cycle', 100),
                    ('scf', 'cphf_refinements', 10))

#: Fields left out of the identity at their default: an identity without one
#: is the identity at that default.
IDENTITY_AT_DEFAULT_OMITTED = (('isdf', 'test_set'), ('isdf', 'pivot_points'),
                               ('solvent', 'loop_force_fraction'),
                               ('scf', 'test_set'), ('scf', 'pivot_points'))

#: (section, field, value): left out of the identity at that value, as the
#: field's absence was before it existed (the Duchemin-Blase fit of a named
#: grid).
IDENTITY_OMITTED_AT = (('isdf', 'test_set', ISDF_DUCHEMIN_BLASE),)


def without_retired(values):
    """`values` ({section: {name: value}}, choices as a record wrote them)
    without the `NUMERICS_RETIRED` fields: one at the value it always held
    is dropped, one at any other value refused. Sections it changes are
    copied; `values` itself is left as it is."""
    out = dict(values)
    for section, name, held in NUMERICS_RETIRED:
        if name not in out.get(section, {}):
            continue
        out[section] = dict(out[section])
        if out[section].pop(name) != held:
            raise ValueError(f'the record sets {section}.{name}, a retired '
                             f'field, to another value than {held!r}: '
                             'the orbital response reads the multiplier_* '
                             'fields; write those')
    return out


def numerics_sections_as_read(record, sections):
    """{section: {field: value}} of a Numerics record for the section names
    `sections`: a later section it lacks reads its
    `NUMERICS_UNRECORDED_SECTIONS` values (else its defaults), a field of
    `NUMERICS_ABSENT` it lacks takes its value, and the retired fields are
    dropped (`without_retired`)."""
    out = {name: dict(record.get(name, NUMERICS_UNRECORDED_SECTIONS.get(name, {})))
           if name in NUMERICS_LATER_SECTIONS
           else dict(record[name]) for name in sections}
    for (section, name), value in NUMERICS_ABSENT.items():
        out[section].setdefault(name, value)
    return without_retired(out)


# ------------------------------------------------------ photophysics stamps
#: A stage stamp's field -> the value its absence means: it is written only
#: away from that value. r0: R0 walked; properties: every property;
#: solvent_scheme: not scheme B (the environment tells gas and scheme A
#: apart); an scf section's own fit (test_set, pivot_points): undeclared.
STAGE_STAMP_ABSENT = {'r0': 'walk', 'properties': 'all', 'solvent_scheme': None,
                      'test_set': None, 'pivot_points': None}


def stamp_unless(stamp, name, value):
    """`stamp` with `name` set to `value`, or without it where `value` is
    what its absence means (`STAGE_STAMP_ABSENT`)."""
    if value == STAGE_STAMP_ABSENT[name]:
        stamp.pop(name, None)
    else:
        stamp[name] = value
    return stamp


def retired_dropped(tree):
    """`tree` (a stamp) with every scf section in it (a stage's, and the
    Hessian stage's inside a later stage's stamp) read through
    `without_retired`."""
    out = {}
    for key, value in tree.items():
        if key == 'scf' and isinstance(value, dict):
            value = without_retired({'scf': value})['scf']
        elif isinstance(value, dict):
            value = retired_dropped(value)
        out[key] = value
    return out


# --------------------------------------------------- the frozen factorization
#: What a factorization record written before a field existed reads it as:
#: the Duchemin-Blase fit on the grid's own points.
FACTORIZATION_ABSENT = {'test_set': ISDF_DUCHEMIN_BLASE, 'pivot_points': None}


# ------------------------------------------------ the solvated driver's record
#: The settings a solvated record may lack, and the value a record that lacks
#: one was made with: the cavity every run used before it was a setting is
#: Bondi radii scaled by 1.2.
SOLVATED_ABSENT = dict(outside_scissor=None, residues=None, xc_grid=None,
                       isdf_test_set=None, pivot_points=None, cavity='bondi-1.2',
                       spin_scaling=None, solvent_arm=None, loop_tolerance='fixed')


def solvated_absent(rec, key):
    """The value of a setting `key` that the solvated record `rec`, lacking
    it, was made with (`SOLVATED_ABSENT`): an ADC(2) record in a solvent ran
    the singlet with its LR term; a key no rule names is '<absent>', which no
    run's setting equals."""
    if key == 'solvent_arm' and rec.get('state_method') == 'adc2' and rec.get('solvent'):
        return 'ss+lr'
    return SOLVATED_ABSENT.get(key, '<absent>')


# ----------------------------------------------------------- the code stamp
def same_code(recorded, current):
    """Whether a record's `code` names the code of this run: equal, or a full
    40-digit commit (the form before edited trees were stamped) that a clean
    `current` short commit abbreviates. A record of that form made on an
    edited tree cannot be told from a clean one; it is accepted with its
    commit."""
    if recorded == current:
        return True
    return (isinstance(recorded, str) and len(recorded) == 40
            and '+' not in current and current != 'unknown'
            and recorded.startswith(current))
