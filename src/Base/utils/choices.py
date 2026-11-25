"""A keyword option checked against the values it may take.

`require_choice` accepts either spelling of a listed value, as
`calc_qp_energy` does for its mode: case is ignored, '_' and '-' are the same
character, and the names in ALIASES stand for the value they map to.
Anything else is refused with the list of valid values.
"""

#: Spellings that name another value (after lower case and '_' -> '-').
#: 'imag-frequency' is the imaginary-frequency route's name in calc_qp_energy
#: (src/SingleReference/GW/qp_energy.py), accepted for 'imagfrequency'.
ALIASES = {'imag-frequency': 'imagfrequency'}


def require_choice(value, choices, keyword):
    """The entry of `choices` that `value` names, refused unless there is one.

    Both are compared normalized (lower case, '_' -> '-', ALIASES), so either
    spelling of a choice is accepted, and the entry is returned as `choices`
    lists it ('crpa_ph' for 'CRPA-PH'). keyword: the option's name, for the
    message of the refusal (ValueError, naming the valid choices).
    """
    def norm(s):
        name = str(s).lower().replace('_', '-')
        return ALIASES.get(name, name)
    listed = {norm(c): c for c in choices}
    if norm(value) not in listed:
        raise ValueError(f"Unknown {keyword}={value!r}; choose one of "
                         + ', '.join(repr(c) for c in choices) + '.')
    return listed[norm(value)]
