"""The keyword options of the downfolding, matched by name: `choice` takes
either spelling of a listed option (case, '-' or '_') and refuses any other,
naming the valid ones."""


def choice(value, choices, keyword):
    """The entry of `choices` that `value` names, refused unless there is one.

    Both are compared normalized (lower case, '_' -> '-', 'imag-frequency'
    -> 'imagfrequency'), so either spelling of a choice is accepted, and the
    entry is returned as `choices` lists it ('crpa_ph' for 'CRPA-PH').
    """
    def norm(s):
        name = str(s).lower().replace('_', '-')
        return {'imag-frequency': 'imagfrequency'}.get(name, name)
    listed = {norm(c): c for c in choices}
    if norm(value) not in listed:
        raise ValueError(f"Unknown {keyword}={value!r}; choose one of "
                         + ', '.join(repr(c) for c in choices) + '.')
    return listed[norm(value)]
