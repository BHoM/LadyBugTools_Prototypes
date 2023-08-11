from warnings import warn

HEAT_GAIN_SWIMMING = 150


def occupant_gain_from_number(
    number_of_people: int,
    gain_per_person: float = HEAT_GAIN_SWIMMING,
) -> float:
    f"""Calculate the heat gain from people in a space.

    Args:
        number_of_people (int): Number of people in the space.
        gain_per_person (float, optional): Heat gain per person in the space in W.
            Defaults to {HEAT_GAIN_SWIMMING} W.

    Returns:
        float: A value representing the heat gain from
            people in W.
    """

    return number_of_people * gain_per_person


def occupant_gain_from_density(
    m2_per_person: float,
    surface_area: float,
    gain_per_person: float = HEAT_GAIN_SWIMMING,
) -> float:
    f"""Calculate the heat gain from people in a space.

    Args:
        m2_per_person (float): The density of people in the space in m2/person.
        surface_area (float): The surface area of the space in m2.
        gain_per_person (float, optional): Heat gain per person in the space in W.
            Defaults to {HEAT_GAIN_SWIMMING} W.

    Returns:
        float: A value representing the heat gain from
            people in W.
    """

    if m2_per_person == 0:
        return surface_area * 0
    return (surface_area / m2_per_person) * gain_per_person


def occupant_gain(
    number_of_people: float = None,
    m2_per_person: float = None,
    surface_area: float = None,
    gain_per_person: float = HEAT_GAIN_SWIMMING,
) -> float:
    f"""Calculate the heat gain from people in a space. This method is
    semi-dynamic in that it will accept either the number of people in the
    space or the density of people in the space and the surface area of the
    space. If the number of people is provided, the density and surface area
    will be ignored. If the density and surface area are provided, the number
    of people will be ignored.

    Args:
        number_of_people (int): Number of people in the space.
        m2_per_person (float): The density of people in the space in m2/person.
        surface_area (float): The surface area of the space in m2.
        gain_per_person (float, optional): Heat gain per person in the space in W.
            Defaults to {HEAT_GAIN_SWIMMING} W.

    Returns:
        float: A value representing the heat gain from
            people in W.
    """
    if number_of_people is not None and m2_per_person is not None:
        raise ValueError(
            "number_of_people and m2_per_person cannot be provided at the same time."
        )

    if number_of_people is not None and surface_area is not None:
        warn(
            "number_of_people and surface_area cannot be provided at the same time. "
            "The surface_area will be ignored and only number_of_people will be "
            "included in the calculation."
        )

    if number_of_people is not None:
        return occupant_gain_from_number(number_of_people, gain_per_person)

    if number_of_people is None and m2_per_person is not None and surface_area is None:
        raise ValueError(
            "If number_of_people is not provided, both m2_per_person and surface_area must be provided."
        )

    if m2_per_person is not None and surface_area is not None:
        return occupant_gain_from_density(m2_per_person, surface_area, gain_per_person)
    else:
        raise ValueError(
            "Either number_of_people or m2_per_person and surface_area must be provided."
        )
