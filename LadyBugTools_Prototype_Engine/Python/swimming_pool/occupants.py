import pandas as pd


def occupant_gain(
    n_people: float,
    gain_per_person: float = 150,
) -> float:
    f"""Calculate the heat gain from people in a body of water.

    Args:
        n_people (float): The number of people in the body of water.
        gain_per_person (float, optional): Heat gain per person in the space in W. Defaults to 150W.

    Returns:
        float: A value representing the heat gain from people in W.
    """
    res = n_people * gain_per_person

    if isinstance(res, pd.Series):
        return res.rename("Q_Occupants (W)")

    return res
