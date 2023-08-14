from ..shortwave import shortwave_gain


def test_shortwave_gain() -> None:
    """_"""
    assert shortwave_gain(1, 1, 0.85) == 0.85
