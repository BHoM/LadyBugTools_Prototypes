from enum import Enum, auto

class BuildingForm(Enum):
    """The form of the building to simulate."""

    CUBOID = auto()
    L_SHAPED = auto()
    U_SHAPED = auto()

class TerrainType(Enum):
    """The type of terrain surrounding the building."""

    OCEAN = "Ocean"
    COUNTRY = "Country"
    SUBURBS = "Suburbs"
    URBAN = "Urban"
    CITY = "City"

def typical_context_distance(terrain_type: TerrainType) -> float:
    """Get the distance to contextual geometry surrounding the building.

    Args:
        terrain_type (TerrainType):
            The type of terrain surrounding the building.

    Returns:
        float:
            The typical distance to contextual geometry surrounding the building, in m.
    """

    match terrain_type:
        case TerrainType.OCEAN:
            context_distance = 1500
        case TerrainType.COUNTRY:
            context_distance = 150
        case TerrainType.SUBURBS:
            context_distance = 50
        case TerrainType.URBAN:
            context_distance = 40
        case TerrainType.CITY:
            context_distance = 30
        case _:
            raise ValueError(
                f"No default context height is available for {terrain_type}."
            )
    return context_distance

class Form:
"""The shape of the building."""

    def __init__(
        self,
        gross_floor_area: float,
        floors: int,
        floor_to_floor_height: float,
        glazing_ratio: list[float],
        skylight_ratio: float = 0,
        building_form: BuildingForm = BuildingForm.CUBOID,
        aspect_ratio: float = 1,
        rotation: float = 0,
        terrain_type: TerrainType = TerrainType.URBAN,
    ):
        self.gross_floor_area = gross_floor_area
        self.floors = floors
        self.floor_to_floor_height = floor_to_floor_height
        self.glazing_ratio = glazing_ratio
        self.skylight_ratio = skylight_ratio
        self.building_form = building_form
        self.aspect_ratio = aspect_ratio
        self.rotation = rotation
        self.terrain_type = terrain_type

        