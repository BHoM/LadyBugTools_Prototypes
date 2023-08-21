from honeybee_energy.boundarycondition import Adiabatic
from honeybee.boundarycondition import Outdoors
from honeybee.model import Model
from honeybee.face import Face
from honeybee.facetype import Floor, Wall
from honeybee.room import Room
from dataclasses import dataclass, field

#for now one file-maybe seperate if it gets longer

@dataclass
class BoxModelGlazing:
  glazing_ratio: float =field(init=True)
  sill_height: float =field(init=True)
  window_height: float=field(init=True)
  bay_width: float=field(init=True)

@dataclass
class BoxModelRoom:
    bay_width: float = field(init=True)
    bay_count: float = field(init=True)
    depth: float = field(init=True)
    height: float = field(init=True)
    glazing_properties: BoxModelGlazing = field(init=True, default=None)
    identifier: str = field(init=True, default='Test')
    
    def __post_init__(self):
        self._width = self.bay_width * self.bay_count
        self.room = Room.from_box(
            identifier=self.identifier,
            width=self._width,
            depth=self.depth,
            height=self.height
        )
        self.assign_boundary_conditions()
        self.assign_glazing_params()
    
    def assign_glazing_params(self):
        if self.glazing_properties is not None:
            self.room.faces[1].apertures_by_ratio_rectangle(
                ratio=self.glazing_properties.glazing_ratio,
                aperture_height=self.glazing_properties.window_height,
                sill_height=self.glazing_properties.sill_height,
                horizontal_separation=self.glazing_properties.bay_width
            )

    def assign_boundary_conditions(self):
        for face in self.room.faces:
            face.boundary_condition = Adiabatic()
        self.room.faces[1].boundary_condition = Outdoors()

    def get_honeybee_room(self):
        return self.room

@dataclass
class BoxModelModel:
    '''generates model from room'''
    room: Room= field(init=True)
    model: Model= field(init=False)

    def generate_honeybee_model(self)->Model:
      model = Model(identifier=self.room.identifier, rooms = [self.room])
      return model
