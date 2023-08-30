from honeybee_energy.boundarycondition import Adiabatic
from honeybee.boundarycondition import Outdoors
from honeybee.model import Model
from honeybee.facetype import Floor
from honeybee.room import Room
from honeybee_radiance.sensorgrid import SensorGrid
from honeybee_vtk.model import Model as VTKModel, SensorGridOptions
from dataclasses import dataclass, field

#for now one file-maybe seperate if it gets longer

@dataclass
class BoxModelGlazing:
  glazing_ratio: float 
  sill_height: float 
  window_height: float
  bay_width: float

@dataclass
class BoxModelRoom:
    bay_width: float 
    bay_count: float 
    depth: float 
    height: float 
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
        self._assign_boundary_conditions()
        self._assign_glazing_params()
    
    def _assign_glazing_params(self):
        if self.glazing_properties is not None:
            self.room.faces[1].apertures_by_ratio_rectangle(
                ratio=self.glazing_properties.glazing_ratio,
                aperture_height=self.glazing_properties.window_height,
                sill_height=self.glazing_properties.sill_height,
                horizontal_separation=self.glazing_properties.bay_width
            )

    def _assign_boundary_conditions(self):
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
    
    @staticmethod
    def generate_VTK_model(model)-> VTKModel:
        model=VTKModel(hb_model=model, grid_options=SensorGridOptions.Mesh)
        return model


@dataclass
class BoxModelSensorGrid:
    model: Model= field(init= True)
    grid_size: float= field(init=True)

    def __post_init__(self):
        room= self.model.rooms[0]
        for face in room.faces:
            if face.type ==Floor():
                floor = face.geometry
        identifier= 'Test'
        sensor_grid = SensorGrid.from_face3d(identifier=identifier,
                                        faces = [floor],
                                        x_dim = self.grid_size,
                                        offset=0.75,
                                        flip=True)
        
        self.model.properties.radiance.add_sensor_grid(sensor_grid)
