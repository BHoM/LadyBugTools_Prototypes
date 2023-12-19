from abc import abstractproperty
from codecs import backslashreplace_errors
from copyreg import constructor
from warnings import filterwarnings
from honeybee_energy.boundarycondition import Adiabatic
from honeybee.boundarycondition import Outdoors
from honeybee.model import Model
from honeybee.shade import Shade
from honeybee.facetype import Floor
from honeybee.room import Room
from honeybee.face import Face
from ladybug_geometry.geometry3d.face import Face3D ####
from ladybug_geometry.geometry3d.pointvector import Vector3D, Point3D ###
from honeybee_radiance.sensorgrid import SensorGrid
from honeybee_vtk.model import Model as VTKModel, SensorGridOptions
from honeybee_energy.material.glazing import EnergyWindowMaterialGlazing
from honeybee_energy.construction.window import WindowConstruction
from honeybee_energy.lib.constructions import window_construction_by_identifier
from honeybee_energy.lib.materials import window_material_by_identifier
from honeybee_energy.constructionset import ApertureConstructionSet
from honeybee_energy.properties.aperture import ApertureEnergyProperties

from honeybee_radiance.modifier.material import Glass
from honeybee.aperture import Aperture
import math
#import glass material, window constuction & apply window construction)
from dataclasses import dataclass, field

#for now one file-maybe seperate if it gets longer

@dataclass
class BoxModelGlazing:
  glazing_ratio: float 
  sill_height: float 
  window_height: float
  bay_width: float
  wall_thickness: float
  glass_transm: float

@dataclass
class BoxModelRoom:
    bay_width: float 
    bay_count: float 
    depth: float 
    height: float 
    glazing_properties: BoxModelGlazing = field(init=True, default=None)
    identifier: str = field(init=True, default='Test')
    north: int = 180
    
    def __post_init__(self):
        self._width = self.bay_width * self.bay_count
        self.room = Room.from_box(
            identifier=self.identifier,
            width=self._width,
            depth=self.depth,
            height=self.height,
            orientation_angle= self.north
        )
        self._assign_boundary_conditions()
        self._assign_glazing_params()
    
    def _assign_glazing_params(self):                                                  
        if self.glazing_properties is not None:
            self.room.faces[1].apertures_by_ratio_rectangle(
                ratio = self.glazing_properties.glazing_ratio,
                aperture_height = self.glazing_properties.window_height,
                sill_height = self.glazing_properties.sill_height,
                horizontal_separation = self.glazing_properties.bay_width
            )
            modifier = Glass.from_single_transmittance("test", self.glazing_properties.glass_transm)
            
            ## Code for creating an inner shade to simulate wall thickness
            z = self.room.faces[1].punched_geometry
            wallShades = Shade("Offset_Wall", z, is_detached=True)
            wallVector = self.room.faces[1].normal * self.glazing_properties.wall_thickness
            self.room.add_indoor_shade(wallShades)
            self.room.move_shades(wallVector)

            for aperture in self.room.faces[1].apertures:
                aperture.extruded_border(self.glazing_properties.wall_thickness)
                aperture.properties.radiance.modifier = modifier
                
    def _assign_boundary_conditions(self):
        for face in self.room.faces:
            face.boundary_condition = Adiabatic()
        self.room.faces[1].boundary_condition = Outdoors()              

    def get_honeybee_room(self):
        return self.room

@dataclass
class BoxModelModel:
    '''generates model from room'''
    room: Room
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
    model: Model
    grid_size: float
    offset_distance: float= field(default=0.1)
    bayAmount: float= field(default=3)
    
    def setModelRoom(modelRoom):
        boxmodelroom = modelRoom

    def __post_init__(self):
        room= self.model.rooms[0]
        for face in room.faces:
            if face.type ==Floor():
                floor = face.geometry
            
        identifier= 'Test'        

        ## Code for changing the size of the simulation grid
        ## Taking input values
        baywidth = self.boxmodelroom.bay_width
        baycount = 3
        depth = 10 
        orientation = -math.pi/180*30 
        offset = 1.3
        
        ## Creating a face that will be used to create a custom sized sensorgrid for the simulation based on user input.
        ySize = depth-offset
        if (self.bayAmount == 1):
            grid_face3D = Face3D.from_rectangle(baywidth, ySize)
            basepoint = [-2*baywidth, -9.5, 0]
        elif(self.bayAmount == 2):
            grid_face3D = Face3D.from_rectangle(2*baywidth, ySize)
            basepoint = [-5/2*baywidth, -9.5, 0]
        else:
            grid_face3D = Face3D.from_rectangle(baywidth * baycount-1, ySize)
            #basepoint = [-(baycount * baywidth - 0.), -9.5, 0]            
            basepoint = [0.71 * math.cos(-orientation/2), 0.71 * math.sin(orientation/2), 0]
            
        originPoint = Point3D.from_array([0, 0, 0])
        
        ## Moving the new face to the right place in the model
        movingVector = Vector3D.from_array(basepoint)
        newFace3D = grid_face3D.rotate_xy(orientation, originPoint)
        newFace3D = newFace3D.move(movingVector)

        self.sensor_grid = SensorGrid.from_face3d(identifier=identifier,
                                        faces = [newFace3D],
                                        x_dim = self.grid_size,
                                        offset=self.offset_distance,
                                        flip=False)

def BoxModel(glazing_ratio, sill_height, window_height, bay_width, bay_count, room_depth, room_height, north, wall_thickness, glass_transm, grid_size, offset_distance):
     glazing_properties= BoxModelGlazing(glazing_ratio = glazing_ratio,
                                         sill_height = sill_height,
                                         window_height = window_height,
                                         bay_width = bay_width,
                                         wall_thickness = -1*wall_thickness,
                                         glass_transm = glass_transm,
                                         )
     
     room= BoxModelRoom(bay_width = bay_width,
                        bay_count = bay_count,
                        depth = room_depth,
                        height = room_height,
                        glazing_properties = glazing_properties,
                        north = north).get_honeybee_room()
            
     model=BoxModelModel(room).generate_honeybee_model()
     #sensor_grid = BoxModelSensorGrid(model=model, grid_size=grid_size, offset_distance = offset_distance, bay_count=bay_count, orientation=north).sensor_grid
     #model.properties.radiance.add_sensorgrid(sensor_grid)
     return model, room

def BoxModelVTK(glazing_ratio, sill_height, window_height, bay_width, bay_count, room_depth, room_height, north, wall_thickness, glass_transm):
    model, room = BoxModel(glazing_ratio, sill_height, window_height, bay_width, bay_count, room_depth, room_height, north, wall_thickness, glass_transm)
    modelVTK = BoxModelModel(room).generate_VTK_model(model)
    return modelVTK
