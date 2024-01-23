from xml.dom import INUSE_ATTRIBUTE_ERR
import numpy as np
import matplotlib.colors as colors
from matplotlib.path import Path as MPLPath
from matplotlib.patches import PathPatch
import math

def vertices_from_grids(grids):
    """Takes a list of grids and returns a list of lists, of patches (mesh faces) per grid"""
    grids_faces_vertices = []
    for grid in grids:
        for g in grid:
            mesh = g.mesh
            datatree_faces_vertices = mesh.face_vertices
            faces = []
            for face in datatree_faces_vertices:
                face_vertices = []
                for vertice in face:
                    x , y = vertice.x, vertice.y
                    face_vertices.append([x,y])
                faces.append(face_vertices) 
            grids_faces_vertices.append(faces) 
    return grids_faces_vertices

def add_starting_vertices_to_end(mesh_face_vertices):
    """add the starting vertice of each list to the end"""
    final_vertices = []
    for face in mesh_face_vertices:
        new_vertices = face
        new_vertices.append(face[0])
        final_vertices.append(new_vertices)
    return final_vertices

def vertices_to_patches(mesh_face_vertices: list):
    """take a list of face vertices that have same start and end point"""
    patches_per_grid = []
    for grid in mesh_face_vertices:
        patches = []
        for face in grid:
            path = MPLPath(face)
            patch = PathPatch(path, rasterized = True)
            patches.append(patch)
        patches_per_grid.append(patches)
    return patches_per_grid

def flatten(l):
    return [item for sublist in l for item in sublist]

# modified from sctriangulate on github
def inter_from_256(x):
    return np.interp(x=x,xp=[0,255],fp=[0,1])

def build_custom_continuous_cmap(rgb):
    '''
    Generating any custom continuous colormap, user should supply a list of (R,G,B) color taking the value from [0,255], because this is
    the format the adobe color will output for you. 
    '''
    all_red = []
    all_green = []
    all_blue = []
    for rgb in rgb:
        all_red.append(rgb[0])
        all_green.append(rgb[1])
        all_blue.append(rgb[2])
    # build each section
    n_section = len(all_red) - 1
    red = tuple([(1/n_section*i,inter_from_256(v),inter_from_256(v)) for i,v in enumerate(all_red)])
    green = tuple([(1/n_section*i,inter_from_256(v),inter_from_256(v)) for i,v in enumerate(all_green)])
    blue = tuple([(1/n_section*i,inter_from_256(v),inter_from_256(v)) for i,v in enumerate(all_blue)])
    cdict = {'red':red,'green':green,'blue':blue}
    new_cmap = colors.LinearSegmentedColormap('new_cmap',segmentdata=cdict)
    return new_cmap

def wireframe_rectangles(model):
    ''' Generates dimensions and anchor points for the rectangle representing the walls of the Box Model.'''
    face = model.faces[0].geometry
    point = face.lower_left_corner
    [x, y] = point.x, point.y
    upper_right = face.upper_right_corner
    [x1, y1] = upper_right.x, upper_right.y
    width = x1-x
    depth = y1-y
    
    return (x,y), width, depth

def wireframe_windows(model):   
    ''' Generates dimensions for rectangles representing windows and wall thickness of the Box Model.'''
    anchor_points=[]
    i=0 
    box_model = model
    # if the simulation is done on one model
    if len(box_model.faces) == 6:        
        is_array=False
        amount = len(box_model.apertures) 
    # if the simulation is done as an array
    else:         
        is_array=True
        amount = len(box_model.apertures) / 8
        
    # taking the anchor point for each window
    while i < amount:
        window = box_model.apertures[i]
        face=window.geometry
        point = face.lower_left_corner
        [x, y] = point.x, point.y        
        anchor_points.append((x,y))
        i+=1
        
    # calculating width and depth of the window reveals
    point1 = face.lower_right_corner
    x1 = point1.x
    y = point1.y
    inner_wall = box_model.shades[0].geometry
    inner_point = inner_wall.lower_left_corner
    y1 = inner_point.y
    width = x1-x
    depth = y1-y    
    return anchor_points, width, depth, inner_point, is_array