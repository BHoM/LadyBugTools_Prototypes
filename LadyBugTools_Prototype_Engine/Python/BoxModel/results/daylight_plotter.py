import numpy as np
import matplotlib.colors as colors
from matplotlib.path import Path as MPLPath
from matplotlib.patches import PathPatch

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
