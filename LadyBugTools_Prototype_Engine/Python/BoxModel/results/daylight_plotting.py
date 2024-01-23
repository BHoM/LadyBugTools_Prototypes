from codecs import xmlcharrefreplace_errors
import os
import zipfile
from ..results.daylight_plotter import (
build_custom_continuous_cmap,
vertices_from_grids,
add_starting_vertices_to_end,
vertices_to_patches,
flatten,
wireframe_rectangles,
wireframe_windows,
)
from dataclasses import dataclass
from ladybug.color import Colorset
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
from honeybee.model import Model

def generate_zip(image_paths, zip_filename):
    '''takes a list of image paths and creates a zipfile with the desired filename'''
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            zipf.write(image_path, image_name)
    with open(zip_filename, "rb") as zip_file:
        return zip_file.read()

@dataclass
class DaylightPlot:
    model: Model
    metric: dict
    grids: list
    lowerLegend: int
    upperLegend: int
    legendOn: bool
    wireframeOn: bool


    def __post_init__(self):
        self.patches = self._generate_patches()
        self.cmap = self._generate_colormap()
        self.rectangles, self.window_reveals, self.inner_wall = self._build_rectangle()

    def _generate_patches(self):
        mesh_vertices = vertices_from_grids(self.grids)
        patch_vertices = []

        for grid in mesh_vertices:
            repeated_vertices = add_starting_vertices_to_end(grid)
            patch_vertices.append(repeated_vertices)
            
        patches_per_grid = vertices_to_patches(patch_vertices)
        patches = flatten(patches_per_grid)
        return patches
    
    def _generate_colormap(self):
        color_set=Colorset()._colors
        index= self.metric['color_index']
        rgb=color_set[index]
        cmap= build_custom_continuous_cmap(rgb)
        return cmap

    def _build_rectangle(self):
        anchor_point, width, depth = wireframe_rectangles(self.model)
        window_point, window_width, window_depth, inner_point, is_array = wireframe_windows(self.model)
        if is_array:
            orientation=[0, 45, 90, 135, 180, 225, 270, 315]
        else:
            orientation=[0]
        rectangles=[]
        window_reveals=[]
        inner_walls=[]
        for angle in orientation:
            rectangle = Rectangle(xy=anchor_point, 
                                  width=width, 
                                  height=depth, 
                                  angle=angle, 
                                  rotation_point=(width/2,-depth/2),
                                  fill=False, 
                                  linewidth=-0.5, 
                                  edgecolor="black",
                                  )
            for (x,y) in window_point:
                window = Rectangle(xy=(x,y), 
                                   width=window_width, 
                                   height=window_depth, 
                                   angle=angle, 
                                   rotation_point=(width/2,-depth/2), 
                                   fill=True,
                                   facecolor="white",
                                   linewidth=-0.5, 
                                   edgecolor="black",
                                   )
                window_reveals.append(window)
            inner_wall = Rectangle(xy=inner_point,
                                   width=-width,
                                   height=-window_depth,
                                   angle=angle,
                                   rotation_point=(width/2,-depth/2),
                                   fill=True,
                                   facecolor="black",
                                   linewidth=-0.5,
                                   edgecolor="black",
                                   )
            inner_walls.append(inner_wall)
            rectangles.append(rectangle)
        return rectangles, window_reveals, inner_walls

    def generate_fig(self):
        p = PatchCollection(self.patches, cmap=self.cmap, alpha=1)
        fig, ax = plt.subplots()
        
        results = []
        for result in self.metric['results']:
            results.append(result)
        p.set_array(flatten(results))
        ax.add_collection(p)
        
        if self.wireframeOn:
            [ax.add_patch(rect) for rect in self.rectangles]
            [ax.add_patch(wall) for wall in self.inner_wall]
            [ax.add_patch(wind) for wind in self.window_reveals]
                   
        if self.legendOn:
            colorbar = fig.colorbar(p,pad=-0.5)
            colorbar.ax.set_title(self.metric['shortened'])
        ax.autoscale(True)
        ax.axis('off')
        #ax.legend()
        plt.axis('square')

        p.set_clim([self.lowerLegend, self.upperLegend])
        #plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)
        return p, fig

    def save_fig(self, output_image_folder):
        metric_name = self.metric['name'].replace(' ', '_')
        image_filepath = os.path.join(output_image_folder, f'{metric_name}.png')
        plt.savefig(image_filepath, dpi=500, bbox_inches='tight', transparent=True)
        return image_filepath
        
