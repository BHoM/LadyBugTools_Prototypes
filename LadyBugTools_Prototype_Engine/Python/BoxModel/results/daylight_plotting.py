import os
import zipfile
from ..results.daylight_plotter import (
build_custom_continuous_cmap,
vertices_from_grids,
add_starting_vertices_to_end,
vertices_to_patches,
flatten
)
from dataclasses import dataclass
from ladybug.color import Colorset
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

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
    metric: dict
    grids: list

    def __post_init__(self):
        self.patches = self._generate_patches()
        self.cmap = self._generate_colormap()

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

    def generate_fig(self):
        p = PatchCollection(self.patches, cmap=self.cmap, alpha=1)
        p.set_array(self.metric['results'])

        fig, ax = plt.subplots()
        ax.add_collection(p)
        ax.autoscale(True)
        ax.axis('equal')
        ax.axis('off')
        p.set_clim([0, 100])
        return p, fig

    def save_fig(self, output_image_folder):
        metric_name = self.metric['name'].replace(' ', '_')
        image_filepath = os.path.join(output_image_folder, f'{metric_name}.png')
        plt.savefig(image_filepath, bbox_inches='tight', dpi=500)
        return image_filepath
        
