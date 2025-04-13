import numpy as np          
import nibabel as nib       
import pygalmesh            
import meshio               
from pathlib import Path    


"""from project.meshing import (
    remove_unused_points,   
    load_mesh_fenics        
)"""


def generate_anatomical_mesh(case, phase, mask_roi, mesh_version):
    
    mask_file = case.totalseg_mask_file(phase, mask_roi) # this is a file path
    mask_nifti = nib.load(mask_file)
    mask = mask_nifti.get_fdata()
    resolution = mask_nifti.header.get_zooms()
    # mask output from segmentation model have the value.
    # create the mesh from the mask
    print('Generating mesh with pygalmesh')
    mesh = pygalmesh.generate_from_array(
        mask.astype(np.uint16), # segmentation mask with multiple regions (e.g. values 0,1,3,4,5,6,7)
        voxel_size=resolution,
        max_cell_circumradius={ # this controls the size of the tetras in each region
            'default': 10.0,
            6: 5.0, # airways
            7: 2.0, # vessels
        },
        max_facet_distance=1.5, # important quality parameter, lower = more facets/better resolution
        lloyd=True, # lloyd and odt are two post processing algorithms that improve the mesh
        odt=True
    )
    print('Postprocessing mesh')
    mesh = project.meshing.remove_unused_points(mesh)

    tetra_cells = mesh.get_cells_type('tetra')
    mesh.cells = [meshio.CellBlock('tetra', tetra_cells)]
    
    case.mesh_dir.mkdir(exist_ok=True)
    mesh_file = case.mesh_file(phase, mask_roi, mesh_version)
    print(f'Saving {mesh_file}')
    meshio.xdmf.write(mesh_file, mesh)

    mesh, cell_labels = project.meshing.load_mesh_fenics(mesh_file)  
    return mesh

generate_anatomical_mesh(case, phase, mask_roi='lung_regions2', mesh_version=11)

"""

find_unique_elements(mesh.cell_data['c_labels'][0])
[4, 7, 1, 3, 5, 2, 6]

points = mesh.points                        # shape (9471, 3)
tetra = mesh.cells_dict["tetra"]            # shape (48024, 4)
tetra_labels = mesh.cell_data["c_labels"][0]  # shape (48024,)
"""