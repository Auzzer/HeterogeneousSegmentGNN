import numpy as np          # 用于数组操作（如mask.astype(np.uint16)）
import nibabel as nib       # 读取医学影像NIfTI文件（如nib.load）
import pygalmesh            # 生成四面体网格（generate_from_array）
import meshio               # 网格文件的读写（如meshio.xdmf.write）
from pathlib import Path    # 处理文件路径（如case.mesh_dir.mkdir）

# 以下为项目自定义模块（假设在项目代码中定义）
from project.meshing import (
    remove_unused_points,   # 清理未使用的网格顶点
    load_mesh_fenics        # 加载网格到FEniCS
)
def generate_anatomical_mesh(case, phase, mask_roi, mesh_version):
    
    mask_file = case.totalseg_mask_file(phase, mask_roi) # this is a file path
    mask_nifti = nib.load(mask_file)
    mask = mask_nifti.get_fdata()
    resolution = mask_nifti.header.get_zooms()

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