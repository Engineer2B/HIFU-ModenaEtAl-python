import numpy
from stl import mesh


def move_stl(focal, tumor, stl_list, um):
    # Using an existing stl file:
    mesh_list = []
    for i in range(0, len(stl_list)):
        # Creation of the 'mesh' object from the STLs list
        mesh_mesh = mesh.Mesh.from_file(stl_list[i])
        # Creation of a list with all the mesh from stl_list
        mesh_list.append(mesh_mesh)

    # Calculations of the translation to align focal point and tumor centre
    delta_x = focal[0] - tumor[0]
    delta_y = focal[1] - tumor[1]
    delta_z = focal[2] - tumor[2]

    # STL files may have different unit of measure, but te coordinates of focal point and tumor centre are in cm, so
    # we have to be consistent:
    if um == 'mm':
        fact = 10
    elif um == 'm':
        fact = 0.1
    elif um == 'cm':
        fact = 1

    # max_x = max((mesh_list[i].v0[:, 0])/fact)

    # For every mesh in 'mesh_list':
    # - from each triangle vertex (v0, v1, v2) import the x,y,z (0, 1, 2) coordinates
    # - divide by 'fact' (according to the unit measure)
    # - add delta_x,y,z to translate the vertex
    # - eventually, a new STL file is saved from the translated vertices
    for i in range(0, len(stl_list)):
        mesh_list[i].v0[:, 0] = (mesh_list[i].v0[:, 0])/fact + delta_x + 0.3
        mesh_list[i].v0[:, 1] = (mesh_list[i].v0[:, 1])/fact + delta_y
        mesh_list[i].v0[:, 2] = (mesh_list[i].v0[:, 2])/fact + delta_z
        mesh_list[i].v1[:, 0] = (mesh_list[i].v1[:, 0])/fact + delta_x+ 0.3
        mesh_list[i].v1[:, 1] = (mesh_list[i].v1[:, 1])/fact + delta_y
        mesh_list[i].v1[:, 2] = (mesh_list[i].v1[:, 2])/fact + delta_z
        mesh_list[i].v2[:, 0] = (mesh_list[i].v2[:, 0])/fact + delta_x+ 0.3
        mesh_list[i].v2[:, 1] = (mesh_list[i].v2[:, 1])/fact + delta_y
        mesh_list[i].v2[:, 2] = (mesh_list[i].v2[:, 2])/fact + delta_z
        name = 'CT-data_' + str(i) + '.stl'
        mesh_list[i].save(name)

