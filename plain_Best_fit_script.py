import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import itertools
from PCAClass import PCABounding, Best_Fit_CPP


#Load the STL file
#mesh = o3d.io.read_triangle_mesh("plane_segments\Circle_mesh.stl")
#mesh = o3d.io.read_triangle_mesh("plane_segments\Skinny_tall_mesh.stl")
#mesh = o3d.io.read_triangle_mesh("plane_segments\Fat_Short_mesh.stl")
mesh = o3d.io.read_triangle_mesh("plane_segments\plane_segment_8_mesh.stl")

# pre-process mesh
tolerance = 1e-5  # Adjust tolerance as needed
mesh.merge_close_vertices(tolerance)

mesh.compute_adjacency_list()
mesh.compute_triangle_normals()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], point_show_normal=False)

# Find boundary edges and identify boundary verticies
boundary = mesh.get_non_manifold_edges(allow_boundary_edges=False) #Non-manifold defined differently if True
boundary_vertices = np.unique(np.array(boundary).flatten())
boundary_vertices_coords = np.asarray(mesh.vertices)[boundary_vertices]
#print("Boundary Vertices Coordinates:")
#print(segment.boundary_vertices_coords)

# Create a visualization object
boundary_pcd=o3d.geometry.PointCloud()
boundary_pcd.points = o3d.utility.Vector3dVector(boundary_vertices_coords)
boundary_pcd.paint_uniform_color([1,0,0])
o3d.visualization.draw_geometries([mesh, boundary_pcd], point_show_normal=False)
