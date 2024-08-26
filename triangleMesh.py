import open3d as o3d
import numpy as np

# Function to create a plane mesh using o3d.t.geometry.TriangleMesh
def create_plane(vertices, triangles):
    # Convert numpy arrays to Open3D tensor
    vertices_o3d = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
    triangles_o3d = o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.Int32)
    
    mesh = o3d.t.geometry.TriangleMesh(vertices_o3d, triangles_o3d)
    mesh.compute_vertex_normals()
    return mesh

# Define the vertices of the first plane (in XY plane)
vertices1 = np.array([
    [0, 0, 0],   # Bottom-left
    [1, 0, 0],   # Bottom-right
    [1, 1, 0],   # Top-right
    [0, 1, 0],   # Top-left
])

# Define the triangles of the first plane
triangles = np.array([
    [0, 1, 2],   # First triangle
    [0, 2, 3],   # Second triangle
])

# Create the first plane mesh
mesh1 = create_plane(vertices1, triangles)

# Rotate the vertices around the X-axis by 90 degrees for the second plane
rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz([np.pi / 2, 0, 0])
vertices2 = np.dot(vertices1, rotation_matrix.T)

# Create the second plane mesh
mesh2 = create_plane(vertices2, triangles)

# Perform boolean intersection using o3d.t.geometry
mesh_intersection = mesh1.boolean_intersection(mesh2)

# Convert the intersection result to legacy geometry for visualization
mesh_intersection_legacy = mesh_intersection.to_legacy()

# Visualize the original meshes and the intersection
o3d.visualization.draw_geometries([
    mesh1.to_legacy(), 
    mesh2.to_legacy(), 
    mesh_intersection_legacy
])

# Check if the intersection is not empty
if mesh_intersection_legacy.has_vertices():
    print("Intersection detected.")
else:
    print("No intersection detected.")
