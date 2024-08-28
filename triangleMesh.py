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


tensor_mesh = o3d.t.io.read_triangle_mesh('plane_segments\plane_segment_8_mesh.stl')
legacy_mesh_converted=tensor_mesh.to_legacy()

# Extract vertices as a point cloud
vertices = tensor_mesh.vertex["positions"].numpy() 
point_cloud = o3d.geometry.PointCloud()  
point_cloud.points = o3d.utility.Vector3dVector(vertices)

o3d.visualization.draw_geometries([legacy_mesh_converted,point_cloud])


print('hello')

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

# Calculate the center of mesh1
center1 = np.mean(vertices1, axis=0)

# Translate mesh2 to rotate around the center of mesh1
vertices2 = vertices1 - center1  # Translate to origin

# Rotate the vertices around the X-axis by 90 degrees
rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz([np.pi / 2, 0, 0])
vertices2 = np.dot(vertices2, rotation_matrix.T)

# Translate vertices2 back to the original center
vertices2 += center1

# Create the second plane mesh
mesh2 = create_plane(vertices2, triangles)


# Perform boolean intersection using o3d.t.geometry
mesh_intersection = mesh1.boolean_intersection(mesh2)

# Convert the intersection result to legacy geometry for visualization
mesh_intersection_legacy = mesh_intersection.to_legacy()
# Color the intersection mesh red
colors = np.array([[1, 0, 0]] * len(mesh_intersection_legacy.vertices))  # Red color (RGB)
mesh_intersection_legacy.vertex_colors = o3d.utility.Vector3dVector(colors)

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
# Disable back-face culling in the visualizer (if necessary)
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh_intersection_legacy)
render_option = vis.get_render_option()
render_option.mesh_show_back_face = True  # Disable back-face culling
vis.run()
vis.destroy_window()


#possibility
'''
# Convert the intersection mesh to an edge mesh to extract the spline
intersection_edges = mesh_intersection_legacy.get_non_manifold_edges()

# You can now visualize the edges or further process them
edge_mesh = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_intersection_legacy)
o3d.visualization.draw_geometries([edge_mesh])
edges=mesh_intersection_legacy.get_non_manifold_edges()
# Create a dictionary to count the number of edges per vertex
vertex_edge_count = {}
print(edges)
for edge in edges:
    for vertex in edge:
        if vertex in vertex_edge_count:
            vertex_edge_count[vertex] += 1
        else:
            vertex_edge_count[vertex] = 1

# Define your threshold
edge_threshold = 1  # Example threshold

# Filter vertices based on the threshold
filtered_vertices = [vertex for vertex, count in vertex_edge_count.items() if count >= edge_threshold]
print(filtered_vertices)
# Get the positions of these vertices
filtered_vertex_positions = np.asarray(mesh_intersection_legacy.vertices)[filtered_vertices]

# Optionally, create a point cloud of the filtered vertices for visualization
filtered_vertex_cloud = o3d.geometry.PointCloud()
filtered_vertex_cloud.points = o3d.utility.Vector3dVector(filtered_vertex_positions)

# Visualize the filtered vertices
o3d.visualization.draw_geometries([filtered_vertex_cloud])
'''