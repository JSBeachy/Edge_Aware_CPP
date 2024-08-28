import open3d as o3d
import numpy as np

# Load a TriangleMesh and convert to .t geometry
tensor_mesh = o3d.t.io.read_triangle_mesh('plane_segments\plane_segment_8_mesh.stl')
legacy_mesh_converted=tensor_mesh.to_legacy()

# Extract vertices as a point cloud
vertices = tensor_mesh.vertex["positions"].numpy() 
point_cloud = o3d.geometry.PointCloud()  
point_cloud.points = o3d.utility.Vector3dVector(vertices)
o3d.visualization.draw_geometries([legacy_mesh_converted,point_cloud])

# Create a scene and add the triangle mesh
scene = o3d.t.geometry.RaycastingScene()
tensor_id = scene.add_triangles(tensor_mesh)


#Creating the rays

# We create two rays:
# The first ray starts at (0.5,0.5,10) and has direction (0,0,-1).
# The second ray start at (-1,-1,-1) and has direction (0,0,-1).
rays = o3d.core.Tensor([[0.5, 0.5, 10, 0, 0, -1], [-1, -1, -1, 0, 0, -1]],
                       dtype=o3d.core.Dtype.Float32)

ans = scene.cast_rays(rays)






print(ans.keys())
print(ans['t_hit'].numpy(), ans['geometry_ids'].numpy())

# Visualize the box mesh
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(tensor_mesh.to_legacy())
#vis.run()
#vis.destroy_window()


# Create and visualize rays
raynp=rays.numpy()
print(raynp)
for ray in raynp:
    origin = ray[:3]
    direction = ray[3:]
    line_points = [origin, origin + direction * 10]  # Extend the ray for visualization
    print(line_points)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    line_set.paint_uniform_color([1, 0, 0])  # Red color for the rays
    vis.add_geometry(line_set)

# Render the scene
vis.run()
vis.destroy_window()
