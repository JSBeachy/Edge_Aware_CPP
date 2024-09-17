import open3d as o3d
from collections import defaultdict
import numpy as np


# Load STL mesh
def load_stl(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_adjacency_list()
    return mesh

# Find edges and check if they are on the boundary
def find_boundary_edges(mesh):

    
    edge_to_face_count = defaultdict(int)

    # Extract triangles from the mesh
    triangles = np.asarray(mesh.triangles)

    # Iterate through each triangle and register its edges
    for tri in triangles:
        # A triangle has 3 edges, each defined by a pair of vertices
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]

        for edge in edges:
            print(edge)
            # Sort edge vertices so that (v0, v1) and (v1, v0) are considered the same
            edge = tuple(sorted(edge))
            print(edge)
            print('\n')
            edge_to_face_count[edge] += 1

    # Identify boundary edges (edges belonging to only 1 face)
    boundary_edges = [edge for edge, count in edge_to_face_count.items() if count == 1]

    return boundary_edges

# Create a LineSet object to visualize boundary edges
def create_boundary_lines(mesh, boundary_edges):
    vertices = np.asarray(mesh.vertices)
    print( 'weeeeee' )
    # Create an Open3D LineSet to display edges
    lines = []
    for edge in boundary_edges:
        print(edge[0],edge[1])
        lines.append([edge[0], edge[1]])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(lines)
    )
    
    # Set the color of the boundary edges to red
    colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def main():
    # Path to your STL file
    file_path = 'plane_segments\Plannertest.stl'

    # Load the mesh
    mesh = load_stl(file_path)

    # Find boundary edges
    boundary_edges = find_boundary_edges(mesh)

    # Create LineSet for boundary edges
    boundary_lines = create_boundary_lines(mesh, boundary_edges)

    # Visualize the mesh and boundary edges together
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh, boundary_lines])

if __name__ == "__main__":
    main()

