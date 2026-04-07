import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import matplotlib.pyplot as plt

mesh_file= r"plane_segments\Airfoil_surface.stl"
mesh = o3d.io.read_triangle_mesh(mesh_file)
print(f"Mesh loaded: {len(mesh.triangles)} triangles")
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


#### K-means Clustering Code ######
k_clusters=7

t_mesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
t_mesh.compute_triangle_normals()
normals=t_mesh.triangle.normals.cpu().numpy()

#Create k_clusters from 
triangle_colors = o3d.core.Tensor(np.random.rand(len(normals), 3), dtype=o3d.core.float32)
t_mesh.triangle.colors=triangle_colors
#o3d.visualization.draw([t_mesh])
kmeans= KMeans(n_clusters=k_clusters, random_state=42, n_init="auto")
cluster_labels = kmeans.fit_predict(normals)
print(f"Mesh loaded: {len(normals)} triangles")
print(f"Clustering complete for {k_clusters} segments")

cmap = cm.get_cmap('Spectral', k_clusters)
cluster_colors = np.array([cmap(i)[:3] for i in range(k_clusters)], dtype=np.float32)
face_colors_np = cluster_colors[cluster_labels]

print(face_colors_np[0])
face_colors_tensor = o3d.core.Tensor(face_colors_np)
t_mesh.triangle.colors = face_colors_tensor

o3d.visualization.draw([t_mesh])
