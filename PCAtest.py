import open3d as o3d
import numpy as np

plane=o3d.io.read_triangle_mesh('plane_segments\plane_segment_8_mesh.stl')
#o3d.visualization.draw_geometries([plane])
bounding_box=plane.get_oriented_bounding_box()
#Calculate PCA manually
PCApoints=np.asarray(plane.vertices)
mean=np.mean(PCApoints, axis=0)
centered_points=PCApoints-mean
cov_matrix=np.cov(centered_points.T)
eigenvalues,eigenvectors=np.linalg.eig(cov_matrix)

x=min(eigenvalues)
print(x//1)


x=[1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0]
print(list(enumerate(x)))
indexes = [i for i, j in enumerate(x) if j == 1]
print(indexes)