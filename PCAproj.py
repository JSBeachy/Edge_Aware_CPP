import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt



mesh = o3d.io.read_triangle_mesh("plane_segments\plane_segment_8_mesh.stl")
mesh.compute_adjacency_list()
mesh.compute_triangle_normals()
mesh.compute_vertex_normals()

PCApoints=np.asarray(mesh.vertices)
mean=np.mean(PCApoints, axis=0)
centered_points=PCApoints-mean

trueCov=(centered_points.T@centered_points)
normalizer=1/(len(PCApoints)-1)
cov_matrix=np.cov(centered_points.T)
PCA_eigenvals,PCA_eigenvecs=np.linalg.eig(cov_matrix)
print(PCA_eigenvecs)
PCA_eigenvecs2D=PCA_eigenvecs[:,:2]


PCApoints2D=PCApoints@PCA_eigenvecs2D
print(PCApoints)
print(PCApoints2D)