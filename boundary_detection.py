import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import itertools
from PCAClass import PCABounding, Best_Fit_CPP
import time

def linearity_score(vec1,vec2):
    cos_theta= np.dot(vec1,vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return abs(cos_theta)

def fitting(len_my_list, edge_list):
    #use list comprehension and sorting to order the list index
    direction=1
    index=sorted([hull_vertices_list.index(i) for i in edge_list])
    counterclockwise = (index[0] - index[1]) % len_my_list
    clockwise = (index[1] - index[0]) % len_my_list
    length=min(clockwise,counterclockwise)
    if counterclockwise<=clockwise:
        direction=-1
    group=[hull_vertices_list[(index[0] + i*direction)] for i in range(length+1)]
    return np.asarray(group)

# Load the STL file
#mesh = o3d.io.read_triangle_mesh("plane_segments\Circle_mesh.stl")
#mesh = o3d.io.read_triangle_mesh("plane_segments\Skinny_tall_mesh.stl")
#mesh = o3d.io.read_triangle_mesh("plane_segments\Fat_Short_mesh.stl")
mesh = o3d.io.read_triangle_mesh("plane_segments\plane_segment_8_mesh.stl")

segment=Best_Fit_CPP("plane_segments\plane_segment_8_mesh.stl")

# Ensure the mesh has edges and triangle information for visualization
segment.mesh.compute_adjacency_list()
segment.mesh.compute_triangle_normals()
segment.mesh.compute_vertex_normals()

# Find boundary edges and identify boundary verticies
boundary = segment.mesh.get_non_manifold_edges(allow_boundary_edges=False) #Non-manifold defined differently if True
#boundary=segment.boundary_edge_finder()
boundary_vertices = np.unique(np.array(boundary).flatten())
segment.boundary_vertices_coords = np.asarray(segment.mesh.vertices)[boundary_vertices]

# Create a visualization object
boundary_pcd=o3d.geometry.PointCloud()
boundary_pcd.points = o3d.utility.Vector3dVector(segment.boundary_vertices_coords)
boundary_pcd.paint_uniform_color([1,0,0])
#o3d.visualization.draw_geometries([segment.mesh, boundary_pcd], point_show_normal=False)

#Take convex hull and find the verticies
#Find_convex_hull projects into 2D along least PCA axis, and takes hull of 2D points
segment.find_convex_hull(2, segment.boundary_vertices_coords)
#Boundary2D=segment.PCA_projection(2,segment.boundary_vertices_coords)

#Compare linearity of groups of 3 consecutive hull points. 
#Least linear groups are centered on corner point
segment.find_corner_points()

plt.plot(segment.PCA_pointsND[:,0], segment.PCA_pointsND[:,1], 'o', label='Edge points')
plt.plot(segment.hull_verticesND[:,0],segment.hull_verticesND[:,1], "r*",markersize=10,label="Convex Hull")
plt.plot(np.array(segment.corner_points)[:,0],np.array(segment.corner_points)[:,1], 'y*',markersize=20,label='Corner Points')
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
#plt.legend()
#plt.show()
print(segment.primary_axis)

## Determine what "edge" (aka between corners) aligns best with the primary scanning axis
segment.find_primary_scanning_edges()

## Take edge and classify all points between as point to best-fit 
hull_vertices_list=[vertex.tolist() for vertex in segment.hull_vertices]
primary_edge=segment.aligned_edges[0]
secondary_edge=segment.aligned_edges[1]
group1=segment.fitting(len(hull_vertices_list), primary_edge, hull_vertices_list)
group2=segment.fitting(len(hull_vertices_list), secondary_edge, hull_vertices_list)

plt.plot(group1[:,0],group1[:,1], "*",markersize=10,)#label="Group 1")
plt.plot(group2[:,0],group2[:,1], "*",markersize=10,)#label="Group 2")

slope1, intercept1 = np.polyfit(group1[:,0],group1[:,1], 1)
slope2, intercept2 = np.polyfit(group2[:,0],group2[:,1], 1)
samplex=np.linspace(-425,425,20)
sampley1=np.zeros(20)
sampley2=np.zeros(20)
for index,x in enumerate(samplex):
    sampley1[index]=slope1*x+intercept1
    sampley2[index]=slope2*x+intercept2
plt.plot(samplex, sampley1,linewidth=4,label="Edge 1")
plt.plot(samplex, sampley2,linewidth=4,label="Edge 2")


#TODO: Need to determine # of necessary passes
n_passes=15
n=n_passes-1
for i in range(1,n):
    plt.plot(samplex, sampley1*(1-i/n)+sampley2*(i/n),linewidth=2.5,label=f"Intermediate step {i}")
plt.legend()
plt.show()


