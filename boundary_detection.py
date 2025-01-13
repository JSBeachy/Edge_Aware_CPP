import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PCAClass import PCABounding, Best_Fit_CPP
import time
import math


#To see the called functions in the Best_Fit_CPP class, see PCAClass.py

s=time.time()

#Import Mesh here!
segment=Best_Fit_CPP("plane_segments\plane_segment_8_mesh.stl")
#segment=Best_Fit_CPP("plane_segments\plane_segment_1_mesh.stl")
#segment=Best_Fit_CPP("plane_segments\plane_segment_7_mesh.stl")

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

# Display bounding-box coordinate frame (from PCA)
transform=np.eye(4)
PCA_coord_frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
transform[:3,:3]=segment.rot
transform[:3,3]=segment.cent
PCA_coord_frame.transform(transform)
#o3d.visualization.draw_geometries([segment.mesh,segment.bounding_box, PCA_coord_frame], point_show_normal=False)

#Take convex hull and find the verticies
#Find_convex_hull projects into 2D along least PCA axis, and takes hull of 2D points
segment.find_convex_hull(2, segment.boundary_vertices_coords)
#Boundary2D=segment.PCA_projection(2,segment.boundary_vertices_coords)

#Compare linearity of groups of 3 consecutive hull points. 
#Least linear groups are centered on corner point
segment.find_corner_points()

# #2D plot of convex hull and corner points
# plt.plot(segment.PCA_pointsND[:,0], segment.PCA_pointsND[:,1], 'o', label='Edge points')
# plt.plot(segment.hull_verticesND[:,0],segment.hull_verticesND[:,1], "r*",markersize=10,label="Convex Hull")
# plt.plot(np.array(segment.corner_points)[:,0],np.array(segment.corner_points)[:,1], 'y*',markersize=20,label='Corner Points')
# plt.xlabel("X-coordinate", fontweight="bold",fontsize=14)
# plt.ylabel("Y-coordinate", fontweight="bold",fontsize=14)
# plt.legend(prop={'size': 14, 'weight': 'bold'})
# plt.xticks(fontsize=11, fontweight='bold')
# plt.yticks(fontsize=11, fontweight='bold')
# plt.show()

## Determine what "edge" (aka between corners) aligns best with the primary scanning axis
segment.find_primary_scanning_edges()

## Take edge and classify all points between as point to best-fit 
hull_vertices_list=[vertex.tolist() for vertex in segment.hull_vertices]
primary_edge=segment.aligned_edges[0]
secondary_edge=segment.aligned_edges[1]
group1=segment.splitting(len(hull_vertices_list), primary_edge, hull_vertices_list)
group2=segment.splitting(len(hull_vertices_list), secondary_edge, hull_vertices_list)
#Find the 1D fitting (line) of points
segment.edge1_cent,segment.edge1_vec=segment.fit_line_3d(group1)
segment.edge2_cent,segment.edge2_vec=segment.fit_line_3d(group2)
#Form the array of points that make up the line
line_points1 = segment.point_creator(segment.edge1_cent,segment.edge1_vec, 100)
line_points2 = segment.point_creator(segment.edge2_cent,segment.edge2_vec, 100)


''' Visualization of interpolated lines
line_pcd1 = o3d.geometry.PointCloud()
line_pcd1.points = o3d.utility.Vector3dVector(line_points1)
line_pcd1.paint_uniform_color([0, 1, 0])
line_pcd2 = o3d.geometry.PointCloud()
line_pcd2.points = o3d.utility.Vector3dVector(line_points2)
line_pcd2.paint_uniform_color([0, 0, 1])
#Code for Interpolating lines
intermediate_points=o3d.geometry.PointCloud()
n=5
for i in range(1,n):
   temp_ptc=o3d.geometry.PointCloud()
   temp_ptc.points=o3d.utility.Vector3dVector(np.asarray(line_pcd1.points)*(1-i/(n-1))+np.asarray(line_pcd2.points)*(i/(n-1)))
   temp_ptc.paint_uniform_color([0, 1-i/(n-1),i/(n-1)])
   intermediate_points+=temp_ptc

o3d.visualization.draw_geometries([segment.mesh, segment.bounding_box, boundary_pcd, line_pcd1, line_pcd2, intermediate_points])
'''

#Scan_information takes: probe width, scan_line1, scan_line2, and the edge offset (optional)
segment.scan_information(64, line_points1, line_points2,)
#Prints out relevant information about scan_passes
segment.print_scan_information()

#Edge Offset: find what direction to move in for each vector
segment.shift_direction()
#shift_vec=offset (default is 1/2 probe width) shift in the direction of the secondary-axis
shift_vec=segment.offset_one*segment.secondary_axis
line_one=np.asarray(line_points1) - segment.offset_dir*shift_vec
line_last=np.asarray(line_points2) + segment.offset_dir*shift_vec

#Get official lines returned; with consideration for number of passes
lines,color=segment.line_interpolator(line_one,line_last)
color_arrays = [np.array(color) * np.ones((lines[1].shape[0], 1)) for points, color in zip(lines, color)]

#Add 10 mm to each point's z-coordinate in the trial lines; TODO: add triangle normal component
adjusted_lines = [line + np.array([0, 0, 10]) for line in lines]
e=time.time()
print(f"{round(e-s,3)} seconds run time")

#Final visualization
trial=o3d.geometry.PointCloud()
trial.points=o3d.utility.Vector3dVector(np.vstack(adjusted_lines))
trial.colors=o3d.utility.Vector3dVector(np.vstack(color_arrays))
o3d.visualization.draw_geometries([segment.mesh,segment.bounding_box, trial, boundary_pcd])

# Define probe dimensions
probe_width = 10  # Direction of Scanning
probe_length = 50 # Direction perp to scanning

scanned_mesh = segment.scanned_area(adjusted_lines, probe_width, probe_length)
legacy_mesh = scanned_mesh.to_legacy()
# Visualize
o3d.visualization.draw([scanned_mesh, trial], show_ui=True)