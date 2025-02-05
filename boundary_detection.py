import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PCAClass import PCABounding, Best_Fit_CPP
import time
import math

#To see the called functions in the Best_Fit_CPP class, see PCAClass.py

s=time.time()

#Import Mesh here!
#segment=Best_Fit_CPP("plane_segments\plane_segment_8_mesh.stl")
#segment=Best_Fit_CPP("plane_segments\plane_segment_1_mesh.stl")
#segment=Best_Fit_CPP("plane_segments\plane_segment_7_mesh.stl")
segment=Best_Fit_CPP("plane_segments\Hyper_meshed_noise.stl")
#segment=Best_Fit_CPP(r"plane_segments\Non-planar.stl")
#segment=Best_Fit_CPP(r"plane_segments\Bowl.stl")

# Ensure the mesh has edges and triangle information for visualization
segment.mesh.compute_adjacency_list()
segment.mesh.compute_triangle_normals()
segment.mesh.compute_vertex_normals()

print(f"Principal Axes:\n {segment.PCA_eigenvecs}")
print(f"Relative Importance:\n {segment.PCA_eigenvals}")

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
#print(segment.hull_vertices)

# # #2D plot of convex hull and corner points
# plt.plot(segment.PCA_pointsND[:,segment.primary_axis_index], segment.PCA_pointsND[:,segment.secondary_axis_index], 'o', label='Edge points')
# plt.plot(segment.hull_verticesND[:,segment.primary_axis_index],segment.hull_verticesND[:,segment.secondary_axis_index], "r*",markersize=10,label="Convex Hull")
# plt.plot(np.array(segment.corner_pointsND)[:,0],np.array(segment.corner_pointsND)[:,1], 'y*',markersize=20,label='Corner Points')
# plt.xlabel("Principle Axis", fontweight="bold",fontsize=14)
# plt.ylabel("Secondary Axis", fontweight="bold",fontsize=14)
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
group1=segment.splitting(primary_edge, hull_vertices_list)
group2=segment.splitting(secondary_edge, hull_vertices_list)

#Find the 1D fitting (line) of points
segment.edge1_cent,segment.edge1_vec=segment.fit_line_3d(group1)
segment.edge2_cent,segment.edge2_vec=segment.fit_line_3d(group2)

#Form the array of points that make up the line
line_points1 = segment.point_creator(segment.edge1_cent,segment.edge1_vec, 100)
line_points2 = segment.point_creator(segment.edge2_cent,segment.edge2_vec, 100)

if np.linalg.norm(line_points1[0]-line_points2[0])>np.linalg.norm(line_points1[0]-line_points2[-1]):
    line_points2=line_points2[::-1]


# # Visualization of interpolated lines
# line_pcd1 = o3d.geometry.PointCloud()
# line_pcd1.points = o3d.utility.Vector3dVector(line_points1)
# line_pcd1.paint_uniform_color([0, 1, 0])
# line_pcd2 = o3d.geometry.PointCloud()
# line_pcd2.points = o3d.utility.Vector3dVector(line_points2)
# line_pcd2.paint_uniform_color([0, 0, 1])
# #Code for Interpolating lines
# intermediate_points=o3d.geometry.PointCloud()
# n=5
# for i in range(1,n):
#    temp_ptc=o3d.geometry.PointCloud()
#    temp_ptc.points=o3d.utility.Vector3dVector(np.asarray(line_pcd1.points)*(1-i/(n-1))+np.asarray(line_pcd2.points)*(i/(n-1)))
#    temp_ptc.paint_uniform_color([0, 1-i/(n-1),i/(n-1)])
#    intermediate_points+=temp_ptc

# o3d.visualization.draw_geometries([segment.mesh, segment.bounding_box, boundary_pcd, line_pcd1, line_pcd2, intermediate_points])


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
color_arrays = [np.array(color) * np.ones((lines[0].shape[0], 1)) for points, color in zip(lines, color)]

#Add 10 mm to each point's z-coordinate in the trial lines; TODO: add triangle normal component
adjusted_lines = [line + np.array([0, 0, 10]) for line in lines]
e=time.time()
print(f"{round(e-s,3)} seconds run time")

#Path visualization
trial=o3d.geometry.PointCloud()
trial.points=o3d.utility.Vector3dVector(np.vstack(adjusted_lines))
trial.colors=o3d.utility.Vector3dVector(np.vstack(color_arrays))
o3d.visualization.draw_geometries([segment.mesh,segment.bounding_box, trial, boundary_pcd])


s2=time.time()

'''
# Define probe dimensions
probe_width = 10  # Direction of Scanning
probe_length = 50 # Direction perp to scanning

scanned_mesh = segment.scanned_area(adjusted_lines, probe_width, probe_length)
legacy_mesh = scanned_mesh.to_legacy()
# Visualize
e1=time.time()
o3d.visualization.draw([scanned_mesh, trial], show_ui=True)
print(f"Scan calculation time: {e1-s1}")
'''

#Format mesh for ray-casting
tensor_plane = o3d.t.geometry.TriangleMesh.from_legacy(segment.mesh) #OGplane= Surface to Path-Plan on
tensor_plane.compute_triangle_normals()
tensor_plane.compute_vertex_normals()
scene = o3d.t.geometry.RaycastingScene()
tensor_cast_id = scene.add_triangles(tensor_plane)

poses=[]
names=[]

#direction for ray-casting should be along tertiary axis, but pointing "down"
direction=np.array(segment.tertiary_axis) * np.ones((lines[0].shape[0], 1))*-1
for i, point_set in enumerate(lines):
    #move points "up" on z axis by max z bb height to ensure projections have intersections
    z_offset=segment.bounding_box.extent[-1]*segment.tertiary_axis
    ray_origins=point_set+z_offset
    #Ray Cast to test for intersection
    LocVec=np.hstack((ray_origins.astype(dtype=np.float32), direction.astype(np.float32)))
    #print(LocVec)
    ans = scene.cast_rays(LocVec)


    #Calculate "principal vector" for interpolated
    principal_vector=point_set[0]-point_set[-1]
    #print(point_set[0],point_set[-1],principal_vector)
    principal_vector=principal_vector/np.linalg.norm(principal_vector)
    sign=np.dot(principal_vector,segment.primary_axis)
    if sign<0:
        principal_vector=-1*principal_vector

    localposes=[]
    localnames=[]
    #Determine the index of ray-casting to use for RoboDK targets
    index_hits=[p for p,q in enumerate(ans['geometry_ids']) if q==0]
    #Numpoints calculation may scale quite inappropriately, this is not refined equation
    #TODO: Test further to determine how useful numpoints actually is 
    numpoints=5+int(segment.PCA_eigenvals[-1]//10)
    #Set first robot pose in roughly 10 mm from edge
    RoboIndex=np.linspace(index_hits[10],index_hits[-10],numpoints).round().astype(int)
    
    color_updates = np.full((segment.points.shape[0],3), -1.0)

    for i in index_hits:
        dist=ans['t_hit'].numpy()[i]
        delta=dist*segment.tertiary_axis*(-1)
        onSurface=LocVec[i][:3]+delta
        intersection_index=ans['primitive_ids'].numpy()[i]
        #print(f"Intersected triangle index: {intersection_index}")
        average_normal = segment.compute_average_normal_t(tensor_plane, intersection_index)
        #print(f"Average normal of all neighbors: {average_normal}")

        #Creates RoboDK points; Uses principal_vector from above instead of primary axis due to interpolation between edges
        transformation_matrix=np.eye(3)
        transformation_matrix[0:3,0]=principal_vector
        transformation_matrix[0:3,1]=np.cross(average_normal, principal_vector)
        transformation_matrix[0:3,2]=average_normal
        #print(transformation_matrix)

        candidate_indices = np.array(segment.kd_tree.query_ball_point(onSurface, 32))
        if len(candidate_indices)>0:
            inv_rotation_matrix = np.linalg.inv(transformation_matrix)
            #print(candidate_indices)
            new_candidate_points= inv_rotation_matrix@segment.points[candidate_indices].T
            new_seed_point=inv_rotation_matrix@onSurface
            probe_mask = ((new_seed_point[0] - 10 <= new_candidate_points[0]) & (new_candidate_points[0] <= new_seed_point[0] + 10) &
                        (new_seed_point[1] - 32 <= new_candidate_points[1]) & (new_candidate_points[1] <= new_seed_point[1] + 32))
            
            scanned_points=candidate_indices[probe_mask]
            color_updates[scanned_points] = [0, 1, 0]

    update_mask = color_updates[:, 0] != -1
    already_marked_mask = (segment.colors[:, 0] != 1) 
    rescanned_mask = update_mask & already_marked_mask
    segment.colors[update_mask] = color_updates[update_mask]
    segment.colors[rescanned_mask]=[0,0,1]

            


segment.mesh.vertex_colors = o3d.utility.Vector3dVector(segment.colors)
e2=time.time()
print(f"total time for checking: {e2-s2}")
o3d.visualization.draw_geometries([segment.mesh],mesh_show_back_face=True)
'''
for k in RoboIndex:
    dist=ans['t_hit'].numpy()[k]
    delta=dist*segment.tertiary_axis*(-1)
    onSurface=LocVec[k][:3]+delta
    intersection_index=ans['primitive_ids'].numpy()[k]
    #print(f"Intersected triangle index: {intersection_index}")
    average_normal = segment.compute_average_normal_t(tensor_plane, intersection_index)
    #print(f"Average normal of all neighbors: {average_normal}")


    #Creates RoboDK points; Uses principal_vector from above instead of primary axis due to interpolation between edges
    transformation_matrix=np.eye(4)
    transformation_matrix[0:3,0]=principal_vector
    transformation_matrix[0:3,1]=np.cross(average_normal, principal_vector)
    transformation_matrix[0:3,2]=average_normal
    transformation_matrix[0:3,3]=onSurface
    
    localnames.append("Target_"+str(i+1)+"_"+str(k+1))  
    localposes.append(transformation_matrix)
print(i)
#constructs raster motion
if i%2!=0:
    localposes=localposes[::-1]
    localnames=localnames[::-1]
for i,j in enumerate(localposes):
    names.append(localnames[i])
    poses.append(localposes[i])
'''
