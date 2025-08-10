import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PCAClass import PCABounding, Best_Fit_CPP
import time
import os


#To see the called functions in the Best_Fit_CPP class, see PCAClass.py
start=time.time()
#Import Mesh here!
#segment=Best_Fit_CPP("plane_segments\plane_segment_8_mesh.stl")
#segment=Best_Fit_CPP("plane_segments\plane_segment_1_mesh.stl")
#segment=Best_Fit_CPP("plane_segments\plane_segment_7_mesh.stl")
#segment=Best_Fit_CPP("plane_segments\Hyper_meshed_noise.stl")
#segment=Best_Fit_CPP(r"plane_segments\Non-planar.stl")
#segment=Best_Fit_CPP(r"plane_segments\Bowl.stl")
#segment=Best_Fit_CPP("plane_segments\Foil.stl")
#segment=Best_Fit_CPP("plane_segments\Concave.stl")
#segment=Best_Fit_CPP("plane_segments\Convex.stl")
#segment=Best_Fit_CPP("plane_segments\BigL.stl")
#segment=Best_Fit_CPP("plane_segments\Curvy.stl")
#segment=Best_Fit_CPP("plane_segments\surjective_xz.stl")
#segment=Best_Fit_CPP("plane_segments\Airfoil_Surface_example.stl")
#segment=Best_Fit_CPP("plane_segments\Airfoil_Surface_example_Hypermeshed.stl")
#segment=Best_Fit_CPP(r"C:\Users\jonas\NDIBARC\pointcloudcpp\plane_segments\Nose_Cone.stl")
#segment=Best_Fit_CPP("plane_segments\Hat_Stringer.stl")
#segment=Best_Fit_CPP(r"plane_segments\trap_mesh2.stl")
segment=Best_Fit_CPP(r"plane_segments\Fake_Real_Mesh.stl")
segment.boundary_edge_calculations()

# Create a visualization object
#boundary_pcd=o3d.geometry.PointCloud()
#boundary_pcd.points = o3d.utility.Vector3dVector(segment.ordered_edge_points)
#boundary_pcd.paint_uniform_color([0,0,0])

# Display bounding-box coordinate frame (from PCA)
#transform=np.eye(4)
#PCA_coord_frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=250.0)
#transform[:3,:3]=segment.PCA_eigenvecs
#transform[:3,3]=segment.cent+10*segment.PCA_eigenvecs[segment.tertiary_axis_index]
#PCA_coord_frame.transform(transform)
#o3d.visualization.draw_geometries([segment.mesh,segment.bounding_box, PCA_coord_frame], mesh_show_back_face=True)

# #2D plot of convex hull and corner points
# plt.plot(segment.ordered_edge_points2D[:,0], segment.ordered_edge_points2D[:,1], 'o', label='Edge points')
# plt.plot(np.array(segment.corner_points2D)[:,0],np.array(segment.corner_points2D)[:,1], 'y*',markersize=20,label='Corner Points')
# plt.xlabel("Principle Axis", fontweight="bold",fontsize=14)
# plt.ylabel("Secondary Axis", fontweight="bold",fontsize=14)
# plt.legend(prop={'size': 14, 'weight': 'bold'})
# plt.xticks(fontsize=11, fontweight='bold')
# plt.yticks(fontsize=11, fontweight='bold')
# plt.show()


## Take edge and classify all points between as point to best-fit 
segment.edge_fitter(segment.edges)

##Fit Check!
# edges1=o3d.geometry.PointCloud()
# edges1.points=o3d.utility.Vector3dVector(np.vstack(segment.ordered_edge_points))
# edges1.paint_uniform_color([1,0,0])
# edge_cp=[segment.edge1_CP,segment.edge2_CP]
# target_length=20
# edges=[np.vstack([segment.bezier_curve3N(segment.Bezier_order,t,Pi) for t in segment.find_t_newton(segment.Bezier_order, Pi,target_length)]) for Pi in edge_cp]
# color=[[0,1,0],[0,0,1]]
# colors=[col*np.ones((len(edge), 1)) for col, edge in zip(color,edges)]
# edge_fit=o3d.geometry.PointCloud()
# edge_fit.points=o3d.utility.Vector3dVector(np.vstack(edges))
# edge_fit.colors=o3d.utility.Vector3dVector(np.vstack(colors))
# bezier_fit=o3d.geometry.PointCloud()
# bezier_fit.points=o3d.utility.Vector3dVector(np.vstack(segment.edge1_CP))
# bezier_fit.paint_uniform_color([1,0,0])
# segment.fancy_viz([segment.mesh,bezier_fit, edge_fit, segment.bounding_box])
#o3d.visualization.draw_geometries([segment.mesh, edge_fit])
#exit()

#Determine scanning width and required pass info
Probe_width=64
slice_resolution=5 # mm per sampled point
segment.scan_information(Probe_width, slice_resolution)

#Prints out relevant information about scan_passes
segment.print_scan_information()

#Edge Offset: find what direction to move in for each vector
segment.shift_direction()

#shift_vec
shift_vec=segment.offset_one*segment.secondary_axis
segment.edge1_CP=segment.edge1_CP - segment.offset_dir*shift_vec
segment.edge2_CP=segment.edge2_CP + segment.offset_dir*shift_vec

#Interpolate Bézier Curves
passes,colors=segment.line_interpolator(5)

#Add 10 mm to each point's z-coordinate in the trial lines; TODO: could add normal of closes point via kd-tree
adjusted_lines = np.vstack(passes)+ np.array([0, 0, 10])


#Path visualization
#trial=o3d.geometry.PointCloud()
#trial.points=o3d.utility.Vector3dVector(adjusted_lines)
#trial.colors=o3d.utility.Vector3dVector(np.vstack(colors))
#segment.fancy_viz([segment.mesh, trial])
#o3d.visualization.draw_geometries([segment.mesh,segment.bounding_box, trial, boundary_pcd],mesh_show_back_face=True)
#exit()

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
n=0
#while n<20:
#start2=time.time()
redundant, On_surface, On_surface_color=segment.local_scanned_area(Redundancy=True, Elimination=False)
#end=time.time()
#print(f"total time for checking: {end-start2}")
#print(f"Total Path-Planning runtime: {end-start}")
#print(On_surface)

if len(redundant)>0:
    redundant_points=o3d.geometry.PointCloud()
    redundant_points.points=o3d.utility.Vector3dVector(np.vstack(redundant))
    redundant_points.colors=o3d.utility.Vector3dVector(np.full((len(redundant), 3), [1, 0.5, 0]))
    pass_points=o3d.geometry.PointCloud()
    pass_points.points=o3d.utility.Vector3dVector(np.vstack(On_surface))
    pass_points.colors=o3d.utility.Vector3dVector(np.vstack(On_surface_color))
    #forest_mask= (segment.colors == [0, 1, 0]).all(axis=1)
    #print("Green Mask Count:", np.sum(forest_mask)) 
    #blue_mask=(segment.colors == [0, 0, 1]).all(axis=1)
    #segment.mesh.paint_uniform_color([87/255,108/255, 67/255])
    #segment.colors[blue_mask]=np.array([0.10,0.6,0.25])  #[19,71,39]
    #segment.mesh.vertex_colors[forest_mask] = [0,1,0]
    if n==0:
        segment.mesh.vertex_colors = o3d.utility.Vector3dVector(segment.colors)
        segment.fancy_viz([segment.mesh, pass_points, redundant_points])
    #o3d.visualization.draw_geometries([segment.mesh,pass_points],mesh_show_back_face=True)
else:
    pass_points=o3d.geometry.PointCloud()
    pass_points.points=o3d.utility.Vector3dVector(np.vstack(On_surface))
    pass_points.colors=o3d.utility.Vector3dVector(np.vstack(On_surface_color))
    #forest_mask= (segment.colors == [0, 1, 0]).all(axis=1)
    #blue_mask=(segment.colors == [0, 0, 1]).all(axis=1)
    #segment.colors[forest_mask]= [87/255,108/255, 67/255]
    #segment.colors[blue_mask]=np.array([0.10,0.6,0.25])  #[19,71,39]
    #segment.mesh.vertex_colors = o3d.utility.Vector3dVector(segment.colors)
    #segment.fancy_viz([segment.mesh, pass_points])
    #o3d.visualization.draw_geometries([segment.mesh],mesh_show_back_face=True)
    if n==0:
        #segment.mesh.vertex_colors = o3d.utility.Vector3dVector(segment.colors)
        #segment.fancy_viz([segment.mesh, pass_points,])
        pass

#segment.passes=segment.Potential_Field(On_surface)
n+=1
segment.mesh.vertex_colors = o3d.utility.Vector3dVector(segment.colors)
#segment.mesh.paint_uniform_color([0.5, 0.5, 0.5])
#o3d.visualization.draw_geometries([segment.mesh,pass_points],mesh_show_back_face=True)
segment.fancy_viz([segment.mesh, pass_points])




########### Interface with RoboDK ##################3
from robolink import * # RoboDK API
from robodk import *
import os

### Use on-surface
poses = []
names = []
RDK = Robolink()
print("Connected to RoboDK.")

#Clear Previous Items from the Station
print("Clearing existing objects and targets...")
all_items = RDK.ItemList()
for item in all_items:
    # Delete any previous objects (meshes) or targets
    item_type = item.Type()
    if item_type == ITEM_TYPE_OBJECT or item_type == ITEM_TYPE_TARGET:
        item.Delete()

#Load the Current Mesh
absolute_mesh_path = os.path.abspath(segment.relative_file_name)
print(f"Loading mesh from absolute path: {absolute_mesh_path}")
mesh_object = RDK.AddFile(absolute_mesh_path)
mesh_object.setName("SegmentMesh")
base_frame = RDK.Item('', robolink.ITEM_TYPE_FRAME)

# This loop iterates through the calculated passes to generate target poses
# It uses the logic from your provided framework.
for i, point_set in enumerate(On_surface):
    localposes = []
    localnames = []

    principal_vector = point_set[-1] - point_set[0]
    principal_vector /= np.linalg.norm(principal_vector)
    if np.dot(principal_vector, segment.primary_axis) < 0:
        principal_vector *= -1

    # Project path points onto the mesh surface using ray casting
    direction = np.array(segment.tertiary_axis) * -1 * np.ones((point_set.shape[0], 1))
    z_offset = min(segment.bounding_box.extent) * segment.tertiary_axis
    ray_origins = point_set + z_offset
    LocVec = np.hstack((ray_origins.astype(dtype=np.float32), direction.astype(np.float32)))
    ans = segment.scene.cast_rays(LocVec)
    index_hits = [p for p, q in enumerate(ans['geometry_ids']) if q == 0]

    if not index_hits:
        continue # Skip passes with no valid hits

    # --- Generate a selection of points along the path for robot targets ---
    # Numpoints calculation may scale quite inappropriately, this is not a refined equation
    numpoints = min(20, 5 + int(segment.PCA_eigenvals[-1] // 10))
    # Set first robot pose in roughly 10 mm from edge
    # Ensure there are enough hit points to select from
    if len(index_hits) > 20:
        RoboIndex = np.linspace(index_hits[10], index_hits[-10], numpoints).round().astype(int)
    else:
        RoboIndex = np.linspace(index_hits[0], index_hits[-1], numpoints).round().astype(int)


    for k in RoboIndex:
        dist = ans['t_hit'].numpy()[k]
        delta = dist * segment.tertiary_axis * (-1)
        onSurface = LocVec[k][:3] + delta
        intersection_index = ans['primitive_ids'].numpy()[k]
        
        # Compute the average normal at the intersection point
        average_normal = segment.compute_average_normal_t(segment.tensor_plane, intersection_index)

        p_vec = principal_vector.flatten()
        a_norm = average_normal.flatten()
        o_surf = onSurface.flatten()
        y_vec = np.cross(a_norm, p_vec).flatten()
        
        # Creates RoboDK points; Uses principal_vector from above instead of primary axis due to interpolation between edges
        transformation_matrix = np.eye(4)
        transformation_matrix[0:3, 0] = principal_vector
        transformation_matrix[0:3, 1] = y_vec
        transformation_matrix[0:3, 2] = a_norm
        transformation_matrix[0:3, 3] = o_surf

        localnames.append("Target_" + str(i + 1) + "_" + str(k + 1))
        localposes.append(transformation_matrix) # Convert numpy array to RoboDK matrix

    # Constructs raster motion by reversing every other pass
    if i % 2 != 0:
        localposes = localposes[::-1]
        localnames = localnames[::-1]

    poses.extend(localposes)
    names.extend(localnames)

# --- Add the calculated targets to the RoboDK station ---
print(f"Adding {len(poses)} new targets to RoboDK...")

for i, pose in enumerate(poses):
    target=Mat(pose.tolist())
    target_name = names[i]
    new_target = RDK.AddTarget(target_name,base_frame)
    new_target.setPose(target)

print("RoboDK update complete.")





