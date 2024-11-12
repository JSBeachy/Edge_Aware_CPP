import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import itertools
import math
from PCAClass import PCABounding, Best_Fit_CPP

def distance_theshold(new_point,cornerpoints):
    #TODO: Link threshold to with the length of primary/secondary axis 
    threshold=20
    for corner in cornerpoints:
        if np.linalg.norm(new_point-corner)<threshold:
            return False
    return True

def linearity_score(vec1,vec2):
    cos_theta= np.dot(vec1,vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return abs(cos_theta)

def splitting(len_my_list, edge_list, hull_vertices_list):
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

def fit_line_3d(points):
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)
    # Use Singular Value Decomposition (SVD) to find the principal direction
    _, _, vh = np.linalg.svd(points - centroid)
    direction_vector = vh[0]  # First right-singular vector is the principal direction
    return centroid, direction_vector

def bounding_box_interior_points(points):
    bb_min=bounding_box.get_min_bound()
    bb_max=bounding_box.get_max_bound()
    p1=np.asarray(points.points)
    inside_bbox = (p1 >= bb_min) & (p1 <= bb_max) #& for element-wise logic in numpy
    within_bbox = np.all(inside_bbox, axis=1)
    filtered_points=p1[within_bbox]
    if filtered_points.size > 0:
        #print( filtered_points[0] ) # Return the first point inside the bounding box
        #print( filtered_points[-1] ) # Return the last point inside the bounding box
        return(filtered_points[0],filtered_points[-1])
    else:
        print("No Points in Bounding Box")
        return None 
    
def scan_width_determination(line_pcd1,line_pcd2):
    p0_0,p0_1=bounding_box_interior_points(line_pcd1)
    p1_0,p1_1=bounding_box_interior_points(line_pcd2)
    scan_width=max(np.linalg.norm(p0_0-p1_0),np.linalg.norm(p0_1-p1_1))
    return scan_width

#Load the STL file
#mesh = o3d.io.read_triangle_mesh("plane_segments\Circle_mesh.stl")
#mesh = o3d.io.read_triangle_mesh("plane_segments\Skinny_tall_mesh.stl")
#mesh = o3d.io.read_triangle_mesh("plane_segments\Fat_Short_mesh.stl")
mesh = o3d.io.read_triangle_mesh("plane_segments\plane_segment_8_mesh.stl")

# pre-process mesh
tolerance = 1e-5  # Adjust tolerance as needed
mesh.merge_close_vertices(tolerance)

mesh.compute_adjacency_list()
mesh.compute_triangle_normals()
mesh.compute_vertex_normals()
#o3d.visualization.draw_geometries([mesh], point_show_normal=False)

#Take bounding box and find principal components
bounding_box=mesh.get_oriented_bounding_box()
bounding_box.color=([1,0,0])
bblen=list(bounding_box.extent)
primary_axis_index=bblen.index(max(bblen))
popped_bblen=bblen[0:primary_axis_index]+bblen[primary_axis_index+1:]
secondary_axis_index=bblen.index(max(popped_bblen))
tertiary_axis_index=3-(primary_axis_index+secondary_axis_index)
bb_len=bounding_box.extent[primary_axis_index]
rot=bounding_box.R
cent= bounding_box.center 
primary_axis=bounding_box.R[:,primary_axis_index]
secondary_axis=bounding_box.R[:,secondary_axis_index]
tertiary_axis=bounding_box.R[:,tertiary_axis_index]



# Find boundary edges and identify boundary verticies
boundary = mesh.get_non_manifold_edges(allow_boundary_edges=False) #Non-manifold defined differently if True
boundary_vertices = np.unique(np.array(boundary).flatten())
boundary_vertices_coords = np.asarray(mesh.vertices)[boundary_vertices]
#print("Boundary Vertices Coordinates:")
#print(segment.boundary_vertices_coords)

# Create a visualization object
boundary_pcd=o3d.geometry.PointCloud()
boundary_pcd.points = o3d.utility.Vector3dVector(boundary_vertices_coords)
boundary_pcd.paint_uniform_color([1,0,0])
o3d.visualization.draw_geometries([mesh, boundary_pcd], point_show_normal=False)

#Take convex hull and find the verticies
Coords2D=boundary_vertices_coords[:,:2]
hull = ConvexHull(Coords2D)
hull_vertices=boundary_vertices_coords[hull.vertices]
hull_vertices2D=Coords2D[hull.vertices]
hull_vertices_list=[vertex.tolist() for vertex in hull_vertices]

##Calculate relative linearity of consecutive hull points (corners are angled)
linscore=[]
for index, val in enumerate(hull_vertices):
    prev_point=hull_vertices[index-1]
    curr_point=val
    next_point=hull_vertices[(index+1) % len(hull_vertices)]
    vec1=curr_point-prev_point
    vec2=next_point-curr_point
    linscore.append(linearity_score(vec1,vec2))

## Determine which coordinates are the corner points and plot
#TODO implement a non-np.array type for verticies (may be okay now)
paired=sorted(zip(linscore, hull_vertices)) #sort pairwise points by pairings
re_ordered_pairs=[coord for score,coord in paired] #returns the pairs ordered from longest to shortest
cornerpoints=re_ordered_pairs[:4]
#print(cornerpoints)

plt.plot(Coords2D[:,0], Coords2D[:,1], 'o', label='Edge points')
plt.plot(hull_vertices2D[:,0],hull_vertices2D[:,1], "r*",markersize=10,label="Convex Hull")
plt.plot(np.array(cornerpoints)[:,0],np.array(cornerpoints)[:,1], 'y*',markersize=20,label='Corner Points')
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
#plt.legend()
#plt.show()

##Calculate relative linearity of consecutive hull points (corners are angled)
linscore=[]
for index, val in enumerate(hull_vertices2D):
    prev_point=hull_vertices2D[index-1]
    curr_point=val
    next_point=hull_vertices2D[(index+1) % len(hull_vertices2D)]
    vec1=curr_point-prev_point
    vec2=next_point-curr_point
    linscore.append(linearity_score(vec1,vec2))
noise = np.random.uniform(1e-4, 1e-5, len(linscore))
linscore=linscore+noise
# Determine which coordinates are the corner points and plot
paired=sorted(zip(linscore, hull_vertices)) #sort pairwise points by pairings
re_ordered_pairs=[coord for score,coord in paired] #returns the pairs ordered from longest to shortest
cornerpoints=re_ordered_pairs[:4]

## Determine what "edge" (aka between corners) aligns best with the primary scanning axis
edges=list(itertools.combinations(cornerpoints, 2))
edgescore=[]
vec2=primary_axis #should be primary axis of point cloud
for index, val in enumerate(edges):
    curr_point=val
    vec1=curr_point[1]-curr_point[0]
    edgescore.append(linearity_score(vec1,vec2))
edges_lists=[[vertex.tolist() for vertex in edge] for edge in edges]
paired=sorted(zip(edgescore, edges_lists), reverse=True) #sort pairwise points by pairings
re_ordered_pairs=[edge for score,edge in paired] #returns the pairs ordered from longest to shortest
aligned_edges=re_ordered_pairs[:2]

## Take edge and classify all points between as point to best-fit 
hull_vertices_list=[vertex.tolist() for vertex in hull_vertices]
primary_edge=aligned_edges[0]
secondary_edge=aligned_edges[1]
group1=splitting(len(hull_vertices_list), primary_edge, hull_vertices_list)
group2=splitting(len(hull_vertices_list), secondary_edge, hull_vertices_list)

# Form point-cloud of fitting
prop_len=int(bb_len/5)
centroid1,vec1=fit_line_3d(group1)
line_points1 = np.array([centroid1 + t * vec1 for t in np.linspace(-bb_len/2-prop_len, bb_len/2+prop_len, 100)])
centroid2,vec2=fit_line_3d(group2)
line_points2 = np.array([centroid2 + t * vec2 for t in np.linspace(-bb_len/2-prop_len, bb_len/2+prop_len, 100)])
#print(vec1,vec2)

line_pcd1 = o3d.geometry.PointCloud()
line_pcd1.points = o3d.utility.Vector3dVector(line_points1)
line_pcd1.paint_uniform_color([0, 1, 0])
line_pcd2 = o3d.geometry.PointCloud()
line_pcd2.points = o3d.utility.Vector3dVector(line_points2)
line_pcd2.paint_uniform_color([0, 0, 1])
'''
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

probe_width=64
half_probe=probe_width/2
tot_width=scan_width_determination(line_pcd1,line_pcd2)
print(f"Total width to be scanned: {tot_width}")

#setting both edges
offset_one=probe_width/2
offset_two=tot_width-offset_one
tot_passes=math.ceil(tot_width/probe_width)
per_pass_width=round(tot_width/tot_passes,2)
print(f"Width allocated to each pass: {per_pass_width}")

real_distance=tot_width-2*probe_width
passes_left=math.ceil(real_distance/probe_width)
real_per_pass_width=real_distance/passes_left
print(f"Number of total passes: {tot_passes}")
print(f"Number of passes remaining after setting edges: {passes_left}")
print(f"Approximate pass width after setting edges: {real_per_pass_width}")

interp_one=probe_width+real_per_pass_width/2
interp_two=(tot_width-probe_width)-real_per_pass_width/2

#Edge Offset: find what direction to move in for each vector
#print(direction1/np.linalg.norm(direction1), segment.secondary_axis)
#print(np.dot(direction1/np.linalg.norm(direction1),segment.secondary_axis))
#print(np.dot(direction2/np.linalg.norm(direction2),segment.secondary_axis))
direction1=centroid1-cent
direction2=centroid2-cent
dot_1=np.dot(direction1/np.linalg.norm(direction1),secondary_axis)
dot_2=np.dot(direction2/np.linalg.norm(direction2),secondary_axis)

#shift_vec=1/2 probe width shift in the direction of the secondary-axis
shift_vec=half_probe*secondary_axis
if dot_1<0:
    dot_1=-1
    start=1
else:
    dot_1=1 
    start=0 #should build the opposite way

#Shift the Vectors 1/2 probe width
line_one=np.asarray(line_pcd1.points) - dot_1*shift_vec
line_last=np.asarray(line_pcd2.points) - dot_2*shift_vec
color_one=np.asarray([0, 1, 0])
color_two=np.asarray([0, 0, 1])
'''
#Option 1 (Option 2 from Num_pass_calc - Intermediate Interpolation) -NOT Prefered
interp_one=np.asarray(line_pcd1.points) - dot_1*(probe_width+real_per_pass_width/2)*segment.secondary_axis
interp_two=np.asarray(line_pcd2.points) - dot_2*(probe_width+real_per_pass_width/2)*segment.secondary_axis
lines=[]
for i in range(tot_passes):
    if i==0:
        lines.append(line_one)
    elif i==tot_passes-1:
        lines.append(line_last)
    else:
        lines.append(interp_one*(1-(i-1)/(passes_left-1))+interp_two*((i-1)/(passes_left-1)))
'''
#Option 2 (Option 3 from Num_pass_calc - Direct interpolation)
lines=[]
color=[]
for i in range(tot_passes):
    lines.append(line_one*(1-(i)/(tot_passes-1))+line_last*((i)/(tot_passes-1)))
    color.append(color_one*(1-(i)/(tot_passes-1))+color_two*((i)/(tot_passes-1)))
color_arrays = [np.array(color) * np.ones((lines[1].shape[0], 1)) for points, color in zip(lines, color)]
trial=o3d.geometry.PointCloud()
trial.points=o3d.utility.Vector3dVector(np.vstack(lines))
trial.colors=o3d.utility.Vector3dVector(np.vstack(color_arrays))
o3d.visualization.draw_geometries([mesh, bounding_box, trial])












## For calculating corners based on pairwise distance.... Not guaranteed much with this method
'''
pairs=itertools.combinations(hull.vertices, 2)
pairs=list(pairs)
distances=[]
for i in pairs:
    coords=[boundary_vertices_coords[j] for j in i]
    #print(coords)
    distances.append(np.linalg.norm(coords[0]-coords[1]))
#print(distances)
paired=sorted(zip(distances,pairs), reverse=True) #sort pairwise points by pairings
re_ordered_pairs=[pair for dist,pair in paired] #returns the pairs ordered from longest to shortest
#print(re_ordered_pairs)


#determine corners
cornerpoints=[]
counter=0
index=[]
while len(cornerpoints)<4:
    point_tuple=re_ordered_pairs[counter]
    for j in point_tuple:
        new_point = boundary_vertices_coords[j]
        if j not in index and distance_theshold(new_point, cornerpoints):
            index.append(j)
            cornerpoints.append(new_point)  

    counter+=1
'''