import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import itertools
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
paired=sorted(zip(linscore, hull_vertices2D)) #sort pairwise points by pairings
re_ordered_pairs=[coord for score,coord in paired] #returns the pairs ordered from longest to shortest
cornerpoints=re_ordered_pairs[:4]

## Determine what "edge" (aka between corners) aligns best with the primary scanning axis
edges=list(itertools.combinations(cornerpoints, 2))
edgescore=[]
vec2=np.array([1,0,0]) #should be primary axis of point cloud
for index, val in enumerate(edges):
    curr_point=val
    vec1=curr_point[1]-curr_point[0]
    edgescore.append(linearity_score(vec1,vec2))
edges_lists=[[vertex.tolist() for vertex in edge] for edge in edges]
paired=sorted(zip(edgescore, edges_lists), reverse=True) #sort pairwise points by pairings
re_ordered_pairs=[edge for score,edge in paired] #returns the pairs ordered from longest to shortest
alignededges=re_ordered_pairs[:2]





















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