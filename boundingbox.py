import numpy as np
import open3d as o3d

#import os



#trying to get the average mesh normals
def compute_average_normal_t(mesh, triangle_index):
    
    #Make sure that the normals of mesh have been computed
    #Get tensor of triangles_indicies and normals_vectors of mesh
    triangles = mesh.triangle['indices']
    normals = mesh.triangle['normals']

    # Get the vertices of the target triangle
    target_triangle = triangles[triangle_index].numpy()


    # Find neighboring triangles (those sharing any vertex with the target triangle)
    mask = np.isin(triangles.numpy(), target_triangle).any(axis=1)
    
    # Get the normals of the neighboring triangles
    neighboring_normals = normals[mask]

    # Compute the average normal
    average_normal = neighboring_normals.mean(dim=0)
    
    # Normalize the average normal
    average_normal_np = average_normal.numpy()
    norm = np.linalg.norm(average_normal_np)
    average_normal = average_normal_np / norm
    
    return average_normal




'''
print(os.listdir('plane_segments'))
plane_nums=len([name for name in os.listdir('plane_segments')])
plane_int=[]
print(plane_nums)
for i in range(plane_nums):
    plane=o3d.io.read_triangle_mesh('plane_segments\plane_segment_'+str(i+1)+'_mesh.stl')
    #print(len(np.asarray(plane.vertices)))
    plane_int.append(len(np.asarray(plane.vertices)))

print(plane_int)
'''
plane=o3d.io.read_triangle_mesh('plane_segments\plane_segment_8_mesh.stl')
#o3d.visualization.draw_geometries([plane])

bounding_box=plane.get_oriented_bounding_box()

#min_bounding_box=plane.get_minimal_oriented_bounding_box()

#bounding_box.color=[1,0,0]

#min_bounding_box.color=(0,1,0)
#o3d.visualization.draw_geometries([plane,bounding_box,min_bounding_box])


# frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
# transformation=np.eye(4)
# transformation[:3,:3]=bounding_box.R
# transformation[:3, 3]=bounding_box.center
# frame.transform(transformation)
#o3d.visualization.draw_geometries([plane,bounding_box,frame])



bblen=list(bounding_box.extent)
primary_axis_index=bblen.index(max(bblen))
popped_bblen=bblen[0:primary_axis_index]+bblen[primary_axis_index+1:]
secondary_axis_index=bblen.index(max(popped_bblen))
tertiary_axis_index=3-(primary_axis_index+secondary_axis_index)



rot=bounding_box.R
cent= bounding_box.center 
primary_axis=bounding_box.R[:,primary_axis_index]
secondary_axis=bounding_box.R[:,secondary_axis_index]
tertiary_axis=bounding_box.R[:,tertiary_axis_index]

#plane equation is:  normal * ([x,y,z]-cent)=0 where * is the dot product and x,y,z are coordiantes on the plane
#since these are really just the plane normal to the secondary axis we just fill in plane along primary(u) and tertiary(v) axis

plane_points=[]
#loop through all the options 
primary_axis_length=bblen[primary_axis_index]
secondary_axis_length=bblen[secondary_axis_index]
tertiary_axis_length=bblen[tertiary_axis_index]
probe_width=66.22 #unit in mm like the rest of the script
# probe_pass_area is secondary_axis_width - 1 probe width becuase there is extra half probe length covered on both the 1st and last passes
probe_pass_area=secondary_axis_length-probe_width

# num passes is probe_pass_area/probe_width rounded up for full, slightly overlapping coverage
#TODO Can be used to enforce certain amount of overlap as well (future work)


#define the primary and teriary axis and number of points along both
u=rot[:,primary_axis_index]
v=rot[:,tertiary_axis_index]
num_u=int(primary_axis_length//10)
num_v=int(tertiary_axis_length//1)
range_u=np.linspace(-int(primary_axis_length/2)-10,int(primary_axis_length/2)+10,num_u )
range_v=np.linspace(-int(tertiary_axis_length/2)-10,int(tertiary_axis_length/2+10),num_v)

points=[]
off=-secondary_axis_length/2+probe_width/2
offset=rot[:,secondary_axis_index]*off
for i in range_u:
    for j in range_v:
        point=i*u+j*v+offset+cent
        points.append(point)
points=np.array(points)

# print(points)
# point_cloud = o3d.geometry.PointCloud()  
# point_cloud.points = o3d.utility.Vector3dVector(points)
#o3d.visualization.draw_geometries([plane,bounding_box,frame,point_cloud])
faces=[]
for i in range(len(range_u)-1):
    for j in range(len(range_v)-1):
        idx1 = i * num_v + j
        idx2 = i * num_v + (j + 1)
        idx3 = (i + 1) * num_v + j
        idx4 = (i + 1) * num_v + (j + 1)
        #Form the two triangles from the one square
        faces.append([idx1, idx2, idx3])
        faces.append([idx2, idx4, idx3])
faces = np.array(faces)

#form o3d tensor geometry (technically not needed)
vertices_o3d = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
triangles_o3d = o3d.core.Tensor(faces, dtype=o3d.core.Dtype.Int32)
slice = o3d.t.geometry.TriangleMesh(vertices_o3d, triangles_o3d)
slice.compute_triangle_normals()

#Ray Cast to test for intersection
tensor_plane = o3d.t.geometry.TriangleMesh.from_legacy(plane)
tensor_plane.compute_triangle_normals()
scene = o3d.t.geometry.RaycastingScene()
tensor_cast_id = scene.add_triangles(tensor_plane)
LocVec=[]
for i in range(len(range_u)):
    idx=i*num_v + len(range_v)-1
    tertiary_vector=-1*rot[:,tertiary_axis_index]
    LocVec.append(np.concatenate((points[idx],tertiary_vector)))
LocVec=o3d.core.Tensor(np.array(LocVec).astype(np.float32))

ans = scene.cast_rays(LocVec)
#Distance to intersection from "start-point"
#print(ans['t_hit'].numpy())
#Geometric ID (whatever that means)
#print(ans['geometry_ids'].numpy())
#index of triangle hit
#print(ans['primitive_ids'].numpy())
#barycetnric coordinates of hit-points within hit-triangles
#print(ans['primitive_uvs'].numpy())
#Normals of the hit trianges
#print(ans['primitive_normals'].numpy())

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(plane)


for i in range(len(range_u)):
    if ans['geometry_ids'][i]==0:
        dist=ans['t_hit'].numpy()[i]
        delta=dist*tertiary_vector
        onSurface=LocVec.numpy()[i][:3]+delta
        #pose=ans['primitive_normals'].numpy()[i]
        intersection_index=ans['primitive_ids'].numpy()[i]
        print(f"Intersected triangle index: {intersection_index}")
        #make sure to use the actual plane here lol
        average_normal = compute_average_normal_t(tensor_plane, intersection_index)
        print(f"Average normal of all neighbors: {average_normal}")
        
        line_points = [onSurface, onSurface+average_normal*30]
        line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector([[0, 1]]))
        line_set.paint_uniform_color([0, 1, 0])  # Red color for the rays
        vis.add_geometry(line_set)
        





# Visualize rays
raynp=LocVec.numpy()
#print(raynp)
for ray in raynp:
    origin = ray[:3]
    direction = ray[3:]
    line_points = [origin, origin + direction * 30]  # Extend the ray for visualization
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    line_set.paint_uniform_color([1, 0, 0])  # Red color for the rays
    vis.add_geometry(line_set)

# Render the scene
vis.run()
vis.destroy_window()





'''Mesh Intersection attempt
ans = slice.boolean_intersection(tensor_plane)
o3d.visualization.draw([{'name': 'intersection', 'geometry': ans}])
'''


''' Original plane formation
num_interior_passes= -int(-probe_pass_area//probe_width)
scan_lane_width=probe_pass_area/num_interior_passes
#print(probe_width/2+scan_lane_width/2+scan_lane_width+scan_lane_width+scan_lane_width/2+probe_width/2,secondary_axis_length)
j=[i for i in range(num_interior_passes+2)]
off=0
for i in j:
    bbedge=cent-secondary_axis/2
    #TODO add bbedge as refernce point instead of offset+center
    if j.index(i)==0:
        off=-secondary_axis_length/2+probe_width/2
    elif j.index(i)==1 or j.index(i)==j[-1]:
        off=off+scan_lane_width/2
    else:
        off=off+scan_lane_width
    offset=rot[:,secondary_axis_index]*off
    for u in np.linspace(-int(primary_axis_length/2+10),int(primary_axis_length/2 +10),100):
        for v in np.linspace(-int(tertiary_axis_length+10),int(tertiary_axis_length+10),int(tertiary_axis_length)):
            #creates points on plane along primary axis with tertiary for depth
            point_on_plane=offset+cent+ u*rot[:,primary_axis_index]+v*rot[:,tertiary_axis_index]
            plane_points.append(point_on_plane)
    plane_pcd=o3d.geometry.PointCloud()
    plane_pcd.points=o3d.utility.Vector3dVector(plane_points)

    plane_pcd.paint_uniform_color([0,0,1])
    o3d.visualization.draw_geometries([plane,bounding_box, plane_pcd])
'''