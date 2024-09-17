import numpy as np
import open3d as o3d
from robodk import robolink, robomath
import time

#Function to get the average mesh normals, make sure normals have already been computed
def compute_average_normal_t(mesh, triangle_index):
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

def build_a_mesh(num_v,range_u,range_v):
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

    return(np.array(faces))

#Start RoboDK SDK and clear any previous targets or paths
stationpath='segmentscan.rdk'
RL=robolink.Robolink()
station=RL.AddFile(stationpath)
#RL.AddFile('plane_segments\Plannertest.stl')\
RL.AddFile('plane_segments\Plannertest.stl')
time.sleep(1)
station_items=RL.ItemList()
for item in station_items:
    if item.Type()==robolink.ITEM_TYPE_TARGET or item.Type()==robolink.ITEM_TYPE_PROGRAM:
        RL.Delete(item)

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
plane=o3d.io.read_triangle_mesh('plane_segments\Plannertest.stl')
#plane=o3d.io.read_triangle_mesh('plane_segments\Plannertest.stl')
#o3d.visualization.draw_geometries([plane])
bounding_box=plane.get_oriented_bounding_box()

#Calculate PCA manually
PCApoints=np.asarray(plane.vertices)
mean=np.mean(PCApoints, axis=0)
centered_points=PCApoints-mean
cov_matrix=np.cov(centered_points.T)
eigenvalues,eigenvectors=np.linalg.eig(cov_matrix)
spaceing_eig=min(eigenvalues)

#min_bounding_box=plane.get_minimal_oriented_bounding_box()
#min_bounding_box.color=(0,1,0)
bounding_box.color=[1,0,0]


#o3d.visualization.draw_geometries([plane,bounding_box])


# frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
# transformation=np.eye(4)
# transformation[:3,:3]=bounding_box.R
# transformation[:3, 3]=bounding_box.center
# frame.transform(transformation)
#o3d.visualization.draw_geometries([plane,bounding_box,frame])



bblen=list(bounding_box.extent)
print(bounding_box.center)
print(bblen)
primary_axis_index=bblen.index(max(bblen))
popped_bblen=bblen[0:primary_axis_index]+bblen[primary_axis_index+1:]
secondary_axis_index=bblen.index(max(popped_bblen))
tertiary_axis_index=3-(primary_axis_index+secondary_axis_index)



rot=bounding_box.R
print(rot)
cent= bounding_box.center 
primary_axis=bounding_box.R[:,primary_axis_index]
secondary_axis=bounding_box.R[:,secondary_axis_index]
tertiary_axis=bounding_box.R[:,tertiary_axis_index]

#plane equation is:  normal * ([x,y,z]-cent)=0 where * is the dot product and x,y,z are coordiantes on the plane
#since these are really just the plane normal to the secondary axis we just fill in plane along primary(u) and tertiary(v) axis

plane_points=[]
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
num_u=int(primary_axis_length//1+20)
num_v=int(tertiary_axis_length//1)
#with num_u, each index represents spacing of almost exactly 1 mm
range_u=np.linspace(-int(primary_axis_length/2)-10,int(primary_axis_length/2)+10,num_u )
range_v=np.linspace(-int(tertiary_axis_length/2)-10,int(tertiary_axis_length/2+10),num_v)

#Format Scanable plane for ray-casting
tensor_plane = o3d.t.geometry.TriangleMesh.from_legacy(plane) #OGplane= Surface to Path-Plan on
tensor_plane.compute_triangle_normals()
tensor_plane.compute_vertex_normals()
scene = o3d.t.geometry.RaycastingScene()
tensor_cast_id = scene.add_triangles(tensor_plane)

#Original plane formation
num_interior_passes= -int(-probe_pass_area//probe_width)
scan_lane_width=probe_pass_area/num_interior_passes
#print(probe_width/2+scan_lane_width/2+scan_lane_width+scan_lane_width+scan_lane_width/2+probe_width/2,secondary_axis_length)
passes=[i for i in range(num_interior_passes+2)]
off=0
for i in passes:
    points=[]
    bbedge=cent-secondary_axis/2
    #TODO add bbedge as refernce point instead of offset+center
    if passes.index(i)==0:
        off=-secondary_axis_length/2+probe_width/2
    elif passes.index(i)==1 or passes.index(i)==passes[-1]:
        off=off+scan_lane_width/2
    else:
        off=off+scan_lane_width
    offset=rot[:,secondary_axis_index]*off
    for n in range_u:
        for m in range_v:
            point=n*u+m*v+offset+cent
            points.append(point)
    points=np.array(points)

    #plane_pcd=o3d.geometry.PointCloud()
    #plane_pcd.points=o3d.utility.Vector3dVector(points)
    #plane_pcd.paint_uniform_color([0,0,1])
    #o3d.visualization.draw_geometries([plane,bounding_box, plane_pcd])
    #o3d.visualization.draw_geometries([plane, bounding_box, frame, plane_pcd])


    #form o3d tensor geometry of plane slice (technically not needed)
    #faces=build_a_mesh(num_v,range_u,range_v)
    #vertices_o3d = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    #triangles_o3d = o3d.core.Tensor(faces, dtype=o3d.core.Dtype.Int32)
    #slice = o3d.t.geometry.TriangleMesh(vertices_o3d, triangles_o3d)
    #slice.compute_triangle_normals()

    #Ray Cast to test for intersection
    LocVec=[]
    for j in range(len(range_u)):
        idx=j*num_v + len(range_v)-1
        tertiary_vector=-1*rot[:,tertiary_axis_index]
        LocVec.append(np.concatenate((points[idx],tertiary_vector)))
    LocVec=o3d.core.Tensor(np.array(LocVec).astype(np.float32))
    #print(LocVec)
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

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(plane)


    
    poses=[]
    #Determine the index of ray-casting to use for RoboDK targets
    index=[p for p,q in enumerate(ans['geometry_ids']) if q==0]
    EEwidth=32
    spread=index[-(EEwidth//2-1)]-index[EEwidth//2-1]
    #Numpoints may be completely off
    numpoints=5+int(spaceing_eig//10)
    RoboIndex=np.linspace(index[EEwidth//2],index[-EEwidth//2],numpoints).round().astype(int)
    
    for k in RoboIndex:
        dist=ans['t_hit'].numpy()[k]
        delta=dist*tertiary_vector
        onSurface=LocVec.numpy()[k][:3]+delta
        #pose=ans['primitive_normals'].numpy()[i]
        intersection_index=ans['primitive_ids'].numpy()[k]
        #print(f"Intersected triangle index: {intersection_index}")
        #make sure to use the actual plane here lol
        average_normal = compute_average_normal_t(tensor_plane, intersection_index)
        #print(f"Average normal of all neighbors: {average_normal}")
        
        # line_points = [onSurface, onSurface+average_normal*30]
        # line_set = o3d.geometry.LineSet(
        # points=o3d.utility.Vector3dVector(line_points),
        # lines=o3d.utility.Vector2iVector([[0, 1]]))
        # line_set.paint_uniform_color([0, 1, 0])  # Red color for the rays
        # vis.add_geometry(line_set)

        #Creates RoboDK points
        target_pos=RL.AddTarget("Target_"+str(passes.index(i)+1)+"_"+str(k+1))

        transformation_matrix=np.eye(4)
        transformation_matrix[0:3,0]=rot[:,primary_axis_index]
        transformation_matrix[0:3,1]=rot[:,secondary_axis_index]
        transformation_matrix[0:3,2]=average_normal
        transformation_matrix[0:3,3]=onSurface

        pose=robomath.Mat(transformation_matrix.tolist())
        target_pos.setPose(pose)
        poses.append(target_pos)

    print(poses)
    print(type(poses[0]))
    program=RL.AddProgram("Path_"+str(passes.index(i)+1))
    print(i)
    if i%2!=0:
        poses=poses[::-1]
        for pose in poses:
            program.MoveL(pose)
    else:
        for pose in poses:
            program.MoveL(pose)

    
    print("Program created")

    # Visualize rays, create a Line Set from the average vectors
    raynp=LocVec.numpy()

    # for ray in raynp:
    #     origin = ray[:3]
    #     direction = ray[3:]
    #     line_points = [origin, origin + direction * 30]  # Extend the ray for visualization
    #     line_set = o3d.geometry.LineSet(
    #         points=o3d.utility.Vector3dVector(line_points),
    #         lines=o3d.utility.Vector2iVector([[0, 1]])
    #     )
    #     line_set.paint_uniform_color([1, 0, 0])  # Red color for the rays
    #     vis.add_geometry(line_set)



    # Render the scene
    #vis.run()
    #vis.destroy_window()


input("Ready to CLose?:    ")

#RL.Save('segmentscan.rdk')
RL.CloseStation()
RL.CloseRoboDK()


'''Mesh Intersection attempt
ans = slice.boolean_intersection(tensor_plane)
o3d.visualization.draw([{'name': 'intersection', 'geometry': ans}])
'''


