import numpy as np
import open3d as o3d
#import matplotlib
#import os


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

bounding_box.color=[1,0,0]

#min_bounding_box.color=(0,1,0)
#o3d.visualization.draw_geometries([plane,bounding_box,min_bounding_box])


frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
transformation=np.eye(4)
transformation[:3,:3]=bounding_box.R
transformation[:3, 3]=bounding_box.center
frame.transform(transformation)
o3d.visualization.draw_geometries([plane,bounding_box,frame])



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
plane_size=400

#loop through all the options 
secondary_axis_length=bblen[secondary_axis_index]
tertiary_axis_length=bblen[tertiary_axis_index]
probe_width=66.22 #unit in mm like the rest of the script
# probe_pass_area is secondary_axis_width - 1 probe width becuase there is extra half probe length covered on both the 1st and last passes
probe_pass_area=secondary_axis_length-probe_width

# num passes is probe_pass_area/probe_width rounded up for full, slightly overlapping coverage
#TODO Can be used to enforce certain amount of overlap as well (future work)

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
    for u in np.linspace(-plane_size,plane_size,100):
        for v in np.linspace(-int(tertiary_axis_length+10),int(tertiary_axis_length+10),int(tertiary_axis_length)):
            
            #creates points on plane along primary axis with tertiary for depth
            point_on_plane=offset+cent+ u*rot[:,primary_axis_index]+v*rot[:,tertiary_axis_index]
            plane_points.append(point_on_plane)
    plane_pcd=o3d.geometry.PointCloud()
    plane_pcd.points=o3d.utility.Vector3dVector(plane_points)

    plane_pcd.paint_uniform_color([0,0,1])
    o3d.visualization.draw_geometries([plane,bounding_box, plane_pcd])