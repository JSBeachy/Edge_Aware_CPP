import numpy as np
import open3d as o3d
import matplotlib
import os


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

bounding_box.color=(1,0,0)
#min_bounding_box.color=(0,1,0)
#o3d.visualization.draw_geometries([plane,bounding_box,min_bounding_box])

#o3d.visualization.draw_geometries([plane,bounding_box])
bblen=list(bounding_box.extent)
primary_axis_index=bblen.index(max(bblen))
popped_bblen=bblen[0:primary_axis_index]+bblen[primary_axis_index+1:]
secondary_axis_index=bblen.index(max(popped_bblen))
tertiary_axis_index=3-(primary_axis_index+secondary_axis_index)
print(tertiary_axis)



rot=bounding_box.R
cent= bounding_box.center 
normal=bounding_box.R[:,0]##I do not think this is the principal axis

#plane equation is:  normal * ([x,y,z]-cent)=0 where * is the dot product and x,y,z are coordiantes on the plane
#since these are really just the plane normal to the secondary axis we just fill in plane along primary(u) and tertiary(v) axis

plane_points=[]
plane_size=400
for u in np.linspace(-plane_size,plane_size,100):
    for v in np.linspace(-plane_size/2,plane_size/2,int(100/2)):
        #creates points on plane along primary axis with tertiary for depth
        point_on_plane=cent+u*rot[:,primary_axis_index]+v*rot[:,tertiary_axis_index]
        plane_points.append(point_on_plane)

plane_pcd=o3d.geometry.PointCloud()
plane_pcd.points=o3d.utility.Vector3dVector(plane_points)

plane_pcd.paint_uniform_color([0,0,1])
o3d.visualization.draw_geometries([plane,bounding_box, plane_pcd])