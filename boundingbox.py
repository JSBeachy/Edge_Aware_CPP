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

bounding_box.color=(1,0,0)

o3d.visualization.draw_geometries([plane,bounding_box])