import open3d as o3d
import numpy as np

class PCABounding:

    def __init__(self, file):
        self.relative_file_name=file
        self.mesh=o3d.io.read_triangle_mesh(file)
        self.bounding_box=self.mesh.get_oriented_bounding_box('QJ')
        self.index_determination()
        self.axis_determination()
    
    def PCA_Calculation(self):
        PCApoints=np.asarray(self.mesh.vertices)
        mean=np.mean(PCApoints, axis=0)
        centered_points=PCApoints-mean
        cov_matrix=np.cov(centered_points.T)
        self.PCA_eigenvals,self.PCA_eigenvecs=np.linalg.eig(cov_matrix)

    def index_determination(self):
        #based off o3d bounding box, but easily tranferable to PCA_calculation method
        bblen=list(self.bounding_box.extent)
        self.primary_axis_index=bblen.index(max(bblen))
        popped_bblen=bblen[0:self.primary_axis_index]+bblen[self.primary_axis_index+1:]
        self.secondary_axis_index=bblen.index(max(popped_bblen))
        self.tertiary_axis_index=3-(self.primary_axis_index+self.secondary_axis_index)
    
    def axis_determination(self):
        self.rot=self.bounding_box.R
        self.cent= bounding_box.center 
        self.primary_axis=bounding_box.R[:,self.primary_axis_index]
        self.secondary_axis=bounding_box.R[:,self.secondary_axis_index]
        self.tertiary_axis=bounding_box.R[:,self.tertiary_axis_index]


plane=o3d.io.read_triangle_mesh('plane_segments\plane_segment_8_mesh.stl')
bounding_box=plane.get_oriented_bounding_box()







