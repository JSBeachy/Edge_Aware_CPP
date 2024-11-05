import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
from collections import Counter
import itertools

class PCABounding:
    tolerance = 1e-5  # Adjust tolerance as needed

    def __init__(self, file):
        self.relative_file_name=file
        self.mesh=o3d.io.read_triangle_mesh(file)
        self.mesh.merge_close_vertices(self.tolerance)
        self.bounding_box=self.mesh.get_oriented_bounding_box()
        self.index_determination()
        self.axis_determination()
        self.PCApoints=np.asarray(self.mesh.vertices)
        self.PCA_Calculation()
    
    def PCA_Calculation(self):
        mean=np.mean(self.PCApoints, axis=0)
        centered_points=self.PCApoints-mean
        cov_matrix=np.cov(centered_points.T)
        eigenvals, eigenvecs=np.linalg.eig(cov_matrix)
        idx = eigenvals.argsort()[::-1]  
        self.PCA_eigenvals = eigenvals[idx]
        self.PCA_eigenvecs= eigenvecs[:,idx]
        for i in range(self.PCA_eigenvecs.shape[1]):
            if self.PCA_eigenvecs[:,i].sum()<0:
                self.PCA_eigenvecs[:,i] *= -1

    def PCA_projection(self, num_dimentions=2, points_to_project=None): #Project the data into the n-most relevent dimentions
        if np.all(points_to_project)==None:
            points_to_project=self.PCApoints
        PCA_eigenvecs2D=self.PCA_eigenvecs[:,:num_dimentions]
        if np.shape(points_to_project)[1]==np.shape(self.PCApoints)[1]:
            return points_to_project@PCA_eigenvecs2D
        else:
            print("Incorrect point dimentions for projection")
            return

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


class Best_Fit_CPP(PCABounding):

    def __init__(self, file):
        super().__init__(file)
        self.boundary_vertices_coords=None
        self.PCA_pointsND=None
        self.corner_points=None
        self.aligned_edges=None
        #self.boundary_pcd=o3d.geometry.PointCloud()

    def boundary_edge_finder(self):
        # o3d.get_non_manifold_edges() is roughly 4x faster, if possible use it
        triangles=np.asarray(self.mesh.triangles)
        edges = [tuple(sorted((triangle[i], triangle[j]))) for triangle in triangles for i, j in [(0, 1), (1, 2), (2, 0)]]       
        self.edge_counts=Counter(edges)
        boundary_edges=[edge for edge in self.edge_counts if self.edge_counts[edge]==1]
        return boundary_edges

    def find_convex_hull(self, num_dim, point_set):
        #project into 2D
        self.PCA_pointsND=super().PCA_projection(num_dim, point_set)
        self.hull_index=ConvexHull(self.PCA_pointsND).vertices
        self.hull_verticesND=self.PCA_pointsND[self.hull_index]
        self.hull_vertices=point_set[self.hull_index]
        #Defines both 2D boundary vertices and 3D vertices
        return

    def find_corner_points(self, number_of_corners=4):
        linscore=[]
        #Calculate "linearity" score for groups of 3 consecutive points
        for index,val in enumerate(self.hull_verticesND):
            prev_point=self.hull_verticesND[index-1]
            curr_point=val
            next_point=self.hull_verticesND[(index+1)%len(self.hull_verticesND)]
            vec1=curr_point-prev_point
            vec2=next_point-curr_point
            linscore.append(self.linearity_score(vec1,vec2))
        noise = np.random.uniform(1e-4, 1e-5, len(linscore)) 
        linscore=linscore+noise #to avoid issues with exactly matching scores
        paired=sorted(zip(linscore,self.hull_vertices))
        re_ordered_pairs=[coord for score,coord in paired]
        #Take the #corners-least linear points (lowest dot products) as corners
        #Returns corner points in 3D, for ND, would need to re-calculate paired
        self.corner_points=re_ordered_pairs[:number_of_corners]
        return
    
    def find_primary_scanning_edges(self):
        edges=list(itertools.combinations(self.corner_points,2)) #tuples of arrays for valid edges
        edges_lists=[[vertex.tolist() for vertex in edge] for edge in edges]
        edgescore=[]
        vec2=self.primary_axis
        for index,val in enumerate(edges):
            vec1=val[1]-val[0]
            edgescore.append(self.linearity_score(vec1,vec2))
        
        noise = np.random.uniform(1e-4, 1e-5, len(edgescore)) 
        edgescore=edgescore+noise #to avoid issues with exactly matching scores
        paired=sorted(zip(edgescore, edges_lists), reverse=True) #sort pairwise points by pairings
        re_ordered_pairs=[edge for score,edge in paired] 
        #returns the edges(lines between corners that) best match the principal axis
        self.aligned_edges=re_ordered_pairs[:2]
        return


    def linearity_score(self, vec1,vec2): #returns absolute value of dot-product between two vectors
        cos_theta= np.dot(vec1,vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
        return abs(cos_theta)

    def fitting(self, len_my_list, edge_list):
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

    


plane=o3d.io.read_triangle_mesh('plane_segments\plane_segment_8_mesh.stl')
bounding_box=plane.get_oriented_bounding_box()







