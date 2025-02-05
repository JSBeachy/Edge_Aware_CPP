import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull, KDTree
from collections import Counter
import itertools
import math

class PCABounding:
    tolerance = 1e-5  # Adjust tolerance as needed

    def __init__(self, file): 
        #Initializes the parent class, core functionality includes importing mesh as o3d object and finding the bounding box and PCA of the mesh
        #Can be used for any mesh/PCA calculation
        self.relative_file_name=file
        self.mesh=o3d.io.read_triangle_mesh(file)
        self.mesh.merge_close_vertices(self.tolerance) #Merges redunant mesh verticies (we found this to be a necessary step for zivid scans)
        #self.bounding_box=self.mesh.get_oriented_bounding_box()
        self.bounding_box=self.mesh.get_minimal_oriented_bounding_box()
        self.bounding_box.color=([1,0,0])
        self.index_determination()
        self.axis_determination()
        self.PCApoints=np.asarray(self.mesh.vertices)
        self.PCA_Calculation()
        self.points=np.asarray(self.mesh.vertices)
        self.kd_tree=KDTree(self.points)
    
    def PCA_Calculation(self):
        #Standard PCA calculation with mean-centered data
        mean=np.mean(self.PCApoints, axis=0)
        centered_points=self.PCApoints-mean
        cov_matrix=np.cov(centered_points.T)
        eigenvals, eigenvecs=np.linalg.eig(cov_matrix)
        idx = eigenvals.argsort()[::-1]  
        #Add the PCA components to the object as attributes
        self.PCA_eigenvals = eigenvals[idx]
        self.PCA_eigenvecs= eigenvecs[:,idx]
        for i in range(self.PCA_eigenvecs.shape[1]): #This code ensures the coordiantes point in the positive direction as the PCA vectors are un-signed
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
        self.bb_len=self.bounding_box.extent[self.primary_axis_index]
    
    def axis_determination(self):
        #Adds the PCA results
        self.rot=self.bounding_box.R
        self.cent= self.bounding_box.center 
        self.primary_axis=self.bounding_box.R[:,self.primary_axis_index]
        self.secondary_axis=self.bounding_box.R[:,self.secondary_axis_index]
        self.tertiary_axis=self.bounding_box.R[:,self.tertiary_axis_index]


class Best_Fit_CPP(PCABounding):

    def __init__(self, mesh):
        #Initializes the child Best_Fit_CPP class, which contains all functionality of the PCABoudning parent class with added path planning functionality
        #Object attributes are declared below (as null) and then assigned in the functions
        super().__init__(mesh)
        self.boundary_vertices_coords=None
        self.PCA_pointsND=None
        self.corner_points=None
        self.aligned_edges=None
        self.edge1_vec=None
        self.edge1_cent=None
        self.edge2_vec=None
        self.edge2_cent=None
        self.probe_width=64
        self.tot_width=None
        self.tot_passes=None
        self.per_pass_width=None
        self.real_per_pass_width=None
        self.read_distance=None
        self.edge_offset=None
        self.offset_dir=None
        self.scan_lines=None
        self.colors= np.full((self.points.shape[0], 3), (1,0,0))

    def boundary_edge_finder(self):
        # o3d.get_non_manifold_edges() is roughly 4x faster, if possible use it
        # this function is not included in main script
        triangles=np.asarray(self.mesh.triangles)
        edges = [tuple(sorted((triangle[i], triangle[j]))) for triangle in triangles for i, j in [(0, 1), (1, 2), (2, 0)]]       
        self.edge_counts=Counter(edges)
        boundary_edges=[edge for edge in self.edge_counts if self.edge_counts[edge]==1]
        return boundary_edges

    def find_convex_hull(self, num_dim, point_set):
        #project point_set into N-Dimentions
        self.PCA_pointsND=super().PCA_projection(num_dim, point_set)
        self.hull_index=ConvexHull(self.PCA_pointsND).vertices
        #Find the ND (will always be 2D) and 3D points of convex hull
        self.hull_verticesND=self.PCA_pointsND[self.hull_index]
        self.hull_vertices=point_set[self.hull_index]
        #Defines both 2D boundary vertices and 3D vertices in convex hull as object attribues
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
        linscore=linscore+noise #Add random noise to avoid issues with exactly matching scores
        paired=sorted(zip(linscore,self.hull_vertices))
        pairedND=sorted(zip(linscore,self.hull_verticesND))
        re_ordered_pairs=[coord for score,coord in paired]
        re_ordered_pairsND=[coord for score,coord in pairedND]
        #Take the #corners-least linear points (lowest dot products) as corners
        #Returns corner points in 3D, for ND, would need to re-calculate paired
        self.corner_points=re_ordered_pairs[:number_of_corners]
        self.corner_pointsND=re_ordered_pairsND[:number_of_corners]
        return
    
    def find_primary_scanning_edges(self):
        #Iterates through all possible "principal edges" that can be made with corner points
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
        #returns the edges(lines between corners that) best align with the principal axis
        self.aligned_edges=re_ordered_pairs[:2]
        return


    def linearity_score(self, vec1,vec2): #returns absolute value of dot-product between two vectors
        cos_theta= np.dot(vec1,vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
        return abs(cos_theta)

    def splitting(self, edge_list, hull_vertices_list):
            #splits the points between the two relevant corner points
            len_my_list=len(hull_vertices_list)
            direction=-1
            index=sorted([hull_vertices_list.index(i) for i in edge_list])
            dir1_len = (index[0] - index[1]) % len_my_list +1
            dir2_len = (index[1] - index[0]) % len_my_list +1

            ''' #Old-way of doing this based on num. points between in each direction
            length=min(clockwise,counterclockwise)
            if counterclockwise<=clockwise:
                direction=-1
            group=[hull_vertices_list[(index[0] + i*direction)] for i in range(length+1)]
            '''
            midpoint1=(index[1]+((index[0]-index[1]+1)%len_my_list)//2)%len_my_list 
            midpoint2=(index[0]+((index[1]-index[0]+1)%len_my_list)//2)%len_my_list 

            true_vec=np.asarray(edge_list[1])-np.asarray(edge_list[0])
            vector1=np.asarray(hull_vertices_list[midpoint1])-np.asarray(edge_list[0])
            vector2=np.asarray(hull_vertices_list[midpoint2])-np.asarray(edge_list[0])
            
            direction1=self.linearity_score(true_vec,vector1)
            direction2=self.linearity_score(true_vec,vector2)
            if direction1<direction2:
                direction=1
                group=[hull_vertices_list[(index[0] + i*direction)] for i in range(dir2_len)]
            else:
                group=[hull_vertices_list[(index[0] + i*direction)] for i in range(dir1_len)]
            
            #print(true_vec)
            #print(f"base point at: {edge_list[0]}")
            #print(f"length around dir2 direction: {dir2_len}")
            #print(f"length around dir1 direction: {dir1_len}")
            #print(f"midpoint in dir2 direction is {midpoint2}")
            #print(f"midpoint in dir1 direction is {midpoint1}")
            #print(vector1, vector2)
            #print(f"linearity in dir1: {direction1}, linearity in dir2 {direction2} ")
            #print(f"group of edge points is {group}")
            
            return np.asarray(group)
    
    def fit_line_3d(self, points):
        #Use SVD to find principal direction in 1D (not set up to deal with curving segments yet)
        centroid=np.mean(points, axis=0)
        _,_,v=np.linalg.svd(points-centroid)
        direction_vector=v[0]
        corner_midpoint=0.5*(points[0]+points[-1])
        #P=true midpoint, A=base corner point, N is direction vector (see point onto vector projection)
        aminusp=points[0]-corner_midpoint
        vec_to_line=aminusp-aminusp.dot(direction_vector)*direction_vector
        true_centroid=corner_midpoint+vec_to_line
        #print(f"true midpoint: {corner_midpoint}, Edge points midpoint: {centroid}")
        #print(f"along vector {direction_vector} with magnitue {np.linalg.norm(direction_vector)}")
        #print(f"vector to line is {vec_to_line}, putting the vector_midpoint at {corner_midpoint+vec_to_line}")
        
        return true_centroid, direction_vector
    
    def point_creator(self, cent, vec, num_points, prop=5):
        #fills out a vector from the center in both diretions, typically the points are spaced by roughly 1 mm
        length=self.bb_len
        prop_len=int(length/prop)
        points=np.array([cent + t * vec for t in np.linspace(-length/2-prop_len, length/2+prop_len, num_points)])
        return points
    
    def bounding_box_interior_points(self, points):
        #should take np.array of points, finds the points closed to the bounding box, but still inside
        bb_min=self.bounding_box.get_min_bound()
        bb_max=self.bounding_box.get_max_bound()
        p1=np.asarray(points)
        inside_bbox = (p1 >= bb_min) & (p1 <= bb_max) # & for element-wise logic in numpy
        within_bbox = np.all(inside_bbox, axis=1)
        filtered_points=p1[within_bbox]
        if filtered_points.size > 0:
            #print( filtered_points[0] ) # Return the first point inside the bounding box
            #print( filtered_points[-1] ) # Return the last point inside the bounding box
            return(filtered_points[0],filtered_points[-1])
        else:
            print("No Points in Bounding Box")
            return None 
    
    def scan_width_determination(self,line_pcd1,line_pcd2):
        #Find the points on each line closest to bounding box boundaries
        p0_0,p0_1=self.bounding_box_interior_points(line_pcd1)
        p1_0,p1_1=self.bounding_box_interior_points(line_pcd2)
        #Calculate all between other two corners for each point, take third distance (primary axis distance should be largest 2)
        scan_width=[np.linalg.norm(p0_0-p1_0),np.linalg.norm(p0_1-p1_1), np.linalg.norm(p0_0-p1_1),np.linalg.norm(p0_1-p1_0)]
        #print(scan_width)
        return sorted(scan_width)[-3]
    
    def print_scan_information(self):
        #Prints the Scan details to main console
        print(f"Total width to be scanned: {self.tot_width}")
        print(f"Width allocated to each pass: {self.per_pass_width}")
        print(f"Number of total passes: {self.tot_passes}")
        real_distance=self.tot_width-2*self.probe_width
        print(f"Number of passes remaining after setting edges: {math.ceil(real_distance/self.probe_width)}")
        print(f"Approximate pass width after setting edges: {self.real_per_pass_width}")

    def scan_information(self, probe_width, scan_line1, scan_line2, edge_offset=None):
        #calculates the necessary number of passes for the width, and the resulting real-pass width
        if type(probe_width)==int or type(probe_width)==float:
            self.probe_width=probe_width
        else:
            print("Invalid probe width entered")
        half_probe=self.probe_width/2
        
        #Finds width of scan (only at corner points, should be fine for linear fits, wouldn't work for curved scanning edges)
        self.tot_width=self.scan_width_determination(scan_line1,scan_line2)
        self.tot_passes=math.ceil(self.tot_width/self.probe_width)
        self.per_pass_width=round(self.tot_width/self.tot_passes,2)
        #setting both edges
        #with edges set, calculate updated stats
        self.real_distance=self.tot_width-2*self.probe_width
        self.real_passes_left=math.ceil(self.real_distance/self.probe_width)
        if self.real_passes_left>0:
            self.real_per_pass_width=self.real_distance/self.real_passes_left
        else:
            self.real_per_pass_width="No Interior Passes"
        if edge_offset==None:
            self.offset_one=half_probe
            offset_two=self.tot_width-self.offset_one
        else:
            self.offset_one=edge_offset
            offset_two=self.tot_width-self.offset_one

    def shift_direction(self):
        #Shifts both edges "in" along the secondary axis by one 1/2 probe width
        direction1=self.edge1_cent-self.cent
        direction2=self.edge2_cent-self.cent
        dot_1=np.dot(direction1/np.linalg.norm(direction1),self.secondary_axis)
        dot_2=np.dot(direction2/np.linalg.norm(direction2),self.secondary_axis)
        if dot_1<0:
            self.offset_dir=-1
        else:
            self.offset_dir=1 

    def line_interpolator(self, line_one, line_last):
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
        #line_one, line_last are the edge_offset edges, this returns lines of paths
        color_one=np.asarray([0, 1, 0])
        color_two=np.asarray([0, 0, 1])
        if self.tot_passes==1:
            lines=[line_one]
            color=[color_one]
        elif self.tot_passes==2:
            lines=[line_one, line_last]
            color=[color_one, color_two]
        else:
            lines=[]
            color=[]
            for i in range(self.tot_passes):
                lines.append(line_one*(1-(i)/(self.tot_passes-1))+line_last*((i)/(self.tot_passes-1)))
                color.append(color_one*(1-(i)/(self.tot_passes-1))+color_two*((i)/(self.tot_passes-1)))
        self.scan_lines=lines
        return lines, color
    
    def compute_average_normal_t(self, mesh, triangle_index):
            #From Scantoplan
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






























    def scanned_area(self, trial_lines, probe_width, probe_length):
        """
        Marks triangles in the mesh that are within the rectangular probe coverage area.
        
        Parameters:
        - trial_lines: List of arrays, each containing the points of a trial line.
        - probe_width: Float, the width of the rectangular probe coverage area.
        - probe_height: Float, the height of the rectangular probe coverage area.
        """
        # Convert mesh to triangle representation
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        triangles= t_mesh.triangle.indices.numpy()
        vertices = t_mesh.vertex.positions.numpy()

        # Initialize triangle colors (default to light gray)
        colors = np.zeros((len(triangles), 3)) + 0.8

        # Define half-dimensions for rectangular coverage
        half_width = probe_width / 2
        half_length = probe_length / 2

        # Iterate through trial lines
        for line in trial_lines:
            line = np.asarray(line)
            print(len(line))
            for point in line:
                # Define bounds of the rectangular prism
                x_min, x_max = point[0] - half_width, point[0] + half_width
                y_min, y_max = point[1] - half_length, point[1] + half_length
                z_min, z_max = point[2] - half_width, point[2] + half_width  # not implemented yet

                # Check each triangle
                for triangle_index, triangle in enumerate(triangles):
                    tri_vertices = vertices[triangle]
                    # Check if any vertex of the triangle falls within the prism
                    for vertex in tri_vertices:
                        if (
                            x_min <= vertex[0] <= x_max and
                            y_min <= vertex[1] <= y_max #and
                            #z_min <= vertex[2] <= z_max
                        ):
                            colors[triangle_index] = [1, 0, 0]  # Mark as scanned (red)
                            break  # No need to check further vertices

        # Apply the colors to the mesh
         #t_mesh.triangle.material_ids = None  # Reset material IDs
        t_mesh.triangle.colors = o3d.core.Tensor(colors, dtype=o3d.core.Dtype.Float32)
        return t_mesh


        
#plane=o3d.io.read_triangle_mesh('plane_segments\plane_segment_8_mesh.stl')
#bounding_box=plane.get_oriented_bounding_box()







