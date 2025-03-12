import open3d as o3d
import trimesh
import numpy as np
from scipy.spatial import KDTree
from scipy.special import binom
from scipy.integrate import quad
from scipy.optimize import newton
from collections import Counter
import math
import time
import cvxpy as cp

class PCABounding:
    tolerance = 1e-5  # Adjust tolerance as needed

    def __init__(self, file): 
        #Initializes the parent class, core functionality includes importing mesh as o3d object and finding the bounding box and PCA of the mesh
        #Can be used for any mesh/PCA calculation
        self.relative_file_name=file
        self.mesh=o3d.io.read_triangle_mesh(file)
        self.mesh.merge_close_vertices(self.tolerance) #Merges redunant mesh verticies (we found this to be a necessary step for zivid scans)
        self.points=np.asarray(self.mesh.vertices)
        self.kd_tree=KDTree(self.points)
        self.bounding_box=self.mesh.get_oriented_bounding_box()
        #TODO: add custom PCA axis-aligned bounding_box
        #self.bounding_box=self.mesh.get_minimal_oriented_bounding_box()
        self.bounding_box.color=([1,0,0])
        self.index_determination()
        self.axis_determination()
        self.PCA_Calculation()

    
    def PCA_Calculation(self):
        #Standard PCA calculation with mean-centered data
        mean=np.mean(self.points, axis=0)
        centered_points=self.points-mean
        cov_matrix=np.cov(centered_points.T)
        eigenvals, eigenvecs=np.linalg.eig(cov_matrix)
        idx = eigenvals.argsort()[::-1]  
        #Add the PCA components to the object as attributes
        self.PCA_eigenvals = eigenvals[idx]
        self.PCA_eigenvecs= eigenvecs[:,idx]
        for i in range(self.PCA_eigenvecs.shape[1]): #This code ensures the coordiantes point in the positive direction as the PCA vectors are un-signed
            if self.PCA_eigenvecs[:,i].sum()<0:
                self.PCA_eigenvecs[:,i] *= -1

    def PCA_projection(self, points_to_project=None, num_dimentions=2,): #Project the data into the n-most relevent dimentions
        if np.all(points_to_project)==None:
            points_to_project=self.points
        if num_dimentions>3:
            num_dimentions=input("Too many dimensions (for 3D data), Please enter (integer) 1,2, or 3")
        PCA_eigenvecs2D=self.PCA_eigenvecs[:,:num_dimentions]
        if np.shape(points_to_project)[1]==np.shape(self.points)[1]:
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
        if np.sum(self.primary_axis)<0: self.primary_axis=self.primary_axis*-1 
        self.secondary_axis=self.bounding_box.R[:,self.secondary_axis_index]
        if np.sum(self.secondary_axis)<0: self.secondary_axis=self.secondary_axis*-1 
        self.tertiary_axis=self.bounding_box.R[:,self.tertiary_axis_index]
        if np.sum(self.tertiary_axis)<0: self.tertiary_axis=self.tertiary_axis*-1 

class Best_Fit_CPP(PCABounding):

    def __init__(self, mesh):
        #Initializes the child Best_Fit_CPP class, which contains all functionality of the PCABoudning parent class with added path planning functionality
        #Object attributes are declared below (as null) and then assigned in the functions
        super().__init__(mesh)
        self.Bezier_order=6
        self.Sample_num=10
        self.ordered_edge_points=None
        self.ordered_edge_points2D=None
        self.corner_points2D=None
        self.edges=None
        self.edge1_CP=None
        self.edge2_CP=None
        self.probe_width=64
        self.tot_width=None
        self.tot_passes=None
        self.per_pass_width=None
        self.real_per_pass_width=None
        self.read_distance=None
        self.edge_offset=None
        self.offset_dir=None
        self.colors= np.full((self.points.shape[0], 3), [1,0,0])
        self.passes=None
        self.passes_colors=None

    def curvilinear_distance(self, segment):
        diffs=np.diff(segment,axis=0)
        distance=np.linalg.norm(diffs,axis=1)
        return np.sum(distance)

    def boundary_edge_finder(self):
        # o3d.get_non_manifold_edges() is roughly 4x faster, if possible use it
        # this function is not included in main script
        triangles=np.asarray(self.mesh.triangles)
        edges = [tuple(sorted((triangle[i], triangle[j]))) for triangle in triangles for i, j in [(0, 1), (1, 2), (2, 0)]]       
        self.edge_counts=Counter(edges)
        boundary_edges=[edge for edge in self.edge_counts if self.edge_counts[edge]==1]
        return boundary_edges
    
    def order_perimeter(self, points, clockwise=True):
        #Sort the 2D projection of edge points into a clockwise order, send back
        centroid=np.mean(points,axis=0)
        angles=np.arctan2(points[:,1]-centroid[1], points[:, 0]-centroid[0])
        sort_order = np.argsort(angles)
        if clockwise:
            sort_order = sort_order[::-1] 

        return points[sort_order], sort_order
    
    def dot_product(self, points):
        forward_vecs=np.roll(points,shift=1, axis=0)-points
        backward_vecs=np.roll(points,shift=-1,axis=0)-points
        forward_vecs /= np.linalg.norm(forward_vecs, axis=1, keepdims=True)
        backward_vecs /= np.linalg.norm(backward_vecs, axis=1, keepdims=True)
        #einstine summation notation, more efficient than np.sum(f*b)
        dot_products=np.einsum("ij,ij->i",forward_vecs,backward_vecs)

        return dot_products

    def split_perimeter(self, ordered_points, corner_indices):
        corner_indices = np.sort(corner_indices)  # Ensure indices are sorted
        segments=[]
        for i in range(len(corner_indices)):
            start_idx=corner_indices[i]
            end_idx=corner_indices[(i+1) % len(corner_indices)]
            if start_idx<end_idx:
                segment= ordered_points[start_idx: end_idx+1]
            else:
                segment=np.vstack([ordered_points[start_idx:],ordered_points[:end_idx+1]])
            segments.append(segment)
        return segments

    
    def boundary_edge_calculations(self):
        edges_trial = self.mesh.get_non_manifold_edges(allow_boundary_edges=False)
        edge_segments=self.points[edges_trial]
        edge_points=np.unique(edge_segments.reshape(-1,3), axis=0)

        proj_points=super().PCA_projection(edge_points)
        self.ordered_edge_points2D, clockwise_order=self.order_perimeter(proj_points)
        self.ordered_edge_points=edge_points[clockwise_order]
        dps=self.dot_product(self.ordered_edge_points2D)

        #Assuming 4 corners, can change to threshold here, prompt users for corners then
        #mask = np.abs(dps) < threshold
        #theshold_dps=ordered[mask]
        min_indices = np.argsort(np.abs(dps))[:4]
        self.corner_points2D=self.ordered_edge_points2D[min_indices]
        split=self.split_perimeter(self.ordered_edge_points, min_indices)
        segment_len=np.array([self.curvilinear_distance(segment) for segment in split])
        top_two_indices = np.argsort(segment_len)[-2:][::-1]
        self.edges=[split[top_two_indices[0]], split[top_two_indices[1]][::-1]]
        return

    def fit_curve3d(self, curve, Bezier_order=6, sample_points=10):
        # Define Bézier degree and number of sample points
        n = Bezier_order
        n_opt = sample_points

        Bpatch=np.zeros([n+1,n+1])
        for i in range(n+1):
            for j in range(n+1):
                Bpatch[i, j]=(-1)**(j-i)*binom(n,j)*binom(j,i)
        
        # Optimization variables for X, Y, and Z coordinates of Bézier Control Points
        Pix = cp.Variable(n+1)
        Piy = cp.Variable(n+1)
        Piz = cp.Variable(n+1)    

        projection = [i/n_opt for i in range(n_opt+1)]
        cost = 0
        
        #Curve should be Nx3 array of edge points
        control_points=np.zeros([n_opt+1,3])
        for i in range(n_opt+1):
            if i==n_opt:
                control_points[i]=curve[-1]
            else:
                index=(i*len(curve))//n_opt
                control_points[i]=curve[index,:]

        # Compute Bézier coefficients
        Kx = Bpatch.T @ Pix
        Ky = Bpatch.T @ Piy
        Kz = Bpatch.T @ Piz

        # Compute optimization cost
        for i in range(n_opt+1):
            v = projection[i]
            V = np.array([v**k for k in range(n+1)])
            rsx = V @ Kx
            rsy = V @ Ky
            rsz = V @ Kz
            cost += cp.square(cp.norm(cp.hstack([rsx - control_points[i][0], 
                                                rsy - control_points[i][1], 
                                                rsz - control_points[i][2]])))

        # Solve the optimization problem
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve()

        # Extract optimized control points
        Pi = np.column_stack([Pix.value, Piy.value, Piz.value])

        return Pi

    def bezier_curve3N(self, n, t, control_points):
        if not (0 <= t <= 1):
            raise ValueError("t must be in the range (0,1)")
        if control_points.shape != (n + 1, 3):
            raise ValueError("control_points must be an (n+1) x 3 array")
        point = np.zeros(3)
        for i in range(n + 1):
            bernstein = binom(n, i) * (1 - t) ** (n - i) * t ** i
            point += bernstein * control_points[i]

        return point

    def edge_fitter(self, edge_set, B_order=6, sample_num=10):
        self.Bezier_order=B_order
        self.Sample_num=sample_num
        self.edge1_CP= self.fit_curve3d(edge_set[0], Bezier_order=B_order, sample_points=sample_num)
        self.edge2_CP= self.fit_curve3d(edge_set[1], Bezier_order=B_order, sample_points=sample_num)
        #TODO: could add a sampler here to get curves for display
        return

    def mesh_slicer(self, plane_normal, plane_origin, iterations=False, step=0):
        max_distance = 0.0
        intersection = True

        while intersection:
            slice_result = trimesh.intersections.mesh_plane(self.trimesh, plane_normal, plane_origin)

            if len(slice_result) == 0:
                intersection = False
                
            else:
                points=np.unique(np.vstack(slice_result), axis=0)
                # Sort points along the secondary axis
                sorted_indices = np.argsort(points @ self.secondary_axis)
                points = points[sorted_indices]
                # Compute distances
                diffs = np.diff(points, axis=0)
                distances = np.linalg.norm(diffs, axis=1)
                total_distance = np.sum(distances)  # Total length of the intersection curve

                max_distance = max(max_distance, total_distance)
                if iterations:
                    return max_distance
                
                plane_origin=plane_origin + step * plane_normal 
        
        return max_distance  

    def mesh_slice_preperation(self, step_size=10):
        self.trimesh=trimesh.Trimesh(vertices=self.points, faces=np.asarray(self.mesh.triangles))
        plane_origin=self.bounding_box.center
        plane_normal=self.primary_axis
        max_forward_distance = self.mesh_slicer(plane_normal ,plane_origin, step=10)
        max_backward_distance = self.mesh_slicer(plane_normal, plane_origin, step=-10)        
        self.max_distance = max(max_forward_distance, max_backward_distance)
        return self.max_distance
    

    def print_scan_information(self):
        #Prints the Scan details to main console
        print(f"Total width to be scanned: {self.tot_width}")
        print(f"Width allocated to each pass: {self.per_pass_width}")
        print(f"Number of total passes: {self.tot_passes}")
        real_distance=self.tot_width-2*self.probe_width
        print(f"Number of passes remaining after setting edges: {math.ceil(real_distance/self.probe_width)}")
        print(f"Approximate pass width after setting edges: {self.real_per_pass_width}")


    def scan_information(self, probe_width, step_size, edge_offset=None):
        #calculates the necessary number of passes for the width, and the resulting real-pass width
        if type(probe_width)==int or type(probe_width)==float:
            self.probe_width=probe_width
        else:
            print("Invalid probe width entered")
        half_probe=self.probe_width/2
        
        #Finds curvilinear length of mesh at interval of 10mm
        self.tot_width=self.mesh_slice_preperation(step_size)
        self.tot_passes=math.ceil(self.tot_width/self.probe_width)
        self.per_pass_width=round(self.tot_width/self.tot_passes,2)

        #with edges set, calculate updated stats
        self.real_distance=self.tot_width-2*self.probe_width
        self.real_passes_left=math.ceil(self.real_distance/self.probe_width)
        if self.real_passes_left>0:
            self.real_per_pass_width=self.real_distance/self.real_passes_left
        else:
            self.real_per_pass_width="No Interior Passes"
        if edge_offset==None:
            self.offset_one=half_probe
        else:
            self.offset_one=edge_offset


    def shift_direction(self):
        #Shifts both edges "in" along the secondary axis by one 1/2 probe width
        direction1=np.mean(self.edge1_CP, axis=0)-self.cent
        direction2=np.mean(self.edge2_CP, axis=0)-self.cent
        dot_1=np.dot(direction1/np.linalg.norm(direction1),self.secondary_axis)
        if dot_1<0:
            self.offset_dir=-1
        else:
            self.offset_dir=1 

    def bezier_derivative(self, n, t, control_points):
        #Compute P'(t) for a Bezier curve
        derivative = np.zeros(3)
        for i in range(n):
            bernstein = n * binom(n-1, i) * (1-t)**(n-1-i) * t**i
            derivative += bernstein * (control_points[i+1] - control_points[i])
        return derivative
    
    def arc_length_integrand(self,t, n, control_points):
        #Returns ||P'(t)|| for integration
        return np.linalg.norm(self.bezier_derivative(n, t, control_points))

    def bezier_arc_length(self, n, t_start, t_end, control_points):
        #Compute arc length S(t) from t_start to t_end using numerical integration. """
        return quad(self.arc_length_integrand, t_start, t_end, args=(n, control_points))[0]

    def find_t_newton(self, n, control_points, target_length, tol=1e-5):
        cumulative_t=[]
        t_conservative=0.2
        t_min=0
        while t_min<=1:
            cumulative_t.append(t_min)
            def f(t): return self.bezier_arc_length(n, t_min, t, control_points) - target_length
            def f_prime(t): return np.linalg.norm(self.bezier_derivative(n,t,control_points))
            if len(cumulative_t)==2:
                t_conservative=cumulative_t[1]-cumulative_t[0]
            t_guess=min(1,t_conservative+t_min)
            t_new = newton(f, t_guess, f_prime, tol=tol)
            t_min=t_new
        cumulative_t.append(1)    
        
        return cumulative_t

    def line_interpolator(self, interp_dist):
        color_one=np.asarray([0, 1, 0])
        color_two=np.asarray([0, 0, 1])
        n=self.Bezier_order
        target_length=interp_dist
        Control_Point_sets=[]
        color=[]
        if self.tot_passes==1:
            Control_Point_sets=[self.edge1_CP]
            color=[color_one]
        elif self.tot_passes==2:
            Control_Point_sets=[self.edge1_CP,self.edge2_CP]
            color=[color_one, color_two]
        else:
            for i in range(self.tot_passes):
                Control_Point_sets.append(self.edge1_CP*(1-(i)/(self.tot_passes-1))+self.edge2_CP*((i)/(self.tot_passes-1)))
                color.append(color_one*(1-(i)/(self.tot_passes-1))+color_two*((i)/(self.tot_passes-1)))
        
        passes=[np.vstack([self.bezier_curve3N(n,t,Pi) for t in self.find_t_newton(n, Pi,target_length)]) for Pi in Control_Point_sets]
        colors=[col*np.ones((len(pas), 1)) for col, pas in zip(color,passes)]

        self.passes=passes
        self.passes_colors=colors
        return passes, colors
    
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
    
    def pseudo_binary_order(self, passes):
        #Generates a Psuedo-Binary point order from array for local_scanned_area function
        n=len(passes)
        if n == 0:
            return []
        order = [0, n-1]
        queue = [(0, n-1)]
        while queue:
            start, end = queue.pop(0)
            mid = (start + end) // 2
            if mid not in order:
                order.append(mid)
            if start + 1 < mid:
                queue.append((start, mid))
            if mid + 1 < end:
                queue.append((mid, end))
        
        return [i for i in order if i < n]
    
    def ray_cast_prep(self):
        #maybe class instance variable tensor isn't right idea?
        self.tensor_plane = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh) #OGplane= Surface to Path-Plan on
        self.tensor_plane.compute_triangle_normals()
        self.tensor_plane.compute_vertex_normals()
        self.scene = o3d.t.geometry.RaycastingScene()
        tensor_cast_id = self.scene.add_triangles(self.tensor_plane)


    def local_scanned_area(self, Redundancy=True, Elimination=False):
        #determine the order the passes are given to the projectulator
        if self.passes==None:
            print("Path not yet defined")
            return
        self.ray_cast_prep()
        Redundancy_order=self.pseudo_binary_order(self.passes)
        
        #direction for ray-casting should be along tertiary axis, but pointing "down"
        ray_unit_direction=np.array(self.tertiary_axis)*-1
        redundant=[]

        #reset all vertex colors to red
        self.colors= np.full((self.points.shape[0], 3), [1,0,0])

        for index in Redundancy_order:
            point_set=self.passes[index]
            direction = ray_unit_direction * np.ones((point_set.shape[0], 1))
            #move points "up" on z axis by max z bb height to ensure projections have intersections
            z_offset=min(self.bounding_box.extent)*self.tertiary_axis
            ray_origins=point_set+z_offset

            #Ray Cast to test for intersection
            LocVec=np.hstack((ray_origins.astype(dtype=np.float32), direction.astype(np.float32)))
            ans = self.scene.cast_rays(LocVec)

            #Calculate "principal vector" for interpolated
            #TODO: change from principal_vector to linear fit between consecutive points
            principal_vector=point_set[0]-point_set[-1]
            principal_vector=principal_vector/np.linalg.norm(principal_vector)

            sign=np.dot(principal_vector,self.primary_axis)
            if sign<0:
                principal_vector=-1*principal_vector

            #Analyze where rays hit
            index_hits=[p for p,q in enumerate(ans['geometry_ids']) if q==0]
            #Default color is red, color_updates calculates on a per-pass basis
            color_updates = np.full((self.points.shape[0],3), -1.0)
           
            for i in index_hits:
                dist=ans['t_hit'].numpy()[i]
                delta=dist*self.tertiary_axis*(-1)
                onSurface=LocVec[i][:3]+delta
                intersection_index=ans['primitive_ids'].numpy()[i]
                #print(f"Intersected triangle index: {intersection_index}")
                average_normal = self.compute_average_normal_t(self.tensor_plane, intersection_index)
                #print(f"Average normal of all neighbors: {average_normal}")

                #Creates Transformation matrix of each intersection
                transformation_matrix=np.eye(3)
                transformation_matrix[0:3,0]=principal_vector
                transformation_matrix[0:3,1]=np.cross(average_normal, principal_vector)
                transformation_matrix[0:3,2]=average_normal
                #print(transformation_matrix)

                candidate_indices = np.array(self.kd_tree.query_ball_point(onSurface, 32))
                if len(candidate_indices)>0:
                    inv_rotation_matrix = np.linalg.inv(transformation_matrix)
                    #print(candidate_indices)
                    new_candidate_points= inv_rotation_matrix@self.points[candidate_indices].T
                    new_seed_point=inv_rotation_matrix@onSurface
                    probe_mask = ((new_seed_point[0] - 10 <= new_candidate_points[0]) & (new_candidate_points[0] <= new_seed_point[0] + 10) &
                                (new_seed_point[1] - 32 <= new_candidate_points[1]) & (new_candidate_points[1] <= new_seed_point[1] + 32))
                    scanned_points=candidate_indices[probe_mask]
                    if Redundancy==True:
                        already_scanned_mask = (self.colors[scanned_points] == [0, 1, 0]).all(axis=1) | (self.colors[scanned_points] == [0, 0, 1]).all(axis=1)
                        if np.all(already_scanned_mask):
                            redundant.append(point_set[i])
                        if Elimination==True:
                            self.passes[index]= np.delete(self.passes[index],i)
                        else:
                            color_updates[scanned_points] = [0, 1, 0]
                    else:
                        color_updates[scanned_points] = [0, 1, 0]
                        

            update_mask = color_updates[:, 0] != -1
            already_marked_mask = (self.colors[:, 0] != 1) 
            rescanned_mask = update_mask & already_marked_mask
            self.colors[update_mask] = color_updates[update_mask]
            self.colors[rescanned_mask]=[0,0,1]

        return redundant







        
#plane=o3d.io.read_triangle_mesh('plane_segments\plane_segment_8_mesh.stl')
#bounding_box=plane.get_oriented_bounding_box()

