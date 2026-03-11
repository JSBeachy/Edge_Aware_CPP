import pybullet as p
import pybullet_data
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial import cKDTree
import os
import multiprocessing
#from UR5Kinematics import UR5Kinematics

ROBOT_PROFILES = {
    "abb_irb120": {
        "urdf_path": r"C:\Users\jonas\BARC_NDI\pointcloudcpp\abbIrb120.urdf", #Can adjust path as needed (probably should make it relative with permanant location)
        "elbow_singularity": -1.34,  # -76.9 degrees
        "wrist_singularity": 0.0,
        "shoulder_front_hint": 0.0,  # Reaching Forward (normal)
        "shoulder_back_hint": -2.0,  # Reaching Overhead
        "elbow_up_hint": -1.57,      # Elbow bent up 
        "elbow_down_hint": 0,        # Elbow bend down (normal)
        "wrist_up_hint": -1.57,      # Wrist fipped up
        "wrist_down_hint": 1.57      # Wrist flipped down (normal)
    },

    "ur5": {
        "urdf_path": r"C:\Users\jonas\BARC_NDI\pointcloudcpp\ur5.urdf",
        "elbow_singularity": 0.0,    # Straight line
        "wrist_singularity": 0.0,
        "shoulder_front_hint": 0,   # Reaching Forward
        "shoulder_back_hint": 1.57, # Reaching Backwards
        "elbow_up_hint": -1.57,     # Elbow up
        "elbow_down_hint": 1.57,    # Elbow down
        "wrist_up_hint": -1.57,
        "wrist_down_hint": 1.57
    }
}

def worker_process(chunk_indices, chunk_points, chunk_normals, init_kwards, keys, wrist_only):
    init_kwards["connection_mode"] = p.DIRECT
    init_kwards["sample_points"] = False
    local_env = RobotReachability(**init_kwards)
    results=[]
    for i in range(len(chunk_points)):
        pt = chunk_points[i]
        nm = chunk_normals[i]
        res=[]
        for key in keys:
            success, msg = local_env.check_reachability(pt,nm, key, wrist_only)
            res.append(success)
        results.append((chunk_indices[i], tuple(res)))

    p.disconnect(local_env.client)
    return results

class RobotReachability:
    
    def __init__(self, num_points, mesh_path, mesh_position=[0,0.2,0], 
                 mesh_orientation=[0,0,0], base_position=[0,0,0], robot_name="abb_irb120",
                 connection_mode=p.GUI, sample_points=True, shared_mesh_path=None, ik_mode="numerical"):
        
        # Load robot profile
        if robot_name not in ROBOT_PROFILES:
            raise ValueError(f"Unsupported robot '{robot_name}'. Available profiles: {list(ROBOT_PROFILES.keys())}")
        self.profile = ROBOT_PROFILES[robot_name]  
        if sample_points: print("Loaded profile for", robot_name)   

        #save kwagrs for worker processes
        self.urdf_path=self.profile['urdf_path']
        self.mesh_path=mesh_path
        self.mesh_position=mesh_position
        self.mesh_orientation=mesh_orientation
        self.base_position=base_position
        self.robot_name=robot_name
        self.shared_mesh_path = shared_mesh_path
        self.ik_mode=ik_mode

        #self.ur5_kin=UR5Kinematics() if (ik_mode=="analytical" and robot_name=="ur5") else None

        self.num_processes=1
        self.signatures=[]
        self.cell_ids=[]

        #Setup PyBullet
        self.client=p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)
        
        #Load Robot
        self.robot_id = p.loadURDF(self.urdf_path, basePosition=base_position, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)        
        self.ee_index= -1
        self.movable_joints = [] 
        #print(f"Total Joints Found: {self.num_joints}")
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            joint_type = info[2]
            # Identify End Effector by name (Usually 'link_6' or 'tool0')
            if connection_mode == p.GUI:
                print(f"ID {i}: Joint='{joint_name}' Link='{link_name}' Type={joint_type}")
            if "link_6" in link_name or "tool0" in link_name:
                self.ee_index = i
            # Store movable joints for IK control
            if joint_type != p.JOINT_FIXED:
                self.movable_joints.append(i)

        if self.ee_index == -1:
            print("WARNING: Could not find 'link_6'. Defaulting to last index.")
            self.ee_index = self.num_joints - 1

        if sample_points:
            #Mesh Processing
            self.mesh=o3d.io.read_triangle_mesh(mesh_path)
            if not self.mesh.has_triangles():
                raise ValueError("CRITICAL ERROR: Failed to load mesh. File path may be wrong.")
            self.mesh.remove_duplicated_vertices()
            self.mesh.remove_duplicated_triangles()
            self.mesh.remove_degenerate_triangles()
            self.mesh.compute_vertex_normals()

            self.mesh = self.mesh.subdivide_midpoint(number_of_iterations=2)

            self.mesh.scale(0.001, center=(0,0,0))
            center_offset = self.mesh.get_center()
            self.mesh.translate(-center_offset)

            #save file for each parallel processes to load
            self.shared_mesh_path = os.path.abspath("shared_temp_mesh.stl")
            o3d.io.write_triangle_mesh(self.shared_mesh_path, self.mesh)

            #Mesh Positioning and Sampling
            mesh_quat = p.getQuaternionFromEuler(mesh_orientation)
            rot_matrix_flat=p.getMatrixFromQuaternion(mesh_quat)
            R=np.array(rot_matrix_flat).reshape(3,3)
            self.mesh.rotate(R, center=(0, 0, 0))
            self.mesh.translate(mesh_position)

            self.pcd=self.mesh.sample_points_poisson_disk(number_of_points=num_points)
            self.points = np.asarray(self.pcd.points)
            self.normals= np.asarray(self.pcd.normals)
            print(f"Loaded mesh with with {len(self.points)} subsampled points")
            self.max_reach= self.calculate_max_reach()
            self.reachable_mask = self.filter_unreachable_by_distance(self.max_reach)

        else:
            self.mesh=None
            self.points=[]
            self.normals=[]
            self.max_reach=None
            self.reachable_mask=None

        #Load Mesh into PyBullet
        mesh_quat = p.getQuaternionFromEuler(mesh_orientation)
        visual_id = p.createVisualShape(shapeType = p.GEOM_MESH, fileName=self.shared_mesh_path, rgbaColor=[0.6, 0.6, 0.6, 1], meshScale=[1,1,1])
        collision_id = p.createCollisionShape(shapeType = p.GEOM_MESH, fileName = self.shared_mesh_path, meshScale=[1,1,1], flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        
        self.mesh_body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=mesh_position,
            baseOrientation=mesh_quat
        )
    
    def generate_all_seeds(self):
        #Not super generalized
        self.seeds={}
        prof=self.profile
        shoulder_opts=["front", "back"]
        elbow_opts= ['elbow_up', 'elbow_down']
        wrist_opts= ['wrist_up', "wrist_down"]
        for s in shoulder_opts:
            for e in elbow_opts:
                for w in wrist_opts:
                    name=f"{s}_{e}_{w}"
                    j1= prof["shoulder_front_hint"] if s =="front" else prof["shoulder_back_hint"]
                    j3= prof["elbow_up_hint"] if e == "elbow_up" else prof["elbow_down_hint"]
                    j5= prof["wrist_up_hint"] if w == "wrist_up" else prof["wrist_down_hint"]
                    j2 = 0.0 if s=="front" else -1.57
                    self.seeds[name]=[j1, j2, j3, 0, j5, 0]
        print(f"Generated {len(self.seeds)} seeds for {len(shoulder_opts)*len(elbow_opts)*len(wrist_opts)} regions.")

    def align_vector_to_normal(self,normal):
        #Calculate Quaternion to align robot z-axis to tool
        tool_axis=np.array([0,0,1])
        target_axis = normal
        rotation_axis=np.cross(tool_axis,target_axis)
        if np.linalg.norm(rotation_axis)<1e-6: #already aligned
            if np.dot(tool_axis, target_axis) > 0: 
                return [0, 0, 0, 1]
            return [0,0,0,1] 
                
        rotation_axis= rotation_axis/np.linalg.norm(rotation_axis)
        angle = np.arccos(np.dot(tool_axis, target_axis)) 
        q_xyz=rotation_axis*np.sin(angle/2)
        q_w=np.cos(angle/2)
        return list(q_xyz) + [q_w]   

    def calculate_max_reach(self):
        #Calculating maximum reach from URDF (summing joint distances) - could add saftey factor value
        current_joint_index = self.ee_index
        total_length=0.0
        while current_joint_index !=-1:
            joint_info = p.getJointInfo(self.robot_id, current_joint_index)
            parent_link_index=joint_info[16]
            offset_vec= joint_info[14]
            dist=np.linalg.norm(offset_vec)
            #print(f"Link {current_joint_index} is child of Link {parent_link_index}, and is offset by {dist}")
            total_length+=dist
            current_joint_index=parent_link_index  
        #print(f"Calculated Chain Length: {total_length:.3f}m")
        safety_factor = 1.5
        return(total_length*safety_factor)
    
    def filter_unreachable_by_distance(self, max_reach):
        dists=np.linalg.norm(self.points, axis=1)
        mask = dists < (max_reach + 0.1)
        print(f"Distance Culling: {len(self.points)-np.sum(mask)} points skipped due to being completely unreachable")
        return mask
    
    def check_manipulability(self, joint_positions): #TODO: Need to integrate!
        #Calculate Yoshikawa manipulability Index, lower values indicate proximity to singularity
        jac_t, jac_r = p.calculateJacobian(
            self.robot_id, 
            self.ee_index, 
            localPosition=[0,0,0], 
            objPositions=joint_positions, 
            objVelocities=[0]*len(joint_positions), 
            objAccelerations=[0]*len(joint_positions)
        )
        jacobian = np.vstack((jac_t, jac_r))
        manipulability = np.sqrt(np.linalg.det(jacobian @ jacobian.T))
        return manipulability

    def check_collision(self):
        #Return true if colliding with self or environment
        contact_points=p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id)
        for contact in contact_points:
            linkA, linkB = contact[3], contact[4]
            if abs(linkA-linkB) > 1 and contact[8] < -0.005: #Eliminates same-link overlap as false positive
                return True, "Self Collision"
        env_contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.mesh_body_id)
        for contact in env_contacts:
            distance = contact[8]
            if distance < -0.005: 
                return True, "Part Collision"
        return False, "No Collision"
    
    def check_reachability(self, target_pos, target_normal, seed_name, wrist_only=False):
        prof=self.profile
        #Set up joint cage
        ll, ul=[], [] #lower/upper limit
        jr, rp=[], [] #joint range, rest pose
        is_front= "front" in seed_name
        is_elbow_up= "elbow_up" in seed_name
        is_wrist_up= "wrist_up" in seed_name
        
        seed_conf=self.seeds[seed_name]
        target_azimuth = np.arctan2(target_pos[1], target_pos[0]) #Calculate the angle in the XY plane for shoulder hinting

        if wrist_only:
            new_seed_conf= [0]*len(self.movable_joints)
            new_seed_conf[4] = seed_conf[4] #Only set wrist joint,
            seed_conf=new_seed_conf

        singularity_buffer= 0.05
        for i, joint_id in enumerate(self.movable_joints):
            info=p.getJointInfo(self.robot_id, joint_id)
            lower, upper=info[8], info[9]
            current_rest= seed_conf[i]
            #Wrist Joint limits (Assumed J5, split usually at 0 deg- Our convention: wrist up means J5 < split)
            if joint_id==4:
                split = prof["wrist_singularity"]
                if is_wrist_up: 
                    upper = split - singularity_buffer
                else:
                    lower = split + singularity_buffer
            if not wrist_only: #Only assign check elbow/sholder joint limits for full seeds

                #Elbow Joint limits (Assumed J3, split varies - Our convention: wrist up means J5 < split)
                if joint_id==2:
                    split = prof["elbow_singularity"]
                    if is_elbow_up: 
                        upper = -singularity_buffer + split
                    else:
                        lower = singularity_buffer + split

                #Shoulder Joint rests (Assumed J1, rest depends on target azimuth)
                if joint_id==0:
                    if is_front:
                        current_rest=target_azimuth
                    else:
                        current_rest=(target_azimuth%(2*np.pi)-np.pi) #flip to opposite side of the circle

                    #For ABB, joint limit is actually +- 165 degrees, this ensure rest pose respects limits
                    if current_rest > upper:
                        current_rest= upper-0.1
                    if current_rest < lower:
                        current_rest= lower+0.1

            ll.append(lower)
            ul.append(upper)
            jr.append(upper-lower)
            rp.append(current_rest)

        for i, joint_id in enumerate(self.movable_joints):
             p.resetJointState(self.robot_id, joint_id, seed_conf[i])

        target_orient = self.align_vector_to_normal(-target_normal)

        # Call router for IK solver
        joint_poses = self.solve_ik(target_pos, target_orient, seed_conf, ll, ul, jr, rp)
        if joint_poses is None:
            return False, f"IK Failed / Unreachable in {seed_name}"   

        #all_joint_positions=[]
        for i, joint_id in enumerate(self.movable_joints):
            p.resetJointState(self.robot_id, joint_id, joint_poses[i])
            #all_joint_positions.append(joint_poses[i])
        
        #input(f"PAUSED: Inspecting seed '{seed_name}'. Press Enter in the terminal to continue...")    

        #quality=self.check_manipulability(all_joint_positions)
        actual_pos = p.getLinkState(self.robot_id, self.ee_index)[4]
        dist = np.linalg.norm(np.array(actual_pos) - np.array(target_pos))
        if dist > 0.01: 
            return False, f"Unreachable in {seed_name} ({dist:.3f}m)"
        is_collision, _ = self.check_collision()
        if is_collision:
            return False, "Collision"
        return True, "Success"

    def solve_ik(self, target_pos, target_orient, seed_conf, ll, ul, jr, rp):
        # #Route IK requests to analytical solver or PyBullet's DLS
        # if self.ik_mode=="analytical" and self.ur5_kin is not None:
        #     #Handle potential tool offsets (0 for bare flange)
        #     tcp_pos_offset = [0.0, 0.0, 0.0] 
        #     tcp_ori_offset = p.getQuaternionFromEuler([0,np.pi/2,0])
        #     inv_tcp_pos, inv_tcp_ori = p.invertTransform(tcp_pos_offset, tcp_ori_offset)
        #     link6_pos, link6_ori = p.multiplyTransforms(target_pos, target_orient, inv_tcp_pos, inv_tcp_ori)

        #     rot_matrix=np.array(p.getMatrixFromQuaternion(link6_ori)).reshape(3,3)
        #     T_target = np.eye(4)
        #     T_target[:3,:3]=rot_matrix
        #     T_target[:3,3] = link6_pos

        #     #Calculate analytical solutions
        #     joint_solutions=self.ur5_kin.inverse(T_target)
        #     #Run through joint limits
        #     best_sol = None
        #     min_dist = float('inf')
            
        #     for sol in joint_solutions:
        #         is_valid = True
        #         for i in range(6):
        #             if sol[i] < ll[i] or sol[i] > ul[i]:
        #                 is_valid = False; break
                
        #         if is_valid:
        #             dist = np.linalg.norm(np.array(sol) - np.array(seed_conf))
        #             if dist < min_dist:
        #                 min_dist = dist
        #                 best_sol = sol
        #     return best_sol
        
        #PyBullet Numerical Solver Fallback
        return p.calculateInverseKinematics(
            self.robot_id, self.ee_index, target_pos, target_orient,
            lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=rp,
            maxNumIterations=150, residualThreshold=1e-3
        )

    def run_parallel_analysis(self, wrist_only=False, num_processes=None):
        if num_processes is None:
            num_processes= max(1, multiprocessing.cpu_count() - 1)
        self.num_processes = num_processes
        
        self.generate_all_seeds()
        keys=list(self.seeds.keys())
        if wrist_only:
            keys=keys[:2]
        print(f"Running parallel analysis with {num_processes} processes with orientations: {keys}")
        valid_indices=np.where(self.reachable_mask)[0]
        valid_points=self.points[valid_indices]
        valid_normals=self.normals[valid_indices]
        
        #Shuffle data
        suffle_idx=np.random.permutation(len(valid_indices))
        valid_indices=valid_indices[suffle_idx]
        valid_points=valid_points[suffle_idx]
        valid_normals=valid_normals[suffle_idx]

        #Chunk Data
        chunks_idx = np.array_split(valid_indices, num_processes)
        chunks_points = np.array_split(valid_points, num_processes)
        chunks_normals = np.array_split(valid_normals, num_processes)

        init_kwargs={
            'num_points': 0,
            'mesh_path': self.mesh_path,
            'mesh_position': self.mesh_position,
            'mesh_orientation': self.mesh_orientation,
            'base_position': self.base_position,
            'robot_name': self.robot_name,
            'shared_mesh_path': self.shared_mesh_path,
            'ik_mode': self.ik_mode
        }
        
        tasks=[]
        for i in range(num_processes):
            if len(chunks_idx[i])>0:
                tasks.append((chunks_idx[i], chunks_points[i], chunks_normals[i], init_kwargs, keys, wrist_only))
        
        start_time=time.time()
        with multiprocessing.Pool(processes=num_processes) as pool:
            results= pool.starmap(worker_process, tasks)
            
        self.signatures = [tuple([False]*len(keys))]*len(self.points)
        for worker_res in results:
            for original_idx,sig in worker_res:
                self.signatures[original_idx]=sig

        print(f"Parallel analysis complete in {time.time()-start_time:.2f} seconds")

    def run_analysis(self, wrist_only=False):
        self.num_processes=1
        self.generate_all_seeds()
        self.signatures=[]
        keys = list(self.seeds.keys())
        if wrist_only: #Only Keep 1 wrist up and 1 wrist down seed, and don't check elbow/sholder joint limits 
            keys = keys[:2] 
        start_time=time.time()
        for i in range(len(self.points)):
            if not self.reachable_mask[i]:
                self.signatures.append(tuple([False]*len(keys))) #mark all false as default for unreachable points
                continue
            pt = self.points[i]
            nm = self.normals[i]
            res=[]
            for key in keys:
                result, msg = self.check_reachability(pt, nm, key, wrist_only)
                res.append(result)
            self.signatures.append(tuple(res))
        print(f"Analysis complete in {time.time()-start_time:.2f} seconds")

    def refine_boundaries(self, total_dense_points=10000, boundary_radius=0.04, wrist_only=False, num_processes=None):
        print("\n--- Starting Adaptive Boundary Refinement ---")
        if num_processes is None:
            num_processes = self.num_processes if hasattr(self, 'num_processes') else max(1, multiprocessing.cpu_count() - 1)
        #Find boundary points in sparse set        
        tree = cKDTree(self.points)
        boundary_indices = set()
        distances, indices = tree.query(self.points, k=6) #check 5 nearest neighbors (+self))
        for i, neighbors in enumerate(indices):
            my_sig = self.signatures[i] #get valid configs for neighboring points
            for n_idx in neighbors[1:]:
                if self.signatures[n_idx] != my_sig:
                    boundary_indices.add(i)
                    break
        if len(boundary_indices)==0:
            print("No boundary points found for refinement")
            return
        boundary_points = self.points[list(boundary_indices)]

        #Generate new dense points around boundaries
        dense_pcd = self.mesh.sample_points_poisson_disk(number_of_points=total_dense_points)
        dense_points = np.asarray(dense_pcd.points)
        dense_normals = np.asarray(dense_pcd.normals)
        boundary_tree = cKDTree(boundary_points)
        dists, _ = boundary_tree.query(dense_points)
        near_boundary_mask = dists < boundary_radius
        new_points = dense_points[near_boundary_mask]
        new_normals = dense_normals[near_boundary_mask] 
        print(f"Filtered {total_dense_points} dense points down to {len(new_points)} points near boundaries")
        if len(new_points)==0: return

        #Run IK on near-boundary points 
        keys = list(self.seeds.keys())
        if wrist_only: keys = keys[:2]
        valid_indices=np.arange(len(new_points))
        valid_new_points=new_points
        valid_new_normals=new_normals
        new_signatures = [tuple([False]*len(keys))]*len(new_points)
        start_time = time.time()
        if num_processes <=1:
            print(f"Running single-core evaluation on {len(new_points)} points")
            for i, original_idx in enumerate(valid_indices):
                pt = valid_new_points[i]
                nm = valid_new_normals[i]
                res=[]
                for key in keys:
                    result, msg = self.check_reachability(pt, nm, key, wrist_only)
                    res.append(result)
                new_signatures[i]=tuple(res)
        else:
            print(f"Running parallel evaluation on {len(new_points)} points with {num_processes} processes")
            chunks_idx = np.array_split(valid_indices, num_processes)
            chunks_points = np.array_split(valid_new_points, num_processes)
            chunks_normals = np.array_split(valid_new_normals, num_processes)

            init_kwargs={
                'num_points': 0,
                'mesh_path': self.mesh_path,
                'mesh_position': self.mesh_position,
                'mesh_orientation': self.mesh_orientation,
                'base_position': self.base_position,
                'robot_name': self.robot_name,
                'shared_mesh_path': self.shared_mesh_path,
                'ik_mode': self.ik_mode
            }
            
            tasks=[]
            for i in range(num_processes):
                if len(chunks_idx[i])>0:
                    tasks.append((chunks_idx[i], chunks_points[i], chunks_normals[i], init_kwargs, keys, wrist_only))
            
            with multiprocessing.Pool(processes=num_processes) as pool:
                results= pool.starmap(worker_process, tasks)
                
            for worker_res in results:
                for original_idx,sig in worker_res:
                    new_signatures[original_idx]=sig
            
        print(f"Boundary refinement complete in {time.time()-start_time:.2f} seconds")
        #Merge data
        self.points = np.vstack((self.points, new_points))
        self.normals = np.vstack((self.normals, new_normals))
        self.signatures.extend(new_signatures)
        
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.points)
        self.pcd.normals = o3d.utility.Vector3dVector(self.normals)
        print(f"Refinment complete. Total resolution is now {len(self.points)} points")

    def visualize_signatures(self):
        #Colors points strictly by what they can do - wrist only.
        colors = []
        for sig in self.signatures:
            up, down = sig
            if up and down:
                colors.append([0, 1, 0]) # Green (Both)
            elif up and not down:
                colors.append([1, 1, 0]) # Yellow (Up Only)
            elif down and not up:
                colors.append([0, 0, 1]) # Blue (Down Only)
            else:
                colors.append([1, 0, 0]) # Red (Neither)
        self.pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        mesh_wire = o3d.geometry.LineSet.create_from_triangle_mesh(self.mesh)
        o3d.visualization.draw_geometries([self.pcd, mesh_wire, frame])  

    def segment_into_cells(self, radius=0.02):
        #print("Segmeneting surface into cells")
        tree= cKDTree(self.points)
        self.cell_ids = -1 * np.ones(len(self.points), dtype = int)
        current_cell_id=0
        for i in range(len(self.points)):
            if self.cell_ids[i] != -1: continue
            current_sig=self.signatures[i]
            queue=[i]
            self.cell_ids[i] = current_cell_id
            while len(queue)>0:
                idx=queue.pop(0)
                neighbors=tree.query_ball_point(self.points[idx], r=radius)
                for n_idx in neighbors:
                    if self.cell_ids[n_idx] == -1:
                        if self.signatures[n_idx] == current_sig:
                            self.cell_ids[n_idx]=current_cell_id
                            queue.append(n_idx)

            current_cell_id +=1
        print(f"Segmentation complete. Found {current_cell_id+1} unique cells")

    def map_cells_to_mesh(self, distance_threshold=0.05):
        mesh_vertices = np.asarray(self.mesh.vertices)
        valid_mask= self.cell_ids != -1
        valid_points = self.points[valid_mask] #Only consider points that were assigned to a reachable cell
        valid_cell_ids= self.cell_ids[valid_mask]
        self.vertex_cell_ids = -1 *np.ones(len(mesh_vertices), dtype=int) #All vertices start as unassigned
        if len(valid_points)==0: 
            print("No valid points to map to mesh")
            return
        #Build KDTree with reachable points
        tree = cKDTree(valid_points)
        distances, nearest_indices = tree.query(mesh_vertices) #return nearest point for each vertex

        valid_mapping_mask = distances < distance_threshold #Only assign if vertex is close enough to a valid point
        self.vertex_cell_ids[valid_mapping_mask] = valid_cell_ids[nearest_indices[valid_mapping_mask]]
        mapped_count = np.sum(valid_mapping_mask)
        print(f"Mapped {mapped_count} out of {len(mesh_vertices)} vertices to {len(np.unique(valid_cell_ids))} unique cells.")

    def visualize_result(self):
        num_cells=self.cell_ids.max() +1
        cell_colors_map = np.random.uniform(0,1, (num_cells,3))
        final_colors=[]
        for i in range((len(self.points))):
            if self.signatures[i] == (False,False):
                final_colors.append([1,0,0])
            else:
                c_id=self.cell_ids[i]
                final_colors.append(cell_colors_map[c_id])
        self.pcd.colors=o3d.utility.Vector3dVector(np.array(final_colors))
        mesh_wire=o3d.geometry.LineSet.create_from_triangle_mesh(self.mesh)
        o3d.visualization.draw_geometries([self.pcd, mesh_wire])

    def visualize_with_plotly(self, filename="reachability_map.html"):
        import plotly.graph_objects as go
        import plotly.colors as pc
        fig=go.Figure()
        keys=list(self.seeds.keys())
        v=np.asarray(self.mesh.vertices)
        t=np.asarray(self.mesh.triangles)
        fig.add_trace(go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=t[:,0], j=t[:,1], k=t[:,2], color='lightgrey', opacity=0.5))
        #colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#FFA500', '#800080']
        unique_cells = np.unique(self.cell_ids)
        keys= list(self.seeds.keys())
        if len(unique_cells)>1:
            palette=pc.sample_colorscale("Turbo", len(unique_cells))
        else:
            palette=['rgb(17, 157, 255)']

        for i, cell_id in enumerate(unique_cells):
            if cell_id == -1:
                continue
            indices = np.where(self.cell_ids == cell_id)[0]
            pts = self.points[indices]
            sig= self.signatures[indices[0]]
            valid_configs=[keys[k] for k, is_valid in enumerate(sig) if is_valid]
            # Format the hover text
            # We use <br> to break lines in the hover box
            hover_text = f"<b>Cell ID: {cell_id}</b><br>Points: {len(pts)}<br><br><b>Valid Configs:</b><br>" + "<br>".join(valid_configs)
            
            fig.add_trace(go.Scatter3d(
                x=pts[:,0], 
                y=pts[:,1], 
                z=pts[:,2],
                mode='markers',
                marker=dict(
                    size=4, 
                    color=palette[i], # Assign unique color
                    line=dict(width=0)
                ),
                name=f"Cell {cell_id}",
                text=[hover_text] * len(pts), # Apply text to all points
                hoverinfo='text' # Only show our custom text
            ))
            
        fig.update_layout(
            title=f"Reachability Segmentation ({len(unique_cells)} Unique Regions)",
            scene=dict(aspectmode='data'),
            legend_title_text='Segments'
        )
        
        fig.write_html(filename)

    def visualize_solid_mesh(self):
            print("Visualizing solid mapped mesh...")
            num_cells = np.max(self.cell_ids) + 1
            np.random.seed(42) 
            cell_colors_map = np.random.uniform(0.1, 0.9, (num_cells, 3))
            vertex_colors = np.zeros((len(self.vertex_cell_ids), 3))
            for i, v_id in enumerate(self.vertex_cell_ids):
                if v_id == -1:
                    vertex_colors[i] = [1, 0, 0] # Red for unreachable
                else:
                    vertex_colors[i] = cell_colors_map[v_id]
            self.mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)           
            o3d.visualization.draw_geometries([self.mesh], mesh_show_back_face=True)

    def cleanup(self):
        try:
            if self.shared_mesh_path and os.path.exists(self.shared_mesh_path):
                os.remove(self.shared_mesh_path)
        except OSError:
            pass
        if p.isConnected(self.client):
            p.disconnect(self.client)

if __name__ == "__main__":
    s=time.time()
    multiprocessing.freeze_support()
    mesh_file = r"C:\Users\jonas\BARC_NDI\pointcloudcpp\plane_segments\Airfoil_Surface_example.stl"
    
    # Placement: 0.5m forward, 0.2m up. Rotated upright.
    part_pos = [0.5, 0, 0.3] 
    part_euler = [0, 1.57, 3.14] 
    num_points= 5000 #5000
    chosen_solver="numerical"
    rob="abb_irb120" # or "ur5"
    segmenter = RobotReachability(num_points, mesh_file, mesh_position=part_pos, mesh_orientation=part_euler, robot_name=rob, ik_mode=chosen_solver)
    
    # Reachability Check
    wrist_only=False
    #segmenter.run_analysis(wrist_only)
    segmenter.run_parallel_analysis(wrist_only=False)
    #if wrist_only:
    #    segmenter.visualize_signatures()
    
    segmenter.refine_boundaries(total_dense_points=15000, boundary_radius=0.05, wrist_only=wrist_only)

    # Cell Formation (Radius = 5cm neighbor search)
    segmenter.segment_into_cells(radius=0.05)
    segmenter.map_cells_to_mesh(distance_threshold=0.05)

    print(f"Total execution time: {time.time()-s:.2f} seconds")
    # 3. View Results
    segmenter.visualize_result()
    segmenter.visualize_with_plotly("my_robot_cells.html")
    segmenter.visualize_solid_mesh()
    
    segmenter.cleanup()

