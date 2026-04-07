import pybullet as p
import pybullet_data
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial import cKDTree
import os

ROBOT_PROFILES = {
    "abb_irb120": {
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

class RobotReachability:
    
    def __init__(self, num_points, urdf_path, mesh_path, mesh_position=[0,0.2,0], mesh_orientation=[0,0,0], base_position=[0,0,0], robot_name="abb_irb120"):
        
        # Load robotProfile
        if robot_name not in ROBOT_PROFILES:
            raise ValueError(f"Unsupported robot '{robot_name}'. Available profiles: {list(ROBOT_PROFILES.keys())}")
        self.profile = ROBOT_PROFILES[robot_name]  
        print("Loaded profile for", robot_name)    

        #Setup PyBullet
        #self.client = p.connect(p.DIRECT)
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)
        
        #Load Robot
        self.robot_id = p.loadURDF(urdf_path, basePosition=base_position, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)
        print("-" * 30)
        print(f"Total Joints Found: {self.num_joints}")
        self.ee_index= -1
        self.movable_joints = [] 
        
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            joint_type = info[2]
            
            print(f"ID {i}: Joint='{joint_name}' Link='{link_name}' Type={joint_type}")
            # Identify End Effector by name (Usually 'link_6' or 'tool0')
            if "link_6" in link_name or "tool0" in link_name:
                self.ee_index = i
            # Store movable joints for IK control
            if joint_type != p.JOINT_FIXED:
                self.movable_joints.append(i)

        if self.ee_index == -1:
            print("WARNING: Could not find 'link_6'. Defaulting to last index.")
            self.ee_index = self.num_joints - 1
        print(f"SELECTED END EFFECTOR INDEX: {self.ee_index}")
        print("-" * 30)

        
        #Define Seeds for positional configuration
        # self.seeds = {
        #     "wrist_up" : [0, 0, 0, 0, -1.57, 0],
        #     "wrist_down" : [0, 0, 0, 0, 1.57, 0]
        # }

        #Mesh Processing
        self.mesh=o3d.io.read_triangle_mesh(mesh_path)
        if not self.mesh.has_triangles():
            raise ValueError("CRITICAL ERROR: Failed to load mesh. File path may be wrong.")
        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_duplicated_triangles()
        self.mesh.remove_degenerate_triangles()
        self.mesh.compute_vertex_normals()

        self.mesh.scale(0.001, center=(0,0,0))
        center_offset = self.mesh.get_center()
        self.mesh.translate(-center_offset)
        temp_mesh_path = os.path.abspath("temp_mesh_loader.stl")
        o3d.io.write_triangle_mesh(temp_mesh_path, self.mesh)

        #Load Mesh into PyBullet
        mesh_quat = p.getQuaternionFromEuler(mesh_orientation)
        visual_id = p.createVisualShape(shapeType = p.GEOM_MESH, fileName=temp_mesh_path, rgbaColor=[0.6, 0.6, 0.6, 1], meshScale=[1,1,1])
        collision_id = p.createCollisionShape(shapeType = p.GEOM_MESH, fileName = temp_mesh_path, meshScale=[1,1,1], flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        
        self.mesh_body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=mesh_position,
            baseOrientation=mesh_quat
        )

        rot_matrix_flat=p.getMatrixFromQuaternion(mesh_quat)
        R=np.array(rot_matrix_flat).reshape(3,3)
        self.mesh.rotate(R, center=(0, 0, 0))
        self.mesh.translate(mesh_position)

        self.pcd=self.mesh.sample_points_poisson_disk(number_of_points=num_points)
        self.points = np.asarray(self.pcd.points)
        self.normals= np.asarray(self.pcd.normals)
        print(f"Loaded mesh with with {len(self.points)} subsampled points")

        #Storage for Cell Divisions
        self.signatures=[]
        self.cell_ids=[]
        self.wristup_count=0
        self.wristdown_count=0
        self.max_reach= self.calculate_max_reach()
        self.reachable_mask = self.filter_unreachable_by_distance(self.max_reach)
    
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
        print("Calculating maximum reach from URDF (summing joint distances)")
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
        print(f"Distance Culling: {len(self.points)-np.sum(mask)} points ignored due to being completely unreachable")
        return mask
    
    def check_collision(self):
        #Return true if colliding with self or environment
        contact_points=p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id)
        for contact in contact_points:
            linkA, linkB = contact[3], contact[4]
            if abs(linkA-linkB) > 1 and contact[8] < -0.005: #Eliminates same-link overlap as false positive
                return True, "Self Collision"
        env_contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.mesh_body_id)
        for contact in env_contacts:
            link_index = contact[3] # The link ID on the robot
            distance = contact[8]
            if distance < -0.001: # Genuine collision
                # Get the name of the link for clarity
                if link_index == -1:
                    link_name = "BASE (Fixed Base)"
                else:
                    link_name = p.getJointInfo(self.robot_id, link_index)[12].decode('utf-8')
                
                print(f"CRITICAL COLLISION: {link_name} (ID {link_index}) is inside the part by {abs(distance):.3f}m")
                return True, f"Collision: {link_name}"
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

        for i, joint_id in enumerate(self.movable_joints):
            info=p.getJointInfo(self.robot_id, joint_id)
            lower, upper=info[8], info[9]
            current_rest= seed_conf[i]
            #Wrist Joint limits (Assumed J5, split usually at 0 deg- Our convention: wrist up means J5 < split)
            if joint_id==4:
                split = prof["wrist_singularity"]
                if is_wrist_up: 
                    upper = -0.05 + split
                else:
                    lower = 0.05 + split
            if not wrist_only: #Only assign check elbow/sholder joint limits for full seeds
                #Elbow Joint limits (Assumed J3, split varies - Our convention: wrist up means J5 < split)
                if joint_id==2:
                    split = prof["elbow_singularity"]
                    if is_elbow_up: 
                        upper = -0.02 + split
                    else:
                        lower = 0.02 - split

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

        # 3. Run IK with the Limits (The Cage)
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_index,
            target_pos,
            target_orient,
            lowerLimits=ll,    # <--- Enforces all joint limits
            upperLimits=ul,    # <--- Enforces all joint limits
            jointRanges=jr,
            restPoses=rp,
            maxNumIterations=200, 
            residualThreshold=1e-5
        )

        for i, joint_id in enumerate(self.movable_joints):
            p.resetJointState(self.robot_id, joint_id, joint_poses[i])

        actual_pos = p.getLinkState(self.robot_id, self.ee_index)[4]
        dist = np.linalg.norm(np.array(actual_pos) - np.array(target_pos))
        if dist > 0.01: 
            return False, f"Unreachable in {seed_name} ({dist:.3f}m)"
        is_collision, _ = self.check_collision()
        if is_collision:
            return False, "Collision"
        actual_j5 = joint_poses[4]
        if "wrist_up" in seed_name and actual_j5 > 0: return False, "Cage Breach"
        if "wrist_down" in seed_name and actual_j5 < 0: return False, "Cage Breach"
        return True, "Success"

    def run_analysis(self, wrist_only=False):
        self.generate_all_seeds()
        feasible_mask = self.reachable_mask
        self.signatures=[]
        keys = list(self.seeds.keys())
        if wrist_only: #Only Keep 1 wrist up and 1 wrist down seed, and don't check elbow/sholder joint limits 
            keys = keys[:2] 
        print(keys)

        #print(f"Starting analysis on {len(self.points)} points")
        start_time=time.time()
        for i in range(len(self.points)):
            if not feasible_mask[i]:
                self.signatures.append(tuple([False]*len(keys)))
                continue
            pt = self.points[i]
            nm = self.normals[i]
            res=[]
            for key in keys:
                result, msg = self.check_reachability(pt, nm, key, wrist_only)
                res.append(result)
            self.signatures.append(tuple(res))

            #if i%200==0: print(f" Processed {i}/{len(self.points)}")
        print(f"Analysis complete in {time.time()-start_time:.2f} seconds")

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

    def visualize_heatmap(self):
        print("Visualizing Reachability Heatmap")
        colors=[]
        for sig in self.signatures:
            count=sum(sig)
            if count==0:
                colors.append([1,0,0])
            else:
                intensity = count/len(sig)
                colors.append([1-intensity, 1, 1-intensity]) # More green for more reachable seeds
        self.pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        mesh_wire = o3d.geometry.LineSet.create_from_triangle_mesh(self.mesh)
        o3d.visualization.draw_geometries([self.pcd, mesh_wire])

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

if __name__ == "__main__":
    abb_urdf_file = r"C:\Users\jonas\BARC_NDI\pointcloudcpp\abbIrb120.urdf" 
    ur_urdf_file=r"C:\Users\jonas\BARC_NDI\pointcloudcpp\ur5.urdf"
    mesh_file = r"C:\Users\jonas\BARC_NDI\pointcloudcpp\plane_segments\Airfoil_Surface_example.stl"
    
    # Placement: 0.5m forward, 0.2m up. Rotated upright.
    part_pos = [0.5, 0, 0.3] 
    part_euler = [0, 1.57, 3.14] 
    num_points=10000
    # --- EXECUTION ---
    segmenter = RobotReachability(num_points, abb_urdf_file, mesh_file, mesh_position=part_pos, mesh_orientation=part_euler)
    
    # 1. Reachability Check
    wrist_only=True
    segmenter.run_analysis(wrist_only)
    if wrist_only:
        segmenter.visualize_signatures()
    segmenter.visualize_heatmap()

    # 2. Cell Formation (Radius = 2cm neighbor search)
    segmenter.segment_into_cells(radius=0.05)
    
    # 3. View Results
    segmenter.visualize_result()
    print(f"Wrist up successes: {segmenter.wristup_count}")
    print(f"Wrist down successes: {segmenter.wristdown_count}")
    if p.isConnected():
        p.disconnect()

