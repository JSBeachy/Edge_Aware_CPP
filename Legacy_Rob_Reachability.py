import pybullet as p
import pybullet_data
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial import cKDTree
import os

class RobotReachability:
    
    def __init__(self, num_points, urdf_path, mesh_path, mesh_position=[0,0.2,0], mesh_orientation=[0,0,0],base_position=[0,0,0]):
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
        self.seeds = {
            "wrist_up" : [0, 0, 0, 0, -1.57, 0],
            "wrist_down" : [0, 0, 0, 0, 1.57, 0]
        }

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

    def check_reachability(self, target_pos, target_normal, seed_conf):
        #Two-Stage IK Solver
        for i, joint_id in enumerate(self.movable_joints):
            if i < len(seed_conf):
                p.resetJointState(self.robot_id, joint_id, seed_conf[i])
        target_orient=self.align_vector_to_normal(-target_normal) #negative so tool can match

        #Coarse Pass
        joint_poses_1 = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_index,
            target_pos,
            target_orient,
            maxNumIterations=500,
            residualThreshold=1e-3
        )
        #Move to Pass 1 Result
        for i, joint_id in enumerate(self.movable_joints):
            p.resetJointState(self.robot_id, joint_id, joint_poses_1[i])
        #Check Coarse Error, abort if not close
        actual_pos_1 = p.getLinkState(self.robot_id, self.ee_index)[4] #EE pos in world frame
        dist_coarse = np.linalg.norm(np.array(actual_pos_1)-np.array(target_pos))
        if dist_coarse > 0.10:
            return False, f"Coarse Fail: ({dist_coarse:.3f} off)"
        # Fine Pass
        joint_poses_2 = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_index,
            target_pos,
            target_orient,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        #Move to Pass 2 Result
        for i, joint_id in enumerate(self.movable_joints):
            p.resetJointState(self.robot_id, joint_id, joint_poses_2[i])
        #Check Fine Error
        actual_pos_final = p.getLinkState(self.robot_id, self.ee_index)[4]
        dist_final = np.linalg.norm(np.array(actual_pos_final)-np.array(target_pos))
        if dist_final>0.01: #Strict 1cm tolerance
            return False, f"Refine Fail: ({dist_final:.3f} off)"
        collision, reason= self.check_collision()
        if collision:
            return False, reason
        
        return True, "Success"
    
    def draw_ee_frame(self, life_time=0.1):
        # Get current position and orientation of the End Effector
        state = p.getLinkState(self.robot_id, self.ee_index)
        pos = state[4]
        orn = state[5]
        # Convert quaternion to rotation matrix
        rot_matrix = p.getMatrixFromQuaternion(orn)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Axis length
        L = 0.15 
        # Calculate endpoints for X, Y, Z axes
        # origin + (Rotation * axis_vector)
        x_axis = np.array(pos) + rot_matrix @ np.array([L, 0, 0])
        y_axis = np.array(pos) + rot_matrix @ np.array([0, L, 0])
        z_axis = np.array(pos) + rot_matrix @ np.array([0, 0, L])
        # Draw Lines
        p.addUserDebugLine(pos, x_axis, [1, 0, 0], lifeTime=life_time, lineWidth=2) # X = Red
        p.addUserDebugLine(pos, y_axis, [0, 1, 0], lifeTime=life_time, lineWidth=2) # Y = Green
        p.addUserDebugLine(pos, z_axis, [0, 0, 1], lifeTime=life_time, lineWidth=2) # Z = Blue
    
    def run(self, config_seeds=None):
        if config_seeds is None:
            # Could define hard-coded seeds
            config_seeds = self.seeds
        colors=[]

        print(f"Processing {len(self.points)} points")
        for i in range(len(self.points)):
            pt=self.points[i]
            nm=self.normals[i]

            can_up, status_up =self.check_reachability(pt, nm, self.seeds["wrist_up"])
            can_down, status_down =self.check_reachability(pt,nm,self.seeds["wrist_down"])

            if can_up and can_down:
                colors.append([0,1,0])
            elif can_up:
                colors.append([1,1,0])
            elif can_down:
                colors.append([0,0,1])
            else:
                colors.append([1,0,0])

        self.pcd.colors=o3d.utility.Vector3dVector(np.array(colors))
        mesh_wire=o3d.geometry.LineSet.create_from_triangle_mesh(self.mesh)
        o3d.visualization.draw_geometries([self.pcd, mesh_wire])
        p.disconnect()

if __name__ == "__main__":
    urdf_file = r"C:\Users\jonas\BARC_NDI\pointcloudcpp\abbIrb120.urdf" 
    mesh_file = r"C:\Users\jonas\BARC_NDI\pointcloudcpp\plane_segments\Airfoil_Surface_example.stl"
        
    mesh=o3d.io.read_triangle_mesh(mesh_file)
    #o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    extent=mesh.get_axis_aligned_bounding_box().get_extent()

    print(extent)
    part_pos = [.5, 0, extent[2]/1000] 
    part_euler= [0,1.57,3.14]
    
    segmenter = RobotReachability(urdf_file, mesh_file, mesh_position=part_pos, mesh_orientation=part_euler)

    dists = np.linalg.norm(segmenter.points, axis=1)
    test_idx = np.argmin(dists)
    target_pt = segmenter.points[test_idx]
    target_nm = segmenter.normals[test_idx]
    p.addUserDebugLine(target_pt, target_pt + (target_nm * 0.2), [1, 0, 1], lifeTime=0, lineWidth=3)
    
    try:
        l=0
        while True:
            flip = l%2
            p.stepSimulation()
            segmenter.draw_ee_frame(life_time=1)
            if flip: 
                seed="wrist_up"
            else:
                seed="wrist_down"
            result, _ = segmenter.check_reachability(target_pt, target_nm, segmenter.seeds[seed])
            print(result, _)
            curr_pos = p.getLinkState(segmenter.robot_id, segmenter.ee_index)[4]
            dist = np.linalg.norm(np.array(curr_pos) - target_pt)
            print(dist)
            p.addUserDebugLine(curr_pos, target_pt, [0, 1, 1], lifeTime=1)
            time.sleep(5)
            l+=1
            
    except KeyboardInterrupt:
        pass


# if __name__ == "__main__":

#     urdf_file = r"C:\Users\jonas\BARC_NDI\pointcloudcpp\abbIrb120.urdf" 
#     mesh_file = r"C:\Users\jonas\BARC_NDI\pointcloudcpp\plane_segments\Airfoil_Surface_example.stl"
    
#     seeds = {
#         "wrist_up" : [0, 0, 0, 0, -1.57, 0],
#         "wrist_down" : [0, 0, 0, 0, 1.57, 0] 
#     }
    
#     mesh=o3d.io.read_triangle_mesh(mesh_file)
#     extent=mesh.get_axis_aligned_bounding_box().get_extent()
    

#     print(extent)
#     part_pos = [.5, 0, extent[2]/1000] 
#     part_euler= [0,1.57,3.14]

#     segmenter = RobotReachability(
#         urdf_file, 
#         mesh_file, 
#         mesh_position=part_pos, 
#         mesh_orientation=part_euler
#     )
#     for _ in range(2400): 
#         p.stepSimulation()
#         time.sleep(1/240.)
#     
#     segmenter.run(seeds)
