import numpy as np
import math


class UR5Kinematics:
    #Analytic IK solver for UR5

    def __init__(self):
        #Exact D-H parameters for UR5 in meters
        self.d1=0.089159
        self.a2=-0.42500
        self.a3=-0.39225
        self.d4=0.10915
        self.d5=0.09465
        self.d6=0.0823

    def inverse(self,T):
        #Takes goal position (in tranformation matrix form) and returns set of IK solutions (joint angle form) 
        solutions=[]
        P05 = T @ np.array([0, 0, -self.d6, 1]).T #Back out where wrist intersection is in base frame?

        #Theta 1
        psi=math.atan2(P05[1], P05[0])
        L = math.sqrt(P05[0]**2 + P05[1]**2)
        if L < self.d4: return [] 
        phi = math.acos(self.d4 / L)
        theta1_opts = [psi + phi + math.pi/2, psi - phi + math.pi/2]

        for th1 in theta1_opts:
            #Theta 5
            P06_x, P06_y= T[0,3], T[1,3]
            p16z = P06_x * math.sin(th1) - P06_y*math.cos(th1)
            val = (p16z - self.d4) / self.d6
            if abs(val) > 1.0: continue
            th5_1 = math.acos(val)
            theta5_opts=[th5_1, -th5_1]

            for th5 in theta5_opts:
                # Theta 6
                T01 = np.array([
                    [math.cos(th1), -math.sin(th1), 0, 0],
                    [math.sin(th1),  math.cos(th1), 0, 0],
                    [0, 0, 1, self.d1], [0, 0, 0, 1]])
                T61 = np.linalg.inv(T01) @ T
                if math.sin(th5) == 0: 
                    th6 = 0 
                else: 
                    th6 = math.atan2(-T61[1, 2] / math.sin(th5), T61[0, 2] / math.sin(th5))

                #Theta 2, 3, and 4
                T45 = np.array([
                    [math.cos(th5), -math.sin(th5), 0, 0],
                    [0, 0, -1, -self.d5],
                    [math.sin(th5),  math.cos(th5), 0, 0], [0, 0, 0, 1]])
                T56 = np.array([
                    [math.cos(th6), -math.sin(th6), 0, 0],
                    [0, 0, 1, self.d6],
                    [-math.sin(th6), -math.cos(th6), 0, 0], [0, 0, 0, 1]])
                T14 = T61 @ np.linalg.inv(T45 @ T56)
                P14 = T14[:3, 3]
                P14_x, P14_z = P14[0], P14[2]
                
                L_sq = P14_x**2 + P14_z**2
                val_th3 = (L_sq - self.a2**2 - self.a3**2) / (2 * self.a2 * self.a3)
                if abs(val_th3) > 1.0: continue    

                th3_1 = math.acos(val_th3)
                theta3_opts = [th3_1, -th3_1]

                for th3 in theta3_opts:
                    th2 = math.atan2(-P14_z, P14_x) - math.asin(-self.a3 * math.sin(th3) / math.sqrt(L_sq))
                    T12 = np.array([
                        [math.cos(th2), -math.sin(th2), 0, 0],
                        [math.sin(th2),  math.cos(th2), 0, 0],
                        [0, 0, 1, 0], [0, 0, 0, 1]
                    ])
                    T23 = np.array([
                        [math.cos(th3), -math.sin(th3), 0, self.a2],
                        [math.sin(th3),  math.cos(th3), 0, 0],
                        [0, 0, 1, 0], [0, 0, 0, 1]
                    ])
                    T34 = np.linalg.inv(T12 @ T23) @ T14
                    th4 = math.atan2(T34[1, 0], T34[0, 0])  

                    sol=[
                        (th1+math.pi)%(2*math.pi) - math.pi,
                        (th2+math.pi)%(2*math.pi) - math.pi,                        
                        (th3+math.pi)%(2*math.pi) - math.pi,
                        (th4+math.pi)%(2*math.pi) - math.pi,                        
                        (th5+math.pi)%(2*math.pi) - math.pi,
                        (th6+math.pi)%(2*math.pi) - math.pi]
                    solutions.append(sol)
        return solutions
              