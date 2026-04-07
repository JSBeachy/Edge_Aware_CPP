import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time
import matplotlib.colors as mcolors

class SyntheticReachability:
    def __init__(self, grid_res=150, min_configs=3, max_configs=6):
        # Randomly decide how many configurations this test will have
        self.num_configs = np.random.randint(min_configs, max_configs + 1)
        print(f"Generating synthetic mesh ({grid_res}x{grid_res}) with {self.num_configs} random configurations...")
        
        #Create a dense 2D grid
        x = np.linspace(-1.5, 1.5, grid_res)
        y = np.linspace(-1.5, 1.5, grid_res)
        xx, yy = np.meshgrid(x, y)
        self.points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()))) #Array of 150*150=22500 points (22500x3 matrix), x y and z columns
        all_masks = []
        all_scores = []

        # Generate shapes
        for c in range(self.num_configs):
            # Randomize center position
            cx = np.random.uniform(-0.8, 0.8)
            cy = np.random.uniform(-0.8, 0.8)
            
            # Randomize shape type (0: Circle, 1: Ellipse, 2: Wavy Blob)
            shape_type = np.random.choice([0, 1, 2])
            
            dist = np.sqrt((self.points[:,0] - cx)**2 + (self.points[:,1] - cy)**2)
            
            if shape_type == 0:
                # Circle
                radius = np.random.uniform(0.4, 0.8)
                boundary = np.full_like(dist, radius)
                
            elif shape_type == 1:
                # Ellipse
                angle = np.random.uniform(0, np.pi)
                rx = np.random.uniform(0.3, 0.6)
                ry = np.random.uniform(0.6, 1.0)
                px = self.points[:,0] - cx
                py = self.points[:,1] - cy
                rot_x = px * np.cos(angle) - py * np.sin(angle) #Multiply px,py by rotation matrix give rotx, roty (rotated by theta around "origin" cx, cy)
                rot_y = px * np.sin(angle) + py * np.cos(angle)
                dist = np.sqrt((rot_x / rx)**2 + (rot_y / ry)**2) # ellipse boundary has value=1, so anything > 1 is outside shape
                boundary = np.ones_like(dist) 
                
            else:
                # Blob
                base_radius = np.random.uniform(0.4, 0.7)
                frequency = np.random.randint(3, 8)
                amplitude = np.random.uniform(0.05, 0.08)
                angles = np.arctan2(self.points[:,1] - cy, self.points[:,0] - cx)
                boundary = base_radius + amplitude * np.sin(frequency * angles) #calculate outer limit of boundary at each angle, compare to overall distance

            # Create boolean mask, evauluating if each grid point is in or out of shape
            mask = dist < boundary
            
            # Create manipulability score (1.0 at center, fading to ~0 at boundary) with added noise
            raw_score = 1.0 - (dist / boundary)
            noise = np.random.normal(0, 0.05, len(self.points))
            score = np.clip(raw_score + noise, 0.001, 1.0)
            
            # If it's outside the boundary, force score to 0
            score[~mask] = 0.0
            
            all_masks.append(mask)
            all_scores.append(score)

        #Assemble the Signatures and Manipulabilities lists natively
        self.signatures = [] #Configuration options available to a single point
        self.manipulabilities = [] #Manipulability scores for each corresponding configuration
        print(np.shape(all_masks))
        
        for i in range(len(self.points)):
            sig = tuple([all_masks[c][i] for c in range(self.num_configs)]) #all_mask[c] is binary mask evaluating what points are inside shape c
            man = tuple([all_scores[c][i] for c in range(self.num_configs)]) #all_scores[c] are values representing distances from shape c center for all points 
            self.signatures.append(sig)
            self.manipulabilities.append(man)
        
        # Generate a distinct color palette for the visualizer
        cmap = plt.get_cmap('tab10')
        self.palette = [cmap(i)[:3] for i in range(self.num_configs)]
        
        print("Data generation complete.")

    def segment_into_cells(self, radius=0.03):
        print("Segmenting synthetic surface into cells...")
        tree = cKDTree(self.points) 
        self.cell_ids = -1 * np.ones(len(self.points), dtype=int)
        current_cell_id = 0
        
        for i in range(len(self.points)):#Breadth-First Search: Search neighbors in "rings" for identical configs and group into 1 cell
            if self.cell_ids[i] != -1: 
                continue
            current_sig = self.signatures[i]
            # Skip unreachable points
            if sum(current_sig) == 0:
                continue
            queue = [i]
            self.cell_ids[i] = current_cell_id
            
            while len(queue) > 0:
                idx = queue.pop(0)
                neighbors = tree.query_ball_point(self.points[idx], r=radius)
                for n_idx in neighbors:
                    if self.cell_ids[n_idx] == -1:
                        if self.signatures[n_idx] == current_sig:
                            self.cell_ids[n_idx] = current_cell_id #If the neighbor has identical configuration, add it to the same cell
                            queue.append(n_idx)
            current_cell_id += 1
        print(f"Segmentation complete. Found {current_cell_id} unique reachable geographic cells.")    

    def visualize_raw_overlaps(self):
        """Plots the raw data so you can see the Anchors vs. Overlaps."""
        colors = np.zeros((len(self.points), 3))
        
        for i, sig in enumerate(self.signatures):
            valid_count = sum(sig)
            if valid_count == 0:
                colors[i] = [0.9, 0.9, 0.9] # Light grey for unreachable
            elif valid_count == 1:
                # Anchors get the pure configuration color
                config_idx = sig.index(True)
                colors[i] = self.palette[config_idx]
            else:
                # Overlaps blend the colors of all valid configurations
                blended_color = np.zeros(3)
                for c, is_valid in enumerate(sig):
                    if is_valid:
                        blended_color += np.array(self.palette[c])
                colors[i] = blended_color / valid_count

        plt.figure(figsize=(10, 8))
        plt.scatter(self.points[:,0], self.points[:,1], c=colors, s=2)
        plt.title(f"Ground Truth: {self.num_configs} Configurations\nAnchors (Pure) vs. Overlaps (Mixed)")
        plt.axis('equal')
        plt.show()

    def visualize_resolved_cells(self, title="Resolved Cells"):
        """Plots the final resolved configurations."""
        colors = np.zeros((len(self.points), 3))
        
        for i, label in enumerate(self.cell_ids):
            if label == -1: 
                colors[i] = [0.9, 0.9, 0.9]
            else:
                colors[i] = self.palette[label]
            
        plt.figure(figsize=(10, 8))
        plt.scatter(self.points[:,0], self.points[:,1], c=colors, s=2)
        plt.title(title)
        plt.axis('equal')
        plt.show()

    # ==========================================
    # PASTE YOUR RESOLUTION FUNCTIONS HERE
    # - resolve_overlapping_greedy_smooth()
    # - resolve_overlapping_regions() (MRF)
    # ==========================================

if __name__ == "__main__":
    # 1. Generate the test dataset
    testbed = SyntheticReachability(grid_res=150, min_configs=3, max_configs=5)
    
    # Show the problem
    testbed.visualize_raw_overlaps()
    
    # 2. Test the Algorithms
    radius = 0.03
    
    # To test MRF:
    # testbed.resolve_overlapping_regions(radius=radius, alpha=1.0, beta=3.0)
    # testbed.visualize_resolved_cells("MRF Optimization Result")
    
    # To test Greedy + Smoothing:
    # testbed.resolve_overlapping_greedy_smooth(radius=radius, smoothing_iterations=5)
    # testbed.visualize_resolved_cells("Greedy + Smoothing Result")
