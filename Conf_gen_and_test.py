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

    def visualize_cell_ids(self, title="Geographic Cells (Signatures)"):
        """Colors points strictly by their assigned geographic cell_id."""
        unique_cells = np.unique(self.cell_ids)
        
        # Create a distinct random color for each cell
        np.random.seed(42) # Keeps colors consistent across runs
        cell_palette = {cid: np.random.rand(3,) for cid in unique_cells}
        cell_palette[-1] = np.array([0.9, 0.9, 0.9]) # Grey for unreachable
        
        colors = np.array([cell_palette[cid] for cid in self.cell_ids])
        
        plt.figure(figsize=(10, 8))
        plt.scatter(self.points[:,0], self.points[:,1], c=colors, s=2)
        valid_cell_count = len(unique_cells) - 1 if -1 in unique_cells else len(unique_cells)
        plt.title(f"{title}\n{valid_cell_count} Unique Geographic Regions")
        plt.axis('equal')
        plt.show()
        
    def visualize_resolved_cells(self, title="Resolved Configurations"):
        """Plots the final assigned configurations after resolution."""
        if not hasattr(self, 'assigned_configs'):
            print("Error: Run a resolution algorithm first.")
            return
            
        colors = np.zeros((len(self.points), 3))
        for i, config_idx in enumerate(self.assigned_configs):
            if config_idx == -1: 
                colors[i] = [0.9, 0.9, 0.9]
            else:
                colors[i] = self.palette[config_idx]
            
        plt.figure(figsize=(10, 8))
        plt.scatter(self.points[:,0], self.points[:,1], c=colors, s=2)
        plt.title(title)
        plt.axis('equal')
        plt.show()

    def resolve_overlapping_greedy_smooth(self, radius=0.03, smoothing_iterations=5):
        print(f"\n--- Resolving Multi-Config Cells (Greedy + Smooth) ---")
        start_time = time.time()
        tree = cKDTree(self.points)
        
        # Initialize decisions array
        self.assigned_configs = -1 * np.ones(len(self.points), dtype=int)
        locked = np.zeros(len(self.points), dtype=bool)
        
        anchor_cells = 0
        overlap_cells = 0
        
        # Cell-by-cell initialization
        unique_cells = np.unique(self.cell_ids)
        for cell_id in unique_cells:
            if cell_id == -1: continue
            
            cell_indices = np.where(self.cell_ids == cell_id)[0]
            sig = self.signatures[cell_indices[0]]
            valid_configs = sum(sig)
            
            if valid_configs == 1: # Anchors
                config_idx = sig.index(True)
                self.assigned_configs[cell_indices] = config_idx
                locked[cell_indices] = True
                anchor_cells += 1
            else: # Overlaps
                for idx in cell_indices:
                    valid_manips = [m if v else -1 for m, v in zip(self.manipulabilities[idx], sig)]
                    self.assigned_configs[idx] = np.argmax(valid_manips)
                locked[cell_indices] = False
                overlap_cells += 1
        self.flip_history=[]#convergence tracking
                
        print(f"  Processed {anchor_cells} Anchor Cells and {overlap_cells} Overlap Cells.")
        
        # Majority-vote Smoothing Loop
        for iteration in range(smoothing_iterations):
            flips = 0
            new_configs = np.copy(self.assigned_configs)
            
            for i in range(len(self.points)):
                if locked[i] or self.assigned_configs[i] == -1:
                    continue
                
                neighbors = tree.query_ball_point(self.points[i], r=radius)
                if len(neighbors) <= 1: continue
                
                label_counts = {}
                for n in neighbors:
                    lbl = self.assigned_configs[n]
                    if lbl != -1:
                        label_counts[lbl] = label_counts.get(lbl, 0) + 1
                        
                if not label_counts: continue
                
                best_label = max(label_counts, key=label_counts.get)
                if best_label != self.assigned_configs[i] and self.signatures[i][best_label]:
                    new_configs[i] = best_label
                    flips += 1
                    
            self.assigned_configs = new_configs
            self.flip_history.append(flips)
            print(f"  Smoothing Iteration {iteration+1}: {flips} points flipped.")
            if flips == 0: break
            
        print(f"Resolution complete in {time.time()-start_time:.2f}s.")

    def resolve_overlapping_regions_MRF(self, radius=0.03, alpha=1.0, beta=2.5, iterations=150000):
        print(f"\n--- Resolving Multi-Config Cells (MRF: alpha={alpha}, beta={beta}) ---")
        start_time = time.time()
        
        # Build Graph
        tree = cKDTree(self.points)
        edges = tree.query_pairs(r=radius)
        neighbors = {i: [] for i in range(len(self.points))}
        for i, j in edges:
            neighbors[i].append(j)
            neighbors[j].append(i)
            
        # Initialize decisions array
        self.assigned_configs = -1 * np.ones(len(self.points), dtype=int)
        locked = np.zeros(len(self.points), dtype=bool)
        
        anchor_cells = 0
        overlap_cells = 0
        
        # Cell-by-cell initialization
        unique_cells = np.unique(self.cell_ids)
        for cell_id in unique_cells:
            if cell_id == -1: continue
            
            cell_indices = np.where(self.cell_ids == cell_id)[0]
            sig = self.signatures[cell_indices[0]]
            valid_configs = sum(sig)
            
            if valid_configs == 1: #anchors
                config_idx = sig.index(True)
                self.assigned_configs[cell_indices] = config_idx
                locked[cell_indices] = True
                anchor_cells += 1
            else: #overlaps
                for idx in cell_indices:
                    valid_manips = [m if v else -1 for m, v in zip(self.manipulabilities[idx], sig)]
                    self.assigned_configs[idx] = np.argmax(valid_manips)
                locked[cell_indices] = False
                overlap_cells += 1   
        #print(f"  Processed {anchor_cells} Anchor Cells and {overlap_cells} Overlap Cells.")
        
        #Initial total energy calculation
        current_energy=0.0
        #manipulability cost
        for i in range(len(self.points)):
            lbl = self.assigned_configs[i]
            if lbl != -1:
                current_energy -= alpha * self.manipulabilities[i][lbl]
        #Add smoothness cost
        for i, j in edges:
            if self.assigned_configs[i] != -1 and self.assigned_configs[j] != -1:
                if self.assigned_configs[i] != self.assigned_configs[j]:
                    current_energy += beta
        # Build dynamic boundary list
        boundary_list = []
        for i in range(len(self.points)):
            if locked[i] or self.assigned_configs[i] == -1: continue
            for n in neighbors[i]:
                if self.assigned_configs[n] != -1 and self.assigned_configs[n] != self.assigned_configs[i]:
                    boundary_list.append(i)
                    break

        # --- Convergence Tracking Setup ---
        self.energy_history = [current_energy]
        self.mrf_track_interval = 500 # Record data every 500 attempts
        total_flips = 0
                    
        # ICM Delta-Update Loop
        flips = 0
        for step in range(iterations):
            if not boundary_list: break
            
            idx = boundary_list[np.random.choice(len(boundary_list))]
            old_label = self.assigned_configs[idx]
            
            neighbor_labels = [self.assigned_configs[n] for n in neighbors[idx] if self.assigned_configs[n] != -1]
            if not neighbor_labels: continue
            
            new_label = np.random.choice(neighbor_labels)
            if new_label == old_label: continue
            if not self.signatures[idx][new_label]: continue
            
            # Delta Energy Math
            delta_data = self.manipulabilities[idx][old_label] - self.manipulabilities[idx][new_label]
            matches_old = sum(1 for n in neighbors[idx] if self.assigned_configs[n] == old_label)
            matches_new = sum(1 for n in neighbors[idx] if self.assigned_configs[n] == new_label)
            delta_smooth = matches_old - matches_new
            
            delta_E = (alpha * delta_data) + (beta * delta_smooth)
            
            if delta_E < 0:
                self.assigned_configs[idx] = new_label
                flips += 1
                # Add affected unlocked neighbors back to check list
                for n in neighbors[idx]:
                    if not locked[n] and self.assigned_configs[n] != -1:
                        boundary_list.append(n)

            # Record Energy periodically
            if (step + 1) % self.mrf_track_interval == 0:
                self.energy_history.append(current_energy)

        # Catch the final energy state
        if len(self.energy_history) == 1 or self.energy_history[-1] != current_energy:
            self.energy_history.append(current_energy)
                        
        print(f"Resolution complete in {time.time()-start_time:.2f}s. {flips} boundary flips performed.")

    def plot_convergence(self):
        """Plots the convergence of the resolution algorithm (Energy or Flips)."""
        plt.figure(figsize=(8, 4))
        
        if hasattr(self, 'energy_history') and self.energy_history:
            # We ran the MRF algorithm
            plt.plot(range(len(self.energy_history)), self.energy_history, color='purple', linewidth=2)
            plt.title("MRF Optimization: System Energy over Time")
            plt.xlabel(f'Epochs (1 Epoch = {self.mrf_track_interval} iterations)')
            plt.ylabel('Total System Energy (Lower is Better)')
            
            # Add a marker for the final stabilization point
            plt.scatter(len(self.energy_history)-1, self.energy_history[-1], color='red', zorder=5)
            plt.text(len(self.energy_history)-1, self.energy_history[-1], 
                     f" Final: {self.energy_history[-1]:.1f}", verticalalignment='bottom')
            
        elif hasattr(self, 'flip_history') and self.flip_history:
            # We ran the Greedy+Smooth algorithm
            plt.plot(range(1, len(self.flip_history) + 1), self.flip_history, marker='o', linestyle='-')
            plt.title("Greedy + Smoothing: Boundary Flips per Sweep")
            plt.xlabel('Smoothing Iteration (Full Board Sweep)')
            plt.ylabel('Number of Flips Accepted')
            
        else:
            print("No history found. Run a resolution algorithm first.")
            return

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 1. Generate the test dataset
    testbed = SyntheticReachability(grid_res=150, min_configs=3, max_configs=5)
    testbed.segment_into_cells(radius=0.03)
    # Show the problem
    testbed.visualize_raw_overlaps()
    testbed.visualize_cell_ids()
    
    #Test the Algorithms
    radius = 0.03

    # To test Greedy + Smoothing:
    testbed.resolve_overlapping_greedy_smooth(radius=radius, smoothing_iterations=5)
    testbed.plot_convergence()
    testbed.visualize_resolved_cells("Greedy + Smoothing Result")
    
    # To test MRF:
    testbed.resolve_overlapping_regions_MRF(radius=radius, alpha=1.0, beta=3.0)
    testbed.plot_convergence()
    testbed.visualize_resolved_cells("MRF Optimization Result")
    
