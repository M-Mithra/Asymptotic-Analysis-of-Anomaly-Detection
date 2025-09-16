import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --- REQUIRED IMPORTS FOR VISUALIZATION ---
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

# ===== HELPER FUNCTION FOR KL DIVERGENCE =====
def kl_divergence(p, q):
    """ Calculates KL Divergence D_KL(p || q) for discrete distributions. """
    epsilon = 1e-10
    p_safe = p + epsilon
    q_safe = q + epsilon
    return np.sum(p_safe * np.log(p_safe / q_safe))

# ===== HELPER FUNCTION TO PLOT 3D SOLID CLUSTER SHAPES =====
def plot_solid_cluster_shapes_3d(ax, x, y, z, face_color):
    if x.size == 0: return
    points = np.vstack([x, y, z]).T
    clustering = DBSCAN(eps=step * 3.5, min_samples=20).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)
    unique_labels.discard(-1)
    for k in unique_labels:
        cluster_points = points[labels == k]
        if len(cluster_points) < 4: continue
        try:
            hull = ConvexHull(cluster_points)
            triangles = [cluster_points[s] for s in hull.simplices]
            surface = Poly3DCollection(triangles, facecolor=face_color, edgecolor='k', alpha=0.5, linewidth=0.2)
            ax.add_collection3d(surface)
        except Exception as e:
            print(f"Could not compute 3D hull for a cluster: {e}")

# ===== HELPER FUNCTION TO PLOT 2D SOLID CLUSTER SHAPES (FOR FRAMES) =====
def plot_solid_cluster_shapes_2d(ax, x, y, face_color):
    if x.size == 0: return
    points_2d = np.vstack([x, y]).T
    # For 2D, a hull needs at least 3 points
    if len(points_2d) < 3:
        ax.scatter(x, y, color=face_color, s=15) # Plot as points if no hull can be formed
        return
    try:
        hull = ConvexHull(points_2d)
        # Create a polygon from the hull vertices
        polygon = Polygon(points_2d[hull.vertices,:], facecolor=face_color, alpha=0.5, edgecolor='k', linewidth=1)
        ax.add_patch(polygon)
    except Exception as e:
        # If hull fails, fall back to plotting points
        ax.scatter(x, y, color=face_color, s=15, alpha=0.6)
        print(f"Could not compute 2D hull, plotting points instead: {e}")

# ===== MAIN SCRIPT =====
# --- SETTINGS ---
step = 0.025
save_folder = input("Enter main folder to save all files: ").strip()
os.makedirs(save_folder, exist_ok=True)

# --- DATA GENERATION AND CLASSIFICATION ---
all_points = []
print("Running outlier detection with KL Divergence...")
t_values = np.arange(0, 1.0001, step)
for t1 in t_values:
    for t2 in t_values:
        for t3 in t_values:
            point_data = {'t1': t1, 't2': t2, 't3': t3}
            if np.isclose(t1, t2) and np.isclose(t2, t3):
                point_data['outlier_class'] = 'No Outlier'
            else:
                t = np.array([t1, t2, t3])
                P1, P2, P3 = np.array([t[0], 1-t[0]]), np.array([t[1], 1-t[1]]), np.array([t[2], 1-t[2]])
                Q1, Q2, Q3 = (P2+P3)/2, (P1+P3)/2, (P1+P2)/2
                D = np.array([kl_divergence(P1, Q1), kl_divergence(P2, Q2), kl_divergence(P3, Q3)])
                max_score = np.max(D)
                indices_of_max = np.where(np.isclose(D, max_score))[0]
                if len(indices_of_max) > 1:
                    point_data['outlier_class'] = 'Ambiguous'
                else:
                    outlier_idx = np.argmax(D)
                    point_data['outlier_class'] = f'Outlier {outlier_idx + 1}'
            all_points.append(point_data)
print("Calculation complete.")

# --- DATA HANDLING AND SAVING ---
print("Organizing data and saving to files...")
df = pd.DataFrame(all_points)
df.to_excel(os.path.join(save_folder, 'outlier_classification_data_kl.xlsx'), index=False)
# (omitting the main 3D plots from this snippet to focus on the new frame generation)
print("Main data files saved.")

# =================================================================================
# vv NEW: FRAME-BY-FRAME VISUALIZATION GENERATION vv
# =================================================================================
print("\nStarting frame-by-frame visualization generation...")

# --- 1. CREATE FOLDER STRUCTURE ---
grouped_folder = os.path.join(save_folder, 'Grouped_Frames')
ungrouped_folder = os.path.join(save_folder, 'Ungrouped_Frames')
directions = ['X_Direction', 'Y_Direction', 'Z_Direction']
for base_folder in [grouped_folder, ungrouped_folder]:
    for direction in directions:
        os.makedirs(os.path.join(base_folder, direction), exist_ok=True)
print("Directory structure for frames created.")

# --- 2. DEFINE SLICING PARAMETERS ---
slice_step = 0.1
slice_values = np.arange(0, 1.001, slice_step)
tolerance = step / 2.0 # A small tolerance to grab points on the slice

# Mapping for colors and markers
color_map = {'Outlier 1': 'red', 'Outlier 2': 'green', 'Outlier 3': 'blue', 'Ambiguous': 'orange', 'No Outlier': 'black'}
marker_map = {'Outlier 1': 'x', 'Outlier 2': 'o', 'Outlier 3': 's', 'Ambiguous': 'P', 'No Outlier': 'D'}
class_names = list(color_map.keys())

# --- 3. LOOP THROUGH DIRECTIONS AND SLICES TO CREATE PLOTS ---
axis_map = {
    'X': {'col': 't1', 'labels': ('t2', 't3')},
    'Y': {'col': 't2', 'labels': ('t1', 't3')},
    'Z': {'col': 't3', 'labels': ('t1', 't2')}
}

for direction_char, details in axis_map.items():
    col_to_slice = details['col']
    x_label, y_label = details['labels']
    
    for val in slice_values:
        slice_val_str = f"{val:.1f}"
        print(f"  - Generating frames for slice {direction_char} = {slice_val_str}")
        
        # Filter the dataframe to get the points on this slice
        slice_df = df[np.isclose(df[col_to_slice], val, atol=tolerance)]
        
        if slice_df.empty:
            continue
            
        # --- Create Ungrouped (Scatter) Plot ---
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8))
        for name in class_names:
            subset = slice_df[slice_df['outlier_class'] == name]
            if not subset.empty:
                ax_scatter.scatter(subset[x_label], subset[y_label], c=color_map[name], marker=marker_map[name], label=name, s=50)
        ax_scatter.set_title(f'Ungrouped Slice at {direction_char} = {slice_val_str}'); ax_scatter.set_xlabel(x_label); ax_scatter.set_ylabel(y_label)
        ax_scatter.set_xlim(0, 1); ax_scatter.set_ylim(0, 1); ax_scatter.legend(); ax_scatter.grid(True)
        scatter_path = os.path.join(ungrouped_folder, f'{direction_char}_Direction', f'slice_{direction_char}_{slice_val_str}.png')
        fig_scatter.savefig(scatter_path, dpi=150)
        plt.close(fig_scatter)
        
        # --- Create Grouped (Solid Shape) Plot ---
        fig_grouped, ax_grouped = plt.subplots(figsize=(8, 8))
        for name in class_names:
            subset = slice_df[slice_df['outlier_class'] == name]
            if not subset.empty:
                plot_solid_cluster_shapes_2d(ax_grouped, subset[x_label].values, subset[y_label].values, color_map[name])
        
        ax_grouped.set_title(f'Grouped Slice at {direction_char} = {slice_val_str}'); ax_grouped.set_xlabel(x_label); ax_grouped.set_ylabel(y_label)
        ax_grouped.set_xlim(0, 1); ax_grouped.set_ylim(0, 1)
        legend_patches = [Patch(facecolor=color, alpha=0.5, edgecolor='k', label=name) for name, color in color_map.items()]
        ax_grouped.legend(handles=legend_patches); ax_grouped.grid(True)
        grouped_path = os.path.join(grouped_folder, f'{direction_char}_Direction', f'slice_{direction_char}_{slice_val_str}.png')
        fig_grouped.savefig(grouped_path, dpi=150)
        plt.close(fig_grouped)

print("\nProcess complete! All frame-by-frame visualizations have been saved.")