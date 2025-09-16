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

# ===== HELPER FUNCTION FOR TOTAL VARIATION DISTANCE =====
def total_variation_distance(p, q):
    """ Calculates the Total Variation Distance between two discrete distributions. """
    return 0.5 * np.sum(np.abs(p - q))

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
    if len(points_2d) < 3:
        ax.scatter(x, y, color=face_color, s=15)
        return
    try:
        hull = ConvexHull(points_2d)
        polygon = Polygon(points_2d[hull.vertices,:], facecolor=face_color, alpha=0.5, edgecolor='k', linewidth=1)
        ax.add_patch(polygon)
    except Exception as e:
        ax.scatter(x, y, color=face_color, s=15, alpha=0.6)
        print(f"Could not compute 2D hull, plotting points instead: {e}")

# ===== MAIN SCRIPT =====
# --- SETTINGS ---
step = 0.025
save_folder = input("Enter main folder to save all files (Total Variation version): ").strip()
os.makedirs(save_folder, exist_ok=True)

# --- DATA GENERATION AND CLASSIFICATION ---
all_points = []
print("Running outlier detection with Total Variation Distance...")
t_values = np.arange(0, 1.0001, step)
for t1 in t_values:
    for t2 in t_values:
        for t3 in t_values:
            point_data = {'t1': t1, 't2': t2, 't3': t3}
            if np.isclose(t1, t2) and np.isclose(t2, t3):
                point_data['outlier_class'] = 'No Outlier'
            else:
                # === NEW LOGIC USING TOTAL VARIATION DISTANCE ===
                t = np.array([t1, t2, t3])
                P1, P2, P3 = np.array([t[0], 1-t[0]]), np.array([t[1], 1-t[1]]), np.array([t[2], 1-t[2]])
                Q1, Q2, Q3 = (P2+P3)/2, (P1+P3)/2, (P1+P2)/2
                D = np.array([total_variation_distance(P1, Q1), 
                              total_variation_distance(P2, Q2), 
                              total_variation_distance(P3, Q3)])
                
                # --- AMBIGUITY DETECTION LOGIC ---
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
df.to_excel(os.path.join(save_folder, 'outlier_classification_data_tv.xlsx'), index=False)

df_red = df[df['outlier_class'] == 'Outlier 1']; x_red, y_red, z_red = df_red['t1'].values, df_red['t2'].values, df_red['t3'].values
df_green = df[df['outlier_class'] == 'Outlier 2']; x_green, y_green, z_green = df_green['t1'].values, df_green['t2'].values, df_green['t3'].values
df_blue = df[df['outlier_class'] == 'Outlier 3']; x_blue, y_blue, z_blue = df_blue['t1'].values, df_blue['t2'].values, df_blue['t3'].values
df_black = df[df['outlier_class'] == 'No Outlier']; x_black, y_black, z_black = df_black['t1'].values, df_black['t2'].values, df_black['t3'].values
df_orange = df[df['outlier_class'] == 'Ambiguous']; x_orange, y_orange, z_orange = df_orange['t1'].values, df_orange['t2'].values, df_orange['t3'].values

npz_path = os.path.join(save_folder, 'outlier_classification_data_tv.npz')
np.savez(npz_path, x_red=x_red, y_red=y_red, z_red=z_red, x_green=x_green, y_green=y_green, z_green=z_green, x_blue=x_blue, y_blue=y_blue, z_blue=z_blue, x_black=x_black, y_black=y_black, z_black=z_black, x_orange=x_orange, y_orange=y_orange, z_orange=z_orange)
print("Main data files saved.")

# =================================================================================
# vv GENERATING THE 4 MAIN 3D PLOTS vv
# =================================================================================
print("\nGenerating the four main 3D plots...")
# --- PLOT 1: MAIN-STYLE SCATTER PLOT ---
fig1 = plt.figure(figsize=(12, 10)); ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_title("Main-Style Scatter Plot (Total Variation)"); ax1.set_xlabel("t1"); ax1.set_ylabel("t2"); ax1.set_zlabel("t3")
ax1.scatter(x_red, y_red, z_red, c='red', marker='x', s=15, alpha=0.6, label='Outlier 1')
ax1.scatter(x_green, y_green, z_green, c='green', marker='o', s=15, alpha=0.6, label='Outlier 2')
ax1.scatter(x_blue, y_blue, z_blue, c='blue', marker='s', s=15, alpha=0.6, label='Outlier 3')
ax1.scatter(x_black, y_black, z_black, c='black', marker='D', s=20, label='No Outlier')
ax1.scatter(x_orange, y_orange, z_orange, c='orange', marker='P', s=20, label='Ambiguous')
ax1.legend(loc='upper left'); ax1.view_init(elev=25, azim=-60)
plt.savefig(os.path.join(save_folder, "1_main_scatter_tv.png"), dpi=300); plt.show()

# --- PLOT 2: MAIN-STYLE GROUPED PLOT ---
fig2 = plt.figure(figsize=(12, 12)); ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_title("Main-Style Solid Shapes (Total Variation)"); ax2.set_xlabel("t1"); ax2.set_ylabel("t2"); ax2.set_zlabel("t3")
ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1]); ax2.set_zlim([0, 1]); ax2.view_init(elev=25, azim=-60)
ax2.xaxis.set_pane_color((1,1,1,0)); ax2.yaxis.set_pane_color((1,1,1,0)); ax2.zaxis.set_pane_color((1,1,1,0)); ax2.grid(False)
plot_solid_cluster_shapes_3d(ax2, x_red, y_red, z_red, 'red'); plot_solid_cluster_shapes_3d(ax2, x_green, y_green, z_green, 'green'); plot_solid_cluster_shapes_3d(ax2, x_blue, y_blue, z_blue, 'blue'); plot_solid_cluster_shapes_3d(ax2, x_orange, y_orange, z_orange, 'orange')
ax2.plot([0, 1], [0, 1], [0, 1], color='black', linewidth=3, linestyle='--')
legend_elements = [Patch(fc='red', alpha=0.5, ec='k', label='Outlier 1'), Patch(fc='green', alpha=0.5, ec='k', label='Outlier 2'), Patch(fc='blue', alpha=0.5, ec='k', label='Outlier 3'), Patch(fc='orange', alpha=0.5, ec='k', label='Ambiguous'), Line2D([0], [0], color='black', lw=3, ls='--', label='No Outlier')]
ax2.legend(handles=legend_elements, loc='upper left')
plt.savefig(os.path.join(save_folder, "2_main_grouped_tv.png"), dpi=300); plt.show()

# --- PLOT 3: TEXTBOOK-STYLE SCATTER PLOT ---
fig3 = plt.figure(figsize=(14, 12)); ax3 = fig3.add_subplot(111, projection='3d')
ax3.set_title("Textbook-Style Scatter Plot (Total Variation)"); ax3.view_init(elev=25, azim=-60); ax3.set_axis_off()
ax3.scatter(x_red, y_red, z_red, c='red', marker='x', s=15, alpha=0.6, label='Outlier 1'); ax3.scatter(x_green, y_green, z_green, c='green', marker='o', s=15, alpha=0.6, label='Outlier 2'); ax3.scatter(x_blue, y_blue, z_blue, c='blue', marker='s', s=15, alpha=0.6, label='Outlier 3'); ax3.scatter(x_black, y_black, z_black, c='black', marker='D', s=20, label='No Outlier'); ax3.scatter(x_orange, y_orange, z_orange, c='orange', marker='P', s=20, label='Ambiguous')
ax3.legend(loc='upper left')
axis_len = 1.1; ax3.quiver(0,0,0, axis_len,0,0, color='k', arrow_length_ratio=0.05, lw=1); ax3.quiver(0,0,0, 0,axis_len,0, color='k', arrow_length_ratio=0.05, lw=1); ax3.quiver(0,0,0, 0,0,axis_len, color='k', arrow_length_ratio=0.05, lw=1)
ax3.text(axis_len,0,0, ' t1', color='k', fontsize=14); ax3.text(0,axis_len,0, ' t2', color='k', fontsize=14); ax3.text(0,0,axis_len, ' t3', color='k', fontsize=14)
tick_values = np.arange(0.2, 1.1, 0.2); tick_font_size = 8
for val in tick_values:
    ax3.text(val, -0.05, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center'); ax3.text(-0.05, val, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center'); ax3.text(-0.05, 0, val, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center')
plt.savefig(os.path.join(save_folder, "3_textbook_scatter_tv.png"), dpi=300); plt.show()

# --- PLOT 4: TEXTBOOK-STYLE GROUPED PLOT ---
fig4 = plt.figure(figsize=(14, 14)); ax4 = fig4.add_subplot(111, projection='3d')
ax4.set_title("Textbook-Style Solid Shapes (Total Variation)"); ax4.view_init(elev=25, azim=-60); ax4.set_axis_off()
plot_solid_cluster_shapes_3d(ax4, x_red, y_red, z_red, 'red'); plot_solid_cluster_shapes_3d(ax4, x_green, y_green, z_green, 'green'); plot_solid_cluster_shapes_3d(ax4, x_blue, y_blue, z_blue, 'blue'); plot_solid_cluster_shapes_3d(ax4, x_orange, y_orange, z_orange, 'orange')
ax4.plot([0, 1], [0, 1], [0, 1], color='black', linewidth=3, linestyle='--')
ax4.quiver(0,0,0, axis_len,0,0, color='k', arrow_length_ratio=0.05, lw=1); ax4.quiver(0,0,0, 0,axis_len,0, color='k', arrow_length_ratio=0.05, lw=1); ax4.quiver(0,0,0, 0,0,axis_len, color='k', arrow_length_ratio=0.05, lw=1)
ax4.text(axis_len,0,0, ' t1', color='k', fontsize=14); ax4.text(0,axis_len,0, ' t2', color='k', fontsize=14); ax4.text(0,0,axis_len, ' t3', color='k', fontsize=14)
ax4.legend(handles=legend_elements, loc='upper left')
for val in tick_values:
    ax4.text(val, -0.05, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center'); ax4.text(-0.05, val, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center'); ax4.text(-0.05, 0, val, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center')
plt.savefig(os.path.join(save_folder, "4_textbook_grouped_tv.png"), dpi=300); plt.show()

# =================================================================================
# vv GENERATING FRAME-BY-FRAME VISUALIZATIONS vv
# =================================================================================
print("\nStarting frame-by-frame visualization generation...")
# --- 1. CREATE FOLDER STRUCTURE ---
grouped_folder = os.path.join(save_folder, 'Grouped_Frames'); ungrouped_folder = os.path.join(save_folder, 'Ungrouped_Frames')
directions = ['X_Direction', 'Y_Direction', 'Z_Direction']
for base_folder in [grouped_folder, ungrouped_folder]:
    for direction in directions:
        os.makedirs(os.path.join(base_folder, direction), exist_ok=True)
print("Directory structure for frames created.")

# --- 2. DEFINE SLICING PARAMETERS ---
slice_step = 0.1; slice_values = np.arange(0, 1.001, slice_step); tolerance = step / 2.0
color_map = {'Outlier 1': 'red', 'Outlier 2': 'green', 'Outlier 3': 'blue', 'Ambiguous': 'orange', 'No Outlier': 'black'}
marker_map = {'Outlier 1': 'x', 'Outlier 2': 'o', 'Outlier 3': 's', 'Ambiguous': 'P', 'No Outlier': 'D'}
class_names = list(color_map.keys())
axis_map = {'X': {'col': 't1', 'labels': ('t2', 't3')}, 'Y': {'col': 't2', 'labels': ('t1', 't3')}, 'Z': {'col': 't3', 'labels': ('t1', 't2')}}

# --- 3. LOOP THROUGH DIRECTIONS AND SLICES TO CREATE PLOTS ---
for direction_char, details in axis_map.items():
    col_to_slice, (x_label, y_label) = details['col'], details['labels']
    for val in slice_values:
        slice_val_str = f"{val:.1f}"; print(f"  - Generating frames for slice {direction_char} = {slice_val_str}")
        slice_df = df[np.isclose(df[col_to_slice], val, atol=tolerance)]
        if slice_df.empty: continue
        
        # Create Ungrouped (Scatter) Plot
        fig_s, ax_s = plt.subplots(figsize=(8, 8))
        for name in class_names:
            subset = slice_df[slice_df['outlier_class'] == name]
            if not subset.empty: ax_s.scatter(subset[x_label], subset[y_label], c=color_map[name], marker=marker_map[name], label=name, s=50)
        ax_s.set_title(f'Ungrouped Slice at {direction_char}={slice_val_str} (TV)'); ax_s.set_xlabel(x_label); ax_s.set_ylabel(y_label)
        ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1); ax_s.legend(); ax_s.grid(True)
        scatter_path = os.path.join(ungrouped_folder, f'{direction_char}_Direction', f'slice_{slice_val_str}_tv.png')
        fig_s.savefig(scatter_path, dpi=150); plt.close(fig_s)
        
        # Create Grouped (Solid Shape) Plot
        fig_g, ax_g = plt.subplots(figsize=(8, 8))
        for name in class_names:
            subset = slice_df[slice_df['outlier_class'] == name]
            if not subset.empty: plot_solid_cluster_shapes_2d(ax_g, subset[x_label].values, subset[y_label].values, color_map[name])
        ax_g.set_title(f'Grouped Slice at {direction_char}={slice_val_str} (TV)'); ax_g.set_xlabel(x_label); ax_g.set_ylabel(y_label)
        ax_g.set_xlim(0, 1); ax_g.set_ylim(0, 1)
        legend_patches = [Patch(fc=color, alpha=0.5, ec='k', label=name) for name, color in color_map.items()]
        ax_g.legend(handles=legend_patches); ax_g.grid(True)
        grouped_path = os.path.join(grouped_folder, f'{direction_char}_Direction', f'slice_{slice_val_str}_tv.png')
        fig_g.savefig(grouped_path, dpi=150); plt.close(fig_g)

print(f"\nProcess complete! All files have been saved in the folder: '{save_folder}'")