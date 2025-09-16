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
from matplotlib.patches import Patch

# ===== HELPER FUNCTION FOR MMD (RBF KERNEL) =====
def rbf_kernel(x, y, gamma):
    """ Computes the RBF kernel between two points. """
    return np.exp(-gamma * (x - y)**2)

# ===== HELPER FUNCTION TO PLOT SOLID CLUSTER SHAPES =====
def plot_solid_cluster_shapes(ax, x, y, z, face_color):
    """
    Finds distinct clusters and plots their convex hull as a solid,
    semi-transparent 3D shape.
    """
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
            surface = Poly3DCollection(triangles)
            surface.set_facecolor(face_color)
            surface.set_edgecolor('k')
            surface.set_alpha(0.5)
            surface.set_linewidth(0.2)
            ax.add_collection3d(surface)
        except Exception as e:
            print(f"Could not compute hull for a cluster: {e}")

# ===== MAIN SCRIPT =====
# --- SETTINGS ---
step = 0.05
GAMMA = 10.0 # Kernel sensitivity parameter. Higher values mean more sensitivity to small distances.
save_folder = input("Enter folder to save all images and data (MMD version): ").strip()
os.makedirs(save_folder, exist_ok=True)

# --- DATA GENERATION AND CLASSIFICATION ---
all_points = []
print("Running outlier detection with MMD...")
t_values = np.arange(0, 1.0001, step)
for t1 in t_values:
    for t2 in t_values:
        for t3 in t_values:
            point_data = {'t1': t1, 't2': t2, 't3': t3}
            if t1 == t2 and t2 == t3:
                point_data['outlier_class'] = 'No Outlier'
            else:
                # === NEW LOGIC USING MMD ===
                t = np.array([t1, t2, t3])
                # Calculate kernel similarity for each pair
                k23 = rbf_kernel(t[1], t[2], gamma=GAMMA) # Similarity of pair (t2, t3)
                k13 = rbf_kernel(t[0], t[2], gamma=GAMMA) # Similarity of pair (t1, t3)
                k12 = rbf_kernel(t[0], t[1], gamma=GAMMA) # Similarity of pair (t1, t2)
                
                # The outlier is the one left out of the MOST similar pair (max kernel value)
                # D[0] corresponds to leaving t1 out, D[1] to leaving t2 out, etc.
                D = np.array([k23, k13, k12])
                outlier_idx = np.argmax(D)
                point_data['outlier_class'] = f'Outlier {outlier_idx + 1}'
            all_points.append(point_data)
print("Calculation complete.")

# --- DATA HANDLING AND SAVING ---
print("Organizing data and saving to files...")
df = pd.DataFrame(all_points)

# Save to Excel
excel_path = os.path.join(save_folder, 'outlier_classification_data_mmd.xlsx')
df.to_excel(excel_path, index=False)
print(f"Data saved to {excel_path}")

# Extract coordinate arrays
df_red = df[df['outlier_class'] == 'Outlier 1']; x_red, y_red, z_red = df_red['t1'].values, df_red['t2'].values, df_red['t3'].values
df_green = df[df['outlier_class'] == 'Outlier 2']; x_green, y_green, z_green = df_green['t1'].values, df_green['t2'].values, df_green['t3'].values
df_blue = df[df['outlier_class'] == 'Outlier 3']; x_blue, y_blue, z_blue = df_blue['t1'].values, df_blue['t2'].values, df_blue['t3'].values
df_black = df[df['outlier_class'] == 'No Outlier']; x_black, y_black, z_black = df_black['t1'].values, df_black['t2'].values, df_black['t3'].values

# Save to NPZ file
npz_path = os.path.join(save_folder, 'outlier_classification_data_mmd.npz')
np.savez(npz_path, x_red=x_red, y_red=y_red, z_red=z_red, x_green=x_green, y_green=y_green, z_green=z_green, x_blue=x_blue, y_blue=y_blue, z_blue=z_blue, x_black=x_black, y_black=y_black, z_black=z_black)
print(f"Data also saved to {npz_path}")

# =================================================================================
# vv ALL 4 PLOTS ARE GENERATED BELOW THIS LINE vv
# =================================================================================

# --- PLOT 1: MAIN-STYLE SCATTER PLOT (MMD) ---
print("Generating Plot 1: Main-Style Scatter...")
fig1 = plt.figure(figsize=(12, 10)); ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_title("Main-Style Scatter Plot (MMD)"); ax1.set_xlabel("t1"); ax1.set_ylabel("t2"); ax1.set_zlabel("t3")
ax1.scatter(x_red, y_red, z_red, color='red', marker='x', s=15, alpha=0.6, label='Outlier 1')
ax1.scatter(x_green, y_green, z_green, color='green', marker='o', s=15, alpha=0.6, label='Outlier 2')
ax1.scatter(x_blue, y_blue, z_blue, color='blue', marker='s', s=15, alpha=0.6, label='Outlier 3')
ax1.scatter(x_black, y_black, z_black, color='black', marker='D', s=20, label='No Outlier')
ax1.legend(loc='upper left'); ax1.view_init(elev=25, azim=-60)
plt.savefig(os.path.join(save_folder, "1_main_scatter_mmd.png"), dpi=300); plt.show()

# --- PLOT 2: MAIN-STYLE GROUPED PLOT (MMD) ---
print("Generating Plot 2: Main-Style Grouped Shapes...")
fig2 = plt.figure(figsize=(12, 12)); ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_title("Main-Style Solid Shapes (MMD)"); ax2.set_xlabel("t1"); ax2.set_ylabel("t2"); ax2.set_zlabel("t3")
ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1]); ax2.set_zlim([0, 1]); ax2.view_init(elev=25, azim=-60)
ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)); ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)); ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)); ax2.grid(False)
plot_solid_cluster_shapes(ax2, x_red, y_red, z_red, 'red'); plot_solid_cluster_shapes(ax2, x_green, y_green, z_green, 'green'); plot_solid_cluster_shapes(ax2, x_blue, y_blue, z_blue, 'blue')
ax2.plot([0, 1], [0, 1], [0, 1], color='black', linewidth=3, linestyle='--')
legend_elements = [Patch(facecolor='red', alpha=0.5, edgecolor='k', label='Outlier 1 Region'), Patch(facecolor='green', alpha=0.5, edgecolor='k', label='Outlier 2 Region'), Patch(facecolor='blue', alpha=0.5, edgecolor='k', label='Outlier 3 Region'), Line2D([0], [0], color='black', lw=3, linestyle='--', label='No Outlier')]
ax2.legend(handles=legend_elements, loc='upper left')
plt.savefig(os.path.join(save_folder, "2_main_grouped_mmd.png"), dpi=300); plt.show()

# --- PLOT 3: TEXTBOOK-STYLE SCATTER PLOT (MMD) ---
print("Generating Plot 3: Textbook-Style Scatter...")
fig3 = plt.figure(figsize=(14, 12)); ax3 = fig3.add_subplot(111, projection='3d')
ax3.set_title("Textbook-Style Scatter Plot (MMD)"); ax3.view_init(elev=25, azim=-60); ax3.set_axis_off()
ax3.scatter(x_red, y_red, z_red, color='red', marker='x', s=15, alpha=0.6, label='Outlier 1'); ax3.scatter(x_green, y_green, z_green, color='green', marker='o', s=15, alpha=0.6, label='Outlier 2'); ax3.scatter(x_blue, y_blue, z_blue, color='blue', marker='s', s=15, alpha=0.6, label='Outlier 3'); ax3.scatter(x_black, y_black, z_black, color='black', marker='D', s=20, label='No Outlier')
ax3.legend(loc='upper left')
axis_len = 1.1
ax3.quiver(0, 0, 0, axis_len, 0, 0, color='black', arrow_length_ratio=0.05, linewidth=1); ax3.quiver(0, 0, 0, 0, axis_len, 0, color='black', arrow_length_ratio=0.05, linewidth=1); ax3.quiver(0, 0, 0, 0, 0, axis_len, color='black', arrow_length_ratio=0.05, linewidth=1)
ax3.text(axis_len, 0, 0, ' t1', color='black', fontsize=14); ax3.text(0, axis_len, 0, ' t2', color='black', fontsize=14); ax3.text(0, 0, axis_len, ' t3', color='black', fontsize=14)
tick_values = np.arange(0.2, 1.1, 0.2); tick_font_size = 8
for val in tick_values:
    ax3.text(val, -0.05, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center'); ax3.text(-0.05, val, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center'); ax3.text(-0.05, 0, val, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center')
plt.savefig(os.path.join(save_folder, "3_textbook_scatter_mmd.png"), dpi=300); plt.show()

# --- PLOT 4: TEXTBOOK-STYLE GROUPED PLOT (MMD) ---
print("Generating Plot 4: Textbook-Style Grouped Shapes...")
fig4 = plt.figure(figsize=(14, 14)); ax4 = fig4.add_subplot(111, projection='3d')
ax4.set_title("Textbook-Style Solid Shapes (MMD)"); ax4.view_init(elev=25, azim=-60); ax4.set_axis_off()
plot_solid_cluster_shapes(ax4, x_red, y_red, z_red, 'red'); plot_solid_cluster_shapes(ax4, x_green, y_green, z_green, 'green'); plot_solid_cluster_shapes(ax4, x_blue, y_blue, z_blue, 'blue')
ax4.plot([0, 1], [0, 1], [0, 1], color='black', linewidth=3, linestyle='--')
ax4.quiver(0, 0, 0, axis_len, 0, 0, color='black', arrow_length_ratio=0.05, linewidth=1); ax4.quiver(0, 0, 0, 0, axis_len, 0, color='black', arrow_length_ratio=0.05, linewidth=1); ax4.quiver(0, 0, 0, 0, 0, axis_len, color='black', arrow_length_ratio=0.05, linewidth=1)
ax4.text(axis_len, 0, 0, ' t1', color='black', fontsize=14); ax4.text(0, axis_len, 0, ' t2', color='black', fontsize=14); ax4.text(0, 0, axis_len, ' t3', color='black', fontsize=14)
ax4.legend(handles=legend_elements, loc='upper left')
for val in tick_values:
    ax4.text(val, -0.05, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center'); ax4.text(-0.05, val, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center'); ax4.text(-0.05, 0, val, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center')
plt.savefig(os.path.join(save_folder, "4_textbook_grouped_mmd.png"), dpi=300); plt.show()

print(f"\nProcess complete! All 4 MMD plots and both data files have been saved in the folder: '{save_folder}'")