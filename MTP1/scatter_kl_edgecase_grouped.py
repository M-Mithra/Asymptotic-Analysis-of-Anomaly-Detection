import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --- REQUIRED IMPORTS ---
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ===== HELPER FUNCTION FOR KL DIVERGENCE =====
def kl_divergence(p, q):
    """ Calculates KL Divergence D_KL(p || q) for discrete distributions. """
    epsilon = 1e-10
    p_safe = p + epsilon
    q_safe = q + epsilon
    return np.sum(p_safe * np.log(p_safe / q_safe))

# ===== SETTINGS =====
step = 0.2
t_values = np.arange(0, 1.0001, step)

# Ask user for folder to save plots
save_folder = input("Enter folder to save plots: ").strip()
os.makedirs(save_folder, exist_ok=True)

# Prepare lists, including a category for "no outlier"
x_red, y_red, z_red = [], [], []
x_green, y_green, z_green = [], [], []
x_blue, y_blue, z_blue = [], [], []
x_black, y_black, z_black = [], [], [] # For t1=t2=t3 case (plotted as yellow)

# ===== LOOP OVER TYPE SPACE =====
print("Running outlier detection...")
# (This loop can take a few minutes with step=0.025)
for t1 in t_values:
    for t2 in t_values:
        for t3 in t_values:
            if t1 == t2 and t2 == t3:
                x_black.append(t1); y_black.append(t2); z_black.append(t3)
                continue
            t = np.array([t1, t2, t3])
            P1, P2, P3 = np.array([t[0], 1-t[0]]), np.array([t[1], 1-t[1]]), np.array([t[2], 1-t[2]])
            Q1, Q2, Q3 = (P2+P3)/2, (P1+P3)/2, (P1+P2)/2
            D = np.array([kl_divergence(P1, Q1), kl_divergence(P2, Q2), kl_divergence(P3, Q3)])
            outlier_idx = np.argmax(D)
            if outlier_idx == 0: x_red.append(t1); y_red.append(t2); z_red.append(t3)
            elif outlier_idx == 1: x_green.append(t1); y_green.append(t2); z_green.append(t3)
            else: x_blue.append(t1); y_blue.append(t2); z_blue.append(t3)
print("Calculation complete.")


# ===== HELPER FUNCTION TO PLOT SOLID CLUSTER SHAPES =====
def plot_solid_cluster_shapes(ax, x, y, z, face_color):
    """
    Finds distinct clusters and plots their convex hull as a solid,
    semi-transparent 3D shape.
    """
    if not x: return
    points = np.vstack([x, y, z]).T
    
    # DBSCAN finds the separate groups of points
    clustering = DBSCAN(eps=step * 3.5, min_samples=20).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)
    unique_labels.discard(-1) # Ignore noise

    # For each cluster, compute its hull and draw it as a solid shape
    for k in unique_labels:
        cluster_points = points[labels == k]
        if len(cluster_points) < 4: continue
        try:
            hull = ConvexHull(cluster_points)
            # Get the vertices for each triangular face of the hull
            triangles = [cluster_points[s] for s in hull.simplices]
            
            # Create a 3D polygon collection for the solid shape
            surface = Poly3DCollection(triangles)
            surface.set_facecolor(face_color)
            surface.set_edgecolor('k') # Black edge for definition
            surface.set_alpha(0.5)     # Transparency
            surface.set_linewidth(0.2)
            ax.add_collection3d(surface)
        except Exception as e:
            print(f"Could not compute hull for a cluster: {e}")


# ===== PLOT: VISUALIZATION WITH SOLID, COLORED SHAPES =====
print("Generating final plot with solid shapes...")
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Set up the plot appearance
ax.set_xlabel("t1"); ax.set_ylabel("t2"); ax.set_zlabel("t3")
ax.set_title("3D Outlier Regions Visualized as Solid Shapes")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_zlim([0, 1])
ax.view_init(elev=25, azim=-60)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.grid(False)

# Plot the solid shapes for each outlier group
plot_solid_cluster_shapes(ax, x_red, y_red, z_red, 'red')
plot_solid_cluster_shapes(ax, x_green, y_green, z_green, 'green')
plot_solid_cluster_shapes(ax, x_blue, y_blue, z_blue, 'blue')

# Plot the "no outlier" case as a yellow diagonal line
ax.plot([0, 1], [0, 1], [0, 1], color='yellow', linewidth=5, label='No Outlier (t1=t2=t3)')

# Create a custom legend to represent the solid shapes
legend_elements = [Patch(facecolor='red', alpha=0.5, edgecolor='k', label='Outlier 1 Region'),
                   Patch(facecolor='green', alpha=0.5, edgecolor='k', label='Outlier 2 Region'),
                   Patch(facecolor='blue', alpha=0.5, edgecolor='k', label='Outlier 3 Region'),
                   Line2D([0], [0], color='yellow', lw=4, label='No Outlier')]
ax.legend(handles=legend_elements, loc='upper left')

plt.savefig(os.path.join(save_folder, "outlier_cube_solid_shapes.png"), dpi=300)
plt.show()
print(f"Plot with solid shapes saved in folder: {save_folder}")

# ===== PLOT 2: TEXTBOOK-STYLE WITH SOLID, COLORED SHAPES =====
print("Generating textbook-style plot with solid shapes...")
fig2 = plt.figure(figsize=(12, 12))
ax2 = fig2.add_subplot(111, projection='3d')

# Set up the plot appearance
ax2.set_title("Textbook-Style Plot with Solid Regions")
ax2.view_init(elev=25, azim=-60)
# Turn off the default grid and panes for the textbook look
ax2.set_axis_off()

# Plot the solid shapes for each outlier group
# This uses the same helper function as the first plot
plot_solid_cluster_shapes(ax2, x_red, y_red, z_red, 'red')
plot_solid_cluster_shapes(ax2, x_green, y_green, z_green, 'green')
plot_solid_cluster_shapes(ax2, x_blue, y_blue, z_blue, 'blue')

# Plot the "no outlier" case as a yellow diagonal line
ax2.plot([0, 1], [0, 1], [0, 1], color='yellow', linewidth=5)

# --- Manually draw the clean axes, labels, and ticks ---
axis_len = 1.1
# Draw axis arrows
ax2.quiver(0, 0, 0, axis_len, 0, 0, color='black', arrow_length_ratio=0.05, linewidth=1)
ax2.quiver(0, 0, 0, 0, axis_len, 0, color='black', arrow_length_ratio=0.05, linewidth=1)
ax2.quiver(0, 0, 0, 0, 0, axis_len, color='black', arrow_length_ratio=0.05, linewidth=1)

# Draw axis labels
ax2.text(axis_len, 0, 0, ' t1', color='black', fontsize=14)
ax2.text(0, axis_len, 0, ' t2', color='black', fontsize=14)
ax2.text(0, 0, axis_len, ' t3', color='black', fontsize=14)

# Draw tick values
tick_values = np.arange(0.2, 1.1, 0.2)
tick_font_size = 8
for val in tick_values:
    ax2.text(val, -0.05, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center')
    ax2.text(-0.05, val, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center')
    ax2.text(-0.05, 0, val, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center')

# Set plot limits to ensure all elements are visible
ax2.set_xlim([-0.1, axis_len])
ax2.set_ylim([-0.1, axis_len])
ax2.set_zlim([-0.1, axis_len])

# Create a custom legend
legend_elements = [Patch(facecolor='red', alpha=0.5, edgecolor='k', label='Outlier 1 Region'),
                   Patch(facecolor='green', alpha=0.5, edgecolor='k', label='Outlier 2 Region'),
                   Patch(facecolor='blue', alpha=0.5, edgecolor='k', label='Outlier 3 Region'),
                   Line2D([0], [0], color='yellow', lw=4, label='No Outlier')]
ax2.legend(handles=legend_elements, loc='upper left')

plt.savefig(os.path.join(save_folder, "outlier_textbook_solid_shapes.png"), dpi=300)
plt.show()
print(f"Textbook plot with solid shapes saved in folder: {save_folder}")