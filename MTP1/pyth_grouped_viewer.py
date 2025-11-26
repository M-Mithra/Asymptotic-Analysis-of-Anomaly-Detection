import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# === 1. LOAD DATA ===
data_path = input("Enter path to your .npz file (e.g., outlier_data.npz): ").strip()
data = np.load(data_path)

# Extract coordinates. Using .get() allows it to work even if some keys are missing
x_red, y_red, z_red = data.get('x_red', []), data.get('y_red', []), data.get('z_red', [])
x_green, y_green, z_green = data.get('x_green', []), data.get('y_green', []), data.get('z_green', [])
x_blue, y_blue, z_blue = data.get('x_blue', []), data.get('y_blue', []), data.get('z_blue', [])
# Try to load 'Ambiguous' (orange) and 'No Outlier' (black) if they exist
x_amb, y_amb, z_amb = data.get('x_orange', []), data.get('y_orange', []), data.get('z_orange', [])
x_black, y_black, z_black = data.get('x_black', []), data.get('y_black', []), data.get('z_black', [])

# === 2. SETUP PLOT ===
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Outlier Regions (GLT)", fontsize=16)

# Organize data for looping
groups = [
    (x_red, y_red, z_red, 'red', 'Outlier 1'),
    (x_green, y_green, z_green, 'green', 'Outlier 2'),
    (x_blue, y_blue, z_blue, 'blue', 'Outlier 3'),
    (x_amb, y_amb, z_amb, 'orange', 'Ambiguous')
]

# === 3. PLOT SOLID SHAPES (Convex Hulls) ===
for x, y, z, color, label in groups:
    if len(x) == 0: continue
    
    # Combine points
    points = np.vstack([x, y, z]).T
    
    # Use DBSCAN to separate distinct "islands" of points (eps may need tuning based on your step size)
    # eps=0.1 is a safe default for a unit cube with decent resolution
    clustering = DBSCAN(eps=0.15, min_samples=4).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)
    unique_labels.discard(-1) # Ignore noise

    # Draw solid shape for each cluster
    for k in unique_labels:
        cluster_points = points[labels == k]
        
        # ConvexHull needs at least 4 points to form a 3D volume
        if len(cluster_points) < 4: 
            # If too few points for a shape, just plot them as dots
            ax.scatter(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2], color=color, s=10, alpha=0.5)
            continue
            
        try:
            hull = ConvexHull(cluster_points)
            triangles = [cluster_points[s] for s in hull.simplices]
            
            # Create the 3D mesh
            mesh = Poly3DCollection(triangles)
            mesh.set_facecolor(color)
            mesh.set_edgecolor(color) 
            mesh.set_alpha(0.4) # Semi-transparent
            ax.add_collection3d(mesh)
        except Exception:
            # Fallback to scatter if hull fails (e.g., flat 2D shapes)
            ax.scatter(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2], color=color, s=10, alpha=0.5)

# Plot the "No Outlier" line if it exists
if len(x_black) > 0:
    # Sort by one axis to ensure the line connects in order
    sort_idx = np.argsort(x_black)
    ax.plot(x_black[sort_idx], y_black[sort_idx], z_black[sort_idx], color='black', linewidth=3, linestyle='--', label='No Outlier')
else:
    # Fallback: Draw diagonal from 0 to 1 if data is missing
    ax.plot([0, 1], [0, 1], [0, 1], color='black', linewidth=3, linestyle='--', label='No Outlier')


# === 4. APPLY TEXTBOOK STYLE (Manual Axes) ===
# Turn off the box
ax.set_axis_off()

# Set view
ax.view_init(elev=25, azim=-60)

# Draw arrows
axis_len = 1.1
ax.quiver(0, 0, 0, axis_len, 0, 0, color='k', arrow_length_ratio=0.05, lw=1.5) # X-axis
ax.quiver(0, 0, 0, 0, axis_len, 0, color='k', arrow_length_ratio=0.05, lw=1.5) # Y-axis
ax.quiver(0, 0, 0, 0, 0, axis_len, color='k', arrow_length_ratio=0.05, lw=1.5) # Z-axis

# Labels
ax.text(axis_len, 0, 0, ' t1', fontsize=14, fontweight='bold')
ax.text(0, axis_len, 0, ' t2', fontsize=14, fontweight='bold')
ax.text(0, 0, axis_len, ' t3', fontsize=14, fontweight='bold')

# Ticks
ticks = np.arange(0.2, 1.1, 0.2)
for val in ticks:
    ax.text(val, -0.05, 0, f'{val:.1f}', fontsize=9, ha='center', color='gray')
    ax.text(-0.05, val, 0, f'{val:.1f}', fontsize=9, ha='center', color='gray')
    ax.text(-0.05, 0, val, f'{val:.1f}', fontsize=9, ha='center', color='gray')

# Limits
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_zlim(-0.1, 1.1)

# === 5. CUSTOM LEGEND ===
legend_elements = [
    Patch(facecolor='red', alpha=0.4, label='Outlier 1 Region'),
    Patch(facecolor='green', alpha=0.4, label='Outlier 2 Region'),
    Patch(facecolor='blue', alpha=0.4, label='Outlier 3 Region'),
    Patch(facecolor='orange', alpha=0.4, label='Ambiguous Region'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='No Outlier')
]
ax.legend(handles=legend_elements, loc='upper left')

plt.show()