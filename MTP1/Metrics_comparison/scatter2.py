import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ===== SETTINGS =====
step = 0.1  # precision
t_values = np.arange(0, 1.0001, step)

# Ask user for folder to save plots
save_folder = input("Enter folder to save plots: ").strip()
os.makedirs(save_folder, exist_ok=True)

# Prepare lists to store points for each outlier class
x_red, y_red, z_red = [], [], []
x_green, y_green, z_green = [], [], []
x_blue, y_blue, z_blue = [], [], []

# ===== LOOP OVER TYPE SPACE =====
for t1 in t_values:
    for t2 in t_values:
        for t3 in t_values:
            t = np.array([t1, t2, t3])
            
            # Centroids leave-one-out
            c = np.zeros(3)
            c[0] = (t[1] + t[2]) / 2
            c[1] = (t[0] + t[2]) / 2
            c[2] = (t[0] + t[1]) / 2
            
            # Sum-of-distances
            D = np.zeros(3)
            D[0] = (t[1]-c[0])**2 + (t[2]-c[0])**2
            D[1] = (t[0]-c[1])**2 + (t[2]-c[1])**2
            D[2] = (t[0]-c[2])**2 + (t[1]-c[2])**2
            
            # Outlier index
            outlier_idx = np.argmin(D)
            
            # Store points with symbols/colors
            if outlier_idx == 0:
                x_red.append(t1)
                y_red.append(t2)
                z_red.append(t3)
            elif outlier_idx == 1:
                x_green.append(t1)
                y_green.append(t2)
                z_green.append(t3)
            else:
                x_blue.append(t1)
                y_blue.append(t2)
                z_blue.append(t3)

# ===== 3D SPACIOUS SCATTER PLOT =====
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

# Plot each outlier with different marker and color
ax.scatter(x_red, y_red, z_red, color='red', marker='x', s=15, alpha=0.7, label='Outlier 1')
ax.scatter(x_green, y_green, z_green, color='green', marker='o', s=15, alpha=0.7, label='Outlier 2')
ax.scatter(x_blue, y_blue, z_blue, color='blue', marker='s', s=15, alpha=0.7, label='Outlier 3')

ax.set_xlabel("t1")
ax.set_ylabel("t2")
ax.set_zlabel("t3")
ax.set_title("3D Outlier Classification (Spacious)")
ax.legend()
ax.view_init(elev=30, azim=30)  # nice 3D view

# Save figure
plt.savefig(os.path.join(save_folder, "outlier_cube_3D_spacious.png"), dpi=300)
plt.show()
print(f"3D plot saved in folder: {save_folder}")
