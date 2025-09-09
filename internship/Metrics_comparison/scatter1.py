import numpy as np
import matplotlib.pyplot as plt
import os

# ===== SETTINGS =====
step = 0.1
t_values = np.arange(0, 1.0001, step)  # 0.0 to 1.0 with step 0.05

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

# ===== 3D SCATTER PLOT =====
fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x_red, y_red, z_red, color='red', s=20, marker='x', alpha=0.6, label='Outlier 1')
ax1.scatter(x_green, y_green, z_green, color='green', s=20, marker='o', alpha=0.6, label='Outlier 2')
ax1.scatter(x_blue, y_blue, z_blue, color='blue', s=20, marker='s', alpha=0.6, label='Outlier 3')
ax1.set_xlabel("t1")
ax1.set_ylabel("t2")
ax1.set_zlabel("t3")
ax1.set_title("3D Outlier Classification")
ax1.legend()

# ===== 2D CUBE PLOT =====
ax2 = fig.add_subplot(122)
# Draw cube edges
cube_edges = [
    ((0,0), (1,0)), ((0,0), (0,1)), ((1,0), (1,1)), ((0,1),(1,1)),  # bottom square
    ((0,0), (0,0)), ((1,0), (1,0)), ((0,1),(0,1)), ((1,1),(1,1))   # placeholders
]
# Simulate 3D perspective by projecting onto 2D (x,y) plane
ax2.scatter(x_red, y_red, color='red', s=20, marker='x', alpha=0.6, label='Outlier 1')
ax2.scatter(x_green, y_green, color='green', s=20, marker='o', alpha=0.6, label='Outlier 2')
ax2.scatter(x_blue, y_blue, color='blue', s=20, marker='s', alpha=0.6, label='Outlier 3')
ax2.set_xlabel("t1")
ax2.set_ylabel("t2")
ax2.set_title("2D Cube Projection: t1 vs t2")
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
ax2.legend()

plt.tight_layout()

# ===== SAVE FIGURES =====
plt.savefig(os.path.join(save_folder, "outlier_cube_3D_2D.png"), dpi=300)
plt.show()
print(f"Plots saved in folder: {save_folder}")
