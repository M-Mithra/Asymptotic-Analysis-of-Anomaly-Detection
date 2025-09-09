import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===== SETTINGS =====
step = 0.05
t_values = np.arange(0, 1.0001, step)  # 0.0 to 1.0 with step 0.05

# Prepare lists to store points for each outlier class
x_red, y_red, z_red = [], [], []
x_green, y_green, z_green = [], [], []
x_blue, y_blue, z_blue = [], [], []

# ===== LOOP OVER TYPE SPACE =====
for t1 in t_values:
    for t2 in t_values:
        for t3 in t_values:
            t = np.array([t1, t2, t3])
            
            # Compute centroids of leave-one-out sequences
            c = np.zeros(3)
            c[0] = (t[1] + t[2]) / 2
            c[1] = (t[0] + t[2]) / 2
            c[2] = (t[0] + t[1]) / 2
            
            # Compute sum-of-distances for each leave-one-out
            D = np.zeros(3)
            D[0] = (t[1]-c[0])**2 + (t[2]-c[0])**2
            D[1] = (t[0]-c[1])**2 + (t[2]-c[1])**2
            D[2] = (t[0]-c[2])**2 + (t[1]-c[2])**2
            
            # Outlier index = argmin
            outlier_idx = np.argmin(D)
            
            # Store in corresponding color group
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
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_red, y_red, z_red, color='red', s=10, alpha=0.6, label='Outlier 1')
ax.scatter(x_green, y_green, z_green, color='green', s=10, alpha=0.6, label='Outlier 2')
ax.scatter(x_blue, y_blue, z_blue, color='blue', s=10, alpha=0.6, label='Outlier 3')

ax.set_xlabel("t1")
ax.set_ylabel("t2")
ax.set_zlabel("t3")
ax.set_title("3D Outlier Classification (LOO Sum-of-Distances)")
ax.legend()
plt.tight_layout()
plt.show()
