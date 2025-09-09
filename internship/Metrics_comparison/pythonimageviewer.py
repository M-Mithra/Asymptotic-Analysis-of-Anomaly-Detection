import numpy as np
import matplotlib.pyplot as plt

# === USER INPUT ===
data_path = input("Enter path to outlier_data.npz: ").strip()

# === LOAD DATA ===
data = np.load(data_path)

x_red, y_red, z_red = data['x_red'], data['y_red'], data['z_red']
x_green, y_green, z_green = data['x_green'], data['y_green'], data['z_green']
x_blue, y_blue, z_blue = data['x_blue'], data['y_blue'], data['z_blue']

# === 3D INTERACTIVE PLOT ===
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_red, y_red, z_red, color='red', marker='x', s=15, alpha=0.7, label='Outlier 1')
ax.scatter(x_green, y_green, z_green, color='green', marker='o', s=15, alpha=0.7, label='Outlier 2')
ax.scatter(x_blue, y_blue, z_blue, color='blue', marker='s', s=15, alpha=0.7, label='Outlier 3')

ax.set_xlabel("t1")
ax.set_ylabel("t2")
ax.set_zlabel("t3")
ax.set_title("3D Outlier Classification (Interactive)")
ax.legend()
plt.show()