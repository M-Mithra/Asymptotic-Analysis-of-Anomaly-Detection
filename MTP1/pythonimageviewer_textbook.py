import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === USER INPUT ===
data_path = input("Enter path to outlier_data.npz: ").strip()

# === LOAD DATA ===
data = np.load(data_path)

x_red, y_red, z_red = data['x_red'], data['y_red'], data['z_red']
x_green, y_green, z_green = data['x_green'], data['y_green'], data['z_green']
x_blue, y_blue, z_blue = data['x_blue'], data['y_blue'], data['z_blue']
# Check if 'No Outlier' data exists (from the robust/edge-case handling)
if 'x_black' in data:
    x_black, y_black, z_black = data['x_black'], data['y_black'], data['z_black']
else:
    x_black, y_black, z_black = [], [], []

# === TEXTBOOK STYLE PLOT ===
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 1. Plot the scatter points
ax.scatter(x_red, y_red, z_red, color='red', marker='x', s=15, alpha=0.7, label='Outlier 1')
ax.scatter(x_green, y_green, z_green, color='green', marker='o', s=15, alpha=0.7, label='Outlier 2')
ax.scatter(x_blue, y_blue, z_blue, color='blue', marker='s', s=15, alpha=0.7, label='Outlier 3')
if len(x_black) > 0:
    ax.scatter(x_black, y_black, z_black, color='black', marker='D', s=20, label='No Outlier')

# 2. Turn off the default boxy axes and grid
ax.set_axis_off()

# 3. Set a nice viewing angle (Origin at front)
ax.view_init(elev=20, azim=-60)

# 4. Manually draw the axes as arrows
axis_len = 1.1
# X-axis (t1)
ax.quiver(0, 0, 0, axis_len, 0, 0, color='black', arrow_length_ratio=0.05, linewidth=1)
# Y-axis (t2)
ax.quiver(0, 0, 0, 0, axis_len, 0, color='black', arrow_length_ratio=0.05, linewidth=1)
# Z-axis (t3)
ax.quiver(0, 0, 0, 0, 0, axis_len, color='black', arrow_length_ratio=0.05, linewidth=1)

# 5. Manually add Axis Labels at the tips
ax.text(axis_len, 0, 0, ' t1', color='black', fontsize=14, fontweight='bold')
ax.text(0, axis_len, 0, ' t2', color='black', fontsize=14, fontweight='bold')
ax.text(0, 0, axis_len, ' t3', color='black', fontsize=14, fontweight='bold')

# 6. Manually add Tick Values (0.2, 0.4, etc.)
tick_values = np.arange(0.2, 1.1, 0.2)
tick_font_size = 9
for val in tick_values:
    # t1 axis ticks (offset slightly on y for visibility)
    ax.text(val, -0.05, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center')
    # t2 axis ticks (offset slightly on x)
    ax.text(-0.05, val, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center')
    # t3 axis ticks (offset slightly on x)
    ax.text(-0.05, 0, val, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center')

# 7. Set plot limits to ensure the arrows aren't cut off
ax.set_xlim([-0.1, axis_len])
ax.set_ylim([-0.1, axis_len])
ax.set_zlim([-0.1, axis_len])

ax.set_title("3D Outlier Classification (GLT)")
ax.legend(loc='upper left')

plt.show()