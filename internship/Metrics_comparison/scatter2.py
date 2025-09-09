import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ===== SETTINGS =====
step = 0.025  # precision
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

# After your loop, save data
np.savez(os.path.join(save_folder, "outlier_data.npz"),
         x_red=x_red, y_red=y_red, z_red=z_red,
         x_green=x_green, y_green=y_green, z_green=z_green,
         x_blue=x_blue, y_blue=y_blue, z_blue=z_blue)

# ===== 3D SPACIOUS SCATTER PLOT =====
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

# Plot each outlier with a different marker and color
ax.scatter(x_red, y_red, z_red, color='red', marker='x', s=15, alpha=0.7, label='Outlier 1')
ax.scatter(x_green, y_green, z_green, color='green', marker='o', s=15, alpha=0.7, label='Outlier 2')
ax.scatter(x_blue, y_blue, z_blue, color='blue', marker='s', s=15, alpha=0.7, label='Outlier 3')

# Set labels and title
ax.set_xlabel("t1")
ax.set_ylabel("t2")
ax.set_zlabel("t3")
ax.set_title("3D Outlier Classification (Spacious)")
ax.legend()

# === FINAL FIXES FOR A CLEAR PLOT ===

# 1. Set explicit limits for all axes to ensure they start from 0
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

# 2. Set a viewing angle that brings the (0,0,0) origin to the foreground
ax.view_init(elev=20, azim=-60)

# 3. Make the background panes transparent to remove visual clutter
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# Save the figure
plt.savefig(os.path.join(save_folder, "outlier_cube_3D_final.png"), dpi=300)
plt.show()
print(f"3D plot saved in folder: {save_folder}")

# ===== PLOT 2: TEXTBOOK-STYLE 3D AXES WITH VALUES =====

fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')

# 1. Plot your scatter data
ax2.scatter(x_red, y_red, z_red, color='red', marker='x', s=15, label='Outlier 1')
ax2.scatter(x_green, y_green, z_green, color='green', marker='o', s=15, label='Outlier 2')
ax2.scatter(x_blue, y_blue, z_blue, color='blue', marker='s', s=15, label='Outlier 3')
ax2.legend(loc='upper left')

# 2. Set the viewing angle
ax2.view_init(elev=20, azim=-60)

# 3. Turn off the entire default axis structure
ax2.set_axis_off()

# 4. Manually draw the t1, t2, and t3 axes as black arrows
axis_len = 1.1
ax2.quiver(0, 0, 0, axis_len, 0, 0, color='black', arrow_length_ratio=0.05, linewidth=1)
ax2.quiver(0, 0, 0, 0, axis_len, 0, color='black', arrow_length_ratio=0.05, linewidth=1)
ax2.quiver(0, 0, 0, 0, 0, axis_len, color='black', arrow_length_ratio=0.05, linewidth=1)

# 5. Manually add axis labels at the end of the arrows
ax2.text(axis_len, 0, 0, ' t1', color='black', fontsize=14)
ax2.text(0, axis_len, 0, ' t2', color='black', fontsize=14)
ax2.text(0, 0, axis_len, ' t3', color='black', fontsize=14)

# 6. Manually add tick values on each axis
tick_values = np.arange(0.2, 1.1, 0.2)
tick_font_size = 8  # Small font as requested

# t1-axis ticks (offset on y-axis for visibility)
for val in tick_values:
    ax2.text(val, -0.05, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center')

# t2-axis ticks (offset on x-axis for visibility)
for val in tick_values:
    ax2.text(-0.05, val, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center')
    
# t3-axis ticks (offset on x-axis for visibility)
for val in tick_values:
    ax2.text(-0.05, 0, val, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center')

# 7. Set plot limits to ensure all manually drawn elements are visible
ax2.set_xlim([-0.1, axis_len])
ax2.set_ylim([-0.1, axis_len])
ax2.set_zlim([-0.1, axis_len])

# Save and show this new plot
plt.savefig(os.path.join(save_folder, "outlier_cube_textbook_with_ticks.png"), dpi=300)
plt.show()
print(f"Textbook-style plot with axis values saved in folder: {save_folder}")
