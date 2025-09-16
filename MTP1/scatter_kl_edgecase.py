import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# SETTINGS
step = 0.1
t_values = np.arange(0, 1.0001, step)

save_folder = input("Enter folder to save plots (Edge Case Fixed): ").strip()
os.makedirs(save_folder, exist_ok=True)

# Prepare lists, now including a category for "no outlier"
x_red, y_red, z_red = [], [], []
x_green, y_green, z_green = [], [], []
x_blue, y_blue, z_blue = [], [], []
x_black, y_black, z_black = [], [], [] # For t1=t2=t3 case

# HELPER FUNCTION FOR KL DIVERGENCE
def kl_divergence(p, q):
    epsilon = 1e-10
    p_safe = p + epsilon
    q_safe = q + epsilon
    return np.sum(p_safe * np.log(p_safe / q_safe))

# LOOP OVER TYPE SPACE
print("Running outlier detection (edge case handled)...")
for t1 in t_values:
    for t2 in t_values:
        for t3 in t_values:
            
            # === NEW: CATCH THE EDGE CASE WHERE ALL VALUES ARE EQUAL ===
            if t1 == t2 and t2 == t3:
                x_black.append(t1)
                y_black.append(t2)
                z_black.append(t3)
                continue # Skip the rest of the calculation for this point
            # ==========================================================

            t = np.array([t1, t2, t3])
            
            # KL DIVERGENCE LOGIC
            P1 = np.array([t[0], 1 - t[0]])
            P2 = np.array([t[1], 1 - t[1]])
            P3 = np.array([t[2], 1 - t[2]])
            
            Q1 = (P2 + P3) / 2
            Q2 = (P1 + P3) / 2
            Q3 = (P1 + P2) / 2
            
            D = np.zeros(3)
            D[0] = kl_divergence(P1, Q1)
            D[1] = kl_divergence(P2, Q2)
            D[2] = kl_divergence(P3, Q3)
            
            outlier_idx = np.argmax(D)
            
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

print("Calculation complete. Generating plots...")

# After your loop, save all categories of data
np.savez(os.path.join(save_folder, "outlier_data_fixed.npz"),
         x_red=x_red, y_red=y_red, z_red=z_red,
         x_green=x_green, y_green=y_green, z_green=z_green,
         x_blue=x_blue, y_blue=y_blue, z_blue=z_blue,
         x_black=x_black, y_black=y_black, z_black=z_black) # Added the 'no outlier' data

# ===== PLOT 1: 3D SPACIOUS SCATTER PLOT =====
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_red, y_red, z_red, color='red', marker='x', s=15, alpha=0.7, label='Outlier 1')
ax.scatter(x_green, y_green, z_green, color='green', marker='o', s=15, alpha=0.7, label='Outlier 2')
ax.scatter(x_blue, y_blue, z_blue, color='blue', marker='s', s=15, alpha=0.7, label='Outlier 3')
# Add the "no outlier" points to the plot
ax.scatter(x_black, y_black, z_black, color='black', marker='D', s=20, label='No Outlier')

ax.set_xlabel("t1"); ax.set_ylabel("t2"); ax.set_zlabel("t3")
ax.set_title("3D Outlier Classification (KL Divergence, Edge Case Handled)")
ax.legend()
ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_zlim([0, 1])
ax.view_init(elev=20, azim=-60)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

plt.savefig(os.path.join(save_folder, "outlier_cube_kl_fixed.png"), dpi=300)
plt.show()

# (The textbook-style plot can be updated in the same way, by adding the ax2.scatter call for the black points)
# ===== PLOT 2: TEXTBOOK-STYLE 3D AXES WITH VALUES (FIXED) =====

fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')

# 1. Plot your scatter data
ax2.scatter(x_red, y_red, z_red, color='red', marker='x', s=15, label='Outlier 1')
ax2.scatter(x_green, y_green, z_green, color='green', marker='o', s=15, label='Outlier 2')
ax2.scatter(x_blue, y_blue, z_blue, color='blue', marker='s', s=15, label='Outlier 3')
# === THIS LINE IS ADDED FOR THE EDGE CASE ===
ax2.scatter(x_black, y_black, z_black, color='black', marker='D', s=20, label='No Outlier')
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
plt.savefig(os.path.join(save_folder, "outlier_cube_textbook_fixed.png"), dpi=300)
plt.show()
print(f"Textbook-style plot with fix saved in folder: {save_folder}")