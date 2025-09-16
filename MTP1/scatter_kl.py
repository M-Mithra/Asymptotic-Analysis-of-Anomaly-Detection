import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ===== HELPER FUNCTION FOR KL DIVERGENCE =====
def kl_divergence(p, q):
    """ Calculates KL Divergence D_KL(p || q) for discrete distributions. """
    # Add a small epsilon to avoid division by zero or log(0)
    epsilon = 1e-10
    p_safe = p + epsilon
    q_safe = q + epsilon
    return np.sum(p_safe * np.log(p_safe / q_safe))

# ===== SETTINGS =====
step = 0.025  # precision
t_values = np.arange(0, 1.0001, step)

# Ask user for folder to save plots
save_folder = input("Enter folder to save plots (KL Divergence version): ").strip()
os.makedirs(save_folder, exist_ok=True)

# Prepare lists to store points for each outlier class
x_red, y_red, z_red = [], [], []
x_green, y_green, z_green = [], [], []
x_blue, y_blue, z_blue = [], [], []

# ===== LOOP OVER TYPE SPACE =====
print("Running outlier detection with KL Divergence... This may take a moment.")
for t1 in t_values:
    for t2 in t_values:
        for t3 in t_values:
            t = np.array([t1, t2, t3])
            
            # === NEW LOGIC USING KL DIVERGENCE ===
            
            # 1. Represent each scalar as a Bernoulli distribution [p, 1-p]
            P1 = np.array([t[0], 1 - t[0]])
            P2 = np.array([t[1], 1 - t[1]])
            P3 = np.array([t[2], 1 - t[2]])
            
            # 2. Create leave-one-out reference distributions by averaging the other two
            Q1 = (P2 + P3) / 2  # Reference for P1 (is P1 the outlier?)
            Q2 = (P1 + P3) / 2  # Reference for P2 (is P2 the outlier?)
            Q3 = (P1 + P2) / 2  # Reference for P3 (is P3 the outlier?)
            
            # 3. Calculate the KL Divergence for each point from its reference
            D = np.zeros(3)
            D[0] = kl_divergence(P1, Q1)
            D[1] = kl_divergence(P2, Q2)
            D[2] = kl_divergence(P3, Q3)
            
            # 4. Outlier is the one with the MAXIMUM divergence
            outlier_idx = np.argmax(D)
            
            # =======================================
            
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

print("Calculation complete. Generating plots...")
# After your loop, save data
np.savez(os.path.join(save_folder, "outlier_data_kl.npz"),
         x_red=x_red, y_red=y_red, z_red=z_red,
         x_green=x_green, y_green=y_green, z_green=z_green,
         x_blue=x_blue, y_blue=y_blue, z_blue=z_blue)

# ===== PLOT 1: 3D SPACIOUS SCATTER PLOT =====
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
ax.set_title("3D Outlier Classification (KL Divergence)")
ax.legend()
ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_zlim([0, 1])
ax.view_init(elev=20, azim=-60)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

plt.savefig(os.path.join(save_folder, "outlier_cube_kl_final.png"), dpi=300)
plt.show()
print(f"3D plot saved in folder: {save_folder}")


# ===== PLOT 2: TEXTBOOK-STYLE 3D AXES WITH VALUES =====
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(x_red, y_red, z_red, color='red', marker='x', s=15, label='Outlier 1')
ax2.scatter(x_green, y_green, z_green, color='green', marker='o', s=15, label='Outlier 2')
ax2.scatter(x_blue, y_blue, z_blue, color='blue', marker='s', s=15, label='Outlier 3')
ax2.legend(loc='upper left')
ax2.view_init(elev=20, azim=-60)
ax2.set_axis_off()
axis_len = 1.1
ax2.quiver(0, 0, 0, axis_len, 0, 0, color='black', arrow_length_ratio=0.05, linewidth=1)
ax2.quiver(0, 0, 0, 0, axis_len, 0, color='black', arrow_length_ratio=0.05, linewidth=1)
ax2.quiver(0, 0, 0, 0, 0, axis_len, color='black', arrow_length_ratio=0.05, linewidth=1)
ax2.text(axis_len, 0, 0, ' t1', color='black', fontsize=14)
ax2.text(0, axis_len, 0, ' t2', color='black', fontsize=14)
ax2.text(0, 0, axis_len, ' t3', color='black', fontsize=14)
tick_values = np.arange(0.2, 1.1, 0.2)
tick_font_size = 8
for val in tick_values:
    ax2.text(val, -0.05, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center')
    ax2.text(-0.05, val, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center')
    ax2.text(-0.05, 0, val, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center')
ax2.set_xlim([-0.1, axis_len]); ax2.set_ylim([-0.1, axis_len]); ax2.set_zlim([-0.1, axis_len])

plt.savefig(os.path.join(save_folder, "outlier_cube_kl_textbook.png"), dpi=300)
plt.show()
print(f"Textbook-style KL plot saved in folder: {save_folder}")