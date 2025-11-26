import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --- REQUIRED IMPORTS FOR VISUALIZATION ---
# (We no longer need the ConvexHull/DBSCAN imports for this version)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ==============================================================================
# --- PASTE YOUR SCENARIOS DICTIONARY HERE ---
# ==============================================================================
# 31 scenarios for a high-resolution view
SCENARIOS = {
    # --- "Easy" Region (TVD 0.30 - 0.50) ---
    "Easy_TVD_0.50":    {'params': (0.25, 0.75), 'lengths': np.arange(20, 201, 20)},
    "Easy_TVD_0.45":    {'params': (0.275, 0.725), 'lengths': np.arange(20, 201, 20)},
    "Easy_TVD_0.40":    {'params': (0.3, 0.7), 'lengths': np.arange(20, 201, 20)},
    "Easy_TVD_0.35":    {'params': (0.325, 0.675), 'lengths': np.arange(30, 301, 30)},
    "Easy_TVD_0.32":    {'params': (0.34, 0.66), 'lengths': np.arange(35, 351, 35)},
    "Medium_TVD_0.30":  {'params': (0.35, 0.65), 'lengths': np.arange(40, 401, 40)},
    
    # --- "Dip" Region (TVD 0.20 - 0.29) ---
    "Medium_TVD_0.29":  {'params': (0.355, 0.645), 'lengths': np.arange(41, 411, 41)},
    "Medium_TVD_0.28":  {'params': (0.36, 0.64), 'lengths': np.arange(42, 421, 42)},
    "Medium_TVD_0.27":  {'params': (0.365, 0.635), 'lengths': np.arange(43, 431, 43)},
    "Medium_TVD_0.26":  {'params': (0.37, 0.63), 'lengths': np.arange(44, 441, 44)},
    "Medium_TVD_0.25":  {'params': (0.375, 0.625), 'lengths': np.arange(45, 451, 45)},
    "Medium_TVD_0.24":  {'params': (0.38, 0.62), 'lengths': np.arange(46, 461, 46)},
    "Medium_TVD_0.23":  {'params': (0.385, 0.615), 'lengths': np.arange(47, 471, 47)},
    "Medium_TVD_0.22":  {'params': (0.39, 0.61), 'lengths': np.arange(48, 481, 48)},
    "Medium_TVD_0.21":  {'params': (0.395, 0.605), 'lengths': np.arange(49, 491, 49)},
    "Medium_TVD_0.20":  {'params': (0.4, 0.6), 'lengths': np.arange(50, 501, 50)},

    # --- "Hard" Region (TVD 0.10 - 0.19) ---
    "Medium_TVD_0.19":  {'params': (0.405, 0.595), 'lengths': np.arange(60, 601, 60)},
    "Medium_TVD_0.18":  {'params': (0.41, 0.59), 'lengths': np.arange(70, 701, 70)},
    "Medium_TVD_0.17":  {'params': (0.415, 0.585), 'lengths': np.arange(80, 801, 80)},
    "Hard_TVD_0.16":    {'params': (0.42, 0.58), 'lengths': np.arange(90, 901, 90)},
    "Hard_TVD_0.15":    {'params': (0.425, 0.575), 'lengths': np.arange(100, 1001, 100)},
    "Hard_TVD_0.14":    {'params': (0.43, 0.57), 'lengths': np.arange(120, 1201, 120)},
    "Hard_TVD_0.13":    {'params': (0.435, 0.565), 'lengths': np.arange(140, 1401, 140)},
    "Hard_TVD_0.12":    {'params': (0.44, 0.56), 'lengths': np.arange(160, 1601, 160)},
    "Hard_TVD_0.11":    {'params': (0.445, 0.555), 'lengths': np.arange(180, 1801, 180)},
    "Hard_TVD_0.10":    {'params': (0.45, 0.55), 'lengths': np.arange(200, 2001, 200)},

    # --- "Very Hard" Region (TVD < 0.10) ---
    "VHard_TVD_0.08":   {'params': (0.46, 0.54), 'lengths': np.arange(300, 3001, 300)},
    "VHard_TVD_0.06":   {'params': (0.47, 0.53), 'lengths': np.arange(400, 4001, 400)},
    "XHard_TVD_0.04":   {'params': (0.48, 0.52), 'lengths': np.arange(500, 5001, 500)},
    "XHard_TVD_0.03":   {'params': (0.485, 0.515), 'lengths': np.arange(750, 7501, 750)},
    "XHard_TVD_0.02":   {'params': (0.49, 0.51), 'lengths': np.arange(1000, 10001, 1000)},
}

# ===== HELPER FUNCTION FOR MMD (RBF KERNEL) =====
def rbf_kernel(x, y, gamma):
    """ Computes the RBF kernel between two points. """
    return np.exp(-gamma * (x - y)**2)

# ===== MAIN SCRIPT =====
# --- SETTINGS ---
GAMMA = 10.0 # Kernel sensitivity parameter for MMD
save_folder = input("Enter main folder to save SCENARIO plots: ").strip()
os.makedirs(save_folder, exist_ok=True)

# --- DATA GENERATION AND CLASSIFICATION (MODIFIED) ---
all_points = []
print("Calculating decision boundaries for specific scenarios...")

# --- NEW LOOP 1: Iterate over your 31 scenarios ---
for scenario_name, details in SCENARIOS.items():
    p_pi, p_mu = details['params']

    # For each scenario, create the 3 possible ground-truth points
    points_to_test = [
        # ( (t1, t2, t3),  ground_truth_class )
        ( (p_mu, p_pi, p_pi), 'Outlier 1' ),  # Case 1: t1 is the outlier
        ( (p_pi, p_mu, p_pi), 'Outlier 2' ),  # Case 2: t2 is the outlier
        ( (p_pi, p_pi, p_mu), 'Outlier 3' )   # Case 3: t3 is the outlier
    ]

    # --- NEW LOOP 2: Test just these 3 points ---
    for (t1, t2, t3), ground_truth in points_to_test:
        
        # --- This is your original classification logic ---
        point_data = {'t1': t1, 't2': t2, 't3': t3}
        point_data['ground_truth'] = ground_truth # Store the real answer
        point_data['scenario'] = scenario_name    # Store which scenario it came from

        if np.isclose(t1, t2) and np.isclose(t2, t3):
            point_data['outlier_class'] = 'No Outlier'
        else:
            t = np.array([t1, t2, t3])
            k23 = rbf_kernel(t[1], t[2], gamma=GAMMA)
            k13 = rbf_kernel(t[0], t[2], gamma=GAMMA)
            k12 = rbf_kernel(t[0], t[1], gamma=GAMMA)
            D = np.array([k23, k13, k12])
            
            # --- AMBIGUITY DETECTION LOGIC ---
            max_score = np.max(D)
            indices_of_max = np.where(np.isclose(D, max_score))[0]
            if len(indices_of_max) > 1:
                point_data['outlier_class'] = 'Ambiguous'
            else:
                outlier_idx = np.argmax(D)
                point_data['outlier_class'] = f'Outlier {outlier_idx + 1}'
        
        all_points.append(point_data)

print("Calculation complete.")

# --- DATA HANDLING AND SAVING ---
print("Organizing data and saving to files...")
df = pd.DataFrame(all_points)
df.to_excel(os.path.join(save_folder, 'scenario_classification_data.xlsx'), index=False)

# Create a new column to easily check for errors
df['is_error'] = (df['ground_truth'] != df['outlier_class'])

# --- Separate data by its CLASSIFIED result ---
df_red = df[df['outlier_class'] == 'Outlier 1']; x_red, y_red, z_red = df_red['t1'].values, df_red['t2'].values, df_red['t3'].values
df_green = df[df['outlier_class'] == 'Outlier 2']; x_green, y_green, z_green = df_green['t1'].values, df_green['t2'].values, df_green['t3'].values
df_blue = df[df['outlier_class'] == 'Outlier 3']; x_blue, y_blue, z_blue = df_blue['t1'].values, df_blue['t2'].values, df_blue['t3'].values
df_black = df[df['outlier_class'] == 'No Outlier']; x_black, y_black, z_black = df_black['t1'].values, df_black['t2'].values, df_black['t3'].values
df_orange = df[df['outlier_class'] == 'Ambiguous']; x_orange, y_orange, z_orange = df_orange['t1'].values, df_orange['t2'].values, df_orange['t3'].values

# --- Separate data by ERRORS ---
df_errors = df[df['is_error'] == True]
x_err, y_err, z_err = df_errors['t1'].values, df_errors['t2'].values, df_errors['t3'].values


npz_path = os.path.join(save_folder, 'scenario_classification_data.npz')
np.savez(npz_path, x_red=x_red, y_red=y_red, z_red=z_red, x_green=x_green, y_green=y_green, z_green=z_green, x_blue=x_blue, y_blue=y_blue, z_blue=z_blue, x_black=x_black, y_black=y_black, z_black=z_black, x_orange=x_orange, y_orange=y_orange, z_orange=z_orange, x_err=x_err, y_err=y_err, z_err=z_err)
print("Main data files saved.")

# =================================================================================
# vv GENERATING THE 3D PLOTS vv
# =================================================================================
print("\nGenerating the 3D scenario plots...")

# --- PLOT 1: MAIN-STYLE SCATTER PLOT (Shows Test Decisions) ---
fig1 = plt.figure(figsize=(12, 10)); ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_title(f"Test Decisions for Scenarios (GAMMA={GAMMA})"); ax1.set_xlabel("t1"); ax1.set_ylabel("t2"); ax1.set_zlabel("t3")
ax1.scatter(x_red, y_red, z_red, c='red', marker='x', s=50, label='Decision: Outlier 1')
ax1.scatter(x_green, y_green, z_green, c='green', marker='o', s=50, label='Decision: Outlier 2')
ax1.scatter(x_blue, y_blue, z_blue, c='blue', marker='s', s=50, label='Decision: Outlier 3')
ax1.scatter(x_black, y_black, z_black, c='black', marker='D', s=50, label='Decision: No Outlier')
ax1.scatter(x_orange, y_orange, z_orange, c='orange', marker='P', s=50, label='Decision: Ambiguous')
ax1.plot([0, 1], [0, 1], [0, 1], color='k', linewidth=2, linestyle='--', label='t1=t2=t3') # Add diagonal line
ax1.legend(loc='upper left'); ax1.view_init(elev=25, azim=-60)
plt.savefig(os.path.join(save_folder, "1_scenario_decisions.png"), dpi=300); plt.show()


# --- PLOT 2: TEXTBOOK-STYLE SCATTER PLOT (Shows Test Errors) ---
fig3 = plt.figure(figsize=(14, 12)); ax3 = fig3.add_subplot(111, projection='3d')
ax3.set_title(f"TEST ERRORS for Scenarios (GAMMA={GAMMA})"); ax3.view_init(elev=25, azim=-60); ax3.set_axis_off()

# Plot ALL points as small grey dots
ax3.scatter(df['t1'], df['t2'], df['t3'], c='grey', marker='.', s=20, alpha=0.5, label='All Scenarios')
# Plot the ERRORS as big, bright "X" marks
ax3.scatter(x_err, y_err, z_err, c='magenta', marker='X', s=200, lw=3, label='Classification Errors')

ax3.plot([0, 1], [0, 1], [0, 1], color='k', linewidth=2, linestyle='--', label='t1=t2=t3') # Add diagonal line

ax3.legend(loc='upper left')
axis_len = 1.1; ax3.quiver(0,0,0, axis_len,0,0, color='k', arrow_length_ratio=0.05, lw=1); ax3.quiver(0,0,0, 0,axis_len,0, color='k', arrow_length_ratio=0.05, lw=1); ax3.quiver(0,0,0, 0,0,axis_len, color='k', arrow_length_ratio=0.05, lw=1)
ax3.text(axis_len,0,0, ' t1', color='k', fontsize=14); ax3.text(0,axis_len,0, ' t2', color='k', fontsize=14); ax3.text(0,0,axis_len, ' t3', color='k', fontsize=14)
tick_values = np.arange(0.2, 1.1, 0.2); tick_font_size = 8
for val in tick_values:
    ax3.text(val, -0.05, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center'); ax3.text(-0.05, val, 0, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center'); ax3.text(-0.05, 0, val, f'{val:.1f}', color='gray', fontsize=tick_font_size, ha='center')
plt.savefig(os.path.join(save_folder, "2_scenario_errors.png"), dpi=300); plt.show()


print(f"\nProcess complete! All files have been saved in the folder: '{save_folder}'")