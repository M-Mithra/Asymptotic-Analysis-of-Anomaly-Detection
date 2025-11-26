import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
import itertools # For iterating over combinations of neighbors

# --- HELPER FUNCTIONS ---

def get_neighbors(x, y, grid_step, df_slice, x_label, y_label):
    """
    Returns the classes of immediate neighbors of (x, y) in the grid.
    This assumes a regular grid.
    """
    neighbors_coords = []
    # Direct neighbors
    neighbors_coords.append((x + grid_step, y))
    neighbors_coords.append((x - grid_step, y))
    neighbors_coords.append((x, y + grid_step))
    neighbors_coords.append((x, y - grid_step))
    # Diagonal neighbors (optional, but makes boundaries smoother)
    neighbors_coords.append((x + grid_step, y + grid_step))
    neighbors_coords.append((x + grid_step, y - grid_step))
    neighbors_coords.append((x - grid_step, y + grid_step))
    neighbors_coords.append((x - grid_step, y - grid_step))

    neighbor_classes = []
    for nx, ny in neighbors_coords:
        # Find if a neighbor exists within a small tolerance
        neighbor_row = df_slice[np.isclose(df_slice[x_label], nx) & np.isclose(df_slice[y_label], ny)]
        if not neighbor_row.empty:
            neighbor_classes.append(neighbor_row['outlier_class'].iloc[0])
    return neighbor_classes

def find_boundary_points(df_slice, x_col, y_col, grid_step, x_label, y_label):
    """
    Identifies points that lie on a decision boundary.
    A point is a boundary point if it has neighbors of a different class.
    """
    boundary_points = []
    
    # Get all unique points and their classes in the slice
    unique_points = df_slice[[x_col, y_col, 'outlier_class']].drop_duplicates()
    
    for _, row in unique_points.iterrows():
        px, py, p_class = row[x_col], row[y_col], row['outlier_class']
        
        neighbor_classes = get_neighbors(px, py, grid_step, df_slice, x_label, y_label)
        
        # Check if any neighbor has a different class
        if any(n_class != p_class for n_class in neighbor_classes):
            boundary_points.append(row.to_dict())
            
    return pd.DataFrame(boundary_points)


# ===== MAIN SCRIPT =====
if __name__ == "__main__":
    print("--- Start of Experiment 2: Decision Boundary Comparison (MMD, GLT, KL) ---")
    
    # --- SETTINGS ---
    # The 'step' parameter from your data generation script (e.g., 0.01 for 101x101x101 grid)
    GRID_STEP = 0.025 
    
    # Define which slices you want to visualize for comparison (t1, t2, t3 values)
    # These will be 'X=val', 'Y=val', 'Z=val'
    # Choose values that you found interesting, e.g., where boundaries change a lot or ambiguity is clear.
    comparison_slices = [
        {'direction': 'X', 'value': 0.3},
        {'direction': 'Y', 'value': 0.5},
        {'direction': 'Z', 'value': 0.7},
        # Add more slices if needed
    ]

    base_output_dir = "results_analysis_reports"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 1. Ask for the folder path where the .xlsx data files are
    data_folder = input("\nPlease enter the FULL path to the folder containing 'outlier_classification_data_mmd.xlsx', '..._glt.xlsx', and '..._kl.xlsx': ").strip()
    while not os.path.exists(data_folder):
        print("Folder not found. Please check the path and try again.")
        data_folder = input("Please enter the FULL path to the folder containing classification data: ").strip()

    # File paths for the generated data
    mmd_data_path = os.path.join(data_folder, 'outlier_classification_data_mmd.xlsx')
    glt_data_path = os.path.join(data_folder, 'outlier_classification_data_glt.xlsx')
    kl_data_path = os.path.join(data_folder, 'outlier_classification_data_kl.xlsx')


    # Check if all files exist
    required_files = {
        'MMD': mmd_data_path,
        'GLT': glt_data_path,
        'KL': kl_data_path
    }
    missing_files = [metric for metric, path in required_files.items() if not os.path.exists(path)]

    if missing_files:
        print(f"\nERROR: The following required data files are missing in '{data_folder}':")
        for metric in missing_files:
            print(f"  - '{os.path.basename(required_files[metric])}' for {metric}")
        print("Please ensure your previous scripts were run and generated these files correctly before running this comparison script.")
        exit()

    # Load the full classification dataframes
    print(f"\nLoading MMD data from: {mmd_data_path}")
    df_mmd_full = pd.read_excel(mmd_data_path)
    print(f"Loading GLT data from: {glt_data_path}")
    df_glt_full = pd.read_excel(glt_data_path)
    print(f"Loading KL data from: {kl_data_path}")
    df_kl_full = pd.read_excel(kl_data_path)


    # 2. Ask for the output folder name
    output_subfolder_name = input(f"What do you want to name the output folder for these boundary comparison plots? ")
    output_folder_full_path = os.path.join(base_output_dir, output_subfolder_name)
    os.makedirs(output_folder_full_path, exist_ok=True)
    print(f"Comparison plots will be saved to: {output_folder_full_path}")

    # --- DEFINE PLOTTING PARAMETERS ---
    color_map = {
        'Outlier 1': 'red', 
        'Outlier 2': 'green', 
        'Outlier 3': 'blue', 
        'Ambiguous': 'orange', 
        'No Outlier': 'black'
    }
    class_names = list(color_map.keys())
    
    # Map for axes labels
    axis_map = {
        'X': {'col': 't1', 'labels': ('t2', 't3')}, 
        'Y': {'col': 't2', 'labels': ('t1', 't3')}, 
        'Z': {'col': 't3', 'labels': ('t1', 't2')}
    }

    # Unified legend handles
    legend_handles = [
        Patch(facecolor='red', edgecolor='k', alpha=0.5, label='Outlier 1'),
        Patch(facecolor='green', edgecolor='k', alpha=0.5, label='Outlier 2'),
        Patch(facecolor='blue', edgecolor='k', alpha=0.5, label='Outlier 3'),
        Patch(facecolor='orange', edgecolor='k', alpha=0.7, label='Ambiguous Region'), # Highlight ambiguous
        Patch(facecolor='black', edgecolor='k', alpha=0.5, label='No Outlier')
    ]

    # --- GENERATE COMPARISON PLOTS ---
    print("\nGenerating 2D boundary comparison plots (MMD vs. GLT vs. KL)...")
    for slice_info in comparison_slices:
        direction_char = slice_info['direction']
        slice_val = slice_info['value']
        slice_val_str = f"{slice_val:.1f}"

        col_to_slice, (x_label, y_label) = axis_map[direction_char]['col'], axis_map[direction_char]['labels']
        
        print(f"  - Processing slice: {direction_char}={slice_val_str}")

        # --- Prepare data for all three metrics ---
        datasets = {
            'MMD': df_mmd_full,
            'GLT': df_glt_full,
            'KL': df_kl_full
        }
        
        boundary_dfs = {}
        for metric_name, df_full in datasets.items():
            slice_df = df_full[np.isclose(df_full[col_to_slice], slice_val, atol=GRID_STEP / 2.0)]
            boundary_dfs[metric_name] = find_boundary_points(slice_df, x_label, y_label, GRID_STEP, x_label, y_label)
            datasets[metric_name] = slice_df # Store sliced df for background scatter

        # --- Plotting Side-by-Side-by-Side ---
        fig, axes = plt.subplots(1, 3, figsize=(24, 8)) # Three subplots side-by-side
        
        for i, (metric_name, ax) in enumerate(zip(['MMD', 'GLT', 'KL'], axes)):
            ax.set_title(f'{metric_name} Boundaries at {direction_char}={slice_val_str}')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.grid(True, linestyle=':', alpha=0.6)

            # Plot actual classification regions (background)
            current_slice_df = datasets[metric_name]
            for name in class_names:
                subset = current_slice_df[current_slice_df['outlier_class'] == name]
                if not subset.empty:
                    ax.scatter(subset[x_label], subset[y_label], color=color_map[name], s=5, alpha=0.1, label=name) # Very light background scatter

            # Plot boundary points (more prominently)
            current_boundary_df = boundary_dfs[metric_name]
            if not current_boundary_df.empty:
                for name in class_names:
                    subset_boundary = current_boundary_df[current_boundary_df['outlier_class'] == name]
                    if not subset_boundary.empty:
                        marker_style = 'o'
                        edge_color = color_map[name]
                        face_color = 'none' # Transparent face for general boundaries
                        if name == 'Ambiguous':
                            face_color = color_map[name] # Solid fill for ambiguous
                            marker_style = 'X'
                            ax.scatter(subset_boundary[x_label], subset_boundary[y_label], 
                                    facecolors=face_color, edgecolors='k', marker=marker_style, s=50, alpha=0.7, zorder=3, label=f'Boundary ({name})')
                        else:
                            ax.scatter(subset_boundary[x_label], subset_boundary[y_label], 
                                    color=edge_color, marker=marker_style, s=20, alpha=0.6, zorder=2, label=f'Boundary ({name})')
            
            # Only add legend to the first plot to avoid redundancy, or adjust its position
            if i == 0: # Only on the first subplot
                ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.0, 1.02), ncol=3, borderaxespad=0.)
            else:
                ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.0, 1.02), ncol=3, borderaxespad=0.) # Added to all for clarity per plot

        plt.tight_layout()
        plot_filename = os.path.join(output_folder_full_path, f'boundary_comparison_MMD_GLT_KL_{direction_char}{slice_val_str}.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close(fig)
        print(f"    -> Saved: {plot_filename}")

    print("\nDecision boundary comparison plots (MMD, GLT, KL) generated successfully!")
    print(f"All outputs are in '{output_folder_full_path}'")
    print("--- Script finished. ---")