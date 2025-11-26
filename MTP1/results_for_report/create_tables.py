import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import dataframe_image as dfi
from scipy.interpolate import make_interp_spline
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Helper Function to save DataFrame as image (using dataframe_image) ---
def save_df_as_image(df, filepath, title=""):
    """Saves a pandas DataFrame as an image using dataframe_image or matplotlib fallback."""
    print(f"   -> Attempting to save table image to: {filepath}")
    try:
        styled_df = df.style.set_caption(f"<b>{title}</b>") \
                            .set_table_styles([{'selector': 'caption', 'props': [('font-size', '16px'), ('font-weight', 'bold')]}]) \
                            .format(precision=6)
        
        dfi.export(styled_df, filepath, table_conversion='matplotlib', max_rows=-1, max_cols=-1)
        print(f"   -> Successfully saved table image.")
    except Exception as e:
        print(f"   -> WARNING: Could not save table image using dataframe_image. Error: {e}")
        print("      Falling back to basic matplotlib table plot.")
        
        fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.5))) # Auto-size height
        ax.axis('off')
        ax.axis('tight')
        
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        ax.set_title(title, fontsize=16, weight='bold', loc='center', pad=20)
        
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"   -> Saved basic table image (matplotlib fallback).")

# --- Analysis and Plotting Functions ---

def generate_summary_tables(df_summary_data, output_folder):
    """Generates and saves performance and efficiency summary tables from the provided DataFrame."""
    print("\nGenerating summary tables...")
    
    # Define columns to check
    perf_cols = ['Scenario', 'TVD_Dist', 'c_KL', 'c_MMD', 'c_GLT', '2*DB_Bound']
    ratio_cols = ['Scenario', 'TVD_Dist', 'Ratio_KL', 'Ratio_MMD', 'Ratio_GLT']
    
    # Check if all required columns exist
    if not all(col in df_summary_data.columns for col in perf_cols):
        print(f"   -> ERROR: Missing one or more columns for Performance Summary Table.")
        print(f"      Required: {perf_cols}")
        print(f"      Found: {list(df_summary_data.columns)}")
    else:
        # Table 1: Performance Summary (c values)
        performance_summary = df_summary_data[perf_cols].copy()
        performance_summary = performance_summary.sort_values(by='TVD_Dist', ascending=False).reset_index(drop=True)
        
        csv_path_c = os.path.join(output_folder, "performance_summary_c_values.csv")
        img_path_c = os.path.join(output_folder, "performance_summary_c_values.png")
        performance_summary.to_csv(csv_path_c, index=False)
        save_df_as_image(performance_summary.round(6), img_path_c, "Performance Summary (Error Exponents 'c')")
        print(f"   -> Saved Performance Summary (c values) to: {csv_path_c}")

    if not all(col in df_summary_data.columns for col in ratio_cols):
        print(f"   -> ERROR: Missing one or more columns for Efficiency Ratio Table.")
        print(f"      Required: {ratio_cols}")
    else:
        # Table 2: Efficiency Ratio Summary
        efficiency_summary = df_summary_data[ratio_cols].copy()
        efficiency_summary = efficiency_summary.sort_values(by='TVD_Dist', ascending=False).reset_index(drop=True)

        csv_path_ratio = os.path.join(output_folder, "efficiency_summary_ratios.csv")
        img_path_ratio = os.path.join(output_folder, "efficiency_summary_ratios.png")
        efficiency_summary.to_csv(csv_path_ratio, index=False)
        save_df_as_image(efficiency_summary.round(4), img_path_ratio, "Efficiency Ratio Summary (c / 2*DB Bound)")
        print(f"   -> Saved Efficiency Ratio Summary to: {csv_path_ratio}")

def plot_smooth_curve(ax, x_data, y_data, **kwargs):
    """Helper function to plot data points and a smooth spline."""
    # Plot the original data points
    ax.plot(x_data, y_data, marker=kwargs.get('marker', 'o'), linestyle='None', 
            color=kwargs.get('color'), alpha=0.5, label=f"{kwargs.get('label')} (Data)")
    
    # Create smooth curve
    if len(x_data) > 3:
        try:
            # Sort data for spline (spline requires monotonic x)
            sorted_indices = np.argsort(x_data)
            x_sorted = x_data[sorted_indices]
            y_sorted = y_data[sorted_indices]
            
            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
            spl = make_interp_spline(x_sorted, y_sorted)
            y_smooth = spl(x_smooth)
            ax.plot(x_smooth, y_smooth, color=kwargs.get('color'), label=f"{kwargs.get('label')} (Trend)")
        except Exception as e:
            print(f"   -> Warning: Could not create smooth curve for {kwargs.get('label')}. Plotting linear. Error: {e}")
            ax.plot(x_data, y_data, linestyle='--', color=kwargs.get('color')) # Fallback
    else:
        # Not enough data for a spline, just connect the dots
        ax.plot(x_data, y_data, linestyle='--', color=kwargs.get('color'))

def generate_plots(df_summary_data, output_folder):
    """Generates and saves key plots for the report from the provided DataFrame."""
    print("\nGenerating plots...")

    # Plot 1: Error Exponent vs. TVD (Combined)
    # NEW: Sort by TVD_Dist in descending order
    df_plot_tvd = df_summary_data.sort_values(by='TVD_Dist', ascending=False).reset_index(drop=True)
    try:
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        plot_smooth_curve(ax1, df_plot_tvd['TVD_Dist'], df_plot_tvd['c_KL'], marker='o', color='blue', label='Max Avg KL')
        plot_smooth_curve(ax1, df_plot_tvd['TVD_Dist'], df_plot_tvd['c_MMD'], marker='x', color='green', label='Max Avg MMD')
        plot_smooth_curve(ax1, df_plot_tvd['TVD_Dist'], df_plot_tvd['c_GLT'], marker='s', color='red', label='Universal GLT')
        
        # Plot theoretical bound (also smooth)
        plot_smooth_curve(ax1, df_plot_tvd['TVD_Dist'], df_plot_tvd['2*DB_Bound'], marker='.', color='gray', label='Theoretical Bound (2*DB)')

        plt.title('Error Exponent (c) vs. Total Variation Distance (TVD)')
        plt.xlabel('Total Variation Distance (TVD) -> Larger is Easier')
        plt.ylabel('Error Exponent (c)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_folder, "exponent_vs_tvd_smooth.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"   -> Saved plot: {plot_path}")
    except KeyError as e:
        print(f"   -> ERROR: Could not generate 'Exponent vs. TVD' plot. Missing column: {e}")
    except Exception as e:
        print(f"   -> ERROR: Could not generate 'Exponent vs. TVD' plot: {e}")

    # Plot 2: Efficiency Ratio vs. Problem Difficulty (2*DB_Bound, inverted axis)
    # Sort by 2*DB_Bound for this plot (descending for inverted log axis)
    df_plot_db = df_summary_data.sort_values(by='2*DB_Bound', ascending=False).reset_index(drop=True)
    try:
        plt.figure(figsize=(10, 6))
        ax2 = plt.gca()
        plot_smooth_curve(ax2, df_plot_db['2*DB_Bound'], df_plot_db['Ratio_KL'], marker='o', color='blue', label='Max Avg KL')
        plot_smooth_curve(ax2, df_plot_db['2*DB_Bound'], df_plot_db['Ratio_MMD'], marker='x', color='green', label='Max Avg MMD')
        plot_smooth_curve(ax2, df_plot_db['2*DB_Bound'], df_plot_db['Ratio_GLT'], marker='s', color='red', label='Universal GLT')
        
        plt.axhline(y=1.0, color='r', linestyle=':', label='Theoretical Max (1.0)')
        plt.title('Efficiency Ratio (c / 2*DB) vs. Problem Difficulty')
        plt.xlabel('Problem Difficulty (2 * Bhattacharyya Distance) -> Smaller is Harder')
        plt.ylabel('Efficiency Ratio')
        
        if (df_plot_db['2*DB_Bound'] > 0).all():
            plt.xscale('log')
            plt.gca().invert_xaxis()
        else:
            print("   -> NOTE: 2*DB_Bound contains non-positive values. Using linear scale for x-axis.")
            if df_plot_db['2*DB_Bound'].min() < df_plot_db['2*DB_Bound'].max():
                plt.gca().invert_xaxis()

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_folder, "efficiency_vs_difficulty_smooth.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"   -> Saved plot: {plot_path}")
    except KeyError as e:
        print(f"   -> ERROR: Could not generate 'Efficiency Ratio' plot. Missing column: {e}")
    except Exception as e:
        print(f"   -> ERROR: Could not generate 'Efficiency Ratio' plot: {e}")

    # Plot 3: Maximin Analysis Bar Chart
    try:
        min_efficiency = {
            'Max Avg KL': df_summary_data['Ratio_KL'].min(),
            'Max Avg MMD': df_summary_data['Ratio_MMD'].min(),
            'Universal GLT': df_summary_data['Ratio_GLT'].min()
        }
        metrics = list(min_efficiency.keys())
        values = list(min_efficiency.values())

        plt.figure(figsize=(8, 6))
        bars = plt.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('Maximin Analysis: Most Robust Metric (Worst-Case Efficiency)')
        plt.xlabel('Metric')
        plt.ylabel('Minimum Efficiency (c / 2*DB)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if all(np.isfinite(values)):
             plt.ylim(0, max(0.05, max(values) * 1.2 + 0.05))
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plot_path = os.path.join(output_folder, "maximin_analysis.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"   -> Saved plot: {plot_path}")
    except KeyError as e:
        print(f"   -> ERROR: Could not generate 'Maximin' plot. Missing column: {e}")
    except Exception as e:
        print(f"   -> ERROR: Could not generate 'Maximin' plot: {e}")


# --- Main Function ---

def main():
    """Main function to control the analysis workflow."""
    print("--- Start of Experiment 1 Results Analysis Script ---")
    base_output_dir = "results_analysis_reports"
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"All analysis reports will be saved in the '{base_output_dir}' directory.")

    # 1. Ask for the file path
    excel_file_path = input(f"\nPlease enter the FULL path to your Excel (.xlsx) file: ")
    while not os.path.exists(excel_file_path):
        print("File not found. Please check the path and try again.")
        excel_file_path = input(f"Please enter the FULL path to your Excel (.xlsx) file: ")

    # 2. Hardcode the sheet name
    summary_sheet_name = 'Final_Summary'

    # 3. Ask for the output folder name
    output_subfolder_name = input(f"What do you want to name the output folder for these results? ")
    output_folder_full_path = os.path.join(base_output_dir, output_subfolder_name)
    os.makedirs(output_folder_full_path, exist_ok=True)
    print(f"Output for this analysis will be saved to: {output_folder_full_path}")

    # 4. Run the analysis
    try:
        # Load the specified summary sheet directly
        df_summary_data = pd.read_excel(excel_file_path, sheet_name=summary_sheet_name)
        print(f"\nSuccessfully loaded sheet: '{summary_sheet_name}' from '{os.path.basename(excel_file_path)}'")
        print(f"Found {len(df_summary_data)} scenarios in the summary data.")
        
        # Check if necessary columns exist before proceeding
        required_cols = ['Scenario', 'TVD_Dist', 'c_KL', 'c_MMD', 'c_GLT', '2*DB_Bound', 'Ratio_KL', 'Ratio_MMD', 'Ratio_GLT']
        missing_cols = [col for col in required_cols if col not in df_summary_data.columns]
        
        if missing_cols:
            print(f"\n--- ERROR: The sheet '{summary_sheet_name}' is missing required columns: {missing_cols} ---")
            print("Cannot proceed with analysis.")
        else:
            # Generate tables
            generate_summary_tables(df_summary_data, output_folder_full_path)
            
            # Generate plots
            generate_plots(df_summary_data, output_folder_full_path)
            
            print(f"\nAnalysis complete. All outputs are in '{output_folder_full_path}'")

    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        print(f"An error occurred during analysis: {e}")
        print(f"Please ensure the file path is correct, the sheet name '{summary_sheet_name}' exists, and 'openpyxl' is installed.")

    print("\n--- Script finished. ---")

if __name__ == "__main__":
    main()