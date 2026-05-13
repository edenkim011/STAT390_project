import pandas as pd
import matplotlib.pyplot as plt

def create_trajectory_plot():
    try:
        # 1. Load the data (Ensure it is Tab-Separated)
        df = pd.read_csv('results.tsv', sep='\t')
        
        # 2. Clean numeric data
        df['Metric_RMSE'] = pd.to_numeric(df['Metric_RMSE'], errors='coerce')
        
        # 3. Setup the plot
        plt.figure(figsize=(12, 6))
        
        # Plot the main trajectory line
        plt.plot(df['Experiment_ID'], df['Metric_RMSE'], 
                 linestyle='-', color='#bdc3c7', zorder=1, label='Search Path')

        # 4. Color-code points: Green for KEEP, Red for DISCARD
        for i, row in df.iterrows():
            dot_color = '#27ae60' if row['Decision'] == 'KEEP' else '#e74c3c'
            plt.scatter(row['Experiment_ID'], row['Metric_RMSE'], 
                        color=dot_color, s=100, edgecolors='white', zorder=2)
            
            # Label each point with its RMSE value
            plt.text(row['Experiment_ID'], row['Metric_RMSE'] + 0.02, 
                     f"{row['Metric_RMSE']:.3f}", 
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 5. Aesthetics
        plt.title('Metric Trajectory Plot', fontsize=16, pad=20)
        plt.xlabel('Experiment Sequence', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        # Custom Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='KEEP (Success)',
                   markerfacecolor='#27ae60', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='DISCARD (Regression)',
                   markerfacecolor='#e74c3c', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # 6. Save and Show
        plot_filename = 'trajectory_plot.png'
        plt.savefig(plot_filename, dpi=300)
        print(f"Successfully generated: {plot_filename}")
        
    except Exception as e:
        print(f"Error generating plot: {e}")

if __name__ == "__main__":
    create_trajectory_plot()