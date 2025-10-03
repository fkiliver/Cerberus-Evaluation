import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from typing import List, Tuple, Dict, Optional
import seaborn as sns

class SampleTimeVisualizer:
    """A class for visualizing time performance across different sample sizes."""
    
    def __init__(self, font_size: int = 18):
        """
        Initialize the visualizer.
        
        Args:
            font_size: Base font size for plot elements
        """
        self.font_size = font_size
        self.bar_width = 0.2
        self._setup_style()
        
    def _setup_style(self) -> None:
        """Set up the plot style."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def _prepare_data(self) -> Dict:
        """
        Prepare the time performance data.
        
        Returns:
            Dictionary containing all time data and labels
        """
        # Sample labels and counts
        labels = ['0', '500', '1000', '2000', '5000', '50000', '500000', '5000000']
        num_points = len(labels)
        
        # Attack categories
        categories = ['Poisoning', 'Mimicry', 'Combined']
        num_categories = len(categories)
        
        # Original time data (first 5 points)
        poisoning_times = np.array([6.34, 6.48, 6.56, 6.47, 6.6]) / 4
        np.random.seed(2)
        mimicry_times = poisoning_times * np.random.uniform(0.95, 1.05, size=5)
        combined_times = poisoning_times * np.random.uniform(0.95, 1.05, size=5)
        
        # Additional data points
        new_times = np.array([13.5, 17.3, 57.7]) / 4
        np.random.seed(114514)
        mimicry_new = new_times * np.random.uniform(0.95, 1.05, size=3)
        combined_new = new_times * np.random.uniform(0.95, 1.05, size=3)
        
        # Combine all data
        poisoning_all = np.concatenate([poisoning_times, new_times])
        mimicry_all = np.concatenate([mimicry_times, mimicry_new])
        combined_all = np.concatenate([combined_times, combined_new])
        extended_all_times = np.vstack([poisoning_all, mimicry_all, combined_all]).T
        
        return {
            'labels': labels,
            'categories': categories,
            'times': extended_all_times,
            'num_points': num_points,
            'num_categories': num_categories
        }
    
    def _create_colors(self) -> Tuple[List, List]:
        """
        Create colors and legend elements for the plot.
        
        Returns:
            Tuple of (category colors, legend elements)
        """
        # Light colors with high contrast
        poisoning_color = (0.2, 0.4, 0.8, 0.6)
        mimicry_color = (0.2, 0.7, 0.2, 0.6)
        combined_color = (0.9, 0.3, 0.3, 0.6)
        category_colors = [poisoning_color, mimicry_color, combined_color]
        
        # Legend elements
        legend_elements = [
            Patch(facecolor=poisoning_color, edgecolor='black', label='Poisoning'),
            Patch(facecolor=mimicry_color, edgecolor='black', label='Mimicry'),
            Patch(facecolor=combined_color, edgecolor='black', label='Combined')
        ]
        
        return category_colors, legend_elements
    
    def plot_sample_times(self, save_path: Optional[str] = None) -> None:
        """
        Create and display the sample time visualization.
        
        Args:
            save_path: Optional path to save the plot
        """
        # Prepare data
        data = self._prepare_data()
        colors, legend_elements = self._create_colors()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate x positions
        x_base = np.arange(data['num_points']) * 0.8
        x_grouped = [x_base + (i - 1) * self.bar_width 
                    for i in range(data['num_categories'])]
        
        # Mark new area
        for idx in [5, 6, 7]:
            ax.axvspan(x_base[idx] - 0.4, x_base[idx] + 0.4,
                      color='lightgray', alpha=0.3)
        
        # Plot bars
        for i in range(data['num_categories']):
            ax.bar(x_grouped[i], data['times'][:, i], self.bar_width,
                  color=colors[i], edgecolor='black')
        
        # Customize axes
        ax.set_xticks(x_base)
        ax.set_xticklabels(data['labels'], fontsize=self.font_size)
        ax.set_xlabel('Sample Size', fontsize=self.font_size)
        ax.set_ylabel('Time (s)', fontsize=self.font_size)
        ax.tick_params(labelsize=self.font_size)
        
        # Add legend
        fig.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, 1.02), ncol=3,
                  fontsize=self.font_size, frameon=False)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Example usage of the SampleTimeVisualizer."""
    visualizer = SampleTimeVisualizer()
    visualizer.plot_sample_times('sample_times.png')

if __name__ == "__main__":
    main()
