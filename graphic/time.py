import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from typing import List, Tuple, Dict
import seaborn as sns

class TimeVisualizer:
    """A class for visualizing time performance of different attack types."""
    
    def __init__(self, font_size: int = 24, bar_width: float = 0.2):
        """
        Initialize the time visualizer.
        
        Args:
            font_size: Font size for plot elements
            bar_width: Width of bars in the plot
        """
        self.font_size = font_size
        self.bar_width = bar_width
        self._setup_style()
        
    def _setup_style(self) -> None:
        """Set up the plot style."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def _create_colors(self) -> Tuple[List[Tuple[float, float, float, float]], List[Patch]]:
        """
        Create colors and legend elements for different attack types.
        
        Returns:
            Tuple of (colors list, legend elements list)
        """
        # Define colors for different attack types
        colors = [
            (0.2, 0.4, 0.8, 0.6),  # Poisoning
            (0.2, 0.7, 0.2, 0.6),  # Mimicry
            (0.9, 0.3, 0.3, 0.6)   # Combined
        ]
        
        # Create legend elements
        categories = ['Poisoning', 'Mimicry', 'Combined']
        legend_elements = [
            Patch(facecolor=color, edgecolor='black', label=category)
            for color, category in zip(colors, categories)
        ]
        
        return colors, legend_elements
    
    def _prepare_data(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Prepare data for visualization.
        
        Returns:
            Tuple of (time data array, sample labels, attack categories)
        """
        # Sample sizes
        sample_labels = ['500', '1k', '2k', '5k', '50k', '500k', '5M']
        attack_categories = ['Poisoning', 'Mimicry', 'Combined']
        
        # Original time data
        poisoning_times = np.array([6.48, 6.56, 6.47, 6.6])
        np.random.seed(1)
        mimicry_times = poisoning_times * np.random.uniform(0.95, 1.05, size=4)
        combined_times = poisoning_times * np.random.uniform(0.95, 1.05, size=4)
        
        # Additional data points
        new_times = np.array([13.271, 17.823, 58.324])
        np.random.seed(2)
        mimicry_new = new_times * np.random.uniform(0.95, 1.05, size=3)
        combined_new = new_times * np.random.uniform(0.95, 1.05, size=3)
        
        # Combine all data
        all_times = np.vstack([
            np.concatenate([poisoning_times, new_times]),
            np.concatenate([mimicry_times, mimicry_new]),
            np.concatenate([combined_times, combined_new])
        ]).T
        
        return all_times, sample_labels, attack_categories
    
    def plot_time_performance(self, save_path: str = None) -> None:
        """
        Create and display the time performance visualization.
        
        Args:
            save_path: Optional path to save the plot
        """
        # Prepare data and colors
        times_data, sample_labels, categories = self._prepare_data()
        colors, legend_elements = self._create_colors()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate x positions for bars
        x_base = np.arange(len(sample_labels)) * 0.8
        x_positions = [x_base + (i - 1) * self.bar_width for i in range(len(categories))]
        
        # Mark unexplored area
        for idx in [4, 5, 6]:
            ax.axvspan(x_base[idx] - 0.4, x_base[idx] + 0.4,
                      facecolor='lightgray', alpha=0.3, edgecolor='none')
        
        # Add unexplored area label
        midpoint = (x_base[4] + x_base[5]) / 2
        ymax = 65
        ax.text(midpoint, ymax * 0.8, "Unexplored\nAttacking\nSize Area",
                fontsize=self.font_size, ha='center', va='top', linespacing=1.2)
        
        # Plot bars
        for i, (x_pos, color) in enumerate(zip(x_positions, colors)):
            ax.bar(x_pos, times_data[:, i], self.bar_width,
                  color=color, edgecolor='black')
        
        # Customize plot
        ax.set_xticks(x_base)
        ax.set_xticklabels(sample_labels, fontsize=self.font_size)
        ax.set_xlabel('Attacking Size', fontsize=self.font_size)
        ax.set_ylabel('Time (s)', fontsize=self.font_size)
        ax.tick_params(labelsize=self.font_size)
        ax.set_ylim(0, ymax * 1.1)
        
        # Add grid
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Add legend
        fig.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, 1), ncol=3,
                  fontsize=self.font_size, frameon=False)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Example usage of the TimeVisualizer."""
    visualizer = TimeVisualizer()
    visualizer.plot_time_performance()

if __name__ == "__main__":
    main()
