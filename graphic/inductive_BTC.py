# 导入matplotlib相关库
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from typing import List, Dict, Tuple, Optional
import seaborn as sns

class InductiveBTCVisualizer:
    """A class for creating multi-dimensional robustness assessment visualizations."""
    
    def __init__(self,
                 figsize: Tuple[int, int] = (20, 14),
                 title_fontsize: int = 24,
                 label_fontsize: int = 20,
                 tick_fontsize: int = 16,
                 legend_fontsize: int = 16,
                 line_width: float = 3.0,
                 line_alpha: float = 0.8,
                 bar_alpha: float = 0.1):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size (width, height)
            title_fontsize: Font size for the main title
            label_fontsize: Font size for axis labels
            tick_fontsize: Font size for tick labels
            legend_fontsize: Font size for legend
            line_width: Width of plot lines
            line_alpha: Alpha value for lines
            bar_alpha: Alpha value for background bars
        """
        self.figsize = figsize
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize
        self.legend_fontsize = legend_fontsize
        self.line_width = line_width
        self.line_alpha = line_alpha
        self.bar_alpha = bar_alpha
        
        # Define colors
        self.dad_color = "royalblue"
        self.fpa_color = "teal"
        self.esr_color = "indigo"
        self.bar_color = (58/256, 190/256, 120/256, self.bar_alpha)
        
        # Setup style
        plt.style.use('seaborn')
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    def _prepare_data(self) -> Dict[str, List[float]]:
        """
        Prepare the data for visualization.
        
        Returns:
            Dictionary containing all data arrays
        """
        # Define attack sizes
        self.attack_sizes = ['Baseline', '500', '1000', '2000', '5000']
        self.x = np.arange(len(self.attack_sizes))
        self.sample_sizes = [0, 500, 1000, 2000, 5000]
        
        # Magic training phase data
        magic_train_dad = [0.00, 0.04, 0.10, 0.21, 0.46]
        magic_train_fpa = [0.00, 0.18, 0.42, 0.85, 1.65]
        magic_train_esr = [0.021, 0.024, 0.028, 0.035, 0.045]
        
        # Magic inference phase data
        magic_infer_dad = [0.00, 0.03, 0.06, 0.10, 0.15]
        magic_infer_fpa = [0.00, 0.06, 0.10, 0.15, 0.22]
        magic_infer_esr = [0.021, 0.240, 0.650, 1.450, 3.250]
        
        # Airtag training phase data
        airtag_train_dad = [0.00, 0.08, 0.17, 0.29, 0.58]
        airtag_train_fpa = [0.00, 0.25, 0.58, 1.10, 2.20]
        airtag_train_esr = [0.110, 0.125, 0.145, 0.175, 0.215]
        
        # Airtag inference phase data
        airtag_infer_dad = [0.00, 0.07, 0.14, 0.22, 0.33]
        airtag_infer_fpa = [0.00, 0.12, 0.18, 0.28, 0.40]
        airtag_infer_esr = [0.110, 0.385, 0.880, 1.950, 4.280]
        
        # Calculate maximum values for y-axis limits
        max_dad = max(max(magic_train_dad), max(magic_infer_dad),
                     max(airtag_train_dad), max(airtag_infer_dad))
        max_fpa = max(max(magic_train_fpa), max(magic_infer_fpa),
                     max(airtag_train_fpa), max(airtag_infer_fpa))
        max_esr = max(max(magic_train_esr), max(magic_infer_esr),
                     max(airtag_train_esr), max(airtag_infer_esr))
        self.y_max = max(max_dad, max_fpa, max_esr) * 1.1
        self.sample_y_max = max(self.sample_sizes) * 1.1
        
        return {
            'magic_train': [magic_train_dad, magic_train_fpa, magic_train_esr],
            'magic_infer': [magic_infer_dad, magic_infer_fpa, magic_infer_esr],
            'airtag_train': [airtag_train_dad, airtag_train_fpa, airtag_train_esr],
            'airtag_infer': [airtag_infer_dad, airtag_infer_fpa, airtag_infer_esr]
        }
    
    def _setup_subplot(self, ax: plt.Axes, ax_right: plt.Axes,
                      data: List[List[float]], title: str) -> None:
        """
        Setup a single subplot with its twin axis.
        
        Args:
            ax: Main axis
            ax_right: Twin axis for sample size bars
            data: List of [DAD, FPA, ESR] data arrays
            title: Subplot title
        """
        # Create background bar chart
        ax_right.bar(self.x, self.sample_sizes, width=0.7, color=self.bar_color)
        ax_right.set_ylabel('Attack Sample Size', fontsize=self.label_fontsize, labelpad=10)
        ax_right.yaxis.set_label_position("right")
        ax_right.yaxis.set_tick_params(labelsize=self.tick_fontsize)
        ax_right.set_ylim([0, self.sample_y_max])
        
        # Plot main metrics
        ax.plot(self.x, data[0], linewidth=self.line_width, alpha=self.line_alpha,
               color=self.dad_color, linestyle="-", marker='o', label='DAD')
        ax.plot(self.x, data[1], linewidth=self.line_width, alpha=self.line_alpha,
               color=self.fpa_color, linestyle="-", marker='D', label='FPA')
        ax.plot(self.x, data[2], linewidth=self.line_width, alpha=self.line_alpha,
               color=self.esr_color, linestyle="-", marker='s', label='ESR')
        
        # Configure main axis
        ax.set_ylim([0, self.y_max])
        ax.set_ylabel('Impact Value', fontsize=self.label_fontsize)
        ax.yaxis.set_label_position("left")
        ax.yaxis.set_tick_params(labelsize=self.tick_fontsize)
        ax.grid(True)
        ax.legend(loc='upper left', prop={'size': self.legend_fontsize})
        ax.set_title(title, fontsize=self.label_fontsize + 2)
        ax.set_xticks(self.x)
        ax.set_xticklabels(self.attack_sizes)
        ax.set_xlabel('Attack Sample Size', fontsize=self.label_fontsize)
    
    def plot_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Create and display the comparison plot.
        
        Args:
            save_path: Optional path to save the plot
        """
        # Prepare data
        data = self._prepare_data()
        
        # Create figure and subplots
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Multi-dimensional Robustness Assessment - Magic vs Airtag',
                    fontsize=self.title_fontsize)
        
        # Setup each subplot
        self._setup_subplot(axes[0, 0], axes[0, 0].twinx(),
                          data['magic_train'],
                          'Magic - Training Phase Poisoning')
        self._setup_subplot(axes[0, 1], axes[0, 1].twinx(),
                          data['magic_infer'],
                          'Magic - Inference Phase Mimicry')
        self._setup_subplot(axes[1, 0], axes[1, 0].twinx(),
                          data['airtag_train'],
                          'Airtag - Training Phase Poisoning')
        self._setup_subplot(axes[1, 1], axes[1, 1].twinx(),
                          data['airtag_infer'],
                          'Airtag - Inference Phase Mimicry')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show plot
        if save_path:
            plt.savefig(f'{save_path}.pdf', format="pdf", bbox_inches='tight')
            plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Example usage of the InductiveBTCVisualizer."""
    visualizer = InductiveBTCVisualizer()
    visualizer.plot_comparison('magic_airtag_comparison')

if __name__ == "__main__":
    main()
