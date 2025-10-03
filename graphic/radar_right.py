# --- 9. Create ALL Legend Handles (Easier to manage) ---
# Replace the individual handles with group handles for the new format

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Dict, Optional
import seaborn as sns

class RadarLegendCreator:
    """A class for creating and managing radar plot legends."""
    
    def __init__(self, 
                 marker_size: int = 8,
                 line_width: float = 2.0,
                 legend_font_size: int = 12,
                 legend_handle_length: float = 2.0,
                 legend_zorder: int = 10):
        """
        Initialize the legend creator.
        
        Args:
            marker_size: Size of markers in the legend
            line_width: Width of lines in the legend
            legend_font_size: Font size for legend text
            legend_handle_length: Length of legend handles
            legend_zorder: Z-order for the legend
        """
        self.marker_size = marker_size
        self.line_width = line_width
        self.legend_font_size = legend_font_size
        self.legend_handle_length = legend_handle_length
        self.legend_zorder = legend_zorder
        
        # Define colors
        self.colors = {
            'magic_training': '#1f77b4',  # Blue
            'airtag_training': '#ff7f0e',  # Orange
            'magic_overall': '#2ca02c',   # Green
            'airtag_overall': '#d62728'   # Red
        }
        
        # Define line styles
        self.line_styles = {
            'magic_training': '-',
            'airtag_training': '--',
            'magic_overall': '-.',
            'airtag_overall': ':'
        }
        
    def _create_group_header(self, label: str) -> Line2D:
        """
        Create a header line for a legend group.
        
        Args:
            label: The group header text
            
        Returns:
            Line2D object for the header
        """
        return Line2D([0], [0], label=label,
                     marker='', linestyle='', color='none')
    
    def _create_attack_marker(self,
                            label: str,
                            marker: str,
                            color: str,
                            linestyle: str) -> Line2D:
        """
        Create a marker line for an attack type.
        
        Args:
            label: The marker label
            marker: The marker style
            color: The marker color
            linestyle: The line style
            
        Returns:
            Line2D object for the marker
        """
        return Line2D([0], [0], label=label,
                     marker=marker,
                     markerfacecolor=color,
                     markeredgecolor='black',
                     markersize=self.marker_size,
                     linestyle=linestyle,
                     linewidth=self.line_width,
                     color=color)
    
    def create_legend_handles(self) -> List[Line2D]:
        """
        Create all legend handles in a grouped format.
        
        Returns:
            List of Line2D objects for the legend
        """
        # Data Poisoning Attacks group
        handles = [
            self._create_group_header('Data Poisoning Attacks:'),
            self._create_attack_marker('mark MAGIC', 'o',
                                     self.colors['magic_training'],
                                     self.line_styles['magic_training']),
            self._create_attack_marker('mark Airtag', 'o',
                                     self.colors['airtag_training'],
                                     self.line_styles['airtag_training'])
        ]
        
        # Mimicry Attacks group
        handles.extend([
            self._create_group_header('Mimicry Attacks:'),
            self._create_attack_marker('mark MAGIC', '^',
                                     self.colors['magic_training'], ':'),
            self._create_attack_marker('mark Airtag', '^',
                                     self.colors['airtag_training'], ':')
        ])
        
        # Combined Attacks group
        handles.extend([
            self._create_group_header('Combined Attacks:'),
            self._create_attack_marker('mark MAGIC', 's',
                                     self.colors['magic_overall'],
                                     self.line_styles['magic_overall']),
            self._create_attack_marker('mark Airtag', 's',
                                     self.colors['airtag_overall'],
                                     self.line_styles['airtag_overall'])
        ])
        
        return handles
    
    def add_legend_to_plot(self, ax: plt.Axes) -> None:
        """
        Add the legend to the plot.
        
        Args:
            ax: The axes object to add the legend to
        """
        legend = ax.legend(
            handles=self.create_legend_handles(),
            loc='upper center',
            bbox_to_anchor=(0.5, 1.30),
            fontsize=self.legend_font_size,
            handlelength=self.legend_handle_length,
            ncol=3,
            columnspacing=1.0,
            handletextpad=0.5,
            frameon=True,
            facecolor='white',
            framealpha=0.7
        )
        legend.set_zorder(self.legend_zorder)

def main():
    """Example usage of the RadarLegendCreator."""
    # Create a sample plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create and add the legend
    legend_creator = RadarLegendCreator()
    legend_creator.add_legend_to_plot(ax)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
