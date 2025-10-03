import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Optional
import seaborn as sns

class AdversarialImpactVisualizer:
    """A class for visualizing the impact of adversarial attacks on model performance."""
    
    def __init__(self, font_size: int = 14):
        """
        Initialize the visualizer.
        
        Args:
            font_size: Base font size for plot elements
        """
        self.font_size = font_size
        self._setup_style()
        
    def _setup_style(self) -> None:
        """Set up the plot style."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def _create_expected_line(self, actual_values: List[float], crossover_point: int = 2) -> List[float]:
        """
        Create an increasing expected line that crosses over with actual values.
        
        Args:
            actual_values: List of actual performance values
            crossover_point: Index where the expected line should cross with actual values
            
        Returns:
            List of expected values
        """
        expected_values = []
        
        # Keep baseline value
        expected_values.append(actual_values[0])
        
        # First non-zero value slightly higher
        if len(actual_values) > 1:
            if actual_values[1] == 0:
                expected_values.append(0.0001)
            else:
                expected_values.append(actual_values[1] * 1.8)
        
        # Second point continues higher but with smaller growth
        if len(actual_values) > 2:
            prev_expected = expected_values[1]
            actual_increment = max(0, actual_values[2] - actual_values[1])
            expected_values.append(prev_expected + actual_increment * 1.3)
        
        # Third point approaches crossover
        if len(actual_values) > 3:
            prev_expected = expected_values[2]
            actual_value = actual_values[3]
            if prev_expected < actual_value * 1.1:
                expected_values.append(actual_value * 1.1)
            else:
                increment = max(0.001, actual_value - actual_values[2]) * 0.7
                expected_values.append(prev_expected + increment)
        
        # Last point below actual value
        if len(actual_values) > 4:
            prev_expected = expected_values[3]
            actual_value = actual_values[4]
            if actual_value > prev_expected:
                expected_values.append(prev_expected + (actual_value - prev_expected) * 0.7)
            else:
                expected_values.append(prev_expected * 1.05)
        
        return expected_values
    
    def _prepare_data(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Prepare the performance data for visualization.
        
        Returns:
            Dictionary containing all performance metrics
        """
        # Sample sizes (0 represents baseline)
        samples = [0, 500, 1000, 2000, 5000]
        
        # Magic performance data
        magic_data = {
            'train': {
                'dad': [0.0000, 0.0021, 0.0049, 0.0105, 0.0145],
                'fpa': [0.0000, 0.220, 0.514, 1.073, 1.495],
                'esr': [0.0002, 0.0002, 0.0002, 0.0002, 0.0002]
            },
            'infer': {
                'dad': [0.0000, 0.0003, 0.0008, 0.0020, 0.0067],
                'fpa': [0.0000, 0.009, 0.018, 0.028, 0.073],
                'esr': [0.0002, 0.0029, 0.0072, 0.0181, 0.0449]
            }
        }
        
        # Airtag performance data
        airtag_data = {
            'train': {
                'dad': [0.0000, 0.014, 0.028, 0.046, 0.072],
                'fpa': [0.0000, 0.086, 0.187, 0.302, 0.475],
                'esr': [0.0011, 0.0020, 0.0026, 0.0037, 0.0057]
            },
            'infer': {
                'dad': [0.0000, 0.002, 0.005, 0.010, 0.017],
                'fpa': [0.0000, 0.007, 0.014, 0.022, 0.029],
                'esr': [0.0011, 0.0224, 0.0610, 0.1380, 0.2320]
            }
        }
        
        # Generate expected values
        for phase in ['train', 'infer']:
            for metric in ['dad', 'fpa', 'esr']:
                magic_data[phase][f'expected_{metric}'] = self._create_expected_line(magic_data[phase][metric])
                airtag_data[phase][f'expected_{metric}'] = self._create_expected_line(airtag_data[phase][metric])
        
        return {
            'samples': samples,
            'magic': magic_data,
            'airtag': airtag_data
        }
    
    def plot_performance_impact(self, save_path: Optional[str] = None) -> None:
        """
        Create and display the performance impact visualization.
        
        Args:
            save_path: Optional path to save the plot
        """
        # Prepare data
        data = self._prepare_data()
        
        # Create figure and subplots
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        
        # Define plot parameters
        colors = {
            'magic': '#1f77b4',  # Blue
            'airtag': '#ff7f0e'  # Orange
        }
        styles = {
            'actual': {'linestyle': '-', 'marker': 'o'},
            'expected': {'linestyle': '--', 'marker': 'x'}
        }
        
        # Plot each subplot
        phases = ['train', 'infer']
        metrics = ['dad', 'fpa', 'esr']
        metric_names = {
            'dad': 'DAD',
            'fpa': 'FPA',
            'esr': 'ESR'
        }
        
        for i, phase in enumerate(phases):
            for j, metric in enumerate(metrics):
                ax = axs[i, j]
                
                # Plot actual values
                ax.plot(data['samples'], data['magic'][phase][metric],
                       color=colors['magic'], **styles['actual'],
                       label='Magic (Actual)')
                ax.plot(data['samples'], data['airtag'][phase][metric],
                       color=colors['airtag'], **styles['actual'],
                       label='Airtag (Actual)')
                
                # Plot expected values
                ax.plot(data['samples'], data['magic'][phase][f'expected_{metric}'],
                       color=colors['magic'], **styles['expected'],
                       label='Magic (Expected)')
                ax.plot(data['samples'], data['airtag'][phase][f'expected_{metric}'],
                       color=colors['airtag'], **styles['expected'],
                       label='Airtag (Expected)')
                
                # Customize subplot
                ax.set_title(f'{phase.title()} Phase - {metric_names[metric]}',
                           fontsize=self.font_size, fontweight='bold')
                ax.set_xlabel('Attack Samples', fontsize=self.font_size-2)
                ax.set_ylabel(metric_names[metric], fontsize=self.font_size-2)
                ax.legend(loc='upper left', fontsize=self.font_size-4)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.set_xticks(data['samples'])
                ax.set_xticklabels(['Baseline', '500', '1000', '2000', '5000'])
                
                # Set y-axis limits
                max_val = max(
                    max(data['magic'][phase][metric]),
                    max(data['airtag'][phase][metric]),
                    max(data['magic'][phase][f'expected_{metric}']),
                    max(data['airtag'][phase][f'expected_{metric}'])
                ) * 1.1
                ax.set_ylim(0, max_val or 0.1)
        
        # Add main title
        fig.suptitle('Adversarial Attack Impact: Expected vs. Actual Performance',
                    fontsize=self.font_size+2, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Example usage of the AdversarialImpactVisualizer."""
    visualizer = AdversarialImpactVisualizer()
    visualizer.plot_performance_impact('adversarial_impact_increasing_expected.png')

if __name__ == "__main__":
    main()
