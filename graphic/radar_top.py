import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as PathEffects
from scipy.interpolate import splprep, splev
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple, Optional
import seaborn as sns

AXIS_LIMITS = {
    'DAD': [0.004, 0.5],
    'ESR': [0.00005, 1],
    'FPA': [0.01, 3.0]
}

AXIS_COLORS = {
    'DAD': 'black',
    'ESR': 'black',
    'FPA': 'black'
}

magic_color_training = '#1f77b4'
magic_color_overall = '#08306b'
airtag_color_training = '#e31a1c'
airtag_color_overall = '#d62728'

linestyle_magic_training = '--'  # Long dashed line for poisoning
linestyle_airtag_training = '--' # Short dashed line for mimicry
linestyle_magic_overall = '-'
linestyle_airtag_overall = '-'

RADAR_MIN = 0
RADAR_MAX = 10

SMALLER_IS_BETTER = True

NUM_RINGS = 5

LINE_WIDTH = 2.5
MARKER_SIZE = 8
AXIS_LINE_WIDTH = 2.0
ARROW_SIZE = 3

AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 10
LEGEND_FONT_SIZE = 13 # Increased legend font slightly

FILL_ALPHA_NON_SOLID = 0.1
FILL_ALPHA_SOLID = 0.2

MIN_TICK_RADIUS = 0.25

ARROW_ZORDER = 10
TEXT_ZORDER = 15
LEGEND_ZORDER = 20

SHADOW_COLOR = 'white'
SHADOW_LINEWIDTH = 1.5

INTERP_POINTS = 300
LEGEND_HANDLE_LENGTH = 3.5 # Adjust this value to control line length in legend

class TopRadarVisualizer:
    """A class for creating radar plots to visualize model performance metrics."""
    
    def __init__(self,
                 figsize: Tuple[int, int] = (7, 8),
                 axis_limits: Optional[Dict[str, List[float]]] = None,
                 axis_colors: Optional[Dict[str, str]] = None,
                 radar_min: float = 0,
                 radar_max: float = 10,
                 smaller_is_better: bool = True,
                 num_rings: int = 5,
                 line_width: float = 2.5,
                 marker_size: int = 8,
                 axis_line_width: float = 2.0,
                 arrow_size: int = 3,
                 axis_label_size: int = 15,
                 tick_label_size: int = 10,
                 legend_font_size: int = 13,
                 fill_alpha_non_solid: float = 0.1,
                 fill_alpha_solid: float = 0.2,
                 min_tick_radius: float = 0.25,
                 interp_points: int = 300):
        """
        Initialize the radar visualizer.
        
        Args:
            figsize: Figure size (width, height)
            axis_limits: Dictionary of metric names to [min, max] values
            axis_colors: Dictionary of metric names to colors
            radar_min: Minimum value for radar plot
            radar_max: Maximum value for radar plot
            smaller_is_better: Whether smaller values are better
            num_rings: Number of concentric rings
            line_width: Width of plot lines
            marker_size: Size of plot markers
            axis_line_width: Width of axis lines
            arrow_size: Size of arrow heads
            axis_label_size: Font size for axis labels
            tick_label_size: Font size for tick labels
            legend_font_size: Font size for legend
            fill_alpha_non_solid: Alpha for non-solid fills
            fill_alpha_solid: Alpha for solid fills
            min_tick_radius: Minimum radius for tick labels
            interp_points: Number of points for interpolation
        """
        self.figsize = figsize
        self.axis_limits = axis_limits or {
            'DAD': [0.004, 0.5],
            'ESR': [0.00005, 1],
            'FPA': [0.01, 3.0]
        }
        self.axis_colors = axis_colors or {
            'DAD': 'black',
            'ESR': 'black',
            'FPA': 'black'
        }
        self.radar_min = radar_min
        self.radar_max = radar_max
        self.smaller_is_better = smaller_is_better
        self.num_rings = num_rings
        self.line_width = line_width
        self.marker_size = marker_size
        self.axis_line_width = axis_line_width
        self.arrow_size = arrow_size
        self.axis_label_size = axis_label_size
        self.tick_label_size = tick_label_size
        self.legend_font_size = legend_font_size
        self.fill_alpha_non_solid = fill_alpha_non_solid
        self.fill_alpha_solid = fill_alpha_solid
        self.min_tick_radius = min_tick_radius
        self.interp_points = interp_points
        
        # Style settings
        self.magic_color_training = '#1f77b4'
        self.magic_color_overall = '#08306b'
        self.airtag_color_training = '#e31a1c'
        self.airtag_color_overall = '#d62728'
        
        self.linestyle_magic_training = '--'  # Long dashed line for poisoning
        self.linestyle_airtag_training = '--' # Short dashed line for mimicry
        self.linestyle_magic_overall = '-'
        self.linestyle_airtag_overall = '-'
        
        self._setup_style()
    
    def _setup_style(self) -> None:
        """Set up the visualization style."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def _map_value(self, value: float, metric: str) -> float:
        """
        Map a value to the radar plot scale.
        
        Args:
            value: Original value
            metric: Metric name
            
        Returns:
            Mapped value
        """
        min_val, max_val = self.axis_limits[metric]
        value = max(min(value, max_val), min_val)
        if value <= 0 or min_val <= 0 or max_val <= 0:
            print(f"Warning: Non-positive value encountered for {metric}: {value}. Clamping may occur.")
            value = max(value, 1e-9)
            min_val = max(min_val, 1e-9)
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        log_value = np.log10(value)
        if log_max == log_min:
            normalized = 0.5
        else:
            normalized = (log_value - log_min) / (log_max - log_min)
        if self.smaller_is_better:
            normalized = 1 - normalized
        return self.radar_min + normalized * (self.radar_max - self.radar_min)
    
    def _interpolate_spline(self, angles: np.ndarray, data_closed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate data points using spline.
        
        Args:
            angles: Array of angles
            data_closed: Array of closed data points
            
        Returns:
            Tuple of (theta, radius) arrays
        """
        x = data_closed * np.cos(angles)
        y = data_closed * np.sin(angles)
        tck, u = splprep([x, y], s=0, per=True, k=3)
        u_new = np.linspace(u.min(), u.max(), self.interp_points)
        x_new, y_new = splev(u_new, tck)
        r_new = np.sqrt(x_new**2 + y_new**2)
        theta_new = np.arctan2(y_new, x_new)
        sort_indices = np.argsort(theta_new)
        return theta_new[sort_indices], r_new[sort_indices]
    
    def _create_legend_handles(self) -> List[Line2D]:
        """
        Create legend handles for the plot.
        
        Returns:
            List of Line2D objects for legend
        """
        return [
            Line2D([0], [0], marker='o', label='MAGIC',
                  markerfacecolor=self.magic_color_training,
                  markeredgecolor='black',
                  markersize=self.marker_size,
                  linestyle=self.linestyle_magic_training,
                  linewidth=self.line_width,
                  color=self.magic_color_training),
            Line2D([0], [0], marker='o', label='AirTag',
                  markerfacecolor=self.airtag_color_training,
                  markeredgecolor='black',
                  markersize=self.marker_size,
                  linestyle=self.linestyle_airtag_training,
                  linewidth=self.line_width,
                  color=self.airtag_color_training),
            Line2D([0], [0], marker='s', label='MAGIC          ',
                  markerfacecolor=self.magic_color_overall,
                  markeredgecolor='black',
                  markersize=self.marker_size,
                  linestyle=self.linestyle_magic_overall,
                  linewidth=self.line_width,
                  color=self.magic_color_overall),
            Line2D([0], [0], marker='s', label='AirTag          ',
                  markerfacecolor=self.airtag_color_overall,
                  markeredgecolor='black',
                  markersize=self.marker_size,
                  linestyle=self.linestyle_airtag_overall,
                  linewidth=self.line_width,
                  color=self.airtag_color_overall),
            Line2D([0], [0], marker='^', label='MAGIC',
                  markerfacecolor=self.magic_color_training,
                  markeredgecolor='black',
                  markersize=self.marker_size,
                  linestyle=':',
                  linewidth=self.line_width,
                  color=self.magic_color_training),
            Line2D([0], [0], marker='^', label='AirTag',
                  markerfacecolor=self.airtag_color_training,
                  markeredgecolor='black',
                  markersize=self.marker_size,
                  linestyle=':',
                  linewidth=self.line_width,
                  color=self.airtag_color_training)
        ]
    
    def _draw_custom_legend(self, fig: plt.Figure, ax: plt.Axes) -> None:
        """
        Draw a custom legend for the plot.
        
        Args:
            fig: Figure object
            ax: Axes object
        """
        # Legend position parameters
        x_start = 0.03
        y_start = 1.25
        y_gap = 0.06
        x_gap_label = 0.5
        x_gap_between = 0.25
        line_length = 0.09
        box_x_fig = 0.1
        box_y_fig = 0.8
        box_width = 0.8
        box_height = 0.13
        
        # Create legend box
        legend_box = FancyBboxPatch(
            (box_x_fig, box_y_fig),
            box_width,
            box_height,
            boxstyle="round,pad=0.001",
            transform=fig.transFigure,
            facecolor='white',
            edgecolor='black',
            alpha=0.1,
            linewidth=1.5,
            mutation_scale=10,
            zorder=15
        )
        fig.patches.append(legend_box)
        
        # Create legend handles
        handles = self._create_legend_handles()
        
        # Define legend information
        legend_info = [
            ('Data Poisoning Attacks:', handles[0], handles[1]),
            ('Mimicry Attacks:', handles[4], handles[5]),
            ('Combined Attacks:', handles[2], handles[3])
        ]
        
        # Draw legend elements
        for idx, (attack_label, magic_handle, airtag_handle) in enumerate(legend_info):
            y = y_start - idx * y_gap
            
            # Draw attack type label
            ax.text(x_start, y, attack_label,
                   transform=ax.transAxes,
                   ha='left', va='center',
                   fontsize=self.legend_font_size)
            
            # Draw MAGIC line and marker
            x0 = x_start + x_gap_label
            x1 = x0 + line_length
            ax.plot([x0, x1], [y, y],
                   transform=ax.transAxes,
                   linestyle=magic_handle.get_linestyle(),
                   linewidth=self.line_width,
                   color=magic_handle.get_color(),
                   clip_on=False)
            
            xm = (x0 + x1) / 2
            ax.plot(xm, y,
                   transform=ax.transAxes,
                   marker=magic_handle.get_marker(),
                   markersize=self.marker_size,
                   markerfacecolor=magic_handle.get_markerfacecolor(),
                   markeredgecolor='black',
                   markeredgewidth=0.8,
                   clip_on=False)
            
            # Draw AirTag line and marker
            x0 = x_start + x_gap_label + x_gap_between
            x1 = x0 + line_length
            ax.plot([x0, x1], [y, y],
                   transform=ax.transAxes,
                   linestyle=airtag_handle.get_linestyle(),
                   linewidth=self.line_width,
                   color=airtag_handle.get_color(),
                   clip_on=False)
            
            xm = (x0 + x1) / 2
            ax.plot(xm, y,
                   transform=ax.transAxes,
                   marker=airtag_handle.get_marker(),
                   markersize=self.marker_size,
                   markerfacecolor=airtag_handle.get_markerfacecolor(),
                   markeredgecolor='black',
                   markeredgewidth=0.8,
                   clip_on=False)
    
    def plot_radar(self,
                  data: Dict[str, List[float]],
                  metrics: List[str],
                  title: str,
                  save_path: Optional[str] = None) -> None:
        """
        Create and display the radar plot.
        
        Args:
            data: Dictionary containing performance data
            metrics: List of metric names
            title: Plot title
            save_path: Optional path to save the plot
        """
        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, subplot_kw=dict(polar=True))
        
        # Calculate angles
        N = len(metrics)
        angles = np.array([n/float(N)*2*np.pi + np.pi/2 for n in range(N)])
        
        # Map data
        magic_mapped = [self._map_value(data['magic_training'][i], metrics[i]) for i in range(N)]
        airtag_mapped = [self._map_value(data['airtag_training'][i], metrics[i]) for i in range(N)]
        magic_overall_mapped = [self._map_value(data['magic_overall'][i], metrics[i]) for i in range(N)]
        airtag_overall_mapped = [self._map_value(data['airtag_overall'][i], metrics[i]) for i in range(N)]
        magic_inf_mapped = [self._map_value(data['magic_inference'][i], metrics[i]) for i in range(N)]
        airtag_inf_mapped = [self._map_value(data['airtag_inference'][i], metrics[i]) for i in range(N)]
        
        # Close data loops
        angles_closed = np.concatenate((angles, [angles[0]]))
        magic_closed = np.array(magic_mapped + magic_mapped[:1])
        airtag_closed = np.array(airtag_mapped + airtag_mapped[:1])
        magic_overall_closed = np.array(magic_overall_mapped + magic_overall_mapped[:1])
        airtag_overall_closed = np.array(airtag_overall_mapped + airtag_overall_mapped[:1])
        magic_inf_closed = np.array(magic_inf_mapped + magic_inf_mapped[:1])
        airtag_inf_closed = np.array(airtag_inf_mapped + airtag_inf_mapped[:1])
        
        # Interpolate data
        theta_magic, r_magic = self._interpolate_spline(angles_closed, magic_closed)
        theta_airtag, r_airtag = self._interpolate_spline(angles_closed, airtag_closed)
        theta_magic_ov, r_magic_ov = self._interpolate_spline(angles_closed, magic_overall_closed)
        theta_airtag_ov, r_airtag_ov = self._interpolate_spline(angles_closed, airtag_overall_closed)
        theta_magic_inf, r_magic_inf = self._interpolate_spline(angles_closed, magic_inf_closed)
        theta_airtag_inf, r_airtag_inf = self._interpolate_spline(angles_closed, airtag_inf_closed)
        
        # Setup plot appearance
        ax.set_ylim(self.radar_min, self.radar_max)
        ax.set_xticks(angles)
        ax.set_xticklabels([''] * N)
        ax.set_yticklabels([])
        ax.grid(False)
        
        # Draw rings
        r_grid = np.linspace(self.radar_min, self.radar_max, self.num_rings + 1)
        for i in range(self.num_rings):
            if (self.num_rings - 1 - i) % 2 == 0:
                ax.fill_between(np.linspace(0, 2*np.pi, 100),
                              r_grid[i], r_grid[i+1],
                              color='lightgray', alpha=0.3, zorder=0)
        for r in r_grid:
            ax.plot(np.linspace(0, 2*np.pi, 100), [r]*100,
                   color='grey', ls=':', lw=0.5, zorder=0.5)
        
        # Draw axes and arrows
        for i, metric in enumerate(metrics):
            angle = angles[i]
            color = self.axis_colors[metric]
            ax.plot([angle, angle], [self.radar_min, self.radar_max],
                   color=color, linewidth=self.axis_line_width, zorder=1)
            arrow = FancyArrowPatch(
                (angle, self.radar_max * 1),
                (angle, self.radar_max * 0.9),
                transform=ax.transData,
                color=color,
                linewidth=self.axis_line_width,
                arrowstyle=f'-|>, head_width={self.arrow_size}, head_length={self.arrow_size*1.5}',
                zorder=10)
            ax.add_patch(arrow)
        
        # Plot interpolated lines and fills
        ax.plot(theta_magic, r_magic,
               linestyle=self.linestyle_magic_training,
               linewidth=self.line_width,
               color=self.magic_color_training,
               label='_nolegend_')
        ax.fill(theta_magic, r_magic,
               color=self.magic_color_training,
               alpha=self.fill_alpha_non_solid)
        
        ax.plot(theta_airtag, r_airtag,
               linestyle=self.linestyle_airtag_training,
               linewidth=self.line_width,
               color=self.airtag_color_training,
               label='_nolegend_')
        ax.fill(theta_airtag, r_airtag,
               color=self.airtag_color_training,
               alpha=self.fill_alpha_non_solid)
        
        ax.plot(theta_magic_ov, r_magic_ov,
               linestyle=self.linestyle_magic_overall,
               linewidth=self.line_width,
               color=self.magic_color_overall,
               label='_nolegend_')
        ax.fill(theta_magic_ov, r_magic_ov,
               color=self.magic_color_overall,
               alpha=self.fill_alpha_solid)
        
        ax.plot(theta_airtag_ov, r_airtag_ov,
               linestyle=self.linestyle_airtag_overall,
               linewidth=self.line_width,
               color=self.airtag_color_overall,
               label='_nolegend_')
        ax.fill(theta_airtag_ov, r_airtag_ov,
               color=self.airtag_color_overall,
               alpha=self.fill_alpha_solid)
        
        ax.plot(theta_magic_inf, r_magic_inf,
               linestyle=':',
               linewidth=self.line_width,
               color=self.magic_color_training,
               label='_nolegend_')
        ax.fill(theta_magic_inf, r_magic_inf,
               color=self.magic_color_training,
               alpha=self.fill_alpha_non_solid)
        
        ax.plot(theta_airtag_inf, r_airtag_inf,
               linestyle=':',
               linewidth=self.line_width,
               color=self.airtag_color_training,
               label='_nolegend_')
        ax.fill(theta_airtag_inf, r_airtag_inf,
               color=self.airtag_color_training,
               alpha=self.fill_alpha_non_solid)
        
        # Plot data point markers
        ax.plot(angles, magic_mapped, 'o',
               markersize=self.marker_size,
               color=self.magic_color_training,
               markeredgecolor='black',
               markeredgewidth=0.5,
               label='_nolegend_',
               linestyle='None',
               zorder=5)
        
        ax.plot(angles, airtag_mapped, 'o',
               markersize=self.marker_size,
               color=self.airtag_color_training,
               markeredgecolor='black',
               markeredgewidth=0.5,
               label='_nolegend_',
               linestyle='None',
               zorder=5)
        
        ax.plot(angles, magic_overall_mapped, 's',
               markersize=self.marker_size,
               color=self.magic_color_overall,
               markeredgecolor='black',
               markeredgewidth=0.5,
               label='_nolegend_',
               linestyle='None',
               zorder=5)
        
        ax.plot(angles, airtag_overall_mapped, 's',
               markersize=self.marker_size,
               color=self.airtag_color_overall,
               markeredgecolor='black',
               markeredgewidth=0.5,
               label='_nolegend_',
               linestyle='None',
               zorder=5)
        
        ax.plot(angles, magic_inf_mapped, '^',
               markersize=self.marker_size,
               color=self.magic_color_training,
               markeredgecolor='black',
               markeredgewidth=0.5,
               label='_nolegend_',
               linestyle='None',
               zorder=5)
        
        ax.plot(angles, airtag_inf_mapped, '^',
               markersize=self.marker_size,
               color=self.airtag_color_training,
               markeredgecolor='black',
               markeredgewidth=0.5,
               label='_nolegend_',
               linestyle='None',
               zorder=5)
        
        # Add axis and tick labels
        for i, label in enumerate(metrics):
            angle = angles[i]
            txt = ax.text(angle, self.radar_max * 1.15, label,
                         ha='center', va='center',
                         fontsize=self.axis_label_size,
                         color=self.axis_colors[label],
                         fontweight='bold',
                         zorder=15)
            txt.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                                                       foreground='white')])
        
        # Add tick labels
        for i, metric in enumerate(metrics):
            angle = angles[i]
            min_val, max_val = self.axis_limits[metric]
            tick_values = np.logspace(np.log10(min_val), np.log10(max_val), 5)
            for val in tick_values:
                mapped_val = self._map_value(val, metric)
                if mapped_val < self.radar_max * self.min_tick_radius:
                    continue
                ha = 'left' if np.cos(angle) < -0.1 else ('right' if np.cos(angle) > 0.1 else 'center')
                va = 'top' if np.sin(angle) < -0.1 else ('bottom' if np.sin(angle) > 0.1 else 'center')
                rot = np.degrees(angle)
                if rot > 90 and rot < 270:
                    rot -= 180
                offset_r = 0.3
                offset_angle_factor = 0.05
                label_angle = angle + offset_angle_factor * np.sign(np.cos(angle)) if abs(np.cos(angle)) > 0.1 else angle
                if val < 0.01:
                    tick_label = f"{val:.4f}"
                elif val < 1:
                    tick_label = f"{val:.2f}"
                else:
                    tick_label = f"{val:.1f}"
                txt = ax.text(label_angle, mapped_val + offset_r, tick_label,
                            ha=ha, va=va,
                            fontsize=self.tick_label_size,
                            color='black',
                            rotation=rot,
                            rotation_mode='anchor',
                            zorder=15)
                txt.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                                                          foreground='white')])
        
        # Add custom legend
        self._draw_custom_legend(fig, ax)
        
        # Save or show plot
        if save_path:
            plt.savefig(f'{save_path}.pdf', format="pdf", bbox_inches='tight')
            plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Example usage of the TopRadarVisualizer."""
    # Define metrics and data
    metrics = ['DAD', 'ESR', 'FPA']
    data = {
        'magic_training': [0.0145, 0.0002, 1.495],
        'airtag_training': [0.072, 0.0057, 0.475],
        'magic_overall': [0.0212, 0.0481, 1.568],
        'airtag_overall': [0.089, 0.285, 0.504],
        'magic_inference': [0.0067, 0.0449, 0.073],
        'airtag_inference': [0.017, 0.232, 0.029]
    }
    
    # Create and plot radar visualization
    visualizer = TopRadarVisualizer()
    visualizer.plot_radar(data, metrics, 'Model Performance Comparison',
                         'model_performance_radar_top')

if __name__ == "__main__":
    main()