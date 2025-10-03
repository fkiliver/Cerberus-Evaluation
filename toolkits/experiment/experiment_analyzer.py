import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import seaborn as sns
from pathlib import Path

class ExperimentAnalyzer:
    """A tool for analyzing experiment results and generating visualizations."""
    
    def __init__(self, results_dir: str):
        """
        Initialize the analyzer with the directory containing experiment results.
        
        Args:
            results_dir: Directory containing experiment result files
        """
        self.results_dir = Path(results_dir)
        self.metrics = ['DAD', 'FPA', 'ESR']
        
    def load_results(self, pattern: str = "*.txt") -> Dict[str, pd.DataFrame]:
        """
        Load experiment results from files matching the pattern.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            Dictionary mapping experiment names to their results
        """
        results = {}
        for file in self.results_dir.glob(pattern):
            if file.stem.endswith('_baseline'):
                continue
            results[file.stem] = pd.read_csv(file, sep='\t')
        return results
    
    def generate_summary_statistics(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate summary statistics for all experiments.
        
        Args:
            results: Dictionary of experiment results
            
        Returns:
            DataFrame containing summary statistics
        """
        summary = []
        for exp_name, data in results.items():
            stats = {
                'Experiment': exp_name,
                'Mean DAD': data['DAD'].mean(),
                'Std DAD': data['DAD'].std(),
                'Mean FPA': data['FPA'].mean(),
                'Std FPA': data['FPA'].std(),
                'Mean ESR': data['ESR'].mean(),
                'Std ESR': data['ESR'].std()
            }
            summary.append(stats)
        return pd.DataFrame(summary)
    
    def plot_metric_distribution(self, results: Dict[str, pd.DataFrame], metric: str):
        """
        Plot the distribution of a metric across experiments.
        
        Args:
            results: Dictionary of experiment results
            metric: Metric to plot ('DAD', 'FPA', or 'ESR')
        """
        plt.figure(figsize=(10, 6))
        data = []
        labels = []
        for exp_name, df in results.items():
            data.append(df[metric])
            labels.extend([exp_name] * len(df))
        
        sns.boxplot(data=data)
        plt.xticks(range(len(results)), results.keys(), rotation=45)
        plt.title(f'Distribution of {metric} Across Experiments')
        plt.tight_layout()
        plt.savefig(self.results_dir / f'{metric}_distribution.png')
        plt.close()
    
    def plot_metric_correlation(self, results: Dict[str, pd.DataFrame]):
        """
        Plot correlation matrix between metrics.
        
        Args:
            results: Dictionary of experiment results
        """
        combined_data = pd.concat([df for df in results.values()])
        corr = combined_data[self.metrics].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Between Metrics')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'metric_correlation.png')
        plt.close()
    
    def generate_report(self, results: Dict[str, pd.DataFrame]) -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            results: Dictionary of experiment results
            
        Returns:
            Markdown formatted report
        """
        summary = self.generate_summary_statistics(results)
        
        report = ["# Experiment Analysis Report\n"]
        report.append("## Summary Statistics\n")
        report.append(summary.to_markdown())
        report.append("\n## Key Findings\n")
        
        # Add key findings based on the data
        for metric in self.metrics:
            best_exp = summary[f'Mean {metric}'].idxmin()
            worst_exp = summary[f'Mean {metric}'].idxmax()
            report.append(f"\n### {metric} Analysis")
            report.append(f"- Best performing experiment: {best_exp}")
            report.append(f"- Worst performing experiment: {worst_exp}")
            report.append(f"- Overall mean: {summary[f'Mean {metric}'].mean():.4f}")
            report.append(f"- Overall std: {summary[f'Std {metric}'].mean():.4f}")
        
        return "\n".join(report)

def main():
    """Example usage of the ExperimentAnalyzer."""
    analyzer = ExperimentAnalyzer("results")
    results = analyzer.load_results()
    
    # Generate visualizations
    for metric in ['DAD', 'FPA', 'ESR']:
        analyzer.plot_metric_distribution(results, metric)
    analyzer.plot_metric_correlation(results)
    
    # Generate and save report
    report = analyzer.generate_report(results)
    with open("results/analysis_report.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    main() 