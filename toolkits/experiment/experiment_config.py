import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

class ExperimentConfig:
    """A tool for managing experiment configurations."""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("ExperimentConfig")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("experiment_config.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_experiment_config(self, 
                               name: str,
                               dataset: str,
                               model_params: Dict[str, Any],
                               attack_params: Dict[str, Any],
                               evaluation_params: Dict[str, Any]) -> str:
        """
        Create a new experiment configuration.
        
        Args:
            name: Name of the experiment
            dataset: Dataset to use
            model_params: Model parameters
            attack_params: Attack parameters
            evaluation_params: Evaluation parameters
            
        Returns:
            Path to the created configuration file
        """
        config = {
            'experiment_name': name,
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset,
            'model_parameters': model_params,
            'attack_parameters': attack_params,
            'evaluation_parameters': evaluation_params
        }
        
        config_path = self.config_dir / f"{name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.logger.info(f"Created experiment configuration: {config_path}")
        return str(config_path)
    
    def load_config(self, name: str) -> Dict[str, Any]:
        """
        Load an experiment configuration.
        
        Args:
            name: Name of the experiment
            
        Returns:
            Configuration dictionary
        """
        config_path = self.config_dir / f"{name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.logger.info(f"Loaded experiment configuration: {config_path}")
        return config
    
    def update_config(self, name: str, updates: Dict[str, Any]) -> None:
        """
        Update an existing experiment configuration.
        
        Args:
            name: Name of the experiment
            updates: Dictionary of updates to apply
        """
        config = self.load_config(name)
        config.update(updates)
        
        config_path = self.config_dir / f"{name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.logger.info(f"Updated experiment configuration: {config_path}")
    
    def list_experiments(self) -> list:
        """
        List all available experiment configurations.
        
        Returns:
            List of experiment names
        """
        return [f.stem for f in self.config_dir.glob("*.yaml")]
    
    def export_config(self, name: str, format: str = "json") -> str:
        """
        Export configuration to a different format.
        
        Args:
            name: Name of the experiment
            format: Export format ("json" or "yaml")
            
        Returns:
            Path to the exported file
        """
        config = self.load_config(name)
        export_path = self.config_dir / f"{name}.{format}"
        
        if format == "json":
            with open(export_path, 'w') as f:
                json.dump(config, f, indent=2)
        elif format == "yaml":
            with open(export_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported configuration to: {export_path}")
        return str(export_path)

def main():
    """Example usage of the ExperimentConfig."""
    config_manager = ExperimentConfig()
    
    # Create a sample experiment configuration
    model_params = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100
    }
    
    attack_params = {
        'poisoning_ratio': 0.1,
        'mimicry_threshold': 0.8
    }
    
    eval_params = {
        'metrics': ['DAD', 'FPA', 'ESR'],
        'cross_validation_folds': 5
    }
    
    config_path = config_manager.create_experiment_config(
        name="sample_experiment",
        dataset="ATLAS",
        model_params=model_params,
        attack_params=attack_params,
        evaluation_params=eval_params
    )
    
    # List all experiments
    experiments = config_manager.list_experiments()
    print(f"Available experiments: {experiments}")
    
    # Export configuration
    json_path = config_manager.export_config("sample_experiment", format="json")
    print(f"Exported configuration to: {json_path}")

if __name__ == "__main__":
    main() 