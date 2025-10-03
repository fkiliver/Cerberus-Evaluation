import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import random
from datetime import datetime
import logging
import networkx as nx

class AttackGenerator:
    """A tool for generating various types of attacks for Cerberus framework."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the attack generator.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("AttackGenerator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("attack_generation.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def generate_poisoning_attack(self, 
                                benign_data: List[Dict],
                                malicious_data: List[Dict],
                                attack_ratio: float = 0.1,
                                max_attempts: int = 1000) -> List[Dict]:
        """
        Generate data poisoning attacks by injecting malicious samples into benign data.
        
        Args:
            benign_data: List of benign data samples
            malicious_data: List of malicious data samples
            attack_ratio: Ratio of malicious samples to inject
            max_attempts: Maximum number of attempts to find valid injection points
            
        Returns:
            List of poisoned data samples
        """
        num_poison = int(len(benign_data) * attack_ratio)
        poisoned_data = benign_data.copy()
        
        # Create a graph representation of the data
        G = nx.DiGraph()
        for i, sample in enumerate(benign_data):
            G.add_node(i, data=sample)
            if 'parent_id' in sample:
                G.add_edge(sample['parent_id'], i)
        
        # Find injection points
        injection_points = []
        attempts = 0
        while len(injection_points) < num_poison and attempts < max_attempts:
            # Select a random malicious sample
            malicious_sample = random.choice(malicious_data)
            
            # Find a suitable injection point
            for node in G.nodes():
                if G.out_degree(node) < 3:  # Limit branching factor
                    injection_points.append((node, malicious_sample))
                    break
            
            attempts += 1
        
        # Inject malicious samples
        for node, malicious_sample in injection_points:
            # Create a modified version of the malicious sample
            modified_sample = malicious_sample.copy()
            modified_sample['timestamp'] = poisoned_data[node]['timestamp']
            modified_sample['parent_id'] = node
            
            # Insert the modified sample
            poisoned_data.insert(node + 1, modified_sample)
        
        self.logger.info(f"Generated poisoning attack with {len(injection_points)} samples")
        return poisoned_data
    
    def generate_mimicry_attack(self,
                              benign_data: List[Dict],
                              malicious_data: List[Dict],
                              attack_ratio: float = 0.1,
                              mimicry_threshold: float = 0.8) -> List[Dict]:
        """
        Generate mimicry attacks by embedding malicious behavior within benign patterns.
        
        Args:
            benign_data: List of benign data samples
            malicious_data: List of malicious data samples
            attack_ratio: Ratio of malicious samples to inject
            mimicry_threshold: Threshold for determining successful mimicry
            
        Returns:
            List of data samples with embedded mimicry attacks
        """
        num_attacks = int(len(benign_data) * attack_ratio)
        attacked_data = benign_data.copy()
        
        # Group benign samples by type
        benign_groups = {}
        for sample in benign_data:
            sample_type = sample.get('type', 'unknown')
            if sample_type not in benign_groups:
                benign_groups[sample_type] = []
            benign_groups[sample_type].append(sample)
        
        # Generate mimicry attacks
        for _ in range(num_attacks):
            # Select a random malicious sample
            malicious_sample = random.choice(malicious_data)
            
            # Find a suitable benign pattern to mimic
            target_type = random.choice(list(benign_groups.keys()))
            benign_pattern = random.choice(benign_groups[target_type])
            
            # Create a mimicry attack
            mimicry_attack = self._create_mimicry_attack(malicious_sample, benign_pattern)
            
            # Insert the attack
            insert_pos = random.randint(0, len(attacked_data))
            attacked_data.insert(insert_pos, mimicry_attack)
        
        self.logger.info(f"Generated mimicry attack with {num_attacks} samples")
        return attacked_data
    
    def _create_mimicry_attack(self, 
                             malicious_sample: Dict,
                             benign_pattern: Dict) -> Dict:
        """
        Create a mimicry attack by combining malicious and benign patterns.
        
        Args:
            malicious_sample: Original malicious sample
            benign_pattern: Benign pattern to mimic
            
        Returns:
            Modified malicious sample that mimics benign pattern
        """
        mimicry_attack = malicious_sample.copy()
        
        # Copy relevant benign attributes
        for key in ['timestamp', 'process_name', 'user_id']:
            if key in benign_pattern:
                mimicry_attack[key] = benign_pattern[key]
        
        # Modify malicious attributes to match benign pattern
        if 'parameters' in mimicry_attack and 'parameters' in benign_pattern:
            mimicry_attack['parameters'] = self._blend_parameters(
                mimicry_attack['parameters'],
                benign_pattern['parameters']
            )
        
        return mimicry_attack
    
    def _blend_parameters(self,
                         malicious_params: Dict,
                         benign_params: Dict) -> Dict:
        """
        Blend malicious and benign parameters to create a more subtle attack.
        
        Args:
            malicious_params: Parameters from malicious sample
            benign_params: Parameters from benign sample
            
        Returns:
            Blended parameters
        """
        blended_params = benign_params.copy()
        
        # Preserve critical malicious parameters while adopting benign structure
        for key, value in malicious_params.items():
            if key.startswith('malicious_'):
                blended_params[key] = value
            elif key in benign_params:
                # Blend values if they exist in both
                if isinstance(value, (int, float)) and isinstance(benign_params[key], (int, float)):
                    blended_params[key] = (value + benign_params[key]) / 2
                else:
                    blended_params[key] = value
        
        return blended_params
    
    def generate_combined_attack(self,
                               benign_data: List[Dict],
                               malicious_data: List[Dict],
                               attack_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate a combination of poisoning and mimicry attacks.
        
        Args:
            benign_data: List of benign data samples
            malicious_data: List of malicious data samples
            attack_ratio: Ratio of malicious samples to inject
            
        Returns:
            Tuple of (poisoned training data, attacked test data)
        """
        # Split data into training and test sets
        split_idx = int(len(benign_data) * 0.7)
        train_data = benign_data[:split_idx]
        test_data = benign_data[split_idx:]
        
        # Generate poisoning attack for training data
        poisoned_train = self.generate_poisoning_attack(
            train_data,
            malicious_data,
            attack_ratio=attack_ratio/2
        )
        
        # Generate mimicry attack for test data
        attacked_test = self.generate_mimicry_attack(
            test_data,
            malicious_data,
            attack_ratio=attack_ratio/2
        )
        
        self.logger.info("Generated combined attack (poisoning + mimicry)")
        return poisoned_train, attacked_test
    
    def save_attack_data(self,
                        data: List[Dict],
                        output_path: str,
                        attack_type: str) -> None:
        """
        Save generated attack data to file.
        
        Args:
            data: List of data samples
            output_path: Path to save the data
            attack_type: Type of attack ('poisoning', 'mimicry', or 'combined')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved {attack_type} attack data to: {output_path}")

def main():
    """Example usage of the AttackGenerator."""
    generator = AttackGenerator()
    
    # Load sample data
    try:
        with open("sample_benign.json", "r") as f:
            benign_data = json.load(f)
        with open("sample_malicious.json", "r") as f:
            malicious_data = json.load(f)
        
        # Generate different types of attacks
        poisoned_data = generator.generate_poisoning_attack(
            benign_data,
            malicious_data,
            attack_ratio=0.1
        )
        
        mimicry_data = generator.generate_mimicry_attack(
            benign_data,
            malicious_data,
            attack_ratio=0.1
        )
        
        poisoned_train, attacked_test = generator.generate_combined_attack(
            benign_data,
            malicious_data,
            attack_ratio=0.1
        )
        
        # Save generated attacks
        generator.save_attack_data(poisoned_data, "attacks/poisoning.json", "poisoning")
        generator.save_attack_data(mimicry_data, "attacks/mimicry.json", "mimicry")
        generator.save_attack_data(poisoned_train, "attacks/combined_train.json", "combined")
        generator.save_attack_data(attacked_test, "attacks/combined_test.json", "combined")
        
    except FileNotFoundError:
        print("Sample data files not found. Please provide valid data files.")

if __name__ == "__main__":
    main() 