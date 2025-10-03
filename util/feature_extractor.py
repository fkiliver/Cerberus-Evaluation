import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import networkx as nx
from datetime import datetime

class FeatureExtractor:
    """A tool for extracting features from system traces for Cerberus framework."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the feature extractor.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=1000)
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("FeatureExtractor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("feature_extraction.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def extract_sequence_features(self, 
                                data: List[Dict],
                                window_size: int = 5) -> np.ndarray:
        """
        Extract sequence-based features from system traces.
        
        Args:
            data: List of system trace samples
            window_size: Size of the sliding window for sequence analysis
            
        Returns:
            Array of sequence features
        """
        features = []
        
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            
            # Extract basic sequence features
            seq_features = {
                'avg_duration': np.mean([s.get('duration', 0) for s in window]),
                'std_duration': np.std([s.get('duration', 0) for s in window]),
                'unique_processes': len(set(s.get('process_name', '') for s in window)),
                'unique_users': len(set(s.get('user_id', '') for s in window))
            }
            
            # Extract temporal features
            timestamps = [datetime.fromisoformat(s.get('timestamp', '')) for s in window]
            seq_features['time_gaps'] = np.mean([
                (timestamps[j+1] - timestamps[j]).total_seconds()
                for j in range(len(timestamps)-1)
            ])
            
            features.append(list(seq_features.values()))
        
        return np.array(features)
    
    def extract_graph_features(self, 
                             data: List[Dict]) -> np.ndarray:
        """
        Extract graph-based features from system traces.
        
        Args:
            data: List of system trace samples
            
        Returns:
            Array of graph features
        """
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for i, sample in enumerate(data):
            G.add_node(i, **sample)
            if 'parent_id' in sample:
                G.add_edge(sample['parent_id'], i)
        
        features = []
        for node in G.nodes():
            node_features = {
                'in_degree': G.in_degree(node),
                'out_degree': G.out_degree(node),
                'clustering_coef': nx.clustering(G, node),
                'pagerank': nx.pagerank(G)[node],
                'betweenness': nx.betweenness_centrality(G)[node]
            }
            features.append(list(node_features.values()))
        
        return np.array(features)
    
    def extract_text_features(self, 
                            data: List[Dict]) -> np.ndarray:
        """
        Extract text-based features from system traces.
        
        Args:
            data: List of system trace samples
            
        Returns:
            Array of text features
        """
        # Combine relevant text fields
        texts = []
        for sample in data:
            text = ' '.join([
                str(sample.get('process_name', '')),
                str(sample.get('command', '')),
                str(sample.get('parameters', ''))
            ])
            texts.append(text)
        
        # Extract TF-IDF features
        return self.tfidf.fit_transform(texts).toarray()
    
    def extract_behavior_features(self, 
                                data: List[Dict]) -> np.ndarray:
        """
        Extract behavior-based features from system traces.
        
        Args:
            data: List of system trace samples
            
        Returns:
            Array of behavior features
        """
        features = []
        
        for sample in data:
            behavior_features = {
                'has_network': int('network' in str(sample.get('type', '')).lower()),
                'has_file_io': int('file' in str(sample.get('type', '')).lower()),
                'has_process': int('process' in str(sample.get('type', '')).lower()),
                'has_registry': int('registry' in str(sample.get('type', '')).lower()),
                'is_system': int(sample.get('user_id', '') == 'SYSTEM'),
                'is_admin': int('admin' in str(sample.get('user_id', '')).lower()),
                'is_remote': int('remote' in str(sample.get('source', '')).lower())
            }
            features.append(list(behavior_features.values()))
        
        return np.array(features)
    
    def extract_all_features(self,
                           data: List[Dict],
                           normalize: bool = True) -> Dict[str, np.ndarray]:
        """
        Extract all types of features from system traces.
        
        Args:
            data: List of system trace samples
            normalize: Whether to normalize the features
            
        Returns:
            Dictionary containing different types of features
        """
        # Extract different types of features
        sequence_features = self.extract_sequence_features(data)
        graph_features = self.extract_graph_features(data)
        text_features = self.extract_text_features(data)
        behavior_features = self.extract_behavior_features(data)
        
        # Normalize features if requested
        if normalize:
            sequence_features = self.scaler.fit_transform(sequence_features)
            graph_features = self.scaler.fit_transform(graph_features)
            text_features = self.scaler.fit_transform(text_features)
            behavior_features = self.scaler.fit_transform(behavior_features)
        
        return {
            'sequence': sequence_features,
            'graph': graph_features,
            'text': text_features,
            'behavior': behavior_features
        }
    
    def save_features(self,
                     features: Dict[str, np.ndarray],
                     output_dir: str) -> None:
        """
        Save extracted features to files.
        
        Args:
            features: Dictionary of feature arrays
            output_dir: Directory to save the features
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for feature_type, feature_array in features.items():
            output_path = output_dir / f"{feature_type}_features.npy"
            np.save(output_path, feature_array)
            self.logger.info(f"Saved {feature_type} features to: {output_path}")

def main():
    """Example usage of the FeatureExtractor."""
    extractor = FeatureExtractor()
    
    # Load sample data
    try:
        with open("sample_traces.json", "r") as f:
            data = json.load(f)
        
        # Extract all features
        features = extractor.extract_all_features(data, normalize=True)
        
        # Save features
        extractor.save_features(features, "features")
        
    except FileNotFoundError:
        print("Sample data file not found. Please provide valid data file.")

if __name__ == "__main__":
    main() 