import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """A tool for preprocessing experiment data."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
        self.scalers = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("DataPreprocessor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("data_preprocessing.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        self.logger.info(f"Loaded data from: {file_path}")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values and outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle outliers using IQR method
        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = df[column].clip(lower_bound, upper_bound)
        
        self.logger.info("Data cleaning completed")
        return df
    
    def normalize_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Normalize numerical features.
        
        Args:
            df: Input DataFrame
            method: Normalization method ('standard' or 'minmax')
            
        Returns:
            Normalized DataFrame
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unsupported normalization method: {method}")
            
            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler
        
        self.logger.info(f"Feature normalization completed using {method} method")
        return df
    
    def split_data(self, df: pd.DataFrame, 
                  target_col: str,
                  test_size: float = 0.2,
                  val_size: float = 0.1,
                  random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            test_size: Test set size ratio
            val_size: Validation set size ratio
            random_state: Random seed
            
        Returns:
            Dictionary containing train, validation, and test sets
        """
        # First split: separate test set
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        
        # Second split: separate validation set from training set
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_size_adjusted, random_state=random_state
        )
        
        self.logger.info(f"Data split completed: train={len(train)}, val={len(val)}, test={len(test)}")
        return {
            'train': train,
            'val': val,
            'test': test
        }
    
    def save_processed_data(self, data_dict: Dict[str, pd.DataFrame], 
                          output_dir: str) -> None:
        """
        Save processed data to files.
        
        Args:
            data_dict: Dictionary containing processed DataFrames
            output_dir: Directory to save processed data
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for name, df in data_dict.items():
            output_path = output_dir / f"{name}.csv"
            df.to_csv(output_path, index=False)
            self.logger.info(f"Saved {name} data to: {output_path}")
    
    def generate_preprocessing_report(self, df: pd.DataFrame) -> str:
        """
        Generate a report of the preprocessing steps and data statistics.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Markdown formatted report
        """
        report = ["# Data Preprocessing Report\n"]
        
        # Basic statistics
        report.append("## Basic Statistics\n")
        report.append("### Shape")
        report.append(f"- Rows: {df.shape[0]}")
        report.append(f"- Columns: {df.shape[1]}\n")
        
        # Column information
        report.append("### Column Information")
        for col in df.columns:
            report.append(f"\n#### {col}")
            report.append(f"- Type: {df[col].dtype}")
            if df[col].dtype in [np.number]:
                report.append(f"- Mean: {df[col].mean():.4f}")
                report.append(f"- Std: {df[col].std():.4f}")
                report.append(f"- Min: {df[col].min():.4f}")
                report.append(f"- Max: {df[col].max():.4f}")
            else:
                report.append(f"- Unique values: {df[col].nunique()}")
        
        return "\n".join(report)

def main():
    """Example usage of the DataPreprocessor."""
    preprocessor = DataPreprocessor()
    
    # Load and process sample data
    try:
        df = preprocessor.load_data("sample_data.csv")
        df = preprocessor.clean_data(df)
        df = preprocessor.normalize_features(df, method='standard')
        
        # Split data
        data_splits = preprocessor.split_data(
            df, 
            target_col='label',
            test_size=0.2,
            val_size=0.1
        )
        
        # Save processed data
        preprocessor.save_processed_data(data_splits, "processed_data")
        
        # Generate report
        report = preprocessor.generate_preprocessing_report(df)
        with open("preprocessing_report.md", "w") as f:
            f.write(report)
            
    except FileNotFoundError:
        print("Sample data file not found. Please provide a valid data file.")

if __name__ == "__main__":
    main() 