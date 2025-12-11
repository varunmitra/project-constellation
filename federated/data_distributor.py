#!/usr/bin/env python3
"""
Data Distribution Utility
Splits datasets across multiple devices for federated learning
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDistributor:
    """Distributes datasets across multiple devices for federated learning"""
    
    def __init__(self, output_dir: str = "federated_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def distribute_dataset(
        self, 
        dataset_path: str, 
        device_ids: List[str], 
        distribution_strategy: str = "random",
        test_split: float = 0.2
    ) -> Dict[str, str]:
        """
        Distribute dataset across devices
        
        Args:
            dataset_path: Path to the dataset CSV file
            device_ids: List of device IDs to distribute to
            distribution_strategy: Strategy for distribution (random, stratified, iid)
            test_split: Fraction of data to use for testing
        
        Returns:
            Dictionary mapping device_id to data_path
        """
        
        logger.info(f"📊 Distributing dataset: {dataset_path}")
        logger.info(f"🎯 Devices: {len(device_ids)}")
        logger.info(f"📈 Strategy: {distribution_strategy}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        logger.info(f"📊 Total samples: {len(df)}")
        
        # Split into train and test
        train_df, test_df = self._split_train_test(df, test_split)
        
        # Distribute training data
        device_data_paths = {}
        
        if distribution_strategy == "random":
            device_data_paths = self._distribute_random(train_df, device_ids)
        elif distribution_strategy == "stratified":
            device_data_paths = self._distribute_stratified(train_df, device_ids)
        elif distribution_strategy == "iid":
            device_data_paths = self._distribute_iid(train_df, device_ids)
        else:
            raise ValueError(f"Unknown distribution strategy: {distribution_strategy}")
        
        # Save test data
        test_path = self.output_dir / "test_data.csv"
        test_df.to_csv(test_path, index=False)
        logger.info(f"📊 Test data saved: {test_path}")
        
        return device_data_paths
    
    def _split_train_test(self, df: pd.DataFrame, test_split: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into train and test sets"""
        from sklearn.model_selection import train_test_split
        
        # Assuming the first column is the label
        X = df.iloc[:, 1:]  # Features (text columns)
        y = df.iloc[:, 0]   # Labels
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42, stratify=y
        )
        
        # Reconstruct dataframes
        train_df = pd.concat([y_train, X_train], axis=1)
        test_df = pd.concat([y_test, X_test], axis=1)
        
        # Reset indices
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        
        logger.info(f"📊 Train samples: {len(train_df)}")
        logger.info(f"📊 Test samples: {len(test_df)}")
        
        return train_df, test_df
    
    def _distribute_random(self, df: pd.DataFrame, device_ids: List[str]) -> Dict[str, str]:
        """Distribute data randomly across devices"""
        # Shuffle the dataframe
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate chunk size
        chunk_size = len(df_shuffled) // len(device_ids)
        
        device_data_paths = {}
        
        for i, device_id in enumerate(device_ids):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < len(device_ids) - 1 else len(df_shuffled)
            
            device_data = df_shuffled.iloc[start_idx:end_idx]
            device_path = self.output_dir / f"{device_id}_data.csv"
            device_data.to_csv(device_path, index=False)
            
            device_data_paths[device_id] = str(device_path)
            
            logger.info(f"📱 Device {device_id}: {len(device_data)} samples")
        
        return device_data_paths
    
    def _distribute_stratified(self, df: pd.DataFrame, device_ids: List[str]) -> Dict[str, str]:
        """Distribute data maintaining class distribution across devices"""
        from sklearn.model_selection import train_test_split
        
        # Get unique labels
        labels = df.iloc[:, 0].unique()
        logger.info(f"📊 Classes: {labels}")
        
        device_data_paths = {}
        device_dfs = {device_id: pd.DataFrame() for device_id in device_ids}
        
        # Distribute each class across devices
        for label in labels:
            class_data = df[df.iloc[:, 0] == label]
            
            # Split class data across devices
            class_chunks = np.array_split(class_data, len(device_ids))
            
            for i, device_id in enumerate(device_ids):
                if i < len(class_chunks):
                    device_dfs[device_id] = pd.concat([device_dfs[device_id], class_chunks[i]], ignore_index=True)
        
        # Save device data
        for device_id, device_df in device_dfs.items():
            if not device_df.empty:
                device_path = self.output_dir / f"{device_id}_data.csv"
                device_df.to_csv(device_path, index=False)
                device_data_paths[device_id] = str(device_path)
                
                # Log class distribution
                class_counts = device_df.iloc[:, 0].value_counts().to_dict()
                logger.info(f"📱 Device {device_id}: {len(device_df)} samples, classes: {class_counts}")
        
        return device_data_paths
    
    def _distribute_iid(self, df: pd.DataFrame, device_ids: List[str]) -> Dict[str, str]:
        """Distribute data in IID (Independent and Identically Distributed) manner"""
        # This is similar to random distribution but ensures each device gets
        # a representative sample of the overall distribution
        
        # Calculate samples per device
        samples_per_device = len(df) // len(device_ids)
        
        device_data_paths = {}
        
        for i, device_id in enumerate(device_ids):
            # Sample with replacement to ensure IID
            device_data = df.sample(n=samples_per_device, replace=True, random_state=42+i)
            device_path = self.output_dir / f"{device_id}_data.csv"
            device_data.to_csv(device_path, index=False)
            
            device_data_paths[device_id] = str(device_path)
            
            # Log class distribution
            class_counts = device_data.iloc[:, 0].value_counts().to_dict()
            logger.info(f"📱 Device {device_id}: {len(device_data)} samples, classes: {class_counts}")
        
        return device_data_paths
    
    def create_synthetic_devices(self, num_devices: int, prefix: str = "device") -> List[str]:
        """Create synthetic device IDs for testing"""
        return [f"{prefix}_{i:03d}" for i in range(num_devices)]
    
    def analyze_distribution(self, device_data_paths: Dict[str, str]) -> Dict[str, Dict]:
        """Analyze the distribution of data across devices"""
        analysis = {}
        
        for device_id, data_path in device_data_paths.items():
            df = pd.read_csv(data_path)
            
            # Basic statistics
            total_samples = len(df)
            class_counts = df.iloc[:, 0].value_counts().to_dict()
            class_distribution = {k: v/total_samples for k, v in class_counts.items()}
            
            analysis[device_id] = {
                "total_samples": total_samples,
                "class_counts": class_counts,
                "class_distribution": class_distribution,
                "data_path": data_path
            }
        
        return analysis

# Example usage
def main():
    distributor = DataDistributor()
    
    # Create synthetic devices
    device_ids = distributor.create_synthetic_devices(4, "fed_device")
    
    # Distribute AG News dataset
    dataset_path = "training/data/ag_news_train.csv"
    
    if Path(dataset_path).exists():
        device_paths = distributor.distribute_dataset(
            dataset_path=dataset_path,
            device_ids=device_ids,
            distribution_strategy="stratified",
            test_split=0.2
        )
        
        # Analyze distribution
        analysis = distributor.analyze_distribution(device_paths)
        
        print("\n📊 Distribution Analysis:")
        for device_id, stats in analysis.items():
            print(f"\n{device_id}:")
            print(f"  Samples: {stats['total_samples']}")
            print(f"  Classes: {stats['class_counts']}")
            print(f"  Distribution: {stats['class_distribution']}")
    else:
        print(f"Dataset not found: {dataset_path}")

if __name__ == "__main__":
    main()
