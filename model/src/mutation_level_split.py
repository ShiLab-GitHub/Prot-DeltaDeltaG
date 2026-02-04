#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mutation-level train/test split
Mutation-level split: mutations from the same protein complex can appear in both training and test sets
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
import pickle
import os

class MutationLevelSplitter:
    """Class for mutation-level dataset splitting"""
    
    def __init__(self, ds_file, id2seq_file, n_splits=10, random_state=42):
        """
        Initialize the data splitter
        
        Args:
            ds_file: Path to ΔΔG data file
            id2seq_file: Path to protein sequence file
            n_splits: Number of cross-validation folds
            random_state: Random seed
        """
        self.ds_file = ds_file
        self.id2seq_file = id2seq_file
        self.n_splits = n_splits
        self.random_state = random_state
        
        # Initialize data storage
        self.raw_data = []
        self.raw_ids = []
        self.id2index = {}
        self.seqs = []
        
        # Load data
        self._load_sequences()
        self._load_mutation_data()
        
    def _load_sequences(self):
        """Load protein sequence data"""
        print("Loading protein sequence data...")
        
        index = 0
        for line in open(self.id2seq_file):
            line = line.strip().split('\t')
            if len(line) >= 2:
                self.id2index[line[0]] = index
                self.seqs.append(line[1])
                index += 1
        
        print(f"Loaded {len(self.seqs)} protein sequences")
        
    def _load_mutation_data(self):
        """Load mutation data"""
        print("Loading mutation data...")
        
        skip_head = True  # Skip header
        sid1_index = 0    # First protein sequence ID column index
        sid2_index = 1    # Second protein sequence ID column index
        sid3_index = 2    # Third protein sequence ID column index
        sid4_index = 3    # Fourth protein sequence ID column index
        
        for line in tqdm(open(self.ds_file)):
            if skip_head:
                skip_head = False
                continue
                
            line = line.rstrip('\n').rstrip('\r').replace('\t\t', '\t').split('\t')
            
            # Check if sequences exist
            if len(line) >= 4:
                seq1_exists = line[sid1_index] in self.id2index
                seq2_exists = line[sid2_index] in self.id2index
                seq3_exists = line[sid3_index] in self.id2index
                seq4_exists = line[sid4_index] in self.id2index
                
                # Keep data if all sequences exist
                if seq1_exists and seq2_exists and seq3_exists and seq4_exists:
                    self.raw_ids.append((line[sid1_index], line[sid2_index], 
                                       line[sid3_index], line[sid4_index]))
                    self.raw_data.append(line)
                else:
                    print(f"Skipping row with missing sequences: {line}")
        
        print(f"Loaded {len(self.raw_data)} mutation data entries")
    
    def create_mutation_level_splits(self):
        """
        Create mutation-level cross-validation splits
        
        Returns:
            list: List containing (train_indices, test_indices) tuples
        """
        print(f"Creating {self.n_splits}-fold cross-validation splits...")
        
        # Create data indices
        data_indices = np.arange(len(self.raw_data))
        
        # Use KFold for splitting
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        splits = []
        fold_num = 1
        
        for train_idx, test_idx in kf.split(data_indices):
            print(f"Fold {fold_num}: training set {len(train_idx)} samples, test set {len(test_idx)} samples")
            
            # Validate data integrity
            train_data_check = self._validate_split(train_idx)
            test_data_check = self._validate_split(test_idx)
            
            if train_data_check and test_data_check:
                splits.append((train_idx, test_idx))
                fold_num += 1
            else:
                print(f"Fold {fold_num} data validation failed, skipping...")
        
        print(f"Successfully created {len(splits)} valid data splits")
        return splits
    
    def _validate_split(self, indices):
        """
        Validate data split integrity
        
        Args:
            indices: Data index array
            
        Returns:
            bool: Whether validation passes
        """
        if len(indices) == 0:
            return False
            
        # Check if indices are within valid range
        if np.max(indices) >= len(self.raw_data) or np.min(indices) < 0:
            return False
            
        return True
    
    def get_split_data(self, train_indices, test_indices):
        """
        Get training and test set data based on indices
        
        Args:
            train_indices: Training set indices
            test_indices: Test set indices
            
        Returns:
            tuple: (train_data, train_ids, test_data, test_ids)
        """
        train_data = [self.raw_data[i] for i in train_indices]
        train_ids = [self.raw_ids[i] for i in train_indices]
        
        test_data = [self.raw_data[i] for i in test_indices]
        test_ids = [self.raw_ids[i] for i in test_indices]
        
        return train_data, train_ids, test_data, test_ids
    
    def save_splits(self, splits, output_dir):
        """
        Save data split results
        
        Args:
            splits: Data split results
            output_dir: Output directory
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save split indices
        with open(os.path.join(output_dir, 'mutation_level_splits.pkl'), 'wb') as f:
            pickle.dump(splits, f)
        
        # Save detailed information
        split_info = {
            'method': 'mutation_level',
            'n_splits': self.n_splits,
            'total_samples': len(self.raw_data),
            'random_state': self.random_state
        }
        
        with open(os.path.join(output_dir, 'split_info.pkl'), 'wb') as f:
            pickle.dump(split_info, f)
            
        print(f"Data split results saved to {output_dir}")
    
    def analyze_splits(self, splits):
        """
        Analyze statistics of data splits
        
        Args:
            splits: Data split results
        """
        print("\n=== Mutation-Level Split Statistics ===")
        print(f"Total samples: {len(self.raw_data)}")
        print(f"Cross-validation folds: {len(splits)}")
        
        train_sizes = []
        test_sizes = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            train_size = len(train_idx)
            test_size = len(test_idx)
            train_sizes.append(train_size)
            test_sizes.append(test_size)
            
            print(f"Fold {i+1}: training set {train_size} samples ({train_size/len(self.raw_data)*100:.1f}%), "
                  f"test set {test_size} samples ({test_size/len(self.raw_data)*100:.1f}%)")
        
        print(f"\nAverage training set size: {np.mean(train_sizes):.1f} ± {np.std(train_sizes):.1f}")
        print(f"Average test set size: {np.mean(test_sizes):.1f} ± {np.std(test_sizes):.1f}")


def main():
    """Main function example"""
    # Configure file paths
    ds_file = '../../data/binding_affinity/SKP1402m.single.ddg.txt'
    id2seq_file = '../../data/binding_affinity/SKP1402m.seq.txt'
    output_dir = 'splits/mutation_level'
    
    # Create data splitter
    splitter = MutationLevelSplitter(
        ds_file=ds_file,
        id2seq_file=id2seq_file,
        n_splits=10,
        random_state=42
    )
    
    # Create data splits
    splits = splitter.create_mutation_level_splits()
    
    # Analyze split results
    splitter.analyze_splits(splits)
    
    # Save results
    splitter.save_splits(splits, output_dir)
    
    # Example: get first fold data
    if len(splits) > 0:
        train_data, train_ids, test_data, test_ids = splitter.get_split_data(
            splits[0][0], splits[0][1]
        )
        print(f"\nFirst fold example:")
        print(f"Training set first 3 sample IDs: {train_ids[:3]}")
        print(f"Test set first 3 sample IDs: {test_ids[:3]}")


if __name__ == "__main__":
    main() 