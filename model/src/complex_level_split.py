#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complex-level train/test split
Complex-level split: ensures all mutations from the same protein complex stay together in either training or test set
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from tqdm import tqdm
import pickle
import os
from collections import defaultdict, Counter

class ComplexLevelSplitter:
    """Class for complex-level dataset splitting"""
    
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
        self.complex_groups = []  # Store complex group for each sample
        self.complex_to_samples = defaultdict(list)  # Mapping from complex to samples
        
        # Load data
        self._load_sequences()
        self._load_mutation_data()
        self._identify_complexes()
        
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
    
    def _identify_complexes(self):
        """Identify and group protein complexes"""
        print("Identifying protein complexes...")
        
        # Determine complex group for each sample
        for i, protein_ids in enumerate(self.raw_ids):
            # Method 1: Use first 4 characters of PDB ID as complex identifier
            complex_id = self._extract_complex_id(protein_ids)
            self.complex_groups.append(complex_id)
            self.complex_to_samples[complex_id].append(i)
        
        # Count complex information
        complex_counts = Counter(self.complex_groups)
        print(f"Identified {len(complex_counts)} different protein complexes")
        
        # Show complex distribution statistics
        print("\nComplex sample distribution statistics:")
        print(f"Complexes with most samples: {complex_counts.most_common(5)}")
        print(f"Complexes with least samples: {complex_counts.most_common()[-5:]}")
        
        # Check if there are enough complexes for cross-validation
        if len(complex_counts) < self.n_splits:
            print(f"Warning: Number of complexes ({len(complex_counts)}) is less than CV folds ({self.n_splits})")
            print("Consider reducing CV folds or using mutation-level split")
    
    def _extract_complex_id(self, protein_ids):
        """
        Extract complex identifier from protein IDs
        
        Args:
            protein_ids: Tuple of protein IDs
            
        Returns:
            str: Complex identifier
        """
        # Method 1: Use first 4 characters of the first protein ID (PDB ID standard)
        if len(protein_ids[0]) >= 4:
            return protein_ids[0][:4].upper()
        
        # Method 2: If ID format is non-standard, use complete first ID
        return protein_ids[0].upper()
    
    def create_complex_level_splits(self):
        """
        Create complex-level cross-validation splits
        
        Returns:
            list: List containing (train_indices, test_indices) tuples
        """
        print(f"Creating {self.n_splits}-fold complex-level cross-validation splits...")
        
        # Get unique complex list
        unique_complexes = list(set(self.complex_groups))
        complex_array = np.array(unique_complexes)
        
        print(f"Number of complexes participating in splits: {len(unique_complexes)}")
        
        # Use KFold to ensure all samples from same complex stay in same group
        if len(unique_complexes) >= self.n_splits:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = []
            fold_num = 1
            
            for train_complex_idx, test_complex_idx in kf.split(complex_array):
                train_complexes = complex_array[train_complex_idx]
                test_complexes = complex_array[test_complex_idx]
                
                # Convert complex indices to sample indices
                train_indices = []
                test_indices = []
                
                for complex_id in train_complexes:
                    train_indices.extend(self.complex_to_samples[complex_id])
                
                for complex_id in test_complexes:
                    test_indices.extend(self.complex_to_samples[complex_id])
                
                train_indices = np.array(sorted(train_indices))
                test_indices = np.array(sorted(test_indices))
                
                print(f"Fold {fold_num}: training set {len(train_indices)} samples ({len(train_complexes)} complexes), "
                      f"test set {len(test_indices)} samples ({len(test_complexes)} complexes)")
                
                # Validate data integrity
                if self._validate_complex_split(train_indices, test_indices):
                    splits.append((train_indices, test_indices))
                    fold_num += 1
                else:
                    print(f"Fold {fold_num} data validation failed, skipping...")
        
        else:
            print(f"Error: Number of complexes ({len(unique_complexes)}) insufficient for {self.n_splits}-fold CV")
            print("Consider reducing CV folds")
            return []
        
        print(f"Successfully created {len(splits)} valid complex-level data splits")
        return splits
    
    def _validate_complex_split(self, train_indices, test_indices):
        """
        Validate complex-level data split integrity
        
        Args:
            train_indices: Training set indices
            test_indices: Test set indices
            
        Returns:
            bool: Whether validation passes
        """
        # Basic validation
        if len(train_indices) == 0 or len(test_indices) == 0:
            return False
        
        # Check index range
        all_indices = np.concatenate([train_indices, test_indices])
        if np.max(all_indices) >= len(self.raw_data) or np.min(all_indices) < 0:
            return False
        
        # Check for overlaps
        if len(np.intersect1d(train_indices, test_indices)) > 0:
            print("Error: Training and test sets have overlapping samples")
            return False
        
        # Validate complex-level separation
        train_complexes = set([self.complex_groups[i] for i in train_indices])
        test_complexes = set([self.complex_groups[i] for i in test_indices])
        
        if len(train_complexes.intersection(test_complexes)) > 0:
            print("Error: Training and test sets share the same complexes")
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
        with open(os.path.join(output_dir, 'complex_level_splits.pkl'), 'wb') as f:
            pickle.dump(splits, f)
        
        # Save detailed information
        split_info = {
            'method': 'complex_level',
            'n_splits': self.n_splits,
            'total_samples': len(self.raw_data),
            'total_complexes': len(set(self.complex_groups)),
            'random_state': self.random_state,
            'complex_groups': self.complex_groups
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
        print("\n=== Complex-Level Split Statistics ===")
        print(f"Total samples: {len(self.raw_data)}")
        print(f"Total complexes: {len(set(self.complex_groups))}")
        print(f"Cross-validation folds: {len(splits)}")
        
        train_sizes = []
        test_sizes = []
        train_complex_counts = []
        test_complex_counts = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            train_size = len(train_idx)
            test_size = len(test_idx)
            train_sizes.append(train_size)
            test_sizes.append(test_size)
            
            # Count complexes
            train_complexes = set([self.complex_groups[j] for j in train_idx])
            test_complexes = set([self.complex_groups[j] for j in test_idx])
            train_complex_counts.append(len(train_complexes))
            test_complex_counts.append(len(test_complexes))
            
            print(f"Fold {i+1}: training set {train_size} samples ({len(train_complexes)} complexes), "
                  f"test set {test_size} samples ({len(test_complexes)} complexes)")
        
        print(f"\nAverage training set size: {np.mean(train_sizes):.1f} ± {np.std(train_sizes):.1f}")
        print(f"Average test set size: {np.mean(test_sizes):.1f} ± {np.std(test_sizes):.1f}")
        print(f"Average training set complexes: {np.mean(train_complex_counts):.1f} ± {np.std(train_complex_counts):.1f}")
        print(f"Average test set complexes: {np.mean(test_complex_counts):.1f} ± {np.std(test_complex_counts):.1f}")
    
    def get_complex_statistics(self):
        """
        Get complex statistics
        
        Returns:
            dict: Complex statistics
        """
        complex_counts = Counter(self.complex_groups)
        
        stats = {
            'total_complexes': len(complex_counts),
            'total_samples': len(self.raw_data),
            'avg_samples_per_complex': np.mean(list(complex_counts.values())),
            'std_samples_per_complex': np.std(list(complex_counts.values())),
            'max_samples_per_complex': max(complex_counts.values()),
            'min_samples_per_complex': min(complex_counts.values()),
            'complex_distribution': complex_counts
        }
        
        return stats


def main():
    """Main function example"""
    # Configure file paths
    ds_file = '../../data/binding_affinity/SKP1402m.single.ddg.txt'
    id2seq_file = '../../data/binding_affinity/SKP1402m.seq.txt'
    output_dir = 'splits/complex_level'
    
    # Create data splitter
    splitter = ComplexLevelSplitter(
        ds_file=ds_file,
        id2seq_file=id2seq_file,
        n_splits=5,  # Complex-level split usually requires fewer folds
        random_state=42
    )
    
    # Get complex statistics
    stats = splitter.get_complex_statistics()
    print(f"\nComplex statistics:")
    print(f"Total complexes: {stats['total_complexes']}")
    print(f"Average samples per complex: {stats['avg_samples_per_complex']:.1f} ± {stats['std_samples_per_complex']:.1f}")
    print(f"Sample range: {stats['min_samples_per_complex']} - {stats['max_samples_per_complex']}")
    
    # Create data splits
    splits = splitter.create_complex_level_splits()
    
    if len(splits) > 0:
        # Analyze split results
        splitter.analyze_splits(splits)
        
        # Save results
        splitter.save_splits(splits, output_dir)
        
        # Example: get first fold data
        train_data, train_ids, test_data, test_ids = splitter.get_split_data(
            splits[0][0], splits[0][1]
        )
        print(f"\nFirst fold example:")
        print(f"Training set first 3 sample IDs: {train_ids[:3]}")
        print(f"Test set first 3 sample IDs: {test_ids[:3]}")
        
        # Validate complex separation
        train_complexes = set([splitter.complex_groups[i] for i in splits[0][0]])
        test_complexes = set([splitter.complex_groups[i] for i in splits[0][1]])
        print(f"Training set complex examples: {list(train_complexes)[:5]}")
        print(f"Test set complex examples: {list(test_complexes)[:5]}")
        print(f"Complex separation validation: {len(train_complexes.intersection(test_complexes)) == 0}")
    
    else:
        print("Complex-level split failed, please check data or adjust parameters")


if __name__ == "__main__":
    main() 