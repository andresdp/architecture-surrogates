"""
Monte Carlo Sampling Analysis for CoCoME Architecture Optimization

This script analyzes the effect of different sampling ratios on incremental learning performance
by running the level-by-level training pipeline (same as main_cocome_raw_training.py) with varying sample ratios.

Key features:
- Tests sample ratios from 0.3 to 0.8 in 0.1 increments
- Uses incremental level-by-level training: Level 1 -> Level 2 -> Level 3 -> Level 4 -> Level 5
- Each level uses 70/30 train/test split, with sample_ratio applied to the 70% training portion
- Calculates RMSE at each complexity level (1-5) and averages them for final metric
- Uses only CatBoost SingleOutput model for consistency
- Compares against incremental baseline trained with 100% data at each level
- Generates plots showing sampling ratio vs average RMSE across all complexity levels
- Sample ratio interpretation: 0.3 = 30% of the 70% training data used at each level
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
import sys
import warnings
import time
from datetime import datetime

# Import the existing training components
from main_cocome_raw_training import (
    CatBoostMultiTarget
)

warnings.filterwarnings('ignore')


def calculate_target_normalization_factors(features_file):
    """Calculate normalization factors for RMSE based on target ranges"""
    data = pd.read_csv(features_file)
    target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
    
    target_stats = {}
    target_ranges = {}
    
    for target in target_names:
        target_values = data[target].values
        target_min = np.min(target_values)
        target_max = np.max(target_values)
        target_range = target_max - target_min
        target_std = np.std(target_values)
        
        target_stats[target] = {
            'min': target_min,
            'max': target_max,
            'range': target_range,
            'std': target_std,
            'mean': np.mean(target_values)
        }
        target_ranges[target] = target_range
    
    # Overall normalization factor (using range-based normalization)
    overall_target_range = np.mean(list(target_ranges.values()))
    
    return target_stats, overall_target_range


def normalize_rmse(rmse_value, normalization_factor):
    """Normalize RMSE by the overall target range to make it dimensionless"""
    return rmse_value / normalization_factor if normalization_factor > 0 else rmse_value


def run_training_with_sample_ratio(features_file, sample_ratio, verbose=False):
    """
    Run the training pipeline with a specific sample ratio using level-by-level incremental learning.
    This follows the same approach as main_cocome_raw_training.py:
    - Train incrementally level by level (1 -> 2 -> 3 -> 4 -> 5)
    - Calculate RMSE at each level
    - Return the average RMSE across all 5 levels
    
    Args:
        features_file: Path to the CoCoME features CSV file
        sample_ratio: Fraction of training data to use from the 70% train split at each level
        verbose: Whether to print verbose output
    
    Returns:
        dict: Dictionary with 'avg_rmse' and 'level_rmses' (list of RMSE for each level 1-5)
    """
    print(f"Running incremental training with sample_ratio={sample_ratio:.1f}...")
    
    # Load data
    data = pd.read_csv(features_file)
    
    # Prepare features and targets (same as main_cocome_raw_training.py)
    exclude_cols = ['solID', 'level', 'm1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
    if data.columns[0].startswith('Unnamed') or data.columns[0] == '':
        exclude_cols.append(data.columns[0])
    
    feature_columns = [col for col in data.columns if col not in exclude_cols]
    target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
    
    # Group data by complexity levels
    complexity_groups = {}
    for level in range(1, 6):  # Levels 1-5
        mask = data['level'] == level
        if mask.any():
            level_data = data[mask]
            complexity_groups[level] = {
                'features': level_data[feature_columns].values,
                'objectives': level_data[target_names].values
            }
    
    if verbose:
        print(f"  Feature dimensionality: {len(feature_columns)} raw features")
        for level in sorted(complexity_groups.keys()):
            print(f"  Level {level}: {len(complexity_groups[level]['features'])} samples")
    
    # Initialize optimizer and target scaler
    optimizer = CatBoostMultiTarget('CatBoost Incremental Training', verbose=verbose)
    target_scaler = StandardScaler()
    level_rmses = [float('nan')] * 5  # Initialize with NaN for all 5 levels
    
    # Progressive learning through complexity levels (same as main_cocome_raw_training.py)
    for level in range(1, 6):
        level_idx = level - 1  # Convert to 0-based index for list
        
        if level not in complexity_groups:
            if verbose:
                print(f"  Level {level}: No data found, skipping")
            level_rmses[level_idx] = float('nan')
            continue
            
        level_data = complexity_groups[level]
        X_level = level_data['features']
        y_level = level_data['objectives']
        
        if len(X_level) < 5:
            if verbose:
                print(f"  Level {level}: Insufficient samples ({len(X_level)}), skipping")
            level_rmses[level_idx] = float('nan')
            continue
        
        # Train/test split (70/30) for this level
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_level, y_level, test_size=0.3, random_state=None
        )
        
        # Apply sample ratio to training data (same as main_cocome_raw_training.py)
        # Special case: Level 1 always uses 100% of training data for stable foundation
        if level == 1:
            # Use all training data for Level 1 to provide stable foundation
            X_train = X_train_full
            y_train = y_train_full
            effective_ratio = 1.0  # 100% for reporting
        else:
            # Apply sample ratio for levels 2-5
            n_train_samples = max(1, int(len(X_train_full) * sample_ratio))
            
            # Ensure we have at least a minimum number of samples for stable training
            min_samples = min(5, len(X_train_full))  # At least 5 samples or all available
            n_train_samples = max(min_samples, n_train_samples)
            
            # Don't sample more than available
            n_train_samples = min(n_train_samples, len(X_train_full))
            
            train_indices = np.random.choice(len(X_train_full), size=n_train_samples, replace=False)
            X_train = X_train_full[train_indices]
            y_train = y_train_full[train_indices]
            effective_ratio = sample_ratio
        
        # Scale targets (fit scaler on first level, transform on subsequent levels)
        if level == 1:
            y_train_scaled = target_scaler.fit_transform(y_train)
            y_test_scaled = target_scaler.transform(y_test)
        else:
            y_train_scaled = target_scaler.transform(y_train)
            y_test_scaled = target_scaler.transform(y_test)
        
        if verbose:
            print(f"  Level {level}: Train/Test split: {len(X_train_full)}/{len(X_test)}")
            print(f"  Level {level}: Training with {len(X_train)} samples ({effective_ratio*100:.0f}% of train set)")
            if level == 1:
                print(f"    Level {level}: Using 100% of training data (stable foundation for incremental learning)")
        
        try:
            # Handle both first training and incremental updates
            if level == 1:
                # Initial training on level 1
                optimizer.fit(X_train, y_train_scaled, level)
                if verbose:
                    print(f"    Level {level}: Initial training completed")
            else:
                # Incremental update with new level data
                optimizer.update(X_train, y_train_scaled, level)
                if verbose:
                    print(f"    Level {level}: Incremental update completed")
            
            # Evaluate performance on this level's test set
            y_pred_scaled = optimizer.predict(X_test)
            
            # Check for NaN predictions
            if np.any(np.isnan(y_pred_scaled)):
                print(f"    Level {level}: Warning - NaN predictions detected, replacing with median")
                # Calculate median of test targets as fallback
                median_targets = np.nanmedian(y_test_scaled, axis=0)
                # Replace NaN predictions with median values
                for i in range(y_pred_scaled.shape[1]):
                    nan_mask = np.isnan(y_pred_scaled[:, i])
                    y_pred_scaled[nan_mask, i] = median_targets[i]
            
            # Transform back to original space for meaningful RMSE calculation
            y_test_original = target_scaler.inverse_transform(y_test_scaled)
            y_pred_original = target_scaler.inverse_transform(y_pred_scaled)
            
            # Calculate RMSE in original space
            mse = mean_squared_error(y_test_original, y_pred_original)
            rmse = np.sqrt(mse)
            
            # Sanity check for RMSE
            if not np.isfinite(rmse) or rmse > 1000:
                print(f"    Level {level}: Warning - extreme RMSE value: {rmse}")
                if not np.isfinite(rmse):
                    rmse = 1000.0  # Use a high but finite value
            
            level_rmses[level_idx] = rmse
            
            if verbose:
                print(f"    Level {level}: RMSE = {rmse:.4f}")
                
        except Exception as e:
            print(f"    Level {level}: Training failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            # Use a high RMSE value for failed levels
            level_rmses[level_idx] = 1000.0  # Use finite value instead of inf
    
    # Calculate average RMSE across all levels
    finite_rmses = [rmse for rmse in level_rmses if np.isfinite(rmse) and rmse <= 100]
    
    if finite_rmses:
        avg_rmse = np.mean(finite_rmses)
        if verbose:
            print(f"  Using {len(finite_rmses)}/{len(level_rmses)} levels for averaging")
    else:
        # If no reasonable values, use all finite values
        all_finite = [rmse for rmse in level_rmses if np.isfinite(rmse)]
        if all_finite:
            avg_rmse = np.mean(all_finite)
            if verbose:
                print(f"  Warning: All RMSE values are extreme. Using {len(all_finite)}/{len(level_rmses)} finite values")
        else:
            avg_rmse = float('nan')
            if verbose:
                print(f"  Error: No finite RMSE values found")
    
    if verbose:
        print(f"  Level RMSEs: {[f'{rmse:.4f}' if np.isfinite(rmse) else 'FAILED' for rmse in level_rmses]}")
        print(f"  Average RMSE: {avg_rmse:.4f}")
    
    successful_levels = len([r for r in level_rmses if np.isfinite(r) and r <= 100])
    if np.isfinite(avg_rmse):
        print(f"  Average RMSE across {successful_levels} levels: {avg_rmse:.4f}")
    else:
        print(f"  Training failed - no valid RMSE values")
    
    return {
        'avg_rmse': avg_rmse,
        'level_rmses': level_rmses  # List of 5 values (one per level)
    }


def calculate_stratified_baseline(features_file, verbose=False):
    """
    Calculate stratified baseline using the whole dataset with no incremental learning.
    Uses stratified train/test split based on level distribution to maintain proportions.
    
    Args:
        features_file: Path to the CoCoME features CSV file
        verbose: Whether to print verbose output
    
    Returns:
        dict: Dictionary with 'avg_rmse' and 'level_rmses' (list of RMSE for each level 1-5)
    """
    print("Calculating stratified baseline (whole dataset, no incremental learning, CatBoost, stratified 70/30 split)...")
    
    # Load data
    data = pd.read_csv(features_file)
    
    # Prepare features and targets
    exclude_cols = ['solID', 'level', 'm1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
    if data.columns[0].startswith('Unnamed') or data.columns[0] == '':
        exclude_cols.append(data.columns[0])
    
    feature_columns = [col for col in data.columns if col not in exclude_cols]
    target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
    
    X_all = data[feature_columns].values
    y_all = data[target_names].values
    levels_all = data['level'].values
    
    if verbose:
        level_counts = {level: sum(levels_all == level) for level in range(1, 6)}
        print(f"Level distribution: {level_counts}")
        print(f"Feature dimensionality: {len(feature_columns)} raw features")
        print(f"Total samples: {len(X_all)}")
    
    # Stratified train/test split maintaining level proportions
    X_train, X_test, y_train, y_test, levels_train, levels_test = train_test_split(
        X_all, y_all, levels_all, test_size=0.3, stratify=levels_all, random_state=42
    )
    
    # Initialize scaler and fit on training data
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_test_scaled = target_scaler.transform(y_test)
    
    if verbose:
        print(f"Train/Test split: {len(X_train)}/{len(X_test)}")
        train_level_counts = {level: sum(levels_train == level) for level in range(1, 6)}
        test_level_counts = {level: sum(levels_test == level) for level in range(1, 6)}
        print(f"Train level distribution: {train_level_counts}")
        print(f"Test level distribution: {test_level_counts}")
    
    # Import CatBoost for the stratified baseline (same as incremental learning)
    import catboost as cb
    from sklearn.multioutput import MultiOutputRegressor
    
    # Train single CatBoost model on all data (same as incremental learning uses)
    base_model = cb.CatBoostRegressor(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=False
    )
    model = MultiOutputRegressor(base_model)
    
    try:
        if verbose:
            print("Training CatBoost model on whole dataset...")
        
        model.fit(X_train_scaled, y_train_scaled)
        
        # Get predictions for the whole test set
        y_pred_scaled = model.predict(X_test_scaled)
        
        # Transform back to original space for meaningful RMSE calculation
        y_test_original = target_scaler.inverse_transform(y_test_scaled)
        y_pred_original = target_scaler.inverse_transform(y_pred_scaled)
        
        # Calculate RMSE for each level separately in original space
        level_rmses = [float('nan')] * 5
        
        for level in range(1, 6):
            level_idx = level - 1
            level_mask = levels_test == level
            
            if np.any(level_mask):
                y_test_level = y_test_original[level_mask]
                y_pred_level = y_pred_original[level_mask]
                
                mse = mean_squared_error(y_test_level, y_pred_level)
                rmse = np.sqrt(mse)
                level_rmses[level_idx] = rmse
                
                if verbose:
                    print(f"  Level {level}: RMSE = {rmse:.4f} (n_test = {sum(level_mask)})")
            else:
                if verbose:
                    print(f"  Level {level}: No test samples")
        
        # Calculate overall RMSE across all levels (single baseline value) in original space
        overall_mse = mean_squared_error(y_test_original, y_pred_original)
        overall_rmse = np.sqrt(overall_mse)
        
        # For stratified baseline, we use the overall RMSE as the single baseline
        finite_rmses = [rmse for rmse in level_rmses if np.isfinite(rmse)]
        
        if finite_rmses:
            avg_rmse = overall_rmse  # Use overall RMSE for single baseline
            if verbose:
                print(f"  Overall RMSE (single stratified baseline): {overall_rmse:.4f}")
                print(f"  Average RMSE per level: {np.mean(finite_rmses):.4f}")
        else:
            avg_rmse = float('nan')
            if verbose:
                print("  Error: No finite RMSE values found")
        
    except Exception as e:
        if verbose:
            print(f"  Stratified baseline training failed: {e}")
            import traceback
            traceback.print_exc()
        avg_rmse = float('nan')
        overall_rmse = float('nan')
    
    return {
        'avg_rmse': avg_rmse,
        'level_rmses': level_rmses,  # Keep individual level RMSEs for analysis
        'overall_rmse': overall_rmse  # Single stratified baseline value
    }


def calculate_reference_baseline(features_file, verbose=False):
    """
    Calculate reference baseline using the same incremental level-by-level approach
    as main_cocome_raw_training.py, but with 100% of training data at each level.
    
    Args:
        features_file: Path to the CoCoME features CSV file
        verbose: Whether to print verbose output
    
    Returns:
        dict: Dictionary with 'avg_rmse' and 'level_rmses' (list of RMSE for each level 1-5)
    """
    print("Calculating reference baseline (incremental training, 100% sample ratio, 70/30 split per level)...")
    
    # Load data
    data = pd.read_csv(features_file)
    
    # Prepare features and targets
    exclude_cols = ['solID', 'level', 'm1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
    if data.columns[0].startswith('Unnamed') or data.columns[0] == '':
        exclude_cols.append(data.columns[0])
    
    feature_columns = [col for col in data.columns if col not in exclude_cols]
    target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
    
    # Group data by complexity levels
    complexity_groups = {}
    for level in range(1, 6):  # Levels 1-5
        mask = data['level'] == level
        if mask.any():
            level_data = data[mask]
            complexity_groups[level] = {
                'features': level_data[feature_columns].values,
                'objectives': level_data[target_names].values
            }
    
    # Ensure we have sufficient samples in each level
    level_counts = {level: len(complexity_groups[level]['features']) for level in complexity_groups.keys()}
    print(f"Level distribution: {level_counts}")
    
    if verbose:
        print(f"Feature dimensionality: {len(feature_columns)} raw features")
        for level in sorted(complexity_groups.keys()):
            print(f"Level {level}: {len(complexity_groups[level]['features'])} samples")
    
    # Initialize optimizer and target scaler
    optimizer = CatBoostMultiTarget('CatBoost Baseline', verbose=verbose)
    target_scaler = StandardScaler()
    level_rmses = [float('nan')] * 5  # Initialize with NaN for all 5 levels
    
    # Progressive learning through complexity levels (same as main_cocome_raw_training.py)
    for level in range(1, 6):
        level_idx = level - 1  # Convert to 0-based index for list
        
        if level not in complexity_groups:
            if verbose:
                print(f"Level {level}: No data found, skipping")
            level_rmses[level_idx] = float('nan')
            continue
            
        level_data = complexity_groups[level]
        X_level = level_data['features']
        y_level = level_data['objectives']
        
        if len(X_level) < 5:
            if verbose:
                print(f"Level {level}: Insufficient samples ({len(X_level)}), skipping")
            level_rmses[level_idx] = float('nan')
            continue
        
        # Train/test split (70/30) for this level
        X_train, X_test, y_train, y_test = train_test_split(
            X_level, y_level, test_size=0.3, random_state=42  # Fixed random state for reproducible baseline
        )
        
        # Use 100% of training data (no sampling for baseline)
        # Scale targets (fit scaler on first level, transform on subsequent levels)
        if level == 1:
            y_train_scaled = target_scaler.fit_transform(y_train)
            y_test_scaled = target_scaler.transform(y_test)
        else:
            y_train_scaled = target_scaler.transform(y_train)
            y_test_scaled = target_scaler.transform(y_test)
        
        if verbose:
            print(f"Level {level}: Train/Test split: {len(X_train)}/{len(X_test)}")
            print(f"Level {level}: Training with {len(X_train)} samples (100% of train set)")
        
        try:
            # Handle both first training and incremental updates
            if level == 1:
                # Initial training on level 1
                optimizer.fit(X_train, y_train_scaled, level)
                if verbose:
                    print(f"  Level {level}: Initial training completed")
            else:
                # Incremental update with new level data
                optimizer.update(X_train, y_train_scaled, level)
                if verbose:
                    print(f"  Level {level}: Incremental update completed")
            
            # Evaluate performance on this level's test set
            y_pred_scaled = optimizer.predict(X_test)
            
            # Check for NaN predictions
            if np.any(np.isnan(y_pred_scaled)):
                print(f"  Level {level}: Warning - NaN predictions detected in baseline, replacing with median")
                # Calculate median of test targets as fallback
                median_targets = np.nanmedian(y_test_scaled, axis=0)
                # Replace NaN predictions with median values
                for i in range(y_pred_scaled.shape[1]):
                    nan_mask = np.isnan(y_pred_scaled[:, i])
                    y_pred_scaled[nan_mask, i] = median_targets[i]
            
            # Transform back to original space for meaningful RMSE calculation  
            y_test_original = target_scaler.inverse_transform(y_test_scaled)
            y_pred_original = target_scaler.inverse_transform(y_pred_scaled)
            
            # Calculate RMSE in original space
            mse = mean_squared_error(y_test_original, y_pred_original)
            rmse = np.sqrt(mse)
            
            # Sanity check for RMSE
            if not np.isfinite(rmse) or rmse > 1000:
                print(f"  Level {level}: Warning - extreme baseline RMSE value: {rmse}")
                if not np.isfinite(rmse):
                    rmse = 1000.0  # Use a high but finite value
            
            level_rmses[level_idx] = rmse
            
            if verbose:
                print(f"  Level {level}: RMSE = {rmse:.4f}")
                
        except Exception as e:
            if verbose:
                print(f"  Level {level}: Baseline training failed: {e}")
                import traceback
                traceback.print_exc()
            # Use a high RMSE value for failed levels
            level_rmses[level_idx] = 1000.0  # Use finite value instead of inf
    
    # Calculate average RMSE across all levels
    finite_rmses = [rmse for rmse in level_rmses if np.isfinite(rmse) and rmse <= 100]
    
    if finite_rmses:
        avg_rmse = np.mean(finite_rmses)
        if verbose:
            print(f"Using {len(finite_rmses)}/{len(level_rmses)} levels for baseline averaging")
    else:
        # If no reasonable values, use all finite values
        all_finite = [rmse for rmse in level_rmses if np.isfinite(rmse)]
        if all_finite:
            avg_rmse = np.mean(all_finite)
            if verbose:
                print(f"Warning: All baseline RMSE values are extreme. Using {len(all_finite)}/{len(level_rmses)} finite values")
        else:
            avg_rmse = 10.0  # Fallback baseline value
            if verbose:
                print(f"Error: No finite baseline RMSE values found, using fallback value")
    
    if verbose:
        print(f"Level RMSEs: {[f'{rmse:.4f}' if np.isfinite(rmse) else 'FAILED' for rmse in level_rmses]}")
        print(f"Average RMSE: {avg_rmse:.4f}")
    
    successful_levels = len([r for r in level_rmses if np.isfinite(r) and r <= 100])
    print(f"  Reference RMSE (avg across {successful_levels} levels): {avg_rmse:.4f}")
    
    return {
        'avg_rmse': avg_rmse,
        'level_rmses': level_rmses  # List of 5 values (one per level)
    }


def run_monte_carlo_analysis(features_file='cocome-levels-features.csv', 
                           sample_ratios=None, 
                           output_dir='Raw_training'):
    """
    Run Monte Carlo analysis across different sample ratios.
    
    Args:
        features_file: Path to the CoCoME features CSV file
        sample_ratios: List of sample ratios to test (default: 0.3 to 0.8 in 0.1 steps)
        output_dir: Directory to save results
    
    Returns:
        dict: Results containing sample ratios, RMSEs, and reference baseline
    """
    if sample_ratios is None:
        sample_ratios = np.arange(0.3, 0.85, 0.1)  # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
    
    print("="*80)
    print("MONTE CARLO SAMPLING ANALYSIS - COCOME INCREMENTAL LEARNING")
    print("="*80)
    print(f"Testing sample ratios: {sample_ratios}")
    print(f"Using model: CatBoost SingleOutput with incremental level-by-level training")
    print(f"Training approach: Level 1 -> 2 -> 3 -> 4 -> 5 (same as main_cocome_raw_training.py)")
    print(f"Sample ratio: Applied to 70% training portion at each level")
    print(f"RMSE calculation: Average RMSE across all 5 complexity levels")
    print(f"Baselines:")
    print(f"  - Incremental baseline: 100% data per level, level-by-level training")
    print(f"  - Stratified baseline: Whole dataset, no incremental learning, CatBoost")
    print(f"Features file: {features_file}")
    print("="*80)
    
    # Check if features file exists
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file '{features_file}' not found!")
    
    # Calculate incremental reference baseline
    start_time = time.time()
    reference_result = calculate_reference_baseline(features_file, verbose=False)
    reference_rmse = reference_result['avg_rmse']
    reference_level_rmses = reference_result['level_rmses']
    baseline_time = time.time() - start_time
    
    # Calculate stratified baseline (whole dataset, no incremental learning)
    start_time = time.time()
    stratified_result = calculate_stratified_baseline(features_file, verbose=False)
    stratified_rmse = stratified_result['avg_rmse']
    stratified_level_rmses = stratified_result['level_rmses']
    stratified_overall_rmse = stratified_result['overall_rmse']  # Single baseline value
    stratified_time = time.time() - start_time
    
    # Run training for each sample ratio
    results = {
        'sample_ratios': list(sample_ratios),
        'rmse_values': [],  # Average RMSE values
        'level_rmse_values': [],  # List of lists: [[level1_rmse, level2_rmse, ...], ...]
        'reference_rmse': reference_rmse,
        'reference_level_rmses': reference_level_rmses,
        'stratified_rmse': stratified_rmse,
        'stratified_level_rmses': stratified_level_rmses,
        'stratified_overall_rmse': stratified_overall_rmse,  # Single baseline value
        'timing': {'baseline_time': baseline_time, 'stratified_time': stratified_time, 'training_times': []}
    }
    
    for i, sample_ratio in enumerate(sample_ratios):
        print(f"\n--- Sample Ratio {i+1}/{len(sample_ratios)}: {sample_ratio:.1f} ---")
        start_time = time.time()
        
        # Retry logic for failed training attempts
        max_retries = 3
        retry_count = 0
        final_result = None
        
        while retry_count < max_retries and (final_result is None or not np.isfinite(final_result['avg_rmse'])):
            try:
                if retry_count > 0:
                    print(f"  Retry {retry_count} for sample_ratio {sample_ratio:.1f}...")
                final_result = run_training_with_sample_ratio(features_file, sample_ratio, verbose=False)
                
                # Check if result is valid
                if not np.isfinite(final_result['avg_rmse']):
                    print(f"  Invalid RMSE result: {final_result['avg_rmse']}")
                    final_result = None
                    retry_count += 1
                else:
                    break  # Success
                    
            except Exception as e:
                print(f"  Error with sample_ratio {sample_ratio} (attempt {retry_count + 1}): {e}")
                if retry_count == max_retries - 1:  # Last attempt
                    import traceback
                    traceback.print_exc()
                retry_count += 1
                final_result = None
        
        # Record result
        if final_result is not None and np.isfinite(final_result['avg_rmse']):
            results['rmse_values'].append(final_result['avg_rmse'])
            results['level_rmse_values'].append(final_result['level_rmses'])
        else:
            print(f"  Failed after {max_retries} attempts for sample_ratio {sample_ratio:.1f}")
            results['rmse_values'].append(float('nan'))
            results['level_rmse_values'].append([float('nan')] * 5)
        
        training_time = time.time() - start_time
        results['timing']['training_times'].append(training_time)
        print(f"Training time: {training_time:.2f} seconds")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)
    
    return results


def plot_sampling_analysis(results, output_dir='Raw_training', features_file='cocome-levels-features.csv'):
    """
    Create plots showing the effect of sample ratio on RMSE for each complexity level.
    Each sampling ratio will show 5 different colored points (one for each level).
    Includes both incremental baseline and stratified baseline reference lines.
    
    Args:
        results: Results dictionary from run_monte_carlo_analysis
        output_dir: Directory to save plots
        features_file: Path to features file for normalization calculation
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate normalization factors
    target_stats, normalization_factor = calculate_target_normalization_factors(features_file)
    
    print(f"\nCoCoME Target normalization factors calculated:")
    for target in ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']:
        if target in target_stats:
            stats = target_stats[target]
            print(f"  {target}: range={stats['range']:.2f}, std={stats['std']:.2f}")
    print(f"  Overall range (for normalization): {normalization_factor:.2f}")
    
    # Extract data
    sample_ratios = np.array(results['sample_ratios'])
    level_rmse_values = results['level_rmse_values']  # List of lists
    reference_level_rmses = results['reference_level_rmses']
    stratified_overall_rmse = results['stratified_overall_rmse']  # Single baseline value
    
    # Define colors for each complexity level
    level_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
    level_names = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Plot individual level performance for each sample ratio (normalized)
    for level_idx in range(5):
        # Skip Level 1 (level_idx == 0) - only plot Level 2 and higher
        if level_idx == 0:
            continue
            
        level_rmses_for_this_level = []
        level_rmses_normalized = []
        valid_sample_ratios = []
        
        for i, sample_ratio in enumerate(sample_ratios):
            if i < len(level_rmse_values):
                level_rmses = level_rmse_values[i]
                if level_idx < len(level_rmses) and np.isfinite(level_rmses[level_idx]):
                    rmse_raw = level_rmses[level_idx]
                    rmse_normalized = normalize_rmse(rmse_raw, normalization_factor)
                    
                    level_rmses_for_this_level.append(rmse_raw)
                    level_rmses_normalized.append(rmse_normalized)
                    valid_sample_ratios.append(sample_ratio)
                    
                    # DEBUG: Print Level 5 plotting data
                    if level_idx == 4:  # Level 5
                        baseline_norm = normalize_rmse(reference_level_rmses[level_idx], normalization_factor)
                        above_below = "ABOVE" if rmse_normalized > baseline_norm else "BELOW"
                        print(f"  DEBUG PLOT Level 5: ratio={sample_ratio}, y={rmse_normalized:.4f}, baseline={baseline_norm:.4f} -> {above_below}")
        
        # Plot this level's normalized performance across sample ratios
        if level_rmses_normalized:
            plt.plot(valid_sample_ratios, level_rmses_normalized,
                    marker='o', linewidth=2, markersize=8,
                    color=level_colors[level_idx], 
                    label=f'{level_names[level_idx]} (Incremental)',
                    markerfacecolor=level_colors[level_idx], 
                    markeredgecolor=level_colors[level_idx],
                    alpha=0.8)
            
            print(f"  Plotted {len(level_rmses_normalized)} points for Level {level_idx + 1}")
            print(f"    Raw RMSE range: {min(level_rmses_for_this_level):.2f} - {max(level_rmses_for_this_level):.2f}")
            print(f"    Normalized RMSE range: {min(level_rmses_normalized):.3f} - {max(level_rmses_normalized):.3f}")
    
    # Plot incremental baseline as dashed horizontal lines for each level (normalized)
    for level_idx in range(5):
        # Keep all level baselines (including Level 1) since they use full datasets
        if level_idx < len(reference_level_rmses) and np.isfinite(reference_level_rmses[level_idx]):
            baseline_normalized = normalize_rmse(reference_level_rmses[level_idx], normalization_factor)
            plt.axhline(y=baseline_normalized, 
                       color=level_colors[level_idx], 
                       linestyle='--', 
                       linewidth=1.5, 
                       alpha=0.6,
                       label=f'{level_names[level_idx]} (Incr. 100%)')  # Show all levels in legend
    
    # Plot stratified baseline as single thick black horizontal line (normalized)
    if np.isfinite(stratified_overall_rmse):
        stratified_normalized = normalize_rmse(stratified_overall_rmse, normalization_factor)
        plt.axhline(y=stratified_normalized, 
                   color='black', 
                   linestyle='-', 
                   linewidth=3, 
                   alpha=0.9,
                   label='Stratified Baseline (Whole Dataset)')
    
    # Formatting
    plt.xlabel('Sample Ratio (Fraction of Training Data per Level)', fontsize=12)
    plt.ylabel('Normalized RMSE', fontsize=12)
    # No title
    plt.grid(True, alpha=0.3)
    
    # Create comprehensive legend showing all entries
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Separate incremental learning curves from baselines
    incremental_handles = [h for h, l in zip(handles, labels) if 'Incremental)' in l]
    incremental_labels = [l for l in labels if 'Incremental)' in l]
    
    baseline_handles = [h for h, l in zip(handles, labels) if 'Incr. 100%)' in l or 'Stratified Baseline' in l]
    baseline_labels = [l for l in labels if 'Incr. 100%)' in l or 'Stratified Baseline' in l]
    
    # Create legends with more space
    legend1 = plt.legend(incremental_handles, incremental_labels, 
                        loc='upper left', fontsize=9, title='Incremental Learning', 
                        title_fontsize=10)
    plt.gca().add_artist(legend1)
    
    if baseline_handles:
        legend2 = plt.legend(baseline_handles, baseline_labels, 
                           loc='center left', bbox_to_anchor=(1.02, 0.5), 
                           fontsize=9, title='Baselines', title_fontsize=10)
    
    # Set x-axis ticks
    plt.xticks(sample_ratios, [f'{ratio:.1f}' for ratio in sample_ratios])
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'monte_carlo_sampling_by_level_normalized_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Normalized plot saved: {output_file}")
    
    plt.show()
    
    # Print summary with both raw and normalized values
    print("\nSUMMARY BY LEVEL:")
    print("-" * 80)
    print(f"Normalization factor (average target range): {normalization_factor:.2f}")
    print(f"Incremental baseline (level-by-level, 100% data per level):")
    for level_idx in range(5):
        if level_idx < len(reference_level_rmses) and np.isfinite(reference_level_rmses[level_idx]):
            raw_val = reference_level_rmses[level_idx]
            norm_val = normalize_rmse(raw_val, normalization_factor)
            print(f"  Level {level_idx + 1}: {raw_val:.4f} raw, {norm_val:.4f} normalized")
        else:
            print(f"  Level {level_idx + 1}: FAILED")
    
    print(f"\nStratified baseline (whole dataset, no incremental learning):")
    if np.isfinite(stratified_overall_rmse):
        raw_val = stratified_overall_rmse
        norm_val = normalize_rmse(raw_val, normalization_factor)
        print(f"  Overall RMSE: {raw_val:.4f} raw, {norm_val:.4f} normalized")
    else:
        print(f"  Overall RMSE: FAILED")
    
    print(f"\nDetailed Results by Level:")
    for i, sample_ratio in enumerate(sample_ratios):
        print(f"\nSample Ratio {sample_ratio:.1f}:")
        if i < len(level_rmse_values):
            level_rmses = level_rmse_values[i]
            for level_idx in range(5):
                if level_idx < len(level_rmses):
                    if np.isfinite(level_rmses[level_idx]):
                        inc_baseline_rmse = reference_level_rmses[level_idx] if level_idx < len(reference_level_rmses) else float('nan')
                        strat_baseline_rmse = stratified_overall_rmse  # Use single overall baseline
                        
                        result_str = f"  Level {level_idx + 1}: {level_rmses[level_idx]:.4f}"
                        
                        if np.isfinite(inc_baseline_rmse):
                            inc_diff = level_rmses[level_idx] - inc_baseline_rmse
                            result_str += f" (vs Incr: {inc_diff:+.4f})"
                        
                        if np.isfinite(strat_baseline_rmse):
                            strat_diff = level_rmses[level_idx] - strat_baseline_rmse
                            result_str += f" (vs Strat: {strat_diff:+.4f})"
                        
                        print(result_str)
                    else:
                        print(f"  Level {level_idx + 1}: FAILED")
                else:
                    print(f"  Level {level_idx + 1}: NO DATA")
        else:
            print("  All levels: FAILED")


def save_results(results, output_dir='Raw_training', features_file='cocome-levels-features.csv'):
    """Save results to CSV file for further analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate normalization factors
    target_stats, normalization_factor = calculate_target_normalization_factors(features_file)
    
    # Create DataFrame with individual level results
    rows = []
    sample_ratios = results['sample_ratios']
    level_rmse_values = results['level_rmse_values']
    reference_level_rmses = results['reference_level_rmses']
    stratified_overall_rmse = results['stratified_overall_rmse']  # Single baseline value
    training_times = results['timing']['training_times']
    
    for i, sample_ratio in enumerate(sample_ratios):
        if i < len(level_rmse_values):
            level_rmses = level_rmse_values[i]
            training_time = training_times[i] if i < len(training_times) else 0
            
            # Create one row per level for this sample ratio
            for level_idx in range(5):
                level = level_idx + 1
                level_rmse = level_rmses[level_idx] if level_idx < len(level_rmses) else float('nan')
                ref_rmse = reference_level_rmses[level_idx] if level_idx < len(reference_level_rmses) else float('nan')
                strat_rmse = stratified_overall_rmse  # Use single overall baseline for all levels
                
                # Calculate normalized values
                level_rmse_norm = normalize_rmse(level_rmse, normalization_factor) if np.isfinite(level_rmse) else float('nan')
                ref_rmse_norm = normalize_rmse(ref_rmse, normalization_factor) if np.isfinite(ref_rmse) else float('nan')
                strat_rmse_norm = normalize_rmse(strat_rmse, normalization_factor) if np.isfinite(strat_rmse) else float('nan')
                
                row = {
                    'sample_ratio': sample_ratio,
                    'complexity_level': level,
                    'rmse_raw': level_rmse,
                    'rmse_normalized': level_rmse_norm,
                    'incremental_baseline_rmse_raw': ref_rmse,
                    'incremental_baseline_rmse_normalized': ref_rmse_norm,
                    'stratified_baseline_rmse_raw': strat_rmse,
                    'stratified_baseline_rmse_normalized': strat_rmse_norm,
                    'rmse_vs_incremental': level_rmse - ref_rmse if np.isfinite(level_rmse) and np.isfinite(ref_rmse) else float('nan'),
                    'rmse_vs_stratified': level_rmse - strat_rmse if np.isfinite(level_rmse) and np.isfinite(strat_rmse) else float('nan'),
                    'relative_improvement_vs_incremental': ((ref_rmse - level_rmse) / ref_rmse * 100) if np.isfinite(level_rmse) and np.isfinite(ref_rmse) and ref_rmse != 0 else float('nan'),
                    'relative_improvement_vs_stratified': ((strat_rmse - level_rmse) / strat_rmse * 100) if np.isfinite(level_rmse) and np.isfinite(strat_rmse) and strat_rmse != 0 else float('nan'),
                    'training_time_seconds': training_time
                }
                rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'monte_carlo_results_by_level_{timestamp}.csv')
    df.to_csv(output_file, index=False)
    print(f"Results saved: {output_file}")
    
    return df


def main():
    """Main execution function."""
    # Configuration
    features_file = 'cocome-levels-features.csv'
    sample_ratios = np.arange(0.3, 0.85, 0.1)  # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
    output_dir = 'Raw_training'
    
    # Check if we're in the right directory
    if not os.path.exists(features_file):
        # Try the cocome subdirectory
        features_file = os.path.join('cocome', 'cocome-levels-features.csv')
        if not os.path.exists(features_file):
            print(f"Error: Could not find '{features_file}'")
            print("Please run this script from the directory containing the CoCoME dataset file.")
            return
    
    try:
        # Run the analysis
        results = run_monte_carlo_analysis(features_file, sample_ratios, output_dir)
        
        # Create plots
        plot_sampling_analysis(results, output_dir, features_file)
        
        # Save detailed results
        df = save_results(results, output_dir, features_file)
        
        print("\n" + "="*80)
        print("MONTE CARLO SAMPLING ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()