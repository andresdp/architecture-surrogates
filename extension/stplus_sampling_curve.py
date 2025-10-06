"""
STPlus Dataset Sampling Curve Analysis for Raw Features

This script analyzes how model performance varies with training set size using raw features.
It creates incremental learning curves by training models with increasing amounts of data
and evaluating their performance to understand data efficiency and convergence patterns.

Key differences from CoCoME version:
- Uses STPlus dataset: only 2 complexity levels instead of 5
- 4 targets instead of 8: m1,                    # Calculate RMSE in scaled space (consistent with CoCoME approach)
                    level_rmse = np.sqrt(mean_squared_error(y_test_scaled, y_pred_scaled))1, p2
- Bot types: "Modifiability" and "Performance"
- Adapted for STPlus-specific incremental analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import pickle
import xgboost as xgb
import catboost as cb
warnings.filterwarnings('ignore')

# Global random seed for reproducibility
# np.random.seed(42)


class STPlus_SamplingCurveAnalyzer:
    """
    Analyzes how model performance changes with training set size for STPlus dataset
    using raw features (~700+ columns) across 2 complexity levels and 4 targets.
    """
    
    def __init__(self, data_file, verbose=True):
        self.data_file = data_file
        self.verbose = verbose
        self.data = None
        self.features = None
        self.target_names = ['m1', 'm2', 'p1', 'p2']  # STPlus targets
        self.level_names = [1, 2]  # STPlus levels
        self.bot_types = ["Modifiability", "Performance"]
        self.results = {}
        
        # Sampling configuration - match CoCoME approach
        self.sample_ratios = np.arange(0.3, 0.85, 0.1)  # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
        self.n_repeats = 1  # Single run per sample ratio, like CoCoME
        
        if self.verbose:
            print("STPlus Sampling Curve Analyzer initialized")
            print(f"Target variables: {self.target_names}")
            print(f"Complexity levels: {self.level_names}")
            print(f"Bot types: {self.bot_types}")
    
    def load_and_prepare_data(self):
        """Load and prepare STPlus dataset with raw features"""
        if self.verbose:
            print(f"Loading STPlus data from: {self.data_file}")
        
        self.data = pd.read_csv(self.data_file)
        
        if self.verbose:
            print(f"Raw data shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns[:10])}..." if len(self.data.columns) > 10 else f"Columns: {list(self.data.columns)}")
        
        # Identify feature columns (exclude targets, metadata, and non-numeric columns)
        exclude_columns = ['Unnamed: 0', 'solID', 'level', 'bot'] + self.target_names
        self.features = [col for col in self.data.columns if col not in exclude_columns]
        
        if self.verbose:
            print(f"Feature columns identified: {len(self.features)} features")
            print(f"Sample features: {self.features[:10]}...")
            print(f"Excluded columns: {exclude_columns}")
        
        # Verify all feature columns are numeric
        non_numeric_features = []
        for col in self.features:
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                non_numeric_features.append(col)
        
        if non_numeric_features:
            if self.verbose:
                print(f"Removing non-numeric feature columns: {non_numeric_features}")
            self.features = [col for col in self.features if col not in non_numeric_features]
        
        if self.verbose:
            print(f"Final feature columns: {len(self.features)} numeric features")
        
        # Handle missing values
        if self.data.isnull().sum().sum() > 0:
            if self.verbose:
                print("Handling missing values...")
            self.data = self.data.fillna(self.data.median(numeric_only=True))
        
        # Encode categorical bot types if needed
        if self.data['bot'].dtype == 'object':
            le_bot = LabelEncoder()
            self.data['bot'] = le_bot.fit_transform(self.data['bot'])
            if self.verbose:
                print(f"Bot types encoded: {dict(zip(le_bot.classes_, le_bot.transform(le_bot.classes_)))}")
        
        if self.verbose:
            print("Data preparation completed")
            print(f"Final data shape: {self.data.shape}")
            for level in self.level_names:
                level_count = len(self.data[self.data['level'] == level])
                print(f"Level {level}: {level_count} samples")
        
        # Calculate target statistics for normalization
        self._calculate_target_normalization_factors()
        
        return self
    
    def _calculate_target_normalization_factors(self):
        """Calculate normalization factors for RMSE based on target ranges"""
        self.target_stats = {}
        self.target_ranges = {}
        
        for target in self.target_names:
            target_values = self.data[target].values
            target_min = np.min(target_values)
            target_max = np.max(target_values)
            target_range = target_max - target_min
            target_std = np.std(target_values)
            
            self.target_stats[target] = {
                'min': target_min,
                'max': target_max,
                'range': target_range,
                'std': target_std,
                'mean': np.mean(target_values)
            }
            self.target_ranges[target] = target_range
        
        # Overall normalization factor (using range-based normalization)
        self.overall_target_range = np.mean(list(self.target_ranges.values()))
        
        if self.verbose:
            print(f"\nTarget normalization factors calculated:")
            for target in self.target_names:
                stats = self.target_stats[target]
                print(f"  {target}: range={stats['range']:.2f}, std={stats['std']:.2f}")
            print(f"  Overall range (for normalization): {self.overall_target_range:.2f}")
    
    def normalize_rmse(self, rmse_value):
        """Normalize RMSE by the overall target range to make it dimensionless"""
        return rmse_value / self.overall_target_range if self.overall_target_range > 0 else rmse_value
    
    def create_level_specific_models(self):
        """Create model configurations for each level and target"""
        model_configs = {}
        
        for level in self.level_names:
            model_configs[f'level_{level}'] = {
                # XGBoost configurations - adapted for STPlus
                'xgb_single': {
                    'name': 'XGBoost SingleOutput',
                    'models': {},
                    'scalers': {},
                    'type': 'single_target'
                },
                'xgb_multi': {
                    'name': 'XGBoost MultiOutput',
                    'model': None,
                    'scaler': None,
                    'type': 'multi_target'
                },
                # CatBoost configurations - adapted for STPlus
                'catboost_single': {
                    'name': 'CatBoost SingleOutput',
                    'models': {},
                    'scalers': {},
                    'type': 'single_target'
                },
                'catboost_multi': {
                    'name': 'CatBoost MultiOutput',
                    'model': None,
                    'scaler': None,
                    'type': 'multi_target'
                }
            }
        
        return model_configs
    
    def train_xgboost_single_target(self, X_train, y_train, X_test, y_test, target_name):
        """Train XGBoost model for single target"""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # XGBoost model configuration
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        return {
            'model': model,
            'scaler': scaler,
            'predictions': y_pred,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'mae': mae
        }
    
    def train_xgboost_multi_target(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model for multiple targets"""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # XGBoost MultiOutput model
        from sklearn.multioutput import MultiOutputRegressor
        base_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1  # Use 1 job per estimator since we have multiple estimators
        )
        
        model = MultiOutputRegressor(base_model)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics for each target
        results = {}
        for i, target in enumerate(self.target_names):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            
            results[target] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'mae': mae
            }
        
        # Overall metrics
        overall_mse = mean_squared_error(y_test, y_pred)
        overall_r2 = r2_score(y_test, y_pred)
        
        return {
            'model': model,
            'scaler': scaler,
            'predictions': y_pred,
            'target_results': results,
            'overall_mse': overall_mse,
            'overall_rmse': np.sqrt(overall_mse),
            'overall_r2': overall_r2
        }
    
    def train_catboost_single_target(self, X_train, y_train, X_test, y_test, target_name):
        """Train CatBoost model for single target"""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # CatBoost model configuration
        model = cb.CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        )
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        return {
            'model': model,
            'scaler': scaler,
            'predictions': y_pred,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'mae': mae
        }
    
    def train_catboost_multi_target(self, X_train, y_train, X_test, y_test):
        """Train CatBoost model for multiple targets"""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # CatBoost MultiOutput model
        from sklearn.multioutput import MultiOutputRegressor
        base_model = cb.CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        )
        
        model = MultiOutputRegressor(base_model)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics for each target
        results = {}
        for i, target in enumerate(self.target_names):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            
            results[target] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'mae': mae
            }
        
        # Overall metrics
        overall_mse = mean_squared_error(y_test, y_pred)
        overall_r2 = r2_score(y_test, y_pred)
        
        return {
            'model': model,
            'scaler': scaler,
            'predictions': y_pred,
            'target_results': results,
            'overall_mse': overall_mse,
            'overall_rmse': np.sqrt(overall_mse),
            'overall_r2': overall_r2
        }
    
    def run_sampling_curve_analysis(self):
        """Run incremental sampling curve analysis following CoCoME approach"""
        if self.verbose:
            print("Starting STPlus incremental sampling curve analysis...")
            print(f"Sample ratios: {self.sample_ratios}")
            print(f"Single run per sample ratio (like CoCoME)")
            print(f"Incremental learning: Level 1 -> Level 2")
        
        self.results = {}
        
        # For each sample ratio, do incremental training across levels
        for sample_ratio in self.sample_ratios:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"SAMPLE RATIO {sample_ratio:.1f}")
                print(f"{'='*60}")
            
            # Group data by complexity levels
            complexity_groups = {}
            for level in self.level_names:
                level_data = self.data[self.data['level'] == level].copy()
                if len(level_data) > 0:
                    complexity_groups[level] = {
                        'features': level_data[self.features].values,
                        'objectives': level_data[self.target_names].values,
                        'bot': level_data['bot'].values
                    }
            
            # Initialize target scaler and models
            target_scaler = StandardScaler()
            feature_scaler = StandardScaler()
            level_rmses = {}
            
            # Progressive learning through complexity levels (1 -> 2)
            for level_idx, level in enumerate(self.level_names):
                if level not in complexity_groups:
                    if self.verbose:
                        print(f"\nLevel {level}: No data found, skipping...")
                    level_rmses[level] = float('nan')
                    continue
                
                level_data = complexity_groups[level]
                X_level = level_data['features']
                y_level = level_data['objectives']
                bot_level = level_data['bot']
                
                if len(X_level) < 5:
                    if self.verbose:
                        print(f"\nLevel {level}: Insufficient samples ({len(X_level)}), skipping...")
                    level_rmses[level] = float('nan')
                    continue
                
                # Train/test split (70/30) for this level
                X_train_full, X_test, y_train_full, y_test = train_test_split(
                    X_level, y_level, test_size=0.3, random_state=42, stratify=bot_level
                )
                
                # Apply sample ratio to training data
                n_train_samples = max(1, int(len(X_train_full) * sample_ratio))
                
                # Ensure we have at least a minimum number of samples for stable training
                min_samples = min(5, len(X_train_full))  # At least 5 samples or all available
                n_train_samples = max(min_samples, n_train_samples)
                
                n_train_samples = min(n_train_samples, len(X_train_full))
                
                # Sample training data
                if n_train_samples < len(X_train_full):
                    # Use fixed random seed for reproducible sampling across different sample ratios
                    np.random.seed(42 + level_idx * 100 + int(sample_ratio * 10))
                    train_indices = np.random.choice(len(X_train_full), size=n_train_samples, replace=False)
                    X_train = X_train_full[train_indices]
                    y_train = y_train_full[train_indices]
                else:
                    X_train = X_train_full
                    y_train = y_train_full
                
                if self.verbose:
                    print(f"\nLevel {level}:")
                    print(f"  Train/Test split: {len(X_train_full)}/{len(X_test)}")
                    print(f"  Training with {len(X_train)} samples ({sample_ratio*100:.0f}% of train set)")
                
                # Scale targets and features - FIX: Use all data up to current level for consistent scaling
                if level_idx == 0:  # First level
                    # Fit scalers on current level data
                    y_train_scaled = target_scaler.fit_transform(y_train)
                    y_test_scaled = target_scaler.transform(y_test)
                    X_train_scaled = feature_scaler.fit_transform(X_train)
                    X_test_scaled = feature_scaler.transform(X_test)
                    
                    # Initialize separate models for each target (CatBoost single output)
                    target_models = {}
                    for target_idx, target_name in enumerate(self.target_names):
                        target_model = cb.CatBoostRegressor(
                            iterations=100,
                            depth=6,
                            learning_rate=0.1,
                            random_seed=42,
                            verbose=False
                        )
                        target_model.fit(X_train_scaled, y_train_scaled[:, target_idx])
                        target_models[target_name] = target_model
                        
                        if self.verbose:
                            print(f"    Initial training for {target_name}")
                    
                    # Store raw data for proper scaling in next level
                    accumulated_X_raw = X_train.copy()
                    accumulated_y_raw = y_train.copy()
                
                else:  # Subsequent levels - refit scalers on all accumulated data
                    # Combine raw data from all levels so far
                    accumulated_X_raw = np.vstack([accumulated_X_raw, X_train])
                    accumulated_y_raw = np.vstack([accumulated_y_raw, y_train])
                    
                    # Refit scalers on all accumulated data for consistent scaling
                    target_scaler = StandardScaler()
                    feature_scaler = StandardScaler()
                    
                    # Scale all accumulated data
                    accumulated_y_scaled = target_scaler.fit_transform(accumulated_y_raw)
                    accumulated_X_scaled = feature_scaler.fit_transform(accumulated_X_raw)
                    
                    # Scale current test data with the refitted scaler
                    y_test_scaled = target_scaler.transform(y_test)
                    X_test_scaled = feature_scaler.transform(X_test)
                    
                    # Retrain each target model with all accumulated data
                    for target_idx, target_name in enumerate(self.target_names):
                        target_model = cb.CatBoostRegressor(
                            iterations=100,
                            depth=6,
                            learning_rate=0.1,
                            random_seed=42,
                            verbose=False
                        )
                        target_model.fit(accumulated_X_scaled, accumulated_y_scaled[:, target_idx])
                        target_models[target_name] = target_model
                        
                        if self.verbose:
                            print(f"    Incremental update for {target_name} with level {level} data")
                
                # Evaluate performance on this level's test set
                try:
                    # Get predictions from all target models
                    y_pred_scaled = np.zeros_like(y_test_scaled)
                    for target_idx, target_name in enumerate(self.target_names):
                        y_pred_scaled[:, target_idx] = target_models[target_name].predict(X_test_scaled)
                    
                    # Convert back to original scale for RMSE calculation
                    y_test_original = target_scaler.inverse_transform(y_test_scaled)
                    y_pred_original = target_scaler.inverse_transform(y_pred_scaled)
                    
                    # Calculate RMSE in original scale
                    mse = mean_squared_error(y_test_original, y_pred_original)
                    rmse = np.sqrt(mse)
                    level_rmses[level] = rmse
                    
                    if self.verbose:
                        print(f"    Level {level}: RMSE = {rmse:.4f}")
                
                except Exception as e:
                    if self.verbose:
                        print(f"    Level {level}: Evaluation failed: {e}")
                    level_rmses[level] = float('nan')
            
            # Store results for this sample ratio
            self.results[f'ratio_{sample_ratio:.1f}'] = level_rmses
        
        if self.verbose:
            print(f"\nIncremental sampling curve analysis completed!")
        
        return self
    
    def calculate_stratified_baseline(self):
        """Calculate stratified baseline using the whole dataset with no level-by-level training"""
        if self.verbose:
            print("Calculating stratified baseline (whole dataset, stratified 70/30 split)...")
        
        # Use all data together
        X_all = self.data[self.features].values
        y_all = self.data[self.target_names].values
        levels_all = self.data['level'].values
        
        # Stratified train/test split maintaining level proportions
        X_train, X_test, y_train, y_test, levels_train, levels_test = train_test_split(
            X_all, y_all, levels_all, test_size=0.3, stratify=levels_all, random_state=42
        )
        
        # Scale features and targets
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
        y_train_scaled = target_scaler.fit_transform(y_train)
        y_test_scaled = target_scaler.transform(y_test)
        
        if self.verbose:
            print(f"  Train/Test split: {len(X_train)}/{len(X_test)}")
            train_level_counts = {level: sum(levels_train == level) for level in self.level_names}
            test_level_counts = {level: sum(levels_test == level) for level in self.level_names}
            print(f"  Train level distribution: {train_level_counts}")
            print(f"  Test level distribution: {test_level_counts}")
        
        # Train CatBoost model on whole dataset
        from sklearn.multioutput import MultiOutputRegressor
        
        model = MultiOutputRegressor(cb.CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        ))
        
        try:
            model.fit(X_train_scaled, y_train_scaled)
            y_pred_scaled = model.predict(X_test_scaled)
            
            # Transform back to original space for meaningful RMSE calculation
            y_test_original = target_scaler.inverse_transform(y_test_scaled)
            y_pred_original = target_scaler.inverse_transform(y_pred_scaled)
            
            # Calculate RMSE for each level separately in original space
            level_rmses = {}
            for level in self.level_names:
                level_mask = levels_test == level
                if np.any(level_mask):
                    y_test_level = y_test_original[level_mask]
                    y_pred_level = y_pred_original[level_mask]
                    
                    mse = mean_squared_error(y_test_level, y_pred_level)
                    rmse = np.sqrt(mse)
                    level_rmses[level] = rmse
                    
                    if self.verbose:
                        print(f"    Level {level}: RMSE = {rmse:.4f} ({sum(level_mask)} samples, original space)")
            
            # Calculate overall RMSE in original space
            overall_mse = mean_squared_error(y_test_original, y_pred_original)
            overall_rmse = np.sqrt(overall_mse)
            
            if self.verbose:
                print(f"  Overall stratified baseline RMSE: {overall_rmse:.4f} (original space)")
            
            return {
                'overall_rmse': overall_rmse,
                'level_rmses': level_rmses
            }
            
        except Exception as e:
            if self.verbose:
                print(f"  Stratified baseline training failed: {e}")
            return {
                'overall_rmse': float('nan'),
                'level_rmses': {}
            }
    
    def create_sampling_curve_plots(self):
        """Create comprehensive sampling curve visualizations matching CoCoME style"""
        if self.verbose:
            print("Creating sampling curve visualizations...")
        
        # Calculate stratified baseline
        baseline_result = self.calculate_stratified_baseline()
        
        # Style configuration - match CoCoME colors
        plt.style.use('default')
        level_colors = ['#1f77b4', '#ff7f0e']  # Blue for Level 1, Orange for Level 2
        level_names = ['Level 1', 'Level 2']
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Plot individual level performance for each sample ratio
        for level_idx, level in enumerate(self.level_names):
            # Skip Level 1 (level_idx == 0) - only plot Level 2 and higher
            if level_idx == 0:
                continue
                
            sample_ratios = []
            rmse_values = []
            rmse_normalized = []
            
            # Extract data points from results (new structure: ratio_X.X -> {level: rmse})
            for ratio_key in sorted(self.results.keys()):
                sample_ratio = float(ratio_key.split('_')[1])
                if level in self.results[ratio_key] and np.isfinite(self.results[ratio_key][level]):
                    rmse_raw = self.results[ratio_key][level]
                    rmse_norm = self.normalize_rmse(rmse_raw)
                    
                    sample_ratios.append(sample_ratio)
                    rmse_values.append(rmse_raw)
                    rmse_normalized.append(rmse_norm)
            
            if sample_ratios:
                # Plot normalized RMSE curve for this level
                ax.plot(sample_ratios, rmse_normalized, 'o-', 
                       color=level_colors[level_idx], 
                       label=f'{level_names[level_idx]} (Incremental)',
                       linewidth=2, markersize=8, alpha=0.8)
                
                if self.verbose:
                    print(f"  Plotted {len(sample_ratios)} points for Level {level}")
                    print(f"    Raw RMSE range: {min(rmse_values):.2f} - {max(rmse_values):.2f}")
                    print(f"    Normalized RMSE range: {min(rmse_normalized):.3f} - {max(rmse_normalized):.3f}")
            else:
                if self.verbose:
                    print(f"  No valid data points for Level {level}")
        
        # Debug: Print results structure
        if self.verbose:
            print("  Debug - Results structure:")
            for ratio_key in sorted(self.results.keys()):
                print(f"    {ratio_key}: {self.results[ratio_key]}")
        
        # Plot stratified baseline as thick black horizontal line (normalized)
        if np.isfinite(baseline_result['overall_rmse']):
            baseline_normalized = self.normalize_rmse(baseline_result['overall_rmse'])
            ax.axhline(y=baseline_normalized, 
                      color='black', 
                      linestyle='-', 
                      linewidth=3, 
                      alpha=0.9,
                      label='Stratified Baseline (Whole Dataset)')
        
        # Plot individual level baselines as dashed lines (normalized)
        for level_idx, level in enumerate(self.level_names):
            # Keep all level baselines (including Level 1) since they use full datasets
            if level in baseline_result['level_rmses']:
                level_baseline = baseline_result['level_rmses'][level]
                if np.isfinite(level_baseline):
                    level_baseline_normalized = self.normalize_rmse(level_baseline)
                    ax.axhline(y=level_baseline_normalized, 
                              color=level_colors[level_idx], 
                              linestyle='--', 
                              linewidth=1.5, 
                              alpha=0.6,
                              label=f'{level_names[level_idx]} (100% Data)')
        
        # Formatting
        ax.set_xlabel('Sample Ratio (Fraction of Training Data per Level)', fontsize=12)
        ax.set_ylabel('Normalized RMSE', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis ticks
        ax.set_xticks(self.sample_ratios)
        ax.set_xticklabels([f'{ratio:.1f}' for ratio in self.sample_ratios])
        
        # Create legend
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = 'Raw_training'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plot_filename = f'{output_dir}/stplus_sampling_curve_analysis_normalized.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Normalized sampling curve plot saved: {plot_filename}")
        
        plt.show()
        
        # Print summary with both raw and normalized values
        print("\nSTPlus Sampling Analysis Summary:")
        print("-" * 80)
        print(f"Normalization factor (average target range): {self.overall_target_range:.2f}")
        print(f"Stratified baseline (whole dataset): {baseline_result['overall_rmse']:.4f} raw, {self.normalize_rmse(baseline_result['overall_rmse']):.4f} normalized")
        for level in self.level_names:
            if level in baseline_result['level_rmses']:
                raw_val = baseline_result['level_rmses'][level]
                norm_val = self.normalize_rmse(raw_val)
                print(f"Level {level} baseline: {raw_val:.4f} raw, {norm_val:.4f} normalized")
        
        print(f"\nDetailed Results by Level:")
        for sample_ratio in self.sample_ratios:
            print(f"\nSample Ratio {sample_ratio:.1f}:")
            for level in self.level_names:
                level_key = f'level_{level}'
                if level_key in self.results and sample_ratio in self.results[level_key]['catboost_single']:
                    result = self.results[level_key]['catboost_single'][sample_ratio]
                    print(f"  Level {level}: {result['rmse_mean']:.4f} (scaled)")
                else:
                    print(f"  Level {level}: FAILED")
    
    def print_sampling_summary(self):
        """Print summary of sampling curve results"""
        print("\n" + "="*80)
        print("STPLUS INCREMENTAL SAMPLING CURVE SUMMARY (SCALED SPACE)")
        print("="*80)
        
        # Find best and worst performance for each level
        for level in self.level_names:
            print(f"\nLevel {level}:")
            print("-" * 10)
            
            best_rmse = float('inf')
            worst_rmse = 0
            best_ratio = None
            worst_ratio = None
            
            for ratio_key in self.results.keys():
                if level in self.results[ratio_key]:
                    rmse = self.results[ratio_key][level]
                    if np.isfinite(rmse):
                        sample_ratio = float(ratio_key.split('_')[1])
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_ratio = sample_ratio
                        if rmse > worst_rmse:
                            worst_rmse = rmse
                            worst_ratio = sample_ratio
            
            if best_ratio is not None:
                print(f"  Best RMSE: {best_rmse:.4f} at ratio {best_ratio:.1f}")
                print(f"  Worst RMSE: {worst_rmse:.4f} at ratio {worst_ratio:.1f}")
                if worst_rmse > 0:
                    improvement = ((worst_rmse - best_rmse) / worst_rmse) * 100
                    print(f"  Improvement: {improvement:.1f}%")
            else:
                print(f"  No valid results found")
        
        print("="*80)
    
    def save_sampling_results(self, filename=None):
        """Save sampling curve results to file"""
        output_dir = 'Raw_training'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'stplus_sampling_curve_results_{timestamp}.csv'
        
        results_path = os.path.join(output_dir, filename)
        
        # Prepare data for CSV export with both raw and normalized RMSE
        rows = []
        for ratio_key in sorted(self.results.keys()):
            sample_ratio = float(ratio_key.split('_')[1])
            for level in self.level_names:
                if level in self.results[ratio_key]:
                    rmse_raw = self.results[ratio_key][level]
                    rmse_normalized = self.normalize_rmse(rmse_raw)
                    row = {
                        'Sample_Ratio': sample_ratio,
                        'Level': level,
                        'RMSE_Raw': rmse_raw,
                        'RMSE_Normalized': rmse_normalized,
                        'Model_Type': 'catboost_single_incremental'
                    }
                    rows.append(row)
        
        if rows:
            results_df = pd.DataFrame(rows)
            results_df.to_csv(results_path, index=False)
            print(f"Sampling curve results saved: {results_path}")
        else:
            print("No results to save!")


def main_sampling_curve_stplus():
    """Main function for STPlus sampling curve analysis"""
    print("Starting STPlus Sampling Curve Analysis - Raw Features")
    print("="*60)
    print("Dataset: STPlus (~700+ raw features)")
    print("Levels: 2 (complexity levels)")
    print("Targets: 4 (m1, m2, p1, p2)")
    print("Models: CatBoost (Single Output only)")
    print("Sample ratios: 0.3 to 0.8 in 0.1 steps (single run each)")
    print("Approach: Level-by-level training with stratified baseline")
    print("="*60)
    
    # Create analyzer
    analyzer = STPlus_SamplingCurveAnalyzer('stplus-levels-bots-features.csv', verbose=True)
    
    # Run analysis
    analyzer.load_and_prepare_data()
    analyzer.run_sampling_curve_analysis()
    analyzer.create_sampling_curve_plots()
    analyzer.print_sampling_summary()
    analyzer.save_sampling_results()
    
    print("\nSTPlus sampling curve analysis completed!")
    print("Results saved as 'Raw_training/stplus_sampling_curve_analysis.png' and CSV file")
    
    return analyzer


if __name__ == "__main__":
    # Run STPlus sampling curve analysis
    analyzer = main_sampling_curve_stplus()