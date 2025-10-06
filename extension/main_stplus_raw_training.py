"""
Architecture Optimization Training - STPlus Dataset with Raw Features

This script performs architecture optimization using the original raw features from stplus-levels-bots-features.csv.
It uses all original columns including operation columns (op1_, op2_) and embedding columns (emb_).

Key differences from CoCoME version:
- Uses STPlus features: ~700+ columns instead of 930+ from CoCoME
- Only 2 complexity levels instead of 5
- 4 targets instead of 8: m1, m2, p1, p2 (no m3, m4, p3, p4)
- Bot types: "Modifiability" and "Performance"

Active Learning Features:
- Initial training: Full dataset from complexity level 1
- Incremental updates: Small samples from each subsequent level (only level 2)
- Memory management: Maintains sliding window of recent samples
- Early stopping based on performance improvement
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# If you need reproducibility, uncomment the next line:
# np.random.seed(42)

class XGBoostSingleTarget:
    """XGBoost model for single target with incremental learning"""
    
    def __init__(self, target_name, target_idx, verbose=True):
        self.target_name = target_name
        self.target_idx = target_idx
        self.verbose = verbose
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
        self.X_train_history = None
        self.y_train_history = None
    
    def fit(self, X, y):
        """Fit XGBoost model for single target; if a previous model exists, continue from it."""
        self.X_train = X.copy()
        self.y_train = y[:, self.target_idx].copy()
        
        X_scaled = self.scaler.fit_transform(X)
        
        xgb_params = {
            'n_estimators': 50,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': np.random.randint(0, 10000)
        }
        
        if self.model is not None and hasattr(self.model, 'get_booster'):
            # Continue training from existing model
            self.model.fit(X_scaled, self.y_train, xgb_model=self.model.get_booster())
        else:
            # New model
            self.model = xgb.XGBRegressor(**xgb_params)
            self.model.fit(X_scaled, self.y_train)
        
        return self
    
    def predict(self, X):
        """Predict single target values"""
        if self.model is None:
            raise ValueError("Model not fitted yet!")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def update(self, X_new, y_new):
        """Incremental learning: Continue XGBoost training with ONLY new data"""
        y_new_target = y_new[:, self.target_idx]
        X_new_scaled = self.scaler.transform(X_new)
        
        # Incremental learning: Continue training existing model with ONLY new data
        if hasattr(self.model, 'get_booster'):
            # Create temporary model for incremental update
            temp_xgb_params = {
                'n_estimators': min(25, max(5, len(X_new) // 2)),
                'max_depth': 4,
                'learning_rate': 0.05,  # Lower learning rate for incremental updates
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': np.random.randint(0, 10000)
            }
            
            try:
                # Continue training from existing booster
                self.model.set_params(**temp_xgb_params)
                self.model.fit(X_new_scaled, y_new_target, xgb_model=self.model.get_booster())
            except Exception as e:
                if self.verbose:
                    print(f"    Warning: XGBoost incremental update failed for {self.target_name}: {e}")
                    print(f"    Falling back to regular fit...")
                # Fallback to regular training
                self.model.fit(X_new_scaled, y_new_target)
        
        # Evaluate performance on new data only
        y_pred = self.model.predict(X_new_scaled)
        mse = mean_squared_error(y_new_target, y_pred)
        
        if self.verbose:
            print(f"    XGBoost {self.target_name} update: MSE on new data = {mse:.6f}")
        
        return mse


class CatBoostSingleTarget:
    """CatBoost model for single target with incremental learning"""
    
    def __init__(self, target_name, target_idx, verbose=True, save_models=False, models_dir="models"):
        self.target_name = target_name
        self.target_idx = target_idx
        self.verbose = verbose
        self.save_models = save_models
        self.models_dir = models_dir
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
        self.X_train_history = None
        self.y_train_history = None
        self.current_level = 0
    
    def fit(self, X, y, level=None):
        """Fit CatBoost model for single target; if a previous model exists, continue from it."""
        self.X_train = X.copy()
        self.y_train = y[:, self.target_idx].copy()
        
        if level is not None:
            self.current_level = level
        
        X_scaled = self.scaler.fit_transform(X)
        
        cb_params = {
            'iterations': min(50, max(10, len(X) // 2)),  # Adaptive iterations based on data size
            'depth': 4,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3.0,
            'random_seed': int(np.random.randint(0, 10000)),
            'loss_function': 'RMSE',
            'verbose': False,
            'allow_writing_files': False,  # Prevent CatBoost from writing files
            'thread_count': 1  # Control threading for stability
        }
        
        try:
            if self.model is not None:
                # Continue from existing model if available
                self.model = CatBoostRegressor(**cb_params)
                self.model.fit(X_scaled, self.y_train, init_model=self.model)
            else:
                # New model
                self.model = CatBoostRegressor(**cb_params)
                self.model.fit(X_scaled, self.y_train)
            
            # Save model if requested
            if self.save_models:
                self._save_model(level)
                
        except Exception as e:
            if self.verbose:
                print(f"    Warning: CatBoost training failed for {self.target_name}: {e}")
            return self
        
        return self
    
    def predict(self, X):
        """Predict single target values"""
        if self.model is None:
            if self.verbose:
                print(f"Warning: {self.target_name} model not fitted, returning zeros")
            return np.zeros(len(X))
        
        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Prediction failed for {self.target_name}: {e}, returning zeros")
            return np.zeros(len(X))
    
    def update(self, X_new, y_new, level=None):
        """Incremental learning: Continue CatBoost training with ONLY new data"""
        if self.model is None:
            if self.verbose:
                print(f"    No existing model for {self.target_name}, using fit instead")
            return self.fit(X_new, y_new, level)
        
        if level is not None:
            self.current_level = level
            
        y_new_target = y_new[:, self.target_idx]
        X_new_scaled = self.scaler.transform(X_new)
        
        try:
            # CatBoost incremental learning parameters
            cb_update_params = {
                'iterations': min(25, max(5, len(X_new) // 2)),
                'depth': 4,
                'learning_rate': 0.05,  # Lower learning rate for updates
                'l2_leaf_reg': 3.0,
                'random_seed': int(np.random.randint(0, 10000)),
                'loss_function': 'RMSE',
                'verbose': False,
                'allow_writing_files': False,
                'thread_count': 1
            }
            
            # Create new model and continue from previous
            updated_model = CatBoostRegressor(**cb_update_params)
            updated_model.fit(X_new_scaled, y_new_target, init_model=self.model)
            self.model = updated_model
            
            # Save updated model if requested
            if self.save_models:
                self._save_model(level)
            
            # Evaluate performance on new data
            y_pred = self.model.predict(X_new_scaled)
            mse = mean_squared_error(y_new_target, y_pred)
            
            if self.verbose:
                print(f"    CatBoost {self.target_name} update: MSE on new data = {mse:.6f}")
            
            return mse
                
        except Exception as e:
            if self.verbose:
                print(f"    Warning: CatBoost incremental update failed for {self.target_name}: {e}")
            return float('inf')
    
    def _save_model(self, level):
        """Save the current model to the models directory"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        model_filename = f"CatBoostSingleOutput_{self.target_name}_level-{level}.cbm"
        model_path = os.path.join(self.models_dir, model_filename)
        
        try:
            self.model.save_model(model_path)
            if self.verbose:
                print(f"    Model saved: {model_path}")
        except Exception as e:
            if self.verbose:
                print(f"    Warning: Could not save model {model_path}: {e}")


class XGBoostMultiTarget:
    """XGBoost optimizer with separate models for each target."""
    
    def __init__(self, name, verbose=True):
        self.name = name
        self.verbose = verbose
        self.target_names = ['m1', 'm2', 'p1', 'p2']  # STPlus targets
        self.models = {target: XGBoostSingleTarget(target, i, verbose) 
                      for i, target in enumerate(self.target_names)}
        
    def fit(self, X, y, level=None):
        for target, model in self.models.items():
            model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models.values()])
        return predictions
    
    def update(self, X_new, y_new, level=None):
        for target, model in self.models.items():
            model.update(X_new, y_new)
    
    def recommend_next(self, candidate_pool, n_recommendations=5):
        # Simple uncertainty-based sampling
        predictions = self.predict(candidate_pool)
        uncertainty = np.std(predictions, axis=1)
        top_indices = np.argsort(uncertainty)[-n_recommendations:]
        return candidate_pool[top_indices]


class CatBoostMultiTarget:
    """CatBoost optimizer with separate models for each target."""
    
    def __init__(self, name, verbose=True, save_models=False, models_dir="models"):
        self.name = name
        self.verbose = verbose
        self.target_names = ['m1', 'm2', 'p1', 'p2']  # STPlus targets
        self.models = {target: CatBoostSingleTarget(target, i, verbose, save_models, models_dir) 
                      for i, target in enumerate(self.target_names)}
        
    def fit(self, X, y, level=None):
        for target, model in self.models.items():
            model.fit(X, y, level)
        return self
    
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models.values()])
        return predictions
    
    def update(self, X_new, y_new, level=None):
        for target, model in self.models.items():
            model.update(X_new, y_new, level)
    
    def recommend_next(self, candidate_pool, n_recommendations=5):
        # Simple uncertainty-based sampling
        predictions = self.predict(candidate_pool)
        uncertainty = np.std(predictions, axis=1)
        top_indices = np.argsort(uncertainty)[-n_recommendations:]
        return candidate_pool[top_indices]


class XGBoostMultiOutput:
    """XGBoost optimizer with single multi-output model for all targets"""
    
    def __init__(self, name, verbose=True):
        self.name = name
        self.verbose = verbose
        self.target_names = ['m1', 'm2', 'p1', 'p2']  # STPlus targets
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y, level=None):
        """Fit XGBoost Multi-output model using MultiOutputRegressor"""
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        X_scaled = self.scaler.fit_transform(X)
        
        xgb_params = {
            'n_estimators': 50,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': np.random.randint(0, 10000)
        }
        
        base_model = xgb.XGBRegressor(**xgb_params)
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X):
        """Predict multi-target values"""
        if self.model is None:
            if self.verbose:
                print("Warning: Model not fitted, returning zeros")
            return np.zeros((len(X), len(self.target_names)))
        
        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Prediction failed: {e}, returning zeros")
            return np.zeros((len(X), len(self.target_names)))
    
    def update(self, X_new, y_new, level=None):
        """Incremental learning for multi-output model"""
        if self.model is None:
            return self.fit(X_new, y_new, level=level)
        
        # For multi-output, we retrain on combined data
        X_combined = np.vstack([self.X_train, X_new])
        y_combined = np.vstack([self.y_train, y_new])
        
        # Keep only recent data to prevent memory issues
        max_samples = 1000
        if len(X_combined) > max_samples:
            X_combined = X_combined[-max_samples:]
            y_combined = y_combined[-max_samples:]
        
        return self.fit(X_combined, y_combined, level=level)
    
    def recommend_next(self, candidate_pool, n_recommendations=5):
        predictions = self.predict(candidate_pool)
        uncertainty = np.std(predictions, axis=1)
        top_indices = np.argsort(uncertainty)[-n_recommendations:]
        return candidate_pool[top_indices]


class CatBoostMultiOutput:
    """CatBoost optimizer with single multi-output model (wrapped) and TRUE incremental updates."""
    
    def __init__(self, name, verbose=True):
        self.name = name
        self.verbose = verbose
        self.target_names = ['m1', 'm2', 'p1', 'p2']  # STPlus targets
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y, level=None):
        """Fit CatBoost Multi-output model using MultiOutputRegressor"""
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        X_scaled = self.scaler.fit_transform(X)
        
        cb_params = {
            'iterations': min(50, max(10, len(X) // 2)),
            'depth': 4,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3.0,
            'random_seed': int(np.random.randint(0, 10000)),
            'loss_function': 'RMSE',
            'verbose': False,
            'allow_writing_files': False,
            'thread_count': 1
        }
        
        base_model = CatBoostRegressor(**cb_params)
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X):
        """Predict multi-target values"""
        if self.model is None:
            if self.verbose:
                print("Warning: Model not fitted, returning zeros")
            return np.zeros((len(X), len(self.target_names)))
        
        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Prediction failed: {e}, returning zeros")
            return np.zeros((len(X), len(self.target_names)))
    
    def update(self, X_new, y_new, level=None):
        """Incremental learning for multi-output model"""
        if self.model is None:
            return self.fit(X_new, y_new, level=level)
        
        # For multi-output, we retrain on combined data
        X_combined = np.vstack([self.X_train, X_new])
        y_combined = np.vstack([self.y_train, y_new])
        
        # Keep only recent data to prevent memory issues
        max_samples = 1000
        if len(X_combined) > max_samples:
            X_combined = X_combined[-max_samples:]
            y_combined = y_combined[-max_samples:]
        
        return self.fit(X_combined, y_combined, level=level)
    
    def recommend_next(self, candidate_pool, n_recommendations=5):
        predictions = self.predict(candidate_pool)
        uncertainty = np.std(predictions, axis=1)
        top_indices = np.argsort(uncertainty)[-n_recommendations:]
        return candidate_pool[top_indices]


class RawResultsPlotter:
    """Handles all plotting and summary generation for raw-features runs."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def analyze_results(self, results, feature_columns, target_scaler, timing_info, use_multioutput=False, metric='mse', sample_ratio=0.3):
        """Analyze and visualize results with detailed plots and summary"""
        
        print(f"\n{'='*80}")
        print("STPLUS RAW FEATURES TRAINING ANALYSIS")
        print(f"{'='*80}")
        print(f"Feature columns: {len(feature_columns)}")
        print(f"Sample ratio: {sample_ratio:.0%}")
        print(f"Metric: {metric.upper()}")
        print(f"Models: {list(results.keys())}")
        print(f"{'='*80}")
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'STPlus Architecture Optimization - Raw Features Training\n'
                    f'{len(feature_columns)} features, Sample ratio: {sample_ratio:.0%}, Metric: {metric.upper()}',
                    fontsize=16, fontweight='bold')
        
        # Style mapping for consistent colors
        style_map = {
            'XGBoost SingleOutput': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
            'CatBoost SingleOutput': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},
            'XGBoost MultiOutput': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},
            'CatBoost MultiOutput': {'color': '#d62728', 'marker': 'D', 'linestyle': ':'}
        }
        
        # Plot 1: Overall Performance by Level
        ax1 = axes[0, 0]
        for optimizer_name, data in results.items():
            if optimizer_name not in style_map:
                continue
                
            levels = []
            avg_scores = []
            
            for level in [1, 2]:  # STPlus has only 2 levels
                if f'level_{level}' in data:
                    level_data = data[f'level_{level}']
                    if metric == 'mse':
                        score = level_data['test_mse']
                    elif metric == 'rmse':
                        score = np.sqrt(level_data['test_mse'])
                    else:  # r2
                        score = level_data['test_r2']
                    
                    levels.append(level)
                    avg_scores.append(score)
            
            if levels and avg_scores:
                style = style_map[optimizer_name]
                ax1.plot(levels, avg_scores, label=optimizer_name, **style, linewidth=2, markersize=8)
        
        ax1.set_xlabel('Complexity Level', fontweight='bold')
        ax1.set_ylabel(f'{metric.upper()} Score', fontweight='bold')
        ax1.set_title('Performance by Complexity Level', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks([1, 2])
        
        # Plot 2: Per-Target Performance (Level 2)
        ax2 = axes[0, 1]
        target_names = ['m1', 'm2', 'p1', 'p2']
        x_pos = np.arange(len(target_names))
        width = 0.2
        
        for i, (optimizer_name, data) in enumerate(results.items()):
            if optimizer_name not in style_map:
                continue
                
            if 'level_2' in data:
                level_data = data['level_2']
                target_scores = []
                
                for target in target_names:
                    if f'test_{target}_{metric}' in level_data:
                        if metric == 'mse':
                            score = level_data[f'test_{target}_mse']
                        elif metric == 'rmse':
                            score = np.sqrt(level_data[f'test_{target}_mse'])
                        else:  # r2
                            score = level_data[f'test_{target}_r2']
                        target_scores.append(score)
                    else:
                        target_scores.append(0)
                
                if target_scores and any(score > 0 for score in target_scores):
                    style = style_map[optimizer_name]
                    ax2.bar(x_pos + i*width, target_scores, width, 
                           label=optimizer_name, color=style['color'], alpha=0.7)
        
        ax2.set_xlabel('Target Variables', fontweight='bold')
        ax2.set_ylabel(f'{metric.upper()} Score', fontweight='bold')
        ax2.set_title(f'Per-Target Performance (Level 2)', fontweight='bold')
        ax2.set_xticks(x_pos + width*1.5)
        ax2.set_xticklabels(target_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Training Time Comparison
        ax3 = axes[1, 0]
        if timing_info:
            optimizer_names = []
            training_times = []
            
            for optimizer_name, data in results.items():
                if optimizer_name in timing_info:
                    optimizer_names.append(optimizer_name)
                    training_times.append(timing_info[optimizer_name])
            
            if optimizer_names and training_times:
                colors = [style_map.get(name, {}).get('color', 'gray') for name in optimizer_names]
                bars = ax3.bar(optimizer_names, training_times, color=colors, alpha=0.7)
                ax3.set_xlabel('Optimizer', fontweight='bold')
                ax3.set_ylabel('Training Time (seconds)', fontweight='bold')
                ax3.set_title('Training Time Comparison', fontweight='bold')
                ax3.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, time_val in zip(bars, training_times):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Model Comparison Summary
        ax4 = axes[1, 1]
        summary_data = []
        model_names = []
        
        for optimizer_name, data in results.items():
            if optimizer_name not in style_map:
                continue
                
            # Calculate overall performance (average across levels)
            level_scores = []
            for level in [1, 2]:
                if f'level_{level}' in data:
                    level_data = data[f'level_{level}']
                    if metric == 'mse':
                        score = level_data['test_mse']
                    elif metric == 'rmse':
                        score = np.sqrt(level_data['test_mse'])
                    else:  # r2
                        score = level_data['test_r2']
                    level_scores.append(score)
            
            if level_scores:
                avg_score = np.mean(level_scores)
                summary_data.append(avg_score)
                model_names.append(optimizer_name)
        
        if summary_data and model_names:
            colors = [style_map.get(name, {}).get('color', 'gray') for name in model_names]
            bars = ax4.barh(model_names, summary_data, color=colors, alpha=0.7)
            ax4.set_xlabel(f'Average {metric.upper()} Score', fontweight='bold')
            ax4.set_ylabel('Model', fontweight='bold')
            ax4.set_title(f'Overall Model Comparison', fontweight='bold')
            
            # Add value labels
            for bar, score in zip(bars, summary_data):
                width = bar.get_width()
                ax4.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{score:.4f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_dir = 'Raw_training'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plot_filename = f'{output_dir}/stplus_raw_features_training_analysis.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"\nDetailed analysis plot saved: {plot_filename}")
        
        plt.show()
        
        # Print detailed summary
        self.print_summary_with_timing(results, timing_info)
    
    def print_summary_with_timing(self, results, timing_info):
        """Print comprehensive summary including timing information"""
        print(f"\n{'='*80}")
        print("STPLUS RAW FEATURES TRAINING SUMMARY")
        print(f"{'='*80}")
        
        for optimizer_name, data in results.items():
            print(f"\n{optimizer_name}:")
            print(f"{'-' * (len(optimizer_name) + 1)}")
            
            # Training time
            if timing_info and optimizer_name in timing_info:
                print(f"  Training Time: {timing_info[optimizer_name]:.2f} seconds")
            
            # Performance by level
            for level in [1, 2]:  # STPlus has only 2 levels
                if f'level_{level}' in data:
                    level_data = data[f'level_{level}']
                    print(f"  Level {level}:")
                    print(f"    Test MSE: {level_data.get('test_mse', 0):.6f}")
                    print(f"    Test RMSE: {np.sqrt(level_data.get('test_mse', 0)):.6f}")
                    print(f"    Test R²: {level_data.get('test_r2', 0):.6f}")
                    
                    # Per-target metrics if available
                    target_names = ['m1', 'm2', 'p1', 'p2']
                    for target in target_names:
                        if f'test_{target}_mse' in level_data:
                            mse = level_data[f'test_{target}_mse']
                            r2 = level_data.get(f'test_{target}_r2', 0)
                            print(f"    {target}: MSE={mse:.6f}, RMSE={np.sqrt(mse):.6f}, R²={r2:.6f}")
        
        print(f"{'='*80}")
    
    def save_training_summary(self, results, feature_columns, target_scaler, timing_info, summary_file):
        """Save comprehensive training summary to file"""
        output_dir = 'Raw_training'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        summary_path = os.path.join(output_dir, summary_file)
        
        with open(summary_path, 'w') as f:
            f.write("STPLUS ARCHITECTURE OPTIMIZATION - RAW FEATURES TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Features: {len(feature_columns)} columns\n")
            f.write(f"Targets: m1, m2, p1, p2 (4 targets)\n")
            f.write(f"Complexity Levels: 2\n")
            f.write("=" * 80 + "\n\n")
            
            for optimizer_name, data in results.items():
                f.write(f"{optimizer_name}:\n")
                f.write(f"{'-' * (len(optimizer_name) + 1)}\n")
                
                # Training time
                if timing_info and optimizer_name in timing_info:
                    f.write(f"  Training Time: {timing_info[optimizer_name]:.2f} seconds\n")
                
                # Performance by level
                for level in [1, 2]:
                    if f'level_{level}' in data:
                        level_data = data[f'level_{level}']
                        f.write(f"  Level {level}:\n")
                        f.write(f"    Test MSE: {level_data.get('test_mse', 0):.6f} (scaled)\n")
                        f.write(f"    Test RMSE: {np.sqrt(level_data.get('test_mse', 0)):.6f} (scaled)\n")
                        f.write(f"    Test R²: {level_data.get('test_r2', 0):.6f} (scaled)\n")
                        
                        # Per-target metrics
                        target_names = ['m1', 'm2', 'p1', 'p2']
                        for target in target_names:
                            if f'test_{target}_mse' in level_data:
                                mse = level_data[f'test_{target}_mse']
                                r2 = level_data.get(f'test_{target}_r2', 0)
                                f.write(f"    {target}: MSE={mse:.6f}, RMSE={np.sqrt(mse):.6f}, R²={r2:.6f} (scaled)\n")
                        f.write("\n")
                
                f.write("\n")
        
        if self.verbose:
            print(f"Training summary saved: {summary_path}")


class STPlus_RawOptimizationTrainer:
    """
    Trainer class for STPlus architecture optimization using raw features.
    
    This class handles the complete pipeline for training and evaluating architecture
    optimization models using the original raw features from the STPlus dataset.
    """
    
    def __init__(self, features_file, verbose=True, metric='mse'):
        self.features_file = features_file
        self.verbose = verbose
        self.metric = metric
        self.data = None
        self.feature_columns = None
        self.target_names = ['m1', 'm2', 'p1', 'p2']  # STPlus targets
        self.complexity_groups = {}
        self.optimizers = {}
        self.results = {}
        self.timing_info = {}
        self.target_scaler = StandardScaler()
    
    def prepare_data(self):
        """Load and prepare the STPlus dataset"""
        if self.verbose:
            print("Loading STPlus dataset...")
        
        self.data = pd.read_csv(self.features_file)
        
        if self.verbose:
            print(f"Dataset loaded: {self.data.shape}")
            print(f"Levels: {sorted(self.data.level.unique())}")
            print(f"Bot types: {sorted(self.data.bot.unique())}")
        
        # Prepare feature columns
        exclude_cols = ['solID', 'level', 'm1', 'm2', 'p1', 'p2', 'bot']
        if self.data.columns[0].startswith('Unnamed') or self.data.columns[0] == '':
            exclude_cols.append(self.data.columns[0])
        
        self.feature_columns = [col for col in self.data.columns if col not in exclude_cols]
        
        if self.verbose:
            print(f"Feature columns: {len(self.feature_columns)}")
            op_cols = len([col for col in self.feature_columns if col.startswith('op')])
            emb_cols = len([col for col in self.feature_columns if col.startswith('emb')])
            print(f"  Operation features: {op_cols}")
            print(f"  Embedding features: {emb_cols}")
        
        # Group data by complexity levels (only 2 levels for STPlus)
        for level in [1, 2]:
            level_data = self.data[self.data['level'] == level]
            if len(level_data) > 0:
                self.complexity_groups[level] = {
                    'features': level_data[self.feature_columns].values,
                    'targets': level_data[self.target_names].values,
                    'level': level
                }
                if self.verbose:
                    print(f"Level {level}: {len(level_data)} samples")
    
    def setup_optimizers(self, save_models=False, models_dir="models", use_multioutput=False):
        """Initialize the optimization models - CatBoost only for performance"""
        if self.verbose:
            print("Setting up optimizers (CatBoost only)...")
        
        self.optimizers = {
            'CatBoost SingleOutput': CatBoostMultiTarget('CatBoost Single', self.verbose, save_models, models_dir)
        }
        
        if use_multioutput:
            self.optimizers.update({
                'CatBoost MultiOutput': CatBoostMultiOutput('CatBoost Multi', self.verbose)
            })
        
        if self.verbose:
            print(f"Optimizers ready: {list(self.optimizers.keys())}")
    
    def run_optimization_comparison(self, max_level=2, train_ratio=0.7, sample_ratio=0.3, save_models=False, models_dir="models", use_multioutput=False):
        """
        Run the optimization comparison with incremental learning.
        
        STPlus version:
        - Only 2 levels instead of 5
        - 4 targets instead of 8
        - Level-by-level incremental training: Level 1 -> Level 2
        """
        
        if not self.optimizers:
            self.setup_optimizers(save_models, models_dir, use_multioutput)
        
        print("="*80)
        print("STPLUS ARCHITECTURE OPTIMIZATION - INCREMENTAL LEARNING")
        print("="*80)
        print(f"Levels: 1 -> 2 (incremental)")
        print(f"Targets: {self.target_names}")
        print(f"Features: {len(self.feature_columns)}")
        print(f"Train ratio: {train_ratio}")
        print(f"Sample ratio: {sample_ratio} (applied to training portion)")
        print(f"Models: {list(self.optimizers.keys())}")
        print("="*80)
        
        # Initialize results storage
        for optimizer_name in self.optimizers.keys():
            self.results[optimizer_name] = {}
        
        # Progressive learning through complexity levels (1 -> 2)
        for current_level in [1, 2]:
            print(f"\n{'='*60}")
            print(f"COMPLEXITY LEVEL {current_level}")
            print(f"{'='*60}")
            
            if current_level not in self.complexity_groups:
                print(f"No data available for level {current_level}, skipping...")
                continue
            
            level_group = self.complexity_groups[current_level]
            X_level = level_group['features']
            y_level = level_group['targets']
            
            print(f"Level {current_level} data: {len(X_level)} samples")
            
            # Split data for this level
            X_train_level, X_test_level, y_train_level, y_test_level = train_test_split(
                X_level, y_level, test_size=(1-train_ratio), random_state=42
            )
            
            # Apply sample ratio to training data
            if sample_ratio < 1.0:
                sample_size = int(len(X_train_level) * sample_ratio)
                indices = np.random.choice(len(X_train_level), sample_size, replace=False)
                X_train_sampled = X_train_level[indices]
                y_train_sampled = y_train_level[indices]
                print(f"Training with {len(X_train_sampled)} samples ({sample_ratio:.0%} of training data)")
            else:
                X_train_sampled = X_train_level
                y_train_sampled = y_train_level
                print(f"Training with {len(X_train_sampled)} samples (100% of training data)")
            
            print(f"Testing with {len(X_test_level)} samples")
            
            # Feature scaling
            if current_level == 1:
                # Initialize scalers on first level
                feature_scaler = StandardScaler()
                X_train_scaled = feature_scaler.fit_transform(X_train_sampled)
                X_test_scaled = feature_scaler.transform(X_test_level)
                
                # Fit target scaler on first level
                self.target_scaler.fit(y_train_sampled)
                y_train_scaled = self.target_scaler.transform(y_train_sampled)
                y_test_scaled = self.target_scaler.transform(y_test_level)
            else:
                # Use existing scalers
                X_train_scaled = feature_scaler.transform(X_train_sampled)
                X_test_scaled = feature_scaler.transform(X_test_level)
                y_train_scaled = self.target_scaler.transform(y_train_sampled)
                y_test_scaled = self.target_scaler.transform(y_test_level)
            
            # Train/Update models
            for optimizer_name, optimizer in self.optimizers.items():
                print(f"\n{optimizer_name}:")
                print("-" * (len(optimizer_name) + 1))
                
                start_time = time.time()
                
                try:
                    if current_level == 1:
                        # Initial training
                        print("  Initial training on Level 1...")
                        optimizer.fit(X_train_scaled, y_train_scaled, level=current_level)
                    else:
                        # Incremental update
                        print(f"  Incremental update with Level {current_level} data...")
                        if hasattr(optimizer, 'update'):
                            optimizer.update(X_train_scaled, y_train_scaled, level=current_level)
                        else:
                            # Fallback to retraining if update not available
                            print("  Update not available, retraining...")
                            optimizer.fit(X_train_scaled, y_train_scaled, level=current_level)
                    
                    # Predict on test set
                    y_pred_scaled = optimizer.predict(X_test_scaled)
                    
                    # Calculate metrics in scaled space (consistent with CoCoME approach)
                    test_mse = mean_squared_error(y_test_scaled, y_pred_scaled)
                    test_rmse = np.sqrt(test_mse)
                    test_r2 = r2_score(y_test_scaled, y_pred_scaled)
                    
                    # Per-target metrics in scaled space
                    per_target_metrics = {}
                    for i, target in enumerate(self.target_names):
                        target_mse = mean_squared_error(y_test_scaled[:, i], y_pred_scaled[:, i])
                        target_r2 = r2_score(y_test_scaled[:, i], y_pred_scaled[:, i])
                        per_target_metrics[f'test_{target}_mse'] = target_mse
                        per_target_metrics[f'test_{target}_r2'] = target_r2
                    
                    # Store results
                    self.results[optimizer_name][f'level_{current_level}'] = {
                        'test_mse': test_mse,
                        'test_rmse': test_rmse,
                        'test_r2': test_r2,
                        **per_target_metrics
                    }
                    
                    # Timing
                    training_time = time.time() - start_time
                    if optimizer_name not in self.timing_info:
                        self.timing_info[optimizer_name] = 0
                    self.timing_info[optimizer_name] += training_time
                    
                    print(f"  Test MSE: {test_mse:.6f} (scaled)")
                    print(f"  Test RMSE: {test_rmse:.6f} (scaled)")
                    print(f"  Test R²: {test_r2:.6f} (scaled)")
                    print(f"  Training time: {training_time:.2f}s")
                    
                    # Per-target performance (scaled space)
                    for target in self.target_names:
                        target_mse = per_target_metrics[f'test_{target}_mse']
                        target_r2 = per_target_metrics[f'test_{target}_r2']
                        print(f"  {target}: MSE={target_mse:.6f}, R²={target_r2:.6f} (scaled)")
                
                except Exception as e:
                    print(f"  Error with {optimizer_name}: {e}")
                    # Store error results
                    self.results[optimizer_name][f'level_{current_level}'] = {
                        'test_mse': float('inf'),
                        'test_rmse': float('inf'),
                        'test_r2': float('-inf'),
                        'error': str(e)
                    }
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETED")
        print(f"{'='*80}")
        
        # Save results summary
        summary_file = f'stplus_raw_training_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        self.save_training_summary(summary_file)
    
    def analyze_results(self, metric='mse', sample_ratio=0.3, use_multioutput=False):
        """Generate comprehensive analysis and visualization"""
        if not self.results:
            print("No results available for analysis!")
            return
        
        plotter = RawResultsPlotter(self.verbose)
        plotter.analyze_results(
            self.results,
            self.feature_columns,
            self.target_scaler,
            self.timing_info,
            use_multioutput,
            metric,
            sample_ratio
        )
    
    def save_training_summary(self, summary_file):
        """Save training summary to file"""
        plotter = RawResultsPlotter(self.verbose)
        plotter.save_training_summary(
            self.results,
            self.feature_columns,
            self.target_scaler,
            self.timing_info,
            summary_file
        )


def main(metric='mse', sample_ratio=0.3, save_models=False, models_dir="models", use_multioutput=False):
    """Main execution function
    
    Args:
        metric (str): Metric to use for evaluation and plotting ('mse', 'rmse', 'r2').
        sample_ratio (float): Fraction of training data to use per level (0.3 = 30%).
        save_models (bool): If True, save all models from each level to the models directory.
        models_dir (str): Directory name where models will be saved (default: "models").
        use_multioutput (bool): If True, include multi-output models in comparison.
    """
    print("STPlus Architecture Optimization Training - RAW FEATURES - CATBOOST SINGLEOUTPUT")
    print("="*80)
    print("Using RAW features from stplus-levels-bots-features.csv (~700+ columns)")
    print("Training CatBoost SingleOutput approach:")
    print("- SingleOutput: Separate models for each target (4 models)")
    print("- Level-specific evaluation: Each level tested on its own complexity")
    print(f"Metric: {metric.upper()}")
    print(f"Training sample ratio: {sample_ratio:.0%} (of 70% train split)")
    print(f"Multi-output models: {'Included' if use_multioutput else 'Not included'}")
    if save_models:
        print(f"Model saving: Enabled (directory: {models_dir})")
    else:
        print("Model saving: Disabled")
    print("="*80)
    
    # Check if raw features file exists
    features_file = 'stplus-levels-bots-features.csv'
    if not os.path.exists(features_file):
        print(f"Error: {features_file} not found!")
        print("Please ensure the STPlus features file is in the current directory.")
        return None
    
    # Initialize trainer
    trainer = STPlus_RawOptimizationTrainer(features_file, verbose=True, metric=metric)
    
    # Run complete training pipeline
    try:
        # Prepare data
        trainer.prepare_data()
        
        # Setup optimizers
        trainer.setup_optimizers(
            save_models=save_models, 
            models_dir=models_dir,
            use_multioutput=use_multioutput
        )
        
        # Run optimization comparison
        trainer.run_optimization_comparison(
            max_level=2,  # STPlus has only 2 levels
            train_ratio=0.7,
            sample_ratio=sample_ratio,
            save_models=save_models,
            models_dir=models_dir,
            use_multioutput=use_multioutput
        )
        
        # Analyze and visualize results
        trainer.analyze_results(
            metric=metric,
            sample_ratio=sample_ratio,
            use_multioutput=use_multioutput
        )
        
        print(f"\n{'='*80}")
        print("STPlus RAW FEATURES TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Results saved in 'Raw_training/' directory")
        print(f"{'='*80}")
        
        return trainer
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Default parameters - modify as needed
    main(
        metric='rmse',          # Metric for evaluation: 'mse', 'rmse', or 'r2'
        sample_ratio=0.8,       # Use 80% of training data at each level
        save_models=False,      # Set to True to save trained models
        models_dir="models",    # Directory for saved models
        use_multioutput=False   # Set to True to include multi-output models
    )