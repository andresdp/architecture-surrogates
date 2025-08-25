"""
Architecture Optimization Training - CoCoME Dataset with Raw Features

This script performs architecture optimization using the original raw features from cocome-levels-features.csv
instead of the engineered features. It uses all 930+ original columns including operation columns (op1_, op2_, op3_)
and embedding columns (emb_).

Key differences from main_cocome_training.py:
- Uses raw features: 930+ columns instead of 16 engineered features
- Includes all boolean operation columns and high-dimensional embeddings
- May require different preprocessing and model configurations due to high dimensionality

Active Learning Features:
- Initial training: Full dataset from complexity level 1
- Incremental updates: Small samples from each subsequent level
- Memory management: Maintains sliding window of recent samples
- Early stopping based on performance improvement
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
import warnings
import os
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
            incremental_model = xgb.XGBRegressor(**xgb_params)
            incremental_model.fit(X_scaled, self.y_train, xgb_model=self.model.get_booster())
            self.model = incremental_model
        else:
            self.model = xgb.XGBRegressor(**xgb_params)
            self.model.fit(X_scaled, self.y_train)
        
        return self
    
    def predict(self, X):
        """Predict single target values"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def update(self, X_new, y_new):
        """ incremental learning: Continue XGBoost training with ONLY new data"""
        y_new_target = y_new[:, self.target_idx]
        X_new_scaled = self.scaler.transform(X_new)
        
        #  incremental learning: Continue training existing model with ONLY new data
        if hasattr(self.model, 'get_booster'):
            # Use xgb_model parameter to continue training from existing model
            xgb_params = {
                'n_estimators': 25,  # Reduced additional trees for high-dim data
                'max_depth': 4, 
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': np.random.randint(0, 10000)
            }
            
            
            
            # Continue training from existing model with ONLY new data
            incremental_model = xgb.XGBRegressor(**xgb_params)
            incremental_model.fit(X_new_scaled, y_new_target, xgb_model=self.model.get_booster())
            
            # Replace the model with the updated one
            self.model = incremental_model
            
            # Update data tracking (for monitoring only - NOT used for retraining)
            if hasattr(self, 'X_train_history') and self.X_train_history is not None:
                self.X_train_history = np.vstack([self.X_train_history, X_new])
                self.y_train_history = np.concatenate([self.y_train_history, y_new_target])
            else:
                self.X_train_history = X_new.copy()
                self.y_train_history = y_new_target.copy()
        
        # Evaluate performance on new data only
        y_pred = self.model.predict(X_new_scaled)
        mse = mean_squared_error(y_new_target, y_pred)
        
        if self.verbose:
            print(f"      {self.target_name}:  incremental update (MSE: {mse:.4f}, New samples: {len(X_new)})")


class XGBoostMultiTarget:
    """XGBoost optimizer with separate models for each target."""
    
    def __init__(self, name, verbose=True):
        self.name = name
        self.target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        self.verbose = verbose
        self.models = {target_name: XGBoostSingleTarget(target_name, i, verbose)
                       for i, target_name in enumerate(self.target_names)}
        
    def fit(self, X, y):
        for target_name in self.target_names:
            self.models[target_name].fit(X, y)
        return self
    
    def predict(self, X):
        predictions = []
        for target_name in self.target_names:
            predictions.append(self.models[target_name].predict(X))
        return np.column_stack(predictions)
    
    def update(self, X_new, y_new):
        for target_name in self.target_names:
            self.models[target_name].update(X_new, y_new)
    
    def recommend_next(self, candidate_pool, n_recommendations=5):
        predictions = self.predict(candidate_pool)
        weights = np.array([1/8] * 8)
        scores = np.sum(predictions * weights, axis=1)
        top_indices = np.argsort(scores)[:n_recommendations]
        return candidate_pool[top_indices], predictions[top_indices]


    


class XGBoostMultiOutput:
    """XGBoost optimizer with single multi-output model for all targets"""
    
    def __init__(self, name, verbose=True):
        self.name = name
        self.target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        self.verbose = verbose
        self.model = None
        self.scaler = StandardScaler()
        self.X_train_history = None
        self.y_train_history = None
        
    def fit(self, X, y):
        """Fit single XGBoost model for all targets simultaneously"""
        self.X_train_history = X.copy()
        self.y_train_history = y.copy()
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Adjusted parameters for high-dimensional data
        xgb_params = {
            'n_estimators': 50,  # Reduced for high-dim data
            'max_depth': 4,      # Reduced to prevent overfitting
            'learning_rate': 0.1,
            'subsample': 0.8,    # Added subsampling
            'colsample_bytree': 0.8,  # Added column subsampling
            'reg_alpha': 0.1,    # Added L1 regularization
            'reg_lambda': 1.0,   # Added L2 regularization
            'random_state': np.random.randint(0, 10000),
            'objective': 'reg:squarederror'
        }
        
        # Use MultiOutputRegressor to handle multiple targets properly
        base_model = xgb.XGBRegressor(**xgb_params)
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_scaled, y)
        
        if self.verbose:
            print(f"      Multi-output XGBoost: Trained on all {y.shape[1]} targets simultaneously")
        
        return self
    
    def predict(self, X):
        """Predict all targets using single multi-output model"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Ensure predictions have correct shape for multi-output
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        if predictions.shape[1] != len(self.target_names):
            # Handle case where XGBoost returns single column for multi-output
            predictions = np.tile(predictions, (1, len(self.target_names)))
        
        return predictions
    
    def update(self, X_new, y_new):
        """ incremental learning: Continue XGBoost training with ONLY new data"""
        X_new_scaled = self.scaler.transform(X_new)
        
        # Update history for monitoring only (not used for retraining)
        if self.X_train_history is not None:
            self.X_train_history = np.vstack([self.X_train_history, X_new])
            self.y_train_history = np.vstack([self.y_train_history, y_new])
        else:
            self.X_train_history = X_new.copy()
            self.y_train_history = y_new.copy()
        
        # Continue training each target estimator from previous booster with ONLY new data
        # MultiOutputRegressor wraps estimators_ list after fitting
        new_estimators = []
        for i, estimator in enumerate(self.model.estimators_):
            # Prepare per-target y
            y_new_target = y_new[:, i]
            # Create a new estimator and continue from previous booster
            xgb_params = estimator.get_params()
            incremental_model = xgb.XGBRegressor(**xgb_params)
            incremental_model.fit(X_new_scaled, y_new_target, xgb_model=estimator.get_booster())
            new_estimators.append(incremental_model)
        self.model.estimators_ = new_estimators
        
        # Evaluate performance on new data only
        y_pred = self.model.predict(X_new_scaled)
        mse = mean_squared_error(y_new, y_pred)
        
        if self.verbose:
            print(f"      Multi-output XGBoost: TRUE incremental update (MSE: {mse:.4f}, New samples: {len(X_new)})")
    
    def recommend_next(self, candidate_pool, n_recommendations=5):
        """Recommend next architectures based on predictions"""
        predictions = self.predict(candidate_pool)
        weights = np.array([1/8] * 8)
        scores = np.sum(predictions * weights, axis=1)
        top_indices = np.argsort(scores)[:n_recommendations]
        return candidate_pool[top_indices], predictions[top_indices]


class LGBMMultiOutput:
    """LightGBM optimizer with single multi-output model (wrapped) and TRUE incremental updates."""
    
    def __init__(self, name, verbose=True):
        self.name = name
        self.target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        self.verbose = verbose
        self.model = None
        self.scaler = StandardScaler()
        self.X_train_history = None
        self.y_train_history = None
    
    def fit(self, X, y):
        self.X_train_history = X.copy()
        self.y_train_history = y.copy()
        X_scaled = self.scaler.fit_transform(X)
        lgbm_params = {
            'n_estimators': 50,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': np.random.randint(0, 10000)
        }
        base_model = LGBMRegressor(**lgbm_params)
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_scaled, y)
        if self.verbose:
            print(f"      Multi-output LGBM: Trained on all {y.shape[1]} targets simultaneously")
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        if predictions.shape[1] != len(self.target_names):
            predictions = np.tile(predictions, (1, len(self.target_names)))
        return predictions
    
    def update(self, X_new, y_new):
        X_new_scaled = self.scaler.transform(X_new)
        # Monitor-only history
        if self.X_train_history is not None:
            self.X_train_history = np.vstack([self.X_train_history, X_new])
            self.y_train_history = np.vstack([self.y_train_history, y_new])
        else:
            self.X_train_history = X_new.copy()
            self.y_train_history = y_new.copy()
        # Continue training each target using init_model
        new_estimators = []
        for i, estimator in enumerate(self.model.estimators_):
            y_new_target = y_new[:, i]
            params = estimator.get_params()
            incremental_model = LGBMRegressor(**params)
            # Prefer passing booster_ if available
            init_model = getattr(estimator, 'booster_', None) or estimator
            incremental_model.fit(X_new_scaled, y_new_target, init_model=init_model)
            new_estimators.append(incremental_model)
        self.model.estimators_ = new_estimators
        y_pred = self.model.predict(X_new_scaled)
        mse = mean_squared_error(y_new, y_pred)
        if self.verbose:
            print(f"      Multi-output LGBM: TRUE incremental update (MSE: {mse:.4f}, New samples: {len(X_new)})")
    
    def recommend_next(self, candidate_pool, n_recommendations=5):
        predictions = self.predict(candidate_pool)
        weights = np.array([1/8] * 8)
        scores = np.sum(predictions * weights, axis=1)
        top_indices = np.argsort(scores)[:n_recommendations]
        return candidate_pool[top_indices], predictions[top_indices]


class CatBoostMultiOutput:
    """CatBoost optimizer with single multi-output model (wrapped) and TRUE incremental updates."""
    
    def __init__(self, name, verbose=True):
        self.name = name
        self.target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        self.verbose = verbose
        self.model = None
        self.scaler = StandardScaler()
        self.X_train_history = None
        self.y_train_history = None
    
    def fit(self, X, y):
        self.X_train_history = X.copy()
        self.y_train_history = y.copy()
        X_scaled = self.scaler.fit_transform(X)
        cb_params = {
            'iterations': 50,
            'depth': 4,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3.0,
            'random_seed': int(np.random.randint(0, 10000)),
            'loss_function': 'RMSE',
            'verbose': False
        }
        base_model = CatBoostRegressor(**cb_params)
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_scaled, y)
        if self.verbose:
            print(f"      Multi-output CatBoost: Trained on all {y.shape[1]} targets simultaneously")
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        if predictions.shape[1] != len(self.target_names):
            predictions = np.tile(predictions, (1, len(self.target_names)))
        return predictions
    
    def update(self, X_new, y_new):
        X_new_scaled = self.scaler.transform(X_new)
        # Monitor-only history
        if self.X_train_history is not None:
            self.X_train_history = np.vstack([self.X_train_history, X_new])
            self.y_train_history = np.vstack([self.y_train_history, y_new])
        else:
            self.X_train_history = X_new.copy()
            self.y_train_history = y_new.copy()
        # Continue training per-target using init_model
        new_estimators = []
        for i, estimator in enumerate(self.model.estimators_):
            y_new_target = y_new[:, i]
            params = estimator.get_params()
            incremental_model = CatBoostRegressor(**params)
            incremental_model.fit(X_new_scaled, y_new_target, init_model=estimator)
            new_estimators.append(incremental_model)
        self.model.estimators_ = new_estimators
        y_pred = self.model.predict(X_new_scaled)
        mse = mean_squared_error(y_new, y_pred)
        if self.verbose:
            print(f"      Multi-output CatBoost: TRUE incremental update (MSE: {mse:.4f}, New samples: {len(X_new)})")
    
    def recommend_next(self, candidate_pool, n_recommendations=5):
        predictions = self.predict(candidate_pool)
        weights = np.array([1/8] * 8)
        scores = np.sum(predictions * weights, axis=1)
        top_indices = np.argsort(scores)[:n_recommendations]
        return candidate_pool[top_indices], predictions[top_indices]


class RawResultsPlotter:
    """Handles all plotting and summary generation for raw-features runs."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def analyze_results(self, results, feature_columns, target_scaler, timing_info, use_multioutput=False):
        if self.verbose:
            print("\n" + "="*80)
            print("ANALYZING RAW FEATURES RESULTS")
            print("="*80)
        
        style_map = {}
        for name in results.keys():
            if name == 'XGBoost_MultiOutput':
                style_map[name] = {'linestyle': '-', 'marker': 's', 'color': 'green', 'label': 'XGBoost_MultiOutput'}
            elif name == 'XGBoost_SingleOutput':
                style_map[name] = {'linestyle': '-', 'marker': 'o', 'color': 'blue', 'label': 'XGBoost_SingleOutput'}
            elif name == 'LightGBM_MultiOutput':
                style_map[name] = {'linestyle': '-', 'marker': 'D', 'color': 'purple', 'label': 'LightGBM_MultiOutput'}
            elif name == 'CatBoost_MultiOutput':
                style_map[name] = {'linestyle': '-', 'marker': '^', 'color': 'orange', 'label': 'CatBoost_MultiOutput'}
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        axes[0, 0].set_title('Overall Test MSE (Raw Features, Standardized Targets)', fontsize=14, fontweight='bold')
        for name, res in results.items():
            if res['mse_scores_test_scaled'] and name in style_map:
                style = style_map[name]
                axes[0, 0].plot(res['mse_scores_test_scaled'],
                                label=style.get('label', name),
                                linestyle=style['linestyle'],
                                marker=style['marker'],
                                color=style['color'],
                                markersize=6,
                                linewidth=2)
        axes[0, 0].set_xlabel('Complexity Level')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        objectives = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        plot_positions = [(0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)]
        for i, obj in enumerate(objectives):
            row, col = plot_positions[i]
            axes[row, col].set_title(f'{obj} - MSE (Raw Features)', fontsize=14, fontweight='bold')
            for name, res in results.items():
                if res['test_mse_scaled'][obj] and name in style_map:
                    style = style_map[name]
                    axes[row, col].plot(res['test_mse_scaled'][obj],
                                        label=style.get('label', name),
                                        linestyle=style['linestyle'],
                                        marker=style['marker'],
                                        color=style['color'],
                                        markersize=6,
                                        linewidth=2)
            axes[row, col].set_xlabel('Complexity Level')
            axes[row, col].set_ylabel('MSE')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_yscale('log')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        approach_suffix = "Combined_Raw"
        summary_file = f'Raw_training/cocome_training_summary_{approach_suffix}_{timestamp}.txt'
        self.save_training_summary(results, feature_columns, target_scaler, timing_info, summary_file)
        
        fig.delaxes(axes[0, 1])
        fig.delaxes(axes[2, 2])
        fig.delaxes(axes[2, 3])
        
        plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'Raw_training/cocome_training_results_{approach_suffix}_{timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if self.verbose:
            print(f"Results visualization saved: {output_file}")
        plt.show()
        
        self.print_summary(results)
    
    def save_training_summary(self, results, feature_columns, target_scaler, timing_info, filename):
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COCOME RAW FEATURES TRAINING SUMMARY -  INCREMENTAL LEARNING\n")
            f.write("="*80 + "\n")
            f.write("IMPORTANT: This uses RAW FEATURES (930+ columns) with  incremental learning:\n")
            f.write("- XGBoost: Continues training from previous model with ONLY new level data\n")
            f.write("- Historical data is tracked for monitoring only, NOT used for retraining\n")
            f.write("- Each level trains/updates models using ONLY that level's new samples\n")
            f.write("- Targets (8 metrics) are standardized globally via StandardScaler before training;\n")
            f.write("  evaluations and plots are reported in original target units (inverse-transformed).\n")
            f.write(f"- Feature dimensionality: {len(feature_columns)} raw features\n")
            f.write("="*80 + "\n\n")
            
            op1_cols = [col for col in feature_columns if col.startswith('op1_')]
            op2_cols = [col for col in feature_columns if col.startswith('op2_')]
            op3_cols = [col for col in feature_columns if col.startswith('op3_')]
            emb_cols = [col for col in feature_columns if col.startswith('emb_')]
            other_cols = [col for col in feature_columns if not any(col.startswith(prefix) for prefix in ['op1_', 'op2_', 'op3_', 'emb_'])]
            f.write("FEATURE BREAKDOWN:\n")
            f.write("="*80 + "\n")
            f.write(f"Total Features: {len(feature_columns)}\n")
            f.write(f"  • op1_ columns (boolean operations): {len(op1_cols)}\n")
            f.write(f"  • op2_ columns (boolean operations): {len(op2_cols)}\n")
            f.write(f"  • op3_ columns (boolean operations): {len(op3_cols)}\n")
            f.write(f"  • emb_ columns (embeddings): {len(emb_cols)}\n")
            f.write(f"  • other columns: {len(other_cols)}\n")
            f.write("="*80 + "\n\n")
            
            if timing_info:
                f.write("TRAINING TIMING ANALYSIS\n")
                f.write("="*80 + "\n")
                overall_time = timing_info.get('overall_time', 0)
                f.write(f"Total Training Time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)\n")
                f.write(f"Data Preparation Time: {timing_info.get('data_preparation_time', 0):.2f} seconds\n")
                f.write(f"Start Time: {datetime.fromtimestamp(timing_info.get('overall_start_time', 0)).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"End Time: {datetime.fromtimestamp(timing_info.get('overall_end_time', 0)).strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                if 'level_times' in timing_info:
                    f.write("PER-LEVEL TIMING:\n")
                    f.write("-" * 40 + "\n")
                    for level, level_time in sorted(timing_info['level_times'].items()):
                        f.write(f"Level {level}: {level_time:.2f} seconds\n")
                    f.write("\n")
                if 'optimizer_times' in timing_info:
                    f.write("PER-OPTIMIZER TIMING:\n")
                    f.write("-" * 40 + "\n")
                    for optimizer_name, timing_data in timing_info['optimizer_times'].items():
                        total_time = timing_data.get('total_time', 0)
                        f.write(f"{optimizer_name}:\n")
                        f.write(f"  Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
                        f.write(f"  Average Time per Level: {total_time/len(timing_data.get('level_times', [1])):.2f} seconds\n")
                        if 'level_times' in timing_data:
                            f.write("  Level Breakdown:\n")
                            for level, time_taken in sorted(timing_data['level_times'].items()):
                                f.write(f"    Level {level}: {time_taken:.2f} seconds\n")
                        f.write("\n")
                f.write("="*80 + "\n\n")
            
            first_optimizer = list(results.keys())[0]
            sample_details = results[first_optimizer]['sample_details']
            f.write("SAMPLE USAGE BY LEVEL ( INCREMENTAL LEARNING - RAW FEATURES):\n")
            f.write("-" * 60 + "\n")
            f.write("NOTE: Models train ONLY on current level data, previous data is tracked for monitoring only\n\n")
            for details in sample_details:
                level = details['level']
                if details.get('skipped', False):
                    f.write(f"Level {level}: SKIPPED\n")
                    f.write(f"  Available: {details['total_level_samples']} (insufficient for train/test split)\n\n")
            else:
                    f.write(f"Level {level}:\n")
                    f.write(f"  Available: {details['total_level_samples']} samples\n")
                    f.write(f"  Training:  {details['current_level_train']} (ONLY current level data used)\n")
                    f.write(f"  Testing:   {details['current_level_test']} (ONLY current level data used)\n")
                    f.write(f"  Validation: {details['current_level_validation']} (held out, not used)\n")
                    if details['previous_level_retained'] > 0:
                        f.write(f"  Historical: {details['previous_level_retained']} (tracked for monitoring only)\n")
                        f.write(f"  Total Seen: {details['total_model_samples']} (historical tracking only)\n")
                    f.write("\n")
            total_train = sum([d['current_level_train'] for d in sample_details])
            total_test = sum([d['current_level_test'] for d in sample_details])
            total_validation = sum([d['current_level_validation'] for d in sample_details])
            f.write("OVERALL TOTALS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Training: {total_train} samples\n")
            f.write(f"Testing:  {total_test} samples\n")
            f.write(f"Validation: {total_validation} samples\n\n")
            target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
            for name, res in results.items():
                f.write("="*80 + "\n")
                f.write(f"{name.upper()} RESULTS (RAW FEATURES)\n")
                f.write("="*80 + "\n")
                if res['best_scores']:
                    f.write("Overall Performance:\n")
                    f.write(f"  Final Best Score: {res['best_scores'][-1]:.4f}\n")
                    f.write(f"  Final Training MSE: {res['mse_scores'][-1]:.4f}\n")
                    if all(res['test_mse'][target] for target in target_names):
                        final_test_mse = np.mean([res['test_mse'][target][-1] for target in target_names])
                        f.write(f"  Final Test MSE (avg): {final_test_mse:.4f}\n")
                f.write("\nIndividual Model Performance (Test Set):\n")
                for target in target_names:
                    if res['test_mse'][target]:
                        final_test_mse = res['test_mse'][target][-1]
                        f.write(f"  {target:4}: Final Test MSE = {final_test_mse:.4f}\n")
                f.write("\n")
        if self.verbose:
            print(f"Training summary saved: {filename}")
    
    def print_summary(self, results):
        print("\n" + "="*80)
        print("DETAILED RAW FEATURES OPTIMIZATION RESULTS SUMMARY")
        print("="*80)
        target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        for name, res in results.items():
            print(f"\n{name}:")
            print("-" * 50)
            if res['best_scores']:
                print("  Overall Performance:")
                print(f"    Final Best Score: {res['best_scores'][-1]:.4f}")
                print(f"    Final Training MSE: {res['mse_scores'][-1]:.4f}")
                if all(res['test_mse'][target] for target in target_names):
                    final_test_mse = np.mean([res['test_mse'][target][-1] for target in target_names])
                    print(f"    Final Test MSE (avg): {final_test_mse:.4f}")
            print("\n  Individual Model Performance (Test Set):")
            for target in target_names:
                if res['test_mse'][target]:
                    final_test_mse = res['test_mse'][target][-1]
                    print(f"    {target:4}: Final Test MSE = {final_test_mse:.4f}")
 


class CoCoMERawOptimizationTrainer:
    """Training pipeline for CoCoME optimization using raw features"""
    
    def __init__(self, raw_features_file, verbose=True):
        self.data = pd.read_csv(raw_features_file)
        self.features = None
        self.objectives = None
        self.objectives_scaled = None
        self.optimizers = {}
        self.results = {}
        self.verbose = verbose
        self.feature_columns = None
        self.timing_info = {}
        self.target_scaler = StandardScaler()
        
        if verbose:
            print(f"Loaded raw features: {self.data.shape}")
            
    def prepare_data(self):
        """Prepare features and objectives from raw dataset"""
        if self.verbose:
            print("Preparing data from raw features...")
        
        # Identify feature columns (exclude metadata and targets)
        # The first column is unnamed index, then solID, target metrics, level
        exclude_cols = ['solID', 'level', 'm1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        
        # Also exclude the unnamed index column (first column)
        if self.data.columns[0].startswith('Unnamed') or self.data.columns[0] == '':
            exclude_cols.append(self.data.columns[0])
        
        self.feature_columns = [col for col in self.data.columns if col not in exclude_cols]
        
        # Extract features and objectives
        self.features = self.data[self.feature_columns].values
        target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        self.objectives = self.data[target_names].values
        # Fit global scaler on targets and store scaled version for training
        self.objectives_scaled = self.target_scaler.fit_transform(self.objectives)
        
        if self.verbose:
            print(f"Features: {self.features.shape} ({len(self.feature_columns)} raw features)")
            print(f"Objectives: {self.objectives.shape} (8 target metrics)")
            print(f"Feature types breakdown:")
            
            # Count different types of features
            op1_cols = [col for col in self.feature_columns if col.startswith('op1_')]
            op2_cols = [col for col in self.feature_columns if col.startswith('op2_')]
            op3_cols = [col for col in self.feature_columns if col.startswith('op3_')]
            emb_cols = [col for col in self.feature_columns if col.startswith('emb_')]
            other_cols = [col for col in self.feature_columns if not any(col.startswith(prefix) for prefix in ['op1_', 'op2_', 'op3_', 'emb_'])]
            
            print(f"  • op1_ columns: {len(op1_cols)}")
            print(f"  • op2_ columns: {len(op2_cols)}")
            print(f"  • op3_ columns: {len(op3_cols)}")
            print(f"  • emb_ columns: {len(emb_cols)}")
            print(f"  • other columns: {len(other_cols)}")
        
        return self
    
    def setup_optimizers(self, use_multioutput=False):
        """Setup optimization algorithms.
        Now configures both Single-output (separate models) and Multi-output models
        to enable side-by-side comparison in plots.
        """
        if self.verbose:
            print("Setting up optimizers for raw features...")
            print("Comparing models: XGBoost (Single/Multi-output), LightGBM (Multi-output), CatBoost (Multi-output)")
        
        self.optimizers = {
            'XGBoost_SingleOutput': XGBoostMultiTarget('XGBoost Separate Models (Raw)', verbose=self.verbose),
            'XGBoost_MultiOutput': XGBoostMultiOutput('XGBoost Multi-Output (Raw)', verbose=self.verbose),
            'LightGBM_MultiOutput': LGBMMultiOutput('LightGBM Multi-Output (Raw)', verbose=self.verbose),
            'CatBoost_MultiOutput': CatBoostMultiOutput('CatBoost Multi-Output (Raw)', verbose=self.verbose)
        }
        
        return self

    def group_by_complexity(self):
        """Group data by complexity levels"""
        complexity_groups = {}
        for level in range(1, 6):  # Levels 1-5
            mask = self.data['level'] == level
            if mask.any():
                complexity_groups[level] = {
                    'features': self.features[mask],
                    'objectives': self.objectives[mask],
                    'objectives_scaled': self.objectives_scaled[mask],
                    'indices': self.data[mask].index.tolist()
                }
        return complexity_groups

    def run_optimization_comparison(self, max_level=5, train_ratio=0.7, sample_ratio=0.3, use_multioutput=False):
        """Run the full optimization comparison with hierarchical active learning"""
        # Start overall timing
        overall_start_time = time.time()
        
        if self.verbose:
            print("\n" + "="*80)
            print("STARTING COCOME RAW FEATURES OPTIMIZATION COMPARISON")
            print("="*80)
            print(f"Max levels: {max_level}")
            print(f"Train/test ratio: {train_ratio:.0%}/{1-train_ratio:.0%}")
            print(f"Training sample ratio: {sample_ratio:.0%}")
            print(f"Using RAW FEATURES: {len(self.feature_columns)} columns")
            print()
        
        complexity_groups = self.group_by_complexity()
        
        # Initialize timing tracking
        self.timing_info = {
            'overall_start_time': overall_start_time,
            'level_times': {},
            'optimizer_times': {name: {'total_time': 0, 'level_times': {}} for name in self.optimizers.keys()},
            'data_preparation_time': 0,
            'evaluation_time': 0
        }
        
        # Initialize results with enhanced sample tracking
        target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        for name in self.optimizers.keys():
            self.results[name] = {
                'best_scores': [],
                'mse_scores': [],  # overall train MSE (raw)
                'mse_scores_test': [],  # overall test MSE (raw)
                'mse_scores_scaled': [],  # overall train MSE (scaled)
                'mse_scores_test_scaled': [],  # overall test MSE (scaled)
                'level_performances': {},
                'convergence': [],
                'individual_mse': {target: [] for target in target_names},  # per-target train MSE (raw)
                'individual_mse_scaled': {target: [] for target in target_names},  # per-target train MSE (scaled)
                'test_mse': {target: [] for target in target_names},  # per-target test MSE (raw)
                'test_mse_scaled': {target: [] for target in target_names},  # per-target test MSE (scaled)
                'training_samples_used': [],
                'test_samples_used': [],
                'sample_details': []  # New: detailed sample tracking per level
            }
        
        # Progressive learning through complexity levels
        for level in range(1, max_level + 1):
            level_start_time = time.time()
            
            if self.verbose:
                print(f"\n=== LEVEL {level} ===")
            
            if level not in complexity_groups:
                if self.verbose:
                    print(f"No architectures found for level {level}")
                continue
                
            level_data = complexity_groups[level]
            X_level = level_data['features']
            y_level = level_data['objectives']
            y_level_scaled = level_data['objectives_scaled']
            
            if self.verbose:
                print(f"Total architectures at level {level}: {len(X_level)}")
            
            # Skip levels with insufficient samples
            if len(X_level) < 5:
                if self.verbose:
                    print(f"  Skipping level {level} - insufficient samples ({len(X_level)})")
                for name in self.optimizers.keys():
                    self.results[name]['best_scores'].append(float('inf'))
                    self.results[name]['mse_scores'].append(float('inf'))
                    for target in target_names:
                        self.results[name]['test_mse'][target].append(float('inf'))
                        self.results[name]['individual_mse'][target].append(float('inf'))
                    self.results[name]['training_samples_used'].append(0)
                    self.results[name]['test_samples_used'].append(0)
                    
                    # Add sample details for skipped level
                    skipped_sample_details = {
                        'level': level,
                        'total_level_samples': len(X_level),
                        'current_level_train': 0,
                        'current_level_test': 0,
                        'current_level_validation': 0,
                        'previous_level_retained': 0,
                        'total_model_samples': 0,
                        'skipped': True
                    }
                    self.results[name]['sample_details'].append(skipped_sample_details)
                continue
            
            # Data preparation timing
            data_prep_start = time.time()
            
            # Train/test split (use different random state each time)
            X_train_full, X_test, y_train_full, y_test, y_train_full_scaled, y_test_scaled = train_test_split(
                X_level, y_level, y_level_scaled, test_size=(1-train_ratio), random_state=None
            )
            
            # Sample from training data (truly random sampling)
            n_train_samples = max(1, int(len(X_train_full) * sample_ratio))
            train_indices = np.random.choice(len(X_train_full), size=n_train_samples, replace=False)
            X_train = X_train_full[train_indices]
            y_train = y_train_full[train_indices]
            y_train_scaled = y_train_full_scaled[train_indices]
            
            data_prep_time = time.time() - data_prep_start
            self.timing_info['data_preparation_time'] += data_prep_time
            
            if self.verbose:
                print(f"  Train/Test split: {len(X_train_full)}/{len(X_test)} samples")
                print(f"  Training with {len(X_train)} samples ({sample_ratio*100:.0f}% of train set)")
                print(f"  Feature dimensionality: {X_train.shape[1]} raw features")
            
            # Train each optimizer
            for name, optimizer in self.optimizers.items():
                optimizer_start_time = time.time()
                
                if self.verbose:
                    print(f"  Training {name}...")
                
                try:
                    # Handle both single-model and multi-model optimizers
                    if hasattr(optimizer, 'models'):
                        # Multi-target optimizer with separate models
                        first_training = all(model.model is None for model in optimizer.models.values())
                    else:
                        # Single multi-output model
                        first_training = optimizer.model is None
                    
                    if first_training:
                        optimizer.fit(X_train, y_train_scaled)
                        if self.verbose:
                            print(f"    Trained from scratch: {len(X_train)} samples")
                    else:
                        optimizer.update(X_train, y_train_scaled)
                    
                    # Evaluate performance
                    y_train_pred_scaled = optimizer.predict(X_train)
                    # Inverse-transform predictions to original target scale
                    y_train_pred = self.target_scaler.inverse_transform(y_train_pred_scaled)
                    train_mse = mean_squared_error(y_train, y_train_pred)
                    train_mse_scaled = mean_squared_error(y_train_scaled, y_train_pred_scaled)
                    
                    y_test_pred_scaled = optimizer.predict(X_test)
                    y_test_pred = self.target_scaler.inverse_transform(y_test_pred_scaled)
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    test_mse_scaled = mean_squared_error(y_test_scaled, y_test_pred_scaled)
                    
                    # Individual model performance
                    individual_test_mse = {}
                    for i, target in enumerate(target_names):
                        target_train_mse = mean_squared_error(y_train[:, i], y_train_pred[:, i])
                        target_train_mse_scaled = mean_squared_error(y_train_scaled[:, i], y_train_pred_scaled[:, i])
                        self.results[name]['individual_mse'][target].append(target_train_mse)
                        self.results[name]['individual_mse_scaled'][target].append(target_train_mse_scaled)
                        
                        target_test_mse = mean_squared_error(y_test[:, i], y_test_pred[:, i])
                        target_test_mse_scaled = mean_squared_error(y_test_scaled[:, i], y_test_pred_scaled[:, i])
                        individual_test_mse[target] = target_test_mse
                        self.results[name]['test_mse'][target].append(target_test_mse)
                        self.results[name]['test_mse_scaled'][target].append(target_test_mse_scaled)
                        
                        if self.verbose:
                            print(f"      {target}: Train MSE={target_train_mse:.4f}, "
                                  f"Test MSE={target_test_mse:.4f}")
                    
                    # Track best scores - handle both optimizer types
                    best_scores = []
                    if hasattr(optimizer, 'models'):
                        # Multi-target optimizer with separate models
                        best_scaled = []
                        for target_name in target_names:
                            model = optimizer.models[target_name]
                            if hasattr(model, 'y_train') and model.y_train is not None and len(model.y_train) > 0:
                                best_scaled.append(np.min(model.y_train))
                            else:
                                best_scaled.append(float('inf'))
                        # Convert best scaled targets back to raw scale where possible
                        if not any(np.isinf(best_scaled)):
                            best_raw = self.target_scaler.inverse_transform(np.array(best_scaled).reshape(1, -1))[0]
                            best_scores = list(best_raw)
                        else:
                            best_scores = [float('inf')] * len(target_names)
                    else:
                        # Single multi-output model
                        if hasattr(optimizer, 'y_train_history') and optimizer.y_train_history is not None and len(optimizer.y_train_history) > 0:
                            # For multi-output, calculate best score for each target column (currently scaled)
                            best_scaled = [np.min(optimizer.y_train_history[:, i]) for i in range(len(target_names))]
                            best_raw = self.target_scaler.inverse_transform(np.array(best_scaled).reshape(1, -1))[0]
                            best_scores = list(best_raw)
                        else:
                            best_scores = [float('inf')] * len(target_names)
                    
                    current_best = np.sum(best_scores)
                    
                    # Calculate detailed sample tracking
                    current_level_train_samples = len(X_train)
                    current_level_test_samples = len(X_test)
                    validation_samples = len(X_train_full) - len(X_train)  # Unused training samples
                    
                    # Get accumulated samples from incremental learning - handle both optimizer types
                    previous_level_samples = 0
                    total_model_samples = 0
                    
                    if hasattr(optimizer, 'models'):
                        # Multi-target optimizer with separate models
                        first_model = list(optimizer.models.values())[0]
                        if hasattr(first_model, 'X_train_history') and first_model.X_train_history is not None:
                            total_model_samples = len(first_model.X_train_history)
                            if level > 1:  # Only count previous samples if not first level
                                previous_level_samples = total_model_samples - current_level_train_samples
                        elif hasattr(first_model, 'y_train_history') and first_model.y_train_history is not None:
                            total_model_samples = len(first_model.y_train_history)
                            if level > 1:  # Only count previous samples if not first level
                                previous_level_samples = total_model_samples - current_level_train_samples
                    else:
                        # Single multi-output model
                        if hasattr(optimizer, 'X_train_history') and optimizer.X_train_history is not None:
                            total_model_samples = len(optimizer.X_train_history)
                            if level > 1:  # Only count previous samples if not first level
                                previous_level_samples = total_model_samples - current_level_train_samples
                        elif hasattr(optimizer, 'y_train_history') and optimizer.y_train_history is not None:
                            total_model_samples = len(optimizer.y_train_history)
                            if level > 1:  # Only count previous samples if not first level
                                previous_level_samples = total_model_samples - current_level_train_samples
                    
                    sample_details = {
                        'level': level,
                        'total_level_samples': len(X_level),
                        'current_level_train': current_level_train_samples,
                        'current_level_test': current_level_test_samples,
                        'current_level_validation': validation_samples,
                        'previous_level_retained': previous_level_samples,
                        'total_model_samples': total_model_samples
                    }
                    
                    self.results[name]['best_scores'].append(current_best)
                    self.results[name]['mse_scores'].append(train_mse)
                    self.results[name]['mse_scores_scaled'].append(train_mse_scaled)
                    self.results[name]['mse_scores_test'].append(test_mse)
                    self.results[name]['mse_scores_test_scaled'].append(test_mse_scaled)
                    self.results[name]['training_samples_used'].append(len(X_train))
                    self.results[name]['test_samples_used'].append(len(X_test))
                    self.results[name]['sample_details'].append(sample_details)
                    self.results[name]['level_performances'][level] = {
                        'train_mse': train_mse,
                        'test_mse': test_mse,
                        'individual_test_mse': individual_test_mse,
                        'best_score': current_best,
                        'num_architectures': len(X_level),
                        'train_samples': len(X_train),
                        'test_samples': len(X_test),
                        'sample_details': sample_details
                    }
                    
                    if self.verbose:
                        print(f"    Test MSE (standardized): {test_mse_scaled:.4f}, Train MSE (standardized): {train_mse_scaled:.4f}")
                    
                    # Record timing for this optimizer at this level
                    optimizer_time = time.time() - optimizer_start_time
                    self.timing_info['optimizer_times'][name]['total_time'] += optimizer_time
                    self.timing_info['optimizer_times'][name]['level_times'][level] = optimizer_time
                    
                    if self.verbose:
                        print(f"    Training time: {optimizer_time:.2f} seconds")
                    
                except Exception as e:
                    if self.verbose:
                        print(f"    Error with {name}: {e}")
                        import traceback
                        traceback.print_exc()
                    # Handle errors gracefully
                    self.results[name]['best_scores'].append(float('inf'))
                    self.results[name]['mse_scores'].append(float('inf'))
                    for target in target_names:
                        self.results[name]['test_mse'][target].append(float('inf'))
                        self.results[name]['individual_mse'][target].append(float('inf'))
                    self.results[name]['training_samples_used'].append(0)
                    self.results[name]['test_samples_used'].append(0)
                    
                    # Add empty sample details for error case
                    error_sample_details = {
                        'level': level,
                        'total_level_samples': len(X_level) if 'X_level' in locals() else 0,
                        'current_level_train': 0,
                        'current_level_test': 0,
                        'current_level_validation': 0,
                        'previous_level_retained': 0,
                        'total_model_samples': 0
                    }
                    self.results[name]['sample_details'].append(error_sample_details)
            
            # Record timing for this level
            level_time = time.time() - level_start_time
            self.timing_info['level_times'][level] = level_time
            
            if self.verbose:
                print(f"  Level {level} completed in {level_time:.2f} seconds")
        
        # Record overall timing
        overall_time = time.time() - overall_start_time
        self.timing_info['overall_time'] = overall_time
        self.timing_info['overall_end_time'] = time.time()
        
        if self.verbose:
            print(f"\n" + "="*80)
            print(f"RAW FEATURES TRAINING COMPLETED - Total time: {overall_time:.2f} seconds")
            print("="*80)
    
    def analyze_results(self, use_multioutput=False):
        plotter = RawResultsPlotter(verbose=self.verbose)
        plotter.analyze_results(self.results, self.feature_columns, self.target_scaler, self.timing_info, use_multioutput=use_multioutput)


def main(use_multioutput=False):
    """Main execution function
    
    Args:
        use_multioutput (bool): If True, use single multi-output XGBoost model.
                              If False, use separate XGBoost models for each target.
    """
    print("CoCoME Architecture Optimization Training - RAW FEATURES")
    print("="*80)
    print("Using RAW features from cocome-levels-features.csv (930+ columns)")
    print(f"XGBoost approach: {'Multi-output (single model)' if use_multioutput else 'Single-output (separate models)'}")
    print("="*80)
    
    # Check if raw features file exists
    features_file = 'cocome-levels-features.csv'
    if not os.path.exists(features_file):
        print(f"Error: Raw features file '{features_file}' not found!")
        print("Please make sure the CoCoME raw dataset file is available.")
        return
    
    # Initialize trainer
    trainer = CoCoMERawOptimizationTrainer(features_file, verbose=True)
    
    # Run complete training pipeline
    try:
        trainer.prepare_data()
        trainer.setup_optimizers(use_multioutput=use_multioutput)
        trainer.run_optimization_comparison(max_level=5, train_ratio=0.7, sample_ratio=0.3, use_multioutput=use_multioutput)
        trainer.analyze_results(use_multioutput=use_multioutput)
        
        print("\n" + "="*80)
        print("RAW FEATURES TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Key Features:")
        print("- incremental learning with XGBoost")
        print(f"- Trained on {len(trainer.feature_columns)} RAW features (vs 16 engineered features)")
        print("- High-dimensional feature space with regularization")
        print("- Separate models for each of 8 target metrics")
        print("- Active learning: models updated with ONLY new data per level")
        print("- XGBoost: continues training from previous model")
        print("- Comprehensive performance evaluation and visualization")
        print("- Training summary saved to .txt file")
        
        # Print timing summary
        if hasattr(trainer, 'timing_info') and trainer.timing_info:
            print("\nTIMING SUMMARY:")
            print("-" * 50)
            overall_time = trainer.timing_info.get('overall_time', 0)
            print(f"Total Training Time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
            
            if 'optimizer_times' in trainer.timing_info:
                for optimizer_name, timing_data in trainer.timing_info['optimizer_times'].items():
                    total_time = timing_data.get('total_time', 0)
                    print(f"{optimizer_name}: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print("="*80)
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    # Check command line arguments for configuration
    use_multioutput = False
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'multioutput':
        use_multioutput = True
    
    # You can change these parameters to test different configurations:
    # use_multioutput=True for single multi-output model, False for separate models per target
    main(use_multioutput=use_multioutput)
