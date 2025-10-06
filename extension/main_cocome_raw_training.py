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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
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
            'iterations': min(50, max(10, len(X) // 2)),  # Adaptive iterations based on sample size
            'depth': 4,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3.0,
            'random_seed': int(np.random.randint(0, 10000)),
            'loss_function': 'RMSE',
            'verbose': False,
            'allow_writing_files': False,  # Prevent temp file issues
            'thread_count': 1  # Avoid threading issues with large samples
        }
        
        try:
            if self.model is not None:
                # Continue training from existing model
                incremental_model = CatBoostRegressor(**cb_params)
                incremental_model.fit(X_scaled, self.y_train, init_model=self.model)
                self.model = incremental_model
            else:
                # Train new model
                self.model = CatBoostRegressor(**cb_params)
                self.model.fit(X_scaled, self.y_train)
            
            # Save model if requested
            if self.save_models and self.model is not None and level is not None:
                self._save_model(level)
                
        except Exception as e:
            if self.verbose:
                print(f"      {self.target_name}: CatBoost training failed: {e}")
            self.model = None
        
        return self
    
    def predict(self, X):
        """Predict single target values"""
        if self.model is None:
            # Return NaN predictions if model failed to train
            return np.full(len(X), np.nan)
        
        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        except Exception as e:
            if self.verbose:
                print(f"      {self.target_name}: CatBoost prediction failed: {e}")
            return np.full(len(X), np.nan)
    
    def update(self, X_new, y_new, level=None):
        """Incremental learning: Continue CatBoost training with ONLY new data"""
        if self.model is None:
            if self.verbose:
                print(f"      {self.target_name}: Skipping update - model failed to train initially")
            return
        
        if level is not None:
            self.current_level = level
            
        y_new_target = y_new[:, self.target_idx]
        X_new_scaled = self.scaler.transform(X_new)
        
        try:
            # Continue training from existing model with ONLY new data
            cb_params = {
                'iterations': min(25, max(5, len(X_new) // 2)),  # Reduced additional iterations for high-dim data
                'depth': 4,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3.0,
                'random_seed': int(np.random.randint(0, 10000)),
                'loss_function': 'RMSE',
                'verbose': False,
                'allow_writing_files': False,
                'thread_count': 1
            }
            
            # Continue training from existing model with ONLY new data
            incremental_model = CatBoostRegressor(**cb_params)
            incremental_model.fit(X_new_scaled, y_new_target, init_model=self.model)
            
            # Replace the model with the updated one
            self.model = incremental_model
            
            # Save model if requested
            if self.save_models and self.model is not None and level is not None:
                self._save_model(level)
            
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
                print(f"      {self.target_name}: CatBoost incremental update (MSE: {mse:.4f}, New samples: {len(X_new)})")
                
        except Exception as e:
            if self.verbose:
                print(f"      {self.target_name}: CatBoost update failed: {e}")
            self.model = None  # Mark as failed
    
    def _save_model(self, level):
        """Save the current model to the models directory"""
        try:
            # Create models directory if it doesn't exist
            if not os.path.exists(self.models_dir):
                os.makedirs(self.models_dir)
                if self.verbose:
                    print(f"      Created models directory: {self.models_dir}")
            
            # Create filename with the required naming convention
            model_filename = f"CatBoostSingleOutput_{self.target_name}_level-{level}.cbm"
            model_path = os.path.join(self.models_dir, model_filename)
            
            # Save the model
            self.model.save_model(model_path)
            
            if self.verbose:
                print(f"      {self.target_name}: Model saved to {model_path}")
                
        except Exception as e:
            if self.verbose:
                print(f"      {self.target_name}: Failed to save model: {e}")


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


class CatBoostMultiTarget:
    """CatBoost optimizer with separate models for each target."""
    
    def __init__(self, name, verbose=True, save_models=False, models_dir="models"):
        self.name = name
        self.target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        self.verbose = verbose
        self.save_models = save_models
        self.models_dir = models_dir
        self.models = {target_name: CatBoostSingleTarget(target_name, i, verbose, save_models, models_dir)
                       for i, target_name in enumerate(self.target_names)}
        
    def fit(self, X, y, level=None):
        for target_name in self.target_names:
            self.models[target_name].fit(X, y, level)
        return self
    
    def predict(self, X):
        predictions = []
        for target_name in self.target_names:
            predictions.append(self.models[target_name].predict(X))
        return np.column_stack(predictions)
    
    def update(self, X_new, y_new, level=None):
        for target_name in self.target_names:
            self.models[target_name].update(X_new, y_new, level)
    
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
        
        # More conservative CatBoost parameters for stability with varying sample sizes
        cb_params = {
            'iterations': min(50, max(10, len(X) // 2)),  # Adaptive iterations based on sample size
            'depth': 4,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3.0,
            'random_seed': int(np.random.randint(0, 10000)),
            'loss_function': 'RMSE',
            'verbose': False,
            'allow_writing_files': False,  # Prevent temp file issues
            'thread_count': 1  # Avoid threading issues with large samples
        }
        
        try:
            base_model = CatBoostRegressor(**cb_params)
            self.model = MultiOutputRegressor(base_model)
            self.model.fit(X_scaled, y)
            if self.verbose:
                print(f"      Multi-output CatBoost: Trained on all {y.shape[1]} targets simultaneously")
        except Exception as e:
            if self.verbose:
                print(f"      Multi-output CatBoost: Training failed: {e}")
            # Set model to None to indicate failure
            self.model = None
        return self
    
    def predict(self, X):
        if self.model is None:
            # Return NaN predictions if model failed to train
            return np.full((len(X), len(self.target_names)), np.nan)
        X_scaled = self.scaler.transform(X)
        try:
            predictions = self.model.predict(X_scaled)
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)
            if predictions.shape[1] != len(self.target_names):
                predictions = np.tile(predictions, (1, len(self.target_names)))
            return predictions
        except Exception as e:
            if self.verbose:
                print(f"      Multi-output CatBoost: Prediction failed: {e}")
            return np.full((len(X), len(self.target_names)), np.nan)
    
    def update(self, X_new, y_new):
        if self.model is None:
            if self.verbose:
                print(f"      Multi-output CatBoost: Skipping update - model failed to train initially")
            return
            
        X_new_scaled = self.scaler.transform(X_new)
        # Monitor-only history
        if self.X_train_history is not None:
            self.X_train_history = np.vstack([self.X_train_history, X_new])
            self.y_train_history = np.vstack([self.y_train_history, y_new])
        else:
            self.X_train_history = X_new.copy()
            self.y_train_history = y_new.copy()
        
        try:
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
        except Exception as e:
            if self.verbose:
                print(f"      Multi-output CatBoost: Update failed: {e}")
            self.model = None  # Mark as failed
    
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
    
    def analyze_results(self, results, feature_columns, target_scaler, timing_info, use_multioutput=False, metric='mse', sample_ratio=0.3):
        if self.verbose:
            print("\n" + "="*80)
            print("ANALYZING RAW FEATURES CATBOOST SINGLEOUTPUT RESULTS")
            print("="*80)
            
            # Debug: Check what results we have
            metric_lower = (metric or 'mse').lower()
            print(f"Available optimizers: {list(results.keys())}")
            for name, res in results.items():
                if metric_lower == 'mse':
                    series = res.get('mse_scores_test_scaled', [])
                elif metric_lower == 'rmse':
                    mse_values = res.get('mse_scores_test_scaled', [])
                    series = []
                    for v in mse_values:
                        if np.isfinite(v) and v >= 0:
                            series.append(np.sqrt(v))
                        else:
                            series.append(float('nan'))
                    print(f"  MSE values: {mse_values[:3]}...")
                elif metric_lower == 'r2':
                    series = res.get('r2_scores_test_scaled', [])
                print(f"{name}: {len(series)} values, finite values: {sum(1 for v in series if np.isfinite(v))}")
                if series:
                    print(f"  Sample values: {series[:3]}...")
                    
                # Special debugging for both CatBoost variants
                if 'CatBoost' in name:
                    print(f"  {name} detailed analysis:")
                    print(f"    All MSE values: {res.get('mse_scores_test_scaled', [])}")
                    print(f"    All R2 values: {res.get('r2_scores_test_scaled', [])}")
                    print(f"    Will be plotted: {series and len(series) > 0 and any(np.isfinite(v) for v in series)}")
        
        style_map = {}
        for name in results.keys():
            if name == 'XGBoost_MultiOutput':
                style_map[name] = {'linestyle': '-', 'marker': 's', 'color': 'green', 'label': 'XGBoost_MultiOutput'}
            elif name == 'XGBoost_SingleOutput':
                style_map[name] = {'linestyle': '-', 'marker': 'o', 'color': 'blue', 'label': 'XGBoost_SingleOutput'}
            elif name == 'CatBoost_SingleOutput':
                style_map[name] = {'linestyle': '--', 'marker': 'd', 'color': 'red', 'label': 'CatBoost_SingleOutput'}
            elif name == 'CatBoost_MultiOutput':
                style_map[name] = {'linestyle': '-', 'marker': '^', 'color': 'orange', 'label': 'CatBoost_MultiOutput'}
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # Determine metric label and series selector
        metric_lower = (metric or 'mse').lower()
        overall_title_metric = 'MSE' if metric_lower == 'mse' else ('RMSE' if metric_lower == 'rmse' else 'R^2')
        sample_pct = int(sample_ratio * 100)
        axes[0, 0].set_title(f'Overall Test {overall_title_metric} (Raw Features, {sample_pct}% Sample)', fontsize=14, fontweight='bold')
        for name, res in results.items():
            if name not in style_map:
                continue
            series = None
            if metric_lower == 'mse':
                series = res.get('mse_scores_test_scaled', [])
            elif metric_lower == 'rmse':
                mse_values = res.get('mse_scores_test_scaled', [])
                series = []
                for v in mse_values:
                    if np.isfinite(v) and v >= 0:
                        series.append(np.sqrt(v))
                    else:
                        series.append(float('nan'))  # Use NaN instead of inf for invalid values
            elif metric_lower == 'r2':
                series = res.get('r2_scores_test_scaled', [])
            if series and len(series) > 0 and any(np.isfinite(v) for v in series):
                style = style_map[name]
                axes[0, 0].plot(series,
                                label=style.get('label', name),
                                linestyle=style['linestyle'],
                                marker=style['marker'],
                                color=style['color'],
                                markersize=6,
                                linewidth=2)
        axes[0, 0].set_xlabel('Complexity Level')
        axes[0, 0].set_ylabel(overall_title_metric)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        if metric_lower in ['mse', 'rmse']:
            axes[0, 0].set_yscale('log')
        
        objectives = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        plot_positions = [(0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)]
        for i, obj in enumerate(objectives):
            row, col = plot_positions[i]
            per_title_metric = overall_title_metric
            axes[row, col].set_title(f'{obj} - {per_title_metric} (Raw Features, {sample_pct}% Sample)', fontsize=14, fontweight='bold')
            

            for name, res in results.items():
                if name not in style_map:
                    continue
                series = None
                if metric_lower == 'mse':
                    series = res.get('test_mse', {}).get(obj, [])
                elif metric_lower == 'rmse':
                    mse_list = res.get('test_mse', {}).get(obj, [])
                    series = []
                    for v in mse_list:
                        if np.isfinite(v) and v >= 0:
                            series.append(np.sqrt(v))
                        else:
                            series.append(float('nan'))  # Use NaN instead of inf for invalid values
                elif metric_lower == 'r2':
                    # For individual targets, we need to calculate R2 from MSE or use a separate storage
                    # For now, let's use MSE data and convert to a simple performance metric
                    series = res.get('test_mse', {}).get(obj, [])
                if series and len(series) > 0 and any(np.isfinite(v) for v in series):
                    style = style_map[name]
                    axes[row, col].plot(series,
                                        label=style.get('label', name),
                                        linestyle=style['linestyle'],
                                        marker=style['marker'],
                                        color=style['color'],
                                        markersize=6,
                                        linewidth=2)
            axes[row, col].set_xlabel('Complexity Level')
            axes[row, col].set_ylabel(per_title_metric)
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
            if metric_lower in ['mse', 'rmse']:
                axes[row, col].set_yscale('log')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        approach_suffix = "CatBoost_SingleOutput_Raw_LevelSpecific"
        summary_file = f'Raw_training/cocome_training_summary_{approach_suffix}_{timestamp}.txt'
        self.save_training_summary(results, feature_columns, target_scaler, timing_info, summary_file)
        
        fig.delaxes(axes[0, 1])
        fig.delaxes(axes[2, 2])
        fig.delaxes(axes[2, 3])
        
        plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sample_pct = int(sample_ratio * 100)
        output_file = f'Raw_training/cocome_training_results_{approach_suffix}_{metric_lower}_{sample_pct}pct_{timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if self.verbose:
            print(f"Results visualization saved: {output_file}")
        plt.show()
        
        self.print_summary_with_timing(results, timing_info)
    
    def print_summary_with_timing(self, results, timing_info):
        print("\n" + "="*80)
        print("DETAILED RAW FEATURES CATBOOST SINGLEOUTPUT RESULTS SUMMARY")
        print("="*80)
        
        # Print timing information
        if timing_info and 'optimizer_times' in timing_info:
            print("\nTIMING INFORMATION:")
            print("-" * 50)
            
            for optimizer_name, timing_data in timing_info['optimizer_times'].items():
                total_time = timing_data.get('total_time', 0)
                level_times = timing_data.get('level_times', {})
                num_levels = len(level_times) if level_times else 1
                avg_time_per_level = total_time / num_levels if num_levels > 0 else 0
                
                print(f"{optimizer_name}:")
                print(f"  Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
                print(f"  Average Time per Level: {avg_time_per_level:.2f} seconds")
            print()
        
        # Print performance summary
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
    
    def save_training_summary(self, results, feature_columns, target_scaler, timing_info, summary_file):
        """Save comprehensive training summary to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(summary_file), exist_ok=True)
            
            with open(summary_file, 'w') as f:
                f.write("CoCoME Raw Features CatBoost Comparison Training Summary\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Feature Count: {len(feature_columns)} raw features\n")
                f.write(f"Target Count: 8 targets (m1-m4, p1-p4)\n\n")
                
                # Timing information
                if timing_info:
                    f.write("TIMING INFORMATION:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Total Training Time: {timing_info.get('overall_time', 0):.2f} seconds\n")
                    if 'optimizer_times' in timing_info:
                        for name, data in timing_info['optimizer_times'].items():
                            f.write(f"{name}: {data.get('total_time', 0):.2f} seconds\n")
                    f.write("\n")
                
                # Results summary
                f.write("RESULTS SUMMARY:\n")
                f.write("-" * 40 + "\n")
                for name, res in results.items():
                    f.write(f"\n{name}:\n")
                    if res['best_scores']:
                        f.write(f"  Final Best Score: {res['best_scores'][-1]:.4f}\n")
                        f.write(f"  Final Training MSE: {res['mse_scores'][-1]:.4f}\n")
                    
                    f.write("  Individual Target Performance:\n")
                    for target in ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']:
                        if res['test_mse'][target]:
                            f.write(f"    {target}: {res['test_mse'][target][-1]:.4f}\n")
                
            if self.verbose:
                print(f"Training summary saved: {summary_file}")
                
        except Exception as e:
            if self.verbose:
                print(f"Failed to save training summary: {e}")


class CoCoMERawOptimizationTrainer:
    """
    Trainer class for CoCoME architecture optimization using raw features.
    
    This class handles the complete pipeline for training and evaluating architecture
    optimization models using the original raw features from the CoCoME dataset.
    """
    
    def __init__(self, features_file, verbose=True, metric='mse'):
        self.features_file = features_file
        self.verbose = verbose
        self.metric = metric
        self.data = None
        self.feature_columns = None
        self.target_columns = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.optimizers = {}
        self.results = {}
        self.timing_info = {}
        self.train_data = None
        self.test_data = None
        self.plotter = RawResultsPlotter(verbose=verbose)
    
    def prepare_data(self):
        """Load and prepare the raw CoCoME dataset."""
        if self.verbose:
            print("Loading raw CoCoME dataset...")
        
        # Load the dataset
        self.data = pd.read_csv(self.features_file)
        
        if self.verbose:
            print(f"Loaded dataset: {self.data.shape[0]} samples, {self.data.shape[1]} columns")
        
        # Identify feature columns (exclude targets and level)
        exclude_columns = self.target_columns + ['level']
        self.feature_columns = [col for col in self.data.columns if col not in exclude_columns]
        
        if self.verbose:
            print(f"Feature columns: {len(self.feature_columns)} raw features")
            print(f"Target columns: {len(self.target_columns)} targets")
            print(f"Complexity levels: {sorted(self.data['level'].unique())}")
        
        # Check for missing values
        if self.data.isnull().any().any():
            if self.verbose:
                print("Warning: Missing values found, filling with 0...")
            self.data.fillna(0, inplace=True)
        
        # Handle categorical/string features
        categorical_columns = []
        for col in self.feature_columns:
            if self.data[col].dtype == 'object' or self.data[col].dtype.name == 'string':
                categorical_columns.append(col)
        
        if categorical_columns:
            if self.verbose:
                print(f"Found {len(categorical_columns)} categorical columns, applying label encoding...")
                print(f"Sample categorical columns: {categorical_columns[:5]}")
            
            from sklearn.preprocessing import LabelEncoder
            
            # Apply label encoding to categorical columns
            for col in categorical_columns:
                le = LabelEncoder()
                # Handle any remaining NaN values
                self.data[col] = self.data[col].fillna('unknown')
                self.data[col] = le.fit_transform(self.data[col].astype(str))
        
        # Ensure all feature columns are numeric
        for col in self.feature_columns:
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                if self.verbose:
                    print(f"Converting column {col} to numeric...")
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                # Fill any conversion errors with 0
                self.data[col] = self.data[col].fillna(0)
        
        if self.verbose:
            print("Data preprocessing completed successfully!")
        
        return self
    
    def setup_optimizers(self, save_models=False, models_dir="models"):
        """Setup CatBoost optimizers: both Single and Multi-output variants."""
        if self.verbose:
            print("Setting up CatBoost optimizers for raw features...")
            print("- CatBoost SingleOutput: Separate models for each target (8 models)")
            print("- CatBoost MultiOutput: Single model for all targets (1 model)")
            if save_models:
                print(f"Model saving ENABLED - models will be saved to '{models_dir}' directory")
            else:
                print("Model saving DISABLED")
        
        self.optimizers = {
            'CatBoost_SingleOutput': CatBoostMultiTarget('CatBoost Single Output (Raw)', verbose=self.verbose, save_models=save_models, models_dir=models_dir),
            'CatBoost_MultiOutput': CatBoostMultiOutput('CatBoost Multi Output (Raw)', verbose=self.verbose)
        }
        
        return self
    
    def run_optimization_comparison(self, max_level=5, train_ratio=0.7, sample_ratio=0.3, save_models=False, models_dir="models"):
        """Run the complete optimization comparison pipeline."""
        start_time = time.time()
        self.timing_info = {'optimizer_times': {}}
        
        if self.verbose:
            print(f"\nStarting CatBoost comparison training with raw features...")
            print(f"Max level: {max_level}, Train ratio: {train_ratio}, Sample ratio: {sample_ratio}")
            print("NOTE: Each level will be evaluated on its own test data for proper incremental learning")
        
        # Initialize results tracking
        for name in self.optimizers.keys():
            self.results[name] = {
                'best_scores': [],
                'mse_scores': [],
                'mse_scores_test_scaled': [],
                'r2_scores_test_scaled': [],
                'test_mse': {target: [] for target in self.target_columns}
            }
            self.timing_info['optimizer_times'][name] = {
                'total_time': 0,
                'level_times': {}
            }
        
        # Train optimizers level by level
        for level in range(1, max_level + 1):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"TRAINING LEVEL {level}")
                print(f"{'='*60}")
            
            # Get level data and split into train/test for this level
            level_data = self.data[self.data['level'] == level].copy()
            X_level_raw = level_data[self.feature_columns].values
            y_level_raw = level_data[self.target_columns].values
            
            # Handle NaN values
            X_level_raw = np.nan_to_num(X_level_raw, nan=0.0)
            y_level_raw = np.nan_to_num(y_level_raw, nan=0.0)
            
            # Split current level data into train/test
            X_level_train, X_level_test, y_level_train, y_level_test = train_test_split(
                X_level_raw, y_level_raw, test_size=0.3, random_state=42
            )
            
            # Scale using the global scalers (fit on level 1, transform others)
            if level == 1:
                # Fit scalers on level 1
                X_level_train_scaled = self.feature_scaler.fit_transform(X_level_train)
                y_level_train_scaled = self.target_scaler.fit_transform(y_level_train)
            else:
                # Transform using existing scalers
                X_level_train_scaled = self.feature_scaler.transform(X_level_train)
                y_level_train_scaled = self.target_scaler.transform(y_level_train)
            
            # Transform test data for this level
            X_level_test_scaled = self.feature_scaler.transform(X_level_test)
            y_level_test_scaled = self.target_scaler.transform(y_level_test)
            
            # Sample from training data if needed
            n_samples = min(len(X_level_train_scaled), int(len(X_level_train_scaled) * sample_ratio))
            if n_samples < len(X_level_train_scaled):
                indices = np.random.choice(len(X_level_train_scaled), n_samples, replace=False)
                X_level_train_scaled = X_level_train_scaled[indices]
                y_level_train_scaled = y_level_train_scaled[indices]
            
            if self.verbose:
                print(f"Level {level} training samples: {len(X_level_train_scaled)}")
                print(f"Level {level} test samples: {len(X_level_test_scaled)}")
            
            # Store current level test data
            current_test_data = {'X': X_level_test_scaled, 'y': y_level_test_scaled}
            
            # Train each optimizer
            for name, optimizer in self.optimizers.items():
                level_start_time = time.time()
                
                if self.verbose:
                    print(f"\n  Training {name}...")
                
                try:
                    if level == 1:
                        # Initial training
                        if 'MultiOutput' in name:
                            # MultiOutput doesn't use level parameter
                            optimizer.fit(X_level_train_scaled, y_level_train_scaled)
                        else:
                            optimizer.fit(X_level_train_scaled, y_level_train_scaled, level=level)
                    else:
                        # Incremental update
                        if 'MultiOutput' in name:
                            # MultiOutput doesn't use level parameter
                            optimizer.update(X_level_train_scaled, y_level_train_scaled)
                        else:
                            optimizer.update(X_level_train_scaled, y_level_train_scaled, level=level)
                    
                    # Evaluate on current level's test set
                    y_pred_test = optimizer.predict(current_test_data['X'])
                    
                    # Check for NaN values and handle them
                    if np.isnan(y_pred_test).any():
                        if self.verbose:
                            print(f"    Warning: NaN values found in predictions, replacing with 0")
                        y_pred_test = np.nan_to_num(y_pred_test, nan=0.0)
                    
                    # Calculate metrics on current level's test data
                    if y_pred_test.shape == current_test_data['y'].shape and not np.isnan(y_pred_test).all():
                        test_mse_scaled = mean_squared_error(current_test_data['y'], y_pred_test)
                        test_r2_scaled = r2_score(current_test_data['y'], y_pred_test)
                        
                        # Per-target MSE on current level
                        for i, target in enumerate(self.target_columns):
                            if i < y_pred_test.shape[1]:
                                target_mse = mean_squared_error(current_test_data['y'][:, i], y_pred_test[:, i])
                                self.results[name]['test_mse'][target].append(target_mse)
                            else:
                                self.results[name]['test_mse'][target].append(float('inf'))
                    else:
                        test_mse_scaled = float('inf')
                        test_r2_scaled = -float('inf')
                        for target in self.target_columns:
                            self.results[name]['test_mse'][target].append(float('inf'))
                    
                    # Store results
                    self.results[name]['mse_scores_test_scaled'].append(test_mse_scaled)
                    self.results[name]['r2_scores_test_scaled'].append(test_r2_scaled)
                    self.results[name]['best_scores'].append(test_mse_scaled)
                    self.results[name]['mse_scores'].append(test_mse_scaled)
                    
                    if self.verbose:
                        print(f"    Test MSE: {test_mse_scaled:.4f}, R2: {test_r2_scaled:.4f}")
                
                except Exception as e:
                    if self.verbose:
                        print(f"    Training failed: {e}")
                    
                    # Store failure results
                    self.results[name]['mse_scores_test_scaled'].append(float('inf'))
                    self.results[name]['r2_scores_test_scaled'].append(-float('inf'))
                    self.results[name]['best_scores'].append(float('inf'))
                    self.results[name]['mse_scores'].append(float('inf'))
                    for target in self.target_columns:
                        self.results[name]['test_mse'][target].append(float('inf'))
                
                # Record timing
                level_time = time.time() - level_start_time
                self.timing_info['optimizer_times'][name]['level_times'][level] = level_time
                self.timing_info['optimizer_times'][name]['total_time'] += level_time
        
        # Record overall time
        self.timing_info['overall_time'] = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("TRAINING COMPLETED")
            print(f"Total time: {self.timing_info['overall_time']:.2f} seconds")
            print(f"{'='*60}")
        
        return self
    
    def analyze_results(self, metric='mse', sample_ratio=0.3):
        """Analyze and visualize results."""
        self.plotter.analyze_results(
            self.results, 
            self.feature_columns, 
            self.target_scaler, 
            self.timing_info, 
            use_multioutput=True, 
            metric=metric, 
            sample_ratio=sample_ratio
        )
        return self


def main(metric='mse', sample_ratio=0.3, save_models=False, models_dir="models"):
    """Main execution function
    
    Args:
        metric (str): Metric to use for evaluation and plotting ('mse', 'rmse', 'r2').
        sample_ratio (float): Fraction of training data to use per level (0.3 = 30%).
        save_models (bool): If True, save all models from each level to the models directory.
        models_dir (str): Directory name where models will be saved (default: "models").
    """
    print("CoCoME Architecture Optimization Training - RAW FEATURES - CATBOOST SINGLEOUTPUT")
    print("="*80)
    print("Using RAW features from cocome-levels-features.csv (930+ columns)")
    print("Training CatBoost SingleOutput approach:")
    print("- SingleOutput: Separate models for each target (8 models)")
    print("- Level-specific evaluation: Each level tested on its own complexity")
    print(f"Metric: {metric.upper()}")
    print(f"Training sample ratio: {sample_ratio:.0%} (of 70% train split)")
    if save_models:
        print(f"Model saving: ENABLED - models will be saved to '{models_dir}' directory")
    else:
        print("Model saving: DISABLED")
    print("="*80)
    
    # Check if raw features file exists
    features_file = 'cocome-levels-features.csv'
    if not os.path.exists(features_file):
        print(f"Error: Raw features file '{features_file}' not found!")
        print("Please make sure the CoCoME raw dataset file is available.")
        return
    
    # Initialize trainer
    trainer = CoCoMERawOptimizationTrainer(features_file, verbose=True, metric=metric)
    
    # Run complete training pipeline
    try:
        trainer.prepare_data()
        trainer.setup_optimizers(save_models=save_models, models_dir=models_dir)
        trainer.run_optimization_comparison(max_level=5, train_ratio=0.7, sample_ratio=sample_ratio, save_models=save_models, models_dir=models_dir)
        trainer.analyze_results(metric=metric, sample_ratio=sample_ratio)
        
        print("\n" + "="*80)
        print("RAW FEATURES CATBOOST SINGLEOUTPUT TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Key Features:")
        print("- Incremental learning with CatBoost SingleOutput approach")
        print(f"- Trained on {len(trainer.feature_columns)} RAW features (vs 16 engineered features)")
        print("- High-dimensional feature space with regularization")
        print("- SingleOutput: 8 separate CatBoost models (one per target metric)")
        print("- Level-specific evaluation: Each level tested on its own complexity")
        print("- Active learning: models updated with ONLY new data per level")
        print("- CatBoost: continues training from previous model")
        if save_models:
            print(f"- Model saving: ENABLED for SingleOutput models in '{models_dir}' directory")
        else:
            print("- Model saving: DISABLED")
        print("- Comprehensive performance evaluation and visualization")
        print("- Training summary saved to .txt file")
        
        # Print timing summary
        if hasattr(trainer, 'timing_info') and trainer.timing_info:
            print("\nTIMING INFORMATION:")
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
    # Default parameters - modify as needed
    main(
        metric='rmse',          # Options: 'mse', 'rmse', 'r2'
        sample_ratio=0.8,       # Use 30% of training data per level
        save_models=False,      # Disable model saving
        models_dir="models"     # Directory for saved models
    )
