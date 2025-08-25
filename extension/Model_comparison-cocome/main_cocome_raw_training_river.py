"""
Architecture Optimization Training - CoCoME Dataset with Raw Features (River Online Learning)

This script mirrors main_cocome_raw_training.py but replaces GBDT-based models
with River's online learning regressors. Models are updated incrementally in the
update method by streaming new level data sample-by-sample.

Models included (regression-capable only):
- River LinearRegression
- River ExtremelyFastDecisionTreeRegressor
- River KNNRegressor

Other requested models that are classifier-only or unsupervised (LogisticRegression,
GaussianNB, KMeans) are omitted here since this pipeline predicts continuous targets
and evaluates with MSE.

Active Learning Features:
- Initial training: Only on the current level's randomly sampled training set
- Incremental updates: On subsequent calls, learn from ONLY new level data
- Memory management: Historical samples tracked for monitoring only
- Early stopping is not implemented; models remain online-ready
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
import os
import time
from datetime import datetime

# River imports
from river import linear_model, tree, neighbors, optim

warnings.filterwarnings('ignore')


class RiverSingleTargetRegressor:
    """Wraps a single River regressor for one target, with batch helpers.

    - Uses sklearn StandardScaler for features (fit on first training batch).
    - Learns sample-by-sample via model.learn_one on scaled features.
    - Predicts with model.predict_one on scaled features.
    """

    def __init__(self, target_name, target_idx, model_factory, feature_names=None, verbose=True):
        self.target_name = target_name
        self.target_idx = target_idx
        self.model_factory = model_factory
        self.verbose = verbose
        self.model = None
        self.feature_scaler = StandardScaler()
        self.feature_names = feature_names  # optional
        self.X_train_history = None
        self.y_train_history = None

    def _ensure_model(self):
        if self.model is None:
            self.model = self.model_factory()

    def _row_to_dict(self, x_row):
        # Use provided feature names if available, else index-based keys
        if self.feature_names is not None and len(self.feature_names) == len(x_row):
            return {self.feature_names[i]: float(x_row[i]) for i in range(len(x_row))}
        return {f"f{i}": float(val) for i, val in enumerate(x_row)}

    def fit(self, X, y_scaled):
        """Fit on a batch by streaming samples to River model."""
        self._ensure_model()
        # Fit scaler on this batch and stream-learn
        X_scaled = self.feature_scaler.fit_transform(X)
        y_target = y_scaled[:, self.target_idx]

        for xi, yi in zip(X_scaled, y_target):
            self.model.learn_one(self._row_to_dict(xi), float(yi))

        # Track history for monitoring only
        self.X_train_history = X.copy()
        self.y_train_history = y_target.copy()
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet")
        X_scaled = self.feature_scaler.transform(X)
        preds = []
        for xi in X_scaled:
            y_hat = self.model.predict_one(self._row_to_dict(xi))
            # Some River models may return None before sufficient data; fall back to 0.0
            preds.append(0.0 if y_hat is None else float(y_hat))
        return np.array(preds)

    def update(self, X_new, y_new_scaled):
        """Incremental update using ONLY new data (streamed)."""
        self._ensure_model()

        # Transform with existing scaler (do not refit to preserve consistency)
        X_new_scaled = self.feature_scaler.transform(X_new)
        y_target_new = y_new_scaled[:, self.target_idx]

        for xi, yi in zip(X_new_scaled, y_target_new):
            self.model.learn_one(self._row_to_dict(xi), float(yi))

        # Monitoring: update training history (append)
        if self.X_train_history is not None:
            self.X_train_history = np.vstack([self.X_train_history, X_new])
            self.y_train_history = np.concatenate([self.y_train_history, y_target_new])
        else:
            self.X_train_history = X_new.copy()
            self.y_train_history = y_target_new.copy()

        # Evaluate performance on new data only (scaled target space)
        y_pred_new = self.predict(X_new)
        mse = mean_squared_error(y_target_new, y_pred_new)
        if self.verbose:
            print(f"      {self.target_name}:  incremental update (MSE: {mse:.4f}, New samples: {len(X_new)})")


class RiverMultiTargetRegressor:
    """Manages separate River regressors for each of the 8 target metrics."""

    def __init__(self, name, model_factory, feature_names=None, verbose=True):
        self.name = name
        self.target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        self.verbose = verbose
        self.feature_names = feature_names
        self.models = {
            tn: RiverSingleTargetRegressor(tn, i, model_factory, feature_names=feature_names, verbose=verbose)
            for i, tn in enumerate(self.target_names)
        }

    def fit(self, X, y_scaled):
        for tn in self.target_names:
            self.models[tn].fit(X, y_scaled)
        if self.verbose:
            print(f"      {self.name}: Trained separate River regressors for all targets")
        return self

    def predict(self, X):
        preds = []
        for tn in self.target_names:
            preds.append(self.models[tn].predict(X))
        return np.column_stack(preds)

    def update(self, X_new, y_new_scaled):
        for tn in self.target_names:
            self.models[tn].update(X_new, y_new_scaled)

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
            print("ANALYZING RAW FEATURES RESULTS (River models)")
            print("="*80)
        
        style_map = {}
        for name in results.keys():
            if name == 'River_HoeffdingTreeRegressor':
                style_map[name] = {'linestyle': '-', 'marker': 's', 'color': 'green', 'label': 'River_HoeffdingTreeRegressor'}
            elif name == 'River_KNNRegressor':
                style_map[name] = {'linestyle': '-', 'marker': 'D', 'color': 'purple', 'label': 'River_KNNRegressor'}
        
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
        approach_suffix = "River_Raw"
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
            f.write("COCOME RAW FEATURES TRAINING SUMMARY - River Online Regressors\n")
            f.write("="*80 + "\n")
            f.write("IMPORTANT: This uses RAW FEATURES (930+ columns) with true online incremental learning via River.\n")
            f.write("- Models learn from ONLY current level's new samples in update()\n")
            f.write("- Historical data is tracked for monitoring only, NOT used for retraining\n")
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
                        level_times = timing_data.get('level_times', {})
                        level_count = max(1, len(level_times))
                        f.write(f"  Average Time per Level: {total_time/level_count:.2f} seconds\n")
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
        print("DETAILED RAW FEATURES OPTIMIZATION RESULTS SUMMARY (River)")
        print("="*80)
        target_names = ['m1', 'm2', 'm3', 'p1', 'p2', 'p3', 'p4']
        # Correct order should be 8 targets; fix oversight
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
    """Training pipeline for CoCoME optimization using raw features with River models"""
    
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
        
        exclude_cols = ['solID', 'level', 'm1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        
        if self.data.columns[0].startswith('Unnamed') or self.data.columns[0] == '':
            exclude_cols.append(self.data.columns[0])
        
        self.feature_columns = [col for col in self.data.columns if col not in exclude_cols]
        
        self.features = self.data[self.feature_columns].values
        target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        self.objectives = self.data[target_names].values
        self.objectives_scaled = self.target_scaler.fit_transform(self.objectives)
        
        if self.verbose:
            print(f"Features: {self.features.shape} ({len(self.feature_columns)} raw features)")
            print(f"Objectives: {self.objectives.shape} (8 target metrics)")
            print(f"Feature types breakdown:")
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
        """Setup River regressors (separate model per target)."""
        if self.verbose:
            print("Setting up River online regressors for raw features...")
            print("Models: HoeffdingTreeRegressor, KNNRegressor")
        
        # Define model factories
        def efdt_factory():
            return tree.HoeffdingTreeRegressor()
        
        def knn_factory():
            return neighbors.KNNRegressor(n_neighbors=5)
        
        self.optimizers = {
            'River_HoeffdingTreeRegressor': RiverMultiTargetRegressor('River HoeffdingTree Regressor (Raw)', efdt_factory, feature_names=self.feature_columns, verbose=self.verbose),
            'River_KNNRegressor': RiverMultiTargetRegressor('River KNN Regressor (Raw)', knn_factory, feature_names=self.feature_columns, verbose=self.verbose)
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
        overall_start_time = time.time()
        
        if self.verbose:
            print("\n" + "="*80)
            print("STARTING COCOME RAW FEATURES OPTIMIZATION COMPARISON (River)")
            print("="*80)
            print(f"Max levels: {max_level}")
            print(f"Train/test ratio: {train_ratio:.0%}/{1-train_ratio:.0%}")
            print(f"Training sample ratio: {sample_ratio:.0%}")
            print(f"Using RAW FEATURES: {len(self.feature_columns)} columns")
            print()
        
        complexity_groups = self.group_by_complexity()
        
        self.timing_info = {
            'overall_start_time': overall_start_time,
            'level_times': {},
            'optimizer_times': {name: {'total_time': 0, 'level_times': {}} for name in self.optimizers.keys()},
            'data_preparation_time': 0,
            'evaluation_time': 0
        }
        
        target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        for name in self.optimizers.keys():
            self.results[name] = {
                'best_scores': [],
                'mse_scores': [],
                'mse_scores_test': [],
                'mse_scores_scaled': [],
                'mse_scores_test_scaled': [],
                'level_performances': {},
                'convergence': [],
                'individual_mse': {target: [] for target in target_names},
                'individual_mse_scaled': {target: [] for target in target_names},
                'test_mse': {target: [] for target in target_names},
                'test_mse_scaled': {target: [] for target in target_names},
                'training_samples_used': [],
                'test_samples_used': [],
                'sample_details': []
            }
        
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
            
            data_prep_start = time.time()
            X_train_full, X_test, y_train_full, y_test, y_train_full_scaled, y_test_scaled = train_test_split(
                X_level, y_level, y_level_scaled, test_size=(1-train_ratio), random_state=None
            )
            
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
            
            for name, optimizer in self.optimizers.items():
                optimizer_start_time = time.time()
                
                if self.verbose:
                    print(f"  Training {name}...")
                
                try:
                    first_training = all(model.model is None for model in optimizer.models.values())
                    if first_training:
                        optimizer.fit(X_train, y_train_scaled)
                        if self.verbose:
                            print(f"    Trained from scratch: {len(X_train)} samples")
                    else:
                        optimizer.update(X_train, y_train_scaled)
                    
                    # Evaluate performance
                    y_train_pred_scaled = optimizer.predict(X_train)
                    y_train_pred = self.target_scaler.inverse_transform(y_train_pred_scaled)
                    train_mse = mean_squared_error(y_train, y_train_pred)
                    train_mse_scaled = mean_squared_error(y_train_scaled, y_train_pred_scaled)
                    
                    y_test_pred_scaled = optimizer.predict(X_test)
                    y_test_pred = self.target_scaler.inverse_transform(y_test_pred_scaled)
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    test_mse_scaled = mean_squared_error(y_test_scaled, y_test_pred_scaled)
                    
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
                            print(f"      {target}: Train MSE={target_train_mse:.4f}, Test MSE={target_test_mse:.4f}")
                    
                    # Best scores: min observed scaled target values from training history, then inverse-transform
                    best_scaled = []
                    for target_name in target_names:
                        model = optimizer.models[target_name]
                        if hasattr(model, 'y_train_history') and model.y_train_history is not None and len(model.y_train_history) > 0:
                            best_scaled.append(np.min(model.y_train_history))
                        else:
                            best_scaled.append(float('inf'))
                    if not any(np.isinf(best_scaled)):
                        best_raw = self.target_scaler.inverse_transform(np.array(best_scaled).reshape(1, -1))[0]
                        best_scores = list(best_raw)
                    else:
                        best_scores = [float('inf')] * len(target_names)
                    current_best = np.sum(best_scores)
                    
                    current_level_train_samples = len(X_train)
                    current_level_test_samples = len(X_test)
                    validation_samples = len(X_train_full) - len(X_train)
                    
                    first_model = list(optimizer.models.values())[0]
                    previous_level_samples = 0
                    total_model_samples = 0
                    if hasattr(first_model, 'X_train_history') and first_model.X_train_history is not None:
                        total_model_samples = len(first_model.X_train_history)
                        if level > 1:
                            previous_level_samples = total_model_samples - current_level_train_samples
                    elif hasattr(first_model, 'y_train_history') and first_model.y_train_history is not None:
                        total_model_samples = len(first_model.y_train_history)
                        if level > 1:
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
                    self.results[name]['best_scores'].append(float('inf'))
                    self.results[name]['mse_scores'].append(float('inf'))
                    for target in target_names:
                        self.results[name]['test_mse'][target].append(float('inf'))
                        self.results[name]['individual_mse'][target].append(float('inf'))
                    self.results[name]['training_samples_used'].append(0)
                    self.results[name]['test_samples_used'].append(0)
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
            
            level_time = time.time() - level_start_time
            self.timing_info['level_times'][level] = level_time
            
            if self.verbose:
                print(f"  Level {level} completed in {level_time:.2f} seconds")
        
        overall_time = time.time() - overall_start_time
        self.timing_info['overall_time'] = overall_time
        self.timing_info['overall_end_time'] = time.time()
        
        if self.verbose:
            print(f"\n" + "="*80)
            print(f"RAW FEATURES TRAINING COMPLETED (River) - Total time: {overall_time:.2f} seconds")
            print("="*80)
    
    def analyze_results(self, use_multioutput=False):
        plotter = RawResultsPlotter(verbose=self.verbose)
        plotter.analyze_results(self.results, self.feature_columns, self.target_scaler, self.timing_info, use_multioutput=use_multioutput)


def main(use_multioutput=False):
    """Main execution function (River online learning version)"""
    print("CoCoME Architecture Optimization Training - RAW FEATURES (River)")
    print("="*80)
    print("Using RAW features from cocome-levels-features.csv (930+ columns)")
    print("Models: River HoeffdingTree Regressor, River KNN Regressor")
    print("="*80)
    
    features_file = 'cocome-levels-features.csv'
    if not os.path.exists(features_file):
        print(f"Error: Raw features file '{features_file}' not found!")
        print("Please make sure the CoCoME raw dataset file is available.")
        return
    
    trainer = CoCoMERawOptimizationTrainer(features_file, verbose=True)
    
    try:
        trainer.prepare_data()
        trainer.setup_optimizers(use_multioutput=False)
        trainer.run_optimization_comparison(max_level=5, train_ratio=0.7, sample_ratio=0.3, use_multioutput=False)
        trainer.analyze_results(use_multioutput=False)
        
        print("\n" + "="*80)
        print("RAW FEATURES TRAINING COMPLETED SUCCESSFULLY! (River)")
        print("="*80)
        print("Key Features:")
        print("- True incremental learning with River (learn_one per sample)")
        print(f"- Trained on {len(trainer.feature_columns)} RAW features (vs 16 engineered features)")
        print("- High-dimensional feature space; per-target separate regressors")
        print("- Active learning: models updated with ONLY new data per level")
        print("- Comprehensive performance evaluation and visualization")
        print("- Training summary saved to .txt file")
        
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
    main(use_multioutput=False)


