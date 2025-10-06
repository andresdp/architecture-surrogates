"""
Monte Carlo Analysis for Raw Features Architecture Optimization - STPlus Dataset

This script runs multi        # Setup all optimizers (enable both single and multi output)
        trainer.setup_optimizers(use_multioutput=True)  # Enable both types
        trainer.run_optimization_comparison(max_level=2, train_ratio=0.7, sample_ratio=0.3, use_multioutput=True)
        
        return trainer.resultsions of the architecture optimization comparison
using raw features from STPlus dataset to analyze the variability introduced by random sampling.
It provides statistical analysis including mean, standard deviation, and confidence intervals.

Key differences from CoCoME version:
- Uses STPlus dataset: only 2 Levels instead of 5
- 4 targets instead of 8: m1, m2, p1, p2
- Bot types: "Modifiability" and "Performance"
- Adapted for STPlus-specific analysis
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import multiprocessing as mp
from scipy.optimize import minimize
from scipy.interpolate import griddata
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
warnings.filterwarnings('ignore')

# Import the classes from the raw training file
from main_stplus_raw_training import (
    STPlus_RawOptimizationTrainer
)

# Note: Global random seed removed to allow variation between runs
# Individual simulations still use different seeds for comparison
# If you need reproducibility, uncomment the next line:
# np.random.seed(42)


def run_simulation_worker(args):
    """Worker function for parallel simulation execution"""
    simulation_id, data_file, enable_early_stopping, patience, use_gpu, use_multioutput = args
    
    # Set different random seed for each simulation
    np.random.seed(42 + simulation_id * 100)
    
    # Create trainer instance with verbose=False to reduce output
    trainer = STPlus_RawOptimizationTrainer(data_file, verbose=False)
    
    # Run the comparison using STPlus_RawOptimizationTrainer methods
    trainer.prepare_data()
    # Setup all optimizers (enable both single and multi output)
    trainer.setup_optimizers(use_multioutput=True)
    trainer.run_optimization_comparison(max_level=2, train_ratio=0.7, sample_ratio=0.3, use_multioutput=True)
    
    return simulation_id, trainer.results


class MonteCarloSTPlus_RawOptimizationAnalysis:
    """Monte Carlo analysis for STPlus raw features optimization comparison"""
    
    def __init__(self, data_file, n_simulations=10, use_parallel=False, use_gpu=False, use_multioutput=False):
        self.data_file = data_file
        self.n_simulations = n_simulations
        self.use_parallel = use_parallel
        self.use_gpu = use_gpu
        self.use_multioutput = use_multioutput
        self.simulation_results = {}
        self.statistical_summary = {}
        self.timing_info = {}
        
    def configure_gpu_settings(self):
        """Configure GPU settings for XGBoost if available"""
        if self.use_gpu:
            try:
                import xgboost as xgb
                # Check if GPU is available
                if xgb.get_config()['use_rmm']:
                    print("GPU acceleration available for XGBoost")
                    return True
                else:
                    print("GPU not available, using CPU")
                    return False
            except:
                print("GPU configuration failed, using CPU")
                return False
        return False
        
    def run_single_simulation(self, simulation_id, enable_early_stopping=True, patience=3):
        """Run a single simulation with different random seed"""
        print(f"\n{'='*60}")
        print(f"STPLUS RAW FEATURES SIMULATION {simulation_id + 1}/{self.n_simulations}")
        print(f"{'='*60}")
        
        # Set different random seed for each simulation
        np.random.seed(42 + simulation_id * 100)
        
        # Create trainer instance with verbose=False to reduce output
        trainer = STPlus_RawOptimizationTrainer(self.data_file, verbose=False)
        
        # Run the comparison using STPlus_RawOptimizationTrainer methods
        trainer.prepare_data()
        # Setup all optimizers per current training code (enable both single and multi output)
        trainer.setup_optimizers(use_multioutput=True)  # Enable both types
        trainer.run_optimization_comparison(max_level=2, train_ratio=0.7, sample_ratio=0.3, use_multioutput=True)
        
        # Debug: Print what results we got
        if self.verbose:
            print(f"Simulation {simulation_id + 1} results keys: {list(trainer.results.keys())}")
            for opt_name, opt_results in trainer.results.items():
                print(f"  {opt_name}: {list(opt_results.keys())}")
        
        return trainer.results
    
    def run_monte_carlo_analysis(self, enable_early_stopping=True, patience=3):
        """Run multiple simulations and collect results"""
        print("Starting Monte Carlo Analysis - STPLUS RAW FEATURES")
        print("="*60)
        print(f"Running {self.n_simulations} simulations")
        print(f"Early stopping: {'enabled' if enable_early_stopping else 'disabled'}")
        print(f"Patience: {patience}")
        print(f"Parallel processing: {'enabled' if self.use_parallel else 'disabled'}")
        print(f"GPU acceleration: {'enabled' if self.use_gpu else 'disabled'}")
        print(f"XGBoost approach: {'Multi-output (single model)' if self.use_multioutput else 'Single-output (separate models)'}")
        if self.use_parallel:
            print(f"Max workers: {min(mp.cpu_count(), self.n_simulations)}")
        print("="*60)
        
        # Start timing
        start_time = time.time()
        
        # Initialize storage for all simulations (dynamic across all optimizers in trainer)
        # Note: STPlus_RawOptimizationTrainer uses 4 targets: m1, m2, p1, p2
        target_names = ['m1', 'm2', 'p1', 'p2']
        all_results = {}
        
        # Run simulations (parallel or sequential)
        if self.use_parallel:
            self._run_parallel_simulations(all_results, enable_early_stopping, patience)
        else:
            self._run_sequential_simulations(all_results, enable_early_stopping, patience)
        
        # End timing
        end_time = time.time()
        total_time = end_time - start_time
        
        # Store timing information
        self.timing_info = {
            'total_time': total_time,
            'time_per_simulation': total_time / self.n_simulations,
            'start_time': start_time,
            'end_time': end_time,
            'parallel_processing': self.use_parallel,
            'gpu_acceleration': self.use_gpu
        }
        
        self._print_timing_summary()
        
        self.simulation_results = all_results
        return self
    
    def _run_sequential_simulations(self, all_results, enable_early_stopping, patience):
        """Run simulations sequentially"""
        for sim_id in range(self.n_simulations):
            try:
                results = self.run_single_simulation(sim_id, enable_early_stopping, patience)
                self._store_simulation_results(all_results, results)
                print(f"Simulation {sim_id + 1}/{self.n_simulations} completed successfully")
            except Exception as e:
                print(f"Simulation {sim_id + 1} failed: {e}")
                continue
    
    def _run_parallel_simulations(self, all_results, enable_early_stopping, patience):
        """Run simulations in parallel"""
        # Prepare arguments for parallel execution
        args_list = [
            (sim_id, self.data_file, enable_early_stopping, patience, self.use_gpu, self.use_multioutput)
            for sim_id in range(self.n_simulations)
        ]
        
        # Use ProcessPoolExecutor for parallel execution
        max_workers = min(mp.cpu_count(), self.n_simulations)
        print(f"Running STPlus raw features simulations with {max_workers} parallel workers...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_simulation_worker, args): args[0] for args in args_list}
            
            for future in as_completed(futures):
                sim_id = futures[future]
                try:
                    simulation_id, results = future.result()
                    self._store_simulation_results(all_results, results)
                    print(f"Parallel simulation {simulation_id + 1}/{self.n_simulations} completed")
                except Exception as e:
                    print(f"Parallel simulation {sim_id + 1} failed: {e}")
                    continue
    
    def _store_simulation_results(self, all_results, results):
        """Store results from a single simulation across all optimizers"""
        target_names = ['m1', 'm2', 'p1', 'p2']
        
        for optimizer_name, res in results.items():
            # Initialize structure on first encounter - match CoCoME structure
            if optimizer_name not in all_results:
                all_results[optimizer_name] = {
                    'mse_scores_test_scaled': [],  # overall test MSE (scaled) per simulation
                    'test_mse_scaled': {target: [] for target in target_names},  # per-target test MSE (scaled)
                }
            
            # Convert STPlus level-based results to time-series format expected by Monte Carlo
            overall_mse_series = []
            target_mse_series = {target: [] for target in target_names}
            
            # Extract MSE values for each level (1, 2 for STPlus)
            for level in [1, 2]:
                level_key = f'level_{level}'
                if level_key in res:
                    level_data = res[level_key]
                    
                    # Overall MSE for this level
                    if 'test_mse' in level_data and np.isfinite(level_data['test_mse']):
                        overall_mse_series.append(level_data['test_mse'])
                    else:
                        overall_mse_series.append(float('nan'))
                    
                    # Per-target MSE for this level
                    for target in target_names:
                        target_mse_key = f'test_{target}_mse'
                        if target_mse_key in level_data and np.isfinite(level_data[target_mse_key]):
                            target_mse_series[target].append(level_data[target_mse_key])
                        else:
                            target_mse_series[target].append(float('nan'))
                else:
                    # Missing level data
                    overall_mse_series.append(float('nan'))
                    for target in target_names:
                        target_mse_series[target].append(float('nan'))
            
            # Store the series for this simulation
            if overall_mse_series:
                all_results[optimizer_name]['mse_scores_test_scaled'].append(overall_mse_series)
            
            for target in target_names:
                if target_mse_series[target]:
                    all_results[optimizer_name]['test_mse_scaled'][target].append(target_mse_series[target])
    
    def _print_timing_summary(self):
        """Print timing summary"""
        print("\n" + "="*60)
        print("STPLUS RAW FEATURES TIMING SUMMARY")
        print("="*60)
        print(f"Total execution time: {self.timing_info['total_time']:.2f} seconds")
        print(f"Average time per simulation: {self.timing_info['time_per_simulation']:.2f} seconds")
        print(f"Parallel processing: {'Yes' if self.timing_info['parallel_processing'] else 'No'}")
        print(f"GPU acceleration: {'Yes' if self.timing_info['gpu_acceleration'] else 'No'}")
        
        # Calculate speedup estimate if parallel
        if self.timing_info['parallel_processing']:
            estimated_sequential_time = self.timing_info['time_per_simulation'] * self.n_simulations
            speedup = estimated_sequential_time / self.timing_info['total_time']
            print(f"Estimated speedup: {speedup:.2f}x")
        
        print("="*60)
    
    def calculate_statistics(self):
        """Calculate statistical measures across simulations for each optimizer"""
        print("\nCalculating statistical measures for STPlus raw features (scaled space test MSE)...")
        
        self.statistical_summary = {}
        
        for optimizer_name, sim_data in self.simulation_results.items():
            self.statistical_summary[optimizer_name] = {}
            
            # Overall test MSE (scaled) - match CoCoME structure
            overall_data = sim_data.get('mse_scores_test_scaled', [])
            if overall_data:
                # Pad variable-length sequences with NaN then compute nan-aware stats
                max_len = max(len(seq) for seq in overall_data)
                padded = np.full((len(overall_data), max_len), np.nan)
                for i, seq in enumerate(overall_data):
                    padded[i, :len(seq)] = seq
                self.statistical_summary[optimizer_name]['mse_scores_test_scaled'] = {
                    'mean': np.nanmean(padded, axis=0),
                    'std': np.nanstd(padded, axis=0),
                    'min': np.nanmin(padded, axis=0),
                    'max': np.nanmax(padded, axis=0),
                    'median': np.nanmedian(padded, axis=0),
                    'q25': np.nanpercentile(padded, 25, axis=0),
                    'q75': np.nanpercentile(padded, 75, axis=0),
                    'raw_data': padded
                }
            
            # Per-target test MSE (scaled) - match CoCoME structure
            self.statistical_summary[optimizer_name]['test_mse_scaled'] = {}
            target_names = ['m1', 'm2', 'p1', 'p2']
            for target in target_names:
                target_data = sim_data.get('test_mse_scaled', {}).get(target, [])
                if target_data:
                    max_len = max(len(seq) for seq in target_data)
                    padded = np.full((len(target_data), max_len), np.nan)
                    for i, seq in enumerate(target_data):
                        padded[i, :len(seq)] = seq
                    self.statistical_summary[optimizer_name]['test_mse_scaled'][target] = {
                        'mean': np.nanmean(padded, axis=0),
                        'std': np.nanstd(padded, axis=0),
                        'min': np.nanmin(padded, axis=0),
                        'max': np.nanmax(padded, axis=0),
                        'median': np.nanmedian(padded, axis=0),
                        'q25': np.nanpercentile(padded, 25, axis=0),
                        'q75': np.nanpercentile(padded, 75, axis=0),
                        'raw_data': padded
                    }
        
        return self
    
    def create_monte_carlo_visualizations(self):
        """Create two separate visualizations: Overall RMSE and per-target RMSE plots (scaled space)"""
        print("Creating Monte Carlo visualizations for STPlus raw features (scaled space RMSE)...")
        
        # Style map for CatBoost models only - updated for STPlus (using spaces not underscores)
        style_map = {
            'CatBoost SingleOutput': {'linestyle': '--', 'marker': 'd', 'color': 'red', 'label': 'CatBoost SingleOutput'},
            'CatBoost MultiOutput': {'linestyle': '-', 'marker': '^', 'color': 'orange', 'label': 'CatBoost MultiOutput'},
        }
        
        # PLOT 1: Overall Test RMSE
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        for optimizer_name, stats in self.statistical_summary.items():
            # Only process CatBoost models (using space not underscore)
            if not optimizer_name.startswith('CatBoost') or 'mse_scores_test_scaled' not in stats or optimizer_name not in style_map:
                continue
            # Convert MSE to RMSE
            mse_series = stats['mse_scores_test_scaled']
            rmse_mean = np.sqrt(mse_series['mean'])
            rmse_std = np.sqrt(mse_series['std'])  # Approximate RMSE std
            
            levels = range(1, len(rmse_mean) + 1)
            style = style_map[optimizer_name]
            ax1.plot(levels, rmse_mean, label=style.get('label', optimizer_name),
                    linestyle=style['linestyle'], marker=style['marker'], color=style['color'],
                    markersize=8, linewidth=3)
            ax1.fill_between(levels, rmse_mean - rmse_std, rmse_mean + rmse_std,
                            color=style['color'], alpha=0.15)
        
        ax1.set_xlabel('Level', fontsize=12)
        ax1.set_ylabel('RMSE', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.set_xticks(range(1, 3))  # Set x-axis to show only integers 1-2 for STPlus
        
        plt.tight_layout()
        filename1 = f'Raw_training/monte_carlo_catboost_comparison_STPlus_Raw.png'
        plt.savefig(filename1, dpi=300, bbox_inches='tight')
        print(f"CatBoost comparison RMSE visualization saved: {filename1}")
        plt.show()
        
        # PLOT 2: Per-target Test RMSE (2x2 grid for 4 targets) - CatBoost only
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
        
        objectives = ['m1', 'm2', 'p1', 'p2']
        
        for i, obj in enumerate(objectives):
            row = i // 2  # 0 for first 2, 1 for last 2
            col = i % 2   # 0,1 for each row
            ax = axes2[row, col]
            
            for optimizer_name, stats in self.statistical_summary.items():
                # Only process CatBoost models (using space not underscore)
                if not optimizer_name.startswith('CatBoost') or optimizer_name not in style_map:
                    continue
                target_stats = stats.get('test_mse_scaled', {}).get(obj, None)
                if target_stats is None or len(target_stats.get('mean', [])) == 0:
                    continue
                
                # Convert MSE to RMSE
                mse_mean = target_stats['mean']
                mse_std = target_stats['std']
                rmse_mean = np.sqrt(mse_mean)
                rmse_std = np.sqrt(mse_std)  # Approximate RMSE std
                
                levels = range(1, len(rmse_mean) + 1)
                style = style_map[optimizer_name]
                ax.plot(levels, rmse_mean, label=style.get('label', optimizer_name),
                       linestyle=style['linestyle'], marker=style['marker'], color=style['color'],
                       markersize=6, linewidth=2)
                ax.fill_between(levels, rmse_mean - rmse_std, rmse_mean + rmse_std,
                               color=style['color'], alpha=0.15)
            
            ax.set_xlabel('Level', fontsize=10)
            ax.set_ylabel('RMSE', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.set_xticks(range(1, 3))  # Set x-axis to show only integers 1-2 for STPlus
        
        plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
        filename2 = f'Raw_training/monte_carlo_catboost_per_target_STPlus_Raw.png'
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        print(f"CatBoost per-target RMSE visualization saved: {filename2}")
        plt.show()
        
        return self
    
    def print_statistical_summary(self):
        """Print detailed statistical summary (standardized test MSE)"""
        print("\n" + "="*80)
        print("STPLUS MONTE CARLO STATISTICAL SUMMARY - RAW FEATURES (SCALED SPACE)")
        print("="*80)
        
        for optimizer_name, stats in self.statistical_summary.items():
            # Only process CatBoost models (using space not underscore)
            if not optimizer_name.startswith('CatBoost') or 'mse_scores_test_scaled' not in stats:
                continue
            print(f"\n{optimizer_name}:")
            print("-" * 60)
            final_mse_mean = stats['mse_scores_test_scaled']['mean'][-1]
            final_mse_std = stats['mse_scores_test_scaled']['std'][-1]
            print(f"  Final Test MSE (overall, scaled space): {final_mse_mean:.4f} ± {final_mse_std:.4f}")
            
            print(f"\n  Individual Target Performance (Final Level, scaled space):")
            target_names = ['m1', 'm2', 'p1', 'p2']
            for target in target_names:
                target_stats = stats.get('test_mse_scaled', {}).get(target, None)
                if target_stats is None:
                    continue
                test_mean = target_stats['mean'][-1]
                test_std = target_stats['std'][-1]
                print(f"    {target:4}: Test MSE = {test_mean:.4f} ± {test_std:.4f}")
        
        return self
    
    def save_results(self, filename=None):
        """Save aggregated Monte Carlo results (scaled space test MSE) to CSV"""
        if filename is None:
            filename = f'Raw_training/monte_carlo_results_Combined_STPlus_Raw.csv'
        
        print(f"\nSaving STPlus raw features Monte Carlo results to {filename}...")
        
        results_data = []
        for optimizer_name, stats in self.statistical_summary.items():
            # Only process CatBoost models (using space not underscore)
            if not optimizer_name.startswith('CatBoost'):
                continue
            # Overall metrics: scaled space test MSE
            if 'mse_scores_test_scaled' in stats:
                overall = stats['mse_scores_test_scaled']
                for level in range(len(overall['mean'])):
                    results_data.append({
                        'optimizer': optimizer_name,
                        'level': level + 1,
                        'metric': 'mse_scores_test_scaled',
                        'target': 'overall',
                        'mean': overall['mean'][level],
                        'std': overall['std'][level],
                        'min': overall['min'][level],
                        'max': overall['max'][level],
                        'median': overall['median'][level],
                        'q25': overall['q25'][level],
                        'q75': overall['q75'][level]
                    })
            # Per-target scaled space test MSE
            for target, target_stats in stats.get('test_mse_scaled', {}).items():
                for level in range(len(target_stats['mean'])):
                    results_data.append({
                        'optimizer': optimizer_name,
                        'level': level + 1,
                        'metric': 'test_mse_scaled',
                        'target': target,
                        'mean': target_stats['mean'][level],
                        'std': target_stats['std'][level],
                        'min': target_stats['min'][level],
                        'max': target_stats['max'][level],
                        'median': target_stats['median'][level],
                        'q25': target_stats['q25'][level],
                        'q75': target_stats['q75'][level]
                    })
        
        df = pd.DataFrame(results_data)
        df.to_csv(filename, index=False)
        print(f"STPlus raw features results saved to {filename}")
        
        return self


def main_monte_carlo_stplus_raw(n_simulations=10, enable_early_stopping=True, patience=3, use_parallel=False, use_gpu=False, use_multioutput=False):
    """Main function for Monte Carlo analysis with STPlus raw features"""
    print("Starting Monte Carlo Architecture Optimization Analysis - STPLUS RAW FEATURES (CatBoost Only)")
    print("="*60)
    print(f"Number of simulations: {n_simulations}")
    print(f"Early stopping: {'enabled' if enable_early_stopping else 'disabled'}")
    print(f"Patience: {patience}")
    print(f"Parallel processing: {'enabled' if use_parallel else 'disabled'}")
    print(f"GPU acceleration: {'enabled' if use_gpu else 'disabled'}")
    print("Optimizers: CatBoost (Single-output vs Multi-output comparison)")
    print(f"Features: RAW STPlus features (~700+ columns)")
    print(f"Levels: 2 (instead of 5 for CoCoME)")
    print(f"Targets: 4 (m1, m2, p1, p2)")
    print("="*60)
    
    # Create analyzer
    analyzer = MonteCarloSTPlus_RawOptimizationAnalysis('stplus-levels-bots-features.csv', 
                                                      n_simulations=n_simulations,
                                                      use_parallel=use_parallel,
                                                      use_gpu=use_gpu,
                                                      use_multioutput=use_multioutput)
    
    # Run analysis
    analyzer.run_monte_carlo_analysis(enable_early_stopping=enable_early_stopping, patience=patience)
    analyzer.calculate_statistics()
    analyzer.create_monte_carlo_visualizations()
    analyzer.print_statistical_summary()
    analyzer.save_results()
    
    print(f"\nMonte Carlo STPlus CatBoost comparison analysis completed!")
    print(f"Results saved as 'Raw_training/monte_carlo_catboost_comparison_STPlus_Raw.png', 'Raw_training/monte_carlo_catboost_per_target_STPlus_Raw.png' and 'Raw_training/monte_carlo_results_Combined_STPlus_Raw.csv'")
    
    # Print final timing summary
    if hasattr(analyzer, 'timing_info') and analyzer.timing_info:
        print(f"\nFinal Performance Summary:")
        print(f"Total time: {analyzer.timing_info['total_time']:.2f} seconds")
        print(f"Time per simulation: {analyzer.timing_info['time_per_simulation']:.2f} seconds")
    
    return analyzer


if __name__ == "__main__":
    import sys
    
    # Check command line arguments for configuration
    use_multioutput = False
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'multioutput':
        use_multioutput = True
    
    # Run Monte Carlo analysis with STPlus raw features
    # You can change use_multioutput=True to test single multi-output model vs separate models per target
    analyzer = main_monte_carlo_stplus_raw(n_simulations=5, enable_early_stopping=False, patience=3, use_parallel=True, use_gpu=False, use_multioutput=use_multioutput)