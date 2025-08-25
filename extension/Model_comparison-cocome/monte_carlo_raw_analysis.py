"""
Monte Carlo Analysis for Raw Features Architecture Optimization

This script runs multiple simulations of the architecture optimization comparison
using raw features (930+ columns) to analyze the variability introduced by random sampling.
It provides statistical analysis including mean, standard deviation, and confidence intervals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import xgboost as xgb
from scipy.optimize import minimize
from scipy.interpolate import griddata
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import the classes from the raw training file
from main_cocome_raw_training import (
    CoCoMERawOptimizationTrainer
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
    trainer = CoCoMERawOptimizationTrainer(data_file, verbose=False)
    
    # Run the comparison using CoCoMERawOptimizationTrainer methods
    trainer.prepare_data()
    # Setup all optimizers as defined in the current training pipeline
    trainer.setup_optimizers(use_multioutput=use_multioutput)
    trainer.run_optimization_comparison(max_level=5, train_ratio=0.7, sample_ratio=0.3, use_multioutput=use_multioutput)
    
    return simulation_id, trainer.results


class MonteCarloRawOptimizationAnalysis:
    """Monte Carlo analysis for raw features optimization comparison"""
    
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
                gpu_available = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1)
                print("GPU acceleration enabled for XGBoost")
                return True
            except Exception as e:
                print(f"GPU not available, falling back to CPU: {e}")
                self.use_gpu = False
                return False
        return False
        
    def run_single_simulation(self, simulation_id, enable_early_stopping=True, patience=3):
        """Run a single simulation with different random seed"""
        print(f"\n{'='*60}")
        print(f"RAW FEATURES SIMULATION {simulation_id + 1}/{self.n_simulations}")
        print(f"{'='*60}")
        
        # Set different random seed for each simulation
        np.random.seed(42 + simulation_id * 100)
        
        # Create trainer instance with verbose=False to reduce output
        trainer = CoCoMERawOptimizationTrainer(self.data_file, verbose=False)
        
        # Run the comparison using CoCoMERawOptimizationTrainer methods
        trainer.prepare_data()
        # Setup all optimizers per current training code (multioutput flag is accepted but all models are created)
        trainer.setup_optimizers(use_multioutput=self.use_multioutput)
        trainer.run_optimization_comparison(max_level=5, train_ratio=0.7, sample_ratio=0.3, use_multioutput=self.use_multioutput)
        
        return trainer.results
    
    def run_monte_carlo_analysis(self, enable_early_stopping=True, patience=3):
        """Run multiple simulations and collect results"""
        print("Starting Monte Carlo Analysis - RAW FEATURES")
        print("="*60)
        print(f"Running {self.n_simulations} simulations")
        print(f"Early stopping: {'enabled' if enable_early_stopping else 'disabled'}")
        print(f"Patience: {patience}")
        print(f"Parallel processing: {'enabled' if self.use_parallel else 'disabled'}")
        print(f"GPU acceleration: {'enabled' if self.use_gpu else 'disabled'}")
        print(f"XGBoost approach: {'Multi-output (single model)' if self.use_multioutput else 'Single-output (separate models)'}")
        if self.use_parallel:
            print(f"CPU cores available: {mp.cpu_count()}")
        print("="*60)
        
        # Start timing
        start_time = time.time()
        
        # Initialize storage for all simulations (dynamic across all optimizers in trainer)
        # Note: CoCoMERawOptimizationTrainer uses 8 targets: m1, m2, m3, m4, p1, p2, p3, p4
        target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
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
                print(f"Raw features simulation {sim_id + 1} completed successfully")
                
            except Exception as e:
                print(f"Error in raw features simulation {sim_id + 1}: {e}")
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
        print(f"Running raw features simulations with {max_workers} parallel workers...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_sim = {executor.submit(run_simulation_worker, args): args[0] 
                           for args in args_list}
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_sim):
                sim_id = future_to_sim[future]
                try:
                    simulation_id, results = future.result()
                    self._store_simulation_results(all_results, results)
                    completed += 1
                    print(f"Raw features simulation {simulation_id + 1} completed successfully ({completed}/{self.n_simulations})")
                    
                except Exception as e:
                    print(f"Error in raw features simulation {sim_id + 1}: {e}")
                    continue
    
    def _store_simulation_results(self, all_results, results):
        """Store results from a single simulation across all optimizers"""
        target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        
        for optimizer_name, res in results.items():
            # Initialize structure on first encounter
            if optimizer_name not in all_results:
                all_results[optimizer_name] = {
                    'mse_scores_test_scaled': [],  # overall test MSE (scaled) per simulation
                    'test_mse_scaled': {target: [] for target in target_names},  # per-target test MSE (scaled)
                }
            # Append overall test MSE (scaled) series for this simulation
            if 'mse_scores_test_scaled' in res:
                all_results[optimizer_name]['mse_scores_test_scaled'].append(res['mse_scores_test_scaled'])
            # Append per-target test MSE (scaled) series for this simulation
            if 'test_mse_scaled' in res:
                for target in target_names:
                    if target in res['test_mse_scaled']:
                        all_results[optimizer_name]['test_mse_scaled'][target].append(res['test_mse_scaled'][target])
    
    def _print_timing_summary(self):
        """Print timing summary"""
        print("\n" + "="*60)
        print("RAW FEATURES TIMING SUMMARY")
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
        print("\nCalculating statistical measures for raw features (standardized test MSE)...")
        
        self.statistical_summary = {}
        
        for optimizer_name, sim_data in self.simulation_results.items():
            self.statistical_summary[optimizer_name] = {}
            
            # Overall test MSE (scaled)
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
            
            # Per-target test MSE (scaled)
            self.statistical_summary[optimizer_name]['test_mse_scaled'] = {}
            target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
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
        """Create visualizations matching training plots: overall and per-target standardized test MSE"""
        print("Creating Monte Carlo visualizations for raw features (standardized test MSE)...")
        
        # Style map matching main training visualization
        style_map = {
            'XGBoost_MultiOutput': {'linestyle': '-', 'marker': 's', 'color': 'green', 'label': 'XGBoost_MultiOutput'},
            'XGBoost_SingleOutput': {'linestyle': '-', 'marker': 'o', 'color': 'blue', 'label': 'XGBoost_SingleOutput'},
            'LightGBM_MultiOutput': {'linestyle': '-', 'marker': 'D', 'color': 'purple', 'label': 'LightGBM_MultiOutput'},
            'CatBoost_MultiOutput': {'linestyle': '-', 'marker': '^', 'color': 'orange', 'label': 'CatBoost_MultiOutput'},
        }
        
        # Create 3x4 grid similar to training plots
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'Monte Carlo Analysis - Raw Features - {self.n_simulations} Simulations', fontsize=16, fontweight='bold')
        
        # Overall standardized Test MSE with confidence intervals
        ax_overall = axes[0, 0]
        ax_overall.set_title('Overall Test MSE (Raw Features, Standardized Targets)', fontsize=14, fontweight='bold')
        for optimizer_name, stats in self.statistical_summary.items():
            if 'mse_scores_test_scaled' not in stats or optimizer_name not in style_map:
                continue
            series = stats['mse_scores_test_scaled']
            levels = range(1, len(series['mean']) + 1)
            style = style_map[optimizer_name]
            ax_overall.plot(levels, series['mean'], label=style.get('label', optimizer_name),
                            linestyle=style['linestyle'], marker=style['marker'], color=style['color'],
                            markersize=6, linewidth=2)
            ax_overall.fill_between(levels, series['mean'] - series['std'], series['mean'] + series['std'],
                                    color=style['color'], alpha=0.15)
        ax_overall.set_xlabel('Complexity Level')
        ax_overall.set_ylabel('MSE')
        ax_overall.legend()
        ax_overall.grid(True, alpha=0.3)
        ax_overall.set_yscale('log')
        
        # Per-target standardized Test MSE with confidence intervals
        objectives = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
        plot_positions = [(0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)]
        for i, obj in enumerate(objectives):
            row, col = plot_positions[i]
            ax = axes[row, col]
            ax.set_title(f'{obj} - MSE (Raw Features)', fontsize=14, fontweight='bold')
            for optimizer_name, stats in self.statistical_summary.items():
                if optimizer_name not in style_map:
                    continue
                target_stats = stats.get('test_mse_scaled', {}).get(obj, None)
                if target_stats is None or len(target_stats.get('mean', [])) == 0:
                    continue
                levels = range(1, len(target_stats['mean']) + 1)
                style = style_map[optimizer_name]
                ax.plot(levels, target_stats['mean'], label=style.get('label', optimizer_name),
                        linestyle=style['linestyle'], marker=style['marker'], color=style['color'],
                        markersize=6, linewidth=2)
                ax.fill_between(levels, target_stats['mean'] - target_stats['std'], target_stats['mean'] + target_stats['std'],
                                color=style['color'], alpha=0.15)
            ax.set_xlabel('Complexity Level')
            ax.set_ylabel('MSE')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        # Remove unused subplots to match training layout
        fig.delaxes(axes[0, 1])
        fig.delaxes(axes[2, 2])
        fig.delaxes(axes[2, 3])
        
        plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
        
        filename = f'Raw_training/monte_carlo_optimization_analysis_Combined_Raw.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Monte Carlo visualization saved: {filename}")
        plt.show()
        
        return self
    
    def print_statistical_summary(self):
        """Print detailed statistical summary (standardized test MSE)"""
        print("\n" + "="*80)
        print("MONTE CARLO STATISTICAL SUMMARY - RAW FEATURES")
        print("="*80)
        
        for optimizer_name, stats in self.statistical_summary.items():
            if 'mse_scores_test_scaled' not in stats:
                continue
            print(f"\n{optimizer_name}:")
            print("-" * 60)
            final_mse_mean = stats['mse_scores_test_scaled']['mean'][-1]
            final_mse_std = stats['mse_scores_test_scaled']['std'][-1]
            print(f"  Final Test MSE (overall, standardized): {final_mse_mean:.4f} ± {final_mse_std:.4f}")
            
            print(f"\n  Individual Target Performance (Final Level, standardized):")
            target_names = ['m1', 'm2', 'm3', 'm4', 'p1', 'p2', 'p3', 'p4']
            for target in target_names:
                target_stats = stats.get('test_mse_scaled', {}).get(target, None)
                if target_stats is None:
                    continue
                test_mean = target_stats['mean'][-1]
                test_std = target_stats['std'][-1]
                print(f"    {target:4}: Test MSE = {test_mean:.4f} ± {test_std:.4f}")
        
        return self
    
    def save_results(self, filename=None):
        """Save aggregated Monte Carlo results (standardized test MSE) to CSV"""
        if filename is None:
            filename = f'Raw_training/monte_carlo_results_Combined_Raw.csv'
        
        print(f"\nSaving raw features Monte Carlo results to {filename}...")
        
        results_data = []
        for optimizer_name, stats in self.statistical_summary.items():
            # Overall metrics: standardized test MSE
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
            # Per-target standardized test MSE
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
        print(f"Raw features results saved to {filename}")
        
        return self


def main_monte_carlo_raw(n_simulations=10, enable_early_stopping=True, patience=3, use_parallel=False, use_gpu=False, use_multioutput=False):
    """Main function for Monte Carlo analysis with raw features"""
    print("Starting Monte Carlo Architecture Optimization Analysis - RAW FEATURES")
    print("="*60)
    print(f"Number of simulations: {n_simulations}")
    print(f"Early stopping: {'enabled' if enable_early_stopping else 'disabled'}")
    print(f"Patience: {patience}")
    print(f"Parallel processing: {'enabled' if use_parallel else 'disabled'}")
    print(f"GPU acceleration: {'enabled' if use_gpu else 'disabled'}")
    print("Optimizers: XGBoost (Single/Multi-output), LightGBM (Multi-output), CatBoost (Multi-output)")
    print(f"Features: RAW (930+ columns)")
    print("="*60)
    
    # Create analyzer
    analyzer = MonteCarloRawOptimizationAnalysis('cocome-levels-features.csv', 
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
    
    print(f"\nMonte Carlo raw features analysis completed!")
    print(f"Results saved as 'Raw_training/monte_carlo_optimization_analysis_Combined_Raw.png' and 'Raw_training/monte_carlo_results_Combined_Raw.csv'")
    
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
    
    # Run Monte Carlo analysis with raw features
    # You can change use_multioutput=True to test single multi-output model vs separate models per target
    analyzer = main_monte_carlo_raw(n_simulations=20, enable_early_stopping=False, patience=3, use_parallel=True, use_gpu=False, use_multioutput=use_multioutput)
