This site corresponds to the reproducibility kit for the paper *"Architecture Optimization using Surrogate-based Incremental Learning for Quality-attribute Analyses"* accepted at ICSA 2025.

---

We provide the following assets:
* [Datasets](https://github.com/andresdp/architecture-surrogates/tree/main/datasets) for the *ST+* and *CoCoME* case studies
* *Jupyter* [notebooks](https://github.com/andresdp/architecture-surrogates/tree/main/notebooks) to run the Machine Learning models, including the surrogate model
* Additional scripts to compute metrics and charts based on the (intermediate) [results](https://github.com/andresdp/architecture-surrogates/tree/main/results) of the notebooks

We recommend running the notebooks in *Google Colab* using Python 3.10.12.

Alternatively, a local Python environment can be set up using the *requirements.txt* file.

----

## Datasets
We already assembled 2 datasets that contain architectural instances along with their features and quality-attribute values for the *ST+* and *CoCoME* case studies.
These systems are taken from the [SQuAT-Viz](https://github.com/SQuAT-Team/squat-vis) project. A particular feature is used to indicate the level (of search) in the tree for the architecturral instances.

For computing the graph embeddings for each architectural instance, we relied on the *FEATHER-N* algorithm from Rozemberczki et al., whose implementation is provided by the [karateclub](https://github.com/benedekrozemberczki/karateclub/tree/master) library.

## Notebooks
The notebooks should be executed in the following order:
1. *[eda.ipynb](https://github.com/andresdp/architecture-surrogates/blob/main/notebooks/eda.ipynb)*: it reads the two datasets and provides some statistics of the quality-attribute objectives, such as value distributions and correlations.
2. *[naive-approach.ipynb](https://github.com/andresdp/architecture-surrogates/blob/main/notebooks/naive-approach.ipynb)*: it implements the so-called naive approach, in which the whole dataset is split and trained with an ML model.
3. *[ial-approach.ipynb](https://github.com/andresdp/architecture-surrogates/blob/main/notebooks/ial-approach.ipynb)*: it implement the *Incremental Active-Learning (IAL)* approach, which divides the dataset in batches according to the tree level of the architectural instance and then runs the surrogate model.
4. *[reports.ipynb](https://github.com/andresdp/architecture-surrogates/blob/main/notebooks/reports.ipynb)*: it computes performance metrics for the naive and *IAL* approaches, based on the CSV results saved by the two previous notebooks. Additional test results are included in the *[scottknott_test.ipynb](https://github.com/andresdp/architecture-surrogates/blob/main/notebooks/scottknott_test.ipynb)* notebook.

The *IAL* approach can be configured to run either with a *RandomForest* or *XGBoost* (multi-output) regressor.

The active learning strategy relies on the [moDAL](https://github.com/modAL-python/modAL) library.

## Extension Scripts
Additional training and analysis scripts are provided in the [extension](https://github.com/andresdp/architecture-surrogates/tree/main/extension) folder to run advanced raw features analysis with CatBoost models:

### CoCoME Raw Features Training
- **[main_cocome_raw_training.py](https://github.com/andresdp/architecture-surrogates/blob/main/extension/main_cocome_raw_training.py)**: Trains CatBoost models using all 930+ raw features from the CoCoME dataset instead of engineered features. Uses incremental level-by-level training (Level 1→2→3→4→5) with separate models for each target metric.

**How to run:**
```bash
cd extension
python main_cocome_raw_training.py
```

**Generated plots:** Shows RMSE performance across complexity levels for CatBoost SingleOutput and MultiOutput models. The plots demonstrate how CatBoost models perform when trained on high-dimensional raw feature space (operation columns and embedding features) compared to the engineered features approach.

### STPlus Raw Features Training  
- **[main_stplus_raw_training.py](https://github.com/andresdp/architecture-surrogates/blob/main/extension/main_stplus_raw_training.py)**: Similar to CoCoME version but adapted for STPlus dataset with ~700+ raw features, 2 complexity levels (Level 1→2), and 4 target metrics (m1, m2, p1, p2).

**How to run:**
```bash
cd extension
python main_stplus_raw_training.py
```

**Generated plots:** Displays per-target performance comparison between CatBoost SingleOutput and MultiOutput approaches for the STPlus case study using raw features.

### Monte Carlo Analysis Scripts
- **[monte_carlo_raw_analysis.py](https://github.com/andresdp/architecture-surrogates/blob/main/extension/monte_carlo_raw_analysis.py)**: Runs multiple simulations of the CoCoME raw features training to analyze performance variability and statistical significance. Provides confidence intervals and mean/standard deviation across multiple runs.

**How to run:**
```bash
cd extension
python monte_carlo_raw_analysis.py
```

**Generated plots:** Monte Carlo analysis showing CatBoost model performance distribution across multiple runs with error bars and confidence intervals for overall and per-target RMSE.

- **[monte_carlo_stplus_raw_analysis.py](https://github.com/andresdp/architecture-surrogates/blob/main/extension/monte_carlo_stplus_raw_analysis.py)**: STPlus version of Monte Carlo analysis for statistical validation of raw features training results.

**How to run:**
```bash
cd extension
python monte_carlo_stplus_raw_analysis.py
```

**Generated plots:** Statistical analysis of CatBoost performance on STPlus dataset showing variability across different random seeds and training configurations.

### Sampling Curve Analysis
- **[sampling_curve.py](https://github.com/andresdp/architecture-surrogates/blob/main/extension/sampling_curve.py)**: Analyzes how CatBoost model performance varies with different training data sample ratios (30%-80%) for CoCoME dataset. Tests incremental learning efficiency with reduced training data.

**How to run:**
```bash
cd extension
python sampling_curve.py
```

**Generated plots:** Sampling curves showing normalized RMSE vs sample ratio for each complexity level, with baselines for comparison. Demonstrates data efficiency and convergence patterns of CatBoost models on raw features.

- **[stplus_sampling_curve.py](https://github.com/andresdp/architecture-surrogates/blob/main/extension/stplus_sampling_curve.py)**: STPlus version of sampling curve analysis adapted for 2 complexity levels and 4 target metrics.

**How to run:**
```bash
cd extension
python stplus_sampling_curve.py
```

**Generated plots:** STPlus-specific sampling curves comparing incremental learning performance against stratified baselines, showing how training data size affects model quality.

### Key Features of CatBoost Analysis
The extension scripts focus specifically on **CatBoost** models because they:
- Handle high-dimensional raw features (930+ for CoCoME, 700+ for STPlus) effectively
- Support true incremental learning through model continuation  
- Provide robust performance on architecture optimization tasks
- Show superior convergence compared to XGBoost on these datasets

The generated plots demonstrate **incremental learning curves** where models are trained progressively through complexity levels, showing how performance evolves as architectural complexity increases. All RMSE measurements are performed in original target space for meaningful comparison across different approaches.

## Other sources
Additional details and code can be found in the Masters Thesis *"An Empirical Study on the Effectiveness of Surrogate Model Solving for Efficient Architecture-based Quality Optimization"* (Vadim Titov, University of Hamburg, 2023) at this [repository](https://git.informatik.uni-hamburg.de/6titov/masters-thesis).
