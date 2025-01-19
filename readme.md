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
4. *[reports.ipynb](https://github.com/andresdp/architecture-surrogates/blob/main/notebooks/reports.ipynb)*: it computes performance metrics for the naive and *IAL* approaches, based on the CSV results saved by the two previous notebooks.

The *IAL* approach can be configured to run either with a *RandomForest* or *XGBoost* (multi-output) regressor.

The active learning strategy relies on the [moDAL](https://github.com/modAL-python/modAL) library.

## Other sources
Additional details and code can be found in the Masters Thesis *"An Empirical Study on the Effectiveness of Surrogate Model Solving for Efficient Architecture-based Quality Optimization"* (Vadim Titov, University of Hamburg, 2023) at this [repository](https://git.informatik.uni-hamburg.de/6titov/masters-thesis).
