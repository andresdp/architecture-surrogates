from fileinput import filename
import subprocess
import os
import pandas as pd
import numpy as np
from typing import List, Any, Tuple


# This class is ony for testing purposes and to exemplify how the
# file-based communication protocol with the surrogate model works.
# When implemented in EASIER, this class will be replaced by a Java implementation
class OptimizationEngine:

    def __init__(self, problem: str, algorithm: str):
        self.algorithm = algorithm
        self.problem = problem
        self.objectives = []
        self.surrogate_objectives = []

    @staticmethod
    def call_python_script(script_path="surrogatesdriver.py", script_args=[]):
        # Construct the command list
        command = ["python", script_path] + script_args
        # Execute the script
        result = subprocess.run(command, check=True) # capture_output=True, text=True, check=True)
        return result

    def generate_population(self, filename: str, n_round: int, previous_population: str=None) -> None:
        """Generate a population of solutions and save to a file."""
        
        print(f"Generating population - {self.algorithm} on {self.problem} and saving to {filename}.")

        # TODO: Here we should implement the logic to generate a new population 
        # either from scratch or based on the previous population
        
        # Get the filename without folder information
        filename_ = filename.split("/")[-1] if "/" in filename else filename
        folder_ = filename.removesuffix(filename_)
        if n_round is None:
          output_name = folder_ + "all_"+filename_
          dataset = self.assemble_dataframe(self.X, self.y)
        else:
          output_name = folder_ + str(n_round)+"_"+filename_
          dataset = self.assemble_dataframe(self.X[n_round-1], self.y[n_round-1])
        
        dataset.to_csv(output_name, index=False)
        print(f">> New population saved to {output_name}.")
        
        return output_name

    INITIAL_TRAINING_ARGS = ["-i", "-o"] 
    def initial_train(self, filename: str, n_round: int=0) -> str:
        """Perform initial training of the surrogate model."""
        
        print(f">> Passing population for initial training - {self.algorithm} on {self.problem} for round {n_round}")
        
        filename_ = filename.split("/")[-1] if "/" in filename else filename
        folder_ = filename.removesuffix(filename_)
        if n_round is not None:
          filename_ = filename_.removeprefix(str(n_round)+"_") 
          output_name = folder_ + str(n_round) + "_" + filename_
          dataset = self.assemble_dataframe(self.X[0], self.y[0])
        else:
          filename_ = filename_.removeprefix("all_") 
          output_name = folder_ + "all_" + filename_
          dataset = self.assemble_dataframe(self.X, self.y)

        dataset.to_csv(output_name, index=False)
        print(f">> initial data saved to {output_name}")
        
        # 1. Invoke the surrogate to perform initial training
        args = [OptimizationEngine.INITIAL_TRAINING_ARGS[0], output_name, OptimizationEngine.INITIAL_TRAINING_ARGS[1]] + self.surrogate_objectives
        self.call_python_script(script_args=args)
        
        return output_name

    CANDIDATE_SELECTION_ARGS = ["-s", "-f", "-o"] # Check if we need parameters for this script
    def select_candidates_for_solvers(self, filename: str, n_round: int=0) -> None:
        """Select candidates for the optimization solvers."""
        
        # 2. Invoke the surrogate to select best candidates to evaluate with solvers
        print(f">> Preparing population for candidates selection for {self.algorithm} on {self.problem} for round {n_round}")
        
        filename_ = filename.split("/")[-1] if "/" in filename else filename
        folder_ = filename.removesuffix(filename_)
        filename_ = filename_.removeprefix(str(n_round)+"_") 
        output_name = folder_ + str(n_round) + "_" + filename_

        # In this case, the values of the Y column are not needed
        dataset = self.assemble_dataframe(self.X[n_round-1]) #, self.y[n_round-1])
        dataset.to_csv(output_name, index=False)
        print(f">> data to select saved to {output_name}")

        args = [OptimizationEngine.CANDIDATE_SELECTION_ARGS[0], output_name, OptimizationEngine.CANDIDATE_SELECTION_ARGS[1], str(self.fraction), OptimizationEngine.CANDIDATE_SELECTION_ARGS[2]] + self.surrogate_objectives
        self.call_python_script(script_args=args)
        
        return output_name

    CANDIDATES_SELECTION_PREFIX2 = "marked"
    CANDIDATES_SELECTION_PREFIX3 = "evaluated"
    def run_solvers_on_candidates(self, filename: str, n_round: int=0) -> None:
        """Run optimization solvers on the selected candidates."""

        # 3. Read the population file with candidate updates
        # Ideally, this file should be created by the surrogate model        
        filename_ = filename.split("/")[-1] if "/" in filename else filename
        folder_ = filename.removesuffix(filename_)
        filename_ = filename_.removeprefix(str(n_round)+"_").removesuffix(".csv")
        output_name = folder_ + str(n_round) + "_" + filename_ + "_" + self.CANDIDATES_SELECTION_PREFIX2 + ".csv"
        print(f">> Read population (with marked candidates) from {output_name}")
        input_df = pd.read_csv(output_name)
        # Get a  mask for all rows from input_df whose objective values are all not NaN
        mask = input_df[self.surrogate_objectives].notna().all(axis=1)

        # 4. Apply solvers only on rows that are candidates
        # TODO: Call Easier code here on candidates from population data
        # Add values for the objectives that do not belong to surrogate objectives
        # additional_objectives = list(set(self.objectives) - set(self.surrogate_objectives))
        # original_dataset = self.assemble_dataframe(self.X[n_round-1], self.y[n_round-1])
        # input_df[additional_objectives] = original_dataset[additional_objectives]  # Add additional objectives to the input dataframe

        print(f">> Passing population with evaluated candidates - {self.algorithm} on {self.problem} for round {n_round}")
        output_name = output_name.replace(OptimizationEngine.CANDIDATES_SELECTION_PREFIX2, OptimizationEngine.CANDIDATES_SELECTION_PREFIX3)
        dataset = self.assemble_dataframe(self.X[n_round-1], self.y[n_round-1])
        # All objective values that are not True in mask must be assigned to NaN
  
        dataset.loc[~mask, self.surrogate_objectives] = np.nan

        dataset.to_csv(output_name, index=False)
        print(f">> data evaluated saved to {output_name}")
        
        return output_name

    RETRAIN_ARGS = ["-r", "-o"] 
    PREDICT_ARGS = ["-p", "-o"] 
    PREFIX_PREDICTION = "predicted"
    def retrain_and_predict(self, filename: str, n_round: int) -> None:
        """Perform training of the surrogate model (beyond round 0)."""
        
        print(f">> Asking to re-train surrogate model - {self.algorithm} on {self.problem} for round {n_round}")
        
        filename_ = filename.split("/")[-1] if "/" in filename else filename
        folder_ = filename.removesuffix(filename_)
        filename_ = filename_.removeprefix(str(n_round)+"_") 
        output_name = folder_ + str(n_round) + "_" + filename_
        
        # 5. Invoke the surrogate to perform re-training
        args = [OptimizationEngine.RETRAIN_ARGS[0], output_name, OptimizationEngine.RETRAIN_ARGS[1]] + self.surrogate_objectives
        self.call_python_script(script_args=args)
    
        # 6. Invoke the surrogate to perform predictions
        args = [OptimizationEngine.PREDICT_ARGS[0], output_name, OptimizationEngine.PREDICT_ARGS[1]] + self.surrogate_objectives
        self.call_python_script(script_args=args)
        
        # 7. Read the population file with everything predicted
        # Ideally, this file should be created by the surrogate model        
        output_name = output_name.replace(OptimizationEngine.CANDIDATES_SELECTION_PREFIX3, OptimizationEngine.PREFIX_PREDICTION)
        print(f">> Read population (with predictions) from {output_name}")

        return output_name
    
    FOLDER = "./runs/"
    def run_engine(self, max_evaluations: int = 10, incremental: bool=True, training_percent: float = 0.1) -> dict:
        """Run the optimization algorithm."""
        print(f"Running {self.algorithm} on {self.problem} for {max_evaluations} evaluations.")
        # Here you would implement the logic to run the optimization algorithm

        # Check if the folder exists, if not create it
        if not os.path.exists(OptimizationEngine.FOLDER):
            os.makedirs(OptimizationEngine.FOLDER)

        if not incremental:
          print(f"It's NON-incremental mode!")
          print(f"Round all ======")
          population_filename = OptimizationEngine.FOLDER+f"{self.problem}_{self.algorithm}.csv"
          last_population_filename = self.generate_population(population_filename, n_round=None)
          _ = self.initial_train(last_population_filename, n_round=None)
          print(f"==========")
          print("Done!") 
          return

        for n_round in range(1, max_evaluations + 1):
            print(f"Round {n_round} of {max_evaluations} ======")
            population_filename = OptimizationEngine.FOLDER+f"{self.problem}_{self.algorithm}.csv"
            if n_round == 1:
                last_population_filename = self.generate_population(population_filename, n_round)
                _ = self.initial_train(last_population_filename, n_round)
            else:
                current_filename = self.generate_population(population_filename, n_round, last_population_filename)
                current_filename = self.select_candidates_for_solvers(current_filename, n_round)
                current_filename = self.run_solvers_on_candidates(current_filename, n_round)
                last_population_filename = self.retrain_and_predict(current_filename, n_round)
        print(f"==========")
        print("Done!")
        return
        #  For now, we just return a dummy result
        # return {"result": "dummy_result", "evaluations": max_evaluations}

    def load_file(self, filename: str, objectives: List[str], surrogate_objectives: List[str]=None,
                  split_by_level: bool=True, tactics: bool=True, embeddings: bool=True, fraction: float=0.2) -> None:
        """Function to simulate loading a file."""
        self.df = pd.read_csv(filename, index_col=0)
        self.objectives = objectives
        self.surrogate_objectives = surrogate_objectives if surrogate_objectives is not None else objectives
        print(f"Loading file: {filename} {self.df.shape}")
        self.fraction = fraction

        self.X, self.y, self.feature_names = get_regression_data(self.df, target=self.objectives, 
                                                        tactics=tactics, embeddings=embeddings, 
                                                        split_by_level=split_by_level)
        # Note that self.X and self.y are lists of numpy arrays, one for each level

    def assemble_dataframe(self, X: np.ndarray, y: np.ndarray=None) -> pd.DataFrame:
        """Assembles a DataFrame from the feature matrix X and target vector y."""
        df = pd.DataFrame(data=X, columns=self.feature_names)
        # Get the indices of the surrogate objectives
        if y is not None:
            indices = [self.objectives.index(obj) for obj in self.surrogate_objectives]
            df[self.surrogate_objectives] = y[:,indices] # Assign ony the positions for surrogates
        else:
            df[self.surrogate_objectives] = np.nan
        return df #.head(20)

#############################################################

# Utility function for accessing and formating the input dataframes.
# Dataframe df is assumed to be preprocessed already
def get_regression_data(df, target, cols=None, # Target columns
                        tactics=True, embeddings=True, # Flags to select tactic encoding, embeddings encoding, or both
                        split_by_level=True) -> Tuple[np.ndarray, np.ndarray, List[str]]: # If True, the dataset is split into batches (for active learning)
  '''Prepares data for regression tasks.

  This function extracts features and target variables from a dataframe,
  optionally splitting the data into batches for active learning.

  Args:
    df: The input dataframe containing features and target variables.
    target: A list of column names representing the target variables.
    cols: A list of column names to include as features (default: None).
    tactics: A boolean flag indicating whether to include tactic encoding features (default: True).
    embeddings: A boolean flag indicating whether to include embeddings encoding features (default: True).
    split_by_level: A boolean flag indicating whether to split the data into batches based on the 'level' column (default: True).

  Returns:
    A tuple containing:
      - X_list: A list of feature matrices, or a single feature matrix if split_by_level is False.
      - y_list: A list of target variable arrays, or a single target variable array if split_by_level is False.
      - fnames: A list of feature names.
  '''

  if cols is None:
    cols = target
  temp = df.columns
  cols = cols+['solID']
  for t in target:
    if t in cols:
      cols.remove(t)
  if 'bot' in temp:
    cols.append('bot')
  if not tactics:
    tactic_cols = [c for c in temp if c.startswith('op')]
    for c in tactic_cols:
      cols.append(c)
  if not embeddings:
    emb_cols = [c for c in temp if c.startswith('emb')]
    for c in emb_cols:
      cols.append(c)

  X = df.drop(cols, axis=1)
  if len(target) == 1:
    y = df[target[0]]
  else:
    y = df[target]

  if split_by_level:
    levels = set(X['level'])
    X_list = []
    y_list = []
    for l in levels:
      #print(l)
      df_level = X[X['level'] == l]
      X_list.append(df_level.drop(['level']+target, axis=1).values)
      y_list.append(df_level[target].values)
  else:
    # X_list = X.drop(['level']+target, axis=1).values
    X_list = X.drop(target, axis=1).values
    if len(target) == 1:
      y_list = X[target[0]].values
    else:
      y_list = X[target].values

  fnames = list(X.columns)
  if split_by_level:
    fnames.remove('level')
  for t in target:
    fnames.remove(t)

  return X_list, y_list, fnames



STPLUS_DATAPATH = './datasets/stplus-levels-bots-features.csv'
COCOME_DATAPATH = './datasets/cocome-levels-features.csv'

OBJ_STPLUS= ['m1', 'm2', 'p1', 'p2']
OBJ_COCOME = ['m1', 'm2', 'm3', 'm4','p1', 'p2', 'p3', 'p4']

if __name__ == "__main__":
    """Main method to run the optimization engine."""

    n_steps = 5
    optimization_engine = OptimizationEngine(problem="cocome", algorithm="nsgaii")
    optimization_engine.load_file(COCOME_DATAPATH, OBJ_COCOME, 
                                  # surrogate_objectives=['m1', 'p1', 'p4'], fraction=0.4,
                                  split_by_level=(n_steps is not None)
                                )
    
    # n_steps = 2
    # optimization_engine = OptimizationEngine(problem="stplus", algorithm="nsgaii")
    # optimization_engine.load_file(STPLUS_DATAPATH, OBJ_STPLUS, 
    #                               # embeddings=False, tactics=True,
    #                               split_by_level=(n_steps is not None),
    #                               #, surrogate_objectives=['p1', 'p2'], fraction=0.3
    #                             )

    if n_steps is not None:
        optimization_engine.run_engine(max_evaluations=n_steps, incremental=True)
    else:
        optimization_engine.run_engine(max_evaluations=n_steps, incremental=False)
