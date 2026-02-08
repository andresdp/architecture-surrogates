from __future__ import annotations

import pandas as pd
import numpy as np

import xgboost as xgb

import json
import itertools
import re
from pathlib import Path
from typing import Dict, List, Union, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

import copy
import json

# from alibi_detect.cd import KSDrift

import random
random.seed(3)


class EasierEncoder:

    ACTIONS = ["mcnn", "moc", "moncnn", "clone"]  # fixed vocabulary of actions
    BIGRAMS = [f"{a1}_{a2}" for a1, a2 in itertools.product(ACTIONS, repeat=2)]

    # --- Node family/size vocab (fixed at time zero) ---
    FAMILY_PREFIXES = {"t2", "t3", "m5", "m5ad", "m6i", "c5", "c6i", "r5", "r6i", "d2"}
    SIZE_CLASSES = {
        "nano", "micro", "small", "medium", "large",
        "xlarge", "2xlarge", "4xlarge", "8xlarge",
        "12xlarge", "16xlarge", "24xlarge", "32xlarge"
    }
    FAM_OTHER = "OTHER"
    SIZE_OTHER = "OTHER"

    # Matches pieces like 'm6i.xlarge', 'd2.2xlarge' anywhere in a token
    _family_size_re = re.compile(
        r"([a-z]+\d+[a-z]*)\.(\d*x?large|nano|micro|small|medium|large)", re.I
    )

    @staticmethod
    def _pos_feature_names(max_len: int = 4) -> List[str]:
        return [f"pos{p}_{a}" for p in range(1, max_len + 1) for a in EasierEncoder.ACTIONS]

    @staticmethod
    def _all_family_prefixes() -> List[str]:
        return sorted(EasierEncoder.FAMILY_PREFIXES | {EasierEncoder.FAM_OTHER})

    @staticmethod
    def _all_size_classes() -> List[str]:
        return sorted(EasierEncoder.SIZE_CLASSES | {EasierEncoder.SIZE_OTHER})

    @staticmethod
    def _node_feature_names() -> List[str]:
        fams = EasierEncoder._all_family_prefixes()
        sizes = EasierEncoder._all_size_classes()
        names: List[str] = []
        # global family counts
        names += [f"count_family_{fam}" for fam in fams]
        # per-action-to-family counts (only actions that place on a node)
        for a in ("mcnn", "moncnn", "clone"):
            names += [f"count_{a}_to_family_{fam}" for fam in fams]
        # size counts
        names += [f"count_size_{s}" for s in sizes]
        # diversity
        names += ["uniq_families_used", "uniq_sizes_used"]
        return names

    @staticmethod
    def extract_family_size(token: str) -> tuple[str, str]:
        """
        Find '<family>.<size>' in a token, normalize into fixed vocab with OTHER fallback.
        Examples: 'm6i.xlarge', 'd2.2xlarge', possibly inside 'New-Node_m6i.xlarge_xyz'
        """
        if not token:
            return (EasierEncoder.FAM_OTHER, EasierEncoder.SIZE_OTHER)
        m = EasierEncoder._family_size_re.search(token)
        if not m:
            # try substrings split by '_' if embedded
            for part in token.split("_"):
                m = EasierEncoder._family_size_re.search(part)
                if m:
                    break
        if m:
            fam = m.group(1).lower()
            size = m.group(2).lower()
            fam_norm = fam if fam in EasierEncoder.FAMILY_PREFIXES else EasierEncoder.FAM_OTHER
            size_norm = size if size in EasierEncoder.SIZE_CLASSES else EasierEncoder.SIZE_OTHER
            return (fam_norm, size_norm)
        return (EasierEncoder.FAM_OTHER, EasierEncoder.SIZE_OTHER)

    @staticmethod
    def add_node_family_features(encoding: Dict, refactoring_steps: List[Dict]) -> None:
        """
        Mutates `encoding` in place, adding:
        - count_family_{fam}, count_size_{size}
        - count_{action}_to_family_{fam} for actions that place on a node (mcnn/moncnn/clone)
        - uniq_families_used, uniq_sizes_used
        """
        # Initialize all node-related columns for stability
        for fam in EasierEncoder._all_family_prefixes():
            encoding[f"count_family_{fam}"] = 0
            encoding[f"count_mcnn_to_family_{fam}"] = 0
            encoding[f"count_moncnn_to_family_{fam}"] = 0
            encoding[f"count_clone_to_family_{fam}"] = 0
        for size in EasierEncoder._all_size_classes():
            encoding[f"count_size_{size}"] = 0
        encoding["uniq_families_used"] = 0
        encoding["uniq_sizes_used"] = 0

        fam_seen, size_seen = set(), set()

        for step in refactoring_steps:
            name = (step.get("name") or "").strip()
            # Determine where the node info lives
            node_token = ""
            if name == "mcnn":
                node_token = step.get("to", "")
            elif name == "moncnn":
                node_token = step.get("where", "")
            elif name == "clone":
                node_token = step.get("to", "")
            else:
                # moc: no new node destination; skip
                continue

            fam, size = EasierEncoder.extract_family_size(node_token)

            # global counts
            encoding[f"count_family_{fam}"] += 1
            encoding[f"count_size_{size}"] += 1
            fam_seen.add(fam)
            size_seen.add(size)

            # per-action cross
            if name in ("mcnn", "moncnn", "clone"):
                encoding[f"count_{name}_to_family_{fam}"] += 1

        encoding["uniq_families_used"] = len(fam_seen)
        encoding["uniq_sizes_used"] = len(size_seen)


    def encode_solution(self,
        solution: Dict,
        include_bigrams: bool = True,
        include_positional: bool = True,
        positional_len: int = 4,
    ) -> Dict:
        """
        Encode a single solution into a fixed-size vector.

        Features:
        - count_*: unigram counts over ACTIONS
        - bigram_*: pair counts over ACTIONS x ACTIONS (if include_bigrams=True)
        - pos{k}_{action}: one-hot for action at position k (1..positional_len) (if include_positional=True)
        - num_new_nodes_created, num_new_components_created: derived from action semantics
        - node family/size features parsed from destination/where tokens
        - sequence_len, solID
        - y__<objective>: considered objectives as targets/labels (prefixed)

        Assumptions:
        - mcnn -> creates a new node
        - moncnn -> creates a new component + new node
        - clone -> creates a new node
        - moc -> moves operation to existing component (no new elements)
        """
        ref = solution.get("refactoring", []) or []
        actions = [step.get("name", "").strip() for step in ref]
        # considered = solution.get("consideredObjectives", {}) or {}
        considered = solution.get("objectives", {}) or {}

        feats: Dict[str, Union[int, float, str]] = {}

        # Base: IDs / metadata
        feats["solID"] = solution.get("solID", None)
        feats["sequence_len"] = len(actions)

        # Unigrams
        for a in EasierEncoder.ACTIONS:
            feats[f"count_{a}"] = 0
        for a in actions:
            if a in EasierEncoder.ACTIONS:
                feats[f"count_{a}"] += 1

        # Bigrams
        if include_bigrams:
            for b in EasierEncoder.BIGRAMS:
                feats[f"bigram_{b}"] = 0
            for a1, a2 in zip(actions, actions[1:]):
                if a1 in EasierEncoder.ACTIONS and a2 in EasierEncoder.ACTIONS:
                    feats[f"bigram_{a1}_{a2}"] += 1

        # Positional one-hots (fixed first N positions; missing remain 0)
        if include_positional:
            for name in EasierEncoder._pos_feature_names(positional_len):
                feats[name] = 0
            for idx, a in enumerate(actions[:positional_len], start=1):
                if a in EasierEncoder.ACTIONS:
                    feats[f"pos{idx}_{a}"] = 1

        # Creation counters (derived from action semantics)
        feats["num_new_components_created"] = feats["count_moncnn"]
        feats["num_new_nodes_created"] = (
            feats["count_mcnn"] + feats["count_moncnn"] + feats["count_clone"]
        )

        # --- Node family/size features from tokens like 'm6i.xlarge' / 'd2.2xlarge' ---
        EasierEncoder.add_node_family_features(feats, ref)

        # Targets (considered objectives)
        for k, v in considered.items():
            feats[f"y__{k}"] = v

        return feats


    def encode_population(self,
        population: Dict,
        include_bigrams: bool = True,
        include_positional: bool = True,
        positional_len: int = 4,
        save_csv: Optional[Union[str, Path]] = None,
        family_filter: bool = False,
    ) -> pd.DataFrame:
        """Encode all solutions in a population JSON to a DataFrame."""
        solutions = population.get("solutions", []) or []
        rows = [
            self.encode_solution(
                sol,
                include_bigrams=include_bigrams,
                include_positional=include_positional,
                positional_len=positional_len,
            )
            for sol in solutions
        ]
        df = pd.DataFrame(rows)

        # Order columns for readability (keep stable)
        col_order: List[str] = []
        col_order += ["solID", "sequence_len", "num_new_nodes_created", "num_new_components_created"]
        col_order += [f"count_{a}" for a in EasierEncoder.ACTIONS]
        if include_bigrams:
            col_order += [f"bigram_{b}" for b in EasierEncoder.BIGRAMS]
        if include_positional:
            col_order += EasierEncoder._pos_feature_names(positional_len)
        # node family/size block
        col_order += EasierEncoder._node_feature_names()

        # targets at the end
        target_cols = sorted([c for c in df.columns if c.startswith("y__")])

        # plus any columns not covered (safe fallback)
        remaining = [c for c in df.columns if c not in set(col_order + target_cols)]
        df = df[col_order + remaining + target_cols]

        if save_csv is not None:
            save_csv = Path(save_csv)
            save_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_csv, index=False)

        if family_filter:
            encoding_df = df[[x for x in df.columns if 'family' in x]]
            return encoding_df
        
        return df


    def load_population(self, src: Union[str, Path, Dict]) -> Dict:
        """Load a population JSON from a dict, string path, or raw JSON string."""
        if isinstance(src, dict):
            return src
        p = Path(src)
        if p.exists():
            return json.loads(Path(p).read_text(encoding="utf-8"))
        # if not a file, try to parse as raw JSON string
        return json.loads(str(src))


class EasierSurrogate:
    
    XGBOOST_DEFAULT_PARAMS = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'random_state': 42,
        # 'multi_strategy': 'one_output_per_tree' # 'multi_output_tree'
    }
    
    def __init__(self, objectives, model_id=None, strategy='one_output_per_tree'):
        self.objectives = objectives
        self.reset(model_id, strategy=strategy)
    
    @staticmethod
    def create_xgb_model(objectives, model_id, strategy='one_output_per_tree'):
        model = None
        if model_id is not None:
            model = dict()
            for obj in objectives:
                json_filename = model_id+"__"+obj.lower()
                print("Loading XGBoot model as "+json_filename)
                if not json_filename.endswith('json'):
                    json_filename = json_filename + ".json"
                json_filename = Path(json_filename)
                loaded_model = xgb.XGBRegressor(**EasierSurrogate.XGBOOST_DEFAULT_PARAMS, multi_strategy=strategy)
                if json_filename.exists():
                    loaded_model.load_model(json_filename)
                model[obj] = loaded_model
        return model
 
    def reset(self, model_id=None, strategy='one_output_per_tree'):
        if model_id is None:
            self.model = {obj:None for obj in self.objectives}
        else:
            self.model = EasierSurrogate.create_xgb_model(self.objectives, model_id, strategy=strategy)
        
        self.df = None
        self.features = None
        self.strategy = strategy
    
    @staticmethod
    def get_data(df):
        x_cols = [col for col in df.columns if not col.startswith("y")]
        y_cols = [col for col in df.columns if col.startswith("y")]

        X = df[x_cols].drop(columns=["solID", "population_number"], axis=1)
        y = df[["solID"] + y_cols].drop(columns=["solID"], axis=1)
        print(X.shape, "objectives:", len(y_cols))

        return X, y
    
    @staticmethod
    def train_xgboost(X, y, test_size=None, stratify=None, strategy='one_output_per_tree', previous_model=None):

        if test_size is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=42)
            print(test_size, X_train.shape, X_test.shape)
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None
        
        model = xgb.XGBRegressor(**EasierSurrogate.XGBOOST_DEFAULT_PARAMS, multi_strategy=strategy)
        model.fit(X_train, y_train, xgb_model=previous_model)

        return model, X_test, y_test
    
    @staticmethod
    def validate_xgboost(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse) 
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        # print(f"\tTest metrics: R2={r2} -- RMSE={rmse} -- MAPE={(mape * 100):.2f}% ")
        return r2, rmse, mape
    
    @staticmethod
    def encode_as_dataframe(batch_dict, population_id):
        
        encoder = EasierEncoder()
        population_df = encoder.encode_population(
                batch_dict,
                include_bigrams=True,
                include_positional=True,
                positional_len=4,
                # save_csv=args.out,
            )
        population_df["population_number"] = population_id
        return population_df
    
    # This method should be called externally
    def encode_and_retain(self, batch_dict, population_id, accumulate=False):

        population_df = EasierSurrogate.encode_as_dataframe(batch_dict, population_id)        
        self.retain(population_df, accumulate=accumulate)
    
    def retain(self, batch_df, accumulate=False):
        if not accumulate:
            self.df = batch_df  
        elif self.df is None:
            self.df = batch_df
        else:
            result = pd.concat([self.df, batch_df], ignore_index=True) 
            self.df = result
        print("Retain:", self.df.shape)

    # This method should be called externally
    def train(self, incremental_learning=False, test_size=0.25, model_id=None):

        print(model_id)
        X, y = EasierSurrogate.get_data(self.df)
        # print(X.shape, y.shape)
        self.features = X.columns

        if self.model is None:
            self.model = dict()

        if self.strategy == 'one_output_per_tree':
            # Have one regressor per objective
            for obj in self.objectives:
                print(f"Training model for objective: {obj}")
                previous_model = self.model[obj] if incremental_learning else None
                # For simplicity, we have one model (surrogate) per objective
                model, X_test, y_test = EasierSurrogate.train_xgboost(
                                X, y[obj], 
                                test_size=test_size, 
                                stratify=self.df["population_number"],
                                previous_model=previous_model,
                                strategy=self.strategy
                            )
                if (X_test is not None) and (y_test is not None):
                    r2, rmse, mape = EasierSurrogate.validate_xgboost(model, X_test, y_test)
                    print(f"\tTest metrics: R2={r2} -- RMSE={rmse} -- MAPE={(mape * 100):.2f}% ")
                else:
                    print("Skipping validation")
                self.model[obj] = model
                if model_id is not None:
                    json_filename = model_id+"__"+obj.lower()
                    if not json_filename.endswith('json'):
                        json_filename = json_filename + ".json"
                    print("Saving XGBoot model as "+json_filename)
                    model.save_model(json_filename)
        
        else: # Have only one regressor for all the objectives
            print(f"Training multi-output model for objectives: {self.objectives}")
            previous_model = self.model['all'] if incremental_learning else None
            model, X_test, y_test = EasierSurrogate.train_xgboost(
                            X, y[self.objectives], 
                            test_size=test_size, 
                            stratify=self.df["population_number"],
                            previous_model=previous_model,
                            strategy=self.strategy
                        )
            if (X_test is not None) and (y_test is not None):
                r2, rmse, mape = EasierSurrogate.validate_xgboost(model, X_test, y_test)
                print(f"\tTest metrics for all: R2={r2} -- RMSE={rmse} -- MAPE={(mape * 100):.2f}% ")
            else:
                print("Skipping validation")
            self.model = {'all': model}
            if model_id is not None:
                json_filename = model_id+"__all"
                if not json_filename.endswith('json'):
                    json_filename = json_filename + ".json"
                print("Saving XGBoot multi-output model as "+json_filename)
                model.save_model(json_filename)

    def encode_and_validate(self, batch_dict, population_id, model_id=None):

        population_df = EasierSurrogate.encode_as_dataframe(batch_dict, population_id)
        self.validate(population_df, model_id=model_id+"_"+str(population_id))
        
    def validate(self, batch_df, model_id=None):

        if self.model is None:
            self.model = EasierSurrogate.create_xgb_model(self.objectives, model_id)

        X, y = EasierSurrogate.get_data(batch_df)


        if self.strategy == 'one_output_per_tree':
            for obj in self.objectives:
                print(f"Validating for objective: {obj}")
                r2, rmse, mape = EasierSurrogate.validate_xgboost(self.model[obj], X, y[obj])
                print(f"\tTest metrics: R2={r2} -- RMSE={rmse} -- MAPE={(mape * 100):.2f}% ")
        else:
            print(f"Validating for all objectives")
            r2, rmse, mape = EasierSurrogate.validate_xgboost(self.model['all'], X, y[self.objectives])
            print(f"\tTest metrics for all: R2={r2} -- RMSE={rmse} -- MAPE={(mape * 100):.2f}% ")
    
    @staticmethod
    def decode(batch_dict, df, objectives):
        preds_batch_dict = copy.deepcopy(batch_dict)
        idx_sol = 0
        for idx, sol in enumerate(preds_batch_dict['solutions']):
            if len(sol['objectives']) == 0: # The solution has no objective values yet
                for obj in objectives:
                    preds_batch_dict['solutions'][idx]['objectives'][obj] = float(df.at[idx_sol,obj])
                idx_sol += 1
        return preds_batch_dict
    
    # This method should be called externally
    def encode_and_active_learning(self, batch_dict, population_id, model_id=None):
                
        population_df = EasierSurrogate.encode_as_dataframe(batch_dict, population_id)
        preds = self.active_learning(population_df, model_id=model_id+"_"+str(population_id))

        return EasierSurrogate.decode(batch_dict, preds, self.objectives)
            
    def active_learning(self, batch_df, model_id=None):
        # sample_population = batch_df.sample(frac=sampling_factor, random_state=random_state)
        # Select only those raws with values (not NaN) in their objectives
        sample_population = batch_df[batch_df[self.objectives].notna().all(axis=1)]
        self.retain(sample_population, accumulate=True)
        if self.model is None:
            self.model = EasierSurrogate.create_xgb_model(self.objectives, model_id)
        self.train(incremental_learning=True, test_size=None, model_id=model_id)

        # Get the remaining rows from population3 that are not in sample
        remaining_indices = batch_df.index.difference(sample_population.index)
        remaining_population = batch_df.loc[remaining_indices]

        return self.predict(remaining_population[surrogate.features])
    
    # This method should be called externally
    def encode_and_predict(self, batch_dict, population_id, model_id=None):

        population_df = EasierSurrogate.encode_as_dataframe(batch_dict, population_id)
        preds = self.predict(population_df, model_id+"_"+str(population_id))
        
        return EasierSurrogate.decode(batch_dict, preds, self.objectives)

    def predict(self, batch_df, model_id=None):
        if self.model is None:
            self.model = EasierSurrogate.create_xgb_model(self.objectives, model_id)
        
        predictions_df = pd.DataFrame()
        if self.strategy == 'one_output_per_tree':
            for obj in self.objectives:
                predictions_df[obj] = self.model[obj].predict(batch_df)
        else:
            all_preds = self.model['all'].predict(batch_df)
            print(all_preds.shape)
            predictions_df = pd.DataFrame(all_preds, columns=self.objectives)
        
        return predictions_df

    def check_drift(self, batch_df, reference_df=None):

        if reference_df is None:
            x_ref = self.df[self.features].values
        else:
            x_ref = reference_df[self.features].values
        cd = KSDrift(x_ref, p_val=0.05)

        x = batch_df[self.features].values
        preds = cd.predict(x, drift_type='batch', return_p_val=True, return_distance=True)
        # print("Is drift?", preds['data']['is_drift'])
        # return (preds['data']['is_drift'] == 1)
        return preds


# Test ----------------------------------------------------------------

# JSON_FILENAME = "./easier_pop_json/population"
# JSON_FILENAME = "./nsgaii-ccm-eval-102-surrogate-false/run1/population"
# MODEL_NAME = './models/cocome'
# N = 102 # N = 196

# TRAINTICKET
JSON_FILENAME = "./nsgaii-ttbs-eval-102-surrogate-false/run1/population"
MODEL_NAME = './models/trainticket'
N = 102

def load_populations_to_dictionary(prefix, n):
    encoder = EasierEncoder()
     
    populations = dict()
    for idx in range(1,n+1):
        json_filename = prefix + '__' + str(idx) + '.json'
        # print("Loading population from "+json_filename)
        population = encoder.load_population(json_filename)
        populations[json_filename] = population

    return populations

def select_random_indices(population, sampling_factor):
    n_range = len(population['solutions'])
    my_list = range(0, n_range) 
    num_to_select = round(n_range * sampling_factor)
    return random.sample(my_list, num_to_select)


if __name__ == "__main__":

    # Load all populations
    all_populations_dict = load_populations_to_dictionary(prefix=JSON_FILENAME, n=N)
    all_populations = list(all_populations_dict.values())

    # Define a list of objectives for the surrogate
    target_objectives = [
        'y__changes',
        'y__energy',
        # 'y__energyPerScenario__ProcessSale_job_class',
        # 'y__energyPerScenario__ReceivedOrderedProducts_job_class',
        # 'y__energyPerScenario__ShowDeliveryReports_job_class',
        'y__pas',
        'y__perfq',
        'y__power',
        'y__price',
        # 'y__pricePerScenario__ProcessSale_job_class',
        # 'y__pricePerScenario__ReceivedOrderedProducts_job_class',
        # 'y__pricePerScenario__ShowDeliveryReports_job_class',
        'y__reliability'
        # 'y__responseTimePerScenario__ProcessSale_job_class',
        # 'y__responseTimePerScenario__ReceivedOrderedProducts_job_class',
        # 'y__responseTimePerScenario__ShowDeliveryReports_job_class',
        # 'y__sysRespT'
    ]

    surrogate = EasierSurrogate(target_objectives, model_id=None, strategy='one_output_per_tree')
    # surrogate = EasierSurrogate(target_objectives, model_id=MODEL_NAME, strategy='multi_output_tree')

    # 0. Train the surrogate with the first 20 populations
    n_populations = 20
    for idx in range(0,n_populations):
        population_idx = all_populations[idx]
        surrogate.encode_and_retain(population_idx, population_id=idx+1, accumulate=True)
    print(surrogate.df.shape)
    surrogate.train(model_id=MODEL_NAME)
    print()

    # 1. Re-train the surrogate with the (whole) 21st population
    print('=='*10)
    n_populations = 21
    population2 = all_populations[n_populations]
    surrogate.encode_and_retain(population2, population_id=n_populations, accumulate=False)
    surrogate.train(incremental_learning=True)
    print()

    frac = 0.7 # This is the fraction to remove objectives

    # 1. Do active learning (re-training) on the surrogate with the 30% of the 22nd population
    print('=='*10)
    n_populations = 21
    population3 = copy.deepcopy(all_populations[n_populations])
    # batch_df = EasierSurrogate.encode_as_dataframe(population3, population_id=n_populations)
    # print("Drift?", surrogate.check_drift(batch_df))

    random_selection = select_random_indices(population3, sampling_factor=frac) 
    # Randomly remove values of objectives for some solutions
    for idx in random_selection: 
        population3['solutions'][idx]['objectives'] = {}
    # with open('input-population.json', 'w') as file:
    #     json.dump(population3, file, indent=4)
    
    predicted_population = surrogate.encode_and_active_learning(population3, population_id=n_populations, model_id=MODEL_NAME)
    # print(predicted_population)
    # with open('output-population.json', 'w') as file:
    #     json.dump(predicted_population, file, indent=4)
    print()

    surrogate.encode_and_validate(all_populations[n_populations], population_id=n_populations, model_id=MODEL_NAME)
    print()

    # 1. Do active learning (re-training) on the surrogate with the 30% of the 23nd population
    print('=='*10)
    n_populations = 22
    population3 = copy.deepcopy(all_populations[n_populations])
    # batch_df = EasierSurrogate.encode_as_dataframe(population3, population_id=n_populations)
    # print("Drift?", surrogate.check_drift(batch_df))

    random_selection = select_random_indices(population3, sampling_factor=frac) 
    # Randomly remove values of objectives for some solutions
    for idx in random_selection: 
        population3['solutions'][idx]['objectives'] = {}

    predicted_population = surrogate.encode_and_active_learning(population3, population_id=n_populations, model_id=MODEL_NAME)
    # print(predicted_population)
    print()

    surrogate.encode_and_validate(all_populations[n_populations], population_id=n_populations, model_id=MODEL_NAME)
    print()
